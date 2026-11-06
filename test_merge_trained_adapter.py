"""Synthetic streaming merge validation. No downloaded weights or GPU required."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import merge_trained_adapter as merge


def write_tensor_file(path, entries):
    header, parts, offset = {}, [], 0
    for name, dtype, shape, raw in entries:
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + len(raw)]}
        parts.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * ((-len(encoded)) % 8)
    Path(path).write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(parts))


def f32(*values):
    return struct.pack('<' + 'f' * len(values), *values)


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_valid_header_reads_exact_tensor_bytes(self):
        file = self.root / 'model.safetensors'
        write_tensor_file(file, [('x', 'F32', [2, 2], f32(1, 2, 3, 4))])
        _, tensors = merge.read_header(file)
        self.assertEqual(tensors['x'].read(), f32(1, 2, 3, 4))
        self.assertEqual(tensors['x'].shape, (2, 2))

    def test_duplicate_json_keys_and_bad_offsets_are_rejected(self):
        file = self.root / 'bad.safetensors'
        for header, data in ((b'{"x":1,"x":2}', b''),
                             (b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}', b'12345678'),
                             (b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}', b'12345')):
            file.write_bytes(struct.pack('<Q', len(header)) + header + data)
            with self.subTest(header=header), self.assertRaises(ValueError):
                merge.read_header(file)

    def test_overlap_and_trailing_data_are_rejected(self):
        file = self.root / 'bad.safetensors'
        entries = {'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
                   'y': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}
        header = json.dumps(entries).encode()
        file.write_bytes(struct.pack('<Q', len(header)) + header + f32(1))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            merge.read_header(file)
        write_tensor_file(file, [('x', 'F32', [1], f32(1))])
        with file.open('ab') as output:
            output.write(b'extra')
        with self.assertRaisesRegex(ValueError, 'trailing'):
            merge.read_header(file)

    def test_index_cannot_escape_base_folder(self):
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': '../outside.safetensors'}}))
        with self.assertRaises(ValueError):
            merge.base_shards(self.root)

    def test_adapter_variant_and_missing_pair_are_rejected(self):
        config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 1, 'lora_alpha': 2, 'use_dora': True}
        (self.root / 'adapter_config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'variant'):
            merge.adapter_pairs(self.root)
        config['use_dora'] = False
        (self.root / 'adapter_config.json').write_text(json.dumps(config))
        write_tensor_file(self.root / 'adapter_model.safetensors', [
            ('base_model.model.layer.lora_A.weight', 'F32', [1, 2], f32(1, 2))])
        with self.assertRaisesRegex(ValueError, 'pair'):
            merge.adapter_pairs(self.root)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'CPU PyTorch unavailable in this interpreter')
class NumericalMergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base, self.adapter, self.output = (self.root / name for name in ('base', 'adapter', 'merged'))
        self.base.mkdir()
        self.adapter.mkdir()
        (self.base / 'config.json').write_text('{"architectures":["Qwen3_5ForConditionalGeneration"]}')
        (self.base / 'tokenizer_config.json').write_text('{"original":true}')
        (self.adapter / 'tokenizer_config.json').write_text('{"original":false}')
        self.config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 1, 'lora_alpha': 2,
                       'base_model_name_or_path': 'synthetic', 'bias': 'none'}
        (self.adapter / 'adapter_config.json').write_text(json.dumps(self.config))
        self.write_adapter()
        write_tensor_file(self.base / 'model.safetensors', [
            ('layer.weight', 'F32', [2, 2], f32(1, 2, 3, 4)),
            ('model.visual.weight', 'F32', [2], f32(8, 9))])

    def write_adapter(self, left=(1, 2), right=(3, 4)):
        write_tensor_file(self.adapter / 'adapter_model.safetensors', [
            ('base_model.model.layer.lora_A.weight', 'F32', [1, 2], f32(*left)),
            ('base_model.model.layer.lora_B.weight', 'F32', [2, 1], f32(*right))])

    def test_cpu_math_matches_formula_and_preserves_vision_metadata_and_inputs(self):
        original = (self.base / 'model.safetensors').read_bytes()
        report = merge.merge(self.base, self.adapter, self.output, threads=1)
        _, tensors = merge.read_header(self.output / 'model.safetensors')
        self.assertEqual(struct.unpack('<4f', tensors['layer.weight'].read()), (7, 14, 11, 20))
        self.assertEqual(tensors['model.visual.weight'].read(), f32(8, 9))
        self.assertEqual((self.base / 'model.safetensors').read_bytes(), original)
        self.assertEqual((self.output / 'tokenizer_config.json').read_text(), '{"original":true}')
        self.assertEqual((report['merged_tensors'], report['unchanged_tensors']), (1, 1))
        self.assertEqual(report['vision_adapter_tensors'], 0)

    def test_sharded_input_preserves_index_and_each_shard(self):
        (self.base / 'model.safetensors').unlink()
        write_tensor_file(self.base / 'part-1.safetensors', [('layer.weight', 'F32', [2, 2], f32(1, 2, 3, 4))])
        write_tensor_file(self.base / 'part-2.safetensors', [('model.visual.weight', 'F32', [2], f32(8, 9))])
        original_vision = (self.base / 'part-2.safetensors').read_bytes()
        index = {'weight_map': {'layer.weight': 'part-1.safetensors', 'model.visual.weight': 'part-2.safetensors'}}
        (self.base / 'model.safetensors.index.json').write_text(json.dumps(index))
        merge.merge(self.base, self.adapter, self.output, threads=1)
        self.assertEqual((self.output / 'part-2.safetensors').read_bytes(), original_vision)
        self.assertEqual(merge.read_json(self.output / 'model.safetensors.index.json'), index)

    def test_bfloat16_output_retains_bfloat16_dtype(self):
        import torch
        values = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
        write_tensor_file(self.base / 'model.safetensors', [('layer.weight', 'BF16', [2, 2], values)])
        merge.merge(self.base, self.adapter, self.output, threads=1)
        _, tensors = merge.read_header(self.output / 'model.safetensors')
        self.assertEqual(tensors['layer.weight'].dtype, 'BF16')
        self.assertEqual(merge.tensor_from_bytes(tensors['layer.weight']).float().tolist(), [[7, 14], [11, 20]])

    def test_nonfinite_adapter_is_rejected_without_publishing_output(self):
        self.write_adapter(left=(float('nan'), 2))
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            merge.merge(self.base, self.adapter, self.output, threads=1)
        self.assertFalse(self.output.exists())

    def test_existing_nonempty_output_and_in_place_merge_are_refused(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'not empty'):
            merge.merge(self.base, self.adapter, self.output, threads=1)
        self.assertEqual(marker.read_text(), 'keep')
        with self.assertRaisesRegex(ValueError, 'separate'):
            merge.merge(self.base, self.adapter, self.base / 'out', threads=1)

    def test_missing_target_and_shape_mismatch_fail_before_output_creation(self):
        write_tensor_file(self.base / 'model.safetensors', [('different.weight', 'F32', [2, 2], f32(1, 2, 3, 4))])
        with self.assertRaisesRegex(ValueError, 'missing'):
            merge.merge(self.base, self.adapter, self.output, threads=1)
        self.assertFalse(self.output.exists())
        write_tensor_file(self.base / 'model.safetensors', [('layer.weight', 'F32', [1, 4], f32(1, 2, 3, 4))])
        with self.assertRaisesRegex(ValueError, 'shape'):
            merge.merge(self.base, self.adapter, self.output, threads=1)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
