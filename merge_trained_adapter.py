"""Development-only CPU LoRA merge, streaming original Safetensors shards.

Never imports adapter Python modules or pickle checkpoints. Requires CPU PyTorch
and NumPy for actual math; header inspection and metadata validation use stdlib.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import uuid

DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'I32': 4, 'I64': 8,
               'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8}
FLOAT_DTYPES = {'F16', 'BF16', 'F32', 'F64'}
METADATA_FILES = {'config.json', 'generation_config.json', 'tokenizer.json',
                  'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja',
                  'chat_template.json', 'processor_config.json', 'preprocessor_config.json',
                  'video_preprocessor_config.json', 'vocab.json', 'merges.txt',
                  'tokenizer.model', 'added_tokens.json', 'model.safetensors.index.json'}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def member(root: Path, relative: str) -> Path:
    given = Path(relative)
    if given.is_absolute() or '..' in given.parts or ':' in relative:
        raise ValueError('Model member must be a relative path inside its folder')
    path = (root / given).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Model member escapes its folder or is not a file')
    return path


@dataclass(frozen=True)
class TensorInfo:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int
    data_start: int

    def read(self):
        with self.path.open('rb') as source:
            source.seek(self.data_start + self.start)
            raw = source.read(self.end - self.start)
        if len(raw) != self.end - self.start:
            raise ValueError(f'Truncated tensor data: {self.path.name}')
        return raw


def read_header(path: Path):
    """Validate offsets, nonoverlap, dtype sizes and exact file data coverage."""
    path = Path(path)
    with path.open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('Missing Safetensors header length')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= 64 * 1024 * 1024:
            raise ValueError('Safetensors header length outside supported bounds')
        raw_header = source.read(length)
    if len(raw_header) != length:
        raise ValueError('Truncated Safetensors header')
    header = json.loads(raw_header, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError('Safetensors header must be an object')
    data_start = length + 8
    available = path.stat().st_size - data_start
    tensors = {}
    ranges = []
    for name, item in header.items():
        if name == '__metadata__':
            if not isinstance(item, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in item.items()):
                raise ValueError('Safetensors metadata must contain only strings')
            continue
        if not isinstance(item, dict) or set(item) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError(f'Invalid tensor metadata: {name}')
        dtype, shape, offsets = item['dtype'], item['shape'], item['data_offsets']
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'Unsupported tensor dtype or shape: {name}')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError(f'Invalid data offsets: {name}')
        start, end = offsets
        if start < 0 or end < start or end > available or end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f'Tensor byte size or bounds mismatch: {name}')
        tensors[name] = TensorInfo(path, dtype, tuple(shape), start, end, data_start)
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError('Safetensors data has overlap or holes')
        cursor = end
    if cursor != available:
        raise ValueError('Safetensors file has trailing or missing tensor data')
    return prefix + raw_header, tensors


def base_shards(root: Path):
    index_path = root / 'model.safetensors.index.json'
    weight_map = None
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map or any(not isinstance(v, str) for v in weight_map.values()):
            raise ValueError('Invalid Safetensors shard index')
        names = sorted(set(weight_map.values()))
    else:
        names = [p.name for p in sorted(root.glob('*.safetensors')) if p.name != 'adapter_model.safetensors']
    if not names:
        raise ValueError('No base Safetensors weights found')
    shards, all_tensors = {}, {}
    for name in names:
        if not name.endswith('.safetensors'):
            raise ValueError('Base index must reference Safetensors files only')
        prefix, tensors = read_header(member(root, name))
        for key in tensors:
            if key in all_tensors or (weight_map is not None and weight_map.get(key) != name):
                raise ValueError(f'Duplicate tensor or shard-index mismatch: {key}')
        shards[name] = (prefix, tensors)
        all_tensors.update(tensors)
    if weight_map is not None and set(weight_map) != set(all_tensors):
        raise ValueError('Shard index does not match actual tensor names')
    return shards, all_tensors


def adapter_pairs(root: Path):
    config = read_json(member(root, 'adapter_config.json'))
    if config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM':
        raise ValueError('Only vanilla causal-LM LoRA adapters are supported')
    for key in ('use_dora', 'use_rslora', 'fan_in_fan_out', 'use_qalora', 'use_bdlora'):
        if config.get(key):
            raise ValueError(f'Unsupported LoRA variant: {key}')
    if config.get('bias', 'none') != 'none' or config.get('modules_to_save'):
        raise ValueError('Adapter includes base parameters or bias updates')
    if config.get('rank_pattern') or config.get('alpha_pattern') or config.get('target_parameters'):
        raise ValueError('Per-parameter/per-module rank or alpha overrides are unsupported')
    rank, alpha = config.get('r'), config.get('lora_alpha')
    if type(rank) is not int or not 1 <= rank <= 4096 or type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Invalid LoRA rank or alpha')
    _, tensors = read_header(member(root, 'adapter_model.safetensors'))
    pairs = {}
    for key, info in tensors.items():
        if not key.startswith('base_model.model.'):
            raise ValueError(f'Unexpected LoRA tensor prefix: {key}')
        for component in ('A', 'B'):
            suffix = f'.lora_{component}.weight'
            if key.endswith(suffix):
                name = key.removeprefix('base_model.model.').removesuffix(suffix) + '.weight'
                if component in pairs.setdefault(name, {}):
                    raise ValueError(f'Duplicate LoRA component: {name}')
                pairs[name][component] = info
                break
        else:
            raise ValueError(f'Unexpected non-LoRA tensor: {key}')
    if not pairs:
        raise ValueError('Adapter contains no LoRA weights')
    for name, pair in pairs.items():
        if set(pair) != {'A', 'B'} or pair['A'].dtype not in FLOAT_DTYPES or pair['B'].dtype not in FLOAT_DTYPES:
            raise ValueError(f'Missing/non-floating LoRA pair: {name}')
        if len(pair['A'].shape) != 2 or len(pair['B'].shape) != 2 or pair['A'].shape[0] != rank or pair['B'].shape[1] != rank:
            raise ValueError(f'LoRA rank/shape mismatch: {name}')
    return config, pairs, alpha / rank


def tensor_from_bytes(info: TensorInfo, raw=None):
    import torch
    types = {'F16': torch.float16, 'BF16': torch.bfloat16, 'F32': torch.float32, 'F64': torch.float64}
    if info.dtype not in types:
        raise ValueError('LoRA math requires floating base and adapter weights')
    tensor = torch.frombuffer(bytearray(info.read() if raw is None else raw), dtype=types[info.dtype]).reshape(info.shape)
    if not torch.isfinite(tensor).all().item():
        raise ValueError('Non-finite input tensor')
    return tensor


def merge_tensor(base: TensorInfo, pair, scale, raw):
    import torch
    original = tensor_from_bytes(base, raw)
    left = tensor_from_bytes(pair['A']).float()
    right = tensor_from_bytes(pair['B']).float()
    delta = torch.matmul(right, left) * scale
    merged = (original.float() + delta).to(original.dtype).contiguous()
    if not torch.isfinite(merged).all().item():
        raise ValueError('Merged tensor contains non-finite values')
    # uint8 view supports bfloat16, whose direct NumPy representation is absent.
    result = merged.view(torch.uint8).numpy().tobytes()
    if len(result) != len(raw):
        raise ValueError('Merged tensor changed its byte size')
    return result


def merge(base, adapter, output, *, threads=6, base_revision=None):
    import torch
    base, adapter = Path(base).resolve(strict=True), Path(adapter).resolve(strict=True)
    output = Path(output).resolve()
    if output == base or output.is_relative_to(base) or output == adapter or output.is_relative_to(adapter):
        raise ValueError('Output must be separate from both input folders')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output already exists and is not empty')
    read_json(member(base, 'config.json'))
    shards, tensors = base_shards(base)
    config, pairs, scale = adapter_pairs(adapter)
    for name, pair in pairs.items():
        if name not in tensors:
            raise ValueError(f'Adapter target missing from base: {name}')
        info = tensors[name]
        if info.dtype not in FLOAT_DTYPES or len(info.shape) != 2 or info.shape != (pair['B'].shape[0], pair['A'].shape[1]):
            raise ValueError(f'Base/LoRA shape or dtype mismatch: {name}')
    if type(threads) is not int or not 1 <= threads <= 256:
        raise ValueError('Use 1-256 CPU threads')
    torch.set_num_threads(threads)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.' + output.name + '-merge-stage-' + uuid.uuid4().hex)
    stage.mkdir()
    report = {'status': 'MERGED_CANDIDATE_NOT_EVALUATED', 'base_model': config.get('base_model_name_or_path'),
              'base_revision': base_revision, 'rank': config['r'], 'alpha': config['lora_alpha'],
              'adapter_modules': len(pairs), 'merged_tensors': 0, 'unchanged_tensors': 0,
              'vision_adapter_tensors': sum('vision' in name or 'visual' in name for name in pairs),
              'adapter_sha256': hashlib.sha256(member(adapter, 'adapter_model.safetensors').read_bytes()).hexdigest(),
              'shards': {}}
    try:
        for relative, (prefix, items) in shards.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
            source_hash.update(prefix)
            output_hash.update(prefix)
            with next(iter(items.values())).path.open('rb') as source, destination.open('xb') as target:
                source.seek(len(prefix))
                target.write(prefix)
                for name, info in sorted(items.items(), key=lambda item: item[1].start):
                    if name in pairs:
                        raw = source.read(info.end - info.start)
                        if len(raw) != info.end - info.start:
                            raise ValueError('Base weights changed or were truncated during merge')
                        source_hash.update(raw)
                        changed = merge_tensor(info, pairs[name], scale, raw)
                        target.write(changed)
                        output_hash.update(changed)
                        report['merged_tensors'] += 1
                    else:
                        remaining = info.end - info.start
                        while remaining:
                            chunk = source.read(min(4 * 1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError('Base weights changed or were truncated during merge')
                            source_hash.update(chunk)
                            target.write(chunk)
                            output_hash.update(chunk)
                            remaining -= len(chunk)
                        report['unchanged_tensors'] += 1
            report['shards'][relative] = {'base_sha256': source_hash.hexdigest(), 'merged_sha256': output_hash.hexdigest()}
            print(json.dumps({'shard': relative, 'merged_tensors': report['merged_tensors'], 'unchanged_tensors': report['unchanged_tensors']}), flush=True)
        for name in METADATA_FILES:
            if (base / name).is_file():
                shutil.copy2(member(base, name), stage / name)
        (stage / 'merge-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if output.exists():
            # Only the prevalidated empty output may be removed; never recursive.
            output.rmdir()
        stage.rename(output)
    except Exception as error:
        raise RuntimeError(f'Merge failed; incomplete staging files retained at {stage}: {error}') from error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--base-revision')
    args = parser.parse_args()
    result = merge(args.base, args.adapter, args.output, threads=args.threads, base_revision=args.base_revision)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
