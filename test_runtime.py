"""Local model and performance helpers; all network/import operations are mocked."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import runtime

MIB = 1048576
QWEN_INFO = {"block_count": 32, "size_bytes": 3324173934,
             "context_length": 262144, "kv_bytes_per_token": 32768}


def hardware(free=1952, total=8151):
    return {"gpus": [{"name": "Detected GPU", "total_mb": total, "free_mb": free,
                       "used_mb": total - free}], "cpu": "Test CPU"}


class AllocationTests(unittest.TestCase):
    def test_all_gpu_profiles_use_engine_dynamic_allocation_for_any_model(self):
        for profile in ('auto', 'rtx5060', 'rtx4070_laptop', 'rtx5080'):
            for model, info in ((runtime.MODEL, QWEN_INFO), ('custom:20b', {**QWEN_INFO, 'size_bytes':20*1024**3}), ('custom:8k', {**QWEN_INFO, 'context_length':8192})):
                with self.subTest(profile=profile, model=model), patch.dict(os.environ, {}, clear=True), patch.object(runtime, 'get_model_info', return_value=info), patch.object(runtime, 'get_hardware_info', return_value=hardware(800)):
                    result=runtime.get_inference_options(model, profile)
                    self.assertEqual(-1,result['num_gpu'])
                    self.assertEqual(min(16384,info['context_length']),result['num_ctx'])

    def test_no_detected_gpu_uses_cpu(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runtime, 'get_model_info', return_value=QWEN_INFO), patch.object(runtime,'get_hardware_info',return_value={'gpus':[]}):
            self.assertEqual({'num_ctx':16384,'num_gpu':0},runtime.get_inference_options(runtime.MODEL))

    def test_cpu_and_launcher_override_skip_gpu_probe_but_validate_model(self):
        with patch.dict(os.environ, {"LOCAL_AGENT_GPU_LAYERS": "26"}), \
                patch.object(runtime, "get_model_info", return_value=QWEN_INFO) as metadata, \
                patch.object(runtime, "get_hardware_info") as get:
            self.assertEqual(0, runtime.get_inference_options(runtime.MODEL, "cpu")["num_gpu"])
            self.assertEqual(26, runtime.get_inference_options(runtime.MODEL, "auto")["num_gpu"])
            get.assert_not_called()
            self.assertEqual(2, metadata.call_count)
        with patch.dict(os.environ, {"LOCAL_AGENT_GPU_LAYERS": "0"}), \
                patch.object(runtime, "get_model_info", return_value=QWEN_INFO):
            self.assertEqual(0, runtime.get_inference_options(runtime.MODEL, "rtx5080")["num_gpu"])

    def test_too_small_context_rejected_for_every_profile(self):
        info = {**QWEN_INFO, "context_length": 4096}
        with patch.object(runtime, "get_model_info", return_value=info):
            for profile in runtime.PERFORMANCE_PROFILES:
                with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "at least 8K"):
                    runtime.get_inference_options("small-model", profile)

    def test_8k_custom_model_context_applies_to_cpu_and_environment_override(self):
        info = {**QWEN_INFO, "context_length": 8192}
        with patch.object(runtime, "get_model_info", return_value=info), \
                patch.dict(os.environ, {"LOCAL_AGENT_GPU_LAYERS": "12"}):
            self.assertEqual({"num_ctx": 8192, "num_gpu": 0}, runtime.get_inference_options("custom", "cpu"))
            self.assertEqual({"num_ctx": 8192, "num_gpu": 12}, runtime.get_inference_options("custom", "auto"))

    def test_cpu_cannot_bypass_remote_model_rejection(self):
        with patch.object(runtime, "get_model_info", side_effect=ValueError("remote service")):
            with self.assertRaisesRegex(ValueError, "remote service"):
                runtime.get_inference_options("remote-alias", "cpu")

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            runtime.get_inference_options(runtime.MODEL, "imaginary GPU")

    def test_dynamic_allocation_does_not_poll_resident_model_memory(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runtime,'get_model_info',return_value=QWEN_INFO), patch.object(runtime,'get_hardware_info',return_value=hardware()), patch.object(runtime,'_get') as request:
            self.assertEqual(-1,runtime.get_inference_options(runtime.MODEL)['num_gpu'])
            request.assert_not_called()


class ModelMetadataTests(unittest.TestCase):
    def setUp(self):
        runtime._model_info.cache_clear()

    def test_qwen_coding_sampling_requires_reasoning_and_matching_family(self):
        expected = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "presence_penalty": 0.0, "repeat_penalty": 1.0}
        self.assertEqual(runtime.get_generation_options("qwen3.5:4b-q4_K_M", True), expected)
        for model, think in (("qwen3.5:4b-q4_K_M", False), ("qwen2.5-coder:7b", True),
                             ("custom-local:4b", True)):
            self.assertEqual(runtime.get_generation_options(model, think), {"temperature": 0.2})

    def test_model_name_rejects_cloud_shell_and_url_inputs(self):
        for name in ("model:cloud", "model:4b-cloud", "../model", "model\nPARAMETER x 1",
                     "https://host/model", "model; echo bad", "--help"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime._validate_model_name(name)
        self.assertEqual("hf.co/owner/model:Q4_K_M", runtime._validate_model_name("hf.co/owner/model:Q4_K_M"))

    def test_hybrid_model_kv_cache_uses_full_attention_layers(self):
        response = {"model_info": {"general.architecture": "qwen35", "qwen35.block_count": 32,
            "qwen35.context_length": 262144, "qwen35.attention.recurrent_layers": [True, True, True, False] * 8,
            "qwen35.attention.head_count": 16, "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256},
            "capabilities": ["completion", "tools", "thinking"]}
        with patch.object(runtime, "_post", return_value=response), patch.object(runtime, "_get", return_value={
                "models": [{"name": runtime.MODEL, "size": QWEN_INFO["size_bytes"]}]}):
            result = runtime.get_model_info(runtime.MODEL)
        self.assertEqual(32768, result["kv_bytes_per_token"])
        self.assertEqual(QWEN_INFO["size_bytes"], result["size_bytes"])
        self.assertIn("tools", result["capabilities"])
        self.assertTrue(result["capabilities_known"])

    def test_remote_model_metadata_is_rejected_even_without_cloud_suffix(self):
        with patch.object(runtime, "_post", return_value={"remote_host": "https://example.invalid"}):
            with self.assertRaisesRegex(ValueError, "remote service"):
                runtime.get_model_info("remote-alias")

    def test_cloud_download_rejected_before_network(self):
        with patch.object(runtime, "ensure_server") as ensure:
            with self.assertRaises(ValueError):
                runtime.pull_model("huge:cloud")
            ensure.assert_not_called()

    def test_hardware_detection_parses_actual_vram(self):
        runtime._hardware_cache = (0.0, None)
        completed = Mock(returncode=0, stdout="NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 2400, 5788\n")
        with patch.object(runtime.shutil, "which", return_value="nvidia-smi"), \
                patch.object(runtime.subprocess, "run", return_value=completed) as run:
            info = runtime.get_hardware_info()
        self.assertEqual(8188, info["gpus"][0]["total_mb"])
        self.assertEqual(2400, info["gpus"][0]["free_mb"])
        self.assertIn("memory.free", run.call_args.args[0][1])
        runtime._hardware_cache = (0.0, None)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="local-model-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "my local weights.gguf"
        self.source.write_bytes(b"GGUF" + b"\x00" * 32)

    def test_import_uses_local_cli_and_quoted_modelfile_without_shell(self):
        captured = {}
        def create(args, **kwargs):
            captured.update(args=args, kwargs=kwargs, modelfile=Path(args[-1]).read_text(encoding="utf-8"))
            return Mock(stdout=io.StringIO("creating model\nsuccess\n"), wait=Mock(return_value=0), poll=Mock(return_value=0))
        progress = []
        with patch.object(runtime, "ROOT", self.root), patch.object(runtime, "ensure_server"), \
                patch.object(runtime, "list_models", return_value=[]), \
                patch.object(runtime, "get_model_info", return_value={"capabilities": ["tools"]}), \
                patch.object(runtime.subprocess, "Popen", side_effect=create):
            runtime.import_gguf("my-model", self.source, progress=progress.append)
        self.assertEqual(["create", "my-model", "--file"], captured["args"][1:4])
        self.assertEqual(f'FROM "{self.source.as_posix()}"\n', captured["modelfile"])
        self.assertNotIn("shell", captured["kwargs"])
        self.assertEqual("1", captured["kwargs"]["env"]["OLLAMA_NO_CLOUD"])
        self.assertEqual(f"127.0.0.1:{runtime.PORT}", captured["kwargs"]["env"]["OLLAMA_HOST"])
        self.assertEqual({"status": "success"}, progress[-1])
        self.assertFalse(Path(captured["args"][-1]).exists())

    def test_import_does_not_overwrite_existing_alias(self):
        with patch.object(runtime, "ensure_server"), patch.object(runtime, "list_models", return_value=["my-model:latest"]), \
                patch.object(runtime.subprocess, "Popen") as create:
            with self.assertRaisesRegex(ValueError, "already exists"):
                runtime.import_gguf("my-model", self.source)
            create.assert_not_called()

    def test_bad_file_rejected_before_server_or_cli(self):
        self.source.write_bytes(b"not weights")
        with patch.object(runtime, "ensure_server") as ensure, patch.object(runtime.subprocess, "Popen") as create:
            with self.assertRaisesRegex(ValueError, "GGUF header"):
                runtime.import_gguf("my-model", self.source)
            ensure.assert_not_called()
            create.assert_not_called()

    def test_cli_failure_is_reported_and_temporary_modelfile_removed(self):
        with patch.object(runtime, "ROOT", self.root), patch.object(runtime, "ensure_server"), \
                patch.object(runtime, "list_models", return_value=[]), \
                patch.object(runtime.subprocess, "Popen", return_value=Mock(
                    stdout=io.StringIO("unsupported architecture\n"), wait=Mock(return_value=1), poll=Mock(return_value=1))):
            with self.assertRaisesRegex(RuntimeError, "unsupported architecture"):
                runtime.import_gguf("my-model", self.source)
        self.assertEqual([], list((self.root / "data").iterdir()))


if __name__ == "__main__":
    unittest.main()
