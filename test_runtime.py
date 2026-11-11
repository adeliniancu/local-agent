"""Local model and performance helpers; all network/import operations are mocked."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
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

    def test_gpu_allocation_does_not_depend_on_nvidia_hardware_probe(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(runtime, "get_model_info", return_value=QWEN_INFO), \
                patch.object(runtime, "get_hardware_info", return_value={"gpus": []}) as probe:
            self.assertEqual({"num_ctx": 16384, "num_gpu": -1}, runtime.get_inference_options(runtime.MODEL))
            probe.assert_not_called()

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


class EngineTests(unittest.TestCase):
    def test_engine_starts_with_multi_gpu_scheduling_and_cloud_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ollama = root / "ollama.exe"
            ollama.touch()
            with patch.object(runtime, "ROOT", root), patch.object(runtime, "OLLAMA", ollama), \
                    patch.object(runtime, "_get", side_effect=[OSError(), {"version": "test"}]), \
                    patch.object(runtime.subprocess, "Popen", return_value=Mock(pid=123)) as start:
                self.assertEqual(runtime.BASE_URL, runtime.ensure_server())
            env = start.call_args.kwargs["env"]
            self.assertEqual("1", env["OLLAMA_SCHED_SPREAD"])
            self.assertEqual("1", env["OLLAMA_NO_CLOUD"])


@unittest.skipUnless(os.name == "nt", "Windows engine migration")
class EngineMigrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.marker = self.root / 'data/engine.pid'
        self.marker.parent.mkdir()
        self.marker.write_text('1234')
        self.module = self.root / 'runtime.py'
        self.module.write_text('# new runtime')
        self.executable = self.root / 'runtime/ollama/ollama.exe'
        self.executable.parent.mkdir(parents=True)
        self.executable.write_bytes(b'fixture')
        self.stamp = 1_700_000_000_000_000_000
        os.utime(self.marker, ns=(self.stamp, self.stamp))
        os.utime(self.module, ns=(self.stamp + 1_000_000_000,) * 2)
        self.created_ns = self.stamp - 1_000_000_000
        self.image_path = self.executable
        self.kernel = Mock()
        self.kernel.OpenProcess.return_value = 77
        self.kernel.WaitForSingleObject.side_effect = [258, 0]
        def query(_handle, _flags, buffer, _size):
            buffer.value = str(self.image_path)
            return True
        def process_times(_handle, created, *_unused):
            ticks = self.created_ns // 100 + 116444736000000000
            created._obj.dwLowDateTime = ticks & 0xFFFFFFFF
            created._obj.dwHighDateTime = ticks >> 32
            return True
        self.kernel.QueryFullProcessImageNameW.side_effect = query
        self.kernel.GetProcessTimes.side_effect = process_times
        self.kernel.TerminateProcess.return_value = True
        self.enterContext(patch.object(runtime, 'ROOT', self.root))
        self.enterContext(patch.object(runtime, 'OLLAMA', self.executable))
        self.enterContext(patch.object(runtime, '__file__', str(self.module)))
        self.factory = self.enterContext(patch.object(runtime, '_engine_process_api', return_value=self.kernel))
        self.clients = self.enterContext(patch.object(runtime, '_engine_has_clients', return_value=False))

    def test_owned_process_uses_same_verified_handle_and_next_start_is_not_restarted(self):
        with patch.object(runtime, '_get', return_value={}), \
                patch.object(runtime.subprocess, 'Popen', return_value=Mock(pid=777)) as spawn:
            self.assertEqual(runtime.ensure_server(), runtime.BASE_URL)
            self.assertEqual(runtime.ensure_server(), runtime.BASE_URL)
        spawn.assert_called_once()
        self.assertEqual(spawn.call_args.kwargs['env']['OLLAMA_SCHED_SPREAD'], '1')
        self.kernel.OpenProcess.assert_called_once_with(0x00101001, False, 1234)
        self.kernel.TerminateProcess.assert_called_once_with(77, 0)
        self.kernel.CloseHandle.assert_called_once_with(77)
        self.assertEqual(self.kernel.QueryFullProcessImageNameW.call_args.args[0], 77)
        self.assertEqual(self.kernel.GetProcessTimes.call_args.args[0], 77)
        self.assertEqual(self.marker.read_text(), '777')

    def test_missing_invalid_or_new_marker_never_opens_process(self):
        for content in ('', '-1', '4294967296', '1' * 65, 'not a pid', str(os.getpid())):
            with self.subTest(content=content):
                self.marker.write_text(content)
                os.utime(self.marker, ns=(self.stamp,) * 2)
                self.assertFalse(runtime._restart_outdated_owned_engine())
        self.marker.write_text('1234')
        self.assertFalse(runtime._restart_outdated_owned_engine())
        self.marker.unlink()
        self.assertFalse(runtime._restart_outdated_owned_engine())
        self.factory.assert_not_called()

    def test_foreign_executable_never_terminated(self):
        self.image_path = self.root.parent / 'other-install/ollama.exe'
        self.assertFalse(runtime._restart_outdated_owned_engine())
        self.kernel.TerminateProcess.assert_not_called()
        self.clients.assert_not_called()
        self.kernel.CloseHandle.assert_called_once_with(77)

    def test_reused_pid_for_same_executable_never_terminated(self):
        self.created_ns = self.stamp + 100
        self.assertFalse(runtime._restart_outdated_owned_engine())
        self.kernel.TerminateProcess.assert_not_called()
        self.clients.assert_not_called()
        self.kernel.CloseHandle.assert_called_once_with(77)

    def test_active_request_leaves_existing_engine_and_marker_untouched(self):
        self.clients.return_value = True
        original = self.marker.read_bytes(), self.marker.stat().st_mtime_ns
        with patch.object(runtime, '_get', return_value={}), patch.object(runtime.subprocess, 'Popen') as spawn:
            self.assertEqual(runtime.ensure_server(), runtime.BASE_URL)
        spawn.assert_not_called()
        self.kernel.TerminateProcess.assert_not_called()
        self.kernel.CloseHandle.assert_called_once_with(77)
        self.assertEqual((self.marker.read_bytes(), self.marker.stat().st_mtime_ns), original)

    def test_inaccessible_or_exited_process_does_not_block_reuse(self):
        self.kernel.OpenProcess.return_value = None
        self.assertFalse(runtime._restart_outdated_owned_engine())
        self.kernel.CloseHandle.assert_not_called()
        self.kernel.OpenProcess.return_value = 77
        self.kernel.WaitForSingleObject.side_effect = [0]
        self.assertFalse(runtime._restart_outdated_owned_engine())
        self.kernel.TerminateProcess.assert_not_called()
        self.kernel.CloseHandle.assert_called_once_with(77)

    def test_identity_or_client_query_failure_never_terminates(self):
        for failure in ('path', 'time', 'clients'):
            with self.subTest(failure=failure):
                self.kernel.WaitForSingleObject.side_effect = [258]
                self.kernel.QueryFullProcessImageNameW.return_value = False
                original_path = self.kernel.QueryFullProcessImageNameW.side_effect
                original_times = self.kernel.GetProcessTimes.side_effect
                if failure == 'path':
                    self.kernel.QueryFullProcessImageNameW.side_effect = None
                elif failure == 'time':
                    self.kernel.GetProcessTimes.side_effect = None
                    self.kernel.GetProcessTimes.return_value = False
                else:
                    self.clients.side_effect = subprocess.TimeoutExpired('probe', 10)
                self.assertFalse(runtime._restart_outdated_owned_engine())
                self.kernel.QueryFullProcessImageNameW.side_effect = original_path
                self.kernel.GetProcessTimes.side_effect = original_times
        self.kernel.TerminateProcess.assert_not_called()
        self.assertEqual(self.kernel.CloseHandle.call_count, 3)

    def test_failed_termination_reuses_server_and_timeout_is_actionable(self):
        self.kernel.TerminateProcess.return_value = False
        self.kernel.WaitForSingleObject.side_effect = [258, 258]
        self.assertFalse(runtime._restart_outdated_owned_engine())
        self.kernel.TerminateProcess.return_value = True
        self.kernel.WaitForSingleObject.side_effect = [258, 258]
        with self.assertRaisesRegex(RuntimeError, 'Try Connect'):
            runtime._restart_outdated_owned_engine()
        self.assertEqual(self.kernel.CloseHandle.call_count, 2)

    def test_failed_replacement_start_preserves_old_marker(self):
        with patch.object(runtime, '_get', side_effect=[{}, OSError('not ready')]), \
                patch.object(runtime.subprocess, 'Popen', return_value=Mock(pid=777, poll=Mock(return_value=1))):
            with self.assertRaisesRegex(RuntimeError, 'could not start'):
                runtime.ensure_server()
        self.assertEqual(self.marker.read_text(), '1234')


class EngineClientProbeTests(unittest.TestCase):
    def test_checks_owned_listener_and_clients_on_the_configured_port(self):
        for count in (0, 1):
            with patch.object(runtime.subprocess, 'run', return_value=Mock(returncode=0,
                    stdout=json.dumps({'listening': True, 'clients': count}))) as run:
                self.assertEqual(runtime._engine_has_clients(1234), bool(count))
        command = run.call_args.args[0][-1]
        self.assertIn('OwningProcess -eq 1234', command)
        self.assertIn(f'LocalPort -eq {runtime.PORT}', command)
        self.assertIn("State -eq 'Established'", command)
        self.assertIn("State -eq 'Listen'", command)
        self.assertIn("LocalAddress -eq '127.0.0.1'", command)
        self.assertIn("ErrorActionPreference = 'Stop'", command)

    def test_probe_errors_or_missing_listener_are_not_mistaken_for_idle(self):
        for code, output in ((1, '{}'), (0, 'null'), (0, 'true'), (0, 'invalid'),
                             (0, '{"listening": false, "clients": 0}'),
                             (0, '{"listening": true, "clients": true}')):
            with self.subTest(code=code, output=output), \
                    patch.object(runtime.subprocess, 'run', return_value=Mock(returncode=code, stdout=output)):
                with self.assertRaises((OSError, ValueError)):
                    runtime._engine_has_clients(1234)


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


class CodingPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory(prefix="coding-v2-test-"))
        self.root = Path(temporary).resolve()
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "settings.json").write_text('{"model": "original:4b"}', encoding="utf-8")
        notices = self.root / "Third Party Notices"
        notices.mkdir()
        license_name = "QWEN3.5-APACHE-2.0-LICENSE.txt"
        bundled_license = (runtime.ROOT / "distribution-notices" / license_name).read_bytes()
        (notices / license_name).write_bytes(bundled_license)
        self.assets = {"language.part01": b"GGUF-language-", "language.part02": b"weights", "projector.gguf": b"GGUF-projector"}
        self.files = (self.file_spec("model.gguf", "language.part01", "language.part02"),
                      self.file_spec("mmproj.gguf", "projector.gguf"))
        self.models = ["original:4b"]
        self.opener = Mock(open=Mock(side_effect=self.response))
        self.enterContext(patch.object(runtime, "ROOT", self.root))
        self.enterContext(patch.object(runtime, "_CODING_V2_FILES", self.files))
        self.total = sum(item[1] for item in self.files)
        self.enterContext(patch.object(runtime, "CODING_V2_DOWNLOAD_BYTES", self.total))
        self.enterContext(patch.object(runtime, "ensure_server"))
        self.enterContext(patch.object(runtime, "list_models", side_effect=lambda: list(self.models)))
        self.enterContext(patch.object(runtime.shutil, "disk_usage", return_value=Mock(free=9 * 1024**3)))
        self.enterContext(patch.object(runtime.urllib.request, "build_opener", return_value=self.opener))
        self.create = self.enterContext(patch.object(runtime.subprocess, "run"))

    def file_spec(self, name, *assets):
        parts = tuple((asset, len(self.assets[asset]), hashlib.sha256(self.assets[asset]).hexdigest()) for asset in assets)
        assembled = b"".join(self.assets[asset] for asset in assets)
        return (name, len(assembled), hashlib.sha256(assembled).hexdigest(), parts)

    def response(self, request, timeout):
        self.assertTrue(request.full_url.startswith(runtime._CODING_V2_RELEASE))
        data = self.assets[request.full_url.removeprefix(runtime._CODING_V2_RELEASE)]
        response = io.BytesIO(data)
        response.headers = {"Content-Length": str(len(data))}
        return response

    def test_verified_parts_register_exact_tag_with_cached_projector(self):
        cached = self.root / "models" / "blobs" / ("sha256-" + self.files[1][2])
        cached.parent.mkdir(parents=True)
        cached.write_bytes(self.assets["projector.gguf"])

        def register(args, **kwargs):
            self.assertEqual(["create", runtime.CODING_V2_MODEL, "--file"], args[1:4])
            self.assertEqual(f"127.0.0.1:{runtime.PORT}", kwargs["env"]["OLLAMA_HOST"])
            self.assertGreater(kwargs["timeout"], 0)
            modelfile = Path(args[-1])
            self.assertEqual((modelfile.parent / "model.gguf").read_bytes(), b"GGUF-language-weights")
            text = modelfile.read_text(encoding="utf-8")
            self.assertEqual(2, text.count('FROM "'))
            self.assertIn(f'FROM "{cached.as_posix()}"', text)
            self.assertIn("RENDERER qwen3.5\nPARSER qwen3.5", text)
            self.assertIn('LICENSE """', text)
            self.models.append(runtime.CODING_V2_MODEL)
            return Mock(stdout="success\n", returncode=0)

        self.create.side_effect = register
        events = []
        runtime.install_coding_v2(progress=events.append)
        self.assertEqual(2, self.opener.open.call_count)
        self.assertIn({"kind": "progress", "done": self.total, "total": self.total}, events)
        self.assertEqual(["importing", "success"], [event["kind"] for event in events[-2:]])
        self.assertEqual(["original:4b", runtime.CODING_V2_MODEL], self.models)
        self.assertEqual('{"model": "original:4b"}', (self.data / "settings.json").read_text(encoding="utf-8"))
        self.assertFalse(list(self.data.glob("coding-v2-*")))
        self.opener.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            runtime.install_coding_v2()
        self.opener.open.assert_not_called()

    def test_bad_part_hash_stops_before_registration_and_cleans_download(self):
        self.assets["language.part01"] = b"GGUF-language?"
        with self.assertRaisesRegex(RuntimeError, "failed verification"):
            runtime.install_coding_v2()
        self.create.assert_not_called()
        self.assertFalse(list(self.data.glob("coding-v2-*")))

    def test_cancel_during_download_stops_before_registration_and_cleans_download(self):
        cancelled = threading.Event()

        def progress(event):
            if event["kind"] == "progress":
                cancelled.set()

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            runtime.install_coding_v2(progress=progress, cancel_event=cancelled)
        self.create.assert_not_called()
        self.assertEqual(1, self.opener.open.call_count)
        self.assertFalse(list(self.data.glob("coding-v2-*")))

    def test_registration_timeout_cleans_verified_download_and_reports_error(self):
        self.create.side_effect = subprocess.TimeoutExpired("ollama create", 900)
        events = []
        with self.assertRaisesRegex(RuntimeError, "registration timed out"):
            runtime.install_coding_v2(progress=events.append)
        self.assertEqual("importing", events[-1]["kind"])
        self.assertFalse(list(self.data.glob("coding-v2-*")))


if __name__ == "__main__":
    unittest.main()
