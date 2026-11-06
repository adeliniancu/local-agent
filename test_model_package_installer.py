"""Offline integrity and isolation checks for the optional trained-model installer."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import model_package_installer as installer


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture():
    payloads = {"language.001": b"GGUF-language-first-", "language.002": b"language-last", "projector.gguf": b"GGUF-projector"}
    def item(name, parts):
        data = b"".join(payloads[part] for part in parts)
        return {"name": name, "size": len(data), "sha256": sha(data), "parts": [
            {"name": part, "url": installer.RELEASE_PREFIX + part, "size": len(payloads[part]), "sha256": sha(payloads[part])}
            for part in parts]}
    return ({"schema": 1, "model": installer.MODEL, "label": "Experimental trained 4B pilot",
        "files": [item("model.gguf", ["language.001", "language.002"]), item("mmproj.gguf", ["projector.gguf"])],
        "parameters": dict(installer.PARAMETERS), "renderer": "qwen3.5", "parser": "qwen3.5",
        "license_file": "LICENSE.txt", "quality": {"trained_passed": 1, "reference_passed": 2, "tasks": 3},
        "experimental": True}, payloads)


class Response(io.BytesIO):
    def __init__(self, data, length=True):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))} if length else {}


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.app = self.root / "Local Agent"
        for relative in ("app.py", "runtime.py", "runtime/python/python.exe", "runtime/ollama/ollama.exe"):
            path = self.app / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"trusted test placeholder")
        (self.app / "data").mkdir()
        self.original = self.app / "data/settings.json"
        self.original.write_text('{"model":"original:4b"}', encoding="utf-8")
        self.chat = self.app / "data/chat.json"
        self.chat.write_text('{"messages":["private chat"]}', encoding="utf-8")
        self.license = self.root / "LICENSE.txt"
        self.license.write_text("Apache License, Version 2.0\nCopyright original model authors\n", encoding="utf-8")
        self.manifest, self.payloads = fixture()
        self.cancel = threading.Event()
        self.requests = []
        self.events = []

    def opener(self, url):
        self.requests.append(url)
        return Response(self.payloads[url.removeprefix(installer.RELEASE_PREFIX)])

    def download(self, manifest=None, opener=None):
        folder = self.root / "downloads"
        folder.mkdir(exist_ok=True)
        return installer.download_files(manifest or self.manifest, folder, self.app, self.cancel,
                                        lambda *args: self.events.append(args), opener=opener or self.opener)

    def runtime(self, names=None):
        models = list(names or ["original:4b"])
        return SimpleNamespace(PORT=11435, ensure_server=lambda: None,
            list_models=lambda: list(models), _get=lambda path: {"version": "0.40.15"}, models=models)

    def install(self, runtime, process):
        free = SimpleNamespace(free=installer.MIN_FREE_BYTES + 1)
        with patch.object(installer.shutil, "disk_usage", return_value=free), patch.object(installer.subprocess, "run", side_effect=process):
            installer.install_model(self.app, self.manifest, self.license, self.cancel,
                lambda *args: self.events.append(args), runtime_module=runtime, opener=self.opener)

    def test_manifest_rejects_untrusted_addresses_and_traversal(self):
        self.assertIs(installer.validate_manifest(self.manifest), self.manifest)
        for field, value in (("name", "../escape"), ("name", "folder\\escape"),
                ("url", "https://evil.example/model"),
                ("url", installer.RELEASE_PREFIX + "language.001?token=x"),
                ("url", "https://github.com@evil.example/language.001")):
            bad = deepcopy(self.manifest)
            bad["files"][0]["parts"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(installer.InstallError):
                installer.validate_manifest(bad)

    def test_manifest_rejects_changed_commands_parameters_and_sizes(self):
        for field, value in (("model", "other:4b"), ("renderer", "other"), ("schema", True),
                ("experimental", False), ("license_file", "../LICENSE"),
                ("parameters", {**installer.PARAMETERS, "temperature": True}),
                ("quality", {"trained_passed": True, "reference_passed": 2, "tasks": 3})):
            bad = deepcopy(self.manifest)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(installer.InstallError):
                installer.validate_manifest(bad)
        for size in (-1, True, 2 * 1024**3):
            bad = deepcopy(self.manifest)
            bad["files"][0]["parts"][0]["size"] = size
            with self.subTest(size=size), self.assertRaises(installer.InstallError):
                installer.validate_manifest(bad)

    def test_json_duplicate_fields_and_overlarge_input_rejected(self):
        path = self.root / "manifest.json"
        for raw in ('{"schema":1,"schema":1}', " " * 65537):
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(installer.InstallError):
                installer.load_manifest(path)

    def test_split_assembly_hashes_and_progress(self):
        result = self.download()
        self.assertEqual(result["model.gguf"].read_bytes(), b"GGUF-language-first-language-last")
        self.assertEqual(result["mmproj.gguf"].read_bytes(), b"GGUF-projector")
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.events[-1], (sum(item["size"] for item in self.manifest["files"]),) * 2)

    def test_verified_cache_reused_but_corrupt_cache_redownloaded(self):
        blobs = self.app / "models/blobs"
        blobs.mkdir(parents=True)
        for index, item in enumerate(self.manifest["files"]):
            content = b"".join(self.payloads[part["name"]] for part in item["parts"])
            (blobs / ("sha256-" + item["sha256"])).write_bytes(content if index else b"X" * len(content))
        result = self.download()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(result["mmproj.gguf"].parent, blobs)
        self.assertEqual(result["model.gguf"].parent, self.root / "downloads")

    def test_corrupt_part_and_full_hash_rejected(self):
        corrupt = lambda url: Response(b"X" * len(self.payloads[url.removeprefix(installer.RELEASE_PREFIX)]))
        with self.assertRaisesRegex(installer.InstallError, "checksum"):
            self.download(opener=corrupt)
        (self.root / "downloads/model.gguf").unlink()
        self.manifest["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(installer.InstallError, "Assembled"):
            self.download()

    def test_stream_is_bounded_even_without_content_length(self):
        oversize = lambda url: Response(b"X" * 1024, length=False)
        with self.assertRaisesRegex(installer.InstallError, "exceeded"):
            self.download(opener=oversize)
        self.assertEqual((self.root / "downloads/model.gguf").stat().st_size, 0)

    def test_cancel_before_or_during_download(self):
        self.cancel.set()
        with self.assertRaises(installer.Cancelled):
            self.download()
        self.assertEqual(self.requests, [])
        self.cancel.clear()
        def cancel_opener(url):
            self.cancel.set()
            return self.opener(url)
        with self.assertRaises(installer.Cancelled):
            self.download(opener=cancel_opener)

    def test_existing_model_never_overwritten(self):
        runtime = self.runtime([installer.MODEL])
        with patch.object(installer.subprocess, "run") as process, self.assertRaisesRegex(installer.InstallError, "already installed"):
            installer.install_model(self.app, self.manifest, self.license, self.cancel, lambda *args: None,
                                    runtime_module=runtime, opener=self.opener)
        process.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_registration_uses_only_verified_files_and_preserves_user_data(self):
        runtime = self.runtime()
        original, chat = self.original.read_bytes(), self.chat.read_bytes()
        def register(command, **kwargs):
            self.assertEqual(command[1:4], ["create", installer.MODEL, "-f"])
            self.assertNotIn("shell", kwargs)
            self.assertEqual(kwargs["env"]["OLLAMA_MODELS"], str(self.app / "models"))
            self.assertEqual(kwargs["env"]["OLLAMA_NO_CLOUD"], "1")
            text = Path(command[-1]).read_text(encoding="utf-8")
            self.assertEqual(text.count("FROM "), 2)
            self.assertIn("RENDERER qwen3.5\nPARSER qwen3.5", text)
            self.assertIn("LICENSE \"\"\"Apache License", text)
            runtime.models.append(installer.MODEL)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        self.install(runtime, register)
        self.assertEqual(self.events[-1][0], "success")
        self.assertEqual(self.original.read_bytes(), original)
        self.assertEqual(self.chat.read_bytes(), chat)
        self.assertFalse(list((self.app / "data").glob("trained-model-*")))

    def test_failed_checksum_or_registration_cleans_temporary_downloads(self):
        runtime = self.runtime()
        self.manifest["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(installer.InstallError, "checksum"):
            self.install(runtime, lambda *args, **kwargs: self.fail("Unverified import"))
        self.assertFalse(list((self.app / "data").glob("trained-model-*")))
        self.manifest, self.payloads = fixture()
        with self.assertRaisesRegex(installer.InstallError, "registration failed"):
            self.install(runtime, lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failed"))
        self.assertEqual(runtime.models, ["original:4b"])
        self.assertFalse(list((self.app / "data").glob("trained-model-*")))

    def test_old_engine_refused_before_download(self):
        runtime = self.runtime()
        runtime._get = lambda path: {"version": "0.20.0"}
        with self.assertRaisesRegex(installer.InstallError, "Update Local Agent"):
            self.install(runtime, lambda *args, **kwargs: self.fail("Old-engine import"))
        self.assertEqual(self.requests, [])

    def test_temporary_engine_owns_separate_port_and_restores_runtime_after_error(self):
        runtime = self.runtime()
        runtime.BASE_URL = "http://127.0.0.1:11435"
        runtime._get = lambda path, timeout: {"version": "0.40.15"}
        process = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda timeout: None)
        with patch.object(installer.subprocess, "Popen", return_value=process) as spawn, \
                patch.object(process, "terminate") as terminate:
            with self.assertRaisesRegex(installer.InstallError, "test failure"):
                with installer.owned_engine(self.app, runtime, self.cancel):
                    self.assertNotEqual(runtime.PORT, 11435)
                    self.assertEqual(runtime.BASE_URL, f"http://127.0.0.1:{runtime.PORT}")
                    self.assertEqual(spawn.call_args.kwargs["env"]["OLLAMA_MODELS"], str(self.app / "models"))
                    self.assertEqual(spawn.call_args.kwargs["env"]["OLLAMA_HOST"], f"127.0.0.1:{runtime.PORT}")
                    raise installer.InstallError("test failure")
            terminate.assert_called_once()
        self.assertEqual(runtime.PORT, 11435)
        self.assertEqual(runtime.BASE_URL, "http://127.0.0.1:11435")

    def test_untrusted_redirects_rejected(self):
        handler = installer.TrustedRedirects()
        for url in ("http://github.com/model", "https://evil.example/model",
                    "https://github.com:8443/model", "https://user@github.com/model"):
            with self.subTest(url=url), self.assertRaises(installer.InstallError):
                handler.redirect_request(None, None, 302, "", {}, url)


if __name__ == "__main__":
    unittest.main()
