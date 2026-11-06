"""Small release-builder checks; no runtime downloads or model inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import build_release


class ReleaseBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.output = Path(self.temp.name) / "output"
        self.source.mkdir()
        for relative in build_release.APP_FILES:
            (self.source / relative).write_text("sample", encoding="utf-8")
        (self.source / "version.py").write_text("__version__ = '2.0.0'\nGITHUB_REPOSITORY = ''\n", encoding="utf-8")
        for name in build_release.ASSETS:
            path = self.source / "assets" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"image")
        for relative in ("runtime/python/python.exe", "runtime/ollama/ollama.exe", "distribution-notices/LICENSE.txt",
                         "models/private-model", "data/private-chat.json", "logs/private.log", "workspace/private.txt"):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    def test_allowlist_omits_models_and_personal_data(self):
        files = build_release.release_files(self.source)
        names = [name for _, name in files]
        self.assertIn("runtime/python/python.exe", names)
        self.assertIn("runtime/ollama/ollama.exe", names)
        self.assertIn("Third Party Notices/LICENSE.txt", names)
        self.assertFalse(any(name.split("/")[0] in {"models", "data", "logs", "workspace"} for name in names))

    def test_package_injects_repo_without_changing_source(self):
        def build_update(source, output, repository, tag):
            (output / "Local-Agent-update-2.0.0.zip").write_bytes(b"update")
            (output / "update-feed.json").write_text(json.dumps({"version": "2.0.0"}))
            return {"version": "2.0.0"}
        with patch.object(build_release, "ensure_runtimes", return_value={"python": "3.15.0"}), \
                patch.object(build_release, "_load_module", return_value=types.SimpleNamespace(build_update=build_update)):
            report = build_release.build_release(self.source, self.output, "example/local-agent", "v2.0.0", True)
        archive = self.output / report["assets"][0]
        with zipfile.ZipFile(archive) as package:
            prefix = "Local-Agent-Windows-x64-2.0.0/"
            self.assertIn("example/local-agent", package.read(prefix + "version.py").decode())
            manifest = json.loads(package.read(prefix + "PACKAGE.json"))
            self.assertFalse(manifest["models_included"])
            self.assertFalse(manifest["personal_data_included"])
        self.assertIn("GITHUB_REPOSITORY = ''", (self.source / "version.py").read_text())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertTrue((self.output / report["assets"][1]).read_text().startswith(digest))

    def test_tag_mismatch_rejected_before_runtime_preparation(self):
        with patch.object(build_release, "ensure_runtimes") as prepare, self.assertRaises(ValueError):
            build_release.build_release(self.source, self.output, "example/local-agent", "v9.0.0", True)
        prepare.assert_not_called()

    def test_skip_download_never_bootstraps_missing_runtime(self):
        (self.source / "runtime" / "python" / "python.exe").unlink()
        with patch.object(build_release, "_download_ollama") as download, self.assertRaises(RuntimeError):
            build_release.ensure_runtimes(self.source, skip_download=True)
        download.assert_not_called()

    def test_unsafe_archive_rejected_before_extraction(self):
        archive = Path(self.temp.name) / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("../outside.txt", b"no")
        target = Path(self.temp.name) / "engine"
        with self.assertRaises(RuntimeError):
            build_release._extract_ollama(archive, target)
        self.assertFalse(target.exists())
        self.assertFalse((target.parent / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
