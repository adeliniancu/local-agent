"""Focused update-package, rollback and trust-boundary checks; no network/model."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import warnings
import zipfile

import build_update
import update_worker as worker
import updater


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="LocalAgent-update-test-")
        # Windows runners can expose TEMP through an 8.3 alias. Match the
        # installer's resolved paths so fault injection targets the same file.
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        for name in worker.REQUIRED_FILES:
            (self.source / name).write_text(f"# new {name}\n", encoding="utf-8")
        (self.source / "version.py").write_text("APP_ID = 'local-agent-windows'\n__version__ = '2.0.0'\nGITHUB_REPOSITORY = ''\nUPDATE_FEED_URL = ''\n", encoding="utf-8")
        (self.source / "assets").mkdir()
        (self.source / "assets/app.ico").write_bytes(b"test icon")
        self.output = self.root / "dist"
        self.result = build_update.build_update(self.source, self.output, "example/local-agent", "v2.0.0")
        self.archive = Path(self.result["archive"])
        self.target = self.root / "installed"
        self.target.mkdir()
        for name in ("app.py", "bootstrap.py", "agent_core.py", "runtime.py"):
            (self.target / name).write_text(f"# original {name}\n", encoding="utf-8")
        for name in ("runtime/python/pythonw.exe", "runtime/ollama/ollama.exe"):
            path = self.target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"runtime must stay unchanged")
        for name in ("data/last-chat.json", "workspace/precious.py", "models/blobs/model"):
            path = self.target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"personal file must stay unchanged")

    def tearDown(self):
        self.temporary.cleanup()

    def unpack(self):
        source = self.root / "extracted"
        source.mkdir()
        # The fixture was made by our builder and validated before this extraction.
        worker.inspect_archive(self.archive, "1.0.0")
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(source)
        return source

    def rewrite_archive(self, change):
        with zipfile.ZipFile(self.archive) as archive:
            contents = {info.filename: archive.read(info) for info in archive.infolist()}
        change(contents)
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                archive.writestr(name, data)

    def test_build_roundtrip_and_feed(self):
        manifest, payload = worker.inspect_archive(self.archive, "1.0.0")
        self.assertEqual(manifest["version"], "2.0.0")
        self.assertIn(b"example/local-agent", payload["version.py"])
        self.assertNotIn(b"example/local-agent", (self.source / "version.py").read_bytes())
        feed = updater.validate_feed(json.loads(Path(self.result["feed"]).read_text()), "1.0.0")
        self.assertEqual(feed["sha256"], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        self.assertEqual(feed["size"], self.archive.stat().st_size)
        self.assertEqual(feed["download_url"], "https://github.com/example/local-agent/releases/download/v2.0.0/Local-Agent-update-2.0.0.zip")

    def test_installs_app_and_preserves_personal_files_and_runtime(self):
        source = self.unpack()
        original = {path.relative_to(self.target).as_posix(): path.read_bytes()
                    for path in self.target.rglob("*") if path.is_file()}
        backup = worker.apply_update(source, self.target)
        self.assertEqual(worker.installed_version(self.target), "2.0.0")
        self.assertEqual((self.target / "app.py").read_bytes(), (self.source / "app.py").read_bytes())
        self.assertEqual((backup / "files/app.py").read_bytes(), original["app.py"])
        for name, data in original.items():
            if name.startswith(("runtime/", "data/", "workspace/", "models/")):
                self.assertEqual((self.target / name).read_bytes(), data)
        self.assertEqual(json.loads((backup / "result.json").read_text())["status"], "installed")

    def test_failed_replace_rolls_back_existing_and_new_files(self):
        source = self.unpack()
        old_app = (self.target / "app.py").read_bytes()
        old_core = (self.target / "agent_core.py").read_bytes()
        def failing_replace(source, destination):
            if Path(destination) == self.target / "runtime.py":
                raise PermissionError("simulated locked file")
            return os.replace(source, destination)
        with self.assertRaisesRegex(worker.UpdateError, "Previous application files were restored"):
            worker.apply_update(source, self.target, replace=failing_replace)
        self.assertEqual((self.target / "app.py").read_bytes(), old_app)
        self.assertEqual((self.target / "agent_core.py").read_bytes(), old_core)
        self.assertFalse((self.target / "assets/app.ico").exists())
        self.assertFalse((self.target / "version.py").exists())
        self.assertEqual(worker.installed_version(self.target), "1.0.0")
        record = next((self.target / "data/updates").glob("*/result.json"))
        self.assertEqual(json.loads(record.read_text())["status"], "rolled_back")

    def test_rejects_path_traversal_and_user_data_paths(self):
        manifest, _ = worker.inspect_archive(self.archive)
        for path in ("../escape.py", "data/last-chat.json", "runtime/python/python.exe", "C:/escape.py", "assets/../app.py"):
            with self.subTest(path=path):
                modified = json.loads(json.dumps(manifest))
                modified["files"][0]["path"] = path
                with self.assertRaises(worker.UpdateError):
                    worker.validate_manifest(modified, "1.0.0")

    def test_rejects_duplicate_zip_entries(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "a") as archive:
                archive.writestr("manifest.json", "{}")
        with self.assertRaisesRegex(worker.UpdateError, "duplicate"):
            worker.inspect_archive(self.archive)

    def test_rejects_case_collisions_and_extra_files(self):
        self.rewrite_archive(lambda contents: contents.update({"PAYLOAD/APP.PY": b"x"}))
        with self.assertRaises(worker.UpdateError):
            worker.inspect_archive(self.archive)

    def test_rejects_symlink_zip_entries(self):
        info = zipfile.ZipInfo("payload/app.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive) as archive:
            contents = {entry.filename: archive.read(entry) for entry in archive.infolist()}
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in contents.items():
                archive.writestr(info if name == "payload/app.py" else name, data)
        with self.assertRaisesRegex(worker.UpdateError, "regular files"):
            worker.inspect_archive(self.archive)

    def test_rejects_oversized_archive_member_before_reading(self):
        with patch.object(worker, "MAX_FILE_BYTES", 32):
            with self.assertRaisesRegex(worker.UpdateError, "oversized"):
                worker.inspect_archive(self.archive)

    def test_rejects_bad_checksum_and_manifest_version_mismatch(self):
        self.rewrite_archive(lambda contents: contents.update({"payload/app.py": b"changed"}))
        with self.assertRaises(worker.UpdateError):
            worker.inspect_archive(self.archive)
        # Independently verify a self-consistent checksum cannot conceal a version mismatch.
        manifest, payload = worker.inspect_folder(self.unpack_rebuilt())
        payload["version.py"] = b"APP_ID='local-agent-windows'\n__version__='9.0.0'\n"
        entry = next(item for item in manifest["files"] if item["path"] == "version.py")
        entry.update(size=len(payload["version.py"]), sha256=hashlib.sha256(payload["version.py"]).hexdigest())
        with self.assertRaisesRegex(worker.UpdateError, "version does not match"):
            worker._verify_payload(manifest, payload)

    def unpack_rebuilt(self):
        build_update.build_update(self.source, self.output, "example/local-agent", "v2.0.0")
        return self.unpack()

    def test_rejects_downgrade_and_unsupported_source_version(self):
        for version in ("2.0.0", "3.0.0", "0.9.0"):
            with self.subTest(version=version):
                with self.assertRaises(worker.UpdateError):
                    worker.inspect_archive(self.archive, version)

    def test_duplicate_json_fields_are_rejected(self):
        with self.assertRaisesRegex(worker.UpdateError, "Duplicate JSON"):
            worker.load_json(b'{"version":"2.0.0","version":"3.0.0"}')

    def test_version_file_is_parsed_without_running_code(self):
        value = worker.version_constants(b"raise RuntimeError('must not execute')\n__version__='2.0.0'\n")
        self.assertEqual(value["__version__"], "2.0.0")

    def test_non_https_feed_and_redirect_are_rejected(self):
        for url in ("http://example.com/feed.json", "https://user:password@example.com/feed.json", "file:///feed.json"):
            with self.subTest(url=url):
                with self.assertRaises(worker.UpdateError):
                    updater._https_url(url)
        request = urllib.request.Request("https://example.com/feed.json")
        with self.assertRaises(worker.UpdateError):
            updater._HTTPSRedirect().redirect_request(request, None, 302, "Found", {}, "http://example.com/feed.json")

    def test_download_checks_hash_before_accepting_package(self):
        feed = json.loads(Path(self.result["feed"]).read_text())
        folder = self.root / "download"
        folder.mkdir()
        response = io.BytesIO(self.archive.read_bytes())
        response.headers = {"Content-Length": str(feed["size"])}
        import threading
        with patch.object(updater.version, "__version__", "1.0.0"), \
             patch.object(updater, "_open", return_value=response), \
             patch.object(updater.tempfile, "mkdtemp", return_value=str(folder)):
            result = updater.download_update(feed, threading.Event())
        self.assertEqual(result.read_bytes(), self.archive.read_bytes())
        result.unlink()
        feed["sha256"] = "0" * 64
        response = io.BytesIO(self.archive.read_bytes())
        response.headers = {}
        with patch.object(updater.version, "__version__", "1.0.0"), \
             patch.object(updater, "_open", return_value=response), \
             patch.object(updater.tempfile, "mkdtemp", return_value=str(folder)):
            with self.assertRaisesRegex(worker.UpdateError, "damaged"):
                updater.download_update(feed, threading.Event())
        self.assertFalse((folder / "update.zip").exists())

    def test_destination_symlink_is_rejected(self):
        outside = self.root / "outside.py"
        outside.write_bytes(b"do not change")
        destination = self.target / "app.py"
        destination.unlink()
        try:
            destination.symlink_to(outside)
        except OSError:
            self.skipTest("Creating Windows symlinks requires a privilege unavailable here")
        with self.assertRaises(worker.UpdateError):
            worker.apply_update(self.unpack(), self.target)
        self.assertEqual(outside.read_bytes(), b"do not change")

    @unittest.skipUnless(os.name == "nt", "Windows junction integration")
    def test_destination_junction_cannot_redirect_update_into_user_files(self):
        outside = self.root / "private"
        outside.mkdir()
        protected = outside / "app.ico"
        protected.write_bytes(b"private file")
        junction = self.target / "assets"
        result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
                                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            self.skipTest("Junction creation unavailable")
        try:
            with self.assertRaises(worker.UpdateError):
                worker.apply_update(self.unpack(), self.target)
            self.assertEqual(protected.read_bytes(), b"private file")
            self.assertEqual(worker.installed_version(self.target), "1.0.0")
        finally:
            os.rmdir(junction)

    def test_other_running_app_instance_blocks_update_after_launcher_exits(self):
        with patch.object(worker, "wait_for_pid") as wait, \
             patch.object(worker, "running_app_pids", return_value=[9999]):
            with self.assertRaisesRegex(worker.UpdateError, "still open"):
                worker.wait_until_closed(self.target, [1234])
        wait.assert_called_once_with(1234)


if __name__ == "__main__":
    unittest.main()
