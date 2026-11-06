"""Regression tests for attachment boundaries, including native Windows paths."""
import io
import os
from pathlib import Path
import stat
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

import runtime


def png(width, height, include_pixels=True):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    result = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
    if include_pixels:
        result += chunk(b"IDAT", zlib.compress((b"\0" + b"\x22\x88\xCC" * width) * height))
    return result + chunk(b"IEND", b"")


class AttachmentSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="attachment-security-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "app"
        self.root.mkdir()
        self.root_patch = patch.object(runtime, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_file_growth_after_stat_is_bounded_before_storage(self):
        source = self.base / "growing.txt"
        source.write_bytes(b"x" * 100)
        original_stat = Path.stat
        def changed_stat(path, *args, **kwargs):
            if path == source:
                return os.stat_result((stat.S_IFREG | 0o600, 0, 0, 1, 0, 0, 1, 0, 0, 0))
            return original_stat(path, *args, **kwargs)
        with patch.object(runtime, "MAX_ATTACHMENT_BYTES", 8), patch.object(Path, "stat", changed_stat):
            with self.assertRaisesRegex(ValueError, "size limit"):
                runtime.prepare_attachment(source)
        self.assertFalse((self.root / "data").exists())

    def test_read_limit_does_not_request_unbounded_stream(self):
        class GrowingStream(io.BytesIO):
            def read(self, size=-1):
                if size < 0 or size > 9:
                    raise AssertionError("Unbounded file read")
                return super().read(size)
        with patch.object(Path, "open", return_value=GrowingStream(b"x" * 100)):
            with self.assertRaisesRegex(ValueError, "size limit"):
                runtime._read_attachment_bytes(Path("unused"), 8)

    @unittest.skipUnless(os.name == "nt", "Windows junction boundary")
    def test_data_junction_rejected_before_creating_external_directory(self):
        outside = self.base / "private"
        outside.mkdir()
        junction = self.root / "data"
        result = subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
                                capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(0, result.returncode, result.stderr)
        try:
            with self.assertRaisesRegex(ValueError, "storage folder"):
                runtime._attachment_dir()
            self.assertFalse((outside / "attachments").exists())
        finally:
            junction.rmdir()  # Remove only this test junction, not its target.

    def test_unicode_xml_entity_declarations_rejected_before_parser(self):
        xml = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY x "expanded">]><x>&x;</x>'
        for encoding in ("utf-16", "utf-16-be", "utf-32", "utf-32-be"):
            path = self.base / (encoding + ".docx")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", xml.encode(encoding))
            with self.subTest(encoding=encoding), patch.object(runtime.ET, "fromstring") as parse:
                with self.assertRaisesRegex(ValueError, "entity declarations"):
                    runtime.prepare_attachment(path)
                parse.assert_not_called()

    def test_pixel_bomb_is_rejected_before_native_decoder(self):
        path = self.base / "huge.png"
        path.write_bytes(png(50000, 50000, include_pixels=False))
        with patch.object(runtime, "_attachment_powershell") as native:
            with self.assertRaisesRegex(ValueError, "too many pixels"):
                runtime.prepare_attachment(path)
            native.assert_not_called()

    def test_raster_headers_reject_invalid_or_unbounded_dimensions(self):
        for payload in (b"not an image", png(0, 2, False), b"GIF89a" + struct.pack("<HH", 65535, 65535)):
            with self.subTest(payload=payload[:12]), self.assertRaises(ValueError):
                runtime._attachment_image_dimensions(payload)
        self.assertEqual((2, 3), runtime._attachment_image_dimensions(png(2, 3)))
        jpeg = b"\xff\xd8\xff\xe0\x00\x04xx\xff\xc0\x00\x0b\x08\x00\x03\x00\x02\x01\x01\x11\x00\xff\xd9"
        self.assertEqual((2, 3), runtime._attachment_image_dimensions(jpeg))

    def test_storage_extension_cannot_escape_cache(self):
        with self.assertRaisesRegex(ValueError, "storage extension"):
            runtime._store_attachment_bytes(b"content", ".txt/../../outside")
        self.assertFalse((self.root / "data").exists())

    @unittest.skipUnless(os.name == "nt", "Native PowerShell payload test")
    def test_powershell_payload_is_data_with_unicode_and_metacharacters(self):
        value = "quoted '\"; $(1+1); Romanian \u0103 and \u00e9\nnext line"
        result = runtime._attachment_powershell("@{value=$p.value; count=$p.files.Count} | ConvertTo-Json -Compress",
                                               {"value": value, "files": ["a", "b"]}, timeout=15)
        self.assertEqual({"value": value, "count": 2}, result)

    @unittest.skipUnless(os.name == "nt", "Native Windows image conversion")
    def test_native_image_is_resized_to_bounded_png_and_thumbnail(self):
        source = self.base / "large image.png"
        source.write_bytes(png(2000, 1000))
        metadata = runtime.prepare_attachment(source)
        image = runtime.resolve_attachment_path({"path": metadata["image"]})
        preview = runtime.resolve_attachment_path(metadata, preview=True)
        self.assertEqual((1280, 640), runtime._attachment_image_dimensions(image.read_bytes()))
        self.assertEqual((256, 128), runtime._attachment_image_dimensions(preview.read_bytes()))
        self.assertEqual("image", metadata["kind"])


if __name__ == "__main__":
    unittest.main()
