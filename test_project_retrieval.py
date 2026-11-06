"""Local retrieval returns bounded, current, scoped source references."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

import runtime


class ProjectRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_ranked_excerpts_have_exact_source_offsets_and_refresh_after_edit(self):
        (self.root / 'noise.py').write_text('print("Hello world")\n' * 50)
        source = '# Persist records using atomic JSON replacement.\n' + 'def atomic_save_json():\n    pass\n'
        file = self.root / 'storage.py'
        file.write_text(source, encoding='utf-8')
        result = runtime.retrieve_project_context(self.root, 'atomic JSON records')
        self.assertEqual(result['passages'][0]['path'], 'storage.py')
        for item in result['passages']:
            text = (self.root / item['path']).read_text(encoding='utf-8')
            self.assertEqual(text[item['start']:item['end']], item['content'])
        file.write_text('# rewritten with no relevant words\n')
        self.assertFalse(runtime.retrieve_project_context(self.root, 'atomic JSON records')['passages'])

    def test_scoping_dependencies_binary_files_and_output_byte_limit(self):
        folder = self.root / 'src'
        folder.mkdir()
        (folder / 'code.py').write_text('# atomic JSON 🙂\n' * 500, encoding='utf-8')
        (self.root / 'outside.py').write_text('# atomic JSON outside\n' * 20)
        (folder / 'binary.bin').write_bytes(b'\x00atomic JSON')
        (folder / '.venv').mkdir()
        (folder / '.venv' / 'private.py').write_text('# atomic JSON\n' * 50)
        result = runtime.retrieve_project_context(self.root, 'atomic JSON', path='src', max_bytes=1600)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False).encode()), 1600)
        self.assertTrue(result['passages'])
        self.assertTrue(all(item['path'] == 'src/code.py' for item in result['passages']))
        with self.assertRaises(ValueError):
            runtime.retrieve_project_context(self.root, 'atomic JSON', path='..')

    def test_skipped_binary_files_still_obey_file_scan_limit(self):
        for i in range(205):
            (self.root / f'{i:03}.bin').write_bytes(b'\x00')
        result = runtime.retrieve_project_context(self.root, 'atomic JSON')
        self.assertEqual(result['files_scanned'], 200)
        self.assertTrue(result['limited'])

    def test_cancellation_does_not_return_partial_context(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            runtime.retrieve_project_context(self.root, 'atomic JSON', cancel_event=cancel)

    def test_automatic_retrieval_skips_obvious_credentials_and_keys(self):
        for name in ('.env', '.env.local', 'credentials.json', 'service_account.json', 'private.key'):
            (self.root / name).write_text('atomic JSON fixture-private-marker')
        (self.root / '.AWS').mkdir()
        (self.root / '.AWS' / 'config').write_text('atomic JSON fixture-private-marker')
        (self.root / 'main.py').write_text('# atomic JSON records\n')
        result = runtime.retrieve_project_context(self.root, 'atomic JSON')
        self.assertEqual([item['path'] for item in result['passages']], ['main.py'])
        self.assertNotIn('fixture-private-marker', json.dumps(result))

    def test_link_to_outside_workspace_cannot_supply_context(self):
        with tempfile.TemporaryDirectory() as outside:
            private = Path(outside) / 'private.py'
            private.write_text('# atomic JSON secret')
            try:
                (self.root / 'link.py').symlink_to(private)
            except OSError:
                self.skipTest('Symlink privilege unavailable')
            self.assertFalse(runtime.retrieve_project_context(self.root, 'atomic JSON')['passages'])


if __name__ == '__main__':
    unittest.main()
