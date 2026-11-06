"""Repeated owned patches are idempotent only for the exact resulting revision."""
from pathlib import Path
import tempfile
import threading
import unittest

from agent_core import Agent, BACKUP_DIR


class EditIdempotencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.target = self.root / 'main.py'
        self.target.write_text('answer = 1\n')
        self.agent = Agent(self.root, lambda *_: None, lambda *_: True, threading.Event())
        self.agent._read_file(self.target)

    def apply(self):
        return self.agent._edit_file('main.py', 'answer = 1', 'answer = 2')

    def test_identical_patch_does_not_change_file_or_create_another_backup(self):
        first = self.apply()
        self.assertIn('earlier test results describe the old revision', first['note'])
        before = self.target.read_bytes()
        backups = sorted(str(path) for path in (self.root / BACKUP_DIR).rglob('*'))
        second = self.apply()
        self.assertTrue(second['unchanged'])
        self.assertTrue(second['already_applied'])
        self.assertEqual(second['replacements'], 0)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(sorted(str(path) for path in (self.root / BACKUP_DIR).rglob('*')), backups)
        self.assertIsNone(self.agent._progress_signature('edit_file', {'path':'main.py'}, second))

    def test_external_edit_requires_fresh_read_and_different_revision_is_not_already_applied(self):
        self.apply()
        self.target.write_text('answer = 2\n# external change\n')
        with self.assertRaisesRegex(ValueError, 'changed since your last read'):
            self.apply()
        self.agent._read_file(self.target)
        with self.assertRaisesRegex(ValueError, 'No exact match'):
            self.apply()
        self.assertIn('external change', self.target.read_text())

    def test_new_run_and_explicit_steering_reset_previous_patch_cache(self):
        self.apply()
        self.agent._chat = lambda: {'role':'assistant','content':'Ready.'}
        self.agent.run('Start a new task')
        with self.assertRaisesRegex(ValueError, 'No exact match'):
            self.apply()
        # Reapply from a fresh original revision, then deliver a new instruction.
        self.target.write_text('answer = 1\n')
        self.agent._read_file(self.target)
        self.apply()
        self.agent.poll_messages = lambda: ['Reconsider that edit']
        self.agent._drain_messages()
        with self.assertRaisesRegex(ValueError, 'No exact match'):
            self.apply()

    def test_oldest_cache_entry_is_evicted_after_thirty_two_successful_edits(self):
        for value in range(1, 35):
            result = self.agent._edit_file('main.py', f'answer = {value}', f'answer = {value+1}')
            self.assertEqual(result['replacements'], 1)
        self.assertEqual(len(self.agent._applied_edits), 32)
        self.assertTrue(self.agent._edit_file('main.py', 'answer = 34', 'answer = 35')['already_applied'])


if __name__ == '__main__':
    unittest.main()
