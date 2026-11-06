import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

from agent_core import Agent
from test_context import tool_group


class RepairMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.approve = Mock(return_value=True)
        self.agent = Agent(self.root, lambda *_: None, self.approve, threading.Event())

    def test_latest_file_page_and_exception_survive_large_old_chat(self):
        original = "self.load_tasks()\nself.create_widgets()\n" + "# irrelevant line\n" * 800
        (self.root / "main.py").write_text(original)
        records = [{"role": "user", "content": "Fix the app startup and preserve existing tasks."}]
        for i in range(30):
            records += tool_group("list_files", {}, {"entries": "older information" * 500})
        records += tool_group("run_command", {"command": "python main.py"},
                              {"exit_code": 1, "output": "Traceback\nAttributeError: TodoApp has no status_label\n\n----------\nRan 5 tests in 0.2s\n\nFAILED (errors=5)"})
        page = self.agent._execute("read_file", {"path": "main.py"})
        records += tool_group("read_file", {"path": "main.py"}, page)
        self.agent.history = records
        messages = self.agent._messages()
        self.assertTrue(any(m.get("role") == "tool" and "self.load_tasks()" in m.get("content", "") for m in messages))
        self.assertIn("AttributeError", json.dumps(messages))
        self.assertIn("status_label", json.dumps(messages))
        self.assertLessEqual(self.agent._request_size(messages), self.agent._context_budgets()[0])
        self.assertEqual(len(self.agent._records()), len(records))

    def test_unicode_pages_reassemble_without_skipped_characters(self):
        original = "日本語🙂\n" * 3000
        (self.root / "text.py").write_text(original, encoding="utf-8")
        parts, start = [], 0
        while start is not None:
            page = self.agent._execute("read_file", {"path": "text.py", "start": start})
            self.assertLessEqual(len(page["content"].encode()), 4000)
            parts.append(page["content"])
            start = page["next_start"]
        self.assertEqual("".join(parts), original)
        (self.root / "tiny.txt").write_text("🙂x", encoding="utf-8")
        first = self.agent._read_file(self.root / "tiny.txt", 0, 1)
        self.assertEqual(first["content"], "🙂")
        self.assertEqual(first["next_start"], 1)

    def test_related_source_pages_survive_together_after_compaction(self):
        original = "# startup\n" + "x = 'first section'\n" * 230
        original += "# actions\n" + "y = 'second section'\n" * 230
        (self.root / "main.py").write_text(original, encoding="utf-8")
        self.agent.history = [{"role": "user", "content": "Fix startup, then verify the task actions."}]
        for _ in range(15):
            self.agent.history += tool_group("list_files", {}, {"entries": "old exploration " * 400})
        pages, start = [], 0
        while start is not None:
            page = self.agent._read_file(self.root / "main.py", start)
            pages.append(page)
            self.agent.history += tool_group("read_file", {"path": "main.py", "start": start}, page)
            self.agent._messages()
            start = page["next_start"]
        messages = self.agent._messages(force_compact=True)
        visible = [json.loads(item["content"]) for item in messages if item.get("tool_name") == "read_file"]
        self.assertEqual([page["content"] for page in visible], [page["content"] for page in pages])
        self.assertLessEqual(self.agent._request_size(messages), self.agent._context_budgets()[0])

    def test_full_replacement_requires_all_pages_of_same_version(self):
        target = self.root / "large.py"
        target.write_text("# original\n" * 1000)
        self.agent._read_file(target, 0)
        self.agent._read_file(target, 8000)
        with self.assertRaisesRegex(ValueError, "partial read"):
            self.agent._write(target, "# replacement\n")
        self.agent._read_file(target, 4000)
        self.agent._write(target, "# replacement\n")
        self.assertEqual(target.read_text(), "# replacement\n")
        # Reading a page after an external edit must reset previous coverage.
        target.write_text("# changed!\n" * 1000)
        self.agent._read_file(target, 4000)
        with self.assertRaisesRegex(ValueError, "partial read"):
            self.agent._write(target, "# replacement\n")
        self.agent._read_file(target, 0)
        self.agent._read_file(target, 8000)
        self.agent._write(target, "# replacement\n")
        self.assertEqual(target.read_text(), "# replacement\n")

    def test_launch_is_nonblocking_and_does_not_duplicate(self):
        (self.root / "app.py").write_text("import time\ntime.sleep(30)\n")
        first = self.agent._execute("launch_python_app", {"path": "app.py"})
        process = next(iter(self.agent._launched_apps.values()))[0]
        self.addCleanup(self.agent._kill_process, process)
        self.assertTrue(first["running"])
        second = self.agent._execute("launch_python_app", {"path": "app.py"})
        self.assertEqual(first["pid"], second["pid"])
        self.assertTrue(second["already_running"])
        self.assertEqual(self.approve.call_count, 1)

    def test_overlapping_reads_do_not_evade_loop_detection(self):
        target = self.root / "main.py"
        target.write_text("# example\n" * 1000)
        def progress(start, size):
            result = self.agent._read_file(target, start, size)
            return self.agent._progress_signature("read_file", {"path": "main.py"}, result)
        self.assertIsNotNone(progress(0, 4000))
        for start in (100, 350, 500, 800, 950, 1050):
            self.assertIsNone(progress(start, 150))
        self.assertIsNotNone(progress(4000, 4000))
        self.assertIsNone(progress(3800, 400))
        target.write_text("# changed\n" * 1000)
        self.assertIsNotNone(progress(0, 4000))

    def test_different_test_failures_with_identical_footer_are_not_a_loop(self):
        for diagnostic in ('AttributeError: no label', 'TclError: duplicate ID', 'AssertionError: expected dark theme'):
            self.agent._record_command_result({'exit_code': 1, 'output': diagnostic + '\n\nRan 5 tests\nFAILED (errors=1)'})
        self.assertFalse(self.agent._stuck)
        for _ in range(2):
            self.agent._record_command_result({'exit_code': 1, 'output': 'AssertionError: expected dark theme\n\nRan 5 tests\nFAILED (errors=1)'})
        self.assertTrue(self.agent._stuck)

    def test_launch_reports_startup_failure_and_respects_denial(self):
        (self.root / "bad.py").write_text("raise RuntimeError('startup-broken')\n")
        self.approve.return_value = False
        with self.assertRaises(PermissionError):
            self.agent._execute("launch_python_app", {"path": "bad.py"})
        self.assertFalse(self.agent._launched_apps)
        self.approve.return_value = True
        result = self.agent._execute("launch_python_app", {"path": "bad.py"})
        self.assertFalse(result["running"])
        self.assertIn("startup-broken", result["output"])


if __name__ == "__main__":
    unittest.main()
