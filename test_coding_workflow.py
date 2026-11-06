"""Precise edits preserve concurrent changes and unrelated file bytes."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from agent_core import Agent, BACKUP_DIR, MAX_TOOL_OUTPUT


class CodingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = Agent(self.root, lambda *_: None, lambda *_: True, threading.Event())
        self.path = self.root / "source.py"

    def read(self):
        return self.agent._execute("read_file", {"path": "source.py"})

    def edit(self, search, replace):
        return self.agent._execute("edit_file", {"path": "source.py", "search": search, "replace": replace})

    def test_existing_file_requires_read_before_overwrite(self):
        self.path.write_text("original")
        with self.assertRaisesRegex(ValueError, "read_file first"):
            self.agent._execute("write_file", {"path": "source.py", "content": "replacement"})
        self.assertEqual(self.path.read_text(), "original")
        self.assertFalse((self.root / BACKUP_DIR).exists())

    def test_stale_read_blocks_whole_file_overwrite_and_edit(self):
        self.path.write_text("old = 1\n")
        self.read()
        self.path.write_text("old = 1\nuser_change = True\n")
        for action in (lambda: self.edit("old = 1", "old = 2"),
                       lambda: self.agent._execute("write_file", {"path": "source.py", "content": "old = 2\n"})):
            with self.assertRaisesRegex(ValueError, "changed since your last read"):
                action()
        self.assertIn("user_change", self.path.read_text())
        self.read()
        self.edit("old = 1", "old = 2")
        self.assertEqual(self.path.read_text(), "old = 2\nuser_change = True\n")

    def test_unique_match_including_overlaps_and_missing_block(self):
        self.path.write_text("aaa\nvalue = 1\nvalue = 1\n")
        self.read()
        for search in ("aa", "value = 1", "missing"):
            with self.subTest(search=search), self.assertRaisesRegex(ValueError, "No changes were written"):
                self.edit(search, "replacement")
        self.assertEqual(self.path.read_text(), "aaa\nvalue = 1\nvalue = 1\n")

    def test_edit_preserves_bom_newlines_unrelated_bytes_and_backup(self):
        original = b"\xef\xbb\xbf# untouched\r\nvalue = 1\r\nkeep = 3\r\n"
        self.path.write_bytes(original)
        result = self.read()
        self.assertEqual(result["sha256"], hashlib.sha256(original).hexdigest())
        edited = self.edit("value = 1\nkeep = 3", "value = 2\nkeep = 3")
        self.assertEqual(self.path.read_bytes(), original.replace(b"value = 1", b"value = 2"))
        self.assertEqual((self.root / edited["backup"]).read_bytes(), original)
        # A single-line search with a multiline replacement uses the same CRLF style.
        self.edit("value = 2", "value = 2\nadded = 4")
        self.assertIn(b"value = 2\r\nadded = 4\r\n", self.path.read_bytes())

    def test_partial_read_allows_precise_edit_but_not_whole_file_replacement(self):
        original = "target = 1\n" + "# preserved tail\n" * MAX_TOOL_OUTPUT
        self.path.write_text(original, encoding="utf-8")
        self.assertTrue(self.read()["truncated"])
        self.edit("target = 1", "target = 2")
        self.assertEqual(self.path.read_text(), original.replace("target = 1", "target = 2"))
        self.edit("target = 2", "target = 2")  # A no-op edit must not pretend the full file was read.
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.agent._execute("write_file", {"path": "source.py", "content": "short replacement"})

    def test_change_during_write_preparation_is_preserved(self):
        self.path.write_text("old")
        self.read()
        original_check = self.agent._check_cancelled
        checks = 0
        def interleave():
            nonlocal checks
            original_check()
            checks += 1
            # _execute then _write entry, then the final pre-temp check.
            if checks == 3:
                self.path.write_text("new user edit")
        with patch.object(self.agent, "_check_cancelled", side_effect=interleave):
            with self.assertRaisesRegex(ValueError, "changed while preparing"):
                self.agent._execute("write_file", {"path": "source.py", "content": "agent edit"})
        self.assertEqual(self.path.read_text(), "new user edit")
        self.assertFalse([path for path in self.root.glob(".localagent-*") if path.is_file()])

    def test_new_file_appearing_during_preparation_is_not_overwritten(self):
        original_check = self.agent._check_cancelled
        checks = 0
        def interleave():
            nonlocal checks
            original_check()
            checks += 1
            if checks == 3:
                self.path.write_text("created by user")
        with patch.object(self.agent, "_check_cancelled", side_effect=interleave):
            with self.assertRaisesRegex(ValueError, "appeared at this path"):
                self.agent._execute("write_file", {"path": "source.py", "content": "agent content"})
        self.assertEqual(self.path.read_text(), "created by user")

    def test_own_new_file_can_be_edited_without_redundant_read(self):
        self.agent._execute("write_file", {"path": "source.py", "content": "value = 1"})
        self.edit("value = 1", "value = 2")
        self.assertEqual(self.path.read_text(), "value = 2")

    def test_qwen_bare_tool_json_executes_through_normal_file_guards(self):
        self.agent.model = "qwen2.5-coder:7b"
        responses = iter([{"role": "assistant", "content": json.dumps({"name": "write_file", "arguments": {
            "path": "source.py", "content": "value = 1"}})}, {"role": "assistant", "content": "Created source.py."}])
        self.agent._chat = lambda: next(responses)
        self.agent.run("Create source.py")
        self.assertEqual(self.path.read_text(), "value = 1")
        self.assertEqual(sum(item["role"] == "tool" for item in self.agent.history), 1)

    def test_qwen_tool_fallback_never_extracts_examples_or_invalid_requests(self):
        self.agent.model = "qwen2.5-coder:7b"
        valid = json.dumps({"name": "write_file", "arguments": {"path": "source.py", "content": "x"}})
        cases = ["Here is an example: " + valid, "```json\n" + valid + "\n```", valid + valid,
                 json.dumps({"name": "write_file", "arguments": {"path": "source.py", "content": "x"}, "explanation": "example"}),
                 json.dumps({"name": "unknown_tool", "arguments": {}}),
                 json.dumps({"name": "write_file", "arguments": {"path": "source.py"}}),
                 json.dumps({"name": "write_file", "arguments": {"path": 3, "content": "x"}}),
                 json.dumps({"name": "write_file", "arguments": {"path": "source.py", "content": "x", "extra": True}}),
                 json.dumps({"name": "run_command", "arguments": {"command": "echo example", "timeout_seconds": True}}),
                 json.dumps({"name": "read_file", "arguments": "source.py"}),
                 '{"name":"read_file","name":"write_file","arguments":{"path":"source.py","content":"x"}}',
                 '{"value": 42}', '[{"name":"read_file","arguments":{"path":"source.py"}}]']
        for content in cases:
            with self.subTest(content=content):
                message = {"role": "assistant", "content": content}
                self.assertEqual(self.agent._normalize_tool_message(message), message)
        self.assertFalse(self.path.exists())

    def test_qwen_wrapped_call_is_supported_but_other_models_unchanged(self):
        content = '<tool_call>{"name":"read_file","arguments":{"path":"source.py"}}</tool_call>'
        message = {"role": "assistant", "content": content}
        self.assertEqual(self.agent._normalize_tool_message(message), message)
        self.agent.model = "qwen2.5-coder:7b"
        self.assertEqual(self.agent._normalize_tool_message(message)["tool_calls"][0]["function"]["name"], "read_file")

    def test_qwen_json_command_still_requires_approval(self):
        self.agent.model = "qwen2.5-coder:7b"
        approved = []
        self.agent.approve = lambda command, cwd: approved.append(command) or False
        self.agent._chat = lambda: {"role": "assistant", "content": '{"name":"run_command","arguments":{"command":"echo example"}}'}
        with patch("agent_core.subprocess.Popen") as process:
            self.agent.run("Run the command")
        self.assertEqual(approved, ["echo example"])
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
