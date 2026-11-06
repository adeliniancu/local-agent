"""Long tasks continue; repeated actions and finite budgets pause without errors."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from agent_core import Agent, BACKUP_DIR, MAX_RUN_STEPS


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def actions(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.events = []
        self.cancel = threading.Event()
        self.agent = Agent(self.root, lambda kind, data: self.events.append((kind, data)),
                           lambda *_: True, self.cancel)

    def assert_clean_end(self):
        self.assertEqual(self.events[-1][0], "done")
        self.assertFalse([data for kind, data in self.events if kind == "error"])
        for _, _, valid in self.agent._groups(self.agent._records()):
            self.assertTrue(valid, self.agent.history)

    def test_productive_task_completes_beyond_twenty_four_model_calls(self):
        responses = iter([actions(call("write_file", path=f"file-{i}.txt", content=str(i)))
                          for i in range(30)] + [{"role": "assistant", "content": "Created all 30 files."}])
        self.agent._chat = lambda: next(responses)
        self.agent.run("Create thirty files")
        self.assertEqual(len(list(self.root.glob("file-*.txt"))), 30)
        self.assertEqual([data["steps"] for kind, data in self.events if kind == "continuation"], [24])
        self.assertEqual(self.agent.history[-1]["content"], "Created all 30 files.")
        self.assert_clean_end()

    def test_repeated_reads_warn_then_pause_and_preserve_every_result(self):
        (self.root / "input.txt").write_text("Already known")
        self.agent._chat = lambda: actions(call("read_file", path="input.txt"))
        self.agent.run("Inspect the file")
        results = [json.loads(item["content"]) for item in self.agent.history if item["role"] == "tool"]
        self.assertEqual(len(results), 9)  # First read is new evidence; eight repeats are not.
        self.assertTrue(any("recovery_hint" in result for result in results))
        self.assertIn("no new result", self.agent.history[-1]["content"])
        self.assert_clean_end()

    def test_identical_rewrites_do_not_make_backups_or_reset_progress(self):
        (self.root / "same.txt").write_text("same", encoding="utf-8")
        self.agent._chat = lambda: actions(call("write_file", path="same.txt", content="same"))
        self.agent.run("Update the file")
        self.assertFalse((self.root / BACKUP_DIR).exists())
        results = [json.loads(item["content"]) for item in self.agent.history if item["role"] == "tool"]
        self.assertEqual(len(results), 8)
        self.assertTrue(all(result["unchanged"] for result in results))
        self.assert_clean_end()

    def test_alternating_file_states_eventually_pause(self):
        count = 0
        def chat():
            nonlocal count
            count += 1
            return actions(call("write_file", path="loop.txt", content=str(count % 2)))
        self.agent._chat = chat
        self.agent.run("Make the change")
        self.assertEqual(count, 10)
        self.assertIn("no new result", self.agent.history[-1]["content"])
        self.assert_clean_end()

    def test_changed_read_results_are_progress(self):
        count = 0
        def chat():
            nonlocal count
            count += 1
            if count > 26:
                return {"role": "assistant", "content": "Read all updates."}
            (self.root / "changing.txt").write_text(str(count))
            return actions(call("read_file", path="changing.txt"))
        self.agent._chat = chat
        self.agent.run("Follow the file updates")
        self.assertEqual(count, 27)
        self.assert_clean_end()

    def test_three_identical_command_failures_remain_a_clean_pause(self):
        count = 0
        def command(**_):
            nonlocal count
            count += 1
            result = {"exit_code": 1, "output": "ModuleNotFoundError: missing package"}
            self.agent._record_command_result(result)
            return result
        self.agent._run_command = command
        self.agent._chat = lambda: actions(call("run_command", command="python check.py"))
        self.agent.run("Check the project")
        self.assertEqual(count, 3)
        self.assertIn("same command error", self.agent.history[-1]["content"])
        self.assert_clean_end()

    def test_repeated_successful_commands_do_not_fake_progress(self):
        self.agent._run_command = lambda **_: {"exit_code": 0, "output": "same result"}
        self.agent._chat = lambda: actions(call("run_command", command="python check.py"))
        self.agent.run("Check the project")
        self.assertEqual(sum(item["role"] == "tool" for item in self.agent.history), 9)
        self.assertIn("no new result", self.agent.history[-1]["content"])
        self.assert_clean_end()

    def test_repeated_plan_updates_do_not_fake_progress_and_batch_is_completed(self):
        self.agent._chat = lambda: actions(*[call("set_plan", steps=[{"text": "Inspect", "status": "in_progress"}])
                                              for _ in range(10)])
        self.agent.run("Do the work")
        results = [json.loads(item["content"]) for item in self.agent.history if item["role"] == "tool"]
        self.assertEqual(len(results), 10)
        self.assertTrue(all(result.get("skipped") for result in results[8:]))
        self.assertEqual(len([kind for kind, _ in self.events if kind == "plan"]), 8)
        self.assert_clean_end()

    def test_productive_task_still_has_a_clean_finite_step_budget_and_can_resume(self):
        counter = 0
        def chat():
            nonlocal counter
            counter += 1
            return actions(call("write_file", path="progress.txt", content=str(counter)))
        self.agent._chat = chat
        with patch("agent_core.MAX_RUN_STEPS", 3):
            self.agent.run("Long work")
        self.assertEqual(counter, 3)
        self.assertIn("3-step run budget", self.agent.history[-1]["content"])
        self.assertIn("Continue", self.agent.history[-1]["content"])
        before = self.agent.history.copy()
        self.agent._chat = lambda: {"role": "assistant", "content": "Finished remaining work."}
        self.agent.run("Continue")
        self.assertEqual(self.agent.history[:len(before)], before)
        self.assertEqual(self.agent.history[-1]["content"], "Finished remaining work.")
        self.assertGreater(MAX_RUN_STEPS, 24)
        self.assert_clean_end()

    def test_elapsed_budget_skips_actions_but_preserves_tool_pairing(self):
        self.agent._chat = lambda: actions(call("write_file", path="never.txt", content="no"))
        with patch("agent_core.time.monotonic", side_effect=[0, 0, 3601]):
            self.agent.run("Do work")
        self.assertFalse((self.root / "never.txt").exists())
        self.assertIn("one-hour", self.agent.history[-1]["content"])
        self.assert_clean_end()

    def test_queued_direction_resets_stall_counter(self):
        (self.root / "input.txt").write_text("known")
        pending = []
        count = 0
        def poll():
            result = pending[:]
            pending.clear()
            return result
        def chat():
            nonlocal count
            count += 1
            if count == 9:
                pending.append("Create done.txt now")
            if count == 10:
                return actions(call("write_file", path="done.txt", content="done"))
            if count == 11:
                return {"role": "assistant", "content": "Done."}
            return actions(call("read_file", path="input.txt"))
        self.agent.poll_messages = poll
        self.agent._chat = chat
        self.agent.run("Inspect input")
        self.assertEqual(count, 11)
        self.assertTrue((self.root / "done.txt").exists())
        self.assert_clean_end()

    def test_cancellation_after_continuation_keeps_unexecuted_tool_result(self):
        counter = 0
        def chat():
            nonlocal counter
            counter += 1
            if counter == 26:
                self.cancel.set()
            return actions(call("write_file", path="progress.txt", content=str(counter)))
        self.agent._chat = chat
        self.agent.run("Do long work")
        self.assertEqual(counter, 26)
        self.assertEqual((self.root / "progress.txt").read_text(), "25")
        self.assertIn("Stopped by user", self.agent.history[-1]["content"])
        self.assert_clean_end()


if __name__ == "__main__":
    unittest.main()
