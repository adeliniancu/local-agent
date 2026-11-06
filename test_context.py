"""Context continuation tests: complete tool groups, exact recall and bounded retries."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest

from agent_core import Agent, Cancelled, CONTEXT_STATE, CONTEXT_BUDGET_BYTES


def tool_group(name, arguments, result):
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_name": name, "content": json.dumps(result, ensure_ascii=False)}]


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events = []
        self.cancel = threading.Event()
        self.agent = Agent(self.temp.name, lambda kind, data: self.events.append((kind, data)),
                           lambda *_: True, self.cancel)

    def assert_valid_protocol(self, messages):
        waiting = []
        for message in messages:
            if message.get("role") == "tool":
                self.assertTrue(waiting, "orphan tool result")
                self.assertEqual(message["tool_name"], waiting.pop(0))
            else:
                self.assertFalse(waiting, "assistant tool calls lost their results")
                waiting = [call["function"]["name"] for call in message.get("tool_calls", [])]
        self.assertFalse(waiting)

    def test_large_single_tool_heavy_turn_compacts_and_continues(self):
        records = [{"role": "user", "content": "Build the report generator. Preserve existing input files."}]
        for index in range(5):
            records += tool_group("read_file", {"path": f"input{index}.txt"}, {"content": "large diagnostic output " * 1600})
        records += tool_group("write_file", {"path": "report.py", "content": "print('ready')"},
                              {"path": "report.py", "bytes_written": 14, "backup": ".localagent-backups/saved/report.py"})
        original = copy.deepcopy(records)
        self.agent.history = records
        messages = self.agent._messages()
        self.assertEqual(self.agent._records(), original)
        self.assertLessEqual(self.agent._request_size(messages), CONTEXT_BUDGET_BYTES)
        self.assert_valid_protocol(messages)
        handoff = self.agent.history[0]["content"]
        self.assertIn("report generator", handoff)
        self.assertIn("report.py", handoff)
        self.assertIn("Preserve existing", handoff)
        self.assertEqual(self.agent.history[0][CONTEXT_STATE]["compactions"], 1)
        self.assertTrue(any(kind == "memory" for kind, _ in self.events))

    def test_giant_unicode_request_kept_exact_and_middle_limit_recoverable(self):
        request = "Construiește raportul. " + "測試😀" * 12000 + " Never delete originals. " + "界" * 10000
        self.agent.history = [{"role": "user", "content": request}]
        messages = self.agent._messages()
        self.assertLessEqual(self.agent._request_size(messages), CONTEXT_BUDGET_BYTES)
        self.assertEqual(self.agent._records()[0]["content"], request)
        self.assertIn("Never delete originals", self.agent.history[0]["content"])
        found = self.agent._read_working_history(query="Never delete originals")
        self.assertIn("Never delete originals", found["matches"][0]["excerpt"])
        self.assertEqual(found["matches"][0]["message_index"], 0)

    def test_exact_unicode_paging_has_no_gaps_or_duplicates(self):
        text = "aș界😀" * 1000
        self.agent.history = [{"role": "user", "content": text}]
        offset, parts = 0, []
        while offset is not None:
            result = self.agent._read_working_history(message_index=0, start=offset, max_chars=601)
            parts.append(result["text"])
            self.assertLessEqual(len(result["text"].encode("utf-8")), 601)
            offset = result["next_start"]
        self.assertEqual("".join(parts), text)

    def test_plan_and_steering_survive_save_restore_and_repeated_compaction(self):
        steps = [{"text": "Inspect source", "status": "completed"}, {"text": "Test the report", "status": "in_progress"}]
        self.agent.history = [{"role": "user", "content": "Build reports. Do not modify input.csv."}]
        self.agent.history += tool_group("set_plan", {"steps": steps}, {"updated": True})
        self.agent.history += tool_group("read_file", {"path": "input.csv"}, {"content": "large " * 10000})
        self.agent.history.append({"role": "user", "content": "Use Romanian labels and keep the old output."})
        self.agent._messages()
        saved = json.loads(json.dumps(self.agent.history, ensure_ascii=False))
        restored = Agent(self.temp.name, lambda *_: None, lambda *_: True, threading.Event())
        restored.history = saved
        self.assert_valid_protocol(restored._messages())
        self.assertIn("Test the report", restored.history[0]["content"])
        self.assertIn("Romanian labels", restored.history[0]["content"])
        before = copy.deepcopy(restored._records())
        restored.history += tool_group("run_command", {"command": "python report.py"}, {"exit_code": 0, "output": "PASS " * 20000})
        messages = restored._messages()
        self.assertEqual(restored._records()[:len(before)], before)
        self.assertEqual(restored.history[0][CONTEXT_STATE]["compactions"], 2)
        self.assertLessEqual(restored._request_size(messages), CONTEXT_BUDGET_BYTES)
        self.assert_valid_protocol(messages)

    def test_cancel_during_compaction_preserves_full_history(self):
        original = [{"role": "user", "content": "task " * 20000}]
        self.agent.history = copy.deepcopy(original)
        self.agent.emit = lambda kind, data: self.cancel.set() if kind == "status" else None
        with self.assertRaises(Cancelled):
            self.agent._messages()
        self.assertEqual(self.agent.history, original)

    def test_context_error_retries_once_without_replaying_completed_write(self):
        calls = []
        def model():
            calls.append(1)
            if len(calls) == 1:
                return tool_group("write_file", {"path": "done.txt", "content": "done"}, {})[0]
            if len(calls) == 2:
                raise RuntimeError("Ollama HTTP 400: the input length exceeds the context length")
            return {"role": "assistant", "content": "Created done.txt."}
        self.agent._chat_once = model
        self.agent.run("Create done.txt")
        self.assertEqual(len(calls), 3)
        self.assertEqual((Path(self.temp.name) / "done.txt").read_text(), "done")
        self.assertEqual(sum(kind == "tool" for kind, _ in self.events), 1)
        self.assertFalse((Path(self.temp.name) / ".localagent-backups").exists())
        self.assertFalse(any(kind == "error" for kind, _ in self.events))

    def test_second_context_error_stops_without_unbounded_retry(self):
        calls = []
        def model():
            calls.append(1)
            raise RuntimeError("context length exceeded")
        self.agent._chat_once = model
        self.agent.run("Continue the existing task")
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("after one retry" in data.get("text", "") for kind, data in self.events if kind == "error"))
        self.assertEqual(self.events[-1][0], "done")

    def test_partial_stream_is_reset_before_context_retry(self):
        calls = []
        def model():
            calls.append(1)
            self.agent._request_streamed = True
            if len(calls) == 1:
                self.agent.emit("token", {"text": "discard partial"})
                raise RuntimeError("input length exceeds context length")
            self.agent.emit("token", {"text": "Complete reply"})
            return {"role": "assistant", "content": "Complete reply"}
        self.agent._chat_once = model
        self.agent.run("Explain the result")
        visible = ""
        for kind, data in self.events:
            if kind == "stream_reset":
                visible = ""
            elif kind == "token":
                visible += data["text"]
        self.assertEqual(visible, "Complete reply")
        self.assertNotIn("discard partial", json.dumps(self.agent.history))

    def test_network_errors_are_not_misclassified_as_context_errors(self):
        calls = []
        def model():
            calls.append(1)
            raise RuntimeError("Connection refused")
        self.agent._chat_once = model
        self.agent.run("Continue")
        self.assertEqual(len(calls), 1)
        self.assertFalse(any(kind == "memory" for kind, _ in self.events))

    def test_incomplete_tools_and_corrupt_cut_never_produce_orphan_results(self):
        records = [{"role": "user", "content": "Continue"}]
        records += tool_group("read_file", {"path": "a"}, {"content": "read"})
        marker = {"role": "system", "content": "old", CONTEXT_STATE: {
            "schema": 1, "covered_messages": 2, "compactions": 1, "budget_bytes": 24000}}
        self.agent.history = [marker] + records
        self.assert_valid_protocol(self.agent._messages())
        self.agent.history = records + [{"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": {}}}]}]
        self.assert_valid_protocol(self.agent._messages())

    def test_malformed_saved_state_is_rebuilt_without_losing_messages(self):
        original = [{"role": "user", "content": "Keep this goal. " * 5000}]
        for changes in ({"compactions": "invalid"}, {"compactions": -1}, {"budget_bytes": "invalid"}):
            state = {"schema": 1, "covered_messages": 1, "compactions": 1, "budget_bytes": 24000, **changes}
            self.agent.history = [{"role": "system", "content": "bad marker", CONTEXT_STATE: state}] + copy.deepcopy(original)
            messages = self.agent._messages()
            self.assertEqual(self.agent._records(), original)
            self.assertEqual(self.agent.history[0][CONTEXT_STATE]["compactions"], 1)
            self.assert_valid_protocol(messages)

    def test_eight_k_context_scales_budgets_and_survives_reload(self):
        self.agent._inference_options_cache = {"num_ctx": 8192, "num_gpu": 0}
        (Path(self.temp.name) / "AGENTS.md").write_text('Preserve user files. "\\"\n' * 3000, encoding="utf-8")
        original = [{"role": "user", "content": "Keep the original goal. " + '日本語 "\\" ' * 8000}]
        self.agent.history = copy.deepcopy(original)
        messages = self.agent._messages()
        self.assertLessEqual(self.agent._request_size(messages), 14000)
        self.assertEqual(self.agent._context_state()[CONTEXT_STATE]["context_tokens"], 8192)
        messages = self.agent._messages(force_compact=True, aggressive=True)
        self.assertLessEqual(self.agent._request_size(messages), 10000)
        self.assertEqual(self.agent._records(), original)
        self.agent.history = json.loads(json.dumps(self.agent.history))
        self.assertIsNotNone(self.agent._context_state())
        self.assert_valid_protocol(self.agent._messages())

    def test_eight_k_context_retry_preserves_completed_tools(self):
        self.agent._inference_options_cache = {"num_ctx": 8192, "num_gpu": 0}
        records = [{"role": "user", "content": "Continue the existing task. " * 2000}]
        records += tool_group("write_file", {"path": "saved.txt", "content": "saved"}, {"path": "saved.txt", "bytes_written": 5})
        self.agent.history = copy.deepcopy(records)
        requests = []
        def model():
            requests.append(self.agent._messages())
            if len(requests) == 1:
                raise RuntimeError("context length exceeded")
            return {"role": "assistant", "content": "Continue from the saved work."}
        self.agent._chat_once = model
        self.assertEqual(self.agent._chat()["content"], "Continue from the saved work.")
        self.assertEqual(len(requests), 2)
        self.assertLessEqual(self.agent._request_size(requests[-1]), 10000)
        self.assertEqual(self.agent._records(), records)
        self.assert_valid_protocol(requests[-1])

    def test_eight_k_exact_history_page_survives_compaction(self):
        self.agent._inference_options_cache = {"num_ctx": 8192, "num_gpu": 0}
        self.agent.history = [{"role": "user", "content": "Inspect the file and preserve the exact requirements."}]
        self.agent.history += tool_group("read_file", {"path": "source.py"}, {"content": "EXACT_PAGE_SENTINEL " + '日本語 "\\" ' * 4000})
        self.agent._messages(force_compact=True, aggressive=True)
        result = self.agent._read_working_history(message_index=2, max_chars=6000)
        self.agent.history += tool_group("read_working_history", {"message_index": 2, "max_chars": 6000}, result)
        messages = self.agent._messages()
        self.assertLessEqual(self.agent._request_size(messages), 10000)
        self.assertTrue(any(item.get("role") == "tool" and item.get("tool_name") == "read_working_history"
                            and "EXACT_PAGE_SENTINEL" in item["content"] for item in messages))
        self.assert_valid_protocol(messages)

    def test_large_read_explains_exact_retrieval_instead_of_repeating_read(self):
        self.agent.history = [{"role": "user", "content": "Inspect the source"}]
        self.agent.history += tool_group("read_file", {"path": "source.py"}, {"content": "important_start = 1\n" + "large " * 10000})
        self.agent._messages()
        summary = self.agent.history[0]["content"]
        self.assertIn("important_start", summary)
        self.assertIn("page result #2", summary)
        self.assertIn("instead of repeating", summary)

    def test_exact_retrieval_survives_aggressive_compaction_with_escaped_text(self):
        (Path(self.temp.name) / "AGENTS.md").write_text('Keep files. "quoted" \\ path\n' * 1000, encoding="utf-8")
        self.agent.history = [{"role": "user", "content": 'Retrieve this exact data: "quoted" \\ value\n' * 5000}]
        self.agent._messages(force_compact=True, aggressive=True)
        result = self.agent._read_working_history(message_index=0, max_chars=6000)
        self.assertIsNotNone(result["next_start"])
        self.agent.history += tool_group("read_working_history", {"message_index": 0, "max_chars": 6000}, result)
        messages = self.agent._messages()
        self.assertTrue(any(item.get("tool_name") == "read_working_history" for item in messages))
        self.assertLessEqual(self.agent._request_size(messages), 16_000)
        counter = self.agent.history[0][CONTEXT_STATE]["compactions"]
        self.agent._messages()
        self.assertEqual(self.agent.history[0][CONTEXT_STATE]["compactions"], counter)
        self.assert_valid_protocol(messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
