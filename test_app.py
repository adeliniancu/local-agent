"""Desktop behavior tests using a withdrawn Tk window; no model or network."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import app


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name).resolve()
        self.patches = [patch.object(app, "DATA", self.data), patch.object(app, "STATE", self.data / "last-chat.json")]
        for item in self.patches:
            item.start()
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            for item in reversed(self.patches):
                item.stop()
            self.temp.cleanup()
            self.skipTest(f"Tk is unavailable: {exc}")
        self.root.withdraw()
        self.ui = app.LocalAgentApp(self.root, start_engine=False)
        self.ui.ready = True
        self.ui.base_url = "http://127.0.0.1:11435"
        self.root.update_idletasks()

    def tearDown(self):
        if self.root.winfo_exists():
            self.ui.worker = None
            self.ui.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_enter_sends_and_shift_enter_inserts_newline(self):
        with patch.object(self.ui, "send") as send:
            self.assertEqual(self.ui._send_key(None), "break")
            send.assert_called_once()
        self.ui.input.insert("1.0", "line one")
        self.ui.input.mark_set("insert", "end-1c")
        self.assertEqual(self.ui._newline_key(None), "break")
        self.assertEqual(self.ui.input.get("1.0", "end-1c"), "line one\n")

    def test_busy_send_is_queued_and_does_not_split_stream(self):
        self.ui._set_busy(True)
        self.ui._handle_event("token", {"text": "Hello "}, 0)
        self.ui.input.insert("1.0", "Also add tests")
        self.ui.send()
        self.assertEqual(self.ui.input.cget("state"), "normal")
        self.assertEqual(self.ui.incoming.get_nowait(), "Also add tests")
        self.ui._handle_event("token", {"text": "world"}, 0)
        self.ui._handle_event("assistant", {"text": "Hello world"}, 0)
        transcript = self.ui.transcript.get("1.0", "end")
        self.assertEqual(transcript.count("Hello world"), 1)
        self.assertEqual(transcript.count("Also add tests"), 1)
        self.assertLess(transcript.index("Hello world"), transcript.index("Also add tests"))
        self.ui._handle_event("steering", {"text": "Also add tests"}, 0)
        self.assertEqual(self.ui.queued_count, 0)

    def test_completion_race_preserves_late_messages(self):
        self.ui.busy = True
        self.ui.incoming.put("late instruction")
        with patch.object(self.ui, "_start_run") as start:
            self.ui._handle_event("worker_complete", {"history": [{"role": "assistant", "content": "Done"}]}, 0)
            start.assert_called_once_with("late instruction", continuation=True)
        self.assertEqual(self.ui.history[0]["content"], "Done")

    def test_stop_drops_pending_and_never_auto_runs(self):
        self.ui._set_busy(True)
        self.ui.incoming.put("Do not run after Stop")
        self.ui.queued_count = 1
        request = {"event": threading.Event(), "answer": None}
        self.ui.question_request = request
        self.ui.stop()
        self.assertTrue(request["event"].is_set())
        self.assertIsNone(request["answer"])
        self.assertTrue(self.ui.incoming.empty())
        with patch.object(self.ui, "_start_run") as start:
            self.ui._handle_event("worker_complete", {"history": []}, 0)
            start.assert_not_called()
        self.assertFalse(self.ui.busy)

    def test_main_composer_answers_pending_question(self):
        self.ui._set_busy(True)
        request = {"event": threading.Event(), "answer": None}
        self.ui.question_request = request
        self.ui.input.insert("1.0", "Use Python")
        self.ui.send()
        self.assertEqual(request["answer"], "Use Python")
        self.assertTrue(request["event"].is_set())
        self.assertTrue(self.ui.incoming.empty())

    def test_speed_uses_model_metrics_and_elapsed_uses_wall_time(self):
        self.ui._set_busy(True)
        self.ui.work_started = 100.0
        self.ui._handle_event("token", {"text": "one arbitrary stream chunk"}, 0)
        self.assertIsNone(self.ui.last_tokens_per_second)
        self.ui._handle_event("metrics", {"tokens_per_second": 12.75, "eval_count": 51}, 0)
        with patch("app.time.monotonic", return_value=107.0):
            self.ui._tick()
        self.assertIn("00:07", self.ui.metrics_label.cget("text"))
        self.assertIn("12.8 tokens/s", self.ui.metrics_label.cget("text"))
        self.ui._handle_event("metrics", {"tokens_per_second": None}, 0)
        self.assertIsNone(self.ui.last_tokens_per_second)

    def test_archive_preserves_history_and_reason_setting(self):
        self.ui.history = [{"role": "user", "content": "Keep this"}]
        self.ui.think_var.set(True)
        self.ui.new_chat()
        archives = list((self.data / "chats").glob("*.json"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(json.loads(archives[0].read_text())["history"][0]["content"], "Keep this")
        self.assertTrue(json.loads(app.STATE.read_text())["reason_more"])

    def test_update_is_blocked_while_working(self):
        self.ui._set_busy(True)
        self.assertEqual(self.ui.update_button.cget("state"), "disabled")
        self.ui.open_update()
        self.assertFalse(self.ui.updating)

    def test_memory_notice_preserves_chat_and_does_not_interrupt_work(self):
        self.ui.history = [{"role": "user", "content": "Original instructions stay saved"}]
        self.ui._set_busy(True)
        self.ui._handle_event("memory", {"compactions": 1, "covered_messages": 12}, 0)
        self.ui._handle_event("memory", {"compactions": 1, "covered_messages": 12}, 0)
        self.assertTrue(self.ui.busy)
        self.assertEqual(self.ui.input.cget("state"), "normal")
        self.assertEqual(self.ui.memory_label.winfo_manager(), "pack")
        transcript = self.ui.transcript.get("1.0", "end")
        self.assertEqual(transcript.count("Working memory summarized so the task can continue."), 1)
        self.ui._save_state()
        saved = json.loads(app.STATE.read_text())
        self.assertEqual(saved["memory_state"], {"compactions": 1, "covered_messages": 12})
        self.assertEqual(saved["history"][0]["content"], "Original instructions stay saved")
        self.ui.busy = False
        self.ui.new_chat()
        self.assertFalse(self.ui.memory_state)
        self.assertEqual(self.ui.memory_label.winfo_manager(), "")
        archived = json.loads(next((self.data / "chats").glob("*.json")).read_text())
        self.assertEqual(archived["memory_state"]["compactions"], 1)

    def test_context_retry_discards_only_unfinished_assistant_card(self):
        self.ui._handle_event("assistant", {"text": "Earlier completed reply"}, 0)
        self.ui._handle_event("token", {"text": "unfinished output to discard"}, 0)
        self.ui._append("YOU · QUEUED\n", "user_role")
        self.ui._append("Keep this queued instruction\n\n")
        self.ui._handle_event("stream_reset", {"text": "Retrying after memory compaction"}, 0)
        transcript = self.ui.transcript.get("1.0", "end")
        self.assertIn("Earlier completed reply", transcript)
        self.assertIn("Keep this queued instruction", transcript)
        self.assertNotIn("unfinished output to discard", transcript)
        self.ui._handle_event("token", {"text": "Successful retry"}, 0)
        self.ui._handle_event("assistant", {"text": "Successful retry"}, 0)
        self.assertEqual(self.ui.transcript.get("1.0", "end").count("Successful retry"), 1)

    def test_missing_workspace_retains_history_and_blocks_sending(self):
        missing = self.data / "moved-project"
        history = [{"role": "user", "content": "Important previous task"}]
        app.STATE.write_text(json.dumps({"workspace": str(missing), "history": history}))
        self.ui._load_state()
        self.ui._render_history()
        self.assertEqual(self.ui.workspace, missing)
        self.assertEqual(self.ui.history, history)
        self.ui._handle_event("connected", {"base_url": self.ui.base_url, "models": [app.DEFAULT_MODEL]}, None)
        self.assertEqual(self.ui.send_button.cget("state"), "disabled")
        self.ui.input.insert("1.0", "continue")
        with patch.object(self.ui, "_start_run") as start:
            self.ui.send()
            start.assert_not_called()
        self.ui._save_state()
        saved = json.loads(app.STATE.read_text())
        self.assertEqual(saved["history"], history)
        self.assertEqual(saved["workspace"], str(missing))

    def test_missing_workspace_archive_remains_readable(self):
        archive = self.data / "old-chat.json"
        archive.write_text(json.dumps({"workspace": str(self.data / "gone"),
                                      "history": [{"role": "assistant", "content": "A saved result"}]}))
        with patch("app.filedialog.askopenfilename", return_value=str(archive)):
            self.ui.open_saved_chat()
        self.assertIn("A saved result", self.ui.transcript.get("1.0", "end"))
        self.assertEqual(self.ui.send_button.cget("state"), "disabled")

    def test_context_marker_restores_badge_without_displaying_metadata(self):
        marker = {"role": "system", "content": "Hidden working summary", "_local_agent_context": {
            "schema": 1, "covered_messages": 25, "compactions": 3, "budget_bytes": 16000}}
        app.STATE.write_text(json.dumps({"workspace": str(self.ui.workspace),
                                        "history": [marker, {"role": "user", "content": "Visible request"}]}))
        self.ui._load_state()
        self.ui._render_history()
        self.ui._refresh_memory_indicator()
        self.assertEqual(self.ui.memory_state["compactions"], 3)
        self.assertEqual(self.ui.memory_label.winfo_manager(), "pack")
        transcript = self.ui.transcript.get("1.0", "end")
        self.assertIn("Visible request", transcript)
        self.assertNotIn("Hidden working summary", transcript)
        self.assertNotIn("budget_bytes", transcript)

    def test_empty_models_only_shows_download_without_starting_it(self):
        with patch.object(self.ui, "download_model") as download:
            self.ui._handle_event("connected", {"base_url": self.ui.base_url, "models": []}, None)
            self.assertFalse(self.ui.ready)
            self.assertEqual(self.ui.setup_bar.winfo_manager(), "pack")
            download.assert_not_called()
        self.ui._handle_event("model_download_status", {"status": "pulling model", "total": 100, "completed": 25}, None)
        self.assertIn("25%", self.ui.status_label.cget("text"))
        self.ui._handle_event("connected", {"base_url": self.ui.base_url, "models": [app.DEFAULT_MODEL]}, None)
        self.assertTrue(self.ui.ready)
        self.assertEqual(self.ui.setup_bar.winfo_manager(), "")

    def test_question_callback_integrates_with_agent_and_saved_history(self):
        from agent_core import Agent
        responses = [
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "ask_user", "arguments": {"question": "Which language?", "options": ["Python", "Java"]}}}]},
            {"role": "assistant", "content": "I will use Python."},
        ]

        def respond(request):
            self.ui.question_request = request
            self.ui._answer_question("Python")

        self.ui.think_var.set(True)
        with patch.object(Agent, "_chat", side_effect=responses), patch.object(self.ui, "_show_question", side_effect=respond):
            self.ui.input.insert("1.0", "Help me choose a language")
            self.ui.send()
            deadline = time.monotonic() + 5
            while self.ui.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
            self.assertFalse(self.ui.busy, "The question callback did not finish")
            self.assertTrue(self.ui.agent.think)
        saved = json.loads(app.STATE.read_text())
        result = next(item for item in saved["history"] if item.get("tool_name") == "ask_user")
        self.assertEqual(json.loads(result["content"])["answer"], "Python")
        self.ui._render_history()
        self.assertIn("Which language?", self.ui.transcript.get("1.0", "end"))
        self.assertIn("YOU · ANSWER", self.ui.transcript.get("1.0", "end"))


if __name__ == "__main__":
    unittest.main()
