"""Desktop behavior tests using a withdrawn Tk window; no model or network."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import app
import runtime


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name).resolve()
        self.patches = [patch.object(app, "DATA", self.data), patch.object(app, "STATE", self.data / "last-chat.json"),
                        patch.object(runtime, "ROOT", self.data)]
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

    def test_reasoning_default_and_explicit_saved_choice(self):
        self.assertTrue(self.ui.think_var.get())
        app.STATE.write_text(json.dumps({"reason_more": False}), encoding="utf-8")
        self.ui._load_state()
        self.assertFalse(self.ui.reason_more)
        app.STATE.write_text(json.dumps({}), encoding="utf-8")
        self.ui._load_state()
        self.assertTrue(self.ui.reason_more)

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

    @staticmethod
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from DesktopTests.descendants(child)

    def test_command_approval_is_inline_and_always_allow_is_visible(self):
        self.ui._set_busy(True)
        command = "Write-Output '" + "complete command text " * 100 + "'"
        request = {"command": command, "cwd": str(self.ui.workspace),
                   "event": threading.Event(), "allowed": False}
        with patch("app.tk.Toplevel", side_effect=AssertionError("AI approvals must not open popups")):
            self.ui._show_approval(request)
        card = self.ui.approval_window
        self.assertIsInstance(card, tk.Frame)
        self.assertIs(self.ui.prompt_panel.master, self.ui.footer_frame)
        command_text = next(child for child in self.descendants(card) if isinstance(child, tk.Text))
        self.assertEqual(command_text.get("1.0", "end-1c"), command)
        self.assertFalse(request["event"].is_set())
        self.assertEqual(self.ui.always_allow_box.winfo_manager(), "pack")
        self.ui.always_allow_box.invoke()
        self.assertTrue(request["event"].is_set())
        self.assertTrue(request["allowed"])
        self.assertTrue(self.ui.allow_commands.is_set())
        self.assertIsNone(self.ui.approval_window)
        self.ui.busy = False
        self.ui.new_chat()
        self.assertFalse(self.ui.always_allow_var.get())
        self.assertFalse(self.ui.allow_commands.is_set())

    def test_question_choices_are_inline_and_submit_without_popup(self):
        self.ui._set_busy(True)
        request = {"question": "Which language should I use?", "options": ["Python", "Java"],
                   "event": threading.Event(), "answer": None}
        with patch("app.tk.Toplevel", side_effect=AssertionError("AI questions must not open popups")):
            self.ui._show_question(request)
        card = self.ui.question_window
        self.assertIsInstance(card, tk.Frame)
        self.assertIs(self.ui.prompt_panel.master, self.ui.footer_frame)
        widgets = list(self.descendants(self.ui.prompt_panel))
        next(child for child in widgets if isinstance(child, tk.Radiobutton) and child.cget("text") == "Python").invoke()
        next(child for child in widgets if isinstance(child, tk.Button) and child.cget("text") == "Send answer").invoke()
        self.assertTrue(request["event"].is_set())
        self.assertEqual(request["answer"], "Python")
        self.assertIsNone(self.ui.question_window)

    def test_question_delegation_does_not_grant_command_permission(self):
        request = {"question": "Which color?", "options": ["Blue", "Green"],
                   "event": threading.Event(), "answer": None}
        self.ui._show_question(request)
        self.ui.decide_questions_box.invoke()
        self.assertTrue(request["event"].is_set())
        self.assertIn("best judgment", request["answer"])
        self.assertFalse(self.ui.allow_commands.is_set())
        self.assertIsNone(self.ui.prompt_panel)
        another = {"question": "Which font?", "options": [], "event": threading.Event(), "answer": None}
        self.ui._show_question(another)
        self.assertTrue(another["event"].is_set())
        self.assertIsNone(self.ui.prompt_panel)
        self.ui.new_chat()
        self.assertFalse(self.ui.decide_questions_var.get())

    def test_small_window_keeps_question_actions_visible_when_chat_scrolled(self):
        self.root.attributes("-alpha", 0)
        self.root.deiconify()
        self.root.geometry("840x620")
        self.root.update()
        self.ui._append("Old output\n" * 200)
        request = {"question": "Choose carefully. " * 100, "options": ["Option " + str(i) + " text" * 50 for i in range(4)],
                   "event": threading.Event(), "answer": None}
        self.ui._show_question(request)
        self.root.update()
        button = next(w for w in self.descendants(self.ui.prompt_panel)
                      if isinstance(w, tk.Button) and w.cget("text") == "Send answer")
        position = button.winfo_rooty()
        self.ui.transcript.yview_moveto(0)
        self.root.update()
        self.assertEqual(position, button.winfo_rooty())
        self.assertGreaterEqual(position, self.root.winfo_rooty())
        self.assertLessEqual(position + button.winfo_height(), self.root.winfo_rooty() + self.root.winfo_height())
        self.assertLessEqual(self.ui.send_button.winfo_rooty() + self.ui.send_button.winfo_height(),
                             self.root.winfo_rooty() + self.root.winfo_height())
        self.ui._answer_question("Blue")
        self.root.withdraw()

    def test_pinned_command_checkbox_allows_current_and_future_commands(self):
        request = {"command": "Write-Output test", "cwd": str(self.ui.workspace),
                   "event": threading.Event(), "allowed": False}
        self.ui._show_approval(request)
        box = next(w for w in self.descendants(self.ui.prompt_panel) if isinstance(w, tk.Checkbutton))
        box.invoke()
        self.assertTrue(request["allowed"])
        self.assertTrue(request["event"].is_set())
        self.assertIsNone(self.ui.prompt_panel)
        self.assertFalse(self.ui.decide_questions_var.get())

    def test_stop_resolves_inline_approval_without_running_command(self):
        self.ui._set_busy(True)
        request = {"command": "Write-Output test", "cwd": str(self.ui.workspace),
                   "event": threading.Event(), "allowed": False}
        self.ui._show_approval(request)
        self.ui.stop()
        self.assertTrue(request["event"].is_set())
        self.assertFalse(request["allowed"])
        self.assertIsNone(self.ui.approval_window)

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

    def model_info(self, info, *, model=None, serial=None):
        self.ui._handle_event("model_info", {
            "model": model or self.ui.model_var.get(),
            "serial": self.ui.model_check_serial if serial is None else serial,
            "info": info,
        }, None)

    def test_profile_is_saved_and_model_controls_wait_for_task_completion(self):
        self.ui.profile_var.set(app.PROFILE_LABELS["rtx4070_laptop"])
        self.ui._profile_changed()
        self.assertEqual(json.loads(app.STATE.read_text())["performance_profile"], "rtx4070_laptop")
        self.ui.performance_profile = "auto"
        self.ui._load_state()
        self.assertEqual(self.ui.performance_profile, "rtx4070_laptop")
        self.ui.import_path = "example.gguf"
        self.ui._set_busy(True)
        for widget in (self.ui.profile_box, self.ui.download_name_entry, self.ui.custom_download_button,
                       self.ui.import_name_entry, self.ui.choose_gguf_button, self.ui.import_button):
            self.assertEqual(str(widget.cget("state")), "disabled")
        self.assertEqual(self.ui.input.cget("state"), "normal")
        self.ui._set_busy(False)
        self.assertEqual(str(self.ui.profile_box.cget("state")), "readonly")
        self.assertEqual(self.ui.import_button.cget("state"), "normal")

    def test_model_capabilities_block_tools_and_disable_unsupported_thinking(self):
        self.ui.think_var.set(True)
        self.model_info({"capabilities": ["completion"], "capabilities_known": True, "context_length": 16384})
        self.assertFalse(self.ui.think_var.get())
        self.assertEqual(self.ui.think_box.cget("state"), "disabled")
        self.assertEqual(self.ui.send_button.cget("state"), "disabled")
        self.ui.input.insert("1.0", "Edit a file")
        with patch.object(self.ui, "_start_run") as start:
            self.ui.send()
            start.assert_not_called()
        self.assertIn("does not support tools", self.ui.status_label.cget("text"))
        self.model_info({"capabilities": ["completion", "tools"], "context_length": 8192})
        self.assertEqual(self.ui.send_button.cget("state"), "normal")
        self.assertIn("Reason more is unavailable", self.ui.model_capability_label.cget("text"))

    def test_small_context_blocked_and_unknown_capabilities_explained(self):
        self.model_info({"capabilities": ["tools", "thinking"], "context_length": 4096})
        self.assertFalse(self.ui._can_send())
        self.assertIn("8,192", self.ui.model_capability_label.cget("text"))
        self.model_info({"capabilities": [], "capabilities_known": False})
        self.assertTrue(self.ui._can_send())
        self.assertIn("Tool support is unknown", self.ui.model_capability_label.cget("text"))
        self.model_info({"capabilities": [], "capabilities_known": True})
        self.assertFalse(self.ui._can_send())

    def test_late_model_info_cannot_overwrite_new_selection(self):
        self.ui.model_check_serial = 4
        self.ui.model_checking = True
        self.model_info({"capabilities": ["completion"]}, serial=3)
        self.assertTrue(self.ui.model_checking)
        self.assertIsNone(self.ui.model_tool_support)
        self.model_info({"capabilities": ["completion"]}, model="old-model:latest")
        self.assertTrue(self.ui.model_checking)
        self.model_info({"capabilities": ["tools", "thinking"], "context_length": 16384})
        self.assertFalse(self.ui.model_checking)
        self.assertTrue(self.ui.model_tool_support)

    def test_hardware_label_reports_real_memory_without_estimated_speed(self):
        self.ui._handle_event("hardware_info", {"gpus": [{"name": "NVIDIA RTX 5060",
                              "total_mb": 8192, "free_mb": 5632, "used_mb": 2560}]}, None)
        label = self.ui.hardware_label.cget("text")
        self.assertIn("NVIDIA RTX 5060", label)
        self.assertIn("5.5 / 8.0 GB", label)
        self.assertNotIn("tokens", label)
        self.assertIsNone(self.ui.last_tokens_per_second)

    def wait_for_model_operation(self):
        deadline = time.monotonic() + 3
        while self.ui.downloading and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.ui.downloading, "Model operation did not finish")

    def test_custom_download_requires_click_and_selects_downloaded_model(self):
        with patch("runtime.pull_model") as download, patch.object(self.ui, "connect") as connect:
            self.ui.toggle_settings()
            self.ui.download_name_var.set("custom-model:4b")
            self.ui.profile_var.set(app.PROFILE_LABELS["rtx5080"])
            self.ui._profile_changed()
            download.assert_not_called()
            self.ui.custom_download_button.invoke()
            self.wait_for_model_operation()
            download.assert_called_once()
            self.assertEqual(download.call_args.args, ("custom-model:4b",))
            connect.assert_called_once()
        self.ui._handle_event("connected", {"base_url": self.ui.base_url,
                              "models": [app.DEFAULT_MODEL, "custom-model:4b"]}, None)
        self.assertEqual(self.ui.model_var.get(), "custom-model:4b")
        self.assertEqual(self.ui.setup_bar.winfo_manager(), "")

    def test_gguf_file_selection_waits_for_explicit_import(self):
        gguf = self.data / "weights.gguf"
        with patch("app.filedialog.askopenfilename", return_value=str(gguf)), \
             patch("runtime.import_gguf") as importer, patch.object(self.ui, "connect") as connect:
            self.ui.choose_gguf()
            importer.assert_not_called()
            self.ui.import_name_var.set("my-agent")
            self.ui.import_button.invoke()
            self.wait_for_model_operation()
            importer.assert_called_once()
            self.assertEqual(importer.call_args.args, ("my-agent", str(gguf)))
            connect.assert_called_once()
        self.ui._handle_event("connected", {"base_url": self.ui.base_url,
                              "models": [app.DEFAULT_MODEL, "my-agent:latest"]}, None)
        self.assertEqual(self.ui.model_var.get(), "my-agent:latest")

    def test_model_operation_error_restores_controls(self):
        self.ui.downloading = True
        self.ui.pending_model_selection = "broken-model"
        self.ui.model_operation_label = "Importing model"
        self.ui._handle_event("model_download_error", {"text": "That file could not be imported."}, None)
        self.assertFalse(self.ui.downloading)
        self.assertIsNone(self.ui.pending_model_selection)
        self.assertEqual(self.ui.update_button.cget("state"), "normal")
        self.assertEqual(self.ui.custom_download_button.cget("state"), "normal")
        self.assertIn("That file could not be imported.", self.ui.transcript.get("1.0", "end"))

    def attachment(self, name="example.txt", content="Reference text"):
        source = self.data / name
        source.write_text(content, encoding="utf-8")
        return runtime.prepare_attachment(source)

    def test_attachment_only_send_passes_metadata_and_clears_chips(self):
        item = self.attachment()
        self.ui._add_attachments([item])
        self.assertEqual(self.ui.pending_frame.winfo_manager(), "pack")
        with patch.object(self.ui, "_start_run") as start:
            self.ui.send()
            start.assert_called_once_with("", attachments=[item])
        self.assertFalse(self.ui.pending_attachments)
        self.assertEqual(self.ui.pending_frame.winfo_manager(), "")
        self.assertTrue(any(isinstance(widget, tk.Label) and widget.cget("text") == "example.txt"
                            for widget in self.descendants(self.ui.transcript)))

    def test_busy_attachments_queue_and_stop_discards_them(self):
        item = self.attachment()
        self.ui._set_busy(True)
        self.ui._add_attachments([item])
        self.ui.input.insert("1.0", "Read this too")
        self.ui.send()
        queued = self.ui.incoming.get_nowait()
        self.assertEqual(queued, {"content": "Read this too", "attachments": [item]})
        self.ui.incoming.put(queued)
        self.ui.stop()
        self.assertTrue(self.ui.incoming.empty())
        self.assertIn("queued message(s) were not sent", self.ui.transcript.get("1.0", "end"))

    def test_late_attachment_queue_keeps_each_message_and_attachment_pair(self):
        first = self.attachment("first.txt", "First")
        second = self.attachment("second.txt", "Second")
        self.ui.busy = True
        self.ui.incoming.put({"content": "First request", "attachments": [first]})
        self.ui.incoming.put({"content": "Second request", "attachments": [second]})
        with patch.object(self.ui, "_start_run") as start:
            self.ui._handle_event("worker_complete", {"history": []}, 0)
            start.assert_called_once_with("First request", attachments=[first], continuation=True)
        self.assertEqual(self.ui.incoming.get_nowait(), {"content": "Second request", "attachments": [second]})
        self.assertEqual(self.ui.queued_count, 1)

    def test_known_nonvision_model_keeps_unsent_images_and_explains_inline(self):
        image = {"id": "a" * 64, "name": "picture.png", "kind": "image", "path": "a" * 64 + ".png"}
        self.ui._add_attachments([image])
        self.ui.model_vision_support = False
        self.ui.input.insert("1.0", "Describe this")
        with patch.object(self.ui, "_start_run") as start:
            self.ui.send()
            start.assert_not_called()
        self.assertIn("cannot read images", self.ui.status_label.cget("text"))
        self.assertEqual(self.ui.pending_attachments, [image])
        self.assertEqual(self.ui.input.get("1.0", "end-1c"), "Describe this")

    def test_smaller_context_limits_images_without_consuming_the_message(self):
        images = [{"id": digit * 64, "name": digit + ".png", "kind": "image", "path": digit * 64 + ".png"}
                  for digit in "ab"]
        self.ui._add_attachments(images)
        self.ui.model_vision_support = True
        self.ui.model_context_tokens = 8192
        with patch.object(self.ui, "_start_run") as start:
            self.ui.send()
            start.assert_not_called()
        self.assertIn("one image per message", self.ui.status_label.cget("text"))
        self.assertEqual(len(self.ui.pending_attachments), 2)

    def test_chips_remove_clear_and_ignore_previous_chat_preparation(self):
        first, second = self.attachment("a.txt", "A"), self.attachment("b.txt", "B")
        self.ui._add_attachments([first, second, first])
        self.assertEqual(len(self.ui.pending_attachments), 2)
        self.ui._remove_attachment(first["id"])
        self.assertEqual(self.ui.pending_attachments, [second])
        old_epoch = self.ui.attachment_epoch
        self.ui.new_chat()
        self.ui._handle_event("attachments_prepared", {"epoch": old_epoch, "items": [first]}, None)
        self.assertFalse(self.ui.pending_attachments)
        self.assertEqual(self.ui.attachment_jobs, 0)

    def wait_for_attachments(self):
        deadline = time.monotonic() + 5
        while self.ui.attachment_jobs and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertEqual(self.ui.attachment_jobs, 0, "Attachment preparation did not finish")

    def test_clipboard_plain_text_replaces_selection_without_attachment(self):
        self.ui.input.insert("1.0", "replace me")
        self.ui.input.tag_add("sel", "1.0", "end-1c")
        with patch.object(self.root, "clipboard_get", return_value="pasted text"), \
             patch("runtime.clipboard_attachments", return_value=[]):
            self.assertEqual(self.ui._paste_input(), "break")
            self.wait_for_attachments()
        self.assertEqual(self.ui.input.get("1.0", "end-1c"), "pasted text")
        self.assertFalse(self.ui.pending_attachments)

    def test_clipboard_files_are_attached_instead_of_pasted_as_text(self):
        item = self.attachment()
        with patch.object(self.root, "clipboard_get", return_value="ignored file name"), \
             patch("runtime.clipboard_attachments", return_value=[item]):
            self.ui._paste_input()
            self.wait_for_attachments()
        self.assertEqual(self.ui.pending_attachments, [item])
        self.assertEqual(self.ui.input.get("1.0", "end-1c"), "")

    def test_plain_text_paste_keeps_position_while_clipboard_worker_finishes(self):
        release = threading.Event()
        def read_clipboard():
            release.wait(2)
            return []
        with patch.object(self.root, "clipboard_get", return_value="pasted "), \
             patch("runtime.clipboard_attachments", side_effect=read_clipboard):
            self.ui._paste_input()
            self.ui.input.insert("insert", "typed later")
            release.set()
            self.wait_for_attachments()
        self.assertEqual(self.ui.input.get("1.0", "end-1c"), "pasted typed later")

    def test_transcript_menu_copies_and_composer_menu_pastes(self):
        event = SimpleNamespace(x_root=0, y_root=0)
        with patch("app.tk.Menu") as menu_type, patch.object(self.ui.transcript, "event_generate") as generate:
            menu = menu_type.return_value
            self.ui._text_menu(event, self.ui.transcript)
            commands = {call.kwargs["label"]: call.kwargs["command"] for call in menu.add_command.call_args_list}
            commands["Copy"]()
            generate.assert_called_once_with("<<Copy>>")
            self.assertNotIn("Paste", commands)
        with patch("app.tk.Menu") as menu_type, patch.object(self.ui, "_paste_input") as paste:
            self.ui._text_menu(event, self.ui.input, editable=True)
            commands = {call.kwargs["label"]: call.kwargs["command"] for call in menu_type.return_value.add_command.call_args_list}
            commands["Paste"]()
            paste.assert_called_once()

    def test_two_quick_pastes_keep_their_order(self):
        with patch.object(self.root, "clipboard_get", side_effect=["first ", "second"]), \
             patch.object(self.ui, "_prepare_attachments") as prepare:
            self.ui._paste_input()
            self.ui._paste_input()
        requests = [call.kwargs for call in prepare.call_args_list]
        for request in requests:
            self.ui._handle_event("attachments_prepared", {"epoch": self.ui.attachment_epoch, "items": [],
                                  "paste": request["fallback_text"], "paste_marks": request["paste_marks"]}, None)
        self.assertEqual(self.ui.input.get("1.0", "end-1c"), "first second")

    def test_oversized_selection_is_rejected_before_any_file_preparation(self):
        self.ui._add_attachments([self.attachment()])
        with patch("runtime.prepare_attachment") as prepare:
            self.ui._prepare_attachments([str(self.data / f"file{index}.txt") for index in range(100)])
            prepare.assert_not_called()
        self.assertEqual(self.ui.attachment_jobs, 0)
        self.assertIn("up to 7 more", self.ui.status_label.cget("text"))

    def test_unsupported_file_preparation_fails_inline_without_clearing_text(self):
        source = self.data / "paper.pdf"
        source.write_bytes(b"%PDF-test")
        self.ui.input.insert("1.0", "Keep my message")
        self.ui._prepare_attachments([str(source)])
        self.wait_for_attachments()
        self.assertFalse(self.ui.pending_attachments)
        self.assertIn("PDF reading is not available", self.ui.transcript.get("1.0", "end"))
        self.assertEqual(self.ui.input.get("1.0", "end-1c"), "Keep my message")

    def test_stored_chat_shows_original_prompt_and_missing_file_error(self):
        item = self.attachment(content="DO NOT DISPLAY THE EXPANDED FILE CONTENTS")
        self.ui.history = [runtime.build_user_message("Review this file", [item])]
        self.ui._save_state()
        self.ui.history = []
        self.ui._load_state()
        self.ui._render_history()
        transcript = self.ui.transcript.get("1.0", "end")
        self.assertIn("Review this file", transcript)
        self.assertNotIn("DO NOT DISPLAY THE EXPANDED FILE CONTENTS", transcript)
        runtime.resolve_attachment_path(item).unlink()
        self.ui._open_attachment(item)
        self.assertIn("missing", self.ui.status_label.cget("text"))

    def test_script_attachment_opens_in_notepad_without_execution(self):
        item = self.attachment("example.ps1", "Write-Output 'Do not run this'")
        with patch("app.subprocess.Popen") as launch, patch("app.os.startfile") as startfile:
            self.ui._open_attachment(item)
            launch.assert_called_once()
            command = launch.call_args.args[0]
            self.assertEqual(Path(command[0]).name.lower(), "notepad.exe")
            self.assertEqual(command[1], str(runtime.resolve_attachment_path(item)))
            startfile.assert_not_called()

    def test_saved_image_label_cannot_execute_a_script_attachment(self):
        item = self.attachment("pretend-image.ps1", "Write-Output test")
        item["kind"] = "image"
        with patch("app.os.startfile") as startfile:
            self.ui._open_attachment(item)
            startfile.assert_not_called()
        self.assertIn("not an image", self.ui.status_label.cget("text"))

    def test_actual_image_normalization_and_inline_preview(self):
        source = self.data / "small.png"
        picture = tk.PhotoImage(width=32, height=24)
        picture.put("#4ca889", to=(0, 0, 32, 24))
        picture.write(str(source), format="png")
        item = runtime.prepare_attachment(source)
        self.assertTrue(runtime.resolve_attachment_path(item, preview=True).is_file())
        self.ui._add_attachments([item])
        self.assertEqual(len(self.ui.pending_previews), 1)
        self.ui._render_attachments([item])
        self.assertEqual(len(self.ui.chat_previews), 1)
        self.assertLessEqual(self.ui.chat_previews[0].width(), 220)

    def test_coding_preset_fills_exact_model_without_download(self):
        with patch("runtime.pull_model") as download:
            self.ui.coding_preset_var.set("Coding + images · Qwen 3.5 9B")
            self.ui._coding_preset_changed()
            download.assert_not_called()
        self.assertEqual(self.ui.download_name_var.get(), "qwen3.5:9b-q4_K_M")
        self.assertIn("8 GB", self.ui.coding_preset_note.cget("text"))

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
        self.ui.profile_var.set(app.PROFILE_LABELS["rtx4070_laptop"])
        with patch.object(Agent, "_chat", side_effect=responses), patch.object(self.ui, "_show_question", side_effect=respond):
            self.ui.input.insert("1.0", "Help me choose a language")
            self.ui.send()
            deadline = time.monotonic() + 5
            while self.ui.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
            self.assertFalse(self.ui.busy, "The question callback did not finish")
            self.assertTrue(self.ui.agent.think)
            self.assertEqual(self.ui.agent.performance_profile, "rtx4070_laptop")
        saved = json.loads(app.STATE.read_text())
        result = next(item for item in saved["history"] if item.get("tool_name") == "ask_user")
        self.assertEqual(json.loads(result["content"])["answer"], "Python")
        self.ui._render_history()
        self.assertIn("Which language?", self.ui.transcript.get("1.0", "end"))
        self.assertIn("YOU · ANSWER", self.ui.transcript.get("1.0", "end"))

    def test_attachment_reaches_agent_and_is_retained_in_saved_history(self):
        from agent_core import Agent
        item = self.attachment(content="A small code reference")
        self.ui.model_vision_support = True
        self.ui._add_attachments([item])
        with patch.object(Agent, "_chat", return_value={"role": "assistant", "content": "I read the attachment."}):
            self.ui.send()
            deadline = time.monotonic() + 5
            while self.ui.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
            self.assertFalse(self.ui.busy, "Attachment task did not finish")
        saved = json.loads(app.STATE.read_text())
        message = next(entry for entry in saved["history"] if entry.get("attachments"))
        self.assertEqual(message["attachments"][0]["id"], item["id"])
        self.assertEqual(message["user_text"], "")
        self.assertIn("A small code reference", message["content"])
        self.assertTrue(self.ui.agent.vision_supported)

    def test_attachment_preflight_failure_restores_retry_draft(self):
        from agent_core import Agent
        item = self.attachment()
        self.ui._add_attachments([item])
        self.ui.input.insert("1.0", "Read the file")
        with patch.object(Agent, "run", side_effect=ValueError("Attachment rejected before sending")):
            self.ui.send()
            deadline = time.monotonic() + 5
            while self.ui.busy and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
        self.assertFalse(self.ui.busy)
        self.assertEqual(self.ui.pending_attachments, [item])
        self.assertEqual(self.ui.input.get("1.0", "end-1c"), "Read the file")
        self.assertIn("returned to the composer", self.ui.status_label.cget("text"))


if __name__ == "__main__":
    unittest.main()
