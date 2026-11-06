"""Behavior tests for workspace access, execution approval, streaming and cancellation."""
from __future__ import annotations

import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_core import Agent, BASE_URL, MODEL, BACKUP_DIR, MAX_TOOL_OUTPUT
import runtime


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events = []
        self.cancel = threading.Event()
        self.agent = Agent(self.root, lambda kind, data: self.events.append((kind, data)),
                           lambda command, cwd: True, self.cancel)

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_outside_paths_and_alternate_streams(self):
        for path in ("../outside.txt", str(self.root.parent / "outside.txt"), "file.txt:stream"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.agent._execute("write_file", {"path": path, "content": "no"})

    def test_rejects_symlink_outside_workspace(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "secret.txt"
            target.write_text("secret", encoding="utf-8")
            link = self.root / "link.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("Windows symlink permission unavailable")
            with self.assertRaises(ValueError):
                self.agent._execute("read_file", {"path": "link.txt"})

    def test_overwrite_preserves_original_backup(self):
        original = self.root / "nested" / "example.txt"
        original.parent.mkdir()
        original.write_bytes(b"old\r\ntext")
        result = self.agent._execute("write_file", {"path": "nested/example.txt", "content": "new\ntext"})
        self.assertEqual(original.read_bytes(), b"new\ntext")
        backup = self.root / result["backup"]
        self.assertEqual(backup.read_bytes(), b"old\r\ntext")
        with self.assertRaises(ValueError):
            self.agent._execute("write_file", {"path": result["backup"], "content": "no"})

    @unittest.skipUnless(os.name == "nt", "Windows junction integration")
    def test_junction_cannot_escape_search_or_file_tools(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret.txt"
            secret.write_text("private-junction-sentinel", encoding="utf-8")
            junction = self.root / "junction"
            result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), outside],
                                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode:
                self.skipTest("Junction creation unavailable")
            try:
                search = self.agent._execute("search_files", {"query": "private-junction-sentinel"})
                self.assertEqual(search["matches"], "")
                self.assertNotIn("junction", self.agent._execute("list_files", {})["entries"])
                with self.assertRaises(ValueError):
                    self.agent._execute("read_file", {"path": "junction/secret.txt"})
                with self.assertRaises(ValueError):
                    self.agent._read(junction / "secret.txt")
                with self.assertRaises(ValueError):
                    self.agent._execute("write_file", {"path": "junction/new.txt", "content": "no"})
            finally:
                os.rmdir(junction)

    def test_replace_requires_unique_exact_text(self):
        path = self.root / "example.txt"
        path.write_text("same same", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.agent._execute("replace_text", {"path": "example.txt", "old": "same", "new": "other"})
        self.assertEqual(path.read_text(), "same same")
        self.agent._execute("replace_text", {"path": "example.txt", "old": "same same", "new": "one"})
        self.assertEqual(path.read_text(), "one")

    def test_listing_depth_one_includes_folders_and_skips_dependencies(self):
        (self.root / "source").mkdir()
        (self.root / "source" / "inside.py").write_text("x=1")
        (self.root / "node_modules").mkdir()
        result = self.agent._execute("list_files", {"max_depth": 1})
        self.assertIn("source/", result["entries"])
        self.assertNotIn("inside.py", result["entries"])
        self.assertNotIn("node_modules", result["entries"])

    def test_read_truncation_is_explicit(self):
        (self.root / "big.txt").write_text("x" * (MAX_TOOL_OUTPUT + 30))
        result = self.agent._execute("read_file", {"path": "big.txt"})
        self.assertTrue(result["truncated"])
        self.assertIn("truncated", result["content"])

    def test_denied_command_never_starts_process(self):
        self.agent.approve = lambda command, cwd: False
        with patch("agent_core.subprocess.Popen") as popen, self.assertRaises(PermissionError):
            self.agent._execute("run_command", {"command": "Write-Output 'test'"})
        popen.assert_not_called()
        self.assertTrue(self.agent._denied)

    def test_cancel_after_approval_never_starts_process(self):
        def approve(command, cwd):
            self.cancel.set()
            return True
        self.agent.approve = approve
        with patch("agent_core.subprocess.Popen") as popen:
            self.agent._chat = lambda: {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_command", "arguments": {"command": "Write-Output 'test'"}}},
                {"function": {"name": "write_file", "arguments": {"path": "never.txt", "content": "no"}}}]}
            self.agent.run("Make a file")
        popen.assert_not_called()
        self.assertFalse((self.root / "never.txt").exists())
        self.assertEqual(sum(item["role"] == "tool" for item in self.agent.history), 2)
        self.assertEqual(self.events[-1][0], "done")

    def test_tool_loop_records_results_and_final_response(self):
        responses = iter([
            {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "write_file", "arguments": {"path": "hello.txt", "content": "hello"}}}]},
            {"role": "assistant", "content": "Created hello.txt."}])
        self.agent._chat = lambda: next(responses)
        self.agent.run("Create hello.txt")
        self.assertEqual((self.root / "hello.txt").read_text(), "hello")
        self.assertEqual([item["role"] for item in self.agent.history], ["user", "assistant", "tool", "assistant"])
        self.assertFalse(any(kind == "error" for kind, _ in self.events))
        self.assertEqual(self.events[-1][0], "done")

    def test_declined_command_pauses_remaining_tools(self):
        self.agent.approve = lambda command, cwd: False
        self.agent._chat = lambda: {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "run_command", "arguments": {"command": "Write-Output 'test'"}}},
            {"function": {"name": "write_file", "arguments": {"path": "never.txt", "content": "no"}}}]}
        self.agent.run("Do a task")
        self.assertFalse((self.root / "never.txt").exists())
        self.assertEqual(sum(item["role"] == "tool" for item in self.agent.history), 2)
        self.assertTrue(any("declined" in data.get("text", "") for kind, data in self.events))

    def test_context_compaction_preserves_saved_turns(self):
        self.agent.history = [{"role": "user", "content": "x" * 45000},
                              {"role": "assistant", "content": "old"},
                              {"role": "user", "content": "new"}]
        self.agent._messages()
        self.assertEqual(self.agent._records()[0]["content"], "x" * 45000)
        self.assertEqual(self.agent._records()[-1], {"role": "user", "content": "new"})
        self.assertTrue(any(kind == "memory" for kind, data in self.events))

    def test_local_only_models_and_endpoint(self):
        for kwargs in ({"model": "example:cloud"}, {"model": "example:9b-cloud"}, {"base_url": "https://cloud.example"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Agent(self.root, lambda *_: None, lambda *_: True, self.cancel, **kwargs)

    def test_portable_command_environment_uses_bundle_and_clears_python_overrides(self):
        bundle = self.root / "runtime" / "python"
        bundle.mkdir(parents=True)
        (bundle / "python.exe").touch()
        (bundle / "Scripts").mkdir()
        with patch("agent_core.APP_ROOT", self.root), patch.dict(os.environ, {
                "PATH": "old-path", "PYTHONHOME": "another-install", "PYTHONPATH": "another-library"}):
            env = self.agent._command_environment()
        self.assertEqual(env["PATH"].split(os.pathsep), [str(bundle), str(bundle / "Scripts"), "old-path"])
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_runtime_port_validation_and_shared_endpoint(self):
        self.assertEqual(BASE_URL, runtime.BASE_URL)
        self.assertEqual(runtime._local_port("11436"), 11436)
        for value in ("0", "65536", "not-a-port", "127.0.0.1:11435"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime._local_port(value)

    def test_question_returns_selected_or_written_answer(self):
        asked = []
        self.agent.ask_user = lambda question, options: asked.append((question, options)) or "Use Python"
        result = self.agent._execute("ask_user", {"question": "Which language?", "options": ["Python", "JavaScript"]})
        self.assertEqual(asked, [("Which language?", ["Python", "JavaScript"])])
        self.assertEqual(result, {"answered": True, "answer": "Use Python"})
        with self.assertRaises(ValueError):
            self.agent._execute("ask_user", {"question": "Too many?", "options": ["1", "2", "3", "4", "5"]})

    def test_unanswered_question_pauses_remaining_actions(self):
        self.agent.ask_user = lambda *_: None
        self.agent._chat = lambda: {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "ask_user", "arguments": {"question": "Which file should I change?"}}},
            {"function": {"name": "write_file", "arguments": {"path": "never.txt", "content": "no"}}}]}
        self.agent.run("Make that change")
        self.assertFalse((self.root / "never.txt").exists())
        self.assertEqual(sum(item["role"] == "tool" for item in self.agent.history), 2)
        self.assertTrue(any("unanswered" in data.get("text", "") for _, data in self.events))
        self.assertEqual(self.events[-1][0], "done")

    def test_stop_interrupts_question_callback(self):
        release = threading.Event()
        self.agent.ask_user = lambda *_: release.wait(5) or "answer"
        self.agent._chat = lambda: {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "ask_user", "arguments": {"question": "Which option?"}}}]}
        timer = threading.Timer(0.2, self.cancel.set)
        timer.start()
        started = time.monotonic()
        try:
            self.agent.run("Ask me")
        finally:
            timer.cancel()
            release.set()
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(self.events[-1][0], "done")
        self.assertFalse(any(kind == "error" for kind, _ in self.events))

    def test_plan_reports_validated_steps(self):
        steps = [{"text": "Inspect", "status": "completed"}, {"text": "Edit and test", "status": "in_progress"}]
        result = self.agent._execute("set_plan", {"steps": steps})
        self.assertEqual(result, {"updated": True, "step_count": 2})
        self.assertIn(("plan", {"steps": steps}), self.events)
        with self.assertRaises(ValueError):
            self.agent._execute("set_plan", {"steps": [{"text": "One", "status": "in_progress"}, {"text": "Two", "status": "in_progress"}]})

    def test_messages_before_model_and_during_final_are_processed(self):
        pending = ["Use blue"]
        snapshots = []
        def poll():
            result = pending[:]
            pending.clear()
            return result
        def chat():
            snapshots.append([item.copy() for item in self.agent.history])
            if len(snapshots) == 1:
                pending.append("Also add a title")
                return {"role": "assistant", "content": "Applied blue."}
            return {"role": "assistant", "content": "Added the title."}
        self.agent.poll_messages = poll
        self.agent._chat = chat
        self.agent.run("Change the design")
        self.assertEqual(len(snapshots), 2)
        self.assertEqual([item["content"] for item in snapshots[0]], ["Change the design", "Use blue"])
        self.assertEqual(snapshots[1][-1], {"role": "user", "content": "Also add a title"})
        self.assertEqual([data["text"] for kind, data in self.events if kind == "steering"], ["Use blue", "Also add a title"])

    def test_steering_waits_for_complete_tool_group(self):
        pending, snapshots = [], []
        def poll():
            result = pending[:]
            pending.clear()
            return result
        original_execute = self.agent._execute
        def execute(name, arguments):
            result = original_execute(name, arguments)
            if name == "write_file":
                pending.append("Now explain the files")
            return result
        def chat():
            snapshots.append([item.copy() for item in self.agent.history])
            if len(snapshots) == 1:
                return {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "write_file", "arguments": {"path": "one.txt", "content": "one"}}},
                    {"function": {"name": "list_files", "arguments": {}}}]}
            return {"role": "assistant", "content": "The file contains one."}
        self.agent.poll_messages = poll
        self.agent._execute = execute
        self.agent._chat = chat
        self.agent.run("Make a file")
        self.assertEqual([item["role"] for item in snapshots[1]], ["user", "assistant", "tool", "tool", "user"])

    def test_missing_or_zero_metrics_do_not_invent_token_speed(self):
        self.agent._emit_metrics({"eval_count": 5, "eval_duration": 0})
        metrics = self.events[-1][1]
        self.assertEqual(metrics["eval_count"], 5)
        self.assertEqual(metrics["eval_duration_seconds"], 0)
        self.assertIsNone(metrics["tokens_per_second"])
        self.assertIsNone(metrics["prompt_eval_count"])
        self.agent._emit_metrics({})
        self.assertIsNone(self.events[-1][1]["eval_count"])

    @unittest.skipUnless(os.name == "nt", "Windows PowerShell integration")
    def test_real_command_and_stop(self):
        result = self.agent._execute("run_command", {"command": "Write-Output 'local-agent-ok'"})
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("local-agent-ok", result["output"])
        timer = threading.Timer(0.5, self.cancel.set)
        timer.start()
        started = time.monotonic()
        try:
            result = self.agent._execute("run_command", {"command": "Start-Sleep -Seconds 30"})
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 6)
        self.assertEqual(result.get("error"), "Stopped by user.")


class StreamTests(unittest.TestCase):
    def run_server(self, handler_function):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.request_payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                handler_function(self)
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_stream_assembles_tools_and_content(self):
        def respond(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "application/x-ndjson")
            handler.end_headers()
            for item in [{"message": {"content": "Hi "}}, {"message": {"content": "there"}},
                         {"message": {"tool_calls": [{"function": {"index": 0, "name": "list_files", "arguments": {}}}]}},
                         {"done": True, "done_reason": "stop"}]:
                handler.wfile.write(json.dumps(item).encode() + b"\n")
        endpoint = self.run_server(respond)
        with tempfile.TemporaryDirectory() as workspace:
            events = []
            agent = Agent(workspace, lambda kind, data: events.append((kind, data)), lambda *_: True,
                          threading.Event(), base_url=endpoint)
            result = agent._chat()
        self.assertEqual(result["content"], "Hi there")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "list_files")
        self.assertEqual([data["text"] for kind, data in events if kind == "token"], ["Hi ", "there"])

    def test_incomplete_stream_is_error(self):
        def respond(handler):
            handler.send_response(200)
            handler.end_headers()
            handler.wfile.write(b'{"message":{"content":"partial"}}\n')
        endpoint = self.run_server(respond)
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(workspace, lambda *_: None, lambda *_: True, threading.Event(), base_url=endpoint)
            with self.assertRaisesRegex(RuntimeError, "ended early"):
                agent._chat()

    def test_actual_token_metrics_and_optional_thinking(self):
        requests = []
        def respond(handler):
            requests.append(handler.request_payload)
            handler.send_response(200)
            handler.end_headers()
            chunks = [{"message": {"thinking": "Check the task."}},
                      {"message": {"content": "Done."}},
                      {"done": True, "eval_count": 12, "eval_duration": 500_000_000,
                       "prompt_eval_count": 32, "total_duration": 2_000_000_000}]
            for chunk in chunks:
                handler.wfile.write(json.dumps(chunk).encode() + b"\n")
        endpoint = self.run_server(respond)
        with tempfile.TemporaryDirectory() as workspace:
            events = []
            agent = Agent(workspace, lambda kind, data: events.append((kind, data)), lambda *_: True,
                          threading.Event(), base_url=endpoint, think=True, ask_user=lambda *_: "Yes")
            result = agent._chat()
        self.assertTrue(requests[0]["think"])
        self.assertIn("ask_user", [tool["function"]["name"] for tool in requests[0]["tools"]])
        self.assertEqual(result["thinking"], "Check the task.")
        self.assertEqual(result["content"], "Done.")
        metrics = [data for kind, data in events if kind == "metrics"]
        self.assertEqual(metrics, [{"eval_count": 12, "eval_duration_seconds": 0.5,
                                    "tokens_per_second": 24.0, "prompt_eval_count": 32,
                                    "total_duration_seconds": 2.0}])
        self.assertEqual([data["text"] for kind, data in events if kind == "token"], ["Done."])

    def test_stop_during_silent_inference_returns_promptly(self):
        release = threading.Event()
        def respond(handler):
            handler.send_response(200)
            handler.end_headers()
            release.wait(5)
        endpoint = self.run_server(respond)
        with tempfile.TemporaryDirectory() as workspace:
            cancel = threading.Event()
            events = []
            agent = Agent(workspace, lambda kind, data: events.append((kind, data)), lambda *_: True,
                          cancel, base_url=endpoint)
            timer = threading.Timer(0.2, cancel.set)
            timer.start()
            started = time.monotonic()
            try:
                agent.run("Hello")
            finally:
                timer.cancel()
                release.set()
            self.assertLess(time.monotonic() - started, 1.5)
            self.assertEqual(events[-1][0], "done")
            self.assertFalse(any(kind == "error" for kind, _ in events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
