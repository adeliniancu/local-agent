"""Exercise the real local model, file tools, and one approved terminal command.

Run only after the model download finishes. Each run uses a new folder beneath
workspace/smoke-test and never edits an existing project. No Codex dependency.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from agent_core import Agent, BASE_URL, MODEL


ROOT = Path(__file__).resolve().parent
CHECKER = '''from pathlib import Path
import importlib.util

path = Path(__file__).with_name("word_count.py")
spec = importlib.util.spec_from_file_location("smoke_word_count", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = [
    ("", 0),
    ("  \\t\\n", 0),
    ("one", 1),
    ("one two three", 3),
    ("  one\\ttwo\\nthree  ", 3),
    ("naive\\u2003cafe", 2),
    ("hello, world!", 2),
]
for value, expected in cases:
    actual = module.count_words(value)
    assert actual == expected, (value, expected, actual)
print("LOCAL_AGENT_SMOKE_PASS: 7 cases")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.timeout < 30:
        parser.error("--timeout must be at least 30 seconds")

    workspace = ROOT / "workspace" / "smoke-test" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    workspace.mkdir(parents=True, exist_ok=False)
    checker = workspace / "check_word_count.py"
    checker.write_text(CHECKER, encoding="utf-8")
    checker_digest = hashlib.sha256(checker.read_bytes()).digest()
    executable = str(Path(sys.executable).resolve()).replace("'", "''")
    command = f"& '{executable}' -I '.\\check_word_count.py'"
    events = []
    approvals = []
    cancel = threading.Event()
    started = time.monotonic()

    def emit(name, payload):
        events.append({"event": name, "payload": payload})
        if name == "tool":
            arguments = payload.get("arguments", {})
            target = arguments.get("path", arguments.get("command", ""))
            print(f"TOOL {payload.get('name')}: {target}", flush=True)
        elif name in {"error", "status"}:
            print(f"{name.upper()}: {payload.get('text', '')}", flush=True)
        elif name == "tool_result" and payload.get("name") == "run_command":
            print("COMMAND RESULT: " + json.dumps(payload.get("result")), flush=True)

    def approve(proposed, cwd):
        allowed = (
            proposed.strip() == command
            and Path(cwd).resolve() == workspace.resolve()
            and checker.is_file()
            and hashlib.sha256(checker.read_bytes()).digest() == checker_digest
            and not cancel.is_set()
        )
        approvals.append({"command": proposed, "cwd": cwd, "allowed": allowed})
        print("APPROVAL: " + ("allowed exact test command" if allowed else "rejected unexpected command"), flush=True)
        return allowed

    prompt = (
        "Complete this small coding task using your tools. First read check_word_count.py. "
        "Create word_count.py in this folder defining count_words(text: str) -> int. "
        "It must count whitespace-separated words, including tabs, newlines, and Unicode whitespace. "
        "Empty or whitespace-only input returns zero. Use only Python's standard library. "
        "Do not modify check_word_count.py. After creating the module, use run_command to run "
        "this exact PowerShell command, with no additions or alterations:\n"
        + command
        + "\nThen summarize the actual result. Do not merely print proposed code or commands."
    )
    timer = threading.Timer(args.timeout, cancel.set)
    timer.daemon = True
    agent = Agent(workspace, emit, approve, cancel, model=args.model, base_url=BASE_URL)
    print(f"WORKSPACE: {workspace}", flush=True)
    timer.start()
    try:
        agent.run(prompt)
    finally:
        timer.cancel()

    writes = [item["payload"]["result"] for item in events
              if item["event"] == "tool_result"
              and item["payload"].get("name") == "write_file"]
    executions = [item["payload"]["result"] for item in events
                  if item["event"] == "tool_result"
                  and item["payload"].get("name") == "run_command"]
    errors = [item["payload"].get("text") for item in events if item["event"] == "error"]
    checks = {
        "module_created_by_tool": any(item.get("path") == "word_count.py" and "error" not in item for item in writes),
        "module_exists": (workspace / "word_count.py").is_file(),
        "checker_unchanged": checker.is_file() and hashlib.sha256(checker.read_bytes()).digest() == checker_digest,
        "exact_command_approved": any(item["allowed"] for item in approvals),
        "no_unexpected_commands": all(item["allowed"] for item in approvals),
        "command_passed_seven_cases": any(item.get("exit_code") == 0
            and "LOCAL_AGENT_SMOKE_PASS: 7 cases" in item.get("output", "")
            and "error" not in item for item in executions),
        "no_agent_error": not errors,
        "finished_before_timeout": not cancel.is_set(),
    }
    passed = all(checks.values())
    report = {"passed": passed, "checks": checks, "elapsed_seconds": round(time.monotonic() - started, 2),
              "model": args.model, "workspace": str(workspace), "errors": errors,
              "approvals": approvals, "events": events, "history": agent.history}
    (workspace / "smoke-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "checks", "elapsed_seconds", "workspace")}, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
