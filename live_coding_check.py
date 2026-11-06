"""Opt-in local-model coding check. Not part of unittest discovery or the app package."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

from agent_core import Agent
import runtime


SOURCE = "def total_with_tax(amount, tax_rate):\n    return amount + tax_rate\n"
TESTS = """import unittest
from math_utils import total_with_tax

class TaxTests(unittest.TestCase):
    def test_percentage(self):
        self.assertAlmostEqual(total_with_tax(100, 0.2), 120)
    def test_zero(self):
        self.assertAlmostEqual(total_with_tax(37, 0), 37)
    def test_fractional(self):
        self.assertAlmostEqual(total_with_tax(12.5, 0.08), 13.5)
"""


def run_check(model: str) -> dict:
    events, approvals = [], []
    cancel = threading.Event()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="local-agent-coding-check-") as temporary:
        project = Path(temporary).resolve()
        source = project / "math_utils.py"
        tests = project / "test_math_utils.py"
        source.write_text(SOURCE, encoding="utf-8")
        tests.write_text(TESTS, encoding="utf-8")

        def safe_code():
            # This check intentionally asks for simple arithmetic, so the test
            # process need not run imports, calls, or new test modules authored
            # by the model.
            if tests.read_text(encoding="utf-8") != TESTS:
                return False
            if set(project.glob("test*.py")) != {tests}:
                return False
            try:
                tree = ast.parse(source.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                return False
            allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
                       ast.Expr, ast.Constant, ast.Name, ast.Load, ast.Store, ast.BinOp,
                       ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UnaryOp, ast.USub, ast.UAdd)
            return all(isinstance(node, allowed) for node in ast.walk(tree))

        allowed_commands = {"python -m unittest", "python -m unittest -v",
                            "python -m unittest test_math_utils", "python -m unittest test_math_utils -v",
                            "python -m unittest -v test_math_utils", "python -m unittest test_math_utils.py",
                            "python -m unittest test_math_utils.py -v", "python -m unittest -v test_math_utils.py"}
        def approve(command, cwd):
            allowed = Path(cwd).resolve() == project and " ".join(command.strip().split()) in allowed_commands and safe_code()
            approvals.append({"command": command, "allowed": allowed})
            return allowed

        agent = Agent(project, lambda kind, data: events.append((kind, data)), approve, cancel,
                      model=model, performance_profile="auto", think=False)
        timer = threading.Timer(360, cancel.set)
        timer.start()
        try:
            with patch("agent_core.MAX_RUN_STEPS", 8), patch("agent_core.MAX_RUN_SECONDS", 360):
                agent.run("Fix the bug in math_utils.py: total_with_tax(amount, tax_rate) must add amount multiplied by "
                          "the fractional tax rate to the amount (100 at 0.2 must return 120). Read the source and tests, "
                          "use edit_file for the focused fix, then run python -m unittest -v and report the actual result. "
                          "Do not modify test_math_utils.py or create extra files. All work is confined to this temporary project.")
        finally:
            timer.cancel()
        checked = None
        if safe_code():
            checked = subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=project, capture_output=True,
                                     text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tool_names = [data.get("name") for kind, data in events if kind == "tool"]
        model_test_passed = any(kind == "tool_result" and data.get("name") == "run_command"
                                and data.get("result", {}).get("exit_code") == 0 for kind, data in events)
        result = {"model": model, "passed": bool(checked and checked.returncode == 0 and model_test_passed
                                                  and "read_file" in tool_names and "edit_file" in tool_names),
                  "elapsed_seconds": round(time.monotonic() - started, 2), "tools": tool_names,
                  "approvals": approvals, "source": source.read_text(encoding="utf-8"),
                  "external_test_returncode": checked.returncode if checked else None,
                  "external_test_output": (checked.stdout + checked.stderr)[-4000:] if checked else "Code preflight failed",
                  "errors": [data.get("text") for kind, data in events if kind == "error"],
                  "final": next((item.get("content") for item in reversed(agent.history) if item.get("role") == "assistant"), ""),
                  "metrics": [data for kind, data in events if kind == "metrics"]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    args = parser.parse_args()
    result = run_check(args.model)
    output = runtime.ROOT / "data" / "live-coding-check.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
