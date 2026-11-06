"""Validate a relocated copy without installed Python or Codex on PATH."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dist" / "Local-Agent-Windows-x64"
TARGET = ROOT / "dist" / "portability-check" / "A different folder" / "Local Agent"


def main():
    if not TARGET.exists():
        print("Copying to a different folder with spaces", flush=True)
        shutil.copytree(SOURCE, TARGET)
    executable = TARGET / "runtime" / "python" / "python.exe"
    env = os.environ.copy()
    system_root = os.environ["SYSTEMROOT"]
    env["PATH"] = os.pathsep.join((str(executable.parent), os.path.join(system_root, "System32"),
                                  os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0")))
    env["PYTHONHOME"] = r"C:\Nonexistent-Python-For-Portability-Test"
    env["PYTHONPATH"] = r"C:\Nonexistent-Python-Modules-For-Portability-Test"
    env["LOCAL_AGENT_PORT"] = "11438"
    env["LOCAL_AGENT_GPU_LAYERS"] = "0"
    for flag in ("--check", "--smoke-test"):
        result = subprocess.run([str(executable), "-I", str(TARGET / "bootstrap.py"), flag],
                                cwd=system_root, env=env, capture_output=True, text=True, timeout=60)
        print(f"Bootstrap {flag}: exit {result.returncode} {result.stdout.strip()}", flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr or "Bootstrap check failed")
    code = r'''
import sys, pathlib, json, threading, urllib.request, time
sys.path.insert(0, sys.argv[1])
import bootstrap
bootstrap.prepare_environment()
import runtime
from agent_core import Agent
root = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(sys.prefix).resolve() == root / 'runtime' / 'python'
runtime.ensure_server()
models = runtime.list_models()
assert runtime.MODEL in models, models
workspace = root / 'workspace'
workspace.mkdir(exist_ok=True)
agent = Agent(workspace, lambda *_: None, lambda *_: True, threading.Event(), base_url=runtime.BASE_URL)
command = 'python -I -c "import sys; print(sys.executable)"'
result = agent._run_command(command)
assert result['exit_code'] == 0, result
assert str(root / 'runtime' / 'python' / 'python.exe').lower() in result['output'].lower(), result
body = {'model': runtime.MODEL, 'messages': [{'role': 'user', 'content': 'Reply with exactly: PORTABLE READY'}],
        'think': False, 'stream': False, 'keep_alive': 0,
        'options': {'num_gpu': 0, 'num_ctx': 16384, 'num_predict': 16}}
request = urllib.request.Request(runtime.BASE_URL + '/api/chat', json.dumps(body).encode(), {'Content-Type': 'application/json'})
started = time.monotonic()
with urllib.request.urlopen(request, timeout=240) as response:
    result_chat = json.load(response)
assert 'PORTABLE READY' in result_chat['message']['content'], result_chat
report = {'passed': True, 'isolated_python': sys.executable, 'models': models,
          'command_result': result, 'model_reply': result_chat['message']['content'],
          'model_seconds': round(time.monotonic()-started, 2), 'server': runtime.BASE_URL}
print(json.dumps(report, indent=2), flush=True)
(root.parent.parent / 'validation-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
'''
    try:
        result = subprocess.run([str(executable), "-I", "-u", "-c", code, str(TARGET)],
                                cwd=system_root, env=env, timeout=300)
        if result.returncode:
            raise RuntimeError("Relocated engine/command check failed")
    finally:
        pid_file = TARGET / "data" / "engine.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text())
            subprocess.run(["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                           creationflags=subprocess.CREATE_NO_WINDOW)
    print("Relocated package validation passed", flush=True)


if __name__ == '__main__':
    main()
