"""Start the app's private, local-only Ollama engine. No hosted services required."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


def _local_port(value: str | None = None) -> int:
    value = os.environ.get("LOCAL_AGENT_PORT", "11435") if value is None else value
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("LOCAL_AGENT_PORT must be a port number from 1 to 65535.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("LOCAL_AGENT_PORT must be a port number from 1 to 65535.")
    return port


PORT = _local_port()
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL = "qwen3.5:4b-q4_K_M"
OLLAMA = ROOT / "runtime" / "ollama" / "ollama.exe"
_start_lock = threading.Lock()


def _get(path: str, timeout: float = 3):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(BASE_URL + path, timeout=timeout) as response:
        return json.load(response)


def ensure_server() -> str:
    """Return the local endpoint, starting our portable engine when necessary."""
    with _start_lock:
        try:
            _get("/api/version")
            return BASE_URL
        except (OSError, ValueError, urllib.error.URLError):
            pass
        if not OLLAMA.is_file():
            raise RuntimeError("The local AI engine is missing. Restore the Ollama runtime in the app folder.")
        for directory in ("models", "logs", "data", "workspace"):
            (ROOT / directory).mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update({
            "OLLAMA_HOST": f"127.0.0.1:{PORT}",
            "OLLAMA_MODELS": str(ROOT / "models"),
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_CONTEXT_LENGTH": "16384",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_KEEP_ALIVE": "2m",
        })
        with (ROOT / "logs" / "ollama.log").open("ab") as log:
            process = subprocess.Popen(
                [str(OLLAMA), "serve"], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        for _ in range(100):
            try:
                _get("/api/version", timeout=1)
                (ROOT / "data" / "engine.pid").write_text(str(process.pid), encoding="utf-8")
                return BASE_URL
            except (OSError, ValueError, urllib.error.URLError):
                if process.poll() is not None:
                    raise RuntimeError("The local AI engine could not start. See logs/ollama.log.")
                time.sleep(0.2)
        raise RuntimeError("The local AI engine is still starting. Try Connect again shortly.")


def list_models() -> list[str]:
    return [item["name"] for item in _get("/api/tags").get("models", [])
            if not item["name"].endswith(("-cloud", ":cloud"))]


def pull_model(model: str = MODEL, *, progress=None):
    """Download an explicitly chosen model and print sparse setup progress."""
    ensure_server()
    request = urllib.request.Request(
        BASE_URL + "/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last = None
    with opener.open(request, timeout=1800) as response:
        for line in response:
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            if progress is not None:
                progress(event)
            total = event.get("total", 0)
            percent = int(100 * event.get("completed", 0) / total) if total else None
            state = (event.get("status", ""), (percent // 10) * 10 if percent is not None else None)
            if state != last:
                print(state[0], f"{state[1]}%" if state[1] is not None else "", flush=True)
                last = state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull", action="store_true")
    args = parser.parse_args()
    if args.pull:
        pull_model()
    else:
        print(ensure_server())
        print(list_models())
