"""Start the app's private, local-only Ollama engine. No hosted services required."""
from __future__ import annotations

import json
import csv
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import base64
import hashlib
import zipfile
import xml.etree.ElementTree as ET

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
PERFORMANCE_PROFILES = {
    "auto": "Auto (detected hardware)", "rtx5060": "RTX 5060 desktop",
    "rtx4070_laptop": "RTX 4070 laptop", "rtx5080": "RTX 5080 desktop", "cpu": "CPU only",
}
_hardware_cache = (0.0, None)
_hardware_lock = threading.Lock()
_option_cache = {}
_option_lock = threading.Lock()


def _validate_model_name(model: str) -> str:
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?", model):
        raise ValueError("Enter an Ollama model name such as qwen3.5:4b-q4_K_M.")
    if model.lower().endswith((":cloud", "-cloud")) or any(part in {"", ".", ".."} for part in model.split("/")):
        raise ValueError("Choose a local model; cloud models are disabled.")
    return model


def _post(path: str, payload: dict, timeout: float = 30):
    request = urllib.request.Request(BASE_URL + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def get_hardware_info() -> dict:
    """Read actual hardware. GPU profile names never invent VRAM or clock rates."""
    global _hardware_cache
    with _hardware_lock:
        if _hardware_cache[1] is not None and time.monotonic() - _hardware_cache[0] < 3:
            return json.loads(json.dumps(_hardware_cache[1]))
        result = {"gpus": [], "cpu": platform.processor() or "Unknown CPU",
                  "ram_total_mb": 0, "ram_available_mb": 0}
        if os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                    result["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                class MemoryStatus(ctypes.Structure):
                    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                        (name, ctypes.c_ulonglong) for name in
                        ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
                memory = MemoryStatus()
                memory.length = ctypes.sizeof(memory)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                    result.update(ram_total_mb=memory.total // 1048576, ram_available_mb=memory.available // 1048576)
            except (OSError, AttributeError):
                pass
        system_smi = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/nvidia-smi.exe"
        smi = str(system_smi) if system_smi.is_file() else shutil.which("nvidia-smi")
        if smi:
            try:
                process = subprocess.run([smi, "--query-gpu=name,memory.total,memory.free,memory.used",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                         timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if process.returncode == 0:
                    for row in csv.reader(process.stdout.splitlines()):
                        if len(row) == 4:
                            result["gpus"].append({"name": row[0].strip(), "total_mb": int(row[1]),
                                                   "free_mb": int(row[2]), "used_mb": int(row[3])})
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        _hardware_cache = (time.monotonic(), result)
        return json.loads(json.dumps(result))


@lru_cache(maxsize=24)
def _model_info(model: str) -> dict:
    response = _post("/api/show", {"model": model})
    if response.get("remote_model") or response.get("remote_host") or "cloud" in response.get("capabilities", []):
        raise ValueError("This model uses a remote service. Choose downloaded local weights.")
    raw = response.get("model_info") or {}
    architecture = raw.get("general.architecture", "")
    def field(name, default=0):
        return raw.get(f"{architecture}.{name}", default)
    blocks = int(field("block_count"))
    recurrent = field("attention.recurrent_layers", [])
    attention_blocks = sum(not item for item in recurrent) if isinstance(recurrent, list) and len(recurrent) == blocks else blocks
    heads = max(1, int(field("attention.head_count", 1)))
    key_length = int(field("attention.key_length", max(1, int(field("embedding_length", 128)) // heads)))
    value_length = int(field("attention.value_length", key_length))
    kv_heads = max(1, int(field("attention.head_count_kv", heads)))
    tags = _get("/api/tags", timeout=10).get("models", [])
    canonical = model if ":" in model else model + ":latest"
    size = next((int(item.get("size", 0)) for item in tags
                 if item.get("name") in {model, canonical} or item.get("model") in {model, canonical}), 0)
    return {"name": model, "size_bytes": size, "block_count": blocks,
            "context_length": int(field("context_length", 16384)),
            "kv_bytes_per_token": attention_blocks * kv_heads * (key_length + value_length) * 2,
            "capabilities": list(response.get("capabilities") or []),
            "capabilities_known": isinstance(response.get("capabilities"), list),
            "details": response.get("details") or {}}


def get_model_info(model: str) -> dict:
    return json.loads(json.dumps(_model_info(_validate_model_name(model))))


def _context_length(metadata: dict) -> int:
    maximum = int(metadata.get("context_length") or 16384)
    if maximum < 8192:
        raise ValueError("This model supports fewer than 8,192 context tokens. Choose a model with at least 8K context for this agent.")
    return min(16384, maximum)


def _calculate_options(model: str, profile: str, hardware: dict, metadata: dict, loaded_vram: int = 0) -> dict:
    context = _context_length(metadata)
    options = {"num_ctx": context, "num_gpu": 0}
    if profile == "cpu" or not hardware.get("gpus"):
        return options
    gpu = hardware["gpus"][0]
    # Add back an existing model allocation once, then keep 512 MiB for the display
    # and other apps. Profile labels never override actual detected memory.
    available_mb = min(float(gpu["total_mb"]), float(gpu["free_mb"]) + loaded_vram / 1048576)
    budget_mb = max(0.0, available_mb - 512)
    blocks = int(metadata.get("block_count") or 0)
    size_mb = int(metadata.get("size_bytes") or 0) / 1048576
    if blocks <= 0 or size_mb <= 0:
        return options
    layers = blocks + 1
    kv_mb = int(metadata.get("kv_bytes_per_token") or 32768) * context / 1048576
    full_mb = size_mb * 1.08 + kv_mb + 512
    if budget_mb >= full_mb:
        options["num_gpu"] = layers
    elif model == MODEL and context == 16384 and blocks == 32:
        # Measured Windows Q4_K_M allocations at 10/20/26 layers: 1488/2253/2760 MiB.
        # Keep margin and cap partial offload at the tested 26 layers. Full offload
        # still requires the separate full-model budget above.
        options["num_gpu"] = max(0, min(26, int((budget_mb - 700) / 84)))
    else:
        per_layer = (size_mb * 1.08 + kv_mb) / layers
        options["num_gpu"] = max(0, min(blocks, int((budget_mb - 512) / per_layer)))
    return options


def get_inference_options(model: str, profile: str = "auto") -> dict:
    model = _validate_model_name(model)
    if profile not in PERFORMANCE_PROFILES:
        raise ValueError("Choose a recognized performance profile.")
    metadata = get_model_info(model)
    context = _context_length(metadata)
    if profile == "cpu":
        return {"num_ctx": context, "num_gpu": 0}
    override = os.environ.get("LOCAL_AGENT_GPU_LAYERS")
    if override is not None:
        try:
            return {"num_ctx": context, "num_gpu": max(0, min(999, int(override)))}
        except ValueError:
            pass
    hardware = get_hardware_info()
    loaded = _get("/api/ps", timeout=5).get("models", [])
    # With several GPUs, do not attribute an unknown allocation to the first card.
    loaded_vram = sum(int(item.get("size_vram") or 0) for item in loaded) if len(hardware.get("gpus", [])) == 1 else 0
    proposed = _calculate_options(model, profile, hardware, metadata, loaded_vram)
    key = (model, profile)
    with _option_lock:
        previous = _option_cache.get(key)
        # A reduced budget takes effect immediately. Small upward fluctuations wait
        # 60s so every model turn does not trigger an expensive model reload.
        if previous and proposed["num_gpu"] > previous[1]["num_gpu"] and time.monotonic() - previous[0] < 60:
            return dict(previous[1])
        if previous and proposed == previous[1]:
            return dict(proposed)
        _option_cache[key] = (time.monotonic(), dict(proposed))
    return proposed


def import_gguf(model: str, path: str | Path, *, progress=None) -> None:
    """Import user-selected local GGUF weights through the bundled official CLI."""
    model = _validate_model_name(model)
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".gguf":
        raise ValueError("Select a local .gguf model file.")
    with source.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("The selected file does not have a GGUF header.")
    if any(character in str(source) for character in ('"', "\r", "\n")):
        raise ValueError("The model path contains unsupported quote or newline characters.")
    ensure_server()
    existing = set(list_models())
    canonical = model if ":" in model else model + ":latest"
    if model in existing or canonical in existing:
        raise ValueError("That model name already exists. Choose a new import name.")
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OLLAMA_HOST=f"127.0.0.1:{PORT}", OLLAMA_MODELS=str(ROOT / "models"), OLLAMA_NO_CLOUD="1")
    with tempfile.TemporaryDirectory(prefix="model-import-", dir=data) as temporary:
        modelfile = Path(temporary) / "Modelfile"
        modelfile.write_text(f'FROM "{source.as_posix()}"\n', encoding="utf-8")
        process = subprocess.Popen([str(OLLAMA), "create", model, "--file", str(modelfile)], env=env,
                                   cwd=temporary, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = []
        try:
            for line in process.stdout:
                text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).strip()
                if text:
                    lines = (lines + [text])[-8:]
                    if progress is not None:
                        progress({"status": text})
            if process.wait() != 0:
                raise RuntimeError("Model import failed: " + "\n".join(lines)[-2000:])
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    _model_info.cache_clear()
    get_model_info(model)


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
    model = _validate_model_name(model)
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
                if sys.stdout is not None:
                    print(state[0], f"{state[1]}%" if state[1] is not None else "", flush=True)
                last = state
    _model_info.cache_clear()
    get_model_info(model)


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
