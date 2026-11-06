"""Install the explicitly selected experimental model into an existing Local Agent."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
import urllib.request

MODEL = "qwen3.5:4b-local-agent-trained"
RELEASE_PREFIX = "https://github.com/adeliniancu/local-agent/releases/download/v2.0.100/"
PARAMETERS = {"top_p": 0.95, "presence_penalty": 1.5, "temperature": 1, "top_k": 20}
MAX_BYTES = 8 * 1024**3
MIN_FREE_BYTES = 8 * 1024**3


class InstallError(Exception):
    pass


class Cancelled(InstallError):
    pass


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InstallError("Duplicate manifest field")
        result[key] = value
    return result


def load_manifest(path):
    raw = Path(path).read_bytes()
    if len(raw) > 65536:
        raise InstallError("Model manifest is too large")
    return validate_manifest(json.loads(raw, object_pairs_hook=unique_json))


def validate_manifest(value):
    required = {"schema", "model", "label", "files", "parameters", "renderer", "parser", "license_file", "quality", "experimental"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise InstallError("Incomplete model manifest")
    if type(value["schema"]) is not int or value["schema"] != 1 or value["model"] != MODEL or value["experimental"] is not True:
        raise InstallError("This installer supports only the experimental trained 4B pilot")
    if value["renderer"] != "qwen3.5" or value["parser"] != "qwen3.5" or value["license_file"] != "LICENSE.txt":
        raise InstallError("Unexpected model format")
    if (not isinstance(value["label"], str) or len(value["label"]) > 150
            or any(ord(c) < 32 for c in value["label"])):
        raise InstallError("Invalid model label")
    params = value["parameters"]
    if (not isinstance(params, dict) or params != PARAMETERS
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in params.values())):
        raise InstallError("Unexpected model parameters")
    quality = value["quality"]
    if (not isinstance(quality, dict) or quality != {"trained_passed": 1, "reference_passed": 2, "tasks": 3}
            or any(type(v) is not int for v in quality.values())):
        raise InstallError("Missing experimental quality disclosure")
    files = value["files"]
    if not isinstance(files, list) or len(files) != 2 or {f.get("name") for f in files if isinstance(f, dict)} != {"model.gguf", "mmproj.gguf"}:
        raise InstallError("Model package must contain a language model and image projector")
    assets = set()
    for item in files:
        if type(item.get("size")) is not int or not 4 <= item["size"] <= MAX_BYTES or not digest_ok(item.get("sha256")):
            raise InstallError("Invalid model file size or checksum")
        parts = item.get("parts")
        if not isinstance(parts, list) or not 1 <= len(parts) <= 8:
            raise InstallError("Invalid model file parts")
        for part in parts:
            if not isinstance(part, dict):
                raise InstallError("Invalid download part")
            name = part.get("name", "")
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,139}", name)
                    or ".." in name or name in assets or part.get("url") != RELEASE_PREFIX + name):
                raise InstallError("Download address or filename is not trusted")
            assets.add(name)
            if type(part.get("size")) is not int or not 1 <= part["size"] < 2 * 1024**3 or not digest_ok(part.get("sha256")):
                raise InstallError("Invalid download part size or checksum")
        if sum(part["size"] for part in parts) != item["size"]:
            raise InstallError("Parts do not add up to the model file size")
    if sum(item["size"] for item in files) > MAX_BYTES:
        raise InstallError("Model package is too large")
    return value


def digest_ok(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_cancel(cancel):
    if cancel.is_set():
        raise Cancelled("Download cancelled. Your original models were kept.")


class TrustedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "https" or parsed.hostname not in {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
                or parsed.username or parsed.password or parsed.fragment):
            raise InstallError("The download redirected to an untrusted address")
        return super().redirect_request(request, fp, code, message, headers, url)


def open_download(url):
    return urllib.request.build_opener(TrustedRedirects()).open(
        urllib.request.Request(url, headers={"User-Agent": "Local-Agent-Model-Installer"}), timeout=60)


def file_matches(path, item, cancel):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != item["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            check_cancel(cancel)
            digest.update(chunk)
    return digest.hexdigest() == item["sha256"]


def download_files(manifest, folder, app_dir, cancel, progress, *, opener=open_download):
    paths, completed = {}, 0
    total = sum(item["size"] for item in manifest["files"])
    for item in manifest["files"]:
        check_cancel(cancel)
        cached = app_dir / "models" / "blobs" / ("sha256-" + item["sha256"])
        if file_matches(cached, item, cancel):
            paths[item["name"]] = cached.resolve()
            completed += item["size"]
            progress(completed, total)
            continue
        destination = folder / item["name"]
        full_hash = hashlib.sha256()
        with destination.open("xb") as output:
            for part in item["parts"]:
                check_cancel(cancel)
                part_hash, count = hashlib.sha256(), 0
                with opener(part["url"]) as source:
                    length = source.headers.get("Content-Length")
                    if length is not None and int(length) != part["size"]:
                        raise InstallError("Download size does not match its manifest")
                    while chunk := source.read(1024 * 1024):
                        check_cancel(cancel)
                        count += len(chunk)
                        if count > part["size"]:
                            raise InstallError("Download exceeded its expected size")
                        output.write(chunk)
                        part_hash.update(chunk)
                        full_hash.update(chunk)
                        completed += len(chunk)
                        progress(completed, total)
                if count != part["size"] or part_hash.hexdigest() != part["sha256"]:
                    raise InstallError("Download checksum failed. No model was imported; try again.")
        if destination.stat().st_size != item["size"] or full_hash.hexdigest() != item["sha256"]:
            raise InstallError("Assembled model checksum failed")
        with destination.open("rb") as source:
            if source.read(4) != b"GGUF":
                raise InstallError("The model file is not a GGUF model")
        paths[item["name"]] = destination
    return paths


def validate_app(app_dir):
    app_dir = Path(app_dir).resolve(strict=True)
    if any(c in str(app_dir) for c in ('"', "\r", "\n")):
        raise InstallError("Unsupported application folder name")
    for name in ("app.py", "runtime.py", "runtime/python/python.exe", "runtime/ollama/ollama.exe"):
        if not (app_dir / name).is_file():
            raise InstallError("Select the existing Local Agent folder containing app.py and runtime")
    data = app_dir / "data"
    if data.is_symlink() or data.is_junction() or not data.resolve().is_relative_to(app_dir):
        raise InstallError("The application's data folder must stay inside its folder")
    return app_dir


def app_runtime(app_dir):
    spec = importlib.util.spec_from_file_location("_model_installer_runtime", app_dir / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_model(app_dir, manifest, license_path, cancel, notify, *, runtime_module=None, opener=open_download):
    app_dir, manifest = validate_app(app_dir), validate_manifest(manifest)
    license_text = Path(license_path).read_text(encoding="utf-8")
    if not license_text.strip() or len(license_text.encode()) > 65536 or '"""' in license_text:
        raise InstallError("Missing or invalid model license")
    runtime = runtime_module or app_runtime(app_dir)
    runtime.ensure_server()
    if MODEL in runtime.list_models():
        raise InstallError("The trained pilot is already installed. Click Refresh in Local Agent and select it.")
    engine = runtime._get("/api/version").get("version", "")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", engine)
    if not match or tuple(map(int, match.groups())) < (0, 30, 0):
        raise InstallError("Update Local Agent first; its engine must support Qwen3.5.")
    if shutil.disk_usage(app_dir).free < MIN_FREE_BYTES:
        raise InstallError("Allow at least 8 GB of free disk space for download and installation.")
    data = app_dir / "data"
    data.mkdir(exist_ok=True)
    notify("status", "Downloading and verifying model files (up to 3.4 GB)…")
    with tempfile.TemporaryDirectory(prefix="trained-model-", dir=data) as temporary:
        folder = Path(temporary).resolve()
        if not folder.is_relative_to(data.resolve()):
            raise InstallError("Temporary model folder left application storage")
        paths = download_files(manifest, folder, app_dir, cancel, lambda a, b: notify("progress", (a, b)), opener=opener)
        check_cancel(cancel)
        if MODEL in runtime.list_models():
            raise InstallError("This model was installed by another window; nothing was overwritten.")
        lines = [f'FROM "{paths[name].as_posix()}"' for name in ("model.gguf", "mmproj.gguf")]
        lines += ["TEMPLATE {{ .Prompt }}", "RENDERER qwen3.5", "PARSER qwen3.5"]
        lines += [f"PARAMETER {name} {value}" for name, value in PARAMETERS.items()]
        lines += ['LICENSE """' + license_text + '"""']
        modelfile = folder / "Modelfile"
        modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment.update(OLLAMA_HOST=f"127.0.0.1:{runtime.PORT}", OLLAMA_MODELS=str(app_dir / "models"), OLLAMA_NO_CLOUD="1")
        notify("importing", "Registering the verified model. Please wait until this finishes…")
        process = subprocess.run([str(app_dir / "runtime/ollama/ollama.exe"), "create", MODEL, "-f", str(modelfile)],
            cwd=folder, env=environment, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=900, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode:
            detail = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", process.stdout + process.stderr)[-1200:]
            raise InstallError("Model registration failed: " + detail)
        if MODEL not in runtime.list_models():
            raise InstallError("The engine did not confirm the new model")
    notify("success", "Installed. In Local Agent click Refresh, then choose " + MODEL + ". Your original model and chats were kept.")


def main():
    import tkinter as tk
    from tkinter import ttk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model-manifest.json"))
    args = parser.parse_args()
    window = tk.Tk()
    window.title("Install experimental Local Agent model")
    window.geometry("680x440")
    window.minsize(650, 400)
    frame = ttk.Frame(window, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Experimental trained 4B pilot", font=("Segoe UI", 17, "bold")).pack(anchor="w")
    explanation = ttk.Label(frame, text="This pilot passed 1 of 3 coding tasks; the untrained reference passed 2 of 3.\nIt is not a demonstrated improvement. Your original model stays available.\n\nDownloads up to 3.4 GB. Allow 8 GB of free disk space.\nNo Codex account, API key or payment is needed by this installer.", wraplength=625)
    explanation.pack(anchor="w", pady=(12, 14))
    status = tk.StringVar(value="Choose Install to download this optional model.")
    status_label = ttk.Label(frame, textvariable=status, wraplength=625)
    status_label.pack(anchor="w")
    frame.bind("<Configure>", lambda event: [label.configure(wraplength=max(150, event.width - 48)) for label in (explanation, status_label)])
    bar = ttk.Progressbar(frame, mode="determinate")
    bar.pack(fill="x", pady=12)
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")
    events, cancel = queue.Queue(), threading.Event()
    state = {"busy": False, "importing": False}
    def close():
        if state["importing"]:
            status.set("Finishing model registration; please wait.")
        elif state["busy"]:
            cancel.set()
            status.set("Cancelling download…")
        else:
            window.destroy()
    def start():
        state["busy"] = True
        install.configure(state="disabled")
        def worker():
            try:
                manifest = load_manifest(args.manifest)
                install_model(args.app_dir, manifest, args.manifest.with_name("LICENSE.txt"), cancel,
                              lambda kind, payload: events.put((kind, payload)))
            except Exception as error:
                events.put(("error", str(error)))
        threading.Thread(target=worker, daemon=True).start()
    install = ttk.Button(buttons, text="Install experimental model", command=start)
    install.pack(side="left")
    close_button = ttk.Button(buttons, text="Cancel / Close", command=close)
    close_button.pack(side="right")
    def poll():
        while not events.empty():
            kind, payload = events.get_nowait()
            if kind == "progress":
                bar.configure(maximum=payload[1], value=payload[0])
            else:
                status.set(payload)
                if kind == "importing":
                    state["importing"] = True
                    close_button.configure(state="disabled")
                elif kind in {"success", "error"}:
                    state.update(busy=False, importing=False)
                    close_button.configure(state="normal", text="Close")
        window.after(100, poll)
    window.protocol("WM_DELETE_WINDOW", close)
    window.after(100, poll)
    window.mainloop()


if __name__ == "__main__":
    main()
