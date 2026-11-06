"""Build a clean Windows ZIP from an explicit application/runtime allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
NAME = "Local-Agent-Windows-x64"
STAGE = DIST / NAME
APP_FILES = ["app.py", "agent_core.py", "runtime.py", "bootstrap.py",
             "version.py", "updater.py", "update_worker.py",
             "Start Local Agent.cmd", "Start Local Agent (CPU).cmd", "START HERE.txt"]
ASSETS = ("app.ico", "app.png", "logo.png")
EXCLUDED = {"__pycache__", "site-packages", "test", "tests", "idlelib", "turtledemo"}


def copy_tree(source: Path, target: Path):
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        if any(part in EXCLUDED for part in relative.parts) or item.suffix in {".pyc", ".pyo"}:
            continue
        if item.is_symlink() or item.is_junction():
            raise RuntimeError(f"Unexpected linked file in runtime: {item}")
        if item.is_file():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


def prepare_python():
    source = Path(sys.base_prefix)
    target = ROOT / "runtime" / "python"
    if target.exists():
        raise RuntimeError("Bundled Python already exists; inspect before replacing it.")
    target.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file() and (item.name in {"python.exe", "pythonw.exe", "LICENSE.txt"}
                               or item.suffix.lower() == ".dll"):
            shutil.copy2(item, target / item.name)
    for directory in ("Lib", "DLLs"):
        copy_tree(source / directory, target / directory)
    (target / "Lib" / "site-packages").mkdir(exist_ok=True)
    print(f"Prepared independent Python runtime: {target}", flush=True)


def stage():
    if STAGE.exists():
        raise RuntimeError(f"Staging directory already exists: {STAGE}")
    STAGE.mkdir(parents=True)
    for name in APP_FILES:
        shutil.copy2(ROOT / name, STAGE / name)
    (STAGE / "assets").mkdir()
    for name in ASSETS:
        shutil.copy2(ROOT / "assets" / name, STAGE / "assets" / name)
    copy_tree(ROOT / "distribution-notices", STAGE / "Third Party Notices")
    for name in ("python", "ollama"):
        print(f"Copying {name} runtime", flush=True)
        copy_tree(ROOT / "runtime" / name, STAGE / "runtime" / name)
    source_models = ROOT / "models"
    # Only the selected model is shipped, not other models or account state.
    manifests = [Path("manifests/registry.ollama.ai/library/qwen3.5/4b-q4_K_M"),
                 Path("manifests-v2/ollama.com/library/qwen3.5/4b-q4_K_M")]
    first = json.loads((source_models / manifests[0]).read_text())
    digests = {first["config"]["digest"]} | {layer["digest"] for layer in first["layers"]}
    for relative in manifests:
        target = STAGE / "models" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_models / relative, target)
    for digest in sorted(digests):
        source = source_models / "blobs" / digest.replace(":", "-")
        with source.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest.split(":", 1)[1]:
            raise RuntimeError(f"Model blob checksum mismatch: {source.name}")
        target = STAGE / "models" / "blobs" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f"Staged clean app at {STAGE}", flush=True)


def archive():
    archive_path = DIST / (NAME + ".zip")
    if archive_path.exists():
        raise RuntimeError(f"Archive already exists: {archive_path}")
    files = sorted(item for item in STAGE.rglob("*") if item.is_file())
    forbidden = {"data", "logs", "workspace", "__pycache__"}
    paths = [item.relative_to(STAGE) for item in files]
    for relative in paths:
        if relative.parts[0] in forbidden or "__pycache__" in relative.parts:
            raise RuntimeError(f"Personal/generated file in staging: {relative}")
    print(f"Creating ZIP64 archive of {len(files)} files", flush=True)
    total = sum(item.stat().st_size for item in files)
    manifest = {"application": "Local Agent", "platform": "Windows x64",
                "python": "3.15.0", "ollama": "0.40.15", "model": "qwen3.5:4b-q4_K_M",
                "unpacked_bytes": total, "personal_data_included": False,
                "files": [{"path": str(path).replace("\\", "/"), "size": file.stat().st_size}
                          for file, path in zip(files, paths)]}
    (STAGE / "PACKAGE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(STAGE / "PACKAGE.json")
    written = 0
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as package:
        for item in files:
            package.write(item, arcname=NAME + "/" + item.relative_to(STAGE).as_posix())
            written += item.stat().st_size
            if item.stat().st_size > 20_000_000:
                print(f"Packed {written / total:.0%}: {item.name}", flush=True)
    print(f"Archive ready: {archive_path} ({archive_path.stat().st_size:,} bytes)", flush=True)
    with archive_path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    (DIST / (NAME + ".sha256.txt")).write_text(checksum + "  " + archive_path.name + "\n", encoding="ascii")
    print("SHA256: " + checksum, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["python", "stage", "archive"])
    args = parser.parse_args()
    {"python": prepare_python, "stage": stage, "archive": archive}[args.action]()
