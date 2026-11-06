"""Validate and install app-only Local Agent updates using Python's standard library.

Checksums detect damaged packages. They do not authenticate a package's author.
Only install ZIP files from a source you trust, or the configured HTTPS feed.
This module is also copied beside the payload for the initial version-1 upgrade.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zipfile

APP_ID = "local-agent-windows"
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
APP_FILES = frozenset({
    "app.py", "agent_core.py", "runtime.py", "bootstrap.py", "updater.py",
    "update_worker.py", "version.py", "Start Local Agent.cmd",
    "Start Local Agent (CPU).cmd", "START HERE.txt",
    "assets/app.ico", "assets/app.png", "assets/logo.png",
})
REQUIRED_FILES = frozenset({"app.py", "agent_core.py", "runtime.py", "bootstrap.py",
                            "updater.py", "update_worker.py", "version.py"})
EXTRA_ARCHIVE_FILES = frozenset({"Install Update.cmd", "update_worker.py"})


class UpdateError(ValueError):
    pass


def parse_version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value):
        raise UpdateError("Update versions must have the form 2.0.0.")
    return tuple(int(part) for part in value.split("."))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise UpdateError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json(data: bytes):
    if len(data) > MAX_MANIFEST_BYTES:
        raise UpdateError("The update manifest is too large.")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise UpdateError("The update manifest is not valid JSON.") from error


def version_constants(data: bytes) -> dict:
    try:
        tree = ast.parse(data.decode("utf-8-sig"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                        values[target.id] = node.value.value
        return values
    except (UnicodeError, SyntaxError) as error:
        raise UpdateError("The application's version file is invalid.") from error


def installed_version(target: Path) -> str:
    path = safe_path(target, "version.py")
    if not path.exists():
        return "1.0.0"
    value = version_constants(path.read_bytes()).get("__version__")
    parse_version(value)
    return value


def validate_manifest(manifest: dict, current_version: str | None = None) -> dict:
    if not isinstance(manifest, dict) or manifest.get("app_id") != APP_ID:
        raise UpdateError("This update is not for Local Agent.")
    new = parse_version(manifest.get("version"))
    minimum = parse_version(manifest.get("min_version"))
    if minimum >= new:
        raise UpdateError("The update's minimum version is invalid.")
    if current_version is not None:
        current = parse_version(current_version)
        if new <= current:
            raise UpdateError(f"Version {current_version} is already installed, or is newer than this update.")
        if current < minimum:
            raise UpdateError(f"Install version {manifest['min_version']} before this update.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files or len(files) > len(APP_FILES):
        raise UpdateError("The update file list is invalid.")
    seen, total = set(), 0
    for entry in files:
        if not isinstance(entry, dict):
            raise UpdateError("The update contains an invalid file entry.")
        path = entry.get("path")
        if not isinstance(path, str) or path not in APP_FILES or path.casefold() in seen:
            raise UpdateError(f"Disallowed or duplicate update path: {path!r}")
        if PurePosixPath(path).is_absolute() or "\\" in path or ":" in path:
            raise UpdateError("The update contains an unsafe file path.")
        seen.add(path.casefold())
        size = entry.get("size")
        if type(size) is not int or size < 0 or size > MAX_FILE_BYTES:
            raise UpdateError(f"Invalid file size: {path}")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise UpdateError(f"Invalid file checksum: {path}")
        total += size
    if total > MAX_EXPANDED_BYTES:
        raise UpdateError("The expanded update is too large.")
    if not REQUIRED_FILES.issubset({entry["path"] for entry in files}):
        raise UpdateError("The update is missing required application files.")
    return manifest


def _verify_payload(manifest: dict, payload: dict[str, bytes]) -> None:
    for entry in manifest["files"]:
        data = payload[entry["path"]]
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise UpdateError(f"The update is damaged: {entry['path']}")
    values = version_constants(payload["version.py"])
    if values.get("APP_ID") != APP_ID or values.get("__version__") != manifest["version"]:
        raise UpdateError("The packaged application version does not match its manifest.")


def inspect_archive(archive_path: str | Path, current_version: str | None = None) -> tuple[dict, dict[str, bytes]]:
    path = Path(archive_path)
    if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise UpdateError("Choose a Local Agent update ZIP smaller than 32 MB.")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > len(APP_FILES) + 3:
                raise UpdateError("The update contains too many files.")
            seen = set()
            for info in infos:
                name = info.filename
                if name.casefold() in seen:
                    raise UpdateError("The ZIP contains duplicate file names.")
                seen.add(name.casefold())
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise UpdateError("The update must contain regular files only.")
                if info.flag_bits & 1 or info.file_size > MAX_FILE_BYTES:
                    raise UpdateError("Encrypted or oversized update files are not supported.")
            if sum(info.file_size for info in infos) > MAX_EXPANDED_BYTES:
                raise UpdateError("The expanded update is too large.")
            manifest = validate_manifest(load_json(archive.read("manifest.json")), current_version)
            expected = {"manifest.json"} | {"payload/" + item["path"] for item in manifest["files"]}
            actual = {info.filename for info in infos}
            if not expected.issubset(actual) or not actual.issubset(expected | EXTRA_ARCHIVE_FILES):
                raise UpdateError("The ZIP contains unexpected or missing files.")
            payload = {}
            for entry in manifest["files"]:
                info = archive.getinfo("payload/" + entry["path"])
                if info.file_size != entry["size"]:
                    raise UpdateError(f"The ZIP file size is inconsistent: {entry['path']}")
                payload[entry["path"]] = archive.read(info)
            _verify_payload(manifest, payload)
            if "update_worker.py" in actual and archive.read("update_worker.py") != payload["update_worker.py"]:
                raise UpdateError("The offline installer does not match the update payload.")
            return manifest, payload
    except (zipfile.BadZipFile, KeyError, RuntimeError, EOFError) as error:
        raise UpdateError("The update ZIP is incomplete or damaged.") from error


def safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise UpdateError("An update path points outside the application folder.")
    current = root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink() or current.is_junction():
            raise UpdateError(f"Linked update paths are not supported: {relative}")
    return path


def inspect_folder(source: Path, current_version: str | None = None) -> tuple[dict, dict[str, bytes]]:
    source = source.resolve()
    manifest_path = safe_path(source, "manifest.json")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise UpdateError("The update manifest is too large.")
    manifest = validate_manifest(load_json(manifest_path.read_bytes()), current_version)
    payload = {}
    for entry in manifest["files"]:
        path = safe_path(source, "payload/" + entry["path"])
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise UpdateError(f"The extracted update is incomplete: {entry['path']}")
        payload[entry["path"]] = path.read_bytes()
    _verify_payload(manifest, payload)
    return manifest, payload


def stage_archive(archive: Path, current_version: str) -> tuple[Path, dict]:
    manifest, payload = inspect_archive(archive, current_version)
    folder = Path(tempfile.mkdtemp(prefix="LocalAgent-update-"))
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, data in payload.items():
        target = folder / "payload" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Run this installed verifier, independent of files about to be replaced.
    shutil.copy2(Path(__file__).resolve(), folder / "installer.py")
    return folder, manifest


def validate_target(target: Path) -> Path:
    target = target.resolve(strict=True)
    for relative in ("app.py", "bootstrap.py", "runtime/python/pythonw.exe", "runtime/ollama/ollama.exe"):
        if not safe_path(target, relative).is_file():
            raise UpdateError("Select the extracted Local Agent application folder, containing its runtime and models folders.")
    return target


def wait_for_pid(pid: int, timeout: float = 120) -> None:
    if pid <= 0 or pid == os.getpid():
        raise UpdateError("Invalid application process ID.")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # process already exited
                return
            raise UpdateError("The updater could not wait for Local Agent to close.")
        try:
            if kernel.WaitForSingleObject(handle, int(timeout * 1000)) != 0:
                raise UpdateError("Local Agent is still running. Close it before installing the update.")
        finally:
            kernel.CloseHandle(handle)
    else:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        raise UpdateError("The application is still running.")


def _atomic_write(path: Path, data: bytes, replace=os.replace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".localagent-update-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def apply_update(source: Path, target: Path, *, replace=os.replace) -> Path:
    target = validate_target(target)
    previous = installed_version(target)
    manifest, payload = inspect_folder(source, previous)
    # Resolve and validate every destination before any application file changes.
    destinations = {name: safe_path(target, name) for name in payload}
    for path in destinations.values():
        if path.exists() and not path.is_file():
            raise UpdateError(f"An update destination is not a file: {path.name}")
    backup_root = safe_path(target, "data/updates")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    backup.mkdir()
    old = {}
    for name, path in destinations.items():
        if path.exists() and path.stat().st_size > MAX_FILE_BYTES:
            raise UpdateError(f"The existing application file is unexpectedly large: {name}")
        old[name] = path.read_bytes() if path.exists() else None
        if old[name] is not None:
            copy = backup / "files" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(old[name])
    record = {"from_version": previous, "to_version": manifest["version"],
              "previously_absent": [name for name, data in old.items() if data is None],
              "status": "installing"}
    record_file = backup / "result.json"
    record_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    changed = []
    try:
        for name in sorted(payload, key=lambda name: (name == "version.py", name)):
            changed.append(name)
            _atomic_write(destinations[name], payload[name], replace=replace)
    except Exception as error:
        failures = []
        for name in reversed(changed):
            try:
                if old[name] is None:
                    destinations[name].unlink(missing_ok=True)
                else:
                    _atomic_write(destinations[name], old[name])
            except Exception as rollback_error:
                failures.append(f"{name}: {rollback_error}")
        record.update(status="rollback_incomplete" if failures else "rolled_back", error=str(error), rollback_errors=failures)
        record_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        message = f"The update failed. {'Some files need recovery' if failures else 'Previous application files were restored'}. Backups: {backup}"
        raise UpdateError(message) from error
    record["status"] = "installed"
    record_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return backup


def running_app_pids(target: Path) -> list[int]:
    if os.name != "nt":
        return []
    script = "Get-CimInstance Win32_Process -Filter \"Name = 'pythonw.exe' OR Name = 'python.exe'\" | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise UpdateError("Could not check whether Local Agent is closed. Close it, then try again.")
    rows = json.loads(result.stdout.strip() or "[]")
    if isinstance(rows, dict):
        rows = [rows]
    executables = {str((target / "runtime/python" / name).resolve()).casefold() for name in ("python.exe", "pythonw.exe")}
    names = (str(target / "bootstrap.py").casefold(), str(target / "app.py").casefold())
    return [int(row["ProcessId"]) for row in rows
            if row.get("ProcessId") != os.getpid()
            and str(row.get("ExecutablePath", "")).casefold() in executables
            and any(name in str(row.get("CommandLine", "")).casefold() for name in names)]


def restart_app(target: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    subprocess.Popen([str(target / "runtime/python/pythonw.exe"), "-I", str(target / "bootstrap.py")],
                     cwd=target, env=environment, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def wait_until_closed(target: Path, pids: list[int]) -> None:
    for pid in pids:
        wait_for_pid(pid)
    if running_app_pids(target):
        raise UpdateError("Local Agent is still open. Close every Local Agent window, then run Install Update again.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    import tkinter as tk
    from tkinter import messagebox
    window = tk.Tk()
    window.withdraw()
    target = None
    try:
        target = validate_target(args.target)
        manifest, _ = inspect_folder(args.source, installed_version(target))
        if args.offline:
            if not messagebox.askokcancel("Install Local Agent update", f"Install Local Agent {manifest['version']}?\n\nFolder: {target}\n\nClose every Local Agent window first. Your chats, projects and model will be kept. Only continue if this update came from a source you trust.", parent=window):
                return 0
        wait_until_closed(target, args.wait_pid)
        backup = apply_update(args.source, target)
        try:
            restart_app(target)
        except OSError as error:
            raise UpdateError(f"The update installed, but the app could not restart. Open Start Local Agent.cmd. Backups: {backup}") from error
        return 0
    except Exception as error:
        detail = str(error)
        if target is not None:
            try:
                log_path = safe_path(target, "logs/update.log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Update failed\n")
                    traceback.print_exc(file=log)
                detail += f"\n\nDetails: {log_path}"
            except (OSError, UpdateError):
                pass
        messagebox.showerror("Local Agent update", detail, parent=window)
        return 1
    finally:
        window.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
