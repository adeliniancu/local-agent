"""Portable entry point. Uses only the Python runtime inside this bundle."""
from __future__ import annotations

import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "runtime" / "python"
LOG_PATH = ROOT / "logs" / "desktop.log"
MANIFEST = ROOT / "models" / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "4b-q4_K_M"


def prepare_environment() -> None:
    """Keep the app and its commands independent of an installed Python."""
    os.environ.pop("PYTHONHOME", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ["PATH"] = os.pathsep.join((str(PYTHON), str(PYTHON / "Scripts"), os.environ.get("PATH", "")))
    # -I deliberately excludes the script directory. Add only our known app
    # directory, so imports work without relying on the launcher's current dir.
    root_string = str(ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != root_string]
    sys.path.insert(0, root_string)
    os.chdir(ROOT)


def validate_bundle() -> None:
    required = (
        PYTHON / "python.exe", PYTHON / "pythonw.exe",
        ROOT / "runtime" / "ollama" / "ollama.exe",
        ROOT / "app.py", ROOT / "agent_core.py", ROOT / "runtime.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"A required file is missing: {path.relative_to(ROOT)}. "
                "Extract the entire ZIP into a writable folder, then open Start Local Agent.cmd."
            )
    # GitHub first-install releases carry the runtime but download the large
    # model on the app's explicit setup button. Existing offline bundles work
    # unchanged and their model files are still checked when present.
    if not MANIFEST.is_file():
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    blobs = [manifest["config"], *manifest["layers"]]
    for entry in blobs:
        digest = entry.get("digest", "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("The bundled model manifest is invalid. Extract a fresh copy of the ZIP.")
        path = ROOT / "models" / "blobs" / digest.replace(":", "-")
        if not path.is_file() or path.stat().st_size != entry.get("size"):
            raise FileNotFoundError("The bundled AI model is incomplete. Extract the entire ZIP again.")


def show_error(error: BaseException, log_available: bool) -> None:
    details = f"\n\nDetails were saved to:\n{LOG_PATH}" if log_available else "\n\nUse a writable folder and extract the entire ZIP before opening the app."
    message = f"Local Agent couldn't start.\n\n{type(error).__name__}: {error}{details}"
    try:
        ctypes.windll.user32.MessageBoxW(None, message, "Local Agent", 0x10)
    except (AttributeError, OSError):
        if sys.stderr is not None:
            print(message, file=sys.stderr)


def main() -> int:
    log = None
    original_stdout, original_stderr = sys.stdout, sys.stderr
    original_thread_hook = threading.excepthook
    check_only = "--check" in sys.argv
    try:
        prepare_environment()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log = LOG_PATH.open("a", encoding="utf-8", buffering=1)
        print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Local Agent starting", file=log)
        print(f"Python: {sys.version.split()[0]} | Folder: {ROOT}", file=log)
        # pythonw has no console streams. Keep background and Tk callback
        # tracebacks available even when the app was opened by double-click.
        sys.stdout = log
        sys.stderr = log

        def thread_error(args):
            print(f"Background thread failed: {args.thread.name if args.thread else 'unknown'}", file=log)
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=log)

        threading.excepthook = thread_error
        validate_bundle()
        import tkinter  # noqa: F401 -- verify the bundled Tcl/Tk installation
        import agent_core  # noqa: F401
        import runtime  # noqa: F401
        import app
        if check_only:
            print("Bundle check passed.", file=log)
            if original_stdout is not None:
                print("Bundle check passed.", file=original_stdout)
        else:
            app.main()
        return 0
    except Exception as error:
        if log is not None:
            traceback.print_exc(file=log)
            log.flush()
        elif original_stderr is not None:
            traceback.print_exc(file=original_stderr)
        if check_only and original_stderr is not None:
            print(f"Bundle check failed: {error}", file=original_stderr)
        else:
            show_error(error, log is not None)
        return 1
    finally:
        threading.excepthook = original_thread_hook
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if log is not None:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
