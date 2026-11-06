"""Small, app-only updates from a trusted ZIP or the configured public HTTPS feed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import urllib.error
import urllib.parse
import urllib.request

import version
from update_worker import (APP_ID, MAX_ARCHIVE_BYTES, MAX_MANIFEST_BYTES, UpdateError,
                           inspect_archive, load_json, parse_version, stage_archive)

ROOT = Path(__file__).resolve().parent


def _https_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 4096:
        raise UpdateError("The update address is invalid.")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise UpdateError("Updates require an HTTPS address without embedded credentials.")
    return url


def configured_feed() -> str:
    explicit = getattr(version, "UPDATE_FEED_URL", "").strip()
    if explicit:
        return _https_url(explicit)
    repository = getattr(version, "GITHUB_REPOSITORY", "").strip()
    if not repository:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise UpdateError("The configured GitHub repository is invalid.")
    return f"https://github.com/{repository}/releases/latest/download/update-feed.json"


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _open(url: str):
    request = urllib.request.Request(_https_url(url), headers={"User-Agent": "Local-Agent-Updater/" + version.__version__, "Accept": "application/json, application/octet-stream"})
    return urllib.request.build_opener(_HTTPSRedirect()).open(request, timeout=25)


def validate_feed(feed: dict, current_version: str) -> dict:
    if not isinstance(feed, dict) or feed.get("app_id") != APP_ID:
        raise UpdateError("The update feed is not for Local Agent.")
    parse_version(feed.get("version"))
    parse_version(feed.get("min_version"))
    if parse_version(feed["min_version"]) >= parse_version(feed["version"]):
        raise UpdateError("The update feed has an invalid minimum version.")
    if parse_version(feed["version"]) > parse_version(current_version) and parse_version(current_version) < parse_version(feed["min_version"]):
        raise UpdateError(f"Install version {feed['min_version']} before this update.")
    _https_url(feed.get("download_url"))
    if type(feed.get("size")) is not int or not 0 < feed["size"] <= MAX_ARCHIVE_BYTES:
        raise UpdateError("The update download size is invalid.")
    if not isinstance(feed.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", feed["sha256"]):
        raise UpdateError("The update download checksum is invalid.")
    return feed


def read_feed(url: str, current_version: str) -> dict:
    try:
        with _open(url) as response:
            data = response.read(MAX_MANIFEST_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise UpdateError("No update has been published at this address yet.") from error
        raise UpdateError(f"The update server returned HTTP {error.code}.") from error
    return validate_feed(load_json(data), current_version)


def download_update(feed: dict, cancel: threading.Event, progress=None) -> Path:
    validate_feed(feed, version.__version__)
    folder = Path(tempfile.mkdtemp(prefix="LocalAgent-download-"))
    destination = folder / "update.zip"
    total = 0
    digest = hashlib.sha256()
    try:
        with _open(feed["download_url"]) as response, destination.open("wb") as output:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != feed["size"]:
                raise UpdateError("The update download size does not match its feed.")
            while True:
                if cancel.is_set():
                    raise UpdateError("Update download cancelled.")
                chunk = response.read(128 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > feed["size"] or total > MAX_ARCHIVE_BYTES:
                    raise UpdateError("The update download is larger than expected.")
                output.write(chunk)
                digest.update(chunk)
                if progress:
                    progress(total, feed["size"])
        if total != feed["size"] or digest.hexdigest() != feed["sha256"]:
            raise UpdateError("The update download is damaged. Try downloading it again.")
        manifest, _ = inspect_archive(destination, version.__version__)
        if manifest["version"] != feed["version"] or manifest["min_version"] != feed["min_version"]:
            raise UpdateError("The downloaded package does not match the update feed.")
        return destination
    except Exception:
        destination.unlink(missing_ok=True)
        raise


class _UpdateDialog:
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.feed_url = configured_feed()
        self.window = tk.Toplevel(parent)
        self.window.title("Update Local Agent")
        self.window.geometry("600x360")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()
        self.cancel = threading.Event()
        self.events = queue.Queue()
        self.archive = None
        self.closed = False
        self.busy = False
        self.poll_id = None
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(self.window, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Update Local Agent", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(frame, text=f"Installed version: {version.__version__}").pack(anchor="w", pady=(8, 15))
        source = f"Updates from {getattr(version, 'GITHUB_REPOSITORY', '') or urllib.parse.urlsplit(self.feed_url).hostname}" if self.feed_url else "Online updates have not been configured for this copy."
        ttk.Label(frame, text=source, wraplength=540).pack(anchor="w")
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(15, 10))
        self.online = ttk.Button(controls, text="Check online", command=self.check_online)
        self.online.pack(side="left")
        if not self.feed_url:
            self.online.configure(state="disabled")
        self.local = ttk.Button(controls, text="Choose update ZIP…", command=self.choose_zip)
        self.local.pack(side="left", padx=8)
        self.status = tk.StringVar(value="Your model, chats and project files will be kept.")
        ttk.Label(frame, textvariable=self.status, wraplength=540).pack(anchor="w", fill="x", pady=(6, 8))
        ttk.Label(frame, text="Only use update files supplied by the app's author.", wraplength=540).pack(anchor="w")
        actions = ttk.Frame(frame)
        actions.pack(side="bottom", fill="x", pady=(15, 0))
        self.install = ttk.Button(actions, text="Install and restart", command=self.install_update, state="disabled")
        self.install.pack(side="right")
        ttk.Button(actions, text="Close", command=self.close).pack(side="right", padx=8)
        self.poll_id = self.window.after(50, self.poll)

    def set_busy(self, busy: bool):
        self.busy = busy
        self.local.configure(state="disabled" if busy else "normal")
        self.online.configure(state="normal" if self.feed_url and not busy else "disabled")
        self.install.configure(state="normal" if self.archive and not busy else "disabled")

    def background(self, operation):
        self.set_busy(True)
        def run():
            try:
                self.events.put(("result", operation()))
            except Exception as error:
                self.events.put(("error", str(error)))
        threading.Thread(target=run, daemon=True, name="local-agent-update-check").start()

    def choose_zip(self):
        if self.busy:
            return
        selected = filedialog.askopenfilename(parent=self.window, title="Choose a Local Agent update ZIP", filetypes=[("Update ZIP", "*.zip")])
        if selected:
            self.archive = None
            self.status.set("Checking the update…")
            def inspect():
                manifest, _ = inspect_archive(selected, version.__version__)
                return {"archive": Path(selected), "manifest": manifest}
            self.background(inspect)

    def check_online(self):
        if self.busy or not self.feed_url:
            return
        self.archive = None
        self.status.set("Checking for updates…")
        def check():
            feed = read_feed(self.feed_url, version.__version__)
            if parse_version(feed["version"]) <= parse_version(version.__version__):
                return {"current": True}
            if self.cancel.is_set():
                return {"cancelled": True}
            self.events.put(("status", f"Downloading version {feed['version']}…"))
            archive = download_update(feed, self.cancel)
            manifest, _ = inspect_archive(archive, version.__version__)
            return {"archive": archive, "manifest": manifest}
        self.background(check)

    def install_update(self):
        if self.busy or not self.archive:
            return
        self.status.set("Preparing the update. Local Agent will close and restart…")
        selected = self.archive
        def prepare():
            folder, manifest = stage_archive(selected, version.__version__)
            return {"prepared": folder, "manifest": manifest}
        self.background(prepare)

    def poll(self):
        self.poll_id = None
        if self.closed:
            return
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "status":
                    self.status.set(payload)
                elif event == "error":
                    self.set_busy(False)
                    self.status.set(payload)
                elif event == "result":
                    self.set_busy(False)
                    if payload.get("current"):
                        self.status.set("You're using the latest published version.")
                    elif "archive" in payload:
                        self.archive = payload["archive"]
                        manifest = payload["manifest"]
                        self.status.set(f"Version {manifest['version']} is ready. Install and restart when you're ready.")
                        self.install.configure(state="normal")
                    elif "prepared" in payload:
                        folder = payload["prepared"]
                        try:
                            subprocess.Popen([str(ROOT / "runtime/python/pythonw.exe"), "-I", str(folder / "installer.py"),
                                              "--source", str(folder), "--target", str(ROOT), "--wait-pid", str(os.getpid())],
                                             cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        except OSError as error:
                            self.status.set(f"The update could not start: {error}")
                        else:
                            self.close()
                            self.parent.destroy()
                            return
        except queue.Empty:
            pass
        self.poll_id = self.window.after(75, self.poll)

    def close(self):
        if not self.closed:
            self.closed = True
            self.cancel.set()
            if self.poll_id is not None:
                self.window.after_cancel(self.poll_id)
                self.poll_id = None
            self.window.destroy()


def open_update_dialog(parent: tk.Tk) -> tk.Toplevel:
    """Return the nonblocking update window; do not open while an agent task runs."""
    return _UpdateDialog(parent).window
