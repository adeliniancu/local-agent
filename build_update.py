"""Build a small app-only update ZIP and a public HTTPS release-feed document."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from update_worker import APP_FILES, APP_ID, REQUIRED_FILES, UpdateError, inspect_archive, validate_manifest, version_constants

INSTALL_COMMAND = r'''@echo off
setlocal
set "PYTHONHOME="
set "PYTHONPATH="
set "AGENT_FOLDER="
echo Select the folder where Local Agent is already installed.
for /f "usebackq delims=" %%I in (`powershell.exe -NoProfile -STA -Command "Add-Type -AssemblyName System.Windows.Forms; $picker = New-Object System.Windows.Forms.FolderBrowserDialog; $picker.Description = 'Select your existing Local Agent folder (contains runtime and models)'; $picker.ShowNewFolderButton = $false; if ($picker.ShowDialog() -eq 'OK') { Write-Output $picker.SelectedPath }"`) do set "AGENT_FOLDER=%%I"
if not defined AGENT_FOLDER exit /b 0
if not exist "%AGENT_FOLDER%\runtime\python\pythonw.exe" goto missing
if not exist "%AGENT_FOLDER%\app.py" goto missing
start "Local Agent Update" "%AGENT_FOLDER%\runtime\python\pythonw.exe" -I "%~dp0update_worker.py" --offline --source "%~dp0." --target "%AGENT_FOLDER%"
exit /b 0
:missing
echo This folder does not contain Local Agent. Choose its extracted application folder.
pause
exit /b 1
'''


def build_update(source: Path, output: Path, repository: str = "", tag: str = "", min_version: str = "1.0.0") -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    constants = version_constants((source / "version.py").read_bytes())
    release_version = constants.get("__version__")
    repository = repository or constants.get("GITHUB_REPOSITORY", "")
    if repository and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise UpdateError("Use a GitHub repository in owner/name form.")
    tag = tag or "v" + release_version
    if tag not in (release_version, "v" + release_version):
        raise UpdateError("The release tag must match version.py.")
    payload = {}
    for relative in sorted(APP_FILES):
        path = source / relative
        if path.is_file():
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(source):
                raise UpdateError("Linked application files cannot be packaged.")
            payload[relative] = path.read_bytes()
    if not REQUIRED_FILES.issubset(payload):
        raise UpdateError("The source folder is missing required application files.")
    # Record the release repository in the artifact without changing the source.
    payload["version.py"] = (f'"""Application version and public update channel; no account credentials."""\n'
        f'APP_ID = {APP_ID!r}\n__version__ = {release_version!r}\nGITHUB_REPOSITORY = {repository!r}\n'
        f'UPDATE_FEED_URL = {constants.get("UPDATE_FEED_URL", "")!r}\n').encode("utf-8")
    manifest = {"app_id": APP_ID, "version": release_version, "min_version": min_version,
                "files": [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in sorted(payload.items())]}
    validate_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f"Local-Agent-update-{release_version}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, data in sorted(payload.items()):
            archive.writestr("payload/" + name, data)
        archive.writestr("Install Update.cmd", INSTALL_COMMAND.replace("\n", "\r\n"))
        archive.writestr("update_worker.py", payload["update_worker.py"])
    inspect_archive(archive_path)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    size = archive_path.stat().st_size
    feed = {"app_id": APP_ID, "version": release_version, "min_version": min_version,
            "download_url": f"https://github.com/{repository}/releases/download/{tag}/{archive_path.name}" if repository else "",
            "sha256": digest, "size": size}
    feed_path = output / "update-feed.json"
    feed_path.write_text(json.dumps(feed, indent=2) + "\n", encoding="utf-8")
    (output / (archive_path.name + ".sha256.txt")).write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    return {"archive": str(archive_path), "feed": str(feed_path), "sha256": digest, "size": size,
            "version": release_version, "online_configured": bool(repository or constants.get("UPDATE_FEED_URL"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "dist")
    parser.add_argument("--repository", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--min-version", default="1.0.0")
    args = parser.parse_args()
    print(json.dumps(build_update(args.source_dir, args.output_dir, args.repository, args.tag, args.min_version), indent=2))


if __name__ == "__main__":
    main()
