"""Build the first-install Windows release without models, accounts or chat data."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
OLLAMA_VERSION = "0.40.15"
OLLAMA_SIZE = 1_614_941_966
OLLAMA_SHA256 = "9ca74a7ab66ec86870b1eecac7df160df08dc4e3a291ab5d2d59bcf9310221c6"
OLLAMA_URL = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-windows-amd64.zip"
MAX_RELEASE_ASSET = 2 * 1024**3
APP_FILES = (
    "app.py", "agent_core.py", "runtime.py", "bootstrap.py", "version.py",
    "updater.py", "update_worker.py", "Start Local Agent.cmd",
    "Start Local Agent (CPU).cmd", "START HERE.txt",
)
ASSETS = ("app.ico", "app.png", "logo.png")
EXCLUDED = {"__pycache__", "site-packages", "test", "tests", "idlelib", "turtledemo"}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("local_agent_build_" + path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load build helper: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_version(source: Path) -> dict:
    values = {}
    for statement in ast.parse((source / "version.py").read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in {"__version__", "GITHUB_REPOSITORY"}:
                    values[target.id] = ast.literal_eval(statement.value)
    version = values.get("__version__", "")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError("version.py must define a numeric __version__, such as 2.0.0.")
    return values


def _download_ollama(archive: Path):
    if archive.exists():
        if archive.stat().st_size != OLLAMA_SIZE or _sha256(archive) != OLLAMA_SHA256:
            raise RuntimeError(f"Existing Ollama archive failed verification: {archive}")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    if partial.exists():
        raise RuntimeError(f"An unfinished download already exists; inspect it before retrying: {partial}")
    print(f"Downloading official Ollama {OLLAMA_VERSION} runtime", flush=True)
    digest = hashlib.sha256()
    total = 0
    last_report = -1
    try:
        request = urllib.request.Request(OLLAMA_URL, headers={"User-Agent": "Local-Agent-release-builder"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as target:
            while chunk := response.read(4 * 1024**2):
                target.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total > OLLAMA_SIZE:
                    raise RuntimeError("Ollama download exceeded the pinned release size.")
                report = int(total * 10 / OLLAMA_SIZE)
                if report != last_report:
                    print(f"Download {report * 10}%", flush=True)
                    last_report = report
        if total != OLLAMA_SIZE or digest.hexdigest() != OLLAMA_SHA256:
            raise RuntimeError("Official Ollama archive checksum or size mismatch.")
        partial.rename(archive)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _extract_ollama(archive: Path, destination: Path):
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite an existing Ollama runtime: {destination}")
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        for member in members:
            relative = PurePosixPath(member.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or any(":" in part for part in relative.parts):
                raise RuntimeError("Unsafe path in Ollama archive.")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError("Linked file in Ollama archive.")
        destination.mkdir(parents=True)
        package.extractall(destination)
    if not (destination / "ollama.exe").is_file():
        raise RuntimeError("Ollama archive did not contain the expected Windows executable.")


def ensure_runtimes(source: Path, skip_download=False):
    python = source / "runtime" / "python" / "python.exe"
    if not python.is_file():
        if skip_download or os.environ.get("GITHUB_ACTIONS") != "true":
            raise RuntimeError("Bundled Python is missing. CI prepares it from setup-python; local builds reuse the existing runtime.")
        helper = _load_module(source / "build_shareable.py")
        helper.prepare_python()
        # Some full CPython distributions keep Tcl scripts in a separate tree;
        # newer ones embed them in the Tcl DLL. Support both layouts.
        tcl_source = Path(sys.base_prefix) / "tcl"
        if tcl_source.is_dir():
            helper.copy_tree(tcl_source, python.parent / "tcl")
    ollama = source / "runtime" / "ollama" / "ollama.exe"
    if not ollama.is_file():
        if skip_download:
            raise RuntimeError("Ollama runtime is missing and --skip-download was requested.")
        archive = source / "runtime" / "ollama-windows-amd64.zip"
        _download_ollama(archive)
        _extract_ollama(archive, ollama.parent)
    # The probe imports Tk/Tcl without displaying a window and never loads a model.
    probe = subprocess.run(
        [str(python), "-I", "-c", "import sys,tkinter,ssl,json; assert sys.version_info[:2]==(3,15); t=tkinter.Tcl(); print(json.dumps({'python':sys.version.split()[0],'tcl':t.eval('info patchlevel')}))"],
        capture_output=True, text=True, timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if probe.returncode:
        raise RuntimeError("Bundled Python/Tcl verification failed: " + probe.stderr.strip())
    return json.loads(probe.stdout)


def _tree_files(source: Path):
    if not source.is_dir():
        raise RuntimeError(f"Required release directory is missing: {source}")
    for current, directories, filenames in os.walk(source):
        current = Path(current)
        for name in directories:
            item = current / name
            if item.is_symlink() or item.is_junction():
                raise RuntimeError(f"Linked folder is not allowed in a release: {item}")
        directories[:] = sorted(name for name in directories if name not in EXCLUDED)
        for name in sorted(filenames):
            item = current / name
            if item.is_symlink() or item.is_junction():
                raise RuntimeError(f"Linked file is not allowed in a release: {item}")
            if item.suffix not in {".pyc", ".pyo"}:
                yield item


def release_files(source: Path) -> list[tuple[Path, str]]:
    files = []
    for name in APP_FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Required application file is missing or linked: {name}")
        files.append((path, name))
    for name in ASSETS:
        path = source / "assets" / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Required image is missing or linked: {name}")
        files.append((path, "assets/" + name))
    for directory, archive_directory in (("distribution-notices", "Third Party Notices"),
                                          ("runtime/python", "runtime/python"),
                                          ("runtime/ollama", "runtime/ollama")):
        base = source / directory
        files.extend((item, archive_directory + "/" + item.relative_to(base).as_posix())
                     for item in _tree_files(base))
    return files


def build_release(source: Path, output: Path, repository: str = "", tag: str = "", skip_download=False):
    source, output = source.resolve(), output.resolve()
    values = read_version(source)
    version = values["__version__"]
    repository = repository or os.environ.get("GITHUB_REPOSITORY", "") or values.get("GITHUB_REPOSITORY", "")
    if repository and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must have the form owner/repository.")
    tag = tag or f"v{version}"
    if tag != f"v{version}":
        raise ValueError(f"Release tag {tag!r} must match version.py: v{version}.")
    if os.environ.get("GITHUB_ACTIONS") == "true" and not repository:
        raise RuntimeError("GitHub release builds require GITHUB_REPOSITORY.")
    output.mkdir(parents=True, exist_ok=True)
    name = f"Local-Agent-Windows-x64-{version}"
    archive = output / (name + ".zip")
    checksum_file = output / (name + ".zip.sha256.txt")
    if archive.exists() or checksum_file.exists():
        raise RuntimeError("Release output already exists. Choose a new output directory; existing packages will not be overwritten.")
    runtime_info = ensure_runtimes(source, skip_download)
    files = release_files(source)
    package_version = (source / "version.py").read_text(encoding="utf-8")
    if repository:
        package_version, replacements = re.subn(r"(?m)^GITHUB_REPOSITORY\s*=.*$", f"GITHUB_REPOSITORY = {repository!r}", package_version)
        if replacements != 1:
            raise RuntimeError("version.py must contain one GITHUB_REPOSITORY assignment.")
    manifest = {"application": "Local Agent", "version": version, "platform": "Windows x64",
                "python": runtime_info["python"], "ollama": OLLAMA_VERSION,
                "models_included": False, "personal_data_included": False,
                "repository": repository,
                "files": [{"path": name, "size": len(package_version.encode("utf-8")) if name == "version.py" else path.stat().st_size}
                          for path, name in files]}
    print(f"Packing {len(files)} application/runtime files; model download happens on first use", flush=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as package:
        for path, relative in files:
            if relative == "version.py":
                package.writestr(name + "/version.py", package_version)
            else:
                package.write(path, arcname=name + "/" + relative)
            if path.stat().st_size > 30_000_000:
                print(f"Packed {relative}", flush=True)
        package.writestr(name + "/PACKAGE.json", json.dumps(manifest, indent=2))
    size = archive.stat().st_size
    if size >= MAX_RELEASE_ASSET:
        raise RuntimeError(f"First-install ZIP exceeds GitHub's 2 GiB per-asset limit ({size:,} bytes).")
    checksum_file.write_text(_sha256(archive) + "  " + archive.name + "\n", encoding="ascii")
    update = _load_module(source / "build_update.py").build_update(source, output, repository, tag)
    assets = [archive.name, checksum_file.name, f"Local-Agent-update-{version}.zip", "update-feed.json"]
    for asset in assets:
        path = output / asset
        if not path.is_file() or path.stat().st_size >= MAX_RELEASE_ASSET:
            raise RuntimeError(f"Release asset is missing or too large: {asset}")
    notes = (
        f"Local Agent {version} for Windows x64\n\n"
        f"First installation: download {archive.name}, extract it, and open Start Local Agent.cmd. "
        "Python and Ollama are included; the local model downloads on first use.\n\n"
        f"Existing installation: use the app's update check, or download Local-Agent-update-{version}.zip "
        "and run its update installer. Projects, models, settings and chat history are preserved.\n\n"
        "All model inference runs locally. No account or API key is required.\n"
    )
    (output / "release-notes.txt").write_text(notes, encoding="utf-8")
    report = {"version": version, "tag": tag, "repository": repository, "assets": assets,
              "installer_bytes": size, "update": update}
    (output / "release-assets.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"version": version, "installer_bytes": size, "assets": assets}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist" / "release")
    parser.add_argument("--repository", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing portable runtimes; never download or replace them.")
    arguments = parser.parse_args()
    build_release(arguments.source_dir, arguments.output_dir, arguments.repository,
                  arguments.tag, arguments.skip_download)
