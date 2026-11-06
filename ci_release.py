"""Version and guard releases built automatically from saved source pushes."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request


def parse_version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError("Release versions must have three numeric components.")
    return tuple(int(part) for part in value.split("."))


def automatic_version(base: str, run_number: int) -> str:
    if type(run_number) is not int or run_number < 1:
        raise ValueError("The workflow run number must be a positive integer.")
    major, minor, patch = parse_version(base)
    return f"{major}.{minor}.{patch + run_number}"


def _source_version(text: str) -> str:
    values = [ast.literal_eval(node.value) for node in ast.parse(text).body
              if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
    if len(values) != 1:
        raise ValueError("version.py must define __version__ exactly once.")
    parse_version(values[0])
    return values[0]


def prepare_release(source: Path, run_number: int) -> dict:
    # Always use committed source; installed/generated metadata must not compound.
    committed = subprocess.run(["git", "show", "HEAD:version.py"], cwd=source,
                               check=True, capture_output=True, text=True, encoding="utf-8").stdout
    base = _source_version(committed)
    generated = automatic_version(base, run_number)
    replaced, count = re.subn(r"(?m)^__version__\s*=.*$", f"__version__ = {generated!r}", committed)
    if count != 1:
        raise ValueError("Cannot replace the application's version assignment.")
    (source / "version.py").write_text(replaced, encoding="utf-8")
    return {"base_version": base, "version": generated, "tag": "v" + generated}


def github_json(path: str):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Local-Agent-Release", "Cache-Control": "no-cache"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request("https://api.github.com" + path, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"GitHub release check returned HTTP {error.code}.") from error


def check_publication(repository: str, expected_sha: str, version: str, *, get=github_json,
                      assets: dict[str, int] | None = None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Use repository owner/name.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_sha):
        raise ValueError("The tested source commit must be a full Git SHA.")
    candidate = parse_version(version)
    prefix = "/repos/" + repository
    branch = get(prefix + "/branches/main")
    if not branch or branch.get("commit", {}).get("sha") != expected_sha:
        return {"allowed": False, "reason": "A newer source push superseded this build."}
    latest = get(prefix + "/releases/latest")
    if latest:
        latest_tag = str(latest.get("tag_name", ""))
        latest_version = latest_tag[1:] if latest_tag.startswith("v") else latest_tag
        if parse_version(latest_version) >= candidate:
            return {"allowed": False, "reason": "This version or a newer version is already published."}
    tag = "v" + version
    existing = get(prefix + "/releases/tags/" + tag)
    # GitHub's tag lookup can omit drafts until the tag is published. The
    # authenticated release list includes our prepared, still-unpublished draft.
    if existing is None:
        for page in range(1, 21):
            releases = get(prefix + f"/releases?per_page=100&page={page}") or []
            if not isinstance(releases, list):
                raise RuntimeError("GitHub returned an invalid release list.")
            matching = [release for release in releases if release.get("tag_name") == tag]
            if len(matching) > 1:
                raise RuntimeError("Multiple releases use the expected tag; publication stopped.")
            if matching:
                existing = matching[0]
                break
            if len(releases) < 100:
                break
        else:
            raise RuntimeError("The release history could not be fully checked.")
    if existing and not existing.get("draft"):
        return {"allowed": False, "reason": "The release is already published and will not be changed."}
    if existing and existing.get("target_commitish") != expected_sha:
        raise RuntimeError("An existing draft belongs to a different source commit; it will not be overwritten.")
    reference = get(prefix + "/git/ref/tags/" + tag)
    if reference:
        obj = reference.get("object", {})
        for _ in range(5):
            if obj.get("type") != "tag":
                break
            annotated = get(prefix + "/git/tags/" + str(obj.get("sha", "")))
            obj = (annotated or {}).get("object", {})
        if obj.get("type") != "commit" or obj.get("sha") != expected_sha:
            raise RuntimeError("The release tag belongs to another commit; it will not be moved.")
    if assets is not None:
        if not existing:
            raise RuntimeError("The prepared draft is missing.")
        uploaded = {asset.get("name"): asset for asset in existing.get("assets", [])}
        for name, size in assets.items():
            asset = uploaded.get(name)
            if not asset or asset.get("size") != size or asset.get("state") != "uploaded":
                raise RuntimeError(f"Draft asset is missing or incomplete: {name}")
    return {"allowed": True, "existing_draft": existing is not None, "tag": tag, "reason": "Tested source is current and version is newer."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    prepare.add_argument("--run-number", type=int, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--repository", required=True)
    check.add_argument("--sha", required=True)
    check.add_argument("--version", required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--require-assets", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_release(args.source_dir.resolve(), args.run_number)
    else:
        assets = None
        if args.require_assets:
            manifest = json.loads(args.require_assets.read_text(encoding="utf-8"))
            assets = {name: (args.require_assets.parent / name).stat().st_size for name in manifest["assets"]}
        result = check_publication(args.repository, args.sha, args.version, assets=assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
