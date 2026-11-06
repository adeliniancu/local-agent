"""Owner-only background source syncing. No tests, release tags, or runtime uploads."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REMOTE = "https://github.com/adeliniancu/local-agent"
MARKER = "Local-Agent-Source-Sync: 1"
POLL_SECONDS = 2.0
SETTLE_SECONDS = 2.0
MAX_FILE_BYTES = 16 * 1024 * 1024
DOCUMENTS = {
    ".gitattributes", ".gitignore", "README.md", "AGENTS.md", "CHANGELOG.md",
    "START HERE.txt", "Start Local Agent.cmd", "Start Local Agent (CPU).cmd",
    "Start GitHub Sync.cmd", "Stop GitHub Sync.cmd", "requirements.txt", "pyproject.toml",
}
ASSETS = {"assets/app.ico", "assets/app.png", "assets/logo.png"}
PRIVATE_PARTS = {
    ".git", "runtime", "models", "logs", "data", "workspace", "dist", "dev-tools",
    "devbackups", "__pycache__", ".updates", ".localagent-backups", ".venv", "venv",
    "credentials", "secrets", "accounts", ".ssh", ".aws", ".azure",
}
SECRET_PATTERN = re.compile(
    rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|"
    rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b|"
    rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b|"
    rb"https?://[^\s/@:]+:[^\s/@]+@", re.IGNORECASE
)


class SyncError(RuntimeError):
    pass


class NetworkError(SyncError):
    pass


def safe_error(error: BaseException) -> str:
    """Never put credential-shaped output in the persistent status/log."""
    value = str(error)
    value = re.sub(r"https?://\S+", "[remote URL]", value)
    value = re.sub(r"(?:gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]+", "[redacted]", value)
    return value[:800]


def new_credential_values(old: bytes, new: bytes) -> bool:
    """Existing reviewed test literals are permitted; newly added values are not."""
    return bool(set(SECRET_PATTERN.findall(new)) - set(SECRET_PATTERN.findall(old)))


def allowed_source(name: str, reviewed: set[str] | None = None) -> bool:
    if not name or "\\" in name or ":" in name:
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")):
        return False
    lowered = [p.casefold() for p in path.parts]
    if any(p in PRIVATE_PARTS or p.startswith(".env") for p in lowered):
        return False
    leaf = lowered[-1]
    if any(word in leaf for word in ("credential", "secret", "token", "password")):
        return False
    if leaf in {"auth.json", "id_rsa", "id_ed25519"}:
        return False
    if name in DOCUMENTS or name in ASSETS:
        return True
    if len(path.parts) == 1 and path.suffix == ".py":
        return True
    if len(path.parts) == 3 and path.parts[:2] == (".github", "workflows"):
        return path.suffix in {".yml", ".yaml"}
    return bool(reviewed and name in reviewed and len(path.parts) == 2
                and path.parts[0] == "distribution-notices"
                and path.suffix.lower() in {".txt", ".md", ".json", ".html", ".docx"})


def safe_path(root: Path, name: str) -> Path:
    path = root / name
    current = path
    while current != root:
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise SyncError(f"Linked source path is excluded: {name}")
        if current.parent == current:
            raise SyncError("Source path leaves the repository")
        current = current.parent
    if not path.resolve().is_relative_to(root):
        raise SyncError(f"Source path leaves the repository: {name}")
    return path


def version_metadata_only(old: bytes, new: bytes) -> bool:
    def normalize(value: bytes) -> str | None:
        tree = ast.parse(value.decode("utf-8-sig"))
        found = 0
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else []
            if len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == "__version__":
                if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                    return None
                node.value = ast.Constant(value="<installed version>")
                found += 1
        return ast.dump(tree, include_attributes=False) if found == 1 else None
    try:
        before, after = normalize(old), normalize(new)
        return before is not None and before == after
    except (SyntaxError, UnicodeError, ValueError):
        return False


@dataclass
class Snapshot:
    head: str
    files: dict[str, bytes | None]
    ignored_version: bool = False

    @property
    def fingerprint(self) -> tuple:
        return (self.head, tuple((name, None if value is None else hashlib.sha256(value).hexdigest())
                                 for name, value in sorted(self.files.items())))


class GitRepository:
    def __init__(self, root: Path, expected_remote: str = REMOTE, git: str | None = None):
        self.root = root.resolve()
        bundled = self.root / "dev-tools/git/cmd/git.exe"
        self.git = git or (str(bundled) if bundled.is_file() else shutil.which("git"))
        if not self.git:
            raise SyncError("Git is missing. Restore dev-tools/git/cmd/git.exe.")
        self.expected_remote = expected_remote.rstrip("/").removesuffix(".git").casefold()
        self.gh = self.root / "dev-tools/gh/bin/gh.exe"
        self.reviewed = set(self.run("ls-files", "-z").decode("utf-8").split("\0"))

    def run(self, *args: str, data: bytes | None = None, network: bool = False,
            okay: tuple[int, ...] = (0,), index: Path | None = None) -> bytes:
        command = [self.git, "-C", str(self.root)]
        if network and self.gh.is_file():
            helper = "!" + shlex.quote(self.gh.as_posix()) + " auth git-credential"
            command += ["-c", "credential.helper=", "-c", "credential.helper=" + helper]
        command.extend(args)
        env = os.environ.copy()
        for name in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
            env.pop(name, None)
        env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", GH_PROMPT_DISABLED="1")
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        try:
            result = subprocess.run(command, input=data, capture_output=True, env=env,
                                    timeout=45 if network else 20,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as exc:
            kind = NetworkError if network else SyncError
            raise kind(f"Git {args[0]} could not finish: {safe_error(exc)}") from exc
        if result.returncode not in okay:
            kind = NetworkError if network else SyncError
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise kind(f"Git {args[0]} failed: {safe_error(SyncError(detail))}")
        return result.stdout

    def text(self, *args: str, **kwargs) -> str:
        return self.run(*args, **kwargs).decode("utf-8", "replace").strip()

    def validate(self) -> None:
        if Path(self.text("rev-parse", "--show-toplevel")).resolve() != self.root:
            raise SyncError("Source sync must run at the expected repository root.")
        if self.text("symbolic-ref", "--quiet", "--short", "HEAD") != "main":
            raise SyncError("Source sync is paused: the checked-out branch must be main.")
        for options in (("--all", "origin"), ("--push", "--all", "origin")):
            urls = self.text("remote", "get-url", *options).splitlines()
            if not urls or any(u.rstrip("/").removesuffix(".git").casefold() != self.expected_remote for u in urls):
                raise SyncError("Source sync is paused: origin does not match the approved repository.")
        self.require_clean_index()

    def require_clean_index(self) -> None:
        if self.run("diff", "--cached", "--name-only", "-z", "HEAD"):
            raise SyncError("Source sync is paused: staged changes belong to a manual Git operation. The index was preserved.")

    def head(self) -> str:
        return self.text("rev-parse", "HEAD")

    def snapshot(self) -> Snapshot:
        self.validate()
        head = self.head()
        changed = self.run("diff", "--name-only", "--no-renames", "-z", "HEAD")
        changed += self.run("ls-files", "--others", "--exclude-standard", "-z")
        files: dict[str, bytes | None] = {}
        ignored_version = False
        for name in sorted(set(changed.decode("utf-8").split("\0"))):
            if not allowed_source(name, self.reviewed):
                continue
            path = safe_path(self.root, name)
            if not path.exists():
                files[name] = None
                continue
            before = path.stat()
            if not path.is_file() or before.st_size > MAX_FILE_BYTES:
                raise SyncError(f"Source file is not a regular file under 16 MiB: {name}")
            value = path.read_bytes()
            after = path.stat()
            safe_path(self.root, name)
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise SyncError(f"Waiting for a source write to finish: {name}")
            old = self.run("show", f"{head}:{name}", okay=(0, 128))
            if new_credential_values(old, value):
                raise SyncError(f"Source sync is paused: credential-shaped content in {name}. Nothing was staged.")
            if name == "version.py":
                if old and version_metadata_only(old, value):
                    ignored_version = True
                    continue
            files[name] = value
        return Snapshot(head, files, ignored_version)

    def commit(self, snapshot: Snapshot) -> str | None:
        self.validate()
        if self.head() != snapshot.head:
            raise SyncError("Repository HEAD changed; source sync will inspect it again.")
        if not snapshot.files:
            return None
        old_entries = self.tree_entries(snapshot.head)
        expected: dict[str, tuple[str, str] | None] = {}
        for name, value in snapshot.files.items():
            safe_path(self.root, name)
            if value is None:
                expected[name] = None
            else:
                blob = self.text("hash-object", "-w", "--stdin", "--path=" + name, data=value)
                mode = old_entries.get(name, ("100644", ""))[0]
                if mode not in {"100644", "100755"}:
                    raise SyncError(f"Unsupported source file mode: {name}")
                expected[name] = (mode, blob)
        index = Path(self.text("rev-parse", "--path-format=absolute", "--git-path", "index"))
        index_lock = index.with_name(index.name + ".lock")
        try:
            stream = index_lock.open("xb")
        except FileExistsError as exc:
            raise SyncError("Another Git operation holds the index lock; source sync will retry.") from exc
        owns_lock = True
        try:
            # Git checkout/add also require this lock. The real index remains untouched
            # until both the immutable commit and the guarded HEAD transaction succeed.
            with stream:
                stream.write(index.read_bytes())
            self.require_clean_index()
            self.validate_branch_and_head(snapshot.head)
            entries = []
            for name, entry in expected.items():
                mode, blob = entry if entry else ("0", "0" * 40)
                entries.append(f"{mode} {blob}\t{name}\0".encode("utf-8"))
            self.run("update-index", "-z", "--index-info", data=b"".join(entries), index=index_lock)
            tree = self.text("write-tree", index=index_lock)
            tree_entries = self.tree_entries(tree)
            changed = set(self.run("diff-tree", "--no-commit-id", "--name-only", "--no-renames", "-r", "-z", snapshot.head, tree).decode("utf-8").split("\0")) - {""}
            if not changed.issubset(expected) or any(tree_entries.get(name) != value for name, value in expected.items()):
                raise SyncError("Concurrent staged changes detected. The original index was preserved.")
            if not changed:
                return None
            message = "Auto-sync saved source changes\n\n" + MARKER
            commit = self.text("commit-tree", tree, "-p", snapshot.head, "-m", message)
            self.advance_head(snapshot.head, commit)
            try:
                os.replace(index_lock, index)
                owns_lock = False
            except OSError:
                # No worktree mutation occurred. Restore only our guarded ref update.
                self.advance_head(commit, snapshot.head)
                raise
            return commit
        finally:
            if owns_lock:
                index_lock.with_name(index_lock.name + ".lock").unlink(missing_ok=True)
                index_lock.unlink(missing_ok=True)

    def advance_head(self, old: str, new: str) -> None:
        # Canonical index.lock blocks checkout/switch/add throughout this operation.
        # Validate on both sides as direct symbolic-ref can bypass that index lock.
        self.validate_branch_and_head(old)
        self.run("update-ref", "--no-deref", "-m", "Local Agent source sync",
                 "refs/heads/main", new, old)
        try:
            self.validate_branch_and_head(new)
        except SyncError:
            # Restore only our own CAS update; never rewrite another process's commit.
            self.run("update-ref", "--no-deref", "-m", "Local Agent source sync rollback",
                     "refs/heads/main", old, new)
            raise

    def validate_branch_and_head(self, head: str) -> None:
        if self.text("symbolic-ref", "--quiet", "--short", "HEAD") != "main" or self.head() != head:
            raise SyncError("Branch or HEAD changed during capture. Staged files were preserved.")

    def tree_entries(self, ref: str) -> dict[str, tuple[str, str]]:
        result = {}
        for item in self.run("ls-tree", "-r", "-z", ref).split(b"\0"):
            if item:
                metadata, name = item.split(b"\t", 1)
                mode, kind, blob = metadata.decode().split()
                result[name.decode("utf-8")] = (mode, blob)
        return result

    def publish(self) -> str | None:
        self.validate()
        head = self.head()
        self.run("fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main", network=True)
        remote = self.text("rev-parse", "refs/remotes/origin/main")
        if head == remote:
            return None
        ancestor = self.text("merge-base", head, remote)
        if ancestor != remote:
            raise SyncError("Source sync is paused: GitHub has changes that need a manual merge. No force push was attempted.")
        # Inspect every outgoing commit, including files added then deleted later.
        for commit in self.text("rev-list", f"{remote}..{head}").splitlines():
            if MARKER not in self.text("show", "-s", "--format=%B", commit).splitlines():
                raise SyncError("Source sync is paused: unpushed commits were not created by this watcher.")
            paths = self.run("diff-tree", "--root", "--no-commit-id", "--name-only", "--no-renames", "-r", "-z", commit).decode("utf-8").split("\0")
            entries = self.tree_entries(commit)
            parents = self.text("rev-list", "--parents", "-n", "1", commit).split()[1:]
            if len(parents) != 1:
                raise SyncError("Source sync is paused: an outgoing commit has unexpected parent history.")
            parent_entries = self.tree_entries(parents[0])
            for name in filter(None, paths):
                if not allowed_source(name, self.reviewed):
                    raise SyncError(f"Source sync is paused: an outgoing commit contains an excluded path: {name}")
                entry = entries.get(name)
                if entry:
                    mode, blob = entry
                    if mode not in {"100644", "100755"} or int(self.text("cat-file", "-s", blob)) > MAX_FILE_BYTES:
                        raise SyncError(f"Source sync is paused: an outgoing file is not allowed: {name}")
                    previous = parent_entries.get(name)
                    old = self.run("cat-file", "blob", previous[1]) if previous else b""
                    if new_credential_values(old, self.run("cat-file", "blob", blob)):
                        raise SyncError(f"Source sync is paused: credential-shaped content in an outgoing file: {name}")
        self.validate()
        # Explicit SHA prevents a simultaneous manual commit from being published.
        self.run("push", "--porcelain", "origin", f"{head}:refs/heads/main", network=True)
        return head


class SourceSync:
    def __init__(self, repository: GitRepository, report=lambda message: None, clock=time.monotonic):
        self.repository, self.report, self.clock = repository, report, clock
        self.observed = None
        self.stable_since = 0.0
        self.next_network = 0.0
        self.retry_seconds = 5.0
        self.network_failed = False
        self.last_commit = None
        self.last_push = None
        self.message = "Starting source sync"

    def status(self, message: str) -> None:
        if message != self.message:
            self.message = message
            self.report(message)

    def step(self) -> None:
        now = self.clock()
        try:
            snapshot = self.repository.snapshot()
            if snapshot.fingerprint != self.observed:
                self.observed, self.stable_since = snapshot.fingerprint, now
            if snapshot.files and now - self.stable_since >= SETTLE_SECONDS:
                commit = self.repository.commit(snapshot)
                self.observed = None
                if commit:
                    self.last_commit = commit
                    self.status(f"Committed saved source changes: {commit[:12]}")
                    if not self.network_failed:
                        self.next_network = now
            elif snapshot.files:
                self.status("Waiting for saved source files to settle")
            elif not self.network_failed:
                self.status("Watching source changes; installed release version is ignored" if snapshot.ignored_version else "Watching source changes")
            if now >= self.next_network:
                pushed = self.repository.publish()
                self.last_push = pushed or self.last_push
                self.network_failed, self.retry_seconds = False, 5.0
                self.next_network = self.clock() + 30.0
                if pushed:
                    self.status(f"Pushed source changes to GitHub: {pushed[:12]}")
        except NetworkError as exc:
            self.network_failed = True
            self.next_network = self.clock() + self.retry_seconds
            self.status(f"Offline or GitHub unavailable; retry in {self.retry_seconds:g}s. {safe_error(exc)}")
            self.retry_seconds = min(60.0, self.retry_seconds * 2)
        except SyncError as exc:
            self.observed = None
            self.status(safe_error(exc))
        except OSError as exc:
            self.observed = None
            self.status("Waiting for a source file operation to finish: " + safe_error(exc))


class ProcessLock:
    def __init__(self, path: Path):
        self.path, self.stream = path, None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self.stream = stream
        return True

    def release(self) -> None:
        if self.stream:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()
            self.stream = None


def write_status(path: Path, **values) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "updated": time.time(), **values}, indent=2), encoding="utf-8")
    temporary.replace(path)


def console(message: str) -> None:
    if sys.stdout is not None:
        print(message, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="Settle and sync once, then exit")
    group.add_argument("--status", action="store_true", help="Show background sync status")
    group.add_argument("--stop", action="store_true", help="Request a clean background stop")
    args = parser.parse_args(argv)
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    status_path, stop_path = data / "source-sync-status.json", data / "source-sync.stop"
    lock = ProcessLock(data / "source-sync.lock")
    if args.status:
        running = not lock.acquire()
        lock.release()
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        status["running"] = running
        console(json.dumps(status, indent=2))
        return 0
    if args.stop:
        stop_path.write_text("stop", encoding="ascii")
        console("Stop requested. An active Git operation may take up to 45 seconds to finish.")
        return 0
    if not lock.acquire():
        console("Source sync is already running.")
        return 1
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logger = logging.getLogger("local-agent-source-sync")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(logs / "source-sync.log", maxBytes=512 * 1024, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    sync = None
    fatal_message = None
    try:
        stop_path.unlink(missing_ok=True)
        sync = SourceSync(GitRepository(ROOT), logger.info)
        while not stop_path.exists():
            sync.step()
            write_status(status_path, running=True, message=sync.message,
                         last_commit=sync.last_commit, last_push=sync.last_push)
            if args.once:
                time.sleep(SETTLE_SECONDS)
                sync.step()
                console(sync.message)
                break
            time.sleep(POLL_SECONDS)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        message = "Source sync stopped: " + safe_error(exc)
        fatal_message = message
        logger.error(message)
        write_status(status_path, running=False, message=message)
        console(message)
        return 1
    finally:
        try:
            if sync:
                write_status(status_path, running=False, message=fatal_message or sync.message,
                             last_commit=sync.last_commit, last_push=sync.last_push)
            logger.info("Source sync stopped")
        finally:
            handler.close()
            logger.removeHandler(handler)
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
