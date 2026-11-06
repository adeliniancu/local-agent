"""Source sync tests use isolated temporary repositories, never GitHub."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from source_sync import (GitRepository, MARKER, NetworkError, ProcessLock, Snapshot,
                         SourceSync, SyncError, allowed_source, safe_path,
                         version_metadata_only)

GIT = Path(__file__).parent / "dev-tools/git/cmd/git.exe"
GIT = str(GIT) if GIT.is_file() else shutil.which("git")


class AllowlistTests(unittest.TestCase):
    def test_only_public_source_paths(self):
        for path in ("source_sync.py", "test_source_sync.py", "ci_release.py", "AGENTS.md",
                     "README.md", "START HERE.txt", "Start GitHub Sync.cmd", "Stop GitHub Sync.cmd",
                     ".github/workflows/release.yml", "assets/app.ico"):
            self.assertTrue(allowed_source(path), path)
        for path in ("data/chat.json", "runtime/app.py", "models/model.gguf", "workspace/main.py",
                     "dev-tools/gh.exe", "dist/app.py", "logs/debug.py", ".env", ".env.py",
                     "credentials.py", "my_secret.py", "auth.json", "assets/private.png",
                     "nested/main.py", "../main.py", "C:/main.py", "main.py:secret",
                     "distribution-notices/unknown.txt", ".github/workflows/deep/private.yml"):
            self.assertFalse(allowed_source(path), path)

    def test_reviewed_notices_only(self):
        name = "distribution-notices/OLLAMA-MIT-LICENSE.txt"
        self.assertTrue(allowed_source(name, {name}))
        self.assertFalse(allowed_source("data/chat.json", {"data/chat.json"}))

    def test_version_ast_ignores_only_literal_version(self):
        original = b'APP_ID = "local"\n__version__ = "2.0.0"\nGITHUB_REPOSITORY = "owner/repo"\n'
        updated = b"# installed release\nAPP_ID='local'\n__version__='2.0.17'\nGITHUB_REPOSITORY='owner/repo'\n"
        self.assertTrue(version_metadata_only(original, updated))
        self.assertFalse(version_metadata_only(original, updated + b"print('different code')\n"))
        self.assertFalse(version_metadata_only(original, updated.replace(b"owner/repo", b"other/repo")))
        self.assertFalse(version_metadata_only(original, updated.replace(b"'2.0.17'", b"get_version()")))

    def test_duplicate_process_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            first = ProcessLock(Path(folder) / "sync.lock")
            second = ProcessLock(Path(folder) / "sync.lock")
            self.assertTrue(first.acquire())
            try:
                self.assertFalse(second.acquire())
            finally:
                first.release()
            self.assertTrue(second.acquire())
            second.release()


@unittest.skipUnless(GIT, "Git is required for isolated repository tests")
class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="source-sync-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "source"
        self.remote = self.base / "remote.git"
        self.root.mkdir()
        self.git("init", "--bare", str(self.remote), at=self.base)
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "Source Sync Test")
        self.git("config", "user.email", "source-sync-test@example.invalid")
        self.git("config", "core.autocrlf", "false")
        (self.root / "app.py").write_text("print('initial')\n", encoding="utf-8")
        (self.root / "version.py").write_text('__version__="2.0.0"\nAPP_ID="local"\n', encoding="utf-8")
        self.git("add", "app.py", "version.py")
        self.git("commit", "-m", "Initial test source")
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-u", "origin", "main")
        self.repo = GitRepository(self.root, expected_remote=str(self.remote), git=GIT)

    def git(self, *args, at=None):
        result = subprocess.run([GIT, "-C", str(at or self.root), *args], capture_output=True,
                                text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            self.fail(result.stderr)
        return result.stdout.strip()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_captured_bytes_commit_and_push_preserves_newer_write(self):
        self.write("app.py", "print('captured')\n")
        snapshot = self.repo.snapshot()
        self.write("app.py", "print('newer write')\n")
        commit = self.repo.commit(snapshot)
        self.assertIn(MARKER, self.git("show", "-s", "--format=%B", commit))
        self.assertEqual("print('captured')", self.git("show", f"{commit}:app.py"))
        self.assertEqual("print('newer write')\n", (self.root / "app.py").read_text())
        self.assertEqual(commit, self.repo.publish())
        self.assertEqual(commit, self.git("rev-parse", "refs/heads/main", at=self.remote))
        self.assertIn("app.py", self.repo.snapshot().files)

    def test_preexisting_staged_changes_preserved(self):
        self.write("app.py", "print('manually staged')\n")
        self.git("add", "app.py")
        index = self.git("write-tree")
        head = self.repo.head()
        with self.assertRaisesRegex(SyncError, "staged changes"):
            self.repo.snapshot()
        self.assertEqual(index, self.git("write-tree"))
        self.assertEqual(head, self.repo.head())

    def test_mixed_index_after_snapshot_is_preserved(self):
        self.write("app.py", "print('captured')\n")
        snapshot = self.repo.snapshot()
        self.write("manual.py", "print('manual staged')\n")
        self.git("add", "manual.py")
        index = self.git("write-tree")
        with self.assertRaisesRegex(SyncError, "staged changes"):
            self.repo.commit(snapshot)
        self.assertEqual(index, self.git("write-tree"))

    def test_commit_failure_leaves_original_index_and_head(self):
        self.write("app.py", "print('captured')\n")
        snapshot = self.repo.snapshot()
        original_index = self.git("write-tree")
        original_run = self.repo.run
        def fail_commit(*args, **kwargs):
            if args[0] == "commit-tree":
                raise SyncError("Injected commit failure")
            return original_run(*args, **kwargs)
        with patch.object(self.repo, "run", side_effect=fail_commit):
            with self.assertRaisesRegex(SyncError, "Injected"):
                self.repo.commit(snapshot)
        self.assertEqual(snapshot.head, self.repo.head())
        self.assertEqual(original_index, self.git("write-tree"))
        self.assertFalse((self.root / ".git/index.lock").exists())

    def test_branch_switch_before_ref_transaction_is_rejected(self):
        self.git("branch", "other")
        self.write("app.py", "print('captured')\n")
        snapshot = self.repo.snapshot()
        original_index = self.git("write-tree")
        original_advance = self.repo.advance_head
        def switch_branch(old, new):
            # Direct symref mutation bypasses checkout's index lock; transaction
            # verification must still prevent advancing main in this race.
            self.git("symbolic-ref", "HEAD", "refs/heads/other")
            return original_advance(old, new)
        with patch.object(self.repo, "advance_head", side_effect=switch_branch):
            with self.assertRaises(SyncError):
                self.repo.commit(snapshot)
        self.assertEqual(snapshot.head, self.git("rev-parse", "main"))
        self.assertEqual(original_index, self.git("write-tree"))

    def test_failed_index_promotion_rolls_back_own_ref(self):
        self.write("app.py", "print('captured')\n")
        snapshot = self.repo.snapshot()
        original_index = self.git("write-tree")
        with patch("source_sync.os.replace", side_effect=OSError("Injected promotion failure")):
            with self.assertRaisesRegex(OSError, "promotion"):
                self.repo.commit(snapshot)
        self.assertEqual(snapshot.head, self.repo.head())
        self.assertEqual(original_index, self.git("write-tree"))
        self.assertFalse((self.root / ".git/index.lock").exists())

    def test_direct_branch_change_after_ref_update_rolls_back_only_own_commit(self):
        self.git("branch", "other")
        self.write("app.py", "print('captured')\n")
        snapshot = self.repo.snapshot()
        original_index = self.git("write-tree")
        original_run = self.repo.run
        switched = []
        def switch_after_update(*args, **kwargs):
            result = original_run(*args, **kwargs)
            if args[0] == "update-ref" and not switched:
                switched.append(True)
                self.git("symbolic-ref", "HEAD", "refs/heads/other")
            return result
        with patch.object(self.repo, "run", side_effect=switch_after_update):
            with self.assertRaises(SyncError):
                self.repo.commit(snapshot)
        self.assertEqual(snapshot.head, self.git("rev-parse", "main"))
        self.assertEqual(original_index, self.git("write-tree"))

    def test_manual_staging_is_blocked_during_private_index_transaction(self):
        self.write("app.py", "print('captured')\n")
        snapshot = self.repo.snapshot()
        original_run = self.repo.run
        attempted = []
        def manual_stage(*args, **kwargs):
            if args[0] == "write-tree" and kwargs.get("index"):
                self.write("app.py", "print('new manual write')\n")
                attempt = subprocess.run([GIT, "-C", str(self.root), "add", "app.py"], capture_output=True,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                attempted.append(attempt.returncode)
            return original_run(*args, **kwargs)
        with patch.object(self.repo, "run", side_effect=manual_stage):
            commit = self.repo.commit(snapshot)
        self.assertTrue(attempted and all(code != 0 for code in attempted))
        self.assertEqual("print('captured')", self.git("show", f"{commit}:app.py"))
        self.assertIn("new manual write", (self.root / "app.py").read_text())
        self.assertEqual("", self.git("diff", "--cached", "--name-only"))

    def test_generated_version_not_staged_with_source_change(self):
        self.write("version.py", "# installed\n__version__='2.0.17'\nAPP_ID = 'local'\n")
        self.write("app.py", "print('changed')\n")
        snapshot = self.repo.snapshot()
        self.assertTrue(snapshot.ignored_version)
        self.assertEqual({"app.py"}, set(snapshot.files))
        commit = self.repo.commit(snapshot)
        self.assertIn('"2.0.0"', self.git("show", f"{commit}:version.py"))
        self.assertIn("2.0.17", (self.root / "version.py").read_text())
        remaining = self.repo.snapshot()
        self.assertTrue(remaining.ignored_version)
        self.assertFalse(remaining.files)

    def test_secret_content_never_staged(self):
        self.write("utility.py", "API_KEY='" + "ghp_" + "a" * 36 + "'\n")
        with self.assertRaisesRegex(SyncError, "credential-shaped"):
            self.repo.snapshot()
        self.assertEqual("", self.git("diff", "--cached", "--name-only"))

    def test_reviewed_credential_test_literal_can_remain_unchanged(self):
        literal = "https://" + "user:password" + "@example.com/feed.json"
        self.write("url_test.py", f"URL = {literal!r}\n")
        self.git("add", "url_test.py")
        self.git("commit", "-m", "Reviewed URL parser test")
        self.git("push", "origin", "main")
        self.write("url_test.py", f"URL = {literal!r}\n# updated test\n")
        self.repo.commit(self.repo.snapshot())
        self.assertIsNotNone(self.repo.publish())

    def test_private_files_ignored(self):
        self.write("data/chat.json", '{"private":"chat"}')
        self.write("credentials.py", "password = 'private'\n")
        self.write("helper.py", "print('public')\n")
        self.assertEqual({"helper.py"}, set(self.repo.snapshot().files))

    def test_renames_deletions_and_new_source(self):
        (self.root / "app.py").rename(self.root / "new_app.py")
        self.write("README.md", "Public documentation\n")
        snapshot = self.repo.snapshot()
        self.assertEqual({"app.py", "new_app.py", "README.md"}, set(snapshot.files))
        self.repo.commit(snapshot)
        self.assertEqual("", self.git("status", "--porcelain"))

    def test_no_empty_commit_when_no_source_changes(self):
        head = self.repo.head()
        self.assertIsNone(self.repo.commit(self.repo.snapshot()))
        self.assertEqual(head, self.repo.head())
        self.assertIsNone(self.repo.publish())

    def test_wrong_branch_or_remote_refused(self):
        self.git("checkout", "-b", "manual-work")
        with self.assertRaisesRegex(SyncError, "branch must be main"):
            self.repo.snapshot()
        self.git("checkout", "main")
        self.git("remote", "set-url", "origin", "https://example.invalid/other")
        with self.assertRaisesRegex(SyncError, "approved repository"):
            self.repo.snapshot()

    def test_manual_outgoing_commit_not_published(self):
        self.write("app.py", "print('manual commit')\n")
        self.git("add", "app.py")
        self.git("commit", "-m", "Manual commit")
        before = self.git("rev-parse", "refs/heads/main", at=self.remote)
        with self.assertRaisesRegex(SyncError, "not created by this watcher"):
            self.repo.publish()
        self.assertEqual(before, self.git("rev-parse", "refs/heads/main", at=self.remote))

    def test_private_file_added_then_deleted_in_outgoing_history_refused(self):
        self.write("data/private.json", '{"private":true}')
        self.git("add", "data/private.json")
        self.git("commit", "-m", "Pretend sync\n\n" + MARKER)
        self.git("rm", "data/private.json")
        self.git("commit", "-m", "Pretend sync\n\n" + MARKER)
        with self.assertRaisesRegex(SyncError, "excluded path"):
            self.repo.publish()

    def test_source_symlink_is_rejected(self):
        outside = self.base / "private.txt"
        outside.write_text("private", encoding="utf-8")
        link = self.root / "linked.py"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Symlink creation requires Windows developer mode/privilege")
        with self.assertRaisesRegex(SyncError, "Linked source"):
            self.repo.snapshot()

    def test_junction_parent_is_rejected(self):
        import os
        if os.name != "nt":
            self.skipTest("Windows junction test")
        outside = self.base / "external"
        outside.mkdir()
        (outside / "app.ico").write_bytes(b"private")
        junction = self.root / "assets"
        result = subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
                                capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(0, result.returncode, result.stderr)
        try:
            with self.assertRaisesRegex(SyncError, "Linked source"):
                safe_path(self.root, "assets/app.ico")
        finally:
            junction.rmdir()  # Remove the junction itself, never its target.


class FakeRepository:
    def __init__(self):
        self.current = b"first"
        self.committed = b"old"
        self.head_value = "initial"
        self.published = "initial"
        self.commit_count = 0
        self.publish_calls = 0
        self.offline = False
        self.after_push = None

    def snapshot(self):
        files = {} if self.current == self.committed else {"app.py": self.current}
        return Snapshot(self.head_value, files)

    def commit(self, snapshot):
        self.committed = snapshot.files["app.py"]
        self.commit_count += 1
        self.head_value = f"auto-{self.commit_count}"
        return self.head_value

    def publish(self):
        self.publish_calls += 1
        if self.offline:
            raise NetworkError("Network unavailable")
        if self.published == self.head_value:
            return None
        self.published = self.head_value
        if self.after_push:
            callback, self.after_push = self.after_push, None
            callback()
        return self.published


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.repo = FakeRepository()
        self.sync = SourceSync(self.repo, clock=lambda: self.now)

    def step(self, seconds):
        self.now += seconds
        self.sync.step()

    def test_edit_while_push_is_seen_by_next_capture(self):
        self.repo.after_push = lambda: setattr(self.repo, "current", b"saved during push")
        self.step(0)
        self.step(2)
        self.assertEqual(1, self.repo.commit_count)
        self.step(2)
        self.step(2)
        self.assertEqual(2, self.repo.commit_count)
        self.assertEqual(b"saved during push", self.repo.committed)
        self.assertEqual(self.repo.head_value, self.repo.published)

    def test_pending_push_retries_without_new_edits(self):
        self.repo.offline = True
        self.step(0)
        self.step(2)
        self.assertEqual(1, self.repo.commit_count)
        self.assertEqual(1, self.repo.publish_calls)
        self.step(3)
        self.assertEqual(2, self.repo.publish_calls)
        self.repo.offline = False
        self.step(10)
        self.assertEqual(self.repo.head_value, self.repo.published)
        self.assertEqual(1, self.repo.commit_count)

    def test_continuous_writes_wait_for_settle(self):
        self.step(0)
        self.repo.current = b"partial write"
        self.step(1)
        self.repo.current = b"complete write"
        self.step(1)
        self.step(1)
        self.assertEqual(0, self.repo.commit_count)
        self.step(1)
        self.assertEqual(1, self.repo.commit_count)
        self.assertEqual(b"complete write", self.repo.committed)

    def test_transient_atomic_save_error_keeps_watcher_running(self):
        with patch.object(self.repo, "snapshot", side_effect=FileNotFoundError("Atomic save")):
            self.step(0)
        self.step(2)
        self.step(2)
        self.assertEqual(1, self.repo.commit_count)


if __name__ == "__main__":
    unittest.main()
