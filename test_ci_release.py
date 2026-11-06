"""Automatic release versioning and current-source publication checks; no network."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import ci_release


class AutomaticReleaseTests(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.prefix = "/repos/example/local-agent"
        self.data = {
            self.prefix + "/branches/main": {"commit": {"sha": self.sha}},
            self.prefix + "/releases/latest": {"tag_name": "v2.0.0"},
        }

    def check(self, version="2.0.2", assets=None):
        return ci_release.check_publication("example/local-agent", self.sha, version,
                                            get=self.data.get, assets=assets)

    def test_each_run_gets_a_newer_deterministic_version(self):
        self.assertEqual(ci_release.automatic_version("2.0.0", 2), "2.0.2")
        self.assertEqual(ci_release.automatic_version("2.1.3", 7), "2.1.10")
        self.assertLess(ci_release.parse_version(ci_release.automatic_version("2.0.0", 2)),
                        ci_release.parse_version(ci_release.automatic_version("2.0.0", 3)))
        for invalid in (0, -1, True):
            with self.assertRaises(ValueError):
                ci_release.automatic_version("2.0.0", invalid)

    def test_prepare_uses_committed_base_and_is_idempotent(self):
        committed = '"""Keep this metadata."""\n__version__ = "2.0.0"\nGITHUB_REPOSITORY = "example/local-agent"\n'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "version.py").write_text(committed.replace("2.0.0", "2.0.999"))
            with patch("ci_release.subprocess.run", return_value=types.SimpleNamespace(stdout=committed)) as git:
                first = ci_release.prepare_release(source, 4)
                first_text = (source / "version.py").read_text()
                second = ci_release.prepare_release(source, 4)
            self.assertEqual(first, {"base_version": "2.0.0", "version": "2.0.4", "tag": "v2.0.4"})
            self.assertEqual(first, second)
            self.assertEqual((source / "version.py").read_text(), first_text)
            self.assertIn('GITHUB_REPOSITORY = "example/local-agent"', first_text)
            self.assertEqual(git.call_args.args[0], ["git", "show", "HEAD:version.py"])

    def test_current_new_version_may_create_draft(self):
        result = self.check()
        self.assertTrue(result["allowed"])
        self.assertFalse(result["existing_draft"])

    def test_superseded_source_is_not_published(self):
        self.data[self.prefix + "/branches/main"]["commit"]["sha"] = "b" * 40
        self.assertFalse(self.check()["allowed"])

    def test_published_version_never_regresses(self):
        self.data[self.prefix + "/releases/latest"] = {"tag_name": "v2.0.3"}
        self.assertFalse(self.check("2.0.2")["allowed"])
        self.assertFalse(self.check("2.0.3")["allowed"])

    def test_same_commit_draft_can_resume_only_after_all_assets_upload(self):
        self.data[self.prefix + "/releases/tags/v2.0.2"] = {
            "draft": True, "target_commitish": self.sha,
            "assets": [{"name": "app.zip", "size": 123, "state": "uploaded"}]}
        self.assertTrue(self.check()["existing_draft"])
        self.assertTrue(self.check(assets={"app.zip": 123})["allowed"])
        with self.assertRaisesRegex(RuntimeError, "missing or incomplete"):
            self.check(assets={"app.zip": 124})
        with self.assertRaisesRegex(RuntimeError, "missing or incomplete"):
            self.check(assets={"feed.json": 123})

    def test_other_commit_draft_and_tag_are_never_overwritten(self):
        self.data[self.prefix + "/releases/tags/v2.0.2"] = {"draft": True, "target_commitish": "b" * 40}
        with self.assertRaisesRegex(RuntimeError, "different source commit"):
            self.check()
        del self.data[self.prefix + "/releases/tags/v2.0.2"]
        self.data[self.prefix + "/git/ref/tags/v2.0.2"] = {"object": {"type": "commit", "sha": "b" * 40}}
        with self.assertRaisesRegex(RuntimeError, "another commit"):
            self.check()

    def test_annotated_tag_resolves_to_tested_commit(self):
        self.data[self.prefix + "/git/ref/tags/v2.0.2"] = {"object": {"type": "tag", "sha": "c" * 40}}
        self.data[self.prefix + "/git/tags/" + "c" * 40] = {"object": {"type": "commit", "sha": self.sha}}
        self.assertTrue(self.check()["allowed"])

    def test_new_push_during_upload_blocks_final_promotion(self):
        self.assertTrue(self.check()["allowed"])
        self.data[self.prefix + "/releases/tags/v2.0.2"] = {"draft": True, "target_commitish": self.sha, "assets": []}
        self.data[self.prefix + "/branches/main"]["commit"]["sha"] = "b" * 40
        self.assertFalse(self.check(assets={})["allowed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
