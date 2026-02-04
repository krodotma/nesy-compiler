import json
import unittest
from unittest import mock

import sys
import pathlib

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import github_auth  # noqa: E402


class TestGitHubAuth(unittest.TestCase):
    def test_status_not_logged_in(self) -> None:
        def fake_run(argv, *, stdin_text=None):
            self.assertEqual(argv[:2], ["auth", "status"])
            return 1, "", "You are not logged in"

        with mock.patch.object(github_auth, "run_gh", side_effect=fake_run):
            st = github_auth.status(host="github.com")
            self.assertFalse(st["logged_in"])
            self.assertEqual(st["host"], "github.com")

    def test_status_parses_login_account_format(self) -> None:
        out = "github.com\\n  ✓ Logged in to github.com account krodotma (/root/.config/gh/hosts.yml)\\n"

        def fake_run(argv, *, stdin_text=None):
            self.assertEqual(argv[:3], ["auth", "status", "-h"])
            return 0, out, ""

        with mock.patch.object(github_auth, "run_gh", side_effect=fake_run):
            st = github_auth.status(host="github.com")
            self.assertTrue(st["logged_in"])
            self.assertEqual(st["login"], "krodotma")

    def test_login_with_missing_token_env(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            res = github_auth.login_with_token(host="github.com", scopes="repo", token_env_order=["GH_TOKEN"])
            self.assertFalse(res["ok"])
            self.assertEqual(res["reason"], "missing_token_env")

    def test_repo_permissions_parses(self) -> None:
        payload = {"permissions": {"pull": True, "push": False, "admin": False}, "default_branch": "main", "private": False}

        def fake_run(argv, *, stdin_text=None):
            self.assertEqual(argv[0], "api")
            self.assertIn("repos/krodotma/pluribus", argv[1])
            return 0, json.dumps(payload), ""

        with mock.patch.object(github_auth, "run_gh", side_effect=fake_run):
            res = github_auth.repo_permissions(repo="krodotma/pluribus")
            self.assertTrue(res["ok"])
            self.assertEqual(res["permissions"]["pull"], True)
            self.assertEqual(res["permissions"]["push"], False)
            self.assertEqual(res["default_branch"], "main")

    def test_login_with_token_does_not_return_token(self) -> None:
        def fake_run(argv, *, stdin_text=None):
            # Ensure the token goes to stdin (not argv), but never returns.
            self.assertEqual(argv[0:3], ["auth", "login", "--hostname"])
            self.assertTrue(stdin_text and "s3cr3t" in stdin_text)
            return 0, "ok", ""

        with mock.patch.dict("os.environ", {"GH_TOKEN": "s3cr3t"}, clear=True):
            with mock.patch.object(github_auth, "run_gh", side_effect=fake_run):
                res = github_auth.login_with_token(host="github.com", scopes="repo", token_env_order=["GH_TOKEN"])
                self.assertTrue(res["ok"])
                self.assertNotIn("s3cr3t", json.dumps(res))


if __name__ == "__main__":
    unittest.main()
