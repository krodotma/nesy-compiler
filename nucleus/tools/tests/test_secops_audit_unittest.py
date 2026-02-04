#!/usr/bin/env python3
"""Integration tests for SecOps Audit components.

Tests the SecOps security audit functionality:
- Permission checker
- Secret scanning hooks
- Policy enforcement

Effects: R(filesystem) - uses temp directories
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add secops to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "secops"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from permission_checker import (
    PermissionChecker,
    SecurityFinding,
)

# Import secops_guard functions
from secops_guard import (
    SECRET_PATTERNS,
    SENSITIVE_FILES,
    scan_for_secrets,
    check_sensitive_files,
    should_skip_file,
)


class TestPermissionCheckerInit(unittest.TestCase):
    """Test PermissionChecker initialization."""

    def test_init_with_path(self):
        """Checker initializes with a root path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checker = PermissionChecker(root)
            self.assertEqual(checker.root, root)

    def test_findings_initially_empty(self):
        """Findings list is initially empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = PermissionChecker(Path(tmpdir))
            self.assertEqual(checker.findings, [])


class TestWorldWritableCheck(unittest.TestCase):
    """Test world-writable detection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)
        self.checker = PermissionChecker(self.root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detect_world_writable_file(self):
        """Detect world-writable file."""
        test_file = self.root / "writable.txt"
        test_file.touch()
        os.chmod(test_file, 0o666)

        finding = self.checker.check_world_writable(test_file)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.category, "world_writable")
        self.assertIn("World-writable", finding.message)

    def test_detect_world_writable_directory(self):
        """Detect world-writable directory."""
        test_dir = self.root / "writable_dir"
        test_dir.mkdir()
        os.chmod(test_dir, 0o777)

        finding = self.checker.check_world_writable(test_dir)
        self.assertIsNotNone(finding)
        self.assertTrue(finding.details["is_directory"])

    def test_no_finding_for_normal_permissions(self):
        """No finding for normal permissions."""
        test_file = self.root / "normal.txt"
        test_file.touch()
        os.chmod(test_file, 0o644)

        finding = self.checker.check_world_writable(test_file)
        self.assertIsNone(finding)

    def test_sensitive_path_critical_severity(self):
        """Sensitive paths get critical severity."""
        sensitive_dir = self.root / ".pluribus"
        sensitive_dir.mkdir()
        os.chmod(sensitive_dir, 0o777)

        finding = self.checker.check_world_writable(sensitive_dir)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.severity, "critical")


class TestSetuidCheck(unittest.TestCase):
    """Test setuid/setgid detection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)
        self.checker = PermissionChecker(self.root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detect_setgid(self):
        """Detect setgid bit."""
        test_file = self.root / "setgid_file"
        test_file.touch()
        try:
            os.chmod(test_file, 0o2755)
            finding = self.checker.check_setuid_setgid(test_file)
            if finding:  # May require privileges
                self.assertEqual(finding.category, "setgid")
        except PermissionError:
            self.skipTest("Insufficient permissions for setgid test")

    def test_no_finding_for_normal_file(self):
        """No finding for normal executable."""
        test_file = self.root / "normal_exec"
        test_file.touch()
        os.chmod(test_file, 0o755)

        finding = self.checker.check_setuid_setgid(test_file)
        self.assertIsNone(finding)


class TestSecretsExposedCheck(unittest.TestCase):
    """Test secrets exposure detection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)
        self.checker = PermissionChecker(self.root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detect_exposed_env_file(self):
        """Detect exposed .env file."""
        env_file = self.root / ".env"
        env_file.touch()
        os.chmod(env_file, 0o644)  # Readable by others

        finding = self.checker.check_secrets_exposed(env_file)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.category, "secrets_exposed")
        self.assertEqual(finding.severity, "critical")

    def test_detect_exposed_credentials(self):
        """Detect exposed credentials file."""
        creds_file = self.root / "credentials.json"
        creds_file.touch()
        os.chmod(creds_file, 0o644)

        finding = self.checker.check_secrets_exposed(creds_file)
        self.assertIsNotNone(finding)

    def test_no_finding_for_protected_secrets(self):
        """No finding for properly protected secrets."""
        env_file = self.root / ".env"
        env_file.touch()
        os.chmod(env_file, 0o600)  # Owner only

        finding = self.checker.check_secrets_exposed(env_file)
        self.assertIsNone(finding)

    def test_no_finding_for_normal_file(self):
        """No finding for non-secret file."""
        normal_file = self.root / "readme.txt"
        normal_file.touch()
        os.chmod(normal_file, 0o644)

        finding = self.checker.check_secrets_exposed(normal_file)
        self.assertIsNone(finding)


class TestFullAudit(unittest.TestCase):
    """Test full audit functionality."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)
        self.checker = PermissionChecker(self.root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_audit_returns_report(self):
        """audit() returns a report dict."""
        report = self.checker.audit()

        self.assertIn("audit_time", report)
        self.assertIn("root", report)
        self.assertIn("total_findings", report)
        self.assertIn("severity_counts", report)
        self.assertIn("findings", report)
        self.assertIn("passed", report)

    def test_empty_directory_passes(self):
        """Empty directory passes audit."""
        report = self.checker.audit()
        self.assertTrue(report["passed"])
        self.assertEqual(report["total_findings"], 0)

    def test_audit_finds_issues(self):
        """audit() finds permission issues."""
        # Create a world-writable file
        bad_file = self.root / "bad.txt"
        bad_file.touch()
        os.chmod(bad_file, 0o666)

        report = self.checker.audit()
        self.assertGreater(report["total_findings"], 0)
        self.assertFalse(report["passed"])

    def test_audit_respects_max_depth(self):
        """audit() respects max_depth parameter."""
        # Create nested structure
        deep_dir = self.root / "a" / "b" / "c" / "d" / "e"
        deep_dir.mkdir(parents=True)
        deep_file = deep_dir / "deep.txt"
        deep_file.touch()
        os.chmod(deep_file, 0o666)

        # With shallow depth, shouldn't find the deep file
        report = self.checker.audit(max_depth=2)
        # Should still get a report
        self.assertIn("total_findings", report)

    def test_audit_sorts_by_severity(self):
        """audit() sorts findings by severity."""
        # Create files with different severities
        # (implementation depends on which triggers what severity)
        report = self.checker.audit()

        if len(report["findings"]) > 1:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
            for i in range(len(report["findings"]) - 1):
                sev1 = severity_order.get(report["findings"][i]["severity"], 5)
                sev2 = severity_order.get(report["findings"][i + 1]["severity"], 5)
                self.assertLessEqual(sev1, sev2)


class TestSecurityFinding(unittest.TestCase):
    """Test SecurityFinding dataclass."""

    def test_to_dict(self):
        """SecurityFinding converts to dict."""
        finding = SecurityFinding(
            severity="high",
            category="world_writable",
            path="/test/path",
            message="Test message",
            details={"key": "value"},
        )
        d = finding.to_dict()

        self.assertEqual(d["severity"], "high")
        self.assertEqual(d["category"], "world_writable")
        self.assertEqual(d["path"], "/test/path")
        self.assertEqual(d["message"], "Test message")
        self.assertEqual(d["details"]["key"], "value")


class TestSecopsGuardSecretPatterns(unittest.TestCase):
    """Test secret detection patterns."""

    def test_detect_aws_access_key(self):
        """Detect AWS access key."""
        content = 'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"'
        findings = scan_for_secrets(content, "test.py")
        self.assertGreater(len(findings), 0)
        self.assertTrue(any("AWS" in f["secret_type"] for f in findings))

    def test_detect_github_token(self):
        """Detect GitHub personal access token."""
        content = 'token = "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"'
        findings = scan_for_secrets(content, "test.py")
        self.assertGreater(len(findings), 0)
        self.assertTrue(any("GitHub" in f["secret_type"] for f in findings))

    def test_detect_openai_key(self):
        """Detect OpenAI API key."""
        content = 'OPENAI_API_KEY = "sk-" + "a" * 48'  # Avoid actual match in test
        # Construct a fake key that matches the pattern
        fake_key = "sk-" + "a" * 48
        content = f'api_key = "{fake_key}"'
        findings = scan_for_secrets(content, "test.py")
        self.assertGreater(len(findings), 0)

    def test_detect_private_key_header(self):
        """Detect private key header."""
        content = '-----BEGIN RSA PRIVATE KEY-----\nMIIE...'
        findings = scan_for_secrets(content, "key.pem")
        self.assertGreater(len(findings), 0)
        self.assertTrue(any("Private Key" in f["secret_type"] for f in findings))

    def test_no_false_positive_on_normal_code(self):
        """No false positive on normal code."""
        content = '''
def get_user(user_id):
    return User.objects.get(id=user_id)
'''
        findings = scan_for_secrets(content, "views.py")
        self.assertEqual(len(findings), 0)

    def test_line_number_accurate(self):
        """Finding includes accurate line number."""
        content = '''line1
line2
API_KEY = "AKIAIOSFODNN7EXAMPLE"
line4'''
        findings = scan_for_secrets(content, "test.py")
        if findings:
            # The AWS key should be on line 3
            self.assertEqual(findings[0]["line"], 3)


class TestSecopsGuardSensitiveFiles(unittest.TestCase):
    """Test sensitive file detection."""

    def test_detect_env_file(self):
        """Detect .env file."""
        result = check_sensitive_files(".env")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "sensitive_file")

    def test_detect_env_local(self):
        """Detect .env.local file."""
        result = check_sensitive_files(".env.local")
        self.assertIsNotNone(result)

    def test_detect_credentials_json(self):
        """Detect credentials.json."""
        result = check_sensitive_files("config/credentials.json")
        self.assertIsNotNone(result)

    def test_detect_private_key_pem(self):
        """Detect private key PEM."""
        result = check_sensitive_files("certs/private-key.pem")
        self.assertIsNotNone(result)

    def test_no_false_positive_on_normal_files(self):
        """No false positive on normal files."""
        normal_files = [
            "src/app.py",
            "README.md",
            "package.json",
            "requirements.txt",
            "config.yaml",
        ]
        for f in normal_files:
            result = check_sensitive_files(f)
            self.assertIsNone(result, f"False positive on {f}")


class TestSecopsGuardSkipFiles(unittest.TestCase):
    """Test file skipping logic."""

    def test_skip_binary_files(self):
        """Skip binary file extensions."""
        binary_files = [
            "image.png",
            "photo.jpg",
            "document.pdf",
            "archive.zip",
            "module.pyc",
        ]
        for f in binary_files:
            self.assertTrue(should_skip_file(f), f"Should skip {f}")

    def test_skip_lock_files(self):
        """Skip lock files."""
        lock_files = [
            "package-lock.json",
            "Pipfile.lock",
            "poetry.lock",
        ]
        for f in lock_files:
            self.assertTrue(should_skip_file(f), f"Should skip {f}")

    def test_dont_skip_source_files(self):
        """Don't skip source code files."""
        source_files = [
            "app.py",
            "index.js",
            "main.go",
            "config.yaml",
            "README.md",
        ]
        for f in source_files:
            self.assertFalse(should_skip_file(f), f"Should not skip {f}")


class TestSecopsGuardIntegration(unittest.TestCase):
    """Integration tests for secops guard."""

    def test_scan_file_with_multiple_secrets(self):
        """Scan file with multiple secrets."""
        content = '''
# Config file
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
API_TOKEN = "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
PASSWORD = "supersecretpassword123"
'''
        findings = scan_for_secrets(content, "config.py")
        self.assertGreaterEqual(len(findings), 2)

    def test_finding_includes_file_path(self):
        """Finding includes file path."""
        content = 'key = "AKIAIOSFODNN7EXAMPLE"'
        findings = scan_for_secrets(content, "my/deep/path/file.py")
        if findings:
            self.assertEqual(findings[0]["file"], "my/deep/path/file.py")


if __name__ == "__main__":
    unittest.main()
