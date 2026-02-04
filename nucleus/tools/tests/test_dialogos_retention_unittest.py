#!/usr/bin/env python3
"""Unit tests for dialogos_retention.py"""
from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools import dialogos_retention as dr


class TestRetentionPolicy(unittest.TestCase):
    """Tests for RetentionPolicy dataclass."""

    def test_defaults(self):
        policy = dr.RetentionPolicy()
        self.assertEqual(policy.max_age_days, 30)
        self.assertEqual(policy.max_size_mb, 500)
        self.assertEqual(policy.compress_after_days, 7)
        self.assertIsInstance(policy.archive_path, Path)

    def test_validate_ok(self):
        policy = dr.RetentionPolicy()
        warnings = policy.validate()
        self.assertEqual(warnings, [])

    def test_validate_max_age_below_min(self):
        policy = dr.RetentionPolicy(max_age_days=3, compress_after_days=2)
        warnings = policy.validate()
        # One warning for max_age_days below minimum
        self.assertTrue(any("below minimum" in w for w in warnings))

    def test_validate_compress_after_exceeds_max_age(self):
        policy = dr.RetentionPolicy(max_age_days=10, compress_after_days=15)
        warnings = policy.validate()
        self.assertEqual(len(warnings), 1)
        self.assertIn("exceeds max_age_days", warnings[0])


class TestLoadPolicy(unittest.TestCase):
    """Tests for load_policy function."""

    def test_load_policy_nonexistent(self):
        policy = dr.load_policy(Path("/nonexistent/policy.json"))
        self.assertEqual(policy.max_age_days, 30)

    def test_load_policy_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "max_age_days": 60,
                "max_size_mb": 1000,
                "compress_after_days": 14,
                "archive_path": "/tmp/test_archive",
            }, f)
            f.flush()
            policy = dr.load_policy(Path(f.name))
            os.unlink(f.name)

        self.assertEqual(policy.max_age_days, 60)
        self.assertEqual(policy.max_size_mb, 1000)
        self.assertEqual(policy.compress_after_days, 14)
        self.assertEqual(str(policy.archive_path), "/tmp/test_archive")


class TestUtilities(unittest.TestCase):
    """Tests for utility functions."""

    def test_human_bytes(self):
        self.assertEqual(dr.human_bytes(500), "500.0B")
        self.assertEqual(dr.human_bytes(1024), "1.0KB")
        self.assertEqual(dr.human_bytes(1024 * 1024), "1.0MB")
        self.assertEqual(dr.human_bytes(1024 * 1024 * 1024), "1.0GB")

    def test_parse_duration(self):
        self.assertEqual(dr.parse_duration("30d"), 30)
        self.assertEqual(dr.parse_duration("90"), 90)
        self.assertEqual(dr.parse_duration("  7d  "), 7)

    def test_days_ago(self):
        now = time.time()
        self.assertAlmostEqual(dr.days_ago(now), 0.0, places=2)
        self.assertAlmostEqual(dr.days_ago(now - 86400), 1.0, places=2)


class TestRotate(unittest.TestCase):
    """Tests for rotate_trace function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trace_path = Path(self.tmpdir) / "trace.ndjson"
        self.trace_path.write_text('{"test": "data"}\n')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rotate_not_needed(self):
        policy = dr.RetentionPolicy(max_size_mb=1)
        result = dr.rotate_trace(self.trace_path, policy, dry_run=True)
        self.assertFalse(result["rotated"])
        self.assertIn("no rotation needed", result.get("message", ""))

    def test_rotate_needed_dry_run(self):
        # Write a file larger than threshold
        with self.trace_path.open("w") as f:
            # Write ~1.5MB to exceed 1MB threshold
            f.write("x" * (1024 * 1024 + 512 * 1024))
        policy = dr.RetentionPolicy(max_size_mb=1)
        result = dr.rotate_trace(self.trace_path, policy, dry_run=True)
        self.assertTrue(result["rotated"])
        self.assertIsNotNone(result.get("new_file"))

    def test_rotate_missing_file(self):
        policy = dr.RetentionPolicy()
        missing = Path(self.tmpdir) / "missing.ndjson"
        result = dr.rotate_trace(missing, policy, dry_run=True)
        self.assertIn("error", result)


class TestCompress(unittest.TestCase):
    """Tests for compress_traces function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trace_dir = Path(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_compress_no_files(self):
        policy = dr.RetentionPolicy(compress_after_days=0)
        result = dr.compress_traces(self.trace_dir, policy, dry_run=True)
        self.assertEqual(result["compressed"], [])

    def test_compress_old_file_dry_run(self):
        # Create a rotated file with old mtime
        old_file = self.trace_dir / "trace.ndjson.20240101_000000"
        old_file.write_text('{"old": "data"}\n')
        # Set mtime to 30 days ago
        old_mtime = time.time() - (30 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        policy = dr.RetentionPolicy(compress_after_days=7)
        result = dr.compress_traces(self.trace_dir, policy, dry_run=True)
        self.assertEqual(len(result["compressed"]), 1)
        self.assertEqual(result["compressed"][0]["file"], "trace.ndjson.20240101_000000")

    def test_compress_actual(self):
        # Create a rotated file with old mtime
        old_file = self.trace_dir / "trace.ndjson.20240101_000000"
        old_file.write_text('{"old": "data"}\n' * 100)
        old_mtime = time.time() - (30 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        policy = dr.RetentionPolicy(compress_after_days=7)
        result = dr.compress_traces(self.trace_dir, policy, dry_run=False)

        self.assertEqual(len(result["compressed"]), 1)
        gz_file = self.trace_dir / "trace.ndjson.20240101_000000.gz"
        self.assertTrue(gz_file.exists())
        self.assertFalse(old_file.exists())  # Original should be deleted


class TestPrune(unittest.TestCase):
    """Tests for prune_archives function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.archive_dir = Path(self.tmpdir) / "archive"
        self.archive_dir.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prune_no_files(self):
        policy = dr.RetentionPolicy(archive_path=self.archive_dir)
        result = dr.prune_archives(policy, dry_run=True)
        self.assertEqual(result["pruned"], [])

    def test_prune_old_file_dry_run(self):
        # Create old archive file
        old_archive = self.archive_dir / "trace.ndjson.20240101_000000.gz"
        old_archive.write_bytes(b"compressed data")
        old_mtime = time.time() - (60 * 86400)  # 60 days old
        os.utime(old_archive, (old_mtime, old_mtime))

        policy = dr.RetentionPolicy(max_age_days=30, archive_path=self.archive_dir)
        result = dr.prune_archives(policy, dry_run=True)
        self.assertEqual(len(result["pruned"]), 1)

    def test_prune_respects_min_retention(self):
        # Create a file that's 5 days old
        recent_archive = self.archive_dir / "trace.ndjson.recent.gz"
        recent_archive.write_bytes(b"compressed data")
        recent_mtime = time.time() - (5 * 86400)  # 5 days old
        os.utime(recent_archive, (recent_mtime, recent_mtime))

        # Try to prune with 3 days threshold (below MIN_RETENTION_DAYS=7)
        policy = dr.RetentionPolicy(max_age_days=3, archive_path=self.archive_dir)
        result = dr.prune_archives(policy, older_than_days=3, dry_run=True)

        # Should use MIN_RETENTION_DAYS=7, so 5-day-old file should be skipped
        self.assertEqual(len(result["pruned"]), 0)
        self.assertEqual(result["threshold_days"], dr.MIN_RETENTION_DAYS)


class TestStatus(unittest.TestCase):
    """Tests for get_status function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trace_dir = Path(self.tmpdir)
        self.trace_path = self.trace_dir / "trace.ndjson"
        self.trace_path.write_text('{"test": "data"}\n')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_status_basic(self):
        policy = dr.RetentionPolicy(archive_path=self.trace_dir / "archive")
        status = dr.get_status(self.trace_dir, policy)

        self.assertEqual(status["trace_dir"], str(self.trace_dir))
        self.assertIsNotNone(status["active_trace"])
        self.assertEqual(status["active_trace"]["size_bytes"], len('{"test": "data"}\n'))

    def test_status_stale_trace_warning(self):
        # Set mtime to 25 hours ago
        old_mtime = time.time() - (25 * 3600)
        os.utime(self.trace_path, (old_mtime, old_mtime))

        policy = dr.RetentionPolicy(archive_path=self.trace_dir / "archive")
        status = dr.get_status(self.trace_dir, policy)

        self.assertTrue(any(w["type"] == "stale_trace" for w in status["warnings"]))

    def test_status_missing_trace(self):
        self.trace_path.unlink()
        policy = dr.RetentionPolicy(archive_path=self.trace_dir / "archive")
        status = dr.get_status(self.trace_dir, policy)

        self.assertIsNone(status["active_trace"])
        self.assertTrue(any(w["type"] == "missing_trace" for w in status["warnings"]))


class TestCLI(unittest.TestCase):
    """Tests for CLI interface."""

    def test_cli_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir)
            (trace_dir / "trace.ndjson").write_text('{"test": 1}\n')

            result = dr.main(["--trace-dir", str(trace_dir), "status"])
            self.assertEqual(result, 0)

    def test_cli_rotate_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir)
            (trace_dir / "trace.ndjson").write_text('{"test": 1}\n')

            result = dr.main(["--trace-dir", str(trace_dir), "rotate", "--dry-run"])
            self.assertEqual(result, 0)

    def test_cli_all_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir)
            (trace_dir / "trace.ndjson").write_text('{"test": 1}\n')

            result = dr.main(["--trace-dir", str(trace_dir), "all", "--dry-run"])
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
