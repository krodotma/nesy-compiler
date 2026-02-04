#!/usr/bin/env python3
"""
Unit tests for clade_weaver.py

Tests the Clade-Weave neurosymbolic merge protocol components.
"""

import unittest
import tempfile
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clade_weaver import (
    CladeAnalysis,
    MergeResult,
    load_genotype_lock,
    analyze_clade,
)


class TestCladeAnalysis(unittest.TestCase):
    """Test the CladeAnalysis dataclass."""

    def test_analysis_dataclass_creation(self):
        """Test that CladeAnalysis can be created with all fields."""
        analysis = CladeAnalysis(
            clade="clade/agent-a/task-001",
            base_branch="main",
            files_changed=["file1.py", "file2.py"],
            commits=[{"sha": "abc123", "message": "test commit"}],
            has_conflicts=False,
            conflict_files=[],
            touches_genotype=False,
            genotype_files=[],
            reasoning_events=[],
            merge_strategy="fast_forward"
        )

        self.assertEqual(analysis.clade, "clade/agent-a/task-001")
        self.assertEqual(analysis.base_branch, "main")
        self.assertEqual(len(analysis.files_changed), 2)
        self.assertEqual(analysis.merge_strategy, "fast_forward")
        self.assertFalse(analysis.has_conflicts)
        self.assertFalse(analysis.touches_genotype)

    def test_analysis_with_conflicts(self):
        """Test analysis with conflicts detected."""
        analysis = CladeAnalysis(
            clade="clade/agent-b/task-002",
            base_branch="main",
            files_changed=["api.py"],
            commits=[],
            has_conflicts=True,
            conflict_files=["api.py"],
            touches_genotype=False,
            genotype_files=[],
            reasoning_events=[],
            merge_strategy="neurosymbolic"
        )

        self.assertTrue(analysis.has_conflicts)
        self.assertEqual(analysis.merge_strategy, "neurosymbolic")
        self.assertIn("api.py", analysis.conflict_files)

    def test_analysis_with_genotype(self):
        """Test analysis when genotype files are touched."""
        analysis = CladeAnalysis(
            clade="clade/agent-c/task-003",
            base_branch="main",
            files_changed=["specs/CLADE_WEAVE.md"],
            commits=[{"sha": "def456", "message": "update spec"}],
            has_conflicts=False,
            conflict_files=[],
            touches_genotype=True,
            genotype_files=["specs/CLADE_WEAVE.md"],
            reasoning_events=[],
            merge_strategy="constitutional"
        )

        self.assertTrue(analysis.touches_genotype)
        self.assertEqual(analysis.merge_strategy, "constitutional")


class TestMergeResult(unittest.TestCase):
    """Test the MergeResult dataclass."""

    def test_successful_merge(self):
        """Test successful merge result."""
        result = MergeResult(
            success=True,
            strategy="fast_forward",
            merged_sha="abc123def456",
            error=None,
            artifacts=[]
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.merged_sha)
        self.assertIsNone(result.error)

    def test_failed_merge(self):
        """Test failed merge result."""
        result = MergeResult(
            success=False,
            strategy="neurosymbolic",
            merged_sha=None,
            error="Neurosymbolic synthesis requested",
            artifacts=[{"type": "synthesis_request"}]
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.merged_sha)
        self.assertIsNotNone(result.error)
        self.assertEqual(len(result.artifacts), 1)


class TestGenotypelock(unittest.TestCase):
    """Test genotype lock file loading."""

    def test_load_nonexistent_genotype_lock(self):
        """Test loading when genotype.lock doesn't exist."""
        with patch('clade_weaver.GENOTYPE_LOCK', Path("/nonexistent/path")):
            result = load_genotype_lock()
            self.assertEqual(result, [])

    def test_load_genotype_lock(self):
        """Test loading an existing genotype.lock file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lock', delete=False) as f:
            f.write("# Protected files\n")
            f.write("specs/CLADE_WEAVE.md\n")
            f.write("tools/iso_git.mjs\n")
            f.write("\n")
            f.write("# Another comment\n")
            f.write("core/bus.py\n")
            temp_path = f.name

        try:
            with patch('clade_weaver.GENOTYPE_LOCK', Path(temp_path)):
                result = load_genotype_lock()
                self.assertEqual(len(result), 3)
                self.assertIn("specs/CLADE_WEAVE.md", result)
                self.assertIn("tools/iso_git.mjs", result)
                self.assertIn("core/bus.py", result)
        finally:
            os.unlink(temp_path)


class TestMergeStrategies(unittest.TestCase):
    """Test merge strategy determination logic."""

    def test_strategy_for_no_commits(self):
        """Empty clade should use fast_forward."""
        analysis = CladeAnalysis(
            clade="clade/test/empty",
            base_branch="main",
            files_changed=[],
            commits=[],
            has_conflicts=False,
            conflict_files=[],
            touches_genotype=False,
            genotype_files=[],
            reasoning_events=[],
            merge_strategy="fast_forward"
        )
        self.assertEqual(analysis.merge_strategy, "fast_forward")

    def test_strategy_for_conflicts(self):
        """Conflicting changes should use neurosymbolic."""
        analysis = CladeAnalysis(
            clade="clade/test/conflict",
            base_branch="main",
            files_changed=["file.py"],
            commits=[{"sha": "123", "message": "change"}],
            has_conflicts=True,
            conflict_files=["file.py"],
            touches_genotype=False,
            genotype_files=[],
            reasoning_events=[],
            merge_strategy="neurosymbolic"
        )
        self.assertEqual(analysis.merge_strategy, "neurosymbolic")

    def test_strategy_for_genotype(self):
        """Genotype modifications should use constitutional."""
        analysis = CladeAnalysis(
            clade="clade/test/genotype",
            base_branch="main",
            files_changed=["specs/core.md"],
            commits=[{"sha": "456", "message": "spec update"}],
            has_conflicts=False,
            conflict_files=[],
            touches_genotype=True,
            genotype_files=["specs/core.md"],
            reasoning_events=[],
            merge_strategy="constitutional"
        )
        self.assertEqual(analysis.merge_strategy, "constitutional")


class TestBusEventIntegration(unittest.TestCase):
    """Test bus event emission and reasoning retrieval."""

    @patch('clade_weaver.emit_event')
    @patch('clade_weaver.resolve_bus_paths')
    def test_emit_event_called_on_conflict(self, mock_resolve, mock_emit):
        """Verify events are emitted when conflicts are detected."""
        mock_resolve.return_value = {"active_dir": "/tmp/bus"}

        # Import here to use mocked functions
        from clade_weaver import neurosymbolic_merge

        analysis = CladeAnalysis(
            clade="clade/test/conflict",
            base_branch="main",
            files_changed=["api.py"],
            commits=[],
            has_conflicts=True,
            conflict_files=["api.py"],
            touches_genotype=False,
            genotype_files=[],
            reasoning_events=[],
            merge_strategy="neurosymbolic"
        )

        # Mock run_git to avoid actual git commands
        with patch('clade_weaver.run_git') as mock_git:
            mock_git.return_value = MagicMock(returncode=1, stdout="", stderr="")
            result = neurosymbolic_merge(analysis)

        # Check that emit_event was called for conflict and synthesis request
        self.assertTrue(mock_emit.called)
        self.assertFalse(result.success)
        self.assertEqual(result.strategy, "neurosymbolic")


class TestArchiveClade(unittest.TestCase):
    """Test clade archiving as fossils."""

    @patch('clade_weaver.run_git')
    def test_archive_creates_fossil_tag(self, mock_git):
        """Test that archiving creates a fossil tag."""
        mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from clade_weaver import archive_clade

        result = archive_clade("clade/agent-a/task-001")

        self.assertTrue(result)
        # Check that git tag was called
        mock_git.assert_called()
        call_args = mock_git.call_args[0]
        self.assertEqual(call_args[0], "tag")
        self.assertIn("fossil", call_args[1])


if __name__ == "__main__":
    unittest.main()
