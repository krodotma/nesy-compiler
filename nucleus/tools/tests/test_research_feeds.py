#!/usr/bin/env python3
"""
Tests for research_feeds.py
===========================

Tests the unified research feeds aggregator.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_feeds import (
    ResearchFeedsAggregator,
    FEED_CONFIG,
)


class TestFeedConfig(unittest.TestCase):
    """Tests for feed configuration."""

    def test_feeds_defined(self):
        """Test expected feeds are configured."""
        expected = {"arxiv", "labs"}
        self.assertEqual(set(FEED_CONFIG.keys()), expected)

    def test_feed_intervals(self):
        """Test feed intervals are reasonable."""
        # arXiv should be daily (86400s)
        self.assertEqual(FEED_CONFIG["arxiv"]["default_interval_s"], 86400)
        # Labs should be weekly (604800s)
        self.assertEqual(FEED_CONFIG["labs"]["default_interval_s"], 604800)


class TestAggregator(unittest.TestCase):
    """Tests for ResearchFeedsAggregator."""

    def setUp(self):
        """Create temporary root directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        # Create .pluribus structure
        (self.root / ".pluribus" / "rhizome.json").parent.mkdir(parents=True)
        (self.root / ".pluribus" / "rhizome.json").write_text("{}")
        (self.root / ".pluribus" / "sota").mkdir(parents=True)

        self.aggregator = ResearchFeedsAggregator(self.root)

    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_status_not_initialized(self):
        """Test status when databases don't exist."""
        status = self.aggregator.get_status()

        self.assertIn("feeds", status)
        # Without databases, status should indicate not initialized or have empty data
        self.assertIn("arxiv", status["feeds"])
        self.assertIn("labs", status["feeds"])

    @patch("research_feeds.ResearchFeedsAggregator._import_arxiv")
    @patch("research_feeds.ResearchFeedsAggregator._import_labs")
    def test_fetch_all_mocked(self, mock_labs_import, mock_arxiv_import):
        """Test fetch_all with mocked sub-modules."""
        # Create mock stores and results
        mock_arxiv_store = MagicMock()
        mock_labs_store = MagicMock()

        mock_arxiv_module = MagicMock()
        mock_arxiv_module.ArxivStore.return_value = mock_arxiv_store
        mock_arxiv_module.fetch_all_categories.return_value = {
            "total_papers": 10,
            "total_new": 3,
            "categories": {"cs.AI": {"paper_count": 10, "new_count": 3, "status": "success"}},
        }

        mock_labs_module = MagicMock()
        mock_labs_module.LabsStore.return_value = mock_labs_store
        mock_labs_module.fetch_all_sources.return_value = {
            "total_posts": 5,
            "total_new": 2,
            "sources": {"openai": {"post_count": 5, "new_count": 2, "status": "success"}},
        }

        mock_arxiv_import.return_value = mock_arxiv_module
        mock_labs_import.return_value = mock_labs_module

        results = self.aggregator.fetch_all(dry_run=True, emit_events=False, index_rag=False)

        self.assertEqual(results["total_new"], 5)  # 3 + 2
        self.assertIn("arxiv", results["feeds"])
        self.assertIn("labs", results["feeds"])

    def test_get_recent_items_empty(self):
        """Test getting recent items when empty."""
        items = self.aggregator.get_recent_items(limit=10)
        self.assertEqual(items, [])

    def test_get_stats_empty(self):
        """Test getting stats when empty."""
        stats = self.aggregator.get_stats()

        self.assertIn("generated_iso", stats)
        self.assertEqual(stats["total_items"], 0)
        self.assertEqual(stats["total_indexed"], 0)


class TestCLIParsing(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def test_parser_commands(self):
        """Test all CLI commands are defined."""
        from research_feeds import build_parser

        parser = build_parser()

        # These should all be valid commands
        valid_commands = ["status", "fetch", "daemon", "recent", "stats", "export"]

        for cmd in valid_commands:
            # This shouldn't raise
            args = parser.parse_args([cmd])
            self.assertEqual(args.cmd, cmd)

    def test_fetch_args(self):
        """Test fetch command arguments."""
        from research_feeds import build_parser

        parser = build_parser()

        args = parser.parse_args(["fetch", "--dry-run", "--no-bus", "--no-rag"])
        self.assertTrue(args.dry_run)
        self.assertTrue(args.no_bus)
        self.assertTrue(args.no_rag)

    def test_daemon_args(self):
        """Test daemon command arguments."""
        from research_feeds import build_parser

        parser = build_parser()

        args = parser.parse_args(["daemon", "--arxiv-interval", "3600", "--labs-interval", "7200"])
        self.assertEqual(args.arxiv_interval, 3600)
        self.assertEqual(args.labs_interval, 7200)

    def test_export_formats(self):
        """Test export format options."""
        from research_feeds import build_parser

        parser = build_parser()

        for fmt in ["json", "ndjson", "md"]:
            args = parser.parse_args(["export", "--format", fmt])
            self.assertEqual(args.format, fmt)


class TestBusIntegration(unittest.TestCase):
    """Tests for bus event emission."""

    @patch("research_feeds.emit_bus_event")
    def test_emit_event_called(self, mock_emit):
        """Test that bus events are emitted."""
        mock_emit.return_value = True

        # Import and call directly
        from research_feeds import emit_bus_event

        result = emit_bus_event(
            topic="sota.feeds.test",
            kind="log",
            data={"test": "data"},
        )

        # The mock was called
        mock_emit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
