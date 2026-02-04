#!/usr/bin/env python3
"""
Tests for arxiv_feed.py
=======================

Tests RSS parsing, storage deduplication, and bus event emission.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_feed import (
    ArxivPaper,
    ArxivStore,
    CATEGORIES,
    parse_arxiv_rss,
    fetch_all_categories,
)


class TestArxivPaper(unittest.TestCase):
    """Tests for ArxivPaper dataclass."""

    def test_paper_creation(self):
        """Test creating a paper with auto-generated hash."""
        paper = ArxivPaper(
            arxiv_id="2312.12345",
            title="Test Paper Title",
            abstract="This is a test abstract for the paper.",
            authors=["Alice", "Bob"],
            categories=["cs.AI", "cs.LG"],
            link="https://arxiv.org/abs/2312.12345",
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
            source_category="cs.AI",
        )

        self.assertEqual(paper.arxiv_id, "2312.12345")
        self.assertEqual(paper.title, "Test Paper Title")
        self.assertEqual(len(paper.content_hash), 16)  # Auto-generated

    def test_paper_hash_deterministic(self):
        """Test that paper hash is deterministic."""
        paper1 = ArxivPaper(
            arxiv_id="2312.99999",
            title="Same Title",
            abstract="Same abstract content",
            authors=[],
            categories=["cs.AI"],
            link="https://arxiv.org/abs/2312.99999",
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
            source_category="cs.AI",
        )

        paper2 = ArxivPaper(
            arxiv_id="2312.99999",
            title="Same Title",
            abstract="Same abstract content",
            authors=[],
            categories=["cs.AI"],
            link="https://arxiv.org/abs/2312.99999",
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
            source_category="cs.AI",
        )

        self.assertEqual(paper1.content_hash, paper2.content_hash)


class TestArxivStore(unittest.TestCase):
    """Tests for ArxivStore SQLite storage."""

    def setUp(self):
        """Create temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_arxiv.sqlite3"
        self.store = ArxivStore(self.db_path)

    def tearDown(self):
        """Clean up."""
        self.store.close()
        if self.db_path.exists():
            self.db_path.unlink()
        Path(self.temp_dir).rmdir()

    def test_insert_and_dedup(self):
        """Test paper insertion and deduplication."""
        paper = ArxivPaper(
            arxiv_id="2312.00001",
            title="Test Paper",
            abstract="Abstract",
            authors=["Author"],
            categories=["cs.AI"],
            link="https://arxiv.org/abs/2312.00001",
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
            source_category="cs.AI",
            tags=["arxiv", "ai"],
        )

        # First insert should succeed
        result1 = self.store.insert_paper(paper)
        self.assertTrue(result1)

        # Second insert should be deduplicated
        result2 = self.store.insert_paper(paper)
        self.assertFalse(result2)

    def test_paper_exists(self):
        """Test paper existence check."""
        paper = ArxivPaper(
            arxiv_id="2312.00002",
            title="Test Paper 2",
            abstract="Abstract 2",
            authors=["Author"],
            categories=["cs.LG"],
            link="https://arxiv.org/abs/2312.00002",
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
            source_category="cs.LG",
        )

        self.assertFalse(self.store.paper_exists("2312.00002"))
        self.store.insert_paper(paper)
        self.assertTrue(self.store.paper_exists("2312.00002"))

    def test_stats(self):
        """Test statistics gathering."""
        # Insert some papers
        for i in range(3):
            paper = ArxivPaper(
                arxiv_id=f"2312.{1000+i}",
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                authors=[],
                categories=["cs.AI"],
                link=f"https://arxiv.org/abs/2312.{1000+i}",
                published_iso="2023-12-15T00:00:00Z",
                fetched_iso="2023-12-15T12:00:00Z",
                source_category="cs.AI",
            )
            self.store.insert_paper(paper)

        stats = self.store.stats()
        self.assertEqual(stats["total_papers"], 3)
        self.assertEqual(stats["by_category"]["cs.AI"], 3)

    def test_fetch_history(self):
        """Test fetch history recording."""
        self.store.record_fetch("cs.AI", 10, 5, "success")
        self.store.record_fetch("cs.LG", 8, 3, "success")

        history = self.store.get_fetch_history(10)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["category"], "cs.LG")  # Most recent first


class TestRSSParsing(unittest.TestCase):
    """Tests for RSS parsing."""

    def test_parse_empty_content(self):
        """Test parsing empty/invalid content."""
        papers = list(parse_arxiv_rss("", "cs.AI"))
        self.assertEqual(papers, [])

        papers = list(parse_arxiv_rss("not xml", "cs.AI"))
        self.assertEqual(papers, [])

    def test_parse_valid_rss(self):
        """Test parsing valid arXiv RSS format."""
        # Sample arXiv RDF/RSS content
        rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns="http://purl.org/rss/1.0/"
                 xmlns:dc="http://purl.org/dc/elements/1.1/"
                 xmlns:taxo="http://purl.org/rss/1.0/modules/taxonomy/">
            <channel>
                <title>cs.AI updates on arXiv.org</title>
            </channel>
            <item rdf:about="http://arxiv.org/abs/2312.12345">
                <title>2312.12345: A Test Paper on AI</title>
                <link>http://arxiv.org/abs/2312.12345</link>
                <description>This is a test abstract.</description>
                <dc:creator>Test Author</dc:creator>
                <dc:date>2023-12-15</dc:date>
            </item>
        </rdf:RDF>
        """

        papers = list(parse_arxiv_rss(rss_content, "cs.AI"))
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].arxiv_id, "2312.12345")
        self.assertIn("ai", papers[0].tags)


class TestCategories(unittest.TestCase):
    """Tests for category configuration."""

    def test_categories_defined(self):
        """Test all expected categories are configured."""
        expected = {"cs.AI", "cs.LG", "cs.CL", "stat.ML"}
        self.assertEqual(set(CATEGORIES.keys()), expected)

    def test_category_priorities(self):
        """Test category priorities are valid."""
        for cat, config in CATEGORIES.items():
            self.assertIn("priority", config)
            self.assertIn(config["priority"], [1, 2, 3])


class TestIntegration(unittest.TestCase):
    """Integration tests (mocked network)."""

    @patch("arxiv_feed.fetch_rss")
    @patch("arxiv_feed.emit_bus_event")
    @patch("arxiv_feed.index_paper_in_rag")
    def test_fetch_with_mocked_network(self, mock_rag, mock_bus, mock_fetch):
        """Test full fetch cycle with mocked network."""
        # Setup mocks
        mock_fetch.return_value = """<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns="http://purl.org/rss/1.0/"
                 xmlns:dc="http://purl.org/dc/elements/1.1/">
            <item>
                <title>Mock Paper</title>
                <link>http://arxiv.org/abs/2312.99999</link>
                <description>Mock abstract</description>
            </item>
        </rdf:RDF>
        """
        mock_bus.return_value = True
        mock_rag.return_value = True

        # Create temp store
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite3"
            store = ArxivStore(db_path)

            result = fetch_all_categories(
                store,
                categories=["cs.AI"],
                emit_events=True,
                index_rag=True,
            )

            self.assertEqual(result["total_papers"], 1)
            self.assertEqual(result["total_new"], 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
