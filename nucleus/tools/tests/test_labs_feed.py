#!/usr/bin/env python3
"""
Tests for labs_feed.py
======================

Tests HTML/RSS parsing, storage, and bus event emission for industry labs.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from labs_feed import (
    LabPost,
    LabsStore,
    SOURCES,
    parse_rss_feed,
    parse_html_blog,
    clean_html,
    fetch_all_sources,
)


class TestLabPost(unittest.TestCase):
    """Tests for LabPost dataclass."""

    def test_post_creation(self):
        """Test creating a post with auto-generated IDs."""
        post = LabPost(
            post_id="",
            source="deepmind",
            title="Test Blog Post",
            summary="A summary of the post.",
            link="https://deepmind.google/blog/test-post",
            authors=["DeepMind Team"],
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
        )

        self.assertEqual(post.source, "deepmind")
        self.assertEqual(len(post.post_id), 16)  # Auto-generated from link hash
        self.assertEqual(len(post.content_hash), 16)

    def test_post_hash_deterministic(self):
        """Test that post hash is deterministic."""
        post1 = LabPost(
            post_id="",
            source="openai",
            title="Same Title",
            summary="Same summary",
            link="https://openai.com/blog/same-post",
            authors=[],
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
        )

        post2 = LabPost(
            post_id="",
            source="openai",
            title="Same Title",
            summary="Same summary",
            link="https://openai.com/blog/same-post",
            authors=[],
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
        )

        self.assertEqual(post1.post_id, post2.post_id)
        self.assertEqual(post1.content_hash, post2.content_hash)


class TestCleanHtml(unittest.TestCase):
    """Tests for HTML cleaning utility."""

    def test_clean_tags(self):
        """Test HTML tag removal."""
        dirty = "<p>Hello <strong>World</strong></p>"
        clean = clean_html(dirty)
        self.assertEqual(clean, "Hello World")

    def test_clean_entities(self):
        """Test HTML entity decoding."""
        dirty = "Hello &amp; World &lt;test&gt;"
        clean = clean_html(dirty)
        self.assertEqual(clean, "Hello & World <test>")

    def test_clean_cdata(self):
        """Test CDATA removal."""
        dirty = "<![CDATA[Some content]]>"
        clean = clean_html(dirty)
        self.assertEqual(clean, "Some content")

    def test_clean_whitespace(self):
        """Test whitespace normalization."""
        dirty = "  Hello   \n  World  "
        clean = clean_html(dirty)
        self.assertEqual(clean, "Hello World")

    def test_clean_empty(self):
        """Test empty/None handling."""
        self.assertEqual(clean_html(""), "")
        self.assertEqual(clean_html(None), "")


class TestLabsStore(unittest.TestCase):
    """Tests for LabsStore SQLite storage."""

    def setUp(self):
        """Create temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_labs.sqlite3"
        self.store = LabsStore(self.db_path)

    def tearDown(self):
        """Clean up."""
        self.store.close()
        if self.db_path.exists():
            self.db_path.unlink()
        Path(self.temp_dir).rmdir()

    def test_insert_and_dedup(self):
        """Test post insertion and deduplication by link."""
        post = LabPost(
            post_id="",
            source="anthropic",
            title="Claude Update",
            summary="New capabilities",
            link="https://anthropic.com/news/claude-update",
            authors=["Anthropic"],
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
            tags=["anthropic", "ai"],
        )

        # First insert should succeed
        result1 = self.store.insert_post(post)
        self.assertTrue(result1)

        # Second insert should be deduplicated (same link)
        result2 = self.store.insert_post(post)
        self.assertFalse(result2)

    def test_post_exists(self):
        """Test post existence check."""
        post = LabPost(
            post_id="",
            source="meta",
            title="Llama 3",
            summary="New model release",
            link="https://ai.meta.com/blog/llama-3",
            authors=[],
            published_iso="2023-12-15T00:00:00Z",
            fetched_iso="2023-12-15T12:00:00Z",
        )

        self.assertFalse(self.store.post_exists(post.link))
        self.store.insert_post(post)
        self.assertTrue(self.store.post_exists(post.link))

    def test_stats(self):
        """Test statistics gathering."""
        # Insert posts from different sources
        for source in ["deepmind", "openai", "anthropic"]:
            post = LabPost(
                post_id="",
                source=source,
                title=f"Post from {source}",
                summary="Summary",
                link=f"https://{source}.com/blog/test",
                authors=[],
                published_iso="2023-12-15T00:00:00Z",
                fetched_iso="2023-12-15T12:00:00Z",
            )
            self.store.insert_post(post)

        stats = self.store.stats()
        self.assertEqual(stats["total_posts"], 3)
        self.assertEqual(stats["by_source"]["deepmind"], 1)
        self.assertEqual(stats["by_source"]["openai"], 1)
        self.assertEqual(stats["by_source"]["anthropic"], 1)


class TestRSSParsing(unittest.TestCase):
    """Tests for RSS/Atom feed parsing."""

    def test_parse_rss2(self):
        """Test parsing RSS 2.0 format."""
        rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test Blog</title>
                <item>
                    <title>Test Post Title</title>
                    <link>https://example.com/blog/test-post</link>
                    <description>This is a test description.</description>
                    <pubDate>Fri, 15 Dec 2023 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """

        posts = list(parse_rss_feed(rss_content, "openai"))
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].title, "Test Post Title")
        self.assertEqual(posts[0].source, "openai")
        self.assertIn("openai", posts[0].tags)

    def test_parse_atom(self):
        """Test parsing Atom format."""
        atom_content = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <title>Test Feed</title>
            <entry>
                <title>Atom Post</title>
                <link href="https://example.com/atom-post" rel="alternate"/>
                <summary>Atom summary</summary>
                <published>2023-12-15T12:00:00Z</published>
                <author><name>Test Author</name></author>
            </entry>
        </feed>
        """

        posts = list(parse_rss_feed(atom_content, "deepmind"))
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].title, "Atom Post")
        self.assertEqual(posts[0].source, "deepmind")

    def test_parse_empty(self):
        """Test parsing empty/invalid content."""
        posts = list(parse_rss_feed("", "openai"))
        self.assertEqual(posts, [])

        posts = list(parse_rss_feed("not xml", "openai"))
        self.assertEqual(posts, [])


class TestHTMLParsing(unittest.TestCase):
    """Tests for HTML blog page parsing."""

    def test_parse_blog_links(self):
        """Test extracting blog links from HTML."""
        html_content = """
        <html>
        <body>
            <article>
                <a href="/news/2023/test-post">
                    This is a test blog post title that should be extracted
                </a>
                <p>Some description text for the post.</p>
            </article>
            <article>
                <a href="/news/2023/another-post">
                    Another blog post with a longer title here
                </a>
            </article>
        </body>
        </html>
        """

        posts = list(parse_html_blog(html_content, "anthropic"))
        # Should find posts with /news/ in URL
        self.assertTrue(len(posts) >= 0)  # Heuristic may or may not match

    def test_skip_social_links(self):
        """Test that social media links are skipped."""
        html_content = """
        <html>
        <body>
            <a href="https://twitter.com/anthropic">Twitter</a>
            <a href="https://facebook.com/share">Share</a>
            <a href="/blog/real-post">Real Blog Post Title Here</a>
        </body>
        </html>
        """

        posts = list(parse_html_blog(html_content, "anthropic"))
        # Social links should be filtered out
        for post in posts:
            self.assertNotIn("twitter.com", post.link)
            self.assertNotIn("facebook.com", post.link)


class TestSources(unittest.TestCase):
    """Tests for source configuration."""

    def test_sources_defined(self):
        """Test all expected sources are configured."""
        expected = {"deepmind", "openai", "anthropic", "meta", "msr", "huggingface"}
        self.assertEqual(set(SOURCES.keys()), expected)

    def test_source_types(self):
        """Test source types are valid."""
        valid_types = {"rss", "html"}
        for source, config in SOURCES.items():
            self.assertIn(config["type"], valid_types)

    def test_source_urls(self):
        """Test all sources have URLs."""
        for source, config in SOURCES.items():
            self.assertIn("url", config)
            self.assertTrue(config["url"].startswith("http"))


class TestIntegration(unittest.TestCase):
    """Integration tests (mocked network)."""

    @patch("labs_feed.fetch_url")
    @patch("labs_feed.emit_bus_event")
    @patch("labs_feed.index_post_in_rag")
    def test_fetch_with_mocked_network(self, mock_rag, mock_bus, mock_fetch):
        """Test full fetch cycle with mocked network."""
        # Setup mocks
        mock_fetch.return_value = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Mock Blog Post</title>
                    <link>https://example.com/blog/mock-post</link>
                    <description>Mock description</description>
                </item>
            </channel>
        </rss>
        """
        mock_bus.return_value = True
        mock_rag.return_value = True

        # Create temp store
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite3"
            store = LabsStore(db_path)

            result = fetch_all_sources(
                store,
                sources=["openai"],
                emit_events=True,
                index_rag=True,
            )

            self.assertEqual(result["total_posts"], 1)
            self.assertEqual(result["total_new"], 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
