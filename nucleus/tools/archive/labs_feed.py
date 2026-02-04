#!/usr/bin/env python3
"""
Industry Labs Feed Integration for Pluribus SOTA Pipeline
==========================================================

Fetches research updates from major AI labs (DeepMind, OpenAI, Anthropic, Meta AI, MSR).
Parses RSS/Atom feeds or HTML pages, deduplicates, and emits bus events.

Usage:
  python3 labs_feed.py fetch                      # Fetch from all labs
  python3 labs_feed.py fetch --source deepmind    # Single source
  python3 labs_feed.py fetch --dry-run            # Preview
  python3 labs_feed.py daemon --interval 604800   # Weekly daemon
  python3 labs_feed.py sources                    # List configured sources
  python3 labs_feed.py history                    # Show fetch history

Bus Events Emitted:
  - sota.labs.new_post (per post)
  - sota.labs.fetch_complete (batch summary)

Supported Sources:
  - deepmind: Google DeepMind Research Blog
  - openai: OpenAI Research Updates
  - anthropic: Anthropic Research Blog
  - meta: Meta AI Research Blog
  - msr: Microsoft Research Blog
  - huggingface: Hugging Face Blog

Sextet Compliance:
  - Source: Lab blogs/RSS (external)
  - Transducer: RSS/HTML -> structured post metadata
  - Regulator: Rate-limiting, dedup via sqlite
  - Memory: .pluribus/sota/labs_feed.sqlite3
  - Feedback: Bus events for downstream processing
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin, urlparse

sys.dont_write_bytecode = True

# ============================================================================
# Configuration
# ============================================================================

# Weekly default (7 days in seconds)
DEFAULT_INTERVAL_S = 604800

# Rate limit between source fetches (seconds)
RATE_LIMIT_S = 5

# Source configurations
SOURCES = {
    "deepmind": {
        "name": "Google DeepMind",
        "priority": 2,
        "cadence": "weekly",
        "type": "rss",
        "url": "https://deepmind.google/blog/rss.xml",
        "fallback_url": "https://www.deepmind.com/blog",
        "tags": ["deepmind", "google", "ai", "research"],
    },
    "openai": {
        "name": "OpenAI Research",
        "priority": 2,
        "cadence": "weekly",
        "type": "rss",
        "url": "https://openai.com/blog/rss/",
        "fallback_url": "https://openai.com/research",
        "tags": ["openai", "ai", "research"],
    },
    "anthropic": {
        "name": "Anthropic Research",
        "priority": 2,
        "cadence": "weekly",
        "type": "html",  # Anthropic doesn't have RSS
        "url": "https://www.anthropic.com/news",
        "tags": ["anthropic", "ai", "safety", "research"],
    },
    "meta": {
        "name": "Meta AI Research",
        "priority": 2,
        "cadence": "weekly",
        "type": "rss",
        "url": "https://ai.meta.com/blog/rss/",
        "fallback_url": "https://ai.meta.com/blog/",
        "tags": ["meta", "facebook", "ai", "research"],
    },
    "msr": {
        "name": "Microsoft Research",
        "priority": 3,
        "cadence": "biweekly",
        "type": "rss",
        "url": "https://www.microsoft.com/en-us/research/feed/",
        "tags": ["microsoft", "msr", "research"],
    },
    "huggingface": {
        "name": "Hugging Face Blog",
        "priority": 3,
        "cadence": "biweekly",
        "type": "rss",
        "url": "https://huggingface.co/blog/feed.xml",
        "tags": ["huggingface", "ml", "open-source"],
    },
}


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_rhizome_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return Path("/pluribus")


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class LabPost:
    """Represents a parsed blog post from an AI lab."""
    post_id: str  # Derived hash
    source: str
    title: str
    summary: str
    link: str
    authors: list[str]
    published_iso: str
    fetched_iso: str
    content_hash: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.post_id:
            # Deterministic ID from link
            self.post_id = hashlib.sha256(self.link.encode()).hexdigest()[:16]
        if not self.content_hash:
            content = f"{self.link}:{self.title}:{self.summary[:300]}"
            self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================================
# Fetcher Utilities
# ============================================================================


def fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch URL with appropriate headers."""
    headers = {
        "User-Agent": "Pluribus-SOTA/1.0 (Research Feed Aggregator; https://pluribus.dev)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/html, */*",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def clean_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    if not text:
        return ""
    # Remove CDATA
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# RSS/Atom Parser
# ============================================================================


def parse_rss_feed(xml_content: str, source_key: str) -> Generator[LabPost, None, None]:
    """Parse RSS/Atom feed and yield LabPost objects."""
    source_config = SOURCES.get(source_key, {})
    base_tags = source_config.get("tags", [])
    fetched_iso = now_iso_utc()

    # Namespaces for Atom, RSS, Dublin Core, Content
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "dc": "http://purl.org/dc/elements/1.1/",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "media": "http://search.yahoo.com/mrss/",
    }

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return

    # Detect format: Atom vs RSS
    is_atom = "atom" in root.tag.lower() or root.find(".//atom:entry", ns) is not None

    if is_atom:
        entries = root.findall(".//atom:entry", ns) or root.findall(".//{http://www.w3.org/2005/Atom}entry")
        for entry in entries:
            try:
                # Atom format
                title_elem = entry.find("atom:title", ns) or entry.find("{http://www.w3.org/2005/Atom}title")
                title = clean_html(title_elem.text if title_elem is not None and title_elem.text else "")

                # Link (Atom uses href attribute)
                link = ""
                link_elem = entry.find("atom:link[@rel='alternate']", ns) or entry.find("{http://www.w3.org/2005/Atom}link")
                if link_elem is not None:
                    link = link_elem.get("href", "")

                # Summary/Content
                summary_elem = entry.find("atom:summary", ns) or entry.find("{http://www.w3.org/2005/Atom}summary")
                content_elem = entry.find("atom:content", ns) or entry.find("{http://www.w3.org/2005/Atom}content")
                summary = clean_html(
                    (summary_elem.text if summary_elem is not None and summary_elem.text else "") or
                    (content_elem.text if content_elem is not None and content_elem.text else "")
                )[:1000]

                # Published date
                pub_elem = entry.find("atom:published", ns) or entry.find("{http://www.w3.org/2005/Atom}published")
                upd_elem = entry.find("atom:updated", ns) or entry.find("{http://www.w3.org/2005/Atom}updated")
                published_iso = (
                    (pub_elem.text if pub_elem is not None and pub_elem.text else "") or
                    (upd_elem.text if upd_elem is not None and upd_elem.text else "") or
                    fetched_iso
                )

                # Authors
                authors = []
                for author_elem in entry.findall("atom:author/atom:name", ns):
                    if author_elem.text:
                        authors.append(author_elem.text.strip())

                if title and link:
                    yield LabPost(
                        post_id="",
                        source=source_key,
                        title=title,
                        summary=summary,
                        link=link,
                        authors=authors,
                        published_iso=published_iso,
                        fetched_iso=fetched_iso,
                        tags=list(base_tags),
                    )
            except Exception:
                continue
    else:
        # RSS 2.0 format
        items = root.findall(".//item")
        for item in items:
            try:
                title_elem = item.find("title")
                title = clean_html(title_elem.text if title_elem is not None and title_elem.text else "")

                link_elem = item.find("link")
                link = (link_elem.text or "").strip() if link_elem is not None else ""

                # Description
                desc_elem = item.find("description")
                content_elem = item.find("content:encoded", ns)
                summary = clean_html(
                    (desc_elem.text if desc_elem is not None and desc_elem.text else "") or
                    (content_elem.text if content_elem is not None and content_elem.text else "")
                )[:1000]

                # Published date
                pub_elem = item.find("pubDate")
                dc_date = item.find("dc:date", ns)
                published_iso = (
                    (pub_elem.text if pub_elem is not None and pub_elem.text else "") or
                    (dc_date.text if dc_date is not None and dc_date.text else "") or
                    fetched_iso
                )

                # Authors
                authors = []
                creator_elem = item.find("dc:creator", ns)
                author_elem = item.find("author")
                if creator_elem is not None and creator_elem.text:
                    authors.append(creator_elem.text.strip())
                elif author_elem is not None and author_elem.text:
                    authors.append(author_elem.text.strip())

                if title and link:
                    yield LabPost(
                        post_id="",
                        source=source_key,
                        title=title,
                        summary=summary,
                        link=link,
                        authors=authors,
                        published_iso=published_iso,
                        fetched_iso=fetched_iso,
                        tags=list(base_tags),
                    )
            except Exception:
                continue


# ============================================================================
# HTML Scraper (for sources without RSS)
# ============================================================================


def parse_html_blog(html_content: str, source_key: str) -> Generator[LabPost, None, None]:
    """Parse HTML blog page (heuristic extraction)."""
    source_config = SOURCES.get(source_key, {})
    base_tags = source_config.get("tags", [])
    base_url = source_config.get("url", "")
    fetched_iso = now_iso_utc()

    # Simple heuristic extraction for common blog patterns
    # This is intentionally simple - for production you'd use BeautifulSoup

    # Pattern 1: <article> or <div class="post"> with <h2> or <h3> titles
    # Pattern 2: <a href="..."> with card/post container

    # Extract all links with their context
    link_pattern = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)

    seen_links = set()

    for match in link_pattern.finditer(html_content):
        href = match.group(1)
        anchor_content = clean_html(match.group(2))

        # Skip non-blog links
        if not anchor_content or len(anchor_content) < 10:
            continue
        if href.startswith("#") or href.startswith("javascript:"):
            continue
        if any(skip in href.lower() for skip in ["twitter.com", "facebook.com", "linkedin.com", "youtube.com", "/tag/", "/category/", "/author/"]):
            continue

        # Normalize URL
        if href.startswith("/"):
            parsed = urlparse(base_url)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"
        elif not href.startswith("http"):
            href = urljoin(base_url, href)

        # Dedup within this fetch
        if href in seen_links:
            continue
        seen_links.add(href)

        # Heuristic: title is the anchor text if it looks like a title
        title = anchor_content[:200]

        # Try to extract surrounding text for summary
        # Look for nearby <p> content
        pos = match.start()
        context_start = max(0, pos - 500)
        context_end = min(len(html_content), pos + 1000)
        context = html_content[context_start:context_end]

        # Extract paragraphs near the link
        p_pattern = re.compile(r"<p[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
        summaries = [clean_html(p.group(1)) for p in p_pattern.finditer(context)]
        summary = " ".join(s for s in summaries if s and len(s) > 20)[:500]

        # Filter: must look like a blog post (title length, URL pattern)
        if len(title) < 15:
            continue
        if not any(indicator in href.lower() for indicator in ["/blog/", "/post/", "/news/", "/research/", "/article/"]):
            # Also accept if URL ends with year/slug pattern
            if not re.search(r"/20\d\d/", href):
                continue

        yield LabPost(
            post_id="",
            source=source_key,
            title=title,
            summary=summary,
            link=href,
            authors=[],
            published_iso=fetched_iso,  # HTML rarely has dates we can extract simply
            fetched_iso=fetched_iso,
            tags=list(base_tags),
        )


# ============================================================================
# Storage
# ============================================================================


class LabsStore:
    """SQLite storage for lab posts."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        ensure_dir(db_path.parent)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS posts (
                post_id TEXT PRIMARY KEY,
                source TEXT,
                title TEXT,
                summary TEXT,
                link TEXT UNIQUE,
                authors TEXT,  -- JSON array
                published_iso TEXT,
                fetched_iso TEXT,
                content_hash TEXT,
                tags TEXT,  -- JSON array
                indexed_in_rag INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS fetch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                fetched_iso TEXT,
                post_count INTEGER,
                new_count INTEGER,
                status TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_posts_fetched ON posts(fetched_iso);
            CREATE INDEX IF NOT EXISTS idx_posts_source ON posts(source);
        """)
        self.conn.commit()

    def post_exists(self, link: str) -> bool:
        """Check if post already exists (by link)."""
        cur = self.conn.execute("SELECT 1 FROM posts WHERE link = ?", (link,))
        return cur.fetchone() is not None

    def insert_post(self, post: LabPost) -> bool:
        """Insert post if not exists. Returns True if inserted."""
        if self.post_exists(post.link):
            return False

        try:
            self.conn.execute(
                """INSERT INTO posts
                   (post_id, source, title, summary, link, authors,
                    published_iso, fetched_iso, content_hash, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    post.post_id,
                    post.source,
                    post.title,
                    post.summary,
                    post.link,
                    json.dumps(post.authors),
                    post.published_iso,
                    post.fetched_iso,
                    post.content_hash,
                    json.dumps(post.tags),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def mark_indexed(self, post_id: str):
        """Mark post as indexed in RAG."""
        self.conn.execute("UPDATE posts SET indexed_in_rag = 1 WHERE post_id = ?", (post_id,))
        self.conn.commit()

    def record_fetch(self, source: str, post_count: int, new_count: int, status: str):
        """Record fetch history."""
        self.conn.execute(
            "INSERT INTO fetch_history (source, fetched_iso, post_count, new_count, status) VALUES (?, ?, ?, ?, ?)",
            (source, now_iso_utc(), post_count, new_count, status),
        )
        self.conn.commit()

    def get_recent_posts(self, limit: int = 50) -> list[dict]:
        """Get recently fetched posts."""
        cur = self.conn.execute(
            """SELECT post_id, source, title, link, published_iso, fetched_iso, tags
               FROM posts ORDER BY fetched_iso DESC LIMIT ?""",
            (limit,),
        )
        return [
            {
                "post_id": r[0],
                "source": r[1],
                "title": r[2],
                "link": r[3],
                "published_iso": r[4],
                "fetched_iso": r[5],
                "tags": json.loads(r[6]),
            }
            for r in cur.fetchall()
        ]

    def get_fetch_history(self, limit: int = 20) -> list[dict]:
        """Get fetch history."""
        cur = self.conn.execute(
            "SELECT source, fetched_iso, post_count, new_count, status FROM fetch_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [
            {"source": r[0], "fetched_iso": r[1], "post_count": r[2], "new_count": r[3], "status": r[4]}
            for r in cur.fetchall()
        ]

    def stats(self) -> dict:
        """Get storage statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        indexed = self.conn.execute("SELECT COUNT(*) FROM posts WHERE indexed_in_rag = 1").fetchone()[0]
        by_source = {}
        for src in SOURCES:
            cnt = self.conn.execute("SELECT COUNT(*) FROM posts WHERE source = ?", (src,)).fetchone()[0]
            by_source[src] = cnt
        return {
            "total_posts": total,
            "indexed_in_rag": indexed,
            "by_source": by_source,
        }

    def close(self):
        self.conn.close()


# ============================================================================
# Bus Integration
# ============================================================================


def emit_bus_event(topic: str, kind: str, data: dict, level: str = "info"):
    """Emit event to agent bus."""
    try:
        tools_dir = Path(__file__).parent
        sys.path.insert(0, str(tools_dir))
        from agent_bus import emit_event, resolve_bus_paths

        paths = resolve_bus_paths(None)
        emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor="labs-feed",
            data=data,
            trace_id=None,
            run_id=None,
            durable=True,
        )
        return True
    except Exception as e:
        sys.stderr.write(f"[labs_feed] Bus emit failed: {e}\n")
        return False


def index_post_in_rag(post: LabPost) -> bool:
    """Index post in RAG vector store."""
    try:
        tools_dir = Path(__file__).parent
        sys.path.insert(0, str(tools_dir))
        from rag_vector import VectorRAG, DB_PATH

        event = {
            "id": f"labs:{post.post_id}",
            "topic": "ingest.text",
            "kind": "lab_post",
            "iso": post.fetched_iso,
            "data": {
                "text": f"Title: {post.title}\n\nSource: {post.source}\n\nSummary: {post.summary}\n\nLink: {post.link}",
                "source": f"labs:{post.source}",
                "post_id": post.post_id,
            },
        }

        rag = VectorRAG(Path(DB_PATH), load_model=True)
        rag.init_schema()
        result = rag.index_event(event)
        rag.close()
        return result is not None
    except Exception as e:
        sys.stderr.write(f"[labs_feed] RAG index failed: {e}\n")
        return False


# ============================================================================
# Fetch Logic
# ============================================================================


def fetch_source(
    store: LabsStore,
    source_key: str,
    dry_run: bool = False,
    emit_events: bool = True,
    index_rag: bool = True,
) -> dict:
    """Fetch posts from a single source."""
    source_config = SOURCES.get(source_key)
    if not source_config:
        return {"source": source_key, "status": "unknown_source", "post_count": 0, "new_count": 0, "errors": []}

    result = {
        "source": source_key,
        "status": "success",
        "post_count": 0,
        "new_count": 0,
        "errors": [],
    }

    # Fetch content
    try:
        content = fetch_url(source_config["url"])
    except urllib.error.URLError as e:
        # Try fallback URL if available
        fallback = source_config.get("fallback_url")
        if fallback:
            try:
                content = fetch_url(fallback)
            except Exception as e2:
                result["status"] = "fetch_error"
                result["errors"].append(f"Primary: {e}, Fallback: {e2}")
                return result
        else:
            result["status"] = "fetch_error"
            result["errors"].append(str(e))
            return result
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        return result

    # Parse based on source type
    source_type = source_config.get("type", "rss")
    if source_type == "html":
        posts = list(parse_html_blog(content, source_key))
    else:
        posts = list(parse_rss_feed(content, source_key))

    result["post_count"] = len(posts)

    for post in posts:
        if dry_run:
            print(f"  [DRY-RUN] {post.source}: {post.title[:50]}...")
            result["new_count"] += 1
            continue

        is_new = store.insert_post(post)
        if is_new:
            result["new_count"] += 1

            if emit_events:
                emit_bus_event(
                    topic="sota.labs.new_post",
                    kind="artifact",
                    data=asdict(post),
                )

            if index_rag:
                if index_post_in_rag(post):
                    store.mark_indexed(post.post_id)

    if not dry_run:
        store.record_fetch(source_key, result["post_count"], result["new_count"], result["status"])

    return result


def fetch_all_sources(
    store: LabsStore,
    sources: list[str] | None = None,
    dry_run: bool = False,
    emit_events: bool = True,
    index_rag: bool = True,
) -> dict:
    """Fetch posts from all configured sources."""
    target_sources = sources or list(SOURCES.keys())
    results = {
        "fetched_iso": now_iso_utc(),
        "sources": {},
        "total_posts": 0,
        "total_new": 0,
    }

    for i, source_key in enumerate(target_sources):
        print(f"[labs_feed] Fetching {source_key} ({SOURCES.get(source_key, {}).get('name', source_key)})...")
        src_result = fetch_source(store, source_key, dry_run, emit_events, index_rag)
        results["sources"][source_key] = src_result
        results["total_posts"] += src_result["post_count"]
        results["total_new"] += src_result["new_count"]

        if i < len(target_sources) - 1:
            time.sleep(RATE_LIMIT_S)

    if emit_events and not dry_run:
        emit_bus_event(
            topic="sota.labs.fetch_complete",
            kind="metric",
            data={
                "total_posts": results["total_posts"],
                "total_new": results["total_new"],
                "sources": list(results["sources"].keys()),
                "fetched_iso": results["fetched_iso"],
            },
        )

    return results


# ============================================================================
# Daemon Mode
# ============================================================================


_RUNNING = True


def _handle_signal(sig, frame):
    global _RUNNING
    _RUNNING = False


def run_daemon(store: LabsStore, interval_s: int, index_rag: bool):
    """Run in daemon mode with periodic fetches."""
    global _RUNNING

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(f"[labs_feed] Daemon started (interval={interval_s}s)")
    emit_bus_event("sota.labs.daemon_started", "log", {"interval_s": interval_s})

    while _RUNNING:
        try:
            results = fetch_all_sources(store, emit_events=True, index_rag=index_rag)
            print(f"[labs_feed] Fetch complete: {results['total_new']} new posts")
        except Exception as e:
            sys.stderr.write(f"[labs_feed] Daemon error: {e}\n")
            emit_bus_event("sota.labs.daemon_error", "log", {"error": str(e)}, level="error")

        for _ in range(interval_s):
            if not _RUNNING:
                break
            time.sleep(1)

    print("[labs_feed] Daemon stopped")
    emit_bus_event("sota.labs.daemon_stopped", "log", {})


# ============================================================================
# CLI
# ============================================================================


def cmd_fetch(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    db_path = root / ".pluribus" / "sota" / "labs_feed.sqlite3"
    store = LabsStore(db_path)

    sources = [args.source] if args.source else None
    results = fetch_all_sources(
        store,
        sources=sources,
        dry_run=args.dry_run,
        emit_events=not args.no_bus,
        index_rag=not args.no_rag,
    )

    print(f"\n[labs_feed] Summary:")
    print(f"  Total posts: {results['total_posts']}")
    print(f"  New posts:   {results['total_new']}")
    for src, res in results["sources"].items():
        print(f"  {src}: {res['new_count']}/{res['post_count']} new (status: {res['status']})")

    store.close()
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    db_path = root / ".pluribus" / "sota" / "labs_feed.sqlite3"
    store = LabsStore(db_path)

    run_daemon(store, args.interval, index_rag=not args.no_rag)

    store.close()
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    print("[labs_feed] Configured Sources:")
    print(f"{'Key':<15} {'Name':<25} {'Type':<6} {'Priority'} {'Cadence'}")
    print("-" * 70)
    for key, cfg in SOURCES.items():
        print(f"{key:<15} {cfg['name']:<25} {cfg['type']:<6} P{cfg['priority']}       {cfg['cadence']}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    db_path = root / ".pluribus" / "sota" / "labs_feed.sqlite3"
    store = LabsStore(db_path)

    print("[labs_feed] Fetch History:")
    for h in store.get_fetch_history(args.limit):
        print(f"  {h['fetched_iso']} | {h['source']:<12} | {h['new_count']}/{h['post_count']} new | {h['status']}")

    print("\n[labs_feed] Recent Posts:")
    for p in store.get_recent_posts(args.limit):
        print(f"  [{p['source']}] {p['title'][:50]}...")

    print("\n[labs_feed] Stats:")
    stats = store.stats()
    print(f"  Total: {stats['total_posts']} posts ({stats['indexed_in_rag']} indexed)")
    for src, cnt in stats["by_source"].items():
        print(f"  {src}: {cnt}")

    store.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    db_path = root / ".pluribus" / "sota" / "labs_feed.sqlite3"
    store = LabsStore(db_path)

    stats = store.stats()
    print(json.dumps(stats, indent=2))

    store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="labs_feed.py", description="Industry labs feed integration for SOTA pipeline")

    sub = p.add_subparsers(dest="cmd", required=True)

    fetch_p = sub.add_parser("fetch", help="Fetch posts from lab blogs")
    fetch_p.add_argument("--source", choices=list(SOURCES.keys()), help="Single source to fetch")
    fetch_p.add_argument("--dry-run", action="store_true", help="Preview without storing/emitting")
    fetch_p.add_argument("--no-bus", action="store_true", help="Don't emit bus events")
    fetch_p.add_argument("--no-rag", action="store_true", help="Don't index in RAG")
    fetch_p.set_defaults(func=cmd_fetch)

    daemon_p = sub.add_parser("daemon", help="Run as daemon with periodic fetches")
    daemon_p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S, help="Fetch interval in seconds")
    daemon_p.add_argument("--no-rag", action="store_true", help="Don't index in RAG")
    daemon_p.set_defaults(func=cmd_daemon)

    sources_p = sub.add_parser("sources", help="List configured sources")
    sources_p.set_defaults(func=cmd_sources)

    history_p = sub.add_parser("history", help="Show fetch history and recent posts")
    history_p.add_argument("--limit", type=int, default=20, help="Number of items to show")
    history_p.set_defaults(func=cmd_history)

    stats_p = sub.add_parser("stats", help="Show storage statistics (JSON)")
    stats_p.set_defaults(func=cmd_stats)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
