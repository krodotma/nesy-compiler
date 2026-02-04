#!/usr/bin/env python3
"""
arXiv Feed Integration for Pluribus SOTA Pipeline
==================================================

Fetches RSS feeds from arXiv for target categories (cs.AI, cs.LG, cs.CL, stat.ML).
Parses, deduplicates, and emits bus events + optionally indexes into RAG.

Usage:
  python3 arxiv_feed.py fetch                     # Fetch and emit events
  python3 arxiv_feed.py fetch --dry-run           # Preview without emitting
  python3 arxiv_feed.py fetch --category cs.AI    # Single category
  python3 arxiv_feed.py daemon --interval 86400   # Daily daemon mode
  python3 arxiv_feed.py history                   # Show fetch history

Bus Events Emitted:
  - sota.arxiv.new_paper (per paper)
  - sota.arxiv.fetch_complete (batch summary)

Sextet Compliance:
  - Source: arXiv RSS (external)
  - Transducer: XML -> structured paper metadata
  - Regulator: Rate-limiting, dedup via sqlite
  - Memory: .pluribus/sota/arxiv.sqlite3 (history + dedup)
  - Feedback: Bus events for downstream processing
"""
from __future__ import annotations

import argparse
import hashlib
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

sys.dont_write_bytecode = True

# ============================================================================
# Configuration
# ============================================================================

# arXiv RSS base URL
ARXIV_RSS_BASE = "http://export.arxiv.org/rss/"

# Target categories with priority
CATEGORIES = {
    "cs.AI": {"priority": 1, "tags": ["arxiv", "ai"]},
    "cs.LG": {"priority": 1, "tags": ["arxiv", "ml"]},
    "cs.CL": {"priority": 2, "tags": ["arxiv", "nlp"]},
    "stat.ML": {"priority": 1, "tags": ["arxiv", "stats"]},
}

# Default fetch interval (24 hours in seconds)
DEFAULT_INTERVAL_S = 86400

# Rate limit between category fetches (seconds)
RATE_LIMIT_S = 3


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
class ArxivPaper:
    """Represents a parsed arXiv paper from RSS."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    link: str
    published_iso: str
    fetched_iso: str
    source_category: str
    content_hash: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.content_hash:
            # Deterministic hash for deduplication
            content = f"{self.arxiv_id}:{self.title}:{self.abstract[:500]}"
            self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================================
# RSS Parser
# ============================================================================


def fetch_rss(category: str, timeout: int = 30) -> str:
    """Fetch RSS feed for a category."""
    url = f"{ARXIV_RSS_BASE}{category}"
    headers = {
        "User-Agent": "Pluribus-SOTA/1.0 (Research Feed Aggregator; contact@pluribus.dev)",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_arxiv_rss(xml_content: str, source_category: str) -> Generator[ArxivPaper, None, None]:
    """Parse arXiv RSS XML and yield ArxivPaper objects."""
    # arXiv RSS uses RDF format with namespaces
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rss": "http://purl.org/rss/1.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "taxo": "http://purl.org/rss/1.0/modules/taxonomy/",
        "syn": "http://purl.org/rss/1.0/modules/syndication/",
    }

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return

    # Handle both RDF and standard RSS formats
    # Try namespaced first, then fallback to non-namespaced
    items = root.findall(".//{http://purl.org/rss/1.0/}item")
    if not items:
        items = root.findall(".//item")

    cat_config = CATEGORIES.get(source_category, {})
    base_tags = cat_config.get("tags", ["arxiv"])
    fetched_iso = now_iso_utc()

    for item in items:
        try:
            # Extract link (contains arxiv ID)
            # Try namespaced first, then non-namespaced
            link_elem = item.find("{http://purl.org/rss/1.0/}link")
            if link_elem is None:
                link_elem = item.find("link")
            link = ""
            if link_elem is not None and link_elem.text:
                link = link_elem.text.strip()

            # Extract arxiv ID from link (format: http://arxiv.org/abs/XXXX.XXXXX)
            arxiv_id = ""
            if link:
                match = re.search(r"abs/([^\s/]+)", link)
                if match:
                    arxiv_id = match.group(1)

            if not arxiv_id:
                continue

            # Title
            title_elem = item.find("{http://purl.org/rss/1.0/}title")
            if title_elem is None:
                title_elem = item.find("title")
            title = ""
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()
            # Clean up title (remove arxiv ID prefix if present)
            title = re.sub(r"^\s*[\w.-]+\s*[:\-]\s*", "", title)

            # Description/Abstract
            desc_elem = item.find("{http://purl.org/rss/1.0/}description")
            if desc_elem is None:
                desc_elem = item.find("description")
            abstract = ""
            if desc_elem is not None and desc_elem.text:
                abstract = desc_elem.text.strip()
            # Clean HTML tags from abstract
            abstract = re.sub(r"<[^>]+>", "", abstract)

            # Authors (dc:creator)
            authors = []
            for creator in item.findall("{http://purl.org/dc/elements/1.1/}creator"):
                if creator.text:
                    authors.append(creator.text.strip())
            # Fallback: parse from description if no dc:creator
            if not authors and abstract:
                author_match = re.search(r"Authors?:\s*([^<]+)", abstract)
                if author_match:
                    authors = [a.strip() for a in author_match.group(1).split(",")]

            # Categories (taxo:topic or dc:subject)
            categories = [source_category]
            for subj in item.findall("{http://purl.org/dc/elements/1.1/}subject"):
                if subj.text and subj.text not in categories:
                    categories.append(subj.text.strip())

            # Date (dc:date)
            date_elem = item.find("{http://purl.org/dc/elements/1.1/}date")
            published_iso = fetched_iso
            if date_elem is not None and date_elem.text:
                published_iso = date_elem.text.strip()

            yield ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                link=link,
                published_iso=published_iso,
                fetched_iso=fetched_iso,
                source_category=source_category,
                tags=list(base_tags),
            )

        except Exception:
            continue


# ============================================================================
# Storage / Deduplication
# ============================================================================


class ArxivStore:
    """SQLite-based storage for arXiv papers (dedup + history)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        ensure_dir(db_path.parent)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors TEXT,  -- JSON array
                categories TEXT,  -- JSON array
                link TEXT,
                published_iso TEXT,
                fetched_iso TEXT,
                source_category TEXT,
                content_hash TEXT,
                tags TEXT,  -- JSON array
                indexed_in_rag INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS fetch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                fetched_iso TEXT,
                paper_count INTEGER,
                new_count INTEGER,
                status TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_papers_fetched ON papers(fetched_iso);
            CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(source_category);
            CREATE INDEX IF NOT EXISTS idx_papers_hash ON papers(content_hash);
        """)
        self.conn.commit()

    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if paper already exists."""
        cur = self.conn.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        return cur.fetchone() is not None

    def insert_paper(self, paper: ArxivPaper) -> bool:
        """Insert paper if not exists. Returns True if inserted."""
        if self.paper_exists(paper.arxiv_id):
            return False

        self.conn.execute(
            """INSERT INTO papers
               (arxiv_id, title, abstract, authors, categories, link,
                published_iso, fetched_iso, source_category, content_hash, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper.arxiv_id,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                json.dumps(paper.categories),
                paper.link,
                paper.published_iso,
                paper.fetched_iso,
                paper.source_category,
                paper.content_hash,
                json.dumps(paper.tags),
            ),
        )
        self.conn.commit()
        return True

    def mark_indexed(self, arxiv_id: str):
        """Mark paper as indexed in RAG."""
        self.conn.execute("UPDATE papers SET indexed_in_rag = 1 WHERE arxiv_id = ?", (arxiv_id,))
        self.conn.commit()

    def record_fetch(self, category: str, paper_count: int, new_count: int, status: str):
        """Record fetch history."""
        self.conn.execute(
            "INSERT INTO fetch_history (category, fetched_iso, paper_count, new_count, status) VALUES (?, ?, ?, ?, ?)",
            (category, now_iso_utc(), paper_count, new_count, status),
        )
        self.conn.commit()

    def get_recent_papers(self, limit: int = 50) -> list[dict]:
        """Get recently fetched papers."""
        cur = self.conn.execute(
            """SELECT arxiv_id, title, authors, categories, link, published_iso, fetched_iso, source_category, tags
               FROM papers ORDER BY fetched_iso DESC LIMIT ?""",
            (limit,),
        )
        return [
            {
                "arxiv_id": r[0],
                "title": r[1],
                "authors": json.loads(r[2]),
                "categories": json.loads(r[3]),
                "link": r[4],
                "published_iso": r[5],
                "fetched_iso": r[6],
                "source_category": r[7],
                "tags": json.loads(r[8]),
            }
            for r in cur.fetchall()
        ]

    def get_fetch_history(self, limit: int = 20) -> list[dict]:
        """Get fetch history."""
        cur = self.conn.execute(
            "SELECT category, fetched_iso, paper_count, new_count, status FROM fetch_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [
            {"category": r[0], "fetched_iso": r[1], "paper_count": r[2], "new_count": r[3], "status": r[4]}
            for r in cur.fetchall()
        ]

    def stats(self) -> dict:
        """Get storage statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        indexed = self.conn.execute("SELECT COUNT(*) FROM papers WHERE indexed_in_rag = 1").fetchone()[0]
        by_category = {}
        for cat in CATEGORIES:
            cnt = self.conn.execute("SELECT COUNT(*) FROM papers WHERE source_category = ?", (cat,)).fetchone()[0]
            by_category[cat] = cnt
        return {
            "total_papers": total,
            "indexed_in_rag": indexed,
            "by_category": by_category,
        }

    def close(self):
        self.conn.close()


# ============================================================================
# Bus Integration
# ============================================================================


def emit_bus_event(topic: str, kind: str, data: dict, level: str = "info"):
    """Emit event to agent bus."""
    try:
        # Import bus paths resolver
        tools_dir = Path(__file__).parent
        sys.path.insert(0, str(tools_dir))
        from agent_bus import emit_event, resolve_bus_paths

        paths = resolve_bus_paths(None)
        emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor="arxiv-feed",
            data=data,
            trace_id=None,
            run_id=None,
            durable=True,
        )
        return True
    except Exception as e:
        sys.stderr.write(f"[arxiv_feed] Bus emit failed: {e}\n")
        return False


def index_paper_in_rag(paper: ArxivPaper) -> bool:
    """Index paper in RAG vector store."""
    try:
        tools_dir = Path(__file__).parent
        sys.path.insert(0, str(tools_dir))
        from rag_vector import VectorRAG, DB_PATH

        # Construct event-like object for RAG indexer
        event = {
            "id": f"arxiv:{paper.arxiv_id}",
            "topic": "ingest.text",
            "kind": "arxiv_paper",
            "iso": paper.fetched_iso,
            "data": {
                "text": f"Title: {paper.title}\n\nAuthors: {', '.join(paper.authors)}\n\nAbstract: {paper.abstract}\n\nCategories: {', '.join(paper.categories)}\n\nLink: {paper.link}",
                "source": "arxiv",
                "arxiv_id": paper.arxiv_id,
            },
        }

        rag = VectorRAG(Path(DB_PATH), load_model=True)
        rag.init_schema()
        result = rag.index_event(event)
        rag.close()
        return result is not None
    except Exception as e:
        sys.stderr.write(f"[arxiv_feed] RAG index failed: {e}\n")
        return False


# ============================================================================
# Main Fetch Logic
# ============================================================================


def fetch_category(
    store: ArxivStore,
    category: str,
    dry_run: bool = False,
    emit_events: bool = True,
    index_rag: bool = True,
) -> dict:
    """Fetch papers for a single category."""
    result = {
        "category": category,
        "status": "success",
        "paper_count": 0,
        "new_count": 0,
        "errors": [],
    }

    try:
        xml_content = fetch_rss(category)
    except urllib.error.URLError as e:
        result["status"] = "fetch_error"
        result["errors"].append(str(e))
        return result
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        return result

    papers = list(parse_arxiv_rss(xml_content, category))
    result["paper_count"] = len(papers)

    for paper in papers:
        if dry_run:
            # Preview mode
            print(f"  [DRY-RUN] {paper.arxiv_id}: {paper.title[:60]}...")
            result["new_count"] += 1
            continue

        # Insert into store (dedup)
        is_new = store.insert_paper(paper)
        if is_new:
            result["new_count"] += 1

            # Emit bus event
            if emit_events:
                emit_bus_event(
                    topic="sota.arxiv.new_paper",
                    kind="artifact",
                    data=asdict(paper),
                )

            # Index in RAG
            if index_rag:
                if index_paper_in_rag(paper):
                    store.mark_indexed(paper.arxiv_id)

    # Record fetch history
    if not dry_run:
        store.record_fetch(category, result["paper_count"], result["new_count"], result["status"])

    return result


def fetch_all_categories(
    store: ArxivStore,
    categories: list[str] | None = None,
    dry_run: bool = False,
    emit_events: bool = True,
    index_rag: bool = True,
) -> dict:
    """Fetch papers from all configured categories."""
    target_categories = categories or list(CATEGORIES.keys())
    results = {
        "fetched_iso": now_iso_utc(),
        "categories": {},
        "total_papers": 0,
        "total_new": 0,
    }

    for i, category in enumerate(target_categories):
        print(f"[arxiv_feed] Fetching {category}...")
        cat_result = fetch_category(store, category, dry_run, emit_events, index_rag)
        results["categories"][category] = cat_result
        results["total_papers"] += cat_result["paper_count"]
        results["total_new"] += cat_result["new_count"]

        # Rate limiting between fetches
        if i < len(target_categories) - 1:
            time.sleep(RATE_LIMIT_S)

    # Emit batch complete event
    if emit_events and not dry_run:
        emit_bus_event(
            topic="sota.arxiv.fetch_complete",
            kind="metric",
            data={
                "total_papers": results["total_papers"],
                "total_new": results["total_new"],
                "categories": list(results["categories"].keys()),
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


def run_daemon(store: ArxivStore, interval_s: int, index_rag: bool):
    """Run in daemon mode with periodic fetches."""
    global _RUNNING

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(f"[arxiv_feed] Daemon started (interval={interval_s}s)")
    emit_bus_event("sota.arxiv.daemon_started", "log", {"interval_s": interval_s})

    while _RUNNING:
        try:
            results = fetch_all_categories(store, emit_events=True, index_rag=index_rag)
            print(f"[arxiv_feed] Fetch complete: {results['total_new']} new papers")
        except Exception as e:
            sys.stderr.write(f"[arxiv_feed] Daemon error: {e}\n")
            emit_bus_event("sota.arxiv.daemon_error", "log", {"error": str(e)}, level="error")

        # Sleep in small increments to allow signal handling
        for _ in range(interval_s):
            if not _RUNNING:
                break
            time.sleep(1)

    print("[arxiv_feed] Daemon stopped")
    emit_bus_event("sota.arxiv.daemon_stopped", "log", {})


# ============================================================================
# CLI
# ============================================================================


def cmd_fetch(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    db_path = root / ".pluribus" / "sota" / "arxiv.sqlite3"
    store = ArxivStore(db_path)

    categories = [args.category] if args.category else None
    results = fetch_all_categories(
        store,
        categories=categories,
        dry_run=args.dry_run,
        emit_events=not args.no_bus,
        index_rag=not args.no_rag,
    )

    print(f"\n[arxiv_feed] Summary:")
    print(f"  Total papers: {results['total_papers']}")
    print(f"  New papers:   {results['total_new']}")
    for cat, res in results["categories"].items():
        print(f"  {cat}: {res['new_count']}/{res['paper_count']} new (status: {res['status']})")

    store.close()
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    db_path = root / ".pluribus" / "sota" / "arxiv.sqlite3"
    store = ArxivStore(db_path)

    run_daemon(store, args.interval, index_rag=not args.no_rag)

    store.close()
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    db_path = root / ".pluribus" / "sota" / "arxiv.sqlite3"
    store = ArxivStore(db_path)

    print("[arxiv_feed] Fetch History:")
    for h in store.get_fetch_history(args.limit):
        print(f"  {h['fetched_iso']} | {h['category']:<8} | {h['new_count']}/{h['paper_count']} new | {h['status']}")

    print("\n[arxiv_feed] Recent Papers:")
    for p in store.get_recent_papers(args.limit):
        print(f"  {p['arxiv_id']}: {p['title'][:60]}...")

    print("\n[arxiv_feed] Stats:")
    stats = store.stats()
    print(f"  Total: {stats['total_papers']} papers ({stats['indexed_in_rag']} indexed)")
    for cat, cnt in stats["by_category"].items():
        print(f"  {cat}: {cnt}")

    store.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    db_path = root / ".pluribus" / "sota" / "arxiv.sqlite3"
    store = ArxivStore(db_path)

    stats = store.stats()
    print(json.dumps(stats, indent=2))

    store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arxiv_feed.py", description="arXiv RSS feed integration for SOTA pipeline")

    sub = p.add_subparsers(dest="cmd", required=True)

    fetch_p = sub.add_parser("fetch", help="Fetch papers from arXiv RSS")
    fetch_p.add_argument("--category", help="Single category to fetch (default: all)")
    fetch_p.add_argument("--dry-run", action="store_true", help="Preview without storing/emitting")
    fetch_p.add_argument("--no-bus", action="store_true", help="Don't emit bus events")
    fetch_p.add_argument("--no-rag", action="store_true", help="Don't index in RAG")
    fetch_p.set_defaults(func=cmd_fetch)

    daemon_p = sub.add_parser("daemon", help="Run as daemon with periodic fetches")
    daemon_p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S, help="Fetch interval in seconds")
    daemon_p.add_argument("--no-rag", action="store_true", help="Don't index in RAG")
    daemon_p.set_defaults(func=cmd_daemon)

    history_p = sub.add_parser("history", help="Show fetch history and recent papers")
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
