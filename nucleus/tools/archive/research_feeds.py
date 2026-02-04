#!/usr/bin/env python3
"""
Research Feeds Aggregator Daemon
================================

Unified daemon that orchestrates both arXiv and industry labs feeds.
Manages cadence, provides aggregated API, and emits unified events.

Usage:
  python3 research_feeds.py status               # Show feed status
  python3 research_feeds.py fetch                # Fetch all feeds now
  python3 research_feeds.py daemon               # Run aggregated daemon
  python3 research_feeds.py recent --limit 20    # Show recent papers/posts
  python3 research_feeds.py stats                # Aggregated statistics
  python3 research_feeds.py export --format json # Export recent items

Bus Events Emitted:
  - sota.feeds.daemon_started
  - sota.feeds.fetch_cycle_complete
  - sota.feeds.daemon_stopped

Service Registry:
  id: research-feeds
  port: None (process, not port service)
  lineage: core.sota
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def find_rhizome_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return Path("/pluribus")


# ============================================================================
# Feed Configuration
# ============================================================================

FEED_CONFIG = {
    "arxiv": {
        "name": "arXiv Papers",
        "module": "arxiv_feed",
        "default_interval_s": 86400,  # Daily
        "priority": 1,
    },
    "labs": {
        "name": "Industry Labs",
        "module": "labs_feed",
        "default_interval_s": 604800,  # Weekly
        "priority": 2,
    },
}


# ============================================================================
# Aggregator
# ============================================================================


@dataclass
class FeedStatus:
    """Status of a feed."""
    name: str
    last_fetch_iso: str
    next_fetch_iso: str
    total_items: int
    new_items_last: int
    status: str


class ResearchFeedsAggregator:
    """Aggregates and orchestrates research feeds."""

    def __init__(self, root: Path):
        self.root = root
        self.tools_dir = Path(__file__).parent
        sys.path.insert(0, str(self.tools_dir))

    def _import_arxiv(self):
        import arxiv_feed
        return arxiv_feed

    def _import_labs(self):
        import labs_feed
        return labs_feed

    def fetch_arxiv(self, dry_run: bool = False, emit_events: bool = True, index_rag: bool = True) -> dict:
        """Fetch arXiv papers."""
        try:
            arxiv = self._import_arxiv()
            db_path = self.root / ".pluribus" / "sota" / "arxiv.sqlite3"
            store = arxiv.ArxivStore(db_path)
            result = arxiv.fetch_all_categories(store, emit_events=emit_events, index_rag=index_rag, dry_run=dry_run)
            store.close()
            return {"feed": "arxiv", "status": "success", **result}
        except Exception as e:
            return {"feed": "arxiv", "status": "error", "error": str(e)}

    def fetch_labs(self, dry_run: bool = False, emit_events: bool = True, index_rag: bool = True) -> dict:
        """Fetch industry labs posts."""
        try:
            labs = self._import_labs()
            db_path = self.root / ".pluribus" / "sota" / "labs_feed.sqlite3"
            store = labs.LabsStore(db_path)
            result = labs.fetch_all_sources(store, emit_events=emit_events, index_rag=index_rag, dry_run=dry_run)
            store.close()
            return {"feed": "labs", "status": "success", **result}
        except Exception as e:
            return {"feed": "labs", "status": "error", "error": str(e)}

    def fetch_all(self, dry_run: bool = False, emit_events: bool = True, index_rag: bool = True) -> dict:
        """Fetch all feeds."""
        results = {
            "fetched_iso": now_iso_utc(),
            "feeds": {},
            "total_new": 0,
        }

        # arXiv (higher priority, fetch first)
        arxiv_result = self.fetch_arxiv(dry_run, emit_events, index_rag)
        results["feeds"]["arxiv"] = arxiv_result
        results["total_new"] += arxiv_result.get("total_new", 0)

        # Brief pause between different feed types
        time.sleep(2)

        # Labs
        labs_result = self.fetch_labs(dry_run, emit_events, index_rag)
        results["feeds"]["labs"] = labs_result
        results["total_new"] += labs_result.get("total_new", 0)

        return results

    def get_status(self) -> dict:
        """Get aggregated status of all feeds."""
        status = {
            "checked_iso": now_iso_utc(),
            "feeds": {},
        }

        # arXiv status
        try:
            arxiv = self._import_arxiv()
            db_path = self.root / ".pluribus" / "sota" / "arxiv.sqlite3"
            if db_path.exists():
                store = arxiv.ArxivStore(db_path)
                stats = store.stats()
                history = store.get_fetch_history(1)
                store.close()
                status["feeds"]["arxiv"] = {
                    "name": "arXiv Papers",
                    "total_items": stats["total_papers"],
                    "indexed": stats["indexed_in_rag"],
                    "last_fetch": history[0] if history else None,
                    "status": "active",
                }
            else:
                status["feeds"]["arxiv"] = {"status": "not_initialized"}
        except Exception as e:
            status["feeds"]["arxiv"] = {"status": "error", "error": str(e)}

        # Labs status
        try:
            labs = self._import_labs()
            db_path = self.root / ".pluribus" / "sota" / "labs_feed.sqlite3"
            if db_path.exists():
                store = labs.LabsStore(db_path)
                stats = store.stats()
                history = store.get_fetch_history(1)
                store.close()
                status["feeds"]["labs"] = {
                    "name": "Industry Labs",
                    "total_items": stats["total_posts"],
                    "indexed": stats["indexed_in_rag"],
                    "last_fetch": history[0] if history else None,
                    "status": "active",
                }
            else:
                status["feeds"]["labs"] = {"status": "not_initialized"}
        except Exception as e:
            status["feeds"]["labs"] = {"status": "error", "error": str(e)}

        return status

    def get_recent_items(self, limit: int = 20) -> list[dict]:
        """Get recent items from all feeds, sorted by fetch time."""
        items = []

        # arXiv papers
        try:
            arxiv = self._import_arxiv()
            db_path = self.root / ".pluribus" / "sota" / "arxiv.sqlite3"
            if db_path.exists():
                store = arxiv.ArxivStore(db_path)
                for paper in store.get_recent_papers(limit):
                    items.append({
                        "type": "arxiv",
                        "id": paper["arxiv_id"],
                        "title": paper["title"],
                        "link": paper["link"],
                        "fetched_iso": paper["fetched_iso"],
                        "source": paper["source_category"],
                        "tags": paper["tags"],
                    })
                store.close()
        except Exception:
            pass

        # Labs posts
        try:
            labs = self._import_labs()
            db_path = self.root / ".pluribus" / "sota" / "labs_feed.sqlite3"
            if db_path.exists():
                store = labs.LabsStore(db_path)
                for post in store.get_recent_posts(limit):
                    items.append({
                        "type": "labs",
                        "id": post["post_id"],
                        "title": post["title"],
                        "link": post["link"],
                        "fetched_iso": post["fetched_iso"],
                        "source": post["source"],
                        "tags": post["tags"],
                    })
                store.close()
        except Exception:
            pass

        # Sort by fetch time (most recent first)
        items.sort(key=lambda x: x.get("fetched_iso", ""), reverse=True)
        return items[:limit]

    def get_stats(self) -> dict:
        """Get aggregated statistics."""
        stats = {
            "generated_iso": now_iso_utc(),
            "arxiv": {},
            "labs": {},
            "total_items": 0,
            "total_indexed": 0,
        }

        try:
            arxiv = self._import_arxiv()
            db_path = self.root / ".pluribus" / "sota" / "arxiv.sqlite3"
            if db_path.exists():
                store = arxiv.ArxivStore(db_path)
                arxiv_stats = store.stats()
                stats["arxiv"] = arxiv_stats
                stats["total_items"] += arxiv_stats["total_papers"]
                stats["total_indexed"] += arxiv_stats["indexed_in_rag"]
                store.close()
        except Exception:
            pass

        try:
            labs = self._import_labs()
            db_path = self.root / ".pluribus" / "sota" / "labs_feed.sqlite3"
            if db_path.exists():
                store = labs.LabsStore(db_path)
                labs_stats = store.stats()
                stats["labs"] = labs_stats
                stats["total_items"] += labs_stats["total_posts"]
                stats["total_indexed"] += labs_stats["indexed_in_rag"]
                store.close()
        except Exception:
            pass

        return stats


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
            actor="research-feeds",
            data=data,
            trace_id=None,
            run_id=None,
            durable=True,
        )
        return True
    except Exception:
        return False


# ============================================================================
# Daemon Mode
# ============================================================================

_RUNNING = True


def _handle_signal(sig, frame):
    global _RUNNING
    _RUNNING = False


def run_daemon(
    aggregator: ResearchFeedsAggregator,
    arxiv_interval_s: int,
    labs_interval_s: int,
    index_rag: bool,
):
    """Run aggregated daemon with different cadences per feed."""
    global _RUNNING

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(f"[research_feeds] Daemon started")
    print(f"  arXiv interval: {arxiv_interval_s}s ({arxiv_interval_s/3600:.1f}h)")
    print(f"  Labs interval: {labs_interval_s}s ({labs_interval_s/3600:.1f}h)")

    emit_bus_event("sota.feeds.daemon_started", "log", {
        "arxiv_interval_s": arxiv_interval_s,
        "labs_interval_s": labs_interval_s,
    })

    last_arxiv_fetch = 0.0
    last_labs_fetch = 0.0

    while _RUNNING:
        now = time.time()
        cycle_results = {"fetched_iso": now_iso_utc(), "feeds": {}}

        # Check if arXiv needs fetching
        if now - last_arxiv_fetch >= arxiv_interval_s:
            print(f"[research_feeds] Fetching arXiv...")
            try:
                result = aggregator.fetch_arxiv(emit_events=True, index_rag=index_rag)
                cycle_results["feeds"]["arxiv"] = result
                print(f"[research_feeds] arXiv: {result.get('total_new', 0)} new papers")
            except Exception as e:
                print(f"[research_feeds] arXiv error: {e}")
            last_arxiv_fetch = now

        # Check if labs needs fetching
        if now - last_labs_fetch >= labs_interval_s:
            print(f"[research_feeds] Fetching labs...")
            try:
                result = aggregator.fetch_labs(emit_events=True, index_rag=index_rag)
                cycle_results["feeds"]["labs"] = result
                print(f"[research_feeds] Labs: {result.get('total_new', 0)} new posts")
            except Exception as e:
                print(f"[research_feeds] Labs error: {e}")
            last_labs_fetch = now

        # Emit cycle complete if anything was fetched
        if cycle_results["feeds"]:
            emit_bus_event("sota.feeds.fetch_cycle_complete", "metric", cycle_results)

        # Sleep in small increments
        for _ in range(60):  # Check every minute for signals
            if not _RUNNING:
                break
            time.sleep(1)

    print("[research_feeds] Daemon stopped")
    emit_bus_event("sota.feeds.daemon_stopped", "log", {})


# ============================================================================
# CLI
# ============================================================================


def cmd_status(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    aggregator = ResearchFeedsAggregator(root)
    status = aggregator.get_status()

    print(f"[research_feeds] Status ({status['checked_iso']})")
    print("-" * 60)

    for feed_key, feed_status in status["feeds"].items():
        if feed_status.get("status") == "not_initialized":
            print(f"  {feed_key}: Not initialized")
        elif feed_status.get("status") == "error":
            print(f"  {feed_key}: Error - {feed_status.get('error')}")
        else:
            print(f"  {feed_key}: {feed_status.get('name', feed_key)}")
            print(f"    Total: {feed_status.get('total_items', 0)} items ({feed_status.get('indexed', 0)} indexed)")
            last = feed_status.get("last_fetch")
            if last:
                print(f"    Last fetch: {last.get('fetched_iso', 'N/A')} ({last.get('new_count', 0)} new)")

    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    aggregator = ResearchFeedsAggregator(root)

    results = aggregator.fetch_all(
        dry_run=args.dry_run,
        emit_events=not args.no_bus,
        index_rag=not args.no_rag,
    )

    print(f"\n[research_feeds] Summary ({results['fetched_iso']})")
    print(f"  Total new: {results['total_new']}")

    for feed_key, feed_result in results["feeds"].items():
        status = feed_result.get("status", "unknown")
        new = feed_result.get("total_new", 0)
        print(f"  {feed_key}: {new} new (status: {status})")

    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    aggregator = ResearchFeedsAggregator(root)

    run_daemon(
        aggregator,
        arxiv_interval_s=args.arxiv_interval,
        labs_interval_s=args.labs_interval,
        index_rag=not args.no_rag,
    )

    return 0


def cmd_recent(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    aggregator = ResearchFeedsAggregator(root)

    items = aggregator.get_recent_items(args.limit)

    print(f"[research_feeds] Recent Items ({len(items)})")
    print("-" * 80)

    for item in items:
        type_badge = "[arXiv]" if item["type"] == "arxiv" else "[Labs]"
        print(f"{type_badge} {item['source']}: {item['title'][:60]}...")
        print(f"         {item['link']}")

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    aggregator = ResearchFeedsAggregator(root)

    stats = aggregator.get_stats()
    print(json.dumps(stats, indent=2))

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    root = find_rhizome_root(Path.cwd())
    aggregator = ResearchFeedsAggregator(root)

    items = aggregator.get_recent_items(args.limit)

    if args.format == "json":
        print(json.dumps(items, indent=2))
    elif args.format == "ndjson":
        for item in items:
            print(json.dumps(item))
    elif args.format == "md":
        print("# Recent Research Items\n")
        for item in items:
            print(f"## {item['title']}")
            print(f"- **Type**: {item['type']}")
            print(f"- **Source**: {item['source']}")
            print(f"- **Link**: {item['link']}")
            print(f"- **Fetched**: {item['fetched_iso']}")
            print()

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research_feeds.py", description="Unified research feeds aggregator")

    sub = p.add_subparsers(dest="cmd", required=True)

    status_p = sub.add_parser("status", help="Show feed status")
    status_p.set_defaults(func=cmd_status)

    fetch_p = sub.add_parser("fetch", help="Fetch all feeds now")
    fetch_p.add_argument("--dry-run", action="store_true", help="Preview without storing")
    fetch_p.add_argument("--no-bus", action="store_true", help="Don't emit bus events")
    fetch_p.add_argument("--no-rag", action="store_true", help="Don't index in RAG")
    fetch_p.set_defaults(func=cmd_fetch)

    daemon_p = sub.add_parser("daemon", help="Run aggregated daemon")
    daemon_p.add_argument("--arxiv-interval", type=int, default=86400, help="arXiv fetch interval (seconds)")
    daemon_p.add_argument("--labs-interval", type=int, default=604800, help="Labs fetch interval (seconds)")
    daemon_p.add_argument("--no-rag", action="store_true", help="Don't index in RAG")
    daemon_p.set_defaults(func=cmd_daemon)

    recent_p = sub.add_parser("recent", help="Show recent items")
    recent_p.add_argument("--limit", type=int, default=20, help="Number of items")
    recent_p.set_defaults(func=cmd_recent)

    stats_p = sub.add_parser("stats", help="Show aggregated statistics")
    stats_p.set_defaults(func=cmd_stats)

    export_p = sub.add_parser("export", help="Export recent items")
    export_p.add_argument("--format", choices=["json", "ndjson", "md"], default="json", help="Export format")
    export_p.add_argument("--limit", type=int, default=100, help="Number of items")
    export_p.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
