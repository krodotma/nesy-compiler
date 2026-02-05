#!/usr/bin/env python3
"""
Step 128: Test Cache

Provides test result caching for faster re-runs.

PBTSO Phase: TEST, OBSERVE
Bus Topics:
- test.cache.hit (emits)
- test.cache.miss (emits)
- test.cache.store (emits)

Dependencies: Steps 101-127 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Constants
# ============================================================================

class CacheStrategy(Enum):
    """Cache invalidation strategies."""
    CONTENT_HASH = "content_hash"  # Based on file content hash
    MTIME = "mtime"  # Based on file modification time
    DEPENDENCY_HASH = "dependency_hash"  # Hash of test + dependencies
    GIT_HASH = "git_hash"  # Based on git commit/status


class CacheStatus(Enum):
    """Cache lookup status."""
    HIT = "hit"
    MISS = "miss"
    STALE = "stale"
    INVALID = "invalid"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class CacheEntry:
    """
    A cached test result entry.

    Attributes:
        cache_key: Unique cache key
        test_name: Test name/path
        status: Test result status
        duration_ms: Test duration
        content_hash: Hash of test file content
        dependencies_hash: Hash of test dependencies
        created_at: When entry was created
        expires_at: When entry expires
        hits: Number of cache hits
        metadata: Additional metadata
    """
    cache_key: str
    test_name: str
    status: str
    duration_ms: float
    content_hash: str
    dependencies_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0
    output: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_key": self.cache_key,
            "test_name": self.test_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "content_hash": self.content_hash,
            "dependencies_hash": self.dependencies_hash,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hits": self.hits,
            "output": self.output,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            cache_key=data.get("cache_key", ""),
            test_name=data.get("test_name", ""),
            status=data.get("status", "unknown"),
            duration_ms=data.get("duration_ms", 0),
            content_hash=data.get("content_hash", ""),
            dependencies_hash=data.get("dependencies_hash"),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            hits=data.get("hits", 0),
            output=data.get("output"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    stale: int = 0
    size_bytes: int = 0
    oldest_entry: Optional[float] = None
    newest_entry: Optional[float] = None

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entries": self.total_entries,
            "hits": self.hits,
            "misses": self.misses,
            "stale": self.stale,
            "hit_rate": self.hit_rate,
            "size_bytes": self.size_bytes,
            "oldest_entry": self.oldest_entry,
            "newest_entry": self.newest_entry,
        }


@dataclass
class CacheLookupResult:
    """Result of a cache lookup."""
    status: CacheStatus
    entry: Optional[CacheEntry] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "entry": self.entry.to_dict() if self.entry else None,
            "reason": self.reason,
        }


@dataclass
class CacheConfig:
    """
    Configuration for test caching.

    Attributes:
        cache_dir: Directory for cache storage
        strategy: Cache invalidation strategy
        ttl_s: Time-to-live for cache entries
        max_entries: Maximum cache entries
        max_size_mb: Maximum cache size in MB
        cache_failures: Whether to cache failures
        include_output: Whether to include test output
        dependency_patterns: Patterns for dependency tracking
    """
    cache_dir: str = ".pluribus/test-agent/cache"
    strategy: CacheStrategy = CacheStrategy.CONTENT_HASH
    ttl_s: int = 86400  # 24 hours
    max_entries: int = 10000
    max_size_mb: int = 100
    cache_failures: bool = False
    include_output: bool = False
    dependency_patterns: List[str] = field(default_factory=lambda: ["*.py"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_dir": self.cache_dir,
            "strategy": self.strategy.value,
            "ttl_s": self.ttl_s,
            "max_entries": self.max_entries,
            "max_size_mb": self.max_size_mb,
            "cache_failures": self.cache_failures,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class CacheBus:
    """Bus interface for cache with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Test Cache
# ============================================================================

class TestCache:
    """
    Caches test results for faster re-runs.

    Features:
    - Content-based cache keys
    - Multiple invalidation strategies
    - TTL-based expiration
    - Size-based eviction
    - Dependency tracking

    PBTSO Phase: TEST, OBSERVE
    Bus Topics: test.cache.hit, test.cache.miss, test.cache.store
    """

    BUS_TOPICS = {
        "hit": "test.cache.hit",
        "miss": "test.cache.miss",
        "store": "test.cache.store",
        "evict": "test.cache.evict",
    }

    def __init__(self, bus=None, config: Optional[CacheConfig] = None):
        """
        Initialize the test cache.

        Args:
            bus: Optional bus instance
            config: Cache configuration
        """
        self.bus = bus or CacheBus()
        self.config = config or CacheConfig()
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        self._index_file = Path(self.config.cache_dir) / "cache_index.json"

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load cache index
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                for key, entry_data in data.get("entries", {}).items():
                    self._cache[key] = CacheEntry.from_dict(entry_data)

                self._stats = CacheStats(
                    total_entries=len(self._cache),
                    hits=data.get("stats", {}).get("hits", 0),
                    misses=data.get("stats", {}).get("misses", 0),
                )
            except (json.JSONDecodeError, IOError):
                pass

    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self._index_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump({
                    "entries": {k: v.to_dict() for k, v in self._cache.items()},
                    "stats": {
                        "hits": self._stats.hits,
                        "misses": self._stats.misses,
                    },
                    "updated_at": time.time(),
                }, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get(self, test_name: str, content_hash: Optional[str] = None) -> CacheLookupResult:
        """
        Look up a test result in cache.

        Args:
            test_name: Test name/path
            content_hash: Current content hash (for validation)

        Returns:
            CacheLookupResult with cache status and entry
        """
        cache_key = self._compute_cache_key(test_name, content_hash)
        entry = self._cache.get(cache_key)

        if entry is None:
            self._stats.misses += 1
            self._emit_event("miss", {
                "test_name": test_name,
                "cache_key": cache_key,
            })
            return CacheLookupResult(status=CacheStatus.MISS)

        # Check expiration
        if entry.is_expired():
            self._stats.stale += 1
            self._emit_event("miss", {
                "test_name": test_name,
                "cache_key": cache_key,
                "reason": "expired",
            })
            return CacheLookupResult(
                status=CacheStatus.STALE,
                entry=entry,
                reason="Entry expired",
            )

        # Validate content hash if provided
        if content_hash and entry.content_hash != content_hash:
            self._stats.misses += 1
            self._emit_event("miss", {
                "test_name": test_name,
                "cache_key": cache_key,
                "reason": "hash_mismatch",
            })
            return CacheLookupResult(
                status=CacheStatus.INVALID,
                entry=entry,
                reason="Content hash mismatch",
            )

        # Cache hit
        entry.hits += 1
        self._stats.hits += 1

        self._emit_event("hit", {
            "test_name": test_name,
            "cache_key": cache_key,
            "status": entry.status,
        })

        return CacheLookupResult(
            status=CacheStatus.HIT,
            entry=entry,
        )

    def store(
        self,
        test_name: str,
        status: str,
        duration_ms: float,
        content_hash: Optional[str] = None,
        output: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry:
        """
        Store a test result in cache.

        Args:
            test_name: Test name/path
            status: Test result status
            duration_ms: Test duration
            content_hash: Content hash
            output: Test output
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            Created cache entry
        """
        # Skip caching failures if configured
        if status == "failed" and not self.config.cache_failures:
            return None

        # Compute hash if not provided
        if content_hash is None:
            content_hash = self._compute_content_hash(test_name)

        cache_key = self._compute_cache_key(test_name, content_hash)

        # Create entry
        entry = CacheEntry(
            cache_key=cache_key,
            test_name=test_name,
            status=status,
            duration_ms=duration_ms,
            content_hash=content_hash,
            created_at=time.time(),
            expires_at=time.time() + self.config.ttl_s if self.config.ttl_s > 0 else None,
            output=output if self.config.include_output else None,
            error_message=error_message,
            metadata=metadata or {},
        )

        # Store in cache
        self._cache[cache_key] = entry
        self._stats.total_entries = len(self._cache)

        # Check if eviction needed
        self._maybe_evict()

        # Save index
        self._save_index()

        self._emit_event("store", {
            "test_name": test_name,
            "cache_key": cache_key,
            "status": status,
        })

        return entry

    def invalidate(self, test_name: str) -> bool:
        """Invalidate cache entry for a test."""
        # Find all entries for this test
        to_remove = []
        for key, entry in self._cache.items():
            if entry.test_name == test_name:
                to_remove.append(key)

        for key in to_remove:
            del self._cache[key]

        if to_remove:
            self._save_index()
            return True
        return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching a pattern."""
        import fnmatch

        to_remove = []
        for key, entry in self._cache.items():
            if fnmatch.fnmatch(entry.test_name, pattern):
                to_remove.append(key)

        for key in to_remove:
            del self._cache[key]

        if to_remove:
            self._save_index()

        return len(to_remove)

    def clear(self) -> int:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self._stats = CacheStats()
        self._save_index()
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.total_entries = len(self._cache)

        if self._cache:
            self._stats.oldest_entry = min(e.created_at for e in self._cache.values())
            self._stats.newest_entry = max(e.created_at for e in self._cache.values())

        # Calculate size
        self._stats.size_bytes = self._index_file.stat().st_size if self._index_file.exists() else 0

        return self._stats

    def _compute_cache_key(self, test_name: str, content_hash: Optional[str] = None) -> str:
        """Compute cache key for a test."""
        if content_hash:
            key_input = f"{test_name}:{content_hash}"
        else:
            key_input = test_name

        return hashlib.sha256(key_input.encode()).hexdigest()[:16]

    def _compute_content_hash(self, test_path: str) -> str:
        """Compute content hash for a test file."""
        path = Path(test_path)

        if not path.exists():
            return hashlib.sha256(test_path.encode()).hexdigest()

        try:
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except IOError:
            return hashlib.sha256(test_path.encode()).hexdigest()

    def _maybe_evict(self) -> None:
        """Evict entries if cache is too large."""
        if len(self._cache) <= self.config.max_entries:
            return

        # Sort by hits (ascending) then by created_at (ascending)
        entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].hits, x[1].created_at),
        )

        # Remove oldest/least used entries
        to_remove = len(self._cache) - self.config.max_entries + 100  # Remove extra for buffer

        for i in range(min(to_remove, len(entries))):
            key = entries[i][0]
            del self._cache[key]

            self._emit_event("evict", {
                "cache_key": key,
                "test_name": entries[i][1].test_name,
            })

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.cache.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "cache",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Cache."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Cache")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Lookup command
    lookup_parser = subparsers.add_parser("lookup", help="Lookup cached result")
    lookup_parser.add_argument("test", help="Test name/path")

    # Store command
    store_parser = subparsers.add_parser("store", help="Store test result")
    store_parser.add_argument("test", help="Test name/path")
    store_parser.add_argument("--status", default="passed", help="Test status")
    store_parser.add_argument("--duration", type=float, default=1000, help="Duration in ms")

    # Invalidate command
    invalidate_parser = subparsers.add_parser("invalidate", help="Invalidate cache entries")
    invalidate_parser.add_argument("pattern", help="Test pattern to invalidate")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all cache entries")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")

    # List command
    list_parser = subparsers.add_parser("list", help="List cache entries")
    list_parser.add_argument("--limit", type=int, default=20)

    # Common arguments
    parser.add_argument("--cache-dir", default=".pluribus/test-agent/cache")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = CacheConfig(cache_dir=args.cache_dir)
    cache = TestCache(config=config)

    if args.command == "lookup":
        result = cache.get(args.test)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"\nCache Lookup: {args.test}")
            print(f"Status: {result.status.value}")
            if result.entry:
                print(f"Test Status: {result.entry.status}")
                print(f"Duration: {result.entry.duration_ms:.0f}ms")
                print(f"Hits: {result.entry.hits}")
            if result.reason:
                print(f"Reason: {result.reason}")

    elif args.command == "store":
        content_hash = cache._compute_content_hash(args.test)
        entry = cache.store(
            test_name=args.test,
            status=args.status,
            duration_ms=args.duration,
            content_hash=content_hash,
        )

        if args.json:
            print(json.dumps(entry.to_dict(), indent=2))
        else:
            print(f"\nStored: {args.test}")
            print(f"Cache Key: {entry.cache_key}")
            print(f"Status: {entry.status}")

    elif args.command == "invalidate":
        count = cache.invalidate_pattern(args.pattern)
        print(f"Invalidated {count} entries matching '{args.pattern}'")

    elif args.command == "clear":
        count = cache.clear()
        print(f"Cleared {count} cache entries")

    elif args.command == "stats":
        stats = cache.get_stats()

        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("\nCache Statistics:")
            print(f"  Total Entries: {stats.total_entries}")
            print(f"  Hits: {stats.hits}")
            print(f"  Misses: {stats.misses}")
            print(f"  Hit Rate: {stats.hit_rate:.1f}%")
            print(f"  Size: {stats.size_bytes / 1024:.1f} KB")

    elif args.command == "list":
        entries = sorted(
            cache._cache.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )[:args.limit]

        if args.json:
            print(json.dumps([e.to_dict() for e in entries], indent=2))
        else:
            print(f"\nCache Entries ({len(entries)}):")
            for entry in entries:
                dt = datetime.fromtimestamp(entry.created_at)
                print(f"  [{entry.status}] {entry.test_name} "
                      f"({entry.hits} hits) - {dt.strftime('%Y-%m-%d %H:%M')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
