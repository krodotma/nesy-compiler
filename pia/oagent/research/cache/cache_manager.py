#!/usr/bin/env python3
"""
cache_manager.py - Cache Manager (Step 19)

Research result caching with TTL, eviction policies, and persistence.
Optimizes repeated queries and reduces redundant computation.

PBTSO Phase: ITERATE

Bus Topics:
- a2a.research.cache.hit
- a2a.research.cache.miss
- research.cache.evict
- research.cache.persist

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"       # Least Recently Used
    LFU = "lfu"       # Least Frequently Used
    TTL = "ttl"       # Time-To-Live based
    FIFO = "fifo"     # First In First Out


@dataclass
class CacheConfig:
    """Configuration for cache manager."""

    max_memory_items: int = 10000
    max_disk_mb: int = 500
    default_ttl_seconds: int = 3600  # 1 hour
    persist_path: Optional[str] = None
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_persistence: bool = True
    compression: bool = False
    namespace: str = "research"

    def __post_init__(self):
        if self.persist_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.persist_path = f"{pluribus_root}/.pluribus/research/cache"


# ============================================================================
# Data Models
# ============================================================================


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached entry with metadata."""

    key: str
    value: T
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.last_accessed == 0:
            self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes value)."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "size_bytes": self.size_bytes,
            "tags": self.tags,
        }


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_items: int = 0
    disk_items: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "memory_items": self.memory_items,
            "disk_items": self.disk_items,
            "memory_bytes": self.memory_bytes,
            "disk_bytes": self.disk_bytes,
        }


# ============================================================================
# Cache Manager
# ============================================================================


class CacheManager:
    """
    Research result cache with multiple eviction policies and persistence.

    Features:
    - In-memory LRU/LFU caching
    - Disk persistence with SQLite
    - TTL-based expiration
    - Tag-based invalidation
    - Thread-safe operations

    PBTSO Phase: ITERATE

    Example:
        cache = CacheManager()

        # Cache a result
        cache.set("search:query:abc", results, ttl=300)

        # Retrieve if cached
        if cache.has("search:query:abc"):
            results = cache.get("search:query:abc")

        # Tag-based invalidation
        cache.invalidate_by_tag("file:src/main.py")
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the cache manager.

        Args:
            config: Cache configuration
            bus: AgentBus for event emission
        """
        self.config = config or CacheConfig()
        self.bus = bus or AgentBus()

        # In-memory cache
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._stats = CacheStats()

        # Initialize persistence
        self._db: Optional[sqlite3.Connection] = None
        if self.config.enable_persistence:
            self._init_persistence()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            # Check memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]

                if entry.is_expired():
                    self._remove_entry(key)
                    self._stats.misses += 1
                    self._emit_miss(key)
                    return default

                # Touch entry (update access time)
                entry.touch()

                # Move to end for LRU
                if self.config.eviction_policy == EvictionPolicy.LRU:
                    self._memory_cache.move_to_end(key)

                self._stats.hits += 1
                self._emit_hit(key)
                return entry.value

            # Check disk cache
            if self.config.enable_persistence:
                entry = self._load_from_disk(key)
                if entry and not entry.is_expired():
                    # Promote to memory
                    self._add_to_memory(entry)
                    self._stats.hits += 1
                    self._emit_hit(key)
                    return entry.value

            self._stats.misses += 1
            self._emit_miss(key)
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
            tags: Tags for group invalidation
            metadata: Additional metadata

        Returns:
            True if successfully cached
        """
        ttl = ttl if ttl is not None else self.config.default_ttl_seconds
        now = time.time()

        # Estimate size
        try:
            size = len(pickle.dumps(value))
        except Exception:
            size = 0

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl,
            size_bytes=size,
            tags=tags or [],
            metadata=metadata or {},
        )

        with self._lock:
            self._add_to_memory(entry)

            # Also persist to disk
            if self.config.enable_persistence:
                self._save_to_disk(entry)

        return True

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if not entry.is_expired():
                    return True

            if self.config.enable_persistence:
                entry = self._load_from_disk(key)
                if entry and not entry.is_expired():
                    return True

        return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was found and deleted
        """
        with self._lock:
            found = self._remove_entry(key)

            if self.config.enable_persistence:
                self._delete_from_disk(key)

            return found

    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all entries with a specific tag.

        Args:
            tag: Tag to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0

        with self._lock:
            keys_to_remove = []

            for key, entry in self._memory_cache.items():
                if tag in entry.tags:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self._remove_entry(key)
                count += 1

            # Also invalidate on disk
            if self.config.enable_persistence:
                count += self._invalidate_disk_by_tag(tag)

        return count

    def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Invalidate all entries with keys starting with prefix.

        Args:
            prefix: Key prefix

        Returns:
            Number of entries invalidated
        """
        count = 0

        with self._lock:
            keys_to_remove = [
                key for key in self._memory_cache
                if key.startswith(prefix)
            ]

            for key in keys_to_remove:
                self._remove_entry(key)
                count += 1

            if self.config.enable_persistence:
                count += self._invalidate_disk_by_prefix(prefix)

        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._stats.memory_items = 0
            self._stats.memory_bytes = 0

            if self.config.enable_persistence and self._db:
                self._db.execute("DELETE FROM cache")
                self._db.commit()
                self._stats.disk_items = 0
                self._stats.disk_bytes = 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.memory_items = len(self._memory_cache)
            self._stats.memory_bytes = sum(
                e.size_bytes for e in self._memory_cache.values()
            )

            if self.config.enable_persistence and self._db:
                cursor = self._db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache"
                )
                row = cursor.fetchone()
                self._stats.disk_items = row[0]
                self._stats.disk_bytes = row[1]

        return self._stats

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> T:
        """
        Get from cache or compute and cache result.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Time-to-live in seconds
            tags: Tags for invalidation

        Returns:
            Cached or computed value
        """
        value = self.get(key)

        if value is not None:
            return value

        # Compute value
        value = compute_fn()

        # Cache result
        self.set(key, value, ttl=ttl, tags=tags)

        return value

    def make_key(self, *parts: str) -> str:
        """
        Create a cache key from parts.

        Args:
            parts: Key components

        Returns:
            Formatted cache key
        """
        return f"{self.config.namespace}:{':'.join(parts)}"

    def hash_key(self, data: Any) -> str:
        """
        Create a hash-based key from data.

        Args:
            data: Data to hash

        Returns:
            Hash string
        """
        if isinstance(data, str):
            content = data.encode()
        else:
            content = json.dumps(data, sort_keys=True).encode()

        return hashlib.sha256(content).hexdigest()[:16]

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        count = 0

        with self._lock:
            # Memory cache
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                self._remove_entry(key)
                count += 1

            # Disk cache
            if self.config.enable_persistence and self._db:
                cursor = self._db.execute(
                    "DELETE FROM cache WHERE expires_at < ?",
                    (time.time(),)
                )
                count += cursor.rowcount
                self._db.commit()

        return count

    def close(self) -> None:
        """Close the cache manager and persist state."""
        if self._db:
            self._db.close()
            self._db = None

    def _add_to_memory(self, entry: CacheEntry) -> None:
        """Add entry to memory cache, evicting if necessary."""
        # Evict if at capacity
        while len(self._memory_cache) >= self.config.max_memory_items:
            self._evict_one()

        self._memory_cache[entry.key] = entry

    def _remove_entry(self, key: str) -> bool:
        """Remove entry from memory cache."""
        if key in self._memory_cache:
            del self._memory_cache[key]
            return True
        return False

    def _evict_one(self) -> None:
        """Evict one entry based on policy."""
        if not self._memory_cache:
            return

        if self.config.eviction_policy == EvictionPolicy.LRU:
            # Remove oldest (first item)
            key, entry = next(iter(self._memory_cache.items()))
            del self._memory_cache[key]

        elif self.config.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            min_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].access_count
            )
            del self._memory_cache[min_key]

        elif self.config.eviction_policy == EvictionPolicy.TTL:
            # Remove entry closest to expiration
            min_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].expires_at
            )
            del self._memory_cache[min_key]

        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            # Remove first added (oldest by creation time)
            min_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].created_at
            )
            del self._memory_cache[min_key]

        self._stats.evictions += 1

        self.bus.emit({
            "topic": "research.cache.evict",
            "kind": "cache",
            "data": {"policy": self.config.eviction_policy.value}
        })

    def _init_persistence(self) -> None:
        """Initialize SQLite persistence."""
        persist_path = Path(self.config.persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)

        db_path = persist_path / "cache.db"
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at REAL,
                expires_at REAL,
                access_count INTEGER,
                last_accessed REAL,
                size_bytes INTEGER,
                tags TEXT
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_tags ON cache(tags)")
        self._db.commit()

    def _save_to_disk(self, entry: CacheEntry) -> None:
        """Save entry to disk."""
        if not self._db:
            return

        try:
            value_bytes = pickle.dumps(entry.value)
            tags_json = json.dumps(entry.tags)

            self._db.execute("""
                INSERT OR REPLACE INTO cache
                (key, value, created_at, expires_at, access_count, last_accessed, size_bytes, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.key,
                value_bytes,
                entry.created_at,
                entry.expires_at,
                entry.access_count,
                entry.last_accessed,
                entry.size_bytes,
                tags_json,
            ))
            self._db.commit()

        except Exception:
            pass

    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load entry from disk."""
        if not self._db:
            return None

        try:
            cursor = self._db.execute(
                "SELECT * FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if row:
                value = pickle.loads(row["value"])
                tags = json.loads(row["tags"]) if row["tags"] else []

                return CacheEntry(
                    key=row["key"],
                    value=value,
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                    access_count=row["access_count"],
                    last_accessed=row["last_accessed"],
                    size_bytes=row["size_bytes"],
                    tags=tags,
                )

        except Exception:
            pass

        return None

    def _delete_from_disk(self, key: str) -> None:
        """Delete entry from disk."""
        if self._db:
            self._db.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._db.commit()

    def _invalidate_disk_by_tag(self, tag: str) -> int:
        """Invalidate disk entries by tag."""
        if not self._db:
            return 0

        # SQLite JSON search for tag
        cursor = self._db.execute(
            "DELETE FROM cache WHERE tags LIKE ?",
            (f'%"{tag}"%',)
        )
        self._db.commit()
        return cursor.rowcount

    def _invalidate_disk_by_prefix(self, prefix: str) -> int:
        """Invalidate disk entries by key prefix."""
        if not self._db:
            return 0

        cursor = self._db.execute(
            "DELETE FROM cache WHERE key LIKE ?",
            (f"{prefix}%",)
        )
        self._db.commit()
        return cursor.rowcount

    def _emit_hit(self, key: str) -> None:
        """Emit cache hit event."""
        self.bus.emit({
            "topic": "a2a.research.cache.hit",
            "kind": "cache",
            "data": {"key": key}
        })

    def _emit_miss(self, key: str) -> None:
        """Emit cache miss event."""
        self.bus.emit({
            "topic": "a2a.research.cache.miss",
            "kind": "cache",
            "data": {"key": key}
        })


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Cache Manager."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cache Manager (Step 19)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all cache entries")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove expired entries")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get a cached value")
    get_parser.add_argument("key", help="Cache key")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set a cached value")
    set_parser.add_argument("key", help="Cache key")
    set_parser.add_argument("value", help="Value to cache")
    set_parser.add_argument("--ttl", type=int, default=3600, help="TTL in seconds")

    args = parser.parse_args()

    cache = CacheManager()

    if args.command == "stats":
        stats = cache.get_stats()
        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("Cache Statistics:")
            print(f"  Hit Rate: {stats.hit_rate:.1%}")
            print(f"  Hits: {stats.hits}")
            print(f"  Misses: {stats.misses}")
            print(f"  Evictions: {stats.evictions}")
            print(f"  Memory Items: {stats.memory_items}")
            print(f"  Disk Items: {stats.disk_items}")

    elif args.command == "clear":
        cache.clear()
        print("Cache cleared.")

    elif args.command == "cleanup":
        count = cache.cleanup_expired()
        print(f"Removed {count} expired entries.")

    elif args.command == "get":
        value = cache.get(args.key)
        if value is not None:
            print(f"Value: {value}")
        else:
            print("Key not found or expired.")

    elif args.command == "set":
        cache.set(args.key, args.value, ttl=args.ttl)
        print(f"Cached '{args.key}' for {args.ttl} seconds.")

    cache.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
