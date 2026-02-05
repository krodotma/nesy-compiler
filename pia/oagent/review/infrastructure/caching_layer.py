#!/usr/bin/env python3
"""
Review Caching Layer (Step 182)

Multi-tier caching system for the Review Agent. Provides memory, disk, and
distributed cache tiers with automatic promotion/demotion.

PBTSO Phase: BUILD, OBSERVE
Bus Topics: review.cache.hit, review.cache.miss, review.cache.evict

Cache Tiers:
- L1 (Memory): Fast, limited size, LRU eviction
- L2 (Disk): Larger capacity, persistent
- L3 (Distributed): Shared across agents (optional)

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import pickle
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900
DEFAULT_TTL_SECONDS = 3600  # 1 hour

T = TypeVar("T")

# ============================================================================
# Types
# ============================================================================

class CacheTier(Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_DISTRIBUTED = "l3_distributed"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"       # Least Recently Used
    LFU = "lfu"       # Least Frequently Used
    FIFO = "fifo"     # First In First Out
    TTL = "ttl"       # Time To Live based


@dataclass
class CacheEntry(Generic[T]):
    """
    A cache entry.

    Attributes:
        key: Cache key
        value: Cached value
        created_at: Creation timestamp
        expires_at: Expiration timestamp (None = no expiration)
        last_accessed: Last access timestamp
        access_count: Number of accesses
        tier: Cache tier this entry is in
        size_bytes: Approximate size in bytes
        metadata: Additional metadata
    """
    key: str
    value: T
    created_at: float
    expires_at: Optional[float] = None
    last_accessed: float = 0
    access_count: int = 0
    tier: CacheTier = CacheTier.L1_MEMORY
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.last_accessed == 0:
            self.last_accessed = self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without value)."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "tier": self.tier.value,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


@dataclass
class CacheConfig:
    """
    Cache configuration.

    Attributes:
        l1_max_size: Maximum L1 entries
        l1_max_memory_mb: Maximum L1 memory usage (MB)
        l2_enabled: Enable L2 disk cache
        l2_cache_dir: L2 cache directory
        l2_max_size_mb: Maximum L2 size (MB)
        l3_enabled: Enable L3 distributed cache
        l3_endpoint: L3 cache endpoint
        default_ttl_seconds: Default entry TTL
        eviction_policy: Cache eviction policy
        enable_compression: Compress cached values
        enable_stats: Enable statistics collection
    """
    l1_max_size: int = 1000
    l1_max_memory_mb: int = 100
    l2_enabled: bool = True
    l2_cache_dir: str = ""
    l2_max_size_mb: int = 500
    l3_enabled: bool = False
    l3_endpoint: str = ""
    default_ttl_seconds: int = DEFAULT_TTL_SECONDS
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_compression: bool = False
    enable_stats: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "eviction_policy": self.eviction_policy.value,
        }


@dataclass
class CacheStats:
    """
    Cache statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        hit_rate: Hit rate percentage
        evictions: Number of evictions
        entries: Current entry count
        memory_used_mb: Memory usage in MB
        disk_used_mb: Disk usage in MB
        avg_access_time_ms: Average access time
    """
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    evictions: int = 0
    entries: int = 0
    memory_used_mb: float = 0.0
    disk_used_mb: float = 0.0
    avg_access_time_ms: float = 0.0
    l1_entries: int = 0
    l2_entries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "hit_rate": round(self.hit_rate, 2),
            "memory_used_mb": round(self.memory_used_mb, 2),
            "disk_used_mb": round(self.disk_used_mb, 2),
            "avg_access_time_ms": round(self.avg_access_time_ms, 3),
        }

    def update_hit_rate(self) -> None:
        """Recalculate hit rate."""
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total * 100) if total > 0 else 0.0


# ============================================================================
# L1 Memory Cache
# ============================================================================

class L1MemoryCache:
    """In-memory cache with LRU eviction."""

    def __init__(self, max_size: int, max_memory_mb: int):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_used = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        if entry.is_expired:
            self.delete(key)
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        return entry

    def set(self, entry: CacheEntry) -> bool:
        """Set entry in cache."""
        # Remove existing entry if present
        if entry.key in self._cache:
            self.delete(entry.key)

        # Check size limits
        while (
            len(self._cache) >= self.max_size
            or self._memory_used + entry.size_bytes > self.max_memory_bytes
        ):
            if not self._evict_one():
                return False

        entry.tier = CacheTier.L1_MEMORY
        self._cache[entry.key] = entry
        self._memory_used += entry.size_bytes
        return True

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        entry = self._cache.pop(key, None)
        if entry:
            self._memory_used -= entry.size_bytes
            return True
        return False

    def _evict_one(self) -> bool:
        """Evict one entry (LRU)."""
        if not self._cache:
            return False
        key, entry = self._cache.popitem(last=False)
        self._memory_used -= entry.size_bytes
        return True

    def clear(self) -> int:
        """Clear all entries."""
        count = len(self._cache)
        self._cache.clear()
        self._memory_used = 0
        return count

    def size(self) -> int:
        """Get number of entries."""
        return len(self._cache)

    def memory_used(self) -> int:
        """Get memory used in bytes."""
        return self._memory_used

    def keys(self) -> List[str]:
        """Get all keys."""
        return list(self._cache.keys())


# ============================================================================
# L2 Disk Cache
# ============================================================================

class L2DiskCache:
    """Disk-based cache for larger/persistent storage."""

    def __init__(self, cache_dir: Path, max_size_mb: int):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._index: Dict[str, Dict[str, Any]] = {}
        self._disk_used = 0
        self._ensure_cache_dir()
        self._load_index()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _index_path(self) -> Path:
        """Get index file path."""
        return self.cache_dir / "cache_index.json"

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        self._index = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                self._index = {}

        # Calculate disk usage
        self._disk_used = sum(
            info.get("size_bytes", 0) for info in self._index.values()
        )

    def _save_index(self) -> None:
        """Save cache index to disk."""
        index_path = self._index_path()
        with open(index_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(self._index, f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from disk cache."""
        info = self._index.get(key)
        if info is None:
            return None

        # Check expiration
        if info.get("expires_at") and time.time() > info["expires_at"]:
            self.delete(key)
            return None

        file_path = self._key_to_path(key)
        if not file_path.exists():
            del self._index[key]
            self._save_index()
            return None

        try:
            with open(file_path, "rb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    value = pickle.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=info.get("created_at", time.time()),
                expires_at=info.get("expires_at"),
                last_accessed=time.time(),
                access_count=info.get("access_count", 0) + 1,
                tier=CacheTier.L2_DISK,
                size_bytes=info.get("size_bytes", 0),
                metadata=info.get("metadata", {}),
            )

            # Update index
            info["last_accessed"] = entry.last_accessed
            info["access_count"] = entry.access_count
            self._save_index()

            return entry
        except Exception:
            return None

    def set(self, entry: CacheEntry) -> bool:
        """Set entry in disk cache."""
        # Check size limits
        while self._disk_used + entry.size_bytes > self.max_size_bytes:
            if not self._evict_one():
                return False

        file_path = self._key_to_path(entry.key)
        try:
            with open(file_path, "wb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    pickle.dump(entry.value, f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            self._index[entry.key] = {
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "size_bytes": entry.size_bytes,
                "metadata": entry.metadata,
            }
            self._disk_used += entry.size_bytes
            self._save_index()
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        info = self._index.pop(key, None)
        if info:
            self._disk_used -= info.get("size_bytes", 0)
            file_path = self._key_to_path(key)
            if file_path.exists():
                file_path.unlink()
            self._save_index()
            return True
        return False

    def _evict_one(self) -> bool:
        """Evict one entry (oldest by last_accessed)."""
        if not self._index:
            return False

        oldest_key = min(
            self._index.keys(),
            key=lambda k: self._index[k].get("last_accessed", 0)
        )
        return self.delete(oldest_key)

    def clear(self) -> int:
        """Clear all entries."""
        count = len(self._index)
        for key in list(self._index.keys()):
            self.delete(key)
        return count

    def size(self) -> int:
        """Get number of entries."""
        return len(self._index)

    def disk_used(self) -> int:
        """Get disk used in bytes."""
        return self._disk_used


# ============================================================================
# Cache Manager
# ============================================================================

class CacheManager:
    """
    Multi-tier cache manager.

    Coordinates L1 (memory), L2 (disk), and L3 (distributed) cache tiers.

    Example:
        config = CacheConfig(l1_max_size=1000, l2_enabled=True)
        cache = CacheManager(config)

        # Set value
        await cache.set("my_key", {"data": "value"}, ttl=3600)

        # Get value
        value = await cache.get("my_key")

        # Get with fallback
        value = await cache.get_or_set("key", fallback_fn, ttl=600)
    """

    BUS_TOPICS = {
        "hit": "review.cache.hit",
        "miss": "review.cache.miss",
        "evict": "review.cache.evict",
        "stats": "review.cache.stats",
    }

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the cache manager.

        Args:
            config: Cache configuration
            bus_path: Path to event bus file
        """
        self.config = config or CacheConfig()
        self.bus_path = bus_path or self._get_bus_path()

        # Initialize tiers
        self._l1 = L1MemoryCache(
            max_size=self.config.l1_max_size,
            max_memory_mb=self.config.l1_max_memory_mb,
        )

        self._l2: Optional[L2DiskCache] = None
        if self.config.l2_enabled:
            cache_dir = Path(self.config.l2_cache_dir) if self.config.l2_cache_dir else self._get_cache_dir()
            self._l2 = L2DiskCache(
                cache_dir=cache_dir,
                max_size_mb=self.config.l2_max_size_mb,
            )

        # Statistics
        self._stats = CacheStats()
        self._access_times: List[float] = []
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_cache_dir(self) -> Path:
        """Get default cache directory."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "review" / "cache"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "cache") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "cache-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Checks L1 first, then L2, promoting to L1 on hit.

        Args:
            key: Cache key

        Returns:
            Cached value or None

        Emits:
            review.cache.hit or review.cache.miss
        """
        start_time = time.time()

        # Try L1
        entry = self._l1.get(key)
        if entry:
            self._record_hit(key, CacheTier.L1_MEMORY, start_time)
            return entry.value

        # Try L2
        if self._l2:
            entry = self._l2.get(key)
            if entry:
                # Promote to L1
                entry.tier = CacheTier.L1_MEMORY
                self._l1.set(entry)
                self._record_hit(key, CacheTier.L2_DISK, start_time)
                return entry.value

        self._record_miss(key, start_time)
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default)
            metadata: Additional metadata

        Returns:
            True if set succeeded
        """
        now = time.time()
        ttl = ttl if ttl is not None else self.config.default_ttl_seconds

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl if ttl > 0 else None,
            size_bytes=self._estimate_size(value),
            metadata=metadata or {},
        )

        # Set in L1
        success = self._l1.set(entry)

        # Also set in L2 for persistence
        if self._l2 and success:
            self._l2.set(entry)

        return success

    async def get_or_set(
        self,
        key: str,
        fallback: Callable[[], Any],
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Get value or compute and cache it.

        Args:
            key: Cache key
            fallback: Function to compute value if not cached
            ttl: Time to live in seconds
            metadata: Additional metadata

        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute and cache
        if asyncio.iscoroutinefunction(fallback):
            value = await fallback()
        else:
            value = fallback()

        await self.set(key, value, ttl=ttl, metadata=metadata)
        return value

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        deleted_l1 = self._l1.delete(key)
        deleted_l2 = self._l2.delete(key) if self._l2 else False

        if deleted_l1 or deleted_l2:
            self._stats.evictions += 1
            self._emit_event(self.BUS_TOPICS["evict"], {
                "key": key,
                "reason": "manual",
            })

        return deleted_l1 or deleted_l2

    async def clear(self, tier: Optional[CacheTier] = None) -> int:
        """
        Clear cache entries.

        Args:
            tier: Specific tier to clear (None = all)

        Returns:
            Number of entries cleared
        """
        count = 0

        if tier is None or tier == CacheTier.L1_MEMORY:
            count += self._l1.clear()

        if (tier is None or tier == CacheTier.L2_DISK) and self._l2:
            count += self._l2.clear()

        self._stats.evictions += count
        return count

    def _record_hit(self, key: str, tier: CacheTier, start_time: float) -> None:
        """Record a cache hit."""
        duration_ms = (time.time() - start_time) * 1000
        self._stats.hits += 1
        self._stats.update_hit_rate()
        self._access_times.append(duration_ms)
        if len(self._access_times) > 1000:
            self._access_times = self._access_times[-1000:]

        if self.config.enable_stats:
            self._emit_event(self.BUS_TOPICS["hit"], {
                "key": key,
                "tier": tier.value,
                "duration_ms": round(duration_ms, 3),
            })

    def _record_miss(self, key: str, start_time: float) -> None:
        """Record a cache miss."""
        duration_ms = (time.time() - start_time) * 1000
        self._stats.misses += 1
        self._stats.update_hit_rate()

        if self.config.enable_stats:
            self._emit_event(self.BUS_TOPICS["miss"], {
                "key": key,
                "duration_ms": round(duration_ms, 3),
            })

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.entries = self._l1.size() + (self._l2.size() if self._l2 else 0)
        self._stats.l1_entries = self._l1.size()
        self._stats.l2_entries = self._l2.size() if self._l2 else 0
        self._stats.memory_used_mb = self._l1.memory_used() / 1024 / 1024
        self._stats.disk_used_mb = (self._l2.disk_used() / 1024 / 1024) if self._l2 else 0
        self._stats.avg_access_time_ms = (
            sum(self._access_times) / len(self._access_times)
            if self._access_times else 0
        )
        return self._stats

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        stats = self.get_stats()
        status = {
            "agent": "cache-manager",
            "healthy": True,
            "hit_rate": stats.hit_rate,
            "entries": stats.entries,
            "memory_mb": stats.memory_used_mb,
            "disk_mb": stats.disk_used_mb,
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Caching Layer."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Review Caching Layer (Step 182)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Stats command
    subparsers.add_parser("stats", help="Show cache statistics")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--tier", choices=["l1", "l2"], help="Tier to clear")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get cached value")
    get_parser.add_argument("key", help="Cache key")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set cached value")
    set_parser.add_argument("key", help="Cache key")
    set_parser.add_argument("value", help="Value (JSON)")
    set_parser.add_argument("--ttl", type=int, help="TTL in seconds")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    cache = CacheManager()

    if args.command == "stats":
        stats = cache.get_stats()
        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("Cache Statistics")
            print(f"  Hits: {stats.hits}")
            print(f"  Misses: {stats.misses}")
            print(f"  Hit Rate: {stats.hit_rate:.1f}%")
            print(f"  Entries: {stats.entries} (L1: {stats.l1_entries}, L2: {stats.l2_entries})")
            print(f"  Memory: {stats.memory_used_mb:.2f} MB")
            print(f"  Disk: {stats.disk_used_mb:.2f} MB")

    elif args.command == "clear":
        tier = {
            "l1": CacheTier.L1_MEMORY,
            "l2": CacheTier.L2_DISK,
        }.get(args.tier) if args.tier else None
        count = asyncio.run(cache.clear(tier))
        print(f"Cleared {count} entries")

    elif args.command == "get":
        value = asyncio.run(cache.get(args.key))
        if value is not None:
            print(json.dumps(value, indent=2) if args.json else value)
        else:
            print("Not found")
            return 1

    elif args.command == "set":
        value = json.loads(args.value)
        success = asyncio.run(cache.set(args.key, value, ttl=args.ttl))
        print("OK" if success else "Failed")
        return 0 if success else 1

    else:
        # Default: show stats
        stats = cache.get_stats()
        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print(f"Cache: {stats.entries} entries, {stats.hit_rate:.1f}% hit rate")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
