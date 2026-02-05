#!/usr/bin/env python3
"""
Monitor Caching Layer - Step 282

Multi-tier metrics caching for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.cache.hit (emitted)
- monitor.cache.miss (emitted)
- monitor.cache.evict (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar


class CacheTier(Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"       # In-memory hot cache
    L2_LOCAL = "l2_local"         # Local disk cache
    L3_DISTRIBUTED = "l3_distributed"  # Distributed cache (e.g., Redis)


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"      # Least Recently Used
    LFU = "lfu"      # Least Frequently Used
    TTL = "ttl"      # Time To Live
    FIFO = "fifo"    # First In First Out


@dataclass
class CacheEntry:
    """A cache entry.

    Attributes:
        key: Cache key
        value: Cached value
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        access_count: Number of accesses
        last_accessed: Last access timestamp
        size_bytes: Estimated size in bytes
        tier: Cache tier
    """
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    tier: CacheTier = CacheTier.L1_MEMORY

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Cache statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of evictions
        entries: Number of entries
        size_bytes: Total size in bytes
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0
    size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": self.entries,
            "size_bytes": self.size_bytes,
            "hit_rate": self.hit_rate,
        }


T = TypeVar("T")


class CacheTierImpl(Generic[T]):
    """
    Base cache tier implementation.

    Provides:
    - Key-value storage
    - TTL support
    - Eviction policies
    - Statistics tracking
    """

    def __init__(
        self,
        tier: CacheTier,
        max_entries: int = 10000,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
        default_ttl_s: int = 300,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ):
        """Initialize cache tier.

        Args:
            tier: Cache tier level
            max_entries: Maximum number of entries
            max_size_bytes: Maximum size in bytes
            default_ttl_s: Default TTL in seconds
            eviction_policy: Eviction policy
        """
        self.tier = tier
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self.default_ttl_s = default_ttl_s
        self.eviction_policy = eviction_policy

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                self._evict(key)
                self._stats.misses += 1
                return None

            entry.touch()
            self._stats.hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: T,
        ttl_s: Optional[int] = None,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_s: Time to live in seconds

        Returns:
            True if set successfully
        """
        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        expires_at = time.time() + ttl if ttl > 0 else None

        # Estimate size
        try:
            size = len(json.dumps(value, default=str))
        except Exception:
            size = 100  # Default estimate

        with self._lock:
            # Check if we need to evict
            self._ensure_capacity(size)

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size,
                tier=self.tier,
            )

            # Update existing entry size
            if key in self._cache:
                self._stats.size_bytes -= self._cache[key].size_bytes

            self._cache[key] = entry
            self._stats.entries = len(self._cache)
            self._stats.size_bytes += size

        return True

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._cache:
                self._evict(key)
                return True
            return False

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.entries = 0
            self._stats.size_bytes = 0
            return count

    def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Cache key

        Returns:
            True if exists and not expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._evict(key)
                return False
            return True

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                entries=self._stats.entries,
                size_bytes=self._stats.size_bytes,
            )

    def prune_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries pruned
        """
        with self._lock:
            expired = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired:
                self._evict(key)
            return len(expired)

    def _ensure_capacity(self, new_size: int) -> None:
        """Ensure capacity for new entry."""
        # Evict by entry count
        while len(self._cache) >= self.max_entries:
            self._evict_one()

        # Evict by size
        while self._stats.size_bytes + new_size > self.max_size_bytes and self._cache:
            self._evict_one()

    def _evict_one(self) -> Optional[str]:
        """Evict one entry based on policy."""
        if not self._cache:
            return None

        key_to_evict: Optional[str] = None

        if self.eviction_policy == EvictionPolicy.LRU:
            # Least recently used
            key_to_evict = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed,
            )
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Least frequently used
            key_to_evict = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count,
            )
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # First in first out
            key_to_evict = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at,
            )
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Expired first, then oldest
            expired = [
                k for k, e in self._cache.items()
                if e.expires_at is not None
            ]
            if expired:
                key_to_evict = min(
                    expired,
                    key=lambda k: self._cache[k].expires_at or 0,
                )
            else:
                key_to_evict = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at,
                )

        if key_to_evict:
            self._evict(key_to_evict)

        return key_to_evict

    def _evict(self, key: str) -> None:
        """Evict a specific key."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entries = len(self._cache)
            self._stats.evictions += 1


class MonitorCachingLayer:
    """
    Multi-tier metrics caching for the Monitor Agent.

    Provides:
    - L1 memory cache (hot data)
    - L2 local disk cache (warm data)
    - L3 distributed cache (cold data)
    - Cache coordination
    - Statistics and monitoring

    Example:
        cache = MonitorCachingLayer()

        # Set with automatic tier selection
        await cache.set("metric:cpu", {"value": 75.0})

        # Get with tier fallback
        value = await cache.get("metric:cpu")

        # Get stats
        stats = cache.get_all_stats()
    """

    BUS_TOPICS = {
        "hit": "monitor.cache.hit",
        "miss": "monitor.cache.miss",
        "evict": "monitor.cache.evict",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        l1_max_entries: int = 10000,
        l1_max_size_mb: int = 100,
        l1_ttl_s: int = 60,
        l2_max_entries: int = 100000,
        l2_max_size_mb: int = 1000,
        l2_ttl_s: int = 3600,
        cache_dir: Optional[str] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize caching layer.

        Args:
            l1_max_entries: L1 max entries
            l1_max_size_mb: L1 max size in MB
            l1_ttl_s: L1 default TTL
            l2_max_entries: L2 max entries
            l2_max_size_mb: L2 max size in MB
            l2_ttl_s: L2 default TTL
            cache_dir: Directory for L2 cache
            bus_dir: Bus directory
        """
        # L1 Memory Cache
        self._l1 = CacheTierImpl[Any](
            tier=CacheTier.L1_MEMORY,
            max_entries=l1_max_entries,
            max_size_bytes=l1_max_size_mb * 1024 * 1024,
            default_ttl_s=l1_ttl_s,
            eviction_policy=EvictionPolicy.LRU,
        )

        # L2 Local Cache (simulated in memory for simplicity)
        self._l2 = CacheTierImpl[Any](
            tier=CacheTier.L2_LOCAL,
            max_entries=l2_max_entries,
            max_size_bytes=l2_max_size_mb * 1024 * 1024,
            default_ttl_s=l2_ttl_s,
            eviction_policy=EvictionPolicy.LRU,
        )

        self._last_heartbeat = time.time()
        self._emit_events = True

        # Cache directory
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._cache_dir = cache_dir or os.path.join(pluribus_root, ".pluribus", "cache")
        Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

        # Bus path
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    async def get(
        self,
        key: str,
        default: Optional[Any] = None,
    ) -> Optional[Any]:
        """Get value from cache with tier fallback.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        # Try L1 first
        value = self._l1.get(key)
        if value is not None:
            if self._emit_events:
                self._emit_bus_event(
                    self.BUS_TOPICS["hit"],
                    {"key": key, "tier": CacheTier.L1_MEMORY.value},
                )
            return value

        # Try L2
        value = self._l2.get(key)
        if value is not None:
            # Promote to L1
            self._l1.set(key, value)
            if self._emit_events:
                self._emit_bus_event(
                    self.BUS_TOPICS["hit"],
                    {"key": key, "tier": CacheTier.L2_LOCAL.value},
                )
            return value

        # Cache miss
        if self._emit_events:
            self._emit_bus_event(
                self.BUS_TOPICS["miss"],
                {"key": key},
            )

        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl_s: Optional[int] = None,
        tier: Optional[CacheTier] = None,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_s: Time to live in seconds
            tier: Specific tier (default: both L1 and L2)

        Returns:
            True if set successfully
        """
        if tier == CacheTier.L1_MEMORY:
            return self._l1.set(key, value, ttl_s)
        elif tier == CacheTier.L2_LOCAL:
            return self._l2.set(key, value, ttl_s)
        else:
            # Set in both tiers
            self._l1.set(key, value, ttl_s)
            self._l2.set(key, value, ttl_s)
            return True

    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers.

        Args:
            key: Cache key

        Returns:
            True if deleted from any tier
        """
        deleted_l1 = self._l1.delete(key)
        deleted_l2 = self._l2.delete(key)
        return deleted_l1 or deleted_l2

    async def exists(self, key: str) -> bool:
        """Check if key exists in any tier.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        return self._l1.exists(key) or self._l2.exists(key)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl_s: Optional[int] = None,
    ) -> Any:
        """Get from cache or compute and set.

        Args:
            key: Cache key
            factory: Function to compute value if missing
            ttl_s: Time to live in seconds

        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl_s)
        return value

    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def mset(
        self,
        items: Dict[str, Any],
        ttl_s: Optional[int] = None,
    ) -> int:
        """Set multiple values.

        Args:
            items: Dictionary of key-value pairs
            ttl_s: Time to live in seconds

        Returns:
            Number of items set
        """
        count = 0
        for key, value in items.items():
            if await self.set(key, value, ttl_s):
                count += 1
        return count

    def clear_tier(self, tier: CacheTier) -> int:
        """Clear a specific cache tier.

        Args:
            tier: Cache tier

        Returns:
            Number of entries cleared
        """
        if tier == CacheTier.L1_MEMORY:
            return self._l1.clear()
        elif tier == CacheTier.L2_LOCAL:
            return self._l2.clear()
        return 0

    def clear_all(self) -> int:
        """Clear all cache tiers.

        Returns:
            Number of entries cleared
        """
        count = self._l1.clear()
        count += self._l2.clear()
        return count

    def prune_expired(self) -> int:
        """Prune expired entries from all tiers.

        Returns:
            Number of entries pruned
        """
        count = self._l1.prune_expired()
        count += self._l2.prune_expired()
        return count

    def get_stats(self, tier: CacheTier) -> CacheStats:
        """Get statistics for a specific tier.

        Args:
            tier: Cache tier

        Returns:
            Cache statistics
        """
        if tier == CacheTier.L1_MEMORY:
            return self._l1.get_stats()
        elif tier == CacheTier.L2_LOCAL:
            return self._l2.get_stats()
        return CacheStats()

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all tiers.

        Returns:
            Combined statistics
        """
        l1_stats = self._l1.get_stats()
        l2_stats = self._l2.get_stats()

        return {
            "l1_memory": l1_stats.to_dict(),
            "l2_local": l2_stats.to_dict(),
            "combined": {
                "hits": l1_stats.hits + l2_stats.hits,
                "misses": l1_stats.misses + l2_stats.misses,
                "evictions": l1_stats.evictions + l2_stats.evictions,
                "entries": l1_stats.entries + l2_stats.entries,
                "size_bytes": l1_stats.size_bytes + l2_stats.size_bytes,
            },
        }

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Create a cache key from parts.

        Args:
            *parts: Key parts

        Returns:
            Cache key
        """
        key_str = ":".join(str(p) for p in parts)
        return key_str

    @staticmethod
    def hash_key(key: str) -> str:
        """Create a hashed cache key.

        Args:
            key: Original key

        Returns:
            Hashed key
        """
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        stats = self.get_all_stats()

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_cache",
                "status": "healthy",
                "stats": stats["combined"],
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-cache",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_cache: Optional[MonitorCachingLayer] = None


def get_cache() -> MonitorCachingLayer:
    """Get or create the caching layer singleton.

    Returns:
        MonitorCachingLayer instance
    """
    global _cache
    if _cache is None:
        _cache = MonitorCachingLayer()
    return _cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Caching Layer (Step 282)")
    parser.add_argument("--get", metavar="KEY", help="Get a value")
    parser.add_argument("--set", metavar="KEY=VALUE", help="Set a value")
    parser.add_argument("--delete", metavar="KEY", help="Delete a value")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--prune", action="store_true", help="Prune expired entries")
    parser.add_argument("--clear", action="store_true", help="Clear all caches")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    cache = get_cache()

    if args.get:
        async def run_get():
            return await cache.get(args.get)
        value = asyncio.run(run_get())
        if args.json:
            print(json.dumps({"key": args.get, "value": value}))
        else:
            print(f"{args.get}: {value}")

    if args.set:
        key, value = args.set.split("=", 1)
        async def run_set():
            return await cache.set(key, value)
        asyncio.run(run_set())
        if args.json:
            print(json.dumps({"key": key, "value": value, "set": True}))
        else:
            print(f"Set {key}={value}")

    if args.delete:
        async def run_delete():
            return await cache.delete(args.delete)
        deleted = asyncio.run(run_delete())
        if args.json:
            print(json.dumps({"key": args.delete, "deleted": deleted}))
        else:
            print(f"Deleted {args.delete}: {deleted}")

    if args.stats:
        stats = cache.get_all_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Cache Statistics:")
            for tier, tier_stats in stats.items():
                print(f"  {tier}:")
                for k, v in tier_stats.items():
                    print(f"    {k}: {v}")

    if args.prune:
        count = cache.prune_expired()
        if args.json:
            print(json.dumps({"pruned": count}))
        else:
            print(f"Pruned {count} entries")

    if args.clear:
        count = cache.clear_all()
        if args.json:
            print(json.dumps({"cleared": count}))
        else:
            print(f"Cleared {count} entries")
