#!/usr/bin/env python3
"""
caching_layer.py - Multi-Tier Caching Layer (Step 32)

Multi-tier caching with memory, disk, and distributed layers.
Provides automatic tier promotion/demotion and cache warming.

PBTSO Phase: ITERATE

Bus Topics:
- a2a.research.cache.tier_hit
- a2a.research.cache.tier_miss
- a2a.research.cache.promote
- a2a.research.cache.demote
- research.cache.warm

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import socket
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class CacheTier(Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"     # Fastest, smallest (hot data)
    L2_DISK = "l2_disk"         # Medium speed, larger
    L3_DISTRIBUTED = "l3_dist"  # Slowest, largest (Redis/FalkorDB)


@dataclass
class TierConfig:
    """Configuration for a single tier."""

    tier: CacheTier
    enabled: bool = True
    max_items: int = 1000
    max_size_mb: int = 100
    ttl_seconds: int = 3600
    promote_on_hit: bool = True
    demote_on_evict: bool = True


@dataclass
class TieredCacheConfig:
    """Configuration for tiered cache."""

    l1_config: Optional[TierConfig] = None
    l2_config: Optional[TierConfig] = None
    l3_config: Optional[TierConfig] = None
    namespace: str = "research"
    enable_compression: bool = False
    enable_stats: bool = True
    persist_path: Optional[str] = None
    redis_url: Optional[str] = None
    falkordb_host: str = "localhost"
    falkordb_port: int = 6380
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.l1_config is None:
            self.l1_config = TierConfig(
                tier=CacheTier.L1_MEMORY,
                max_items=10000,
                max_size_mb=100,
                ttl_seconds=300,  # 5 min
            )
        if self.l2_config is None:
            self.l2_config = TierConfig(
                tier=CacheTier.L2_DISK,
                max_items=100000,
                max_size_mb=1000,
                ttl_seconds=3600,  # 1 hour
            )
        if self.l3_config is None:
            self.l3_config = TierConfig(
                tier=CacheTier.L3_DISTRIBUTED,
                enabled=False,  # Disabled by default
                max_items=1000000,
                ttl_seconds=86400,  # 24 hours
            )
        if self.persist_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.persist_path = f"{pluribus_root}/.pluribus/research/cache/tiered"
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


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
    tier: CacheTier
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access stats."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics per tier."""

    tier: CacheTier
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0
    item_count: int = 0
    size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier": self.tier.value,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "item_count": self.item_count,
            "size_bytes": self.size_bytes,
        }


# ============================================================================
# Tier Implementations
# ============================================================================


class CacheTierBackend(ABC):
    """Abstract base class for cache tier backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from tier."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in tier."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from tier."""
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get tier statistics."""
        pass


class MemoryTierBackend(CacheTierBackend):
    """L1 Memory cache backend using LRU OrderedDict."""

    def __init__(self, config: TierConfig):
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(tier=CacheTier.L1_MEMORY)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    del self._cache[key]
                    self._stats.misses += 1
                    return None

                entry.touch()
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return entry.value

            self._stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int) -> bool:
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.config.max_items:
                self._evict_one()

            now = time.time()
            try:
                size = len(pickle.dumps(value))
            except Exception:
                size = 0

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + ttl,
                tier=CacheTier.L1_MEMORY,
                last_accessed=now,
                size_bytes=size,
            )
            self._cache[key] = entry
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def has(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                return not entry.is_expired()
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            self._stats.item_count = len(self._cache)
            self._stats.size_bytes = sum(e.size_bytes for e in self._cache.values())
            return self._stats

    def _evict_one(self) -> Optional[CacheEntry]:
        """Evict the oldest entry."""
        if not self._cache:
            return None
        key, entry = self._cache.popitem(last=False)
        self._stats.evictions += 1
        return entry

    def get_for_demotion(self) -> Optional[CacheEntry]:
        """Get entry for demotion to lower tier."""
        with self._lock:
            if self._cache:
                entry = self._evict_one()
                self._stats.demotions += 1
                return entry
            return None


class DiskTierBackend(CacheTierBackend):
    """L2 Disk cache backend using SQLite."""

    def __init__(self, config: TierConfig, persist_path: str):
        self.config = config
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self._db_path = self.persist_path / "l2_cache.db"
        self._db: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._stats = CacheStats(tier=CacheTier.L2_DISK)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache_l2 (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at REAL,
                expires_at REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                size_bytes INTEGER,
                tags TEXT
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_l2_expires ON cache_l2(expires_at)")
        self._db.commit()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if not self._db:
                return None

            cursor = self._db.execute(
                "SELECT * FROM cache_l2 WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if row:
                if row["expires_at"] < time.time():
                    self._db.execute("DELETE FROM cache_l2 WHERE key = ?", (key,))
                    self._db.commit()
                    self._stats.misses += 1
                    return None

                # Update access stats
                now = time.time()
                self._db.execute(
                    "UPDATE cache_l2 SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                    (now, key)
                )
                self._db.commit()

                self._stats.hits += 1
                return pickle.loads(row["value"])

            self._stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int) -> bool:
        with self._lock:
            if not self._db:
                return False

            now = time.time()
            try:
                value_bytes = pickle.dumps(value)
                size = len(value_bytes)
            except Exception:
                return False

            self._db.execute("""
                INSERT OR REPLACE INTO cache_l2
                (key, value, created_at, expires_at, access_count, last_accessed, size_bytes, tags)
                VALUES (?, ?, ?, ?, 0, ?, ?, '[]')
            """, (key, value_bytes, now, now + ttl, now, size))
            self._db.commit()
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if not self._db:
                return False
            cursor = self._db.execute("DELETE FROM cache_l2 WHERE key = ?", (key,))
            self._db.commit()
            return cursor.rowcount > 0

    def has(self, key: str) -> bool:
        with self._lock:
            if not self._db:
                return False
            cursor = self._db.execute(
                "SELECT 1 FROM cache_l2 WHERE key = ? AND expires_at > ?",
                (key, time.time())
            )
            return cursor.fetchone() is not None

    def clear(self) -> None:
        with self._lock:
            if self._db:
                self._db.execute("DELETE FROM cache_l2")
                self._db.commit()

    def stats(self) -> CacheStats:
        with self._lock:
            if self._db:
                cursor = self._db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache_l2"
                )
                row = cursor.fetchone()
                self._stats.item_count = row[0]
                self._stats.size_bytes = row[1]
            return self._stats

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            if not self._db:
                return 0
            cursor = self._db.execute(
                "DELETE FROM cache_l2 WHERE expires_at < ?",
                (time.time(),)
            )
            self._db.commit()
            return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None


class DistributedTierBackend(CacheTierBackend):
    """L3 Distributed cache backend using Redis or FalkorDB."""

    def __init__(
        self,
        config: TierConfig,
        redis_url: Optional[str] = None,
        falkordb_host: str = "localhost",
        falkordb_port: int = 6380,
    ):
        self.config = config
        self.redis_url = redis_url
        self.falkordb_host = falkordb_host
        self.falkordb_port = falkordb_port

        self._client = None
        self._stats = CacheStats(tier=CacheTier.L3_DISTRIBUTED)
        self._connected = False

        if config.enabled:
            self._connect()

    def _connect(self) -> None:
        """Connect to distributed cache."""
        try:
            import redis
            if self.redis_url:
                self._client = redis.from_url(self.redis_url)
            else:
                self._client = redis.Redis(
                    host=self.falkordb_host,
                    port=self.falkordb_port,
                    decode_responses=False,
                )
            # Test connection
            self._client.ping()
            self._connected = True
        except Exception:
            self._connected = False

    def get(self, key: str) -> Optional[Any]:
        if not self._connected or not self._client:
            self._stats.misses += 1
            return None

        try:
            data = self._client.get(f"cache:l3:{key}")
            if data:
                self._stats.hits += 1
                return pickle.loads(data)
            self._stats.misses += 1
            return None
        except Exception:
            self._stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int) -> bool:
        if not self._connected or not self._client:
            return False

        try:
            data = pickle.dumps(value)
            self._client.setex(f"cache:l3:{key}", ttl, data)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        if not self._connected or not self._client:
            return False

        try:
            return self._client.delete(f"cache:l3:{key}") > 0
        except Exception:
            return False

    def has(self, key: str) -> bool:
        if not self._connected or not self._client:
            return False

        try:
            return self._client.exists(f"cache:l3:{key}") > 0
        except Exception:
            return False

    def clear(self) -> None:
        if not self._connected or not self._client:
            return

        try:
            keys = self._client.keys("cache:l3:*")
            if keys:
                self._client.delete(*keys)
        except Exception:
            pass

    def stats(self) -> CacheStats:
        if self._connected and self._client:
            try:
                info = self._client.info("keyspace")
                if "db0" in info:
                    self._stats.item_count = info["db0"].get("keys", 0)
            except Exception:
                pass
        return self._stats

    def close(self) -> None:
        """Close connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False


# ============================================================================
# Tiered Cache
# ============================================================================


class TieredCache:
    """
    Multi-tier cache with automatic promotion and demotion.

    Features:
    - L1: Fast in-memory LRU cache
    - L2: Persistent disk cache (SQLite)
    - L3: Distributed cache (Redis/FalkorDB)
    - Automatic promotion on hit
    - Demotion to lower tiers on eviction
    - Cache warming

    PBTSO Phase: ITERATE

    Example:
        cache = TieredCache()

        # Set with tier preference
        cache.set("key", value, ttl=300)

        # Get (checks all tiers)
        value = cache.get("key")

        # Warm cache from lower tiers
        cache.warm_cache(["key1", "key2", "key3"])
    """

    def __init__(
        self,
        config: Optional[TieredCacheConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the tiered cache.

        Args:
            config: Tiered cache configuration
            bus: AgentBus for event emission
        """
        self.config = config or TieredCacheConfig()
        self.bus = bus or AgentBus()

        # Initialize tier backends
        self._l1: Optional[MemoryTierBackend] = None
        self._l2: Optional[DiskTierBackend] = None
        self._l3: Optional[DistributedTierBackend] = None

        if self.config.l1_config.enabled:
            self._l1 = MemoryTierBackend(self.config.l1_config)

        if self.config.l2_config.enabled:
            self._l2 = DiskTierBackend(
                self.config.l2_config,
                self.config.persist_path,
            )

        if self.config.l3_config.enabled:
            self._l3 = DistributedTierBackend(
                self.config.l3_config,
                redis_url=self.config.redis_url,
                falkordb_host=self.config.falkordb_host,
                falkordb_port=self.config.falkordb_port,
            )

    def get(self, key: str, promote: bool = True) -> Optional[Any]:
        """
        Get value from cache, checking all tiers.

        Args:
            key: Cache key
            promote: Whether to promote to higher tiers on hit

        Returns:
            Cached value or None
        """
        full_key = self._make_key(key)

        # Check L1
        if self._l1:
            value = self._l1.get(full_key)
            if value is not None:
                self._emit_event("a2a.research.cache.tier_hit", {
                    "key": key,
                    "tier": CacheTier.L1_MEMORY.value,
                })
                return value

        # Check L2
        if self._l2:
            value = self._l2.get(full_key)
            if value is not None:
                self._emit_event("a2a.research.cache.tier_hit", {
                    "key": key,
                    "tier": CacheTier.L2_DISK.value,
                })
                # Promote to L1
                if promote and self._l1 and self.config.l1_config.promote_on_hit:
                    self._l1.set(full_key, value, self.config.l1_config.ttl_seconds)
                    self._emit_event("a2a.research.cache.promote", {
                        "key": key,
                        "from_tier": CacheTier.L2_DISK.value,
                        "to_tier": CacheTier.L1_MEMORY.value,
                    })
                return value

        # Check L3
        if self._l3:
            value = self._l3.get(full_key)
            if value is not None:
                self._emit_event("a2a.research.cache.tier_hit", {
                    "key": key,
                    "tier": CacheTier.L3_DISTRIBUTED.value,
                })
                # Promote to L1/L2
                if promote:
                    if self._l1 and self.config.l1_config.promote_on_hit:
                        self._l1.set(full_key, value, self.config.l1_config.ttl_seconds)
                    if self._l2 and self.config.l2_config.promote_on_hit:
                        self._l2.set(full_key, value, self.config.l2_config.ttl_seconds)
                return value

        self._emit_event("a2a.research.cache.tier_miss", {
            "key": key,
        })
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: Optional[CacheTier] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            tier: Specific tier to use (None = all enabled tiers)

        Returns:
            True if set in at least one tier
        """
        full_key = self._make_key(key)
        success = False

        # Set in L1
        if self._l1 and (tier is None or tier == CacheTier.L1_MEMORY):
            l1_ttl = ttl or self.config.l1_config.ttl_seconds
            success = self._l1.set(full_key, value, l1_ttl) or success

        # Set in L2
        if self._l2 and (tier is None or tier == CacheTier.L2_DISK):
            l2_ttl = ttl or self.config.l2_config.ttl_seconds
            success = self._l2.set(full_key, value, l2_ttl) or success

        # Set in L3
        if self._l3 and (tier is None or tier == CacheTier.L3_DISTRIBUTED):
            l3_ttl = ttl or self.config.l3_config.ttl_seconds
            success = self._l3.set(full_key, value, l3_ttl) or success

        return success

    def delete(self, key: str) -> bool:
        """Delete key from all tiers."""
        full_key = self._make_key(key)
        deleted = False

        if self._l1:
            deleted = self._l1.delete(full_key) or deleted
        if self._l2:
            deleted = self._l2.delete(full_key) or deleted
        if self._l3:
            deleted = self._l3.delete(full_key) or deleted

        return deleted

    def has(self, key: str) -> bool:
        """Check if key exists in any tier."""
        full_key = self._make_key(key)

        if self._l1 and self._l1.has(full_key):
            return True
        if self._l2 and self._l2.has(full_key):
            return True
        if self._l3 and self._l3.has(full_key):
            return True

        return False

    def clear(self, tier: Optional[CacheTier] = None) -> None:
        """Clear cache entries."""
        if tier is None or tier == CacheTier.L1_MEMORY:
            if self._l1:
                self._l1.clear()
        if tier is None or tier == CacheTier.L2_DISK:
            if self._l2:
                self._l2.clear()
        if tier is None or tier == CacheTier.L3_DISTRIBUTED:
            if self._l3:
                self._l3.clear()

    def warm_cache(self, keys: List[str], from_tier: CacheTier = CacheTier.L2_DISK) -> int:
        """
        Warm L1 cache from lower tiers.

        Args:
            keys: Keys to warm
            from_tier: Source tier

        Returns:
            Number of keys warmed
        """
        count = 0

        for key in keys:
            value = self.get(key, promote=True)
            if value is not None:
                count += 1

        self._emit_event("research.cache.warm", {
            "keys_requested": len(keys),
            "keys_warmed": count,
            "from_tier": from_tier.value,
        })

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all tiers."""
        stats = {
            "namespace": self.config.namespace,
            "tiers": {},
        }

        if self._l1:
            s = self._l1.stats()
            stats["tiers"][CacheTier.L1_MEMORY.value] = s.to_dict()

        if self._l2:
            s = self._l2.stats()
            stats["tiers"][CacheTier.L2_DISK.value] = s.to_dict()

        if self._l3:
            s = self._l3.stats()
            stats["tiers"][CacheTier.L3_DISTRIBUTED.value] = s.to_dict()

        # Aggregate stats
        total_hits = sum(t.get("hits", 0) for t in stats["tiers"].values())
        total_misses = sum(t.get("misses", 0) for t in stats["tiers"].values())
        stats["aggregate"] = {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
        }

        return stats

    def close(self) -> None:
        """Close all backends."""
        if self._l2:
            self._l2.close()
        if self._l3:
            self._l3.close()

    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.config.namespace}:{key}"

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "cache",
            "level": "info",
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Caching Layer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Caching Layer (Step 32)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--tier", choices=["l1", "l2", "l3"], help="Specific tier")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get cached value")
    get_parser.add_argument("key", help="Cache key")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set cached value")
    set_parser.add_argument("key", help="Cache key")
    set_parser.add_argument("value", help="Value to cache")
    set_parser.add_argument("--ttl", type=int, default=300, help="TTL in seconds")

    # Warm command
    warm_parser = subparsers.add_parser("warm", help="Warm cache from lower tiers")
    warm_parser.add_argument("keys", nargs="+", help="Keys to warm")

    args = parser.parse_args()

    cache = TieredCache()

    if args.command == "stats":
        stats = cache.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Tiered Cache Statistics:")
            for tier_name, tier_stats in stats.get("tiers", {}).items():
                print(f"\n  {tier_name}:")
                print(f"    Hits: {tier_stats['hits']}")
                print(f"    Misses: {tier_stats['misses']}")
                print(f"    Hit Rate: {tier_stats['hit_rate']:.1%}")
                print(f"    Items: {tier_stats['item_count']}")

    elif args.command == "clear":
        tier = None
        if args.tier == "l1":
            tier = CacheTier.L1_MEMORY
        elif args.tier == "l2":
            tier = CacheTier.L2_DISK
        elif args.tier == "l3":
            tier = CacheTier.L3_DISTRIBUTED

        cache.clear(tier)
        print(f"Cache cleared" + (f" ({args.tier})" if args.tier else " (all tiers)"))

    elif args.command == "get":
        value = cache.get(args.key)
        if value is not None:
            print(f"Value: {value}")
        else:
            print("Key not found")

    elif args.command == "set":
        cache.set(args.key, args.value, ttl=args.ttl)
        print(f"Cached '{args.key}' for {args.ttl} seconds")

    elif args.command == "warm":
        count = cache.warm_cache(args.keys)
        print(f"Warmed {count}/{len(args.keys)} keys")

    cache.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
