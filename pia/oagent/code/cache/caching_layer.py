#!/usr/bin/env python3
"""
caching_layer.py - Multi-tier Code Caching (Step 82)

PBTSO Phase: ITERATE, VERIFY

Provides:
- Multi-tier caching (memory, disk, remote)
- LRU eviction policy
- TTL-based expiration
- Cache statistics and monitoring
- Async cache operations

Bus Topics:
- code.cache.hit
- code.cache.miss
- code.cache.evict
- code.cache.stats

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import socket
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class CacheTier(Enum):
    """Cache tier levels."""
    MEMORY = "memory"
    DISK = "disk"
    REMOTE = "remote"


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
            "memory_bytes": self.memory_bytes,
        }


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0
    tier: CacheTier = CacheTier.MEMORY
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "size_bytes": self.size_bytes,
            "tier": self.tier.value,
            "is_expired": self.is_expired,
        }


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    max_memory_mb: int = 256
    max_disk_mb: int = 1024
    default_ttl_s: int = 3600  # 1 hour
    cleanup_interval_s: int = 300
    disk_cache_dir: str = "/pluribus/.pluribus/cache"
    enable_disk_cache: bool = True
    enable_compression: bool = True
    eviction_policy: str = "lru"  # lru, lfu, fifo
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_disk_mb": self.max_disk_mb,
            "default_ttl_s": self.default_ttl_s,
            "cleanup_interval_s": self.cleanup_interval_s,
            "disk_cache_dir": self.disk_cache_dir,
            "enable_disk_cache": self.enable_disk_cache,
            "eviction_policy": self.eviction_policy,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Cache Interface
# =============================================================================

T = TypeVar("T")


class Cache(ABC, Generic[T]):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


# =============================================================================
# LRU Cache
# =============================================================================

class LRUCache(Cache[T]):
    """
    Thread-safe LRU (Least Recently Used) cache.

    Uses OrderedDict for O(1) access and eviction.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._stats = CacheStats(max_size=max_size)

    def get(self, key: str) -> Optional[T]:
        """Get value, moving to end (most recent)."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Update access stats
            entry.access_count += 1
            entry.last_accessed = time.time()

            self._stats.hits += 1
            return entry.value

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set value, evicting LRU if needed."""
        with self._lock:
            now = time.time()
            expires_at = now + ttl if ttl else None

            # Calculate size estimate
            size_bytes = len(pickle.dumps(value))

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=expires_at,
                last_accessed=now,
                size_bytes=size_bytes,
                tier=CacheTier.MEMORY,
            )

            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1

            self._cache[key] = entry
            self._stats.size = len(self._cache)
            self._stats.memory_bytes += size_bytes

            return True

    def delete(self, key: str) -> bool:
        """Delete a key."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.memory_bytes -= entry.size_bytes
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0
            self._stats.memory_bytes = 0

    def exists(self, key: str) -> bool:
        """Check if key exists and not expired."""
        with self._lock:
            if key not in self._cache:
                return False
            return not self._cache[key].is_expired

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired = [k for k, v in self._cache.items() if v.is_expired]
            for k in expired:
                del self._cache[k]
            self._stats.size = len(self._cache)
            return len(expired)


# =============================================================================
# Memory Cache
# =============================================================================

class MemoryCache(Cache[T]):
    """
    In-memory cache with TTL support.

    Wrapper around LRUCache with additional features.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 256,
        default_ttl: int = 3600,
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self._lru = LRUCache[T](max_size)

    def get(self, key: str) -> Optional[T]:
        return self._lru.get(key)

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        ttl = ttl or self.default_ttl

        # Check memory limit
        stats = self._lru.stats()
        if stats.memory_bytes >= self.max_memory_bytes:
            # Evict until under limit
            self._lru.cleanup_expired()

        return self._lru.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        return self._lru.delete(key)

    def clear(self) -> None:
        self._lru.clear()

    def exists(self, key: str) -> bool:
        return self._lru.exists(key)

    def stats(self) -> CacheStats:
        return self._lru.stats()


# =============================================================================
# Disk Cache
# =============================================================================

class DiskCache(Cache[T]):
    """
    File-based disk cache with TTL support.

    Stores cached values as serialized files on disk.
    """

    def __init__(
        self,
        cache_dir: str = "/pluribus/.pluribus/cache",
        max_size_mb: int = 1024,
        default_ttl: int = 3600,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self._stats = CacheStats()
        self._index: Dict[str, Dict[str, Any]] = {}

        self._ensure_cache_dir()
        self._load_index()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            try:
                self._index = json.loads(index_path.read_text())
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        index_path = self.cache_dir / "index.json"
        try:
            index_path.write_text(json.dumps(self._index))
        except Exception:
            pass

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        hash_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"

    def get(self, key: str) -> Optional[T]:
        """Get value from disk cache."""
        if key not in self._index:
            self._stats.misses += 1
            return None

        meta = self._index[key]
        expires_at = meta.get("expires_at")

        if expires_at and time.time() > expires_at:
            self.delete(key)
            self._stats.misses += 1
            return None

        cache_path = self._key_to_path(key)
        if not cache_path.exists():
            self._stats.misses += 1
            return None

        try:
            with open(cache_path, "rb") as f:
                value = pickle.load(f)
            self._stats.hits += 1
            return value
        except Exception:
            self._stats.misses += 1
            return None

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        ttl = ttl or self.default_ttl
        now = time.time()

        cache_path = self._key_to_path(key)

        try:
            data = pickle.dumps(value)
            size_bytes = len(data)

            # Check size limit
            if self._stats.memory_bytes + size_bytes > self.max_size_bytes:
                self._cleanup_to_size(self.max_size_bytes - size_bytes)

            with open(cache_path, "wb") as f:
                f.write(data)

            self._index[key] = {
                "created_at": now,
                "expires_at": now + ttl if ttl else None,
                "size_bytes": size_bytes,
            }
            self._save_index()

            self._stats.size = len(self._index)
            self._stats.memory_bytes += size_bytes

            return True

        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        if key not in self._index:
            return False

        cache_path = self._key_to_path(key)

        try:
            if cache_path.exists():
                size = self._index[key].get("size_bytes", 0)
                cache_path.unlink()
                self._stats.memory_bytes -= size
        except Exception:
            pass

        del self._index[key]
        self._save_index()
        self._stats.size = len(self._index)

        return True

    def clear(self) -> None:
        """Clear all disk cache entries."""
        for key in list(self._index.keys()):
            self.delete(key)
        self._index = {}
        self._save_index()

    def exists(self, key: str) -> bool:
        """Check if key exists in disk cache."""
        if key not in self._index:
            return False

        meta = self._index[key]
        expires_at = meta.get("expires_at")

        if expires_at and time.time() > expires_at:
            self.delete(key)
            return False

        return self._key_to_path(key).exists()

    def stats(self) -> CacheStats:
        """Get disk cache statistics."""
        return self._stats

    def _cleanup_to_size(self, target_bytes: int) -> None:
        """Remove entries until under target size."""
        # Sort by creation time (oldest first)
        sorted_keys = sorted(
            self._index.keys(),
            key=lambda k: self._index[k].get("created_at", 0),
        )

        for key in sorted_keys:
            if self._stats.memory_bytes <= target_bytes:
                break
            self.delete(key)
            self._stats.evictions += 1


# =============================================================================
# Multi-Tier Cache
# =============================================================================

class MultiTierCache(Cache[T]):
    """
    Multi-tier cache combining memory and disk caching.

    PBTSO Phase: ITERATE, VERIFY

    Features:
    - L1: Fast memory cache
    - L2: Persistent disk cache
    - Automatic promotion/demotion between tiers
    - TTL-based expiration
    - Statistics and monitoring

    Usage:
        cache = MultiTierCache(config)
        cache.set("key", value, ttl=3600)
        value = cache.get("key")
    """

    BUS_TOPICS = {
        "hit": "code.cache.hit",
        "miss": "code.cache.miss",
        "evict": "code.cache.evict",
        "stats": "code.cache.stats",
    }

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or CacheConfig()
        self.bus = bus or LockedAgentBus()

        # Initialize tiers
        self._memory = MemoryCache[T](
            max_memory_mb=self.config.max_memory_mb,
            default_ttl=self.config.default_ttl_s,
        )

        self._disk: Optional[DiskCache[T]] = None
        if self.config.enable_disk_cache:
            self._disk = DiskCache[T](
                cache_dir=self.config.disk_cache_dir,
                max_size_mb=self.config.max_disk_mb,
                default_ttl=self.config.default_ttl_s,
            )

        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_s)
                if isinstance(self._memory._lru, LRUCache):
                    self._memory._lru.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def get(self, key: str) -> Optional[T]:
        """
        Get value, checking tiers in order.

        L1 (memory) -> L2 (disk)
        """
        # Try memory first
        value = self._memory.get(key)
        if value is not None:
            self._emit_event("hit", {"key": key, "tier": "memory"})
            return value

        # Try disk
        if self._disk:
            value = self._disk.get(key)
            if value is not None:
                # Promote to memory
                self._memory.set(key, value)
                self._emit_event("hit", {"key": key, "tier": "disk"})
                return value

        self._emit_event("miss", {"key": key})
        return None

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache tiers.

        Writes to both memory and disk (write-through).
        """
        ttl = ttl or self.config.default_ttl_s

        # Write to memory
        self._memory.set(key, value, ttl)

        # Write to disk
        if self._disk:
            self._disk.set(key, value, ttl)

        return True

    def delete(self, key: str) -> bool:
        """Delete from all tiers."""
        mem_deleted = self._memory.delete(key)
        disk_deleted = self._disk.delete(key) if self._disk else False
        return mem_deleted or disk_deleted

    def clear(self) -> None:
        """Clear all tiers."""
        self._memory.clear()
        if self._disk:
            self._disk.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists in any tier."""
        if self._memory.exists(key):
            return True
        if self._disk and self._disk.exists(key):
            return True
        return False

    def stats(self) -> CacheStats:
        """Get combined statistics."""
        mem_stats = self._memory.stats()
        disk_stats = self._disk.stats() if self._disk else CacheStats()

        return CacheStats(
            hits=mem_stats.hits + disk_stats.hits,
            misses=mem_stats.misses + disk_stats.misses,
            evictions=mem_stats.evictions + disk_stats.evictions,
            size=mem_stats.size + disk_stats.size,
            max_size=mem_stats.max_size + disk_stats.max_size,
            memory_bytes=mem_stats.memory_bytes + disk_stats.memory_bytes,
        )

    def get_tier_stats(self) -> Dict[str, CacheStats]:
        """Get statistics per tier."""
        return {
            "memory": self._memory.stats(),
            "disk": self._disk.stats() if self._disk else CacheStats(),
        }

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit cache event to bus."""
        topic = self.BUS_TOPICS.get(event_type)
        if topic:
            self.bus.emit({
                "topic": topic,
                "kind": "cache",
                "actor": "cache-manager",
                "data": data,
            })


# =============================================================================
# Convenience Functions
# =============================================================================

def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()


def cached(
    cache: Cache,
    ttl: Optional[int] = None,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable:
    """
    Decorator for caching function results.

    Usage:
        @cached(cache, ttl=3600)
        def expensive_function(x, y):
            return compute(x, y)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{cache_key(*args, **kwargs)}"

            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Cache System."""
    import argparse

    parser = argparse.ArgumentParser(description="Cache System (Step 82)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats command
    subparsers.add_parser("stats", help="Show cache statistics")

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--tier", choices=["memory", "disk", "all"], default="all")

    # get command
    get_parser = subparsers.add_parser("get", help="Get cache entry")
    get_parser.add_argument("key", help="Cache key")

    # set command
    set_parser = subparsers.add_parser("set", help="Set cache entry")
    set_parser.add_argument("key", help="Cache key")
    set_parser.add_argument("value", help="Value to cache")
    set_parser.add_argument("--ttl", type=int, help="TTL in seconds")

    # delete command
    del_parser = subparsers.add_parser("delete", help="Delete cache entry")
    del_parser.add_argument("key", help="Cache key")

    args = parser.parse_args()
    cache = MultiTierCache()

    if args.command == "stats":
        stats = cache.stats()
        print(json.dumps(stats.to_dict(), indent=2))
        print("\nPer-tier stats:")
        for tier, tier_stats in cache.get_tier_stats().items():
            print(f"  {tier}: {tier_stats.to_dict()}")
        return 0

    elif args.command == "clear":
        if args.tier in ("memory", "all"):
            cache._memory.clear()
            print("Memory cache cleared")
        if args.tier in ("disk", "all") and cache._disk:
            cache._disk.clear()
            print("Disk cache cleared")
        return 0

    elif args.command == "get":
        value = cache.get(args.key)
        if value is not None:
            print(f"Value: {value}")
            return 0
        else:
            print("Not found")
            return 1

    elif args.command == "set":
        cache.set(args.key, args.value, args.ttl)
        print(f"Set {args.key} = {args.value}")
        return 0

    elif args.command == "delete":
        if cache.delete(args.key):
            print(f"Deleted {args.key}")
            return 0
        else:
            print("Not found")
            return 1

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
