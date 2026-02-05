#!/usr/bin/env python3
"""
layer.py - Deploy Caching Layer (Step 232)

PBTSO Phase: SKILL, ITERATE
A2A Integration: Multi-tier caching for deployment artifacts via deploy.cache.*

Provides:
- CacheTier: Cache tier types
- CachePolicy: Cache eviction policies
- CacheEntry: Cached item
- CacheStats: Cache statistics
- CacheTierConfig: Tier configuration
- DeployCachingLayer: Main caching layer

Bus Topics:
- deploy.cache.hit
- deploy.cache.miss
- deploy.cache.evict
- deploy.cache.invalidate

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import socket
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar


# ==============================================================================
# Bus Emission Helper with File Locking (DKIN v30)
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "cache-layer"
) -> str:
    """Emit an event to the Pluribus bus with fcntl.flock() file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class CacheTier(Enum):
    """Cache tier types."""
    MEMORY = "memory"      # L1 - fastest, smallest
    LOCAL = "local"        # L2 - local disk
    DISTRIBUTED = "distributed"  # L3 - shared across nodes
    REMOTE = "remote"      # L4 - external cache (Redis, etc.)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In First Out
    TTL = "ttl"           # Time To Live only
    SIZE = "size"         # Size-based eviction


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """
    Cached item.

    Attributes:
        key: Cache key
        value: Cached value
        tier: Cache tier
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        expires_at: Expiration timestamp
        access_count: Number of accesses
        size_bytes: Size in bytes
        tags: Cache tags for invalidation
        checksum: Content checksum
    """
    key: str
    value: T
    tier: CacheTier = CacheTier.MEMORY
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    access_count: int = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    checksum: str = ""

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "tier": self.tier.value,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "tags": self.tags,
            "checksum": self.checksum,
        }


@dataclass
class CacheStats:
    """
    Cache statistics.

    Attributes:
        tier: Cache tier
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of evictions
        size_bytes: Current size in bytes
        item_count: Number of cached items
        hit_rate: Cache hit rate (0-1)
    """
    tier: CacheTier
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    item_count: int = 0
    hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size_bytes": self.size_bytes,
            "item_count": self.item_count,
            "hit_rate": self.hit_rate,
        }

    def update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


@dataclass
class CacheTierConfig:
    """
    Configuration for a cache tier.

    Attributes:
        tier: Cache tier type
        enabled: Whether tier is enabled
        max_size_bytes: Maximum size in bytes
        max_items: Maximum number of items
        default_ttl_s: Default TTL in seconds
        policy: Eviction policy
        persist: Whether to persist to disk
        path: Path for disk persistence
    """
    tier: CacheTier
    enabled: bool = True
    max_size_bytes: int = 100 * 1024 * 1024  # 100MB
    max_items: int = 10000
    default_ttl_s: int = 3600
    policy: CachePolicy = CachePolicy.LRU
    persist: bool = False
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "enabled": self.enabled,
            "max_size_bytes": self.max_size_bytes,
            "max_items": self.max_items,
            "default_ttl_s": self.default_ttl_s,
            "policy": self.policy.value,
            "persist": self.persist,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheTierConfig":
        data = dict(data)
        if "tier" in data:
            data["tier"] = CacheTier(data["tier"])
        if "policy" in data:
            data["policy"] = CachePolicy(data["policy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Cache Tier Implementation
# ==============================================================================

class CacheTierImpl:
    """Base implementation for a cache tier."""

    def __init__(self, config: CacheTierConfig):
        self.config = config
        self.stats = CacheStats(tier=config.tier)
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_size = 0

        if config.persist and config.path:
            self._load_from_disk()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get an entry from cache."""
        entry = self._data.get(key)
        if entry is None:
            self.stats.misses += 1
            self.stats.update_hit_rate()
            return None

        if entry.is_expired():
            self._remove(key)
            self.stats.misses += 1
            self.stats.update_hit_rate()
            return None

        # Update access tracking
        entry.accessed_at = time.time()
        entry.access_count += 1

        # Move to end for LRU
        if self.config.policy == CachePolicy.LRU:
            self._data.move_to_end(key)

        self.stats.hits += 1
        self.stats.update_hit_rate()
        return entry

    def put(
        self,
        key: str,
        value: Any,
        ttl_s: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> CacheEntry:
        """Put an entry in cache."""
        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 0

        # Calculate checksum
        try:
            checksum = hashlib.md5(pickle.dumps(value)).hexdigest()[:16]
        except Exception:
            checksum = ""

        # Calculate expiration
        ttl = ttl_s if ttl_s is not None else self.config.default_ttl_s
        expires_at = time.time() + ttl if ttl > 0 else 0.0

        entry = CacheEntry(
            key=key,
            value=value,
            tier=self.config.tier,
            expires_at=expires_at,
            size_bytes=size_bytes,
            tags=tags or [],
            checksum=checksum,
        )

        # Check if we need to evict
        self._ensure_capacity(size_bytes)

        # Remove existing entry if present
        if key in self._data:
            old_entry = self._data[key]
            self._total_size -= old_entry.size_bytes

        self._data[key] = entry
        self._total_size += size_bytes

        self.stats.size_bytes = self._total_size
        self.stats.item_count = len(self._data)

        return entry

    def remove(self, key: str) -> bool:
        """Remove an entry from cache."""
        return self._remove(key)

    def _remove(self, key: str) -> bool:
        """Internal remove."""
        if key not in self._data:
            return False

        entry = self._data.pop(key)
        self._total_size -= entry.size_bytes
        self.stats.size_bytes = self._total_size
        self.stats.item_count = len(self._data)
        return True

    def _ensure_capacity(self, needed_bytes: int) -> int:
        """Ensure capacity by evicting if necessary."""
        evicted = 0

        # Check item count limit
        while len(self._data) >= self.config.max_items:
            self._evict_one()
            evicted += 1

        # Check size limit
        while self._total_size + needed_bytes > self.config.max_size_bytes:
            if not self._evict_one():
                break
            evicted += 1

        self.stats.evictions += evicted
        return evicted

    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self._data:
            return False

        if self.config.policy == CachePolicy.LRU:
            # Remove oldest (first)
            key = next(iter(self._data))
            self._remove(key)
            return True

        elif self.config.policy == CachePolicy.LFU:
            # Remove least frequently used
            min_count = float("inf")
            min_key = None
            for key, entry in self._data.items():
                if entry.access_count < min_count:
                    min_count = entry.access_count
                    min_key = key
            if min_key:
                self._remove(min_key)
                return True

        elif self.config.policy == CachePolicy.FIFO:
            # Remove first
            key = next(iter(self._data))
            self._remove(key)
            return True

        elif self.config.policy == CachePolicy.SIZE:
            # Remove largest
            max_size = 0
            max_key = None
            for key, entry in self._data.items():
                if entry.size_bytes > max_size:
                    max_size = entry.size_bytes
                    max_key = key
            if max_key:
                self._remove(max_key)
                return True

        elif self.config.policy == CachePolicy.TTL:
            # Remove expired or oldest by creation time
            now = time.time()
            for key, entry in list(self._data.items()):
                if entry.expires_at > 0 and entry.expires_at < now:
                    self._remove(key)
                    return True
            # No expired, remove oldest
            if self._data:
                key = next(iter(self._data))
                self._remove(key)
                return True

        return False

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag."""
        removed = 0
        for key in list(self._data.keys()):
            entry = self._data.get(key)
            if entry and tag in entry.tags:
                self._remove(key)
                removed += 1
        return removed

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all entries with key prefix."""
        removed = 0
        for key in list(self._data.keys()):
            if key.startswith(prefix):
                self._remove(key)
                removed += 1
        return removed

    def clear(self) -> int:
        """Clear all entries."""
        count = len(self._data)
        self._data.clear()
        self._total_size = 0
        self.stats.size_bytes = 0
        self.stats.item_count = 0
        return count

    def prune_expired(self) -> int:
        """Remove all expired entries."""
        now = time.time()
        removed = 0
        for key in list(self._data.keys()):
            entry = self._data.get(key)
            if entry and entry.expires_at > 0 and entry.expires_at < now:
                self._remove(key)
                removed += 1
        return removed

    def get_keys(self, prefix: Optional[str] = None) -> List[str]:
        """Get all cache keys."""
        if prefix:
            return [k for k in self._data.keys() if k.startswith(prefix)]
        return list(self._data.keys())

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.config.path:
            return

        cache_file = Path(self.config.path) / f"{self.config.tier.value}_cache.pkl"
        if not cache_file.exists():
            return

        try:
            with open(cache_file, "rb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = pickle.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            self._data = OrderedDict(data.get("entries", {}))
            self._total_size = data.get("total_size", 0)
            self.prune_expired()

        except Exception:
            pass

    def save_to_disk(self) -> None:
        """Save cache to disk."""
        if not self.config.path or not self.config.persist:
            return

        cache_dir = Path(self.config.path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.config.tier.value}_cache.pkl"

        try:
            with open(cache_file, "wb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    pickle.dump({
                        "entries": dict(self._data),
                        "total_size": self._total_size,
                    }, f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass


# ==============================================================================
# Deploy Caching Layer (Step 232)
# ==============================================================================

class DeployCachingLayer:
    """
    Deploy Caching Layer - multi-tier caching for deployment artifacts.

    PBTSO Phase: SKILL, ITERATE

    Responsibilities:
    - Provide multi-tier cache (memory, local, distributed)
    - Manage cache lifecycle (TTL, eviction)
    - Support cache invalidation by tag/prefix
    - Track cache statistics
    - Persist cache across restarts

    Example:
        >>> cache = DeployCachingLayer()
        >>> cache.set("artifact:v1.0.0", artifact_data, ttl_s=3600)
        >>> entry = cache.get("artifact:v1.0.0")
        >>> if entry:
        ...     print(f"Cache hit: {entry.access_count} accesses")
    """

    BUS_TOPICS = {
        "hit": "deploy.cache.hit",
        "miss": "deploy.cache.miss",
        "evict": "deploy.cache.evict",
        "invalidate": "deploy.cache.invalidate",
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "cache-layer",
        tier_configs: Optional[Dict[CacheTier, CacheTierConfig]] = None,
    ):
        """
        Initialize the caching layer.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            tier_configs: Configuration for each tier
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "cache"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Initialize default tier configs
        default_configs = {
            CacheTier.MEMORY: CacheTierConfig(
                tier=CacheTier.MEMORY,
                max_size_bytes=100 * 1024 * 1024,  # 100MB
                max_items=10000,
                default_ttl_s=300,
                policy=CachePolicy.LRU,
            ),
            CacheTier.LOCAL: CacheTierConfig(
                tier=CacheTier.LOCAL,
                max_size_bytes=1024 * 1024 * 1024,  # 1GB
                max_items=100000,
                default_ttl_s=3600,
                policy=CachePolicy.LRU,
                persist=True,
                path=str(self.state_dir / "local"),
            ),
        }

        if tier_configs:
            default_configs.update(tier_configs)

        self._tier_configs = default_configs
        self._tiers: Dict[CacheTier, CacheTierImpl] = {}

        for tier, config in self._tier_configs.items():
            if config.enabled:
                self._tiers[tier] = CacheTierImpl(config)

        # Define tier order (L1 -> L4)
        self._tier_order = [CacheTier.MEMORY, CacheTier.LOCAL, CacheTier.DISTRIBUTED, CacheTier.REMOTE]

        # Custom loaders for cache misses
        self._loaders: Dict[str, Callable[[str], Any]] = {}

    def get(
        self,
        key: str,
        tiers: Optional[List[CacheTier]] = None,
    ) -> Optional[CacheEntry]:
        """
        Get a value from cache.

        Args:
            key: Cache key
            tiers: Specific tiers to check (default: all)

        Returns:
            CacheEntry if found, None otherwise
        """
        search_tiers = tiers or [t for t in self._tier_order if t in self._tiers]

        for tier in search_tiers:
            tier_impl = self._tiers.get(tier)
            if not tier_impl:
                continue

            entry = tier_impl.get(key)
            if entry:
                _emit_bus_event(
                    self.BUS_TOPICS["hit"],
                    {
                        "key": key,
                        "tier": tier.value,
                        "access_count": entry.access_count,
                    },
                    kind="metric",
                    actor=self.actor_id,
                )

                # Promote to higher tiers
                self._promote_to_higher_tiers(key, entry, tier)

                return entry

        _emit_bus_event(
            self.BUS_TOPICS["miss"],
            {"key": key},
            kind="metric",
            actor=self.actor_id,
        )

        return None

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get cached value or default."""
        entry = self.get(key)
        return entry.value if entry else default

    def set(
        self,
        key: str,
        value: Any,
        ttl_s: Optional[int] = None,
        tags: Optional[List[str]] = None,
        tiers: Optional[List[CacheTier]] = None,
    ) -> CacheEntry:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_s: Time to live in seconds
            tags: Tags for invalidation
            tiers: Specific tiers to write to

        Returns:
            CacheEntry
        """
        write_tiers = tiers or [t for t in self._tier_order if t in self._tiers]
        entry = None

        for tier in write_tiers:
            tier_impl = self._tiers.get(tier)
            if tier_impl:
                entry = tier_impl.put(key, value, ttl_s, tags)

        return entry

    def delete(self, key: str) -> bool:
        """Delete a key from all tiers."""
        deleted = False
        for tier_impl in self._tiers.values():
            if tier_impl.remove(key):
                deleted = True
        return deleted

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a tag across all tiers."""
        total = 0
        for tier_impl in self._tiers.values():
            count = tier_impl.invalidate_by_tag(tag)
            total += count

        if total > 0:
            _emit_bus_event(
                self.BUS_TOPICS["invalidate"],
                {"tag": tag, "count": total},
                actor=self.actor_id,
            )

        return total

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all entries with key prefix."""
        total = 0
        for tier_impl in self._tiers.values():
            count = tier_impl.invalidate_by_prefix(prefix)
            total += count

        if total > 0:
            _emit_bus_event(
                self.BUS_TOPICS["invalidate"],
                {"prefix": prefix, "count": total},
                actor=self.actor_id,
            )

        return total

    def invalidate_service(self, service_name: str) -> int:
        """Invalidate all cache entries for a service."""
        return self.invalidate_by_prefix(f"service:{service_name}:")

    def invalidate_deployment(self, deployment_id: str) -> int:
        """Invalidate all cache entries for a deployment."""
        return self.invalidate_by_tag(f"deployment:{deployment_id}")

    def clear(self, tier: Optional[CacheTier] = None) -> int:
        """Clear cache, optionally for specific tier."""
        total = 0
        if tier:
            tier_impl = self._tiers.get(tier)
            if tier_impl:
                total = tier_impl.clear()
        else:
            for tier_impl in self._tiers.values():
                total += tier_impl.clear()
        return total

    def prune_expired(self) -> int:
        """Prune expired entries from all tiers."""
        total = 0
        for tier_impl in self._tiers.values():
            count = tier_impl.prune_expired()
            total += count

            if count > 0:
                _emit_bus_event(
                    self.BUS_TOPICS["evict"],
                    {"tier": tier_impl.config.tier.value, "count": count, "reason": "expired"},
                    actor=self.actor_id,
                )

        return total

    def _promote_to_higher_tiers(
        self,
        key: str,
        entry: CacheEntry,
        found_tier: CacheTier,
    ) -> None:
        """Promote entry to higher (faster) tiers."""
        found_idx = self._tier_order.index(found_tier) if found_tier in self._tier_order else -1

        for i in range(found_idx):
            tier = self._tier_order[i]
            tier_impl = self._tiers.get(tier)
            if tier_impl and tier_impl.config.enabled:
                # Calculate remaining TTL
                remaining_ttl = 0
                if entry.expires_at > 0:
                    remaining_ttl = int(entry.expires_at - time.time())
                    if remaining_ttl <= 0:
                        return

                tier_impl.put(key, entry.value, remaining_ttl or None, entry.tags)

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl_s: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """Get from cache or load using provided function."""
        entry = self.get(key)
        if entry:
            return entry.value

        value = loader()
        self.set(key, value, ttl_s, tags)
        return value

    async def get_or_load_async(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl_s: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """Async version of get_or_load."""
        entry = self.get(key)
        if entry:
            return entry.value

        if asyncio.iscoroutinefunction(loader):
            value = await loader()
        else:
            value = loader()

        self.set(key, value, ttl_s, tags)
        return value

    def register_loader(self, prefix: str, loader: Callable[[str], Any]) -> None:
        """Register a loader for cache misses with specific prefix."""
        self._loaders[prefix] = loader

    def get_stats(self, tier: Optional[CacheTier] = None) -> Dict[str, CacheStats]:
        """Get cache statistics."""
        if tier:
            tier_impl = self._tiers.get(tier)
            if tier_impl:
                return {tier.value: tier_impl.stats}
            return {}

        return {t.value: impl.stats for t, impl in self._tiers.items()}

    def get_total_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all tiers."""
        total_hits = 0
        total_misses = 0
        total_size = 0
        total_items = 0

        for tier_impl in self._tiers.values():
            total_hits += tier_impl.stats.hits
            total_misses += tier_impl.stats.misses
            total_size += tier_impl.stats.size_bytes
            total_items += tier_impl.stats.item_count

        total = total_hits + total_misses
        hit_rate = total_hits / total if total > 0 else 0.0

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": hit_rate,
            "total_size_bytes": total_size,
            "total_items": total_items,
            "tiers": len(self._tiers),
        }

    def get_keys(
        self,
        tier: Optional[CacheTier] = None,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """Get all cache keys."""
        if tier:
            tier_impl = self._tiers.get(tier)
            if tier_impl:
                return tier_impl.get_keys(prefix)
            return []

        # Get keys from all tiers (deduplicated)
        keys = set()
        for tier_impl in self._tiers.values():
            keys.update(tier_impl.get_keys(prefix))
        return list(keys)

    def save(self) -> None:
        """Save persistent tiers to disk."""
        for tier_impl in self._tiers.values():
            tier_impl.save_to_disk()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for caching layer."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Caching Layer (Step 232)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # get command
    get_parser = subparsers.add_parser("get", help="Get cached value")
    get_parser.add_argument("key", help="Cache key")
    get_parser.add_argument("--json", action="store_true", help="JSON output")

    # set command
    set_parser = subparsers.add_parser("set", help="Set cached value")
    set_parser.add_argument("key", help="Cache key")
    set_parser.add_argument("value", help="Value to cache (JSON)")
    set_parser.add_argument("--ttl", "-t", type=int, help="TTL in seconds")
    set_parser.add_argument("--tags", help="Comma-separated tags")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete cached value")
    delete_parser.add_argument("key", help="Cache key")

    # invalidate command
    invalidate_parser = subparsers.add_parser("invalidate", help="Invalidate cache entries")
    invalidate_parser.add_argument("--tag", help="Invalidate by tag")
    invalidate_parser.add_argument("--prefix", help="Invalidate by key prefix")
    invalidate_parser.add_argument("--service", help="Invalidate by service name")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Get cache statistics")
    stats_parser.add_argument("--tier", "-t", choices=[t.value for t in CacheTier], help="Specific tier")
    stats_parser.add_argument("--json", action="store_true", help="JSON output")

    # keys command
    keys_parser = subparsers.add_parser("keys", help="List cache keys")
    keys_parser.add_argument("--prefix", "-p", help="Filter by prefix")
    keys_parser.add_argument("--tier", "-t", choices=[t.value for t in CacheTier], help="Specific tier")

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--tier", "-t", choices=[t.value for t in CacheTier], help="Specific tier")
    clear_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # prune command
    prune_parser = subparsers.add_parser("prune", help="Prune expired entries")
    prune_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    cache = DeployCachingLayer()

    if args.command == "get":
        entry = cache.get(args.key)

        if entry:
            if args.json:
                print(json.dumps({
                    "key": entry.key,
                    "value": entry.value,
                    "tier": entry.tier.value,
                    "access_count": entry.access_count,
                    "expires_at": entry.expires_at,
                }, indent=2))
            else:
                print(f"Key: {entry.key}")
                print(f"Value: {entry.value}")
                print(f"Tier: {entry.tier.value}")
                print(f"Accesses: {entry.access_count}")
        else:
            print(f"Key not found: {args.key}")
            return 1

        return 0

    elif args.command == "set":
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value

        tags = args.tags.split(",") if args.tags else None
        entry = cache.set(args.key, value, args.ttl, tags)

        print(f"Cached: {args.key}")
        print(f"  Size: {entry.size_bytes} bytes")
        if entry.expires_at:
            print(f"  Expires: {entry.expires_at}")

        cache.save()
        return 0

    elif args.command == "delete":
        deleted = cache.delete(args.key)

        if deleted:
            print(f"Deleted: {args.key}")
        else:
            print(f"Key not found: {args.key}")
            return 1

        cache.save()
        return 0

    elif args.command == "invalidate":
        count = 0
        if args.tag:
            count = cache.invalidate_by_tag(args.tag)
            print(f"Invalidated {count} entries with tag: {args.tag}")
        elif args.prefix:
            count = cache.invalidate_by_prefix(args.prefix)
            print(f"Invalidated {count} entries with prefix: {args.prefix}")
        elif args.service:
            count = cache.invalidate_service(args.service)
            print(f"Invalidated {count} entries for service: {args.service}")
        else:
            print("Specify --tag, --prefix, or --service")
            return 1

        cache.save()
        return 0

    elif args.command == "stats":
        tier = CacheTier(args.tier) if args.tier else None

        if tier:
            stats = cache.get_stats(tier)
        else:
            stats = cache.get_stats()

        if args.json:
            print(json.dumps({k: v.to_dict() for k, v in stats.items()}, indent=2))
        else:
            total = cache.get_total_stats()
            print(f"Total hit rate: {total['hit_rate']:.2%}")
            print(f"Total items: {total['total_items']}")
            print(f"Total size: {total['total_size_bytes']:,} bytes")
            print()
            for tier_name, tier_stats in stats.items():
                print(f"{tier_name}:")
                print(f"  Hits: {tier_stats.hits}, Misses: {tier_stats.misses}")
                print(f"  Items: {tier_stats.item_count}")
                print(f"  Size: {tier_stats.size_bytes:,} bytes")

        return 0

    elif args.command == "keys":
        tier = CacheTier(args.tier) if args.tier else None
        keys = cache.get_keys(tier, args.prefix)

        for key in keys:
            print(key)

        print(f"\nTotal: {len(keys)} keys")
        return 0

    elif args.command == "clear":
        tier = CacheTier(args.tier) if args.tier else None

        if not args.force:
            confirm = input("Are you sure you want to clear cache? [y/N] ")
            if confirm.lower() != "y":
                print("Cancelled")
                return 0

        count = cache.clear(tier)
        print(f"Cleared {count} entries")

        cache.save()
        return 0

    elif args.command == "prune":
        count = cache.prune_expired()

        if args.json:
            print(json.dumps({"pruned": count}))
        else:
            print(f"Pruned {count} expired entries")

        cache.save()
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
