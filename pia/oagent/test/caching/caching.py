#!/usr/bin/env python3
"""
Step 132: Test Caching Layer

Multi-tier test result caching system.

PBTSO Phase: TEST, OBSERVE
Bus Topics:
- test.caching.hit (emits)
- test.caching.miss (emits)
- test.caching.store (emits)
- test.caching.evict (emits)

Dependencies: Steps 101-131 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar('T')


# ============================================================================
# Constants
# ============================================================================

class CacheTier(Enum):
    """Cache tier levels."""
    MEMORY = "memory"      # L1: In-memory cache
    DISK = "disk"          # L2: File-based cache
    DATABASE = "database"  # L3: SQLite database


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time-To-Live based
    SIZE = "size"         # Size-based eviction


class CacheStatus(Enum):
    """Cache lookup status."""
    HIT = "hit"
    MISS = "miss"
    STALE = "stale"
    EXPIRED = "expired"
    ERROR = "error"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class CacheEntry:
    """
    A cached entry.

    Attributes:
        key: Cache key
        value: Cached value
        tier: Cache tier
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        expires_at: Expiration timestamp
        hits: Number of hits
        size_bytes: Entry size
        metadata: Additional metadata
    """
    key: str
    value: Any
    tier: CacheTier = CacheTier.MEMORY
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without value)."""
        return {
            "key": self.key,
            "tier": self.tier.value,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "expires_at": self.expires_at,
            "hits": self.hits,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


@dataclass
class CacheStats:
    """Cache statistics."""
    tier: CacheTier
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier": self.tier.value,
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
        }


@dataclass
class CacheLookupResult:
    """Result of a cache lookup."""
    status: CacheStatus
    entry: Optional[CacheEntry] = None
    tier: Optional[CacheTier] = None
    duration_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "tier": self.tier.value if self.tier else None,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class CachingConfig:
    """
    Configuration for the caching layer.

    Attributes:
        enabled_tiers: Enabled cache tiers
        memory_max_entries: Max entries in memory
        memory_max_size_mb: Max memory size
        disk_cache_dir: Disk cache directory
        disk_max_size_mb: Max disk cache size
        db_path: Database path
        default_ttl_s: Default TTL
        strategy: Eviction strategy
        compression: Enable compression
    """
    enabled_tiers: List[CacheTier] = field(default_factory=lambda: [
        CacheTier.MEMORY,
        CacheTier.DISK,
    ])
    memory_max_entries: int = 1000
    memory_max_size_mb: int = 100
    disk_cache_dir: str = ".pluribus/test-agent/cache/disk"
    disk_max_size_mb: int = 500
    db_path: str = ".pluribus/test-agent/cache/cache.db"
    default_ttl_s: int = 3600  # 1 hour
    strategy: CacheStrategy = CacheStrategy.LRU
    compression: bool = False
    output_dir: str = ".pluribus/test-agent/cache"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled_tiers": [t.value for t in self.enabled_tiers],
            "memory_max_entries": self.memory_max_entries,
            "memory_max_size_mb": self.memory_max_size_mb,
            "disk_max_size_mb": self.disk_max_size_mb,
            "default_ttl_s": self.default_ttl_s,
            "strategy": self.strategy.value,
            "compression": self.compression,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class CachingBus:
    """Bus interface for caching with file locking."""

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
# Cache Tiers
# ============================================================================

class MemoryCache:
    """In-memory cache tier (L1)."""

    def __init__(self, max_entries: int = 1000, max_size_mb: int = 100):
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats(tier=CacheTier.MEMORY)

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        entry = self._cache.get(key)
        if entry:
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                return None
            entry.accessed_at = time.time()
            entry.hits += 1
            self._stats.hits += 1
            return entry
        self._stats.misses += 1
        return None

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None, metadata: Optional[Dict] = None) -> CacheEntry:
        """Set entry in cache."""
        # Estimate size
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 0

        entry = CacheEntry(
            key=key,
            value=value,
            tier=CacheTier.MEMORY,
            expires_at=time.time() + ttl_s if ttl_s else None,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )

        self._cache[key] = entry
        self._stats.total_entries = len(self._cache)
        self._maybe_evict()

        return entry

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self._cache:
            del self._cache[key]
            self._stats.total_entries = len(self._cache)
            return True
        return False

    def clear(self) -> int:
        """Clear all entries."""
        count = len(self._cache)
        self._cache.clear()
        self._stats.total_entries = 0
        return count

    def _maybe_evict(self) -> None:
        """Evict entries if needed."""
        while len(self._cache) > self.max_entries:
            # LRU eviction
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
            del self._cache[oldest_key]
            self._stats.evictions += 1
            self._stats.total_entries = len(self._cache)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.total_entries = len(self._cache)
        self._stats.total_size_bytes = sum(e.size_bytes for e in self._cache.values())
        return self._stats


class DiskCache:
    """File-based cache tier (L2)."""

    def __init__(self, cache_dir: str, max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = CacheStats(tier=CacheTier.DISK)
        self._index_file = self.cache_dir / "index.json"
        self._index: Dict[str, Dict] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index."""
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        self._index = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (json.JSONDecodeError, IOError):
                self._index = {}

    def _save_index(self) -> None:
        """Save cache index."""
        with open(self._index_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(self._index, f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash[:2]}" / f"{key_hash}.pkl"

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        if key not in self._index:
            self._stats.misses += 1
            return None

        meta = self._index[key]
        if meta.get("expires_at") and time.time() > meta["expires_at"]:
            self.delete(key)
            self._stats.misses += 1
            return None

        file_path = self._key_to_path(key)
        if not file_path.exists():
            del self._index[key]
            self._save_index()
            self._stats.misses += 1
            return None

        try:
            with open(file_path, "rb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    value = pickle.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Update access time
            self._index[key]["accessed_at"] = time.time()
            self._index[key]["hits"] = self._index[key].get("hits", 0) + 1
            self._save_index()

            entry = CacheEntry(
                key=key,
                value=value,
                tier=CacheTier.DISK,
                created_at=meta.get("created_at", 0),
                accessed_at=time.time(),
                expires_at=meta.get("expires_at"),
                hits=self._index[key]["hits"],
                size_bytes=meta.get("size_bytes", 0),
                metadata=meta.get("metadata", {}),
            )

            self._stats.hits += 1
            return entry

        except Exception:
            self._stats.errors += 1
            return None

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None, metadata: Optional[Dict] = None) -> CacheEntry:
        """Set entry in cache."""
        file_path = self._key_to_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = pickle.dumps(value)
            size_bytes = len(data)

            with open(file_path, "wb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(data)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            now = time.time()
            self._index[key] = {
                "created_at": now,
                "accessed_at": now,
                "expires_at": now + ttl_s if ttl_s else None,
                "hits": 0,
                "size_bytes": size_bytes,
                "metadata": metadata or {},
            }
            self._save_index()

            self._maybe_evict()

            return CacheEntry(
                key=key,
                value=value,
                tier=CacheTier.DISK,
                created_at=now,
                expires_at=now + ttl_s if ttl_s else None,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )

        except Exception:
            self._stats.errors += 1
            raise

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key not in self._index:
            return False

        file_path = self._key_to_path(key)
        if file_path.exists():
            file_path.unlink()

        del self._index[key]
        self._save_index()
        return True

    def clear(self) -> int:
        """Clear all entries."""
        count = len(self._index)
        for key in list(self._index.keys()):
            self.delete(key)
        return count

    def _maybe_evict(self) -> None:
        """Evict entries if needed."""
        total_size = sum(m.get("size_bytes", 0) for m in self._index.values())
        while total_size > self.max_size_bytes and self._index:
            # LRU eviction
            oldest_key = min(self._index.keys(), key=lambda k: self._index[k].get("accessed_at", 0))
            total_size -= self._index[oldest_key].get("size_bytes", 0)
            self.delete(oldest_key)
            self._stats.evictions += 1

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.total_entries = len(self._index)
        self._stats.total_size_bytes = sum(m.get("size_bytes", 0) for m in self._index.values())
        return self._stats


class DatabaseCache:
    """SQLite-based cache tier (L3)."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stats = CacheStats(tier=CacheTier.DATABASE)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at REAL,
                accessed_at REAL,
                expires_at REAL,
                hits INTEGER DEFAULT 0,
                size_bytes INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache(accessed_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)")
        conn.commit()
        conn.close()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT value, created_at, accessed_at, expires_at, hits, size_bytes, metadata
            FROM cache WHERE key = ?
        """, (key,))

        row = cursor.fetchone()
        if not row:
            self._stats.misses += 1
            conn.close()
            return None

        value_blob, created_at, accessed_at, expires_at, hits, size_bytes, metadata_json = row

        if expires_at and time.time() > expires_at:
            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            conn.close()
            self._stats.misses += 1
            return None

        # Update access time
        cursor.execute("""
            UPDATE cache SET accessed_at = ?, hits = hits + 1 WHERE key = ?
        """, (time.time(), key))
        conn.commit()

        try:
            value = pickle.loads(value_blob)
            metadata = json.loads(metadata_json) if metadata_json else {}

            entry = CacheEntry(
                key=key,
                value=value,
                tier=CacheTier.DATABASE,
                created_at=created_at,
                accessed_at=time.time(),
                expires_at=expires_at,
                hits=hits + 1,
                size_bytes=size_bytes,
                metadata=metadata,
            )

            self._stats.hits += 1
            conn.close()
            return entry

        except Exception:
            self._stats.errors += 1
            conn.close()
            return None

    def set(self, key: str, value: Any, ttl_s: Optional[int] = None, metadata: Optional[Dict] = None) -> CacheEntry:
        """Set entry in cache."""
        try:
            value_blob = pickle.dumps(value)
            size_bytes = len(value_blob)
            now = time.time()
            expires_at = now + ttl_s if ttl_s else None
            metadata_json = json.dumps(metadata or {})

            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                INSERT OR REPLACE INTO cache (key, value, created_at, accessed_at, expires_at, hits, size_bytes, metadata)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """, (key, value_blob, now, now, expires_at, size_bytes, metadata_json))
            conn.commit()
            conn.close()

            return CacheEntry(
                key=key,
                value=value,
                tier=CacheTier.DATABASE,
                created_at=now,
                expires_at=expires_at,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )

        except Exception:
            self._stats.errors += 1
            raise

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def clear(self) -> int:
        """Clear all entries."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache")
        row = cursor.fetchone()
        conn.close()

        self._stats.total_entries = row[0] or 0
        self._stats.total_size_bytes = row[1] or 0
        return self._stats


# ============================================================================
# Test Caching Layer
# ============================================================================

class TestCachingLayer:
    """
    Multi-tier test result caching system.

    Features:
    - L1: In-memory cache (fastest)
    - L2: Disk-based cache (persistent)
    - L3: SQLite database (queryable)
    - Automatic tier promotion/demotion
    - TTL-based expiration
    - LRU eviction

    PBTSO Phase: TEST, OBSERVE
    Bus Topics: test.caching.hit, test.caching.miss, test.caching.store, test.caching.evict
    """

    BUS_TOPICS = {
        "hit": "test.caching.hit",
        "miss": "test.caching.miss",
        "store": "test.caching.store",
        "evict": "test.caching.evict",
    }

    def __init__(self, bus=None, config: Optional[CachingConfig] = None):
        """
        Initialize the caching layer.

        Args:
            bus: Optional bus instance
            config: Caching configuration
        """
        self.bus = bus or CachingBus()
        self.config = config or CachingConfig()
        self._tiers: Dict[CacheTier, Any] = {}

        # Initialize tiers
        if CacheTier.MEMORY in self.config.enabled_tiers:
            self._tiers[CacheTier.MEMORY] = MemoryCache(
                max_entries=self.config.memory_max_entries,
                max_size_mb=self.config.memory_max_size_mb,
            )

        if CacheTier.DISK in self.config.enabled_tiers:
            self._tiers[CacheTier.DISK] = DiskCache(
                cache_dir=self.config.disk_cache_dir,
                max_size_mb=self.config.disk_max_size_mb,
            )

        if CacheTier.DATABASE in self.config.enabled_tiers:
            self._tiers[CacheTier.DATABASE] = DatabaseCache(
                db_path=self.config.db_path,
            )

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> CacheLookupResult:
        """
        Get value from cache, checking all tiers.

        Args:
            key: Cache key

        Returns:
            CacheLookupResult with lookup status and entry
        """
        start_time = time.time()

        # Check each tier in order
        for tier in self.config.enabled_tiers:
            cache = self._tiers.get(tier)
            if cache is None:
                continue

            entry = cache.get(key)
            if entry:
                duration_ms = (time.time() - start_time) * 1000

                # Promote to faster tiers
                self._promote(key, entry)

                self._emit_event("hit", {
                    "key": key,
                    "tier": tier.value,
                    "duration_ms": duration_ms,
                })

                return CacheLookupResult(
                    status=CacheStatus.HIT,
                    entry=entry,
                    tier=tier,
                    duration_ms=duration_ms,
                )

        duration_ms = (time.time() - start_time) * 1000

        self._emit_event("miss", {
            "key": key,
            "duration_ms": duration_ms,
        })

        return CacheLookupResult(
            status=CacheStatus.MISS,
            duration_ms=duration_ms,
        )

    def set(
        self,
        key: str,
        value: Any,
        ttl_s: Optional[int] = None,
        tiers: Optional[List[CacheTier]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_s: Time-to-live in seconds
            tiers: Tiers to store in (default: all enabled)
            metadata: Additional metadata

        Returns:
            Created cache entry
        """
        ttl_s = ttl_s or self.config.default_ttl_s
        tiers = tiers or self.config.enabled_tiers
        entry = None

        for tier in tiers:
            cache = self._tiers.get(tier)
            if cache is None:
                continue

            try:
                entry = cache.set(key, value, ttl_s=ttl_s, metadata=metadata)
            except Exception as e:
                self._emit_event("error", {
                    "key": key,
                    "tier": tier.value,
                    "error": str(e),
                })

        if entry:
            self._emit_event("store", {
                "key": key,
                "tiers": [t.value for t in tiers],
                "ttl_s": ttl_s,
            })

        return entry

    def delete(self, key: str) -> bool:
        """
        Delete value from all tiers.

        Args:
            key: Cache key

        Returns:
            True if deleted from any tier
        """
        deleted = False
        for tier, cache in self._tiers.items():
            if cache.delete(key):
                deleted = True

        if deleted:
            self._emit_event("evict", {
                "key": key,
                "reason": "explicit_delete",
            })

        return deleted

    def clear(self, tiers: Optional[List[CacheTier]] = None) -> int:
        """
        Clear all entries from specified tiers.

        Args:
            tiers: Tiers to clear (default: all)

        Returns:
            Total entries cleared
        """
        tiers = tiers or list(self._tiers.keys())
        total_cleared = 0

        for tier in tiers:
            cache = self._tiers.get(tier)
            if cache:
                total_cleared += cache.clear()

        return total_cleared

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all tiers."""
        return {
            tier.value: cache.get_stats()
            for tier, cache in self._tiers.items()
        }

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics across all tiers."""
        stats = self.get_stats()
        combined = {
            "total_entries": sum(s.total_entries for s in stats.values()),
            "total_size_bytes": sum(s.total_size_bytes for s in stats.values()),
            "total_hits": sum(s.hits for s in stats.values()),
            "total_misses": sum(s.misses for s in stats.values()),
            "overall_hit_rate": 0.0,
            "tiers": {k: v.to_dict() for k, v in stats.items()},
        }

        total_requests = combined["total_hits"] + combined["total_misses"]
        if total_requests > 0:
            combined["overall_hit_rate"] = combined["total_hits"] / total_requests * 100

        return combined

    def _promote(self, key: str, entry: CacheEntry) -> None:
        """Promote entry to faster tiers."""
        # Get tier index
        tier_order = list(self.config.enabled_tiers)
        try:
            current_idx = tier_order.index(entry.tier)
        except ValueError:
            return

        # Promote to faster tiers
        for i in range(current_idx):
            faster_tier = tier_order[i]
            cache = self._tiers.get(faster_tier)
            if cache:
                try:
                    remaining_ttl = None
                    if entry.expires_at:
                        remaining_ttl = int(entry.expires_at - time.time())
                        if remaining_ttl <= 0:
                            continue
                    cache.set(key, entry.value, ttl_s=remaining_ttl, metadata=entry.metadata)
                except Exception:
                    pass

    async def get_async(self, key: str) -> CacheLookupResult:
        """Async version of get."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)

    async def set_async(
        self,
        key: str,
        value: Any,
        ttl_s: Optional[int] = None,
        tiers: Optional[List[CacheTier]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry:
        """Async version of set."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.set, key, value, ttl_s, tiers, metadata)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.caching.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "caching",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Caching Layer."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Caching Layer")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get cached value")
    get_parser.add_argument("key", help="Cache key")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set cached value")
    set_parser.add_argument("key", help="Cache key")
    set_parser.add_argument("value", help="Value to cache")
    set_parser.add_argument("--ttl", type=int, default=3600, help="TTL in seconds")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete cached value")
    delete_parser.add_argument("key", help="Cache key")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--tier", choices=["memory", "disk", "database"])

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")

    # Common arguments
    parser.add_argument("--cache-dir", default=".pluribus/test-agent/cache")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = CachingConfig(
        disk_cache_dir=f"{args.cache_dir}/disk",
        db_path=f"{args.cache_dir}/cache.db",
        output_dir=args.cache_dir,
    )
    cache = TestCachingLayer(config=config)

    if args.command == "get":
        result = cache.get(args.key)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Status: {result.status.value}")
            if result.entry:
                print(f"Tier: {result.tier.value}")
                print(f"Value: {result.entry.value}")
                print(f"Hits: {result.entry.hits}")

    elif args.command == "set":
        entry = cache.set(args.key, args.value, ttl_s=args.ttl)

        if args.json:
            print(json.dumps(entry.to_dict(), indent=2))
        else:
            print(f"Cached: {args.key}")
            print(f"TTL: {args.ttl}s")

    elif args.command == "delete":
        if cache.delete(args.key):
            print(f"Deleted: {args.key}")
        else:
            print(f"Not found: {args.key}")

    elif args.command == "clear":
        tiers = None
        if args.tier:
            tiers = [CacheTier(args.tier)]
        count = cache.clear(tiers)
        print(f"Cleared {count} entries")

    elif args.command == "stats":
        stats = cache.get_combined_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("\nCache Statistics:")
            print(f"  Total Entries: {stats['total_entries']}")
            print(f"  Total Size: {stats['total_size_bytes'] / 1024:.1f} KB")
            print(f"  Hits: {stats['total_hits']}")
            print(f"  Misses: {stats['total_misses']}")
            print(f"  Hit Rate: {stats['overall_hit_rate']:.1f}%")

            print("\n  By Tier:")
            for tier, tier_stats in stats['tiers'].items():
                print(f"    {tier}: {tier_stats['total_entries']} entries, "
                      f"{tier_stats['hit_rate']:.1f}% hit rate")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
