"""
Caching Utilities for Creative Section
=======================================

Provides caching infrastructure for expensive computations.

Components:
- CacheEntry: Individual cache entry with TTL
- CacheConfig: Configuration for cache behavior
- MemoryCache: In-memory LRU cache
- DiskCache: File-based persistent cache (optional)
- Decorators: @cached, @memoize for easy use
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
)

T = TypeVar("T")


# =============================================================================
# CACHE ENTRY
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """
    A single cache entry with key, value, and expiry.

    Attributes:
        key: Unique identifier for the cache entry
        value: The cached value
        expiry: Unix timestamp when entry expires (None = never)
        created_at: When the entry was created
        accessed_at: Last access time
        hit_count: Number of times this entry was accessed
    """

    key: str
    value: T
    expiry: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.expiry is None:
            return False
        return time.time() > self.expiry

    def touch(self) -> None:
        """Update access time and hit count."""
        self.accessed_at = time.time()
        self.hit_count += 1

    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL in seconds, or None if no expiry."""
        if self.expiry is None:
            return None
        remaining = self.expiry - time.time()
        return max(0.0, remaining)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "key": self.key,
            "expiry": self.expiry,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "hit_count": self.hit_count,
            # value not included as it may not be serializable
        }


# =============================================================================
# CACHE CONFIG
# =============================================================================


@dataclass
class CacheConfig:
    """
    Configuration for cache behavior.

    Attributes:
        max_size: Maximum number of entries (for LRU eviction)
        default_ttl: Default TTL in seconds (None = no expiry)
        cleanup_interval: Interval for background cleanup in seconds
        enable_stats: Whether to track cache statistics
        disk_path: Path for disk cache (None = memory only)
        serializer: Function to serialize values for disk cache
        deserializer: Function to deserialize values from disk cache
    """

    max_size: int = 1000
    default_ttl: Optional[float] = None
    cleanup_interval: float = 60.0
    enable_stats: bool = True
    disk_path: Optional[Path] = None
    serializer: Callable[[Any], bytes] = pickle.dumps
    deserializer: Callable[[bytes], Any] = pickle.loads

    @classmethod
    def for_creative(cls) -> CacheConfig:
        """Create a default config for Creative subsystems."""
        return cls(
            max_size=500,
            default_ttl=3600.0,  # 1 hour
            cleanup_interval=300.0,  # 5 minutes
            enable_stats=True,
            disk_path=Path("/tmp/pluribus_creative_cache"),
        )

    @classmethod
    def minimal(cls) -> CacheConfig:
        """Create a minimal config for testing."""
        return cls(
            max_size=100,
            default_ttl=60.0,
            cleanup_interval=30.0,
            enable_stats=False,
        )


# =============================================================================
# CACHE STATISTICS
# =============================================================================


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    writes: int = 0
    deletes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "writes": self.writes,
            "deletes": self.deletes,
            "hit_rate": round(self.hit_rate, 2),
            "total_requests": self.total_requests,
        }


# =============================================================================
# MEMORY CACHE
# =============================================================================


class MemoryCache(Generic[T]):
    """
    In-memory LRU cache with TTL support.

    Features:
    - LRU eviction when max_size is reached
    - TTL-based expiration
    - Thread-safe operations
    - Optional statistics tracking

    Example:
        >>> cache = MemoryCache[str](CacheConfig(max_size=100, default_ttl=60))
        >>> cache.set("key1", "value1")
        >>> cache.get("key1")
        'value1'
        >>> cache.get("missing")  # Returns None
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the memory cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self._config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats() if self._config.enable_stats else None

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    @property
    def stats(self) -> Optional[CacheStats]:
        """Get cache statistics (None if disabled)."""
        return self._stats

    def __len__(self) -> int:
        """Return number of entries in cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                if self._stats:
                    self._stats.misses += 1
                return default

            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                if self._stats:
                    self._stats.misses += 1
                    self._stats.expirations += 1
                return default

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            if self._stats:
                self._stats.hits += 1

            return entry.value

    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> CacheEntry[T]:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses config default if None)

        Returns:
            The created cache entry
        """
        with self._lock:
            # Calculate expiry
            effective_ttl = ttl if ttl is not None else self._config.default_ttl
            expiry = time.time() + effective_ttl if effective_ttl else None

            # Create entry
            entry = CacheEntry(key=key, value=value, expiry=expiry)

            # Evict if at capacity
            while len(self._cache) >= self._config.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if self._stats:
                    self._stats.evictions += 1

            # Insert new entry
            self._cache[key] = entry
            self._cache.move_to_end(key)

            if self._stats:
                self._stats.writes += 1

            return entry

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self._stats:
                    self._stats.deletes += 1
                return True
            return False

    def clear(self) -> int:
        """
        Clear all entries from the cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            if self._stats:
                self._stats.expirations += len(expired_keys)
            return len(expired_keys)

    def keys(self) -> list[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    def get_entry(self, key: str) -> Optional[CacheEntry[T]]:
        """
        Get the full cache entry (not just the value).

        Args:
            key: Cache key

        Returns:
            CacheEntry or None
        """
        with self._lock:
            return self._cache.get(key)


# =============================================================================
# DISK CACHE
# =============================================================================


class DiskCache(Generic[T]):
    """
    File-based persistent cache.

    Uses a directory structure with one file per cache entry.
    Suitable for larger objects that should persist across restarts.

    Example:
        >>> cache = DiskCache[dict](Path("/tmp/cache"))
        >>> cache.set("config", {"key": "value"}, ttl=3600)
        >>> cache.get("config")
        {'key': 'value'}
    """

    def __init__(
        self,
        path: Path,
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize the disk cache.

        Args:
            path: Directory path for cache storage
            config: Cache configuration
        """
        self._path = Path(path)
        self._config = config or CacheConfig()
        self._path.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._path / "_metadata.json"
        self._lock = threading.RLock()
        self._stats = CacheStats() if self._config.enable_stats else None
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        self._metadata: Dict[str, Dict[str, Any]] = {}
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    self._metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._metadata = {}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except IOError:
            pass

    def _key_to_filename(self, key: str) -> str:
        """Convert a cache key to a safe filename."""
        # Hash the key for a safe filename
        hash_val = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"cache_{hash_val}.pkl"

    def _key_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self._path / self._key_to_filename(key)

    @property
    def stats(self) -> Optional[CacheStats]:
        """Get cache statistics."""
        return self._stats

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get a value from the disk cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            # Check metadata for expiry
            meta = self._metadata.get(key)
            if meta is None:
                if self._stats:
                    self._stats.misses += 1
                return default

            # Check if expired
            expiry = meta.get("expiry")
            if expiry is not None and time.time() > expiry:
                self.delete(key)
                if self._stats:
                    self._stats.misses += 1
                    self._stats.expirations += 1
                return default

            # Load from disk
            path = self._key_path(key)
            if not path.exists():
                if self._stats:
                    self._stats.misses += 1
                return default

            try:
                with open(path, "rb") as f:
                    value = self._config.deserializer(f.read())
                if self._stats:
                    self._stats.hits += 1
                # Update access time
                self._metadata[key]["accessed_at"] = time.time()
                self._metadata[key]["hit_count"] = meta.get("hit_count", 0) + 1
                self._save_metadata()
                return value
            except (IOError, pickle.PickleError):
                if self._stats:
                    self._stats.misses += 1
                return default

    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> bool:
        """
        Set a value in the disk cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            effective_ttl = ttl if ttl is not None else self._config.default_ttl
            expiry = time.time() + effective_ttl if effective_ttl else None

            path = self._key_path(key)
            try:
                with open(path, "wb") as f:
                    f.write(self._config.serializer(value))

                self._metadata[key] = {
                    "expiry": expiry,
                    "created_at": time.time(),
                    "accessed_at": time.time(),
                    "hit_count": 0,
                }
                self._save_metadata()

                if self._stats:
                    self._stats.writes += 1

                return True
            except (IOError, pickle.PickleError):
                return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from the disk cache.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            path = self._key_path(key)
            deleted = False

            if path.exists():
                try:
                    path.unlink()
                    deleted = True
                except IOError:
                    pass

            if key in self._metadata:
                del self._metadata[key]
                self._save_metadata()
                deleted = True

            if deleted and self._stats:
                self._stats.deletes += 1

            return deleted

    def clear(self) -> int:
        """
        Clear all entries from the disk cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = 0
            for key in list(self._metadata.keys()):
                if self.delete(key):
                    count += 1
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, meta in self._metadata.items()
                if meta.get("expiry") is not None and current_time > meta["expiry"]
            ]
            count = 0
            for key in expired_keys:
                if self.delete(key):
                    count += 1
                    if self._stats:
                        self._stats.expirations += 1
            return count

    def keys(self) -> list[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._metadata.keys())


# =============================================================================
# CACHE DECORATORS
# =============================================================================


def _make_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from function call signature."""
    key_parts = [func.__module__, func.__qualname__]

    # Add positional args
    for arg in args:
        try:
            key_parts.append(repr(arg))
        except Exception:
            key_parts.append(str(id(arg)))

    # Add keyword args (sorted for consistency)
    for k, v in sorted(kwargs.items()):
        try:
            key_parts.append(f"{k}={repr(v)}")
        except Exception:
            key_parts.append(f"{k}={id(v)}")

    key_str = ":".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()


def cached(
    cache: Optional[Union[MemoryCache, DiskCache]] = None,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable:
    """
    Decorator to cache function results.

    Args:
        cache: Cache instance to use (creates MemoryCache if None)
        ttl: Time-to-live for cached results
        key_func: Custom function to generate cache keys

    Returns:
        Decorated function

    Example:
        >>> @cached(ttl=60)
        ... def expensive_computation(x: int) -> int:
        ...     return x ** 2
        >>> expensive_computation(5)  # Computed
        25
        >>> expensive_computation(5)  # From cache
        25
    """
    _cache: Union[MemoryCache, DiskCache] = cache or MemoryCache(CacheConfig.minimal())

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _make_cache_key(func, args, kwargs)

            # Try to get from cache
            result = _cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            _cache.set(key, result, ttl=ttl)
            return result

        # Attach cache for inspection
        wrapper._cache = _cache  # type: ignore
        return wrapper

    return decorator


def memoize(func: Callable[..., T]) -> Callable[..., T]:
    """
    Simple memoization decorator (no TTL, no size limit).

    For simple caching of pure functions.

    Example:
        >>> @memoize
        ... def fibonacci(n: int) -> int:
        ...     if n < 2:
        ...         return n
        ...     return fibonacci(n - 1) + fibonacci(n - 2)
        >>> fibonacci(30)  # Fast due to memoization
    """
    _memo: Dict[str, T] = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        key = _make_cache_key(func, args, kwargs)
        if key not in _memo:
            _memo[key] = func(*args, **kwargs)
        return _memo[key]

    wrapper._memo = _memo  # type: ignore
    wrapper.clear = _memo.clear  # type: ignore
    return wrapper


# =============================================================================
# GLOBAL CACHE INSTANCE
# =============================================================================

_global_cache: Optional[MemoryCache] = None


def get_cache() -> MemoryCache:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MemoryCache(CacheConfig.for_creative())
    return _global_cache


def set_cache(cache: MemoryCache) -> None:
    """Set the global cache instance."""
    global _global_cache
    _global_cache = cache
