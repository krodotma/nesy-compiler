#!/usr/bin/env python3
"""
cache.py - Gate Result Caching

P2-084: Create gate result caching
P2-086: Add bloom filter for duplicate detection

Implements:
- LRU cache for gate results
- Bloom filter for fast duplicate detection
"""

import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from collections import OrderedDict

logger = logging.getLogger("ARK.Perf.Cache")


class BloomFilter:
    """
    Bloom filter for fast duplicate detection.
    
    P2-086: Add bloom filter for duplicate detection
    """
    
    def __init__(self, size: int = 10000, hash_count: int = 5):
        self.size = size
        self.hash_count = hash_count
        self.bits = [False] * size
        self.count = 0
    
    def _hashes(self, item: str) -> List[int]:
        """Generate hash indices for an item."""
        indices = []
        for i in range(self.hash_count):
            h = hashlib.md5(f"{item}_{i}".encode()).hexdigest()
            indices.append(int(h, 16) % self.size)
        return indices
    
    def add(self, item: str) -> None:
        """Add item to filter."""
        for idx in self._hashes(item):
            self.bits[idx] = True
        self.count += 1
    
    def contains(self, item: str) -> bool:
        """Check if item might be in filter (may have false positives)."""
        return all(self.bits[idx] for idx in self._hashes(item))
    
    def clear(self) -> None:
        """Clear the filter."""
        self.bits = [False] * self.size
        self.count = 0
    
    def false_positive_rate(self) -> float:
        """Estimate current false positive rate."""
        if self.count == 0:
            return 0.0
        filled = sum(self.bits) / self.size
        return filled ** self.hash_count


@dataclass
class CacheEntry:
    """A cached gate result."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    hits: int = 0
    ttl: float = 300.0  # 5 minutes default


class GateCache:
    """
    LRU cache for gate results.
    
    P2-084: Create gate result caching
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.bloom = BloomFilter()
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, gate_name: str, inputs: Dict[str, Any]) -> str:
        """Generate cache key from gate name and inputs."""
        input_str = str(sorted(inputs.items()))
        return hashlib.md5(f"{gate_name}:{input_str}".encode()).hexdigest()
    
    def get(self, gate_name: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._make_key(gate_name, inputs)
        
        # Fast path: bloom filter check
        if not self.bloom.contains(key):
            self.misses += 1
            return None
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if time.time() - entry.timestamp > entry.ttl:
            del self.cache[key]
            self.misses += 1
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        entry.hits += 1
        self.hits += 1
        
        return entry.value
    
    def set(
        self, gate_name: str, inputs: Dict[str, Any], 
        value: Any, ttl: Optional[float] = None
    ) -> None:
        """Cache a gate result."""
        key = self._make_key(gate_name, inputs)
        
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Evict oldest if full
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        # Add new entry
        entry = CacheEntry(
            key=key, value=value, 
            ttl=ttl if ttl is not None else self.default_ttl
        )
        self.cache[key] = entry
        self.bloom.add(key)
    
    def invalidate(self, gate_name: str, inputs: Dict[str, Any]) -> bool:
        """Invalidate a cached entry."""
        key = self._make_key(gate_name, inputs)
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.bloom.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "bloom_fp_rate": self.bloom.false_positive_rate()
        }


# Global cache instance
_global_cache = GateCache()


def get_cache() -> GateCache:
    """Get global cache instance."""
    return _global_cache
