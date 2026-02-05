#!/usr/bin/env python3
"""
Cache module for Research Agent.

Contains caching infrastructure for research results.

Steps implemented:
- Step 19: Cache Manager
"""
from __future__ import annotations

from .cache_manager import (
    CacheManager,
    CacheEntry,
    CacheConfig,
    CacheStats,
    EvictionPolicy,
)

__all__ = [
    "CacheManager",
    "CacheEntry",
    "CacheConfig",
    "CacheStats",
    "EvictionPolicy",
]
