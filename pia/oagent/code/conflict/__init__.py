#!/usr/bin/env python3
"""
Conflict module - Merge conflict detection and resolution.

Part of OAGENT 300-Step Plan: Step 62

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""

from .resolver import (
    ConflictResolver,
    MergeConflict,
    ConflictRegion,
    ResolutionStrategy,
    ConflictResolution,
)

__all__ = [
    "ConflictResolver",
    "MergeConflict",
    "ConflictRegion",
    "ResolutionStrategy",
    "ConflictResolution",
]
