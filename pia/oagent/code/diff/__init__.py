#!/usr/bin/env python3
"""
Diff module - Minimal diff generation and optimization.

Part of OAGENT 300-Step Plan: Step 61

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""

from .optimizer import DiffOptimizer, DiffChunk, OptimizedDiff

__all__ = [
    "DiffOptimizer",
    "DiffChunk",
    "OptimizedDiff",
]
