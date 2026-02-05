#!/usr/bin/env python3
"""
Merge module - Context-aware semantic code merging.

Part of OAGENT 300-Step Plan: Step 63

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""

from .semantic import (
    SemanticMerger,
    MergeResult,
    MergeContext,
    MergeStrategy,
    CodeBlock,
)

__all__ = [
    "SemanticMerger",
    "MergeResult",
    "MergeContext",
    "MergeStrategy",
    "CodeBlock",
]
