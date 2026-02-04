#!/usr/bin/env python3
"""
Architecture Consistency Checker Package (Step 155)

Provides architecture validation against defined patterns and rules.
"""

from .checker import (
    ArchitectureChecker,
    ArchitectureViolation,
    ArchitectureRule,
    LayerDefinition,
    ArchitectureCheckResult,
)

__all__ = [
    "ArchitectureChecker",
    "ArchitectureViolation",
    "ArchitectureRule",
    "LayerDefinition",
    "ArchitectureCheckResult",
]
