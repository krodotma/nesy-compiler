#!/usr/bin/env python3
"""
Code Smell Detector Package (Step 154)

Provides detection of code smells and anti-patterns.
"""

from .detector import (
    CodeSmellDetector,
    CodeSmell,
    SmellType,
    SmellSeverity,
    SmellDetectionResult,
)

__all__ = [
    "CodeSmellDetector",
    "CodeSmell",
    "SmellType",
    "SmellSeverity",
    "SmellDetectionResult",
]
