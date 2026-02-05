#!/usr/bin/env python3
"""
Test Comparison Module - Step 124

Provides test run comparison capabilities.

Components:
- TestComparator: Compare test runs
- ComparisonReport: Comparison report
- DiffType: Types of differences

Bus Topics:
- test.compare.request
- test.compare.result
"""

from .comparator import (
    TestComparator,
    CompareConfig,
    CompareResult,
    TestDiff,
    DiffType,
)

__all__ = [
    "TestComparator",
    "CompareConfig",
    "CompareResult",
    "TestDiff",
    "DiffType",
]
