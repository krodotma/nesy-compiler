#!/usr/bin/env python3
"""
Test History Module - Step 123

Provides historical test analysis and tracking capabilities.

Components:
- TestHistoryTracker: Track and analyze test history
- HistoryQuery: Query historical data
- TrendAnalyzer: Analyze trends over time

Bus Topics:
- test.history.record
- test.history.query
"""

from .tracker import (
    TestHistoryTracker,
    HistoryConfig,
    HistoryRecord,
    HistoryQuery,
    TrendData,
)

__all__ = [
    "TestHistoryTracker",
    "HistoryConfig",
    "HistoryRecord",
    "HistoryQuery",
    "TrendData",
]
