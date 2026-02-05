#!/usr/bin/env python3
"""
Regression Detection Module - Step 116

Provides regression identification capabilities.

Components:
- RegressionDetector: Detects test regressions
- RegressionAnalyzer: Analyzes regression patterns
- RegressionReporter: Reports regressions

Bus Topics:
- test.regression.detect
- test.regression.found
- a2a.review.alert
"""

from .detector import (
    RegressionDetector,
    RegressionConfig,
    RegressionResult,
    Regression,
    RegressionType,
)

__all__ = [
    "RegressionDetector",
    "RegressionConfig",
    "RegressionResult",
    "Regression",
    "RegressionType",
]
