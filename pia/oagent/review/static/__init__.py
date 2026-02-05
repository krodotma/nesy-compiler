#!/usr/bin/env python3
"""
Static Analysis Engine Package (Step 152)

Provides static code analysis via external linters and internal checks.
"""

from .engine import (
    StaticAnalysisEngine,
    StaticAnalysisIssue,
    AnalysisSeverity,
    LanguageAnalyzer,
)

__all__ = [
    "StaticAnalysisEngine",
    "StaticAnalysisIssue",
    "AnalysisSeverity",
    "LanguageAnalyzer",
]
