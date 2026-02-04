#!/usr/bin/env python3
"""
Dependency Vulnerability Scanner Package (Step 158)

Provides scanning of project dependencies for known vulnerabilities.
"""

from .vulnerability_scanner import (
    DependencyScanner,
    DependencyVulnerability,
    DependencyInfo,
    DependencyScanResult,
)

__all__ = [
    "DependencyScanner",
    "DependencyVulnerability",
    "DependencyInfo",
    "DependencyScanResult",
]
