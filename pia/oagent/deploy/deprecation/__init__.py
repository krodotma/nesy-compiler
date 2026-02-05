#!/usr/bin/env python3
"""Deprecation manager module for deploy agent."""
from .manager import (
    DeprecationManager,
    DeprecatedItem,
    DeprecationPolicy,
    DeprecationNotice,
    SunsetSchedule,
    MigrationGuide,
)

__all__ = [
    "DeprecationManager",
    "DeprecatedItem",
    "DeprecationPolicy",
    "DeprecationNotice",
    "SunsetSchedule",
    "MigrationGuide",
]
