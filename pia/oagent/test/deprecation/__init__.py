#!/usr/bin/env python3
"""
Step 149: Test Deprecation Manager

Deprecation handling for the Test Agent.
"""

from .deprecation import (
    TestDeprecationManager,
    DeprecationConfig,
    DeprecationNotice,
    DeprecationPolicy,
    DeprecationStatus,
)

__all__ = [
    "TestDeprecationManager",
    "DeprecationConfig",
    "DeprecationNotice",
    "DeprecationPolicy",
    "DeprecationStatus",
]
