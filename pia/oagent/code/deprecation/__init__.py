#!/usr/bin/env python3
"""
Deprecation Module (Step 99) - Deprecation Handling

Provides deprecation management capabilities for the Code Agent.
"""

from .deprecation_manager import (
    DeprecationManager,
    DeprecationConfig,
    Deprecation,
    DeprecationWarning,
    DeprecationLevel,
    deprecated,
)

__all__ = [
    "DeprecationManager",
    "DeprecationConfig",
    "Deprecation",
    "DeprecationWarning",
    "DeprecationLevel",
    "deprecated",
]
