#!/usr/bin/env python3
"""
PAIP subpackage for Code Agent.

Provides PAIP-isolated working copy management.
"""

from .clone_manager import PAIPCloneManager

__all__ = [
    "PAIPCloneManager",
]
