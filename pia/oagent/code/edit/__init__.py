#!/usr/bin/env python3
"""
Edit subpackage for Code Agent.

Provides multi-file edit coordination with atomic operations.
"""

from .coordinator import MultiFileEditCoordinator, FileEdit

__all__ = [
    "MultiFileEditCoordinator",
    "FileEdit",
]
