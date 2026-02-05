#!/usr/bin/env python3
"""
Compile subpackage for Code Agent.

Provides incremental compilation and error detection.
"""

from .incremental import IncrementalCompiler

__all__ = [
    "IncrementalCompiler",
]
