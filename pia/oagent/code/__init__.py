#!/usr/bin/env python3
"""
Code Agent - Neural-Guided Code Generation and AST Manipulation

Part of OAGENT 300-Step Plan: Steps 51-60

This module provides:
- Bootstrap and lifecycle management (Step 51)
- Neural code proposal generation (Step 52)
- AST transformations for Python/TypeScript (Steps 53-55)
- Multi-file edit coordination (Step 56)
- PAIP isolation clone management (Step 57)
- Import dependency resolution (Step 58)
- Code style enforcement (Step 59)
- Incremental compilation (Step 60)

Protocol: DKIN v30, CITIZEN v2, PAIP v16
Bus Topics: a2a.code.*, code.*, paip.*
"""

from .bootstrap import CodeAgentBootstrap, CodeAgentConfig

__all__ = [
    "CodeAgentBootstrap",
    "CodeAgentConfig",
]

__version__ = "0.1.0"
