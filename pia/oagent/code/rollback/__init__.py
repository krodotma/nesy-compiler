#!/usr/bin/env python3
"""
Rollback module - Safe edit rollback management.

Part of OAGENT 300-Step Plan: Step 64

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""

from .manager import (
    RollbackManager,
    RollbackPoint,
    RollbackResult,
    RollbackScope,
)

__all__ = [
    "RollbackManager",
    "RollbackPoint",
    "RollbackResult",
    "RollbackScope",
]
