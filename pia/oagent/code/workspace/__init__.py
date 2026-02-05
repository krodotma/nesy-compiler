#!/usr/bin/env python3
"""
Workspace module - Multi-agent workspace coordination.

Part of OAGENT 300-Step Plan: Step 68

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""

from .sync import (
    WorkspaceSync,
    WorkspaceLock,
    SyncState,
    FileChange,
)

__all__ = [
    "WorkspaceSync",
    "WorkspaceLock",
    "SyncState",
    "FileChange",
]
