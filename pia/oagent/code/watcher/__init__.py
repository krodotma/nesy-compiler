#!/usr/bin/env python3
"""
Watcher module - Real-time file change detection.

Part of OAGENT 300-Step Plan: Step 69

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""

from .file_watcher import (
    FileWatcher,
    FileEvent,
    EventType,
    WatchConfig,
)

__all__ = [
    "FileWatcher",
    "FileEvent",
    "EventType",
    "WatchConfig",
]
