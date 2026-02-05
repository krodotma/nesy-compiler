#!/usr/bin/env python3
"""
Plugin System - Extensible Plugin Architecture (Step 81)

Provides plugin management and extension capabilities.
"""

from .plugin_system import (
    Plugin,
    PluginConfig,
    PluginContext,
    PluginHook,
    PluginInfo,
    PluginManager,
    PluginRegistry,
    PluginState,
    hook,
    main,
)

__all__ = [
    "Plugin",
    "PluginConfig",
    "PluginContext",
    "PluginHook",
    "PluginInfo",
    "PluginManager",
    "PluginRegistry",
    "PluginState",
    "hook",
    "main",
]
