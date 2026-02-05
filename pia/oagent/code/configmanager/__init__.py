#!/usr/bin/env python3
"""
Config Manager - Configuration Management (Step 86)

Provides configuration management capabilities.
"""

from .config_manager import (
    ConfigChangeEvent,
    ConfigManager,
    ConfigSchema,
    ConfigSource,
    ConfigValue,
    ConfigWatcher,
    main,
)

__all__ = [
    "ConfigChangeEvent",
    "ConfigManager",
    "ConfigSchema",
    "ConfigSource",
    "ConfigValue",
    "ConfigWatcher",
    "main",
]
