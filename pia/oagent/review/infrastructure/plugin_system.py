#!/usr/bin/env python3
"""
Review Plugin System (Step 181)

Extensible plugin architecture for the Review Agent. Plugins can hook into
various stages of the review pipeline to add custom analyzers, formatters,
or integrations.

PBTSO Phase: SKILL, BUILD
Bus Topics: review.plugin.load, review.plugin.unload, review.plugin.execute

Plugin Hooks:
- pre_review: Before review starts
- post_analyze: After analysis completes
- pre_comment: Before comments are generated
- post_review: After review completes
- on_error: When an error occurs

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import abc
import asyncio
import fcntl
import importlib
import importlib.util
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Generic

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300  # 5 minutes
A2A_HEARTBEAT_TIMEOUT = 900   # 15 minutes
PLUGIN_API_VERSION = "1.0.0"


# ============================================================================
# Types
# ============================================================================

class PluginHook(Enum):
    """Hook points in the review pipeline."""
    PRE_REVIEW = "pre_review"
    POST_ANALYZE = "post_analyze"
    PRE_COMMENT = "pre_comment"
    POST_REVIEW = "post_review"
    ON_ERROR = "on_error"
    ON_VETO = "on_veto"
    CUSTOM = "custom"


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class PluginPriority(Enum):
    """Plugin execution priority."""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class PluginInfo:
    """
    Plugin metadata.

    Attributes:
        name: Plugin unique name
        version: Plugin version
        description: Plugin description
        author: Plugin author
        api_version: Required API version
        hooks: Hooks this plugin registers
        dependencies: Required plugin dependencies
        config_schema: Configuration schema
    """
    name: str
    version: str
    description: str = ""
    author: str = ""
    api_version: str = PLUGIN_API_VERSION
    hooks: List[PluginHook] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "hooks": [h.value for h in self.hooks],
        }


@dataclass
class PluginConfig:
    """
    Plugin system configuration.

    Attributes:
        plugin_dir: Directory containing plugins
        auto_discover: Auto-discover plugins on startup
        enabled_plugins: List of enabled plugin names
        disabled_plugins: List of disabled plugin names
        sandbox_enabled: Run plugins in sandbox
        timeout_seconds: Plugin execution timeout
        max_plugins: Maximum number of loaded plugins
    """
    plugin_dir: str = ""
    auto_discover: bool = True
    enabled_plugins: List[str] = field(default_factory=list)
    disabled_plugins: List[str] = field(default_factory=list)
    sandbox_enabled: bool = True
    timeout_seconds: int = 30
    max_plugins: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HookContext:
    """
    Context passed to plugin hooks.

    Attributes:
        hook: The hook being executed
        review_id: Current review ID
        files: Files being reviewed
        results: Current results (mutable)
        metadata: Additional metadata
    """
    hook: PluginHook
    review_id: str
    files: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hook": self.hook.value,
            "review_id": self.review_id,
            "files": self.files,
            "metadata": self.metadata,
        }


@dataclass
class HookResult:
    """
    Result from a plugin hook execution.

    Attributes:
        plugin_name: Name of the plugin
        hook: Hook that was executed
        success: Whether execution succeeded
        modified: Whether results were modified
        data: Result data
        error: Error message if failed
        duration_ms: Execution duration
    """
    plugin_name: str
    hook: PluginHook
    success: bool
    modified: bool = False
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plugin_name": self.plugin_name,
            "hook": self.hook.value,
            "success": self.success,
            "modified": self.modified,
            "data": self.data,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


# ============================================================================
# Plugin Base Class
# ============================================================================

class ReviewPlugin(abc.ABC):
    """
    Base class for review plugins.

    Plugins must implement get_info() and can override hook methods.

    Example:
        class MyPlugin(ReviewPlugin):
            def get_info(self) -> PluginInfo:
                return PluginInfo(
                    name="my-plugin",
                    version="1.0.0",
                    description="My custom plugin",
                    hooks=[PluginHook.POST_ANALYZE],
                )

            async def on_post_analyze(self, context: HookContext) -> HookResult:
                # Add custom analysis
                context.results["my_metric"] = compute_metric(context.files)
                return HookResult(
                    plugin_name=self.info.name,
                    hook=PluginHook.POST_ANALYZE,
                    success=True,
                    modified=True,
                )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin.

        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self._info: Optional[PluginInfo] = None
        self._state = PluginState.UNLOADED

    @property
    def info(self) -> PluginInfo:
        """Get plugin info (cached)."""
        if self._info is None:
            self._info = self.get_info()
        return self._info

    @property
    def state(self) -> PluginState:
        """Get plugin state."""
        return self._state

    @abc.abstractmethod
    def get_info(self) -> PluginInfo:
        """
        Get plugin metadata.

        Returns:
            PluginInfo with plugin metadata
        """
        pass

    async def initialize(self) -> bool:
        """
        Initialize the plugin.

        Called once when the plugin is loaded.

        Returns:
            True if initialization succeeded
        """
        return True

    async def shutdown(self) -> None:
        """
        Shutdown the plugin.

        Called when the plugin is unloaded.
        """
        pass

    async def on_pre_review(self, context: HookContext) -> HookResult:
        """Hook: Before review starts."""
        return HookResult(
            plugin_name=self.info.name,
            hook=PluginHook.PRE_REVIEW,
            success=True,
        )

    async def on_post_analyze(self, context: HookContext) -> HookResult:
        """Hook: After analysis completes."""
        return HookResult(
            plugin_name=self.info.name,
            hook=PluginHook.POST_ANALYZE,
            success=True,
        )

    async def on_pre_comment(self, context: HookContext) -> HookResult:
        """Hook: Before comments are generated."""
        return HookResult(
            plugin_name=self.info.name,
            hook=PluginHook.PRE_COMMENT,
            success=True,
        )

    async def on_post_review(self, context: HookContext) -> HookResult:
        """Hook: After review completes."""
        return HookResult(
            plugin_name=self.info.name,
            hook=PluginHook.POST_REVIEW,
            success=True,
        )

    async def on_error(self, context: HookContext) -> HookResult:
        """Hook: When an error occurs."""
        return HookResult(
            plugin_name=self.info.name,
            hook=PluginHook.ON_ERROR,
            success=True,
        )

    async def on_veto(self, context: HookContext) -> HookResult:
        """Hook: When Omega veto is requested."""
        return HookResult(
            plugin_name=self.info.name,
            hook=PluginHook.ON_VETO,
            success=True,
        )


# ============================================================================
# Plugin Manager
# ============================================================================

class PluginManager:
    """
    Manages review plugins.

    Handles plugin discovery, loading, lifecycle, and hook execution.

    Example:
        config = PluginConfig(plugin_dir="/path/to/plugins")
        manager = PluginManager(config)

        # Load all plugins
        await manager.discover_and_load()

        # Execute hooks
        context = HookContext(
            hook=PluginHook.PRE_REVIEW,
            review_id="abc123",
            files=["file.py"],
        )
        results = await manager.execute_hook(PluginHook.PRE_REVIEW, context)
    """

    BUS_TOPICS = {
        "load": "review.plugin.load",
        "unload": "review.plugin.unload",
        "execute": "review.plugin.execute",
        "error": "review.plugin.error",
    }

    def __init__(
        self,
        config: Optional[PluginConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the plugin manager.

        Args:
            config: Plugin system configuration
            bus_path: Path to event bus file
        """
        self.config = config or PluginConfig()
        self.bus_path = bus_path or self._get_bus_path()
        self._plugins: Dict[str, ReviewPlugin] = {}
        self._hook_registry: Dict[PluginHook, List[str]] = {h: [] for h in PluginHook}
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "plugin") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "plugin-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    async def discover_plugins(self) -> List[str]:
        """
        Discover plugins in the plugin directory.

        Returns:
            List of discovered plugin paths
        """
        if not self.config.plugin_dir:
            return []

        plugin_dir = Path(self.config.plugin_dir)
        if not plugin_dir.exists():
            return []

        discovered = []
        for path in plugin_dir.glob("*.py"):
            if path.name.startswith("_"):
                continue
            discovered.append(str(path))

        for path in plugin_dir.glob("*/plugin.py"):
            discovered.append(str(path))

        return discovered

    async def load_plugin(
        self,
        plugin: ReviewPlugin,
        plugin_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Load a plugin instance.

        Args:
            plugin: Plugin instance to load
            plugin_config: Plugin-specific configuration

        Returns:
            True if loading succeeded

        Emits:
            review.plugin.load
        """
        info = plugin.info

        # Check if already loaded
        if info.name in self._plugins:
            return False

        # Check max plugins
        if len(self._plugins) >= self.config.max_plugins:
            return False

        # Check if disabled
        if info.name in self.config.disabled_plugins:
            return False

        # Check dependencies
        for dep in info.dependencies:
            if dep not in self._plugins:
                self._emit_event(self.BUS_TOPICS["error"], {
                    "plugin": info.name,
                    "error": f"Missing dependency: {dep}",
                })
                return False

        # Apply config
        if plugin_config:
            plugin.config.update(plugin_config)

        # Initialize
        plugin._state = PluginState.LOADING
        try:
            success = await asyncio.wait_for(
                plugin.initialize(),
                timeout=self.config.timeout_seconds,
            )
            if not success:
                plugin._state = PluginState.ERROR
                return False
        except Exception as e:
            plugin._state = PluginState.ERROR
            self._emit_event(self.BUS_TOPICS["error"], {
                "plugin": info.name,
                "error": str(e),
            })
            return False

        # Register plugin
        plugin._state = PluginState.ACTIVE
        self._plugins[info.name] = plugin

        # Register hooks
        for hook in info.hooks:
            self._hook_registry[hook].append(info.name)

        self._emit_event(self.BUS_TOPICS["load"], {
            "plugin": info.name,
            "version": info.version,
            "hooks": [h.value for h in info.hooks],
            "status": "loaded",
        })

        return True

    async def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin by name.

        Args:
            name: Plugin name to unload

        Returns:
            True if unloading succeeded

        Emits:
            review.plugin.unload
        """
        if name not in self._plugins:
            return False

        plugin = self._plugins[name]

        # Shutdown
        try:
            await asyncio.wait_for(
                plugin.shutdown(),
                timeout=self.config.timeout_seconds,
            )
        except Exception:
            pass

        # Unregister hooks
        for hook in plugin.info.hooks:
            if name in self._hook_registry[hook]:
                self._hook_registry[hook].remove(name)

        # Remove plugin
        plugin._state = PluginState.UNLOADED
        del self._plugins[name]

        self._emit_event(self.BUS_TOPICS["unload"], {
            "plugin": name,
            "status": "unloaded",
        })

        return True

    async def load_plugin_from_path(self, path: str) -> bool:
        """
        Load a plugin from a file path.

        Args:
            path: Path to plugin file

        Returns:
            True if loading succeeded
        """
        try:
            spec = importlib.util.spec_from_file_location("plugin", path)
            if not spec or not spec.loader:
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, ReviewPlugin)
                    and attr is not ReviewPlugin
                ):
                    plugin = attr()
                    return await self.load_plugin(plugin)

            return False
        except Exception as e:
            self._emit_event(self.BUS_TOPICS["error"], {
                "path": path,
                "error": str(e),
            })
            return False

    async def discover_and_load(self) -> Dict[str, bool]:
        """
        Discover and load all plugins.

        Returns:
            Dict mapping plugin paths to load success
        """
        results = {}
        paths = await self.discover_plugins()

        for path in paths:
            results[path] = await self.load_plugin_from_path(path)

        return results

    async def execute_hook(
        self,
        hook: PluginHook,
        context: HookContext,
    ) -> List[HookResult]:
        """
        Execute all plugins registered for a hook.

        Args:
            hook: Hook to execute
            context: Hook context

        Returns:
            List of hook results

        Emits:
            review.plugin.execute
        """
        results = []
        plugin_names = self._hook_registry.get(hook, [])

        for name in plugin_names:
            plugin = self._plugins.get(name)
            if not plugin or plugin.state != PluginState.ACTIVE:
                continue

            start_time = time.time()
            try:
                # Get hook method
                hook_method = {
                    PluginHook.PRE_REVIEW: plugin.on_pre_review,
                    PluginHook.POST_ANALYZE: plugin.on_post_analyze,
                    PluginHook.PRE_COMMENT: plugin.on_pre_comment,
                    PluginHook.POST_REVIEW: plugin.on_post_review,
                    PluginHook.ON_ERROR: plugin.on_error,
                    PluginHook.ON_VETO: plugin.on_veto,
                }.get(hook)

                if hook_method:
                    result = await asyncio.wait_for(
                        hook_method(context),
                        timeout=self.config.timeout_seconds,
                    )
                    result.duration_ms = (time.time() - start_time) * 1000
                    results.append(result)

            except asyncio.TimeoutError:
                results.append(HookResult(
                    plugin_name=name,
                    hook=hook,
                    success=False,
                    error="Execution timeout",
                    duration_ms=(time.time() - start_time) * 1000,
                ))
            except Exception as e:
                results.append(HookResult(
                    plugin_name=name,
                    hook=hook,
                    success=False,
                    error=str(e),
                    duration_ms=(time.time() - start_time) * 1000,
                ))

        self._emit_event(self.BUS_TOPICS["execute"], {
            "hook": hook.value,
            "review_id": context.review_id,
            "plugins_executed": len(results),
            "successful": sum(1 for r in results if r.success),
        })

        return results

    def get_plugin(self, name: str) -> Optional[ReviewPlugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_loaded_plugins(self) -> List[PluginInfo]:
        """Get info for all loaded plugins."""
        return [p.info for p in self._plugins.values()]

    def is_plugin_active(self, name: str) -> bool:
        """Check if a plugin is active."""
        plugin = self._plugins.get(name)
        return plugin is not None and plugin.state == PluginState.ACTIVE

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "plugin-manager",
            "healthy": True,
            "plugins_loaded": len(self._plugins),
            "plugins_active": sum(1 for p in self._plugins.values() if p.state == PluginState.ACTIVE),
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Plugin System."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Plugin System (Step 181)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List loaded plugins")

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover plugins")
    discover_parser.add_argument("--dir", help="Plugin directory")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get plugin info")
    info_parser.add_argument("name", help="Plugin name")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = PluginConfig(
        plugin_dir=args.dir if hasattr(args, "dir") and args.dir else "",
    )
    manager = PluginManager(config)

    if args.command == "list":
        plugins = manager.get_loaded_plugins()
        if args.json:
            print(json.dumps([p.to_dict() for p in plugins], indent=2))
        else:
            print(f"Loaded Plugins: {len(plugins)}")
            for p in plugins:
                print(f"  {p.name} v{p.version}")
                print(f"    Hooks: {', '.join(h.value for h in p.hooks)}")

    elif args.command == "discover":
        paths = asyncio.run(manager.discover_plugins())
        if args.json:
            print(json.dumps({"discovered": paths}, indent=2))
        else:
            print(f"Discovered Plugins: {len(paths)}")
            for p in paths:
                print(f"  {p}")

    elif args.command == "info":
        plugin = manager.get_plugin(args.name)
        if plugin:
            if args.json:
                print(json.dumps(plugin.info.to_dict(), indent=2))
            else:
                info = plugin.info
                print(f"Plugin: {info.name}")
                print(f"  Version: {info.version}")
                print(f"  Description: {info.description}")
                print(f"  Author: {info.author}")
                print(f"  Hooks: {', '.join(h.value for h in info.hooks)}")
        else:
            print(f"Plugin not found: {args.name}")
            return 1

    else:
        # Default: show status
        status = manager.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("Plugin Manager Status")
            print(f"  Plugins Loaded: {status['plugins_loaded']}")
            print(f"  Plugins Active: {status['plugins_active']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
