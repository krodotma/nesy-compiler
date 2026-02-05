#!/usr/bin/env python3
"""
plugin_system.py - Plugin System (Step 31)

Extensible plugin architecture for Research Agent.
Allows dynamic loading, unloading, and management of plugins.

PBTSO Phase: EXTEND

Bus Topics:
- a2a.research.plugin.load
- a2a.research.plugin.unload
- a2a.research.plugin.error
- research.plugin.register
- research.plugin.hook

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import abc
import fcntl
import importlib
import importlib.util
import inspect
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class PluginState(Enum):
    """Plugin lifecycle states."""
    REGISTERED = "registered"
    LOADED = "loaded"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class PluginHook(Enum):
    """Available plugin hooks."""
    # Lifecycle hooks
    ON_BOOTSTRAP = "on_bootstrap"
    ON_SHUTDOWN = "on_shutdown"

    # Query hooks
    PRE_QUERY = "pre_query"
    POST_QUERY = "post_query"

    # Search hooks
    PRE_SEARCH = "pre_search"
    POST_SEARCH = "post_search"

    # Index hooks
    PRE_INDEX = "pre_index"
    POST_INDEX = "post_index"

    # Cache hooks
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"

    # Analysis hooks
    PRE_ANALYZE = "pre_analyze"
    POST_ANALYZE = "post_analyze"

    # Custom hooks
    CUSTOM = "custom"


@dataclass
class PluginConfig:
    """Configuration for plugin system."""

    plugin_dirs: List[str] = field(default_factory=list)
    auto_discover: bool = True
    auto_load: bool = True
    sandbox_plugins: bool = True
    max_plugins: int = 50
    plugin_timeout_ms: int = 5000
    bus_path: Optional[str] = None

    def __post_init__(self):
        if not self.plugin_dirs:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.plugin_dirs = [
                f"{pluribus_root}/.pluribus/research/plugins",
                f"{pluribus_root}/pia/oagent/research/plugins",
            ]
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class PluginMetadata:
    """Plugin metadata from manifest."""

    name: str
    version: str
    description: str
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    min_agent_version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "hooks": self.hooks,
            "min_agent_version": self.min_agent_version,
        }


@dataclass
class PluginInstance:
    """A loaded plugin instance."""

    metadata: PluginMetadata
    plugin: "Plugin"
    state: PluginState
    path: str
    load_time: float = 0
    error: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "state": self.state.value,
            "path": self.path,
            "load_time": self.load_time,
            "error": self.error,
        }


# ============================================================================
# Plugin Base Class
# ============================================================================


T = TypeVar("T", bound="Plugin")


class Plugin(abc.ABC):
    """
    Base class for all Research Agent plugins.

    Plugins extend agent functionality by implementing hooks
    that are called at various points in the research pipeline.

    Example:
        class MyPlugin(Plugin):
            @classmethod
            def get_metadata(cls) -> PluginMetadata:
                return PluginMetadata(
                    name="my-plugin",
                    version="1.0.0",
                    description="My custom plugin",
                    hooks=["pre_query", "post_search"],
                )

            def activate(self) -> bool:
                self.log("Plugin activated!")
                return True

            def on_pre_query(self, context: Dict[str, Any]) -> Dict[str, Any]:
                # Modify query context
                return context
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin.

        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self._active = False
        self._manager: Optional["PluginManager"] = None

    @classmethod
    @abc.abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """
        Return plugin metadata.

        Must be implemented by all plugins.
        """
        pass

    def activate(self) -> bool:
        """
        Activate the plugin.

        Called when the plugin is loaded and should start operating.

        Returns:
            True if activation successful
        """
        self._active = True
        return True

    def deactivate(self) -> bool:
        """
        Deactivate the plugin.

        Called before the plugin is unloaded.

        Returns:
            True if deactivation successful
        """
        self._active = False
        return True

    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the plugin.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration successful
        """
        self.config.update(config)
        return True

    @property
    def is_active(self) -> bool:
        """Check if plugin is active."""
        return self._active

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message through the plugin manager.

        Args:
            message: Log message
            level: Log level (debug, info, warning, error)
        """
        if self._manager:
            self._manager._emit_event(
                "research.plugin.log",
                {
                    "plugin": self.get_metadata().name,
                    "message": message,
                    "level": level,
                }
            )

    # Hook methods - override in subclasses
    def on_bootstrap(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called during agent bootstrap."""
        return context

    def on_shutdown(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called during agent shutdown."""
        return context

    def on_pre_query(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called before query execution."""
        return context

    def on_post_query(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called after query execution."""
        return context

    def on_pre_search(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called before search."""
        return context

    def on_post_search(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called after search."""
        return context

    def on_pre_index(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called before indexing."""
        return context

    def on_post_index(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called after indexing."""
        return context

    def on_cache_hit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called on cache hit."""
        return context

    def on_cache_miss(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called on cache miss."""
        return context

    def on_pre_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called before analysis."""
        return context

    def on_post_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called after analysis."""
        return context


# ============================================================================
# Plugin Manager
# ============================================================================


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Features:
    - Dynamic plugin discovery from directories
    - Plugin loading and unloading
    - Hook invocation with timeout
    - Plugin dependency resolution
    - Configuration management

    PBTSO Phase: EXTEND

    Example:
        manager = PluginManager()
        manager.discover_plugins()
        manager.load_plugin("my-plugin")

        # Invoke hooks
        context = manager.invoke_hook(PluginHook.PRE_QUERY, {"query": "test"})
    """

    def __init__(
        self,
        config: Optional[PluginConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the plugin manager.

        Args:
            config: Plugin system configuration
            bus: AgentBus for event emission
        """
        self.config = config or PluginConfig()
        self.bus = bus or AgentBus()

        # Plugin storage
        self._plugins: Dict[str, PluginInstance] = {}
        self._hooks: Dict[PluginHook, List[str]] = {hook: [] for hook in PluginHook}
        self._discovered: Dict[str, Path] = {}

    def discover_plugins(self) -> List[str]:
        """
        Discover plugins from configured directories.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        for plugin_dir in self.config.plugin_dirs:
            dir_path = Path(plugin_dir)
            if not dir_path.exists():
                continue

            # Look for Python files and packages
            for item in dir_path.iterdir():
                if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                    # Single file plugin
                    plugin_name = item.stem
                    self._discovered[plugin_name] = item
                    discovered.append(plugin_name)

                elif item.is_dir() and (item / "__init__.py").exists():
                    # Package plugin
                    plugin_name = item.name
                    self._discovered[plugin_name] = item / "__init__.py"
                    discovered.append(plugin_name)

                elif item.is_dir() and (item / "plugin.py").exists():
                    # Plugin with plugin.py
                    plugin_name = item.name
                    self._discovered[plugin_name] = item / "plugin.py"
                    discovered.append(plugin_name)

        self._emit_event(
            "research.plugin.discover",
            {"discovered": discovered, "count": len(discovered)}
        )

        return discovered

    def load_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Load a discovered plugin.

        Args:
            name: Plugin name
            config: Optional plugin configuration

        Returns:
            True if plugin loaded successfully
        """
        if name in self._plugins:
            return True  # Already loaded

        if name not in self._discovered:
            self._emit_event(
                "a2a.research.plugin.error",
                {"plugin": name, "error": "Plugin not discovered"},
                level="error",
            )
            return False

        start_time = time.time()
        plugin_path = self._discovered[name]

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"research_plugin_{name}",
                plugin_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load plugin from {plugin_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Find Plugin subclass
            plugin_class: Optional[Type[Plugin]] = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr)
                    and issubclass(attr, Plugin)
                    and attr is not Plugin
                ):
                    plugin_class = attr
                    break

            if plugin_class is None:
                raise ImportError(f"No Plugin subclass found in {plugin_path}")

            # Get metadata
            metadata = plugin_class.get_metadata()

            # Check dependencies
            for dep in metadata.dependencies:
                if dep not in self._plugins:
                    raise ImportError(f"Missing dependency: {dep}")

            # Instantiate plugin
            plugin = plugin_class(config)
            plugin._manager = self

            # Activate
            if not plugin.activate():
                raise RuntimeError("Plugin activation failed")

            # Register hooks
            for hook_name in metadata.hooks:
                try:
                    hook = PluginHook(hook_name)
                    self._hooks[hook].append(name)
                except ValueError:
                    pass  # Invalid hook name

            # Store instance
            load_time = time.time() - start_time
            instance = PluginInstance(
                metadata=metadata,
                plugin=plugin,
                state=PluginState.ACTIVE,
                path=str(plugin_path),
                load_time=load_time,
                config=config or {},
            )
            self._plugins[name] = instance

            self._emit_event(
                "a2a.research.plugin.load",
                {
                    "plugin": name,
                    "version": metadata.version,
                    "hooks": metadata.hooks,
                    "load_time_ms": load_time * 1000,
                }
            )

            return True

        except Exception as e:
            self._emit_event(
                "a2a.research.plugin.error",
                {"plugin": name, "error": str(e)},
                level="error",
            )

            # Store as errored
            self._plugins[name] = PluginInstance(
                metadata=PluginMetadata(name=name, version="unknown", description=""),
                plugin=None,  # type: ignore
                state=PluginState.ERROR,
                path=str(plugin_path),
                error=str(e),
            )

            return False

    def unload_plugin(self, name: str) -> bool:
        """
        Unload a loaded plugin.

        Args:
            name: Plugin name

        Returns:
            True if plugin unloaded successfully
        """
        if name not in self._plugins:
            return False

        instance = self._plugins[name]

        try:
            if instance.plugin and instance.state == PluginState.ACTIVE:
                instance.plugin.deactivate()

            # Remove from hooks
            for hook_list in self._hooks.values():
                if name in hook_list:
                    hook_list.remove(name)

            del self._plugins[name]

            self._emit_event(
                "a2a.research.plugin.unload",
                {"plugin": name}
            )

            return True

        except Exception as e:
            self._emit_event(
                "a2a.research.plugin.error",
                {"plugin": name, "error": str(e), "action": "unload"},
                level="error",
            )
            return False

    def load_all_discovered(self) -> Dict[str, bool]:
        """
        Load all discovered plugins.

        Returns:
            Dictionary mapping plugin name to load success
        """
        results = {}

        # Sort by dependencies (simple topological sort)
        loading_order = list(self._discovered.keys())

        for name in loading_order:
            results[name] = self.load_plugin(name)

        return results

    def invoke_hook(
        self,
        hook: PluginHook,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke a hook on all registered plugins.

        Args:
            hook: Hook to invoke
            context: Context dictionary passed to plugins

        Returns:
            Modified context dictionary
        """
        plugin_names = self._hooks.get(hook, [])

        for name in plugin_names:
            instance = self._plugins.get(name)
            if not instance or instance.state != PluginState.ACTIVE:
                continue

            plugin = instance.plugin
            method_name = f"on_{hook.value}"

            if hasattr(plugin, method_name):
                try:
                    method = getattr(plugin, method_name)
                    context = method(context) or context
                except Exception as e:
                    self._emit_event(
                        "a2a.research.plugin.error",
                        {
                            "plugin": name,
                            "hook": hook.value,
                            "error": str(e),
                        },
                        level="error",
                    )

        return context

    def get_plugin(self, name: str) -> Optional[PluginInstance]:
        """Get a plugin instance by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[PluginInstance]:
        """List all loaded plugins."""
        return list(self._plugins.values())

    def get_plugins_for_hook(self, hook: PluginHook) -> List[str]:
        """Get plugin names registered for a hook."""
        return self._hooks.get(hook, [])

    def configure_plugin(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> bool:
        """
        Configure a loaded plugin.

        Args:
            name: Plugin name
            config: Configuration dictionary

        Returns:
            True if configuration successful
        """
        instance = self._plugins.get(name)
        if not instance or not instance.plugin:
            return False

        try:
            result = instance.plugin.configure(config)
            if result:
                instance.config.update(config)
            return result
        except Exception:
            return False

    def enable_plugin(self, name: str) -> bool:
        """Enable a disabled plugin."""
        instance = self._plugins.get(name)
        if not instance:
            return False

        if instance.state == PluginState.DISABLED:
            if instance.plugin.activate():
                instance.state = PluginState.ACTIVE
                return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """Disable an active plugin."""
        instance = self._plugins.get(name)
        if not instance:
            return False

        if instance.state == PluginState.ACTIVE:
            if instance.plugin.deactivate():
                instance.state = PluginState.DISABLED
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics."""
        active = sum(1 for p in self._plugins.values() if p.state == PluginState.ACTIVE)
        errored = sum(1 for p in self._plugins.values() if p.state == PluginState.ERROR)

        return {
            "discovered": len(self._discovered),
            "loaded": len(self._plugins),
            "active": active,
            "errored": errored,
            "hooks_registered": {
                hook.value: len(plugins)
                for hook, plugins in self._hooks.items()
                if plugins
            },
        }

    def _emit_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
    ) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "plugin",
            "level": level,
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Plugin System."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plugin System (Step 31)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover plugins")
    discover_parser.add_argument("--dir", action="append", help="Plugin directory")

    # List command
    list_parser = subparsers.add_parser("list", help="List loaded plugins")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load a plugin")
    load_parser.add_argument("name", help="Plugin name")

    # Unload command
    unload_parser = subparsers.add_parser("unload", help="Unload a plugin")
    unload_parser.add_argument("name", help="Plugin name")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = PluginConfig()
    if hasattr(args, "dir") and args.dir:
        config.plugin_dirs = args.dir

    manager = PluginManager(config)

    if args.command == "discover":
        discovered = manager.discover_plugins()
        print(f"Discovered {len(discovered)} plugins:")
        for name in discovered:
            print(f"  - {name}")

    elif args.command == "list":
        manager.discover_plugins()
        manager.load_all_discovered()
        plugins = manager.list_plugins()

        if args.json:
            print(json.dumps([p.to_dict() for p in plugins], indent=2))
        else:
            print(f"Loaded Plugins ({len(plugins)}):")
            for p in plugins:
                status = "OK" if p.state == PluginState.ACTIVE else p.state.value
                print(f"  [{status}] {p.metadata.name} v{p.metadata.version}")

    elif args.command == "load":
        manager.discover_plugins()
        if manager.load_plugin(args.name):
            print(f"Plugin '{args.name}' loaded successfully")
        else:
            print(f"Failed to load plugin '{args.name}'")
            return 1

    elif args.command == "unload":
        if manager.unload_plugin(args.name):
            print(f"Plugin '{args.name}' unloaded")
        else:
            print(f"Failed to unload plugin '{args.name}'")
            return 1

    elif args.command == "stats":
        stats = manager.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Plugin System Statistics:")
            print(f"  Discovered: {stats['discovered']}")
            print(f"  Loaded: {stats['loaded']}")
            print(f"  Active: {stats['active']}")
            print(f"  Errored: {stats['errored']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
