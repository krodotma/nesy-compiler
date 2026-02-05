#!/usr/bin/env python3
"""
Step 131: Test Plugin System

Extensible plugin architecture for the Test Agent.

PBTSO Phase: SKILL, BUILD
Bus Topics:
- test.plugin.load (emits)
- test.plugin.unload (emits)
- test.plugin.hook (emits)

Dependencies: Steps 101-130 (Test Components)
"""
from __future__ import annotations

import abc
import asyncio
import fcntl
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar

T = TypeVar('T')


# ============================================================================
# Constants
# ============================================================================

class PluginHook(Enum):
    """Plugin hook points."""
    PRE_TEST_RUN = "pre_test_run"
    POST_TEST_RUN = "post_test_run"
    PRE_TEST_CASE = "pre_test_case"
    POST_TEST_CASE = "post_test_case"
    ON_TEST_PASS = "on_test_pass"
    ON_TEST_FAIL = "on_test_fail"
    ON_TEST_SKIP = "on_test_skip"
    ON_COVERAGE_COLLECT = "on_coverage_collect"
    ON_REPORT_GENERATE = "on_report_generate"
    ON_ERROR = "on_error"
    CUSTOM = "custom"


class PluginState(Enum):
    """Plugin state."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class PluginPriority(Enum):
    """Plugin execution priority."""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class PluginInfo:
    """
    Information about a plugin.

    Attributes:
        name: Plugin name
        version: Plugin version
        description: Plugin description
        author: Plugin author
        hooks: Supported hooks
        dependencies: Required dependencies
        priority: Execution priority
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    hooks: List[PluginHook] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: PluginPriority = PluginPriority.NORMAL
    config_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "hooks": [h.value for h in self.hooks],
            "dependencies": self.dependencies,
            "priority": self.priority.value,
            "config_schema": self.config_schema,
        }


@dataclass
class PluginContext:
    """
    Context passed to plugin hooks.

    Attributes:
        hook: The hook being called
        data: Hook-specific data
        config: Plugin configuration
        shared: Shared data between plugins
        run_id: Current test run ID
        timestamp: Context creation time
    """
    hook: PluginHook
    data: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    shared: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def get(self, key: str, default: Any = None) -> Any:
        """Get data value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set data value."""
        self.data[key] = value

    def share(self, key: str, value: Any) -> None:
        """Share data with other plugins."""
        self.shared[key] = value


@dataclass
class PluginResult:
    """
    Result from a plugin hook execution.

    Attributes:
        success: Whether execution succeeded
        data: Result data
        error: Error message if failed
        modified_context: Modified context data
        duration_ms: Execution duration
    """
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    modified_context: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0
    skip_remaining: bool = False  # Skip remaining plugins for this hook

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "skip_remaining": self.skip_remaining,
        }


@dataclass
class PluginConfig:
    """
    Configuration for the plugin system.

    Attributes:
        plugin_dirs: Directories to search for plugins
        enabled_plugins: List of enabled plugin names
        disabled_plugins: List of disabled plugin names
        auto_discover: Auto-discover plugins
        fail_on_plugin_error: Fail test run on plugin error
        plugin_timeout_s: Timeout for plugin execution
        output_dir: Output directory
    """
    plugin_dirs: List[str] = field(default_factory=lambda: [
        ".pluribus/test-agent/plugins",
        "plugins",
    ])
    enabled_plugins: List[str] = field(default_factory=list)
    disabled_plugins: List[str] = field(default_factory=list)
    auto_discover: bool = True
    fail_on_plugin_error: bool = False
    plugin_timeout_s: float = 30.0
    output_dir: str = ".pluribus/test-agent/plugins"
    config_file: str = ".pluribus/test-agent/plugins.json"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plugin_dirs": self.plugin_dirs,
            "enabled_plugins": self.enabled_plugins,
            "disabled_plugins": self.disabled_plugins,
            "auto_discover": self.auto_discover,
            "fail_on_plugin_error": self.fail_on_plugin_error,
            "plugin_timeout_s": self.plugin_timeout_s,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class PluginBus:
    """Bus interface for plugins with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Plugin Base Class
# ============================================================================

class Plugin(abc.ABC):
    """
    Base class for Test Agent plugins.

    Subclass this to create a plugin:

    class MyPlugin(Plugin):
        @classmethod
        def get_info(cls) -> PluginInfo:
            return PluginInfo(
                name="my_plugin",
                version="1.0.0",
                hooks=[PluginHook.POST_TEST_RUN],
            )

        def on_post_test_run(self, context: PluginContext) -> PluginResult:
            # Do something after test run
            return PluginResult(success=True)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with configuration."""
        self.config = config or {}
        self._state = PluginState.UNLOADED
        self._load_time: Optional[float] = None
        self._execution_count = 0
        self._error_count = 0

    @classmethod
    @abc.abstractmethod
    def get_info(cls) -> PluginInfo:
        """Return plugin information."""
        pass

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self.get_info().name

    @property
    def state(self) -> PluginState:
        """Get plugin state."""
        return self._state

    def initialize(self) -> bool:
        """
        Initialize the plugin.

        Override this for custom initialization.
        Returns True if successful.
        """
        return True

    def shutdown(self) -> None:
        """
        Shutdown the plugin.

        Override this for custom cleanup.
        """
        pass

    def supports_hook(self, hook: PluginHook) -> bool:
        """Check if plugin supports a hook."""
        return hook in self.get_info().hooks

    def execute_hook(self, hook: PluginHook, context: PluginContext) -> PluginResult:
        """
        Execute a hook.

        Args:
            hook: Hook to execute
            context: Hook context

        Returns:
            PluginResult with execution outcome
        """
        if not self.supports_hook(hook):
            return PluginResult(success=True)

        start_time = time.time()
        self._execution_count += 1

        try:
            handler = self._get_hook_handler(hook)
            if handler:
                result = handler(context)
                result.duration_ms = (time.time() - start_time) * 1000
                return result
            return PluginResult(success=True)

        except Exception as e:
            self._error_count += 1
            return PluginResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _get_hook_handler(self, hook: PluginHook) -> Optional[Callable]:
        """Get handler method for a hook."""
        handler_name = f"on_{hook.value}"
        return getattr(self, handler_name, None)

    # Hook methods - override in subclass
    def on_pre_test_run(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_post_test_run(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_pre_test_case(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_post_test_case(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_test_pass(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_test_fail(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_test_skip(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_coverage_collect(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_report_generate(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)

    def on_error(self, context: PluginContext) -> PluginResult:
        return PluginResult(success=True)


# ============================================================================
# Plugin Manager
# ============================================================================

class TestPluginManager:
    """
    Manages Test Agent plugins.

    Features:
    - Plugin discovery and loading
    - Hook execution
    - Plugin lifecycle management
    - Configuration management

    PBTSO Phase: SKILL, BUILD
    Bus Topics: test.plugin.load, test.plugin.unload, test.plugin.hook
    """

    BUS_TOPICS = {
        "load": "test.plugin.load",
        "unload": "test.plugin.unload",
        "hook": "test.plugin.hook",
        "error": "test.plugin.error",
    }

    def __init__(self, bus=None, config: Optional[PluginConfig] = None):
        """
        Initialize the plugin manager.

        Args:
            bus: Optional bus instance
            config: Plugin configuration
        """
        self.bus = bus or PluginBus()
        self.config = config or PluginConfig()
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self._hook_registry: Dict[PluginHook, List[str]] = {
            hook: [] for hook in PluginHook
        }
        self._shared_context: Dict[str, Any] = {}

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def discover_plugins(self) -> List[PluginInfo]:
        """
        Discover available plugins.

        Returns:
            List of discovered plugin infos
        """
        discovered = []

        for plugin_dir in self.config.plugin_dirs:
            dir_path = Path(plugin_dir)
            if not dir_path.exists():
                continue

            for plugin_file in dir_path.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                try:
                    plugin_class = self._load_plugin_class(plugin_file)
                    if plugin_class:
                        info = plugin_class.get_info()
                        discovered.append(info)
                        self._plugin_classes[info.name] = plugin_class
                except Exception as e:
                    self._emit_event("error", {
                        "plugin_file": str(plugin_file),
                        "error": str(e),
                    })

        return discovered

    def _load_plugin_class(self, plugin_file: Path) -> Optional[Type[Plugin]]:
        """Load plugin class from file."""
        module_name = plugin_file.stem
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find Plugin subclass
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, Plugin) and
                obj is not Plugin):
                return obj

        return None

    def load_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Load a plugin by name.

        Args:
            name: Plugin name
            config: Plugin configuration

        Returns:
            True if loaded successfully
        """
        if name in self._plugins:
            return True

        if name in self.config.disabled_plugins:
            return False

        plugin_class = self._plugin_classes.get(name)
        if plugin_class is None:
            # Try to discover
            self.discover_plugins()
            plugin_class = self._plugin_classes.get(name)

        if plugin_class is None:
            return False

        try:
            plugin = plugin_class(config=config)
            plugin._state = PluginState.LOADING

            # Check dependencies
            info = plugin.get_info()
            for dep in info.dependencies:
                if dep not in self._plugins:
                    if not self.load_plugin(dep):
                        plugin._state = PluginState.ERROR
                        return False

            # Initialize
            if not plugin.initialize():
                plugin._state = PluginState.ERROR
                return False

            plugin._state = PluginState.LOADED
            plugin._load_time = time.time()

            # Register
            self._plugins[name] = plugin

            # Register hooks
            for hook in info.hooks:
                if name not in self._hook_registry[hook]:
                    self._hook_registry[hook].append(name)

            # Sort by priority
            self._sort_hook_registry()

            self._emit_event("load", {
                "plugin": name,
                "version": info.version,
                "hooks": [h.value for h in info.hooks],
            })

            return True

        except Exception as e:
            self._emit_event("error", {
                "plugin": name,
                "error": str(e),
            })
            return False

    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if unloaded successfully
        """
        if name not in self._plugins:
            return False

        plugin = self._plugins[name]

        try:
            plugin.shutdown()
        except Exception:
            pass

        plugin._state = PluginState.UNLOADED

        # Unregister hooks
        for hook in PluginHook:
            if name in self._hook_registry[hook]:
                self._hook_registry[hook].remove(name)

        del self._plugins[name]

        self._emit_event("unload", {
            "plugin": name,
        })

        return True

    def load_all(self) -> int:
        """
        Load all enabled plugins.

        Returns:
            Number of plugins loaded
        """
        if self.config.auto_discover:
            self.discover_plugins()

        loaded = 0
        plugins_to_load = self.config.enabled_plugins or list(self._plugin_classes.keys())

        for name in plugins_to_load:
            if name not in self.config.disabled_plugins:
                if self.load_plugin(name):
                    loaded += 1

        return loaded

    def unload_all(self) -> int:
        """
        Unload all plugins.

        Returns:
            Number of plugins unloaded
        """
        unloaded = 0
        for name in list(self._plugins.keys()):
            if self.unload_plugin(name):
                unloaded += 1
        return unloaded

    def execute_hook(
        self,
        hook: PluginHook,
        data: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> List[PluginResult]:
        """
        Execute a hook across all plugins.

        Args:
            hook: Hook to execute
            data: Hook data
            run_id: Test run ID

        Returns:
            List of plugin results
        """
        results = []
        context = PluginContext(
            hook=hook,
            data=data or {},
            shared=self._shared_context,
            run_id=run_id,
        )

        for plugin_name in self._hook_registry[hook]:
            plugin = self._plugins.get(plugin_name)
            if plugin is None or plugin.state not in (PluginState.LOADED, PluginState.ACTIVE):
                continue

            try:
                plugin._state = PluginState.ACTIVE
                result = plugin.execute_hook(hook, context)
                results.append(result)

                # Update shared context
                self._shared_context.update(result.modified_context)

                self._emit_event("hook", {
                    "plugin": plugin_name,
                    "hook": hook.value,
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                })

                if result.skip_remaining:
                    break

                if not result.success and self.config.fail_on_plugin_error:
                    break

            except Exception as e:
                result = PluginResult(success=False, error=str(e))
                results.append(result)

                self._emit_event("error", {
                    "plugin": plugin_name,
                    "hook": hook.value,
                    "error": str(e),
                })

            finally:
                plugin._state = PluginState.LOADED

        return results

    async def execute_hook_async(
        self,
        hook: PluginHook,
        data: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> List[PluginResult]:
        """Async version of execute_hook."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.execute_hook, hook, data, run_id
        )

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return [p.get_info() for p in self._plugins.values()]

    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        stats = {
            "loaded_plugins": len(self._plugins),
            "available_plugins": len(self._plugin_classes),
            "plugins": {},
        }

        for name, plugin in self._plugins.items():
            stats["plugins"][name] = {
                "state": plugin.state.value,
                "executions": plugin._execution_count,
                "errors": plugin._error_count,
                "load_time": plugin._load_time,
            }

        return stats

    def _sort_hook_registry(self) -> None:
        """Sort hook registry by plugin priority."""
        for hook in PluginHook:
            self._hook_registry[hook].sort(
                key=lambda name: self._plugins[name].get_info().priority.value
                if name in self._plugins else 100
            )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.plugin.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "plugin",
            "actor": "test-agent",
            "data": data,
        })

    def save_config(self) -> None:
        """Save plugin configuration."""
        config_data = {
            "enabled_plugins": list(self._plugins.keys()),
            "disabled_plugins": self.config.disabled_plugins,
            "plugin_configs": {},
        }

        for name, plugin in self._plugins.items():
            config_data["plugin_configs"][name] = plugin.config

        config_file = Path(self.config.config_file)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(config_data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load_config(self) -> bool:
        """Load plugin configuration."""
        config_file = Path(self.config.config_file)
        if not config_file.exists():
            return False

        try:
            with open(config_file) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    config_data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            self.config.enabled_plugins = config_data.get("enabled_plugins", [])
            self.config.disabled_plugins = config_data.get("disabled_plugins", [])

            # Load plugins with saved configs
            for name, plugin_config in config_data.get("plugin_configs", {}).items():
                self.load_plugin(name, config=plugin_config)

            return True

        except (json.JSONDecodeError, IOError):
            return False


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Plugin System."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Plugin Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List plugins")
    list_parser.add_argument("--all", action="store_true", help="Show all discovered plugins")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load a plugin")
    load_parser.add_argument("name", help="Plugin name")

    # Unload command
    unload_parser = subparsers.add_parser("unload", help="Unload a plugin")
    unload_parser.add_argument("name", help="Plugin name")

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover plugins")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show plugin statistics")

    # Common arguments
    parser.add_argument("--plugin-dir", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = PluginConfig()
    if args.plugin_dir:
        config.plugin_dirs.extend(args.plugin_dir)

    manager = TestPluginManager(config=config)

    if args.command == "discover":
        plugins = manager.discover_plugins()

        if args.json:
            print(json.dumps([p.to_dict() for p in plugins], indent=2))
        else:
            print(f"\nDiscovered {len(plugins)} plugins:")
            for plugin in plugins:
                hooks = ", ".join(h.value for h in plugin.hooks)
                print(f"  {plugin.name} v{plugin.version}")
                print(f"    Hooks: {hooks}")

    elif args.command == "list":
        if args.all:
            manager.discover_plugins()
            plugins = list(manager._plugin_classes.keys())
        else:
            plugins = list(manager._plugins.keys())

        if args.json:
            print(json.dumps(plugins, indent=2))
        else:
            print(f"\nPlugins ({len(plugins)}):")
            for name in plugins:
                state = manager._plugins[name].state.value if name in manager._plugins else "unloaded"
                print(f"  [{state.upper()}] {name}")

    elif args.command == "load":
        if manager.load_plugin(args.name):
            print(f"Loaded: {args.name}")
        else:
            print(f"Failed to load: {args.name}")
            exit(1)

    elif args.command == "unload":
        if manager.unload_plugin(args.name):
            print(f"Unloaded: {args.name}")
        else:
            print(f"Failed to unload: {args.name}")
            exit(1)

    elif args.command == "stats":
        stats = manager.get_plugin_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("\nPlugin Statistics:")
            print(f"  Loaded: {stats['loaded_plugins']}")
            print(f"  Available: {stats['available_plugins']}")

            if stats['plugins']:
                print("\n  Plugins:")
                for name, info in stats['plugins'].items():
                    print(f"    {name}: {info['executions']} executions, {info['errors']} errors")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
