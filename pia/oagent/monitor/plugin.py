#!/usr/bin/env python3
"""
Monitor Plugin System - Step 281

Extensible plugin architecture for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.plugin.loaded (emitted)
- monitor.plugin.unloaded (emitted)
- monitor.plugin.error (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import abc
import asyncio
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
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar


class PluginState(Enum):
    """Plugin lifecycle states."""
    REGISTERED = "registered"
    LOADING = "loading"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    UNLOADING = "unloading"


class PluginType(Enum):
    """Types of monitor plugins."""
    COLLECTOR = "collector"        # Metric collection plugins
    PROCESSOR = "processor"        # Metric processing plugins
    ALERTER = "alerter"           # Alert generation plugins
    NOTIFIER = "notifier"         # Notification delivery plugins
    EXPORTER = "exporter"         # Data export plugins
    ANALYZER = "analyzer"         # Analysis plugins
    VISUALIZER = "visualizer"     # Visualization plugins
    INTEGRATION = "integration"   # External integration plugins


@dataclass
class PluginMetadata:
    """Plugin metadata.

    Attributes:
        name: Plugin name
        version: Plugin version
        plugin_type: Type of plugin
        description: Plugin description
        author: Plugin author
        dependencies: Required dependencies
        config_schema: Configuration schema
    """
    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type.value,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "config_schema": self.config_schema,
        }


@dataclass
class PluginInstance:
    """A loaded plugin instance.

    Attributes:
        metadata: Plugin metadata
        state: Current state
        instance: Plugin instance
        config: Plugin configuration
        loaded_at: Load timestamp
        error: Error message if any
    """
    metadata: PluginMetadata
    state: PluginState = PluginState.REGISTERED
    instance: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state.value,
            "config": self.config,
            "loaded_at": self.loaded_at,
            "error": self.error,
        }


class MonitorPlugin(abc.ABC):
    """
    Base class for all monitor plugins.

    Plugins must implement:
    - metadata: Plugin metadata
    - initialize: Setup method
    - shutdown: Cleanup method

    Example:
        class MyPlugin(MonitorPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="my-plugin",
                    version="1.0.0",
                    plugin_type=PluginType.COLLECTOR,
                )

            async def initialize(self, config: Dict[str, Any]) -> None:
                # Setup code
                pass

            async def shutdown(self) -> None:
                # Cleanup code
                pass
    """

    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin.

        Args:
            config: Plugin configuration
        """
        pass

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Perform plugin health check.

        Returns:
            Health status
        """
        return {"healthy": True}


class CollectorPlugin(MonitorPlugin):
    """Base class for metric collector plugins."""

    @abc.abstractmethod
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect metrics.

        Returns:
            List of metric dictionaries
        """
        pass


class ProcessorPlugin(MonitorPlugin):
    """Base class for metric processor plugins."""

    @abc.abstractmethod
    async def process(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process metrics.

        Args:
            metrics: Input metrics

        Returns:
            Processed metrics
        """
        pass


class AlerterPlugin(MonitorPlugin):
    """Base class for alerter plugins."""

    @abc.abstractmethod
    async def evaluate(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate metrics for alerts.

        Args:
            metrics: Current metrics

        Returns:
            List of alerts
        """
        pass


class NotifierPlugin(MonitorPlugin):
    """Base class for notifier plugins."""

    @abc.abstractmethod
    async def notify(self, alert: Dict[str, Any]) -> bool:
        """Send notification.

        Args:
            alert: Alert to notify

        Returns:
            True if sent successfully
        """
        pass


class ExporterPlugin(MonitorPlugin):
    """Base class for exporter plugins."""

    @abc.abstractmethod
    async def export(self, data: Dict[str, Any], destination: str) -> bool:
        """Export data.

        Args:
            data: Data to export
            destination: Export destination

        Returns:
            True if exported successfully
        """
        pass


# Plugin type to base class mapping
PLUGIN_BASES: Dict[PluginType, Type[MonitorPlugin]] = {
    PluginType.COLLECTOR: CollectorPlugin,
    PluginType.PROCESSOR: ProcessorPlugin,
    PluginType.ALERTER: AlerterPlugin,
    PluginType.NOTIFIER: NotifierPlugin,
    PluginType.EXPORTER: ExporterPlugin,
    PluginType.ANALYZER: MonitorPlugin,
    PluginType.VISUALIZER: MonitorPlugin,
    PluginType.INTEGRATION: MonitorPlugin,
}


T = TypeVar("T", bound=MonitorPlugin)


class MonitorPluginSystem:
    """
    Extensible plugin architecture for the Monitor Agent.

    The plugin system provides:
    - Dynamic plugin loading/unloading
    - Plugin lifecycle management
    - Configuration management
    - Plugin discovery
    - Dependency resolution

    Example:
        system = MonitorPluginSystem()

        # Register a plugin class
        system.register_plugin(MyPlugin)

        # Load the plugin
        await system.load_plugin("my-plugin", config={"key": "value"})

        # Use the plugin
        plugin = system.get_plugin("my-plugin")
    """

    BUS_TOPICS = {
        "loaded": "monitor.plugin.loaded",
        "unloaded": "monitor.plugin.unloaded",
        "error": "monitor.plugin.error",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        plugin_dirs: Optional[List[str]] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize plugin system.

        Args:
            plugin_dirs: Directories to search for plugins
            bus_dir: Bus directory
        """
        self._plugins: Dict[str, PluginInstance] = {}
        self._plugin_classes: Dict[str, Type[MonitorPlugin]] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._last_heartbeat = time.time()

        # Plugin directories
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        default_dirs = [
            os.path.join(pluribus_root, "pia", "oagent", "monitor", "plugins"),
            os.path.join(pluribus_root, ".pluribus", "monitor", "plugins"),
        ]
        self._plugin_dirs = plugin_dirs or default_dirs

        # Bus path
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def register_plugin(
        self,
        plugin_class: Type[MonitorPlugin],
    ) -> bool:
        """Register a plugin class.

        Args:
            plugin_class: Plugin class to register

        Returns:
            True if registered successfully
        """
        try:
            # Create temporary instance to get metadata
            temp_instance = plugin_class()
            metadata = temp_instance.metadata

            self._plugin_classes[metadata.name] = plugin_class
            self._plugins[metadata.name] = PluginInstance(
                metadata=metadata,
                state=PluginState.REGISTERED,
            )

            return True
        except Exception as e:
            self._emit_bus_event(
                self.BUS_TOPICS["error"],
                {
                    "action": "register",
                    "plugin_class": plugin_class.__name__,
                    "error": str(e),
                },
                level="error",
            )
            return False

    async def load_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Load and initialize a plugin.

        Args:
            name: Plugin name
            config: Plugin configuration

        Returns:
            True if loaded successfully
        """
        if name not in self._plugins:
            return False

        plugin_info = self._plugins[name]
        if plugin_info.state == PluginState.ACTIVE:
            return True  # Already loaded

        plugin_info.state = PluginState.LOADING
        plugin_info.config = config or {}

        try:
            # Check dependencies
            for dep in plugin_info.metadata.dependencies:
                if dep not in self._plugins or self._plugins[dep].state != PluginState.ACTIVE:
                    raise RuntimeError(f"Missing dependency: {dep}")

            # Create instance
            plugin_class = self._plugin_classes[name]
            instance = plugin_class()

            # Initialize
            await instance.initialize(plugin_info.config)

            plugin_info.instance = instance
            plugin_info.state = PluginState.ACTIVE
            plugin_info.loaded_at = time.time()
            plugin_info.error = None

            self._emit_bus_event(
                self.BUS_TOPICS["loaded"],
                {
                    "plugin": name,
                    "version": plugin_info.metadata.version,
                    "type": plugin_info.metadata.plugin_type.value,
                },
            )

            # Trigger hooks
            await self._trigger_hooks("plugin_loaded", plugin_info)

            return True

        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)

            self._emit_bus_event(
                self.BUS_TOPICS["error"],
                {
                    "plugin": name,
                    "action": "load",
                    "error": str(e),
                },
                level="error",
            )

            return False

    async def unload_plugin(self, name: str) -> bool:
        """Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if unloaded successfully
        """
        if name not in self._plugins:
            return False

        plugin_info = self._plugins[name]
        if plugin_info.state not in (PluginState.ACTIVE, PluginState.ERROR):
            return True

        plugin_info.state = PluginState.UNLOADING

        try:
            if plugin_info.instance:
                await plugin_info.instance.shutdown()

            plugin_info.instance = None
            plugin_info.state = PluginState.DISABLED
            plugin_info.loaded_at = None

            self._emit_bus_event(
                self.BUS_TOPICS["unloaded"],
                {"plugin": name},
            )

            # Trigger hooks
            await self._trigger_hooks("plugin_unloaded", plugin_info)

            return True

        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            return False

    async def reload_plugin(self, name: str) -> bool:
        """Reload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if reloaded successfully
        """
        if name not in self._plugins:
            return False

        config = self._plugins[name].config
        await self.unload_plugin(name)
        return await self.load_plugin(name, config)

    def get_plugin(self, name: str) -> Optional[MonitorPlugin]:
        """Get a loaded plugin instance.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        plugin_info = self._plugins.get(name)
        if plugin_info and plugin_info.state == PluginState.ACTIVE:
            return plugin_info.instance
        return None

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[MonitorPlugin]:
        """Get all plugins of a specific type.

        Args:
            plugin_type: Plugin type

        Returns:
            List of plugin instances
        """
        plugins = []
        for plugin_info in self._plugins.values():
            if (
                plugin_info.state == PluginState.ACTIVE
                and plugin_info.metadata.plugin_type == plugin_type
                and plugin_info.instance
            ):
                plugins.append(plugin_info.instance)
        return plugins

    def list_plugins(
        self,
        state_filter: Optional[PluginState] = None,
        type_filter: Optional[PluginType] = None,
    ) -> List[Dict[str, Any]]:
        """List all registered plugins.

        Args:
            state_filter: Filter by state
            type_filter: Filter by type

        Returns:
            List of plugin info dictionaries
        """
        results = []
        for plugin_info in self._plugins.values():
            if state_filter and plugin_info.state != state_filter:
                continue
            if type_filter and plugin_info.metadata.plugin_type != type_filter:
                continue
            results.append(plugin_info.to_dict())
        return results

    async def discover_plugins(self) -> int:
        """Discover plugins in plugin directories.

        Returns:
            Number of plugins discovered
        """
        discovered = 0

        for plugin_dir in self._plugin_dirs:
            if not os.path.isdir(plugin_dir):
                continue

            for filename in os.listdir(plugin_dir):
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue

                filepath = os.path.join(plugin_dir, filename)
                try:
                    spec = importlib.util.spec_from_file_location(
                        filename[:-3], filepath
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Find plugin classes
                        for name, obj in inspect.getmembers(module):
                            if (
                                inspect.isclass(obj)
                                and issubclass(obj, MonitorPlugin)
                                and obj != MonitorPlugin
                                and not inspect.isabstract(obj)
                            ):
                                if self.register_plugin(obj):
                                    discovered += 1

                except Exception as e:
                    self._emit_bus_event(
                        self.BUS_TOPICS["error"],
                        {
                            "action": "discover",
                            "file": filepath,
                            "error": str(e),
                        },
                        level="error",
                    )

        return discovered

    def register_hook(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register a hook callback.

        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    async def _trigger_hooks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Trigger hooks for an event.

        Args:
            event: Event name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        for callback in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception:
                pass

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all plugins.

        Returns:
            Health status
        """
        plugin_health = {}
        overall_healthy = True

        for name, plugin_info in self._plugins.items():
            if plugin_info.state == PluginState.ACTIVE and plugin_info.instance:
                try:
                    health = await plugin_info.instance.health_check()
                    plugin_health[name] = health
                    if not health.get("healthy", True):
                        overall_healthy = False
                except Exception as e:
                    plugin_health[name] = {"healthy": False, "error": str(e)}
                    overall_healthy = False
            else:
                plugin_health[name] = {
                    "healthy": plugin_info.state != PluginState.ERROR,
                    "state": plugin_info.state.value,
                }

        return {
            "healthy": overall_healthy,
            "total_plugins": len(self._plugins),
            "active_plugins": sum(
                1 for p in self._plugins.values() if p.state == PluginState.ACTIVE
            ),
            "plugins": plugin_health,
        }

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_plugin_system",
                "status": "healthy",
                "plugins": len(self._plugins),
                "active": sum(
                    1 for p in self._plugins.values() if p.state == PluginState.ACTIVE
                ),
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-plugin-system",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_plugin_system: Optional[MonitorPluginSystem] = None


def get_plugin_system() -> MonitorPluginSystem:
    """Get or create the plugin system singleton.

    Returns:
        MonitorPluginSystem instance
    """
    global _plugin_system
    if _plugin_system is None:
        _plugin_system = MonitorPluginSystem()
    return _plugin_system


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Plugin System (Step 281)")
    parser.add_argument("--list", action="store_true", help="List plugins")
    parser.add_argument("--discover", action="store_true", help="Discover plugins")
    parser.add_argument("--load", metavar="NAME", help="Load a plugin")
    parser.add_argument("--unload", metavar="NAME", help="Unload a plugin")
    parser.add_argument("--health", action="store_true", help="Check health")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    system = get_plugin_system()

    if args.discover:
        async def run_discover():
            return await system.discover_plugins()
        count = asyncio.run(run_discover())
        if args.json:
            print(json.dumps({"discovered": count}))
        else:
            print(f"Discovered {count} plugins")

    if args.list:
        plugins = system.list_plugins()
        if args.json:
            print(json.dumps(plugins, indent=2))
        else:
            print("Plugins:")
            for p in plugins:
                print(f"  {p['metadata']['name']}: {p['state']}")

    if args.load:
        async def run_load():
            return await system.load_plugin(args.load)
        success = asyncio.run(run_load())
        if args.json:
            print(json.dumps({"loaded": success}))
        else:
            print(f"Load {args.load}: {'success' if success else 'failed'}")

    if args.unload:
        async def run_unload():
            return await system.unload_plugin(args.unload)
        success = asyncio.run(run_unload())
        if args.json:
            print(json.dumps({"unloaded": success}))
        else:
            print(f"Unload {args.unload}: {'success' if success else 'failed'}")

    if args.health:
        async def run_health():
            return await system.health_check()
        health = asyncio.run(run_health())
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            status = "healthy" if health["healthy"] else "unhealthy"
            print(f"Plugin System Health: {status}")
            print(f"  Total: {health['total_plugins']}")
            print(f"  Active: {health['active_plugins']}")
