#!/usr/bin/env python3
"""
plugin_system.py - Extensible Plugin Architecture (Step 81)

PBTSO Phase: SKILL, ITERATE

Provides:
- Plugin discovery and loading
- Hook-based extension points
- Plugin lifecycle management
- Dependency resolution
- Sandboxed plugin execution

Bus Topics:
- code.plugin.loaded
- code.plugin.unloaded
- code.plugin.error
- code.plugin.hook

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import socket
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class PluginState(Enum):
    """Plugin lifecycle state."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class PluginInfo:
    """Plugin metadata."""
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    entry_point: str = ""
    min_agent_version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "hooks": self.hooks,
            "entry_point": self.entry_point,
            "min_agent_version": self.min_agent_version,
        }


@dataclass
class PluginConfig:
    """Configuration for plugin system."""
    plugin_dirs: List[str] = field(default_factory=lambda: [
        "/pluribus/pia/oagent/code/plugins",
        "/pluribus/.pluribus/plugins",
    ])
    auto_discover: bool = True
    auto_load: bool = True
    sandbox_enabled: bool = True
    max_plugins: int = 100
    hook_timeout_s: float = 30.0
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_dirs": self.plugin_dirs,
            "auto_discover": self.auto_discover,
            "auto_load": self.auto_load,
            "sandbox_enabled": self.sandbox_enabled,
            "max_plugins": self.max_plugins,
            "hook_timeout_s": self.hook_timeout_s,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Plugin Context
# =============================================================================

@dataclass
class PluginContext:
    """
    Context provided to plugins during execution.

    Provides sandboxed access to agent capabilities.
    """
    plugin_id: str
    working_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)
    bus: Optional[LockedAgentBus] = None

    # Services available to plugins
    _services: Dict[str, Any] = field(default_factory=dict)

    def get_service(self, name: str) -> Optional[Any]:
        """Get a service by name."""
        return self._services.get(name)

    def emit_event(self, topic: str, data: Dict[str, Any]) -> str:
        """Emit an event from plugin."""
        if self.bus:
            return self.bus.emit({
                "topic": f"plugin.{self.plugin_id}.{topic}",
                "kind": "plugin_event",
                "actor": self.plugin_id,
                "data": data,
            })
        return ""

    def log(self, level: str, message: str, **kwargs: Any) -> None:
        """Log a message from plugin."""
        if self.bus:
            self.bus.emit({
                "topic": "plugin.log",
                "kind": "log",
                "level": level,
                "actor": self.plugin_id,
                "data": {"message": message, **kwargs},
            })


# =============================================================================
# Hook System
# =============================================================================

T = TypeVar("T")
HookResult = TypeVar("HookResult")


@dataclass
class PluginHook(Generic[T, HookResult]):
    """
    Represents a hook point that plugins can extend.

    Hooks allow plugins to intercept and modify behavior.
    """
    name: str
    description: str = ""
    priority_default: int = 100
    allow_async: bool = True

    _handlers: List[tuple[int, str, Callable]] = field(default_factory=list)

    def register(
        self,
        handler: Callable[[T], HookResult],
        plugin_id: str,
        priority: Optional[int] = None,
    ) -> None:
        """Register a handler for this hook."""
        prio = priority if priority is not None else self.priority_default
        self._handlers.append((prio, plugin_id, handler))
        self._handlers.sort(key=lambda x: x[0])

    def unregister(self, plugin_id: str) -> None:
        """Unregister all handlers from a plugin."""
        self._handlers = [(p, pid, h) for p, pid, h in self._handlers if pid != plugin_id]

    async def execute(self, data: T) -> List[HookResult]:
        """Execute all handlers and collect results."""
        results = []
        for _, _, handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(data)
                else:
                    result = handler(data)
                results.append(result)
            except Exception as e:
                results.append(None)
        return results

    async def execute_chain(self, data: T) -> T:
        """Execute handlers in chain, passing result to next."""
        current = data
        for _, _, handler in self._handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    current = await handler(current)
                else:
                    current = handler(current)
            except Exception:
                pass
        return current


# Decorator for marking hook handlers
def hook(hook_name: str, priority: int = 100) -> Callable:
    """Decorator to mark a method as a hook handler."""
    def decorator(func: Callable) -> Callable:
        func._hook_name = hook_name  # type: ignore
        func._hook_priority = priority  # type: ignore
        return func
    return decorator


# =============================================================================
# Plugin Base Class
# =============================================================================

class Plugin(ABC):
    """
    Base class for all plugins.

    Plugins must inherit from this class and implement
    the required lifecycle methods.

    Usage:
        class MyPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(
                    id="my-plugin",
                    name="My Plugin",
                    version="1.0.0",
                )

            async def initialize(self, context: PluginContext) -> bool:
                # Setup plugin
                return True

            async def shutdown(self) -> None:
                # Cleanup
                pass
    """

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Get plugin metadata."""
        pass

    @abstractmethod
    async def initialize(self, context: PluginContext) -> bool:
        """
        Initialize the plugin.

        Args:
            context: Plugin execution context

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown and cleanup the plugin."""
        pass

    def get_hooks(self) -> List[tuple[str, int, Callable]]:
        """Get all hook handlers defined in this plugin."""
        hooks = []
        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and hasattr(method, "_hook_name"):
                hooks.append((
                    method._hook_name,
                    method._hook_priority,
                    method,
                ))
        return hooks


# =============================================================================
# Plugin Registry
# =============================================================================

class PluginRegistry:
    """
    Central registry for plugins and hooks.

    Manages plugin discovery, loading, and hook registration.
    """

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_states: Dict[str, PluginState] = {}
        self._hooks: Dict[str, PluginHook] = {}
        self._contexts: Dict[str, PluginContext] = {}

    # =========================================================================
    # Hook Management
    # =========================================================================

    def register_hook(self, hook: PluginHook) -> None:
        """Register a hook point."""
        self._hooks[hook.name] = hook

    def get_hook(self, name: str) -> Optional[PluginHook]:
        """Get a hook by name."""
        return self._hooks.get(name)

    def list_hooks(self) -> List[str]:
        """List all registered hooks."""
        return list(self._hooks.keys())

    # =========================================================================
    # Plugin Management
    # =========================================================================

    def register_plugin(self, plugin: Plugin) -> bool:
        """Register a plugin instance."""
        plugin_id = plugin.info.id

        if plugin_id in self._plugins:
            return False

        self._plugins[plugin_id] = plugin
        self._plugin_states[plugin_id] = PluginState.LOADED
        return True

    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin."""
        if plugin_id not in self._plugins:
            return False

        # Unregister all hooks
        for hook in self._hooks.values():
            hook.unregister(plugin_id)

        del self._plugins[plugin_id]
        del self._plugin_states[plugin_id]

        if plugin_id in self._contexts:
            del self._contexts[plugin_id]

        return True

    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)

    def get_state(self, plugin_id: str) -> Optional[PluginState]:
        """Get plugin state."""
        return self._plugin_states.get(plugin_id)

    def set_state(self, plugin_id: str, state: PluginState) -> None:
        """Set plugin state."""
        if plugin_id in self._plugin_states:
            self._plugin_states[plugin_id] = state

    def list_plugins(self) -> List[PluginInfo]:
        """List all registered plugins."""
        return [p.info for p in self._plugins.values()]

    # =========================================================================
    # Context Management
    # =========================================================================

    def set_context(self, plugin_id: str, context: PluginContext) -> None:
        """Set plugin context."""
        self._contexts[plugin_id] = context

    def get_context(self, plugin_id: str) -> Optional[PluginContext]:
        """Get plugin context."""
        return self._contexts.get(plugin_id)


# =============================================================================
# Plugin Manager
# =============================================================================

class PluginManager:
    """
    Main plugin management system.

    PBTSO Phase: SKILL (capability extension)

    Handles:
    - Plugin discovery from directories
    - Plugin loading and initialization
    - Hook execution
    - Plugin lifecycle management

    Usage:
        manager = PluginManager(config)
        await manager.discover()
        await manager.load_all()

        results = await manager.execute_hook("before_edit", data)
    """

    BUS_TOPICS = {
        "loaded": "code.plugin.loaded",
        "unloaded": "code.plugin.unloaded",
        "error": "code.plugin.error",
        "hook": "code.plugin.hook",
    }

    # Standard hooks
    HOOKS = {
        "before_edit": PluginHook(name="before_edit", description="Before file edit"),
        "after_edit": PluginHook(name="after_edit", description="After file edit"),
        "before_generate": PluginHook(name="before_generate", description="Before code generation"),
        "after_generate": PluginHook(name="after_generate", description="After code generation"),
        "before_lint": PluginHook(name="before_lint", description="Before linting"),
        "after_lint": PluginHook(name="after_lint", description="After linting"),
        "before_format": PluginHook(name="before_format", description="Before formatting"),
        "after_format": PluginHook(name="after_format", description="After formatting"),
        "transform_ast": PluginHook(name="transform_ast", description="AST transformation"),
        "validate_code": PluginHook(name="validate_code", description="Code validation"),
    }

    def __init__(
        self,
        config: Optional[PluginConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or PluginConfig()
        self.bus = bus or LockedAgentBus()
        self.registry = PluginRegistry()

        # Register standard hooks
        for hook in self.HOOKS.values():
            self.registry.register_hook(hook)

        # Discovery cache
        self._discovered: Dict[str, Path] = {}

    # =========================================================================
    # Discovery
    # =========================================================================

    async def discover(self) -> List[PluginInfo]:
        """
        Discover plugins in configured directories.

        Returns:
            List of discovered plugin info
        """
        discovered = []

        for plugin_dir in self.config.plugin_dirs:
            dir_path = Path(plugin_dir)
            if not dir_path.exists():
                continue

            for item in dir_path.iterdir():
                if item.is_dir() and (item / "plugin.json").exists():
                    info = self._load_plugin_manifest(item / "plugin.json")
                    if info:
                        self._discovered[info.id] = item
                        discovered.append(info)
                elif item.suffix == ".py" and item.stem != "__init__":
                    info = self._probe_plugin_file(item)
                    if info:
                        self._discovered[info.id] = item
                        discovered.append(info)

        return discovered

    def _load_plugin_manifest(self, manifest_path: Path) -> Optional[PluginInfo]:
        """Load plugin info from manifest file."""
        try:
            data = json.loads(manifest_path.read_text())
            return PluginInfo(
                id=data.get("id", manifest_path.parent.name),
                name=data.get("name", ""),
                version=data.get("version", "0.0.0"),
                description=data.get("description", ""),
                author=data.get("author", ""),
                dependencies=data.get("dependencies", []),
                hooks=data.get("hooks", []),
                entry_point=data.get("entry_point", "main.py"),
                min_agent_version=data.get("min_agent_version", "0.1.0"),
            )
        except Exception:
            return None

    def _probe_plugin_file(self, file_path: Path) -> Optional[PluginInfo]:
        """Probe a Python file for plugin class."""
        try:
            content = file_path.read_text()
            if "class" in content and "Plugin" in content:
                return PluginInfo(
                    id=file_path.stem,
                    name=file_path.stem,
                    version="0.0.0",
                    entry_point=str(file_path),
                )
        except Exception:
            pass
        return None

    # =========================================================================
    # Loading
    # =========================================================================

    async def load(self, plugin_id: str) -> bool:
        """
        Load a discovered plugin.

        Args:
            plugin_id: Plugin ID to load

        Returns:
            True if loaded successfully
        """
        if plugin_id not in self._discovered:
            return False

        plugin_path = self._discovered[plugin_id]

        try:
            plugin = await self._load_plugin_from_path(plugin_path)
            if not plugin:
                return False

            # Register plugin
            if not self.registry.register_plugin(plugin):
                return False

            # Create context
            context = PluginContext(
                plugin_id=plugin_id,
                working_dir=Path(self.config.plugin_dirs[0]),
                bus=self.bus,
            )
            self.registry.set_context(plugin_id, context)

            # Initialize plugin
            success = await plugin.initialize(context)
            if not success:
                self.registry.unregister_plugin(plugin_id)
                return False

            # Register hooks
            for hook_name, priority, handler in plugin.get_hooks():
                hook = self.registry.get_hook(hook_name)
                if hook:
                    hook.register(handler, plugin_id, priority)

            self.registry.set_state(plugin_id, PluginState.ACTIVE)

            # Emit event
            self.bus.emit({
                "topic": self.BUS_TOPICS["loaded"],
                "kind": "plugin",
                "actor": "plugin-manager",
                "data": plugin.info.to_dict(),
            })

            return True

        except Exception as e:
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "error",
                "actor": "plugin-manager",
                "data": {
                    "plugin_id": plugin_id,
                    "error": str(e),
                },
            })
            return False

    async def _load_plugin_from_path(self, path: Path) -> Optional[Plugin]:
        """Load plugin from file or directory."""
        if path.is_dir():
            entry_point = path / "main.py"
            if not entry_point.exists():
                # Check manifest for entry point
                manifest = path / "plugin.json"
                if manifest.exists():
                    data = json.loads(manifest.read_text())
                    entry_point = path / data.get("entry_point", "main.py")
        else:
            entry_point = path

        if not entry_point.exists():
            return None

        # Load module
        spec = importlib.util.spec_from_file_location(
            f"plugin_{path.stem}",
            entry_point,
        )
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Find Plugin class
        for name in dir(module):
            obj = getattr(module, name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, Plugin)
                and obj is not Plugin
            ):
                return obj()

        return None

    async def load_all(self) -> Dict[str, bool]:
        """Load all discovered plugins."""
        results = {}

        # Sort by dependencies (simple topological sort)
        plugin_ids = list(self._discovered.keys())

        for plugin_id in plugin_ids:
            if len(self.registry._plugins) >= self.config.max_plugins:
                results[plugin_id] = False
                continue
            results[plugin_id] = await self.load(plugin_id)

        return results

    # =========================================================================
    # Unloading
    # =========================================================================

    async def unload(self, plugin_id: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_id: Plugin ID to unload

        Returns:
            True if unloaded successfully
        """
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin:
            return False

        try:
            await plugin.shutdown()
        except Exception:
            pass

        self.registry.unregister_plugin(plugin_id)

        self.bus.emit({
            "topic": self.BUS_TOPICS["unloaded"],
            "kind": "plugin",
            "actor": "plugin-manager",
            "data": {"plugin_id": plugin_id},
        })

        return True

    async def unload_all(self) -> None:
        """Unload all plugins."""
        for plugin_id in list(self.registry._plugins.keys()):
            await self.unload(plugin_id)

    # =========================================================================
    # Hook Execution
    # =========================================================================

    async def execute_hook(
        self,
        hook_name: str,
        data: Any,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        Execute a hook with all registered handlers.

        Args:
            hook_name: Name of hook to execute
            data: Data to pass to handlers
            timeout: Execution timeout

        Returns:
            List of results from handlers
        """
        hook = self.registry.get_hook(hook_name)
        if not hook:
            return []

        timeout = timeout or self.config.hook_timeout_s

        try:
            results = await asyncio.wait_for(
                hook.execute(data),
                timeout=timeout,
            )

            self.bus.emit({
                "topic": self.BUS_TOPICS["hook"],
                "kind": "hook",
                "actor": "plugin-manager",
                "data": {
                    "hook": hook_name,
                    "handlers": len(results),
                },
            })

            return results

        except asyncio.TimeoutError:
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "warning",
                "actor": "plugin-manager",
                "data": {
                    "hook": hook_name,
                    "error": "Timeout",
                },
            })
            return []

    async def execute_hook_chain(
        self,
        hook_name: str,
        data: Any,
    ) -> Any:
        """
        Execute a hook as a chain, passing result to each handler.

        Args:
            hook_name: Name of hook to execute
            data: Initial data

        Returns:
            Final transformed data
        """
        hook = self.registry.get_hook(hook_name)
        if not hook:
            return data

        return await hook.execute_chain(data)

    # =========================================================================
    # Queries
    # =========================================================================

    def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return self.registry.list_plugins()

    def get_plugin_state(self, plugin_id: str) -> Optional[PluginState]:
        """Get plugin state."""
        return self.registry.get_state(plugin_id)

    def list_hooks(self) -> List[str]:
        """List all available hooks."""
        return self.registry.list_hooks()

    def get_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics."""
        return {
            "discovered": len(self._discovered),
            "loaded": len(self.registry._plugins),
            "hooks": len(self.registry._hooks),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Plugin System."""
    import argparse

    parser = argparse.ArgumentParser(description="Plugin System (Step 81)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List plugins")
    list_parser.add_argument("--discovered", action="store_true", help="Show discovered")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # load command
    load_parser = subparsers.add_parser("load", help="Load a plugin")
    load_parser.add_argument("plugin_id", help="Plugin ID to load")

    # unload command
    unload_parser = subparsers.add_parser("unload", help="Unload a plugin")
    unload_parser.add_argument("plugin_id", help="Plugin ID to unload")

    # hooks command
    hooks_parser = subparsers.add_parser("hooks", help="List hooks")
    hooks_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()
    manager = PluginManager()

    async def run() -> int:
        if args.command == "list":
            if args.discovered:
                discovered = await manager.discover()
                if args.json:
                    print(json.dumps([p.to_dict() for p in discovered], indent=2))
                else:
                    print(f"Discovered {len(discovered)} plugins:")
                    for p in discovered:
                        print(f"  {p.id}: {p.name} v{p.version}")
            else:
                plugins = manager.list_plugins()
                if args.json:
                    print(json.dumps([p.to_dict() for p in plugins], indent=2))
                else:
                    print(f"Loaded {len(plugins)} plugins:")
                    for p in plugins:
                        state = manager.get_plugin_state(p.id)
                        print(f"  {p.id}: {p.name} [{state.value if state else 'unknown'}]")
            return 0

        elif args.command == "load":
            await manager.discover()
            success = await manager.load(args.plugin_id)
            if success:
                print(f"Loaded plugin: {args.plugin_id}")
                return 0
            else:
                print(f"Failed to load: {args.plugin_id}")
                return 1

        elif args.command == "unload":
            success = await manager.unload(args.plugin_id)
            if success:
                print(f"Unloaded plugin: {args.plugin_id}")
                return 0
            else:
                print(f"Failed to unload: {args.plugin_id}")
                return 1

        elif args.command == "hooks":
            hooks = manager.list_hooks()
            if args.json:
                print(json.dumps(hooks, indent=2))
            else:
                print(f"Available hooks ({len(hooks)}):")
                for h in hooks:
                    print(f"  - {h}")
            return 0

        elif args.command == "stats":
            stats = manager.get_stats()
            print(json.dumps(stats, indent=2))
            return 0

        return 1

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
