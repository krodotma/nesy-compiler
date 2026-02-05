#!/usr/bin/env python3
"""
system.py - Deploy Plugin System (Step 231)

PBTSO Phase: SKILL, DISTRIBUTE
A2A Integration: Manages extensible plugin architecture via deploy.plugin.*

Provides:
- PluginType: Types of deploy plugins
- PluginState: Plugin lifecycle states
- PluginMetadata: Plugin metadata
- PluginConfig: Plugin configuration
- PluginInterface: Base plugin interface
- PluginRegistry: Plugin registry
- DeployPluginSystem: Main plugin system

Bus Topics:
- deploy.plugin.register
- deploy.plugin.load
- deploy.plugin.unload
- deploy.plugin.execute
- deploy.plugin.error

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import abc
import asyncio
import fcntl
import importlib
import importlib.util
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar


# ==============================================================================
# Bus Emission Helper with File Locking (DKIN v30)
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "plugin-system"
) -> str:
    """Emit an event to the Pluribus bus with fcntl.flock() file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class PluginType(Enum):
    """Types of deploy plugins."""
    BUILD = "build"
    TEST = "test"
    PACKAGE = "package"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"
    HEALTH_CHECK = "health_check"
    NOTIFICATION = "notification"
    METRICS = "metrics"
    SECURITY = "security"
    CUSTOM = "custom"


class PluginState(Enum):
    """Plugin lifecycle states."""
    REGISTERED = "registered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    UNLOADED = "unloaded"


class PluginHook(Enum):
    """Plugin hook points in deployment lifecycle."""
    PRE_BUILD = "pre_build"
    POST_BUILD = "post_build"
    PRE_TEST = "pre_test"
    POST_TEST = "post_test"
    PRE_PACKAGE = "pre_package"
    POST_PACKAGE = "post_package"
    PRE_DEPLOY = "pre_deploy"
    POST_DEPLOY = "post_deploy"
    PRE_ROLLBACK = "pre_rollback"
    POST_ROLLBACK = "post_rollback"
    ON_ERROR = "on_error"
    ON_SUCCESS = "on_success"


@dataclass
class PluginMetadata:
    """
    Plugin metadata.

    Attributes:
        plugin_id: Unique plugin identifier
        name: Human-readable plugin name
        version: Plugin version
        author: Plugin author
        description: Plugin description
        plugin_type: Type of plugin
        hooks: Supported hooks
        dependencies: Required plugins
        tags: Plugin tags
        created_at: Registration timestamp
    """
    plugin_id: str
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    plugin_type: PluginType = PluginType.CUSTOM
    hooks: List[PluginHook] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "plugin_type": self.plugin_type.value,
            "hooks": [h.value for h in self.hooks],
            "dependencies": self.dependencies,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        data = dict(data)
        if "plugin_type" in data:
            data["plugin_type"] = PluginType(data["plugin_type"])
        if "hooks" in data:
            data["hooks"] = [PluginHook(h) for h in data["hooks"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PluginConfig:
    """
    Plugin configuration.

    Attributes:
        plugin_id: Plugin identifier
        enabled: Whether plugin is enabled
        priority: Execution priority (lower = earlier)
        settings: Plugin-specific settings
        environments: Target environments (empty = all)
        services: Target services (empty = all)
    """
    plugin_id: str
    enabled: bool = True
    priority: int = 100
    settings: Dict[str, Any] = field(default_factory=dict)
    environments: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PluginExecutionContext:
    """Context passed to plugin execution."""
    hook: PluginHook
    deployment_id: str
    service_name: str
    environment: str
    version: str
    data: Dict[str, Any] = field(default_factory=dict)
    previous_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook": self.hook.value,
            "deployment_id": self.deployment_id,
            "service_name": self.service_name,
            "environment": self.environment,
            "version": self.version,
            "data": self.data,
            "previous_results": self.previous_results,
        }


@dataclass
class PluginExecutionResult:
    """Result of plugin execution."""
    plugin_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Plugin Interface (Abstract Base)
# ==============================================================================

T = TypeVar("T", bound="PluginInterface")


class PluginInterface(abc.ABC):
    """
    Base interface for deploy plugins.

    All plugins must inherit from this class and implement
    the required methods.
    """

    @classmethod
    @abc.abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abc.abstractmethod
    async def initialize(self, config: PluginConfig) -> bool:
        """
        Initialize the plugin with configuration.

        Args:
            config: Plugin configuration

        Returns:
            True if initialization successful
        """
        pass

    @abc.abstractmethod
    async def execute(self, context: PluginExecutionContext) -> PluginExecutionResult:
        """
        Execute the plugin.

        Args:
            context: Execution context

        Returns:
            PluginExecutionResult
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup resources on unload."""
        pass

    def should_execute(self, context: PluginExecutionContext) -> bool:
        """Determine if plugin should execute for given context."""
        metadata = self.get_metadata()
        return context.hook in metadata.hooks


# ==============================================================================
# Plugin Registry
# ==============================================================================

@dataclass
class PluginRegistration:
    """Internal plugin registration entry."""
    metadata: PluginMetadata
    config: PluginConfig
    state: PluginState
    instance: Optional[PluginInterface] = None
    source_path: Optional[str] = None
    load_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_executed: float = 0.0


class PluginRegistry:
    """
    Registry for managing plugin registrations.

    Tracks all registered plugins and their states.
    """

    def __init__(self):
        self._plugins: Dict[str, PluginRegistration] = {}
        self._hooks: Dict[PluginHook, List[str]] = {h: [] for h in PluginHook}

    def register(
        self,
        metadata: PluginMetadata,
        config: Optional[PluginConfig] = None,
        source_path: Optional[str] = None,
    ) -> PluginRegistration:
        """Register a new plugin."""
        if config is None:
            config = PluginConfig(plugin_id=metadata.plugin_id)

        registration = PluginRegistration(
            metadata=metadata,
            config=config,
            state=PluginState.REGISTERED,
            source_path=source_path,
        )

        self._plugins[metadata.plugin_id] = registration

        # Index by hooks
        for hook in metadata.hooks:
            if metadata.plugin_id not in self._hooks[hook]:
                self._hooks[hook].append(metadata.plugin_id)

        return registration

    def unregister(self, plugin_id: str) -> bool:
        """Unregister a plugin."""
        if plugin_id not in self._plugins:
            return False

        registration = self._plugins[plugin_id]

        # Remove from hook index
        for hook in registration.metadata.hooks:
            if plugin_id in self._hooks[hook]:
                self._hooks[hook].remove(plugin_id)

        del self._plugins[plugin_id]
        return True

    def get(self, plugin_id: str) -> Optional[PluginRegistration]:
        """Get a plugin registration."""
        return self._plugins.get(plugin_id)

    def get_by_hook(self, hook: PluginHook) -> List[PluginRegistration]:
        """Get plugins registered for a specific hook."""
        plugin_ids = self._hooks.get(hook, [])
        registrations = [self._plugins[pid] for pid in plugin_ids if pid in self._plugins]
        # Sort by priority
        registrations.sort(key=lambda r: r.config.priority)
        return registrations

    def get_by_type(self, plugin_type: PluginType) -> List[PluginRegistration]:
        """Get plugins of a specific type."""
        return [
            r for r in self._plugins.values()
            if r.metadata.plugin_type == plugin_type
        ]

    def list_all(self) -> List[PluginRegistration]:
        """List all registered plugins."""
        return list(self._plugins.values())

    def get_active(self) -> List[PluginRegistration]:
        """Get all active (enabled and loaded) plugins."""
        return [
            r for r in self._plugins.values()
            if r.state == PluginState.ACTIVE and r.config.enabled
        ]


# ==============================================================================
# Deploy Plugin System (Step 231)
# ==============================================================================

class DeployPluginSystem:
    """
    Deploy Plugin System - extensible plugin architecture.

    PBTSO Phase: SKILL, DISTRIBUTE

    Responsibilities:
    - Register and manage plugins
    - Load plugins from files or classes
    - Execute plugins at defined hooks
    - Handle plugin dependencies
    - Track plugin lifecycle

    Example:
        >>> system = DeployPluginSystem()
        >>> class MyPlugin(PluginInterface):
        ...     @classmethod
        ...     def get_metadata(cls):
        ...         return PluginMetadata(
        ...             plugin_id="my-plugin",
        ...             name="My Plugin",
        ...             plugin_type=PluginType.CUSTOM,
        ...             hooks=[PluginHook.PRE_DEPLOY],
        ...         )
        ...     async def initialize(self, config):
        ...         return True
        ...     async def execute(self, context):
        ...         return PluginExecutionResult(plugin_id="my-plugin", success=True)
        >>> system.register_class(MyPlugin)
        >>> results = await system.execute_hook(PluginHook.PRE_DEPLOY, context)
    """

    BUS_TOPICS = {
        "register": "deploy.plugin.register",
        "load": "deploy.plugin.load",
        "unload": "deploy.plugin.unload",
        "execute": "deploy.plugin.execute",
        "error": "deploy.plugin.error",
    }

    # A2A heartbeat configuration (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "plugin-system",
        auto_discover: bool = False,
        plugin_dirs: Optional[List[str]] = None,
    ):
        """
        Initialize the plugin system.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            auto_discover: Auto-discover plugins in plugin_dirs
            plugin_dirs: Directories to scan for plugins
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "plugins"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.plugin_dirs = plugin_dirs or []

        self._registry = PluginRegistry()
        self._plugin_classes: Dict[str, Type[PluginInterface]] = {}
        self._circuit_breaker: Dict[str, Dict[str, Any]] = {}

        self._load_state()

        if auto_discover:
            self._discover_plugins()

    def register_class(
        self,
        plugin_class: Type[PluginInterface],
        config: Optional[PluginConfig] = None,
    ) -> PluginRegistration:
        """
        Register a plugin class.

        Args:
            plugin_class: Plugin class inheriting PluginInterface
            config: Optional plugin configuration

        Returns:
            PluginRegistration
        """
        metadata = plugin_class.get_metadata()

        registration = self._registry.register(metadata, config)
        self._plugin_classes[metadata.plugin_id] = plugin_class

        _emit_bus_event(
            self.BUS_TOPICS["register"],
            {
                "plugin_id": metadata.plugin_id,
                "name": metadata.name,
                "version": metadata.version,
                "plugin_type": metadata.plugin_type.value,
                "hooks": [h.value for h in metadata.hooks],
            },
            actor=self.actor_id,
        )

        self._save_state()
        return registration

    def register_from_file(
        self,
        file_path: str,
        class_name: str,
        config: Optional[PluginConfig] = None,
    ) -> Optional[PluginRegistration]:
        """
        Register a plugin from a Python file.

        Args:
            file_path: Path to plugin Python file
            class_name: Name of plugin class in file
            config: Optional plugin configuration

        Returns:
            PluginRegistration or None if failed
        """
        try:
            spec = importlib.util.spec_from_file_location("plugin_module", file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            plugin_class = getattr(module, class_name, None)
            if plugin_class is None or not issubclass(plugin_class, PluginInterface):
                return None

            registration = self.register_class(plugin_class, config)
            registration.source_path = file_path
            return registration

        except Exception as e:
            _emit_bus_event(
                self.BUS_TOPICS["error"],
                {
                    "file_path": file_path,
                    "class_name": class_name,
                    "error": str(e),
                },
                level="error",
                actor=self.actor_id,
            )
            return None

    async def load_plugin(self, plugin_id: str) -> bool:
        """
        Load and initialize a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if loaded successfully
        """
        registration = self._registry.get(plugin_id)
        if not registration:
            return False

        if registration.state == PluginState.ACTIVE:
            return True

        plugin_class = self._plugin_classes.get(plugin_id)
        if not plugin_class:
            return False

        try:
            instance = plugin_class()
            success = await instance.initialize(registration.config)

            if success:
                registration.instance = instance
                registration.state = PluginState.ACTIVE
                registration.load_count += 1

                _emit_bus_event(
                    self.BUS_TOPICS["load"],
                    {
                        "plugin_id": plugin_id,
                        "name": registration.metadata.name,
                        "load_count": registration.load_count,
                    },
                    actor=self.actor_id,
                )
            else:
                registration.state = PluginState.ERROR
                registration.last_error = "Initialization returned False"

            self._save_state()
            return success

        except Exception as e:
            registration.state = PluginState.ERROR
            registration.error_count += 1
            registration.last_error = str(e)

            _emit_bus_event(
                self.BUS_TOPICS["error"],
                {
                    "plugin_id": plugin_id,
                    "error": str(e),
                    "error_count": registration.error_count,
                },
                level="error",
                actor=self.actor_id,
            )

            self._save_state()
            return False

    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if unloaded successfully
        """
        registration = self._registry.get(plugin_id)
        if not registration:
            return False

        if registration.instance:
            try:
                await registration.instance.cleanup()
            except Exception:
                pass

        registration.instance = None
        registration.state = PluginState.UNLOADED

        _emit_bus_event(
            self.BUS_TOPICS["unload"],
            {
                "plugin_id": plugin_id,
                "name": registration.metadata.name,
            },
            actor=self.actor_id,
        )

        self._save_state()
        return True

    async def execute_hook(
        self,
        hook: PluginHook,
        context: PluginExecutionContext,
        stop_on_error: bool = False,
    ) -> List[PluginExecutionResult]:
        """
        Execute all plugins registered for a hook.

        Args:
            hook: Hook to execute
            context: Execution context
            stop_on_error: Stop execution on first error

        Returns:
            List of PluginExecutionResult
        """
        results: List[PluginExecutionResult] = []
        registrations = self._registry.get_by_hook(hook)

        for registration in registrations:
            if not registration.config.enabled:
                continue

            if registration.state != PluginState.ACTIVE:
                # Try to load the plugin
                loaded = await self.load_plugin(registration.metadata.plugin_id)
                if not loaded:
                    continue

            # Check circuit breaker (PAIP v16)
            if self._is_circuit_open(registration.metadata.plugin_id):
                results.append(PluginExecutionResult(
                    plugin_id=registration.metadata.plugin_id,
                    success=False,
                    skipped=True,
                    skip_reason="Circuit breaker open",
                ))
                continue

            # Filter by environment/service
            if registration.config.environments:
                if context.environment not in registration.config.environments:
                    results.append(PluginExecutionResult(
                        plugin_id=registration.metadata.plugin_id,
                        success=True,
                        skipped=True,
                        skip_reason="Environment filter",
                    ))
                    continue

            if registration.config.services:
                if context.service_name not in registration.config.services:
                    results.append(PluginExecutionResult(
                        plugin_id=registration.metadata.plugin_id,
                        success=True,
                        skipped=True,
                        skip_reason="Service filter",
                    ))
                    continue

            # Execute plugin
            result = await self._execute_plugin(registration, context)
            results.append(result)

            # Update context with result for next plugin
            context.previous_results[registration.metadata.plugin_id] = result.data

            if not result.success and stop_on_error:
                break

        return results

    async def _execute_plugin(
        self,
        registration: PluginRegistration,
        context: PluginExecutionContext,
    ) -> PluginExecutionResult:
        """Execute a single plugin."""
        plugin_id = registration.metadata.plugin_id

        if not registration.instance:
            return PluginExecutionResult(
                plugin_id=plugin_id,
                success=False,
                error="Plugin not loaded",
            )

        start_time = time.time()

        try:
            result = await registration.instance.execute(context)
            result.duration_ms = (time.time() - start_time) * 1000

            registration.last_executed = time.time()

            # Reset circuit breaker on success
            self._reset_circuit_breaker(plugin_id)

            _emit_bus_event(
                self.BUS_TOPICS["execute"],
                {
                    "plugin_id": plugin_id,
                    "hook": context.hook.value,
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                    "deployment_id": context.deployment_id,
                },
                actor=self.actor_id,
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            registration.error_count += 1
            registration.last_error = str(e)

            # Update circuit breaker
            self._record_circuit_failure(plugin_id)

            _emit_bus_event(
                self.BUS_TOPICS["error"],
                {
                    "plugin_id": plugin_id,
                    "hook": context.hook.value,
                    "error": str(e),
                    "error_count": registration.error_count,
                },
                level="error",
                actor=self.actor_id,
            )

            return PluginExecutionResult(
                plugin_id=plugin_id,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _is_circuit_open(self, plugin_id: str) -> bool:
        """Check if circuit breaker is open for plugin."""
        circuit = self._circuit_breaker.get(plugin_id, {})
        if circuit.get("state") == "open":
            # Check if timeout has passed
            opened_at = circuit.get("opened_at", 0)
            timeout = circuit.get("timeout", 60)
            if time.time() - opened_at > timeout:
                circuit["state"] = "half-open"
                return False
            return True
        return False

    def _record_circuit_failure(self, plugin_id: str) -> None:
        """Record a failure for circuit breaker."""
        if plugin_id not in self._circuit_breaker:
            self._circuit_breaker[plugin_id] = {
                "failures": 0,
                "state": "closed",
                "threshold": 5,
                "timeout": 60,
            }

        circuit = self._circuit_breaker[plugin_id]
        circuit["failures"] = circuit.get("failures", 0) + 1
        circuit["last_failure"] = time.time()

        if circuit["failures"] >= circuit.get("threshold", 5):
            circuit["state"] = "open"
            circuit["opened_at"] = time.time()

    def _reset_circuit_breaker(self, plugin_id: str) -> None:
        """Reset circuit breaker on success."""
        if plugin_id in self._circuit_breaker:
            self._circuit_breaker[plugin_id]["failures"] = 0
            self._circuit_breaker[plugin_id]["state"] = "closed"

    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin."""
        registration = self._registry.get(plugin_id)
        if registration:
            registration.config.enabled = True
            self._save_state()
            return True
        return False

    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin."""
        registration = self._registry.get(plugin_id)
        if registration:
            registration.config.enabled = False
            self._save_state()
            return True
        return False

    def update_config(self, plugin_id: str, config: PluginConfig) -> bool:
        """Update plugin configuration."""
        registration = self._registry.get(plugin_id)
        if registration:
            registration.config = config
            self._save_state()
            return True
        return False

    def get_plugin(self, plugin_id: str) -> Optional[PluginRegistration]:
        """Get plugin registration."""
        return self._registry.get(plugin_id)

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        state: Optional[PluginState] = None,
    ) -> List[PluginRegistration]:
        """List plugins with optional filters."""
        plugins = self._registry.list_all()

        if plugin_type:
            plugins = [p for p in plugins if p.metadata.plugin_type == plugin_type]

        if state:
            plugins = [p for p in plugins if p.state == state]

        return plugins

    def list_by_hook(self, hook: PluginHook) -> List[PluginRegistration]:
        """List plugins registered for a hook."""
        return self._registry.get_by_hook(hook)

    def _discover_plugins(self) -> int:
        """Auto-discover plugins in configured directories."""
        discovered = 0

        for plugin_dir in self.plugin_dirs:
            dir_path = Path(plugin_dir)
            if not dir_path.exists():
                continue

            for py_file in dir_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                # Try to discover plugin class
                try:
                    spec = importlib.util.spec_from_file_location("plugin_mod", str(py_file))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        for name in dir(module):
                            obj = getattr(module, name)
                            if (
                                isinstance(obj, type)
                                and issubclass(obj, PluginInterface)
                                and obj is not PluginInterface
                            ):
                                self.register_class(obj)
                                discovered += 1
                except Exception:
                    continue

        return discovered

    def _save_state(self) -> None:
        """Save plugin state to disk."""
        state = {
            "plugins": {},
        }

        for reg in self._registry.list_all():
            state["plugins"][reg.metadata.plugin_id] = {
                "metadata": reg.metadata.to_dict(),
                "config": reg.config.to_dict(),
                "state": reg.state.value,
                "source_path": reg.source_path,
                "load_count": reg.load_count,
                "error_count": reg.error_count,
                "last_error": reg.last_error,
            }

        state_file = self.state_dir / "plugins_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load plugin state from disk."""
        state_file = self.state_dir / "plugins_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Note: We only load metadata/config, not instances
            # Plugins need to be re-registered and loaded
            for plugin_id, data in state.get("plugins", {}).items():
                metadata = PluginMetadata.from_dict(data["metadata"])
                config = PluginConfig.from_dict(data["config"])
                registration = self._registry.register(metadata, config)
                registration.source_path = data.get("source_path")
                registration.load_count = data.get("load_count", 0)
                registration.error_count = data.get("error_count", 0)
                registration.last_error = data.get("last_error")
                registration.state = PluginState.REGISTERED

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for plugin system."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Plugin System (Step 231)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register command
    register_parser = subparsers.add_parser("register", help="Register a plugin from file")
    register_parser.add_argument("file_path", help="Path to plugin Python file")
    register_parser.add_argument("--class", "-c", dest="class_name", required=True, help="Plugin class name")
    register_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List plugins")
    list_parser.add_argument("--type", "-t", choices=[t.value for t in PluginType], help="Filter by type")
    list_parser.add_argument("--state", "-s", choices=[s.value for s in PluginState], help="Filter by state")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # load command
    load_parser = subparsers.add_parser("load", help="Load a plugin")
    load_parser.add_argument("plugin_id", help="Plugin ID")
    load_parser.add_argument("--json", action="store_true", help="JSON output")

    # unload command
    unload_parser = subparsers.add_parser("unload", help="Unload a plugin")
    unload_parser.add_argument("plugin_id", help="Plugin ID")

    # enable command
    enable_parser = subparsers.add_parser("enable", help="Enable a plugin")
    enable_parser.add_argument("plugin_id", help="Plugin ID")

    # disable command
    disable_parser = subparsers.add_parser("disable", help="Disable a plugin")
    disable_parser.add_argument("plugin_id", help="Plugin ID")

    # info command
    info_parser = subparsers.add_parser("info", help="Get plugin info")
    info_parser.add_argument("plugin_id", help="Plugin ID")
    info_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    system = DeployPluginSystem()

    if args.command == "register":
        registration = system.register_from_file(
            args.file_path,
            args.class_name,
        )

        if registration:
            if args.json:
                print(json.dumps({
                    "plugin_id": registration.metadata.plugin_id,
                    "name": registration.metadata.name,
                    "state": registration.state.value,
                }, indent=2))
            else:
                print(f"Registered plugin: {registration.metadata.plugin_id}")
                print(f"  Name: {registration.metadata.name}")
                print(f"  Type: {registration.metadata.plugin_type.value}")
        else:
            print("Failed to register plugin")
            return 1

        return 0

    elif args.command == "list":
        plugin_type = PluginType(args.type) if args.type else None
        state = PluginState(args.state) if args.state else None

        plugins = system.list_plugins(plugin_type=plugin_type, state=state)

        if args.json:
            print(json.dumps([{
                "plugin_id": p.metadata.plugin_id,
                "name": p.metadata.name,
                "type": p.metadata.plugin_type.value,
                "state": p.state.value,
                "enabled": p.config.enabled,
            } for p in plugins], indent=2))
        else:
            if not plugins:
                print("No plugins registered")
            else:
                for p in plugins:
                    status = "enabled" if p.config.enabled else "disabled"
                    print(f"{p.metadata.plugin_id} ({p.metadata.name}) - {p.state.value} [{status}]")

        return 0

    elif args.command == "load":
        success = asyncio.get_event_loop().run_until_complete(
            system.load_plugin(args.plugin_id)
        )

        if args.json:
            print(json.dumps({"success": success, "plugin_id": args.plugin_id}))
        else:
            if success:
                print(f"Loaded plugin: {args.plugin_id}")
            else:
                print(f"Failed to load plugin: {args.plugin_id}")
                return 1

        return 0

    elif args.command == "unload":
        success = asyncio.get_event_loop().run_until_complete(
            system.unload_plugin(args.plugin_id)
        )

        if success:
            print(f"Unloaded plugin: {args.plugin_id}")
        else:
            print(f"Failed to unload plugin: {args.plugin_id}")
            return 1

        return 0

    elif args.command == "enable":
        success = system.enable_plugin(args.plugin_id)

        if success:
            print(f"Enabled plugin: {args.plugin_id}")
        else:
            print(f"Plugin not found: {args.plugin_id}")
            return 1

        return 0

    elif args.command == "disable":
        success = system.disable_plugin(args.plugin_id)

        if success:
            print(f"Disabled plugin: {args.plugin_id}")
        else:
            print(f"Plugin not found: {args.plugin_id}")
            return 1

        return 0

    elif args.command == "info":
        plugin = system.get_plugin(args.plugin_id)

        if not plugin:
            print(f"Plugin not found: {args.plugin_id}")
            return 1

        if args.json:
            print(json.dumps({
                "metadata": plugin.metadata.to_dict(),
                "config": plugin.config.to_dict(),
                "state": plugin.state.value,
                "load_count": plugin.load_count,
                "error_count": plugin.error_count,
                "last_error": plugin.last_error,
            }, indent=2))
        else:
            print(f"Plugin: {plugin.metadata.plugin_id}")
            print(f"  Name: {plugin.metadata.name}")
            print(f"  Version: {plugin.metadata.version}")
            print(f"  Type: {plugin.metadata.plugin_type.value}")
            print(f"  State: {plugin.state.value}")
            print(f"  Enabled: {plugin.config.enabled}")
            print(f"  Hooks: {', '.join(h.value for h in plugin.metadata.hooks)}")
            print(f"  Load count: {plugin.load_count}")
            print(f"  Error count: {plugin.error_count}")
            if plugin.last_error:
                print(f"  Last error: {plugin.last_error}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
