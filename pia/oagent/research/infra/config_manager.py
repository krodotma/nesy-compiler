#!/usr/bin/env python3
"""
config_manager.py - Configuration Management (Step 36)

Centralized configuration management with multiple sources,
validation, and hot reloading support.

PBTSO Phase: CONFIGURE

Bus Topics:
- a2a.research.config.change
- a2a.research.config.load
- research.config.validate

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class ConfigSource(Enum):
    """Configuration sources in priority order."""
    DEFAULT = "default"         # Built-in defaults
    FILE = "file"               # Configuration files
    ENV = "env"                 # Environment variables
    RUNTIME = "runtime"         # Runtime overrides
    REMOTE = "remote"           # Remote configuration service


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    PATH = "path"


@dataclass
class ConfigSchema:
    """Schema for a configuration value."""

    key: str
    type: ConfigType
    default: Any = None
    required: bool = False
    description: str = ""
    env_var: Optional[str] = None
    validator: Optional[Callable[[Any], bool]] = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    sensitive: bool = False  # Mask in logs

    def validate(self, value: Any) -> bool:
        """Validate a value against this schema."""
        if value is None:
            return not self.required

        # Type check
        if self.type == ConfigType.STRING and not isinstance(value, str):
            return False
        elif self.type == ConfigType.INT and not isinstance(value, int):
            return False
        elif self.type == ConfigType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.type == ConfigType.BOOL and not isinstance(value, bool):
            return False
        elif self.type == ConfigType.LIST and not isinstance(value, list):
            return False
        elif self.type == ConfigType.DICT and not isinstance(value, dict):
            return False
        elif self.type == ConfigType.PATH:
            if not isinstance(value, (str, Path)):
                return False

        # Range check
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False

        # Choices check
        if self.choices is not None and value not in self.choices:
            return False

        # Custom validator
        if self.validator is not None and not self.validator(value):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "type": self.type.value,
            "default": self.default,
            "required": self.required,
            "description": self.description,
            "env_var": self.env_var,
            "choices": self.choices,
            "sensitive": self.sensitive,
        }


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ConfigValue:
    """A configuration value with metadata."""

    key: str
    value: Any
    source: ConfigSource
    schema: Optional[ConfigSchema] = None
    timestamp: float = field(default_factory=time.time)
    version: int = 1

    def to_dict(self, mask_sensitive: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        val = self.value
        if mask_sensitive and self.schema and self.schema.sensitive:
            val = "***MASKED***"

        return {
            "key": self.key,
            "value": val,
            "source": self.source.value,
            "timestamp": self.timestamp,
            "version": self.version,
        }


@dataclass
class ConfigManagerConfig:
    """Configuration for the config manager."""

    config_dir: Optional[str] = None
    config_file: str = "research.json"
    env_prefix: str = "RESEARCH_"
    enable_hot_reload: bool = True
    reload_interval_seconds: int = 30
    validate_on_load: bool = True
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.config_dir is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.config_dir = f"{pluribus_root}/.pluribus/research/config"
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Config Manager
# ============================================================================


T = TypeVar("T")


class ConfigManager:
    """
    Centralized configuration management.

    Features:
    - Multiple configuration sources (files, env, runtime)
    - Schema validation
    - Type coercion
    - Hot reloading
    - Change notifications
    - Secret masking

    PBTSO Phase: CONFIGURE

    Example:
        manager = ConfigManager()

        # Define schema
        manager.define("search.max_results", ConfigType.INT, default=50)
        manager.define("cache.enabled", ConfigType.BOOL, default=True)

        # Get values
        max_results = manager.get("search.max_results")

        # Override at runtime
        manager.set("search.max_results", 100)

        # Watch for changes
        manager.watch("search.*", callback)
    """

    # Default research agent configuration schema
    DEFAULT_SCHEMA: List[ConfigSchema] = [
        ConfigSchema("agent.id", ConfigType.STRING, "research-agent"),
        ConfigSchema("agent.ring_level", ConfigType.INT, 2, min_value=0, max_value=3),
        ConfigSchema("agent.max_context_tokens", ConfigType.INT, 100000),

        ConfigSchema("search.max_results", ConfigType.INT, 50, min_value=1, max_value=1000),
        ConfigSchema("search.timeout_seconds", ConfigType.INT, 30, min_value=1, max_value=300),
        ConfigSchema("search.enable_semantic", ConfigType.BOOL, True),

        ConfigSchema("cache.enabled", ConfigType.BOOL, True),
        ConfigSchema("cache.ttl_seconds", ConfigType.INT, 3600),
        ConfigSchema("cache.max_items", ConfigType.INT, 10000),

        ConfigSchema("index.batch_size", ConfigType.INT, 500),
        ConfigSchema("index.parallel_workers", ConfigType.INT, 4, min_value=1, max_value=32),

        ConfigSchema("log.level", ConfigType.STRING, "INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]),
        ConfigSchema("log.format", ConfigType.STRING, "json", choices=["json", "text"]),

        ConfigSchema("api.host", ConfigType.STRING, "127.0.0.1"),
        ConfigSchema("api.port", ConfigType.INT, 8080, min_value=1, max_value=65535),
        ConfigSchema("api.rate_limit", ConfigType.INT, 100),

        ConfigSchema("falkordb.host", ConfigType.STRING, "localhost"),
        ConfigSchema("falkordb.port", ConfigType.INT, 6380),

        ConfigSchema("bus.heartbeat_interval", ConfigType.INT, 300, description="A2A heartbeat: 300s interval"),
        ConfigSchema("bus.heartbeat_timeout", ConfigType.INT, 900, description="A2A heartbeat: 900s timeout"),
        ConfigSchema("bus.ndjson_rotation_mb", ConfigType.INT, 10, description="NDJSON rotation: 10MB default"),
    ]

    def __init__(
        self,
        config: Optional[ConfigManagerConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the config manager.

        Args:
            config: Manager configuration
            bus: AgentBus for event emission
        """
        self.config = config or ConfigManagerConfig()
        self.bus = bus or AgentBus()

        # Storage
        self._values: Dict[str, ConfigValue] = {}
        self._schemas: Dict[str, ConfigSchema] = {}
        self._watchers: Dict[str, List[Callable[[str, Any, Any], None]]] = {}
        self._lock = threading.RLock()

        # Hot reload
        self._reload_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._file_mtimes: Dict[str, float] = {}

        # Register default schema
        for schema in self.DEFAULT_SCHEMA:
            self.define_schema(schema)

        # Load initial configuration
        self._load_defaults()
        self._load_from_files()
        self._load_from_env()

    def define(
        self,
        key: str,
        type: ConfigType,
        default: Any = None,
        **kwargs,
    ) -> None:
        """
        Define a configuration key.

        Args:
            key: Configuration key
            type: Value type
            default: Default value
            **kwargs: Additional schema options
        """
        schema = ConfigSchema(key=key, type=type, default=default, **kwargs)
        self.define_schema(schema)

    def define_schema(self, schema: ConfigSchema) -> None:
        """Register a configuration schema."""
        with self._lock:
            self._schemas[schema.key] = schema

            # Set default if not already set
            if schema.key not in self._values and schema.default is not None:
                self._values[schema.key] = ConfigValue(
                    key=schema.key,
                    value=schema.default,
                    source=ConfigSource.DEFAULT,
                    schema=schema,
                )

    def get(self, key: str, default: T = None) -> T:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default if not found

        Returns:
            Configuration value
        """
        with self._lock:
            if key in self._values:
                return self._values[key].value

            # Check schema default
            if key in self._schemas:
                return self._schemas[key].default

            return default

    def get_typed(self, key: str, type_class: type, default: T = None) -> T:
        """Get a configuration value with type coercion."""
        value = self.get(key)
        if value is None:
            return default

        try:
            return type_class(value)
        except (ValueError, TypeError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        return self.get_typed(key, int, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get list configuration value."""
        value = self.get(key)
        if value is None:
            return default or []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",")]
        return [value]

    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.RUNTIME,
    ) -> bool:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
            source: Configuration source

        Returns:
            True if successfully set
        """
        with self._lock:
            # Validate if schema exists
            schema = self._schemas.get(key)
            if schema and self.config.validate_on_load:
                if not schema.validate(value):
                    return False

            # Get old value for change notification
            old_value = self._values.get(key)
            old_val = old_value.value if old_value else None

            # Set new value
            if key in self._values:
                self._values[key].value = value
                self._values[key].source = source
                self._values[key].timestamp = time.time()
                self._values[key].version += 1
            else:
                self._values[key] = ConfigValue(
                    key=key,
                    value=value,
                    source=source,
                    schema=schema,
                )

            # Notify watchers
            if value != old_val:
                self._notify_watchers(key, old_val, value)

                # Emit to bus
                if self.config.emit_to_bus:
                    self._emit_change(key, old_val, value)

            return True

    def delete(self, key: str) -> bool:
        """Delete a configuration key."""
        with self._lock:
            if key in self._values:
                del self._values[key]
                return True
            return False

    def has(self, key: str) -> bool:
        """Check if configuration key exists."""
        return key in self._values or key in self._schemas

    def watch(
        self,
        pattern: str,
        callback: Callable[[str, Any, Any], None],
    ) -> None:
        """
        Watch for configuration changes.

        Args:
            pattern: Key pattern (supports * wildcard)
            callback: Callback function(key, old_value, new_value)
        """
        with self._lock:
            if pattern not in self._watchers:
                self._watchers[pattern] = []
            self._watchers[pattern].append(callback)

    def unwatch(self, pattern: str, callback: Callable) -> bool:
        """Remove a watcher."""
        with self._lock:
            if pattern in self._watchers and callback in self._watchers[pattern]:
                self._watchers[pattern].remove(callback)
                return True
            return False

    def reload(self) -> int:
        """
        Reload configuration from all sources.

        Returns:
            Number of values changed
        """
        changed = 0

        # Reload from files
        changed += self._load_from_files()

        # Reload from env
        changed += self._load_from_env()

        return changed

    def start_hot_reload(self) -> None:
        """Start hot reload thread."""
        if self._reload_thread is not None:
            return

        self._stop_event.clear()
        self._reload_thread = threading.Thread(target=self._reload_loop, daemon=True)
        self._reload_thread.start()

    def stop_hot_reload(self) -> None:
        """Stop hot reload thread."""
        self._stop_event.set()
        if self._reload_thread:
            self._reload_thread.join(timeout=5)
            self._reload_thread = None

    def get_all(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """Get all configuration values."""
        with self._lock:
            result = {}
            for key, cv in self._values.items():
                if prefix is None or key.startswith(prefix):
                    result[key] = cv.value
            return result

    def get_metadata(self, key: str) -> Optional[ConfigValue]:
        """Get configuration value with metadata."""
        with self._lock:
            return self._values.get(key)

    def export(self, mask_sensitive: bool = True) -> Dict[str, Any]:
        """Export all configuration."""
        with self._lock:
            return {
                key: cv.to_dict(mask_sensitive)
                for key, cv in self._values.items()
            }

    def validate_all(self) -> Dict[str, str]:
        """Validate all configuration values."""
        errors = {}

        with self._lock:
            for key, schema in self._schemas.items():
                value = self._values.get(key)
                val = value.value if value else None

                if schema.required and val is None:
                    errors[key] = "Required value missing"
                elif val is not None and not schema.validate(val):
                    errors[key] = f"Invalid value: {val}"

        return errors

    def _load_defaults(self) -> None:
        """Load default values from schema."""
        for key, schema in self._schemas.items():
            if schema.default is not None and key not in self._values:
                self._values[key] = ConfigValue(
                    key=key,
                    value=schema.default,
                    source=ConfigSource.DEFAULT,
                    schema=schema,
                )

    def _load_from_files(self) -> int:
        """Load configuration from files."""
        changed = 0
        config_dir = Path(self.config.config_dir)

        if not config_dir.exists():
            return 0

        config_file = config_dir / self.config.config_file

        if config_file.exists():
            # Check if file changed
            mtime = config_file.stat().st_mtime
            if str(config_file) in self._file_mtimes:
                if mtime <= self._file_mtimes[str(config_file)]:
                    return 0

            self._file_mtimes[str(config_file)] = mtime

            try:
                with open(config_file) as f:
                    data = json.load(f)

                for key, value in self._flatten_dict(data).items():
                    if self.set(key, value, ConfigSource.FILE):
                        changed += 1

            except Exception:
                pass

        return changed

    def _load_from_env(self) -> int:
        """Load configuration from environment variables."""
        changed = 0

        for key, schema in self._schemas.items():
            # Check custom env var name
            env_var = schema.env_var or f"{self.config.env_prefix}{key.upper().replace('.', '_')}"

            if env_var in os.environ:
                value = os.environ[env_var]

                # Type coercion
                if schema.type == ConfigType.INT:
                    value = int(value)
                elif schema.type == ConfigType.FLOAT:
                    value = float(value)
                elif schema.type == ConfigType.BOOL:
                    value = value.lower() in ("true", "1", "yes", "on")
                elif schema.type == ConfigType.LIST:
                    value = [v.strip() for v in value.split(",")]

                if self.set(key, value, ConfigSource.ENV):
                    changed += 1

        return changed

    def _flatten_dict(self, d: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary to dot-notation keys."""
        items = {}
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    def _notify_watchers(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify watchers of configuration change."""
        for pattern, callbacks in self._watchers.items():
            if self._matches_pattern(key, pattern):
                for callback in callbacks:
                    try:
                        callback(key, old_value, new_value)
                    except Exception:
                        pass

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        return key == pattern

    def _reload_loop(self) -> None:
        """Background reload loop."""
        while not self._stop_event.wait(self.config.reload_interval_seconds):
            self.reload()

    def _emit_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Emit configuration change event."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Mask sensitive values
        schema = self._schemas.get(key)
        if schema and schema.sensitive:
            old_value = "***MASKED***"
            new_value = "***MASKED***"

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": "a2a.research.config.change",
            "kind": "config",
            "level": "info",
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "key": key,
                "old_value": old_value,
                "new_value": new_value,
            },
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ============================================================================
# Global Config Manager
# ============================================================================


_default_manager: Optional[ConfigManager] = None


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value from default manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConfigManager()
    return _default_manager.get(key, default)


def set_config(key: str, value: Any) -> bool:
    """Set configuration value in default manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConfigManager()
    return _default_manager.set(key, value)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Config Manager."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Config Manager (Step 36)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Get command
    get_parser = subparsers.add_parser("get", help="Get configuration value")
    get_parser.add_argument("key", help="Configuration key")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", help="Configuration value")

    # List command
    list_parser = subparsers.add_parser("list", help="List all configuration")
    list_parser.add_argument("--prefix", help="Filter by prefix")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")

    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Show configuration schema")
    schema_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manager = ConfigManager()

    if args.command == "get":
        value = manager.get(args.key)
        if value is not None:
            print(f"{args.key} = {value}")
        else:
            print(f"Key not found: {args.key}")
            return 1

    elif args.command == "set":
        if manager.set(args.key, args.value):
            print(f"Set {args.key} = {args.value}")
        else:
            print(f"Failed to set {args.key}")
            return 1

    elif args.command == "list":
        config = manager.get_all(args.prefix)

        if args.json:
            print(json.dumps(config, indent=2))
        else:
            print("Configuration:")
            for key in sorted(config.keys()):
                meta = manager.get_metadata(key)
                source = meta.source.value if meta else "?"
                print(f"  {key} = {config[key]} [{source}]")

    elif args.command == "validate":
        errors = manager.validate_all()

        if errors:
            print("Validation errors:")
            for key, error in errors.items():
                print(f"  {key}: {error}")
            return 1
        else:
            print("All configuration valid")

    elif args.command == "schema":
        schemas = [s.to_dict() for s in manager._schemas.values()]

        if args.json:
            print(json.dumps(schemas, indent=2))
        else:
            print("Configuration Schema:")
            for s in sorted(schemas, key=lambda x: x["key"]):
                req = "*" if s["required"] else ""
                print(f"  {s['key']}{req}: {s['type']} (default: {s['default']})")
                if s["description"]:
                    print(f"    {s['description']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
