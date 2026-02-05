#!/usr/bin/env python3
"""
Monitor Config Manager - Step 286

Configuration management for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.config.loaded (emitted)
- monitor.config.changed (emitted)
- monitor.config.error (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
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
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union


class ConfigSource(Enum):
    """Configuration sources."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    OVERRIDE = "override"


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


@dataclass
class ConfigSchema:
    """Schema for a configuration value.

    Attributes:
        key: Configuration key
        config_type: Value type
        default: Default value
        description: Description
        required: Whether required
        min_value: Minimum value (for numbers)
        max_value: Maximum value (for numbers)
        choices: Valid choices
        sensitive: Whether value is sensitive
    """
    key: str
    config_type: ConfigType
    default: Any = None
    description: str = ""
    required: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    sensitive: bool = False

    def validate(self, value: Any) -> bool:
        """Validate a value against the schema.

        Args:
            value: Value to validate

        Returns:
            True if valid
        """
        if value is None:
            return not self.required

        # Type validation
        if self.config_type == ConfigType.STRING:
            if not isinstance(value, str):
                return False
        elif self.config_type == ConfigType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False
        elif self.config_type == ConfigType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
        elif self.config_type == ConfigType.BOOLEAN:
            if not isinstance(value, bool):
                return False
        elif self.config_type == ConfigType.LIST:
            if not isinstance(value, list):
                return False
        elif self.config_type == ConfigType.DICT:
            if not isinstance(value, dict):
                return False

        # Range validation
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False

        # Choices validation
        if self.choices is not None and value not in self.choices:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "type": self.config_type.value,
            "default": self.default if not self.sensitive else "***",
            "description": self.description,
            "required": self.required,
            "sensitive": self.sensitive,
        }


@dataclass
class ConfigValue:
    """A configuration value.

    Attributes:
        key: Configuration key
        value: Configuration value
        source: Value source
        timestamp: When value was set
    """
    key: str
    value: Any
    source: ConfigSource
    timestamp: float = field(default_factory=time.time)

    def to_dict(self, hide_sensitive: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value if not hide_sensitive else "***",
            "source": self.source.value,
            "timestamp": self.timestamp,
        }


class MonitorConfigManager:
    """
    Configuration management for the Monitor Agent.

    Provides:
    - Hierarchical configuration (defaults, file, env, remote)
    - Schema validation
    - Change notifications
    - Configuration reloading
    - Sensitive value handling

    Example:
        manager = MonitorConfigManager()

        # Define schema
        manager.define("metrics.retention_days", ConfigType.INTEGER, default=30)

        # Get configuration
        retention = manager.get("metrics.retention_days")

        # Set configuration
        manager.set("metrics.retention_days", 7)

        # Watch for changes
        manager.watch("metrics.*", lambda key, value: print(f"{key} changed"))
    """

    BUS_TOPICS = {
        "loaded": "monitor.config.loaded",
        "changed": "monitor.config.changed",
        "error": "monitor.config.error",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    # Default configuration schema
    DEFAULT_SCHEMA = [
        ConfigSchema("agent.id", ConfigType.STRING, default="monitor-agent"),
        ConfigSchema("agent.ring_level", ConfigType.INTEGER, default=1, min_value=0, max_value=3),
        ConfigSchema("metrics.retention_days", ConfigType.INTEGER, default=30, min_value=1, max_value=365),
        ConfigSchema("metrics.collection_interval_s", ConfigType.INTEGER, default=60, min_value=1),
        ConfigSchema("alerts.enabled", ConfigType.BOOLEAN, default=True),
        ConfigSchema("alerts.channels", ConfigType.LIST, default=["bus", "slack"]),
        ConfigSchema("slo.tracking_enabled", ConfigType.BOOLEAN, default=True),
        ConfigSchema("anomaly.window_size", ConfigType.INTEGER, default=100, min_value=10),
        ConfigSchema("heartbeat.interval_s", ConfigType.INTEGER, default=300, min_value=60),
        ConfigSchema("heartbeat.timeout_s", ConfigType.INTEGER, default=900, min_value=180),
        ConfigSchema("bus.dir", ConfigType.STRING),
        ConfigSchema("cache.l1_max_entries", ConfigType.INTEGER, default=10000),
        ConfigSchema("cache.l1_ttl_s", ConfigType.INTEGER, default=60),
        ConfigSchema("log.level", ConfigType.STRING, default="info", choices=["debug", "info", "warning", "error"]),
        ConfigSchema("log.format", ConfigType.STRING, default="text", choices=["text", "json"]),
    ]

    def __init__(
        self,
        config_path: Optional[str] = None,
        env_prefix: str = "MONITOR_",
        auto_reload: bool = True,
        bus_dir: Optional[str] = None,
    ):
        """Initialize config manager.

        Args:
            config_path: Path to config file
            env_prefix: Environment variable prefix
            auto_reload: Enable auto-reload on file change
            bus_dir: Bus directory
        """
        self._env_prefix = env_prefix
        self._auto_reload = auto_reload
        self._last_heartbeat = time.time()

        # Configuration storage
        self._schemas: Dict[str, ConfigSchema] = {}
        self._values: Dict[str, ConfigValue] = {}
        self._watchers: Dict[str, List[Callable[[str, Any], None]]] = {}
        self._lock = threading.RLock()

        # Config file
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._config_path = config_path or os.path.join(
            pluribus_root, ".pluribus", "monitor", "config.json"
        )

        # Bus path
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize default schema
        for schema in self.DEFAULT_SCHEMA:
            self._schemas[schema.key] = schema

        # Load configuration
        self._load_all()

    def define(
        self,
        key: str,
        config_type: ConfigType,
        default: Any = None,
        description: str = "",
        required: bool = False,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        choices: Optional[List[Any]] = None,
        sensitive: bool = False,
    ) -> None:
        """Define a configuration schema.

        Args:
            key: Configuration key
            config_type: Value type
            default: Default value
            description: Description
            required: Whether required
            min_value: Minimum value
            max_value: Maximum value
            choices: Valid choices
            sensitive: Whether sensitive
        """
        schema = ConfigSchema(
            key=key,
            config_type=config_type,
            default=default,
            description=description,
            required=required,
            min_value=min_value,
            max_value=max_value,
            choices=choices,
            sensitive=sensitive,
        )

        with self._lock:
            self._schemas[key] = schema

            # Set default value if not already set
            if key not in self._values and default is not None:
                self._values[key] = ConfigValue(
                    key=key,
                    value=default,
                    source=ConfigSource.DEFAULT,
                )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

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

    def get_typed(self, key: str, value_type: Type[T], default: Optional[T] = None) -> Optional[T]:
        """Get a typed configuration value.

        Args:
            key: Configuration key
            value_type: Expected type
            default: Default if not found

        Returns:
            Configuration value
        """
        value = self.get(key, default)
        if value is not None and not isinstance(value, value_type):
            try:
                return value_type(value)
            except Exception:
                return default
        return value

    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.OVERRIDE,
    ) -> bool:
        """Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
            source: Value source

        Returns:
            True if set successfully
        """
        with self._lock:
            # Validate against schema
            if key in self._schemas:
                if not self._schemas[key].validate(value):
                    self._emit_bus_event(
                        self.BUS_TOPICS["error"],
                        {
                            "key": key,
                            "error": "Validation failed",
                        },
                        level="error",
                    )
                    return False

            old_value = self._values.get(key)
            self._values[key] = ConfigValue(
                key=key,
                value=value,
                source=source,
            )

        # Notify watchers
        self._notify_watchers(key, value, old_value.value if old_value else None)

        # Emit change event
        self._emit_bus_event(
            self.BUS_TOPICS["changed"],
            {
                "key": key,
                "source": source.value,
            },
        )

        return True

    def delete(self, key: str) -> bool:
        """Delete a configuration value.

        Args:
            key: Configuration key

        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._values:
                del self._values[key]
                return True
            return False

    def watch(
        self,
        pattern: str,
        callback: Callable[[str, Any], None],
    ) -> None:
        """Watch for configuration changes.

        Args:
            pattern: Key pattern (supports * wildcard)
            callback: Callback function
        """
        with self._lock:
            if pattern not in self._watchers:
                self._watchers[pattern] = []
            self._watchers[pattern].append(callback)

    def unwatch(
        self,
        pattern: str,
        callback: Callable[[str, Any], None],
    ) -> bool:
        """Remove a watcher.

        Args:
            pattern: Key pattern
            callback: Callback to remove

        Returns:
            True if removed
        """
        with self._lock:
            if pattern in self._watchers:
                if callback in self._watchers[pattern]:
                    self._watchers[pattern].remove(callback)
                    return True
            return False

    def reload(self) -> bool:
        """Reload configuration from all sources.

        Returns:
            True if successful
        """
        try:
            self._load_all()
            self._emit_bus_event(
                self.BUS_TOPICS["loaded"],
                {"source": "reload"},
            )
            return True
        except Exception as e:
            self._emit_bus_event(
                self.BUS_TOPICS["error"],
                {"error": str(e)},
                level="error",
            )
            return False

    def save(self, path: Optional[str] = None) -> bool:
        """Save configuration to file.

        Args:
            path: File path (uses default if None)

        Returns:
            True if saved
        """
        file_path = path or self._config_path
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Only save override values
            config = {}
            for key, cv in self._values.items():
                if cv.source == ConfigSource.OVERRIDE:
                    config[key] = cv.value

        try:
            with open(file_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(config, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception:
            return False

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all configuration keys.

        Args:
            prefix: Filter by prefix

        Returns:
            List of keys
        """
        with self._lock:
            keys = set(self._schemas.keys()) | set(self._values.keys())
            if prefix:
                keys = {k for k in keys if k.startswith(prefix)}
            return sorted(keys)

    def get_schema(self, key: str) -> Optional[ConfigSchema]:
        """Get schema for a key.

        Args:
            key: Configuration key

        Returns:
            Schema or None
        """
        return self._schemas.get(key)

    def get_all(
        self,
        prefix: Optional[str] = None,
        include_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """Get all configuration values.

        Args:
            prefix: Filter by prefix
            include_sensitive: Include sensitive values

        Returns:
            Configuration dictionary
        """
        with self._lock:
            result = {}
            for key in self.list_keys(prefix):
                value = self.get(key)
                schema = self._schemas.get(key)

                if schema and schema.sensitive and not include_sensitive:
                    result[key] = "***"
                else:
                    result[key] = value

            return result

    def get_info(self, key: str) -> Dict[str, Any]:
        """Get detailed info about a configuration key.

        Args:
            key: Configuration key

        Returns:
            Configuration info
        """
        with self._lock:
            result = {
                "key": key,
                "value": self.get(key),
            }

            if key in self._values:
                cv = self._values[key]
                result["source"] = cv.source.value
                result["timestamp"] = cv.timestamp

            if key in self._schemas:
                result["schema"] = self._schemas[key].to_dict()

            return result

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration values.

        Returns:
            Dictionary of validation errors
        """
        errors: Dict[str, List[str]] = {}

        with self._lock:
            for key, schema in self._schemas.items():
                value = self.get(key)
                key_errors = []

                if schema.required and value is None:
                    key_errors.append("Required value is missing")
                elif value is not None and not schema.validate(value):
                    key_errors.append(f"Invalid value: {value}")

                if key_errors:
                    errors[key] = key_errors

        return errors

    def _load_all(self) -> None:
        """Load configuration from all sources."""
        # 1. Load defaults from schema
        with self._lock:
            for key, schema in self._schemas.items():
                if schema.default is not None and key not in self._values:
                    self._values[key] = ConfigValue(
                        key=key,
                        value=schema.default,
                        source=ConfigSource.DEFAULT,
                    )

        # 2. Load from file
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r") as f:
                    config = json.load(f)
                    with self._lock:
                        for key, value in config.items():
                            self._values[key] = ConfigValue(
                                key=key,
                                value=value,
                                source=ConfigSource.FILE,
                            )
            except Exception:
                pass

        # 3. Load from environment
        for key in self._schemas:
            env_key = self._env_prefix + key.upper().replace(".", "_")
            env_value = os.environ.get(env_key)
            if env_value is not None:
                # Parse value based on schema type
                schema = self._schemas[key]
                parsed = self._parse_env_value(env_value, schema.config_type)
                if parsed is not None:
                    with self._lock:
                        self._values[key] = ConfigValue(
                            key=key,
                            value=parsed,
                            source=ConfigSource.ENVIRONMENT,
                        )

    def _parse_env_value(
        self,
        value: str,
        config_type: ConfigType,
    ) -> Optional[Any]:
        """Parse environment variable value.

        Args:
            value: String value
            config_type: Expected type

        Returns:
            Parsed value
        """
        try:
            if config_type == ConfigType.STRING:
                return value
            elif config_type == ConfigType.INTEGER:
                return int(value)
            elif config_type == ConfigType.FLOAT:
                return float(value)
            elif config_type == ConfigType.BOOLEAN:
                return value.lower() in ("true", "1", "yes")
            elif config_type == ConfigType.LIST:
                return json.loads(value)
            elif config_type == ConfigType.DICT:
                return json.loads(value)
        except Exception:
            pass
        return None

    def _notify_watchers(
        self,
        key: str,
        new_value: Any,
        old_value: Any,
    ) -> None:
        """Notify watchers of a change.

        Args:
            key: Configuration key
            new_value: New value
            old_value: Old value
        """
        if new_value == old_value:
            return

        with self._lock:
            for pattern, callbacks in self._watchers.items():
                if self._pattern_matches(key, pattern):
                    for callback in callbacks:
                        try:
                            callback(key, new_value)
                        except Exception:
                            pass

    def _pattern_matches(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern.

        Args:
            key: Configuration key
            pattern: Pattern (supports * wildcard)

        Returns:
            True if matches
        """
        if pattern == "*":
            return True
        if "*" not in pattern:
            return key == pattern
        # Simple prefix matching
        prefix = pattern.rstrip("*")
        return key.startswith(prefix)

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
                "component": "monitor_config_manager",
                "status": "healthy",
                "keys": len(self._values),
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
            "actor": "monitor-config-manager",
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


T = TypeVar("T")

# Singleton instance
_manager: Optional[MonitorConfigManager] = None


def get_config_manager() -> MonitorConfigManager:
    """Get or create the config manager singleton.

    Returns:
        MonitorConfigManager instance
    """
    global _manager
    if _manager is None:
        _manager = MonitorConfigManager()
    return _manager


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Config Manager (Step 286)")
    parser.add_argument("--list", action="store_true", help="List all configuration")
    parser.add_argument("--get", metavar="KEY", help="Get a configuration value")
    parser.add_argument("--set", metavar="KEY=VALUE", help="Set a configuration value")
    parser.add_argument("--info", metavar="KEY", help="Get detailed info about a key")
    parser.add_argument("--validate", action="store_true", help="Validate all configuration")
    parser.add_argument("--save", action="store_true", help="Save configuration to file")
    parser.add_argument("--reload", action="store_true", help="Reload configuration")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    manager = get_config_manager()

    if args.list:
        config = manager.get_all()
        if args.json:
            print(json.dumps(config, indent=2))
        else:
            print("Configuration:")
            for key, value in sorted(config.items()):
                print(f"  {key}: {value}")

    if args.get:
        value = manager.get(args.get)
        if args.json:
            print(json.dumps({"key": args.get, "value": value}))
        else:
            print(f"{args.get}: {value}")

    if args.set:
        key, value = args.set.split("=", 1)
        # Try to parse as JSON
        try:
            value = json.loads(value)
        except Exception:
            pass
        success = manager.set(key, value)
        if args.json:
            print(json.dumps({"key": key, "value": value, "success": success}))
        else:
            print(f"Set {key}={value}: {'success' if success else 'failed'}")

    if args.info:
        info = manager.get_info(args.info)
        if args.json:
            print(json.dumps(info, indent=2, default=str))
        else:
            print(f"Configuration Info: {args.info}")
            for k, v in info.items():
                print(f"  {k}: {v}")

    if args.validate:
        errors = manager.validate_all()
        if args.json:
            print(json.dumps(errors, indent=2))
        else:
            if errors:
                print("Validation Errors:")
                for key, errs in errors.items():
                    for err in errs:
                        print(f"  {key}: {err}")
            else:
                print("All configuration is valid")

    if args.save:
        success = manager.save()
        if args.json:
            print(json.dumps({"saved": success}))
        else:
            print(f"Save: {'success' if success else 'failed'}")

    if args.reload:
        success = manager.reload()
        if args.json:
            print(json.dumps({"reloaded": success}))
        else:
            print(f"Reload: {'success' if success else 'failed'}")
