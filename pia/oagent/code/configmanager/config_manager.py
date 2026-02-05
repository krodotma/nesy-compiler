#!/usr/bin/env python3
"""
config_manager.py - Configuration Management (Step 86)

PBTSO Phase: SKILL, ITERATE

Provides:
- Configuration loading from multiple sources
- Schema validation
- Environment variable override
- Hot reloading
- Configuration change notifications

Bus Topics:
- code.config.loaded
- code.config.changed
- code.config.error

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# Configuration Types
# =============================================================================

class ConfigSource(Enum):
    """Source of configuration value."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    OVERRIDE = "override"
    REMOTE = "remote"


@dataclass
class ConfigValue:
    """A configuration value with metadata."""
    key: str
    value: Any
    source: ConfigSource
    timestamp: float = field(default_factory=time.time)
    schema_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source.value,
            "timestamp": self.timestamp,
            "schema_type": self.schema_type,
        }


@dataclass
class ConfigChangeEvent:
    """Configuration change event."""
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "source": self.source.value,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Configuration Schema
# =============================================================================

@dataclass
class SchemaField:
    """Schema field definition."""
    name: str
    type: str  # string, int, float, bool, list, dict
    required: bool = False
    default: Any = None
    description: str = ""
    env_var: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None


class ConfigSchema:
    """
    Configuration schema for validation.

    Usage:
        schema = ConfigSchema()
        schema.add_field(SchemaField("port", "int", default=8080))
        errors = schema.validate({"port": "invalid"})
    """

    def __init__(self):
        self._fields: Dict[str, SchemaField] = {}

    def add_field(self, field: SchemaField) -> None:
        """Add a field to the schema."""
        self._fields[field.name] = field

    def field(
        self,
        name: str,
        type: str,
        required: bool = False,
        default: Any = None,
        description: str = "",
        env_var: Optional[str] = None,
        **kwargs: Any,
    ) -> "ConfigSchema":
        """Fluent method to add a field."""
        self.add_field(SchemaField(
            name=name,
            type=type,
            required=required,
            default=default,
            description=description,
            env_var=env_var,
            **kwargs,
        ))
        return self

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.

        Returns:
            List of validation errors
        """
        errors = []

        for name, field in self._fields.items():
            value = config.get(name)

            # Check required
            if field.required and value is None:
                errors.append(f"Missing required field: {name}")
                continue

            if value is None:
                continue

            # Type check
            type_valid = self._check_type(value, field.type)
            if not type_valid:
                errors.append(f"Invalid type for {name}: expected {field.type}")
                continue

            # Range check
            if field.min_value is not None and value < field.min_value:
                errors.append(f"{name} below minimum: {value} < {field.min_value}")
            if field.max_value is not None and value > field.max_value:
                errors.append(f"{name} above maximum: {value} > {field.max_value}")

            # Choices check
            if field.choices and value not in field.choices:
                errors.append(f"{name} not in allowed values: {field.choices}")

        return errors

    def _check_type(self, value: Any, type_str: str) -> bool:
        """Check if value matches type."""
        type_map = {
            "string": str,
            "str": str,
            "int": int,
            "integer": int,
            "float": (int, float),
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }

        expected = type_map.get(type_str.lower())
        if expected:
            return isinstance(value, expected)
        return True

    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all fields."""
        return {
            name: field.default
            for name, field in self._fields.items()
            if field.default is not None
        }

    def get_env_mappings(self) -> Dict[str, str]:
        """Get environment variable mappings."""
        return {
            name: field.env_var
            for name, field in self._fields.items()
            if field.env_var
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
# Config Watcher
# =============================================================================

class ConfigWatcher:
    """
    Watch configuration files for changes.

    Enables hot reloading of configuration.
    """

    def __init__(
        self,
        file_path: Path,
        callback: Callable[[Dict[str, Any]], None],
        poll_interval_s: float = 5.0,
    ):
        self.file_path = file_path
        self.callback = callback
        self.poll_interval_s = poll_interval_s
        self._last_mtime: Optional[float] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start watching for changes."""
        if self._running:
            return

        self._running = True
        self._last_mtime = self._get_mtime()
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _get_mtime(self) -> Optional[float]:
        """Get file modification time."""
        if self.file_path.exists():
            return self.file_path.stat().st_mtime
        return None

    async def _watch_loop(self) -> None:
        """Watch loop for file changes."""
        while self._running:
            try:
                await asyncio.sleep(self.poll_interval_s)

                mtime = self._get_mtime()
                if mtime and mtime != self._last_mtime:
                    self._last_mtime = mtime
                    self._reload()

            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def _reload(self) -> None:
        """Reload configuration from file."""
        try:
            content = self.file_path.read_text()
            if self.file_path.suffix in (".yaml", ".yml"):
                if YAML_AVAILABLE:
                    config = yaml.safe_load(content)
                else:
                    return
            else:
                config = json.loads(content)

            self.callback(config)
        except Exception:
            pass


# =============================================================================
# Config Manager
# =============================================================================

class ConfigManager:
    """
    Configuration management system.

    PBTSO Phase: SKILL, ITERATE

    Features:
    - Multi-source configuration (files, env, overrides)
    - Schema validation
    - Environment variable override
    - Hot reloading
    - Change notifications

    Usage:
        config = ConfigManager()
        config.load_file("/path/to/config.json")
        value = config.get("key", default="default")
    """

    BUS_TOPICS = {
        "loaded": "code.config.loaded",
        "changed": "code.config.changed",
        "error": "code.config.error",
    }

    # Default configuration paths
    DEFAULT_PATHS = [
        "/pluribus/pia/oagent/code/config.json",
        "/pluribus/.pluribus/code-config.json",
        "~/.config/pluribus/code-config.json",
    ]

    def __init__(
        self,
        schema: Optional[ConfigSchema] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.schema = schema
        self.bus = bus or LockedAgentBus()
        self._values: Dict[str, ConfigValue] = {}
        self._watchers: List[ConfigWatcher] = []
        self._change_handlers: List[Callable[[ConfigChangeEvent], None]] = []
        self._lock = Lock()

    # =========================================================================
    # Loading
    # =========================================================================

    def load_defaults(self) -> None:
        """Load default values from schema."""
        if not self.schema:
            return

        for key, value in self.schema.get_defaults().items():
            self._set_value(key, value, ConfigSource.DEFAULT)

    def load_file(self, file_path: Union[str, Path]) -> bool:
        """
        Load configuration from file.

        Supports JSON and YAML formats.
        """
        path = Path(file_path).expanduser()

        if not path.exists():
            return False

        try:
            content = path.read_text()

            if path.suffix in (".yaml", ".yml"):
                if not YAML_AVAILABLE:
                    return False
                config = yaml.safe_load(content)
            else:
                config = json.loads(content)

            # Validate if schema exists
            if self.schema:
                errors = self.schema.validate(config)
                if errors:
                    self.bus.emit({
                        "topic": self.BUS_TOPICS["error"],
                        "kind": "error",
                        "level": "error",
                        "actor": "config-manager",
                        "data": {"errors": errors, "file": str(path)},
                    })
                    return False

            # Load values
            for key, value in config.items():
                self._set_value(key, value, ConfigSource.FILE)

            # Emit loaded event
            self.bus.emit({
                "topic": self.BUS_TOPICS["loaded"],
                "kind": "config",
                "actor": "config-manager",
                "data": {"file": str(path), "keys": list(config.keys())},
            })

            return True

        except Exception as e:
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "error",
                "actor": "config-manager",
                "data": {"error": str(e), "file": str(path)},
            })
            return False

    def load_environment(self, prefix: str = "CODE_") -> None:
        """Load configuration from environment variables."""
        # Load from schema mappings
        if self.schema:
            for key, env_var in self.schema.get_env_mappings().items():
                if env_var in os.environ:
                    value = self._parse_env_value(os.environ[env_var])
                    self._set_value(key, value, ConfigSource.ENVIRONMENT)

        # Load all with prefix
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                key = env_key[len(prefix):].lower().replace("_", ".")
                value = self._parse_env_value(env_value)
                self._set_value(key, value, ConfigSource.ENVIRONMENT)

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value."""
        # Try to parse as JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        return value

    def load_auto(self) -> int:
        """
        Auto-discover and load configuration.

        Returns:
            Number of files loaded
        """
        loaded = 0

        self.load_defaults()

        for path in self.DEFAULT_PATHS:
            if self.load_file(path):
                loaded += 1

        self.load_environment()

        return loaded

    # =========================================================================
    # Access
    # =========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        with self._lock:
            if key in self._values:
                return self._values[key].value
            return default

    def get_typed(self, key: str, type_: Type, default: Any = None) -> Any:
        """Get configuration value with type conversion."""
        value = self.get(key, default)
        if value is None:
            return default
        try:
            return type_(value)
        except (ValueError, TypeError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        return self.get_typed(key, int, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        return self.get_typed(key, float, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        return bool(value)

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get list configuration value."""
        value = self.get(key, default or [])
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",")]
        return default or []

    def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        """Get dictionary configuration value."""
        value = self.get(key, default or {})
        return value if isinstance(value, dict) else default or {}

    def set(self, key: str, value: Any) -> None:
        """Set configuration value (override)."""
        self._set_value(key, value, ConfigSource.OVERRIDE)

    def _set_value(self, key: str, value: Any, source: ConfigSource) -> None:
        """Internal method to set value and emit change."""
        with self._lock:
            old_value = self._values.get(key)

            self._values[key] = ConfigValue(
                key=key,
                value=value,
                source=source,
            )

            # Check for change
            if old_value and old_value.value != value:
                event = ConfigChangeEvent(
                    key=key,
                    old_value=old_value.value,
                    new_value=value,
                    source=source,
                )

                # Emit bus event
                self.bus.emit({
                    "topic": self.BUS_TOPICS["changed"],
                    "kind": "config",
                    "actor": "config-manager",
                    "data": event.to_dict(),
                })

                # Call handlers
                for handler in self._change_handlers:
                    try:
                        handler(event)
                    except Exception:
                        pass

    def keys(self) -> List[str]:
        """Get all configuration keys."""
        return list(self._values.keys())

    def items(self) -> Dict[str, Any]:
        """Get all configuration as dict."""
        return {k: v.value for k, v in self._values.items()}

    def has(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._values

    # =========================================================================
    # Watching
    # =========================================================================

    def watch_file(self, file_path: Union[str, Path]) -> ConfigWatcher:
        """Add a file watcher for hot reloading."""
        path = Path(file_path).expanduser()
        watcher = ConfigWatcher(
            path,
            lambda config: self._handle_reload(config, path),
        )
        self._watchers.append(watcher)
        return watcher

    def _handle_reload(self, config: Dict[str, Any], path: Path) -> None:
        """Handle file reload."""
        for key, value in config.items():
            self._set_value(key, value, ConfigSource.FILE)

    async def start_watching(self) -> None:
        """Start all file watchers."""
        for watcher in self._watchers:
            await watcher.start()

    async def stop_watching(self) -> None:
        """Stop all file watchers."""
        for watcher in self._watchers:
            await watcher.stop()

    def on_change(self, handler: Callable[[ConfigChangeEvent], None]) -> None:
        """Register a change handler."""
        self._change_handlers.append(handler)

    # =========================================================================
    # Export
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Export all configuration as dict."""
        return {
            k: v.to_dict()
            for k, v in self._values.items()
        }

    def to_json(self) -> str:
        """Export configuration as JSON."""
        return json.dumps(self.items(), indent=2)

    def save(self, file_path: Union[str, Path]) -> bool:
        """Save configuration to file."""
        path = Path(file_path).expanduser()

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            content = self.to_json()
            path.write_text(content)
            return True

        except Exception:
            return False


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Config Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Config Manager (Step 86)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # get command
    get_parser = subparsers.add_parser("get", help="Get configuration value")
    get_parser.add_argument("key", help="Configuration key")
    get_parser.add_argument("--default", "-d", help="Default value")

    # set command
    set_parser = subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", help="Value to set")

    # list command
    list_parser = subparsers.add_parser("list", help="List configuration")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # load command
    load_parser = subparsers.add_parser("load", help="Load configuration file")
    load_parser.add_argument("file", help="Configuration file path")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument("file", help="Configuration file path")

    args = parser.parse_args()

    # Create schema
    schema = ConfigSchema()
    schema.field("port", "int", default=8080, min_value=1, max_value=65535)
    schema.field("host", "string", default="localhost")
    schema.field("debug", "bool", default=False)
    schema.field("workers", "int", default=4, min_value=1)

    config = ConfigManager(schema=schema)
    config.load_auto()

    if args.command == "get":
        value = config.get(args.key, args.default)
        print(value if value is not None else "")
        return 0 if value is not None else 1

    elif args.command == "set":
        config.set(args.key, args.value)
        print(f"Set {args.key} = {args.value}")
        return 0

    elif args.command == "list":
        if args.json:
            print(config.to_json())
        else:
            items = config.to_dict()
            for key, info in items.items():
                print(f"{key}: {info['value']} (from {info['source']})")
        return 0

    elif args.command == "load":
        if config.load_file(args.file):
            print(f"Loaded configuration from {args.file}")
            return 0
        else:
            print(f"Failed to load {args.file}")
            return 1

    elif args.command == "validate":
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {args.file}")
            return 1

        content = path.read_text()
        if path.suffix in (".yaml", ".yml") and YAML_AVAILABLE:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)

        errors = schema.validate(data)
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  - {e}")
            return 1
        else:
            print("Configuration is valid")
            return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
