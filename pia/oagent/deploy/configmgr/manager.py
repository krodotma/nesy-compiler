#!/usr/bin/env python3
"""
manager.py - Deploy Config Manager (Step 236)

PBTSO Phase: SEQUESTER, ITERATE
A2A Integration: Configuration management via deploy.configmgr.*

Provides:
- ConfigScope: Configuration scope levels
- ConfigPriority: Configuration priority levels
- ConfigSource: Configuration sources
- ConfigValue: Configuration value wrapper
- ConfigSchema: Schema for validation
- DeployConfigManager: Main config manager

Bus Topics:
- deploy.configmgr.load
- deploy.configmgr.update
- deploy.configmgr.validate
- deploy.configmgr.override

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union


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
    actor: str = "config-manager"
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

class ConfigScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    ENVIRONMENT = "environment"
    SERVICE = "service"
    DEPLOYMENT = "deployment"
    INSTANCE = "instance"


class ConfigPriority(IntEnum):
    """Configuration priority levels (higher = more important)."""
    DEFAULT = 0
    FILE = 10
    ENVIRONMENT_VAR = 20
    SERVICE = 30
    OVERRIDE = 40
    RUNTIME = 50


class ConfigSource(Enum):
    """Configuration sources."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    COMMAND_LINE = "command_line"
    REMOTE = "remote"
    RUNTIME = "runtime"
    INHERITED = "inherited"


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SECRET = "secret"
    PATH = "path"
    DURATION = "duration"


T = TypeVar("T")


@dataclass
class ConfigValue:
    """
    Configuration value wrapper.

    Attributes:
        key: Configuration key
        value: Configuration value
        config_type: Value type
        source: Value source
        priority: Value priority
        scope: Value scope
        environment: Target environment
        service: Target service
        description: Value description
        secret: Whether value is sensitive
        updated_at: Last update timestamp
    """
    key: str
    value: Any
    config_type: ConfigType = ConfigType.STRING
    source: ConfigSource = ConfigSource.DEFAULT
    priority: ConfigPriority = ConfigPriority.DEFAULT
    scope: ConfigScope = ConfigScope.GLOBAL
    environment: str = ""
    service: str = ""
    description: str = ""
    secret: bool = False
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": "***" if self.secret else self.value,
            "config_type": self.config_type.value,
            "source": self.source.value,
            "priority": int(self.priority),
            "scope": self.scope.value,
            "environment": self.environment,
            "service": self.service,
            "description": self.description,
            "secret": self.secret,
            "updated_at": self.updated_at,
        }

    def get_typed_value(self) -> Any:
        """Get value converted to proper type."""
        if self.config_type == ConfigType.INTEGER:
            return int(self.value)
        elif self.config_type == ConfigType.FLOAT:
            return float(self.value)
        elif self.config_type == ConfigType.BOOLEAN:
            if isinstance(self.value, bool):
                return self.value
            return str(self.value).lower() in ("true", "1", "yes", "on")
        elif self.config_type == ConfigType.LIST:
            if isinstance(self.value, list):
                return self.value
            return self.value.split(",") if self.value else []
        elif self.config_type == ConfigType.DICT:
            if isinstance(self.value, dict):
                return self.value
            return json.loads(self.value) if self.value else {}
        elif self.config_type == ConfigType.DURATION:
            return self._parse_duration(str(self.value))
        return self.value

    def _parse_duration(self, value: str) -> int:
        """Parse duration string to seconds."""
        match = re.match(r"(\d+)(s|m|h|d)?", value)
        if not match:
            return int(value)

        num = int(match.group(1))
        unit = match.group(2) or "s"

        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return num * multipliers.get(unit, 1)


@dataclass
class ConfigSchema:
    """
    Schema for configuration validation.

    Attributes:
        key: Configuration key
        config_type: Expected type
        required: Whether required
        default: Default value
        description: Field description
        min_value: Minimum value (for numbers)
        max_value: Maximum value (for numbers)
        pattern: Regex pattern (for strings)
        choices: Valid choices
        secret: Whether this is sensitive
    """
    key: str
    config_type: ConfigType = ConfigType.STRING
    required: bool = False
    default: Any = None
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    choices: List[Any] = field(default_factory=list)
    secret: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "config_type": self.config_type.value,
            "required": self.required,
            "default": self.default,
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "pattern": self.pattern,
            "choices": self.choices,
            "secret": self.secret,
        }


@dataclass
class ValidationResult:
    """Configuration validation result."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Deploy Config Manager (Step 236)
# ==============================================================================

class DeployConfigManager:
    """
    Deploy Config Manager - comprehensive configuration management.

    PBTSO Phase: SEQUESTER, ITERATE

    Responsibilities:
    - Load configuration from multiple sources
    - Support hierarchical configuration (global -> env -> service)
    - Validate configuration against schemas
    - Support runtime overrides
    - Track configuration changes

    Example:
        >>> config = DeployConfigManager()
        >>> config.define("deploy.timeout_s", ConfigType.INTEGER, default=1800)
        >>> config.set("deploy.timeout_s", 3600, source=ConfigSource.ENVIRONMENT)
        >>> timeout = config.get_int("deploy.timeout_s")
    """

    BUS_TOPICS = {
        "load": "deploy.configmgr.load",
        "update": "deploy.configmgr.update",
        "validate": "deploy.configmgr.validate",
        "override": "deploy.configmgr.override",
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "config-manager",
        env_prefix: str = "DEPLOY_",
        auto_load_env: bool = True,
    ):
        """
        Initialize the config manager.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            env_prefix: Prefix for environment variables
            auto_load_env: Auto-load from environment variables
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "configmgr"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.env_prefix = env_prefix

        # Configuration storage (key -> list of values by priority)
        self._values: Dict[str, List[ConfigValue]] = {}

        # Schemas
        self._schemas: Dict[str, ConfigSchema] = {}

        # Change listeners
        self._listeners: Dict[str, List[Callable[[ConfigValue], None]]] = {}

        self._load_state()

        if auto_load_env:
            self._load_from_environment()

    def define(
        self,
        key: str,
        config_type: ConfigType = ConfigType.STRING,
        default: Any = None,
        required: bool = False,
        description: str = "",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        pattern: Optional[str] = None,
        choices: Optional[List[Any]] = None,
        secret: bool = False,
    ) -> ConfigSchema:
        """
        Define a configuration schema.

        Args:
            key: Configuration key
            config_type: Expected type
            default: Default value
            required: Whether required
            description: Description
            min_value: Minimum value
            max_value: Maximum value
            pattern: Regex pattern
            choices: Valid choices
            secret: Whether sensitive

        Returns:
            ConfigSchema
        """
        schema = ConfigSchema(
            key=key,
            config_type=config_type,
            default=default,
            required=required,
            description=description,
            min_value=min_value,
            max_value=max_value,
            pattern=pattern,
            choices=choices or [],
            secret=secret,
        )

        self._schemas[key] = schema

        # Set default if provided
        if default is not None:
            self.set(
                key,
                default,
                source=ConfigSource.DEFAULT,
                priority=ConfigPriority.DEFAULT,
                config_type=config_type,
                secret=secret,
            )

        return schema

    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.RUNTIME,
        priority: ConfigPriority = ConfigPriority.RUNTIME,
        scope: ConfigScope = ConfigScope.GLOBAL,
        environment: str = "",
        service: str = "",
        config_type: Optional[ConfigType] = None,
        secret: bool = False,
    ) -> ConfigValue:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
            source: Value source
            priority: Value priority
            scope: Value scope
            environment: Target environment
            service: Target service
            config_type: Value type
            secret: Whether sensitive

        Returns:
            ConfigValue
        """
        # Get type from schema if not provided
        if config_type is None:
            schema = self._schemas.get(key)
            config_type = schema.config_type if schema else ConfigType.STRING
            if schema:
                secret = schema.secret

        config_value = ConfigValue(
            key=key,
            value=value,
            config_type=config_type,
            source=source,
            priority=priority,
            scope=scope,
            environment=environment,
            service=service,
            secret=secret,
        )

        if key not in self._values:
            self._values[key] = []

        # Remove existing value with same scope/env/service
        self._values[key] = [
            v for v in self._values[key]
            if not (v.scope == scope and v.environment == environment and v.service == service)
        ]

        self._values[key].append(config_value)
        self._values[key].sort(key=lambda v: v.priority, reverse=True)

        # Emit bus event
        _emit_bus_event(
            self.BUS_TOPICS["update"],
            {
                "key": key,
                "source": source.value,
                "priority": int(priority),
                "scope": scope.value,
            },
            actor=self.actor_id,
        )

        # Notify listeners
        self._notify_listeners(key, config_value)

        self._save_state()
        return config_value

    def get(
        self,
        key: str,
        default: Any = None,
        environment: str = "",
        service: str = "",
    ) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if not found
            environment: Filter by environment
            service: Filter by service

        Returns:
            Configuration value
        """
        config_value = self.get_value(key, environment, service)

        if config_value is None:
            schema = self._schemas.get(key)
            if schema and schema.default is not None:
                return schema.default
            return default

        return config_value.get_typed_value()

    def get_value(
        self,
        key: str,
        environment: str = "",
        service: str = "",
    ) -> Optional[ConfigValue]:
        """
        Get a ConfigValue object.

        Args:
            key: Configuration key
            environment: Filter by environment
            service: Filter by service

        Returns:
            ConfigValue or None
        """
        values = self._values.get(key, [])
        if not values:
            return None

        # Filter by environment and service (most specific first)
        for val in values:
            if val.service == service and val.environment == environment:
                return val

        # Try environment only
        for val in values:
            if val.environment == environment and not val.service:
                return val

        # Try service only
        for val in values:
            if val.service == service and not val.environment:
                return val

        # Return global (highest priority)
        for val in values:
            if not val.environment and not val.service:
                return val

        return values[0] if values else None

    def get_str(self, key: str, default: str = "", **kwargs) -> str:
        """Get string value."""
        value = self.get(key, default, **kwargs)
        return str(value) if value is not None else default

    def get_int(self, key: str, default: int = 0, **kwargs) -> int:
        """Get integer value."""
        value = self.get(key, default, **kwargs)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0, **kwargs) -> float:
        """Get float value."""
        value = self.get(key, default, **kwargs)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False, **kwargs) -> bool:
        """Get boolean value."""
        value = self.get(key, default, **kwargs)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def get_list(self, key: str, default: Optional[List] = None, **kwargs) -> List:
        """Get list value."""
        value = self.get(key, default, **kwargs)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return value.split(",") if value else []
        return default or []

    def get_dict(self, key: str, default: Optional[Dict] = None, **kwargs) -> Dict:
        """Get dict value."""
        value = self.get(key, default, **kwargs)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value) if value else {}
            except json.JSONDecodeError:
                return default or {}
        return default or {}

    def has(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._values and len(self._values[key]) > 0

    def delete(self, key: str) -> bool:
        """Delete a configuration key."""
        if key in self._values:
            del self._values[key]
            self._save_state()
            return True
        return False

    def override(
        self,
        key: str,
        value: Any,
        environment: str = "",
        service: str = "",
    ) -> ConfigValue:
        """
        Override a configuration value with highest priority.

        Args:
            key: Configuration key
            value: Override value
            environment: Target environment
            service: Target service

        Returns:
            ConfigValue
        """
        config_value = self.set(
            key,
            value,
            source=ConfigSource.RUNTIME,
            priority=ConfigPriority.OVERRIDE,
            environment=environment,
            service=service,
        )

        _emit_bus_event(
            self.BUS_TOPICS["override"],
            {
                "key": key,
                "environment": environment,
                "service": service,
            },
            actor=self.actor_id,
        )

        return config_value

    def validate(self) -> ValidationResult:
        """
        Validate all configurations against schemas.

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        for key, schema in self._schemas.items():
            value = self.get(key)

            # Check required
            if schema.required and value is None:
                errors.append(f"Required config '{key}' is not set")
                continue

            if value is None:
                continue

            # Type validation
            try:
                typed_value = self.get_value(key)
                if typed_value:
                    typed_value.get_typed_value()
            except Exception as e:
                errors.append(f"Config '{key}' type error: {e}")
                continue

            # Range validation
            if schema.min_value is not None and isinstance(value, (int, float)):
                if value < schema.min_value:
                    errors.append(f"Config '{key}' is below minimum ({schema.min_value})")

            if schema.max_value is not None and isinstance(value, (int, float)):
                if value > schema.max_value:
                    errors.append(f"Config '{key}' is above maximum ({schema.max_value})")

            # Pattern validation
            if schema.pattern and isinstance(value, str):
                if not re.match(schema.pattern, value):
                    errors.append(f"Config '{key}' does not match pattern: {schema.pattern}")

            # Choices validation
            if schema.choices and value not in schema.choices:
                errors.append(f"Config '{key}' must be one of: {schema.choices}")

        result = ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

        _emit_bus_event(
            self.BUS_TOPICS["validate"],
            {
                "valid": result.valid,
                "error_count": len(errors),
                "warning_count": len(warnings),
            },
            actor=self.actor_id,
        )

        return result

    def _load_from_environment(self) -> int:
        """Load configuration from environment variables."""
        loaded = 0

        for key, value in os.environ.items():
            if not key.startswith(self.env_prefix):
                continue

            # Convert DEPLOY_SOME_KEY to some.key
            config_key = key[len(self.env_prefix):].lower().replace("_", ".")

            self.set(
                config_key,
                value,
                source=ConfigSource.ENVIRONMENT,
                priority=ConfigPriority.ENVIRONMENT_VAR,
            )
            loaded += 1

        if loaded > 0:
            _emit_bus_event(
                self.BUS_TOPICS["load"],
                {"source": "environment", "count": loaded},
                actor=self.actor_id,
            )

        return loaded

    def load_from_file(
        self,
        file_path: str,
        priority: ConfigPriority = ConfigPriority.FILE,
        environment: str = "",
        service: str = "",
    ) -> int:
        """
        Load configuration from a file.

        Args:
            file_path: Path to config file
            priority: Value priority
            environment: Target environment
            service: Target service

        Returns:
            Number of values loaded
        """
        path = Path(file_path)
        if not path.exists():
            return 0

        content = path.read_text()
        loaded = 0

        try:
            if path.suffix in (".json",):
                data = json.loads(content)
            elif path.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(content)
            else:
                # Assume key=value format
                data = {}
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        data[k.strip()] = v.strip()

            def _flatten_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
                items = {}
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        items.update(_flatten_dict(v, key))
                    else:
                        items[key] = v
                return items

            flat_data = _flatten_dict(data)

            for key, value in flat_data.items():
                self.set(
                    key,
                    value,
                    source=ConfigSource.FILE,
                    priority=priority,
                    environment=environment,
                    service=service,
                )
                loaded += 1

        except Exception:
            pass

        if loaded > 0:
            _emit_bus_event(
                self.BUS_TOPICS["load"],
                {"source": "file", "path": str(file_path), "count": loaded},
                actor=self.actor_id,
            )

        return loaded

    def export(
        self,
        environment: str = "",
        service: str = "",
        include_secrets: bool = False,
    ) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Args:
            environment: Filter by environment
            service: Filter by service
            include_secrets: Include secret values

        Returns:
            Configuration dictionary
        """
        result = {}

        for key in self._values.keys():
            config_value = self.get_value(key, environment, service)
            if config_value:
                if config_value.secret and not include_secrets:
                    result[key] = "***"
                else:
                    result[key] = config_value.value

        return result

    def watch(
        self,
        key: str,
        callback: Callable[[ConfigValue], None],
    ) -> None:
        """Register a callback for configuration changes."""
        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(callback)

    def _notify_listeners(self, key: str, value: ConfigValue) -> None:
        """Notify listeners of configuration change."""
        for callback in self._listeners.get(key, []):
            try:
                callback(value)
            except Exception:
                pass

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all configuration keys."""
        keys = list(self._values.keys())
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        return sorted(keys)

    def list_schemas(self) -> List[ConfigSchema]:
        """List all configuration schemas."""
        return list(self._schemas.values())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "schemas": {k: s.to_dict() for k, s in self._schemas.items()},
            "values": {},
        }

        for key, values in self._values.items():
            state["values"][key] = [v.to_dict() for v in values]

        state_file = self.state_dir / "config_manager_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "config_manager_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for key, data in state.get("schemas", {}).items():
                data["config_type"] = ConfigType(data["config_type"])
                self._schemas[key] = ConfigSchema(**data)

            # Note: Values are loaded but not restored to avoid overriding

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for config manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Config Manager (Step 236)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # get command
    get_parser = subparsers.add_parser("get", help="Get configuration value")
    get_parser.add_argument("key", help="Configuration key")
    get_parser.add_argument("--env", "-e", default="", help="Environment")
    get_parser.add_argument("--service", "-s", default="", help="Service")
    get_parser.add_argument("--json", action="store_true", help="JSON output")

    # set command
    set_parser = subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", help="Configuration value")
    set_parser.add_argument("--type", "-t", default="string",
                            choices=["string", "integer", "float", "boolean", "list", "dict"])
    set_parser.add_argument("--env", "-e", default="", help="Environment")
    set_parser.add_argument("--service", "-s", default="", help="Service")
    set_parser.add_argument("--secret", action="store_true", help="Mark as secret")

    # list command
    list_parser = subparsers.add_parser("list", help="List configuration keys")
    list_parser.add_argument("--prefix", "-p", default="", help="Key prefix filter")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # export command
    export_parser = subparsers.add_parser("export", help="Export configuration")
    export_parser.add_argument("--env", "-e", default="", help="Environment")
    export_parser.add_argument("--service", "-s", default="", help="Service")
    export_parser.add_argument("--secrets", action="store_true", help="Include secrets")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--json", action="store_true", help="JSON output")

    # load command
    load_parser = subparsers.add_parser("load", help="Load from file")
    load_parser.add_argument("file", help="Configuration file path")
    load_parser.add_argument("--env", "-e", default="", help="Environment")
    load_parser.add_argument("--service", "-s", default="", help="Service")

    args = parser.parse_args()
    config = DeployConfigManager()

    if args.command == "get":
        value = config.get(args.key, environment=args.env, service=args.service)
        config_value = config.get_value(args.key, args.env, args.service)

        if args.json:
            print(json.dumps({
                "key": args.key,
                "value": value,
                "config_value": config_value.to_dict() if config_value else None,
            }, indent=2))
        else:
            if value is not None:
                print(value)
            else:
                print(f"Key not found: {args.key}")
                return 1

        return 0

    elif args.command == "set":
        config.set(
            args.key,
            args.value,
            config_type=ConfigType(args.type.upper()),
            environment=args.env,
            service=args.service,
            secret=args.secret,
        )
        print(f"Set: {args.key}")
        return 0

    elif args.command == "list":
        keys = config.list_keys(args.prefix)

        if args.json:
            print(json.dumps(keys, indent=2))
        else:
            for key in keys:
                value = config.get(key)
                config_value = config.get_value(key)
                source = config_value.source.value if config_value else "unknown"
                display_value = "***" if (config_value and config_value.secret) else str(value)[:50]
                print(f"{key} = {display_value} [{source}]")

        return 0

    elif args.command == "export":
        exported = config.export(
            environment=args.env,
            service=args.service,
            include_secrets=args.secrets,
        )
        print(json.dumps(exported, indent=2))
        return 0

    elif args.command == "validate":
        result = config.validate()

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.valid:
                print("Configuration is valid")
            else:
                print("Configuration is invalid:")
                for error in result.errors:
                    print(f"  ERROR: {error}")
            for warning in result.warnings:
                print(f"  WARNING: {warning}")

        return 0 if result.valid else 1

    elif args.command == "load":
        count = config.load_from_file(
            args.file,
            environment=args.env,
            service=args.service,
        )
        print(f"Loaded {count} configuration values from {args.file}")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
