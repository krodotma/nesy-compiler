#!/usr/bin/env python3
"""
Review Config Manager (Step 186)

Configuration management system for the Review Agent with multiple sources,
validation, and hot-reloading support.

PBTSO Phase: SKILL, BUILD
Bus Topics: review.config.load, review.config.change, review.config.validate

Config Features:
- Multiple config sources (env, file, defaults)
- Schema validation
- Hot-reload support
- Secret management
- Environment-specific overrides

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class ConfigSource(Enum):
    """Configuration sources (in priority order, highest first)."""
    OVERRIDE = "override"     # Runtime overrides
    ENV = "env"               # Environment variables
    FILE = "file"             # Configuration files
    DEFAULT = "default"       # Default values


class ConfigType(Enum):
    """Configuration value types."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    PATH = "path"
    SECRET = "secret"


class ConfigValidation(Enum):
    """Validation result types."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class ConfigValue:
    """
    A configuration value with metadata.

    Attributes:
        key: Configuration key
        value: Current value
        type: Value type
        source: Where the value came from
        description: Value description
        required: Whether the value is required
        secret: Whether the value is a secret (should be redacted)
        validators: Validation functions
        default: Default value
    """
    key: str
    value: Any
    type: ConfigType = ConfigType.STRING
    source: ConfigSource = ConfigSource.DEFAULT
    description: str = ""
    required: bool = False
    secret: bool = False
    validators: List[str] = field(default_factory=list)
    default: Any = None

    def to_dict(self, include_value: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "key": self.key,
            "type": self.type.value,
            "source": self.source.value,
            "description": self.description,
            "required": self.required,
            "secret": self.secret,
        }

        if include_value:
            if self.secret:
                result["value"] = "***REDACTED***" if self.value else None
            else:
                result["value"] = self.value

        return result


@dataclass
class ConfigSchema:
    """
    Schema for configuration validation.

    Attributes:
        key: Configuration key
        type: Expected type
        description: Value description
        required: Whether required
        default: Default value
        validators: Validator names
        env_var: Environment variable name
        secret: Whether it's a secret
        min_value: Minimum value (for numbers)
        max_value: Maximum value (for numbers)
        pattern: Regex pattern (for strings)
        choices: Valid choices
    """
    key: str
    type: ConfigType = ConfigType.STRING
    description: str = ""
    required: bool = False
    default: Any = None
    validators: List[str] = field(default_factory=list)
    env_var: Optional[str] = None
    secret: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    choices: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    key: str
    status: ConfigValidation
    message: str = ""
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class ConfigChangeEvent:
    """Event emitted when configuration changes."""
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "old_value": "***" if self._is_secret() else self.old_value,
            "new_value": "***" if self._is_secret() else self.new_value,
            "source": self.source.value,
            "timestamp": self.timestamp,
        }

    def _is_secret(self) -> bool:
        """Check if this is likely a secret."""
        return any(s in self.key.lower() for s in ["secret", "password", "token", "key", "api"])


# ============================================================================
# Built-in Validators
# ============================================================================

def validate_positive(value: Any) -> bool:
    """Validate that value is positive."""
    return isinstance(value, (int, float)) and value > 0


def validate_non_negative(value: Any) -> bool:
    """Validate that value is non-negative."""
    return isinstance(value, (int, float)) and value >= 0


def validate_port(value: Any) -> bool:
    """Validate that value is a valid port number."""
    return isinstance(value, int) and 1 <= value <= 65535


def validate_url(value: Any) -> bool:
    """Validate that value is a URL."""
    if not isinstance(value, str):
        return False
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, value))


def validate_path_exists(value: Any) -> bool:
    """Validate that path exists."""
    return isinstance(value, (str, Path)) and Path(value).exists()


def validate_email(value: Any) -> bool:
    """Validate email format."""
    if not isinstance(value, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, value))


BUILT_IN_VALIDATORS = {
    "positive": validate_positive,
    "non_negative": validate_non_negative,
    "port": validate_port,
    "url": validate_url,
    "path_exists": validate_path_exists,
    "email": validate_email,
}


# ============================================================================
# Config Manager
# ============================================================================

class ConfigManager:
    """
    Configuration management system.

    Example:
        manager = ConfigManager()

        # Define schema
        manager.define("review.timeout", ConfigSchema(
            key="review.timeout",
            type=ConfigType.INT,
            description="Review timeout in seconds",
            default=300,
            validators=["positive"],
            env_var="REVIEW_TIMEOUT",
        ))

        # Load configuration
        await manager.load()

        # Get values
        timeout = manager.get("review.timeout")

        # Set values
        manager.set("review.timeout", 600, source=ConfigSource.OVERRIDE)

        # Watch for changes
        manager.on_change("review.timeout", lambda e: print(f"Changed: {e}"))
    """

    BUS_TOPICS = {
        "load": "review.config.load",
        "change": "review.config.change",
        "validate": "review.config.validate",
    }

    def __init__(
        self,
        config_path: Optional[Path] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the config manager.

        Args:
            config_path: Path to config file
            bus_path: Path to event bus file
        """
        self.config_path = config_path or self._get_config_path()
        self.bus_path = bus_path or self._get_bus_path()

        # Schema registry
        self._schemas: Dict[str, ConfigSchema] = {}

        # Value storage by source
        self._values: Dict[ConfigSource, Dict[str, Any]] = {
            source: {} for source in ConfigSource
        }

        # Resolved values cache
        self._resolved: Dict[str, ConfigValue] = {}

        # Change listeners
        self._listeners: Dict[str, List[Callable[[ConfigChangeEvent], None]]] = {}
        self._global_listeners: List[Callable[[ConfigChangeEvent], None]] = []

        # Custom validators
        self._validators: Dict[str, Callable[[Any], bool]] = dict(BUILT_IN_VALIDATORS)

        # State
        self._loaded = False
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_config_path(self) -> Path:
        """Get default config path."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "review" / "config.json"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "config") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "config-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def define(self, key: str, schema: ConfigSchema) -> None:
        """
        Define a configuration schema.

        Args:
            key: Configuration key
            schema: Schema definition
        """
        schema.key = key
        self._schemas[key] = schema

        # Set default value
        if schema.default is not None:
            self._values[ConfigSource.DEFAULT][key] = schema.default

    def define_many(self, schemas: List[ConfigSchema]) -> None:
        """Define multiple schemas."""
        for schema in schemas:
            self.define(schema.key, schema)

    def register_validator(self, name: str, validator: Callable[[Any], bool]) -> None:
        """
        Register a custom validator.

        Args:
            name: Validator name
            validator: Validator function
        """
        self._validators[name] = validator

    async def load(self, reload: bool = False) -> bool:
        """
        Load configuration from all sources.

        Args:
            reload: Force reload even if already loaded

        Returns:
            True if loading succeeded

        Emits:
            review.config.load
        """
        if self._loaded and not reload:
            return True

        # Load from file
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        file_config = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                self._values[ConfigSource.FILE] = file_config
            except Exception as e:
                self._emit_event(self.BUS_TOPICS["load"], {
                    "status": "error",
                    "error": str(e),
                })
                return False

        # Load from environment
        for key, schema in self._schemas.items():
            env_var = schema.env_var or self._key_to_env(key)
            if env_var in os.environ:
                raw_value = os.environ[env_var]
                parsed_value = self._parse_value(raw_value, schema.type)
                self._values[ConfigSource.ENV][key] = parsed_value

        # Resolve all values
        self._resolve_all()

        self._loaded = True

        self._emit_event(self.BUS_TOPICS["load"], {
            "status": "success",
            "keys_loaded": len(self._resolved),
            "sources": {
                source.value: len(values)
                for source, values in self._values.items()
            },
        })

        return True

    def _key_to_env(self, key: str) -> str:
        """Convert config key to environment variable name."""
        return key.upper().replace(".", "_").replace("-", "_")

    def _parse_value(self, raw: str, config_type: ConfigType) -> Any:
        """Parse a raw string value to the appropriate type."""
        if config_type == ConfigType.INT:
            return int(raw)
        elif config_type == ConfigType.FLOAT:
            return float(raw)
        elif config_type == ConfigType.BOOL:
            return raw.lower() in ("true", "1", "yes", "on")
        elif config_type == ConfigType.LIST:
            return json.loads(raw) if raw.startswith("[") else raw.split(",")
        elif config_type == ConfigType.DICT:
            return json.loads(raw)
        elif config_type == ConfigType.PATH:
            return Path(raw)
        else:
            return raw

    def _resolve_all(self) -> None:
        """Resolve all configuration values."""
        self._resolved.clear()

        for key, schema in self._schemas.items():
            self._resolved[key] = self._resolve_key(key, schema)

    def _resolve_key(self, key: str, schema: ConfigSchema) -> ConfigValue:
        """Resolve a single configuration key."""
        # Check sources in priority order
        for source in ConfigSource:
            if key in self._values[source]:
                return ConfigValue(
                    key=key,
                    value=self._values[source][key],
                    type=schema.type,
                    source=source,
                    description=schema.description,
                    required=schema.required,
                    secret=schema.secret,
                    validators=schema.validators,
                    default=schema.default,
                )

        # Return default value
        return ConfigValue(
            key=key,
            value=schema.default,
            type=schema.type,
            source=ConfigSource.DEFAULT,
            description=schema.description,
            required=schema.required,
            secret=schema.secret,
            validators=schema.validators,
            default=schema.default,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        if key in self._resolved:
            return self._resolved[key].value

        # Check direct values
        for source in ConfigSource:
            if key in self._values[source]:
                return self._values[source][key]

        return default

    def get_value(self, key: str) -> Optional[ConfigValue]:
        """Get a ConfigValue object."""
        return self._resolved.get(key)

    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.OVERRIDE,
    ) -> bool:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: New value
            source: Value source

        Returns:
            True if set succeeded

        Emits:
            review.config.change
        """
        old_value = self.get(key)

        # Validate if schema exists
        if key in self._schemas:
            result = self.validate_value(key, value)
            if result.status == ConfigValidation.INVALID:
                return False

        # Set value
        self._values[source][key] = value

        # Re-resolve
        if key in self._schemas:
            self._resolved[key] = self._resolve_key(key, self._schemas[key])

        # Emit change event
        if old_value != value:
            change = ConfigChangeEvent(
                key=key,
                old_value=old_value,
                new_value=value,
                source=source,
            )

            self._emit_event(self.BUS_TOPICS["change"], change.to_dict())

            # Notify listeners
            self._notify_listeners(key, change)

        return True

    def validate_value(self, key: str, value: Any) -> ValidationResult:
        """
        Validate a configuration value.

        Args:
            key: Configuration key
            value: Value to validate

        Returns:
            ValidationResult
        """
        if key not in self._schemas:
            return ValidationResult(key=key, status=ConfigValidation.VALID)

        schema = self._schemas[key]

        # Check required
        if schema.required and value is None:
            return ValidationResult(
                key=key,
                status=ConfigValidation.INVALID,
                message=f"Required configuration '{key}' is missing",
            )

        if value is None:
            return ValidationResult(key=key, status=ConfigValidation.VALID)

        # Check type
        type_valid = self._check_type(value, schema.type)
        if not type_valid:
            return ValidationResult(
                key=key,
                status=ConfigValidation.INVALID,
                message=f"Invalid type for '{key}': expected {schema.type.value}",
            )

        # Check min/max
        if schema.min_value is not None and isinstance(value, (int, float)):
            if value < schema.min_value:
                return ValidationResult(
                    key=key,
                    status=ConfigValidation.INVALID,
                    message=f"Value {value} is below minimum {schema.min_value}",
                )

        if schema.max_value is not None and isinstance(value, (int, float)):
            if value > schema.max_value:
                return ValidationResult(
                    key=key,
                    status=ConfigValidation.INVALID,
                    message=f"Value {value} is above maximum {schema.max_value}",
                )

        # Check pattern
        if schema.pattern and isinstance(value, str):
            if not re.match(schema.pattern, value):
                return ValidationResult(
                    key=key,
                    status=ConfigValidation.INVALID,
                    message=f"Value does not match pattern '{schema.pattern}'",
                )

        # Check choices
        if schema.choices and value not in schema.choices:
            return ValidationResult(
                key=key,
                status=ConfigValidation.INVALID,
                message=f"Value must be one of: {schema.choices}",
            )

        # Run validators
        for validator_name in schema.validators:
            validator = self._validators.get(validator_name)
            if validator and not validator(value):
                return ValidationResult(
                    key=key,
                    status=ConfigValidation.INVALID,
                    message=f"Validation '{validator_name}' failed",
                )

        return ValidationResult(key=key, status=ConfigValidation.VALID, value=value)

    def _check_type(self, value: Any, config_type: ConfigType) -> bool:
        """Check if value matches expected type."""
        type_checks = {
            ConfigType.STRING: lambda v: isinstance(v, str),
            ConfigType.INT: lambda v: isinstance(v, int) and not isinstance(v, bool),
            ConfigType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            ConfigType.BOOL: lambda v: isinstance(v, bool),
            ConfigType.LIST: lambda v: isinstance(v, list),
            ConfigType.DICT: lambda v: isinstance(v, dict),
            ConfigType.PATH: lambda v: isinstance(v, (str, Path)),
            ConfigType.SECRET: lambda v: isinstance(v, str),
        }
        return type_checks.get(config_type, lambda v: True)(value)

    def validate_all(self) -> List[ValidationResult]:
        """
        Validate all configuration values.

        Returns:
            List of validation results

        Emits:
            review.config.validate
        """
        results = []

        for key, schema in self._schemas.items():
            value = self.get(key)
            result = self.validate_value(key, value)
            results.append(result)

        self._emit_event(self.BUS_TOPICS["validate"], {
            "total": len(results),
            "valid": sum(1 for r in results if r.status == ConfigValidation.VALID),
            "invalid": sum(1 for r in results if r.status == ConfigValidation.INVALID),
        })

        return results

    def on_change(
        self,
        key: str,
        callback: Callable[[ConfigChangeEvent], None],
    ) -> None:
        """
        Register a change listener for a specific key.

        Args:
            key: Configuration key
            callback: Callback function
        """
        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(callback)

    def on_any_change(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Register a global change listener."""
        self._global_listeners.append(callback)

    def _notify_listeners(self, key: str, event: ConfigChangeEvent) -> None:
        """Notify listeners of a change."""
        # Key-specific listeners
        for callback in self._listeners.get(key, []):
            try:
                callback(event)
            except Exception:
                pass

        # Global listeners
        for callback in self._global_listeners:
            try:
                callback(event)
            except Exception:
                pass

    async def save(self) -> bool:
        """
        Save configuration to file.

        Returns:
            True if save succeeded
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {}
        for key, config_value in self._resolved.items():
            if config_value.source == ConfigSource.FILE:
                config_data[key] = config_value.value

        try:
            with open(self.config_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(config_data, f, indent=2, default=str)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception:
            return False

    def get_all(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Get all configuration values."""
        result = {}
        for key, config_value in self._resolved.items():
            if config_value.secret and not include_secrets:
                result[key] = "***REDACTED***"
            else:
                result[key] = config_value.value
        return result

    def get_schema(self, key: str) -> Optional[ConfigSchema]:
        """Get schema for a key."""
        return self._schemas.get(key)

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "config-manager",
            "healthy": True,
            "loaded": self._loaded,
            "keys_defined": len(self._schemas),
            "keys_resolved": len(self._resolved),
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
    """CLI entry point for Config Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Config Manager (Step 186)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get configuration value")
    get_parser.add_argument("key", help="Configuration key")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", help="Configuration value")

    # List command
    list_parser = subparsers.add_parser("list", help="List all configuration")
    list_parser.add_argument("--secrets", action="store_true", help="Include secrets")

    # Validate command
    subparsers.add_parser("validate", help="Validate configuration")

    # Schema command
    subparsers.add_parser("schema", help="Show schema")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manager = ConfigManager()

    # Define some default schemas
    manager.define_many([
        ConfigSchema(key="review.timeout", type=ConfigType.INT, default=300,
                    description="Review timeout in seconds", validators=["positive"]),
        ConfigSchema(key="review.max_files", type=ConfigType.INT, default=100,
                    description="Maximum files per review", validators=["positive"]),
        ConfigSchema(key="review.security_enabled", type=ConfigType.BOOL, default=True,
                    description="Enable security scanning"),
        ConfigSchema(key="review.api_key", type=ConfigType.SECRET, secret=True,
                    description="API key", env_var="REVIEW_API_KEY"),
    ])

    asyncio.run(manager.load())

    if args.command == "get":
        value = manager.get(args.key)
        config_value = manager.get_value(args.key)

        if args.json:
            print(json.dumps({"key": args.key, "value": value}, indent=2))
        else:
            if config_value:
                print(f"{args.key} = {value}")
                print(f"  Source: {config_value.source.value}")
                print(f"  Type: {config_value.type.value}")
            else:
                print(f"{args.key} = {value}")

    elif args.command == "set":
        success = manager.set(args.key, args.value)
        print("OK" if success else "Failed")
        return 0 if success else 1

    elif args.command == "list":
        config = manager.get_all(include_secrets=args.secrets)
        if args.json:
            print(json.dumps(config, indent=2))
        else:
            print("Configuration:")
            for key, value in sorted(config.items()):
                print(f"  {key}: {value}")

    elif args.command == "validate":
        results = manager.validate_all()
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            valid = sum(1 for r in results if r.status == ConfigValidation.VALID)
            invalid = sum(1 for r in results if r.status == ConfigValidation.INVALID)
            print(f"Validation: {valid} valid, {invalid} invalid")
            for r in results:
                if r.status != ConfigValidation.VALID:
                    print(f"  {r.key}: {r.message}")

    elif args.command == "schema":
        if args.json:
            schemas = {k: v.to_dict() for k, v in manager._schemas.items()}
            print(json.dumps(schemas, indent=2))
        else:
            print("Configuration Schema:")
            for key, schema in sorted(manager._schemas.items()):
                req = " (required)" if schema.required else ""
                sec = " [SECRET]" if schema.secret else ""
                print(f"  {key}: {schema.type.value}{req}{sec}")
                if schema.description:
                    print(f"    {schema.description}")

    else:
        # Default: show status
        status = manager.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Config Manager: {status['keys_resolved']} keys loaded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
