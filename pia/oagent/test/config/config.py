#!/usr/bin/env python3
"""
Step 136: Test Config Manager

Configuration management for the Test Agent.

PBTSO Phase: PLAN, BUILD
Bus Topics:
- test.config.load (emits)
- test.config.change (emits)
- test.config.validate (emits)

Dependencies: Steps 101-135 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union


# ============================================================================
# Constants
# ============================================================================

class ConfigSource(Enum):
    """Configuration sources."""
    FILE = "file"
    ENV = "env"
    CLI = "cli"
    DEFAULT = "default"
    REMOTE = "remote"


class ConfigValueType(Enum):
    """Configuration value types."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    PATH = "path"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ConfigValue:
    """
    A configuration value.

    Attributes:
        key: Configuration key
        value: Configuration value
        source: Where the value came from
        value_type: Type of the value
        default: Default value
        description: Value description
        required: Whether value is required
        sensitive: Whether value is sensitive
    """
    key: str
    value: Any
    source: ConfigSource = ConfigSource.DEFAULT
    value_type: ConfigValueType = ConfigValueType.STRING
    default: Any = None
    description: str = ""
    required: bool = False
    sensitive: bool = False
    validators: List[Callable[[Any], bool]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": "<redacted>" if self.sensitive else self.value,
            "source": self.source.value,
            "type": self.value_type.value,
            "required": self.required,
            "description": self.description,
        }


@dataclass
class ConfigSchema:
    """
    Schema for configuration validation.

    Attributes:
        key: Configuration key
        value_type: Expected type
        required: Whether required
        default: Default value
        description: Description
        validators: Validation functions
        env_var: Environment variable name
        choices: Allowed values
    """
    key: str
    value_type: ConfigValueType = ConfigValueType.STRING
    required: bool = False
    default: Any = None
    description: str = ""
    validators: List[Callable[[Any], bool]] = field(default_factory=list)
    env_var: Optional[str] = None
    choices: Optional[List[Any]] = None
    sensitive: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None

    def validate(self, value: Any) -> tuple:
        """
        Validate a value against the schema.

        Returns:
            (is_valid, error_message)
        """
        # Type validation
        if self.value_type == ConfigValueType.INT:
            if not isinstance(value, int):
                return False, f"Expected int, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Expected float, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.BOOL:
            if not isinstance(value, bool):
                return False, f"Expected bool, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.LIST:
            if not isinstance(value, list):
                return False, f"Expected list, got {type(value).__name__}"
        elif self.value_type == ConfigValueType.DICT:
            if not isinstance(value, dict):
                return False, f"Expected dict, got {type(value).__name__}"

        # Range validation
        if self.min_value is not None and value < self.min_value:
            return False, f"Value {value} is less than minimum {self.min_value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"Value {value} is greater than maximum {self.max_value}"

        # Choices validation
        if self.choices and value not in self.choices:
            return False, f"Value must be one of: {self.choices}"

        # Pattern validation
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                return False, f"Value does not match pattern: {self.pattern}"

        # Custom validators
        for validator in self.validators:
            if not validator(value):
                return False, "Custom validation failed"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "type": self.value_type.value,
            "required": self.required,
            "default": self.default,
            "description": self.description,
            "choices": self.choices,
        }


@dataclass
class ConfigValidation:
    """
    Result of configuration validation.

    Attributes:
        is_valid: Whether configuration is valid
        errors: Validation errors
        warnings: Validation warnings
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ConfigChangeEvent:
    """
    Configuration change event.

    Attributes:
        key: Changed key
        old_value: Previous value
        new_value: New value
        source: Change source
        timestamp: Change timestamp
    """
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "source": self.source.value,
            "timestamp": self.timestamp,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class ConfigBus:
    """Bus interface for config with file locking."""

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
# Test Config Manager
# ============================================================================

class TestConfigManager:
    """
    Configuration management for the Test Agent.

    Features:
    - Multiple config sources (file, env, CLI)
    - Schema-based validation
    - Change tracking
    - Hot reload support
    - Sensitive value handling

    PBTSO Phase: PLAN, BUILD
    Bus Topics: test.config.load, test.config.change, test.config.validate
    """

    BUS_TOPICS = {
        "load": "test.config.load",
        "change": "test.config.change",
        "validate": "test.config.validate",
    }

    # Default Test Agent configuration schema
    DEFAULT_SCHEMA: List[ConfigSchema] = [
        ConfigSchema("agent_id", ConfigValueType.STRING, default="test-agent"),
        ConfigSchema("ring_level", ConfigValueType.INT, default=2, min_value=0, max_value=3),
        ConfigSchema("parallel_workers", ConfigValueType.INT, default=4, min_value=1, max_value=64),
        ConfigSchema("coverage_threshold", ConfigValueType.FLOAT, default=0.8, min_value=0, max_value=1),
        ConfigSchema("mutation_testing", ConfigValueType.BOOL, default=True),
        ConfigSchema("timeout_s", ConfigValueType.INT, default=300, min_value=1),
        ConfigSchema("output_dir", ConfigValueType.PATH, default=".pluribus/test-agent"),
        ConfigSchema("log_level", ConfigValueType.STRING, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"]),
        ConfigSchema("cache_enabled", ConfigValueType.BOOL, default=True),
        ConfigSchema("cache_ttl_s", ConfigValueType.INT, default=3600, min_value=0),
        ConfigSchema("api_port", ConfigValueType.INT, default=8080, min_value=1, max_value=65535),
        ConfigSchema("api_host", ConfigValueType.STRING, default="127.0.0.1"),
        ConfigSchema("notify_on_failure", ConfigValueType.BOOL, default=True),
        ConfigSchema("frameworks", ConfigValueType.LIST, default=["pytest"]),
        ConfigSchema("test_types", ConfigValueType.LIST, default=["unit", "integration"]),
    ]

    def __init__(
        self,
        bus=None,
        config_file: Optional[str] = None,
        schema: Optional[List[ConfigSchema]] = None,
    ):
        """
        Initialize the config manager.

        Args:
            bus: Optional bus instance
            config_file: Path to config file
            schema: Configuration schema
        """
        self.bus = bus or ConfigBus()
        self.config_file = Path(config_file) if config_file else None
        self.schema = schema or self.DEFAULT_SCHEMA
        self._values: Dict[str, ConfigValue] = {}
        self._change_handlers: List[Callable[[ConfigChangeEvent], None]] = []
        self._loaded = False

        # Build schema map
        self._schema_map: Dict[str, ConfigSchema] = {s.key: s for s in self.schema}

        # Initialize defaults
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        """Apply default values from schema."""
        for schema in self.schema:
            if schema.default is not None:
                self._values[schema.key] = ConfigValue(
                    key=schema.key,
                    value=schema.default,
                    source=ConfigSource.DEFAULT,
                    value_type=schema.value_type,
                    default=schema.default,
                    description=schema.description,
                    required=schema.required,
                    sensitive=schema.sensitive,
                )

    def load(self, config_file: Optional[str] = None) -> ConfigValidation:
        """
        Load configuration from all sources.

        Priority (lowest to highest):
        1. Defaults
        2. Config file
        3. Environment variables
        4. CLI arguments (not handled here)

        Args:
            config_file: Optional config file path

        Returns:
            ConfigValidation result
        """
        validation = ConfigValidation()

        # Load from file
        file_path = Path(config_file) if config_file else self.config_file
        if file_path and file_path.exists():
            file_validation = self._load_from_file(file_path)
            validation.errors.extend(file_validation.errors)
            validation.warnings.extend(file_validation.warnings)

        # Load from environment
        self._load_from_env()

        # Validate final config
        final_validation = self.validate()
        validation.errors.extend(final_validation.errors)
        validation.warnings.extend(final_validation.warnings)
        validation.is_valid = len(validation.errors) == 0

        self._loaded = True

        self._emit_event("load", {
            "is_valid": validation.is_valid,
            "source": str(file_path) if file_path else "defaults",
            "errors": validation.errors,
        })

        return validation

    def _load_from_file(self, file_path: Path) -> ConfigValidation:
        """Load configuration from file."""
        validation = ConfigValidation()

        try:
            with open(file_path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for key, value in data.items():
                self.set(key, value, source=ConfigSource.FILE)

        except json.JSONDecodeError as e:
            validation.errors.append(f"Invalid JSON in config file: {e}")
        except IOError as e:
            validation.errors.append(f"Error reading config file: {e}")

        return validation

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        prefix = "TEST_AGENT_"

        for schema in self.schema:
            # Check explicit env var name
            env_var = schema.env_var or f"{prefix}{schema.key.upper()}"
            value = os.environ.get(env_var)

            if value is not None:
                # Convert value based on type
                converted = self._convert_env_value(value, schema.value_type)
                self.set(schema.key, converted, source=ConfigSource.ENV)

    def _convert_env_value(self, value: str, value_type: ConfigValueType) -> Any:
        """Convert environment variable value to expected type."""
        if value_type == ConfigValueType.INT:
            return int(value)
        elif value_type == ConfigValueType.FLOAT:
            return float(value)
        elif value_type == ConfigValueType.BOOL:
            return value.lower() in ("true", "1", "yes", "on")
        elif value_type == ConfigValueType.LIST:
            return value.split(",")
        elif value_type == ConfigValueType.DICT:
            return json.loads(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        if key in self._values:
            return self._values[key].value
        return default

    def get_value(self, key: str) -> Optional[ConfigValue]:
        """Get a ConfigValue object."""
        return self._values.get(key)

    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.CLI,
    ) -> ConfigValidation:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
            source: Value source

        Returns:
            ConfigValidation result
        """
        validation = ConfigValidation()

        # Get schema
        schema = self._schema_map.get(key)

        # Validate if schema exists
        if schema:
            is_valid, error = schema.validate(value)
            if not is_valid:
                validation.is_valid = False
                validation.errors.append(f"{key}: {error}")
                return validation

        # Track old value for change event
        old_value = self._values.get(key)

        # Set value
        self._values[key] = ConfigValue(
            key=key,
            value=value,
            source=source,
            value_type=schema.value_type if schema else ConfigValueType.STRING,
            default=schema.default if schema else None,
            description=schema.description if schema else "",
            required=schema.required if schema else False,
            sensitive=schema.sensitive if schema else False,
        )

        # Emit change event
        if old_value and old_value.value != value:
            change = ConfigChangeEvent(
                key=key,
                old_value=old_value.value,
                new_value=value,
                source=source,
            )

            self._emit_event("change", change.to_dict())

            # Notify handlers
            for handler in self._change_handlers:
                try:
                    handler(change)
                except Exception:
                    pass

        return validation

    def validate(self) -> ConfigValidation:
        """
        Validate the current configuration.

        Returns:
            ConfigValidation result
        """
        validation = ConfigValidation()

        for schema in self.schema:
            # Check required values
            if schema.required:
                if schema.key not in self._values:
                    validation.is_valid = False
                    validation.errors.append(f"Missing required config: {schema.key}")
                    continue

            # Validate value
            if schema.key in self._values:
                value = self._values[schema.key].value
                is_valid, error = schema.validate(value)
                if not is_valid:
                    validation.is_valid = False
                    validation.errors.append(f"{schema.key}: {error}")

        self._emit_event("validate", validation.to_dict())

        return validation

    def save(self, file_path: Optional[str] = None) -> bool:
        """
        Save configuration to file.

        Args:
            file_path: Output file path

        Returns:
            True if saved successfully
        """
        path = Path(file_path) if file_path else self.config_file
        if path is None:
            return False

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # Build config dict (excluding sensitive values in saved state marker)
            config_data = {}
            for key, config_value in self._values.items():
                if not config_value.sensitive:
                    config_data[key] = config_value.value

            with open(path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(config_data, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            return True

        except IOError:
            return False

    def on_change(self, handler: Callable[[ConfigChangeEvent], None]) -> None:
        """Register a change handler."""
        self._change_handlers.append(handler)

    def remove_change_handler(self, handler: Callable[[ConfigChangeEvent], None]) -> bool:
        """Remove a change handler."""
        if handler in self._change_handlers:
            self._change_handlers.remove(handler)
            return True
        return False

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Args:
            include_sensitive: Include sensitive values

        Returns:
            Configuration dictionary
        """
        result = {}
        for key, config_value in self._values.items():
            if config_value.sensitive and not include_sensitive:
                result[key] = "<redacted>"
            else:
                result[key] = config_value.value
        return result

    def get_all_values(self) -> Dict[str, ConfigValue]:
        """Get all configuration values."""
        return self._values.copy()

    def get_schema(self) -> List[ConfigSchema]:
        """Get configuration schema."""
        return self.schema

    def add_schema(self, schema: ConfigSchema) -> None:
        """Add a configuration schema."""
        self.schema.append(schema)
        self._schema_map[schema.key] = schema
        if schema.default is not None and schema.key not in self._values:
            self._values[schema.key] = ConfigValue(
                key=schema.key,
                value=schema.default,
                source=ConfigSource.DEFAULT,
                value_type=schema.value_type,
                default=schema.default,
            )

    def reset_to_defaults(self) -> None:
        """Reset all values to defaults."""
        self._values.clear()
        self._apply_defaults()

    async def load_async(self, config_file: Optional[str] = None) -> ConfigValidation:
        """Async version of load."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, config_file)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.config.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "config",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Config Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Config Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show configuration")
    show_parser.add_argument("--key", help="Specific key to show")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set a configuration value")
    set_parser.add_argument("key", help="Configuration key")
    set_parser.add_argument("value", help="Configuration value")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")

    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Show configuration schema")

    # Save command
    save_parser = subparsers.add_parser("save", help="Save configuration to file")
    save_parser.add_argument("--output", "-o", help="Output file path")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset to defaults")

    # Common arguments
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config_file = args.config or ".pluribus/test-agent/config.json"
    manager = TestConfigManager(config_file=config_file)
    manager.load()

    if args.command == "show":
        if args.key:
            value = manager.get_value(args.key)
            if value:
                if args.json:
                    print(json.dumps(value.to_dict(), indent=2))
                else:
                    print(f"{args.key}: {value.value}")
                    print(f"  Source: {value.source.value}")
                    print(f"  Type: {value.value_type.value}")
            else:
                print(f"Key not found: {args.key}")
        else:
            config = manager.to_dict()
            if args.json:
                print(json.dumps(config, indent=2))
            else:
                print("\nConfiguration:")
                for key, value in config.items():
                    print(f"  {key}: {value}")

    elif args.command == "set":
        # Try to auto-detect type
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value

        validation = manager.set(args.key, value)
        if validation.is_valid:
            manager.save()
            print(f"Set {args.key} = {value}")
        else:
            print("Validation errors:")
            for error in validation.errors:
                print(f"  - {error}")

    elif args.command == "validate":
        validation = manager.validate()

        if args.json:
            print(json.dumps(validation.to_dict(), indent=2))
        else:
            if validation.is_valid:
                print("Configuration is valid")
            else:
                print("Configuration is invalid:")
                for error in validation.errors:
                    print(f"  - {error}")
            if validation.warnings:
                print("Warnings:")
                for warning in validation.warnings:
                    print(f"  - {warning}")

    elif args.command == "schema":
        schema_list = [s.to_dict() for s in manager.get_schema()]

        if args.json:
            print(json.dumps(schema_list, indent=2))
        else:
            print("\nConfiguration Schema:")
            for schema in manager.get_schema():
                required = "*" if schema.required else ""
                default = f" (default: {schema.default})" if schema.default is not None else ""
                print(f"  {schema.key}{required} [{schema.value_type.value}]{default}")
                if schema.description:
                    print(f"    {schema.description}")

    elif args.command == "save":
        output = args.output or config_file
        if manager.save(output):
            print(f"Configuration saved to: {output}")
        else:
            print("Failed to save configuration")

    elif args.command == "reset":
        manager.reset_to_defaults()
        manager.save()
        print("Configuration reset to defaults")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
