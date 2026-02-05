#!/usr/bin/env python3
"""
injector.py - Config Injector (Step 212)

PBTSO Phase: SEQUESTER
A2A Integration: Injects configuration via deploy.config.inject

Provides:
- ConfigSource: Source types for configuration
- ConfigFormat: Configuration file formats
- ConfigEntry: Configuration entry definition
- InjectionTarget: Target for configuration injection
- ConfigInjector: Runtime configuration injection

Bus Topics:
- deploy.config.inject
- deploy.config.validate
- deploy.config.reload
- deploy.config.diff

Protocol: DKIN v30, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


# ==============================================================================
# Bus Emission Helper
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
    actor: str = "config-injector"
) -> str:
    """Emit an event to the Pluribus bus."""
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
            f.write(json.dumps(event) + "\n")
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class ConfigSource(Enum):
    """Source types for configuration."""
    FILE = "file"
    ENVIRONMENT = "environment"
    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"
    AWS_PARAMETER_STORE = "aws_parameter_store"
    GCP_RUNTIME_CONFIG = "gcp_runtime_config"
    VAULT = "vault"
    INLINE = "inline"


class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"
    PROPERTIES = "properties"


class InjectionTarget(Enum):
    """Target for configuration injection."""
    ENVIRONMENT = "environment"
    FILE = "file"
    CONFIGMAP = "configmap"
    SECRET = "secret"
    CONSUL_KV = "consul_kv"
    ETCD_KV = "etcd_kv"


@dataclass
class ConfigEntry:
    """
    Configuration entry definition.

    Attributes:
        config_id: Unique configuration identifier
        name: Human-readable config name
        source: Configuration source
        source_path: Path or key in source
        format: Configuration format
        environment: Target environment
        service: Target service
        values: Configuration values
        schema: Optional JSON schema for validation
        version: Configuration version
        created_at: Timestamp when created
        updated_at: Timestamp when last updated
    """
    config_id: str
    name: str
    source: ConfigSource = ConfigSource.INLINE
    source_path: str = ""
    format: ConfigFormat = ConfigFormat.JSON
    environment: str = "prod"
    service: str = ""
    values: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[Dict[str, Any]] = None
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "name": self.name,
            "source": self.source.value,
            "source_path": self.source_path,
            "format": self.format.value,
            "environment": self.environment,
            "service": self.service,
            "values": self.values,
            "schema": self.schema,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigEntry":
        data = dict(data)
        if "source" in data:
            data["source"] = ConfigSource(data["source"])
        if "format" in data:
            data["format"] = ConfigFormat(data["format"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InjectionSpec:
    """Specification for a configuration injection."""
    config_id: str
    target: InjectionTarget
    target_path: str  # Path, configmap name, etc.
    keys: List[str] = field(default_factory=list)  # Specific keys to inject (empty = all)
    prefix: str = ""  # Prefix for injected keys
    transform: Optional[str] = None  # Transform function name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InjectionResult:
    """Result of a configuration injection."""
    success: bool
    config_id: str
    target: str
    target_path: str
    keys_injected: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Config Injector (Step 212)
# ==============================================================================

class ConfigInjector:
    """
    Config Injector - runtime configuration injection for deployments.

    PBTSO Phase: SEQUESTER

    Responsibilities:
    - Load configuration from multiple sources
    - Validate configuration against schemas
    - Inject configuration into deployment targets
    - Support hot reload of configurations
    - Track configuration changes

    Example:
        >>> injector = ConfigInjector()
        >>> entry = injector.create_config(
        ...     name="api-config",
        ...     values={"port": 8080, "debug": False},
        ...     service="api-service"
        ... )
        >>> result = await injector.inject(
        ...     config_id=entry.config_id,
        ...     target=InjectionTarget.ENVIRONMENT,
        ...     target_path="deployment/api"
        ... )
    """

    BUS_TOPICS = {
        "inject": "deploy.config.inject",
        "validate": "deploy.config.validate",
        "reload": "deploy.config.reload",
        "diff": "deploy.config.diff",
        "created": "deploy.config.created",
        "updated": "deploy.config.updated",
    }

    # Built-in value transformers
    TRANSFORMERS = {
        "uppercase": lambda v: v.upper() if isinstance(v, str) else v,
        "lowercase": lambda v: v.lower() if isinstance(v, str) else v,
        "string": lambda v: str(v),
        "int": lambda v: int(v),
        "bool": lambda v: bool(v),
        "json": lambda v: json.dumps(v) if not isinstance(v, str) else v,
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "config-injector",
    ):
        """
        Initialize the config injector.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "configs"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._configs: Dict[str, ConfigEntry] = {}
        self._watchers: Dict[str, List[callable]] = {}

        self._load_configs()

    def create_config(
        self,
        name: str,
        values: Dict[str, Any],
        source: ConfigSource = ConfigSource.INLINE,
        source_path: str = "",
        format: ConfigFormat = ConfigFormat.JSON,
        environment: str = "prod",
        service: str = "",
        schema: Optional[Dict[str, Any]] = None,
    ) -> ConfigEntry:
        """
        Create a new configuration entry.

        Args:
            name: Human-readable config name
            values: Configuration values
            source: Configuration source
            source_path: Path in source
            format: Configuration format
            environment: Target environment
            service: Target service
            schema: Optional JSON schema

        Returns:
            Created ConfigEntry
        """
        config_id = f"config-{uuid.uuid4().hex[:12]}"

        entry = ConfigEntry(
            config_id=config_id,
            name=name,
            source=source,
            source_path=source_path,
            format=format,
            environment=environment,
            service=service,
            values=values,
            schema=schema,
        )

        self._configs[config_id] = entry
        self._save_config(entry)

        _emit_bus_event(
            self.BUS_TOPICS["created"],
            {
                "config_id": config_id,
                "name": name,
                "environment": environment,
                "service": service,
                "key_count": len(values),
            },
            actor=self.actor_id,
        )

        return entry

    def update_config(
        self,
        config_id: str,
        values: Dict[str, Any],
        merge: bool = True,
    ) -> Optional[ConfigEntry]:
        """
        Update configuration values.

        Args:
            config_id: Configuration ID
            values: New values
            merge: Whether to merge with existing (True) or replace (False)

        Returns:
            Updated ConfigEntry or None if not found
        """
        entry = self._configs.get(config_id)
        if not entry:
            return None

        old_values = dict(entry.values)

        if merge:
            entry.values.update(values)
        else:
            entry.values = values

        entry.version += 1
        entry.updated_at = time.time()

        self._save_config(entry)

        # Emit diff event
        diff = self._compute_diff(old_values, entry.values)
        _emit_bus_event(
            self.BUS_TOPICS["diff"],
            {
                "config_id": config_id,
                "old_version": entry.version - 1,
                "new_version": entry.version,
                "added": diff["added"],
                "removed": diff["removed"],
                "changed": diff["changed"],
            },
            actor=self.actor_id,
        )

        _emit_bus_event(
            self.BUS_TOPICS["updated"],
            {
                "config_id": config_id,
                "name": entry.name,
                "version": entry.version,
            },
            actor=self.actor_id,
        )

        # Notify watchers
        self._notify_watchers(config_id, entry)

        return entry

    async def load_from_source(
        self,
        config_id: str,
    ) -> Optional[ConfigEntry]:
        """
        Reload configuration from its source.

        Args:
            config_id: Configuration ID

        Returns:
            Updated ConfigEntry or None
        """
        entry = self._configs.get(config_id)
        if not entry:
            return None

        new_values = await self._load_source(entry.source, entry.source_path, entry.format)
        if new_values is None:
            return None

        return self.update_config(config_id, new_values, merge=False)

    async def _load_source(
        self,
        source: ConfigSource,
        path: str,
        format: ConfigFormat,
    ) -> Optional[Dict[str, Any]]:
        """Load values from a configuration source."""
        try:
            if source == ConfigSource.FILE:
                return self._load_from_file(path, format)

            elif source == ConfigSource.ENVIRONMENT:
                return self._load_from_env(path)

            elif source == ConfigSource.INLINE:
                return None  # No external source

            # Simulate loading from external sources
            await asyncio.sleep(0.05)
            return None

        except Exception:
            return None

    def _load_from_file(self, path: str, format: ConfigFormat) -> Optional[Dict[str, Any]]:
        """Load configuration from a file."""
        file_path = Path(path)
        if not file_path.exists():
            return None

        content = file_path.read_text()

        if format == ConfigFormat.JSON:
            return json.loads(content)
        elif format == ConfigFormat.YAML:
            return yaml.safe_load(content)
        elif format == ConfigFormat.ENV:
            return self._parse_env_format(content)
        elif format == ConfigFormat.PROPERTIES:
            return self._parse_properties_format(content)

        return None

    def _load_from_env(self, prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        values = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lstrip("_").lower()
                values[config_key] = value
        return values

    def _parse_env_format(self, content: str) -> Dict[str, Any]:
        """Parse .env format."""
        values = {}
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
        return values

    def _parse_properties_format(self, content: str) -> Dict[str, Any]:
        """Parse Java properties format."""
        values = {}
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
        return values

    def validate(
        self,
        config_id: str,
        additional_rules: Optional[List[callable]] = None,
    ) -> ValidationResult:
        """
        Validate configuration against schema and rules.

        Args:
            config_id: Configuration ID
            additional_rules: Additional validation rules

        Returns:
            ValidationResult
        """
        entry = self._configs.get(config_id)
        if not entry:
            return ValidationResult(valid=False, errors=["Configuration not found"])

        errors = []
        warnings = []

        # Validate against schema if provided
        if entry.schema:
            schema_errors = self._validate_schema(entry.values, entry.schema)
            errors.extend(schema_errors)

        # Run additional rules
        if additional_rules:
            for rule in additional_rules:
                try:
                    result = rule(entry.values)
                    if isinstance(result, str):
                        errors.append(result)
                except Exception as e:
                    errors.append(f"Rule validation error: {e}")

        # Common validations
        for key, value in entry.values.items():
            # Check for empty strings that should probably be None
            if value == "":
                warnings.append(f"Key '{key}' has empty string value")

            # Check for placeholder values
            if isinstance(value, str) and value.startswith("${"):
                errors.append(f"Key '{key}' contains unresolved placeholder: {value}")

        result = ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

        _emit_bus_event(
            self.BUS_TOPICS["validate"],
            {
                "config_id": config_id,
                "valid": result.valid,
                "error_count": len(errors),
                "warning_count": len(warnings),
            },
            actor=self.actor_id,
        )

        return result

    def _validate_schema(self, values: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate values against a JSON schema."""
        errors = []

        # Simple schema validation
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field in required:
            if field not in values:
                errors.append(f"Required field '{field}' is missing")

        # Check property types
        for key, prop_schema in properties.items():
            if key in values:
                value = values[key]
                expected_type = prop_schema.get("type")

                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Field '{key}' should be string, got {type(value).__name__}")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"Field '{key}' should be integer, got {type(value).__name__}")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Field '{key}' should be boolean, got {type(value).__name__}")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{key}' should be number, got {type(value).__name__}")

        return errors

    async def inject(
        self,
        config_id: str,
        target: InjectionTarget,
        target_path: str,
        keys: Optional[List[str]] = None,
        prefix: str = "",
        transform: Optional[str] = None,
    ) -> InjectionResult:
        """
        Inject configuration into a target.

        Args:
            config_id: Configuration ID
            target: Injection target type
            target_path: Path/name of target
            keys: Specific keys to inject (empty = all)
            prefix: Prefix for injected keys
            transform: Transform function name

        Returns:
            InjectionResult
        """
        entry = self._configs.get(config_id)
        if not entry:
            return InjectionResult(
                success=False,
                config_id=config_id,
                target=target.value,
                target_path=target_path,
                keys_injected=[],
                error="Configuration not found",
            )

        # Select keys to inject
        values_to_inject = entry.values
        if keys:
            values_to_inject = {k: v for k, v in entry.values.items() if k in keys}

        # Apply transform
        if transform and transform in self.TRANSFORMERS:
            transformer = self.TRANSFORMERS[transform]
            values_to_inject = {k: transformer(v) for k, v in values_to_inject.items()}

        # Apply prefix
        if prefix:
            values_to_inject = {f"{prefix}{k}": v for k, v in values_to_inject.items()}

        # Perform injection
        try:
            await self._perform_injection(target, target_path, values_to_inject)
            keys_injected = list(values_to_inject.keys())

            _emit_bus_event(
                self.BUS_TOPICS["inject"],
                {
                    "config_id": config_id,
                    "target": target.value,
                    "target_path": target_path,
                    "keys_injected": keys_injected,
                    "key_count": len(keys_injected),
                },
                actor=self.actor_id,
            )

            return InjectionResult(
                success=True,
                config_id=config_id,
                target=target.value,
                target_path=target_path,
                keys_injected=keys_injected,
            )

        except Exception as e:
            return InjectionResult(
                success=False,
                config_id=config_id,
                target=target.value,
                target_path=target_path,
                keys_injected=[],
                error=str(e),
            )

    async def _perform_injection(
        self,
        target: InjectionTarget,
        target_path: str,
        values: Dict[str, Any],
    ) -> None:
        """Perform the actual injection."""
        if target == InjectionTarget.ENVIRONMENT:
            # Set environment variables
            for key, value in values.items():
                os.environ[key] = str(value)

        elif target == InjectionTarget.FILE:
            # Write to file
            path = Path(target_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if target_path.endswith(".json"):
                with open(path, "w") as f:
                    json.dump(values, f, indent=2)
            elif target_path.endswith((".yaml", ".yml")):
                with open(path, "w") as f:
                    yaml.dump(values, f, default_flow_style=False)
            elif target_path.endswith(".env"):
                with open(path, "w") as f:
                    for key, value in values.items():
                        f.write(f'{key}="{value}"\n')
            else:
                # Default to JSON
                with open(path, "w") as f:
                    json.dump(values, f, indent=2)

        elif target in (InjectionTarget.CONFIGMAP, InjectionTarget.SECRET):
            # Simulate Kubernetes resource update
            await asyncio.sleep(0.05)

        else:
            # Simulate other backends
            await asyncio.sleep(0.02)

    def render_template(
        self,
        config_id: str,
        template: str,
    ) -> str:
        """
        Render a template with configuration values.

        Args:
            config_id: Configuration ID
            template: Template string with ${key} placeholders

        Returns:
            Rendered template
        """
        entry = self._configs.get(config_id)
        if not entry:
            return template

        result = template
        for key, value in entry.values.items():
            placeholder = f"${{{key}}}"
            result = result.replace(placeholder, str(value))

        return result

    def get_merged_config(
        self,
        config_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Get merged configuration from multiple sources.

        Later configs override earlier ones.

        Args:
            config_ids: List of configuration IDs to merge

        Returns:
            Merged configuration values
        """
        merged = {}
        for config_id in config_ids:
            entry = self._configs.get(config_id)
            if entry:
                merged.update(entry.values)
        return merged

    def watch(
        self,
        config_id: str,
        callback: callable,
    ) -> None:
        """Register a callback for configuration changes."""
        if config_id not in self._watchers:
            self._watchers[config_id] = []
        self._watchers[config_id].append(callback)

    def _notify_watchers(self, config_id: str, entry: ConfigEntry) -> None:
        """Notify watchers of configuration changes."""
        for callback in self._watchers.get(config_id, []):
            try:
                callback(entry)
            except Exception:
                pass

    def _compute_diff(
        self,
        old: Dict[str, Any],
        new: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """Compute diff between two configurations."""
        old_keys = set(old.keys())
        new_keys = set(new.keys())

        added = list(new_keys - old_keys)
        removed = list(old_keys - new_keys)
        changed = [k for k in old_keys & new_keys if old[k] != new[k]]

        return {"added": added, "removed": removed, "changed": changed}

    def get_config(self, config_id: str) -> Optional[ConfigEntry]:
        """Get configuration by ID."""
        return self._configs.get(config_id)

    def get_config_by_name(
        self,
        name: str,
        environment: Optional[str] = None,
    ) -> Optional[ConfigEntry]:
        """Get configuration by name."""
        for entry in self._configs.values():
            if entry.name == name:
                if environment is None or entry.environment == environment:
                    return entry
        return None

    def list_configs(
        self,
        environment: Optional[str] = None,
        service: Optional[str] = None,
    ) -> List[ConfigEntry]:
        """List configurations with optional filters."""
        configs = list(self._configs.values())

        if environment:
            configs = [c for c in configs if c.environment == environment]

        if service:
            configs = [c for c in configs if c.service == service]

        return configs

    def delete_config(self, config_id: str) -> bool:
        """Delete a configuration."""
        if config_id not in self._configs:
            return False

        del self._configs[config_id]

        config_file = self.state_dir / f"{config_id}.json"
        if config_file.exists():
            config_file.unlink()

        return True

    def _save_config(self, entry: ConfigEntry) -> None:
        """Save configuration to disk."""
        config_file = self.state_dir / f"{entry.config_id}.json"
        with open(config_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def _load_configs(self) -> None:
        """Load configurations from disk."""
        for config_file in self.state_dir.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                entry = ConfigEntry.from_dict(data)
                self._configs[entry.config_id] = entry
            except (json.JSONDecodeError, KeyError, IOError):
                continue


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for config injector."""
    import argparse

    parser = argparse.ArgumentParser(description="Config Injector (Step 212)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a configuration")
    create_parser.add_argument("name", help="Config name")
    create_parser.add_argument("--values", "-v", required=True, help="JSON values or @file")
    create_parser.add_argument("--env", "-e", default="prod", help="Environment")
    create_parser.add_argument("--service", "-s", default="", help="Service name")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # get command
    get_parser = subparsers.add_parser("get", help="Get configuration")
    get_parser.add_argument("config_id", help="Config ID")
    get_parser.add_argument("--key", "-k", help="Get specific key")
    get_parser.add_argument("--json", action="store_true", help="JSON output")

    # update command
    update_parser = subparsers.add_parser("update", help="Update configuration")
    update_parser.add_argument("config_id", help="Config ID")
    update_parser.add_argument("--values", "-v", required=True, help="JSON values")
    update_parser.add_argument("--replace", action="store_true", help="Replace instead of merge")

    # inject command
    inject_parser = subparsers.add_parser("inject", help="Inject configuration")
    inject_parser.add_argument("config_id", help="Config ID")
    inject_parser.add_argument("--target", "-t", default="environment",
                              choices=["environment", "file", "configmap", "secret"])
    inject_parser.add_argument("--path", "-p", required=True, help="Target path")
    inject_parser.add_argument("--prefix", default="", help="Key prefix")
    inject_parser.add_argument("--keys", "-k", help="Comma-separated keys to inject")
    inject_parser.add_argument("--json", action="store_true", help="JSON output")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("config_id", help="Config ID")
    validate_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List configurations")
    list_parser.add_argument("--env", "-e", help="Filter by environment")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # render command
    render_parser = subparsers.add_parser("render", help="Render template")
    render_parser.add_argument("config_id", help="Config ID")
    render_parser.add_argument("--template", "-t", required=True, help="Template string or @file")

    args = parser.parse_args()
    injector = ConfigInjector()

    if args.command == "create":
        # Parse values
        values_str = args.values
        if values_str.startswith("@"):
            with open(values_str[1:], "r") as f:
                values = json.load(f)
        else:
            values = json.loads(values_str)

        entry = injector.create_config(
            name=args.name,
            values=values,
            environment=args.env,
            service=args.service,
        )

        if args.json:
            print(json.dumps(entry.to_dict(), indent=2))
        else:
            print(f"Created config: {entry.config_id}")
            print(f"  Name: {entry.name}")
            print(f"  Environment: {entry.environment}")
            print(f"  Keys: {len(entry.values)}")

        return 0

    elif args.command == "get":
        entry = injector.get_config(args.config_id)
        if not entry:
            print(f"Config not found: {args.config_id}")
            return 1

        if args.key:
            value = entry.values.get(args.key)
            if value is not None:
                print(value)
            else:
                print(f"Key not found: {args.key}")
                return 1
        elif args.json:
            print(json.dumps(entry.to_dict(), indent=2))
        else:
            print(f"Config: {entry.config_id}")
            print(f"  Name: {entry.name}")
            print(f"  Version: {entry.version}")
            print(f"  Values:")
            for k, v in entry.values.items():
                print(f"    {k}: {v}")

        return 0

    elif args.command == "update":
        values = json.loads(args.values)
        entry = injector.update_config(
            args.config_id,
            values,
            merge=not args.replace,
        )

        if entry:
            print(f"Updated config: {entry.config_id} (v{entry.version})")
        else:
            print(f"Config not found: {args.config_id}")
            return 1

        return 0

    elif args.command == "inject":
        target = InjectionTarget(args.target)
        keys = args.keys.split(",") if args.keys else None

        result = asyncio.get_event_loop().run_until_complete(
            injector.inject(
                config_id=args.config_id,
                target=target,
                target_path=args.path,
                keys=keys,
                prefix=args.prefix,
            )
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(f"Injected {len(result.keys_injected)} keys to {args.target}:{args.path}")
            else:
                print(f"Injection failed: {result.error}")

        return 0 if result.success else 1

    elif args.command == "validate":
        result = injector.validate(args.config_id)

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

    elif args.command == "list":
        configs = injector.list_configs(
            environment=args.env,
            service=args.service,
        )

        if args.json:
            print(json.dumps([c.to_dict() for c in configs], indent=2))
        else:
            if not configs:
                print("No configurations found")
            else:
                for c in configs:
                    print(f"{c.config_id} ({c.name}) - {c.environment} v{c.version}")

        return 0

    elif args.command == "render":
        template = args.template
        if template.startswith("@"):
            with open(template[1:], "r") as f:
                template = f.read()

        result = injector.render_template(args.config_id, template)
        print(result)
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
