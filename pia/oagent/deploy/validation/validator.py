#!/usr/bin/env python3
"""
validator.py - Validation Engine (Step 242)

PBTSO Phase: VERIFY
A2A Integration: Validates inputs/outputs via deploy.validation.validate

Provides:
- ValidationType: Types of validation
- ValidationRule: Validation rule definition
- ValidationResult: Validation result
- SchemaValidator: JSON schema validation
- InputValidator: Input validation
- OutputValidator: Output validation
- ValidationEngine: Complete validation engine

Bus Topics:
- deploy.validation.validate
- deploy.validation.passed
- deploy.validation.failed
- deploy.validation.schema

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union


# ==============================================================================
# Bus Emission Helper with fcntl.flock()
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
    actor: str = "validation-engine"
) -> str:
    """Emit an event to the Pluribus bus with file locking."""
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

class ValidationType(Enum):
    """Types of validation."""
    SCHEMA = "schema"
    TYPE = "type"
    RANGE = "range"
    PATTERN = "pattern"
    REQUIRED = "required"
    CUSTOM = "custom"
    ENUM = "enum"
    LENGTH = "length"
    FORMAT = "format"
    DEPENDENCY = "dependency"


class ValidationSeverity(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class FormatType(Enum):
    """Predefined format types."""
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    DATE = "date"
    DATETIME = "datetime"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    HOSTNAME = "hostname"
    URI = "uri"
    SEMVER = "semver"


@dataclass
class ValidationRule:
    """
    Validation rule definition.

    Attributes:
        rule_id: Unique rule identifier
        name: Human-readable rule name
        validation_type: Type of validation
        path: JSONPath or field path
        params: Rule parameters
        message: Error message template
        severity: Validation severity
        enabled: Whether rule is enabled
    """
    rule_id: str
    name: str
    validation_type: ValidationType
    path: str = "$"
    params: Dict[str, Any] = field(default_factory=dict)
    message: str = "Validation failed"
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "validation_type": self.validation_type.value,
            "path": self.path,
            "params": self.params,
            "message": self.message,
            "severity": self.severity.value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationRule":
        data = dict(data)
        if "validation_type" in data:
            data["validation_type"] = ValidationType(data["validation_type"])
        if "severity" in data:
            data["severity"] = ValidationSeverity(data["severity"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ValidationError:
    """
    Single validation error.

    Attributes:
        path: Field path where error occurred
        rule_id: Rule that failed
        message: Error message
        severity: Error severity
        actual_value: Actual value (sanitized)
        expected: Expected value/format
    """
    path: str
    rule_id: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    actual_value: Optional[Any] = None
    expected: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "rule_id": self.rule_id,
            "message": self.message,
            "severity": self.severity.value,
            "actual_value": str(self.actual_value)[:100] if self.actual_value else None,
            "expected": self.expected,
        }


@dataclass
class ValidationResult:
    """
    Validation result.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        info: Informational messages
        validated_at: Validation timestamp
        duration_ms: Validation duration
        rules_applied: Number of rules applied
        metadata: Additional metadata
    """
    valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    validated_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    rules_applied: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
            "validated_at": self.validated_at,
            "duration_ms": self.duration_ms,
            "rules_applied": self.rules_applied,
            "metadata": self.metadata,
        }

    def add_error(self, error: ValidationError) -> None:
        """Add an error and update validity."""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.valid = False
        elif error.severity == ValidationSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info.append(error)


# ==============================================================================
# Schema Validator
# ==============================================================================

class SchemaValidator:
    """
    JSON Schema-style validator.

    Supports:
    - type validation
    - required fields
    - enum values
    - pattern matching
    - min/max constraints
    - nested objects and arrays
    """

    # Format patterns
    FORMAT_PATTERNS = {
        FormatType.EMAIL: re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
        FormatType.URL: re.compile(r"^https?://[^\s]+$"),
        FormatType.UUID: re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
        FormatType.DATE: re.compile(r"^\d{4}-\d{2}-\d{2}$"),
        FormatType.DATETIME: re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
        FormatType.IPV4: re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"),
        FormatType.HOSTNAME: re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"),
        FormatType.SEMVER: re.compile(r"^v?\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"),
    }

    def __init__(self, schema: Dict[str, Any]):
        """Initialize with a schema."""
        self.schema = schema

    def validate(self, data: Any, path: str = "$") -> ValidationResult:
        """Validate data against the schema."""
        result = ValidationResult()
        self._validate_value(data, self.schema, path, result)
        return result

    def _validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate a single value against schema."""
        result.rules_applied += 1

        # Check type
        expected_type = schema.get("type")
        if expected_type:
            if not self._check_type(value, expected_type):
                result.add_error(ValidationError(
                    path=path,
                    rule_id="type",
                    message=f"Expected type {expected_type}, got {type(value).__name__}",
                    actual_value=type(value).__name__,
                    expected=expected_type,
                ))
                return  # Skip further validation on type mismatch

        # Check enum
        enum_values = schema.get("enum")
        if enum_values is not None:
            if value not in enum_values:
                result.add_error(ValidationError(
                    path=path,
                    rule_id="enum",
                    message=f"Value must be one of: {enum_values}",
                    actual_value=value,
                    expected=enum_values,
                ))

        # Check pattern
        pattern = schema.get("pattern")
        if pattern and isinstance(value, str):
            if not re.match(pattern, value):
                result.add_error(ValidationError(
                    path=path,
                    rule_id="pattern",
                    message=f"Value does not match pattern: {pattern}",
                    actual_value=value,
                    expected=pattern,
                ))

        # Check format
        format_type = schema.get("format")
        if format_type and isinstance(value, str):
            try:
                fmt = FormatType(format_type)
                pattern_re = self.FORMAT_PATTERNS.get(fmt)
                if pattern_re and not pattern_re.match(value):
                    result.add_error(ValidationError(
                        path=path,
                        rule_id="format",
                        message=f"Value does not match format: {format_type}",
                        actual_value=value,
                        expected=format_type,
                    ))
            except ValueError:
                pass  # Unknown format, skip

        # Check string constraints
        if isinstance(value, str):
            min_len = schema.get("minLength")
            if min_len is not None and len(value) < min_len:
                result.add_error(ValidationError(
                    path=path,
                    rule_id="minLength",
                    message=f"String too short (min {min_len})",
                    actual_value=len(value),
                    expected=min_len,
                ))

            max_len = schema.get("maxLength")
            if max_len is not None and len(value) > max_len:
                result.add_error(ValidationError(
                    path=path,
                    rule_id="maxLength",
                    message=f"String too long (max {max_len})",
                    actual_value=len(value),
                    expected=max_len,
                ))

        # Check number constraints
        if isinstance(value, (int, float)):
            minimum = schema.get("minimum")
            if minimum is not None and value < minimum:
                result.add_error(ValidationError(
                    path=path,
                    rule_id="minimum",
                    message=f"Value below minimum ({minimum})",
                    actual_value=value,
                    expected=f">= {minimum}",
                ))

            maximum = schema.get("maximum")
            if maximum is not None and value > maximum:
                result.add_error(ValidationError(
                    path=path,
                    rule_id="maximum",
                    message=f"Value above maximum ({maximum})",
                    actual_value=value,
                    expected=f"<= {maximum}",
                ))

        # Check array constraints
        if isinstance(value, list):
            min_items = schema.get("minItems")
            if min_items is not None and len(value) < min_items:
                result.add_error(ValidationError(
                    path=path,
                    rule_id="minItems",
                    message=f"Array too short (min {min_items} items)",
                    actual_value=len(value),
                    expected=min_items,
                ))

            max_items = schema.get("maxItems")
            if max_items is not None and len(value) > max_items:
                result.add_error(ValidationError(
                    path=path,
                    rule_id="maxItems",
                    message=f"Array too long (max {max_items} items)",
                    actual_value=len(value),
                    expected=max_items,
                ))

            # Validate array items
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(value):
                    self._validate_value(item, items_schema, f"{path}[{i}]", result)

        # Check object constraints
        if isinstance(value, dict):
            # Check required fields
            required = schema.get("required", [])
            for req_field in required:
                if req_field not in value:
                    result.add_error(ValidationError(
                        path=f"{path}.{req_field}",
                        rule_id="required",
                        message=f"Required field missing: {req_field}",
                        expected="required",
                    ))

            # Validate properties
            properties = schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_name in value:
                    self._validate_value(
                        value[prop_name],
                        prop_schema,
                        f"{path}.{prop_name}",
                        result,
                    )

            # Check additional properties
            additional = schema.get("additionalProperties")
            if additional is False:
                allowed_keys = set(properties.keys())
                for key in value.keys():
                    if key not in allowed_keys:
                        result.add_error(ValidationError(
                            path=f"{path}.{key}",
                            rule_id="additionalProperties",
                            message=f"Additional property not allowed: {key}",
                            actual_value=key,
                        ))

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True


# ==============================================================================
# Input Validator
# ==============================================================================

class InputValidator:
    """
    Input validation for deployment configurations.

    Validates:
    - Deployment configs
    - Environment variables
    - Service configurations
    - Resource specifications
    """

    # Common validation rules
    COMMON_RULES = {
        "service_name": ValidationRule(
            rule_id="service_name",
            name="Service Name",
            validation_type=ValidationType.PATTERN,
            path="$.service_name",
            params={"pattern": r"^[a-z][a-z0-9-]{0,62}$"},
            message="Service name must be lowercase alphanumeric with hyphens, max 63 chars",
        ),
        "version": ValidationRule(
            rule_id="version",
            name="Version",
            validation_type=ValidationType.PATTERN,
            path="$.version",
            params={"pattern": r"^v?\d+\.\d+\.\d+"},
            message="Version must be semver format (e.g., v1.0.0)",
        ),
        "replicas": ValidationRule(
            rule_id="replicas",
            name="Replicas",
            validation_type=ValidationType.RANGE,
            path="$.replicas",
            params={"min": 1, "max": 100},
            message="Replicas must be between 1 and 100",
        ),
        "port": ValidationRule(
            rule_id="port",
            name="Port",
            validation_type=ValidationType.RANGE,
            path="$.port",
            params={"min": 1, "max": 65535},
            message="Port must be between 1 and 65535",
        ),
        "cpu_limit": ValidationRule(
            rule_id="cpu_limit",
            name="CPU Limit",
            validation_type=ValidationType.PATTERN,
            path="$.resources.limits.cpu",
            params={"pattern": r"^\d+m?$"},
            message="CPU limit must be in millicores (e.g., 500m) or cores (e.g., 2)",
        ),
        "memory_limit": ValidationRule(
            rule_id="memory_limit",
            name="Memory Limit",
            validation_type=ValidationType.PATTERN,
            path="$.resources.limits.memory",
            params={"pattern": r"^\d+[KMGi]+$"},
            message="Memory limit must include unit (e.g., 512Mi, 2Gi)",
        ),
    }

    def __init__(self):
        """Initialize input validator."""
        self._rules: Dict[str, ValidationRule] = dict(self.COMMON_RULES)
        self._custom_validators: Dict[str, Callable] = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self._rules[rule.rule_id] = rule

    def add_custom_validator(
        self,
        name: str,
        validator: Callable[[Any], Union[bool, str]],
    ) -> None:
        """Add a custom validation function."""
        self._custom_validators[name] = validator

    def validate(
        self,
        data: Dict[str, Any],
        rules: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate input data.

        Args:
            data: Input data to validate
            rules: Specific rules to apply (None = all)

        Returns:
            ValidationResult
        """
        start_time = time.time()
        result = ValidationResult()

        rules_to_apply = []
        if rules:
            rules_to_apply = [self._rules[r] for r in rules if r in self._rules]
        else:
            rules_to_apply = list(self._rules.values())

        for rule in rules_to_apply:
            if not rule.enabled:
                continue

            value = self._get_value_at_path(data, rule.path)
            if value is None and rule.validation_type != ValidationType.REQUIRED:
                continue

            self._apply_rule(rule, value, result)
            result.rules_applied += 1

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def _apply_rule(
        self,
        rule: ValidationRule,
        value: Any,
        result: ValidationResult,
    ) -> None:
        """Apply a single validation rule."""
        error = None

        if rule.validation_type == ValidationType.REQUIRED:
            if value is None:
                error = ValidationError(
                    path=rule.path,
                    rule_id=rule.rule_id,
                    message=rule.message,
                    severity=rule.severity,
                )

        elif rule.validation_type == ValidationType.TYPE:
            expected_type = rule.params.get("type")
            if not isinstance(value, eval(expected_type)):
                error = ValidationError(
                    path=rule.path,
                    rule_id=rule.rule_id,
                    message=rule.message,
                    severity=rule.severity,
                    actual_value=type(value).__name__,
                    expected=expected_type,
                )

        elif rule.validation_type == ValidationType.PATTERN:
            pattern = rule.params.get("pattern")
            if pattern and isinstance(value, str):
                if not re.match(pattern, value):
                    error = ValidationError(
                        path=rule.path,
                        rule_id=rule.rule_id,
                        message=rule.message,
                        severity=rule.severity,
                        actual_value=value,
                        expected=pattern,
                    )

        elif rule.validation_type == ValidationType.RANGE:
            min_val = rule.params.get("min")
            max_val = rule.params.get("max")
            if isinstance(value, (int, float)):
                if min_val is not None and value < min_val:
                    error = ValidationError(
                        path=rule.path,
                        rule_id=rule.rule_id,
                        message=rule.message,
                        severity=rule.severity,
                        actual_value=value,
                        expected=f">= {min_val}",
                    )
                elif max_val is not None and value > max_val:
                    error = ValidationError(
                        path=rule.path,
                        rule_id=rule.rule_id,
                        message=rule.message,
                        severity=rule.severity,
                        actual_value=value,
                        expected=f"<= {max_val}",
                    )

        elif rule.validation_type == ValidationType.ENUM:
            allowed = rule.params.get("values", [])
            if value not in allowed:
                error = ValidationError(
                    path=rule.path,
                    rule_id=rule.rule_id,
                    message=rule.message,
                    severity=rule.severity,
                    actual_value=value,
                    expected=allowed,
                )

        elif rule.validation_type == ValidationType.LENGTH:
            min_len = rule.params.get("min")
            max_len = rule.params.get("max")
            length = len(value) if hasattr(value, "__len__") else 0
            if min_len is not None and length < min_len:
                error = ValidationError(
                    path=rule.path,
                    rule_id=rule.rule_id,
                    message=rule.message,
                    severity=rule.severity,
                    actual_value=length,
                    expected=f">= {min_len}",
                )
            elif max_len is not None and length > max_len:
                error = ValidationError(
                    path=rule.path,
                    rule_id=rule.rule_id,
                    message=rule.message,
                    severity=rule.severity,
                    actual_value=length,
                    expected=f"<= {max_len}",
                )

        elif rule.validation_type == ValidationType.CUSTOM:
            validator_name = rule.params.get("validator")
            if validator_name and validator_name in self._custom_validators:
                validator = self._custom_validators[validator_name]
                try:
                    result_val = validator(value)
                    if isinstance(result_val, str):
                        error = ValidationError(
                            path=rule.path,
                            rule_id=rule.rule_id,
                            message=result_val,
                            severity=rule.severity,
                            actual_value=value,
                        )
                    elif result_val is False:
                        error = ValidationError(
                            path=rule.path,
                            rule_id=rule.rule_id,
                            message=rule.message,
                            severity=rule.severity,
                            actual_value=value,
                        )
                except Exception as e:
                    error = ValidationError(
                        path=rule.path,
                        rule_id=rule.rule_id,
                        message=f"Custom validator error: {e}",
                        severity=rule.severity,
                    )

        if error:
            result.add_error(error)

    def _get_value_at_path(self, data: Any, path: str) -> Any:
        """Get value at JSONPath-like path."""
        if path == "$":
            return data

        # Simple path parsing ($.field.subfield)
        parts = path.replace("$.", "").split(".")
        value = data

        for part in parts:
            if not part:
                continue

            # Handle array index
            if "[" in part:
                field = part[:part.index("[")]
                index = int(part[part.index("[") + 1:part.index("]")])
                if field and isinstance(value, dict):
                    value = value.get(field)
                if isinstance(value, list) and 0 <= index < len(value):
                    value = value[index]
                else:
                    return None
            elif isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value


# ==============================================================================
# Output Validator
# ==============================================================================

class OutputValidator:
    """
    Output validation for deployment responses.

    Validates:
    - API responses
    - Deployment results
    - Health check responses
    - Metrics outputs
    """

    def __init__(self):
        """Initialize output validator."""
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register an output schema."""
        self._schemas[name] = schema

    def validate(
        self,
        output: Any,
        schema_name: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate output against a schema.

        Args:
            output: Output to validate
            schema_name: Name of registered schema
            schema: Direct schema definition

        Returns:
            ValidationResult
        """
        start_time = time.time()

        schema_to_use = schema
        if schema_name and schema_name in self._schemas:
            schema_to_use = self._schemas[schema_name]

        if not schema_to_use:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    path="$",
                    rule_id="schema",
                    message="No schema provided for validation",
                )],
            )

        validator = SchemaValidator(schema_to_use)
        result = validator.validate(output)
        result.duration_ms = (time.time() - start_time) * 1000

        return result


# ==============================================================================
# Validation Engine (Step 242)
# ==============================================================================

class ValidationEngine:
    """
    Validation Engine - Complete input/output validation for deployments.

    PBTSO Phase: VERIFY

    Responsibilities:
    - Validate deployment configurations
    - Validate API inputs
    - Validate outputs and responses
    - Support custom validators
    - Track validation metrics

    Example:
        >>> engine = ValidationEngine()
        >>> result = await engine.validate_input(
        ...     data={"service_name": "api", "version": "v1.0.0"},
        ...     rules=["service_name", "version"]
        ... )
        >>> if not result.valid:
        ...     for error in result.errors:
        ...         print(f"{error.path}: {error.message}")
    """

    BUS_TOPICS = {
        "validate": "deploy.validation.validate",
        "passed": "deploy.validation.passed",
        "failed": "deploy.validation.failed",
        "schema": "deploy.validation.schema",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "validation-engine",
    ):
        """
        Initialize the validation engine.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "validation"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Components
        self._input_validator = InputValidator()
        self._output_validator = OutputValidator()

        # Metrics
        self._validation_count = 0
        self._pass_count = 0
        self._fail_count = 0

        self._load_schemas()

    async def validate_input(
        self,
        data: Dict[str, Any],
        rules: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate input data.

        Args:
            data: Input data to validate
            rules: Specific rules to apply
            context: Validation context (e.g., "deployment", "config")

        Returns:
            ValidationResult
        """
        _emit_bus_event(
            self.BUS_TOPICS["validate"],
            {
                "type": "input",
                "context": context,
                "rules_count": len(rules) if rules else "all",
            },
            actor=self.actor_id,
        )

        result = self._input_validator.validate(data, rules)
        self._record_validation(result, "input", context)

        return result

    async def validate_output(
        self,
        output: Any,
        schema_name: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output data.

        Args:
            output: Output data to validate
            schema_name: Name of registered schema
            schema: Direct schema definition
            context: Validation context

        Returns:
            ValidationResult
        """
        _emit_bus_event(
            self.BUS_TOPICS["validate"],
            {
                "type": "output",
                "schema_name": schema_name,
                "context": context,
            },
            actor=self.actor_id,
        )

        result = self._output_validator.validate(output, schema_name, schema)
        self._record_validation(result, "output", context)

        return result

    async def validate_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate data against a JSON schema.

        Args:
            data: Data to validate
            schema: JSON schema
            context: Validation context

        Returns:
            ValidationResult
        """
        validator = SchemaValidator(schema)
        result = validator.validate(data)
        self._record_validation(result, "schema", context)

        return result

    def add_input_rule(self, rule: ValidationRule) -> None:
        """Add an input validation rule."""
        self._input_validator.add_rule(rule)

    def add_custom_validator(
        self,
        name: str,
        validator: Callable[[Any], Union[bool, str]],
    ) -> None:
        """Add a custom validation function."""
        self._input_validator.add_custom_validator(name, validator)

    def register_output_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register an output schema."""
        self._output_validator.register_schema(name, schema)

        _emit_bus_event(
            self.BUS_TOPICS["schema"],
            {
                "action": "registered",
                "name": name,
            },
            actor=self.actor_id,
        )

        # Save schema
        self._save_schema(name, schema)

    def _record_validation(
        self,
        result: ValidationResult,
        validation_type: str,
        context: Optional[str],
    ) -> None:
        """Record validation metrics."""
        self._validation_count += 1

        if result.valid:
            self._pass_count += 1
            _emit_bus_event(
                self.BUS_TOPICS["passed"],
                {
                    "type": validation_type,
                    "context": context,
                    "rules_applied": result.rules_applied,
                    "duration_ms": result.duration_ms,
                },
                actor=self.actor_id,
            )
        else:
            self._fail_count += 1
            _emit_bus_event(
                self.BUS_TOPICS["failed"],
                {
                    "type": validation_type,
                    "context": context,
                    "error_count": len(result.errors),
                    "warning_count": len(result.warnings),
                    "duration_ms": result.duration_ms,
                },
                level="warn",
                actor=self.actor_id,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        return {
            "total_validations": self._validation_count,
            "passed": self._pass_count,
            "failed": self._fail_count,
            "pass_rate": self._pass_count / self._validation_count if self._validation_count > 0 else 0,
        }

    def _save_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Save schema to disk."""
        schema_file = self.state_dir / f"schema_{name}.json"
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

    def _load_schemas(self) -> None:
        """Load schemas from disk."""
        for schema_file in self.state_dir.glob("schema_*.json"):
            try:
                with open(schema_file, "r") as f:
                    schema = json.load(f)
                name = schema_file.stem.replace("schema_", "")
                self._output_validator.register_schema(name, schema)
            except (json.JSONDecodeError, IOError):
                continue


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for validation engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Validation Engine (Step 242)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument("data", help="JSON data or @file")
    validate_parser.add_argument("--type", "-t", default="input", choices=["input", "output", "schema"])
    validate_parser.add_argument("--rules", "-r", help="Comma-separated rules for input validation")
    validate_parser.add_argument("--schema", "-s", help="Schema name or @file for schema validation")
    validate_parser.add_argument("--json", action="store_true", help="JSON output")

    # add-rule command
    rule_parser = subparsers.add_parser("add-rule", help="Add validation rule")
    rule_parser.add_argument("name", help="Rule name")
    rule_parser.add_argument("--type", "-t", required=True,
                            choices=["pattern", "range", "required", "enum", "length"])
    rule_parser.add_argument("--path", "-p", default="$", help="JSONPath")
    rule_parser.add_argument("--params", help="JSON parameters")
    rule_parser.add_argument("--message", "-m", default="Validation failed", help="Error message")

    # register-schema command
    schema_parser = subparsers.add_parser("register-schema", help="Register output schema")
    schema_parser.add_argument("name", help="Schema name")
    schema_parser.add_argument("schema", help="JSON schema or @file")

    # metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Get validation metrics")
    metrics_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    engine = ValidationEngine()

    if args.command == "validate":
        # Parse data
        data_str = args.data
        if data_str.startswith("@"):
            with open(data_str[1:], "r") as f:
                data = json.load(f)
        else:
            data = json.loads(data_str)

        if args.type == "input":
            rules = args.rules.split(",") if args.rules else None
            result = asyncio.get_event_loop().run_until_complete(
                engine.validate_input(data, rules)
            )
        elif args.type == "schema" and args.schema:
            # Parse schema
            if args.schema.startswith("@"):
                with open(args.schema[1:], "r") as f:
                    schema = json.load(f)
            else:
                schema = json.loads(args.schema)
            result = asyncio.get_event_loop().run_until_complete(
                engine.validate_schema(data, schema)
            )
        else:
            result = asyncio.get_event_loop().run_until_complete(
                engine.validate_output(data, args.schema)
            )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.valid:
                print("Validation PASSED")
            else:
                print("Validation FAILED")
                for error in result.errors:
                    print(f"  ERROR [{error.path}]: {error.message}")
                for warning in result.warnings:
                    print(f"  WARNING [{warning.path}]: {warning.message}")

        return 0 if result.valid else 1

    elif args.command == "add-rule":
        params = json.loads(args.params) if args.params else {}
        rule = ValidationRule(
            rule_id=f"rule-{uuid.uuid4().hex[:8]}",
            name=args.name,
            validation_type=ValidationType(args.type),
            path=args.path,
            params=params,
            message=args.message,
        )
        engine.add_input_rule(rule)
        print(f"Added rule: {rule.rule_id}")
        return 0

    elif args.command == "register-schema":
        if args.schema.startswith("@"):
            with open(args.schema[1:], "r") as f:
                schema = json.load(f)
        else:
            schema = json.loads(args.schema)

        engine.register_output_schema(args.name, schema)
        print(f"Registered schema: {args.name}")
        return 0

    elif args.command == "metrics":
        metrics = engine.get_metrics()
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print(f"Total validations: {metrics['total_validations']}")
            print(f"Passed: {metrics['passed']}")
            print(f"Failed: {metrics['failed']}")
            print(f"Pass rate: {metrics['pass_rate']:.1%}")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
