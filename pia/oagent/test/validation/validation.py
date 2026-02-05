#!/usr/bin/env python3
"""
Step 142: Test Validation Module

Input/output validation for the Test Agent.

PBTSO Phase: VERIFY
Bus Topics:
- test.validation.validate (emits)
- test.validation.error (emits)
- test.validation.schema (emits)

Dependencies: Steps 101-141 (Test Components)
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
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Type, Union


# ============================================================================
# Constants
# ============================================================================

class ValidationType(Enum):
    """Types of validation."""
    INPUT = "input"
    OUTPUT = "output"
    CONFIG = "config"
    REQUEST = "request"
    RESPONSE = "response"
    DATA = "data"


class RuleType(Enum):
    """Types of validation rules."""
    REQUIRED = "required"
    TYPE = "type"
    PATTERN = "pattern"
    RANGE = "range"
    LENGTH = "length"
    ENUM = "enum"
    CUSTOM = "custom"
    SCHEMA = "schema"
    NESTED = "nested"


class Severity(Enum):
    """Validation error severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ValidationError:
    """
    A validation error.

    Attributes:
        path: Path to the invalid field
        message: Error message
        rule: Rule that failed
        severity: Error severity
        value: The invalid value
        expected: Expected value/format
    """
    path: str
    message: str
    rule: RuleType = RuleType.CUSTOM
    severity: Severity = Severity.ERROR
    value: Any = None
    expected: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "message": self.message,
            "rule": self.rule.value,
            "severity": self.severity.value,
            "value": str(self.value) if self.value is not None else None,
            "expected": str(self.expected) if self.expected is not None else None,
        }


@dataclass
class ValidationRule:
    """
    A validation rule.

    Attributes:
        name: Rule name
        rule_type: Type of rule
        path: Field path (dot notation)
        check: Validation check function or value
        message: Custom error message
        severity: Error severity
        enabled: Whether rule is enabled
        stop_on_fail: Stop validation on failure
    """
    name: str
    rule_type: RuleType
    path: str
    check: Any = None  # Value, pattern, or callable
    message: Optional[str] = None
    severity: Severity = Severity.ERROR
    enabled: bool = True
    stop_on_fail: bool = False

    def validate(self, value: Any, data: Dict[str, Any]) -> Optional[ValidationError]:
        """
        Validate a value against this rule.

        Args:
            value: Value to validate
            data: Full data object

        Returns:
            ValidationError if invalid, None if valid
        """
        if not self.enabled:
            return None

        error_value = None
        expected = None

        if self.rule_type == RuleType.REQUIRED:
            if value is None or (isinstance(value, str) and not value.strip()):
                return ValidationError(
                    path=self.path,
                    message=self.message or f"{self.path} is required",
                    rule=self.rule_type,
                    severity=self.severity,
                )

        elif self.rule_type == RuleType.TYPE:
            expected_type = self.check
            if not isinstance(value, expected_type):
                return ValidationError(
                    path=self.path,
                    message=self.message or f"{self.path} must be of type {expected_type.__name__}",
                    rule=self.rule_type,
                    severity=self.severity,
                    value=type(value).__name__,
                    expected=expected_type.__name__,
                )

        elif self.rule_type == RuleType.PATTERN:
            pattern = self.check if isinstance(self.check, Pattern) else re.compile(self.check)
            if isinstance(value, str) and not pattern.match(value):
                return ValidationError(
                    path=self.path,
                    message=self.message or f"{self.path} does not match pattern",
                    rule=self.rule_type,
                    severity=self.severity,
                    value=value,
                    expected=pattern.pattern,
                )

        elif self.rule_type == RuleType.RANGE:
            min_val, max_val = self.check
            if value is not None:
                if min_val is not None and value < min_val:
                    return ValidationError(
                        path=self.path,
                        message=self.message or f"{self.path} must be >= {min_val}",
                        rule=self.rule_type,
                        severity=self.severity,
                        value=value,
                        expected=f">= {min_val}",
                    )
                if max_val is not None and value > max_val:
                    return ValidationError(
                        path=self.path,
                        message=self.message or f"{self.path} must be <= {max_val}",
                        rule=self.rule_type,
                        severity=self.severity,
                        value=value,
                        expected=f"<= {max_val}",
                    )

        elif self.rule_type == RuleType.LENGTH:
            min_len, max_len = self.check
            if value is not None:
                length = len(value) if hasattr(value, '__len__') else 0
                if min_len is not None and length < min_len:
                    return ValidationError(
                        path=self.path,
                        message=self.message or f"{self.path} length must be >= {min_len}",
                        rule=self.rule_type,
                        severity=self.severity,
                        value=length,
                        expected=f">= {min_len}",
                    )
                if max_len is not None and length > max_len:
                    return ValidationError(
                        path=self.path,
                        message=self.message or f"{self.path} length must be <= {max_len}",
                        rule=self.rule_type,
                        severity=self.severity,
                        value=length,
                        expected=f"<= {max_len}",
                    )

        elif self.rule_type == RuleType.ENUM:
            allowed = self.check
            if value is not None and value not in allowed:
                return ValidationError(
                    path=self.path,
                    message=self.message or f"{self.path} must be one of: {allowed}",
                    rule=self.rule_type,
                    severity=self.severity,
                    value=value,
                    expected=allowed,
                )

        elif self.rule_type == RuleType.CUSTOM:
            if callable(self.check):
                try:
                    is_valid = self.check(value, data)
                    if not is_valid:
                        return ValidationError(
                            path=self.path,
                            message=self.message or f"{self.path} failed custom validation",
                            rule=self.rule_type,
                            severity=self.severity,
                            value=value,
                        )
                except Exception as e:
                    return ValidationError(
                        path=self.path,
                        message=str(e),
                        rule=self.rule_type,
                        severity=self.severity,
                        value=value,
                    )

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "rule_type": self.rule_type.value,
            "path": self.path,
            "severity": self.severity.value,
            "enabled": self.enabled,
        }


@dataclass
class ValidationResult:
    """
    Result of validation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        validated_data: Sanitized/transformed data
        rules_applied: Rules that were applied
        duration_ms: Validation duration
    """
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    validated_data: Optional[Dict[str, Any]] = None
    rules_applied: List[str] = field(default_factory=list)
    duration_ms: float = 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_error(self, error: ValidationError) -> None:
        """Add an error."""
        if error.severity == Severity.ERROR:
            self.errors.append(error)
            self.is_valid = False
        elif error.severity == Severity.WARNING:
            self.warnings.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "rules_applied": self.rules_applied,
            "duration_ms": self.duration_ms,
        }


@dataclass
class SchemaField:
    """
    A schema field definition.

    Attributes:
        name: Field name
        field_type: Expected type
        required: Whether field is required
        default: Default value
        description: Field description
        validators: Additional validation rules
        nested_schema: Schema for nested objects
    """
    name: str
    field_type: Type = str
    required: bool = False
    default: Any = None
    description: str = ""
    validators: List[ValidationRule] = field(default_factory=list)
    nested_schema: Optional[List] = None  # List[SchemaField]
    enum_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.field_type.__name__,
            "required": self.required,
            "default": self.default,
            "description": self.description,
        }


@dataclass
class ValidationConfig:
    """
    Configuration for validation.

    Attributes:
        output_dir: Output directory
        strict_mode: Fail on unknown fields
        coerce_types: Attempt type coercion
        strip_unknown: Remove unknown fields
        collect_all_errors: Collect all errors vs stop on first
    """
    output_dir: str = ".pluribus/test-agent/validation"
    strict_mode: bool = False
    coerce_types: bool = True
    strip_unknown: bool = False
    collect_all_errors: bool = True
    max_errors: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strict_mode": self.strict_mode,
            "coerce_types": self.coerce_types,
            "strip_unknown": self.strip_unknown,
            "collect_all_errors": self.collect_all_errors,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class ValidationBus:
    """Bus interface for validation with file locking."""

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


# ============================================================================
# Schema Validator
# ============================================================================

class SchemaValidator:
    """
    Schema-based validator.

    Validates data against a defined schema with type checking,
    required fields, and nested validation.
    """

    def __init__(self, schema: List[SchemaField]):
        """
        Initialize schema validator.

        Args:
            schema: List of schema field definitions
        """
        self.schema = schema
        self._field_map = {f.name: f for f in schema}

    def validate(self, data: Dict[str, Any], path_prefix: str = "") -> ValidationResult:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            path_prefix: Path prefix for nested validation

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        # Check required fields
        for field_def in self.schema:
            field_path = f"{path_prefix}{field_def.name}" if path_prefix else field_def.name
            value = data.get(field_def.name)

            # Required check
            if field_def.required and (value is None or value == ""):
                result.add_error(ValidationError(
                    path=field_path,
                    message=f"{field_path} is required",
                    rule=RuleType.REQUIRED,
                ))
                continue

            # Skip further validation if value is None and not required
            if value is None:
                continue

            # Type check
            if not isinstance(value, field_def.field_type):
                result.add_error(ValidationError(
                    path=field_path,
                    message=f"{field_path} must be of type {field_def.field_type.__name__}",
                    rule=RuleType.TYPE,
                    value=type(value).__name__,
                    expected=field_def.field_type.__name__,
                ))
                continue

            # Enum check
            if field_def.enum_values and value not in field_def.enum_values:
                result.add_error(ValidationError(
                    path=field_path,
                    message=f"{field_path} must be one of {field_def.enum_values}",
                    rule=RuleType.ENUM,
                    value=value,
                    expected=field_def.enum_values,
                ))

            # Range check
            if field_def.min_value is not None and value < field_def.min_value:
                result.add_error(ValidationError(
                    path=field_path,
                    message=f"{field_path} must be >= {field_def.min_value}",
                    rule=RuleType.RANGE,
                    value=value,
                ))
            if field_def.max_value is not None and value > field_def.max_value:
                result.add_error(ValidationError(
                    path=field_path,
                    message=f"{field_path} must be <= {field_def.max_value}",
                    rule=RuleType.RANGE,
                    value=value,
                ))

            # Length check
            if hasattr(value, '__len__'):
                if field_def.min_length is not None and len(value) < field_def.min_length:
                    result.add_error(ValidationError(
                        path=field_path,
                        message=f"{field_path} length must be >= {field_def.min_length}",
                        rule=RuleType.LENGTH,
                        value=len(value),
                    ))
                if field_def.max_length is not None and len(value) > field_def.max_length:
                    result.add_error(ValidationError(
                        path=field_path,
                        message=f"{field_path} length must be <= {field_def.max_length}",
                        rule=RuleType.LENGTH,
                        value=len(value),
                    ))

            # Pattern check
            if field_def.pattern and isinstance(value, str):
                if not re.match(field_def.pattern, value):
                    result.add_error(ValidationError(
                        path=field_path,
                        message=f"{field_path} does not match required pattern",
                        rule=RuleType.PATTERN,
                        value=value,
                        expected=field_def.pattern,
                    ))

            # Nested schema check
            if field_def.nested_schema and isinstance(value, dict):
                nested_validator = SchemaValidator(field_def.nested_schema)
                nested_result = nested_validator.validate(value, f"{field_path}.")
                result.errors.extend(nested_result.errors)
                result.warnings.extend(nested_result.warnings)
                if not nested_result.is_valid:
                    result.is_valid = False

            # Custom validators
            for rule in field_def.validators:
                error = rule.validate(value, data)
                if error:
                    result.add_error(error)

        return result


# ============================================================================
# Test Validator
# ============================================================================

class TestValidator:
    """
    Input/output validation for the Test Agent.

    Features:
    - Rule-based validation
    - Schema validation
    - Type coercion
    - Custom validators
    - Validation caching

    PBTSO Phase: VERIFY
    Bus Topics: test.validation.validate, test.validation.error, test.validation.schema
    """

    BUS_TOPICS = {
        "validate": "test.validation.validate",
        "error": "test.validation.error",
        "schema": "test.validation.schema",
    }

    def __init__(self, bus=None, config: Optional[ValidationConfig] = None):
        """
        Initialize the validator.

        Args:
            bus: Optional bus instance
            config: Validation configuration
        """
        self.bus = bus or ValidationBus()
        self.config = config or ValidationConfig()
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._schemas: Dict[str, SchemaValidator] = {}
        self._validators: Dict[str, Callable] = {}

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Register built-in validators
        self._register_builtin_validators()

    def _register_builtin_validators(self) -> None:
        """Register built-in validators."""
        self.register_validator("email", lambda v, _: bool(
            re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(v))
        ))

        self.register_validator("url", lambda v, _: bool(
            re.match(r'^https?://[^\s]+$', str(v))
        ))

        self.register_validator("uuid", lambda v, _: bool(
            re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', str(v).lower())
        ))

        self.register_validator("semver", lambda v, _: bool(
            re.match(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$', str(v))
        ))

        self.register_validator("path_exists", lambda v, _: Path(str(v)).exists())

        self.register_validator("positive", lambda v, _: isinstance(v, (int, float)) and v > 0)

        self.register_validator("non_empty", lambda v, _: v is not None and (
            len(v) > 0 if hasattr(v, '__len__') else True
        ))

    def register_validator(
        self,
        name: str,
        validator: Callable[[Any, Dict], bool],
    ) -> None:
        """
        Register a custom validator.

        Args:
            name: Validator name
            validator: Validation function (value, data) -> bool
        """
        self._validators[name] = validator

    def add_rule(
        self,
        validation_type: ValidationType,
        rule: ValidationRule,
    ) -> None:
        """
        Add a validation rule.

        Args:
            validation_type: Type of validation
            rule: Validation rule to add
        """
        key = validation_type.value
        if key not in self._rules:
            self._rules[key] = []
        self._rules[key].append(rule)

    def add_rules(
        self,
        validation_type: ValidationType,
        rules: List[ValidationRule],
    ) -> None:
        """Add multiple validation rules."""
        for rule in rules:
            self.add_rule(validation_type, rule)

    def register_schema(
        self,
        name: str,
        schema: List[SchemaField],
    ) -> None:
        """
        Register a validation schema.

        Args:
            name: Schema name
            schema: Schema field definitions
        """
        self._schemas[name] = SchemaValidator(schema)

        self._emit_event("schema", {
            "name": name,
            "fields": [f.to_dict() for f in schema],
        })

    def validate(
        self,
        data: Dict[str, Any],
        validation_type: ValidationType = ValidationType.DATA,
        schema_name: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate data.

        Args:
            data: Data to validate
            validation_type: Type of validation
            schema_name: Schema to use for validation

        Returns:
            ValidationResult with validation outcome
        """
        start_time = time.time()
        result = ValidationResult()
        result.validated_data = dict(data) if data else {}

        # Apply schema validation if specified
        if schema_name and schema_name in self._schemas:
            schema_result = self._schemas[schema_name].validate(data)
            result.errors.extend(schema_result.errors)
            result.warnings.extend(schema_result.warnings)
            if not schema_result.is_valid:
                result.is_valid = False
            result.rules_applied.append(f"schema:{schema_name}")

        # Apply type-specific rules
        rules = self._rules.get(validation_type.value, [])
        for rule in rules:
            if not rule.enabled:
                continue

            # Get value at path
            value = self._get_value_at_path(data, rule.path)

            # Apply rule
            error = rule.validate(value, data)
            if error:
                result.add_error(error)
                if rule.stop_on_fail:
                    break

            result.rules_applied.append(rule.name)

            # Check max errors
            if len(result.errors) >= self.config.max_errors:
                break

        result.duration_ms = (time.time() - start_time) * 1000

        # Emit event
        self._emit_event("validate", {
            "validation_type": validation_type.value,
            "is_valid": result.is_valid,
            "error_count": result.error_count,
            "duration_ms": result.duration_ms,
        })

        # Emit errors
        if result.errors:
            self._emit_event("error", {
                "validation_type": validation_type.value,
                "errors": [e.to_dict() for e in result.errors[:10]],
            })

        return result

    def validate_input(self, data: Dict[str, Any], schema_name: Optional[str] = None) -> ValidationResult:
        """Validate input data."""
        return self.validate(data, ValidationType.INPUT, schema_name)

    def validate_output(self, data: Dict[str, Any], schema_name: Optional[str] = None) -> ValidationResult:
        """Validate output data."""
        return self.validate(data, ValidationType.OUTPUT, schema_name)

    def validate_request(self, data: Dict[str, Any], schema_name: Optional[str] = None) -> ValidationResult:
        """Validate API request data."""
        return self.validate(data, ValidationType.REQUEST, schema_name)

    def validate_response(self, data: Dict[str, Any], schema_name: Optional[str] = None) -> ValidationResult:
        """Validate API response data."""
        return self.validate(data, ValidationType.RESPONSE, schema_name)

    def _get_value_at_path(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value from data using dot notation path."""
        if not path:
            return data

        parts = path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list):
                try:
                    index = int(part)
                    value = value[index] if 0 <= index < len(value) else None
                except (ValueError, IndexError):
                    value = None
            else:
                value = None

            if value is None:
                break

        return value

    def coerce_type(self, value: Any, target_type: Type) -> Any:
        """
        Attempt to coerce a value to a target type.

        Args:
            value: Value to coerce
            target_type: Target type

        Returns:
            Coerced value or original if coercion fails
        """
        if not self.config.coerce_types:
            return value

        if value is None:
            return value

        if isinstance(value, target_type):
            return value

        try:
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif target_type == int:
                return int(float(value))
            elif target_type == float:
                return float(value)
            elif target_type == str:
                return str(value)
            elif target_type == list:
                if isinstance(value, str):
                    return value.split(",")
                return list(value)
        except (ValueError, TypeError):
            pass

        return value

    def sanitize(self, data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """
        Sanitize data according to schema.

        Args:
            data: Data to sanitize
            schema_name: Schema to use

        Returns:
            Sanitized data
        """
        if schema_name not in self._schemas:
            return data

        schema = self._schemas[schema_name]
        sanitized = {}

        for field_def in schema.schema:
            value = data.get(field_def.name)

            if value is None:
                if field_def.default is not None:
                    sanitized[field_def.name] = field_def.default
            else:
                # Coerce type if needed
                sanitized[field_def.name] = self.coerce_type(value, field_def.field_type)

        # Include unknown fields if not stripping
        if not self.config.strip_unknown:
            for key, value in data.items():
                if key not in sanitized:
                    sanitized[key] = value

        return sanitized

    def get_rules(self, validation_type: ValidationType) -> List[ValidationRule]:
        """Get rules for a validation type."""
        return self._rules.get(validation_type.value, [])

    def clear_rules(self, validation_type: Optional[ValidationType] = None) -> None:
        """Clear validation rules."""
        if validation_type:
            self._rules[validation_type.value] = []
        else:
            self._rules.clear()

    async def validate_async(
        self,
        data: Dict[str, Any],
        validation_type: ValidationType = ValidationType.DATA,
        schema_name: Optional[str] = None,
    ) -> ValidationResult:
        """Async version of validate."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.validate, data, validation_type, schema_name
        )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.validation.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "validation",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Validator."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Validator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument("file", help="JSON file to validate")
    validate_parser.add_argument("--schema", help="Schema name to use")
    validate_parser.add_argument("--type", default="data",
                                 choices=["input", "output", "request", "response", "data"])

    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Show schema")
    schema_parser.add_argument("name", help="Schema name")

    # Rules command
    rules_parser = subparsers.add_parser("rules", help="List validation rules")
    rules_parser.add_argument("--type", help="Filter by validation type")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/validation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = ValidationConfig(output_dir=args.output)
    validator = TestValidator(config=config)

    if args.command == "validate":
        # Load data
        try:
            with open(args.file) as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading file: {e}")
            exit(1)

        validation_type = ValidationType(args.type)
        result = validator.validate(data, validation_type, args.schema)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "[VALID]" if result.is_valid else "[INVALID]"
            print(f"\nValidation Result: {status}")
            print(f"  Duration: {result.duration_ms:.2f}ms")
            print(f"  Rules Applied: {len(result.rules_applied)}")

            if result.errors:
                print(f"\n  Errors ({result.error_count}):")
                for error in result.errors:
                    print(f"    - {error.path}: {error.message}")

            if result.warnings:
                print(f"\n  Warnings ({result.warning_count}):")
                for warning in result.warnings:
                    print(f"    - {warning.path}: {warning.message}")

        exit(0 if result.is_valid else 1)

    elif args.command == "schema":
        if args.name in validator._schemas:
            schema = validator._schemas[args.name]
            if args.json:
                print(json.dumps([f.to_dict() for f in schema.schema], indent=2))
            else:
                print(f"\nSchema: {args.name}")
                for field_def in schema.schema:
                    required = "*" if field_def.required else ""
                    print(f"  {field_def.name}{required}: {field_def.field_type.__name__}")
                    if field_def.description:
                        print(f"    {field_def.description}")
        else:
            print(f"Schema not found: {args.name}")

    elif args.command == "rules":
        all_rules = []
        for vtype, rules in validator._rules.items():
            if not args.type or vtype == args.type:
                for rule in rules:
                    all_rules.append({
                        "type": vtype,
                        **rule.to_dict(),
                    })

        if args.json:
            print(json.dumps(all_rules, indent=2))
        else:
            print(f"\nValidation Rules ({len(all_rules)}):")
            for rule in all_rules:
                enabled = "[ON]" if rule["enabled"] else "[OFF]"
                print(f"  {enabled} {rule['name']} ({rule['type']})")
                print(f"      Path: {rule['path']}, Type: {rule['rule_type']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
