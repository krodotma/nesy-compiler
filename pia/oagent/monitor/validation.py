#!/usr/bin/env python3
"""
Monitor Validation Module - Step 292

Input/output validation for the Monitor Agent.

PBTSO Phase: VERIFY

Bus Topics:
- monitor.validation.input (emitted)
- monitor.validation.output (emitted)
- monitor.validation.error (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union


class ValidationType(Enum):
    """Validation types."""
    TYPE = "type"
    RANGE = "range"
    PATTERN = "pattern"
    LENGTH = "length"
    ENUM = "enum"
    REQUIRED = "required"
    CUSTOM = "custom"
    SCHEMA = "schema"


class ValidationSeverity(Enum):
    """Validation error severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """A validation error.

    Attributes:
        field: Field name
        validation_type: Type of validation that failed
        message: Error message
        severity: Error severity
        value: Invalid value (sanitized)
        constraint: Constraint that was violated
    """
    field: str
    validation_type: ValidationType
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    value: Any = None
    constraint: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "type": self.validation_type.value,
            "message": self.message,
            "severity": self.severity.value,
            "value": str(self.value)[:100] if self.value is not None else None,
            "constraint": str(self.constraint) if self.constraint else None,
        }


@dataclass
class ValidationResult:
    """Result of validation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        sanitized_data: Sanitized/transformed data
        duration_ms: Validation duration
    """
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "duration_ms": self.duration_ms,
        }


@dataclass
class FieldSchema:
    """Schema for a field.

    Attributes:
        name: Field name
        field_type: Expected type
        required: Whether field is required
        default: Default value
        min_value: Minimum value (for numbers)
        max_value: Maximum value (for numbers)
        min_length: Minimum length (for strings/lists)
        max_length: Maximum length (for strings/lists)
        pattern: Regex pattern (for strings)
        choices: Valid choices
        nested_schema: Schema for nested objects
        validator: Custom validator function
    """
    name: str
    field_type: Type
    required: bool = False
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    choices: Optional[List[Any]] = None
    nested_schema: Optional[Dict[str, "FieldSchema"]] = None
    validator: Optional[Callable[[Any], bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.field_type.__name__,
            "required": self.required,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pattern": self.pattern,
            "choices": self.choices,
        }


class MonitorValidation:
    """
    Validation module for the Monitor Agent.

    Provides:
    - Input validation with schema
    - Output validation
    - Data sanitization
    - Custom validators
    - Validation statistics

    Example:
        validation = MonitorValidation()

        # Define schema
        schema = {
            "metric_name": FieldSchema(
                name="metric_name",
                field_type=str,
                required=True,
                pattern=r"^[a-z][a-z0-9_.]+$",
            ),
            "value": FieldSchema(
                name="value",
                field_type=float,
                required=True,
                min_value=0,
            ),
        }

        # Validate input
        result = validation.validate_input(data, schema)

        if result.valid:
            process(result.sanitized_data)
    """

    BUS_TOPICS = {
        "input": "monitor.validation.input",
        "output": "monitor.validation.output",
        "error": "monitor.validation.error",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    # Common patterns
    PATTERNS = {
        "metric_name": r"^[a-z][a-z0-9_.]{0,255}$",
        "label_name": r"^[a-z_][a-z0-9_]{0,127}$",
        "label_value": r"^.{0,1024}$",
        "agent_id": r"^[a-z][a-z0-9-]{0,63}$",
        "topic": r"^[a-z][a-z0-9.*_-]{0,255}$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "url": r"^https?://[^\s]+$",
        "iso_datetime": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$",
    }

    # Default schemas for common operations
    DEFAULT_SCHEMAS = {
        "metric_point": {
            "name": FieldSchema(
                name="name",
                field_type=str,
                required=True,
                max_length=256,
                pattern=r"^[a-z][a-z0-9_.]+$",
            ),
            "value": FieldSchema(
                name="value",
                field_type=(int, float),
                required=True,
            ),
            "timestamp": FieldSchema(
                name="timestamp",
                field_type=float,
                required=False,
            ),
            "labels": FieldSchema(
                name="labels",
                field_type=dict,
                required=False,
                max_length=32,
            ),
        },
        "alert": {
            "name": FieldSchema(
                name="name",
                field_type=str,
                required=True,
                max_length=256,
            ),
            "severity": FieldSchema(
                name="severity",
                field_type=str,
                required=True,
                choices=["critical", "high", "medium", "low", "info"],
            ),
            "message": FieldSchema(
                name="message",
                field_type=str,
                required=True,
                max_length=4096,
            ),
        },
        "api_request": {
            "method": FieldSchema(
                name="method",
                field_type=str,
                required=True,
                choices=["GET", "POST", "PUT", "DELETE", "PATCH"],
            ),
            "path": FieldSchema(
                name="path",
                field_type=str,
                required=True,
                max_length=2048,
                pattern=r"^/[a-zA-Z0-9/_.-]*$",
            ),
        },
    }

    def __init__(
        self,
        enable_sanitization: bool = True,
        strict_mode: bool = False,
        bus_dir: Optional[str] = None,
    ):
        """Initialize validation module.

        Args:
            enable_sanitization: Enable data sanitization
            strict_mode: Treat warnings as errors
            bus_dir: Bus directory
        """
        self._enable_sanitization = enable_sanitization
        self._strict_mode = strict_mode
        self._last_heartbeat = time.time()

        # Custom validators
        self._custom_validators: Dict[str, Callable[[Any], ValidationResult]] = {}

        # Statistics
        self._stats = {
            "total_validations": 0,
            "successful": 0,
            "failed": 0,
            "by_schema": {},
        }

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def validate_input(
        self,
        data: Dict[str, Any],
        schema: Optional[Dict[str, FieldSchema]] = None,
        schema_name: Optional[str] = None,
    ) -> ValidationResult:
        """Validate input data.

        Args:
            data: Data to validate
            schema: Validation schema
            schema_name: Name of default schema to use

        Returns:
            Validation result
        """
        start_time = time.time()

        # Get schema
        if schema is None:
            if schema_name and schema_name in self.DEFAULT_SCHEMAS:
                schema = self.DEFAULT_SCHEMAS[schema_name]
            else:
                schema = {}

        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        sanitized: Dict[str, Any] = {}

        # Validate each field in schema
        for field_name, field_schema in schema.items():
            field_errors, field_warnings, field_value = self._validate_field(
                data.get(field_name),
                field_schema,
            )
            errors.extend(field_errors)
            warnings.extend(field_warnings)

            if field_value is not None or field_name in data:
                sanitized[field_name] = field_value

        # Add any extra fields not in schema (with warning)
        for field_name in data:
            if field_name not in schema:
                warnings.append(ValidationError(
                    field=field_name,
                    validation_type=ValidationType.SCHEMA,
                    message=f"Unknown field: {field_name}",
                    severity=ValidationSeverity.WARNING,
                ))
                if self._enable_sanitization:
                    sanitized[field_name] = self._sanitize_value(data[field_name])

        # Determine validity
        valid = len(errors) == 0
        if self._strict_mode:
            valid = valid and len(warnings) == 0

        result = ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized if self._enable_sanitization else data,
            duration_ms=(time.time() - start_time) * 1000,
        )

        # Update statistics
        self._stats["total_validations"] += 1
        if valid:
            self._stats["successful"] += 1
        else:
            self._stats["failed"] += 1

        if schema_name:
            if schema_name not in self._stats["by_schema"]:
                self._stats["by_schema"][schema_name] = {"success": 0, "failure": 0}
            if valid:
                self._stats["by_schema"][schema_name]["success"] += 1
            else:
                self._stats["by_schema"][schema_name]["failure"] += 1

        # Emit event
        self._emit_bus_event(
            self.BUS_TOPICS["input"],
            {
                "valid": valid,
                "schema": schema_name,
                "error_count": len(errors),
                "warning_count": len(warnings),
            },
        )

        if not valid:
            self._emit_bus_event(
                self.BUS_TOPICS["error"],
                {
                    "errors": [e.to_dict() for e in errors[:5]],  # Limit to 5
                },
                level="warning",
            )

        return result

    def validate_output(
        self,
        data: Dict[str, Any],
        schema: Optional[Dict[str, FieldSchema]] = None,
    ) -> ValidationResult:
        """Validate output data.

        Args:
            data: Data to validate
            schema: Validation schema

        Returns:
            Validation result
        """
        result = self.validate_input(data, schema)

        self._emit_bus_event(
            self.BUS_TOPICS["output"],
            {
                "valid": result.valid,
                "error_count": len(result.errors),
            },
        )

        return result

    def validate_metric_point(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate a metric point.

        Args:
            data: Metric data

        Returns:
            Validation result
        """
        return self.validate_input(data, schema_name="metric_point")

    def validate_alert(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate an alert.

        Args:
            data: Alert data

        Returns:
            Validation result
        """
        return self.validate_input(data, schema_name="alert")

    def validate_api_request(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate an API request.

        Args:
            data: Request data

        Returns:
            Validation result
        """
        return self.validate_input(data, schema_name="api_request")

    def register_validator(
        self,
        name: str,
        validator: Callable[[Any], ValidationResult],
    ) -> None:
        """Register a custom validator.

        Args:
            name: Validator name
            validator: Validator function
        """
        self._custom_validators[name] = validator

    def validate_with_custom(
        self,
        name: str,
        data: Any,
    ) -> ValidationResult:
        """Validate using custom validator.

        Args:
            name: Validator name
            data: Data to validate

        Returns:
            Validation result
        """
        validator = self._custom_validators.get(name)
        if not validator:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="",
                    validation_type=ValidationType.CUSTOM,
                    message=f"Unknown validator: {name}",
                )],
            )

        return validator(data)

    def sanitize(
        self,
        data: Dict[str, Any],
        schema: Optional[Dict[str, FieldSchema]] = None,
    ) -> Dict[str, Any]:
        """Sanitize data without full validation.

        Args:
            data: Data to sanitize
            schema: Optional schema

        Returns:
            Sanitized data
        """
        sanitized = {}
        for key, value in data.items():
            field_schema = schema.get(key) if schema else None
            if field_schema:
                sanitized[key] = self._coerce_type(value, field_schema.field_type)
            else:
                sanitized[key] = self._sanitize_value(value)
        return sanitized

    def validate_pattern(
        self,
        value: str,
        pattern_name: str,
    ) -> bool:
        """Validate a value against a named pattern.

        Args:
            value: Value to validate
            pattern_name: Pattern name

        Returns:
            True if valid
        """
        pattern = self.PATTERNS.get(pattern_name)
        if not pattern:
            return True

        return bool(re.match(pattern, value))

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics.

        Returns:
            Statistics
        """
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self._stats = {
            "total_validations": 0,
            "successful": 0,
            "failed": 0,
            "by_schema": {},
        }

    def _validate_field(
        self,
        value: Any,
        schema: FieldSchema,
    ) -> tuple[List[ValidationError], List[ValidationError], Any]:
        """Validate a single field.

        Returns:
            Tuple of (errors, warnings, sanitized_value)
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        sanitized_value = value

        # Check required
        if value is None:
            if schema.required:
                errors.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.REQUIRED,
                    message=f"Field '{schema.name}' is required",
                ))
            elif schema.default is not None:
                sanitized_value = schema.default
            return errors, warnings, sanitized_value

        # Check type
        expected_types = schema.field_type if isinstance(schema.field_type, tuple) else (schema.field_type,)
        if not isinstance(value, expected_types):
            # Try to coerce
            coerced = self._coerce_type(value, expected_types[0])
            if coerced is not None:
                sanitized_value = coerced
                warnings.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.TYPE,
                    message=f"Field '{schema.name}' was coerced from {type(value).__name__} to {expected_types[0].__name__}",
                    severity=ValidationSeverity.WARNING,
                ))
            else:
                errors.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.TYPE,
                    message=f"Field '{schema.name}' expected {expected_types[0].__name__}, got {type(value).__name__}",
                    value=value,
                ))
                return errors, warnings, sanitized_value

        # Check range (for numbers)
        if isinstance(sanitized_value, (int, float)):
            if schema.min_value is not None and sanitized_value < schema.min_value:
                errors.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.RANGE,
                    message=f"Field '{schema.name}' must be >= {schema.min_value}",
                    value=sanitized_value,
                    constraint=schema.min_value,
                ))
            if schema.max_value is not None and sanitized_value > schema.max_value:
                errors.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.RANGE,
                    message=f"Field '{schema.name}' must be <= {schema.max_value}",
                    value=sanitized_value,
                    constraint=schema.max_value,
                ))

        # Check length (for strings and lists)
        if isinstance(sanitized_value, (str, list, dict)):
            length = len(sanitized_value)
            if schema.min_length is not None and length < schema.min_length:
                errors.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.LENGTH,
                    message=f"Field '{schema.name}' length must be >= {schema.min_length}",
                    value=sanitized_value,
                    constraint=schema.min_length,
                ))
            if schema.max_length is not None and length > schema.max_length:
                errors.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.LENGTH,
                    message=f"Field '{schema.name}' length must be <= {schema.max_length}",
                    value=sanitized_value,
                    constraint=schema.max_length,
                ))

        # Check pattern (for strings)
        if isinstance(sanitized_value, str) and schema.pattern:
            if not re.match(schema.pattern, sanitized_value):
                errors.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.PATTERN,
                    message=f"Field '{schema.name}' does not match pattern",
                    value=sanitized_value,
                    constraint=schema.pattern,
                ))

        # Check choices
        if schema.choices is not None and sanitized_value not in schema.choices:
            errors.append(ValidationError(
                field=schema.name,
                validation_type=ValidationType.ENUM,
                message=f"Field '{schema.name}' must be one of: {schema.choices}",
                value=sanitized_value,
                constraint=schema.choices,
            ))

        # Custom validator
        if schema.validator:
            try:
                if not schema.validator(sanitized_value):
                    errors.append(ValidationError(
                        field=schema.name,
                        validation_type=ValidationType.CUSTOM,
                        message=f"Field '{schema.name}' failed custom validation",
                        value=sanitized_value,
                    ))
            except Exception as e:
                errors.append(ValidationError(
                    field=schema.name,
                    validation_type=ValidationType.CUSTOM,
                    message=f"Custom validator error: {str(e)}",
                ))

        # Nested schema validation
        if isinstance(sanitized_value, dict) and schema.nested_schema:
            for nested_name, nested_schema in schema.nested_schema.items():
                nested_errors, nested_warnings, nested_value = self._validate_field(
                    sanitized_value.get(nested_name),
                    nested_schema,
                )
                for error in nested_errors:
                    error.field = f"{schema.name}.{error.field}"
                    errors.append(error)
                for warning in nested_warnings:
                    warning.field = f"{schema.name}.{warning.field}"
                    warnings.append(warning)
                sanitized_value[nested_name] = nested_value

        # Sanitize strings
        if isinstance(sanitized_value, str) and self._enable_sanitization:
            sanitized_value = self._sanitize_string(sanitized_value)

        return errors, warnings, sanitized_value

    def _coerce_type(self, value: Any, target_type: Type) -> Any:
        """Try to coerce value to target type."""
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
                    return json.loads(value)
                return list(value)
            elif target_type == dict:
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value)
        except Exception:
            pass
        return None

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a value."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._sanitize_value(v) for v in value]
        return value

    def _sanitize_string(self, value: str) -> str:
        """Sanitize a string value."""
        # Remove null bytes
        value = value.replace("\x00", "")
        # Limit length
        if len(value) > 65536:
            value = value[:65536]
        return value

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
                "component": "monitor_validation",
                "status": "healthy",
                "total_validations": self._stats["total_validations"],
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
            "actor": "monitor-validation",
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


# Singleton instance
_validation: Optional[MonitorValidation] = None


def get_validation() -> MonitorValidation:
    """Get or create the validation module singleton.

    Returns:
        MonitorValidation instance
    """
    global _validation
    if _validation is None:
        _validation = MonitorValidation()
    return _validation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Validation Module (Step 292)")
    parser.add_argument("--validate", metavar="JSON", help="Validate JSON data")
    parser.add_argument("--schema", help="Schema to use (metric_point, alert, api_request)")
    parser.add_argument("--pattern", help="Validate against pattern")
    parser.add_argument("--value", help="Value to validate")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    validation = get_validation()

    if args.validate:
        data = json.loads(args.validate)
        result = validation.validate_input(data, schema_name=args.schema)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "valid" if result.valid else "invalid"
            print(f"Validation: {status}")
            if result.errors:
                print("Errors:")
                for e in result.errors:
                    print(f"  {e.field}: {e.message}")
            if result.warnings:
                print("Warnings:")
                for w in result.warnings:
                    print(f"  {w.field}: {w.message}")

    if args.pattern and args.value:
        valid = validation.validate_pattern(args.value, args.pattern)
        if args.json:
            print(json.dumps({"pattern": args.pattern, "value": args.value, "valid": valid}))
        else:
            print(f"Pattern '{args.pattern}': {'valid' if valid else 'invalid'}")

    if args.stats:
        stats = validation.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Validation Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
