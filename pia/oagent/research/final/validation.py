#!/usr/bin/env python3
"""
validation.py - Input/Output Validation (Step 42)

Comprehensive validation framework for Research Agent inputs and outputs.
Supports schema validation, sanitization, and constraint checking.

PBTSO Phase: PROTECT

Bus Topics:
- a2a.research.validation.input
- a2a.research.validation.output
- a2a.research.validation.error
- research.validation.schema.register

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Pattern, Set, Tuple, Type, TypeVar, Union
)

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class ValidationType(Enum):
    """Types of validation."""
    TYPE = "type"
    RANGE = "range"
    LENGTH = "length"
    PATTERN = "pattern"
    ENUM = "enum"
    REQUIRED = "required"
    UNIQUE = "unique"
    CUSTOM = "custom"


class SanitizationType(Enum):
    """Types of sanitization."""
    TRIM = "trim"
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    STRIP_HTML = "strip_html"
    ESCAPE_HTML = "escape_html"
    NORMALIZE_WHITESPACE = "normalize_whitespace"
    REMOVE_CONTROL_CHARS = "remove_control_chars"


@dataclass
class ValidationConfig:
    """Configuration for validation."""

    strict_mode: bool = True  # Fail on unknown fields
    coerce_types: bool = True  # Attempt type coercion
    strip_unknown: bool = False  # Remove unknown fields
    max_depth: int = 10  # Max nesting depth
    max_string_length: int = 100000  # Max string length
    max_array_length: int = 10000  # Max array length
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ValidationError:
    """A validation error."""

    field: str
    message: str
    type: ValidationType
    value: Any = None
    constraint: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "type": self.type.value,
            "value": str(self.value)[:100] if self.value else None,
            "constraint": str(self.constraint) if self.constraint else None,
        }


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
        }


# ============================================================================
# Field Validators
# ============================================================================


T = TypeVar("T")


class FieldValidator(ABC, Generic[T]):
    """Abstract base for field validators."""

    @abstractmethod
    def validate(self, value: Any, field_name: str) -> Tuple[bool, Optional[ValidationError]]:
        """Validate a value."""
        pass

    def sanitize(self, value: T) -> T:
        """Sanitize a value (default: no-op)."""
        return value


class TypeValidator(FieldValidator[T]):
    """Validates value type."""

    def __init__(self, expected_type: Type[T], allow_none: bool = False):
        self.expected_type = expected_type
        self.allow_none = allow_none

    def validate(self, value: Any, field_name: str) -> Tuple[bool, Optional[ValidationError]]:
        if value is None:
            if self.allow_none:
                return True, None
            return False, ValidationError(
                field=field_name,
                message="Value cannot be None",
                type=ValidationType.REQUIRED,
                value=value,
            )

        if not isinstance(value, self.expected_type):
            return False, ValidationError(
                field=field_name,
                message=f"Expected {self.expected_type.__name__}, got {type(value).__name__}",
                type=ValidationType.TYPE,
                value=value,
                constraint=self.expected_type.__name__,
            )

        return True, None


class RangeValidator(FieldValidator[Union[int, float]]):
    """Validates numeric range."""

    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        exclusive_min: bool = False,
        exclusive_max: bool = False,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.exclusive_min = exclusive_min
        self.exclusive_max = exclusive_max

    def validate(self, value: Any, field_name: str) -> Tuple[bool, Optional[ValidationError]]:
        if not isinstance(value, (int, float)):
            return False, ValidationError(
                field=field_name,
                message="Value must be a number",
                type=ValidationType.TYPE,
                value=value,
            )

        if self.min_value is not None:
            if self.exclusive_min:
                if value <= self.min_value:
                    return False, ValidationError(
                        field=field_name,
                        message=f"Value must be greater than {self.min_value}",
                        type=ValidationType.RANGE,
                        value=value,
                        constraint=f">{self.min_value}",
                    )
            else:
                if value < self.min_value:
                    return False, ValidationError(
                        field=field_name,
                        message=f"Value must be at least {self.min_value}",
                        type=ValidationType.RANGE,
                        value=value,
                        constraint=f">={self.min_value}",
                    )

        if self.max_value is not None:
            if self.exclusive_max:
                if value >= self.max_value:
                    return False, ValidationError(
                        field=field_name,
                        message=f"Value must be less than {self.max_value}",
                        type=ValidationType.RANGE,
                        value=value,
                        constraint=f"<{self.max_value}",
                    )
            else:
                if value > self.max_value:
                    return False, ValidationError(
                        field=field_name,
                        message=f"Value must be at most {self.max_value}",
                        type=ValidationType.RANGE,
                        value=value,
                        constraint=f"<={self.max_value}",
                    )

        return True, None


class LengthValidator(FieldValidator[Union[str, list, dict]]):
    """Validates length of strings, lists, or dicts."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any, field_name: str) -> Tuple[bool, Optional[ValidationError]]:
        if not hasattr(value, "__len__"):
            return False, ValidationError(
                field=field_name,
                message="Value must have a length",
                type=ValidationType.TYPE,
                value=value,
            )

        length = len(value)

        if self.min_length is not None and length < self.min_length:
            return False, ValidationError(
                field=field_name,
                message=f"Length must be at least {self.min_length}, got {length}",
                type=ValidationType.LENGTH,
                value=value,
                constraint=f"min={self.min_length}",
            )

        if self.max_length is not None and length > self.max_length:
            return False, ValidationError(
                field=field_name,
                message=f"Length must be at most {self.max_length}, got {length}",
                type=ValidationType.LENGTH,
                value=value,
                constraint=f"max={self.max_length}",
            )

        return True, None


class PatternValidator(FieldValidator[str]):
    """Validates string against regex pattern."""

    def __init__(self, pattern: Union[str, Pattern], description: str = ""):
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern
        self.description = description or f"Pattern: {self.pattern.pattern}"

    def validate(self, value: Any, field_name: str) -> Tuple[bool, Optional[ValidationError]]:
        if not isinstance(value, str):
            return False, ValidationError(
                field=field_name,
                message="Value must be a string",
                type=ValidationType.TYPE,
                value=value,
            )

        if not self.pattern.match(value):
            return False, ValidationError(
                field=field_name,
                message=f"Value does not match {self.description}",
                type=ValidationType.PATTERN,
                value=value,
                constraint=self.pattern.pattern,
            )

        return True, None


class EnumValidator(FieldValidator[T]):
    """Validates value is in allowed set."""

    def __init__(self, allowed_values: Set[T]):
        self.allowed_values = allowed_values

    def validate(self, value: Any, field_name: str) -> Tuple[bool, Optional[ValidationError]]:
        if value not in self.allowed_values:
            return False, ValidationError(
                field=field_name,
                message=f"Value must be one of: {sorted(str(v) for v in self.allowed_values)}",
                type=ValidationType.ENUM,
                value=value,
                constraint=list(self.allowed_values),
            )

        return True, None


class CustomValidator(FieldValidator[T]):
    """Custom validator with user-defined function."""

    def __init__(
        self,
        validator_fn: Callable[[Any], bool],
        message: str = "Custom validation failed",
    ):
        self.validator_fn = validator_fn
        self.message = message

    def validate(self, value: Any, field_name: str) -> Tuple[bool, Optional[ValidationError]]:
        try:
            if self.validator_fn(value):
                return True, None
        except Exception:
            pass

        return False, ValidationError(
            field=field_name,
            message=self.message,
            type=ValidationType.CUSTOM,
            value=value,
        )


# ============================================================================
# Common Validators
# ============================================================================


# Email validator
EMAIL_VALIDATOR = PatternValidator(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "email address"
)

# URL validator
URL_VALIDATOR = PatternValidator(
    r"^https?://[^\s/$.?#].[^\s]*$",
    "URL"
)

# UUID validator
UUID_VALIDATOR = PatternValidator(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "UUID"
)

# Path validator (no traversal)
SAFE_PATH_VALIDATOR = CustomValidator(
    lambda p: ".." not in p and not p.startswith("/"),
    "Path must be relative and not contain .."
)

# Query validator (basic SQL injection prevention)
SAFE_QUERY_VALIDATOR = CustomValidator(
    lambda q: not any(kw in q.lower() for kw in ["drop", "delete", "truncate", "update", "insert", "--", ";"]),
    "Query contains potentially dangerous keywords"
)


# ============================================================================
# Schema Validation
# ============================================================================


@dataclass
class FieldSchema:
    """Schema for a single field."""

    name: str
    validators: List[FieldValidator] = field(default_factory=list)
    required: bool = True
    default: Any = None
    sanitizers: List[SanitizationType] = field(default_factory=list)
    nested_schema: Optional["Schema"] = None


class Schema:
    """Schema for validating objects."""

    def __init__(
        self,
        fields: List[FieldSchema],
        strict: bool = True,
        allow_extra: bool = False,
    ):
        self.fields = {f.name: f for f in fields}
        self.strict = strict
        self.allow_extra = allow_extra

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema."""
        errors: List[ValidationError] = []
        warnings: List[str] = []
        sanitized: Dict[str, Any] = {}

        # Check required fields
        for name, field_schema in self.fields.items():
            if field_schema.required and name not in data:
                if field_schema.default is not None:
                    sanitized[name] = field_schema.default
                else:
                    errors.append(ValidationError(
                        field=name,
                        message="Required field is missing",
                        type=ValidationType.REQUIRED,
                    ))
                continue

            if name not in data:
                if field_schema.default is not None:
                    sanitized[name] = field_schema.default
                continue

            value = data[name]

            # Apply sanitizers
            if isinstance(value, str):
                value = self._sanitize_string(value, field_schema.sanitizers)

            # Validate nested schema
            if field_schema.nested_schema and isinstance(value, dict):
                nested_result = field_schema.nested_schema.validate(value)
                if not nested_result.valid:
                    for error in nested_result.errors:
                        errors.append(ValidationError(
                            field=f"{name}.{error.field}",
                            message=error.message,
                            type=error.type,
                            value=error.value,
                            constraint=error.constraint,
                        ))
                value = nested_result.sanitized_data

            # Run validators
            for validator in field_schema.validators:
                valid, error = validator.validate(value, name)
                if not valid and error:
                    errors.append(error)

            sanitized[name] = value

        # Check extra fields
        extra_fields = set(data.keys()) - set(self.fields.keys())
        if extra_fields:
            if self.strict and not self.allow_extra:
                for extra in extra_fields:
                    errors.append(ValidationError(
                        field=extra,
                        message="Unknown field",
                        type=ValidationType.CUSTOM,
                    ))
            elif self.allow_extra:
                for extra in extra_fields:
                    sanitized[extra] = data[extra]
            else:
                warnings.append(f"Unknown fields ignored: {extra_fields}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized,
        )

    def _sanitize_string(self, value: str, sanitizers: List[SanitizationType]) -> str:
        """Apply sanitizers to string."""
        for sanitizer in sanitizers:
            if sanitizer == SanitizationType.TRIM:
                value = value.strip()
            elif sanitizer == SanitizationType.LOWERCASE:
                value = value.lower()
            elif sanitizer == SanitizationType.UPPERCASE:
                value = value.upper()
            elif sanitizer == SanitizationType.STRIP_HTML:
                value = re.sub(r"<[^>]+>", "", value)
            elif sanitizer == SanitizationType.ESCAPE_HTML:
                value = (value
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&#39;"))
            elif sanitizer == SanitizationType.NORMALIZE_WHITESPACE:
                value = re.sub(r"\s+", " ", value).strip()
            elif sanitizer == SanitizationType.REMOVE_CONTROL_CHARS:
                value = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)

        return value


# ============================================================================
# Validation Manager
# ============================================================================


class ValidationManager:
    """
    Comprehensive validation manager for Research Agent.

    Features:
    - Schema-based validation
    - Input sanitization
    - Output validation
    - Custom validators
    - Bus event emission

    PBTSO Phase: PROTECT

    Example:
        validator = ValidationManager()

        # Register schema
        validator.register_schema("search_query", Schema([
            FieldSchema(
                name="query",
                validators=[
                    TypeValidator(str),
                    LengthValidator(min_length=1, max_length=1000),
                ],
                sanitizers=[SanitizationType.TRIM],
            ),
            FieldSchema(
                name="limit",
                validators=[TypeValidator(int), RangeValidator(min_value=1, max_value=100)],
                default=10,
            ),
        ]))

        # Validate input
        result = validator.validate_input("search_query", user_input)
        if result.valid:
            # Process sanitized data
            process(result.sanitized_data)
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the validation manager.

        Args:
            config: Validation configuration
            bus: AgentBus for event emission
        """
        self.config = config or ValidationConfig()
        self.bus = bus or AgentBus()

        # Schema registry
        self._schemas: Dict[str, Schema] = {}
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "validations": 0,
            "valid": 0,
            "invalid": 0,
            "sanitizations": 0,
        }

        # Register default schemas
        self._register_default_schemas()

    def register_schema(self, name: str, schema: Schema) -> None:
        """Register a validation schema."""
        with self._lock:
            self._schemas[name] = schema

        self._emit_event("research.validation.schema.register", {
            "name": name,
            "fields": list(schema.fields.keys()),
        })

    def get_schema(self, name: str) -> Optional[Schema]:
        """Get a registered schema."""
        with self._lock:
            return self._schemas.get(name)

    def validate_input(
        self,
        schema_name: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate input data against a schema.

        Args:
            schema_name: Name of the schema
            data: Data to validate
            context: Optional context for logging

        Returns:
            ValidationResult
        """
        self._stats["validations"] += 1

        schema = self.get_schema(schema_name)
        if not schema:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="$schema",
                    message=f"Unknown schema: {schema_name}",
                    type=ValidationType.CUSTOM,
                )],
            )

        # Apply global sanitization
        data = self._global_sanitize(data)

        result = schema.validate(data)

        if result.valid:
            self._stats["valid"] += 1
            self._stats["sanitizations"] += 1
        else:
            self._stats["invalid"] += 1

        # Emit event
        self._emit_event("a2a.research.validation.input", {
            "schema": schema_name,
            "valid": result.valid,
            "error_count": len(result.errors),
            "context": context,
        }, level="info" if result.valid else "warning")

        return result

    def validate_output(
        self,
        schema_name: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate output data against a schema.

        Args:
            schema_name: Name of the schema
            data: Data to validate
            context: Optional context for logging

        Returns:
            ValidationResult
        """
        self._stats["validations"] += 1

        schema = self.get_schema(schema_name)
        if not schema:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="$schema",
                    message=f"Unknown schema: {schema_name}",
                    type=ValidationType.CUSTOM,
                )],
            )

        result = schema.validate(data)

        if result.valid:
            self._stats["valid"] += 1
        else:
            self._stats["invalid"] += 1

        # Emit event
        self._emit_event("a2a.research.validation.output", {
            "schema": schema_name,
            "valid": result.valid,
            "error_count": len(result.errors),
            "context": context,
        }, level="info" if result.valid else "warning")

        return result

    def validate_value(
        self,
        value: Any,
        validators: List[FieldValidator],
        field_name: str = "value",
    ) -> ValidationResult:
        """
        Validate a single value.

        Args:
            value: Value to validate
            validators: List of validators
            field_name: Field name for error messages

        Returns:
            ValidationResult
        """
        errors: List[ValidationError] = []

        for validator in validators:
            valid, error = validator.validate(value, field_name)
            if not valid and error:
                errors.append(error)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_data=value,
        )

    def sanitize_string(
        self,
        value: str,
        sanitizers: Optional[List[SanitizationType]] = None,
    ) -> str:
        """
        Sanitize a string value.

        Args:
            value: String to sanitize
            sanitizers: List of sanitizers to apply

        Returns:
            Sanitized string
        """
        if sanitizers is None:
            sanitizers = [
                SanitizationType.TRIM,
                SanitizationType.NORMALIZE_WHITESPACE,
                SanitizationType.REMOVE_CONTROL_CHARS,
            ]

        for sanitizer in sanitizers:
            if sanitizer == SanitizationType.TRIM:
                value = value.strip()
            elif sanitizer == SanitizationType.LOWERCASE:
                value = value.lower()
            elif sanitizer == SanitizationType.UPPERCASE:
                value = value.upper()
            elif sanitizer == SanitizationType.STRIP_HTML:
                value = re.sub(r"<[^>]+>", "", value)
            elif sanitizer == SanitizationType.ESCAPE_HTML:
                value = (value
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&#39;"))
            elif sanitizer == SanitizationType.NORMALIZE_WHITESPACE:
                value = re.sub(r"\s+", " ", value).strip()
            elif sanitizer == SanitizationType.REMOVE_CONTROL_CHARS:
                value = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)

        self._stats["sanitizations"] += 1
        return value

    def validate_decorator(
        self,
        input_schema: Optional[str] = None,
        output_schema: Optional[str] = None,
    ) -> Callable:
        """
        Decorator for validating function inputs/outputs.

        Args:
            input_schema: Schema name for input validation
            output_schema: Schema name for output validation

        Example:
            @validator.validate_decorator(
                input_schema="search_query",
                output_schema="search_result",
            )
            def search(query: Dict[str, Any]) -> Dict[str, Any]:
                ...
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Validate input
                if input_schema and kwargs:
                    result = self.validate_input(input_schema, kwargs)
                    if not result.valid:
                        raise ValueError(f"Input validation failed: {result.errors}")
                    kwargs = result.sanitized_data

                # Call function
                output = func(*args, **kwargs)

                # Validate output
                if output_schema and isinstance(output, dict):
                    result = self.validate_output(output_schema, output)
                    if not result.valid:
                        raise ValueError(f"Output validation failed: {result.errors}")

                return output

            return wrapper
        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self._stats["validations"]
        return {
            **self._stats,
            "schemas_registered": len(self._schemas),
            "valid_rate": self._stats["valid"] / total if total > 0 else 0.0,
        }

    def _global_sanitize(self, data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Apply global sanitization rules."""
        if depth > self.config.max_depth:
            return data

        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Limit string length
                if len(value) > self.config.max_string_length:
                    value = value[:self.config.max_string_length]
                # Remove null bytes
                value = value.replace("\x00", "")
            elif isinstance(value, list):
                # Limit array length
                if len(value) > self.config.max_array_length:
                    value = value[:self.config.max_array_length]
            elif isinstance(value, dict):
                value = self._global_sanitize(value, depth + 1)

            sanitized[key] = value

        return sanitized

    def _register_default_schemas(self) -> None:
        """Register default validation schemas."""
        # Search query schema
        self.register_schema("search_query", Schema([
            FieldSchema(
                name="query",
                validators=[
                    TypeValidator(str),
                    LengthValidator(min_length=1, max_length=1000),
                ],
                sanitizers=[
                    SanitizationType.TRIM,
                    SanitizationType.NORMALIZE_WHITESPACE,
                ],
            ),
            FieldSchema(
                name="limit",
                validators=[
                    TypeValidator(int),
                    RangeValidator(min_value=1, max_value=1000),
                ],
                required=False,
                default=10,
            ),
            FieldSchema(
                name="offset",
                validators=[
                    TypeValidator(int),
                    RangeValidator(min_value=0),
                ],
                required=False,
                default=0,
            ),
        ], strict=False, allow_extra=True))

        # Index request schema
        self.register_schema("index_request", Schema([
            FieldSchema(
                name="path",
                validators=[
                    TypeValidator(str),
                    LengthValidator(min_length=1, max_length=500),
                    SAFE_PATH_VALIDATOR,
                ],
                sanitizers=[SanitizationType.TRIM],
            ),
            FieldSchema(
                name="recursive",
                validators=[TypeValidator(bool)],
                required=False,
                default=True,
            ),
        ]))

    def _emit_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
    ) -> str:
        """Emit event with file locking."""
        if not self.config.emit_to_bus:
            return ""

        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "validation",
            "level": level,
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Validation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validation (Step 42)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument("--schema", required=True, help="Schema name")
    validate_parser.add_argument("--data", required=True, help="JSON data")

    # Schemas command
    schemas_parser = subparsers.add_parser("schemas", help="List schemas")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run validation demo")

    args = parser.parse_args()

    validator = ValidationManager()

    if args.command == "validate":
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            return 1

        result = validator.validate_input(args.schema, data)
        print(f"Valid: {result.valid}")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error.field}: {error.message}")

        if result.valid and result.sanitized_data:
            print(f"\nSanitized data: {json.dumps(result.sanitized_data, indent=2)}")

        return 0 if result.valid else 1

    elif args.command == "schemas":
        print("Registered schemas:")
        for name, schema in validator._schemas.items():
            print(f"  {name}:")
            for field_name, field in schema.fields.items():
                req = "required" if field.required else "optional"
                print(f"    - {field_name} ({req})")

    elif args.command == "stats":
        stats = validator.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Validation Statistics:")
            print(f"  Total Validations: {stats['validations']}")
            print(f"  Valid: {stats['valid']}")
            print(f"  Invalid: {stats['invalid']}")
            print(f"  Sanitizations: {stats['sanitizations']}")
            print(f"  Schemas: {stats['schemas_registered']}")

    elif args.command == "demo":
        print("Running validation demo...\n")

        # Valid search query
        print("1. Valid search query:")
        result = validator.validate_input("search_query", {
            "query": "  find functions  ",
            "limit": 20,
        })
        print(f"   Valid: {result.valid}")
        print(f"   Sanitized: {result.sanitized_data}")

        # Invalid search query
        print("\n2. Invalid search query (empty query):")
        result = validator.validate_input("search_query", {
            "query": "",
            "limit": 5000,  # Exceeds max
        })
        print(f"   Valid: {result.valid}")
        for error in result.errors:
            print(f"   Error: {error.field} - {error.message}")

        # Custom validation
        print("\n3. Custom email validation:")
        email_result = validator.validate_value(
            "user@example.com",
            [EMAIL_VALIDATOR],
            "email"
        )
        print(f"   Valid email: {email_result.valid}")

        invalid_email_result = validator.validate_value(
            "not-an-email",
            [EMAIL_VALIDATOR],
            "email"
        )
        print(f"   Invalid email: {invalid_email_result.valid}")
        if invalid_email_result.errors:
            print(f"   Error: {invalid_email_result.errors[0].message}")

        # Sanitization
        print("\n4. String sanitization:")
        dirty = "  <script>alert('xss')</script>  Hello\x00World  "
        clean = validator.sanitize_string(dirty, [
            SanitizationType.TRIM,
            SanitizationType.STRIP_HTML,
            SanitizationType.REMOVE_CONTROL_CHARS,
        ])
        print(f"   Input:  '{dirty}'")
        print(f"   Output: '{clean}'")

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
