#!/usr/bin/env python3
"""
Validation Engine (Step 192)

Comprehensive input/output validation system for the Review Agent.
Provides schema validation, sanitization, and type checking.

PBTSO Phase: VERIFY, SEQUESTER
Bus Topics: review.validation.input, review.validation.output, review.validation.error

Validation Features:
- JSON Schema validation
- Input sanitization
- Output validation
- Type coercion
- Custom validators

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
from typing import Any, Callable, Dict, List, Optional, Type, Union, Pattern

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class ValidationSeverity(Enum):
    """Validation error severity."""
    ERROR = "error"       # Blocks operation
    WARNING = "warning"   # Allows operation with notice
    INFO = "info"         # Informational


class DataType(Enum):
    """Supported data types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    ANY = "any"


class SanitizationMode(Enum):
    """Sanitization modes."""
    STRICT = "strict"       # Remove all potentially dangerous content
    MODERATE = "moderate"   # Remove known dangerous patterns
    PERMISSIVE = "permissive"  # Minimal sanitization


@dataclass
class ValidationError:
    """
    A validation error.

    Attributes:
        path: JSON path to the error
        message: Error message
        severity: Error severity
        expected: Expected value/type
        actual: Actual value/type
        code: Error code
    """
    path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    expected: Optional[str] = None
    actual: Optional[str] = None
    code: str = "VALIDATION_ERROR"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "message": self.message,
            "severity": self.severity.value,
            "expected": self.expected,
            "actual": self.actual,
            "code": self.code,
        }


@dataclass
class ValidationResult:
    """
    Result of validation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        warnings: List of warnings
        sanitized_data: Sanitized data (if applicable)
        original_data: Original input data
        duration_ms: Validation duration
    """
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    sanitized_data: Optional[Any] = None
    original_data: Optional[Any] = None
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "duration_ms": round(self.duration_ms, 2),
        }

    def add_error(
        self,
        path: str,
        message: str,
        code: str = "VALIDATION_ERROR",
        expected: Optional[str] = None,
        actual: Optional[str] = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(
            path=path,
            message=message,
            severity=ValidationSeverity.ERROR,
            expected=expected,
            actual=actual,
            code=code,
        ))
        self.valid = False

    def add_warning(
        self,
        path: str,
        message: str,
        code: str = "VALIDATION_WARNING",
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationError(
            path=path,
            message=message,
            severity=ValidationSeverity.WARNING,
            code=code,
        ))


@dataclass
class FieldSchema:
    """
    Schema for a single field.

    Attributes:
        name: Field name
        data_type: Expected data type
        required: Whether field is required
        default: Default value
        min_length: Minimum length (strings/arrays)
        max_length: Maximum length (strings/arrays)
        min_value: Minimum value (numbers)
        max_value: Maximum value (numbers)
        pattern: Regex pattern (strings)
        enum: Valid values
        items: Schema for array items
        properties: Schema for object properties
        validators: Custom validators
        description: Field description
    """
    name: str
    data_type: DataType = DataType.STRING
    required: bool = False
    default: Any = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    enum: Optional[List[Any]] = None
    items: Optional["FieldSchema"] = None
    properties: Optional[Dict[str, "FieldSchema"]] = None
    validators: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "type": self.data_type.value,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.min_length is not None:
            result["minLength"] = self.min_length
        if self.max_length is not None:
            result["maxLength"] = self.max_length
        if self.min_value is not None:
            result["minimum"] = self.min_value
        if self.max_value is not None:
            result["maximum"] = self.max_value
        if self.pattern:
            result["pattern"] = self.pattern
        if self.enum:
            result["enum"] = self.enum
        if self.description:
            result["description"] = self.description
        return result


# ============================================================================
# Built-in Validators
# ============================================================================

def validate_email(value: Any) -> bool:
    """Validate email format."""
    if not isinstance(value, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, value))


def validate_url(value: Any) -> bool:
    """Validate URL format."""
    if not isinstance(value, str):
        return False
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, value))


def validate_uuid(value: Any) -> bool:
    """Validate UUID format."""
    if not isinstance(value, str):
        return False
    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(pattern, value, re.IGNORECASE))


def validate_iso_date(value: Any) -> bool:
    """Validate ISO date format."""
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.rstrip("Z"))
        return True
    except ValueError:
        return False


def validate_path_safe(value: Any) -> bool:
    """Validate path is safe (no traversal)."""
    if not isinstance(value, str):
        return False
    dangerous = ["../", "..\\", "/etc/", "/proc/", "~"]
    return not any(d in value for d in dangerous)


def validate_no_html(value: Any) -> bool:
    """Validate no HTML tags."""
    if not isinstance(value, str):
        return True
    return not bool(re.search(r'<[^>]+>', value))


def validate_alphanumeric(value: Any) -> bool:
    """Validate alphanumeric only."""
    if not isinstance(value, str):
        return False
    return value.isalnum()


BUILT_IN_VALIDATORS: Dict[str, Callable[[Any], bool]] = {
    "email": validate_email,
    "url": validate_url,
    "uuid": validate_uuid,
    "iso_date": validate_iso_date,
    "path_safe": validate_path_safe,
    "no_html": validate_no_html,
    "alphanumeric": validate_alphanumeric,
}


# ============================================================================
# Sanitizers
# ============================================================================

class Sanitizer:
    """Input sanitization utilities."""

    # Patterns to remove in strict mode
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
        r'data:text/html',
    ]

    @classmethod
    def sanitize_string(
        cls,
        value: str,
        mode: SanitizationMode = SanitizationMode.MODERATE,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Sanitize a string value.

        Args:
            value: String to sanitize
            mode: Sanitization mode
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if mode == SanitizationMode.STRICT:
            # Remove all HTML
            value = re.sub(r'<[^>]+>', '', value)
            # Remove dangerous patterns
            for pattern in cls.DANGEROUS_PATTERNS:
                value = re.sub(pattern, '', value, flags=re.IGNORECASE | re.DOTALL)
            # Remove control characters
            value = ''.join(c for c in value if ord(c) >= 32 or c in '\n\r\t')

        elif mode == SanitizationMode.MODERATE:
            # Remove scripts and event handlers
            value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
            value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
            value = re.sub(r'on\w+\s*=', '', value, flags=re.IGNORECASE)

        # Trim and limit length
        value = value.strip()
        if max_length and len(value) > max_length:
            value = value[:max_length]

        return value

    @classmethod
    def sanitize_path(cls, value: str) -> str:
        """Sanitize a file path."""
        # Remove path traversal attempts
        value = value.replace("../", "").replace("..\\", "")
        # Normalize
        value = os.path.normpath(value)
        return value

    @classmethod
    def sanitize_sql_identifier(cls, value: str) -> str:
        """Sanitize a SQL identifier."""
        # Allow only alphanumeric and underscore
        return re.sub(r'[^\w]', '', value)


# ============================================================================
# Schema Validator
# ============================================================================

class SchemaValidator:
    """
    JSON Schema-style validator.

    Example:
        validator = SchemaValidator()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "required": True, "maxLength": 100},
                "age": {"type": "integer", "minimum": 0},
            }
        }

        result = validator.validate(data, schema)
    """

    def __init__(self):
        """Initialize schema validator."""
        self._type_validators = {
            "string": self._validate_string,
            "integer": self._validate_integer,
            "number": self._validate_number,
            "boolean": self._validate_boolean,
            "array": self._validate_array,
            "object": self._validate_object,
            "null": self._validate_null,
        }

    def validate(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str = "$",
    ) -> ValidationResult:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema: Schema definition
            path: Current JSON path

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True, original_data=data)
        self._validate_value(data, schema, path, result)
        return result

    def _validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate a single value against schema."""
        # Check type
        expected_type = schema.get("type", "any")
        if expected_type != "any":
            if expected_type in self._type_validators:
                self._type_validators[expected_type](value, schema, path, result)
            else:
                result.add_error(path, f"Unknown type: {expected_type}", code="UNKNOWN_TYPE")

        # Check enum
        if "enum" in schema and value not in schema["enum"]:
            result.add_error(
                path,
                f"Value must be one of: {schema['enum']}",
                code="ENUM_MISMATCH",
                expected=str(schema["enum"]),
                actual=str(value),
            )

    def _validate_string(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate string value."""
        if value is None and not schema.get("required", False):
            return

        if not isinstance(value, str):
            result.add_error(
                path,
                f"Expected string, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                expected="string",
                actual=type(value).__name__,
            )
            return

        # Check length
        if "minLength" in schema and len(value) < schema["minLength"]:
            result.add_error(
                path,
                f"String too short (min: {schema['minLength']})",
                code="MIN_LENGTH",
            )

        if "maxLength" in schema and len(value) > schema["maxLength"]:
            result.add_error(
                path,
                f"String too long (max: {schema['maxLength']})",
                code="MAX_LENGTH",
            )

        # Check pattern
        if "pattern" in schema:
            if not re.match(schema["pattern"], value):
                result.add_error(
                    path,
                    f"String does not match pattern: {schema['pattern']}",
                    code="PATTERN_MISMATCH",
                )

        # Check format
        if "format" in schema:
            format_validators = {
                "email": validate_email,
                "url": validate_url,
                "uuid": validate_uuid,
                "date-time": validate_iso_date,
            }
            validator = format_validators.get(schema["format"])
            if validator and not validator(value):
                result.add_error(
                    path,
                    f"Invalid format: {schema['format']}",
                    code="FORMAT_INVALID",
                )

    def _validate_integer(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate integer value."""
        if value is None and not schema.get("required", False):
            return

        if not isinstance(value, int) or isinstance(value, bool):
            result.add_error(
                path,
                f"Expected integer, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                expected="integer",
                actual=type(value).__name__,
            )
            return

        self._validate_number_constraints(value, schema, path, result)

    def _validate_number(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate number value."""
        if value is None and not schema.get("required", False):
            return

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            result.add_error(
                path,
                f"Expected number, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                expected="number",
                actual=type(value).__name__,
            )
            return

        self._validate_number_constraints(value, schema, path, result)

    def _validate_number_constraints(
        self,
        value: Union[int, float],
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate number constraints."""
        if "minimum" in schema and value < schema["minimum"]:
            result.add_error(
                path,
                f"Value {value} is below minimum {schema['minimum']}",
                code="MINIMUM",
            )

        if "maximum" in schema and value > schema["maximum"]:
            result.add_error(
                path,
                f"Value {value} is above maximum {schema['maximum']}",
                code="MAXIMUM",
            )

        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            result.add_error(
                path,
                f"Value {value} must be greater than {schema['exclusiveMinimum']}",
                code="EXCLUSIVE_MINIMUM",
            )

        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            result.add_error(
                path,
                f"Value {value} must be less than {schema['exclusiveMaximum']}",
                code="EXCLUSIVE_MAXIMUM",
            )

    def _validate_boolean(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate boolean value."""
        if value is None and not schema.get("required", False):
            return

        if not isinstance(value, bool):
            result.add_error(
                path,
                f"Expected boolean, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                expected="boolean",
                actual=type(value).__name__,
            )

    def _validate_array(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate array value."""
        if value is None and not schema.get("required", False):
            return

        if not isinstance(value, list):
            result.add_error(
                path,
                f"Expected array, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                expected="array",
                actual=type(value).__name__,
            )
            return

        # Check length
        if "minItems" in schema and len(value) < schema["minItems"]:
            result.add_error(
                path,
                f"Array too short (min: {schema['minItems']})",
                code="MIN_ITEMS",
            )

        if "maxItems" in schema and len(value) > schema["maxItems"]:
            result.add_error(
                path,
                f"Array too long (max: {schema['maxItems']})",
                code="MAX_ITEMS",
            )

        # Check unique items
        if schema.get("uniqueItems", False):
            seen = set()
            for i, item in enumerate(value):
                item_repr = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
                if item_repr in seen:
                    result.add_error(
                        f"{path}[{i}]",
                        "Duplicate item in array",
                        code="UNIQUE_ITEMS",
                    )
                seen.add(item_repr)

        # Validate items
        if "items" in schema:
            for i, item in enumerate(value):
                self._validate_value(item, schema["items"], f"{path}[{i}]", result)

    def _validate_object(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate object value."""
        if value is None and not schema.get("required", False):
            return

        if not isinstance(value, dict):
            result.add_error(
                path,
                f"Expected object, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                expected="object",
                actual=type(value).__name__,
            )
            return

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required properties
        for prop in required:
            if prop not in value:
                result.add_error(
                    f"{path}.{prop}",
                    f"Required property '{prop}' is missing",
                    code="REQUIRED",
                )

        # Validate properties
        for prop, prop_schema in properties.items():
            if prop in value:
                self._validate_value(value[prop], prop_schema, f"{path}.{prop}", result)

        # Check additional properties
        if not schema.get("additionalProperties", True):
            for prop in value:
                if prop not in properties:
                    result.add_warning(
                        f"{path}.{prop}",
                        f"Additional property '{prop}' is not allowed",
                        code="ADDITIONAL_PROPERTY",
                    )

    def _validate_null(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate null value."""
        if value is not None:
            result.add_error(
                path,
                f"Expected null, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                expected="null",
                actual=type(value).__name__,
            )


# ============================================================================
# Input Validator
# ============================================================================

class InputValidator:
    """
    Input validation with sanitization.

    Example:
        validator = InputValidator()

        result = validator.validate_request({
            "file_path": "/path/to/file.py",
            "options": {"strict": True},
        })
    """

    def __init__(
        self,
        sanitization_mode: SanitizationMode = SanitizationMode.MODERATE,
        bus_path: Optional[Path] = None,
    ):
        """Initialize input validator."""
        self.sanitization_mode = sanitization_mode
        self.bus_path = bus_path or self._get_bus_path()
        self._schema_validator = SchemaValidator()
        self._custom_validators: Dict[str, Callable[[Any], bool]] = dict(BUILT_IN_VALIDATORS)

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "validation") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "validation-engine",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def register_validator(
        self,
        name: str,
        validator: Callable[[Any], bool],
    ) -> None:
        """Register a custom validator."""
        self._custom_validators[name] = validator

    def validate(
        self,
        data: Any,
        schema: Optional[Dict[str, Any]] = None,
        sanitize: bool = True,
    ) -> ValidationResult:
        """
        Validate input data.

        Args:
            data: Input data to validate
            schema: Optional schema to validate against
            sanitize: Whether to sanitize the data

        Returns:
            ValidationResult
        """
        start_time = time.time()

        result = ValidationResult(valid=True, original_data=data)

        # Sanitize first
        if sanitize:
            data = self._sanitize(data)
            result.sanitized_data = data

        # Schema validation
        if schema:
            schema_result = self._schema_validator.validate(data, schema)
            result.errors.extend(schema_result.errors)
            result.warnings.extend(schema_result.warnings)
            result.valid = schema_result.valid

        result.duration_ms = (time.time() - start_time) * 1000

        # Emit validation event
        self._emit_event("review.validation.input", {
            "valid": result.valid,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "sanitized": sanitize,
            "duration_ms": result.duration_ms,
        })

        return result

    def _sanitize(self, data: Any) -> Any:
        """Recursively sanitize data."""
        if isinstance(data, str):
            return Sanitizer.sanitize_string(data, self.sanitization_mode)
        elif isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return data

    def validate_file_path(self, path: str) -> ValidationResult:
        """Validate a file path for safety."""
        result = ValidationResult(valid=True, original_data=path)

        # Check path traversal
        if not validate_path_safe(path):
            result.add_error("path", "Path contains traversal sequence", code="PATH_TRAVERSAL")

        # Sanitize
        result.sanitized_data = Sanitizer.sanitize_path(path)

        return result


# ============================================================================
# Output Validator
# ============================================================================

class OutputValidator:
    """
    Output validation to ensure safe responses.

    Example:
        validator = OutputValidator()

        result = validator.validate_response({
            "status": "success",
            "data": {...},
        })
    """

    def __init__(
        self,
        bus_path: Optional[Path] = None,
    ):
        """Initialize output validator."""
        self.bus_path = bus_path or self._get_bus_path()
        self._schema_validator = SchemaValidator()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "validation") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "validation-engine",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def validate(
        self,
        data: Any,
        schema: Optional[Dict[str, Any]] = None,
        redact_secrets: bool = True,
    ) -> ValidationResult:
        """
        Validate output data.

        Args:
            data: Output data to validate
            schema: Optional schema to validate against
            redact_secrets: Whether to redact secrets

        Returns:
            ValidationResult
        """
        start_time = time.time()

        result = ValidationResult(valid=True, original_data=data)

        # Check for accidental secret exposure
        if redact_secrets:
            data = self._redact_secrets(data)
            result.sanitized_data = data

        # Schema validation
        if schema:
            schema_result = self._schema_validator.validate(data, schema)
            result.errors.extend(schema_result.errors)
            result.warnings.extend(schema_result.warnings)
            result.valid = schema_result.valid

        result.duration_ms = (time.time() - start_time) * 1000

        # Emit validation event
        self._emit_event("review.validation.output", {
            "valid": result.valid,
            "error_count": len(result.errors),
            "redacted": redact_secrets,
            "duration_ms": result.duration_ms,
        })

        return result

    def _redact_secrets(self, data: Any, path: str = "") -> Any:
        """Redact potential secrets from output."""
        secret_keys = {"password", "secret", "token", "api_key", "apikey", "private_key", "credential"}

        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if k.lower() in secret_keys:
                    result[k] = "***REDACTED***"
                else:
                    result[k] = self._redact_secrets(v, f"{path}.{k}")
            return result
        elif isinstance(data, list):
            return [self._redact_secrets(item, f"{path}[{i}]") for i, item in enumerate(data)]
        elif isinstance(data, str):
            # Check for patterns that look like secrets
            if re.search(r'-----BEGIN.*PRIVATE KEY-----', data):
                return "***REDACTED_KEY***"
            if re.search(r'^[A-Za-z0-9+/]{40,}={0,2}$', data):
                # Base64-encoded data that's long (might be a key)
                return "***REDACTED***"
        return data


# ============================================================================
# Validation Engine (Combined)
# ============================================================================

class ValidationEngine:
    """
    Combined validation engine.

    Example:
        engine = ValidationEngine()

        # Validate input
        input_result = engine.validate_input(request_data, request_schema)

        # Validate output
        output_result = engine.validate_output(response_data, response_schema)
    """

    BUS_TOPICS = {
        "input": "review.validation.input",
        "output": "review.validation.output",
        "error": "review.validation.error",
    }

    def __init__(
        self,
        sanitization_mode: SanitizationMode = SanitizationMode.MODERATE,
        bus_path: Optional[Path] = None,
    ):
        """Initialize validation engine."""
        self.bus_path = bus_path or self._get_bus_path()

        self.input_validator = InputValidator(
            sanitization_mode=sanitization_mode,
            bus_path=self.bus_path,
        )
        self.output_validator = OutputValidator(
            bus_path=self.bus_path,
        )
        self.schema_validator = SchemaValidator()

        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "validation") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "validation-engine",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a named schema."""
        self._schemas[name] = schema

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a registered schema."""
        return self._schemas.get(name)

    def validate_input(
        self,
        data: Any,
        schema: Optional[Union[str, Dict[str, Any]]] = None,
        sanitize: bool = True,
    ) -> ValidationResult:
        """
        Validate input data.

        Args:
            data: Input data
            schema: Schema name or definition
            sanitize: Whether to sanitize

        Returns:
            ValidationResult
        """
        if isinstance(schema, str):
            schema = self._schemas.get(schema)

        return self.input_validator.validate(data, schema, sanitize)

    def validate_output(
        self,
        data: Any,
        schema: Optional[Union[str, Dict[str, Any]]] = None,
        redact_secrets: bool = True,
    ) -> ValidationResult:
        """
        Validate output data.

        Args:
            data: Output data
            schema: Schema name or definition
            redact_secrets: Whether to redact secrets

        Returns:
            ValidationResult
        """
        if isinstance(schema, str):
            schema = self._schemas.get(schema)

        return self.output_validator.validate(data, schema, redact_secrets)

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "validation-engine",
            "healthy": True,
            "schemas_registered": len(self._schemas),
            "custom_validators": len(self.input_validator._custom_validators),
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
    """CLI entry point for Validation Engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Validation Engine (Step 192)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument("data", help="JSON data to validate")
    validate_parser.add_argument("--schema", help="JSON schema")
    validate_parser.add_argument("--input", action="store_true", help="Validate as input")
    validate_parser.add_argument("--output", action="store_true", help="Validate as output")

    # Sanitize command
    sanitize_parser = subparsers.add_parser("sanitize", help="Sanitize data")
    sanitize_parser.add_argument("data", help="Data to sanitize")
    sanitize_parser.add_argument("--mode", choices=["strict", "moderate", "permissive"],
                                 default="moderate", help="Sanitization mode")

    # Path command
    path_parser = subparsers.add_parser("path", help="Validate file path")
    path_parser.add_argument("path", help="Path to validate")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    engine = ValidationEngine()

    if args.command == "validate":
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError:
            data = args.data

        schema = None
        if args.schema:
            schema = json.loads(args.schema)

        if args.output:
            result = engine.validate_output(data, schema)
        else:
            result = engine.validate_input(data, schema)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "VALID" if result.valid else "INVALID"
            print(f"Validation: {status}")
            print(f"  Errors: {len(result.errors)}")
            print(f"  Warnings: {len(result.warnings)}")
            for error in result.errors:
                print(f"  - [{error.code}] {error.path}: {error.message}")

    elif args.command == "sanitize":
        mode = SanitizationMode[args.mode.upper()]
        sanitized = Sanitizer.sanitize_string(args.data, mode)
        if args.json:
            print(json.dumps({"original": args.data, "sanitized": sanitized}, indent=2))
        else:
            print(f"Original: {args.data}")
            print(f"Sanitized: {sanitized}")

    elif args.command == "path":
        result = engine.input_validator.validate_file_path(args.path)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "SAFE" if result.valid else "UNSAFE"
            print(f"Path: {status}")
            if result.sanitized_data != args.path:
                print(f"  Sanitized: {result.sanitized_data}")

    else:
        # Default: show status
        status = engine.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Validation Engine: {status['schemas_registered']} schemas, {status['custom_validators']} validators")

    return 0


if __name__ == "__main__":
    sys.exit(main())
