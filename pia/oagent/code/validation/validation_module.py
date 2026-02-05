#!/usr/bin/env python3
"""
validation_module.py - Validation Module (Step 92)

PBTSO Phase: VERIFY, SKILL

Provides:
- Input validation for API requests
- Output validation for responses
- Schema validation (JSON Schema)
- Code syntax validation
- Path/file validation
- Custom validation rules

Bus Topics:
- code.validation.check
- code.validation.error
- code.validation.schema

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import ast
import json
import os
import re
import socket
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Pattern, Type, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class ValidationType(Enum):
    """Types of validation."""
    INPUT = "input"
    OUTPUT = "output"
    SCHEMA = "schema"
    CODE = "code"
    PATH = "path"
    CUSTOM = "custom"


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationConfig:
    """Configuration for validation module."""
    strict_mode: bool = False
    max_errors: int = 100
    enable_code_validation: bool = True
    enable_path_validation: bool = True
    allowed_paths: List[str] = field(default_factory=lambda: ["/pluribus"])
    blocked_paths: List[str] = field(default_factory=lambda: ["/etc", "/root"])
    max_input_size: int = 10 * 1024 * 1024  # 10MB
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strict_mode": self.strict_mode,
            "max_errors": self.max_errors,
            "enable_code_validation": self.enable_code_validation,
            "enable_path_validation": self.enable_path_validation,
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
# Validation Types
# =============================================================================

@dataclass
class ValidationError:
    """A validation error."""
    field: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    code: str = ""
    path: str = ""
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "path": self.path,
        }


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "duration_ms": self.duration_ms,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }

    def add_error(self, field: str, message: str, **kwargs: Any) -> None:
        """Add an error."""
        self.errors.append(ValidationError(field=field, message=message, **kwargs))
        self.valid = False

    def add_warning(self, field: str, message: str, **kwargs: Any) -> None:
        """Add a warning."""
        self.warnings.append(ValidationError(
            field=field,
            message=message,
            severity=ValidationSeverity.WARNING,
            **kwargs,
        ))


@dataclass
class ValidationRule:
    """A validation rule."""
    name: str
    validator: Callable[[Any], bool]
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    condition: Optional[Callable[[Any], bool]] = None

    def validate(self, value: Any) -> Optional[ValidationError]:
        """Run validation rule."""
        # Check condition
        if self.condition and not self.condition(value):
            return None

        # Run validator
        if not self.validator(value):
            return ValidationError(
                field=self.name,
                message=self.message,
                severity=self.severity,
            )

        return None


# =============================================================================
# Validators
# =============================================================================

class Validator(ABC):
    """Abstract base class for validators."""

    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate data."""
        pass


class SchemaValidator(Validator):
    """
    JSON Schema-style validator.

    Supports a subset of JSON Schema for validation.
    """

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate(self, data: Any) -> ValidationResult:
        """Validate data against schema."""
        start = time.time()
        result = ValidationResult(valid=True)

        self._validate_value(data, self.schema, "", result)

        result.duration_ms = (time.time() - start) * 1000
        return result

    def _validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate a value against schema."""
        # Type validation
        schema_type = schema.get("type")
        if schema_type:
            if not self._check_type(value, schema_type):
                result.add_error(
                    path or "root",
                    f"Expected type {schema_type}, got {type(value).__name__}",
                    path=path,
                )
                return

        # Required validation
        if schema.get("required") and value is None:
            result.add_error(path or "root", "Required field is missing", path=path)
            return

        # Enum validation
        enum_values = schema.get("enum")
        if enum_values and value not in enum_values:
            result.add_error(path or "root", f"Value must be one of: {enum_values}", path=path)

        # String validations
        if schema_type == "string" and isinstance(value, str):
            self._validate_string(value, schema, path, result)

        # Number validations
        if schema_type in ("number", "integer") and isinstance(value, (int, float)):
            self._validate_number(value, schema, path, result)

        # Array validations
        if schema_type == "array" and isinstance(value, list):
            self._validate_array(value, schema, path, result)

        # Object validations
        if schema_type == "object" and isinstance(value, dict):
            self._validate_object(value, schema, path, result)

    def _check_type(self, value: Any, schema_type: str) -> bool:
        """Check if value matches schema type."""
        if value is None:
            return schema_type == "null"

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected = type_map.get(schema_type)
        return expected is None or isinstance(value, expected)

    def _validate_string(
        self,
        value: str,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate string value."""
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")
        format_type = schema.get("format")

        if min_length and len(value) < min_length:
            result.add_error(path, f"String length must be at least {min_length}", path=path)

        if max_length and len(value) > max_length:
            result.add_error(path, f"String length must be at most {max_length}", path=path)

        if pattern:
            if not re.match(pattern, value):
                result.add_error(path, f"String must match pattern: {pattern}", path=path)

        if format_type:
            self._validate_format(value, format_type, path, result)

    def _validate_format(
        self,
        value: str,
        format_type: str,
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate string format."""
        format_patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "uri": r"^https?://",
            "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "date-time": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        }

        pattern = format_patterns.get(format_type)
        if pattern and not re.match(pattern, value, re.IGNORECASE):
            result.add_error(path, f"Invalid {format_type} format", path=path)

    def _validate_number(
        self,
        value: Union[int, float],
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate number value."""
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_min = schema.get("exclusiveMinimum")
        exclusive_max = schema.get("exclusiveMaximum")
        multiple_of = schema.get("multipleOf")

        if minimum is not None and value < minimum:
            result.add_error(path, f"Value must be >= {minimum}", path=path)

        if maximum is not None and value > maximum:
            result.add_error(path, f"Value must be <= {maximum}", path=path)

        if exclusive_min is not None and value <= exclusive_min:
            result.add_error(path, f"Value must be > {exclusive_min}", path=path)

        if exclusive_max is not None and value >= exclusive_max:
            result.add_error(path, f"Value must be < {exclusive_max}", path=path)

        if multiple_of is not None and value % multiple_of != 0:
            result.add_error(path, f"Value must be multiple of {multiple_of}", path=path)

    def _validate_array(
        self,
        value: list,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate array value."""
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        unique_items = schema.get("uniqueItems")
        items_schema = schema.get("items")

        if min_items and len(value) < min_items:
            result.add_error(path, f"Array must have at least {min_items} items", path=path)

        if max_items and len(value) > max_items:
            result.add_error(path, f"Array must have at most {max_items} items", path=path)

        if unique_items and len(value) != len(set(str(v) for v in value)):
            result.add_error(path, "Array items must be unique", path=path)

        if items_schema:
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                self._validate_value(item, items_schema, item_path, result)

    def _validate_object(
        self,
        value: dict,
        schema: Dict[str, Any],
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate object value."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)

        # Check required properties
        for req in required:
            if req not in value:
                prop_path = f"{path}.{req}" if path else req
                result.add_error(prop_path, f"Required property '{req}' is missing", path=prop_path)

        # Validate properties
        for key, prop_value in value.items():
            prop_path = f"{path}.{key}" if path else key

            if key in properties:
                self._validate_value(prop_value, properties[key], prop_path, result)
            elif not additional:
                result.add_warning(prop_path, f"Unknown property '{key}'", path=prop_path)


class CodeValidator(Validator):
    """
    Code syntax validator.

    Validates Python code syntax.
    """

    def __init__(self, language: str = "python"):
        self.language = language

    def validate(self, code: str) -> ValidationResult:
        """Validate code syntax."""
        start = time.time()
        result = ValidationResult(valid=True)

        if self.language == "python":
            self._validate_python(code, result)
        else:
            result.add_warning("language", f"Unsupported language: {self.language}")

        result.duration_ms = (time.time() - start) * 1000
        return result

    def _validate_python(self, code: str, result: ValidationResult) -> None:
        """Validate Python code syntax."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            result.add_error(
                "syntax",
                f"Syntax error at line {e.lineno}: {e.msg}",
                path=f"line:{e.lineno}",
            )

        # Additional checks
        self._check_dangerous_patterns(code, result)

    def _check_dangerous_patterns(self, code: str, result: ValidationResult) -> None:
        """Check for potentially dangerous code patterns."""
        dangerous_patterns = [
            (r"\beval\s*\(", "Use of eval() is dangerous"),
            (r"\bexec\s*\(", "Use of exec() is dangerous"),
            (r"\b__import__\s*\(", "Dynamic import may be dangerous"),
            (r"\bos\.system\s*\(", "os.system() may be dangerous"),
            (r"\bsubprocess\.(call|run|Popen)", "subprocess calls may be dangerous"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                result.add_warning("security", message)


class PathValidator(Validator):
    """
    File path validator.

    Validates file paths for security.
    """

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        blocked_paths: Optional[List[str]] = None,
    ):
        self.allowed_paths = allowed_paths or ["/pluribus"]
        self.blocked_paths = blocked_paths or ["/etc", "/root", "/var"]

    def validate(self, path: str) -> ValidationResult:
        """Validate file path."""
        start = time.time()
        result = ValidationResult(valid=True)

        # Normalize path
        try:
            normalized = str(Path(path).resolve())
        except Exception as e:
            result.add_error("path", f"Invalid path: {e}")
            return result

        # Check for path traversal
        if ".." in path:
            result.add_error("path", "Path traversal detected", code="PATH_TRAVERSAL")

        # Check blocked paths
        for blocked in self.blocked_paths:
            if normalized.startswith(blocked):
                result.add_error("path", f"Access to {blocked} is blocked", code="BLOCKED_PATH")

        # Check allowed paths (if specified)
        if self.allowed_paths:
            allowed = any(normalized.startswith(a) for a in self.allowed_paths)
            if not allowed:
                result.add_error("path", "Path not in allowed directories", code="NOT_ALLOWED")

        result.duration_ms = (time.time() - start) * 1000
        return result


class InputValidator(Validator):
    """
    Generic input validator.

    Combines multiple validation rules.
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> "InputValidator":
        """Add a validation rule."""
        self.rules.append(rule)
        return self

    def required(self, field: str, message: str = "Field is required") -> "InputValidator":
        """Add required rule."""
        return self.add_rule(ValidationRule(
            name=field,
            validator=lambda v: v is not None and v != "",
            message=message,
        ))

    def string(self, field: str, min_length: int = 0, max_length: int = 0) -> "InputValidator":
        """Add string validation rule."""
        def validator(v: Any) -> bool:
            if not isinstance(v, str):
                return False
            if min_length and len(v) < min_length:
                return False
            if max_length and len(v) > max_length:
                return False
            return True

        return self.add_rule(ValidationRule(
            name=field,
            validator=validator,
            message=f"Invalid string (min: {min_length}, max: {max_length})",
        ))

    def pattern(self, field: str, pattern: str, message: str = "Invalid format") -> "InputValidator":
        """Add pattern validation rule."""
        regex = re.compile(pattern)
        return self.add_rule(ValidationRule(
            name=field,
            validator=lambda v: bool(regex.match(str(v))),
            message=message,
        ))

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against all rules."""
        start = time.time()
        result = ValidationResult(valid=True)

        for rule in self.rules:
            value = data.get(rule.name)
            error = rule.validate(value)
            if error:
                if error.severity == ValidationSeverity.ERROR:
                    result.add_error(error.field, error.message)
                else:
                    result.add_warning(error.field, error.message)

        result.duration_ms = (time.time() - start) * 1000
        return result


# =============================================================================
# Validation Module
# =============================================================================

class ValidationModule:
    """
    Validation module for input/output validation.

    PBTSO Phase: VERIFY, SKILL

    Features:
    - Schema validation (JSON Schema subset)
    - Code syntax validation
    - Path security validation
    - Custom validation rules
    - Validation caching

    Usage:
        validator = ValidationModule()

        # Schema validation
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = validator.validate_schema(data, schema)

        # Code validation
        result = validator.validate_code("def foo(): pass")

        # Path validation
        result = validator.validate_path("/pluribus/file.py")
    """

    BUS_TOPICS = {
        "check": "code.validation.check",
        "error": "code.validation.error",
        "schema": "code.validation.schema",
    }

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or ValidationConfig()
        self.bus = bus or LockedAgentBus()

        # Validators
        self._path_validator = PathValidator(
            allowed_paths=self.config.allowed_paths,
            blocked_paths=self.config.blocked_paths,
        )
        self._code_validator = CodeValidator()

        # Statistics
        self._validations = 0
        self._errors = 0
        self._lock = Lock()

    # =========================================================================
    # Schema Validation
    # =========================================================================

    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate data against JSON Schema.

        Args:
            data: Data to validate
            schema: JSON Schema definition

        Returns:
            ValidationResult
        """
        validator = SchemaValidator(schema)
        result = validator.validate(data)

        self._record_validation("schema", result)
        return result

    # =========================================================================
    # Code Validation
    # =========================================================================

    def validate_code(
        self,
        code: str,
        language: str = "python",
    ) -> ValidationResult:
        """
        Validate code syntax.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            ValidationResult
        """
        if not self.config.enable_code_validation:
            return ValidationResult(valid=True)

        validator = CodeValidator(language)
        result = validator.validate(code)

        self._record_validation("code", result)
        return result

    # =========================================================================
    # Path Validation
    # =========================================================================

    def validate_path(self, path: str) -> ValidationResult:
        """
        Validate file path for security.

        Args:
            path: File path to validate

        Returns:
            ValidationResult
        """
        if not self.config.enable_path_validation:
            return ValidationResult(valid=True)

        result = self._path_validator.validate(path)

        self._record_validation("path", result)
        return result

    def validate_paths(self, paths: List[str]) -> ValidationResult:
        """Validate multiple paths."""
        combined = ValidationResult(valid=True)

        for path in paths:
            result = self.validate_path(path)
            combined.errors.extend(result.errors)
            combined.warnings.extend(result.warnings)
            if not result.valid:
                combined.valid = False

        return combined

    # =========================================================================
    # Input Validation
    # =========================================================================

    def validate_input(
        self,
        data: Any,
        rules: Optional[List[ValidationRule]] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate input data.

        Args:
            data: Input data
            rules: Custom validation rules
            schema: Optional JSON Schema

        Returns:
            ValidationResult
        """
        start = time.time()
        combined = ValidationResult(valid=True)

        # Size check
        if isinstance(data, (str, bytes)):
            if len(data) > self.config.max_input_size:
                combined.add_error("size", f"Input exceeds maximum size ({self.config.max_input_size})")
                return combined

        # Schema validation
        if schema:
            schema_result = self.validate_schema(data, schema)
            combined.errors.extend(schema_result.errors)
            combined.warnings.extend(schema_result.warnings)
            if not schema_result.valid:
                combined.valid = False

        # Custom rules
        if rules and isinstance(data, dict):
            for rule in rules:
                value = data.get(rule.name)
                error = rule.validate(value)
                if error:
                    if error.severity == ValidationSeverity.ERROR:
                        combined.add_error(error.field, error.message)
                    else:
                        combined.add_warning(error.field, error.message)

        combined.duration_ms = (time.time() - start) * 1000
        self._record_validation("input", combined)
        return combined

    # =========================================================================
    # Output Validation
    # =========================================================================

    def validate_output(
        self,
        data: Any,
        schema: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate output data.

        Args:
            data: Output data
            schema: Optional JSON Schema

        Returns:
            ValidationResult
        """
        if schema:
            result = self.validate_schema(data, schema)
        else:
            result = ValidationResult(valid=True)

        self._record_validation("output", result)
        return result

    # =========================================================================
    # Builder
    # =========================================================================

    def input_validator(self) -> InputValidator:
        """Get a new input validator builder."""
        return InputValidator()

    # =========================================================================
    # Internal
    # =========================================================================

    def _record_validation(self, type_: str, result: ValidationResult) -> None:
        """Record validation statistics."""
        with self._lock:
            self._validations += 1
            if not result.valid:
                self._errors += 1

        # Emit event
        self.bus.emit({
            "topic": self.BUS_TOPICS["check"],
            "kind": "validation",
            "actor": "validation-module",
            "data": {
                "type": type_,
                "valid": result.valid,
                "error_count": len(result.errors),
                "duration_ms": result.duration_ms,
            },
        })

        # Emit errors
        if not result.valid:
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "warning",
                "actor": "validation-module",
                "data": {
                    "type": type_,
                    "errors": [e.to_dict() for e in result.errors[:10]],
                },
            })

    def stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        with self._lock:
            return {
                "total_validations": self._validations,
                "total_errors": self._errors,
                "error_rate": self._errors / max(self._validations, 1),
                "config": self.config.to_dict(),
            }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Validation Module."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validation Module (Step 92)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # schema command
    schema_parser = subparsers.add_parser("schema", help="Validate against schema")
    schema_parser.add_argument("data_file", help="JSON data file")
    schema_parser.add_argument("schema_file", help="JSON schema file")

    # code command
    code_parser = subparsers.add_parser("code", help="Validate code syntax")
    code_parser.add_argument("file", help="Code file")
    code_parser.add_argument("--language", "-l", default="python")

    # path command
    path_parser = subparsers.add_parser("path", help="Validate file path")
    path_parser.add_argument("paths", nargs="+", help="Paths to validate")

    # demo command
    subparsers.add_parser("demo", help="Run validation demo")

    args = parser.parse_args()
    validator = ValidationModule()

    if args.command == "schema":
        with open(args.data_file) as f:
            data = json.load(f)
        with open(args.schema_file) as f:
            schema = json.load(f)

        result = validator.validate_schema(data, schema)
        print(f"Valid: {result.valid}")
        if result.errors:
            print("Errors:")
            for e in result.errors:
                print(f"  - {e.field}: {e.message}")
        return 0 if result.valid else 1

    elif args.command == "code":
        with open(args.file) as f:
            code = f.read()

        result = validator.validate_code(code, args.language)
        print(f"Valid: {result.valid}")
        if result.errors:
            print("Errors:")
            for e in result.errors:
                print(f"  - {e.field}: {e.message}")
        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w.field}: {w.message}")
        return 0 if result.valid else 1

    elif args.command == "path":
        for path in args.paths:
            result = validator.validate_path(path)
            status = "[OK]" if result.valid else "[BLOCKED]"
            print(f"{status} {path}")
            for e in result.errors:
                print(f"       {e.message}")
        return 0

    elif args.command == "demo":
        print("Validation Module Demo\n")

        # Schema validation
        print("1. Schema Validation")
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "format": "email"},
            },
        }

        valid_data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
        invalid_data = {"name": "", "age": -5}

        result1 = validator.validate_schema(valid_data, schema)
        print(f"  Valid data: {result1.valid}")

        result2 = validator.validate_schema(invalid_data, schema)
        print(f"  Invalid data: {result2.valid}")
        for e in result2.errors:
            print(f"    - {e.field}: {e.message}")

        # Code validation
        print("\n2. Code Validation")
        valid_code = "def hello(name):\n    return f'Hello, {name}!'"
        invalid_code = "def hello(name:\n    return"

        result3 = validator.validate_code(valid_code)
        print(f"  Valid code: {result3.valid}")

        result4 = validator.validate_code(invalid_code)
        print(f"  Invalid code: {result4.valid}")
        for e in result4.errors:
            print(f"    - {e.message}")

        # Path validation
        print("\n3. Path Validation")
        for path in ["/pluribus/file.py", "/etc/passwd", "../../../etc/passwd"]:
            result = validator.validate_path(path)
            status = "OK" if result.valid else "BLOCKED"
            print(f"  {path}: {status}")

        print("\nStatistics:")
        print(json.dumps(validator.stats(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
