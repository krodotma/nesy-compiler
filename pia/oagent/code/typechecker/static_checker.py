#!/usr/bin/env python3
"""
static_checker.py - Static Type Checker (Step 77)

PBTSO Phase: VERIFY

Provides:
- Static type checking for Python (mypy)
- Type inference suggestions
- Type annotation validation
- Integration with type stubs
- Incremental type checking

Bus Topics:
- code.typecheck.check
- code.typecheck.issues
- code.typecheck.suggest

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class TypeSeverity(Enum):
    """Severity levels for type issues."""
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


class TypeCheckMode(Enum):
    """Type checking strictness modes."""
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"


@dataclass
class TypeCheckerConfig:
    """Configuration for the type checker."""
    # Python/mypy settings
    python_version: str = "3.9"
    check_mode: TypeCheckMode = TypeCheckMode.NORMAL
    strict: bool = False
    ignore_missing_imports: bool = True
    show_error_codes: bool = True
    warn_return_any: bool = False
    warn_unused_ignores: bool = True

    # General settings
    incremental: bool = True
    cache_dir: str = ".mypy_cache"
    parallel_jobs: int = 4
    timeout_s: int = 120
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "check_mode": self.check_mode.value,
            "strict": self.strict,
            "ignore_missing_imports": self.ignore_missing_imports,
            "show_error_codes": self.show_error_codes,
            "incremental": self.incremental,
            "timeout_s": self.timeout_s,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class TypeIssue:
    """A single type checking issue."""
    file_path: str
    line: int
    column: int
    severity: TypeSeverity
    code: str
    message: str
    source: str  # type checker that found it
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "suggestion": self.suggestion,
        }


@dataclass
class TypeCheckResult:
    """Result of type checking."""
    success: bool
    total_issues: int
    errors: int
    warnings: int
    files_checked: int
    issues: List[TypeIssue] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_issues": self.total_issues,
            "errors": self.errors,
            "warnings": self.warnings,
            "files_checked": self.files_checked,
            "issues": [i.to_dict() for i in self.issues],
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


@dataclass
class TypeSuggestion:
    """A type annotation suggestion."""
    file_path: str
    line: int
    symbol: str
    suggested_type: str
    confidence: float
    context: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line": self.line,
            "symbol": self.symbol,
            "suggested_type": self.suggested_type,
            "confidence": self.confidence,
            "context": self.context,
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
# Type Checker
# =============================================================================

class TypeChecker:
    """
    Static type checker.

    PBTSO Phase: VERIFY

    Responsibilities:
    - Run type checking with mypy
    - Parse and aggregate type errors
    - Suggest type annotations
    - Support incremental checking

    Usage:
        checker = TypeChecker(config)
        result = checker.check_file(path)
    """

    BUS_TOPICS = {
        "check": "code.typecheck.check",
        "issues": "code.typecheck.issues",
        "suggest": "code.typecheck.suggest",
        "heartbeat": "code.typecheck.heartbeat",
    }

    def __init__(
        self,
        config: Optional[TypeCheckerConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or TypeCheckerConfig()
        self.bus = bus or LockedAgentBus()
        self._mypy_available: Optional[bool] = None

    def _check_mypy(self) -> bool:
        """Check if mypy is available."""
        if self._mypy_available is None:
            self._mypy_available = shutil.which("mypy") is not None
        return self._mypy_available

    def check_file(self, file_path: Path) -> TypeCheckResult:
        """
        Type check a single file.

        Args:
            file_path: Path to Python file

        Returns:
            TypeCheckResult with issues found
        """
        start_time = time.time()

        if not file_path.exists():
            return TypeCheckResult(
                success=False,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=f"File not found: {file_path}",
            )

        if not file_path.suffix == ".py":
            return TypeCheckResult(
                success=True,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=1,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        if not self._check_mypy():
            return TypeCheckResult(
                success=False,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                error="mypy not available",
            )

        try:
            issues = self._run_mypy([file_path])

            errors = sum(1 for i in issues if i.severity == TypeSeverity.ERROR)
            warnings = sum(1 for i in issues if i.severity == TypeSeverity.WARNING)

            # Emit check event
            self.bus.emit({
                "topic": self.BUS_TOPICS["check"],
                "kind": "typecheck",
                "actor": "type-checker",
                "data": {
                    "file": str(file_path),
                    "issues": len(issues),
                    "errors": errors,
                    "warnings": warnings,
                },
            })

            return TypeCheckResult(
                success=True,
                total_issues=len(issues),
                errors=errors,
                warnings=warnings,
                files_checked=1,
                issues=issues,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return TypeCheckResult(
                success=False,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=1,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def check_files(self, file_paths: List[Path]) -> TypeCheckResult:
        """
        Type check multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            TypeCheckResult with all issues
        """
        start_time = time.time()

        python_files = [f for f in file_paths if f.suffix == ".py" and f.exists()]

        if not python_files:
            return TypeCheckResult(
                success=True,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=0,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        if not self._check_mypy():
            return TypeCheckResult(
                success=False,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                error="mypy not available",
            )

        try:
            issues = self._run_mypy(python_files)

            errors = sum(1 for i in issues if i.severity == TypeSeverity.ERROR)
            warnings = sum(1 for i in issues if i.severity == TypeSeverity.WARNING)

            return TypeCheckResult(
                success=True,
                total_issues=len(issues),
                errors=errors,
                warnings=warnings,
                files_checked=len(python_files),
                issues=issues,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return TypeCheckResult(
                success=False,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=len(python_files),
                elapsed_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def check_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> TypeCheckResult:
        """
        Type check all Python files in a directory.

        Args:
            directory: Directory to check
            recursive: Search recursively

        Returns:
            TypeCheckResult with all issues
        """
        pattern = "**/*.py" if recursive else "*.py"
        files = list(directory.glob(pattern))

        # Filter out hidden directories and common exclusions
        files = [
            f for f in files
            if not any(part.startswith('.') for part in f.parts)
            and 'venv' not in f.parts
            and 'node_modules' not in f.parts
            and '__pycache__' not in f.parts
        ]

        return self.check_files(files)

    def _run_mypy(self, file_paths: List[Path]) -> List[TypeIssue]:
        """Run mypy and parse output."""
        issues: List[TypeIssue] = []

        cmd = ["mypy"]

        # Add configuration options
        cmd.extend(["--python-version", self.config.python_version])

        if self.config.strict:
            cmd.append("--strict")

        if self.config.ignore_missing_imports:
            cmd.append("--ignore-missing-imports")

        if self.config.show_error_codes:
            cmd.append("--show-error-codes")

        if self.config.warn_return_any:
            cmd.append("--warn-return-any")

        if self.config.warn_unused_ignores:
            cmd.append("--warn-unused-ignores")

        if self.config.incremental:
            cmd.extend(["--cache-dir", self.config.cache_dir])
        else:
            cmd.append("--no-incremental")

        # Add file paths
        cmd.extend([str(f) for f in file_paths])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            # Parse mypy output
            # Format: file:line:col: severity: message [error-code]
            pattern = re.compile(
                r'^(.+):(\d+):(\d+): (error|warning|note): (.+?)(?:\s+\[(\w+(?:-\w+)*)\])?$'
            )

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                match = pattern.match(line)
                if match:
                    severity_map = {
                        "error": TypeSeverity.ERROR,
                        "warning": TypeSeverity.WARNING,
                        "note": TypeSeverity.NOTE,
                    }

                    issues.append(TypeIssue(
                        file_path=match.group(1),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        severity=severity_map.get(match.group(4), TypeSeverity.ERROR),
                        code=match.group(6) or "",
                        message=match.group(5),
                        source="mypy",
                    ))

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"mypy timed out after {self.config.timeout_s}s")
        except OSError as e:
            raise RuntimeError(f"Failed to run mypy: {e}")

        return issues

    def suggest_types(self, file_path: Path) -> List[TypeSuggestion]:
        """
        Suggest type annotations for untyped code.

        Args:
            file_path: Path to Python file

        Returns:
            List of type annotation suggestions
        """
        suggestions: List[TypeSuggestion] = []

        if not file_path.exists() or file_path.suffix != ".py":
            return suggestions

        try:
            content = file_path.read_text()
            lines = content.split('\n')

            # Find function definitions without return type
            func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)\s*:')

            for line_num, line in enumerate(lines, 1):
                match = func_pattern.match(line)
                if match and '->' not in line:
                    func_name = match.group(2)
                    params = match.group(3)

                    # Infer return type from function body (simplified)
                    suggested_type = self._infer_return_type(lines, line_num)

                    suggestions.append(TypeSuggestion(
                        file_path=str(file_path),
                        line=line_num,
                        symbol=func_name,
                        suggested_type=suggested_type,
                        confidence=0.6,
                        context=line.strip(),
                    ))

            # Find variables without type annotation
            assign_pattern = re.compile(r'^(\s*)(\w+)\s*=\s*(.+)$')

            for line_num, line in enumerate(lines, 1):
                # Skip if it's a type annotated assignment
                if ':' in line.split('=')[0]:
                    continue

                match = assign_pattern.match(line)
                if match:
                    var_name = match.group(2)
                    value = match.group(3).strip()

                    # Skip common patterns
                    if var_name.startswith('_') or var_name.isupper():
                        continue

                    # Infer type from value
                    suggested_type = self._infer_value_type(value)

                    if suggested_type and suggested_type != "Any":
                        suggestions.append(TypeSuggestion(
                            file_path=str(file_path),
                            line=line_num,
                            symbol=var_name,
                            suggested_type=suggested_type,
                            confidence=0.7,
                            context=line.strip(),
                        ))

            # Emit suggest event
            if suggestions:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["suggest"],
                    "kind": "typecheck",
                    "actor": "type-checker",
                    "data": {
                        "file": str(file_path),
                        "suggestions": len(suggestions),
                    },
                })

        except (OSError, UnicodeDecodeError):
            pass

        return suggestions

    def _infer_return_type(self, lines: List[str], func_line: int) -> str:
        """Infer return type from function body."""
        # Simple heuristic - look for return statements
        indent_level = len(lines[func_line - 1]) - len(lines[func_line - 1].lstrip())

        for i in range(func_line, min(func_line + 50, len(lines))):
            line = lines[i]

            # Check for end of function
            if line.strip() and not line.startswith(' ' * (indent_level + 1)):
                if not line.strip().startswith('#'):
                    break

            # Look for return statements
            if 'return ' in line:
                return_val = line.split('return')[1].strip()

                if return_val.startswith('"') or return_val.startswith("'"):
                    return "str"
                elif return_val.isdigit() or (return_val.startswith('-') and return_val[1:].isdigit()):
                    return "int"
                elif return_val in {'True', 'False'}:
                    return "bool"
                elif return_val.startswith('['):
                    return "List"
                elif return_val.startswith('{'):
                    return "Dict"
                elif return_val == 'None':
                    return "None"
                else:
                    return "Any"

        return "None"

    def _infer_value_type(self, value: str) -> Optional[str]:
        """Infer type from assigned value."""
        value = value.strip()

        # String literals
        if value.startswith('"') or value.startswith("'"):
            return "str"

        # Numbers
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return "int"

        # Floats
        try:
            if '.' in value:
                float(value)
                return "float"
        except ValueError:
            pass

        # Booleans
        if value in {'True', 'False'}:
            return "bool"

        # Collections
        if value.startswith('['):
            return "List"
        if value.startswith('{') and ':' in value:
            return "Dict"
        if value.startswith('{'):
            return "Set"
        if value.startswith('('):
            return "Tuple"

        # None
        if value == 'None':
            return "Optional"

        # Function calls
        if '(' in value:
            func_name = value.split('(')[0]
            type_hints = {
                'str': 'str',
                'int': 'int',
                'float': 'float',
                'bool': 'bool',
                'list': 'List',
                'dict': 'Dict',
                'set': 'Set',
                'tuple': 'Tuple',
            }
            return type_hints.get(func_name)

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get type checker statistics."""
        return {
            "mypy_available": self._check_mypy(),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Type Checker."""
    import argparse

    parser = argparse.ArgumentParser(description="Static Type Checker (Step 77)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check command
    check_parser = subparsers.add_parser("check", help="Type check files")
    check_parser.add_argument("paths", nargs="+", help="Files or directories to check")
    check_parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    # suggest command
    suggest_parser = subparsers.add_parser("suggest", help="Suggest type annotations")
    suggest_parser.add_argument("file", help="File to analyze")
    suggest_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    subparsers.add_parser("stats", help="Show checker stats")

    args = parser.parse_args()

    config = TypeCheckerConfig(
        strict=getattr(args, "strict", False),
    )
    checker = TypeChecker(config)

    if args.command == "check":
        all_issues: List[TypeIssue] = []
        total_files = 0
        total_errors = 0
        total_warnings = 0

        for path_str in args.paths:
            path = Path(path_str)

            if path.is_dir():
                result = checker.check_directory(path)
            else:
                result = checker.check_file(path)

            all_issues.extend(result.issues)
            total_files += result.files_checked
            total_errors += result.errors
            total_warnings += result.warnings

        if args.json:
            print(json.dumps({
                "total_issues": len(all_issues),
                "errors": total_errors,
                "warnings": total_warnings,
                "files_checked": total_files,
                "issues": [i.to_dict() for i in all_issues],
            }, indent=2))
        else:
            print(f"Checked {total_files} files")
            print(f"  Errors: {total_errors}")
            print(f"  Warnings: {total_warnings}")

            if all_issues:
                print("\nIssues:")
                for issue in all_issues[:50]:
                    severity = issue.severity.value.upper()
                    code_str = f"[{issue.code}] " if issue.code else ""
                    print(f"  {issue.file_path}:{issue.line}:{issue.column} [{severity}] {code_str}{issue.message}")

                if len(all_issues) > 50:
                    print(f"  ... and {len(all_issues) - 50} more issues")

        return 1 if total_errors > 0 else 0

    elif args.command == "suggest":
        path = Path(args.file)
        suggestions = checker.suggest_types(path)

        if args.json:
            print(json.dumps([s.to_dict() for s in suggestions], indent=2))
        else:
            if suggestions:
                print(f"Type suggestions for {args.file}:")
                for s in suggestions:
                    print(f"  Line {s.line}: {s.symbol}: {s.suggested_type} (confidence: {s.confidence:.2f})")
                    print(f"    Context: {s.context}")
            else:
                print("No type suggestions found")
        return 0

    elif args.command == "stats":
        stats = checker.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
