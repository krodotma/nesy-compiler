#!/usr/bin/env python3
"""
refactoring_engine.py - Automated Refactoring Engine (Step 74)

PBTSO Phase: ITERATE, VERIFY

Provides:
- Automated code refactoring operations
- Extract method/class refactoring
- Rename with reference updates
- Move code between files
- Inline function/variable refactoring
- Safety checks before refactoring

Bus Topics:
- code.refactor.start
- code.refactor.complete
- code.refactor.error

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import ast
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class RefactoringType(Enum):
    """Types of refactoring operations."""
    RENAME = "rename"
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    EXTRACT_VARIABLE = "extract_variable"
    INLINE_VARIABLE = "inline_variable"
    INLINE_METHOD = "inline_method"
    MOVE = "move"
    CHANGE_SIGNATURE = "change_signature"
    CONVERT_TO_ASYNC = "convert_to_async"
    INTRODUCE_PARAMETER = "introduce_parameter"
    ENCAPSULATE_FIELD = "encapsulate_field"
    PULL_UP = "pull_up"
    PUSH_DOWN = "push_down"


class SafetyLevel(Enum):
    """Safety level for refactoring."""
    STRICT = "strict"      # All safety checks must pass
    NORMAL = "normal"      # Most safety checks
    LENIENT = "lenient"    # Minimal checks


@dataclass
class RefactoringConfig:
    """Configuration for the refactoring engine."""
    safety_level: SafetyLevel = SafetyLevel.NORMAL
    backup_before_refactor: bool = True
    validate_syntax_after: bool = True
    update_imports: bool = True
    update_docstrings: bool = True
    preserve_comments: bool = True
    max_files_per_operation: int = 50
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safety_level": self.safety_level.value,
            "backup_before_refactor": self.backup_before_refactor,
            "validate_syntax_after": self.validate_syntax_after,
            "update_imports": self.update_imports,
            "update_docstrings": self.update_docstrings,
            "preserve_comments": self.preserve_comments,
            "max_files_per_operation": self.max_files_per_operation,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class RefactoringOperation:
    """Definition of a refactoring operation."""
    id: str
    refactor_type: RefactoringType
    target: str  # File path or symbol
    options: Dict[str, Any] = field(default_factory=dict)
    scope: List[str] = field(default_factory=list)  # Files to search for references

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.refactor_type.value,
            "target": self.target,
            "options": self.options,
            "scope": self.scope,
        }


@dataclass
class RefactoringChange:
    """A single change made during refactoring."""
    file_path: str
    old_content: str
    new_content: str
    description: str
    line_changes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "description": self.description,
            "line_changes": self.line_changes,
        }


@dataclass
class RefactoringResult:
    """Result of a refactoring operation."""
    operation_id: str
    success: bool
    changes: List[RefactoringChange] = field(default_factory=list)
    files_modified: int = 0
    elapsed_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "success": self.success,
            "changes": [c.to_dict() for c in self.changes],
            "files_modified": self.files_modified,
            "elapsed_ms": self.elapsed_ms,
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass
class SymbolReference:
    """A reference to a symbol in code."""
    file_path: str
    line: int
    column: int
    context: str  # Surrounding code
    is_definition: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "context": self.context,
            "is_definition": self.is_definition,
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
# Refactoring Engine
# =============================================================================

class RefactoringEngine:
    """
    Automated refactoring engine.

    PBTSO Phase: ITERATE, VERIFY

    Responsibilities:
    - Execute refactoring operations
    - Find all references to symbols
    - Update references across files
    - Validate refactoring safety
    - Create backups before changes

    Usage:
        engine = RefactoringEngine(config)
        result = engine.refactor(operation)
    """

    BUS_TOPICS = {
        "start": "code.refactor.start",
        "complete": "code.refactor.complete",
        "error": "code.refactor.error",
        "heartbeat": "code.refactor.heartbeat",
    }

    def __init__(
        self,
        config: Optional[RefactoringConfig] = None,
        bus: Optional[LockedAgentBus] = None,
        working_dir: Optional[Path] = None,
    ):
        self.config = config or RefactoringConfig()
        self.bus = bus or LockedAgentBus()
        self.working_dir = working_dir or Path("/pluribus")
        self._handlers: Dict[RefactoringType, Callable] = {}
        self._backups: Dict[str, Dict[str, str]] = {}  # operation_id -> {file: content}

        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register refactoring type handlers."""
        self._handlers = {
            RefactoringType.RENAME: self._refactor_rename,
            RefactoringType.EXTRACT_METHOD: self._refactor_extract_method,
            RefactoringType.EXTRACT_VARIABLE: self._refactor_extract_variable,
            RefactoringType.INLINE_VARIABLE: self._refactor_inline_variable,
            RefactoringType.MOVE: self._refactor_move,
            RefactoringType.CONVERT_TO_ASYNC: self._refactor_convert_to_async,
            RefactoringType.CHANGE_SIGNATURE: self._refactor_change_signature,
        }

    def refactor(self, operation: RefactoringOperation) -> RefactoringResult:
        """
        Execute a refactoring operation.

        Args:
            operation: Refactoring operation to execute

        Returns:
            RefactoringResult with changes made
        """
        start_time = time.time()
        changes: List[RefactoringChange] = []
        warnings: List[str] = []

        # Emit start event
        self.bus.emit({
            "topic": self.BUS_TOPICS["start"],
            "kind": "refactor",
            "actor": "refactoring-engine",
            "data": operation.to_dict(),
        })

        try:
            # Validate operation
            validation_errors = self._validate_operation(operation)
            if validation_errors:
                raise ValueError(f"Validation failed: {', '.join(validation_errors)}")

            # Create backups
            if self.config.backup_before_refactor:
                self._create_backup(operation)

            # Get handler
            handler = self._handlers.get(operation.refactor_type)
            if not handler:
                raise ValueError(f"Unsupported refactoring type: {operation.refactor_type.value}")

            # Execute refactoring
            changes, warnings = handler(operation)

            # Validate syntax after
            if self.config.validate_syntax_after:
                for change in changes:
                    if change.file_path.endswith(".py"):
                        try:
                            ast.parse(change.new_content)
                        except SyntaxError as e:
                            warnings.append(f"Syntax error in {change.file_path}: {e}")

            # Apply changes
            for change in changes:
                file_path = Path(change.file_path)
                if file_path.is_absolute():
                    full_path = file_path
                else:
                    full_path = self.working_dir / file_path

                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(change.new_content)

            result = RefactoringResult(
                operation_id=operation.id,
                success=True,
                changes=changes,
                files_modified=len(changes),
                elapsed_ms=(time.time() - start_time) * 1000,
                warnings=warnings,
            )

            # Emit complete event
            self.bus.emit({
                "topic": self.BUS_TOPICS["complete"],
                "kind": "refactor",
                "actor": "refactoring-engine",
                "data": result.to_dict(),
            })

            return result

        except Exception as e:
            # Rollback on error
            if self.config.backup_before_refactor and operation.id in self._backups:
                self._rollback(operation.id)

            result = RefactoringResult(
                operation_id=operation.id,
                success=False,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "error",
                "actor": "refactoring-engine",
                "data": {
                    "operation_id": operation.id,
                    "error": str(e),
                },
            })

            return result

    def _validate_operation(self, operation: RefactoringOperation) -> List[str]:
        """Validate a refactoring operation."""
        errors = []

        # Check target exists
        target_path = self.working_dir / operation.target
        if operation.refactor_type in {
            RefactoringType.RENAME,
            RefactoringType.EXTRACT_METHOD,
            RefactoringType.MOVE,
        }:
            if not target_path.exists() and not operation.options.get("symbol"):
                errors.append(f"Target file not found: {operation.target}")

        # Check scope files exist
        for scope_file in operation.scope:
            scope_path = self.working_dir / scope_file
            if not scope_path.exists():
                errors.append(f"Scope file not found: {scope_file}")

        # Check max files
        if len(operation.scope) > self.config.max_files_per_operation:
            errors.append(f"Scope exceeds maximum files: {len(operation.scope)} > {self.config.max_files_per_operation}")

        return errors

    def _create_backup(self, operation: RefactoringOperation) -> None:
        """Create backup of files that will be modified."""
        backup: Dict[str, str] = {}

        # Backup target file
        target_path = self.working_dir / operation.target
        if target_path.exists():
            backup[operation.target] = target_path.read_text()

        # Backup scope files
        for scope_file in operation.scope:
            scope_path = self.working_dir / scope_file
            if scope_path.exists():
                backup[scope_file] = scope_path.read_text()

        self._backups[operation.id] = backup

    def _rollback(self, operation_id: str) -> None:
        """Rollback changes from a failed operation."""
        backup = self._backups.get(operation_id, {})

        for file_path, content in backup.items():
            full_path = self.working_dir / file_path
            full_path.write_text(content)

        del self._backups[operation_id]

    def find_references(
        self,
        symbol: str,
        file_path: str,
        scope: Optional[List[str]] = None,
    ) -> List[SymbolReference]:
        """
        Find all references to a symbol.

        Args:
            symbol: Symbol name to find
            file_path: File containing the symbol definition
            scope: Files to search (default: all Python files)

        Returns:
            List of symbol references
        """
        references: List[SymbolReference] = []
        scope = scope or self._get_default_scope()

        # Pattern for Python identifiers
        pattern = re.compile(rf'\b{re.escape(symbol)}\b')

        for search_file in scope:
            full_path = self.working_dir / search_file
            if not full_path.exists():
                continue

            try:
                content = full_path.read_text()
                lines = content.split('\n')

                for line_num, line in enumerate(lines, 1):
                    for match in pattern.finditer(line):
                        # Determine if this is the definition
                        is_def = (
                            search_file == file_path and
                            (f"def {symbol}" in line or
                             f"class {symbol}" in line or
                             f"{symbol} =" in line)
                        )

                        references.append(SymbolReference(
                            file_path=search_file,
                            line=line_num,
                            column=match.start(),
                            context=line.strip(),
                            is_definition=is_def,
                        ))

            except (OSError, UnicodeDecodeError):
                continue

        return references

    def _get_default_scope(self) -> List[str]:
        """Get default scope (all Python files in working directory)."""
        scope = []
        for py_file in self.working_dir.rglob("*.py"):
            # Exclude hidden directories and common exclusions
            rel_path = py_file.relative_to(self.working_dir)
            if not any(part.startswith('.') for part in rel_path.parts):
                if 'venv' not in rel_path.parts and 'node_modules' not in rel_path.parts:
                    scope.append(str(rel_path))
        return scope[:self.config.max_files_per_operation]

    # =========================================================================
    # Refactoring Handlers
    # =========================================================================

    def _refactor_rename(
        self,
        operation: RefactoringOperation,
    ) -> Tuple[List[RefactoringChange], List[str]]:
        """Handle rename refactoring."""
        changes: List[RefactoringChange] = []
        warnings: List[str] = []

        old_name = operation.options.get("old_name", "")
        new_name = operation.options.get("new_name", "")

        if not old_name or not new_name:
            raise ValueError("rename requires old_name and new_name options")

        # Find all references
        references = self.find_references(
            old_name,
            operation.target,
            operation.scope or None,
        )

        # Group by file
        files_to_update: Dict[str, List[SymbolReference]] = {}
        for ref in references:
            if ref.file_path not in files_to_update:
                files_to_update[ref.file_path] = []
            files_to_update[ref.file_path].append(ref)

        # Apply renames
        pattern = re.compile(rf'\b{re.escape(old_name)}\b')

        for file_path, refs in files_to_update.items():
            full_path = self.working_dir / file_path
            old_content = full_path.read_text()

            new_content = pattern.sub(new_name, old_content)

            if old_content != new_content:
                changes.append(RefactoringChange(
                    file_path=file_path,
                    old_content=old_content,
                    new_content=new_content,
                    description=f"Renamed {old_name} to {new_name}",
                    line_changes=len(refs),
                ))

        return changes, warnings

    def _refactor_extract_method(
        self,
        operation: RefactoringOperation,
    ) -> Tuple[List[RefactoringChange], List[str]]:
        """Handle extract method refactoring."""
        changes: List[RefactoringChange] = []
        warnings: List[str] = []

        method_name = operation.options.get("method_name", "extracted_method")
        start_line = operation.options.get("start_line", 0)
        end_line = operation.options.get("end_line", 0)

        if not start_line or not end_line:
            raise ValueError("extract_method requires start_line and end_line options")

        target_path = self.working_dir / operation.target
        old_content = target_path.read_text()
        lines = old_content.split('\n')

        # Extract the code to be moved
        extracted_lines = lines[start_line - 1:end_line]
        extracted_code = '\n'.join(extracted_lines)

        # Detect indentation
        first_line = extracted_lines[0] if extracted_lines else ""
        base_indent = len(first_line) - len(first_line.lstrip())

        # Dedent extracted code for the new method
        dedented_lines = []
        for line in extracted_lines:
            if line.strip():
                dedented_lines.append(line[base_indent:] if len(line) > base_indent else line)
            else:
                dedented_lines.append("")

        # Create new method
        new_method = f'\ndef {method_name}(self):\n    """Extracted method."""\n'
        for line in dedented_lines:
            new_method += f'    {line}\n' if line.strip() else '\n'

        # Replace original code with method call
        indent = ' ' * base_indent
        method_call = f'{indent}self.{method_name}()'

        new_lines = (
            lines[:start_line - 1] +
            [method_call] +
            lines[end_line:]
        )

        # Add method at the end of the class (simplified)
        new_lines.append(new_method)

        new_content = '\n'.join(new_lines)

        changes.append(RefactoringChange(
            file_path=operation.target,
            old_content=old_content,
            new_content=new_content,
            description=f"Extracted method {method_name}",
            line_changes=end_line - start_line + 1,
        ))

        return changes, warnings

    def _refactor_extract_variable(
        self,
        operation: RefactoringOperation,
    ) -> Tuple[List[RefactoringChange], List[str]]:
        """Handle extract variable refactoring."""
        changes: List[RefactoringChange] = []
        warnings: List[str] = []

        variable_name = operation.options.get("variable_name", "extracted_var")
        expression = operation.options.get("expression", "")
        line_number = operation.options.get("line", 0)

        if not expression:
            raise ValueError("extract_variable requires expression option")

        target_path = self.working_dir / operation.target
        old_content = target_path.read_text()
        lines = old_content.split('\n')

        if line_number > 0 and line_number <= len(lines):
            target_line = lines[line_number - 1]

            # Detect indentation
            indent = len(target_line) - len(target_line.lstrip())
            indent_str = ' ' * indent

            # Replace expression with variable
            new_line = target_line.replace(expression, variable_name)

            # Insert variable assignment before the line
            assignment = f'{indent_str}{variable_name} = {expression}'

            new_lines = (
                lines[:line_number - 1] +
                [assignment, new_line] +
                lines[line_number:]
            )

            new_content = '\n'.join(new_lines)

            changes.append(RefactoringChange(
                file_path=operation.target,
                old_content=old_content,
                new_content=new_content,
                description=f"Extracted variable {variable_name}",
                line_changes=2,
            ))

        return changes, warnings

    def _refactor_inline_variable(
        self,
        operation: RefactoringOperation,
    ) -> Tuple[List[RefactoringChange], List[str]]:
        """Handle inline variable refactoring."""
        changes: List[RefactoringChange] = []
        warnings: List[str] = []

        variable_name = operation.options.get("variable_name", "")

        if not variable_name:
            raise ValueError("inline_variable requires variable_name option")

        target_path = self.working_dir / operation.target
        old_content = target_path.read_text()

        # Find variable assignment
        assign_pattern = re.compile(rf'^(\s*){re.escape(variable_name)}\s*=\s*(.+)$', re.MULTILINE)
        match = assign_pattern.search(old_content)

        if not match:
            warnings.append(f"Variable assignment not found: {variable_name}")
            return changes, warnings

        expression = match.group(2).strip()

        # Replace variable uses with expression
        use_pattern = re.compile(rf'\b{re.escape(variable_name)}\b')

        # Remove the assignment line
        new_content = assign_pattern.sub('', old_content, count=1)

        # Replace uses with expression
        new_content = use_pattern.sub(expression, new_content)

        # Clean up empty lines
        new_content = re.sub(r'\n\s*\n\s*\n', '\n\n', new_content)

        changes.append(RefactoringChange(
            file_path=operation.target,
            old_content=old_content,
            new_content=new_content,
            description=f"Inlined variable {variable_name}",
            line_changes=1,
        ))

        return changes, warnings

    def _refactor_move(
        self,
        operation: RefactoringOperation,
    ) -> Tuple[List[RefactoringChange], List[str]]:
        """Handle move refactoring."""
        changes: List[RefactoringChange] = []
        warnings: List[str] = []

        destination = operation.options.get("destination", "")
        symbol = operation.options.get("symbol", "")

        if not destination:
            raise ValueError("move requires destination option")

        source_path = self.working_dir / operation.target
        dest_path = self.working_dir / destination

        source_content = source_path.read_text()
        dest_content = dest_path.read_text() if dest_path.exists() else ""

        if symbol:
            # Move specific symbol
            # Find and extract the symbol definition (simplified)
            pattern = re.compile(
                rf'^(def {symbol}|class {symbol}|{symbol}\s*=).*?(?=\n(?:def |class |\Z))',
                re.MULTILINE | re.DOTALL
            )

            match = pattern.search(source_content)
            if not match:
                warnings.append(f"Symbol not found: {symbol}")
                return changes, warnings

            moved_code = match.group(0)

            # Remove from source
            new_source = pattern.sub('', source_content)

            # Add to destination
            new_dest = dest_content + '\n\n' + moved_code

            changes.append(RefactoringChange(
                file_path=operation.target,
                old_content=source_content,
                new_content=new_source,
                description=f"Removed {symbol}",
            ))

            changes.append(RefactoringChange(
                file_path=destination,
                old_content=dest_content,
                new_content=new_dest,
                description=f"Added {symbol}",
            ))

        else:
            # Move entire file content
            new_dest = dest_content + '\n\n' + source_content

            changes.append(RefactoringChange(
                file_path=operation.target,
                old_content=source_content,
                new_content="# Moved to " + destination,
                description="Moved content",
            ))

            changes.append(RefactoringChange(
                file_path=destination,
                old_content=dest_content,
                new_content=new_dest,
                description="Received moved content",
            ))

        return changes, warnings

    def _refactor_convert_to_async(
        self,
        operation: RefactoringOperation,
    ) -> Tuple[List[RefactoringChange], List[str]]:
        """Handle convert to async refactoring."""
        changes: List[RefactoringChange] = []
        warnings: List[str] = []

        function_name = operation.options.get("function_name", "")

        if not function_name:
            raise ValueError("convert_to_async requires function_name option")

        target_path = self.working_dir / operation.target
        old_content = target_path.read_text()

        # Convert def to async def
        def_pattern = re.compile(rf'^(\s*)def ({re.escape(function_name)})\s*\(', re.MULTILINE)
        new_content = def_pattern.sub(r'\1async def \2(', old_content)

        if old_content == new_content:
            warnings.append(f"Function not found or already async: {function_name}")
            return changes, warnings

        changes.append(RefactoringChange(
            file_path=operation.target,
            old_content=old_content,
            new_content=new_content,
            description=f"Converted {function_name} to async",
            line_changes=1,
        ))

        return changes, warnings

    def _refactor_change_signature(
        self,
        operation: RefactoringOperation,
    ) -> Tuple[List[RefactoringChange], List[str]]:
        """Handle change signature refactoring."""
        changes: List[RefactoringChange] = []
        warnings: List[str] = []

        function_name = operation.options.get("function_name", "")
        new_params = operation.options.get("new_params", "")

        if not function_name:
            raise ValueError("change_signature requires function_name option")

        target_path = self.working_dir / operation.target
        old_content = target_path.read_text()

        # Find function definition
        def_pattern = re.compile(
            rf'^(\s*(?:async\s+)?def {re.escape(function_name)})\s*\([^)]*\)',
            re.MULTILINE
        )

        new_content = def_pattern.sub(rf'\1({new_params})', old_content)

        if old_content == new_content:
            warnings.append(f"Function not found: {function_name}")
            return changes, warnings

        changes.append(RefactoringChange(
            file_path=operation.target,
            old_content=old_content,
            new_content=new_content,
            description=f"Changed signature of {function_name}",
            line_changes=1,
        ))

        return changes, warnings

    def get_stats(self) -> Dict[str, Any]:
        """Get refactoring engine statistics."""
        return {
            "supported_types": [t.value for t in self._handlers.keys()],
            "active_backups": len(self._backups),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Refactoring Engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Refactoring Engine (Step 74)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # rename command
    rename_parser = subparsers.add_parser("rename", help="Rename a symbol")
    rename_parser.add_argument("file", help="Target file")
    rename_parser.add_argument("old_name", help="Old symbol name")
    rename_parser.add_argument("new_name", help="New symbol name")
    rename_parser.add_argument("--scope", nargs="+", help="Files to search")
    rename_parser.add_argument("--json", action="store_true", help="JSON output")

    # extract-method command
    extract_parser = subparsers.add_parser("extract-method", help="Extract code to a method")
    extract_parser.add_argument("file", help="Target file")
    extract_parser.add_argument("--name", "-n", required=True, help="New method name")
    extract_parser.add_argument("--start", type=int, required=True, help="Start line")
    extract_parser.add_argument("--end", type=int, required=True, help="End line")
    extract_parser.add_argument("--json", action="store_true", help="JSON output")

    # find-references command
    refs_parser = subparsers.add_parser("find-refs", help="Find references to a symbol")
    refs_parser.add_argument("file", help="File containing symbol")
    refs_parser.add_argument("symbol", help="Symbol name")
    refs_parser.add_argument("--scope", nargs="+", help="Files to search")
    refs_parser.add_argument("--json", action="store_true", help="JSON output")

    # types command
    subparsers.add_parser("types", help="List supported refactoring types")

    # stats command
    subparsers.add_parser("stats", help="Show engine stats")

    args = parser.parse_args()

    engine = RefactoringEngine()

    if args.command == "rename":
        operation = RefactoringOperation(
            id=f"refactor-{uuid.uuid4().hex[:8]}",
            refactor_type=RefactoringType.RENAME,
            target=args.file,
            options={"old_name": args.old_name, "new_name": args.new_name},
            scope=args.scope or [],
        )

        result = engine.refactor(operation)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(f"Renamed {args.old_name} to {args.new_name}")
                print(f"Modified {result.files_modified} files")
                for change in result.changes:
                    print(f"  - {change.file_path}: {change.description}")
            else:
                print(f"Error: {result.error}")
                return 1
        return 0

    elif args.command == "extract-method":
        operation = RefactoringOperation(
            id=f"refactor-{uuid.uuid4().hex[:8]}",
            refactor_type=RefactoringType.EXTRACT_METHOD,
            target=args.file,
            options={
                "method_name": args.name,
                "start_line": args.start,
                "end_line": args.end,
            },
        )

        result = engine.refactor(operation)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(f"Extracted method: {args.name}")
            else:
                print(f"Error: {result.error}")
                return 1
        return 0

    elif args.command == "find-refs":
        refs = engine.find_references(args.symbol, args.file, args.scope)

        if args.json:
            print(json.dumps([r.to_dict() for r in refs], indent=2))
        else:
            print(f"Found {len(refs)} references to {args.symbol}:")
            for ref in refs:
                marker = " (definition)" if ref.is_definition else ""
                print(f"  {ref.file_path}:{ref.line}:{ref.column}{marker}")
                print(f"    {ref.context}")
        return 0

    elif args.command == "types":
        print("Supported refactoring types:")
        for rt in RefactoringType:
            print(f"  {rt.value}")
        return 0

    elif args.command == "stats":
        stats = engine.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
