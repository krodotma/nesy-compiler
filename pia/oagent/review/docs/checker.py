#!/usr/bin/env python3
"""
Documentation Completeness Checker (Step 156)

Checks for documentation completeness in code.

PBTSO Phase: VERIFY
Bus Topics: review.docs.check, review.docs.missing

Checks:
- Module docstrings
- Class docstrings
- Function/method docstrings
- Parameter documentation
- Return value documentation
- Type hints presence
- Example code in docstrings

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Types
# ============================================================================

class DocIssueType(Enum):
    """Types of documentation issues."""
    MISSING_MODULE_DOC = "missing_module_doc"
    MISSING_CLASS_DOC = "missing_class_doc"
    MISSING_FUNCTION_DOC = "missing_function_doc"
    MISSING_PARAM_DOC = "missing_param_doc"
    MISSING_RETURN_DOC = "missing_return_doc"
    MISSING_RAISES_DOC = "missing_raises_doc"
    MISSING_TYPE_HINTS = "missing_type_hints"
    INCOMPLETE_DOC = "incomplete_doc"
    OUTDATED_DOC = "outdated_doc"
    MISSING_EXAMPLE = "missing_example"


class DocSeverity(Enum):
    """Severity of documentation issues."""
    ERROR = "error"    # Must be fixed
    WARNING = "warning"  # Should be fixed
    INFO = "info"      # Nice to have


@dataclass
class DocIssue:
    """
    Represents a documentation issue.

    Attributes:
        issue_type: Type of documentation issue
        severity: Issue severity
        file: File path
        line: Line number
        name: Name of the element with missing docs
        element_type: Type of element (module, class, function, method)
        description: Human-readable description
        suggestion: How to fix the issue
        missing_params: List of undocumented parameters
    """
    issue_type: DocIssueType
    severity: DocSeverity
    file: str
    line: int
    name: str
    element_type: str
    description: str
    suggestion: str
    missing_params: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["issue_type"] = self.issue_type.value
        result["severity"] = self.severity.value
        return result


@dataclass
class DocCheckResult:
    """Result from documentation checking."""
    files_checked: int = 0
    issues: List[DocIssue] = field(default_factory=list)
    duration_ms: float = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    coverage_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files_checked": self.files_checked,
            "issues": [i.to_dict() for i in self.issues],
            "duration_ms": self.duration_ms,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "coverage_stats": self.coverage_stats,
        }


# ============================================================================
# Docstring Parser
# ============================================================================

class DocstringParser:
    """Parse docstrings to extract documented parameters, returns, etc."""

    # Common docstring styles
    PARAM_PATTERNS = [
        # Google style: Args:
        r":param\s+(\w+):",  # Sphinx
        r"Args:\s*\n(?:\s+(\w+)[:\s].*\n)*",  # Google
        r"Parameters\s*\n-+\s*\n(?:\s*(\w+)\s*:.*\n)*",  # NumPy
    ]

    RETURN_PATTERNS = [
        r":returns?:",  # Sphinx
        r"Returns:\s*\n",  # Google
        r"Returns\s*\n-+",  # NumPy
    ]

    RAISES_PATTERNS = [
        r":raises?\s+\w+:",  # Sphinx
        r"Raises:\s*\n",  # Google
        r"Raises\s*\n-+",  # NumPy
    ]

    EXAMPLE_PATTERNS = [
        r"Examples?:\s*\n",  # Google
        r">>>",  # Doctest
        r"```",  # Markdown
    ]

    @classmethod
    def parse(cls, docstring: Optional[str]) -> Dict[str, Any]:
        """
        Parse a docstring.

        Returns:
            Dictionary with:
            - has_description: bool
            - documented_params: List[str]
            - has_return: bool
            - has_raises: bool
            - has_example: bool
        """
        if not docstring:
            return {
                "has_description": False,
                "documented_params": [],
                "has_return": False,
                "has_raises": False,
                "has_example": False,
            }

        # Extract documented parameters
        documented_params = []
        # Sphinx style
        for match in re.finditer(r":param\s+(\w+):", docstring):
            documented_params.append(match.group(1))
        # Google style
        args_match = re.search(r"Args:\s*\n((?:\s+\w+.*\n)*)", docstring)
        if args_match:
            for line in args_match.group(1).split("\n"):
                param_match = re.match(r"\s+(\w+)[:\s]", line)
                if param_match:
                    documented_params.append(param_match.group(1))

        return {
            "has_description": len(docstring.strip()) > 10,
            "documented_params": documented_params,
            "has_return": any(re.search(p, docstring) for p in cls.RETURN_PATTERNS),
            "has_raises": any(re.search(p, docstring) for p in cls.RAISES_PATTERNS),
            "has_example": any(re.search(p, docstring) for p in cls.EXAMPLE_PATTERNS),
        }


# ============================================================================
# Documentation Checker
# ============================================================================

class DocChecker:
    """
    Checks for documentation completeness.

    Analyzes Python code for missing or incomplete documentation.

    Example:
        checker = DocChecker()
        result = checker.check(["/path/to/file.py"])
        for issue in result.issues:
            print(f"{issue.file}:{issue.line}: {issue.description}")
    """

    def __init__(
        self,
        require_module_doc: bool = True,
        require_class_doc: bool = True,
        require_function_doc: bool = True,
        require_param_doc: bool = True,
        require_return_doc: bool = True,
        require_type_hints: bool = False,
        min_public_methods: int = 0,
        skip_private: bool = True,
        skip_magic: bool = True,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the documentation checker.

        Args:
            require_module_doc: Require module-level docstrings
            require_class_doc: Require class docstrings
            require_function_doc: Require function/method docstrings
            require_param_doc: Require parameter documentation
            require_return_doc: Require return value documentation
            require_type_hints: Require type hints
            min_public_methods: Minimum public methods to require class doc
            skip_private: Skip private methods (_name)
            skip_magic: Skip magic methods (__name__)
            bus_path: Path to event bus file
        """
        self.require_module_doc = require_module_doc
        self.require_class_doc = require_class_doc
        self.require_function_doc = require_function_doc
        self.require_param_doc = require_param_doc
        self.require_return_doc = require_return_doc
        self.require_type_hints = require_type_hints
        self.min_public_methods = min_public_methods
        self.skip_private = skip_private
        self.skip_magic = skip_magic
        self.bus_path = bus_path or self._get_bus_path()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "docs",
            "actor": "doc-checker",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _should_skip(self, name: str) -> bool:
        """Check if element should be skipped based on name."""
        if self.skip_magic and name.startswith("__") and name.endswith("__"):
            return True
        if self.skip_private and name.startswith("_") and not name.startswith("__"):
            return True
        return False

    def check(
        self,
        files: List[str],
        content_map: Optional[Dict[str, str]] = None,
    ) -> DocCheckResult:
        """
        Check files for documentation completeness.

        Args:
            files: List of file paths to check
            content_map: Optional pre-loaded file contents

        Returns:
            DocCheckResult with all issues found

        Emits:
            review.docs.check (start)
            review.docs.missing (per issue batch)
        """
        start_time = time.time()

        # Emit start event
        self._emit_event("review.docs.check", {
            "files": files[:20],
            "file_count": len(files),
            "status": "started",
        })

        result = DocCheckResult(
            files_checked=len(files),
        )

        # Coverage statistics
        stats = {
            "modules_total": 0,
            "modules_documented": 0,
            "classes_total": 0,
            "classes_documented": 0,
            "functions_total": 0,
            "functions_documented": 0,
        }

        python_files = [f for f in files if f.endswith(".py")]

        for file_path in python_files:
            # Get file content
            if content_map and file_path in content_map:
                content = content_map[file_path]
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (IOError, OSError):
                    continue

            # Check documentation
            issues, file_stats = self._check_file(file_path, content)
            result.issues.extend(issues)

            # Update stats
            for key, value in file_stats.items():
                stats[key] = stats.get(key, 0) + value

        # Calculate coverage percentages
        result.coverage_stats = {
            "module_coverage": (
                stats["modules_documented"] / stats["modules_total"] * 100
                if stats["modules_total"] > 0 else 100
            ),
            "class_coverage": (
                stats["classes_documented"] / stats["classes_total"] * 100
                if stats["classes_total"] > 0 else 100
            ),
            "function_coverage": (
                stats["functions_documented"] / stats["functions_total"] * 100
                if stats["functions_total"] > 0 else 100
            ),
            **stats,
        }

        # Calculate severity counts
        for issue in result.issues:
            if issue.severity == DocSeverity.ERROR:
                result.error_count += 1
            elif issue.severity == DocSeverity.WARNING:
                result.warning_count += 1
            else:
                result.info_count += 1

        result.duration_ms = (time.time() - start_time) * 1000

        # Emit issues found
        if result.issues:
            self._emit_event("review.docs.missing", {
                "issue_count": len(result.issues),
                "error_count": result.error_count,
                "coverage": result.coverage_stats,
                "issues": [i.to_dict() for i in result.issues[:10]],
            })

        # Emit completion
        self._emit_event("review.docs.check", {
            "status": "completed",
            "files_checked": result.files_checked,
            "issue_count": len(result.issues),
            "coverage": result.coverage_stats,
            "duration_ms": result.duration_ms,
        })

        return result

    def _check_file(
        self,
        file_path: str,
        content: str,
    ) -> Tuple[List[DocIssue], Dict[str, int]]:
        """Check a single file for documentation issues."""
        issues = []
        stats = {
            "modules_total": 1,
            "modules_documented": 0,
            "classes_total": 0,
            "classes_documented": 0,
            "functions_total": 0,
            "functions_documented": 0,
        }

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues, stats

        # Check module docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            stats["modules_documented"] = 1
        elif self.require_module_doc:
            issues.append(DocIssue(
                issue_type=DocIssueType.MISSING_MODULE_DOC,
                severity=DocSeverity.WARNING,
                file=file_path,
                line=1,
                name=Path(file_path).name,
                element_type="module",
                description=f"Module '{Path(file_path).name}' is missing a docstring",
                suggestion="Add a module-level docstring describing the module's purpose",
            ))

        # Check classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_issues, class_stats = self._check_class(file_path, node)
                issues.extend(class_issues)
                stats["classes_total"] += 1
                if ast.get_docstring(node):
                    stats["classes_documented"] += 1
                stats["functions_total"] += class_stats["functions_total"]
                stats["functions_documented"] += class_stats["functions_documented"]

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip methods (handled in class check)
                if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree)):
                    func_issues = self._check_function(file_path, node, is_method=False)
                    issues.extend(func_issues)
                    stats["functions_total"] += 1
                    if ast.get_docstring(node):
                        stats["functions_documented"] += 1

        return issues, stats

    def _check_class(
        self,
        file_path: str,
        node: ast.ClassDef,
    ) -> Tuple[List[DocIssue], Dict[str, int]]:
        """Check a class for documentation issues."""
        issues = []
        stats = {"functions_total": 0, "functions_documented": 0}

        class_name = node.name
        class_doc = ast.get_docstring(node)

        # Count public methods
        public_methods = [
            n for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not n.name.startswith("_")
        ]

        # Check class docstring
        if not class_doc and self.require_class_doc:
            if len(public_methods) >= self.min_public_methods:
                issues.append(DocIssue(
                    issue_type=DocIssueType.MISSING_CLASS_DOC,
                    severity=DocSeverity.WARNING,
                    file=file_path,
                    line=node.lineno,
                    name=class_name,
                    element_type="class",
                    description=f"Class '{class_name}' is missing a docstring",
                    suggestion="Add a class docstring describing its purpose and usage",
                ))

        # Check methods
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self._should_skip(child.name):
                    method_issues = self._check_function(
                        file_path, child, is_method=True, class_name=class_name
                    )
                    issues.extend(method_issues)
                    stats["functions_total"] += 1
                    if ast.get_docstring(child):
                        stats["functions_documented"] += 1

        return issues, stats

    def _check_function(
        self,
        file_path: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_method: bool = False,
        class_name: str = "",
    ) -> List[DocIssue]:
        """Check a function/method for documentation issues."""
        issues = []
        func_name = f"{class_name}.{node.name}" if class_name else node.name
        element_type = "method" if is_method else "function"

        # Get docstring and parse it
        docstring = ast.get_docstring(node)
        doc_info = DocstringParser.parse(docstring)

        # Check for missing docstring
        if not docstring and self.require_function_doc:
            issues.append(DocIssue(
                issue_type=DocIssueType.MISSING_FUNCTION_DOC,
                severity=DocSeverity.WARNING,
                file=file_path,
                line=node.lineno,
                name=func_name,
                element_type=element_type,
                description=f"{element_type.capitalize()} '{func_name}' is missing a docstring",
                suggestion=f"Add a docstring describing what this {element_type} does",
            ))
            return issues  # No point checking params if no docstring

        # Get function parameters
        params = []
        for arg in node.args.args:
            if arg.arg not in ("self", "cls"):
                params.append(arg.arg)
        for arg in node.args.kwonlyargs:
            params.append(arg.arg)

        # Check parameter documentation
        if self.require_param_doc and params:
            documented = set(doc_info["documented_params"])
            missing = [p for p in params if p not in documented]

            if missing:
                issues.append(DocIssue(
                    issue_type=DocIssueType.MISSING_PARAM_DOC,
                    severity=DocSeverity.INFO,
                    file=file_path,
                    line=node.lineno,
                    name=func_name,
                    element_type=element_type,
                    description=f"{element_type.capitalize()} '{func_name}' has undocumented parameters",
                    suggestion=f"Document parameters: {', '.join(missing)}",
                    missing_params=missing,
                ))

        # Check return documentation
        if self.require_return_doc:
            # Check if function has a return statement with a value
            has_return_value = any(
                isinstance(n, ast.Return) and n.value is not None
                for n in ast.walk(node)
            )

            if has_return_value and not doc_info["has_return"]:
                issues.append(DocIssue(
                    issue_type=DocIssueType.MISSING_RETURN_DOC,
                    severity=DocSeverity.INFO,
                    file=file_path,
                    line=node.lineno,
                    name=func_name,
                    element_type=element_type,
                    description=f"{element_type.capitalize()} '{func_name}' is missing return value documentation",
                    suggestion="Add a 'Returns:' section to the docstring",
                ))

        # Check type hints
        if self.require_type_hints:
            missing_hints = []

            # Check parameter type hints
            for arg in node.args.args:
                if arg.arg not in ("self", "cls") and arg.annotation is None:
                    missing_hints.append(arg.arg)

            # Check return type hint
            if node.returns is None:
                missing_hints.append("return")

            if missing_hints:
                issues.append(DocIssue(
                    issue_type=DocIssueType.MISSING_TYPE_HINTS,
                    severity=DocSeverity.INFO,
                    file=file_path,
                    line=node.lineno,
                    name=func_name,
                    element_type=element_type,
                    description=f"{element_type.capitalize()} '{func_name}' is missing type hints",
                    suggestion=f"Add type hints for: {', '.join(missing_hints)}",
                ))

        return issues


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Documentation Checker."""
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Completeness Checker (Step 156)")
    parser.add_argument("files", nargs="+", help="Files to check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode (type hints required)")
    parser.add_argument("--include-private", action="store_true", help="Check private methods")

    args = parser.parse_args()

    checker = DocChecker(
        require_type_hints=args.strict,
        skip_private=not args.include_private,
    )
    result = checker.check(args.files)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.summary:
        print(f"Documentation Check Summary:")
        print(f"  Files checked: {result.files_checked}")
        print(f"  Issues: {len(result.issues)}")
        print(f"  Errors: {result.error_count}")
        print(f"  Warnings: {result.warning_count}")
        print(f"  Info: {result.info_count}")
        print(f"  Duration: {result.duration_ms:.1f}ms")
        print(f"\nCoverage:")
        print(f"  Module: {result.coverage_stats.get('module_coverage', 0):.1f}%")
        print(f"  Class: {result.coverage_stats.get('class_coverage', 0):.1f}%")
        print(f"  Function: {result.coverage_stats.get('function_coverage', 0):.1f}%")
    else:
        for issue in result.issues:
            severity_color = {
                DocSeverity.ERROR: "\033[91m",
                DocSeverity.WARNING: "\033[93m",
                DocSeverity.INFO: "\033[90m",
            }.get(issue.severity, "")
            reset = "\033[0m"

            print(f"{severity_color}[{issue.severity.value.upper()}]{reset} {issue.file}:{issue.line}")
            print(f"  {issue.description}")
            print(f"  Suggestion: {issue.suggestion}")
            if issue.missing_params:
                print(f"  Missing: {', '.join(issue.missing_params)}")
            print()

    return 1 if result.error_count > 0 else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
