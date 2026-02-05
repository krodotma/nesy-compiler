#!/usr/bin/env python3
"""
enforcer.py - Code Style Enforcer (Step 59)

PBTSO Phase: VERIFY

Provides:
- Code formatting using external formatters (black, prettier)
- Linting using external linters (ruff, eslint)
- Style issue detection and auto-fixing
- Language-specific formatting rules

Bus Topics:
- code.style.check
- code.style.fixed
- code.lint.issues

Protocol: DKIN v30
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Types
# =============================================================================

class IssueSeverity(Enum):
    """Severity of a style/lint issue."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class StyleIssue:
    """Represents a style or lint issue."""
    file: str
    line: int
    column: int
    message: str
    rule: str
    severity: IssueSeverity
    fixable: bool = False
    fix_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "rule": self.rule,
            "severity": self.severity.value,
            "fixable": self.fixable,
            "fix_applied": self.fix_applied,
        }


@dataclass
class StyleResult:
    """Result of a style check or format operation."""
    success: bool
    files_checked: int
    files_modified: int
    issues: List[StyleIssue] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "files_checked": self.files_checked,
            "files_modified": self.files_modified,
            "issues": [i.to_dict() for i in self.issues],
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


# =============================================================================
# Code Style Enforcer
# =============================================================================

class CodeStyleEnforcer:
    """
    Enforce code style using external formatters and linters.

    PBTSO Phase: VERIFY

    Supported formatters:
    - Python: black, isort
    - TypeScript/JavaScript: prettier
    - Go: gofmt
    - Rust: rustfmt

    Supported linters:
    - Python: ruff, flake8
    - TypeScript/JavaScript: eslint
    - Go: golint
    - Rust: clippy
    """

    FORMATTERS: Dict[str, List[str]] = {
        ".py": ["black", "--line-length", "100", "-q"],
        ".ts": ["npx", "prettier", "--write"],
        ".tsx": ["npx", "prettier", "--write"],
        ".js": ["npx", "prettier", "--write"],
        ".jsx": ["npx", "prettier", "--write"],
        ".json": ["npx", "prettier", "--write"],
        ".md": ["npx", "prettier", "--write"],
        ".go": ["gofmt", "-w"],
        ".rs": ["rustfmt"],
    }

    LINTERS: Dict[str, Tuple[List[str], str]] = {
        # (command, output format)
        ".py": (["ruff", "check", "--output-format", "json"], "ruff"),
        ".ts": (["npx", "eslint", "--format", "json"], "eslint"),
        ".tsx": (["npx", "eslint", "--format", "json"], "eslint"),
        ".js": (["npx", "eslint", "--format", "json"], "eslint"),
        ".jsx": (["npx", "eslint", "--format", "json"], "eslint"),
    }

    IMPORT_SORTERS: Dict[str, List[str]] = {
        ".py": ["isort", "--profile", "black"],
        ".ts": ["npx", "prettier", "--write"],  # Prettier handles import sorting
    }

    def __init__(
        self,
        bus: Optional[Any] = None,
        timeout_s: int = 30,
        check_only: bool = False,
    ):
        self.bus = bus
        self.timeout_s = timeout_s
        self.check_only = check_only
        self._tool_availability: Dict[str, bool] = {}

    # =========================================================================
    # Format Operations
    # =========================================================================

    def format_file(self, path: Path) -> StyleResult:
        """
        Format a single file.

        Args:
            path: Path to file

        Returns:
            StyleResult with format outcome
        """
        start_time = time.time()

        formatter = self.FORMATTERS.get(path.suffix)
        if not formatter:
            return StyleResult(
                success=True,
                files_checked=1,
                files_modified=0,
                error=f"No formatter for {path.suffix}",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # Check if tool is available
        tool = formatter[0]
        if not self._check_tool(tool):
            return StyleResult(
                success=False,
                files_checked=1,
                files_modified=0,
                error=f"Formatter not available: {tool}",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # Get file content hash before
        original_content = path.read_text() if path.exists() else ""

        try:
            if self.check_only:
                # Check mode - add --check flag for black
                cmd = list(formatter)
                if tool == "black":
                    cmd.append("--check")
                cmd.append(str(path))
            else:
                cmd = formatter + [str(path)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )

            # Check if file was modified
            new_content = path.read_text() if path.exists() else ""
            modified = original_content != new_content

            success = result.returncode == 0 or (self.check_only and result.returncode == 1)

            style_result = StyleResult(
                success=success,
                files_checked=1,
                files_modified=1 if modified else 0,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

            # Emit event
            if self.bus and modified:
                self.bus.emit({
                    "topic": "code.style.fixed",
                    "kind": "style",
                    "actor": "code-agent",
                    "data": {
                        "file": str(path),
                        "formatter": tool,
                        "modified": modified,
                    },
                })

            return style_result

        except subprocess.TimeoutExpired:
            return StyleResult(
                success=False,
                files_checked=1,
                files_modified=0,
                error=f"Formatter timed out after {self.timeout_s}s",
                elapsed_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return StyleResult(
                success=False,
                files_checked=1,
                files_modified=0,
                error=str(e),
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    def format_files(self, paths: List[Path]) -> StyleResult:
        """Format multiple files."""
        start_time = time.time()
        total_checked = 0
        total_modified = 0
        all_issues: List[StyleIssue] = []
        errors = []

        for path in paths:
            result = self.format_file(path)
            total_checked += result.files_checked
            total_modified += result.files_modified
            all_issues.extend(result.issues)
            if result.error:
                errors.append(f"{path}: {result.error}")

        return StyleResult(
            success=len(errors) == 0,
            files_checked=total_checked,
            files_modified=total_modified,
            issues=all_issues,
            error="; ".join(errors) if errors else None,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def format_directory(
        self,
        directory: Path,
        extensions: Optional[List[str]] = None,
    ) -> StyleResult:
        """Format all files in a directory."""
        extensions = extensions or list(self.FORMATTERS.keys())

        files = []
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))

        # Filter out hidden/venv directories
        files = [
            f for f in files
            if not any(part.startswith(".") for part in f.parts)
            and "venv" not in f.parts
            and "node_modules" not in f.parts
        ]

        return self.format_files(files)

    # =========================================================================
    # Lint Operations
    # =========================================================================

    def lint_file(self, path: Path) -> List[StyleIssue]:
        """
        Lint a single file.

        Args:
            path: Path to file

        Returns:
            List of StyleIssues found
        """
        linter_config = self.LINTERS.get(path.suffix)
        if not linter_config:
            return []

        linter_cmd, output_format = linter_config

        # Check if tool is available
        tool = linter_cmd[0]
        if not self._check_tool(tool):
            return []

        try:
            cmd = linter_cmd + [str(path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )

            issues = self._parse_lint_output(result.stdout, output_format, path)

            # Emit event
            if self.bus and issues:
                self.bus.emit({
                    "topic": "code.lint.issues",
                    "kind": "style",
                    "actor": "code-agent",
                    "data": {
                        "file": str(path),
                        "linter": tool,
                        "issue_count": len(issues),
                        "issues": [i.to_dict() for i in issues[:10]],  # Limit to 10
                    },
                })

            return issues

        except subprocess.TimeoutExpired:
            return []
        except Exception:
            return []

    def _parse_lint_output(
        self,
        output: str,
        format_type: str,
        path: Path,
    ) -> List[StyleIssue]:
        """Parse linter output to StyleIssues."""
        issues = []

        if not output.strip():
            return issues

        if format_type == "ruff":
            try:
                data = json.loads(output)
                for item in data:
                    issues.append(StyleIssue(
                        file=item.get("filename", str(path)),
                        line=item.get("location", {}).get("row", 1),
                        column=item.get("location", {}).get("column", 1),
                        message=item.get("message", ""),
                        rule=item.get("code", ""),
                        severity=IssueSeverity.ERROR if item.get("code", "").startswith("E") else IssueSeverity.WARNING,
                        fixable=item.get("fix") is not None,
                    ))
            except json.JSONDecodeError:
                pass

        elif format_type == "eslint":
            try:
                data = json.loads(output)
                for file_result in data:
                    for msg in file_result.get("messages", []):
                        issues.append(StyleIssue(
                            file=file_result.get("filePath", str(path)),
                            line=msg.get("line", 1),
                            column=msg.get("column", 1),
                            message=msg.get("message", ""),
                            rule=msg.get("ruleId", ""),
                            severity=IssueSeverity.ERROR if msg.get("severity", 1) == 2 else IssueSeverity.WARNING,
                            fixable=msg.get("fix") is not None,
                        ))
            except json.JSONDecodeError:
                pass

        return issues

    def lint_files(self, paths: List[Path]) -> List[StyleIssue]:
        """Lint multiple files."""
        all_issues = []
        for path in paths:
            all_issues.extend(self.lint_file(path))
        return all_issues

    def lint_and_fix(self, path: Path) -> StyleResult:
        """
        Lint a file and apply auto-fixes.

        Args:
            path: Path to file

        Returns:
            StyleResult with fix outcome
        """
        start_time = time.time()

        # Get linter config
        linter_config = self.LINTERS.get(path.suffix)
        if not linter_config:
            return StyleResult(
                success=True,
                files_checked=1,
                files_modified=0,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        linter_cmd, output_format = linter_config
        tool = linter_cmd[0]

        # Build fix command
        if tool == "ruff":
            fix_cmd = ["ruff", "check", "--fix", str(path)]
        elif tool == "npx":  # eslint
            fix_cmd = ["npx", "eslint", "--fix", str(path)]
        else:
            return StyleResult(
                success=True,
                files_checked=1,
                files_modified=0,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        original_content = path.read_text()

        try:
            result = subprocess.run(
                fix_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )

            new_content = path.read_text()
            modified = original_content != new_content

            # Get remaining issues
            remaining_issues = self.lint_file(path)

            return StyleResult(
                success=True,
                files_checked=1,
                files_modified=1 if modified else 0,
                issues=remaining_issues,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return StyleResult(
                success=False,
                files_checked=1,
                files_modified=0,
                error=str(e),
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    # =========================================================================
    # Import Sorting
    # =========================================================================

    def sort_imports(self, path: Path) -> StyleResult:
        """Sort imports in a file."""
        start_time = time.time()

        sorter = self.IMPORT_SORTERS.get(path.suffix)
        if not sorter:
            return StyleResult(
                success=True,
                files_checked=1,
                files_modified=0,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        tool = sorter[0]
        if not self._check_tool(tool):
            return StyleResult(
                success=False,
                files_checked=1,
                files_modified=0,
                error=f"Import sorter not available: {tool}",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        original_content = path.read_text()

        try:
            cmd = sorter + [str(path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )

            new_content = path.read_text()
            modified = original_content != new_content

            return StyleResult(
                success=result.returncode == 0,
                files_checked=1,
                files_modified=1 if modified else 0,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return StyleResult(
                success=False,
                files_checked=1,
                files_modified=0,
                error=str(e),
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    # =========================================================================
    # Combined Check
    # =========================================================================

    def check_file(self, path: Path) -> StyleResult:
        """
        Run all style checks on a file.

        Args:
            path: Path to file

        Returns:
            Combined StyleResult
        """
        start_time = time.time()
        all_issues: List[StyleIssue] = []

        # Check formatting
        format_result = self.format_file(path)

        # Lint
        lint_issues = self.lint_file(path)
        all_issues.extend(lint_issues)

        return StyleResult(
            success=format_result.success and len(lint_issues) == 0,
            files_checked=1,
            files_modified=format_result.files_modified,
            issues=all_issues,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def fix_all(self, path: Path) -> StyleResult:
        """
        Run all formatters and auto-fixes on a file.

        Args:
            path: Path to file

        Returns:
            Combined StyleResult
        """
        start_time = time.time()
        total_modified = 0
        all_issues: List[StyleIssue] = []

        # Sort imports first
        import_result = self.sort_imports(path)
        total_modified += import_result.files_modified

        # Format
        format_result = self.format_file(path)
        total_modified += format_result.files_modified

        # Fix lint issues
        lint_result = self.lint_and_fix(path)
        total_modified += lint_result.files_modified
        all_issues.extend(lint_result.issues)

        return StyleResult(
            success=True,
            files_checked=1,
            files_modified=min(total_modified, 1),  # Cap at 1
            issues=all_issues,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    # =========================================================================
    # Utilities
    # =========================================================================

    def _check_tool(self, tool: str) -> bool:
        """Check if a tool is available."""
        if tool in self._tool_availability:
            return self._tool_availability[tool]

        try:
            if tool == "npx":
                result = subprocess.run(
                    ["npx", "--version"],
                    capture_output=True,
                    timeout=5,
                )
            else:
                result = subprocess.run(
                    [tool, "--version"],
                    capture_output=True,
                    timeout=5,
                )
            available = result.returncode == 0
        except Exception:
            available = False

        self._tool_availability[tool] = available
        return available

    def get_available_tools(self) -> Dict[str, bool]:
        """Get availability of all tools."""
        tools = set()

        for formatter in self.FORMATTERS.values():
            tools.add(formatter[0])

        for linter_cmd, _ in self.LINTERS.values():
            tools.add(linter_cmd[0])

        for sorter in self.IMPORT_SORTERS.values():
            tools.add(sorter[0])

        return {tool: self._check_tool(tool) for tool in tools}


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Code Style Enforcer."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Style Enforcer (Step 59)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # format command
    format_parser = subparsers.add_parser("format", help="Format files")
    format_parser.add_argument("paths", nargs="+", help="Files or directories to format")
    format_parser.add_argument("--check", action="store_true", help="Check only, don't modify")

    # lint command
    lint_parser = subparsers.add_parser("lint", help="Lint files")
    lint_parser.add_argument("paths", nargs="+", help="Files to lint")
    lint_parser.add_argument("--fix", action="store_true", help="Apply auto-fixes")

    # check command
    check_parser = subparsers.add_parser("check", help="Run all checks")
    check_parser.add_argument("paths", nargs="+", help="Files to check")

    # fix command
    fix_parser = subparsers.add_parser("fix", help="Apply all fixes")
    fix_parser.add_argument("paths", nargs="+", help="Files to fix")

    # tools command
    subparsers.add_parser("tools", help="Show available tools")

    args = parser.parse_args()

    enforcer = CodeStyleEnforcer(check_only=getattr(args, "check", False))

    if args.command == "format":
        for path_str in args.paths:
            path = Path(path_str)
            if path.is_dir():
                result = enforcer.format_directory(path)
            else:
                result = enforcer.format_file(path)
            print(json.dumps(result.to_dict(), indent=2))
        return 0

    elif args.command == "lint":
        for path_str in args.paths:
            path = Path(path_str)
            if args.fix:
                result = enforcer.lint_and_fix(path)
                print(json.dumps(result.to_dict(), indent=2))
            else:
                issues = enforcer.lint_file(path)
                for issue in issues:
                    print(f"{issue.file}:{issue.line}:{issue.column} [{issue.rule}] {issue.message}")
        return 0

    elif args.command == "check":
        for path_str in args.paths:
            path = Path(path_str)
            result = enforcer.check_file(path)
            print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.success else 1

    elif args.command == "fix":
        for path_str in args.paths:
            path = Path(path_str)
            result = enforcer.fix_all(path)
            print(json.dumps(result.to_dict(), indent=2))
        return 0

    elif args.command == "tools":
        tools = enforcer.get_available_tools()
        print("Available tools:")
        for tool, available in sorted(tools.items()):
            status = "OK" if available else "NOT FOUND"
            print(f"  {tool}: {status}")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
