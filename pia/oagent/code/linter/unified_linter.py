#!/usr/bin/env python3
"""
unified_linter.py - Unified Linter Integration (Step 76)

PBTSO Phase: VERIFY

Provides:
- Unified interface for multiple linters
- Multi-language linting support
- Auto-fix capability
- Custom rule configuration
- Issue aggregation and reporting

Bus Topics:
- code.linter.lint
- code.linter.fix
- code.linter.issues

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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class LintSeverity(Enum):
    """Severity levels for lint issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class LinterTool(Enum):
    """Available linting tools."""
    RUFF = "ruff"
    FLAKE8 = "flake8"
    PYLINT = "pylint"
    MYPY = "mypy"
    ESLINT = "eslint"
    TSLINT = "tslint"
    GOLINT = "golint"
    CLIPPY = "clippy"
    SHELLCHECK = "shellcheck"


@dataclass
class LinterConfig:
    """Configuration for the unified linter."""
    # Python settings
    python_linter: str = "ruff"
    python_ignore: List[str] = field(default_factory=lambda: ["E501"])
    python_select: List[str] = field(default_factory=list)

    # JavaScript/TypeScript settings
    js_linter: str = "eslint"
    js_config_file: Optional[str] = None

    # General settings
    auto_fix: bool = False
    max_issues_per_file: int = 100
    parallel_jobs: int = 4
    timeout_s: int = 60
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_linter": self.python_linter,
            "python_ignore": self.python_ignore,
            "python_select": self.python_select,
            "js_linter": self.js_linter,
            "js_config_file": self.js_config_file,
            "auto_fix": self.auto_fix,
            "max_issues_per_file": self.max_issues_per_file,
            "timeout_s": self.timeout_s,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class LintIssue:
    """A single lint issue."""
    file_path: str
    line: int
    column: int
    severity: LintSeverity
    code: str
    message: str
    source: str  # linter that found it
    fixable: bool = False
    fix_text: Optional[str] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "fixable": self.fixable,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass
class LintResult:
    """Result of linting a file or directory."""
    success: bool
    total_issues: int
    errors: int
    warnings: int
    files_checked: int
    issues: List[LintIssue] = field(default_factory=list)
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
class FixResult:
    """Result of fixing lint issues."""
    success: bool
    files_modified: int
    issues_fixed: int
    issues_remaining: int
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "files_modified": self.files_modified,
            "issues_fixed": self.issues_fixed,
            "issues_remaining": self.issues_remaining,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
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
# Unified Linter
# =============================================================================

class UnifiedLinter:
    """
    Unified linter for multiple languages.

    PBTSO Phase: VERIFY

    Responsibilities:
    - Run linters for different languages
    - Aggregate issues from multiple linters
    - Apply auto-fixes
    - Configure lint rules

    Usage:
        linter = UnifiedLinter(config)
        result = linter.lint_file(path)
    """

    BUS_TOPICS = {
        "lint": "code.linter.lint",
        "fix": "code.linter.fix",
        "issues": "code.linter.issues",
        "heartbeat": "code.linter.heartbeat",
    }

    # Language to file extension mapping
    LANGUAGE_EXTENSIONS: Dict[str, List[str]] = {
        "python": [".py", ".pyi"],
        "javascript": [".js", ".jsx", ".mjs"],
        "typescript": [".ts", ".tsx"],
        "go": [".go"],
        "rust": [".rs"],
        "shell": [".sh", ".bash"],
    }

    def __init__(
        self,
        config: Optional[LinterConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or LinterConfig()
        self.bus = bus or LockedAgentBus()
        self._tool_cache: Dict[str, bool] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_language(self, file_path: Path) -> Optional[str]:
        """Detect language from file extension."""
        suffix = file_path.suffix.lower()
        for lang, extensions in self.LANGUAGE_EXTENSIONS.items():
            if suffix in extensions:
                return lang
        return None

    def _check_tool(self, tool: str) -> bool:
        """Check if a linting tool is available."""
        if tool in self._tool_cache:
            return self._tool_cache[tool]

        available = shutil.which(tool) is not None

        # Check npx for node tools
        if not available and tool in {"eslint", "tslint"}:
            available = shutil.which("npx") is not None

        self._tool_cache[tool] = available
        return available

    def lint_file(self, file_path: Path) -> LintResult:
        """
        Lint a single file.

        Args:
            file_path: Path to file

        Returns:
            LintResult with issues found
        """
        start_time = time.time()

        if not file_path.exists():
            return LintResult(
                success=False,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=f"File not found: {file_path}",
            )

        language = self._get_language(file_path)
        if not language:
            return LintResult(
                success=True,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=1,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        try:
            issues = self._lint_content(file_path, language)

            # Limit issues per file
            if len(issues) > self.config.max_issues_per_file:
                issues = issues[:self.config.max_issues_per_file]

            errors = sum(1 for i in issues if i.severity == LintSeverity.ERROR)
            warnings = sum(1 for i in issues if i.severity == LintSeverity.WARNING)

            # Emit lint event
            self.bus.emit({
                "topic": self.BUS_TOPICS["lint"],
                "kind": "lint",
                "actor": "unified-linter",
                "data": {
                    "file": str(file_path),
                    "language": language,
                    "issues": len(issues),
                    "errors": errors,
                    "warnings": warnings,
                },
            })

            return LintResult(
                success=True,
                total_issues=len(issues),
                errors=errors,
                warnings=warnings,
                files_checked=1,
                issues=issues,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return LintResult(
                success=False,
                total_issues=0,
                errors=0,
                warnings=0,
                files_checked=1,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _lint_content(self, file_path: Path, language: str) -> List[LintIssue]:
        """Lint content based on language."""
        if language == "python":
            return self._lint_python(file_path)
        elif language in {"javascript", "typescript"}:
            return self._lint_js(file_path)
        elif language == "go":
            return self._lint_go(file_path)
        elif language == "shell":
            return self._lint_shell(file_path)
        else:
            return []

    def _lint_python(self, file_path: Path) -> List[LintIssue]:
        """Lint Python code."""
        issues: List[LintIssue] = []
        linter = self.config.python_linter

        if linter == "ruff" and self._check_tool("ruff"):
            issues.extend(self._run_ruff(file_path))
        elif linter == "flake8" and self._check_tool("flake8"):
            issues.extend(self._run_flake8(file_path))
        elif linter == "pylint" and self._check_tool("pylint"):
            issues.extend(self._run_pylint(file_path))

        return issues

    def _run_ruff(self, file_path: Path) -> List[LintIssue]:
        """Run ruff linter."""
        issues: List[LintIssue] = []

        try:
            cmd = ["ruff", "check", "--output-format", "json"]

            if self.config.python_ignore:
                cmd.extend(["--ignore", ",".join(self.config.python_ignore)])

            if self.config.python_select:
                cmd.extend(["--select", ",".join(self.config.python_select)])

            cmd.append(str(file_path))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data:
                        severity = LintSeverity.ERROR if item.get("code", "").startswith("E") else LintSeverity.WARNING

                        issues.append(LintIssue(
                            file_path=str(file_path),
                            line=item.get("location", {}).get("row", 1),
                            column=item.get("location", {}).get("column", 1),
                            severity=severity,
                            code=item.get("code", ""),
                            message=item.get("message", ""),
                            source="ruff",
                            fixable=item.get("fix") is not None,
                            end_line=item.get("end_location", {}).get("row"),
                            end_column=item.get("end_location", {}).get("column"),
                        ))
                except json.JSONDecodeError:
                    pass

        except (subprocess.TimeoutExpired, OSError):
            pass

        return issues

    def _run_flake8(self, file_path: Path) -> List[LintIssue]:
        """Run flake8 linter."""
        issues: List[LintIssue] = []

        try:
            cmd = ["flake8", "--format=json"]

            if self.config.python_ignore:
                cmd.extend(["--ignore", ",".join(self.config.python_ignore)])

            cmd.append(str(file_path))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            # Parse flake8 default output format
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                # Format: path:line:col: code message
                match = re.match(r'(.+):(\d+):(\d+): ([A-Z]\d+) (.+)', line)
                if match:
                    severity = LintSeverity.ERROR if match.group(4).startswith('E') else LintSeverity.WARNING

                    issues.append(LintIssue(
                        file_path=str(file_path),
                        line=int(match.group(2)),
                        column=int(match.group(3)),
                        severity=severity,
                        code=match.group(4),
                        message=match.group(5),
                        source="flake8",
                    ))

        except (subprocess.TimeoutExpired, OSError):
            pass

        return issues

    def _run_pylint(self, file_path: Path) -> List[LintIssue]:
        """Run pylint linter."""
        issues: List[LintIssue] = []

        try:
            result = subprocess.run(
                [
                    "pylint",
                    "--output-format=json",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data:
                        severity_map = {
                            "error": LintSeverity.ERROR,
                            "fatal": LintSeverity.ERROR,
                            "warning": LintSeverity.WARNING,
                            "convention": LintSeverity.INFO,
                            "refactor": LintSeverity.HINT,
                        }

                        issues.append(LintIssue(
                            file_path=str(file_path),
                            line=item.get("line", 1),
                            column=item.get("column", 1),
                            severity=severity_map.get(item.get("type", ""), LintSeverity.WARNING),
                            code=item.get("message-id", ""),
                            message=item.get("message", ""),
                            source="pylint",
                        ))
                except json.JSONDecodeError:
                    pass

        except (subprocess.TimeoutExpired, OSError):
            pass

        return issues

    def _lint_js(self, file_path: Path) -> List[LintIssue]:
        """Lint JavaScript/TypeScript code."""
        issues: List[LintIssue] = []

        if not self._check_tool("eslint"):
            return issues

        try:
            cmd = ["npx", "eslint", "--format", "json"]

            if self.config.js_config_file:
                cmd.extend(["--config", self.config.js_config_file])

            cmd.append(str(file_path))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for file_result in data:
                        for msg in file_result.get("messages", []):
                            severity = LintSeverity.ERROR if msg.get("severity", 1) == 2 else LintSeverity.WARNING

                            issues.append(LintIssue(
                                file_path=str(file_path),
                                line=msg.get("line", 1),
                                column=msg.get("column", 1),
                                severity=severity,
                                code=msg.get("ruleId", "") or "",
                                message=msg.get("message", ""),
                                source="eslint",
                                fixable=msg.get("fix") is not None,
                                end_line=msg.get("endLine"),
                                end_column=msg.get("endColumn"),
                            ))
                except json.JSONDecodeError:
                    pass

        except (subprocess.TimeoutExpired, OSError):
            pass

        return issues

    def _lint_go(self, file_path: Path) -> List[LintIssue]:
        """Lint Go code."""
        issues: List[LintIssue] = []

        # Try golint first
        if self._check_tool("golint"):
            try:
                result = subprocess.run(
                    ["golint", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                )

                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue

                    # Format: path:line:col: message
                    match = re.match(r'(.+):(\d+):(\d+): (.+)', line)
                    if match:
                        issues.append(LintIssue(
                            file_path=str(file_path),
                            line=int(match.group(2)),
                            column=int(match.group(3)),
                            severity=LintSeverity.WARNING,
                            code="golint",
                            message=match.group(4),
                            source="golint",
                        ))

            except (subprocess.TimeoutExpired, OSError):
                pass

        return issues

    def _lint_shell(self, file_path: Path) -> List[LintIssue]:
        """Lint shell scripts."""
        issues: List[LintIssue] = []

        if not self._check_tool("shellcheck"):
            return issues

        try:
            result = subprocess.run(
                ["shellcheck", "--format=json", str(file_path)],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data:
                        severity_map = {
                            "error": LintSeverity.ERROR,
                            "warning": LintSeverity.WARNING,
                            "info": LintSeverity.INFO,
                            "style": LintSeverity.HINT,
                        }

                        issues.append(LintIssue(
                            file_path=str(file_path),
                            line=item.get("line", 1),
                            column=item.get("column", 1),
                            severity=severity_map.get(item.get("level", ""), LintSeverity.WARNING),
                            code=f"SC{item.get('code', '')}",
                            message=item.get("message", ""),
                            source="shellcheck",
                            end_line=item.get("endLine"),
                            end_column=item.get("endColumn"),
                        ))
                except json.JSONDecodeError:
                    pass

        except (subprocess.TimeoutExpired, OSError):
            pass

        return issues

    def fix_file(self, file_path: Path) -> FixResult:
        """
        Apply auto-fixes to a file.

        Args:
            file_path: Path to file

        Returns:
            FixResult with fix outcome
        """
        start_time = time.time()

        if not file_path.exists():
            return FixResult(
                success=False,
                files_modified=0,
                issues_fixed=0,
                issues_remaining=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=f"File not found: {file_path}",
            )

        language = self._get_language(file_path)
        if not language:
            return FixResult(
                success=True,
                files_modified=0,
                issues_fixed=0,
                issues_remaining=0,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # Get issue count before
        before_result = self.lint_file(file_path)
        issues_before = before_result.total_issues

        try:
            modified = self._apply_fixes(file_path, language)

            # Get issue count after
            after_result = self.lint_file(file_path)
            issues_after = after_result.total_issues

            issues_fixed = max(0, issues_before - issues_after)

            # Emit fix event
            self.bus.emit({
                "topic": self.BUS_TOPICS["fix"],
                "kind": "lint",
                "actor": "unified-linter",
                "data": {
                    "file": str(file_path),
                    "language": language,
                    "issues_fixed": issues_fixed,
                    "issues_remaining": issues_after,
                },
            })

            return FixResult(
                success=True,
                files_modified=1 if modified else 0,
                issues_fixed=issues_fixed,
                issues_remaining=issues_after,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return FixResult(
                success=False,
                files_modified=0,
                issues_fixed=0,
                issues_remaining=issues_before,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _apply_fixes(self, file_path: Path, language: str) -> bool:
        """Apply auto-fixes based on language."""
        if language == "python":
            return self._fix_python(file_path)
        elif language in {"javascript", "typescript"}:
            return self._fix_js(file_path)
        return False

    def _fix_python(self, file_path: Path) -> bool:
        """Apply Python auto-fixes."""
        if self.config.python_linter == "ruff" and self._check_tool("ruff"):
            try:
                result = subprocess.run(
                    ["ruff", "check", "--fix", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                pass
        return False

    def _fix_js(self, file_path: Path) -> bool:
        """Apply JavaScript/TypeScript auto-fixes."""
        if self._check_tool("eslint"):
            try:
                result = subprocess.run(
                    ["npx", "eslint", "--fix", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                pass
        return False

    async def lint_files(self, file_paths: List[Path]) -> LintResult:
        """
        Lint multiple files in parallel.

        Args:
            file_paths: List of file paths

        Returns:
            Aggregated LintResult
        """
        start_time = time.time()

        if not self._semaphore:
            self._semaphore = asyncio.Semaphore(self.config.parallel_jobs)

        async def lint_one(path: Path) -> LintResult:
            async with self._semaphore:
                return await asyncio.to_thread(self.lint_file, path)

        tasks = [lint_one(path) for path in file_paths]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        all_issues: List[LintIssue] = []
        total_errors = 0
        total_warnings = 0

        for result in results:
            all_issues.extend(result.issues)
            total_errors += result.errors
            total_warnings += result.warnings

        return LintResult(
            success=all(r.success for r in results),
            total_issues=len(all_issues),
            errors=total_errors,
            warnings=total_warnings,
            files_checked=len(file_paths),
            issues=all_issues,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def lint_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> LintResult:
        """
        Lint all files in a directory.

        Args:
            directory: Directory to lint
            recursive: Search recursively
            extensions: File extensions to include

        Returns:
            LintResult with all issues
        """
        if extensions is None:
            extensions = []
            for ext_list in self.LANGUAGE_EXTENSIONS.values():
                extensions.extend(ext_list)

        files: List[Path] = []
        pattern = "**/*" if recursive else "*"

        for ext in extensions:
            files.extend(directory.glob(f"{pattern}{ext}"))

        # Filter out hidden directories and common exclusions
        files = [
            f for f in files
            if not any(part.startswith('.') for part in f.parts)
            and 'venv' not in f.parts
            and 'node_modules' not in f.parts
            and '__pycache__' not in f.parts
        ]

        return asyncio.run(self.lint_files(files))

    def get_available_linters(self) -> Dict[str, bool]:
        """Get availability of all linters."""
        tools = [
            "ruff", "flake8", "pylint", "mypy",
            "eslint", "golint", "shellcheck",
        ]
        return {tool: self._check_tool(tool) for tool in tools}

    def get_stats(self) -> Dict[str, Any]:
        """Get linter statistics."""
        return {
            "available_linters": self.get_available_linters(),
            "supported_languages": list(self.LANGUAGE_EXTENSIONS.keys()),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Unified Linter."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Linter Integration (Step 76)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # lint command
    lint_parser = subparsers.add_parser("lint", help="Lint files")
    lint_parser.add_argument("paths", nargs="+", help="Files or directories to lint")
    lint_parser.add_argument("--json", action="store_true", help="JSON output")

    # fix command
    fix_parser = subparsers.add_parser("fix", help="Fix lint issues")
    fix_parser.add_argument("paths", nargs="+", help="Files to fix")
    fix_parser.add_argument("--json", action="store_true", help="JSON output")

    # tools command
    subparsers.add_parser("tools", help="Show available linters")

    # stats command
    subparsers.add_parser("stats", help="Show linter stats")

    args = parser.parse_args()

    linter = UnifiedLinter()

    if args.command == "lint":
        all_issues: List[LintIssue] = []
        total_files = 0
        total_errors = 0
        total_warnings = 0

        for path_str in args.paths:
            path = Path(path_str)

            if path.is_dir():
                result = linter.lint_directory(path)
            else:
                result = linter.lint_file(path)

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
                for issue in all_issues[:50]:  # Limit display
                    severity = issue.severity.value.upper()
                    print(f"  {issue.file_path}:{issue.line}:{issue.column} [{severity}] {issue.code}: {issue.message}")

                if len(all_issues) > 50:
                    print(f"  ... and {len(all_issues) - 50} more issues")

        return 1 if total_errors > 0 else 0

    elif args.command == "fix":
        for path_str in args.paths:
            path = Path(path_str)

            if path.is_file():
                result = linter.fix_file(path)

                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    if result.success:
                        print(f"{path}: Fixed {result.issues_fixed} issues")
                    else:
                        print(f"{path}: Error - {result.error}")

        return 0

    elif args.command == "tools":
        tools = linter.get_available_linters()
        print("Available linters:")
        for tool, available in sorted(tools.items()):
            status = "OK" if available else "NOT FOUND"
            print(f"  {tool}: {status}")
        return 0

    elif args.command == "stats":
        stats = linter.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
