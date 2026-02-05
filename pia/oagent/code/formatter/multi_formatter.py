#!/usr/bin/env python3
"""
multi_formatter.py - Multi-Language Code Formatter (Step 75)

PBTSO Phase: VERIFY

Provides:
- Multi-language code formatting
- Configurable formatting rules
- Integration with external formatters
- Incremental formatting
- Format-on-save support

Bus Topics:
- code.formatter.format
- code.formatter.check
- code.formatter.error

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
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

class FormatterTool(Enum):
    """Available formatting tools."""
    BLACK = "black"
    RUFF = "ruff"
    ISORT = "isort"
    PRETTIER = "prettier"
    GOFMT = "gofmt"
    RUSTFMT = "rustfmt"
    CLANG_FORMAT = "clang-format"
    AUTOPEP8 = "autopep8"
    YAPF = "yapf"


@dataclass
class FormatterRule:
    """A formatting rule configuration."""
    name: str
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "options": self.options,
        }


@dataclass
class FormatterConfig:
    """Configuration for the code formatter."""
    # Python settings
    python_formatter: str = "black"
    python_line_length: int = 100
    python_target_version: str = "py39"
    sort_imports: bool = True

    # JavaScript/TypeScript settings
    js_formatter: str = "prettier"
    js_tab_width: int = 2
    js_use_tabs: bool = False
    js_single_quote: bool = True

    # Go settings
    go_formatter: str = "gofmt"

    # Rust settings
    rust_formatter: str = "rustfmt"

    # General settings
    format_on_save: bool = False
    timeout_s: int = 30
    parallel_jobs: int = 4
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_formatter": self.python_formatter,
            "python_line_length": self.python_line_length,
            "python_target_version": self.python_target_version,
            "sort_imports": self.sort_imports,
            "js_formatter": self.js_formatter,
            "js_tab_width": self.js_tab_width,
            "js_use_tabs": self.js_use_tabs,
            "js_single_quote": self.js_single_quote,
            "format_on_save": self.format_on_save,
            "timeout_s": self.timeout_s,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class FormatResult:
    """Result of a formatting operation."""
    success: bool
    file_path: str
    original_content: str
    formatted_content: str
    was_modified: bool
    formatter_used: str
    elapsed_ms: float
    error: Optional[str] = None

    @property
    def diff_lines(self) -> int:
        """Calculate number of lines changed."""
        if not self.was_modified:
            return 0
        orig_lines = self.original_content.count('\n')
        fmt_lines = self.formatted_content.count('\n')
        return abs(fmt_lines - orig_lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "was_modified": self.was_modified,
            "formatter_used": self.formatter_used,
            "elapsed_ms": self.elapsed_ms,
            "diff_lines": self.diff_lines,
            "error": self.error,
        }


@dataclass
class BatchFormatResult:
    """Result of batch formatting."""
    total_files: int
    successful: int
    failed: int
    modified: int
    elapsed_ms: float
    results: List[FormatResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "modified": self.modified,
            "elapsed_ms": self.elapsed_ms,
            "results": [r.to_dict() for r in self.results],
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
# Code Formatter
# =============================================================================

class CodeFormatter:
    """
    Multi-language code formatter.

    PBTSO Phase: VERIFY

    Responsibilities:
    - Format code in multiple languages
    - Integrate with external formatters
    - Check formatting without modifying
    - Support incremental formatting

    Usage:
        formatter = CodeFormatter(config)
        result = formatter.format_file(path)
    """

    BUS_TOPICS = {
        "format": "code.formatter.format",
        "check": "code.formatter.check",
        "error": "code.formatter.error",
        "heartbeat": "code.formatter.heartbeat",
    }

    # Language to file extension mapping
    LANGUAGE_EXTENSIONS: Dict[str, List[str]] = {
        "python": [".py", ".pyi"],
        "javascript": [".js", ".jsx", ".mjs"],
        "typescript": [".ts", ".tsx"],
        "go": [".go"],
        "rust": [".rs"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".hpp", ".cc", ".hh"],
        "json": [".json"],
        "yaml": [".yaml", ".yml"],
        "markdown": [".md"],
    }

    def __init__(
        self,
        config: Optional[FormatterConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or FormatterConfig()
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
        """Check if a formatting tool is available."""
        if tool in self._tool_cache:
            return self._tool_cache[tool]

        available = shutil.which(tool) is not None

        # Check npx for node tools
        if not available and tool in {"prettier"}:
            available = shutil.which("npx") is not None

        self._tool_cache[tool] = available
        return available

    def format_file(
        self,
        file_path: Path,
        check_only: bool = False,
    ) -> FormatResult:
        """
        Format a single file.

        Args:
            file_path: Path to file
            check_only: Only check, don't modify

        Returns:
            FormatResult with formatting outcome
        """
        start_time = time.time()

        if not file_path.exists():
            return FormatResult(
                success=False,
                file_path=str(file_path),
                original_content="",
                formatted_content="",
                was_modified=False,
                formatter_used="",
                elapsed_ms=(time.time() - start_time) * 1000,
                error=f"File not found: {file_path}",
            )

        language = self._get_language(file_path)
        if not language:
            return FormatResult(
                success=True,
                file_path=str(file_path),
                original_content="",
                formatted_content="",
                was_modified=False,
                formatter_used="none",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        original_content = file_path.read_text()

        try:
            formatted_content, formatter_used = self._format_content(
                original_content,
                language,
                file_path,
            )

            was_modified = original_content != formatted_content

            if was_modified and not check_only:
                file_path.write_text(formatted_content)

            # Emit format event
            self.bus.emit({
                "topic": self.BUS_TOPICS["format" if not check_only else "check"],
                "kind": "format",
                "actor": "code-formatter",
                "data": {
                    "file": str(file_path),
                    "language": language,
                    "formatter": formatter_used,
                    "modified": was_modified,
                },
            })

            return FormatResult(
                success=True,
                file_path=str(file_path),
                original_content=original_content,
                formatted_content=formatted_content,
                was_modified=was_modified,
                formatter_used=formatter_used,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "error",
                "actor": "code-formatter",
                "data": {
                    "file": str(file_path),
                    "error": str(e),
                },
            })

            return FormatResult(
                success=False,
                file_path=str(file_path),
                original_content=original_content,
                formatted_content=original_content,
                was_modified=False,
                formatter_used="",
                elapsed_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _format_content(
        self,
        content: str,
        language: str,
        file_path: Path,
    ) -> Tuple[str, str]:
        """Format content based on language."""
        if language == "python":
            return self._format_python(content, file_path)
        elif language in {"javascript", "typescript", "json", "yaml", "markdown"}:
            return self._format_with_prettier(content, file_path)
        elif language == "go":
            return self._format_go(content, file_path)
        elif language == "rust":
            return self._format_rust(content, file_path)
        elif language in {"c", "cpp"}:
            return self._format_cpp(content, file_path)
        else:
            return content, "none"

    def _format_python(
        self,
        content: str,
        file_path: Path,
    ) -> Tuple[str, str]:
        """Format Python code."""
        formatted = content
        formatter_used = []

        # Sort imports first
        if self.config.sort_imports and self._check_tool("isort"):
            try:
                result = subprocess.run(
                    [
                        "isort",
                        "--profile", "black",
                        "--line-length", str(self.config.python_line_length),
                        "-",
                    ],
                    input=formatted,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                )
                if result.returncode == 0:
                    formatted = result.stdout
                    formatter_used.append("isort")
            except (subprocess.TimeoutExpired, OSError):
                pass

        # Format with primary formatter
        formatter = self.config.python_formatter

        if formatter == "black" and self._check_tool("black"):
            try:
                result = subprocess.run(
                    [
                        "black",
                        "--line-length", str(self.config.python_line_length),
                        "--target-version", self.config.python_target_version,
                        "-q",
                        "-",
                    ],
                    input=formatted,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                )
                if result.returncode == 0:
                    formatted = result.stdout
                    formatter_used.append("black")
            except (subprocess.TimeoutExpired, OSError):
                pass

        elif formatter == "ruff" and self._check_tool("ruff"):
            try:
                result = subprocess.run(
                    [
                        "ruff",
                        "format",
                        "--line-length", str(self.config.python_line_length),
                        "-",
                    ],
                    input=formatted,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                )
                if result.returncode == 0:
                    formatted = result.stdout
                    formatter_used.append("ruff")
            except (subprocess.TimeoutExpired, OSError):
                pass

        elif formatter == "autopep8" and self._check_tool("autopep8"):
            try:
                result = subprocess.run(
                    [
                        "autopep8",
                        "--max-line-length", str(self.config.python_line_length),
                        "-",
                    ],
                    input=formatted,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                )
                if result.returncode == 0:
                    formatted = result.stdout
                    formatter_used.append("autopep8")
            except (subprocess.TimeoutExpired, OSError):
                pass

        return formatted, "+".join(formatter_used) if formatter_used else "none"

    def _format_with_prettier(
        self,
        content: str,
        file_path: Path,
    ) -> Tuple[str, str]:
        """Format with Prettier."""
        if not self._check_tool("prettier"):
            return content, "none"

        try:
            cmd = ["npx", "prettier", "--stdin-filepath", str(file_path)]

            if self.config.js_use_tabs:
                cmd.append("--use-tabs")
            else:
                cmd.extend(["--tab-width", str(self.config.js_tab_width)])

            if self.config.js_single_quote:
                cmd.append("--single-quote")

            result = subprocess.run(
                cmd,
                input=content,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            if result.returncode == 0:
                return result.stdout, "prettier"

        except (subprocess.TimeoutExpired, OSError):
            pass

        return content, "none"

    def _format_go(
        self,
        content: str,
        file_path: Path,
    ) -> Tuple[str, str]:
        """Format Go code."""
        if not self._check_tool("gofmt"):
            return content, "none"

        try:
            result = subprocess.run(
                ["gofmt"],
                input=content,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            if result.returncode == 0:
                return result.stdout, "gofmt"

        except (subprocess.TimeoutExpired, OSError):
            pass

        return content, "none"

    def _format_rust(
        self,
        content: str,
        file_path: Path,
    ) -> Tuple[str, str]:
        """Format Rust code."""
        if not self._check_tool("rustfmt"):
            return content, "none"

        try:
            result = subprocess.run(
                ["rustfmt", "--edition", "2021"],
                input=content,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            if result.returncode == 0:
                return result.stdout, "rustfmt"

        except (subprocess.TimeoutExpired, OSError):
            pass

        return content, "none"

    def _format_cpp(
        self,
        content: str,
        file_path: Path,
    ) -> Tuple[str, str]:
        """Format C/C++ code."""
        if not self._check_tool("clang-format"):
            return content, "none"

        try:
            result = subprocess.run(
                ["clang-format"],
                input=content,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            if result.returncode == 0:
                return result.stdout, "clang-format"

        except (subprocess.TimeoutExpired, OSError):
            pass

        return content, "none"

    async def format_files(
        self,
        file_paths: List[Path],
        check_only: bool = False,
    ) -> BatchFormatResult:
        """
        Format multiple files in parallel.

        Args:
            file_paths: List of file paths
            check_only: Only check, don't modify

        Returns:
            BatchFormatResult with all outcomes
        """
        start_time = time.time()

        if not self._semaphore:
            self._semaphore = asyncio.Semaphore(self.config.parallel_jobs)

        async def format_one(path: Path) -> FormatResult:
            async with self._semaphore:
                return await asyncio.to_thread(self.format_file, path, check_only)

        tasks = [format_one(path) for path in file_paths]
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        modified = sum(1 for r in results if r.was_modified)

        return BatchFormatResult(
            total_files=len(file_paths),
            successful=successful,
            failed=failed,
            modified=modified,
            elapsed_ms=(time.time() - start_time) * 1000,
            results=results,
        )

    def format_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        check_only: bool = False,
    ) -> BatchFormatResult:
        """
        Format all files in a directory.

        Args:
            directory: Directory to format
            recursive: Search recursively
            extensions: File extensions to include
            check_only: Only check, don't modify

        Returns:
            BatchFormatResult with all outcomes
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

        return asyncio.run(self.format_files(files, check_only))

    def check_file(self, file_path: Path) -> FormatResult:
        """Check if a file needs formatting."""
        return self.format_file(file_path, check_only=True)

    def get_available_formatters(self) -> Dict[str, bool]:
        """Get availability of all formatters."""
        tools = [
            "black", "ruff", "isort", "autopep8", "yapf",
            "prettier", "gofmt", "rustfmt", "clang-format",
        ]
        return {tool: self._check_tool(tool) for tool in tools}

    def get_stats(self) -> Dict[str, Any]:
        """Get formatter statistics."""
        return {
            "available_formatters": self.get_available_formatters(),
            "supported_languages": list(self.LANGUAGE_EXTENSIONS.keys()),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Code Formatter."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Language Code Formatter (Step 75)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # format command
    fmt_parser = subparsers.add_parser("format", help="Format files")
    fmt_parser.add_argument("paths", nargs="+", help="Files or directories to format")
    fmt_parser.add_argument("--check", action="store_true", help="Check only, don't modify")
    fmt_parser.add_argument("--json", action="store_true", help="JSON output")

    # check command
    check_parser = subparsers.add_parser("check", help="Check formatting")
    check_parser.add_argument("paths", nargs="+", help="Files or directories to check")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    # tools command
    subparsers.add_parser("tools", help="Show available formatters")

    # stats command
    subparsers.add_parser("stats", help="Show formatter stats")

    args = parser.parse_args()

    formatter = CodeFormatter()

    if args.command in {"format", "check"}:
        check_only = args.command == "check" or getattr(args, "check", False)
        all_results: List[FormatResult] = []

        for path_str in args.paths:
            path = Path(path_str)

            if path.is_dir():
                batch = formatter.format_directory(path, check_only=check_only)
                all_results.extend(batch.results)
            else:
                result = formatter.format_file(path, check_only=check_only)
                all_results.append(result)

        if args.json:
            print(json.dumps([r.to_dict() for r in all_results], indent=2))
        else:
            modified = sum(1 for r in all_results if r.was_modified)
            failed = sum(1 for r in all_results if not r.success)

            print(f"Processed {len(all_results)} files")
            print(f"  Modified: {modified}")
            print(f"  Failed: {failed}")

            if failed > 0:
                print("\nErrors:")
                for r in all_results:
                    if not r.success:
                        print(f"  {r.file_path}: {r.error}")

            if check_only and modified > 0:
                print("\nFiles needing format:")
                for r in all_results:
                    if r.was_modified:
                        print(f"  {r.file_path}")
                return 1

        return 0

    elif args.command == "tools":
        tools = formatter.get_available_formatters()
        print("Available formatters:")
        for tool, available in sorted(tools.items()):
            status = "OK" if available else "NOT FOUND"
            print(f"  {tool}: {status}")
        return 0

    elif args.command == "stats":
        stats = formatter.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
