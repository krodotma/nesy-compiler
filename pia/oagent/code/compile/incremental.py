#!/usr/bin/env python3
"""
incremental.py - Incremental Compiler (Step 60)

PBTSO Phase: TEST

Provides:
- Incremental compilation for changed files
- Multi-language support (Python, TypeScript)
- Compile error parsing and reporting
- File hash caching for change detection

Bus Topics:
- code.compile.incremental
- code.compile.error
- a2a.test.compile.result

Protocol: DKIN v30
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Types
# =============================================================================

class CompileStatus(Enum):
    """Status of a compile operation."""
    SUCCESS = "success"
    ERROR = "error"
    CACHED = "cached"
    SKIPPED = "skipped"


@dataclass
class CompileError:
    """Represents a compilation error."""
    file: str
    line: int
    column: int
    message: str
    error_type: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "error_type": self.error_type,
            "severity": self.severity,
        }

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}: {self.error_type}: {self.message}"


@dataclass
class CompileResult:
    """Result of a compilation operation."""
    file: str
    status: CompileStatus
    errors: List[CompileError] = field(default_factory=list)
    elapsed_ms: float = 0.0
    output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "status": self.status.value,
            "errors": [e.to_dict() for e in self.errors],
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class IncrementalCompileResult:
    """Result of an incremental compilation run."""
    compiled: List[str]
    cached: List[str]
    errors: List[CompileError]
    total_files: int
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compiled": self.compiled,
            "cached": self.cached,
            "errors": [e.to_dict() for e in self.errors],
            "total_files": self.total_files,
            "elapsed_ms": self.elapsed_ms,
            "success": len(self.errors) == 0,
        }


# =============================================================================
# Incremental Compiler
# =============================================================================

class IncrementalCompiler:
    """
    Incremental compilation for changed files.

    PBTSO Phase: TEST

    Features:
    - File hash caching for change detection
    - Multi-language support (Python, TypeScript, Go, Rust)
    - Detailed error parsing
    - Dependency-aware compilation order

    Compilation methods:
    - Python: py_compile + optional mypy
    - TypeScript: tsc
    - Go: go build
    - Rust: cargo check
    """

    COMPILERS: Dict[str, Tuple[str, List[str]]] = {
        ".py": ("python", [sys.executable, "-m", "py_compile"]),
        ".ts": ("typescript", ["npx", "tsc", "--noEmit"]),
        ".tsx": ("typescript", ["npx", "tsc", "--noEmit"]),
        ".go": ("go", ["go", "build", "-o", "/dev/null"]),
        ".rs": ("rust", ["cargo", "check"]),
    }

    def __init__(
        self,
        project_root: Path,
        bus: Optional[Any] = None,
        use_cache: bool = True,
        cache_file: Optional[Path] = None,
    ):
        self.project_root = Path(project_root)
        self.bus = bus
        self.use_cache = use_cache
        self.cache_file = cache_file or self.project_root / ".pluribus" / "compile_cache.json"

        self.file_hashes: Dict[str, str] = {}
        self.compile_cache: Dict[str, bool] = {}  # file -> last compile success

        if use_cache:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load file hash cache from disk."""
        if self.cache_file.exists():
            try:
                with self.cache_file.open() as f:
                    data = json.load(f)
                    self.file_hashes = data.get("hashes", {})
                    self.compile_cache = data.get("compile", {})
            except (json.JSONDecodeError, IOError):
                pass

    def _save_cache(self) -> None:
        """Save file hash cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("w") as f:
            json.dump({
                "hashes": self.file_hashes,
                "compile": self.compile_cache,
                "updated": time.time(),
            }, f, indent=2)

    # =========================================================================
    # Compilation
    # =========================================================================

    def compile_changed(self) -> IncrementalCompileResult:
        """
        Compile all changed files in the project.

        Returns:
            IncrementalCompileResult with compilation outcomes
        """
        start_time = time.time()
        changed_files = self._get_changed_files()

        compiled = []
        cached = []
        all_errors: List[CompileError] = []

        for file_path in changed_files:
            if self._needs_compile(file_path):
                result = self.compile_file(file_path)

                if result.status == CompileStatus.SUCCESS:
                    compiled.append(str(file_path))
                elif result.status == CompileStatus.CACHED:
                    cached.append(str(file_path))

                all_errors.extend(result.errors)

                # Update cache
                self.file_hashes[str(file_path)] = self._get_file_hash(file_path)
                self.compile_cache[str(file_path)] = len(result.errors) == 0
            else:
                cached.append(str(file_path))

        # Save cache
        if self.use_cache:
            self._save_cache()

        elapsed_ms = (time.time() - start_time) * 1000

        result = IncrementalCompileResult(
            compiled=compiled,
            cached=cached,
            errors=all_errors,
            total_files=len(changed_files),
            elapsed_ms=elapsed_ms,
        )

        # Emit result
        if self.bus:
            self.bus.emit({
                "topic": "code.compile.incremental",
                "kind": "compile",
                "actor": "code-agent",
                "data": result.to_dict(),
            })

            # Also emit to test agent
            if all_errors:
                self.bus.emit({
                    "topic": "a2a.test.compile.result",
                    "kind": "compile",
                    "actor": "code-agent",
                    "data": {
                        "success": False,
                        "error_count": len(all_errors),
                        "errors": [e.to_dict() for e in all_errors[:10]],
                    },
                })

        return result

    def compile_file(self, path: Path) -> CompileResult:
        """
        Compile a single file.

        Args:
            path: Path to file

        Returns:
            CompileResult with compilation outcome
        """
        start_time = time.time()

        # Check if file type is supported
        compiler_config = self.COMPILERS.get(path.suffix)
        if not compiler_config:
            return CompileResult(
                file=str(path),
                status=CompileStatus.SKIPPED,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        lang, cmd = compiler_config

        # Compile based on language
        if lang == "python":
            return self._compile_python(path, start_time)
        elif lang == "typescript":
            return self._compile_typescript(path, start_time)
        elif lang == "go":
            return self._compile_go(path, start_time)
        elif lang == "rust":
            return self._compile_rust(path, start_time)

        return CompileResult(
            file=str(path),
            status=CompileStatus.SKIPPED,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def _compile_python(self, path: Path, start_time: float) -> CompileResult:
        """Compile Python file using py_compile."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.SUCCESS,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )
            else:
                errors = self._parse_python_errors(result.stderr, path)
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.ERROR,
                    errors=errors,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

        except subprocess.TimeoutExpired:
            return CompileResult(
                file=str(path),
                status=CompileStatus.ERROR,
                errors=[CompileError(
                    file=str(path),
                    line=0,
                    column=0,
                    message="Compilation timed out",
                    error_type="TimeoutError",
                )],
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    def _compile_typescript(self, path: Path, start_time: float) -> CompileResult:
        """Compile TypeScript file using tsc."""
        try:
            result = subprocess.run(
                ["npx", "tsc", "--noEmit", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.SUCCESS,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )
            else:
                errors = self._parse_typescript_errors(result.stdout, path)
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.ERROR,
                    errors=errors,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

        except FileNotFoundError:
            return CompileResult(
                file=str(path),
                status=CompileStatus.SKIPPED,
                errors=[CompileError(
                    file=str(path),
                    line=0,
                    column=0,
                    message="TypeScript compiler (tsc) not found",
                    error_type="ToolNotFound",
                )],
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    def _compile_go(self, path: Path, start_time: float) -> CompileResult:
        """Compile Go file."""
        try:
            result = subprocess.run(
                ["go", "build", "-o", "/dev/null", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=path.parent,
            )

            if result.returncode == 0:
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.SUCCESS,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )
            else:
                errors = self._parse_go_errors(result.stderr, path)
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.ERROR,
                    errors=errors,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

        except FileNotFoundError:
            return CompileResult(
                file=str(path),
                status=CompileStatus.SKIPPED,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    def _compile_rust(self, path: Path, start_time: float) -> CompileResult:
        """Compile Rust file using cargo check."""
        try:
            # Find Cargo.toml directory
            cargo_dir = path.parent
            while cargo_dir != cargo_dir.parent:
                if (cargo_dir / "Cargo.toml").exists():
                    break
                cargo_dir = cargo_dir.parent
            else:
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.SKIPPED,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            result = subprocess.run(
                ["cargo", "check", "--message-format=json"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cargo_dir,
            )

            if result.returncode == 0:
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.SUCCESS,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )
            else:
                errors = self._parse_rust_errors(result.stdout, path)
                return CompileResult(
                    file=str(path),
                    status=CompileStatus.ERROR,
                    errors=errors,
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

        except FileNotFoundError:
            return CompileResult(
                file=str(path),
                status=CompileStatus.SKIPPED,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    # =========================================================================
    # Error Parsing
    # =========================================================================

    def _parse_python_errors(self, stderr: str, path: Path) -> List[CompileError]:
        """Parse Python compilation errors."""
        errors = []

        # Pattern: File "path", line N
        import re
        pattern = r'File "([^"]+)", line (\d+)'
        matches = re.findall(pattern, stderr)

        for file_path, line in matches:
            # Extract error message (line after the file reference)
            lines = stderr.split("\n")
            message = ""
            for i, line_text in enumerate(lines):
                if file_path in line_text and f"line {line}" in line_text:
                    if i + 1 < len(lines):
                        message = lines[i + 1].strip()
                    break

            errors.append(CompileError(
                file=str(path),
                line=int(line),
                column=0,
                message=message or "Syntax error",
                error_type="SyntaxError",
            ))

        if not errors and stderr.strip():
            # Generic error
            errors.append(CompileError(
                file=str(path),
                line=0,
                column=0,
                message=stderr.strip()[:200],
                error_type="CompileError",
            ))

        return errors

    def _parse_typescript_errors(self, stdout: str, path: Path) -> List[CompileError]:
        """Parse TypeScript compilation errors."""
        errors = []
        import re

        # Pattern: file.ts(line,col): error TS####: message
        pattern = r"([^(]+)\((\d+),(\d+)\): (error|warning) TS\d+: (.+)"
        for match in re.finditer(pattern, stdout):
            file_path, line, col, severity, message = match.groups()
            errors.append(CompileError(
                file=file_path,
                line=int(line),
                column=int(col),
                message=message,
                error_type="TypeScriptError",
                severity=severity,
            ))

        return errors

    def _parse_go_errors(self, stderr: str, path: Path) -> List[CompileError]:
        """Parse Go compilation errors."""
        errors = []
        import re

        # Pattern: file.go:line:col: message
        pattern = r"([^:]+):(\d+):(\d+): (.+)"
        for match in re.finditer(pattern, stderr):
            file_path, line, col, message = match.groups()
            errors.append(CompileError(
                file=file_path,
                line=int(line),
                column=int(col),
                message=message,
                error_type="GoError",
            ))

        return errors

    def _parse_rust_errors(self, stdout: str, path: Path) -> List[CompileError]:
        """Parse Rust compilation errors (JSON format)."""
        errors = []

        for line in stdout.split("\n"):
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
                if msg.get("reason") == "compiler-message":
                    compiler_msg = msg.get("message", {})
                    if compiler_msg.get("level") in ("error", "warning"):
                        spans = compiler_msg.get("spans", [])
                        if spans:
                            span = spans[0]
                            errors.append(CompileError(
                                file=span.get("file_name", str(path)),
                                line=span.get("line_start", 0),
                                column=span.get("column_start", 0),
                                message=compiler_msg.get("message", ""),
                                error_type="RustError",
                                severity=compiler_msg.get("level", "error"),
                            ))
            except json.JSONDecodeError:
                continue

        return errors

    # =========================================================================
    # Change Detection
    # =========================================================================

    def _get_changed_files(self) -> List[Path]:
        """Get all files that have changed since last compilation."""
        changed = []

        for ext in self.COMPILERS.keys():
            for file_path in self.project_root.rglob(f"*{ext}"):
                # Skip hidden directories and venv
                if any(part.startswith(".") for part in file_path.parts):
                    continue
                if "venv" in file_path.parts or "node_modules" in file_path.parts:
                    continue

                current_hash = self._get_file_hash(file_path)
                cached_hash = self.file_hashes.get(str(file_path))

                if current_hash != cached_hash:
                    changed.append(file_path)

        return changed

    def _needs_compile(self, path: Path) -> bool:
        """Check if file needs compilation."""
        current_hash = self._get_file_hash(path)
        cached_hash = self.file_hashes.get(str(path))

        return current_hash != cached_hash

    def _get_file_hash(self, path: Path) -> str:
        """Get MD5 hash of file contents."""
        try:
            content = path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except IOError:
            return ""

    # =========================================================================
    # Utilities
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear the compilation cache."""
        self.file_hashes.clear()
        self.compile_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_files": len(self.file_hashes),
            "successful_compiles": sum(1 for v in self.compile_cache.values() if v),
            "failed_compiles": sum(1 for v in self.compile_cache.values() if not v),
            "cache_file": str(self.cache_file),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Incremental Compiler."""
    import argparse

    parser = argparse.ArgumentParser(description="Incremental Compiler (Step 60)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compile command
    compile_parser = subparsers.add_parser("compile", help="Compile changed files")
    compile_parser.add_argument("--project", default=".", help="Project root")
    compile_parser.add_argument("--no-cache", action="store_true", help="Disable cache")

    # file command
    file_parser = subparsers.add_parser("file", help="Compile a single file")
    file_parser.add_argument("path", help="File to compile")
    file_parser.add_argument("--project", default=".", help="Project root")

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument("--project", default=".", help="Project root")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache stats")
    stats_parser.add_argument("--project", default=".", help="Project root")

    args = parser.parse_args()

    compiler = IncrementalCompiler(
        Path(args.project),
        use_cache=not getattr(args, "no_cache", False),
    )

    if args.command == "compile":
        result = compiler.compile_changed()
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if len(result.errors) == 0 else 1

    elif args.command == "file":
        result = compiler.compile_file(Path(args.path))
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.status == CompileStatus.SUCCESS else 1

    elif args.command == "clear":
        compiler.clear_cache()
        print("Cache cleared")
        return 0

    elif args.command == "stats":
        stats = compiler.get_cache_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
