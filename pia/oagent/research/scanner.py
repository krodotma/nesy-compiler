#!/usr/bin/env python3
"""
scanner.py - Codebase Scanner Core (Step 2)

Scans filesystem to discover and catalog source files for indexing.

PBTSO Phase: RESEARCH

Bus Topics:
- research.scan.start
- research.scan.progress
- research.scan.complete
- research.index.update

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from .bootstrap import AgentBus

# ============================================================================
# Configuration
# ============================================================================

# Default patterns to ignore during scanning
DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    ".build",
    ".cache",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".eggs",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.class",
]

# File extensions of interest for code analysis
CODE_EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c_header",
    ".hpp": "cpp_header",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".clj": "clojure",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".fs": "fsharp",
    ".cs": "csharp",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".jl": "julia",
    ".dart": "dart",
    ".vue": "vue",
    ".svelte": "svelte",
}

DOC_EXTENSIONS = {
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "plaintext",
    ".adoc": "asciidoc",
}

CONFIG_EXTENSIONS = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "conf",
    ".xml": "xml",
}


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class FileInfo:
    """Metadata about a scanned file."""

    path: str  # Relative path from root
    abs_path: str  # Absolute path
    size: int  # Size in bytes
    ext: str  # File extension
    language: Optional[str]  # Detected language
    category: str  # code, doc, config, other
    mtime: float  # Modification timestamp
    content_hash: Optional[str] = None  # SHA256 hash of content
    line_count: Optional[int] = None  # Number of lines


@dataclass
class ScanResult:
    """Result of a codebase scan."""

    root: str
    total_files: int
    code_files: int
    doc_files: int
    config_files: int
    other_files: int
    total_size_bytes: int
    scan_duration_s: float
    files_by_language: Dict[str, int] = field(default_factory=dict)
    files_by_extension: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# ============================================================================
# Codebase Scanner
# ============================================================================


class CodebaseScanner:
    """
    Core codebase scanner for file discovery and cataloging.

    Scans a directory tree and yields file metadata for indexing.
    Reports progress via A2A bus events.

    Example:
        scanner = CodebaseScanner(Path("/project"))
        for file_info in scanner.scan():
            print(f"Found: {file_info.path}")
    """

    def __init__(
        self,
        root: Path,
        ignore_patterns: Optional[List[str]] = None,
        bus: Optional[AgentBus] = None,
        progress_interval: int = 100,
    ):
        """
        Initialize the codebase scanner.

        Args:
            root: Root directory to scan
            ignore_patterns: Patterns to ignore (default: DEFAULT_IGNORE_PATTERNS)
            bus: AgentBus for event emission (optional)
            progress_interval: Emit progress every N files
        """
        self.root = Path(root).resolve()
        self.ignore_patterns = set(ignore_patterns or DEFAULT_IGNORE_PATTERNS)
        self.bus = bus or AgentBus()
        self.progress_interval = progress_interval

        self._file_count = 0
        self._scan_start_ts: Optional[float] = None
        self._result: Optional[ScanResult] = None

    def scan(
        self,
        compute_hash: bool = False,
        count_lines: bool = False,
    ) -> Iterator[FileInfo]:
        """
        Scan the codebase and yield file metadata.

        Args:
            compute_hash: Whether to compute content hash (slower)
            count_lines: Whether to count lines in files (slower)

        Yields:
            FileInfo objects for each discovered file
        """
        self._file_count = 0
        self._scan_start_ts = time.time()

        # Emit scan start
        self.bus.emit({
            "topic": "research.scan.start",
            "kind": "lifecycle",
            "data": {
                "root": str(self.root),
                "compute_hash": compute_hash,
                "count_lines": count_lines,
            }
        })

        # Initialize counters
        code_files = 0
        doc_files = 0
        config_files = 0
        other_files = 0
        total_size = 0
        files_by_language: Dict[str, int] = {}
        files_by_extension: Dict[str, int] = {}
        errors: List[str] = []

        try:
            for path in self._walk_files():
                try:
                    file_info = self._process_file(
                        path,
                        compute_hash=compute_hash,
                        count_lines=count_lines,
                    )

                    if file_info is None:
                        continue

                    self._file_count += 1
                    total_size += file_info.size

                    # Update counters
                    if file_info.category == "code":
                        code_files += 1
                        if file_info.language:
                            files_by_language[file_info.language] = (
                                files_by_language.get(file_info.language, 0) + 1
                            )
                    elif file_info.category == "doc":
                        doc_files += 1
                    elif file_info.category == "config":
                        config_files += 1
                    else:
                        other_files += 1

                    files_by_extension[file_info.ext] = (
                        files_by_extension.get(file_info.ext, 0) + 1
                    )

                    # Emit progress
                    if self._file_count % self.progress_interval == 0:
                        self._emit_progress()

                    yield file_info

                except Exception as e:
                    error_msg = f"Error processing {path}: {e}"
                    errors.append(error_msg)

        finally:
            # Create scan result
            scan_duration = time.time() - self._scan_start_ts

            self._result = ScanResult(
                root=str(self.root),
                total_files=self._file_count,
                code_files=code_files,
                doc_files=doc_files,
                config_files=config_files,
                other_files=other_files,
                total_size_bytes=total_size,
                scan_duration_s=scan_duration,
                files_by_language=files_by_language,
                files_by_extension=files_by_extension,
                errors=errors,
            )

            # Emit scan complete
            self.bus.emit({
                "topic": "research.scan.complete",
                "kind": "lifecycle",
                "data": self._result.to_dict(),
            })

    def scan_all(
        self,
        compute_hash: bool = False,
        count_lines: bool = False,
    ) -> List[FileInfo]:
        """
        Scan the codebase and return all file metadata.

        This consumes the iterator and returns a list.

        Returns:
            List of FileInfo objects
        """
        return list(self.scan(compute_hash=compute_hash, count_lines=count_lines))

    def get_result(self) -> Optional[ScanResult]:
        """Get the scan result (available after scan completes)."""
        return self._result

    def _walk_files(self) -> Iterator[Path]:
        """Walk directory tree, yielding file paths."""
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Filter out ignored directories in-place
            dirnames[:] = [
                d for d in dirnames
                if not self._should_ignore(d)
            ]

            for filename in filenames:
                if self._should_ignore(filename):
                    continue

                yield Path(dirpath) / filename

    def _should_ignore(self, name: str) -> bool:
        """Check if a file/directory name should be ignored."""
        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                # Wildcard pattern
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True
        return False

    def _process_file(
        self,
        path: Path,
        compute_hash: bool = False,
        count_lines: bool = False,
    ) -> Optional[FileInfo]:
        """Process a single file and return metadata."""
        try:
            stat = path.stat()
        except (OSError, PermissionError):
            return None

        # Skip if not a regular file
        if not path.is_file():
            return None

        rel_path = str(path.relative_to(self.root))
        ext = path.suffix.lower()

        # Determine language and category
        language = None
        if ext in CODE_EXTENSIONS:
            language = CODE_EXTENSIONS[ext]
            category = "code"
        elif ext in DOC_EXTENSIONS:
            language = DOC_EXTENSIONS[ext]
            category = "doc"
        elif ext in CONFIG_EXTENSIONS:
            language = CONFIG_EXTENSIONS[ext]
            category = "config"
        else:
            category = "other"

        # Compute optional fields
        content_hash = None
        line_count = None

        if compute_hash or count_lines:
            try:
                content = path.read_bytes()
                if compute_hash:
                    content_hash = hashlib.sha256(content).hexdigest()
                if count_lines:
                    # Try to decode and count lines
                    try:
                        text = content.decode("utf-8", errors="ignore")
                        line_count = len(text.splitlines())
                    except Exception:
                        pass
            except (OSError, PermissionError):
                pass

        return FileInfo(
            path=rel_path,
            abs_path=str(path),
            size=stat.st_size,
            ext=ext,
            language=language,
            category=category,
            mtime=stat.st_mtime,
            content_hash=content_hash,
            line_count=line_count,
        )

    def _emit_progress(self) -> None:
        """Emit progress event to bus."""
        elapsed = time.time() - self._scan_start_ts if self._scan_start_ts else 0

        self.bus.emit({
            "topic": "research.scan.progress",
            "kind": "metric",
            "data": {
                "files_scanned": self._file_count,
                "elapsed_s": round(elapsed, 2),
                "files_per_second": round(self._file_count / elapsed, 1) if elapsed > 0 else 0,
            }
        })


# ============================================================================
# Incremental Scanner
# ============================================================================


class IncrementalScanner:
    """
    Incremental codebase scanner that tracks changes.

    Uses file modification times and hashes to detect changes
    since the last scan.
    """

    def __init__(
        self,
        root: Path,
        state_file: Optional[Path] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the incremental scanner.

        Args:
            root: Root directory to scan
            state_file: Path to store scan state (default: .pluribus/research/scan_state.json)
            bus: AgentBus for event emission
        """
        self.root = Path(root).resolve()
        self.bus = bus or AgentBus()

        if state_file is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            state_file = Path(pluribus_root) / ".pluribus" / "research" / "scan_state.json"

        self.state_file = Path(state_file)
        self._state: Dict[str, Dict[str, Any]] = {}
        self._load_state()

    def scan_changes(self) -> Iterator[tuple[str, FileInfo]]:
        """
        Scan for changed files since last scan.

        Yields:
            Tuples of (change_type, file_info) where change_type is one of:
            - "added": New file
            - "modified": File content changed
            - "deleted": File was removed
        """
        scanner = CodebaseScanner(self.root, bus=self.bus)
        current_files: Set[str] = set()

        for file_info in scanner.scan():
            rel_path = file_info.path
            current_files.add(rel_path)

            if rel_path not in self._state:
                # New file
                yield ("added", file_info)
            elif file_info.mtime > self._state[rel_path].get("mtime", 0):
                # Modified file
                yield ("modified", file_info)

            # Update state
            self._state[rel_path] = {
                "mtime": file_info.mtime,
                "size": file_info.size,
            }

        # Check for deleted files
        deleted = set(self._state.keys()) - current_files
        for rel_path in deleted:
            yield ("deleted", FileInfo(
                path=rel_path,
                abs_path=str(self.root / rel_path),
                size=0,
                ext=Path(rel_path).suffix,
                language=None,
                category="deleted",
                mtime=0,
            ))
            del self._state[rel_path]

        self._save_state()

    def _load_state(self) -> None:
        """Load scan state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self._state = json.load(f)
            except Exception:
                self._state = {}

    def _save_state(self) -> None:
        """Save scan state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Codebase Scanner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Codebase Scanner (Step 2)"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory)"
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute content hashes"
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Count lines in files"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary only"
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Error: Directory not found: {root}")
        return 1

    scanner = CodebaseScanner(root)

    if args.summary:
        # Consume scan and show summary
        list(scanner.scan(compute_hash=args.hash, count_lines=args.lines))
        result = scanner.get_result()
        if result:
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(f"Scan Summary for: {result.root}")
                print(f"  Total Files: {result.total_files}")
                print(f"  Code Files: {result.code_files}")
                print(f"  Doc Files: {result.doc_files}")
                print(f"  Config Files: {result.config_files}")
                print(f"  Other Files: {result.other_files}")
                print(f"  Total Size: {result.total_size_bytes:,} bytes")
                print(f"  Scan Time: {result.scan_duration_s:.2f}s")
                if result.files_by_language:
                    print(f"  Languages: {dict(sorted(result.files_by_language.items(), key=lambda x: -x[1]))}")
    else:
        # Stream files
        for file_info in scanner.scan(compute_hash=args.hash, count_lines=args.lines):
            if args.json:
                print(json.dumps(asdict(file_info)))
            else:
                print(f"{file_info.category:8} {file_info.language or 'unknown':12} {file_info.path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
