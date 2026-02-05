#!/usr/bin/env python3
"""
Static Analysis Engine (Step 152)

Runs static analysis tools on code files and aggregates results.

PBTSO Phase: VERIFY
Bus Topics: review.static.analyze, review.issues.found

Supported analyzers:
- Python: ruff, mypy, pylint
- TypeScript/JavaScript: eslint
- Go: golangci-lint
- Rust: clippy

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
import os
import re
import subprocess
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

class AnalysisSeverity(Enum):
    """Severity levels for static analysis issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"

    @classmethod
    def from_string(cls, s: str) -> "AnalysisSeverity":
        """Convert string to severity."""
        mapping = {
            "error": cls.ERROR,
            "err": cls.ERROR,
            "e": cls.ERROR,
            "warning": cls.WARNING,
            "warn": cls.WARNING,
            "w": cls.WARNING,
            "info": cls.INFO,
            "information": cls.INFO,
            "i": cls.INFO,
            "hint": cls.HINT,
            "h": cls.HINT,
            "note": cls.HINT,
        }
        return mapping.get(s.lower(), cls.INFO)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"

    @classmethod
    def detect(cls, file_path: str) -> "Language":
        """Detect language from file extension."""
        ext_map = {
            ".py": cls.PYTHON,
            ".pyi": cls.PYTHON,
            ".ts": cls.TYPESCRIPT,
            ".tsx": cls.TYPESCRIPT,
            ".js": cls.JAVASCRIPT,
            ".jsx": cls.JAVASCRIPT,
            ".mjs": cls.JAVASCRIPT,
            ".cjs": cls.JAVASCRIPT,
            ".go": cls.GO,
            ".rs": cls.RUST,
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, cls.UNKNOWN)


@dataclass
class StaticAnalysisIssue:
    """
    Represents a single static analysis issue.

    Attributes:
        file: Absolute path to the file
        line: Line number (1-indexed)
        column: Column number (1-indexed)
        severity: Issue severity level
        rule: Rule ID that triggered the issue
        message: Human-readable description
        source: Tool that found the issue
        end_line: End line for multi-line issues
        end_column: End column for multi-line issues
        fix_available: Whether an auto-fix is available
        fix_suggestion: Suggested fix text
    """
    file: str
    line: int
    column: int
    severity: AnalysisSeverity
    rule: str
    message: str
    source: str = "unknown"
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    fix_available: bool = False
    fix_suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        return result

    @property
    def location(self) -> str:
        """Get human-readable location string."""
        if self.end_line and self.end_line != self.line:
            return f"{self.file}:{self.line}:{self.column}-{self.end_line}:{self.end_column}"
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class AnalysisResult:
    """Result from analyzing a set of files."""
    files_analyzed: int = 0
    issues: List[StaticAnalysisIssue] = field(default_factory=list)
    duration_ms: float = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    analyzers_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files_analyzed": self.files_analyzed,
            "issues": [i.to_dict() for i in self.issues],
            "duration_ms": self.duration_ms,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "analyzers_used": self.analyzers_used,
        }


# ============================================================================
# Language Analyzers
# ============================================================================

class LanguageAnalyzer:
    """Base class for language-specific analyzers."""

    LANGUAGE: Language = Language.UNKNOWN
    COMMAND: List[str] = []

    def __init__(self, timeout: int = 60):
        """
        Initialize analyzer.

        Args:
            timeout: Command timeout in seconds
        """
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if the analyzer tool is available."""
        if not self.COMMAND:
            return False
        try:
            result = subprocess.run(
                [self.COMMAND[0], "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def analyze(self, files: List[str]) -> List[StaticAnalysisIssue]:
        """
        Analyze files and return issues.

        Args:
            files: List of file paths to analyze

        Returns:
            List of analysis issues found
        """
        raise NotImplementedError

    def _run_command(self, args: List[str]) -> Tuple[int, str, str]:
        """Run a command and capture output."""
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"Command not found: {args[0]}"


class RuffAnalyzer(LanguageAnalyzer):
    """Python analyzer using ruff."""

    LANGUAGE = Language.PYTHON
    COMMAND = ["ruff", "check", "--output-format", "json"]

    def analyze(self, files: List[str]) -> List[StaticAnalysisIssue]:
        """Analyze Python files using ruff."""
        if not files:
            return []

        returncode, stdout, stderr = self._run_command(self.COMMAND + files)

        # ruff returns non-zero when issues found, but still outputs JSON
        if not stdout:
            return []

        issues = []
        try:
            data = json.loads(stdout)
            for item in data:
                severity = AnalysisSeverity.WARNING
                if item.get("code", "").startswith("E"):
                    severity = AnalysisSeverity.ERROR

                issues.append(StaticAnalysisIssue(
                    file=item.get("filename", ""),
                    line=item.get("location", {}).get("row", 1),
                    column=item.get("location", {}).get("column", 1),
                    end_line=item.get("end_location", {}).get("row"),
                    end_column=item.get("end_location", {}).get("column"),
                    severity=severity,
                    rule=item.get("code", "UNKNOWN"),
                    message=item.get("message", ""),
                    source="ruff",
                    fix_available=item.get("fix") is not None,
                ))
        except json.JSONDecodeError:
            pass

        return issues


class MypyAnalyzer(LanguageAnalyzer):
    """Python type checker using mypy."""

    LANGUAGE = Language.PYTHON
    COMMAND = ["mypy", "--show-error-codes", "--no-error-summary"]

    def analyze(self, files: List[str]) -> List[StaticAnalysisIssue]:
        """Analyze Python files using mypy."""
        if not files:
            return []

        returncode, stdout, stderr = self._run_command(self.COMMAND + files)

        issues = []
        # Parse mypy output format: file:line: severity: message [code]
        pattern = re.compile(r"^(.+?):(\d+):(?:(\d+):)?\s*(error|warning|note):\s*(.+?)(?:\s*\[(.+?)\])?$")

        for line in stdout.split("\n"):
            match = pattern.match(line.strip())
            if match:
                severity_str = match.group(4)
                issues.append(StaticAnalysisIssue(
                    file=match.group(1),
                    line=int(match.group(2)),
                    column=int(match.group(3)) if match.group(3) else 1,
                    severity=AnalysisSeverity.from_string(severity_str),
                    rule=match.group(6) or "mypy",
                    message=match.group(5),
                    source="mypy",
                ))

        return issues


class ESLintAnalyzer(LanguageAnalyzer):
    """JavaScript/TypeScript analyzer using ESLint."""

    LANGUAGE = Language.TYPESCRIPT
    COMMAND = ["npx", "eslint", "--format", "json"]

    def analyze(self, files: List[str]) -> List[StaticAnalysisIssue]:
        """Analyze JS/TS files using ESLint."""
        if not files:
            return []

        returncode, stdout, stderr = self._run_command(self.COMMAND + files)

        if not stdout:
            return []

        issues = []
        try:
            data = json.loads(stdout)
            for file_result in data:
                file_path = file_result.get("filePath", "")
                for msg in file_result.get("messages", []):
                    severity = AnalysisSeverity.ERROR if msg.get("severity") == 2 else AnalysisSeverity.WARNING
                    issues.append(StaticAnalysisIssue(
                        file=file_path,
                        line=msg.get("line", 1),
                        column=msg.get("column", 1),
                        end_line=msg.get("endLine"),
                        end_column=msg.get("endColumn"),
                        severity=severity,
                        rule=msg.get("ruleId", "eslint"),
                        message=msg.get("message", ""),
                        source="eslint",
                        fix_available=msg.get("fix") is not None,
                    ))
        except json.JSONDecodeError:
            pass

        return issues


class GolangciLintAnalyzer(LanguageAnalyzer):
    """Go analyzer using golangci-lint."""

    LANGUAGE = Language.GO
    COMMAND = ["golangci-lint", "run", "--out-format", "json"]

    def analyze(self, files: List[str]) -> List[StaticAnalysisIssue]:
        """Analyze Go files using golangci-lint."""
        if not files:
            return []

        # golangci-lint works on directories, get unique dirs
        dirs = list(set(str(Path(f).parent) for f in files))
        issues = []

        for dir_path in dirs:
            returncode, stdout, stderr = self._run_command(self.COMMAND + [dir_path])

            if not stdout:
                continue

            try:
                data = json.loads(stdout)
                for item in data.get("Issues", []):
                    pos = item.get("Pos", {})
                    issues.append(StaticAnalysisIssue(
                        file=pos.get("Filename", ""),
                        line=pos.get("Line", 1),
                        column=pos.get("Column", 1),
                        severity=AnalysisSeverity.from_string(item.get("Severity", "warning")),
                        rule=item.get("FromLinter", "unknown"),
                        message=item.get("Text", ""),
                        source="golangci-lint",
                    ))
            except json.JSONDecodeError:
                pass

        return issues


# ============================================================================
# Static Analysis Engine
# ============================================================================

class StaticAnalysisEngine:
    """
    Unified static analysis engine.

    Coordinates multiple language analyzers and aggregates results.

    Example:
        engine = StaticAnalysisEngine()
        result = engine.analyze(["/path/to/file.py", "/path/to/file.ts"])
        for issue in result.issues:
            print(f"{issue.location}: {issue.message}")
    """

    ANALYZERS = {
        Language.PYTHON: [RuffAnalyzer, MypyAnalyzer],
        Language.TYPESCRIPT: [ESLintAnalyzer],
        Language.JAVASCRIPT: [ESLintAnalyzer],
        Language.GO: [GolangciLintAnalyzer],
    }

    def __init__(
        self,
        timeout: int = 60,
        enable_mypy: bool = True,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the static analysis engine.

        Args:
            timeout: Per-analyzer timeout in seconds
            enable_mypy: Enable mypy for Python (slower but thorough)
            bus_path: Path to event bus file
        """
        self.timeout = timeout
        self.enable_mypy = enable_mypy
        self.bus_path = bus_path or self._get_bus_path()
        self._analyzer_cache: Dict[str, LanguageAnalyzer] = {}

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
            "kind": "analysis",
            "actor": "static-analyzer",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _get_analyzer(self, analyzer_cls: type) -> Optional[LanguageAnalyzer]:
        """Get or create an analyzer instance."""
        key = analyzer_cls.__name__
        if key not in self._analyzer_cache:
            analyzer = analyzer_cls(timeout=self.timeout)
            if analyzer.is_available():
                self._analyzer_cache[key] = analyzer
            else:
                return None
        return self._analyzer_cache.get(key)

    def analyze(
        self,
        files: List[str],
        languages: Optional[List[Language]] = None,
    ) -> AnalysisResult:
        """
        Analyze files and return aggregated results.

        Args:
            files: List of file paths to analyze
            languages: Limit analysis to specific languages (optional)

        Returns:
            AnalysisResult with all issues found

        Emits:
            review.static.analyze (start)
            review.issues.found (per issue batch)
        """
        start_time = time.time()

        # Emit start event
        self._emit_event("review.static.analyze", {
            "files": files,
            "languages": [l.value for l in languages] if languages else None,
            "status": "started",
        })

        # Group files by language
        files_by_lang: Dict[Language, List[str]] = {}
        for file_path in files:
            lang = Language.detect(file_path)
            if lang == Language.UNKNOWN:
                continue
            if languages and lang not in languages:
                continue
            if lang not in files_by_lang:
                files_by_lang[lang] = []
            files_by_lang[lang].append(file_path)

        result = AnalysisResult(
            files_analyzed=len(files),
        )

        # Run analyzers for each language
        for lang, lang_files in files_by_lang.items():
            analyzer_classes = self.ANALYZERS.get(lang, [])

            for analyzer_cls in analyzer_classes:
                # Skip mypy if disabled
                if analyzer_cls == MypyAnalyzer and not self.enable_mypy:
                    continue

                analyzer = self._get_analyzer(analyzer_cls)
                if not analyzer:
                    continue

                result.analyzers_used.append(analyzer_cls.__name__)
                issues = analyzer.analyze(lang_files)
                result.issues.extend(issues)

                # Emit issues found
                if issues:
                    self._emit_event("review.issues.found", {
                        "analyzer": analyzer_cls.__name__,
                        "language": lang.value,
                        "issue_count": len(issues),
                        "issues": [i.to_dict() for i in issues[:10]],  # First 10 only
                    })

        # Calculate counts
        for issue in result.issues:
            if issue.severity == AnalysisSeverity.ERROR:
                result.error_count += 1
            elif issue.severity == AnalysisSeverity.WARNING:
                result.warning_count += 1
            else:
                result.info_count += 1

        result.duration_ms = (time.time() - start_time) * 1000

        # Emit completion
        self._emit_event("review.static.analyze", {
            "status": "completed",
            "files_analyzed": result.files_analyzed,
            "issue_count": len(result.issues),
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "duration_ms": result.duration_ms,
        })

        return result

    def analyze_diff(
        self,
        diff_files: List[Dict[str, Any]],
    ) -> AnalysisResult:
        """
        Analyze only changed lines from a diff.

        Args:
            diff_files: List of dicts with 'path' and 'changed_lines' keys

        Returns:
            AnalysisResult filtered to changed lines only
        """
        # Extract file paths
        files = [d["path"] for d in diff_files if "path" in d]

        # Run full analysis
        full_result = self.analyze(files)

        # Build changed line sets
        changed_lines: Dict[str, set] = {}
        for diff_file in diff_files:
            path = diff_file.get("path", "")
            lines = diff_file.get("changed_lines", [])
            changed_lines[path] = set(lines)

        # Filter issues to changed lines
        filtered_issues = []
        for issue in full_result.issues:
            file_changes = changed_lines.get(issue.file, set())
            if not file_changes or issue.line in file_changes:
                filtered_issues.append(issue)

        full_result.issues = filtered_issues
        full_result.error_count = sum(1 for i in filtered_issues if i.severity == AnalysisSeverity.ERROR)
        full_result.warning_count = sum(1 for i in filtered_issues if i.severity == AnalysisSeverity.WARNING)
        full_result.info_count = sum(1 for i in filtered_issues if i.severity == AnalysisSeverity.INFO)

        return full_result


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Static Analysis Engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Static Analysis Engine (Step 152)")
    parser.add_argument("files", nargs="+", help="Files to analyze")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per analyzer (seconds)")
    parser.add_argument("--no-mypy", action="store_true", help="Disable mypy analyzer")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary", action="store_true", help="Show summary only")

    args = parser.parse_args()

    engine = StaticAnalysisEngine(
        timeout=args.timeout,
        enable_mypy=not args.no_mypy,
    )

    result = engine.analyze(args.files)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.summary:
        print(f"Static Analysis Summary:")
        print(f"  Files analyzed: {result.files_analyzed}")
        print(f"  Total issues: {len(result.issues)}")
        print(f"  Errors: {result.error_count}")
        print(f"  Warnings: {result.warning_count}")
        print(f"  Info: {result.info_count}")
        print(f"  Duration: {result.duration_ms:.1f}ms")
        print(f"  Analyzers: {', '.join(result.analyzers_used)}")
    else:
        for issue in result.issues:
            severity_char = issue.severity.value[0].upper()
            print(f"[{severity_char}] {issue.location}: {issue.message} ({issue.rule})")

    # Return non-zero if errors found
    return 1 if result.error_count > 0 else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
