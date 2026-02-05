#!/usr/bin/env python3
"""
Review Diff Analyzer (Step 165)

Analyzes diffs to extract meaningful changes, identify patterns,
and provide context for code review.

PBTSO Phase: RESEARCH, VERIFY
Bus Topics: review.diff.analyze, review.diff.context, review.diff.patterns

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

class ChangeType(Enum):
    """Types of changes in a diff."""
    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"
    RENAMED = "renamed"
    COPIED = "copied"
    MODE_CHANGE = "mode_change"


class HunkType(Enum):
    """Types of diff hunks."""
    FUNCTION_ADDED = "function_added"
    FUNCTION_MODIFIED = "function_modified"
    FUNCTION_DELETED = "function_deleted"
    CLASS_ADDED = "class_added"
    CLASS_MODIFIED = "class_modified"
    IMPORT_CHANGE = "import_change"
    CONFIG_CHANGE = "config_change"
    COMMENT_CHANGE = "comment_change"
    WHITESPACE = "whitespace"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk level of changes."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class DiffLine:
    """A single line in a diff."""
    line_type: str  # '+', '-', ' '
    content: str
    old_line_num: Optional[int]
    new_line_num: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DiffHunk:
    """
    A contiguous section of changes in a diff.

    Attributes:
        hunk_id: Unique hunk identifier
        file_path: Path to the file
        old_start: Start line in old file
        old_count: Number of lines in old file
        new_start: Start line in new file
        new_count: Number of lines in new file
        lines: Lines in the hunk
        hunk_type: Classified type of change
        context: Surrounding context
        risk_level: Assessed risk level
        function_name: Enclosing function if detected
        class_name: Enclosing class if detected
    """
    hunk_id: str
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[DiffLine] = field(default_factory=list)
    hunk_type: HunkType = HunkType.UNKNOWN
    context: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    function_name: Optional[str] = None
    class_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["hunk_type"] = self.hunk_type.value
        result["risk_level"] = self.risk_level.value
        result["lines"] = [l.to_dict() if isinstance(l, DiffLine) else l for l in self.lines]
        return result

    @property
    def additions(self) -> int:
        """Count of added lines."""
        return sum(1 for l in self.lines if l.line_type == "+")

    @property
    def deletions(self) -> int:
        """Count of deleted lines."""
        return sum(1 for l in self.lines if l.line_type == "-")


@dataclass
class FileDiff:
    """
    Diff for a single file.

    Attributes:
        file_path: Path to the file
        change_type: Type of change
        old_path: Previous path if renamed
        hunks: List of diff hunks
        additions: Total lines added
        deletions: Total lines deleted
        binary: Whether file is binary
        language: Detected language
    """
    file_path: str
    change_type: ChangeType
    old_path: Optional[str] = None
    hunks: List[DiffHunk] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    binary: bool = False
    language: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["change_type"] = self.change_type.value
        result["hunks"] = [h.to_dict() for h in self.hunks]
        return result


@dataclass
class DiffAnalysis:
    """
    Complete diff analysis result.

    Attributes:
        analysis_id: Unique analysis ID
        files: List of file diffs
        total_additions: Total lines added
        total_deletions: Total lines deleted
        total_files: Number of files changed
        risk_summary: Summary of risk levels
        patterns: Detected patterns
        stats: Additional statistics
        duration_ms: Analysis duration
    """
    analysis_id: str
    files: List[FileDiff] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    total_files: int = 0
    risk_summary: Dict[str, int] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "files": [f.to_dict() for f in self.files],
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "total_files": self.total_files,
            "risk_summary": self.risk_summary,
            "patterns": self.patterns,
            "stats": self.stats,
            "duration_ms": self.duration_ms,
        }

    @property
    def high_risk_files(self) -> List[FileDiff]:
        """Get files with high-risk changes."""
        return [
            f for f in self.files
            if any(h.risk_level == RiskLevel.HIGH for h in f.hunks)
        ]


# ============================================================================
# Diff Analyzer
# ============================================================================

class DiffAnalyzer:
    """
    Analyzes diffs to extract meaningful information.

    Parses unified diff format, classifies changes, and assesses risk.

    Example:
        analyzer = DiffAnalyzer()

        # Analyze git diff
        analysis = analyzer.analyze_git_diff("main", "HEAD")

        # Or analyze raw diff
        analysis = analyzer.analyze_diff(diff_text)

        for file_diff in analysis.files:
            print(f"{file_diff.file_path}: +{file_diff.additions} -{file_diff.deletions}")
    """

    BUS_TOPICS = {
        "analyze": "review.diff.analyze",
        "context": "review.diff.context",
        "patterns": "review.diff.patterns",
    }

    # Patterns for detecting high-risk changes
    HIGH_RISK_PATTERNS = [
        r"(?:password|secret|api_key|token)\s*=",
        r"(?:exec|eval)\s*\(",
        r"subprocess\.(?:call|run|Popen)",
        r"\.(?:remove|delete|drop)\s*\(",
        r"os\.(?:remove|rmdir|system)",
    ]

    # Patterns for function/class detection
    FUNCTION_PATTERN = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(")
    CLASS_PATTERN = re.compile(r"^class\s+(\w+)")
    IMPORT_PATTERN = re.compile(r"^(?:from|import)\s+")

    def __init__(self, bus_path: Optional[Path] = None):
        """
        Initialize the diff analyzer.

        Args:
            bus_path: Path to event bus file
        """
        self.bus_path = bus_path or self._get_bus_path()
        self._compiled_risk_patterns = [re.compile(p, re.IGNORECASE) for p in self.HIGH_RISK_PATTERNS]

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "diff") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "diff-analyzer",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def analyze_git_diff(
        self,
        base_ref: str = "main",
        head_ref: str = "HEAD",
        repo_path: Optional[str] = None,
    ) -> DiffAnalysis:
        """
        Analyze diff between two git refs.

        Args:
            base_ref: Base reference (branch, tag, SHA)
            head_ref: Head reference
            repo_path: Repository path (defaults to cwd)

        Returns:
            DiffAnalysis result
        """
        cmd = ["git", "diff", "--unified=3", f"{base_ref}...{head_ref}"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=repo_path,
            )

            if result.returncode == 0:
                return self.analyze_diff(result.stdout)
            else:
                # Try alternative format
                cmd = ["git", "diff", "--unified=3", base_ref, head_ref]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=repo_path,
                )
                return self.analyze_diff(result.stdout)

        except subprocess.TimeoutExpired:
            return DiffAnalysis(analysis_id=str(uuid.uuid4())[:8])

    def analyze_diff(self, diff_text: str) -> DiffAnalysis:
        """
        Analyze a unified diff text.

        Args:
            diff_text: Unified diff text

        Returns:
            DiffAnalysis result

        Emits:
            review.diff.analyze
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())[:8]

        self._emit_event(self.BUS_TOPICS["analyze"], {
            "analysis_id": analysis_id,
            "status": "started",
        })

        # Parse the diff
        files = self._parse_diff(diff_text)

        # Classify hunks and assess risk
        for file_diff in files:
            for hunk in file_diff.hunks:
                hunk.hunk_type = self._classify_hunk(hunk)
                hunk.risk_level = self._assess_risk(hunk)

        # Calculate totals
        total_additions = sum(f.additions for f in files)
        total_deletions = sum(f.deletions for f in files)

        # Risk summary
        risk_summary = {level.value: 0 for level in RiskLevel}
        for file_diff in files:
            for hunk in file_diff.hunks:
                risk_summary[hunk.risk_level.value] += 1

        # Detect patterns
        patterns = self._detect_patterns(files)

        analysis = DiffAnalysis(
            analysis_id=analysis_id,
            files=files,
            total_additions=total_additions,
            total_deletions=total_deletions,
            total_files=len(files),
            risk_summary=risk_summary,
            patterns=patterns,
            stats={
                "avg_hunk_size": sum(h.additions + h.deletions for f in files for h in f.hunks) / max(1, sum(len(f.hunks) for f in files)),
                "files_by_language": self._count_by_language(files),
            },
            duration_ms=(time.time() - start_time) * 1000,
        )

        self._emit_event(self.BUS_TOPICS["analyze"], {
            "analysis_id": analysis_id,
            "total_files": len(files),
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "high_risk_count": risk_summary.get("high", 0),
            "status": "completed",
        })

        if patterns:
            self._emit_event(self.BUS_TOPICS["patterns"], {
                "analysis_id": analysis_id,
                "patterns": patterns,
            })

        return analysis

    def _parse_diff(self, diff_text: str) -> List[FileDiff]:
        """Parse unified diff into structured format."""
        files = []
        current_file: Optional[FileDiff] = None
        current_hunk: Optional[DiffHunk] = None
        hunk_counter = 0

        for line in diff_text.split("\n"):
            # New file header
            if line.startswith("diff --git"):
                if current_file:
                    files.append(current_file)

                # Extract file path
                match = re.search(r"diff --git a/(.+?) b/(.+?)$", line)
                if match:
                    old_path = match.group(1)
                    new_path = match.group(2)
                    current_file = FileDiff(
                        file_path=new_path,
                        change_type=ChangeType.MODIFIED,
                        old_path=old_path if old_path != new_path else None,
                        language=self._detect_language(new_path),
                    )
                    current_hunk = None

            # New file indicator
            elif line.startswith("new file"):
                if current_file:
                    current_file.change_type = ChangeType.ADDED

            # Deleted file indicator
            elif line.startswith("deleted file"):
                if current_file:
                    current_file.change_type = ChangeType.DELETED

            # Renamed file indicator
            elif line.startswith("rename from") or line.startswith("rename to"):
                if current_file:
                    current_file.change_type = ChangeType.RENAMED

            # Binary file
            elif line.startswith("Binary files"):
                if current_file:
                    current_file.binary = True

            # Hunk header
            elif line.startswith("@@"):
                if current_hunk and current_file:
                    current_file.hunks.append(current_hunk)

                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:\s+(.*))?", line)
                if match and current_file:
                    hunk_counter += 1
                    current_hunk = DiffHunk(
                        hunk_id=f"h{hunk_counter}",
                        file_path=current_file.file_path,
                        old_start=int(match.group(1)),
                        old_count=int(match.group(2) or 1),
                        new_start=int(match.group(3)),
                        new_count=int(match.group(4) or 1),
                        context=match.group(5) or "",
                    )

            # Diff lines
            elif current_hunk and (line.startswith("+") or line.startswith("-") or line.startswith(" ")):
                line_type = line[0]
                content = line[1:] if len(line) > 1 else ""

                diff_line = DiffLine(
                    line_type=line_type,
                    content=content,
                    old_line_num=None,  # Would need to track
                    new_line_num=None,
                )
                current_hunk.lines.append(diff_line)

                if line_type == "+":
                    if current_file:
                        current_file.additions += 1
                elif line_type == "-":
                    if current_file:
                        current_file.deletions += 1

        # Add last file and hunk
        if current_hunk and current_file:
            current_file.hunks.append(current_hunk)
        if current_file:
            files.append(current_file)

        return files

    def _classify_hunk(self, hunk: DiffHunk) -> HunkType:
        """Classify the type of a hunk."""
        added_lines = [l.content for l in hunk.lines if l.line_type == "+"]
        deleted_lines = [l.content for l in hunk.lines if l.line_type == "-"]
        all_content = "\n".join(added_lines + deleted_lines)

        # Check for function changes
        for line in added_lines:
            if self.FUNCTION_PATTERN.match(line.strip()):
                return HunkType.FUNCTION_ADDED

        for line in deleted_lines:
            if self.FUNCTION_PATTERN.match(line.strip()):
                return HunkType.FUNCTION_DELETED

        if any(self.FUNCTION_PATTERN.match(l.strip()) for l in added_lines + deleted_lines):
            return HunkType.FUNCTION_MODIFIED

        # Check for class changes
        if any(self.CLASS_PATTERN.match(l.strip()) for l in added_lines):
            return HunkType.CLASS_ADDED
        if any(self.CLASS_PATTERN.match(l.strip()) for l in added_lines + deleted_lines):
            return HunkType.CLASS_MODIFIED

        # Check for import changes
        if all(self.IMPORT_PATTERN.match(l.strip()) for l in added_lines + deleted_lines if l.strip()):
            return HunkType.IMPORT_CHANGE

        # Check for config changes
        if hunk.file_path.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".conf")):
            return HunkType.CONFIG_CHANGE

        # Check for comment changes
        if all(l.strip().startswith(("#", "//", "/*", "*", "'''", '"""')) for l in added_lines + deleted_lines if l.strip()):
            return HunkType.COMMENT_CHANGE

        # Check for whitespace only
        if all(not l.strip() for l in added_lines + deleted_lines):
            return HunkType.WHITESPACE

        return HunkType.UNKNOWN

    def _assess_risk(self, hunk: DiffHunk) -> RiskLevel:
        """Assess the risk level of a hunk."""
        content = "\n".join(l.content for l in hunk.lines if l.line_type == "+")

        # Check for high-risk patterns
        for pattern in self._compiled_risk_patterns:
            if pattern.search(content):
                return RiskLevel.HIGH

        # Size-based risk
        if hunk.additions > 100 or hunk.deletions > 100:
            return RiskLevel.MEDIUM

        # Whitespace only is minimal risk
        if hunk.hunk_type == HunkType.WHITESPACE:
            return RiskLevel.MINIMAL

        # Comment only is low risk
        if hunk.hunk_type == HunkType.COMMENT_CHANGE:
            return RiskLevel.LOW

        return RiskLevel.MEDIUM

    def _detect_patterns(self, files: List[FileDiff]) -> List[str]:
        """Detect patterns in the changes."""
        patterns = []

        # Large refactoring
        if len(files) > 10:
            patterns.append("large_refactoring")

        # New feature
        if any(f.change_type == ChangeType.ADDED and not f.file_path.endswith("_test.py") for f in files):
            patterns.append("new_feature")

        # Test additions
        if any("test" in f.file_path.lower() and f.change_type == ChangeType.ADDED for f in files):
            patterns.append("test_additions")

        # Configuration changes
        if any(f.file_path.endswith((".json", ".yaml", ".yml", ".toml")) for f in files):
            patterns.append("config_changes")

        # Dependency changes
        if any(f.file_path in ("package.json", "requirements.txt", "pyproject.toml", "Cargo.toml") for f in files):
            patterns.append("dependency_changes")

        return patterns

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file path."""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "unknown")

    def _count_by_language(self, files: List[FileDiff]) -> Dict[str, int]:
        """Count files by language."""
        counts: Dict[str, int] = {}
        for f in files:
            counts[f.language] = counts.get(f.language, 0) + 1
        return counts


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Diff Analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Diff Analyzer (Step 165)")
    parser.add_argument("--base", default="main", help="Base ref")
    parser.add_argument("--head", default="HEAD", help="Head ref")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--files", action="store_true", help="Show files only")

    args = parser.parse_args()

    analyzer = DiffAnalyzer()
    analysis = analyzer.analyze_git_diff(args.base, args.head)

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
    elif args.files:
        for f in analysis.files:
            risk = max((h.risk_level for h in f.hunks), default=RiskLevel.MINIMAL, key=lambda r: list(RiskLevel).index(r))
            print(f"[{risk.value:^6}] {f.file_path} (+{f.additions} -{f.deletions})")
    else:
        print(f"Diff Analysis")
        print(f"  Files: {analysis.total_files}")
        print(f"  Additions: {analysis.total_additions}")
        print(f"  Deletions: {analysis.total_deletions}")
        print(f"  Risk Summary: {analysis.risk_summary}")
        if analysis.patterns:
            print(f"  Patterns: {', '.join(analysis.patterns)}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
