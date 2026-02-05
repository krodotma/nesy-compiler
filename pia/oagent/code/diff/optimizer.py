#!/usr/bin/env python3
"""
optimizer.py - Diff Optimizer (Step 61)

PBTSO Phase: VERIFY, ITERATE

Provides:
- Minimal diff generation between file versions
- Diff chunk optimization for reduced noise
- Context-aware diff formatting
- Semantic diff hints for code changes
- Unified and context diff formats

Bus Topics:
- code.diff.generate
- code.diff.optimize
- code.diff.ready

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import difflib
import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


# =============================================================================
# Types
# =============================================================================

class DiffFormat(Enum):
    """Output format for diffs."""
    UNIFIED = "unified"
    CONTEXT = "context"
    NDIFF = "ndiff"
    HTML = "html"
    JSON = "json"


class ChangeType(Enum):
    """Type of change in a diff chunk."""
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"
    MOVE = "move"
    RENAME = "rename"


@dataclass
class DiffLine:
    """Represents a single line in a diff."""
    content: str
    line_number_old: Optional[int] = None
    line_number_new: Optional[int] = None
    operation: str = " "  # ' ', '+', '-', '?'

    @property
    def is_addition(self) -> bool:
        return self.operation == "+"

    @property
    def is_deletion(self) -> bool:
        return self.operation == "-"

    @property
    def is_context(self) -> bool:
        return self.operation == " "


@dataclass
class DiffChunk:
    """
    A contiguous chunk of diff changes.

    Represents a hunk in unified diff terminology.
    """
    start_old: int
    count_old: int
    start_new: int
    count_new: int
    lines: List[DiffLine] = field(default_factory=list)
    change_type: ChangeType = ChangeType.MODIFY
    semantic_hint: Optional[str] = None  # e.g., "function definition changed"

    @property
    def header(self) -> str:
        """Generate unified diff hunk header."""
        return f"@@ -{self.start_old},{self.count_old} +{self.start_new},{self.count_new} @@"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_old": self.start_old,
            "count_old": self.count_old,
            "start_new": self.start_new,
            "count_new": self.count_new,
            "lines": [{"content": l.content, "op": l.operation} for l in self.lines],
            "change_type": self.change_type.value,
            "semantic_hint": self.semantic_hint,
        }


@dataclass
class OptimizedDiff:
    """
    An optimized diff between two file versions.

    Contains minimal changes needed to transform old to new.
    """
    id: str
    path: str
    old_hash: str
    new_hash: str
    chunks: List[DiffChunk]
    created_at: float = field(default_factory=time.time)
    lines_added: int = 0
    lines_deleted: int = 0
    lines_modified: int = 0
    optimization_applied: bool = False
    format: DiffFormat = DiffFormat.UNIFIED

    @property
    def total_changes(self) -> int:
        return self.lines_added + self.lines_deleted + self.lines_modified

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "chunks": [c.to_dict() for c in self.chunks],
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "lines_modified": self.lines_modified,
            "optimization_applied": self.optimization_applied,
            "format": self.format.value,
        }

    def to_unified(self, context_lines: int = 3) -> str:
        """Render as unified diff format."""
        lines = [
            f"--- a/{self.path}",
            f"+++ b/{self.path}",
        ]
        for chunk in self.chunks:
            lines.append(chunk.header)
            for diff_line in chunk.lines:
                lines.append(f"{diff_line.operation}{diff_line.content}")
        return "\n".join(lines)


# =============================================================================
# Diff Optimizer
# =============================================================================

class DiffOptimizer:
    """
    Generate and optimize minimal diffs between file versions.

    PBTSO Phase: VERIFY, ITERATE

    Features:
    - Minimal diff generation using sequence matching
    - Chunk merging to reduce fragmentation
    - Whitespace-aware diffing options
    - Semantic hints for code changes (function/class boundaries)
    - Multiple output formats

    Usage:
        optimizer = DiffOptimizer()
        diff = optimizer.generate_diff(old_content, new_content, "file.py")
        optimized = optimizer.optimize(diff)
    """

    BUS_TOPICS = {
        "generate": "code.diff.generate",
        "optimize": "code.diff.optimize",
        "ready": "code.diff.ready",
    }

    # Patterns for semantic hints
    SEMANTIC_PATTERNS = {
        "python": {
            "function": r"^\s*(?:async\s+)?def\s+(\w+)",
            "class": r"^\s*class\s+(\w+)",
            "method": r"^\s+(?:async\s+)?def\s+(\w+)",
        },
        "typescript": {
            "function": r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)",
            "class": r"^\s*(?:export\s+)?class\s+(\w+)",
            "method": r"^\s+(?:async\s+)?(\w+)\s*\(",
        },
    }

    def __init__(
        self,
        bus: Optional[Any] = None,
        context_lines: int = 3,
        ignore_whitespace: bool = False,
        ignore_blank_lines: bool = False,
        merge_adjacent_chunks: bool = True,
        merge_threshold: int = 3,
    ):
        self.bus = bus
        self.context_lines = context_lines
        self.ignore_whitespace = ignore_whitespace
        self.ignore_blank_lines = ignore_blank_lines
        self.merge_adjacent_chunks = merge_adjacent_chunks
        self.merge_threshold = merge_threshold
        self._diffs: Dict[str, OptimizedDiff] = {}

    # =========================================================================
    # Diff Generation
    # =========================================================================

    def generate_diff(
        self,
        old_content: str,
        new_content: str,
        path: str,
        language: Optional[str] = None,
    ) -> OptimizedDiff:
        """
        Generate a diff between old and new content.

        Args:
            old_content: Original file content
            new_content: Modified file content
            path: File path (for header and language detection)
            language: Programming language (auto-detected if not provided)

        Returns:
            OptimizedDiff with all changes
        """
        start_time = time.time()

        # Detect language from extension
        if language is None:
            ext = Path(path).suffix.lower()
            language = {".py": "python", ".ts": "typescript", ".tsx": "typescript",
                       ".js": "javascript"}.get(ext, "text")

        # Preprocess content if needed
        old_lines = self._preprocess(old_content)
        new_lines = self._preprocess(new_content)

        # Generate sequence matcher
        matcher = difflib.SequenceMatcher(
            isjunk=self._is_junk_line if self.ignore_blank_lines else None,
            a=old_lines,
            b=new_lines,
        )

        # Extract opcodes and build chunks
        chunks = self._build_chunks(matcher, old_lines, new_lines, language)

        # Calculate statistics
        lines_added = sum(l.is_addition for c in chunks for l in c.lines)
        lines_deleted = sum(l.is_deletion for c in chunks for l in c.lines)
        lines_modified = min(lines_added, lines_deleted)

        diff = OptimizedDiff(
            id=f"diff-{uuid.uuid4().hex[:8]}",
            path=path,
            old_hash=self._hash_content(old_content),
            new_hash=self._hash_content(new_content),
            chunks=chunks,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            lines_modified=lines_modified,
            optimization_applied=False,
        )

        self._diffs[diff.id] = diff

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["generate"],
                "kind": "diff",
                "actor": "code-agent",
                "data": {
                    "diff_id": diff.id,
                    "path": path,
                    "chunks": len(chunks),
                    "lines_changed": diff.total_changes,
                    "elapsed_ms": (time.time() - start_time) * 1000,
                },
            })

        return diff

    def generate_diff_from_files(
        self,
        old_path: Path,
        new_path: Path,
        output_path: Optional[str] = None,
    ) -> OptimizedDiff:
        """Generate diff between two files."""
        old_content = old_path.read_text() if old_path.exists() else ""
        new_content = new_path.read_text() if new_path.exists() else ""
        path = output_path or str(new_path)
        return self.generate_diff(old_content, new_content, path)

    def _preprocess(self, content: str) -> List[str]:
        """Preprocess content for diffing."""
        lines = content.splitlines(keepends=True)

        if self.ignore_whitespace:
            lines = [re.sub(r'\s+', ' ', line.strip()) + '\n' for line in lines]

        return lines

    def _is_junk_line(self, line: str) -> bool:
        """Check if a line should be treated as junk."""
        return line.strip() == ""

    def _hash_content(self, content: str) -> str:
        """Generate hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _build_chunks(
        self,
        matcher: difflib.SequenceMatcher,
        old_lines: List[str],
        new_lines: List[str],
        language: str,
    ) -> List[DiffChunk]:
        """Build diff chunks from sequence matcher opcodes."""
        chunks: List[DiffChunk] = []
        grouped_opcodes = matcher.get_grouped_opcodes(self.context_lines)

        for group in grouped_opcodes:
            chunk_lines: List[DiffLine] = []
            start_old = group[0][1]
            end_old = group[-1][2]
            start_new = group[0][3]
            end_new = group[-1][4]

            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    for i, line in enumerate(old_lines[i1:i2]):
                        chunk_lines.append(DiffLine(
                            content=line.rstrip('\n'),
                            line_number_old=i1 + i + 1,
                            line_number_new=j1 + i + 1,
                            operation=" ",
                        ))
                elif tag == "replace":
                    for i, line in enumerate(old_lines[i1:i2]):
                        chunk_lines.append(DiffLine(
                            content=line.rstrip('\n'),
                            line_number_old=i1 + i + 1,
                            operation="-",
                        ))
                    for j, line in enumerate(new_lines[j1:j2]):
                        chunk_lines.append(DiffLine(
                            content=line.rstrip('\n'),
                            line_number_new=j1 + j + 1,
                            operation="+",
                        ))
                elif tag == "delete":
                    for i, line in enumerate(old_lines[i1:i2]):
                        chunk_lines.append(DiffLine(
                            content=line.rstrip('\n'),
                            line_number_old=i1 + i + 1,
                            operation="-",
                        ))
                elif tag == "insert":
                    for j, line in enumerate(new_lines[j1:j2]):
                        chunk_lines.append(DiffLine(
                            content=line.rstrip('\n'),
                            line_number_new=j1 + j + 1,
                            operation="+",
                        ))

            # Determine change type
            has_adds = any(l.is_addition for l in chunk_lines)
            has_dels = any(l.is_deletion for l in chunk_lines)
            if has_adds and has_dels:
                change_type = ChangeType.MODIFY
            elif has_adds:
                change_type = ChangeType.ADD
            else:
                change_type = ChangeType.DELETE

            # Generate semantic hint
            semantic_hint = self._generate_semantic_hint(
                chunk_lines, old_lines, start_old, language
            )

            chunk = DiffChunk(
                start_old=start_old + 1,
                count_old=end_old - start_old,
                start_new=start_new + 1,
                count_new=end_new - start_new,
                lines=chunk_lines,
                change_type=change_type,
                semantic_hint=semantic_hint,
            )
            chunks.append(chunk)

        return chunks

    def _generate_semantic_hint(
        self,
        chunk_lines: List[DiffLine],
        all_lines: List[str],
        start_line: int,
        language: str,
    ) -> Optional[str]:
        """Generate semantic hint for a chunk based on context."""
        patterns = self.SEMANTIC_PATTERNS.get(language, {})
        if not patterns:
            return None

        # Look backwards for function/class definition
        for i in range(start_line, max(0, start_line - 50), -1):
            if i < len(all_lines):
                line = all_lines[i]
                for kind, pattern in patterns.items():
                    match = re.match(pattern, line)
                    if match:
                        return f"In {kind} '{match.group(1)}'"

        return None

    # =========================================================================
    # Optimization
    # =========================================================================

    def optimize(self, diff: OptimizedDiff) -> OptimizedDiff:
        """
        Optimize a diff to minimize noise.

        Optimizations:
        - Merge adjacent chunks within threshold
        - Remove redundant context lines
        - Consolidate whitespace-only changes
        """
        if diff.optimization_applied:
            return diff

        start_time = time.time()
        optimized_chunks = diff.chunks

        # Merge adjacent chunks if enabled
        if self.merge_adjacent_chunks:
            optimized_chunks = self._merge_chunks(optimized_chunks)

        # Remove trailing whitespace changes
        optimized_chunks = self._filter_trivial_changes(optimized_chunks)

        diff.chunks = optimized_chunks
        diff.optimization_applied = True

        # Recalculate stats
        diff.lines_added = sum(l.is_addition for c in diff.chunks for l in c.lines)
        diff.lines_deleted = sum(l.is_deletion for c in diff.chunks for l in c.lines)
        diff.lines_modified = min(diff.lines_added, diff.lines_deleted)

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["optimize"],
                "kind": "diff",
                "actor": "code-agent",
                "data": {
                    "diff_id": diff.id,
                    "chunks_after": len(optimized_chunks),
                    "elapsed_ms": (time.time() - start_time) * 1000,
                },
            })

        return diff

    def _merge_chunks(self, chunks: List[DiffChunk]) -> List[DiffChunk]:
        """Merge adjacent chunks within threshold."""
        if len(chunks) <= 1:
            return chunks

        merged: List[DiffChunk] = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            gap = next_chunk.start_old - (current.start_old + current.count_old)

            if gap <= self.merge_threshold:
                # Merge chunks
                # Add context lines for the gap
                gap_lines = [
                    DiffLine(content="...", operation=" ")
                    for _ in range(min(gap, 1))
                ]
                current = DiffChunk(
                    start_old=current.start_old,
                    count_old=next_chunk.start_old + next_chunk.count_old - current.start_old,
                    start_new=current.start_new,
                    count_new=next_chunk.start_new + next_chunk.count_new - current.start_new,
                    lines=current.lines + gap_lines + next_chunk.lines,
                    change_type=ChangeType.MODIFY,
                    semantic_hint=current.semantic_hint or next_chunk.semantic_hint,
                )
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged

    def _filter_trivial_changes(self, chunks: List[DiffChunk]) -> List[DiffChunk]:
        """Filter out trivial whitespace-only changes."""
        filtered = []
        for chunk in chunks:
            # Check if all changes are whitespace-only
            significant_changes = False
            for line in chunk.lines:
                if line.operation in ("+", "-"):
                    if line.content.strip():
                        significant_changes = True
                        break

            if significant_changes or not self.ignore_whitespace:
                filtered.append(chunk)

        return filtered

    # =========================================================================
    # Output Formatting
    # =========================================================================

    def format_diff(
        self,
        diff: OptimizedDiff,
        format: DiffFormat = DiffFormat.UNIFIED,
    ) -> str:
        """Format diff for output."""
        diff.format = format

        if format == DiffFormat.UNIFIED:
            return diff.to_unified(self.context_lines)

        elif format == DiffFormat.CONTEXT:
            return self._format_context(diff)

        elif format == DiffFormat.NDIFF:
            return self._format_ndiff(diff)

        elif format == DiffFormat.HTML:
            return self._format_html(diff)

        elif format == DiffFormat.JSON:
            import json
            return json.dumps(diff.to_dict(), indent=2)

        return diff.to_unified()

    def _format_context(self, diff: OptimizedDiff) -> str:
        """Format as context diff."""
        lines = [
            f"*** a/{diff.path}",
            f"--- b/{diff.path}",
        ]
        for chunk in diff.chunks:
            lines.append("***************")
            lines.append(f"*** {chunk.start_old},{chunk.start_old + chunk.count_old - 1} ****")
            for line in chunk.lines:
                if line.is_deletion:
                    lines.append(f"- {line.content}")
                elif line.is_context:
                    lines.append(f"  {line.content}")
            lines.append(f"--- {chunk.start_new},{chunk.start_new + chunk.count_new - 1} ----")
            for line in chunk.lines:
                if line.is_addition:
                    lines.append(f"+ {line.content}")
                elif line.is_context:
                    lines.append(f"  {line.content}")
        return "\n".join(lines)

    def _format_ndiff(self, diff: OptimizedDiff) -> str:
        """Format as ndiff (more detailed character-level diff)."""
        lines = []
        for chunk in diff.chunks:
            for line in chunk.lines:
                lines.append(f"{line.operation} {line.content}")
        return "\n".join(lines)

    def _format_html(self, diff: OptimizedDiff) -> str:
        """Format as HTML for web display."""
        html = ['<div class="diff">']
        html.append(f'<div class="diff-header">--- a/{diff.path}<br>+++ b/{diff.path}</div>')

        for chunk in diff.chunks:
            html.append(f'<div class="diff-chunk">')
            html.append(f'<div class="chunk-header">{chunk.header}</div>')
            for line in chunk.lines:
                css_class = {
                    "+": "addition",
                    "-": "deletion",
                    " ": "context",
                }.get(line.operation, "context")
                escaped = line.content.replace("<", "&lt;").replace(">", "&gt;")
                html.append(f'<div class="line {css_class}">{line.operation}{escaped}</div>')
            html.append('</div>')

        html.append('</div>')
        return "\n".join(html)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_diff(self, diff_id: str) -> Optional[OptimizedDiff]:
        """Get a diff by ID."""
        return self._diffs.get(diff_id)

    def list_diffs(self) -> List[OptimizedDiff]:
        """List all generated diffs."""
        return list(self._diffs.values())

    def apply_diff(self, diff: OptimizedDiff, old_content: str) -> str:
        """Apply a diff to reconstruct new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines: List[str] = []

        old_idx = 0
        for chunk in diff.chunks:
            # Add unchanged lines before chunk
            while old_idx < chunk.start_old - 1:
                if old_idx < len(old_lines):
                    new_lines.append(old_lines[old_idx])
                old_idx += 1

            # Process chunk lines
            for line in chunk.lines:
                if line.is_context or line.is_addition:
                    new_lines.append(line.content + "\n")
                if line.is_deletion or line.is_context:
                    old_idx += 1

        # Add remaining lines
        while old_idx < len(old_lines):
            new_lines.append(old_lines[old_idx])
            old_idx += 1

        return "".join(new_lines)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Diff Optimizer."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Diff Optimizer (Step 61)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # diff command
    diff_parser = subparsers.add_parser("diff", help="Generate diff between files")
    diff_parser.add_argument("old_file", help="Old file path")
    diff_parser.add_argument("new_file", help="New file path")
    diff_parser.add_argument("--format", choices=["unified", "context", "ndiff", "html", "json"],
                            default="unified", help="Output format")
    diff_parser.add_argument("--context", type=int, default=3, help="Context lines")
    diff_parser.add_argument("--optimize", action="store_true", help="Apply optimizations")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show diff statistics")
    stats_parser.add_argument("old_file", help="Old file path")
    stats_parser.add_argument("new_file", help="New file path")

    args = parser.parse_args()

    if args.command == "diff":
        optimizer = DiffOptimizer(context_lines=args.context)
        diff = optimizer.generate_diff_from_files(
            Path(args.old_file),
            Path(args.new_file),
        )

        if args.optimize:
            diff = optimizer.optimize(diff)

        output = optimizer.format_diff(diff, DiffFormat(args.format))
        print(output)
        return 0

    elif args.command == "stats":
        optimizer = DiffOptimizer()
        diff = optimizer.generate_diff_from_files(
            Path(args.old_file),
            Path(args.new_file),
        )

        print(f"Diff ID: {diff.id}")
        print(f"Path: {diff.path}")
        print(f"Chunks: {len(diff.chunks)}")
        print(f"Lines added: {diff.lines_added}")
        print(f"Lines deleted: {diff.lines_deleted}")
        print(f"Lines modified: {diff.lines_modified}")
        print(f"Total changes: {diff.total_changes}")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
