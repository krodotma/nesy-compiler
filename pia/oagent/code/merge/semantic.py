#!/usr/bin/env python3
"""
semantic.py - Semantic Merger (Step 63)

PBTSO Phase: ITERATE, DISTRIBUTE

Provides:
- Context-aware code merging using AST analysis
- Function/class-level merge granularity
- Import deduplication and ordering
- Docstring and comment preservation
- Semantic equivalence detection

Bus Topics:
- code.merge.start
- code.merge.complete
- code.merge.conflict

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import ast
import difflib
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# =============================================================================
# Types
# =============================================================================

class MergeStrategy(Enum):
    """Strategy for merging code blocks."""
    PRESERVE_BOTH = "preserve_both"    # Keep both versions
    PREFER_BASE = "prefer_base"        # Prefer base version
    PREFER_INCOMING = "prefer_incoming" # Prefer incoming version
    INTERLEAVE = "interleave"          # Interleave changes
    SEMANTIC_MERGE = "semantic_merge"  # AST-aware merge


class BlockType(Enum):
    """Type of code block."""
    IMPORT = "import"
    CLASS = "class"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    VARIABLE = "variable"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    OTHER = "other"


@dataclass
class CodeBlock:
    """
    A semantic code block (function, class, import group, etc.)
    """
    name: str
    block_type: BlockType
    source: str
    start_line: int
    end_line: int
    signature: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    hash: Optional[str] = None

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def semantic_equals(self, other: "CodeBlock") -> bool:
        """Check if blocks are semantically equivalent."""
        if self.block_type != other.block_type:
            return False
        if self.name != other.name:
            return False
        # Normalize whitespace for comparison
        self_normalized = re.sub(r'\s+', ' ', self.source.strip())
        other_normalized = re.sub(r'\s+', ' ', other.source.strip())
        return self_normalized == other_normalized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "block_type": self.block_type.value,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_count": self.line_count,
            "signature": self.signature,
            "dependencies": self.dependencies,
        }


@dataclass
class MergeContext:
    """Context for a merge operation."""
    base_path: str
    incoming_path: Optional[str] = None
    output_path: Optional[str] = None
    preserve_comments: bool = True
    preserve_formatting: bool = True
    sort_imports: bool = True
    remove_duplicate_imports: bool = True
    language: str = "python"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_path": self.base_path,
            "incoming_path": self.incoming_path,
            "output_path": self.output_path,
            "language": self.language,
        }


@dataclass
class MergeResult:
    """Result of a merge operation."""
    id: str
    success: bool
    merged_content: str
    base_blocks: int
    incoming_blocks: int
    merged_blocks: int
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "success": self.success,
            "base_blocks": self.base_blocks,
            "incoming_blocks": self.incoming_blocks,
            "merged_blocks": self.merged_blocks,
            "has_conflicts": self.has_conflicts,
            "conflict_count": len(self.conflicts),
            "warnings": self.warnings,
            "elapsed_ms": self.elapsed_ms,
        }


# =============================================================================
# Semantic Merger
# =============================================================================

class SemanticMerger:
    """
    Context-aware code merging using semantic analysis.

    PBTSO Phase: ITERATE, DISTRIBUTE

    Features:
    - AST-based code block extraction
    - Function/class-level merge granularity
    - Intelligent import merging
    - Docstring preservation
    - Semantic equivalence detection

    Usage:
        merger = SemanticMerger()
        result = merger.merge(base_content, incoming_content, context)
    """

    BUS_TOPICS = {
        "start": "code.merge.start",
        "complete": "code.merge.complete",
        "conflict": "code.merge.conflict",
    }

    def __init__(
        self,
        bus: Optional[Any] = None,
        default_strategy: MergeStrategy = MergeStrategy.SEMANTIC_MERGE,
    ):
        self.bus = bus
        self.default_strategy = default_strategy
        self._results: Dict[str, MergeResult] = {}

    # =========================================================================
    # Main Merge Operation
    # =========================================================================

    def merge(
        self,
        base_content: str,
        incoming_content: str,
        context: MergeContext,
        strategy: Optional[MergeStrategy] = None,
    ) -> MergeResult:
        """
        Merge base and incoming content semantically.

        Args:
            base_content: Original file content
            incoming_content: Content to merge in
            context: Merge context and options
            strategy: Merge strategy to use

        Returns:
            MergeResult with merged content
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy

        # Emit start event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["start"],
                "kind": "merge",
                "actor": "code-agent",
                "data": {
                    "path": context.base_path,
                    "strategy": strategy.value,
                },
            })

        try:
            # Extract code blocks from both versions
            base_blocks = self._extract_blocks(base_content, context.language)
            incoming_blocks = self._extract_blocks(incoming_content, context.language)

            # Perform the merge
            merged_content, conflicts, warnings = self._merge_blocks(
                base_blocks, incoming_blocks, strategy, context
            )

            result = MergeResult(
                id=f"merge-{uuid.uuid4().hex[:8]}",
                success=len(conflicts) == 0,
                merged_content=merged_content,
                base_blocks=len(base_blocks),
                incoming_blocks=len(incoming_blocks),
                merged_blocks=self._count_blocks(merged_content, context.language),
                conflicts=conflicts,
                warnings=warnings,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            result = MergeResult(
                id=f"merge-{uuid.uuid4().hex[:8]}",
                success=False,
                merged_content=base_content,
                base_blocks=0,
                incoming_blocks=0,
                merged_blocks=0,
                warnings=[f"Merge failed: {e}"],
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        self._results[result.id] = result

        # Emit completion event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["complete"],
                "kind": "merge",
                "actor": "code-agent",
                "data": result.to_dict(),
            })

        return result

    # =========================================================================
    # Block Extraction
    # =========================================================================

    def _extract_blocks(self, content: str, language: str) -> List[CodeBlock]:
        """Extract semantic code blocks from content."""
        if language == "python":
            return self._extract_python_blocks(content)
        else:
            # Fallback to line-based extraction
            return self._extract_generic_blocks(content)

    def _extract_python_blocks(self, content: str) -> List[CodeBlock]:
        """Extract code blocks from Python source."""
        blocks: List[CodeBlock] = []
        lines = content.splitlines()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._extract_generic_blocks(content)

        # Track imports
        import_lines: List[int] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(node.lineno)

            elif isinstance(node, ast.ClassDef):
                start = node.lineno
                end = node.end_lineno or start
                source = "\n".join(lines[start - 1:end])

                # Extract method signatures
                methods = [n.name for n in ast.walk(node)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

                blocks.append(CodeBlock(
                    name=node.name,
                    block_type=BlockType.CLASS,
                    source=source,
                    start_line=start,
                    end_line=end,
                    signature=f"class {node.name}",
                    dependencies=methods,
                ))

            elif isinstance(node, ast.FunctionDef):
                start = node.lineno
                end = node.end_lineno or start
                source = "\n".join(lines[start - 1:end])

                # Build signature
                args = [a.arg for a in node.args.args]
                sig = f"def {node.name}({', '.join(args)})"

                blocks.append(CodeBlock(
                    name=node.name,
                    block_type=BlockType.FUNCTION,
                    source=source,
                    start_line=start,
                    end_line=end,
                    signature=sig,
                ))

            elif isinstance(node, ast.AsyncFunctionDef):
                start = node.lineno
                end = node.end_lineno or start
                source = "\n".join(lines[start - 1:end])

                args = [a.arg for a in node.args.args]
                sig = f"async def {node.name}({', '.join(args)})"

                blocks.append(CodeBlock(
                    name=node.name,
                    block_type=BlockType.ASYNC_FUNCTION,
                    source=source,
                    start_line=start,
                    end_line=end,
                    signature=sig,
                ))

            elif isinstance(node, ast.Assign):
                start = node.lineno
                end = node.end_lineno or start
                source = "\n".join(lines[start - 1:end])

                # Get variable names
                names = []
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)

                if names:
                    blocks.append(CodeBlock(
                        name=names[0],
                        block_type=BlockType.VARIABLE,
                        source=source,
                        start_line=start,
                        end_line=end,
                    ))

        # Group imports
        if import_lines:
            import_start = min(import_lines)
            import_end = max(import_lines)
            import_source = "\n".join(lines[import_start - 1:import_end])
            blocks.insert(0, CodeBlock(
                name="imports",
                block_type=BlockType.IMPORT,
                source=import_source,
                start_line=import_start,
                end_line=import_end,
            ))

        # Sort by start line
        blocks.sort(key=lambda b: b.start_line)

        return blocks

    def _extract_generic_blocks(self, content: str) -> List[CodeBlock]:
        """Fallback block extraction for non-Python files."""
        blocks: List[CodeBlock] = []
        lines = content.splitlines()

        current_block: List[str] = []
        start_line = 1

        for i, line in enumerate(lines, 1):
            if line.strip() == "" and current_block:
                # End of block
                blocks.append(CodeBlock(
                    name=f"block_{len(blocks)}",
                    block_type=BlockType.OTHER,
                    source="\n".join(current_block),
                    start_line=start_line,
                    end_line=i - 1,
                ))
                current_block = []
                start_line = i + 1
            else:
                current_block.append(line)

        if current_block:
            blocks.append(CodeBlock(
                name=f"block_{len(blocks)}",
                block_type=BlockType.OTHER,
                source="\n".join(current_block),
                start_line=start_line,
                end_line=len(lines),
            ))

        return blocks

    def _count_blocks(self, content: str, language: str) -> int:
        """Count blocks in merged content."""
        return len(self._extract_blocks(content, language))

    # =========================================================================
    # Block Merging
    # =========================================================================

    def _merge_blocks(
        self,
        base_blocks: List[CodeBlock],
        incoming_blocks: List[CodeBlock],
        strategy: MergeStrategy,
        context: MergeContext,
    ) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """Merge two sets of code blocks."""
        conflicts: List[Dict[str, Any]] = []
        warnings: List[str] = []
        merged_parts: List[str] = []

        # Index blocks by name and type
        base_index = {(b.name, b.block_type): b for b in base_blocks}
        incoming_index = {(b.name, b.block_type): b for b in incoming_blocks}

        # Track processed blocks
        processed: Set[Tuple[str, BlockType]] = set()

        # Process base blocks first
        for block in base_blocks:
            key = (block.name, block.block_type)

            if key in incoming_index:
                incoming = incoming_index[key]

                if block.semantic_equals(incoming):
                    # Identical - keep one
                    merged_parts.append(block.source)
                else:
                    # Different - need to merge
                    merged, conflict = self._merge_single_block(
                        block, incoming, strategy
                    )
                    merged_parts.append(merged)
                    if conflict:
                        conflicts.append(conflict)
            else:
                # Only in base - keep it
                merged_parts.append(block.source)

            processed.add(key)

        # Add blocks only in incoming
        for block in incoming_blocks:
            key = (block.name, block.block_type)

            if key not in processed:
                merged_parts.append(block.source)
                processed.add(key)

        # Handle imports specially
        if context.sort_imports or context.remove_duplicate_imports:
            merged_parts = self._process_imports(
                merged_parts, context.sort_imports, context.remove_duplicate_imports
            )

        # Join parts with newlines
        merged_content = "\n\n".join(part for part in merged_parts if part.strip())

        return merged_content, conflicts, warnings

    def _merge_single_block(
        self,
        base: CodeBlock,
        incoming: CodeBlock,
        strategy: MergeStrategy,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Merge a single block that differs between versions."""
        conflict = None

        if strategy == MergeStrategy.PREFER_BASE:
            return base.source, None

        elif strategy == MergeStrategy.PREFER_INCOMING:
            return incoming.source, None

        elif strategy == MergeStrategy.PRESERVE_BOTH:
            combined = f"# === BASE VERSION ===\n{base.source}\n\n# === INCOMING VERSION ===\n{incoming.source}"
            conflict = {
                "name": base.name,
                "type": base.block_type.value,
                "message": "Both versions preserved",
            }
            return combined, conflict

        elif strategy == MergeStrategy.INTERLEAVE:
            # Try line-by-line interleaving
            return self._interleave_blocks(base, incoming)

        elif strategy == MergeStrategy.SEMANTIC_MERGE:
            # Try semantic merge
            return self._semantic_merge_block(base, incoming)

        return base.source, None

    def _interleave_blocks(
        self,
        base: CodeBlock,
        incoming: CodeBlock,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Interleave two blocks line by line."""
        base_lines = base.source.splitlines()
        incoming_lines = incoming.source.splitlines()

        matcher = difflib.SequenceMatcher(None, base_lines, incoming_lines)
        result: List[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                result.extend(base_lines[i1:i2])
            elif tag == "replace":
                result.extend(base_lines[i1:i2])
                result.extend(incoming_lines[j1:j2])
            elif tag == "delete":
                result.extend(base_lines[i1:i2])
            elif tag == "insert":
                result.extend(incoming_lines[j1:j2])

        return "\n".join(result), None

    def _semantic_merge_block(
        self,
        base: CodeBlock,
        incoming: CodeBlock,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Semantic merge using AST when possible."""
        if base.block_type in (BlockType.FUNCTION, BlockType.ASYNC_FUNCTION):
            # Try to merge function implementations
            return self._merge_functions(base, incoming)

        elif base.block_type == BlockType.CLASS:
            # Try to merge class definitions
            return self._merge_classes(base, incoming)

        # Fallback to interleave
        return self._interleave_blocks(base, incoming)

    def _merge_functions(
        self,
        base: CodeBlock,
        incoming: CodeBlock,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Merge two function definitions."""
        try:
            base_tree = ast.parse(base.source).body[0]
            incoming_tree = ast.parse(incoming.source).body[0]

            # Compare signatures
            if base.signature != incoming.signature:
                # Signatures differ - this is a conflict
                conflict = {
                    "name": base.name,
                    "type": "function",
                    "message": f"Signature changed: {base.signature} vs {incoming.signature}",
                }
                # Prefer incoming (newer signature)
                return incoming.source, conflict

            # Signatures match - merge bodies
            # For now, prefer longer implementation (likely more complete)
            if len(incoming.source) > len(base.source):
                return incoming.source, None
            return base.source, None

        except Exception:
            return self._interleave_blocks(base, incoming)

    def _merge_classes(
        self,
        base: CodeBlock,
        incoming: CodeBlock,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Merge two class definitions."""
        try:
            # Parse both class definitions
            base_tree = ast.parse(base.source).body[0]
            incoming_tree = ast.parse(incoming.source).body[0]

            # Extract methods
            base_methods = {n.name: n for n in ast.walk(base_tree)
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            incoming_methods = {n.name: n for n in ast.walk(incoming_tree)
                              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

            # If incoming has more methods, prefer it
            if len(incoming_methods) > len(base_methods):
                return incoming.source, None

            return base.source, None

        except Exception:
            return self._interleave_blocks(base, incoming)

    def _process_imports(
        self,
        parts: List[str],
        sort_imports: bool,
        remove_duplicates: bool,
    ) -> List[str]:
        """Process and clean up imports in merged content."""
        import_lines: List[str] = []
        other_parts: List[str] = []

        for part in parts:
            lines = part.splitlines()
            part_imports: List[str] = []
            part_other: List[str] = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    part_imports.append(line)
                else:
                    part_other.append(line)

            import_lines.extend(part_imports)
            if part_other:
                other_parts.append("\n".join(part_other))

        # Remove duplicates
        if remove_duplicates:
            import_lines = list(dict.fromkeys(import_lines))

        # Sort imports
        if sort_imports:
            # Group: standard lib, third party, local
            stdlib = []
            third_party = []
            local = []

            for line in import_lines:
                stripped = line.strip()
                if stripped.startswith("from .") or stripped.startswith("import ."):
                    local.append(line)
                elif any(stripped.startswith(f"import {m}") or stripped.startswith(f"from {m}")
                        for m in ["os", "sys", "re", "json", "time", "datetime", "pathlib",
                                 "typing", "dataclasses", "enum", "abc", "collections", "functools"]):
                    stdlib.append(line)
                else:
                    third_party.append(line)

            import_lines = sorted(stdlib) + [""] + sorted(third_party) + [""] + sorted(local)
            import_lines = [l for l in import_lines if l or import_lines.index(l) > 0]

        # Rebuild parts with imports first
        result = []
        if import_lines:
            result.append("\n".join(import_lines))
        result.extend(other_parts)

        return result

    # =========================================================================
    # Three-Way Merge
    # =========================================================================

    def merge_three_way(
        self,
        base_content: str,
        ours_content: str,
        theirs_content: str,
        context: MergeContext,
    ) -> MergeResult:
        """
        Perform a three-way merge (base, ours, theirs).

        Uses base to detect what changed in each branch.
        """
        # Extract blocks from all three
        base_blocks = self._extract_blocks(base_content, context.language)
        ours_blocks = self._extract_blocks(ours_content, context.language)
        theirs_blocks = self._extract_blocks(theirs_content, context.language)

        # Index by name
        base_index = {(b.name, b.block_type): b for b in base_blocks}
        ours_index = {(b.name, b.block_type): b for b in ours_blocks}
        theirs_index = {(b.name, b.block_type): b for b in theirs_blocks}

        merged_parts: List[str] = []
        conflicts: List[Dict[str, Any]] = []

        all_keys = set(base_index.keys()) | set(ours_index.keys()) | set(theirs_index.keys())

        for key in sorted(all_keys, key=lambda k: base_index.get(k, CodeBlock("", BlockType.OTHER, "", 0, 0)).start_line):
            base = base_index.get(key)
            ours = ours_index.get(key)
            theirs = theirs_index.get(key)

            if base and ours and theirs:
                # All three exist
                ours_changed = not (base.semantic_equals(ours) if ours else True)
                theirs_changed = not (base.semantic_equals(theirs) if theirs else True)

                if not ours_changed and not theirs_changed:
                    # No changes
                    merged_parts.append(base.source)
                elif ours_changed and not theirs_changed:
                    # Only ours changed
                    merged_parts.append(ours.source)
                elif not ours_changed and theirs_changed:
                    # Only theirs changed
                    merged_parts.append(theirs.source)
                else:
                    # Both changed - conflict
                    if ours.semantic_equals(theirs):
                        # Same change
                        merged_parts.append(ours.source)
                    else:
                        # Different changes - true conflict
                        merged, conflict = self._merge_single_block(
                            ours, theirs, self.default_strategy
                        )
                        merged_parts.append(merged)
                        if conflict:
                            conflicts.append(conflict)

            elif ours and not theirs:
                # Deleted in theirs
                if base and base.semantic_equals(ours):
                    # Unchanged in ours, deleted in theirs - delete
                    pass
                else:
                    # Changed in ours, deleted in theirs - conflict
                    merged_parts.append(ours.source)
                    conflicts.append({
                        "name": ours.name,
                        "type": "delete_conflict",
                        "message": "Modified in ours, deleted in theirs",
                    })

            elif theirs and not ours:
                # Deleted in ours
                if base and base.semantic_equals(theirs):
                    # Unchanged in theirs, deleted in ours - delete
                    pass
                else:
                    # Changed in theirs, deleted in ours - conflict
                    merged_parts.append(theirs.source)
                    conflicts.append({
                        "name": theirs.name,
                        "type": "delete_conflict",
                        "message": "Modified in theirs, deleted in ours",
                    })

            elif ours:
                # New in ours
                merged_parts.append(ours.source)

            elif theirs:
                # New in theirs
                merged_parts.append(theirs.source)

        merged_content = "\n\n".join(part for part in merged_parts if part.strip())

        return MergeResult(
            id=f"merge3-{uuid.uuid4().hex[:8]}",
            success=len(conflicts) == 0,
            merged_content=merged_content,
            base_blocks=len(base_blocks),
            incoming_blocks=len(theirs_blocks),
            merged_blocks=self._count_blocks(merged_content, context.language),
            conflicts=conflicts,
        )

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_result(self, result_id: str) -> Optional[MergeResult]:
        """Get a merge result by ID."""
        return self._results.get(result_id)

    def analyze_diff(
        self,
        base_content: str,
        incoming_content: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """Analyze differences between two versions."""
        base_blocks = self._extract_blocks(base_content, language)
        incoming_blocks = self._extract_blocks(incoming_content, language)

        base_index = {(b.name, b.block_type): b for b in base_blocks}
        incoming_index = {(b.name, b.block_type): b for b in incoming_blocks}

        added = []
        removed = []
        modified = []
        unchanged = []

        for key, block in incoming_index.items():
            if key not in base_index:
                added.append(block.to_dict())
            elif not block.semantic_equals(base_index[key]):
                modified.append(block.to_dict())
            else:
                unchanged.append(block.name)

        for key, block in base_index.items():
            if key not in incoming_index:
                removed.append(block.to_dict())

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "unchanged_count": len(unchanged),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Semantic Merger."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Semantic Merger (Step 63)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # merge command
    merge_parser = subparsers.add_parser("merge", help="Merge two files")
    merge_parser.add_argument("base", help="Base file")
    merge_parser.add_argument("incoming", help="Incoming file")
    merge_parser.add_argument("--output", "-o", help="Output file")
    merge_parser.add_argument("--strategy",
                             choices=["preserve_both", "prefer_base", "prefer_incoming", "semantic_merge"],
                             default="semantic_merge", help="Merge strategy")
    merge_parser.add_argument("--sort-imports", action="store_true", help="Sort imports")
    merge_parser.add_argument("--json", action="store_true", help="JSON output")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze differences")
    analyze_parser.add_argument("base", help="Base file")
    analyze_parser.add_argument("incoming", help="Incoming file")

    # blocks command
    blocks_parser = subparsers.add_parser("blocks", help="Extract code blocks")
    blocks_parser.add_argument("file", help="File to analyze")
    blocks_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    merger = SemanticMerger()

    if args.command == "merge":
        base_content = Path(args.base).read_text()
        incoming_content = Path(args.incoming).read_text()

        context = MergeContext(
            base_path=args.base,
            incoming_path=args.incoming,
            output_path=args.output,
            sort_imports=args.sort_imports,
        )

        result = merger.merge(
            base_content, incoming_content, context,
            MergeStrategy(args.strategy)
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if args.output:
                Path(args.output).write_text(result.merged_content)
                print(f"Merged to {args.output}")
            else:
                print(result.merged_content)

            if result.conflicts:
                print(f"\nConflicts: {len(result.conflicts)}")
                for conflict in result.conflicts:
                    print(f"  - {conflict['name']}: {conflict.get('message', '')}")

        return 0 if result.success else 1

    elif args.command == "analyze":
        base_content = Path(args.base).read_text()
        incoming_content = Path(args.incoming).read_text()

        analysis = merger.analyze_diff(base_content, incoming_content)
        print(json.dumps(analysis, indent=2))
        return 0

    elif args.command == "blocks":
        content = Path(args.file).read_text()
        blocks = merger._extract_blocks(content, "python")

        if args.json:
            print(json.dumps([b.to_dict() for b in blocks], indent=2))
        else:
            for block in blocks:
                print(f"{block.block_type.value}: {block.name} (lines {block.start_line}-{block.end_line})")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
