#!/usr/bin/env python3
"""
resolver.py - Conflict Resolver (Step 62)

PBTSO Phase: ITERATE, VERIFY

Provides:
- Merge conflict detection in files
- Conflict region extraction
- Multiple resolution strategies
- Automatic resolution for simple conflicts
- Conflict tracking and reporting

Bus Topics:
- code.conflict.detected
- code.conflict.resolved
- code.conflict.manual

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Types
# =============================================================================

class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    OURS = "ours"              # Keep our version
    THEIRS = "theirs"          # Keep their version
    UNION = "union"            # Keep both (concatenate)
    MANUAL = "manual"          # Require human intervention
    SEMANTIC = "semantic"      # Use semantic analysis
    LATEST = "latest"          # Use most recent change
    SHORTEST = "shortest"      # Use shorter version
    LONGEST = "longest"        # Use longer version


class ConflictType(Enum):
    """Type of merge conflict."""
    CONTENT = "content"        # Line content differs
    ADD_ADD = "add_add"        # Both added lines at same location
    MODIFY_DELETE = "modify_delete"  # One modified, one deleted
    RENAME = "rename"          # File renamed differently
    MODE = "mode"              # File mode/permissions differ


class ConflictSeverity(Enum):
    """Severity of the conflict."""
    LOW = "low"       # Trivial, auto-resolvable
    MEDIUM = "medium" # May need review
    HIGH = "high"     # Requires manual resolution
    CRITICAL = "critical"  # Structural conflict


@dataclass
class ConflictRegion:
    """
    A region within a file where conflict exists.

    Represents the conflicting content from different sources.
    """
    start_line: int
    end_line: int
    ours: List[str]
    theirs: List[str]
    base: Optional[List[str]] = None  # Original content (3-way merge)
    marker_type: str = "standard"  # standard, diff3

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def is_trivial(self) -> bool:
        """Check if conflict is trivially resolvable."""
        # Whitespace-only differences
        ours_stripped = [l.strip() for l in self.ours]
        theirs_stripped = [l.strip() for l in self.theirs]
        return ours_stripped == theirs_stripped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "ours": self.ours,
            "theirs": self.theirs,
            "base": self.base,
            "line_count": self.line_count,
            "is_trivial": self.is_trivial,
        }


@dataclass
class MergeConflict:
    """
    Represents a complete merge conflict for a file.

    Contains all conflict regions and metadata.
    """
    id: str
    path: str
    regions: List[ConflictRegion]
    conflict_type: ConflictType = ConflictType.CONTENT
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    resolution_strategy: Optional[ResolutionStrategy] = None
    our_branch: str = "HEAD"
    their_branch: str = "incoming"

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def total_conflict_lines(self) -> int:
        return sum(r.line_count for r in self.regions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "regions": [r.to_dict() for r in self.regions],
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "is_resolved": self.is_resolved,
            "region_count": self.region_count,
            "total_conflict_lines": self.total_conflict_lines,
            "our_branch": self.our_branch,
            "their_branch": self.their_branch,
        }


@dataclass
class ConflictResolution:
    """Result of resolving a conflict."""
    conflict_id: str
    strategy: ResolutionStrategy
    resolved_content: str
    regions_resolved: int
    manual_edits: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "strategy": self.strategy.value,
            "resolved_content_length": len(self.resolved_content),
            "regions_resolved": self.regions_resolved,
            "manual_edits": self.manual_edits,
            "warnings": self.warnings,
        }


# =============================================================================
# Conflict Resolver
# =============================================================================

class ConflictResolver:
    """
    Detect and resolve merge conflicts in files.

    PBTSO Phase: ITERATE, VERIFY

    Features:
    - Detect Git-style conflict markers
    - Parse conflict regions with context
    - Multiple resolution strategies
    - Semantic conflict analysis
    - Auto-resolution for trivial conflicts

    Usage:
        resolver = ConflictResolver()
        conflicts = resolver.detect(file_content, "path/to/file.py")
        resolution = resolver.resolve(conflicts[0], ResolutionStrategy.OURS)
    """

    # Git conflict markers
    MARKER_OURS = "<<<<<<<"
    MARKER_SEPARATOR = "======="
    MARKER_THEIRS = ">>>>>>>"
    MARKER_BASE = "|||||||"  # For diff3 style

    # Pattern to match conflict blocks
    CONFLICT_PATTERN = re.compile(
        r'^<<<<<<<\s*(\S*).*?\n'
        r'(.*?)'
        r'^(?:\|\|\|\|\|\|\|\s*.*?\n(.*?))?'
        r'^=======\n'
        r'(.*?)'
        r'^>>>>>>>\s*(\S*).*?\n',
        re.MULTILINE | re.DOTALL
    )

    BUS_TOPICS = {
        "detected": "code.conflict.detected",
        "resolved": "code.conflict.resolved",
        "manual": "code.conflict.manual",
    }

    def __init__(
        self,
        bus: Optional[Any] = None,
        default_strategy: ResolutionStrategy = ResolutionStrategy.MANUAL,
        auto_resolve_trivial: bool = True,
    ):
        self.bus = bus
        self.default_strategy = default_strategy
        self.auto_resolve_trivial = auto_resolve_trivial
        self._conflicts: Dict[str, MergeConflict] = {}

    # =========================================================================
    # Detection
    # =========================================================================

    def detect(
        self,
        content: str,
        path: str,
        our_branch: str = "HEAD",
        their_branch: str = "incoming",
    ) -> List[MergeConflict]:
        """
        Detect all merge conflicts in file content.

        Args:
            content: File content with potential conflict markers
            path: File path
            our_branch: Name of our branch
            their_branch: Name of their branch

        Returns:
            List of MergeConflict objects
        """
        conflicts: List[MergeConflict] = []
        regions: List[ConflictRegion] = []

        # Find all conflict markers
        lines = content.splitlines(keepends=True)
        current_line = 1
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.startswith(self.MARKER_OURS):
                # Found start of conflict
                region = self._parse_conflict_region(lines, i, current_line)
                if region:
                    regions.append(region)
                    # Skip past the conflict
                    skip_lines = region.end_line - region.start_line + 1
                    i += skip_lines
                    current_line += skip_lines
                    continue

            i += 1
            current_line += 1

        if regions:
            # Determine severity based on regions
            severity = self._calculate_severity(regions)

            conflict = MergeConflict(
                id=f"conflict-{uuid.uuid4().hex[:8]}",
                path=path,
                regions=regions,
                severity=severity,
                our_branch=our_branch,
                their_branch=their_branch,
            )

            conflicts.append(conflict)
            self._conflicts[conflict.id] = conflict

            # Emit detection event
            if self.bus:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["detected"],
                    "kind": "conflict",
                    "actor": "code-agent",
                    "data": {
                        "conflict_id": conflict.id,
                        "path": path,
                        "regions": len(regions),
                        "severity": severity.value,
                        "total_lines": conflict.total_conflict_lines,
                    },
                })

        return conflicts

    def _parse_conflict_region(
        self,
        lines: List[str],
        start_idx: int,
        start_line: int,
    ) -> Optional[ConflictRegion]:
        """Parse a single conflict region from lines."""
        ours: List[str] = []
        theirs: List[str] = []
        base: Optional[List[str]] = None
        marker_type = "standard"

        i = start_idx + 1  # Skip the <<<<<<< marker
        section = "ours"

        while i < len(lines):
            line = lines[i]

            if line.startswith(self.MARKER_BASE):
                # diff3 style - base section
                marker_type = "diff3"
                section = "base"
                base = []
            elif line.startswith(self.MARKER_SEPARATOR):
                section = "theirs"
            elif line.startswith(self.MARKER_THEIRS):
                # End of conflict
                end_line = start_line + (i - start_idx)
                return ConflictRegion(
                    start_line=start_line,
                    end_line=end_line,
                    ours=ours,
                    theirs=theirs,
                    base=base,
                    marker_type=marker_type,
                )
            else:
                # Content line
                if section == "ours":
                    ours.append(line.rstrip('\n'))
                elif section == "theirs":
                    theirs.append(line.rstrip('\n'))
                elif section == "base" and base is not None:
                    base.append(line.rstrip('\n'))

            i += 1

        return None  # Malformed conflict

    def _calculate_severity(self, regions: List[ConflictRegion]) -> ConflictSeverity:
        """Calculate overall severity based on regions."""
        if all(r.is_trivial for r in regions):
            return ConflictSeverity.LOW

        total_lines = sum(r.line_count for r in regions)

        if total_lines > 100:
            return ConflictSeverity.CRITICAL
        elif total_lines > 30:
            return ConflictSeverity.HIGH
        elif total_lines > 10:
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW

    def has_conflicts(self, content: str) -> bool:
        """Quick check if content has any conflict markers."""
        return self.MARKER_OURS in content and self.MARKER_THEIRS in content

    # =========================================================================
    # Resolution
    # =========================================================================

    def resolve(
        self,
        conflict: MergeConflict,
        strategy: Optional[ResolutionStrategy] = None,
        content: Optional[str] = None,
    ) -> ConflictResolution:
        """
        Resolve a merge conflict using the specified strategy.

        Args:
            conflict: The conflict to resolve
            strategy: Resolution strategy to use
            content: Original file content with conflicts

        Returns:
            ConflictResolution with resolved content
        """
        strategy = strategy or self.default_strategy

        if content is None:
            # Try to read from path
            path = Path(conflict.path)
            if path.exists():
                content = path.read_text()
            else:
                return ConflictResolution(
                    conflict_id=conflict.id,
                    strategy=strategy,
                    resolved_content="",
                    regions_resolved=0,
                    warnings=["Could not read file content"],
                )

        warnings: List[str] = []
        resolved_content = content

        # Process each region
        for region in conflict.regions:
            resolved_region = self._resolve_region(region, strategy)

            if resolved_region is None:
                warnings.append(f"Could not auto-resolve region at line {region.start_line}")
                continue

            # Replace the conflict markers with resolved content
            resolved_content = self._replace_region(
                resolved_content, region, resolved_region
            )

        conflict.resolved_at = time.time()
        conflict.resolution_strategy = strategy

        resolution = ConflictResolution(
            conflict_id=conflict.id,
            strategy=strategy,
            resolved_content=resolved_content,
            regions_resolved=len(conflict.regions) - len(warnings),
            warnings=warnings,
        )

        # Emit resolution event
        if self.bus:
            topic = self.BUS_TOPICS["resolved"] if not warnings else self.BUS_TOPICS["manual"]
            self.bus.emit({
                "topic": topic,
                "kind": "conflict",
                "actor": "code-agent",
                "data": {
                    "conflict_id": conflict.id,
                    "strategy": strategy.value,
                    "regions_resolved": resolution.regions_resolved,
                    "warnings": len(warnings),
                },
            })

        return resolution

    def _resolve_region(
        self,
        region: ConflictRegion,
        strategy: ResolutionStrategy,
    ) -> Optional[List[str]]:
        """Resolve a single conflict region."""
        if strategy == ResolutionStrategy.OURS:
            return region.ours

        elif strategy == ResolutionStrategy.THEIRS:
            return region.theirs

        elif strategy == ResolutionStrategy.UNION:
            # Combine both, removing duplicates while preserving order
            seen = set()
            result = []
            for line in region.ours + region.theirs:
                if line not in seen:
                    seen.add(line)
                    result.append(line)
            return result

        elif strategy == ResolutionStrategy.LATEST:
            # Prefer theirs (incoming changes are typically newer)
            return region.theirs

        elif strategy == ResolutionStrategy.SHORTEST:
            return region.ours if len(region.ours) <= len(region.theirs) else region.theirs

        elif strategy == ResolutionStrategy.LONGEST:
            return region.ours if len(region.ours) >= len(region.theirs) else region.theirs

        elif strategy == ResolutionStrategy.SEMANTIC:
            return self._semantic_resolution(region)

        elif strategy == ResolutionStrategy.MANUAL:
            # Cannot auto-resolve
            return None

        return None

    def _semantic_resolution(self, region: ConflictRegion) -> Optional[List[str]]:
        """
        Attempt semantic resolution based on code analysis.

        This is a simplified version - production would use AST analysis.
        """
        # If one side is empty, use the other
        if not region.ours:
            return region.theirs
        if not region.theirs:
            return region.ours

        # If trivial (whitespace only), use ours
        if region.is_trivial:
            return region.ours

        # Check for import additions (can often be merged)
        ours_imports = all(l.strip().startswith(("import ", "from ")) for l in region.ours if l.strip())
        theirs_imports = all(l.strip().startswith(("import ", "from ")) for l in region.theirs if l.strip())

        if ours_imports and theirs_imports:
            # Merge imports
            all_imports = sorted(set(region.ours + region.theirs))
            return all_imports

        # Check for comment additions
        ours_comments = all(l.strip().startswith(("#", "//", "/*")) for l in region.ours if l.strip())
        theirs_comments = all(l.strip().startswith(("#", "//", "/*")) for l in region.theirs if l.strip())

        if ours_comments or theirs_comments:
            # Keep both comments
            return region.ours + region.theirs

        # Cannot semantically resolve
        return None

    def _replace_region(
        self,
        content: str,
        region: ConflictRegion,
        resolved: List[str],
    ) -> str:
        """Replace a conflict region with resolved content."""
        lines = content.splitlines(keepends=True)
        result_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith(self.MARKER_OURS):
                # Skip to end of conflict
                while i < len(lines) and not lines[i].startswith(self.MARKER_THEIRS):
                    i += 1
                i += 1  # Skip the >>>>>>> line

                # Insert resolved content
                for resolved_line in resolved:
                    result_lines.append(resolved_line + "\n")
            else:
                result_lines.append(line)
                i += 1

        return "".join(result_lines)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def resolve_all(
        self,
        conflicts: List[MergeConflict],
        strategy: ResolutionStrategy,
        content_map: Dict[str, str],
    ) -> Dict[str, ConflictResolution]:
        """Resolve multiple conflicts with the same strategy."""
        resolutions = {}
        for conflict in conflicts:
            content = content_map.get(conflict.path)
            resolution = self.resolve(conflict, strategy, content)
            resolutions[conflict.id] = resolution
        return resolutions

    def detect_in_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> List[MergeConflict]:
        """Detect conflicts in all files in a directory."""
        conflicts = []
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
                    content = file_path.read_text()
                    if self.has_conflicts(content):
                        file_conflicts = self.detect(
                            content,
                            str(file_path.relative_to(directory)),
                        )
                        conflicts.extend(file_conflicts)
                except (UnicodeDecodeError, PermissionError):
                    pass  # Skip binary or inaccessible files

        return conflicts

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_conflict(self, conflict_id: str) -> Optional[MergeConflict]:
        """Get a conflict by ID."""
        return self._conflicts.get(conflict_id)

    def list_conflicts(self, path: Optional[str] = None) -> List[MergeConflict]:
        """List all conflicts, optionally filtered by path."""
        if path:
            return [c for c in self._conflicts.values() if c.path == path]
        return list(self._conflicts.values())

    def list_unresolved(self) -> List[MergeConflict]:
        """List all unresolved conflicts."""
        return [c for c in self._conflicts.values() if not c.is_resolved]


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Conflict Resolver."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Conflict Resolver (Step 62)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # detect command
    detect_parser = subparsers.add_parser("detect", help="Detect conflicts in file")
    detect_parser.add_argument("file", help="File to scan")
    detect_parser.add_argument("--json", action="store_true", help="JSON output")

    # resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve conflicts")
    resolve_parser.add_argument("file", help="File to resolve")
    resolve_parser.add_argument("--strategy", choices=["ours", "theirs", "union", "semantic"],
                               default="ours", help="Resolution strategy")
    resolve_parser.add_argument("--output", help="Output file (default: overwrite)")
    resolve_parser.add_argument("--dry-run", action="store_true", help="Show result without writing")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan directory for conflicts")
    scan_parser.add_argument("directory", help="Directory to scan")
    scan_parser.add_argument("--recursive", action="store_true", default=True)

    args = parser.parse_args()

    resolver = ConflictResolver()

    if args.command == "detect":
        content = Path(args.file).read_text()
        conflicts = resolver.detect(content, args.file)

        if args.json:
            print(json.dumps([c.to_dict() for c in conflicts], indent=2))
        else:
            if not conflicts:
                print(f"No conflicts detected in {args.file}")
            else:
                for conflict in conflicts:
                    print(f"Conflict: {conflict.id}")
                    print(f"  Path: {conflict.path}")
                    print(f"  Regions: {conflict.region_count}")
                    print(f"  Severity: {conflict.severity.value}")
                    print(f"  Total lines: {conflict.total_conflict_lines}")
        return 0

    elif args.command == "resolve":
        content = Path(args.file).read_text()
        conflicts = resolver.detect(content, args.file)

        if not conflicts:
            print("No conflicts to resolve")
            return 0

        strategy = ResolutionStrategy(args.strategy)
        resolution = resolver.resolve(conflicts[0], strategy, content)

        if args.dry_run:
            print(resolution.resolved_content)
        else:
            output_path = args.output or args.file
            Path(output_path).write_text(resolution.resolved_content)
            print(f"Resolved {resolution.regions_resolved} regions using {strategy.value}")
            if resolution.warnings:
                for warning in resolution.warnings:
                    print(f"  Warning: {warning}")
        return 0

    elif args.command == "scan":
        conflicts = resolver.detect_in_directory(Path(args.directory), args.recursive)
        print(f"Found {len(conflicts)} files with conflicts:")
        for conflict in conflicts:
            print(f"  {conflict.path}: {conflict.region_count} regions ({conflict.severity.value})")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
