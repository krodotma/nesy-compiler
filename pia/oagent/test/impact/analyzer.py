#!/usr/bin/env python3
"""
Step 119: Test Impact Analyzer

Provides change-based test selection by analyzing code dependencies.

PBTSO Phase: PLAN, TEST
Bus Topics:
- test.impact.analyze (subscribes)
- test.impact.result (emits)

Dependencies: Step 117 (Test Prioritizer)
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Constants
# ============================================================================

class ChangeType(Enum):
    """Types of code changes."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class ImpactLevel(Enum):
    """Level of impact on tests."""
    DIRECT = "direct"  # Test directly tests changed code
    INDIRECT = "indirect"  # Test depends on changed code
    POTENTIAL = "potential"  # Test might be affected
    NONE = "none"  # Test is not affected


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class FileChange:
    """Represents a changed file."""
    path: str
    change_type: ChangeType
    old_path: Optional[str] = None  # For renames
    added_lines: int = 0
    deleted_lines: int = 0
    changed_functions: List[str] = field(default_factory=list)
    changed_classes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "change_type": self.change_type.value,
            "old_path": self.old_path,
            "added_lines": self.added_lines,
            "deleted_lines": self.deleted_lines,
            "changed_functions": self.changed_functions,
            "changed_classes": self.changed_classes,
        }


@dataclass
class ImpactMapping:
    """Mapping of a test to its impacted code."""
    test_name: str
    test_path: str
    impact_level: ImpactLevel
    impacted_files: List[str] = field(default_factory=list)
    impacted_functions: List[str] = field(default_factory=list)
    impacted_classes: List[str] = field(default_factory=list)
    confidence: float = 1.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "test_path": self.test_path,
            "impact_level": self.impact_level.value,
            "impacted_files": self.impacted_files,
            "impacted_functions": self.impacted_functions,
            "impacted_classes": self.impacted_classes,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class ImpactConfig:
    """
    Configuration for impact analysis.

    Attributes:
        repo_root: Root of the repository
        base_ref: Base git reference for comparison
        head_ref: Head git reference for comparison
        test_patterns: Patterns to identify test files
        coverage_map_file: File with test-to-code coverage mapping
        dependency_map_file: File with code dependency mapping
        include_indirect: Include indirectly affected tests
        output_dir: Directory for reports
    """
    repo_root: str = "."
    base_ref: str = "HEAD~1"
    head_ref: str = "HEAD"
    test_patterns: List[str] = field(default_factory=lambda: [
        "**/test_*.py", "**/*_test.py", "**/tests/**/*.py"
    ])
    coverage_map_file: Optional[str] = None
    dependency_map_file: Optional[str] = None
    include_indirect: bool = True
    output_dir: str = ".pluribus/test-agent/impact"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repo_root": self.repo_root,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "test_patterns": self.test_patterns,
            "include_indirect": self.include_indirect,
        }


@dataclass
class ImpactResult:
    """Result of impact analysis."""
    run_id: str
    analyzed_at: float
    changes: List[FileChange] = field(default_factory=list)
    impacted_tests: List[ImpactMapping] = field(default_factory=list)
    all_tests: List[str] = field(default_factory=list)
    selected_tests: List[str] = field(default_factory=list)
    selection_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_changes(self) -> int:
        """Get total number of changed files."""
        return len(self.changes)

    @property
    def direct_impact_count(self) -> int:
        """Get count of directly impacted tests."""
        return sum(1 for t in self.impacted_tests if t.impact_level == ImpactLevel.DIRECT)

    @property
    def indirect_impact_count(self) -> int:
        """Get count of indirectly impacted tests."""
        return sum(1 for t in self.impacted_tests if t.impact_level == ImpactLevel.INDIRECT)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "analyzed_at": self.analyzed_at,
            "total_changes": self.total_changes,
            "changes": [c.to_dict() for c in self.changes],
            "impacted_tests": [t.to_dict() for t in self.impacted_tests],
            "selected_tests": self.selected_tests,
            "selection_ratio": self.selection_ratio,
            "direct_impact_count": self.direct_impact_count,
            "indirect_impact_count": self.indirect_impact_count,
            "metadata": self.metadata,
        }


# ============================================================================
# Bus Interface
# ============================================================================

class ImpactBus:
    """Bus interface for impact analysis."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event_with_meta) + "\n")
        except IOError:
            pass


# ============================================================================
# Impact Analyzer
# ============================================================================

class ImpactAnalyzer:
    """
    Analyzes code changes to determine test impact.

    Impact analysis strategies:
    1. Direct mapping: Test file tests specific module
    2. Coverage-based: Test covers changed lines
    3. Dependency-based: Test imports changed modules
    4. Name heuristics: Test name matches changed code

    PBTSO Phase: PLAN, TEST
    Bus Topics: test.impact.analyze, test.impact.result
    """

    BUS_TOPICS = {
        "analyze": "test.impact.analyze",
        "result": "test.impact.result",
    }

    def __init__(self, bus=None):
        """
        Initialize the impact analyzer.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or ImpactBus()
        self._coverage_map: Dict[str, Set[str]] = {}
        self._dependency_map: Dict[str, Set[str]] = {}

    def analyze(self, config: Optional[ImpactConfig] = None) -> ImpactResult:
        """
        Analyze code changes and determine test impact.

        Args:
            config: Impact analysis configuration

        Returns:
            ImpactResult with impacted tests
        """
        config = config or ImpactConfig()
        run_id = str(uuid.uuid4())

        result = ImpactResult(
            run_id=run_id,
            analyzed_at=time.time(),
        )

        # Emit start event
        self._emit_event("analyze", {
            "run_id": run_id,
            "status": "started",
            "base_ref": config.base_ref,
            "head_ref": config.head_ref,
        })

        # Load coverage and dependency maps
        if config.coverage_map_file:
            self._load_coverage_map(config.coverage_map_file)
        if config.dependency_map_file:
            self._load_dependency_map(config.dependency_map_file)

        # Get code changes
        changes = self._get_git_changes(config)
        result.changes = changes

        # Get all tests
        all_tests = self._discover_tests(config)
        result.all_tests = all_tests

        # Build reverse mapping (file -> tests that cover it)
        file_to_tests = self._build_file_to_tests_map(all_tests, config)

        # Analyze impact for each change
        impacted_tests: Dict[str, ImpactMapping] = {}

        for change in changes:
            # Skip test files themselves
            if self._is_test_file(change.path, config):
                # If test file changed, include it
                mapping = ImpactMapping(
                    test_name=change.path,
                    test_path=change.path,
                    impact_level=ImpactLevel.DIRECT,
                    reason="Test file was modified",
                )
                impacted_tests[change.path] = mapping
                continue

            # Find tests impacted by this change
            tests_for_file = self._find_impacted_tests(change, file_to_tests, config)

            for test_name, impact in tests_for_file.items():
                if test_name not in impacted_tests:
                    impacted_tests[test_name] = impact
                else:
                    # Merge - keep higher impact level
                    existing = impacted_tests[test_name]
                    if self._impact_level_priority(impact.impact_level) > \
                       self._impact_level_priority(existing.impact_level):
                        impacted_tests[test_name] = impact
                    else:
                        existing.impacted_files.extend(impact.impacted_files)

        result.impacted_tests = list(impacted_tests.values())

        # Filter based on impact level
        selected = []
        for mapping in result.impacted_tests:
            if mapping.impact_level in (ImpactLevel.DIRECT, ImpactLevel.INDIRECT):
                selected.append(mapping.test_name)
            elif config.include_indirect and mapping.impact_level == ImpactLevel.POTENTIAL:
                selected.append(mapping.test_name)

        result.selected_tests = selected
        result.selection_ratio = len(selected) / len(all_tests) if all_tests else 0

        # Emit result event
        self._emit_event("result", {
            "run_id": run_id,
            "status": "completed",
            "total_changes": result.total_changes,
            "impacted_tests": len(result.impacted_tests),
            "selected_tests": len(result.selected_tests),
            "selection_ratio": result.selection_ratio,
        })

        # Save report
        self._save_report(result, config.output_dir)

        return result

    def _get_git_changes(self, config: ImpactConfig) -> List[FileChange]:
        """Get changes between git refs."""
        changes = []

        try:
            # Get diff summary
            result = subprocess.run(
                ["git", "diff", "--name-status", config.base_ref, config.head_ref],
                capture_output=True,
                text=True,
                cwd=config.repo_root,
            )

            if result.returncode != 0:
                return changes

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                status = parts[0][0]  # First character
                path = parts[-1]

                if status == 'A':
                    change_type = ChangeType.ADDED
                elif status == 'M':
                    change_type = ChangeType.MODIFIED
                elif status == 'D':
                    change_type = ChangeType.DELETED
                elif status == 'R':
                    change_type = ChangeType.RENAMED
                else:
                    change_type = ChangeType.MODIFIED

                change = FileChange(
                    path=path,
                    change_type=change_type,
                    old_path=parts[1] if len(parts) > 2 else None,
                )

                # Get changed functions/classes for Python files
                if path.endswith('.py') and change_type == ChangeType.MODIFIED:
                    change.changed_functions, change.changed_classes = \
                        self._get_changed_symbols(path, config)

                changes.append(change)

        except Exception:
            pass

        return changes

    def _get_changed_symbols(
        self,
        file_path: str,
        config: ImpactConfig,
    ) -> Tuple[List[str], List[str]]:
        """Get changed functions and classes in a file."""
        functions = []
        classes = []

        try:
            # Get diff for specific file
            result = subprocess.run(
                ["git", "diff", config.base_ref, config.head_ref, "--", file_path],
                capture_output=True,
                text=True,
                cwd=config.repo_root,
            )

            if result.returncode != 0:
                return functions, classes

            diff_text = result.stdout

            # Find function definitions in diff
            func_pattern = r'^\+\s*(?:async\s+)?def\s+(\w+)'
            class_pattern = r'^\+\s*class\s+(\w+)'

            for match in re.finditer(func_pattern, diff_text, re.MULTILINE):
                functions.append(match.group(1))

            for match in re.finditer(class_pattern, diff_text, re.MULTILINE):
                classes.append(match.group(1))

        except Exception:
            pass

        return functions, classes

    def _discover_tests(self, config: ImpactConfig) -> List[str]:
        """Discover all test files."""
        tests = []
        root = Path(config.repo_root)

        for pattern in config.test_patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    tests.append(str(path.relative_to(root)))

        return tests

    def _is_test_file(self, path: str, config: ImpactConfig) -> bool:
        """Check if a file is a test file."""
        import fnmatch

        for pattern in config.test_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    def _build_file_to_tests_map(
        self,
        tests: List[str],
        config: ImpactConfig,
    ) -> Dict[str, Set[str]]:
        """Build mapping from source files to tests that cover them."""
        file_to_tests: Dict[str, Set[str]] = {}

        # Use coverage map if available
        for test, covered_files in self._coverage_map.items():
            for file in covered_files:
                if file not in file_to_tests:
                    file_to_tests[file] = set()
                file_to_tests[file].add(test)

        # Also use naming conventions
        for test in tests:
            # Extract module name from test
            test_path = Path(test)
            test_name = test_path.stem

            # Try common naming patterns
            module_name = test_name.replace("test_", "").replace("_test", "")

            # Map to potential source files
            potential_sources = [
                f"{module_name}.py",
                f"src/{module_name}.py",
                f"lib/{module_name}.py",
            ]

            for source in potential_sources:
                if source not in file_to_tests:
                    file_to_tests[source] = set()
                file_to_tests[source].add(test)

        return file_to_tests

    def _find_impacted_tests(
        self,
        change: FileChange,
        file_to_tests: Dict[str, Set[str]],
        config: ImpactConfig,
    ) -> Dict[str, ImpactMapping]:
        """Find tests impacted by a change."""
        impacted: Dict[str, ImpactMapping] = {}

        # Direct match from coverage map
        if change.path in file_to_tests:
            for test in file_to_tests[change.path]:
                impacted[test] = ImpactMapping(
                    test_name=test,
                    test_path=test,
                    impact_level=ImpactLevel.DIRECT,
                    impacted_files=[change.path],
                    reason=f"Covers changed file: {change.path}",
                )

        # Check dependencies
        if change.path in self._dependency_map:
            dependents = self._dependency_map[change.path]
            for dep in dependents:
                if dep in file_to_tests:
                    for test in file_to_tests[dep]:
                        if test not in impacted:
                            impacted[test] = ImpactMapping(
                                test_name=test,
                                test_path=test,
                                impact_level=ImpactLevel.INDIRECT,
                                impacted_files=[change.path],
                                reason=f"Depends on changed file via {dep}",
                            )

        # Heuristic: filename matching
        change_module = Path(change.path).stem
        for test in file_to_tests.get(change.path, set()):
            pass  # Already handled

        # Check all tests for naming match
        for file, tests in file_to_tests.items():
            if change_module in file:
                for test in tests:
                    if test not in impacted:
                        impacted[test] = ImpactMapping(
                            test_name=test,
                            test_path=test,
                            impact_level=ImpactLevel.POTENTIAL,
                            impacted_files=[change.path],
                            confidence=0.7,
                            reason=f"Name match with {change_module}",
                        )

        return impacted

    def _impact_level_priority(self, level: ImpactLevel) -> int:
        """Get priority for impact level (higher = more important)."""
        priorities = {
            ImpactLevel.DIRECT: 3,
            ImpactLevel.INDIRECT: 2,
            ImpactLevel.POTENTIAL: 1,
            ImpactLevel.NONE: 0,
        }
        return priorities.get(level, 0)

    def _load_coverage_map(self, path: str) -> None:
        """Load test-to-file coverage mapping."""
        try:
            with open(path) as f:
                data = json.load(f)
                for test, files in data.items():
                    self._coverage_map[test] = set(files)
        except (IOError, json.JSONDecodeError):
            pass

    def _load_dependency_map(self, path: str) -> None:
        """Load file dependency mapping."""
        try:
            with open(path) as f:
                data = json.load(f)
                for file, deps in data.items():
                    self._dependency_map[file] = set(deps)
        except (IOError, json.JSONDecodeError):
            pass

    def _save_report(self, result: ImpactResult, output_dir: str) -> None:
        """Save impact analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        report_file = output_path / f"impact_report_{result.run_id}.json"
        with open(report_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save selected tests for easy use
        tests_file = output_path / "selected_tests.txt"
        with open(tests_file, "w") as f:
            f.write("\n".join(result.selected_tests))

        # Generate markdown
        report_md = output_path / f"impact_report_{result.run_id}.md"
        with open(report_md, "w") as f:
            f.write(self._generate_markdown_report(result))

    def _generate_markdown_report(self, result: ImpactResult) -> str:
        """Generate markdown impact report."""
        lines = [
            "# Test Impact Analysis Report",
            "",
            f"**Run ID**: {result.run_id}",
            f"**Date**: {datetime.fromtimestamp(result.analyzed_at).isoformat()}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Files Changed | {result.total_changes} |",
            f"| Total Tests | {len(result.all_tests)} |",
            f"| Impacted Tests | {len(result.impacted_tests)} |",
            f"| Selected Tests | {len(result.selected_tests)} |",
            f"| Selection Ratio | {result.selection_ratio:.1%} |",
            f"| Direct Impact | {result.direct_impact_count} |",
            f"| Indirect Impact | {result.indirect_impact_count} |",
        ]

        if result.changes:
            lines.extend([
                "",
                "## Changes",
                "",
                "| File | Type | Functions | Classes |",
                "|------|------|-----------|---------|",
            ])

            for change in result.changes:
                funcs = ", ".join(change.changed_functions[:3]) or "-"
                classes = ", ".join(change.changed_classes[:3]) or "-"
                lines.append(
                    f"| {change.path} | {change.change_type.value} | {funcs} | {classes} |"
                )

        if result.impacted_tests:
            lines.extend([
                "",
                "## Impacted Tests",
                "",
                "### Direct Impact",
                "",
            ])

            for mapping in result.impacted_tests:
                if mapping.impact_level == ImpactLevel.DIRECT:
                    lines.append(f"- **{mapping.test_name}**")
                    lines.append(f"  - {mapping.reason}")

            lines.extend([
                "",
                "### Indirect Impact",
                "",
            ])

            for mapping in result.impacted_tests:
                if mapping.impact_level == ImpactLevel.INDIRECT:
                    lines.append(f"- {mapping.test_name}")
                    lines.append(f"  - {mapping.reason}")

        return "\n".join(lines)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.impact.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "impact_analysis",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Impact Analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Impact Analyzer")
    parser.add_argument("--base", default="HEAD~1", help="Base git reference")
    parser.add_argument("--head", default="HEAD", help="Head git reference")
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--coverage-map", help="Coverage mapping file")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/impact")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = ImpactConfig(
        repo_root=args.repo,
        base_ref=args.base,
        head_ref=args.head,
        coverage_map_file=args.coverage_map,
        output_dir=args.output,
    )

    analyzer = ImpactAnalyzer()
    result = analyzer.analyze(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Test Impact Analysis Complete")
        print(f"{'='*60}")
        print(f"Files Changed: {result.total_changes}")
        print(f"Total Tests: {len(result.all_tests)}")
        print(f"Selected Tests: {len(result.selected_tests)}")
        print(f"Selection Ratio: {result.selection_ratio:.1%}")
        print(f"\nSelected tests saved to: {config.output_dir}/selected_tests.txt")

        if result.selected_tests:
            print(f"\nTop 10 Selected Tests:")
            for test in result.selected_tests[:10]:
                print(f"  - {test}")


if __name__ == "__main__":
    main()
