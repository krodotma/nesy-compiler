#!/usr/bin/env python3
"""
Step 124: Test Comparison

Compares test runs to identify differences and changes.

PBTSO Phase: VERIFY, OBSERVE
Bus Topics:
- test.compare.request (subscribes)
- test.compare.result (emits)

Dependencies: Steps 101-123 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
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

class DiffType(Enum):
    """Types of differences between test runs."""
    NEW_TEST = "new_test"
    REMOVED_TEST = "removed_test"
    STATUS_CHANGE = "status_change"
    DURATION_CHANGE = "duration_change"
    NEW_FAILURE = "new_failure"
    FIXED_FAILURE = "fixed_failure"
    FLAKY_DETECTED = "flaky_detected"
    COVERAGE_CHANGE = "coverage_change"


class ChangeDirection(Enum):
    """Direction of change."""
    IMPROVED = "improved"
    DEGRADED = "degraded"
    UNCHANGED = "unchanged"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TestRunData:
    """Data from a test run for comparison."""
    run_id: str
    timestamp: float
    tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    coverage: Optional[float] = None
    commit_sha: Optional[str] = None
    branch: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestRunData":
        """Create from dictionary."""
        tests = {}
        for result in data.get("results", data.get("test_results", [])):
            test_name = result.get("test_name", result.get("name", ""))
            tests[test_name] = result

        return cls(
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", 0),
            tests=tests,
            summary=data.get("summary", {}),
            coverage=data.get("coverage_percent", data.get("coverage")),
            commit_sha=data.get("commit_sha"),
            branch=data.get("branch"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "test_count": len(self.tests),
            "coverage": self.coverage,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
        }


@dataclass
class TestDiff:
    """A difference between two test runs."""
    diff_type: DiffType
    test_name: str
    base_value: Any = None
    compare_value: Any = None
    change_direction: ChangeDirection = ChangeDirection.UNCHANGED
    significance: float = 0.0  # 0-1 scale
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "diff_type": self.diff_type.value,
            "test_name": self.test_name,
            "base_value": self.base_value,
            "compare_value": self.compare_value,
            "change_direction": self.change_direction.value,
            "significance": self.significance,
            "details": self.details,
        }


@dataclass
class CompareConfig:
    """
    Configuration for test comparison.

    Attributes:
        duration_threshold_percent: Threshold for duration change detection
        coverage_threshold_percent: Threshold for coverage change detection
        include_unchanged: Include unchanged tests in report
        group_by_type: Group differences by type
        output_dir: Output directory for reports
    """
    duration_threshold_percent: float = 20.0
    coverage_threshold_percent: float = 1.0
    include_unchanged: bool = False
    group_by_type: bool = True
    output_dir: str = ".pluribus/test-agent/compare"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration_threshold_percent": self.duration_threshold_percent,
            "coverage_threshold_percent": self.coverage_threshold_percent,
            "include_unchanged": self.include_unchanged,
            "group_by_type": self.group_by_type,
        }


@dataclass
class CompareResult:
    """Result of comparing two test runs."""
    base_run_id: str
    compare_run_id: str
    compared_at: float
    total_diffs: int = 0
    diffs: List[TestDiff] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    base_summary: Dict[str, Any] = field(default_factory=dict)
    compare_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        """Check if there are regressions."""
        return any(
            d.change_direction == ChangeDirection.DEGRADED
            for d in self.diffs
        )

    @property
    def regression_count(self) -> int:
        """Count regressions."""
        return sum(
            1 for d in self.diffs
            if d.change_direction == ChangeDirection.DEGRADED
        )

    @property
    def improvement_count(self) -> int:
        """Count improvements."""
        return sum(
            1 for d in self.diffs
            if d.change_direction == ChangeDirection.IMPROVED
        )

    def get_diffs_by_type(self, diff_type: DiffType) -> List[TestDiff]:
        """Get diffs of a specific type."""
        return [d for d in self.diffs if d.diff_type == diff_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_run_id": self.base_run_id,
            "compare_run_id": self.compare_run_id,
            "compared_at": self.compared_at,
            "total_diffs": self.total_diffs,
            "has_regressions": self.has_regressions,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
            "diffs": [d.to_dict() for d in self.diffs],
            "summary": self.summary,
            "base_summary": self.base_summary,
            "compare_summary": self.compare_summary,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class CompareBus:
    """Bus interface for comparison with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Test Comparator
# ============================================================================

class TestComparator:
    """
    Compares test runs to identify differences.

    Features:
    - Status change detection
    - Duration regression detection
    - Coverage change analysis
    - New/removed test detection
    - Report generation

    PBTSO Phase: VERIFY, OBSERVE
    Bus Topics: test.compare.request, test.compare.result
    """

    BUS_TOPICS = {
        "request": "test.compare.request",
        "result": "test.compare.result",
    }

    def __init__(self, bus=None, config: Optional[CompareConfig] = None):
        """
        Initialize the test comparator.

        Args:
            bus: Optional bus instance
            config: Comparison configuration
        """
        self.bus = bus or CompareBus()
        self.config = config or CompareConfig()

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def compare(
        self,
        base: TestRunData,
        compare: TestRunData,
    ) -> CompareResult:
        """
        Compare two test runs.

        Args:
            base: Base (older) test run
            compare: Compare (newer) test run

        Returns:
            CompareResult with differences
        """
        result = CompareResult(
            base_run_id=base.run_id,
            compare_run_id=compare.run_id,
            compared_at=time.time(),
            base_summary=base.to_dict(),
            compare_summary=compare.to_dict(),
        )

        diffs = []

        base_tests = set(base.tests.keys())
        compare_tests = set(compare.tests.keys())

        # New tests
        for test_name in compare_tests - base_tests:
            diffs.append(TestDiff(
                diff_type=DiffType.NEW_TEST,
                test_name=test_name,
                compare_value=compare.tests[test_name].get("status"),
                change_direction=ChangeDirection.UNCHANGED,
                significance=0.3,
            ))

        # Removed tests
        for test_name in base_tests - compare_tests:
            diffs.append(TestDiff(
                diff_type=DiffType.REMOVED_TEST,
                test_name=test_name,
                base_value=base.tests[test_name].get("status"),
                change_direction=ChangeDirection.UNCHANGED,
                significance=0.3,
            ))

        # Changed tests
        common_tests = base_tests & compare_tests
        for test_name in common_tests:
            base_test = base.tests[test_name]
            compare_test = compare.tests[test_name]

            # Status changes
            base_status = base_test.get("status")
            compare_status = compare_test.get("status")

            if base_status != compare_status:
                direction = ChangeDirection.UNCHANGED
                diff_type = DiffType.STATUS_CHANGE

                if base_status == "passed" and compare_status == "failed":
                    direction = ChangeDirection.DEGRADED
                    diff_type = DiffType.NEW_FAILURE
                elif base_status == "failed" and compare_status == "passed":
                    direction = ChangeDirection.IMPROVED
                    diff_type = DiffType.FIXED_FAILURE

                diffs.append(TestDiff(
                    diff_type=diff_type,
                    test_name=test_name,
                    base_value=base_status,
                    compare_value=compare_status,
                    change_direction=direction,
                    significance=0.8 if direction == ChangeDirection.DEGRADED else 0.5,
                    details={
                        "base_status": base_status,
                        "compare_status": compare_status,
                    },
                ))

            # Duration changes
            base_duration = base_test.get("duration_ms", 0)
            compare_duration = compare_test.get("duration_ms", 0)

            if base_duration > 0:
                duration_change = ((compare_duration - base_duration) / base_duration) * 100

                if abs(duration_change) >= self.config.duration_threshold_percent:
                    direction = ChangeDirection.UNCHANGED
                    if duration_change > 0:
                        direction = ChangeDirection.DEGRADED
                    else:
                        direction = ChangeDirection.IMPROVED

                    diffs.append(TestDiff(
                        diff_type=DiffType.DURATION_CHANGE,
                        test_name=test_name,
                        base_value=base_duration,
                        compare_value=compare_duration,
                        change_direction=direction,
                        significance=min(abs(duration_change) / 100, 1.0),
                        details={
                            "change_percent": duration_change,
                            "base_duration_ms": base_duration,
                            "compare_duration_ms": compare_duration,
                        },
                    ))

        # Coverage change
        if base.coverage is not None and compare.coverage is not None:
            coverage_change = compare.coverage - base.coverage

            if abs(coverage_change) >= self.config.coverage_threshold_percent:
                direction = ChangeDirection.UNCHANGED
                if coverage_change > 0:
                    direction = ChangeDirection.IMPROVED
                elif coverage_change < 0:
                    direction = ChangeDirection.DEGRADED

                diffs.append(TestDiff(
                    diff_type=DiffType.COVERAGE_CHANGE,
                    test_name="__overall__",
                    base_value=base.coverage,
                    compare_value=compare.coverage,
                    change_direction=direction,
                    significance=abs(coverage_change) / 10,
                    details={
                        "base_coverage": base.coverage,
                        "compare_coverage": compare.coverage,
                        "change": coverage_change,
                    },
                ))

        result.diffs = diffs
        result.total_diffs = len(diffs)

        # Generate summary
        result.summary = self._generate_summary(result, base, compare)

        # Emit result
        self._emit_event("result", {
            "base_run_id": base.run_id,
            "compare_run_id": compare.run_id,
            "total_diffs": result.total_diffs,
            "has_regressions": result.has_regressions,
        })

        return result

    def compare_from_files(
        self,
        base_path: str,
        compare_path: str,
    ) -> CompareResult:
        """Compare test runs from JSON files."""
        with open(base_path) as f:
            base_data = json.load(f)
        with open(compare_path) as f:
            compare_data = json.load(f)

        base = TestRunData.from_dict(base_data)
        compare = TestRunData.from_dict(compare_data)

        return self.compare(base, compare)

    def _generate_summary(
        self,
        result: CompareResult,
        base: TestRunData,
        compare: TestRunData,
    ) -> Dict[str, Any]:
        """Generate comparison summary."""
        summary = {
            "total_diffs": result.total_diffs,
            "regressions": result.regression_count,
            "improvements": result.improvement_count,
            "new_tests": len(result.get_diffs_by_type(DiffType.NEW_TEST)),
            "removed_tests": len(result.get_diffs_by_type(DiffType.REMOVED_TEST)),
            "new_failures": len(result.get_diffs_by_type(DiffType.NEW_FAILURE)),
            "fixed_failures": len(result.get_diffs_by_type(DiffType.FIXED_FAILURE)),
            "duration_changes": len(result.get_diffs_by_type(DiffType.DURATION_CHANGE)),
        }

        # Test count changes
        summary["base_test_count"] = len(base.tests)
        summary["compare_test_count"] = len(compare.tests)
        summary["test_count_change"] = summary["compare_test_count"] - summary["base_test_count"]

        # Status breakdown
        base_passed = sum(1 for t in base.tests.values() if t.get("status") == "passed")
        compare_passed = sum(1 for t in compare.tests.values() if t.get("status") == "passed")
        summary["pass_rate_change"] = 0
        if len(base.tests) > 0 and len(compare.tests) > 0:
            base_rate = base_passed / len(base.tests) * 100
            compare_rate = compare_passed / len(compare.tests) * 100
            summary["pass_rate_change"] = compare_rate - base_rate

        return summary

    def generate_report(
        self,
        result: CompareResult,
        format: str = "markdown",
    ) -> str:
        """Generate comparison report."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == "json":
            output_file = output_path / f"compare_{result.base_run_id[:8]}_{result.compare_run_id[:8]}.json"
            with open(output_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            return str(output_file)

        elif format == "markdown":
            output_file = output_path / f"compare_{result.base_run_id[:8]}_{result.compare_run_id[:8]}.md"

            status = "[REGRESSION]" if result.has_regressions else "[OK]"

            lines = [
                "# Test Run Comparison",
                "",
                f"**Status**: {status}",
                f"**Base Run**: `{result.base_run_id}`",
                f"**Compare Run**: `{result.compare_run_id}`",
                f"**Compared At**: {datetime.fromtimestamp(result.compared_at).isoformat()}",
                "",
                "## Summary",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]

            for key, value in result.summary.items():
                lines.append(f"| {key.replace('_', ' ').title()} | {value} |")

            # Group diffs by type
            if self.config.group_by_type:
                for diff_type in DiffType:
                    type_diffs = result.get_diffs_by_type(diff_type)
                    if type_diffs:
                        lines.extend([
                            "",
                            f"## {diff_type.value.replace('_', ' ').title()}",
                            "",
                        ])

                        if diff_type in (DiffType.NEW_FAILURE, DiffType.FIXED_FAILURE,
                                        DiffType.STATUS_CHANGE):
                            lines.append("| Test | Base | Compare |")
                            lines.append("|------|------|---------|")
                            for diff in type_diffs:
                                lines.append(f"| {diff.test_name} | {diff.base_value} | {diff.compare_value} |")

                        elif diff_type == DiffType.DURATION_CHANGE:
                            lines.append("| Test | Base | Compare | Change |")
                            lines.append("|------|------|---------|--------|")
                            for diff in type_diffs:
                                change = diff.details.get("change_percent", 0)
                                lines.append(
                                    f"| {diff.test_name} | "
                                    f"{diff.base_value:.0f}ms | "
                                    f"{diff.compare_value:.0f}ms | "
                                    f"{change:+.1f}% |"
                                )

                        elif diff_type in (DiffType.NEW_TEST, DiffType.REMOVED_TEST):
                            for diff in type_diffs:
                                lines.append(f"- `{diff.test_name}`")

            with open(output_file, "w") as f:
                f.write("\n".join(lines))

            return str(output_file)

        return ""

    async def compare_async(
        self,
        base: TestRunData,
        compare: TestRunData,
    ) -> CompareResult:
        """Async version of compare."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.compare, base, compare)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.compare.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "comparison",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Comparator."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Run Comparator")
    parser.add_argument("base", help="Base run JSON file")
    parser.add_argument("compare", help="Compare run JSON file")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/compare")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--duration-threshold", type=float, default=20.0,
                       help="Duration change threshold percent")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    config = CompareConfig(
        duration_threshold_percent=args.duration_threshold,
        output_dir=args.output,
    )

    comparator = TestComparator(config=config)

    try:
        result = comparator.compare_from_files(args.base, args.compare)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading files: {e}")
        exit(1)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # Console output
        status = "[REGRESSION]" if result.has_regressions else "[OK]"

        print(f"\n{'='*60}")
        print(f"Test Run Comparison {status}")
        print(f"{'='*60}")
        print(f"Base Run: {result.base_run_id}")
        print(f"Compare Run: {result.compare_run_id}")
        print()

        print("Summary:")
        print(f"  Total Differences: {result.total_diffs}")
        print(f"  Regressions: {result.regression_count}")
        print(f"  Improvements: {result.improvement_count}")
        print(f"  New Failures: {len(result.get_diffs_by_type(DiffType.NEW_FAILURE))}")
        print(f"  Fixed Failures: {len(result.get_diffs_by_type(DiffType.FIXED_FAILURE))}")

        # Show new failures
        new_failures = result.get_diffs_by_type(DiffType.NEW_FAILURE)
        if new_failures:
            print(f"\nNew Failures:")
            for diff in new_failures[:10]:
                print(f"  [FAIL] {diff.test_name}")

        # Show fixed failures
        fixed = result.get_diffs_by_type(DiffType.FIXED_FAILURE)
        if fixed:
            print(f"\nFixed Failures:")
            for diff in fixed[:10]:
                print(f"  [FIXED] {diff.test_name}")

        # Generate report
        report_path = comparator.generate_report(result, args.format)
        print(f"\nReport: {report_path}")

        print(f"{'='*60}\n")

        if result.has_regressions:
            exit(1)


if __name__ == "__main__":
    main()
