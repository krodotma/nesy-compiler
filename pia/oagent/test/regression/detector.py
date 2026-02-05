#!/usr/bin/env python3
"""
Step 116: Regression Detector

Identifies test regressions by comparing current results with history.

PBTSO Phase: VERIFY
Bus Topics:
- test.regression.detect (subscribes)
- test.regression.found (emits)
- a2a.review.alert (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

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

DEFAULT_HISTORY_LOOKBACK = 10  # Number of runs to analyze
DEFAULT_FLAKE_THRESHOLD = 3  # Times a test must fail/pass to be considered flaky
PERFORMANCE_REGRESSION_THRESHOLD = 0.2  # 20% slowdown


class RegressionType(Enum):
    """Types of regressions."""
    NEW_FAILURE = "new_failure"  # Previously passing test now fails
    PERFORMANCE = "performance"  # Test became significantly slower
    FLAKY = "flaky"  # Test became flaky
    COVERAGE_DROP = "coverage_drop"  # Code coverage decreased
    ERROR_RATE = "error_rate"  # Error rate increased
    NEW_ERROR = "new_error"  # New type of error appeared


class RegressionSeverity(Enum):
    """Severity levels for regressions."""
    CRITICAL = "critical"  # Blocking regression
    HIGH = "high"  # Significant regression
    MEDIUM = "medium"  # Notable regression
    LOW = "low"  # Minor regression


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TestHistory:
    """Historical data for a test."""
    test_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    avg_duration_ms: float = 0
    pass_rate: float = 1.0
    last_status: str = "passed"
    flake_count: int = 0

    def add_result(self, status: str, duration_ms: float, run_id: str):
        """Add a result to history."""
        self.results.append({
            "status": status,
            "duration_ms": duration_ms,
            "run_id": run_id,
            "timestamp": time.time(),
        })
        self._recalculate_stats()

    def _recalculate_stats(self):
        """Recalculate statistics from history."""
        if not self.results:
            return

        durations = [r["duration_ms"] for r in self.results if r.get("duration_ms")]
        self.avg_duration_ms = sum(durations) / len(durations) if durations else 0

        passed = sum(1 for r in self.results if r["status"] == "passed")
        self.pass_rate = passed / len(self.results)

        self.last_status = self.results[-1]["status"]

        # Count flakes (status changes)
        self.flake_count = sum(
            1 for i in range(1, len(self.results))
            if self.results[i]["status"] != self.results[i-1]["status"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "avg_duration_ms": self.avg_duration_ms,
            "pass_rate": self.pass_rate,
            "last_status": self.last_status,
            "flake_count": self.flake_count,
            "result_count": len(self.results),
        }


@dataclass
class Regression:
    """A detected regression."""
    id: str
    regression_type: RegressionType
    severity: RegressionSeverity
    test_name: str
    description: str
    current_value: Any
    previous_value: Any
    delta: float
    commit_sha: Optional[str] = None
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "regression_type": self.regression_type.value,
            "severity": self.severity.value,
            "test_name": self.test_name,
            "description": self.description,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "delta": self.delta,
            "commit_sha": self.commit_sha,
            "detected_at": self.detected_at,
            "resolved": self.resolved,
            "metadata": self.metadata,
        }


@dataclass
class RegressionConfig:
    """
    Configuration for regression detection.

    Attributes:
        history_lookback: Number of historical runs to analyze
        performance_threshold: Threshold for performance regression
        flake_threshold: Threshold for flaky test detection
        coverage_threshold: Threshold for coverage regression
        history_dir: Directory for test history storage
        output_dir: Directory for regression reports
    """
    history_lookback: int = DEFAULT_HISTORY_LOOKBACK
    performance_threshold: float = PERFORMANCE_REGRESSION_THRESHOLD
    flake_threshold: int = DEFAULT_FLAKE_THRESHOLD
    coverage_threshold: float = 0.05  # 5% drop
    history_dir: str = ".pluribus/test-agent/history"
    output_dir: str = ".pluribus/test-agent/regressions"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "history_lookback": self.history_lookback,
            "performance_threshold": self.performance_threshold,
            "flake_threshold": self.flake_threshold,
            "coverage_threshold": self.coverage_threshold,
        }


@dataclass
class RegressionResult:
    """Result of regression detection."""
    run_id: str
    analyzed_at: float
    current_run_id: str
    regressions: List[Regression] = field(default_factory=list)
    tests_analyzed: int = 0
    new_failures: int = 0
    performance_regressions: int = 0
    flaky_tests: int = 0
    coverage_regressions: int = 0

    @property
    def total_regressions(self) -> int:
        """Get total number of regressions."""
        return len(self.regressions)

    @property
    def has_critical(self) -> bool:
        """Check if there are critical regressions."""
        return any(r.severity == RegressionSeverity.CRITICAL for r in self.regressions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "analyzed_at": self.analyzed_at,
            "current_run_id": self.current_run_id,
            "total_regressions": self.total_regressions,
            "has_critical": self.has_critical,
            "tests_analyzed": self.tests_analyzed,
            "new_failures": self.new_failures,
            "performance_regressions": self.performance_regressions,
            "flaky_tests": self.flaky_tests,
            "coverage_regressions": self.coverage_regressions,
            "regressions": [r.to_dict() for r in self.regressions],
        }


# ============================================================================
# Bus Interface
# ============================================================================

class RegressionBus:
    """Bus interface for regression detection."""

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
# Regression Detector
# ============================================================================

class RegressionDetector:
    """
    Detects test regressions by analyzing test history.

    Features:
    - New failure detection
    - Performance regression detection
    - Flaky test identification
    - Coverage regression detection
    - Historical trend analysis

    PBTSO Phase: VERIFY
    Bus Topics: test.regression.detect, test.regression.found
    """

    BUS_TOPICS = {
        "detect": "test.regression.detect",
        "found": "test.regression.found",
        "alert": "a2a.review.alert",
    }

    def __init__(self, bus=None, config: Optional[RegressionConfig] = None):
        """
        Initialize the regression detector.

        Args:
            bus: Optional bus instance for event emission
            config: Detection configuration
        """
        self.bus = bus or RegressionBus()
        self.config = config or RegressionConfig()
        self._history: Dict[str, TestHistory] = {}
        self._load_history()

    def _load_history(self) -> None:
        """Load test history from storage."""
        history_path = Path(self.config.history_dir)
        history_file = history_path / "test_history.json"

        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    for test_name, hist_data in data.items():
                        history = TestHistory(test_name=test_name)
                        history.results = hist_data.get("results", [])
                        history._recalculate_stats()
                        self._history[test_name] = history
            except (json.JSONDecodeError, IOError):
                pass

    def _save_history(self) -> None:
        """Save test history to storage."""
        history_path = Path(self.config.history_dir)
        history_path.mkdir(parents=True, exist_ok=True)
        history_file = history_path / "test_history.json"

        data = {}
        for test_name, history in self._history.items():
            # Keep only recent results
            data[test_name] = {
                "results": history.results[-self.config.history_lookback:],
            }

        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def detect_regressions(
        self,
        test_results: List[Dict[str, Any]],
        run_id: str,
        coverage_data: Optional[Dict[str, Any]] = None,
        commit_sha: Optional[str] = None,
    ) -> RegressionResult:
        """
        Detect regressions in test results.

        Args:
            test_results: List of test result dictionaries
            run_id: ID of the current test run
            coverage_data: Optional coverage data
            commit_sha: Optional commit SHA

        Returns:
            RegressionResult with detected regressions
        """
        detection_id = str(uuid.uuid4())
        result = RegressionResult(
            run_id=detection_id,
            analyzed_at=time.time(),
            current_run_id=run_id,
            tests_analyzed=len(test_results),
        )

        # Analyze each test
        for test_data in test_results:
            test_name = test_data.get("name") or test_data.get("test_name", "")
            status = test_data.get("status", "unknown")
            duration_ms = test_data.get("duration_ms", 0)

            # Get or create history
            if test_name not in self._history:
                self._history[test_name] = TestHistory(test_name=test_name)

            history = self._history[test_name]

            # Check for regressions
            regressions = self._analyze_test(
                test_name, status, duration_ms, history, commit_sha
            )
            result.regressions.extend(regressions)

            # Update history
            history.add_result(status, duration_ms, run_id)

            # Update counters
            for reg in regressions:
                if reg.regression_type == RegressionType.NEW_FAILURE:
                    result.new_failures += 1
                elif reg.regression_type == RegressionType.PERFORMANCE:
                    result.performance_regressions += 1
                elif reg.regression_type == RegressionType.FLAKY:
                    result.flaky_tests += 1

        # Check coverage regression
        if coverage_data:
            coverage_regressions = self._check_coverage_regression(
                coverage_data, commit_sha
            )
            result.regressions.extend(coverage_regressions)
            result.coverage_regressions = len(coverage_regressions)

        # Emit events for regressions
        for regression in result.regressions:
            self._emit_event("found", {
                "regression_id": regression.id,
                "type": regression.regression_type.value,
                "severity": regression.severity.value,
                "test_name": regression.test_name,
            })

            # Alert for critical regressions
            if regression.severity == RegressionSeverity.CRITICAL:
                self._emit_event("alert", {
                    "alert_type": "regression",
                    "severity": "critical",
                    "message": regression.description,
                    "regression_id": regression.id,
                })

        # Save updated history
        self._save_history()

        # Save regression report
        self._save_report(result)

        return result

    def _analyze_test(
        self,
        test_name: str,
        status: str,
        duration_ms: float,
        history: TestHistory,
        commit_sha: Optional[str],
    ) -> List[Regression]:
        """Analyze a single test for regressions."""
        regressions = []

        # No history = no regression detection possible
        if not history.results:
            return regressions

        # Check for new failure
        if status == "failed" and history.last_status == "passed":
            # Check if it's a real new failure (not just flaky)
            recent_passes = sum(
                1 for r in history.results[-self.config.flake_threshold:]
                if r["status"] == "passed"
            )

            if recent_passes >= self.config.flake_threshold - 1:
                regressions.append(Regression(
                    id=str(uuid.uuid4()),
                    regression_type=RegressionType.NEW_FAILURE,
                    severity=RegressionSeverity.CRITICAL,
                    test_name=test_name,
                    description=f"Test '{test_name}' started failing",
                    current_value="failed",
                    previous_value="passed",
                    delta=0,
                    commit_sha=commit_sha,
                ))

        # Check for performance regression
        if duration_ms > 0 and history.avg_duration_ms > 0:
            slowdown = (duration_ms - history.avg_duration_ms) / history.avg_duration_ms

            if slowdown > self.config.performance_threshold:
                severity = RegressionSeverity.HIGH if slowdown > 0.5 else RegressionSeverity.MEDIUM

                regressions.append(Regression(
                    id=str(uuid.uuid4()),
                    regression_type=RegressionType.PERFORMANCE,
                    severity=severity,
                    test_name=test_name,
                    description=f"Test '{test_name}' became {slowdown:.0%} slower",
                    current_value=duration_ms,
                    previous_value=history.avg_duration_ms,
                    delta=slowdown,
                    commit_sha=commit_sha,
                    metadata={
                        "current_ms": duration_ms,
                        "avg_ms": history.avg_duration_ms,
                    },
                ))

        # Check for flakiness
        if history.flake_count >= self.config.flake_threshold:
            # Only report once
            if not any(
                r["status"] == "flaky_reported"
                for r in history.results[-self.config.flake_threshold:]
            ):
                regressions.append(Regression(
                    id=str(uuid.uuid4()),
                    regression_type=RegressionType.FLAKY,
                    severity=RegressionSeverity.MEDIUM,
                    test_name=test_name,
                    description=f"Test '{test_name}' is flaky ({history.flake_count} status changes)",
                    current_value=history.pass_rate,
                    previous_value=1.0,
                    delta=1.0 - history.pass_rate,
                    commit_sha=commit_sha,
                    metadata={
                        "pass_rate": history.pass_rate,
                        "flake_count": history.flake_count,
                    },
                ))

        return regressions

    def _check_coverage_regression(
        self,
        coverage_data: Dict[str, Any],
        commit_sha: Optional[str],
    ) -> List[Regression]:
        """Check for coverage regressions."""
        regressions = []

        current_coverage = coverage_data.get("line_coverage_percent", 0)

        # Load previous coverage
        history_path = Path(self.config.history_dir) / "coverage_history.json"
        previous_coverage = 0

        if history_path.exists():
            try:
                with open(history_path) as f:
                    history = json.load(f)
                    if history:
                        previous_coverage = history[-1].get("coverage", 0)
            except (json.JSONDecodeError, IOError):
                pass

        # Check for regression
        if previous_coverage > 0:
            drop = (previous_coverage - current_coverage) / previous_coverage

            if drop > self.config.coverage_threshold:
                regressions.append(Regression(
                    id=str(uuid.uuid4()),
                    regression_type=RegressionType.COVERAGE_DROP,
                    severity=RegressionSeverity.HIGH if drop > 0.1 else RegressionSeverity.MEDIUM,
                    test_name="__coverage__",
                    description=f"Code coverage dropped by {drop:.1%}",
                    current_value=current_coverage,
                    previous_value=previous_coverage,
                    delta=drop,
                    commit_sha=commit_sha,
                ))

        # Update coverage history
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if history_path.exists():
            try:
                with open(history_path) as f:
                    history = json.load(f)
            except:
                pass

        history.append({
            "coverage": current_coverage,
            "timestamp": time.time(),
            "commit_sha": commit_sha,
        })
        history = history[-self.config.history_lookback:]

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        return regressions

    def _save_report(self, result: RegressionResult) -> None:
        """Save regression report."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        report_file = output_path / f"regression_report_{result.run_id}.json"
        with open(report_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Generate markdown
        report_md = output_path / f"regression_report_{result.run_id}.md"
        with open(report_md, "w") as f:
            f.write(self._generate_markdown_report(result))

    def _generate_markdown_report(self, result: RegressionResult) -> str:
        """Generate markdown regression report."""
        lines = [
            "# Regression Report",
            "",
            f"**Run ID**: {result.run_id}",
            f"**Date**: {datetime.fromtimestamp(result.analyzed_at).isoformat()}",
            "",
            "## Summary",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Tests Analyzed | {result.tests_analyzed} |",
            f"| Total Regressions | {result.total_regressions} |",
            f"| New Failures | {result.new_failures} |",
            f"| Performance Regressions | {result.performance_regressions} |",
            f"| Flaky Tests | {result.flaky_tests} |",
            f"| Coverage Regressions | {result.coverage_regressions} |",
        ]

        if result.regressions:
            lines.extend([
                "",
                "## Regressions",
                "",
            ])

            # Group by severity
            for severity in RegressionSeverity:
                severity_regressions = [
                    r for r in result.regressions if r.severity == severity
                ]
                if severity_regressions:
                    lines.append(f"### {severity.value.upper()}")
                    lines.append("")

                    for reg in severity_regressions:
                        lines.append(f"- **{reg.test_name}** ({reg.regression_type.value})")
                        lines.append(f"  - {reg.description}")
                        if reg.commit_sha:
                            lines.append(f"  - Commit: {reg.commit_sha}")
                        lines.append("")

        return "\n".join(lines)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.regression.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "regression_detection",
            "actor": "test-agent",
            "data": data,
        })

    def get_test_history(self, test_name: str) -> Optional[TestHistory]:
        """Get history for a specific test."""
        return self._history.get(test_name)

    def get_all_flaky_tests(self) -> List[str]:
        """Get list of all flaky tests."""
        return [
            name for name, history in self._history.items()
            if history.flake_count >= self.config.flake_threshold
        ]


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Regression Detector."""
    import argparse

    parser = argparse.ArgumentParser(description="Regression Detector")
    parser.add_argument("results_file", help="Path to test results JSON")
    parser.add_argument("--run-id", default=str(uuid.uuid4()), help="Test run ID")
    parser.add_argument("--coverage", help="Path to coverage data JSON")
    parser.add_argument("--commit", help="Commit SHA")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/regressions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Load test results
    try:
        with open(args.results_file) as f:
            test_results = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading results: {e}")
        exit(1)

    # Load coverage if provided
    coverage_data = None
    if args.coverage:
        try:
            with open(args.coverage) as f:
                coverage_data = json.load(f)
        except:
            pass

    config = RegressionConfig(output_dir=args.output)
    detector = RegressionDetector(config=config)

    result = detector.detect_regressions(
        test_results=test_results,
        run_id=args.run_id,
        coverage_data=coverage_data,
        commit_sha=args.commit,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Regression Detection Complete")
        print(f"{'='*60}")
        print(f"Tests Analyzed: {result.tests_analyzed}")
        print(f"Regressions Found: {result.total_regressions}")
        print(f"  - New Failures: {result.new_failures}")
        print(f"  - Performance: {result.performance_regressions}")
        print(f"  - Flaky: {result.flaky_tests}")
        print(f"  - Coverage: {result.coverage_regressions}")
        print(f"{'='*60}")

        if result.has_critical:
            print(f"\nCRITICAL regressions detected!")
            exit(1)


if __name__ == "__main__":
    main()
