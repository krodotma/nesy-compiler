#!/usr/bin/env python3
"""
Step 118: Flaky Test Detector

Identifies and manages flaky tests through multiple run analysis.

PBTSO Phase: VERIFY, TEST
Bus Topics:
- test.flaky.detect (subscribes)
- test.flaky.detected (emits)
- test.flaky.quarantine (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import asyncio
import json
import os
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

DEFAULT_RETRY_COUNT = 3
DEFAULT_FLAKY_THRESHOLD = 0.2  # 20% failure rate considered flaky
DEFAULT_CONSISTENT_RUNS = 5  # Runs to consider a test stable


class FlakyClassification(Enum):
    """Classification of flaky test root cause."""
    TIMING = "timing"  # Race condition or timing issue
    ORDERING = "ordering"  # Test order dependency
    RESOURCE = "resource"  # Shared resource contention
    NETWORK = "network"  # Network-related flakiness
    RANDOM = "random"  # Random data generation
    ENVIRONMENT = "environment"  # Environment differences
    UNKNOWN = "unknown"  # Cause undetermined


class QuarantineStatus(Enum):
    """Quarantine status for a flaky test."""
    ACTIVE = "active"  # Test is quarantined
    MONITORING = "monitoring"  # Test is being monitored
    RELEASED = "released"  # Test is no longer flaky
    PERMANENT = "permanent"  # Test is permanently quarantined


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TestRun:
    """Result of a single test run."""
    run_id: str
    status: str
    duration_ms: float
    timestamp: float
    error_message: Optional[str] = None
    stdout: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
        }


@dataclass
class FlakyTest:
    """Information about a flaky test."""
    test_name: str
    test_path: str
    flakiness_score: float
    classification: FlakyClassification
    run_history: List[TestRun] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    first_detected: float = field(default_factory=time.time)
    last_flake: Optional[float] = None
    quarantine_status: QuarantineStatus = QuarantineStatus.MONITORING
    root_cause_hints: List[str] = field(default_factory=list)
    related_tests: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_runs(self) -> int:
        """Get total number of runs."""
        return self.pass_count + self.fail_count

    @property
    def failure_rate(self) -> float:
        """Get failure rate."""
        if self.total_runs == 0:
            return 0.0
        return self.fail_count / self.total_runs

    def add_run(self, status: str, duration_ms: float, error_message: Optional[str] = None):
        """Add a test run result."""
        run = TestRun(
            run_id=str(uuid.uuid4()),
            status=status,
            duration_ms=duration_ms,
            timestamp=time.time(),
            error_message=error_message,
        )
        self.run_history.append(run)

        if status == "passed":
            self.pass_count += 1
        else:
            self.fail_count += 1
            self.last_flake = time.time()

        # Update flakiness score
        self._update_flakiness_score()

    def _update_flakiness_score(self):
        """Update flakiness score based on recent history."""
        if not self.run_history:
            self.flakiness_score = 0.0
            return

        # Calculate recent failure rate with recency weighting
        recent = self.run_history[-10:]
        weighted_failures = 0
        total_weight = 0

        for i, run in enumerate(recent):
            weight = (i + 1) / len(recent)  # More recent = higher weight
            total_weight += weight
            if run.status != "passed":
                weighted_failures += weight

        if total_weight > 0:
            weighted_rate = weighted_failures / total_weight
        else:
            weighted_rate = 0

        # Also consider status changes (actual flakiness)
        status_changes = sum(
            1 for i in range(1, len(recent))
            if recent[i].status != recent[i-1].status
        )
        change_rate = status_changes / max(len(recent) - 1, 1)

        # Combine metrics
        self.flakiness_score = (weighted_rate + change_rate) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "test_path": self.test_path,
            "flakiness_score": self.flakiness_score,
            "classification": self.classification.value,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "total_runs": self.total_runs,
            "failure_rate": self.failure_rate,
            "first_detected": self.first_detected,
            "last_flake": self.last_flake,
            "quarantine_status": self.quarantine_status.value,
            "root_cause_hints": self.root_cause_hints,
            "related_tests": self.related_tests,
        }


@dataclass
class FlakyConfig:
    """
    Configuration for flaky test detection.

    Attributes:
        retry_count: Number of retries to detect flakiness
        flaky_threshold: Failure rate to consider a test flaky
        consistent_runs: Consecutive passes to release from quarantine
        quarantine_enabled: Whether to quarantine flaky tests
        rerun_flaky: Whether to rerun flaky tests automatically
        history_dir: Directory for flaky test history
        output_dir: Directory for reports
    """
    retry_count: int = DEFAULT_RETRY_COUNT
    flaky_threshold: float = DEFAULT_FLAKY_THRESHOLD
    consistent_runs: int = DEFAULT_CONSISTENT_RUNS
    quarantine_enabled: bool = True
    rerun_flaky: bool = True
    history_dir: str = ".pluribus/test-agent/flaky"
    output_dir: str = ".pluribus/test-agent/flaky/reports"
    test_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "retry_count": self.retry_count,
            "flaky_threshold": self.flaky_threshold,
            "consistent_runs": self.consistent_runs,
            "quarantine_enabled": self.quarantine_enabled,
            "rerun_flaky": self.rerun_flaky,
        }


@dataclass
class FlakyResult:
    """Result of flaky test detection."""
    run_id: str
    analyzed_at: float
    tests_analyzed: int = 0
    flaky_tests: List[FlakyTest] = field(default_factory=list)
    newly_detected: List[str] = field(default_factory=list)
    quarantined: List[str] = field(default_factory=list)
    released: List[str] = field(default_factory=list)

    @property
    def total_flaky(self) -> int:
        """Get total number of flaky tests."""
        return len(self.flaky_tests)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "analyzed_at": self.analyzed_at,
            "tests_analyzed": self.tests_analyzed,
            "total_flaky": self.total_flaky,
            "newly_detected": self.newly_detected,
            "quarantined": self.quarantined,
            "released": self.released,
            "flaky_tests": [t.to_dict() for t in self.flaky_tests],
        }


# ============================================================================
# Bus Interface
# ============================================================================

class FlakyBus:
    """Bus interface for flaky test detection."""

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
# Flaky Test Detector
# ============================================================================

class FlakyDetector:
    """
    Detects and manages flaky tests.

    Detection strategies:
    - Multiple retries: Run test multiple times to detect inconsistency
    - Historical analysis: Analyze past run history for patterns
    - Classification: Attempt to determine root cause

    PBTSO Phase: VERIFY, TEST
    Bus Topics: test.flaky.detect, test.flaky.detected
    """

    BUS_TOPICS = {
        "detect": "test.flaky.detect",
        "detected": "test.flaky.detected",
        "quarantine": "test.flaky.quarantine",
    }

    # Patterns for classifying flaky tests
    CLASSIFICATION_PATTERNS = {
        FlakyClassification.TIMING: [
            "timeout", "deadline", "slow", "race", "concurrent",
            "sleep", "wait", "async", "eventual",
        ],
        FlakyClassification.NETWORK: [
            "connection", "socket", "http", "api", "request",
            "network", "dns", "host", "timeout",
        ],
        FlakyClassification.RESOURCE: [
            "file", "database", "lock", "memory", "disk",
            "permission", "resource", "pool",
        ],
        FlakyClassification.RANDOM: [
            "random", "uuid", "generate", "faker",
        ],
        FlakyClassification.ORDERING: [
            "order", "sequence", "depend", "before", "after",
            "setup", "teardown", "fixture",
        ],
    }

    def __init__(self, bus=None, config: Optional[FlakyConfig] = None):
        """
        Initialize the flaky test detector.

        Args:
            bus: Optional bus instance for event emission
            config: Detection configuration
        """
        self.bus = bus or FlakyBus()
        self.config = config or FlakyConfig()
        self._flaky_tests: Dict[str, FlakyTest] = {}
        self._load_history()

    def _load_history(self) -> None:
        """Load flaky test history from storage."""
        history_path = Path(self.config.history_dir) / "flaky_tests.json"

        if history_path.exists():
            try:
                with open(history_path) as f:
                    data = json.load(f)
                    for test_name, test_data in data.items():
                        flaky = FlakyTest(
                            test_name=test_name,
                            test_path=test_data.get("test_path", ""),
                            flakiness_score=test_data.get("flakiness_score", 0),
                            classification=FlakyClassification(
                                test_data.get("classification", "unknown")
                            ),
                            pass_count=test_data.get("pass_count", 0),
                            fail_count=test_data.get("fail_count", 0),
                            first_detected=test_data.get("first_detected", time.time()),
                            last_flake=test_data.get("last_flake"),
                            quarantine_status=QuarantineStatus(
                                test_data.get("quarantine_status", "monitoring")
                            ),
                            root_cause_hints=test_data.get("root_cause_hints", []),
                        )
                        self._flaky_tests[test_name] = flaky
            except (json.JSONDecodeError, IOError):
                pass

    def _save_history(self) -> None:
        """Save flaky test history to storage."""
        history_path = Path(self.config.history_dir)
        history_path.mkdir(parents=True, exist_ok=True)
        history_file = history_path / "flaky_tests.json"

        data = {name: test.to_dict() for name, test in self._flaky_tests.items()}

        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def detect_flaky_tests(
        self,
        tests: List[str],
        failed_tests: Optional[List[str]] = None,
    ) -> FlakyResult:
        """
        Detect flaky tests through multiple runs.

        Args:
            tests: List of test names/paths to analyze
            failed_tests: Optional list of initially failed tests

        Returns:
            FlakyResult with detection results
        """
        run_id = str(uuid.uuid4())
        result = FlakyResult(
            run_id=run_id,
            analyzed_at=time.time(),
            tests_analyzed=len(tests),
        )

        # Determine which tests to analyze
        tests_to_analyze = failed_tests if failed_tests else tests

        for test_name in tests_to_analyze:
            is_flaky = self._analyze_test(test_name)

            if is_flaky:
                flaky = self._flaky_tests.get(test_name)
                if flaky:
                    result.flaky_tests.append(flaky)

                    if flaky.total_runs == self.config.retry_count:
                        # Newly detected
                        result.newly_detected.append(test_name)

                        self._emit_event("detected", {
                            "test_name": test_name,
                            "flakiness_score": flaky.flakiness_score,
                            "classification": flaky.classification.value,
                        })

        # Update quarantine status
        for name, flaky in self._flaky_tests.items():
            if self._should_quarantine(flaky):
                if flaky.quarantine_status != QuarantineStatus.ACTIVE:
                    flaky.quarantine_status = QuarantineStatus.ACTIVE
                    result.quarantined.append(name)

                    self._emit_event("quarantine", {
                        "test_name": name,
                        "status": "quarantined",
                        "flakiness_score": flaky.flakiness_score,
                    })

            elif self._should_release(flaky):
                flaky.quarantine_status = QuarantineStatus.RELEASED
                result.released.append(name)

                self._emit_event("quarantine", {
                    "test_name": name,
                    "status": "released",
                })

        # Save updated history
        self._save_history()

        # Generate report
        self._save_report(result)

        return result

    def _analyze_test(self, test_name: str) -> bool:
        """
        Analyze a single test for flakiness.

        Returns True if test is determined to be flaky.
        """
        # Get or create flaky test record
        if test_name not in self._flaky_tests:
            self._flaky_tests[test_name] = FlakyTest(
                test_name=test_name,
                test_path=test_name,
                flakiness_score=0,
                classification=FlakyClassification.UNKNOWN,
            )

        flaky = self._flaky_tests[test_name]

        # Run test multiple times
        results = self._run_test_multiple(test_name)

        # Record results
        for status, duration, error in results:
            flaky.add_run(status, duration, error)

        # Classify if flaky
        if flaky.flakiness_score >= self.config.flaky_threshold:
            flaky.classification = self._classify_flakiness(flaky)
            return True

        return False

    def _run_test_multiple(
        self,
        test_name: str,
    ) -> List[Tuple[str, float, Optional[str]]]:
        """Run a test multiple times."""
        results = []

        for _ in range(self.config.retry_count):
            status, duration, error = self._run_single_test(test_name)
            results.append((status, duration, error))

        return results

    def _run_single_test(
        self,
        test_name: str,
    ) -> Tuple[str, float, Optional[str]]:
        """Run a single test and return result."""
        if self.config.test_command:
            command = self.config.test_command.format(test=test_name)
        else:
            command = f"python -m pytest {test_name} -x -v --tb=short"

        start = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            duration = (time.time() - start) * 1000

            if result.returncode == 0:
                return ("passed", duration, None)
            else:
                return ("failed", duration, result.stdout + result.stderr)

        except subprocess.TimeoutExpired:
            return ("timeout", 60000, "Test timed out")
        except Exception as e:
            return ("error", (time.time() - start) * 1000, str(e))

    def _classify_flakiness(self, flaky: FlakyTest) -> FlakyClassification:
        """Attempt to classify the root cause of flakiness."""
        # Collect error messages
        error_text = " ".join(
            run.error_message or ""
            for run in flaky.run_history
            if run.status != "passed"
        ).lower()

        # Add test name for context
        error_text += " " + flaky.test_name.lower()

        # Check patterns
        best_match = FlakyClassification.UNKNOWN
        best_score = 0

        for classification, patterns in self.CLASSIFICATION_PATTERNS.items():
            score = sum(1 for p in patterns if p in error_text)
            if score > best_score:
                best_score = score
                best_match = classification

                # Add hint
                matching_patterns = [p for p in patterns if p in error_text]
                if matching_patterns:
                    flaky.root_cause_hints.append(
                        f"Matches {classification.value} patterns: {matching_patterns[:3]}"
                    )

        return best_match

    def _should_quarantine(self, flaky: FlakyTest) -> bool:
        """Determine if a test should be quarantined."""
        if not self.config.quarantine_enabled:
            return False

        return (
            flaky.flakiness_score >= self.config.flaky_threshold and
            flaky.total_runs >= self.config.retry_count
        )

    def _should_release(self, flaky: FlakyTest) -> bool:
        """Determine if a test should be released from quarantine."""
        if flaky.quarantine_status != QuarantineStatus.ACTIVE:
            return False

        # Check recent runs
        recent = flaky.run_history[-self.config.consistent_runs:]

        if len(recent) < self.config.consistent_runs:
            return False

        return all(run.status == "passed" for run in recent)

    def _save_report(self, result: FlakyResult) -> None:
        """Save flaky test report."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        report_file = output_path / f"flaky_report_{result.run_id}.json"
        with open(report_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Generate markdown
        report_md = output_path / f"flaky_report_{result.run_id}.md"
        with open(report_md, "w") as f:
            f.write(self._generate_markdown_report(result))

    def _generate_markdown_report(self, result: FlakyResult) -> str:
        """Generate markdown flaky test report."""
        lines = [
            "# Flaky Test Report",
            "",
            f"**Run ID**: {result.run_id}",
            f"**Date**: {datetime.fromtimestamp(result.analyzed_at).isoformat()}",
            "",
            "## Summary",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Tests Analyzed | {result.tests_analyzed} |",
            f"| Flaky Tests | {result.total_flaky} |",
            f"| Newly Detected | {len(result.newly_detected)} |",
            f"| Quarantined | {len(result.quarantined)} |",
            f"| Released | {len(result.released)} |",
        ]

        if result.flaky_tests:
            lines.extend([
                "",
                "## Flaky Tests",
                "",
                "| Test | Score | Classification | Failure Rate | Status |",
                "|------|-------|----------------|--------------|--------|",
            ])

            for flaky in sorted(result.flaky_tests, key=lambda x: -x.flakiness_score):
                lines.append(
                    f"| {flaky.test_name} | "
                    f"{flaky.flakiness_score:.2f} | "
                    f"{flaky.classification.value} | "
                    f"{flaky.failure_rate:.1%} | "
                    f"{flaky.quarantine_status.value} |"
                )

            lines.extend([
                "",
                "## Root Cause Analysis",
                "",
            ])

            for flaky in result.flaky_tests:
                if flaky.root_cause_hints:
                    lines.append(f"### {flaky.test_name}")
                    for hint in flaky.root_cause_hints:
                        lines.append(f"- {hint}")
                    lines.append("")

        return "\n".join(lines)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.flaky.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "flaky_detection",
            "actor": "test-agent",
            "data": data,
        })

    def get_flaky_test(self, test_name: str) -> Optional[FlakyTest]:
        """Get flaky test record by name."""
        return self._flaky_tests.get(test_name)

    def get_quarantined_tests(self) -> List[str]:
        """Get list of quarantined tests."""
        return [
            name for name, flaky in self._flaky_tests.items()
            if flaky.quarantine_status == QuarantineStatus.ACTIVE
        ]

    def manually_quarantine(self, test_name: str, reason: str = "") -> bool:
        """Manually quarantine a test."""
        if test_name not in self._flaky_tests:
            self._flaky_tests[test_name] = FlakyTest(
                test_name=test_name,
                test_path=test_name,
                flakiness_score=1.0,
                classification=FlakyClassification.UNKNOWN,
            )

        flaky = self._flaky_tests[test_name]
        flaky.quarantine_status = QuarantineStatus.PERMANENT
        flaky.root_cause_hints.append(f"Manually quarantined: {reason}")

        self._save_history()
        return True

    def release_from_quarantine(self, test_name: str) -> bool:
        """Release a test from quarantine."""
        if test_name in self._flaky_tests:
            self._flaky_tests[test_name].quarantine_status = QuarantineStatus.RELEASED
            self._save_history()
            return True
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Flaky Test Detector."""
    import argparse

    parser = argparse.ArgumentParser(description="Flaky Test Detector")
    parser.add_argument("tests", nargs="+", help="Tests to analyze")
    parser.add_argument("--retries", type=int, default=3, help="Retry count")
    parser.add_argument("--threshold", type=float, default=0.2, help="Flaky threshold")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/flaky/reports")
    parser.add_argument("--list-quarantined", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = FlakyConfig(
        retry_count=args.retries,
        flaky_threshold=args.threshold,
        output_dir=args.output,
    )

    detector = FlakyDetector(config=config)

    if args.list_quarantined:
        quarantined = detector.get_quarantined_tests()
        if args.json:
            print(json.dumps(quarantined))
        else:
            print("Quarantined Tests:")
            for test in quarantined:
                print(f"  - {test}")
        return

    result = detector.detect_flaky_tests(args.tests)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Flaky Test Detection Complete")
        print(f"{'='*60}")
        print(f"Tests Analyzed: {result.tests_analyzed}")
        print(f"Flaky Tests Found: {result.total_flaky}")
        print(f"Newly Detected: {len(result.newly_detected)}")
        print(f"Quarantined: {len(result.quarantined)}")
        print(f"Released: {len(result.released)}")

        if result.flaky_tests:
            print(f"\nFlaky Tests:")
            for flaky in result.flaky_tests:
                print(f"  - {flaky.test_name} ({flaky.classification.value}, "
                      f"score: {flaky.flakiness_score:.2f})")


if __name__ == "__main__":
    main()
