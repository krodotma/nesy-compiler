#!/usr/bin/env python3
"""
Step 117: Test Prioritizer

Provides smart test ordering based on various factors.

PBTSO Phase: PLAN, TEST
Bus Topics:
- test.priority.calculate (subscribes)
- test.priority.result (emits)

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

# Default weights for priority factors
DEFAULT_WEIGHTS = {
    "failure_history": 0.30,
    "code_change": 0.25,
    "execution_time": 0.15,
    "dependency": 0.15,
    "flakiness": 0.10,
    "age": 0.05,
}


class PriorityFactor(Enum):
    """Factors that influence test priority."""
    FAILURE_HISTORY = "failure_history"  # Recent failures
    CODE_CHANGE = "code_change"  # Changed code coverage
    EXECUTION_TIME = "execution_time"  # Fast tests first
    DEPENDENCY = "dependency"  # Dependent on changed modules
    FLAKINESS = "flakiness"  # Flaky tests lower priority
    AGE = "age"  # Time since last run
    CRITICALITY = "criticality"  # Test importance


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TestPriority:
    """Priority information for a test."""
    test_name: str
    priority_score: float
    rank: int = 0
    factors: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    estimated_duration_ms: float = 0
    last_run_at: Optional[float] = None
    last_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "priority_score": self.priority_score,
            "rank": self.rank,
            "factors": self.factors,
            "tags": self.tags,
            "estimated_duration_ms": self.estimated_duration_ms,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
        }


@dataclass
class PriorityConfig:
    """
    Configuration for test prioritization.

    Attributes:
        weights: Weights for each priority factor
        changed_files: List of changed files for impact analysis
        time_budget_s: Optional time budget for test selection
        max_tests: Maximum number of tests to select
        include_patterns: Patterns for tests to always include
        exclude_patterns: Patterns for tests to exclude
        history_dir: Directory for test history
    """
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    changed_files: List[str] = field(default_factory=list)
    time_budget_s: Optional[float] = None
    max_tests: Optional[int] = None
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    history_dir: str = ".pluribus/test-agent/history"
    favor_fast_tests: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "changed_files": self.changed_files,
            "time_budget_s": self.time_budget_s,
            "max_tests": self.max_tests,
            "favor_fast_tests": self.favor_fast_tests,
        }


@dataclass
class PriorityResult:
    """Result of test prioritization."""
    run_id: str
    calculated_at: float
    prioritized_tests: List[TestPriority] = field(default_factory=list)
    total_tests: int = 0
    selected_tests: int = 0
    estimated_duration_s: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "calculated_at": self.calculated_at,
            "total_tests": self.total_tests,
            "selected_tests": self.selected_tests,
            "estimated_duration_s": self.estimated_duration_s,
            "prioritized_tests": [t.to_dict() for t in self.prioritized_tests],
            "metadata": self.metadata,
        }

    def get_test_order(self) -> List[str]:
        """Get ordered list of test names."""
        return [t.test_name for t in self.prioritized_tests]


# ============================================================================
# Bus Interface
# ============================================================================

class PriorityBus:
    """Bus interface for test prioritization."""

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
# Test Prioritizer
# ============================================================================

class TestPrioritizer:
    """
    Prioritizes tests based on multiple factors.

    Prioritization factors:
    - Failure history: Recently failed tests get higher priority
    - Code change impact: Tests covering changed code
    - Execution time: Faster tests can run first
    - Dependencies: Tests for modified modules
    - Flakiness: Flaky tests get lower priority
    - Age: Tests not run recently get higher priority

    PBTSO Phase: PLAN, TEST
    Bus Topics: test.priority.calculate, test.priority.result
    """

    BUS_TOPICS = {
        "calculate": "test.priority.calculate",
        "result": "test.priority.result",
    }

    def __init__(self, bus=None):
        """
        Initialize the test prioritizer.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or PriorityBus()
        self._test_history: Dict[str, Dict[str, Any]] = {}
        self._test_coverage: Dict[str, Set[str]] = {}

    def prioritize(
        self,
        tests: List[str],
        config: Optional[PriorityConfig] = None,
    ) -> PriorityResult:
        """
        Prioritize a list of tests.

        Args:
            tests: List of test names/paths
            config: Prioritization configuration

        Returns:
            PriorityResult with ordered tests
        """
        config = config or PriorityConfig()
        run_id = str(uuid.uuid4())

        # Load test history
        self._load_history(config.history_dir)

        result = PriorityResult(
            run_id=run_id,
            calculated_at=time.time(),
            total_tests=len(tests),
        )

        # Calculate priority for each test
        priorities = []
        for test_name in tests:
            # Check exclusion patterns
            if self._matches_patterns(test_name, config.exclude_patterns):
                continue

            priority = self._calculate_priority(test_name, config)
            priorities.append(priority)

        # Sort by priority score (descending)
        priorities.sort(key=lambda p: p.priority_score, reverse=True)

        # Assign ranks
        for i, priority in enumerate(priorities):
            priority.rank = i + 1

        # Apply time budget or max tests constraint
        selected = self._apply_constraints(priorities, config)

        result.prioritized_tests = selected
        result.selected_tests = len(selected)
        result.estimated_duration_s = sum(
            t.estimated_duration_ms / 1000 for t in selected
        )

        # Emit result event
        self._emit_event("result", {
            "run_id": run_id,
            "total_tests": result.total_tests,
            "selected_tests": result.selected_tests,
            "estimated_duration_s": result.estimated_duration_s,
        })

        return result

    def _calculate_priority(
        self,
        test_name: str,
        config: PriorityConfig,
    ) -> TestPriority:
        """Calculate priority for a single test."""
        factors = {}
        history = self._test_history.get(test_name, {})

        # Factor 1: Failure history
        failure_score = self._score_failure_history(history)
        factors[PriorityFactor.FAILURE_HISTORY.value] = failure_score

        # Factor 2: Code change impact
        change_score = self._score_code_change(test_name, config.changed_files)
        factors[PriorityFactor.CODE_CHANGE.value] = change_score

        # Factor 3: Execution time (faster = higher score)
        time_score = self._score_execution_time(history, config.favor_fast_tests)
        factors[PriorityFactor.EXECUTION_TIME.value] = time_score

        # Factor 4: Dependency impact
        dep_score = self._score_dependency(test_name, config.changed_files)
        factors[PriorityFactor.DEPENDENCY.value] = dep_score

        # Factor 5: Flakiness (more flaky = lower score)
        flaky_score = self._score_flakiness(history)
        factors[PriorityFactor.FLAKINESS.value] = flaky_score

        # Factor 6: Age (longer since last run = higher score)
        age_score = self._score_age(history)
        factors[PriorityFactor.AGE.value] = age_score

        # Calculate weighted sum
        total_score = sum(
            factors.get(factor, 0) * config.weights.get(factor, 0)
            for factor in config.weights
        )

        # Boost for include patterns
        if self._matches_patterns(test_name, config.include_patterns):
            total_score += 1.0  # Significant boost

        return TestPriority(
            test_name=test_name,
            priority_score=total_score,
            factors=factors,
            estimated_duration_ms=history.get("avg_duration_ms", 1000),
            last_run_at=history.get("last_run_at"),
            last_status=history.get("last_status"),
        )

    def _score_failure_history(self, history: Dict[str, Any]) -> float:
        """Score based on recent failures."""
        if not history:
            return 0.5  # Neutral for new tests

        recent_results = history.get("recent_results", [])
        if not recent_results:
            return 0.5

        # Count recent failures (more weight to more recent)
        score = 0
        for i, result in enumerate(recent_results[-5:]):
            weight = (i + 1) / 5  # More recent = higher weight
            if result.get("status") == "failed":
                score += weight

        return min(score, 1.0)

    def _score_code_change(
        self,
        test_name: str,
        changed_files: List[str],
    ) -> float:
        """Score based on coverage of changed files."""
        if not changed_files:
            return 0.0

        # Get test's coverage
        covered_files = self._test_coverage.get(test_name, set())
        if not covered_files:
            # No coverage data - assume based on naming convention
            test_path = Path(test_name)
            module_name = test_path.stem.replace("test_", "").replace("_test", "")

            for changed in changed_files:
                if module_name in changed:
                    return 1.0
            return 0.2  # Small score for unknown coverage

        # Calculate overlap
        changed_set = set(changed_files)
        overlap = len(covered_files & changed_set)

        return min(overlap / max(len(changed_files), 1), 1.0)

    def _score_execution_time(
        self,
        history: Dict[str, Any],
        favor_fast: bool,
    ) -> float:
        """Score based on execution time."""
        avg_time = history.get("avg_duration_ms", 1000)

        if not favor_fast:
            return 0.5  # Neutral

        # Normalize: faster tests get higher scores
        # Assume typical test range is 100ms to 10000ms
        if avg_time <= 100:
            return 1.0
        elif avg_time >= 10000:
            return 0.0
        else:
            # Linear interpolation
            return 1.0 - (avg_time - 100) / 9900

    def _score_dependency(
        self,
        test_name: str,
        changed_files: List[str],
    ) -> float:
        """Score based on dependency on changed modules."""
        # Simple heuristic: check if test file imports changed modules
        test_path = Path(test_name)

        for changed in changed_files:
            # Check if module name appears in test name
            changed_module = Path(changed).stem
            if changed_module in test_name:
                return 1.0

        return 0.0

    def _score_flakiness(self, history: Dict[str, Any]) -> float:
        """Score based on flakiness (inverted - flaky = lower score)."""
        flake_rate = history.get("flake_rate", 0)

        # Invert: 0 flakiness = 1.0 score, 100% flakiness = 0.0 score
        return 1.0 - flake_rate

    def _score_age(self, history: Dict[str, Any]) -> float:
        """Score based on time since last run."""
        last_run = history.get("last_run_at")

        if not last_run:
            return 1.0  # Never run = highest priority

        # Calculate age in hours
        age_hours = (time.time() - last_run) / 3600

        # Normalize: >24 hours = 1.0, <1 hour = 0.0
        if age_hours >= 24:
            return 1.0
        elif age_hours <= 1:
            return 0.0
        else:
            return age_hours / 24

    def _apply_constraints(
        self,
        priorities: List[TestPriority],
        config: PriorityConfig,
    ) -> List[TestPriority]:
        """Apply time budget or max tests constraints."""
        selected = []
        total_time_s = 0

        for priority in priorities:
            # Check max tests
            if config.max_tests and len(selected) >= config.max_tests:
                break

            # Check time budget
            test_time_s = priority.estimated_duration_ms / 1000
            if config.time_budget_s:
                if total_time_s + test_time_s > config.time_budget_s:
                    # Check if we should skip or include smaller tests
                    continue

            selected.append(priority)
            total_time_s += test_time_s

        return selected

    def _matches_patterns(self, test_name: str, patterns: List[str]) -> bool:
        """Check if test name matches any pattern."""
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(test_name, pattern):
                return True
        return False

    def _load_history(self, history_dir: str) -> None:
        """Load test history from storage."""
        history_path = Path(history_dir) / "test_history.json"

        if history_path.exists():
            try:
                with open(history_path) as f:
                    data = json.load(f)
                    for test_name, hist_data in data.items():
                        self._test_history[test_name] = {
                            "recent_results": hist_data.get("results", [])[-10:],
                            "avg_duration_ms": self._calculate_avg_duration(hist_data),
                            "last_run_at": self._get_last_run(hist_data),
                            "last_status": self._get_last_status(hist_data),
                            "flake_rate": self._calculate_flake_rate(hist_data),
                        }
            except (json.JSONDecodeError, IOError):
                pass

    def _calculate_avg_duration(self, hist_data: Dict) -> float:
        """Calculate average duration from history."""
        results = hist_data.get("results", [])
        durations = [r.get("duration_ms", 0) for r in results if r.get("duration_ms")]
        return sum(durations) / len(durations) if durations else 1000

    def _get_last_run(self, hist_data: Dict) -> Optional[float]:
        """Get timestamp of last run."""
        results = hist_data.get("results", [])
        if results:
            return results[-1].get("timestamp")
        return None

    def _get_last_status(self, hist_data: Dict) -> Optional[str]:
        """Get status of last run."""
        results = hist_data.get("results", [])
        if results:
            return results[-1].get("status")
        return None

    def _calculate_flake_rate(self, hist_data: Dict) -> float:
        """Calculate flakiness rate."""
        results = hist_data.get("results", [])
        if len(results) < 2:
            return 0.0

        flakes = sum(
            1 for i in range(1, len(results))
            if results[i].get("status") != results[i-1].get("status")
        )
        return flakes / (len(results) - 1)

    def load_coverage_map(self, coverage_file: str) -> None:
        """Load test-to-file coverage mapping."""
        try:
            with open(coverage_file) as f:
                data = json.load(f)
                for test_name, covered_files in data.items():
                    self._test_coverage[test_name] = set(covered_files)
        except (json.JSONDecodeError, IOError):
            pass

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.priority.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "test_prioritization",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Prioritizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Prioritizer")
    parser.add_argument("tests_file", help="Path to tests list (JSON or one per line)")
    parser.add_argument("--changed", nargs="*", help="Changed files")
    parser.add_argument("--time-budget", type=float, help="Time budget in seconds")
    parser.add_argument("--max-tests", type=int, help="Maximum tests to select")
    parser.add_argument("--history-dir", default=".pluribus/test-agent/history")
    parser.add_argument("--coverage-map", help="Test coverage mapping file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Load tests list
    try:
        with open(args.tests_file) as f:
            content = f.read()
            try:
                tests = json.loads(content)
            except json.JSONDecodeError:
                tests = [line.strip() for line in content.split('\n') if line.strip()]
    except IOError as e:
        print(f"Error loading tests: {e}")
        exit(1)

    config = PriorityConfig(
        changed_files=args.changed or [],
        time_budget_s=args.time_budget,
        max_tests=args.max_tests,
        history_dir=args.history_dir,
    )

    prioritizer = TestPrioritizer()

    if args.coverage_map:
        prioritizer.load_coverage_map(args.coverage_map)

    result = prioritizer.prioritize(tests, config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Test Prioritization Complete")
        print(f"{'='*60}")
        print(f"Total Tests: {result.total_tests}")
        print(f"Selected Tests: {result.selected_tests}")
        print(f"Estimated Duration: {result.estimated_duration_s:.2f}s")
        print(f"\nTop 10 Priority Tests:")

        for test in result.prioritized_tests[:10]:
            status = f"[{test.last_status}]" if test.last_status else ""
            print(f"  {test.rank}. {test.test_name} (score: {test.priority_score:.3f}) {status}")


if __name__ == "__main__":
    main()
