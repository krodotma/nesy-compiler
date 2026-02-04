#!/usr/bin/env python3
"""
Step 106: Test Runner Orchestrator

Orchestrates test execution across multiple frameworks and runners.

PBTSO Phase: TEST, DISTRIBUTE
Bus Topics:
- test.run.start (emits)
- test.run.complete (emits)
- test.run.progress (emits)

Dependencies: Steps 102-105 (Generators)
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# Data Types
# ============================================================================

class TestStatus(Enum):
    """Status of a test execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class RunnerType(Enum):
    """Supported test runner types."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    VITEST = "vitest"
    JEST = "jest"
    PLAYWRIGHT = "playwright"
    CYPRESS = "cypress"


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_id: str
    test_name: str
    status: TestStatus
    duration_s: float
    output: str = ""
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "duration_s": self.duration_s,
            "output": self.output,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "artifacts": self.artifacts,
        }


@dataclass
class TestRun:
    """Represents a test run session."""
    run_id: str
    runner_type: RunnerType
    status: TestStatus = TestStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: List[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    coverage_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "runner_type": self.runner_type.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "coverage_percent": self.coverage_percent,
            "metadata": self.metadata,
        }

    @property
    def duration_s(self) -> float:
        """Get total duration of the test run."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def success_rate(self) -> float:
        """Get success rate as a percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100


@dataclass
class TestRunConfig:
    """Configuration for a test run."""
    test_paths: List[str]
    runner_type: RunnerType = RunnerType.PYTEST
    parallel: bool = True
    workers: int = 4
    timeout_s: int = 300
    fail_fast: bool = False
    collect_coverage: bool = True
    verbose: bool = False
    markers: Optional[List[str]] = None  # pytest markers
    pattern: Optional[str] = None  # test file pattern
    exclude_patterns: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None


# ============================================================================
# Bus Interface
# ============================================================================

class RunnerBus:
    """Bus interface for the test runner."""

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
            pass  # Silently ignore bus errors


# ============================================================================
# Test Runner Orchestrator
# ============================================================================

class TestRunnerOrchestrator:
    """
    Orchestrates test execution across multiple runners.

    PBTSO Phase: TEST, DISTRIBUTE
    Bus Topics: test.run.start, test.run.complete, test.run.progress
    """

    BUS_TOPICS = {
        "start": "test.run.start",
        "complete": "test.run.complete",
        "progress": "test.run.progress",
        "result": "test.run.result",
    }

    def __init__(self, bus=None):
        """
        Initialize the test runner orchestrator.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or RunnerBus()
        self._runners: Dict[RunnerType, Any] = {}
        self._active_runs: Dict[str, TestRun] = {}
        self._executor: Optional[ThreadPoolExecutor] = None

    def register_runner(self, runner_type: RunnerType, runner: Any) -> None:
        """Register a test runner."""
        self._runners[runner_type] = runner

    def run_tests(self, config: TestRunConfig) -> TestRun:
        """
        Execute tests based on configuration.

        Args:
            config: Test run configuration

        Returns:
            TestRun with results
        """
        run = TestRun(
            run_id=str(uuid.uuid4()),
            runner_type=config.runner_type,
            metadata={
                "test_paths": config.test_paths,
                "parallel": config.parallel,
                "workers": config.workers,
            },
        )

        self._active_runs[run.run_id] = run

        # Emit start event
        self._emit_event("start", {
            "run_id": run.run_id,
            "runner_type": config.runner_type.value,
            "test_paths": config.test_paths,
        })

        run.status = TestStatus.RUNNING
        run.started_at = time.time()

        try:
            # Get appropriate runner
            runner = self._get_runner(config.runner_type)

            if config.parallel and config.workers > 1:
                results = self._run_parallel(runner, config)
            else:
                results = self._run_sequential(runner, config)

            run.results = results
            run.total_tests = len(results)
            run.passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            run.failed = sum(1 for r in results if r.status == TestStatus.FAILED)
            run.skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
            run.errors = sum(1 for r in results if r.status == TestStatus.ERROR)

            # Determine overall status
            if run.errors > 0:
                run.status = TestStatus.ERROR
            elif run.failed > 0:
                run.status = TestStatus.FAILED
            else:
                run.status = TestStatus.PASSED

            # Collect coverage if enabled
            if config.collect_coverage:
                run.coverage_percent = self._collect_coverage(config)

        except Exception as e:
            run.status = TestStatus.ERROR
            run.metadata["error"] = str(e)

        run.completed_at = time.time()

        # Emit complete event
        self._emit_event("complete", {
            "run_id": run.run_id,
            "status": run.status.value,
            "total_tests": run.total_tests,
            "passed": run.passed,
            "failed": run.failed,
            "duration_s": run.duration_s,
            "coverage_percent": run.coverage_percent,
        })

        return run

    def _get_runner(self, runner_type: RunnerType):
        """Get a runner instance, creating if necessary."""
        if runner_type in self._runners:
            return self._runners[runner_type]

        # Lazy import runners
        if runner_type == RunnerType.PYTEST:
            from .pytest_runner import PytestRunner
            runner = PytestRunner(self.bus)
        elif runner_type == RunnerType.VITEST:
            from .vitest_runner import VitestRunner
            runner = VitestRunner(self.bus)
        else:
            raise ValueError(f"Unsupported runner type: {runner_type}")

        self._runners[runner_type] = runner
        return runner

    def _run_sequential(self, runner, config: TestRunConfig) -> List[TestResult]:
        """Run tests sequentially."""
        results = []

        for test_path in config.test_paths:
            result = runner.run(
                test_path=test_path,
                timeout_s=config.timeout_s,
                verbose=config.verbose,
                markers=config.markers,
                env=config.env,
                working_dir=config.working_dir,
            )
            results.extend(result)

            # Emit progress
            self._emit_event("progress", {
                "run_id": self._get_current_run_id(),
                "completed": len(results),
                "last_test": result[-1].test_name if result else None,
            })

            # Fail fast
            if config.fail_fast and any(r.status == TestStatus.FAILED for r in result):
                break

        return results

    def _run_parallel(self, runner, config: TestRunConfig) -> List[TestResult]:
        """Run tests in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = {}

            for test_path in config.test_paths:
                future = executor.submit(
                    runner.run,
                    test_path=test_path,
                    timeout_s=config.timeout_s,
                    verbose=config.verbose,
                    markers=config.markers,
                    env=config.env,
                    working_dir=config.working_dir,
                )
                futures[future] = test_path

            for future in as_completed(futures):
                test_path = futures[future]
                try:
                    result = future.result()
                    results.extend(result)

                    # Emit progress
                    self._emit_event("progress", {
                        "run_id": self._get_current_run_id(),
                        "completed": len(results),
                        "last_path": test_path,
                    })

                except Exception as e:
                    # Record error as test result
                    results.append(TestResult(
                        test_id=str(uuid.uuid4()),
                        test_name=f"Error: {test_path}",
                        status=TestStatus.ERROR,
                        duration_s=0,
                        error_message=str(e),
                        file_path=test_path,
                    ))

        return results

    def _collect_coverage(self, config: TestRunConfig) -> Optional[float]:
        """Collect coverage data after test run."""
        # Coverage collection is framework-specific
        # For pytest, we'd parse .coverage or coverage.xml
        # For vitest, we'd parse coverage/coverage-summary.json

        coverage_file = Path(config.working_dir or ".") / ".coverage"
        if coverage_file.exists():
            # Would normally parse the coverage database
            # For now, return a placeholder
            return None

        return None

    def _get_current_run_id(self) -> Optional[str]:
        """Get the ID of the currently active run."""
        for run_id, run in self._active_runs.items():
            if run.status == TestStatus.RUNNING:
                return run_id
        return None

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.run.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "test_execution",
            "actor": "test-agent",
            "data": data,
        })

    def get_run(self, run_id: str) -> Optional[TestRun]:
        """Get a test run by ID."""
        return self._active_runs.get(run_id)

    def list_runs(self) -> List[TestRun]:
        """List all test runs."""
        return list(self._active_runs.values())

    def cancel_run(self, run_id: str) -> bool:
        """Cancel an active test run."""
        run = self._active_runs.get(run_id)
        if run and run.status == TestStatus.RUNNING:
            run.status = TestStatus.ERROR
            run.metadata["cancelled"] = True
            run.completed_at = time.time()
            return True
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Runner Orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Runner Orchestrator")
    parser.add_argument("paths", nargs="*", default=["."], help="Test paths")
    parser.add_argument("--runner", choices=["pytest", "vitest"], default="pytest")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Collect coverage")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")

    args = parser.parse_args()

    config = TestRunConfig(
        test_paths=args.paths,
        runner_type=RunnerType(args.runner),
        parallel=args.parallel,
        workers=args.workers,
        timeout_s=args.timeout,
        verbose=args.verbose,
        collect_coverage=args.coverage,
        fail_fast=args.fail_fast,
    )

    orchestrator = TestRunnerOrchestrator()
    run = orchestrator.run_tests(config)

    print(f"\n{'='*60}")
    print(f"Test Run: {run.run_id}")
    print(f"Status: {run.status.value}")
    print(f"Duration: {run.duration_s:.2f}s")
    print(f"Total: {run.total_tests} | Passed: {run.passed} | Failed: {run.failed} | Skipped: {run.skipped}")
    if run.coverage_percent is not None:
        print(f"Coverage: {run.coverage_percent:.1f}%")
    print(f"{'='*60}")

    # Exit with appropriate code
    if run.status in (TestStatus.FAILED, TestStatus.ERROR):
        exit(1)


if __name__ == "__main__":
    main()
