#!/usr/bin/env python3
"""
Step 127: Test Parallelizer

Provides parallel test execution with intelligent partitioning.

PBTSO Phase: TEST, DISTRIBUTE
Bus Topics:
- test.parallel.start (emits)
- test.parallel.progress (emits)
- test.parallel.complete (emits)

Dependencies: Steps 101-126 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import multiprocessing


# ============================================================================
# Constants
# ============================================================================

class PartitionStrategy(Enum):
    """Strategies for partitioning tests across workers."""
    ROUND_ROBIN = "round_robin"
    BY_FILE = "by_file"
    BY_DURATION = "by_duration"
    BY_DIRECTORY = "by_directory"
    BALANCED = "balanced"


class WorkerStatus(Enum):
    """Status of a worker."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TestItem:
    """A test item to be executed."""
    test_id: str
    test_name: str
    file_path: str
    estimated_duration_ms: float = 1000
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "file_path": self.file_path,
            "estimated_duration_ms": self.estimated_duration_ms,
            "priority": self.priority,
            "tags": self.tags,
        }


@dataclass
class TestPartition:
    """A partition of tests for a worker."""
    partition_id: int
    tests: List[TestItem] = field(default_factory=list)
    estimated_duration_ms: float = 0

    def add_test(self, test: TestItem) -> None:
        """Add a test to this partition."""
        self.tests.append(test)
        self.estimated_duration_ms += test.estimated_duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "partition_id": self.partition_id,
            "test_count": len(self.tests),
            "estimated_duration_ms": self.estimated_duration_ms,
            "tests": [t.test_name for t in self.tests],
        }


@dataclass
class WorkerResult:
    """Result from a worker execution."""
    worker_id: int
    partition_id: int
    status: WorkerStatus
    started_at: float
    completed_at: Optional[float] = None
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        """Get execution duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "partition_id": self.partition_id,
            "status": self.status.value,
            "duration_s": self.duration_s,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "error": self.error,
        }


@dataclass
class ParallelConfig:
    """
    Configuration for parallel test execution.

    Attributes:
        workers: Number of parallel workers
        strategy: Partitioning strategy
        timeout_s: Worker timeout in seconds
        fail_fast: Stop all workers on first failure
        use_processes: Use processes instead of threads
        output_dir: Output directory for results
        collect_coverage: Collect coverage data
        duration_estimates: Test duration estimates file
    """
    workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))
    strategy: PartitionStrategy = PartitionStrategy.BALANCED
    timeout_s: int = 3600
    fail_fast: bool = False
    use_processes: bool = True
    output_dir: str = ".pluribus/test-agent/parallel"
    collect_coverage: bool = True
    duration_estimates: Optional[str] = None
    test_command: str = "python -m pytest {tests} --tb=short -q"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workers": self.workers,
            "strategy": self.strategy.value,
            "timeout_s": self.timeout_s,
            "fail_fast": self.fail_fast,
            "use_processes": self.use_processes,
            "collect_coverage": self.collect_coverage,
        }


@dataclass
class ParallelResult:
    """Result of parallel test execution."""
    run_id: str
    started_at: float
    completed_at: Optional[float] = None
    total_workers: int = 0
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    worker_results: List[WorkerResult] = field(default_factory=list)
    partitions: List[TestPartition] = field(default_factory=list)
    coverage_percent: Optional[float] = None

    @property
    def duration_s(self) -> float:
        """Get total duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100

    @property
    def speedup(self) -> float:
        """Calculate speedup vs sequential execution."""
        if self.total_workers == 0:
            return 1.0

        # Sum of all worker durations (what sequential would take)
        sequential_time = sum(w.duration_s for w in self.worker_results)

        if sequential_time == 0:
            return 1.0

        return sequential_time / self.duration_s

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "duration_s": self.duration_s,
            "total_workers": self.total_workers,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.success_rate,
            "speedup": self.speedup,
            "coverage_percent": self.coverage_percent,
            "worker_results": [w.to_dict() for w in self.worker_results],
            "partitions": [p.to_dict() for p in self.partitions],
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class ParallelBus:
    """Bus interface for parallel execution with file locking."""

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
# Test Parallelizer
# ============================================================================

class TestParallelizer:
    """
    Executes tests in parallel across multiple workers.

    Features:
    - Multiple partitioning strategies
    - Process or thread-based workers
    - Duration-based load balancing
    - Progress tracking
    - Fail-fast support

    PBTSO Phase: TEST, DISTRIBUTE
    Bus Topics: test.parallel.start, test.parallel.progress, test.parallel.complete
    """

    BUS_TOPICS = {
        "start": "test.parallel.start",
        "progress": "test.parallel.progress",
        "complete": "test.parallel.complete",
        "worker_complete": "test.parallel.worker.complete",
    }

    def __init__(self, bus=None, config: Optional[ParallelConfig] = None):
        """
        Initialize the test parallelizer.

        Args:
            bus: Optional bus instance
            config: Parallelizer configuration
        """
        self.bus = bus or ParallelBus()
        self.config = config or ParallelConfig()
        self._duration_estimates: Dict[str, float] = {}
        self._abort_flag = False

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load duration estimates
        if self.config.duration_estimates:
            self._load_duration_estimates()

    def _load_duration_estimates(self) -> None:
        """Load test duration estimates from file."""
        estimates_file = Path(self.config.duration_estimates)
        if estimates_file.exists():
            try:
                with open(estimates_file) as f:
                    self._duration_estimates = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

    def run(self, tests: List[str]) -> ParallelResult:
        """
        Run tests in parallel.

        Args:
            tests: List of test paths/patterns

        Returns:
            ParallelResult with execution results
        """
        run_id = str(uuid.uuid4())
        self._abort_flag = False

        # Discover and convert tests
        test_items = self._discover_tests(tests)

        result = ParallelResult(
            run_id=run_id,
            started_at=time.time(),
            total_workers=self.config.workers,
            total_tests=len(test_items),
        )

        if not test_items:
            result.completed_at = time.time()
            return result

        # Partition tests
        partitions = self._partition_tests(test_items)
        result.partitions = partitions

        # Emit start event
        self._emit_event("start", {
            "run_id": run_id,
            "total_tests": len(test_items),
            "workers": self.config.workers,
            "partitions": [p.to_dict() for p in partitions],
        })

        # Execute partitions in parallel
        executor_class = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor

        with executor_class(max_workers=self.config.workers) as executor:
            futures = {}

            for i, partition in enumerate(partitions):
                if partition.tests:
                    future = executor.submit(
                        self._execute_partition,
                        i,
                        partition,
                    )
                    futures[future] = partition

            # Collect results
            for future in as_completed(futures, timeout=self.config.timeout_s):
                if self._abort_flag:
                    break

                partition = futures[future]

                try:
                    worker_result = future.result()
                    result.worker_results.append(worker_result)

                    # Update totals
                    result.passed += worker_result.passed
                    result.failed += worker_result.failed
                    result.skipped += worker_result.skipped

                    # Emit progress
                    self._emit_event("progress", {
                        "run_id": run_id,
                        "completed_workers": len(result.worker_results),
                        "total_workers": len(partitions),
                        "passed": result.passed,
                        "failed": result.failed,
                    })

                    # Fail fast
                    if self.config.fail_fast and worker_result.failed > 0:
                        self._abort_flag = True
                        break

                except Exception as e:
                    result.worker_results.append(WorkerResult(
                        worker_id=partition.partition_id,
                        partition_id=partition.partition_id,
                        status=WorkerStatus.FAILED,
                        started_at=time.time(),
                        completed_at=time.time(),
                        error=str(e),
                    ))

        result.completed_at = time.time()

        # Emit completion
        self._emit_event("complete", {
            "run_id": run_id,
            "duration_s": result.duration_s,
            "total_tests": result.total_tests,
            "passed": result.passed,
            "failed": result.failed,
            "speedup": result.speedup,
        })

        # Save results
        self._save_results(result)

        return result

    def _discover_tests(self, test_paths: List[str]) -> List[TestItem]:
        """Discover tests from paths."""
        items = []

        for path in test_paths:
            p = Path(path)

            if p.is_file():
                items.append(TestItem(
                    test_id=str(uuid.uuid4()),
                    test_name=str(p),
                    file_path=str(p),
                    estimated_duration_ms=self._duration_estimates.get(str(p), 1000),
                ))
            elif p.is_dir():
                # Discover test files
                for test_file in p.rglob("test_*.py"):
                    items.append(TestItem(
                        test_id=str(uuid.uuid4()),
                        test_name=str(test_file),
                        file_path=str(test_file),
                        estimated_duration_ms=self._duration_estimates.get(str(test_file), 1000),
                    ))
                for test_file in p.rglob("*_test.py"):
                    items.append(TestItem(
                        test_id=str(uuid.uuid4()),
                        test_name=str(test_file),
                        file_path=str(test_file),
                        estimated_duration_ms=self._duration_estimates.get(str(test_file), 1000),
                    ))
            else:
                # Treat as pattern
                items.append(TestItem(
                    test_id=str(uuid.uuid4()),
                    test_name=path,
                    file_path=path,
                    estimated_duration_ms=self._duration_estimates.get(path, 1000),
                ))

        return items

    def _partition_tests(self, tests: List[TestItem]) -> List[TestPartition]:
        """Partition tests according to strategy."""
        num_workers = min(self.config.workers, len(tests))

        if num_workers <= 0:
            return []

        partitions = [TestPartition(partition_id=i) for i in range(num_workers)]

        if self.config.strategy == PartitionStrategy.ROUND_ROBIN:
            for i, test in enumerate(tests):
                partitions[i % num_workers].add_test(test)

        elif self.config.strategy == PartitionStrategy.BY_FILE:
            # Group by file directory
            files_by_dir: Dict[str, List[TestItem]] = {}
            for test in tests:
                dir_name = str(Path(test.file_path).parent)
                if dir_name not in files_by_dir:
                    files_by_dir[dir_name] = []
                files_by_dir[dir_name].append(test)

            # Distribute directories round-robin
            for i, (_, dir_tests) in enumerate(files_by_dir.items()):
                for test in dir_tests:
                    partitions[i % num_workers].add_test(test)

        elif self.config.strategy == PartitionStrategy.BY_DURATION:
            # Sort by duration descending
            sorted_tests = sorted(tests, key=lambda t: t.estimated_duration_ms, reverse=True)

            # Greedy assignment to minimize max duration
            for test in sorted_tests:
                # Find partition with lowest total duration
                min_partition = min(partitions, key=lambda p: p.estimated_duration_ms)
                min_partition.add_test(test)

        elif self.config.strategy == PartitionStrategy.BALANCED:
            # Similar to BY_DURATION but also considers test count
            sorted_tests = sorted(tests, key=lambda t: t.estimated_duration_ms, reverse=True)

            for test in sorted_tests:
                # Find partition with lowest total duration
                min_partition = min(partitions, key=lambda p: p.estimated_duration_ms)
                min_partition.add_test(test)

        else:  # Default: round robin
            for i, test in enumerate(tests):
                partitions[i % num_workers].add_test(test)

        return partitions

    def _execute_partition(
        self,
        worker_id: int,
        partition: TestPartition,
    ) -> WorkerResult:
        """Execute a test partition."""
        result = WorkerResult(
            worker_id=worker_id,
            partition_id=partition.partition_id,
            status=WorkerStatus.RUNNING,
            started_at=time.time(),
            total_tests=len(partition.tests),
        )

        try:
            # Build test paths
            test_paths = [t.file_path for t in partition.tests]
            tests_arg = " ".join(test_paths)

            # Execute tests
            command = self.config.test_command.format(tests=tests_arg)

            proc_result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )

            # Parse results (simple parsing, real implementation would parse JSON output)
            output = proc_result.stdout + proc_result.stderr

            # Count results from output
            result.passed = output.count(" passed") + output.count(" PASSED")
            result.failed = output.count(" failed") + output.count(" FAILED")
            result.skipped = output.count(" skipped") + output.count(" SKIPPED")

            if proc_result.returncode == 0:
                result.status = WorkerStatus.COMPLETED
            else:
                result.status = WorkerStatus.FAILED
                if not result.failed:
                    result.error = output[-1000:] if len(output) > 1000 else output

        except subprocess.TimeoutExpired:
            result.status = WorkerStatus.FAILED
            result.error = "Timeout"
        except Exception as e:
            result.status = WorkerStatus.FAILED
            result.error = str(e)

        result.completed_at = time.time()

        return result

    async def run_async(self, tests: List[str]) -> ParallelResult:
        """Async version of parallel execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, tests)

    def abort(self) -> None:
        """Abort the current execution."""
        self._abort_flag = True

    def _save_results(self, result: ParallelResult) -> None:
        """Save execution results."""
        output_path = Path(self.config.output_dir)
        output_file = output_path / f"parallel_{result.run_id}.json"

        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.parallel.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "parallel",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Parallelizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Parallelizer")
    parser.add_argument("tests", nargs="*", default=["tests/"], help="Test paths")
    parser.add_argument("--workers", "-w", type=int, help="Number of workers")
    parser.add_argument("--strategy", choices=["round_robin", "by_file", "by_duration", "balanced"],
                       default="balanced")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--threads", action="store_true", help="Use threads instead of processes")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/parallel")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    config = ParallelConfig(
        workers=args.workers or max(1, multiprocessing.cpu_count() - 1),
        strategy=PartitionStrategy(args.strategy),
        timeout_s=args.timeout,
        fail_fast=args.fail_fast,
        use_processes=not args.threads,
        output_dir=args.output,
        verbose=args.verbose,
    )

    parallelizer = TestParallelizer(config=config)
    result = parallelizer.run(args.tests)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        status = "[PASS]" if result.failed == 0 else "[FAIL]"

        print(f"\n{'='*60}")
        print(f"Parallel Test Execution {status}")
        print(f"{'='*60}")
        print(f"Run ID: {result.run_id}")
        print(f"Duration: {result.duration_s:.2f}s")
        print(f"Workers: {result.total_workers}")
        print(f"Speedup: {result.speedup:.2f}x")
        print()
        print(f"Tests: {result.total_tests}")
        print(f"Passed: {result.passed}")
        print(f"Failed: {result.failed}")
        print(f"Skipped: {result.skipped}")
        print(f"Success Rate: {result.success_rate:.1f}%")

        if args.verbose:
            print("\nWorker Results:")
            for worker in result.worker_results:
                status = "[OK]" if worker.status == WorkerStatus.COMPLETED else "[FAIL]"
                print(f"  Worker {worker.worker_id}: {status} "
                      f"({worker.passed}/{worker.total_tests}) "
                      f"{worker.duration_s:.2f}s")

        print(f"\nOutput: {config.output_dir}/")
        print(f"{'='*60}\n")

        if result.failed > 0:
            exit(1)


if __name__ == "__main__":
    main()
