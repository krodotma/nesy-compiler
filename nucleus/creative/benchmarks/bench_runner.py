"""
Benchmark Runner
================

Core infrastructure for running and collecting benchmark results.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, Optional, TypeVar, Generic

T = TypeVar("T")


@dataclass
class BenchmarkResult:
    """
    Result of a single benchmark.

    Attributes:
        name: Benchmark name/identifier
        iterations: Number of iterations run
        timings_ms: List of timing measurements in milliseconds
        mean_ms: Mean timing
        std_ms: Standard deviation
        min_ms: Minimum timing
        max_ms: Maximum timing
        median_ms: Median timing
        p95_ms: 95th percentile timing
        p99_ms: 99th percentile timing
        success: Whether benchmark completed successfully
        error: Error message if failed
        metadata: Additional metadata
        timestamp: When benchmark was run
    """
    name: str
    iterations: int
    timings_ms: list[float]
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    success: bool = True
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    @classmethod
    def from_timings(
        cls,
        name: str,
        timings_ms: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> "BenchmarkResult":
        """Create result from raw timings."""
        if not timings_ms:
            return cls(
                name=name,
                iterations=0,
                timings_ms=[],
                mean_ms=0.0,
                std_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                median_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                success=False,
                error="No timings collected",
                metadata=metadata or {},
            )

        sorted_timings = sorted(timings_ms)
        n = len(sorted_timings)

        return cls(
            name=name,
            iterations=n,
            timings_ms=timings_ms,
            mean_ms=statistics.mean(timings_ms),
            std_ms=statistics.stdev(timings_ms) if n > 1 else 0.0,
            min_ms=min(timings_ms),
            max_ms=max(timings_ms),
            median_ms=statistics.median(timings_ms),
            p95_ms=sorted_timings[int(n * 0.95)] if n >= 20 else sorted_timings[-1],
            p99_ms=sorted_timings[int(n * 0.99)] if n >= 100 else sorted_timings[-1],
            success=True,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        name: str,
        error: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "BenchmarkResult":
        """Create a failure result."""
        return cls(
            name=name,
            iterations=0,
            timings_ms=[],
            mean_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            median_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            success=False,
            error=error,
            metadata=metadata or {},
        )


class BenchmarkSuite(ABC):
    """
    Abstract base class for benchmark suites.

    Subclasses implement get_benchmarks() to provide a list of
    (name, callable) pairs for benchmarking.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Suite name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Suite description."""
        ...

    @abstractmethod
    def get_benchmarks(self) -> list[tuple[str, Callable[[], Any]]]:
        """
        Get list of benchmarks to run.

        Returns:
            List of (name, callable) tuples
        """
        ...

    def setup(self) -> None:
        """Optional setup before running suite."""
        pass

    def teardown(self) -> None:
        """Optional teardown after running suite."""
        pass

    def before_each(self, name: str) -> None:
        """Optional setup before each benchmark."""
        pass

    def after_each(self, name: str, result: BenchmarkResult) -> None:
        """Optional teardown after each benchmark."""
        pass


class BenchmarkRunner:
    """
    Runs benchmarks and collects results.

    Features:
    - Configurable iterations and warmup
    - Garbage collection between runs
    - Timing with nanosecond precision
    - Statistical analysis of results
    """

    def __init__(
        self,
        iterations: int = 10,
        warmup: int = 2,
        gc_between_runs: bool = True,
        fail_fast: bool = False,
    ):
        """
        Initialize benchmark runner.

        Args:
            iterations: Number of timed iterations
            warmup: Number of warmup iterations (not timed)
            gc_between_runs: Run garbage collection between iterations
            fail_fast: Stop suite on first failure
        """
        self.iterations = iterations
        self.warmup = warmup
        self.gc_between_runs = gc_between_runs
        self.fail_fast = fail_fast

    def run_single(
        self,
        name: str,
        func: Callable[[], Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            name: Benchmark name
            func: Callable to benchmark
            metadata: Optional metadata to include

        Returns:
            BenchmarkResult with timings
        """
        timings: list[float] = []

        try:
            # Warmup runs
            for _ in range(self.warmup):
                if self.gc_between_runs:
                    gc.collect()
                func()

            # Timed runs
            for _ in range(self.iterations):
                if self.gc_between_runs:
                    gc.collect()

                start = time.perf_counter_ns()
                func()
                end = time.perf_counter_ns()

                elapsed_ms = (end - start) / 1_000_000
                timings.append(elapsed_ms)

            return BenchmarkResult.from_timings(name, timings, metadata)

        except Exception as e:
            return BenchmarkResult.failure(
                name,
                f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                metadata,
            )

    def run_suite(self, suite: BenchmarkSuite) -> list[BenchmarkResult]:
        """
        Run all benchmarks in a suite.

        Args:
            suite: BenchmarkSuite instance

        Returns:
            List of BenchmarkResult objects
        """
        results: list[BenchmarkResult] = []

        try:
            suite.setup()

            for name, func in suite.get_benchmarks():
                suite.before_each(name)

                result = self.run_single(
                    f"{suite.name}.{name}",
                    func,
                    metadata={"suite": suite.name},
                )

                suite.after_each(name, result)
                results.append(result)

                if self.fail_fast and not result.success:
                    break

        finally:
            suite.teardown()

        return results

    def run_suites(
        self,
        suites: list[BenchmarkSuite],
    ) -> dict[str, list[BenchmarkResult]]:
        """
        Run multiple benchmark suites.

        Args:
            suites: List of BenchmarkSuite instances

        Returns:
            Dict mapping suite names to results
        """
        return {
            suite.name: self.run_suite(suite)
            for suite in suites
        }

    def save_results(
        self,
        results: dict[str, list[BenchmarkResult]],
        output_dir: str | Path,
    ) -> Path:
        """
        Save results to JSON file.

        Args:
            results: Dict of suite name to results
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"benchmark_results_{timestamp}.json"

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "iterations": self.iterations,
                "warmup": self.warmup,
            },
            "suites": {
                suite: [r.to_dict() for r in res]
                for suite, res in results.items()
            },
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        return filename


def timed(func: Callable[[], T]) -> tuple[T, float]:
    """
    Execute a function and return result with timing.

    Args:
        func: Callable to execute

    Returns:
        Tuple of (result, elapsed_ms)
    """
    start = time.perf_counter_ns()
    result = func()
    end = time.perf_counter_ns()

    return result, (end - start) / 1_000_000


class ParameterizedBenchmark(Generic[T]):
    """
    Helper for running benchmarks with different parameters.

    Example:
        bench = ParameterizedBenchmark("image_size")
        bench.add_case("small", lambda: generate(64, 64))
        bench.add_case("medium", lambda: generate(256, 256))
        bench.add_case("large", lambda: generate(1024, 1024))

        for name, func in bench.cases():
            result = runner.run_single(name, func)
    """

    def __init__(self, param_name: str):
        """Initialize with parameter name."""
        self.param_name = param_name
        self._cases: list[tuple[str, Callable[[], T]]] = []

    def add_case(self, value: str, func: Callable[[], T]) -> None:
        """Add a parameter case."""
        self._cases.append((f"{self.param_name}={value}", func))

    def cases(self) -> list[tuple[str, Callable[[], T]]]:
        """Get all cases."""
        return self._cases.copy()
