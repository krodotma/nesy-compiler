#!/usr/bin/env python3
"""
Step 113: Benchmark Suite

Provides performance benchmarking capabilities for measuring code performance.

PBTSO Phase: TEST, VERIFY
Bus Topics:
- test.benchmark.run (subscribes)
- test.benchmark.result (emits)
- telemetry.benchmark (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic


# ============================================================================
# Constants
# ============================================================================

DEFAULT_ITERATIONS = 100
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_MIN_TIME_S = 0.1
DEFAULT_MAX_TIME_S = 60.0


class BenchmarkMode(Enum):
    """Benchmarking modes."""
    ITERATIONS = "iterations"  # Fixed number of iterations
    TIME = "time"  # Run for fixed time
    AUTO = "auto"  # Automatically determine iterations


class ComparisonResult(Enum):
    """Result of benchmark comparison."""
    FASTER = "faster"
    SLOWER = "slower"
    SAME = "same"
    INCONCLUSIVE = "inconclusive"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class BenchmarkStats:
    """Statistical analysis of benchmark results."""
    mean: float
    median: float
    std_dev: float
    min_time: float
    max_time: float
    percentile_95: float
    percentile_99: float
    iterations: int
    total_time: float
    ops_per_second: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "min": self.min_time,
            "max": self.max_time,
            "p95": self.percentile_95,
            "p99": self.percentile_99,
            "iterations": self.iterations,
            "total_time": self.total_time,
            "ops_per_second": self.ops_per_second,
        }

    @classmethod
    def from_timings(cls, timings: List[float]) -> "BenchmarkStats":
        """Calculate stats from timing data."""
        if not timings:
            return cls(
                mean=0, median=0, std_dev=0, min_time=0, max_time=0,
                percentile_95=0, percentile_99=0, iterations=0,
                total_time=0, ops_per_second=0
            )

        sorted_timings = sorted(timings)
        n = len(sorted_timings)

        mean_val = statistics.mean(timings)
        ops_per_sec = 1 / mean_val if mean_val > 0 else 0

        return cls(
            mean=mean_val,
            median=statistics.median(timings),
            std_dev=statistics.stdev(timings) if n > 1 else 0,
            min_time=sorted_timings[0],
            max_time=sorted_timings[-1],
            percentile_95=sorted_timings[int(n * 0.95)] if n > 20 else sorted_timings[-1],
            percentile_99=sorted_timings[int(n * 0.99)] if n > 100 else sorted_timings[-1],
            iterations=n,
            total_time=sum(timings),
            ops_per_second=ops_per_sec,
        )


@dataclass
class Benchmark:
    """A single benchmark definition."""
    name: str
    func: Callable
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        if not self.description and self.func.__doc__:
            self.description = self.func.__doc__.strip()


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark runs.

    Attributes:
        mode: Benchmarking mode (iterations, time, auto)
        iterations: Number of iterations (for ITERATIONS mode)
        warmup_iterations: Warmup iterations before measurement
        min_time_s: Minimum run time (for TIME/AUTO mode)
        max_time_s: Maximum run time
        gc_collect: Force garbage collection between runs
        output_dir: Directory for benchmark reports
        compare_baseline: Path to baseline results for comparison
    """
    mode: BenchmarkMode = BenchmarkMode.AUTO
    iterations: int = DEFAULT_ITERATIONS
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    min_time_s: float = DEFAULT_MIN_TIME_S
    max_time_s: float = DEFAULT_MAX_TIME_S
    gc_collect: bool = True
    output_dir: str = ".pluribus/test-agent/benchmarks"
    compare_baseline: Optional[str] = None
    tags_filter: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "min_time_s": self.min_time_s,
            "max_time_s": self.max_time_s,
            "gc_collect": self.gc_collect,
            "output_dir": self.output_dir,
            "compare_baseline": self.compare_baseline,
            "tags_filter": self.tags_filter,
        }


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark."""
    benchmark_name: str
    stats: BenchmarkStats
    timings: List[float] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    comparison: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "stats": self.stats.to_dict(),
            "params": self.params,
            "metadata": self.metadata,
            "comparison": self.comparison,
        }


@dataclass
class SuiteResult:
    """Complete result of running a benchmark suite."""
    run_id: str
    started_at: float
    completed_at: Optional[float] = None
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    config: Optional[BenchmarkConfig] = None

    @property
    def duration_s(self) -> float:
        """Get total duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "system_info": self.system_info,
        }

    def get_benchmark(self, name: str) -> Optional[BenchmarkResult]:
        """Get a benchmark result by name."""
        for b in self.benchmarks:
            if b.benchmark_name == name:
                return b
        return None


# ============================================================================
# Bus Interface
# ============================================================================

class BenchmarkBus:
    """Bus interface for benchmarking."""

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
# Benchmark Suite
# ============================================================================

class BenchmarkSuite:
    """
    Orchestrates performance benchmarking.

    Features:
    - Multiple benchmarking modes (iterations, time-based, auto)
    - Statistical analysis with percentiles
    - Warmup iterations
    - Comparison with baseline results
    - Automatic iteration calibration

    PBTSO Phase: TEST, VERIFY
    Bus Topics: test.benchmark.run, test.benchmark.result
    """

    BUS_TOPICS = {
        "run": "test.benchmark.run",
        "result": "test.benchmark.result",
        "telemetry": "telemetry.benchmark",
    }

    def __init__(self, bus=None):
        """
        Initialize the benchmark suite.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or BenchmarkBus()
        self.benchmarks: List[Benchmark] = []

    def add_benchmark(
        self,
        name: str,
        func: Callable,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """
        Add a benchmark to the suite.

        Args:
            name: Benchmark name
            func: Function to benchmark
            setup: Optional setup function
            teardown: Optional teardown function
            params: Parameters to pass to function
            tags: Tags for filtering
            description: Benchmark description
        """
        self.benchmarks.append(Benchmark(
            name=name,
            func=func,
            setup=setup,
            teardown=teardown,
            params=params or {},
            tags=tags or [],
            description=description,
        ))

    def benchmark(
        self,
        name: Optional[str] = None,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Decorator to add a benchmark."""
        def decorator(func: Callable) -> Callable:
            benchmark_name = name or func.__name__
            self.add_benchmark(
                name=benchmark_name,
                func=func,
                setup=setup,
                teardown=teardown,
                params=params,
                tags=tags,
            )
            return func
        return decorator

    def run(self, config: Optional[BenchmarkConfig] = None) -> SuiteResult:
        """
        Run all benchmarks in the suite.

        Args:
            config: Benchmark configuration

        Returns:
            SuiteResult with all benchmark results
        """
        config = config or BenchmarkConfig()
        run_id = str(uuid.uuid4())

        result = SuiteResult(
            run_id=run_id,
            started_at=time.time(),
            system_info=self._get_system_info(),
            config=config,
        )

        # Emit start event
        self._emit_event("run", {
            "run_id": run_id,
            "status": "started",
            "benchmark_count": len(self.benchmarks),
        })

        # Load baseline if specified
        baseline = None
        if config.compare_baseline:
            baseline = self._load_baseline(config.compare_baseline)

        # Filter benchmarks by tags
        benchmarks_to_run = self._filter_benchmarks(config.tags_filter)

        # Run each benchmark
        for benchmark in benchmarks_to_run:
            bench_result = self._run_benchmark(benchmark, config)

            # Compare with baseline
            if baseline:
                baseline_result = baseline.get_benchmark(benchmark.name)
                if baseline_result:
                    bench_result.comparison = self._compare_results(
                        bench_result, baseline_result
                    )

            result.benchmarks.append(bench_result)

            # Emit result event
            self._emit_event("result", {
                "run_id": run_id,
                "benchmark_name": benchmark.name,
                "mean": bench_result.stats.mean,
                "ops_per_second": bench_result.stats.ops_per_second,
            })

        result.completed_at = time.time()

        # Emit completion event
        self._emit_event("telemetry", {
            "run_id": run_id,
            "status": "completed",
            "duration_s": result.duration_s,
            "benchmark_count": len(result.benchmarks),
        })

        # Save results
        self._save_results(result, config.output_dir)

        return result

    def _run_benchmark(
        self,
        benchmark: Benchmark,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        timings = []

        # Run setup
        if benchmark.setup:
            benchmark.setup()

        try:
            # Warmup iterations
            for _ in range(config.warmup_iterations):
                benchmark.func(**benchmark.params)

            # Determine iteration count
            if config.mode == BenchmarkMode.AUTO:
                iterations = self._calibrate_iterations(benchmark, config)
            elif config.mode == BenchmarkMode.TIME:
                iterations = self._estimate_iterations(benchmark, config)
            else:
                iterations = config.iterations

            # Run benchmark
            start_total = time.time()
            for _ in range(iterations):
                if config.gc_collect:
                    gc.collect()

                start = time.perf_counter()
                benchmark.func(**benchmark.params)
                end = time.perf_counter()

                timings.append(end - start)

                # Check time limit
                if time.time() - start_total > config.max_time_s:
                    break

        finally:
            # Run teardown
            if benchmark.teardown:
                benchmark.teardown()

        stats = BenchmarkStats.from_timings(timings)

        return BenchmarkResult(
            benchmark_name=benchmark.name,
            stats=stats,
            timings=timings,
            params=benchmark.params,
            metadata={
                "description": benchmark.description,
                "tags": benchmark.tags,
            },
        )

    def _calibrate_iterations(
        self,
        benchmark: Benchmark,
        config: BenchmarkConfig,
    ) -> int:
        """Calibrate the number of iterations needed."""
        # Run a few iterations to estimate time
        test_iterations = 5
        start = time.perf_counter()
        for _ in range(test_iterations):
            benchmark.func(**benchmark.params)
        elapsed = time.perf_counter() - start

        time_per_iter = elapsed / test_iterations

        # Calculate iterations needed for min_time
        if time_per_iter > 0:
            needed = int(config.min_time_s / time_per_iter)
            return max(10, min(needed, 10000))  # Clamp between 10 and 10000
        return config.iterations

    def _estimate_iterations(
        self,
        benchmark: Benchmark,
        config: BenchmarkConfig,
    ) -> int:
        """Estimate iterations for time-based mode."""
        return self._calibrate_iterations(benchmark, config)

    def _filter_benchmarks(
        self,
        tags_filter: Optional[List[str]],
    ) -> List[Benchmark]:
        """Filter benchmarks by tags."""
        if not tags_filter:
            return self.benchmarks

        return [
            b for b in self.benchmarks
            if any(tag in b.tags for tag in tags_filter)
        ]

    def _compare_results(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
    ) -> Dict[str, Any]:
        """Compare current result with baseline."""
        current_mean = current.stats.mean
        baseline_mean = baseline.stats.mean

        if baseline_mean == 0:
            return {
                "result": ComparisonResult.INCONCLUSIVE.value,
                "speedup": 0,
                "regression_percent": 0,
            }

        speedup = baseline_mean / current_mean
        regression_percent = ((current_mean - baseline_mean) / baseline_mean) * 100

        # Determine result (allowing 5% margin)
        if speedup > 1.05:
            result = ComparisonResult.FASTER
        elif speedup < 0.95:
            result = ComparisonResult.SLOWER
        else:
            result = ComparisonResult.SAME

        return {
            "result": result.value,
            "speedup": speedup,
            "regression_percent": regression_percent,
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
        }

    def _load_baseline(self, path: str) -> Optional[SuiteResult]:
        """Load baseline results from file."""
        try:
            with open(path) as f:
                data = json.load(f)

            # Reconstruct SuiteResult
            result = SuiteResult(
                run_id=data["run_id"],
                started_at=data["started_at"],
                completed_at=data.get("completed_at"),
                system_info=data.get("system_info", {}),
            )

            for b_data in data.get("benchmarks", []):
                stats = BenchmarkStats(**b_data["stats"])
                result.benchmarks.append(BenchmarkResult(
                    benchmark_name=b_data["benchmark_name"],
                    stats=stats,
                    params=b_data.get("params", {}),
                    metadata=b_data.get("metadata", {}),
                ))

            return result

        except (IOError, json.JSONDecodeError, KeyError):
            return None

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility."""
        import platform
        import sys

        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        }

        # Try to get more info
        try:
            import psutil
            info["cpu_count"] = psutil.cpu_count()
            info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            pass

        return info

    def _save_results(self, result: SuiteResult, output_dir: str) -> None:
        """Save benchmark results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_file = output_path / f"benchmark_{result.run_id}.json"
        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save as latest for baseline comparison
        latest_file = output_path / "benchmark_latest.json"
        with open(latest_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Generate markdown report
        report_file = output_path / f"benchmark_{result.run_id}.md"
        with open(report_file, "w") as f:
            f.write(self._generate_report(result))

    def _generate_report(self, result: SuiteResult) -> str:
        """Generate markdown benchmark report."""
        lines = [
            "# Benchmark Report",
            "",
            f"**Run ID**: {result.run_id}",
            f"**Date**: {datetime.fromtimestamp(result.started_at).isoformat()}",
            f"**Duration**: {result.duration_s:.2f}s",
            "",
            "## System Info",
            "",
        ]

        for key, value in result.system_info.items():
            lines.append(f"- **{key}**: {value}")

        lines.extend([
            "",
            "## Results",
            "",
            "| Benchmark | Mean | Median | Std Dev | Min | Max | Ops/s |",
            "|-----------|------|--------|---------|-----|-----|-------|",
        ])

        for b in result.benchmarks:
            s = b.stats
            lines.append(
                f"| {b.benchmark_name} | "
                f"{self._format_time(s.mean)} | "
                f"{self._format_time(s.median)} | "
                f"{self._format_time(s.std_dev)} | "
                f"{self._format_time(s.min_time)} | "
                f"{self._format_time(s.max_time)} | "
                f"{s.ops_per_second:.2f} |"
            )

        # Add comparison section if available
        has_comparisons = any(b.comparison for b in result.benchmarks)
        if has_comparisons:
            lines.extend([
                "",
                "## Comparison with Baseline",
                "",
                "| Benchmark | Result | Speedup | Regression |",
                "|-----------|--------|---------|------------|",
            ])

            for b in result.benchmarks:
                if b.comparison:
                    c = b.comparison
                    lines.append(
                        f"| {b.benchmark_name} | "
                        f"{c['result']} | "
                        f"{c['speedup']:.2f}x | "
                        f"{c['regression_percent']:+.1f}% |"
                    )

        return "\n".join(lines)

    def _format_time(self, seconds: float) -> str:
        """Format time in appropriate units."""
        if seconds < 1e-6:
            return f"{seconds * 1e9:.2f}ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:.2f}us"
        elif seconds < 1:
            return f"{seconds * 1e3:.2f}ms"
        else:
            return f"{seconds:.2f}s"

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.benchmark.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "benchmark",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# Convenience Functions
# ============================================================================

def benchmark(
    name: Optional[str] = None,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP_ITERATIONS,
):
    """
    Decorator for quick benchmarking.

    Usage:
        @benchmark("my_function")
        def my_function():
            # code to benchmark
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Store benchmark metadata
        wrapper._benchmark_name = name or func.__name__
        wrapper._benchmark_iterations = iterations
        wrapper._benchmark_warmup = warmup
        return wrapper

    return decorator


def timeit(func: Callable, *args, iterations: int = 100, **kwargs) -> BenchmarkStats:
    """
    Simple timing utility.

    Args:
        func: Function to time
        *args: Positional arguments
        iterations: Number of iterations
        **kwargs: Keyword arguments

    Returns:
        BenchmarkStats with timing results
    """
    timings = []

    # Warmup
    for _ in range(10):
        func(*args, **kwargs)

    # Timing runs
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        timings.append(end - start)

    return BenchmarkStats.from_timings(timings)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Benchmark Suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Suite")
    parser.add_argument("module", help="Module containing benchmarks")
    parser.add_argument("--iterations", "-n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/benchmarks")
    parser.add_argument("--baseline", help="Baseline file for comparison")
    parser.add_argument("--tags", nargs="+", help="Filter by tags")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Import module and find benchmarks
    import importlib
    try:
        module = importlib.import_module(args.module)
    except ImportError as e:
        print(f"Error importing module: {e}")
        exit(1)

    suite = BenchmarkSuite()

    # Find decorated functions
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "_benchmark_name"):
            suite.add_benchmark(
                name=obj._benchmark_name,
                func=obj,
            )

    if not suite.benchmarks:
        print("No benchmarks found in module")
        exit(1)

    config = BenchmarkConfig(
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        output_dir=args.output,
        compare_baseline=args.baseline,
        tags_filter=args.tags,
    )

    result = suite.run(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Benchmark Results")
        print(f"{'='*60}")
        print(f"Run ID: {result.run_id}")
        print(f"Duration: {result.duration_s:.2f}s")
        print(f"Benchmarks: {len(result.benchmarks)}")
        print()

        for b in result.benchmarks:
            print(f"{b.benchmark_name}:")
            print(f"  Mean: {b.stats.mean * 1000:.3f}ms")
            print(f"  Ops/s: {b.stats.ops_per_second:.2f}")
            if b.comparison:
                print(f"  vs Baseline: {b.comparison['result']} ({b.comparison['speedup']:.2f}x)")
            print()

        print(f"Report saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
