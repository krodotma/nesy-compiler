"""
Benchmarks Package
==================

Performance benchmarks for the Creative section subsystems.

This package provides comprehensive benchmarking tools for:
- grammars: CGP/EGGP evolution, parsing
- visual: Image generation, style transfer, upscaling
- auralux: TTS/STT synthesis
- avatars: 3DGS, SMPLX operations
- dits: Kernel evaluation, narrative generation
- pipelines: Cross-subsystem pipelines
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    # Core
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    # Subsystem benchmarks
    "GrammarsBenchmark",
    "VisualBenchmark",
    "AuraluxBenchmark",
    "AvatarsBenchmark",
    "DiTSBenchmark",
    "PipelinesBenchmark",
    # Utilities
    "run_all_benchmarks",
    "run_benchmark_suite",
]

from .bench_runner import BenchmarkResult, BenchmarkRunner, BenchmarkSuite
from .bench_grammars import GrammarsBenchmark
from .bench_visual import VisualBenchmark
from .bench_auralux import AuraluxBenchmark
from .bench_avatars import AvatarsBenchmark
from .bench_dits import DiTSBenchmark
from .bench_pipelines import PipelinesBenchmark


def run_all_benchmarks(
    iterations: int = 10,
    warmup: int = 2,
    output_dir: str | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """
    Run all subsystem benchmarks.

    Args:
        iterations: Number of iterations per benchmark
        warmup: Number of warmup iterations to discard
        output_dir: Optional directory to save results

    Returns:
        Dict mapping subsystem names to their benchmark results
    """
    runner = BenchmarkRunner(iterations=iterations, warmup=warmup)

    suites = [
        GrammarsBenchmark(),
        VisualBenchmark(),
        AuraluxBenchmark(),
        AvatarsBenchmark(),
        DiTSBenchmark(),
        PipelinesBenchmark(),
    ]

    results: dict[str, list[BenchmarkResult]] = {}

    for suite in suites:
        suite_results = runner.run_suite(suite)
        results[suite.name] = suite_results

    if output_dir:
        runner.save_results(results, output_dir)

    return results


def run_benchmark_suite(
    suite_name: str,
    iterations: int = 10,
    warmup: int = 2,
) -> list[BenchmarkResult]:
    """
    Run a specific benchmark suite.

    Args:
        suite_name: Name of suite ("grammars", "visual", etc.)
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        List of benchmark results
    """
    suites = {
        "grammars": GrammarsBenchmark,
        "visual": VisualBenchmark,
        "auralux": AuraluxBenchmark,
        "avatars": AvatarsBenchmark,
        "dits": DiTSBenchmark,
        "pipelines": PipelinesBenchmark,
    }

    if suite_name not in suites:
        raise ValueError(f"Unknown suite: {suite_name}. Available: {list(suites.keys())}")

    runner = BenchmarkRunner(iterations=iterations, warmup=warmup)
    suite = suites[suite_name]()

    return runner.run_suite(suite)
