"""
Benchmarks CLI Entry Point
==========================

Run benchmarks from the command line:

    python -m nucleus.creative.benchmarks
    python -m nucleus.creative.benchmarks --suite grammars
    python -m nucleus.creative.benchmarks --suite visual --iterations 20
    python -m nucleus.creative.benchmarks --list
    python -m nucleus.creative.benchmarks --output ./results
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bench_runner import BenchmarkResult


def format_duration(ms: float) -> str:
    """Format duration in human-readable form."""
    if ms < 1:
        return f"{ms * 1000:.2f}us"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms / 1000:.2f}s"


def print_result(result: "BenchmarkResult", verbose: bool = False) -> None:
    """Print a single benchmark result."""
    status = "PASS" if result.success else "FAIL"
    status_color = "\033[92m" if result.success else "\033[91m"
    reset = "\033[0m"

    print(f"  [{status_color}{status}{reset}] {result.name}")
    print(f"       Mean: {format_duration(result.mean_ms)}")
    print(f"       Std:  {format_duration(result.std_ms)}")
    print(f"       Min:  {format_duration(result.min_ms)}")
    print(f"       Max:  {format_duration(result.max_ms)}")

    if verbose and result.metadata:
        print(f"       Meta: {result.metadata}")

    if result.error:
        print(f"       Error: {result.error}")

    print()


def print_suite_summary(
    suite_name: str,
    results: list["BenchmarkResult"],
) -> None:
    """Print summary for a benchmark suite."""
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed

    total_time = sum(r.mean_ms for r in results)

    print(f"\n{'=' * 60}")
    print(f"Suite: {suite_name}")
    print(f"{'=' * 60}")
    print(f"Benchmarks: {passed}/{total} passed, {failed} failed")
    print(f"Total time: {format_duration(total_time)}")
    print(f"{'=' * 60}\n")

    for result in results:
        print_result(result)


def list_benchmarks() -> None:
    """List all available benchmarks."""
    from . import (
        GrammarsBenchmark,
        VisualBenchmark,
        AuraluxBenchmark,
        AvatarsBenchmark,
        DiTSBenchmark,
        PipelinesBenchmark,
    )

    suites = [
        GrammarsBenchmark(),
        VisualBenchmark(),
        AuraluxBenchmark(),
        AvatarsBenchmark(),
        DiTSBenchmark(),
        PipelinesBenchmark(),
    ]

    print("\nAvailable Benchmark Suites:")
    print("=" * 60)

    for suite in suites:
        print(f"\n{suite.name}: {suite.description}")
        benchmarks = suite.get_benchmarks()
        for name, _ in benchmarks:
            print(f"  - {name}")

    print()


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Creative section benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m nucleus.creative.benchmarks
    python -m nucleus.creative.benchmarks --suite grammars
    python -m nucleus.creative.benchmarks --iterations 50 --warmup 5
    python -m nucleus.creative.benchmarks --output ./benchmark_results
    python -m nucleus.creative.benchmarks --list
        """,
    )

    parser.add_argument(
        "--suite", "-s",
        type=str,
        default=None,
        choices=["grammars", "visual", "auralux", "avatars", "dits", "pipelines"],
        help="Run a specific benchmark suite",
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Number of iterations per benchmark (default: 10)",
    )

    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results",
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available benchmarks",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return 0

    # Import here to avoid circular imports
    from . import run_all_benchmarks, run_benchmark_suite
    from .bench_runner import BenchmarkRunner

    print(f"\nPluribuS Creative Benchmarks")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")

    try:
        if args.suite:
            results = run_benchmark_suite(
                args.suite,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            print_suite_summary(args.suite, results)
        else:
            all_results = run_all_benchmarks(
                iterations=args.iterations,
                warmup=args.warmup,
                output_dir=args.output,
            )

            for suite_name, results in all_results.items():
                print_suite_summary(suite_name, results)

        if args.output:
            print(f"Results saved to: {args.output}")

        if args.json:
            import json
            if args.suite:
                data = [r.to_dict() for r in results]
            else:
                data = {
                    suite: [r.to_dict() for r in res]
                    for suite, res in all_results.items()
                }
            print(json.dumps(data, indent=2))

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
