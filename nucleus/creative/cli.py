#!/usr/bin/env python3
"""
Creative CLI - Command Line Interface for Creative Section
==========================================================

Main entry point for the Creative subsystem CLI.

Commands:
- generate: Generate content using a specific subsystem
- pipeline: Run a multi-stage pipeline
- health: Check health of subsystems
- benchmark: Run performance benchmarks
- cache: Manage cache (stats, clear)

Usage:
    python -m nucleus.creative.cli generate visual --prompt "sunset"
    python -m nucleus.creative.cli health --all
    python -m nucleus.creative.cli pipeline visual --stages generate,upscale
    python -m nucleus.creative.cli benchmark --subsystem grammars
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


class OutputFormatter:
    """Format output for terminal display."""

    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
    }

    def __init__(self, color: bool = True, json_output: bool = False):
        self.color = color
        self.json_output = json_output

    def _c(self, color: str, text: str) -> str:
        """Apply color if enabled."""
        if not self.color or self.json_output:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def header(self, text: str) -> str:
        """Format a header."""
        return self._c("bold", f"\n{'=' * 60}\n{text}\n{'=' * 60}")

    def success(self, text: str) -> str:
        """Format success message."""
        return self._c("green", f"[OK] {text}")

    def error(self, text: str) -> str:
        """Format error message."""
        return self._c("red", f"[ERROR] {text}")

    def warning(self, text: str) -> str:
        """Format warning message."""
        return self._c("yellow", f"[WARN] {text}")

    def info(self, text: str) -> str:
        """Format info message."""
        return self._c("blue", f"[INFO] {text}")

    def progress(self, current: int, total: int, label: str = "") -> str:
        """Format a progress bar."""
        if self.json_output:
            return json.dumps({"progress": current / total, "label": label})

        width = 40
        filled = int(width * current / total)
        bar = "#" * filled + "-" * (width - filled)
        percent = current / total * 100
        return f"[{bar}] {percent:5.1f}% {label}"

    def table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format a table."""
        if self.json_output:
            return json.dumps(
                [dict(zip(headers, row)) for row in rows], indent=2
            )

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Format header
        header_line = " | ".join(
            h.ljust(widths[i]) for i, h in enumerate(headers)
        )
        separator = "-+-".join("-" * w for w in widths)

        # Format rows
        row_lines = [
            " | ".join(
                str(cell).ljust(widths[i]) for i, cell in enumerate(row)
            )
            for row in rows
        ]

        return "\n".join(
            [
                self._c("bold", header_line),
                separator,
                *row_lines,
            ]
        )

    def json(self, data: Any) -> str:
        """Format as JSON."""
        return json.dumps(data, indent=2, default=str)


# =============================================================================
# COMMAND IMPLEMENTATIONS
# =============================================================================


async def cmd_health(args: argparse.Namespace, fmt: OutputFormatter) -> int:
    """Run health checks on subsystems."""
    from nucleus.creative.health import check_health, HealthStatus

    print(fmt.header("Creative Section Health Check"))

    start_time = time.time()
    result = await check_health()
    duration = (time.time() - start_time) * 1000

    if args.json:
        print(fmt.json(result.to_dict()))
        return 0 if result.overall_status == HealthStatus.HEALTHY else 1

    # Display results
    status_colors = {
        HealthStatus.HEALTHY: "green",
        HealthStatus.DEGRADED: "yellow",
        HealthStatus.UNHEALTHY: "red",
        HealthStatus.UNKNOWN: "magenta",
    }

    print(f"\nOverall Status: {fmt._c(status_colors.get(result.overall_status, 'reset'), result.overall_status.value.upper())}")
    print(f"Check Duration: {duration:.1f}ms")
    print(f"Check ID: {result.check_id}")
    print()

    # Subsystem table
    headers = ["Subsystem", "Status", "Latency", "Error Rate", "Message"]
    rows = []
    for name, health in sorted(result.subsystems.items()):
        status_text = health.status.value.upper()
        rows.append([
            name,
            status_text,
            f"{health.latency_ms:.1f}ms",
            f"{health.error_rate * 100:.1f}%",
            health.message or "",
        ])

    print(fmt.table(headers, rows))
    print()

    # Summary
    print(f"Summary: {result.healthy_count} healthy, {result.degraded_count} degraded, {result.unhealthy_count} unhealthy")

    return 0 if result.overall_status == HealthStatus.HEALTHY else 1


async def cmd_generate(args: argparse.Namespace, fmt: OutputFormatter) -> int:
    """Generate content using a subsystem."""
    from nucleus.creative import create_pipeline, run_pipeline, SUBSYSTEMS

    subsystem = args.subsystem
    if subsystem not in SUBSYSTEMS:
        print(fmt.error(f"Unknown subsystem: {subsystem}"))
        print(f"Available: {', '.join(SUBSYSTEMS.keys())}")
        return 1

    print(fmt.header(f"Generate - {SUBSYSTEMS[subsystem]['name']}"))
    print(fmt.info(f"Subsystem: {subsystem}"))
    print(fmt.info(f"Description: {SUBSYSTEMS[subsystem]['description']}"))

    # Build params from CLI args
    params = {}
    if args.prompt:
        params["prompt"] = args.prompt
    if args.params:
        try:
            params.update(json.loads(args.params))
        except json.JSONDecodeError as e:
            print(fmt.error(f"Invalid JSON params: {e}"))
            return 1

    # Create and run pipeline
    print(fmt.info(f"Parameters: {json.dumps(params, indent=2)}"))
    print()

    job = create_pipeline(subsystem, **params)
    print(fmt.info(f"Job ID: {job.id}"))
    print(fmt.info(f"Status: {job.status}"))

    if args.dry_run:
        print(fmt.warning("Dry run - not executing pipeline"))
        if args.json:
            print(fmt.json({
                "job_id": job.id,
                "mode": job.mode,
                "params": params,
                "dry_run": True,
            }))
        return 0

    # Define progress callback
    def on_progress(stage: str, progress: float) -> None:
        print(fmt.progress(int(progress), 100, f"Stage: {stage}"))

    try:
        asset = await run_pipeline(job)
        print()
        print(fmt.success(f"Generation complete!"))
        print(fmt.info(f"Asset ID: {asset.id}"))
        print(fmt.info(f"Asset Type: {asset.type}"))
        print(fmt.info(f"Output Path: {asset.path}"))

        if args.json:
            print(fmt.json({
                "job_id": job.id,
                "asset_id": asset.id,
                "asset_type": asset.type,
                "path": asset.path,
                "metadata": asset.metadata,
            }))

        return 0

    except Exception as e:
        print(fmt.error(f"Generation failed: {e}"))
        if args.json:
            print(fmt.json({"error": str(e), "job_id": job.id}))
        return 1


async def cmd_pipeline(args: argparse.Namespace, fmt: OutputFormatter) -> int:
    """Run a multi-stage pipeline."""
    from nucleus.creative import create_pipeline, run_pipeline, SUBSYSTEMS

    subsystem = args.subsystem
    if subsystem not in SUBSYSTEMS:
        print(fmt.error(f"Unknown subsystem: {subsystem}"))
        return 1

    print(fmt.header(f"Pipeline - {SUBSYSTEMS[subsystem]['name']}"))

    # Parse stages
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
        print(fmt.info(f"Custom stages: {stages}"))

    # Parse params
    params = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(fmt.error(f"Invalid JSON params: {e}"))
            return 1

    job = create_pipeline(subsystem, stages=stages, **params)

    print(fmt.info(f"Job ID: {job.id}"))
    print(fmt.info(f"Mode: {job.mode}"))
    print(fmt.info(f"Stages: {[s.name for s in job.params.get('_stages', [])]}"))
    print()

    if args.dry_run:
        print(fmt.warning("Dry run - pipeline not executed"))
        return 0

    try:
        start_time = time.time()
        asset = await run_pipeline(job)
        duration = time.time() - start_time

        print()
        print(fmt.success(f"Pipeline complete in {duration:.2f}s"))
        print(fmt.info(f"Output: {asset.path}"))

        if args.json:
            print(fmt.json({
                "job_id": job.id,
                "asset_id": asset.id,
                "duration_s": duration,
                "stages": [s.name for s in job.params.get("_stages", [])],
            }))

        return 0

    except Exception as e:
        print(fmt.error(f"Pipeline failed: {e}"))
        return 1


async def cmd_benchmark(args: argparse.Namespace, fmt: OutputFormatter) -> int:
    """Run performance benchmarks."""
    from nucleus.creative import SUBSYSTEMS

    print(fmt.header("Creative Section Benchmarks"))

    subsystem = args.subsystem
    if subsystem and subsystem not in SUBSYSTEMS:
        print(fmt.error(f"Unknown subsystem: {subsystem}"))
        return 1

    # Determine which subsystems to benchmark
    targets = [subsystem] if subsystem else list(SUBSYSTEMS.keys())
    iterations = args.iterations

    print(fmt.info(f"Benchmarking: {', '.join(targets)}"))
    print(fmt.info(f"Iterations: {iterations}"))
    print()

    results: Dict[str, Dict[str, Any]] = {}

    for target in targets:
        print(fmt.info(f"Benchmarking {target}..."))

        # Simple import timing benchmark
        times: List[float] = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                getter = SUBSYSTEMS[target]["get"]
                _ = getter()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            except Exception as e:
                print(fmt.warning(f"  Iteration {i+1} failed: {e}"))

        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            results[target] = {
                "avg_ms": round(avg_time, 3),
                "min_ms": round(min_time, 3),
                "max_ms": round(max_time, 3),
                "iterations": len(times),
                "status": "ok",
            }
            print(f"  {fmt.success(f'avg={avg_time:.3f}ms min={min_time:.3f}ms max={max_time:.3f}ms')}")
        else:
            results[target] = {
                "status": "failed",
                "iterations": 0,
            }
            print(f"  {fmt.error('All iterations failed')}")

    print()
    print(fmt.header("Benchmark Results"))

    headers = ["Subsystem", "Avg (ms)", "Min (ms)", "Max (ms)", "Status"]
    rows = []
    for name, data in sorted(results.items()):
        if data["status"] == "ok":
            rows.append([
                name,
                f"{data['avg_ms']:.3f}",
                f"{data['min_ms']:.3f}",
                f"{data['max_ms']:.3f}",
                "OK",
            ])
        else:
            rows.append([name, "-", "-", "-", "FAILED"])

    print(fmt.table(headers, rows))

    if args.json:
        print()
        print(fmt.json(results))

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "subsystems": results,
            }, f, indent=2)
        print(fmt.info(f"Results saved to {args.output}"))

    return 0


async def cmd_cache(args: argparse.Namespace, fmt: OutputFormatter) -> int:
    """Manage cache."""
    from nucleus.creative.cache import get_cache

    print(fmt.header("Creative Section Cache"))

    cache = get_cache()

    if args.action == "stats":
        stats = cache.stats
        if stats is None:
            print(fmt.warning("Statistics not enabled"))
            return 0

        print(fmt.table(
            ["Metric", "Value"],
            [
                ["Hits", str(stats.hits)],
                ["Misses", str(stats.misses)],
                ["Hit Rate", f"{stats.hit_rate:.1f}%"],
                ["Writes", str(stats.writes)],
                ["Evictions", str(stats.evictions)],
                ["Expirations", str(stats.expirations)],
                ["Total Requests", str(stats.total_requests)],
                ["Current Size", str(len(cache))],
            ],
        ))

        if args.json:
            print()
            print(fmt.json(stats.to_dict()))

    elif args.action == "clear":
        if args.dry_run:
            print(fmt.warning(f"Would clear {len(cache)} entries (dry run)"))
        else:
            count = cache.clear()
            print(fmt.success(f"Cleared {count} cache entries"))

    elif args.action == "cleanup":
        count = cache.cleanup_expired()
        print(fmt.success(f"Removed {count} expired entries"))

    elif args.action == "keys":
        keys = cache.keys()
        print(fmt.info(f"Cache keys ({len(keys)} entries):"))
        for key in keys[:50]:  # Limit display
            print(f"  - {key[:60]}...")
        if len(keys) > 50:
            print(f"  ... and {len(keys) - 50} more")

        if args.json:
            print(fmt.json({"keys": keys, "count": len(keys)}))

    return 0


# =============================================================================
# ARGUMENT PARSER
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="creative",
        description="Creative Section CLI - Multimodal AI/ML content generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s health --all
  %(prog)s generate visual --prompt "sunset over mountains"
  %(prog)s pipeline cinema --stages storyboard,generate
  %(prog)s benchmark --subsystem grammars --iterations 10
  %(prog)s cache stats
        """,
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Check health of subsystems",
    )
    health_parser.add_argument(
        "--all",
        action="store_true",
        help="Check all subsystems",
    )
    health_parser.add_argument(
        "--subsystem",
        choices=["grammars", "cinema", "visual", "auralux", "avatars", "dits"],
        help="Check specific subsystem",
    )

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate content using a subsystem",
    )
    gen_parser.add_argument(
        "subsystem",
        choices=["grammars", "cinema", "visual", "auralux", "avatars", "dits"],
        help="Subsystem to use for generation",
    )
    gen_parser.add_argument(
        "--prompt", "-p",
        help="Text prompt for generation",
    )
    gen_parser.add_argument(
        "--params",
        help="Additional params as JSON string",
    )
    gen_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without executing",
    )
    gen_parser.add_argument(
        "--output", "-o",
        help="Output path for generated asset",
    )

    # Pipeline command
    pipe_parser = subparsers.add_parser(
        "pipeline",
        help="Run a multi-stage pipeline",
    )
    pipe_parser.add_argument(
        "subsystem",
        choices=["grammars", "cinema", "visual", "auralux", "avatars", "dits"],
        help="Subsystem for pipeline",
    )
    pipe_parser.add_argument(
        "--stages",
        help="Comma-separated list of stages",
    )
    pipe_parser.add_argument(
        "--params",
        help="Pipeline params as JSON string",
    )
    pipe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without executing",
    )

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmarks",
    )
    bench_parser.add_argument(
        "--subsystem",
        choices=["grammars", "cinema", "visual", "auralux", "avatars", "dits"],
        help="Benchmark specific subsystem (default: all)",
    )
    bench_parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=5,
        help="Number of iterations (default: 5)",
    )
    bench_parser.add_argument(
        "--output", "-o",
        help="Save results to file",
    )

    # Cache command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage cache",
    )
    cache_parser.add_argument(
        "action",
        choices=["stats", "clear", "cleanup", "keys"],
        help="Cache action",
    )
    cache_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done",
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================


async def async_main(args: argparse.Namespace) -> int:
    """Async main entry point."""
    fmt = OutputFormatter(
        color=not args.no_color,
        json_output=args.json,
    )

    if args.command is None:
        create_parser().print_help()
        return 0

    commands = {
        "health": cmd_health,
        "generate": cmd_generate,
        "pipeline": cmd_pipeline,
        "benchmark": cmd_benchmark,
        "cache": cmd_cache,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(fmt.error(f"Unknown command: {args.command}"))
        return 1

    return await handler(args, fmt)


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        if hasattr(args, "verbose") and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
