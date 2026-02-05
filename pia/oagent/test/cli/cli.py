#!/usr/bin/env python3
"""
Step 130: Test CLI

Complete CLI interface for all test operations.

PBTSO Phase: All Phases
Bus Topics:
- test.cli.command (emits)
- test.cli.result (emits)

Dependencies: Steps 101-129 (All Test Components)
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Constants
# ============================================================================

VERSION = "1.0.0"
PROGRAM_NAME = "test-agent"


class OutputFormat(Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    TABLE = "table"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class CLIConfig:
    """
    Configuration for the CLI.

    Attributes:
        output_format: Output format
        verbose: Verbose output
        color: Enable color output
        config_file: Path to config file
        working_dir: Working directory
        output_dir: Output directory
    """
    output_format: OutputFormat = OutputFormat.TEXT
    verbose: bool = False
    color: bool = True
    config_file: Optional[str] = None
    working_dir: str = "."
    output_dir: str = ".pluribus/test-agent"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_format": self.output_format.value,
            "verbose": self.verbose,
            "color": self.color,
            "config_file": self.config_file,
            "working_dir": self.working_dir,
            "output_dir": self.output_dir,
        }


@dataclass
class CLIResult:
    """Result of a CLI command."""
    success: bool
    message: str
    data: Any = None
    exit_code: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "exit_code": self.exit_code,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class CLIBus:
    """Bus interface for CLI with file locking."""

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
# Output Formatters
# ============================================================================

class OutputFormatter:
    """Formats CLI output."""

    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
    }

    def __init__(self, use_color: bool = True):
        self.use_color = use_color

    def color(self, text: str, color: str) -> str:
        """Apply color to text."""
        if not self.use_color:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def success(self, message: str) -> str:
        """Format success message."""
        return self.color(f"[OK] {message}", "green")

    def error(self, message: str) -> str:
        """Format error message."""
        return self.color(f"[ERROR] {message}", "red")

    def warning(self, message: str) -> str:
        """Format warning message."""
        return self.color(f"[WARN] {message}", "yellow")

    def info(self, message: str) -> str:
        """Format info message."""
        return self.color(f"[INFO] {message}", "blue")

    def header(self, title: str) -> str:
        """Format section header."""
        line = "=" * 60
        return f"\n{line}\n{self.color(title, 'bold')}\n{line}"

    def table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format data as table."""
        if not rows:
            return "No data"

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Build table
        lines = []

        # Header
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Rows
        for row in rows:
            row_line = " | ".join(
                str(cell).ljust(widths[i]) if i < len(widths) else str(cell)
                for i, cell in enumerate(row)
            )
            lines.append(row_line)

        return "\n".join(lines)


# ============================================================================
# Test CLI
# ============================================================================

class TestCLI:
    """
    Complete CLI interface for test operations.

    Commands:
    - run: Run tests
    - list: List tests
    - report: Generate reports
    - dashboard: Show dashboard
    - history: Query history
    - compare: Compare runs
    - schedule: Manage schedules
    - cache: Manage cache
    - notify: Send notifications
    - coverage: Coverage operations
    - flaky: Flaky test operations
    - parallel: Parallel execution
    - api: Start API server

    PBTSO Phase: All Phases
    Bus Topics: test.cli.command, test.cli.result
    """

    BUS_TOPICS = {
        "command": "test.cli.command",
        "result": "test.cli.result",
    }

    def __init__(self, config: Optional[CLIConfig] = None):
        """
        Initialize the CLI.

        Args:
            config: CLI configuration
        """
        self.config = config or CLIConfig()
        self.bus = CLIBus()
        self.formatter = OutputFormatter(use_color=self.config.color)
        self._commands: Dict[str, Callable] = {}

        # Register commands
        self._register_commands()

    def _register_commands(self) -> None:
        """Register CLI commands."""
        self._commands = {
            "run": self._cmd_run,
            "list": self._cmd_list,
            "report": self._cmd_report,
            "dashboard": self._cmd_dashboard,
            "history": self._cmd_history,
            "compare": self._cmd_compare,
            "schedule": self._cmd_schedule,
            "cache": self._cmd_cache,
            "notify": self._cmd_notify,
            "coverage": self._cmd_coverage,
            "flaky": self._cmd_flaky,
            "parallel": self._cmd_parallel,
            "api": self._cmd_api,
            "status": self._cmd_status,
            "config": self._cmd_config,
        }

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI.

        Args:
            args: Command line arguments

        Returns:
            Exit code
        """
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)

        # Update config from global args
        if hasattr(parsed_args, "json") and parsed_args.json:
            self.config.output_format = OutputFormat.JSON
        if hasattr(parsed_args, "verbose") and parsed_args.verbose:
            self.config.verbose = True
        if hasattr(parsed_args, "no_color") and parsed_args.no_color:
            self.config.color = False
            self.formatter = OutputFormatter(use_color=False)

        # Get command
        command = getattr(parsed_args, "command", None)

        if command is None:
            parser.print_help()
            return 0

        # Emit command event
        self._emit_event("command", {
            "command": command,
            "args": vars(parsed_args),
        })

        # Execute command
        handler = self._commands.get(command)
        if handler is None:
            print(self.formatter.error(f"Unknown command: {command}"))
            return 1

        try:
            result = handler(parsed_args)

            # Output result
            if self.config.output_format == OutputFormat.JSON:
                print(json.dumps(result.to_dict(), indent=2))
            elif not result.success:
                print(self.formatter.error(result.message))

            # Emit result event
            self._emit_event("result", {
                "command": command,
                "success": result.success,
                "exit_code": result.exit_code,
            })

            return result.exit_code

        except KeyboardInterrupt:
            print("\nAborted")
            return 130
        except Exception as e:
            print(self.formatter.error(str(e)))
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog=PROGRAM_NAME,
            description="Test Agent CLI - Comprehensive test automation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s run tests/                    Run tests in directory
  %(prog)s run tests/ --parallel -w 4    Run tests in parallel
  %(prog)s list                          List available tests
  %(prog)s report results.json           Generate report
  %(prog)s dashboard --watch             Show live dashboard
  %(prog)s history stats                 Show test statistics
  %(prog)s compare run1.json run2.json   Compare test runs
  %(prog)s schedule add daily tests/     Add scheduled job
  %(prog)s cache stats                   Show cache statistics
  %(prog)s api --port 8080               Start API server
            """,
        )

        # Global options
        parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
        parser.add_argument("--json", action="store_true", help="Output as JSON")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--no-color", action="store_true", help="Disable color output")
        parser.add_argument("--config", help="Config file path")
        parser.add_argument("--output-dir", "-o", default=".pluribus/test-agent",
                           help="Output directory")

        subparsers = parser.add_subparsers(dest="command", help="Commands")

        # Run command
        run_parser = subparsers.add_parser("run", help="Run tests")
        run_parser.add_argument("tests", nargs="*", default=["tests/"], help="Test paths")
        run_parser.add_argument("--parallel", "-p", action="store_true", help="Run in parallel")
        run_parser.add_argument("--workers", "-w", type=int, default=4, help="Number of workers")
        run_parser.add_argument("--coverage", action="store_true", help="Collect coverage")
        run_parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
        run_parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
        run_parser.add_argument("--marker", "-m", action="append", help="Pytest markers")

        # List command
        list_parser = subparsers.add_parser("list", help="List tests")
        list_parser.add_argument("path", nargs="?", default="tests/", help="Test path")
        list_parser.add_argument("--pattern", default="test_*.py", help="File pattern")

        # Report command
        report_parser = subparsers.add_parser("report", help="Generate report")
        report_parser.add_argument("input", help="Input JSON file")
        report_parser.add_argument("--format", "-f", nargs="*",
                                  choices=["json", "html", "markdown", "junit"],
                                  default=["html", "markdown"])
        report_parser.add_argument("--title", default="Test Report")

        # Dashboard command
        dashboard_parser = subparsers.add_parser("dashboard", help="Show dashboard")
        dashboard_parser.add_argument("--watch", action="store_true", help="Watch mode")
        dashboard_parser.add_argument("--interval", type=int, default=1000, help="Update interval (ms)")

        # History command
        history_parser = subparsers.add_parser("history", help="Query history")
        history_parser.add_argument("action", nargs="?", default="stats",
                                   choices=["stats", "query", "export", "runs"])
        history_parser.add_argument("--days", type=int, default=7, help="Days to analyze")
        history_parser.add_argument("--test", help="Filter by test name")
        history_parser.add_argument("--limit", type=int, default=20)

        # Compare command
        compare_parser = subparsers.add_parser("compare", help="Compare test runs")
        compare_parser.add_argument("base", help="Base run JSON file")
        compare_parser.add_argument("compare", help="Compare run JSON file")
        compare_parser.add_argument("--format", "-f", choices=["json", "markdown"], default="markdown")

        # Schedule command
        schedule_parser = subparsers.add_parser("schedule", help="Manage schedules")
        schedule_parser.add_argument("action", choices=["list", "add", "remove", "run", "enable", "disable"])
        schedule_parser.add_argument("--name", help="Job name")
        schedule_parser.add_argument("--frequency", choices=["hourly", "daily", "weekly"], default="daily")
        schedule_parser.add_argument("--tests", nargs="*", default=["tests/"])
        schedule_parser.add_argument("--job-id", help="Job ID")

        # Cache command
        cache_parser = subparsers.add_parser("cache", help="Manage cache")
        cache_parser.add_argument("action", choices=["stats", "clear", "lookup", "list"])
        cache_parser.add_argument("--test", help="Test name for lookup")
        cache_parser.add_argument("--limit", type=int, default=20)

        # Notify command
        notify_parser = subparsers.add_parser("notify", help="Send notifications")
        notify_parser.add_argument("--title", required=True, help="Notification title")
        notify_parser.add_argument("--message", required=True, help="Notification message")
        notify_parser.add_argument("--channel", choices=["console", "file", "slack"], default="console")
        notify_parser.add_argument("--severity", choices=["info", "warning", "error"], default="info")

        # Coverage command
        coverage_parser = subparsers.add_parser("coverage", help="Coverage operations")
        coverage_parser.add_argument("action", choices=["show", "report", "diff"])
        coverage_parser.add_argument("--file", help="Coverage file")

        # Flaky command
        flaky_parser = subparsers.add_parser("flaky", help="Flaky test operations")
        flaky_parser.add_argument("action", choices=["list", "detect", "quarantine", "release"])
        flaky_parser.add_argument("--test", help="Test name")
        flaky_parser.add_argument("--retries", type=int, default=3)

        # Parallel command
        parallel_parser = subparsers.add_parser("parallel", help="Parallel execution")
        parallel_parser.add_argument("tests", nargs="*", default=["tests/"])
        parallel_parser.add_argument("--workers", "-w", type=int, help="Number of workers")
        parallel_parser.add_argument("--strategy", choices=["round_robin", "balanced"], default="balanced")

        # API command
        api_parser = subparsers.add_parser("api", help="Start API server")
        api_parser.add_argument("--host", default="127.0.0.1")
        api_parser.add_argument("--port", type=int, default=8080)

        # Status command
        status_parser = subparsers.add_parser("status", help="Show agent status")

        # Config command
        config_parser = subparsers.add_parser("config", help="Show configuration")
        config_parser.add_argument("--show", action="store_true", help="Show current config")

        return parser

    # ========================================================================
    # Command Handlers
    # ========================================================================

    def _cmd_run(self, args) -> CLIResult:
        """Run tests."""
        from .parallel.parallelizer import TestParallelizer, ParallelConfig, PartitionStrategy
        from .runner.orchestrator import TestRunnerOrchestrator, TestRunConfig, RunnerType

        if args.parallel:
            # Use parallel runner
            config = ParallelConfig(
                workers=args.workers,
                timeout_s=args.timeout,
                fail_fast=args.fail_fast,
            )
            parallelizer = TestParallelizer(config=config)
            result = parallelizer.run(args.tests)

            if self.config.output_format != OutputFormat.JSON:
                status = "PASS" if result.failed == 0 else "FAIL"
                print(self.formatter.header(f"Test Run [{status}]"))
                print(f"Duration: {result.duration_s:.2f}s")
                print(f"Workers: {result.total_workers}")
                print(f"Speedup: {result.speedup:.2f}x")
                print()
                print(f"Total: {result.total_tests}")
                print(f"Passed: {result.passed}")
                print(f"Failed: {result.failed}")
                print(f"Skipped: {result.skipped}")

            return CLIResult(
                success=result.failed == 0,
                message=f"Tests completed: {result.passed}/{result.total_tests} passed",
                data=result.to_dict(),
                exit_code=0 if result.failed == 0 else 1,
            )
        else:
            # Use sequential runner
            config = TestRunConfig(
                test_paths=args.tests,
                timeout_s=args.timeout,
                fail_fast=args.fail_fast,
                collect_coverage=args.coverage,
                markers=args.marker,
            )
            runner = TestRunnerOrchestrator()
            result = runner.run_tests(config)

            if self.config.output_format != OutputFormat.JSON:
                status = "PASS" if result.failed == 0 else "FAIL"
                print(self.formatter.header(f"Test Run [{status}]"))
                print(f"Duration: {result.duration_s:.2f}s")
                print(f"Total: {result.total_tests}")
                print(f"Passed: {result.passed}")
                print(f"Failed: {result.failed}")
                print(f"Skipped: {result.skipped}")

            return CLIResult(
                success=result.failed == 0,
                message=f"Tests completed: {result.passed}/{result.total_tests} passed",
                data=result.to_dict(),
                exit_code=0 if result.failed == 0 else 1,
            )

    def _cmd_list(self, args) -> CLIResult:
        """List tests."""
        test_path = Path(args.path)
        tests = []

        if test_path.exists():
            if test_path.is_file():
                tests.append(str(test_path))
            else:
                import fnmatch
                for f in test_path.rglob("*.py"):
                    if fnmatch.fnmatch(f.name, args.pattern):
                        tests.append(str(f))

        if self.config.output_format != OutputFormat.JSON:
            print(self.formatter.header(f"Tests in {args.path}"))
            print(f"Found {len(tests)} test files\n")
            for test in sorted(tests)[:50]:
                print(f"  {test}")
            if len(tests) > 50:
                print(f"  ... and {len(tests) - 50} more")

        return CLIResult(
            success=True,
            message=f"Found {len(tests)} tests",
            data={"tests": tests, "count": len(tests)},
        )

    def _cmd_report(self, args) -> CLIResult:
        """Generate report."""
        from .report.generator import TestReportGenerator, ReportConfig, ReportData, ReportFormat, TestResultData

        try:
            with open(args.input) as f:
                input_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            return CLIResult(
                success=False,
                message=f"Error loading input: {e}",
                exit_code=1,
            )

        # Convert to ReportData
        test_results = []
        for result in input_data.get("results", input_data.get("test_results", [])):
            test_results.append(TestResultData(
                test_name=result.get("test_name", result.get("name", "")),
                status=result.get("status", "unknown"),
                duration_ms=result.get("duration_ms", result.get("duration_s", 0) * 1000),
                file_path=result.get("file_path"),
                error_message=result.get("error_message"),
            ))

        data = ReportData(
            run_id=input_data.get("run_id", str(uuid.uuid4())),
            timestamp=input_data.get("timestamp", time.time()),
            test_results=test_results,
        )

        config = ReportConfig(
            formats=[ReportFormat(f) for f in args.format],
            title=args.title,
            output_dir=self.config.output_dir + "/reports",
        )

        generator = TestReportGenerator()
        result = generator.generate(data, config)

        if self.config.output_format != OutputFormat.JSON:
            print(self.formatter.success("Reports generated:"))
            for path in result.output_files:
                print(f"  {path}")

        return CLIResult(
            success=True,
            message="Reports generated",
            data=result.to_dict(),
        )

    def _cmd_dashboard(self, args) -> CLIResult:
        """Show dashboard."""
        from .dashboard.dashboard import TestDashboard, DashboardConfig

        config = DashboardConfig(
            update_interval_ms=args.interval,
            output_dir=self.config.output_dir + "/dashboard",
        )

        dashboard = TestDashboard(config=config)

        if args.watch:
            print("Dashboard watch mode (Ctrl+C to exit)")
            try:
                while True:
                    print("\033[2J\033[H")  # Clear screen
                    print(dashboard.render_console())
                    time.sleep(args.interval / 1000)
            except KeyboardInterrupt:
                pass
        else:
            print(dashboard.render_console())

        return CLIResult(
            success=True,
            message="Dashboard displayed",
            data=dashboard.get_state(),
        )

    def _cmd_history(self, args) -> CLIResult:
        """Query history."""
        from .history.tracker import TestHistoryTracker, HistoryConfig, HistoryQuery

        config = HistoryConfig(
            db_path=self.config.output_dir + "/history/test_history.db",
        )

        tracker = TestHistoryTracker(config=config)

        try:
            if args.action == "stats":
                stats = tracker.get_stats(args.days)

                if self.config.output_format != OutputFormat.JSON:
                    print(self.formatter.header(f"Test Statistics (Last {args.days} days)"))
                    print(f"Total Runs: {stats.total_runs}")
                    print(f"Total Tests: {stats.total_tests}")
                    print(f"Pass Rate: {stats.pass_rate:.1f}%")
                    print(f"Flaky Tests: {stats.flaky_tests}")

                return CLIResult(
                    success=True,
                    message="Statistics retrieved",
                    data=stats.to_dict(),
                )

            elif args.action == "query":
                query = HistoryQuery(
                    test_name=args.test,
                    limit=args.limit,
                )
                records = tracker.query(query)

                if self.config.output_format != OutputFormat.JSON:
                    print(self.formatter.header(f"Query Results ({len(records)} records)"))
                    for record in records:
                        dt = datetime.fromtimestamp(record.timestamp)
                        print(f"  [{record.status}] {record.test_name} - {dt.strftime('%Y-%m-%d %H:%M')}")

                return CLIResult(
                    success=True,
                    message=f"Found {len(records)} records",
                    data={"records": [r.to_dict() for r in records]},
                )

            elif args.action == "runs":
                runs = tracker.get_run_history(args.limit)

                if self.config.output_format != OutputFormat.JSON:
                    print(self.formatter.header("Recent Test Runs"))
                    for run in runs:
                        dt = datetime.fromtimestamp(run["timestamp"])
                        print(f"  {run['run_id'][:8]}... - {run['passed']}/{run['total_tests']} - "
                              f"{dt.strftime('%Y-%m-%d %H:%M')}")

                return CLIResult(
                    success=True,
                    message=f"Found {len(runs)} runs",
                    data={"runs": runs},
                )

            else:
                return CLIResult(
                    success=False,
                    message=f"Unknown action: {args.action}",
                    exit_code=1,
                )
        finally:
            tracker.close()

    def _cmd_compare(self, args) -> CLIResult:
        """Compare test runs."""
        from .compare.comparator import TestComparator, CompareConfig

        config = CompareConfig(
            output_dir=self.config.output_dir + "/compare",
        )

        comparator = TestComparator(config=config)

        try:
            result = comparator.compare_from_files(args.base, args.compare)
        except (IOError, json.JSONDecodeError) as e:
            return CLIResult(
                success=False,
                message=f"Error comparing: {e}",
                exit_code=1,
            )

        # Generate report
        report_path = comparator.generate_report(result, args.format)

        if self.config.output_format != OutputFormat.JSON:
            status = "REGRESSION" if result.has_regressions else "OK"
            print(self.formatter.header(f"Comparison [{status}]"))
            print(f"Total Differences: {result.total_diffs}")
            print(f"Regressions: {result.regression_count}")
            print(f"Improvements: {result.improvement_count}")
            print(f"\nReport: {report_path}")

        return CLIResult(
            success=not result.has_regressions,
            message=f"Comparison complete: {result.total_diffs} differences",
            data=result.to_dict(),
            exit_code=1 if result.has_regressions else 0,
        )

    def _cmd_schedule(self, args) -> CLIResult:
        """Manage schedules."""
        from .schedule.scheduler import TestScheduler, ScheduleConfig, ScheduledJob, ScheduleFrequency

        config = ScheduleConfig(
            output_dir=self.config.output_dir + "/schedule",
            jobs_file=self.config.output_dir + "/schedule/jobs.json",
        )

        scheduler = TestScheduler(config=config)

        if args.action == "list":
            jobs = scheduler.list_jobs()

            if self.config.output_format != OutputFormat.JSON:
                print(self.formatter.header("Scheduled Jobs"))
                for job in jobs:
                    enabled = "[ON]" if job.enabled else "[OFF]"
                    next_run = ""
                    if job.next_run and job.next_run < float("inf"):
                        next_dt = datetime.fromtimestamp(job.next_run)
                        next_run = next_dt.strftime("%Y-%m-%d %H:%M")
                    print(f"  {enabled} {job.job_id[:8]}... {job.name} (Next: {next_run})")

            return CLIResult(
                success=True,
                message=f"Found {len(jobs)} jobs",
                data={"jobs": [j.to_dict() for j in jobs]},
            )

        elif args.action == "add":
            if not args.name:
                return CLIResult(success=False, message="--name required", exit_code=1)

            job = ScheduledJob(
                job_id=str(uuid.uuid4()),
                name=args.name,
                frequency=ScheduleFrequency(args.frequency),
                test_paths=args.tests,
            )
            job_id = scheduler.add_job(job)

            if self.config.output_format != OutputFormat.JSON:
                print(self.formatter.success(f"Added job: {job_id}"))

            return CLIResult(
                success=True,
                message=f"Job added: {job_id}",
                data=job.to_dict(),
            )

        elif args.action == "remove":
            if not args.job_id:
                return CLIResult(success=False, message="--job-id required", exit_code=1)

            if scheduler.remove_job(args.job_id):
                return CLIResult(success=True, message=f"Job removed: {args.job_id}")
            return CLIResult(success=False, message="Job not found", exit_code=1)

        elif args.action == "run":
            if not args.job_id:
                return CLIResult(success=False, message="--job-id required", exit_code=1)

            result = scheduler.run_job(args.job_id)
            if result:
                return CLIResult(
                    success=result.status.value == "completed",
                    message=f"Job executed: {result.status.value}",
                    data=result.to_dict(),
                )
            return CLIResult(success=False, message="Job not found", exit_code=1)

        return CLIResult(success=False, message=f"Unknown action: {args.action}", exit_code=1)

    def _cmd_cache(self, args) -> CLIResult:
        """Manage cache."""
        from .cache.cache import TestCache, CacheConfig

        config = CacheConfig(
            cache_dir=self.config.output_dir + "/cache",
        )

        cache = TestCache(config=config)

        if args.action == "stats":
            stats = cache.get_stats()

            if self.config.output_format != OutputFormat.JSON:
                print(self.formatter.header("Cache Statistics"))
                print(f"Total Entries: {stats.total_entries}")
                print(f"Hits: {stats.hits}")
                print(f"Misses: {stats.misses}")
                print(f"Hit Rate: {stats.hit_rate:.1f}%")

            return CLIResult(
                success=True,
                message="Cache stats retrieved",
                data=stats.to_dict(),
            )

        elif args.action == "clear":
            count = cache.clear()

            if self.config.output_format != OutputFormat.JSON:
                print(self.formatter.success(f"Cleared {count} entries"))

            return CLIResult(
                success=True,
                message=f"Cleared {count} entries",
                data={"cleared": count},
            )

        elif args.action == "lookup":
            if not args.test:
                return CLIResult(success=False, message="--test required", exit_code=1)

            result = cache.get(args.test)

            if self.config.output_format != OutputFormat.JSON:
                print(f"Status: {result.status.value}")
                if result.entry:
                    print(f"Test Status: {result.entry.status}")
                    print(f"Hits: {result.entry.hits}")

            return CLIResult(
                success=True,
                message=f"Cache lookup: {result.status.value}",
                data=result.to_dict(),
            )

        elif args.action == "list":
            entries = sorted(
                cache._cache.values(),
                key=lambda e: e.created_at,
                reverse=True,
            )[:args.limit]

            if self.config.output_format != OutputFormat.JSON:
                print(self.formatter.header(f"Cache Entries ({len(entries)})"))
                for entry in entries:
                    print(f"  [{entry.status}] {entry.test_name} ({entry.hits} hits)")

            return CLIResult(
                success=True,
                message=f"Listed {len(entries)} entries",
                data={"entries": [e.to_dict() for e in entries]},
            )

        return CLIResult(success=False, message=f"Unknown action: {args.action}", exit_code=1)

    def _cmd_notify(self, args) -> CLIResult:
        """Send notifications."""
        from .notify.notifier import TestNotifier, NotifyConfig, NotificationChannel, AlertSeverity

        config = NotifyConfig(
            output_dir=self.config.output_dir + "/notifications",
        )

        notifier = TestNotifier(config=config)

        result = notifier.notify(
            title=args.title,
            message=args.message,
            severity=AlertSeverity(args.severity),
            channels=[NotificationChannel(args.channel)],
        )

        return CLIResult(
            success=result.sent > 0,
            message=f"Sent {result.sent}/{result.total_notifications} notifications",
            data=result.to_dict(),
        )

    def _cmd_coverage(self, args) -> CLIResult:
        """Coverage operations."""
        return CLIResult(
            success=True,
            message="Coverage command (not fully implemented)",
            data={"action": args.action},
        )

    def _cmd_flaky(self, args) -> CLIResult:
        """Flaky test operations."""
        from .flaky.detector import FlakyDetector, FlakyConfig

        config = FlakyConfig(
            retry_count=args.retries,
            history_dir=self.config.output_dir + "/flaky",
            output_dir=self.config.output_dir + "/flaky/reports",
        )

        detector = FlakyDetector(config=config)

        if args.action == "list":
            quarantined = detector.get_quarantined_tests()

            if self.config.output_format != OutputFormat.JSON:
                print(self.formatter.header("Quarantined Tests"))
                for test in quarantined:
                    print(f"  - {test}")

            return CLIResult(
                success=True,
                message=f"Found {len(quarantined)} quarantined tests",
                data={"quarantined": quarantined},
            )

        return CLIResult(
            success=True,
            message=f"Flaky command: {args.action}",
            data={"action": args.action},
        )

    def _cmd_parallel(self, args) -> CLIResult:
        """Parallel execution."""
        from .parallel.parallelizer import TestParallelizer, ParallelConfig, PartitionStrategy

        config = ParallelConfig(
            workers=args.workers or 4,
            strategy=PartitionStrategy(args.strategy),
            output_dir=self.config.output_dir + "/parallel",
        )

        parallelizer = TestParallelizer(config=config)
        result = parallelizer.run(args.tests)

        if self.config.output_format != OutputFormat.JSON:
            status = "PASS" if result.failed == 0 else "FAIL"
            print(self.formatter.header(f"Parallel Execution [{status}]"))
            print(f"Duration: {result.duration_s:.2f}s")
            print(f"Speedup: {result.speedup:.2f}x")
            print(f"Passed: {result.passed}/{result.total_tests}")

        return CLIResult(
            success=result.failed == 0,
            message=f"Parallel execution complete",
            data=result.to_dict(),
            exit_code=0 if result.failed == 0 else 1,
        )

    def _cmd_api(self, args) -> CLIResult:
        """Start API server."""
        from .api.server import TestAPI, APIConfig

        config = APIConfig(
            host=args.host,
            port=args.port,
            output_dir=self.config.output_dir + "/api",
        )

        api = TestAPI(config=config)

        print(f"Starting API server at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")

        try:
            api.start()
        except KeyboardInterrupt:
            api.stop()

        return CLIResult(
            success=True,
            message="API server stopped",
        )

    def _cmd_status(self, args) -> CLIResult:
        """Show agent status."""
        status = {
            "version": VERSION,
            "output_dir": self.config.output_dir,
            "components": {
                "runner": "available",
                "report": "available",
                "dashboard": "available",
                "history": "available",
                "compare": "available",
                "schedule": "available",
                "cache": "available",
                "notify": "available",
                "parallel": "available",
                "api": "available",
            },
        }

        if self.config.output_format != OutputFormat.JSON:
            print(self.formatter.header("Test Agent Status"))
            print(f"Version: {VERSION}")
            print(f"Output Directory: {self.config.output_dir}")
            print("\nComponents:")
            for name, state in status["components"].items():
                print(f"  [{state.upper()}] {name}")

        return CLIResult(
            success=True,
            message="Status retrieved",
            data=status,
        )

    def _cmd_config(self, args) -> CLIResult:
        """Show configuration."""
        if self.config.output_format != OutputFormat.JSON:
            print(self.formatter.header("Configuration"))
            for key, value in self.config.to_dict().items():
                print(f"  {key}: {value}")

        return CLIResult(
            success=True,
            message="Configuration displayed",
            data=self.config.to_dict(),
        )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.cli.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "cli",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# Main Entry Point
# ============================================================================

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    cli = TestCLI()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
