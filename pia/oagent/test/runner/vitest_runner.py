#!/usr/bin/env python3
"""
Step 108: Vitest Integration

Vitest test runner for JavaScript/TypeScript tests.

PBTSO Phase: TEST
Bus Topics:
- test.vitest.run (subscribes)
- test.vitest.result (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Import from orchestrator for type compatibility
try:
    from .orchestrator import TestResult, TestStatus
except ImportError:
    from enum import Enum

    class TestStatus(Enum):
        PENDING = "pending"
        RUNNING = "running"
        PASSED = "passed"
        FAILED = "failed"
        SKIPPED = "skipped"
        ERROR = "error"
        TIMEOUT = "timeout"

    @dataclass
    class TestResult:
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


# ============================================================================
# Vitest Result Parser
# ============================================================================

class VitestResultParser:
    """Parses Vitest output in various formats."""

    def parse_json_output(self, json_path: Path) -> List[TestResult]:
        """Parse Vitest JSON reporter output."""
        results = []

        if not json_path.exists():
            return results

        try:
            with open(json_path) as f:
                data = json.load(f)

            for test_file in data.get("testResults", []):
                file_path = test_file.get("name", "")

                for assertion in test_file.get("assertionResults", []):
                    status_map = {
                        "passed": TestStatus.PASSED,
                        "failed": TestStatus.FAILED,
                        "skipped": TestStatus.SKIPPED,
                        "pending": TestStatus.SKIPPED,
                        "todo": TestStatus.SKIPPED,
                    }

                    outcome = assertion.get("status", "unknown")
                    status = status_map.get(outcome, TestStatus.ERROR)

                    # Build full test name from ancestor titles
                    ancestors = assertion.get("ancestorTitles", [])
                    title = assertion.get("title", "unknown")
                    full_name = " > ".join(ancestors + [title])

                    error_message = None
                    stack_trace = None

                    failure_messages = assertion.get("failureMessages", [])
                    if failure_messages:
                        error_message = failure_messages[0]
                        if len(failure_messages) > 1:
                            stack_trace = "\n".join(failure_messages[1:])

                    result = TestResult(
                        test_id=str(uuid.uuid4()),
                        test_name=full_name,
                        status=status,
                        duration_s=assertion.get("duration", 0) / 1000,  # ms to s
                        output="",
                        error_message=error_message,
                        stack_trace=stack_trace,
                        file_path=file_path,
                    )
                    results.append(result)

        except (json.JSONDecodeError, KeyError) as e:
            pass

        return results

    def parse_tap_output(self, output: str) -> List[TestResult]:
        """Parse TAP (Test Anything Protocol) output."""
        results = []

        for line in output.split("\n"):
            line = line.strip()

            # TAP format: ok 1 - test name
            #             not ok 2 - failing test
            if line.startswith("ok "):
                match = line[3:].split(" - ", 1)
                if len(match) == 2:
                    test_name = match[1]
                    results.append(TestResult(
                        test_id=str(uuid.uuid4()),
                        test_name=test_name,
                        status=TestStatus.PASSED,
                        duration_s=0,
                    ))

            elif line.startswith("not ok "):
                match = line[7:].split(" - ", 1)
                if len(match) == 2:
                    test_name = match[1]
                    results.append(TestResult(
                        test_id=str(uuid.uuid4()),
                        test_name=test_name,
                        status=TestStatus.FAILED,
                        duration_s=0,
                    ))

        return results

    def parse_console_output(self, output: str) -> List[TestResult]:
        """Parse Vitest console output (fallback)."""
        results = []

        # Pattern for Vitest output: checkmark/cross + test name
        import re

        # Match lines like: ✓ test name (10ms)
        # or: × test name
        passed_pattern = r"[✓✔]\s+(.+?)(?:\s+\(\d+\s*m?s\))?$"
        failed_pattern = r"[×✗]\s+(.+?)(?:\s+\(\d+\s*m?s\))?$"
        skipped_pattern = r"[○-]\s+(.+?)(?:\s+\(\d+\s*m?s\))?$"

        for line in output.split("\n"):
            line = line.strip()

            match = re.match(passed_pattern, line)
            if match:
                results.append(TestResult(
                    test_id=str(uuid.uuid4()),
                    test_name=match.group(1),
                    status=TestStatus.PASSED,
                    duration_s=0,
                ))
                continue

            match = re.match(failed_pattern, line)
            if match:
                results.append(TestResult(
                    test_id=str(uuid.uuid4()),
                    test_name=match.group(1),
                    status=TestStatus.FAILED,
                    duration_s=0,
                ))
                continue

            match = re.match(skipped_pattern, line)
            if match:
                results.append(TestResult(
                    test_id=str(uuid.uuid4()),
                    test_name=match.group(1),
                    status=TestStatus.SKIPPED,
                    duration_s=0,
                ))

        return results


# ============================================================================
# Vitest Runner
# ============================================================================

class VitestRunner:
    """
    Vitest test runner for JavaScript/TypeScript tests.

    PBTSO Phase: TEST
    Bus Topics: test.vitest.run, test.vitest.result
    """

    BUS_TOPICS = {
        "run": "test.vitest.run",
        "result": "test.vitest.result",
    }

    def __init__(self, bus=None):
        """
        Initialize the Vitest runner.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus
        self._parser = VitestResultParser()

    def run(
        self,
        test_path: str,
        timeout_s: int = 300,
        verbose: bool = False,
        markers: Optional[List[str]] = None,  # For API compatibility (unused)
        env: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        coverage: bool = False,
        watch: bool = False,
        ui: bool = False,
    ) -> List[TestResult]:
        """
        Run Vitest on a test path.

        Args:
            test_path: Path to test file or directory
            timeout_s: Timeout in seconds
            verbose: Enable verbose output
            markers: Unused (for API compatibility)
            env: Environment variables
            working_dir: Working directory
            coverage: Collect coverage data
            watch: Run in watch mode
            ui: Run with UI

        Returns:
            List of test results
        """
        run_id = str(uuid.uuid4())

        # Emit run start event
        self._emit_event("run", {
            "run_id": run_id,
            "test_path": test_path,
            "status": "started",
        })

        # Create temp file for JSON output
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json_path = Path(f.name)

        # Build vitest command
        cmd = self._build_command(
            test_path=test_path,
            json_path=json_path,
            verbose=verbose,
            coverage=coverage,
            watch=watch,
            ui=ui,
        )

        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        results = []
        start_time = time.time()

        try:
            # Determine package manager
            pkg_manager = self._detect_package_manager(working_dir)

            if pkg_manager == "pnpm":
                cmd = ["pnpm", "exec"] + cmd
            elif pkg_manager == "yarn":
                cmd = ["yarn"] + cmd
            else:
                cmd = ["npx"] + cmd

            # Run vitest
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=working_dir,
                env=run_env,
            )

            # Parse results from JSON output
            results = self._parser.parse_json_output(json_path)

            # Fallback to console parsing
            if not results:
                results = self._parser.parse_console_output(process.stdout)

            # Add stdout to results
            for result in results:
                if not result.output:
                    result.output = process.stdout

        except subprocess.TimeoutExpired:
            results.append(TestResult(
                test_id=str(uuid.uuid4()),
                test_name=f"Timeout: {test_path}",
                status=TestStatus.TIMEOUT,
                duration_s=timeout_s,
                error_message=f"Test execution timed out after {timeout_s}s",
                file_path=test_path,
            ))

        except FileNotFoundError:
            results.append(TestResult(
                test_id=str(uuid.uuid4()),
                test_name=f"Error: {test_path}",
                status=TestStatus.ERROR,
                duration_s=time.time() - start_time,
                error_message="Vitest not found. Ensure vitest is installed.",
                file_path=test_path,
            ))

        except Exception as e:
            results.append(TestResult(
                test_id=str(uuid.uuid4()),
                test_name=f"Error: {test_path}",
                status=TestStatus.ERROR,
                duration_s=time.time() - start_time,
                error_message=str(e),
                file_path=test_path,
            ))

        finally:
            # Cleanup temp file
            if json_path.exists():
                json_path.unlink()

        # Emit results
        for result in results:
            self._emit_event("result", {
                "run_id": run_id,
                "test_id": result.test_id,
                "test_name": result.test_name,
                "status": result.status.value,
                "duration_s": result.duration_s,
            })

        return results

    def _build_command(
        self,
        test_path: str,
        json_path: Path,
        verbose: bool = False,
        coverage: bool = False,
        watch: bool = False,
        ui: bool = False,
    ) -> List[str]:
        """Build the vitest command."""
        cmd = ["vitest", "run"]  # "run" for non-watch mode

        # Add test path if specific
        if test_path and test_path != ".":
            cmd.append(test_path)

        # JSON reporter
        cmd.extend(["--reporter", "json", "--outputFile", str(json_path)])

        # Also add default reporter for console
        cmd.extend(["--reporter", "default"])

        # Coverage
        if coverage:
            cmd.append("--coverage")

        # Watch mode
        if watch:
            cmd[1] = "watch"  # Replace "run" with "watch"

        # UI mode
        if ui:
            cmd.append("--ui")

        # Disable colors for easier parsing
        cmd.append("--no-color")

        return cmd

    def _detect_package_manager(self, working_dir: Optional[str]) -> str:
        """Detect the package manager in use."""
        cwd = Path(working_dir) if working_dir else Path.cwd()

        if (cwd / "pnpm-lock.yaml").exists():
            return "pnpm"
        elif (cwd / "yarn.lock").exists():
            return "yarn"
        else:
            return "npm"

    def list_tests(self, test_path: str, working_dir: Optional[str] = None) -> List[str]:
        """
        List available tests without running them.

        Args:
            test_path: Path to test file or directory
            working_dir: Working directory

        Returns:
            List of test names
        """
        cmd = ["vitest", "list", "--reporter", "json"]

        if test_path and test_path != ".":
            cmd.append(test_path)

        pkg_manager = self._detect_package_manager(working_dir)

        if pkg_manager == "pnpm":
            cmd = ["pnpm", "exec"] + cmd
        elif pkg_manager == "yarn":
            cmd = ["yarn"] + cmd
        else:
            cmd = ["npx"] + cmd

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=working_dir,
            )

            # Parse test list
            tests = []
            try:
                data = json.loads(process.stdout)
                for file_data in data:
                    file_name = file_data.get("file", "")
                    for test in file_data.get("tests", []):
                        test_name = test.get("name", "")
                        tests.append(f"{file_name} > {test_name}")
            except json.JSONDecodeError:
                pass

            return tests

        except Exception:
            return []

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        if self.bus:
            topic = self.BUS_TOPICS.get(event_type, f"test.vitest.{event_type}")
            self.bus.emit({
                "topic": topic,
                "kind": "test_execution",
                "actor": "test-agent",
                "data": data,
            })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Vitest Runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Vitest Runner")
    parser.add_argument("path", nargs="?", default=".", help="Test path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--coverage", action="store_true", help="Collect coverage")
    parser.add_argument("--watch", action="store_true", help="Watch mode")
    parser.add_argument("--list", action="store_true", help="List tests only")
    parser.add_argument("--cwd", help="Working directory")

    args = parser.parse_args()

    runner = VitestRunner()

    if args.list:
        tests = runner.list_tests(args.path, working_dir=args.cwd)
        print(f"Found {len(tests)} tests:")
        for test in tests:
            print(f"  {test}")
    else:
        results = runner.run(
            test_path=args.path,
            timeout_s=args.timeout,
            verbose=args.verbose,
            coverage=args.coverage,
            watch=args.watch,
            working_dir=args.cwd,
        )

        print(f"\nResults ({len(results)} tests):")
        for result in results:
            status_icon = {
                TestStatus.PASSED: "[PASS]",
                TestStatus.FAILED: "[FAIL]",
                TestStatus.SKIPPED: "[SKIP]",
                TestStatus.ERROR: "[ERR]",
                TestStatus.TIMEOUT: "[TIME]",
            }.get(result.status, "[???]")

            print(f"  {status_icon} {result.test_name} ({result.duration_s:.2f}s)")
            if result.error_message:
                print(f"         Error: {result.error_message[:100]}...")


if __name__ == "__main__":
    main()
