#!/usr/bin/env python3
"""
Step 107: Pytest Integration

Pytest test runner with result parsing and coverage integration.

PBTSO Phase: TEST
Bus Topics:
- test.pytest.run (subscribes)
- test.pytest.result (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree


# Import from orchestrator for type compatibility
try:
    from .orchestrator import TestResult, TestStatus
except ImportError:
    # Fallback for standalone usage
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
# Pytest Result Parser
# ============================================================================

class PytestResultParser:
    """Parses pytest output in various formats."""

    def parse_junit_xml(self, xml_path: Path) -> List[TestResult]:
        """Parse JUnit XML output from pytest."""
        results = []

        if not xml_path.exists():
            return results

        try:
            tree = ElementTree.parse(xml_path)
            root = tree.getroot()

            for testsuite in root.iter("testsuite"):
                for testcase in testsuite.iter("testcase"):
                    result = self._parse_testcase(testcase)
                    results.append(result)

        except ElementTree.ParseError:
            pass

        return results

    def _parse_testcase(self, testcase: ElementTree.Element) -> TestResult:
        """Parse a single testcase element."""
        name = testcase.get("name", "unknown")
        classname = testcase.get("classname", "")
        file_path = testcase.get("file")
        line = testcase.get("line")
        time_s = float(testcase.get("time", 0))

        # Determine status from child elements
        status = TestStatus.PASSED
        error_message = None
        stack_trace = None
        output = ""

        # Check for failure
        failure = testcase.find("failure")
        if failure is not None:
            status = TestStatus.FAILED
            error_message = failure.get("message", "")
            stack_trace = failure.text

        # Check for error
        error = testcase.find("error")
        if error is not None:
            status = TestStatus.ERROR
            error_message = error.get("message", "")
            stack_trace = error.text

        # Check for skip
        skipped = testcase.find("skipped")
        if skipped is not None:
            status = TestStatus.SKIPPED
            error_message = skipped.get("message", "")

        # Get output
        system_out = testcase.find("system-out")
        if system_out is not None and system_out.text:
            output = system_out.text

        full_name = f"{classname}::{name}" if classname else name

        return TestResult(
            test_id=str(uuid.uuid4()),
            test_name=full_name,
            status=status,
            duration_s=time_s,
            output=output,
            error_message=error_message,
            stack_trace=stack_trace,
            file_path=file_path,
            line_number=int(line) if line else None,
        )

    def parse_json_output(self, json_path: Path) -> List[TestResult]:
        """Parse pytest JSON output (pytest-json-report)."""
        results = []

        if not json_path.exists():
            return results

        try:
            with open(json_path) as f:
                data = json.load(f)

            for test in data.get("tests", []):
                status_map = {
                    "passed": TestStatus.PASSED,
                    "failed": TestStatus.FAILED,
                    "skipped": TestStatus.SKIPPED,
                    "error": TestStatus.ERROR,
                }

                outcome = test.get("outcome", "unknown")
                status = status_map.get(outcome, TestStatus.ERROR)

                call = test.get("call", {})
                error_message = None
                stack_trace = None

                if "longrepr" in call:
                    stack_trace = call["longrepr"]
                if "crash" in call:
                    error_message = call["crash"].get("message")

                result = TestResult(
                    test_id=test.get("nodeid", str(uuid.uuid4())),
                    test_name=test.get("nodeid", "unknown"),
                    status=status,
                    duration_s=test.get("duration", 0),
                    output=test.get("stdout", ""),
                    error_message=error_message,
                    stack_trace=stack_trace,
                    file_path=test.get("location", [None])[0],
                    line_number=test.get("location", [None, None])[1],
                )
                results.append(result)

        except (json.JSONDecodeError, KeyError):
            pass

        return results

    def parse_console_output(self, output: str) -> List[TestResult]:
        """Parse pytest console output (fallback)."""
        results = []

        # Pattern for test results: test_file.py::test_name PASSED/FAILED/SKIPPED
        pattern = r"(\S+\.py)::([\w_]+)\s+(PASSED|FAILED|SKIPPED|ERROR)"

        for match in re.finditer(pattern, output):
            file_path, test_name, outcome = match.groups()

            status_map = {
                "PASSED": TestStatus.PASSED,
                "FAILED": TestStatus.FAILED,
                "SKIPPED": TestStatus.SKIPPED,
                "ERROR": TestStatus.ERROR,
            }

            result = TestResult(
                test_id=str(uuid.uuid4()),
                test_name=f"{file_path}::{test_name}",
                status=status_map.get(outcome, TestStatus.ERROR),
                duration_s=0,  # Not available from console output
                file_path=file_path,
            )
            results.append(result)

        return results


# ============================================================================
# Pytest Runner
# ============================================================================

class PytestRunner:
    """
    Pytest test runner with comprehensive result handling.

    PBTSO Phase: TEST
    Bus Topics: test.pytest.run, test.pytest.result
    """

    BUS_TOPICS = {
        "run": "test.pytest.run",
        "result": "test.pytest.result",
    }

    def __init__(self, bus=None):
        """
        Initialize the pytest runner.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus
        self._parser = PytestResultParser()

    def run(
        self,
        test_path: str,
        timeout_s: int = 300,
        verbose: bool = False,
        markers: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        coverage: bool = False,
        collect_only: bool = False,
    ) -> List[TestResult]:
        """
        Run pytest on a test path.

        Args:
            test_path: Path to test file or directory
            timeout_s: Timeout in seconds
            verbose: Enable verbose output
            markers: Pytest markers to filter tests
            env: Environment variables
            working_dir: Working directory
            coverage: Collect coverage data
            collect_only: Only collect tests, don't run

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

        # Build pytest command
        cmd = self._build_command(
            test_path=test_path,
            verbose=verbose,
            markers=markers,
            coverage=coverage,
            collect_only=collect_only,
        )

        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Create temp file for JUnit XML output
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_path = Path(f.name)

        cmd.extend(["--junit-xml", str(xml_path)])

        results = []
        start_time = time.time()

        try:
            # Run pytest
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=working_dir,
                env=run_env,
            )

            # Parse results
            results = self._parser.parse_junit_xml(xml_path)

            # Fallback to console parsing if no XML results
            if not results:
                results = self._parser.parse_console_output(process.stdout)

            # Add stdout/stderr to results
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
            if xml_path.exists():
                xml_path.unlink()

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
        verbose: bool = False,
        markers: Optional[List[str]] = None,
        coverage: bool = False,
        collect_only: bool = False,
    ) -> List[str]:
        """Build the pytest command."""
        cmd = ["python", "-m", "pytest"]

        # Add test path
        cmd.append(test_path)

        # Verbosity
        if verbose:
            cmd.append("-v")

        # Markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        # Coverage
        if coverage:
            cmd.extend(["--cov", "--cov-report=xml"])

        # Collect only (no execution)
        if collect_only:
            cmd.append("--collect-only")

        # Additional useful options
        cmd.extend([
            "--tb=short",  # Short tracebacks
            "-q",  # Quieter output (less noise)
        ])

        return cmd

    def collect_tests(self, test_path: str) -> List[str]:
        """
        Collect test names without running them.

        Args:
            test_path: Path to test file or directory

        Returns:
            List of test node IDs
        """
        cmd = ["python", "-m", "pytest", test_path, "--collect-only", "-q"]

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse collected tests
            tests = []
            for line in process.stdout.split("\n"):
                line = line.strip()
                if "::" in line and not line.startswith("<"):
                    tests.append(line)

            return tests

        except Exception:
            return []

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        if self.bus:
            topic = self.BUS_TOPICS.get(event_type, f"test.pytest.{event_type}")
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
    """CLI entry point for Pytest Runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Pytest Runner")
    parser.add_argument("path", nargs="?", default=".", help="Test path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-m", "--marker", action="append", help="Pytest markers")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--coverage", action="store_true", help="Collect coverage")
    parser.add_argument("--collect-only", action="store_true", help="Only collect tests")

    args = parser.parse_args()

    runner = PytestRunner()

    if args.collect_only:
        tests = runner.collect_tests(args.path)
        print(f"Collected {len(tests)} tests:")
        for test in tests:
            print(f"  {test}")
    else:
        results = runner.run(
            test_path=args.path,
            timeout_s=args.timeout,
            verbose=args.verbose,
            markers=args.marker,
            coverage=args.coverage,
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
                print(f"         Error: {result.error_message}")


if __name__ == "__main__":
    main()
