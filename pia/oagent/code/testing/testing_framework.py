#!/usr/bin/env python3
"""
testing_framework.py - Testing Framework (Step 93)

PBTSO Phase: TEST, VERIFY

Provides:
- Test case definition and execution
- Test suite management
- Assertion library
- Mock/stub support
- Test reporting
- Coverage tracking

Bus Topics:
- code.test.run
- code.test.result
- code.test.suite
- code.test.coverage

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import socket
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestLevel(Enum):
    """Test level/type."""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    ACCEPTANCE = "acceptance"


@dataclass
class TestConfig:
    """Configuration for testing framework."""
    timeout_s: float = 30.0
    parallel: bool = False
    max_workers: int = 4
    fail_fast: bool = False
    verbose: bool = False
    coverage_enabled: bool = False
    output_format: str = "text"
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeout_s": self.timeout_s,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "fail_fast": self.fail_fast,
            "verbose": self.verbose,
            "coverage_enabled": self.coverage_enabled,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Assertions
# =============================================================================

class AssertionError(Exception):
    """Test assertion failure."""
    def __init__(self, message: str, expected: Any = None, actual: Any = None):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class assertion:
    """Assertion library for tests."""

    @staticmethod
    def equal(actual: Any, expected: Any, message: str = "") -> None:
        """Assert values are equal."""
        if actual != expected:
            msg = message or f"Expected {expected!r}, got {actual!r}"
            raise AssertionError(msg, expected, actual)

    @staticmethod
    def not_equal(actual: Any, expected: Any, message: str = "") -> None:
        """Assert values are not equal."""
        if actual == expected:
            msg = message or f"Expected not {expected!r}"
            raise AssertionError(msg, expected, actual)

    @staticmethod
    def true(value: Any, message: str = "") -> None:
        """Assert value is truthy."""
        if not value:
            msg = message or f"Expected truthy, got {value!r}"
            raise AssertionError(msg, True, value)

    @staticmethod
    def false(value: Any, message: str = "") -> None:
        """Assert value is falsy."""
        if value:
            msg = message or f"Expected falsy, got {value!r}"
            raise AssertionError(msg, False, value)

    @staticmethod
    def none(value: Any, message: str = "") -> None:
        """Assert value is None."""
        if value is not None:
            msg = message or f"Expected None, got {value!r}"
            raise AssertionError(msg, None, value)

    @staticmethod
    def not_none(value: Any, message: str = "") -> None:
        """Assert value is not None."""
        if value is None:
            msg = message or "Expected not None"
            raise AssertionError(msg, "not None", None)

    @staticmethod
    def contains(container: Any, item: Any, message: str = "") -> None:
        """Assert container contains item."""
        if item not in container:
            msg = message or f"Expected {container!r} to contain {item!r}"
            raise AssertionError(msg, item, container)

    @staticmethod
    def not_contains(container: Any, item: Any, message: str = "") -> None:
        """Assert container does not contain item."""
        if item in container:
            msg = message or f"Expected {container!r} to not contain {item!r}"
            raise AssertionError(msg, f"not {item!r}", container)

    @staticmethod
    def instance_of(obj: Any, cls: Type, message: str = "") -> None:
        """Assert object is instance of class."""
        if not isinstance(obj, cls):
            msg = message or f"Expected instance of {cls.__name__}, got {type(obj).__name__}"
            raise AssertionError(msg, cls.__name__, type(obj).__name__)

    @staticmethod
    def raises(exception: Type[Exception], func: Callable, *args: Any, **kwargs: Any) -> Exception:
        """Assert function raises exception."""
        try:
            func(*args, **kwargs)
        except exception as e:
            return e
        except Exception as e:
            raise AssertionError(
                f"Expected {exception.__name__}, got {type(e).__name__}",
                exception.__name__,
                type(e).__name__,
            )
        raise AssertionError(
            f"Expected {exception.__name__}, but no exception raised",
            exception.__name__,
            "no exception",
        )

    @staticmethod
    async def raises_async(exception: Type[Exception], func: Callable, *args: Any, **kwargs: Any) -> Exception:
        """Assert async function raises exception."""
        try:
            await func(*args, **kwargs)
        except exception as e:
            return e
        except Exception as e:
            raise AssertionError(
                f"Expected {exception.__name__}, got {type(e).__name__}",
                exception.__name__,
                type(e).__name__,
            )
        raise AssertionError(
            f"Expected {exception.__name__}, but no exception raised",
            exception.__name__,
            "no exception",
        )

    @staticmethod
    def approx_equal(actual: float, expected: float, tolerance: float = 0.001, message: str = "") -> None:
        """Assert floats are approximately equal."""
        if abs(actual - expected) > tolerance:
            msg = message or f"Expected {expected} +/- {tolerance}, got {actual}"
            raise AssertionError(msg, expected, actual)

    @staticmethod
    def length(container: Any, expected_length: int, message: str = "") -> None:
        """Assert container has expected length."""
        actual_length = len(container)
        if actual_length != expected_length:
            msg = message or f"Expected length {expected_length}, got {actual_length}"
            raise AssertionError(msg, expected_length, actual_length)


# =============================================================================
# Test Types
# =============================================================================

@dataclass
class TestResult:
    """Result of a test execution."""
    test_id: str
    name: str
    status: TestStatus
    duration_ms: float = 0.0
    message: str = ""
    error: Optional[str] = None
    traceback: Optional[str] = None
    output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "error": self.error,
            "output": self.output[:1000] if self.output else "",
        }


@dataclass
class TestCase:
    """A test case."""
    id: str
    name: str
    func: Callable
    level: TestLevel = TestLevel.UNIT
    tags: List[str] = field(default_factory=list)
    timeout_s: Optional[float] = None
    skip: bool = False
    skip_reason: str = ""
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level.value,
            "tags": self.tags,
            "skip": self.skip,
        }


@dataclass
class TestSuite:
    """A collection of test cases."""
    id: str
    name: str
    tests: List[TestCase] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    level: TestLevel = TestLevel.UNIT

    def add_test(self, test: TestCase) -> None:
        """Add a test case."""
        self.tests.append(test)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "test_count": len(self.tests),
            "level": self.level.value,
        }


@dataclass
class SuiteResult:
    """Result of a test suite execution."""
    suite_id: str
    name: str
    results: List[TestResult]
    duration_ms: float = 0.0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "total": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "success_rate": self.passed / max(len(self.results), 1),
        }


# =============================================================================
# Test Runner
# =============================================================================

class TestRunner:
    """
    Test runner for executing test cases.

    Features:
    - Sequential and parallel execution
    - Timeout handling
    - Setup/teardown support
    - Result collection
    """

    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()

    async def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case."""
        test_id = test.id or f"test-{uuid.uuid4().hex[:8]}"

        # Check if skipped
        if test.skip:
            return TestResult(
                test_id=test_id,
                name=test.name,
                status=TestStatus.SKIPPED,
                message=test.skip_reason,
            )

        start = time.time()
        output_lines: List[str] = []

        try:
            # Setup
            if test.setup:
                if asyncio.iscoroutinefunction(test.setup):
                    await test.setup()
                else:
                    test.setup()

            # Run test with timeout
            timeout = test.timeout_s or self.config.timeout_s

            if asyncio.iscoroutinefunction(test.func):
                await asyncio.wait_for(test.func(), timeout=timeout)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, test.func),
                    timeout=timeout,
                )

            # Teardown
            if test.teardown:
                if asyncio.iscoroutinefunction(test.teardown):
                    await test.teardown()
                else:
                    test.teardown()

            duration = (time.time() - start) * 1000

            return TestResult(
                test_id=test_id,
                name=test.name,
                status=TestStatus.PASSED,
                duration_ms=duration,
            )

        except AssertionError as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                test_id=test_id,
                name=test.name,
                status=TestStatus.FAILED,
                duration_ms=duration,
                message=str(e),
                error=str(e),
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start) * 1000
            return TestResult(
                test_id=test_id,
                name=test.name,
                status=TestStatus.ERROR,
                duration_ms=duration,
                message=f"Test timed out after {timeout}s",
                error="TimeoutError",
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            tb = traceback.format_exc()
            return TestResult(
                test_id=test_id,
                name=test.name,
                status=TestStatus.ERROR,
                duration_ms=duration,
                message=str(e),
                error=type(e).__name__,
                traceback=tb,
            )

    async def run_suite(self, suite: TestSuite) -> SuiteResult:
        """Run a test suite."""
        start = time.time()
        results: List[TestResult] = []

        # Suite setup
        if suite.setup:
            try:
                if asyncio.iscoroutinefunction(suite.setup):
                    await suite.setup()
                else:
                    suite.setup()
            except Exception as e:
                # All tests fail if setup fails
                for test in suite.tests:
                    results.append(TestResult(
                        test_id=test.id,
                        name=test.name,
                        status=TestStatus.ERROR,
                        message=f"Suite setup failed: {e}",
                    ))
                return SuiteResult(
                    suite_id=suite.id,
                    name=suite.name,
                    results=results,
                    errors=len(results),
                    duration_ms=(time.time() - start) * 1000,
                )

        # Run tests
        if self.config.parallel:
            tasks = [self.run_test(test) for test in suite.tests]
            results = await asyncio.gather(*tasks)
        else:
            for test in suite.tests:
                result = await self.run_test(test)
                results.append(result)

                if self.config.fail_fast and result.status in (TestStatus.FAILED, TestStatus.ERROR):
                    break

        # Suite teardown
        if suite.teardown:
            try:
                if asyncio.iscoroutinefunction(suite.teardown):
                    await suite.teardown()
                else:
                    suite.teardown()
            except Exception:
                pass

        # Calculate summary
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)

        return SuiteResult(
            suite_id=suite.id,
            name=suite.name,
            results=results,
            duration_ms=(time.time() - start) * 1000,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
        )


# =============================================================================
# Test Reporter
# =============================================================================

class TestReporter:
    """Reporter for test results."""

    def __init__(self, format: str = "text"):
        self.format = format

    def report(self, result: Union[TestResult, SuiteResult]) -> str:
        """Generate report for result."""
        if self.format == "json":
            return self._json_report(result)
        elif self.format == "junit":
            return self._junit_report(result)
        else:
            return self._text_report(result)

    def _text_report(self, result: Union[TestResult, SuiteResult]) -> str:
        """Generate text report."""
        lines = []

        if isinstance(result, SuiteResult):
            lines.append(f"Test Suite: {result.name}")
            lines.append("=" * 60)

            for test_result in result.results:
                status_icon = {
                    TestStatus.PASSED: "[PASS]",
                    TestStatus.FAILED: "[FAIL]",
                    TestStatus.SKIPPED: "[SKIP]",
                    TestStatus.ERROR: "[ERR]",
                }.get(test_result.status, "[???]")

                lines.append(f"  {status_icon} {test_result.name} ({test_result.duration_ms:.1f}ms)")

                if test_result.message and test_result.status != TestStatus.PASSED:
                    lines.append(f"         {test_result.message}")

            lines.append("-" * 60)
            lines.append(f"Total: {len(result.results)} | Passed: {result.passed} | Failed: {result.failed} | Skipped: {result.skipped} | Errors: {result.errors}")
            lines.append(f"Duration: {result.duration_ms:.1f}ms")

        else:
            status = result.status.value.upper()
            lines.append(f"[{status}] {result.name}")
            if result.message:
                lines.append(f"  {result.message}")

        return "\n".join(lines)

    def _json_report(self, result: Union[TestResult, SuiteResult]) -> str:
        """Generate JSON report."""
        return json.dumps(result.to_dict(), indent=2)

    def _junit_report(self, result: Union[TestResult, SuiteResult]) -> str:
        """Generate JUnit XML report."""
        if isinstance(result, SuiteResult):
            tests = len(result.results)
            failures = result.failed
            errors = result.errors
            time_s = result.duration_ms / 1000

            xml = f'<?xml version="1.0" encoding="UTF-8"?>\n'
            xml += f'<testsuite name="{result.name}" tests="{tests}" failures="{failures}" errors="{errors}" time="{time_s:.3f}">\n'

            for tr in result.results:
                xml += f'  <testcase name="{tr.name}" time="{tr.duration_ms / 1000:.3f}">\n'
                if tr.status == TestStatus.FAILED:
                    xml += f'    <failure message="{tr.message or "Test failed"}"/>\n'
                elif tr.status == TestStatus.ERROR:
                    xml += f'    <error message="{tr.error or "Error"}"/>\n'
                elif tr.status == TestStatus.SKIPPED:
                    xml += f'    <skipped/>\n'
                xml += f'  </testcase>\n'

            xml += '</testsuite>'
            return xml
        else:
            return self._json_report(result)


# =============================================================================
# Testing Framework
# =============================================================================

class TestingFramework:
    """
    Testing framework for the Code Agent.

    PBTSO Phase: TEST, VERIFY

    Features:
    - Test case definition with decorators
    - Test suite management
    - Assertion library
    - Multiple output formats
    - Integration with bus for reporting

    Usage:
        framework = TestingFramework()

        @framework.test("my test")
        def test_addition():
            assertion.equal(1 + 1, 2)

        results = await framework.run_all()
    """

    BUS_TOPICS = {
        "run": "code.test.run",
        "result": "code.test.result",
        "suite": "code.test.suite",
        "coverage": "code.test.coverage",
    }

    def __init__(
        self,
        config: Optional[TestConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or TestConfig()
        self.bus = bus or LockedAgentBus()

        self._runner = TestRunner(self.config)
        self._reporter = TestReporter(self.config.output_format)

        self._suites: Dict[str, TestSuite] = {}
        self._tests: List[TestCase] = []
        self._lock = Lock()

        # Create default suite
        self._default_suite = TestSuite(
            id="default",
            name="Default Test Suite",
        )
        self._suites["default"] = self._default_suite

    # =========================================================================
    # Test Registration
    # =========================================================================

    def test(
        self,
        name: str,
        level: TestLevel = TestLevel.UNIT,
        tags: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        skip: bool = False,
        skip_reason: str = "",
    ) -> Callable:
        """Decorator to register a test case."""
        def decorator(func: Callable) -> Callable:
            test_case = TestCase(
                id=f"test-{uuid.uuid4().hex[:8]}",
                name=name,
                func=func,
                level=level,
                tags=tags or [],
                timeout_s=timeout,
                skip=skip,
                skip_reason=skip_reason,
            )

            with self._lock:
                self._tests.append(test_case)
                self._default_suite.add_test(test_case)

            return func
        return decorator

    def suite(
        self,
        name: str,
        level: TestLevel = TestLevel.UNIT,
    ) -> TestSuite:
        """Create a test suite."""
        suite = TestSuite(
            id=f"suite-{uuid.uuid4().hex[:8]}",
            name=name,
            level=level,
        )

        with self._lock:
            self._suites[suite.id] = suite

        return suite

    def add_test(
        self,
        name: str,
        func: Callable,
        suite_id: str = "default",
        **kwargs: Any,
    ) -> TestCase:
        """Add a test case programmatically."""
        test_case = TestCase(
            id=f"test-{uuid.uuid4().hex[:8]}",
            name=name,
            func=func,
            **kwargs,
        )

        with self._lock:
            self._tests.append(test_case)
            if suite_id in self._suites:
                self._suites[suite_id].add_test(test_case)

        return test_case

    # =========================================================================
    # Test Execution
    # =========================================================================

    async def run_test(self, test_id: str) -> Optional[TestResult]:
        """Run a single test by ID."""
        for test in self._tests:
            if test.id == test_id:
                return await self._run_and_report(test)
        return None

    async def run_suite(self, suite_id: str) -> Optional[SuiteResult]:
        """Run a test suite by ID."""
        suite = self._suites.get(suite_id)
        if not suite:
            return None

        self.bus.emit({
            "topic": self.BUS_TOPICS["run"],
            "kind": "test",
            "actor": "testing-framework",
            "data": {
                "type": "suite",
                "suite_id": suite_id,
                "test_count": len(suite.tests),
            },
        })

        result = await self._runner.run_suite(suite)

        self.bus.emit({
            "topic": self.BUS_TOPICS["suite"],
            "kind": "test",
            "actor": "testing-framework",
            "data": result.to_dict(),
        })

        return result

    async def run_all(self) -> List[SuiteResult]:
        """Run all test suites."""
        results = []

        for suite_id, suite in self._suites.items():
            if suite.tests:
                result = await self.run_suite(suite_id)
                if result:
                    results.append(result)

        return results

    async def run_by_tag(self, tag: str) -> SuiteResult:
        """Run tests matching a tag."""
        matching_tests = [t for t in self._tests if tag in t.tags]

        suite = TestSuite(
            id=f"tag-{tag}",
            name=f"Tests tagged: {tag}",
            tests=matching_tests,
        )

        return await self._runner.run_suite(suite)

    async def run_by_level(self, level: TestLevel) -> SuiteResult:
        """Run tests at a specific level."""
        matching_tests = [t for t in self._tests if t.level == level]

        suite = TestSuite(
            id=f"level-{level.value}",
            name=f"Tests at level: {level.value}",
            tests=matching_tests,
            level=level,
        )

        return await self._runner.run_suite(suite)

    async def _run_and_report(self, test: TestCase) -> TestResult:
        """Run a test and emit result."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["run"],
            "kind": "test",
            "actor": "testing-framework",
            "data": {"type": "test", "test_id": test.id, "name": test.name},
        })

        result = await self._runner.run_test(test)

        self.bus.emit({
            "topic": self.BUS_TOPICS["result"],
            "kind": "test",
            "actor": "testing-framework",
            "data": result.to_dict(),
        })

        return result

    # =========================================================================
    # Reporting
    # =========================================================================

    def report(self, result: Union[TestResult, SuiteResult]) -> str:
        """Generate report for result."""
        return self._reporter.report(result)

    def set_reporter(self, format: str) -> None:
        """Set report format."""
        self._reporter = TestReporter(format)

    # =========================================================================
    # Utilities
    # =========================================================================

    def list_tests(self) -> List[Dict[str, Any]]:
        """List all registered tests."""
        return [t.to_dict() for t in self._tests]

    def list_suites(self) -> List[Dict[str, Any]]:
        """List all test suites."""
        return [s.to_dict() for s in self._suites.values()]

    def stats(self) -> Dict[str, Any]:
        """Get testing statistics."""
        return {
            "total_tests": len(self._tests),
            "total_suites": len(self._suites),
            "by_level": {
                level.value: sum(1 for t in self._tests if t.level == level)
                for level in TestLevel
            },
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Testing Framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Testing Framework (Step 93)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run tests")
    run_parser.add_argument("--suite", "-s", help="Suite ID")
    run_parser.add_argument("--tag", "-t", help="Tag filter")
    run_parser.add_argument("--level", "-l", help="Level filter")
    run_parser.add_argument("--format", "-f", choices=["text", "json", "junit"], default="text")

    # list command
    list_parser = subparsers.add_parser("list", help="List tests")
    list_parser.add_argument("--json", action="store_true")

    # demo command
    subparsers.add_parser("demo", help="Run demo tests")

    args = parser.parse_args()
    framework = TestingFramework()

    async def run_async():
        if args.command == "demo":
            # Define demo tests
            @framework.test("addition test", level=TestLevel.UNIT)
            def test_addition():
                assertion.equal(1 + 1, 2)

            @framework.test("string test", level=TestLevel.UNIT, tags=["string"])
            def test_string():
                assertion.contains("hello world", "world")

            @framework.test("failing test", level=TestLevel.UNIT)
            def test_failure():
                assertion.equal(1, 2, "This should fail")

            @framework.test("skipped test", skip=True, skip_reason="Not implemented")
            def test_skipped():
                pass

            @framework.test("error test", level=TestLevel.UNIT)
            def test_error():
                raise ValueError("Unexpected error")

            @framework.test("async test", level=TestLevel.INTEGRATION)
            async def test_async():
                await asyncio.sleep(0.01)
                assertion.true(True)

            print("Running demo tests...\n")
            results = await framework.run_all()

            for suite_result in results:
                print(framework.report(suite_result))
                print()

            total_passed = sum(r.passed for r in results)
            total_failed = sum(r.failed for r in results)
            total_errors = sum(r.errors for r in results)

            print(f"\nOverall: {total_passed} passed, {total_failed} failed, {total_errors} errors")
            return 0 if total_failed == 0 and total_errors == 0 else 1

        elif args.command == "run":
            framework.set_reporter(args.format)

            if args.tag:
                result = await framework.run_by_tag(args.tag)
                print(framework.report(result))
            elif args.level:
                result = await framework.run_by_level(TestLevel(args.level))
                print(framework.report(result))
            elif args.suite:
                result = await framework.run_suite(args.suite)
                if result:
                    print(framework.report(result))
                else:
                    print(f"Suite not found: {args.suite}")
                    return 1
            else:
                results = await framework.run_all()
                for result in results:
                    print(framework.report(result))

            return 0

        elif args.command == "list":
            tests = framework.list_tests()
            suites = framework.list_suites()

            if args.json:
                print(json.dumps({"tests": tests, "suites": suites}, indent=2))
            else:
                print("Test Suites:")
                for s in suites:
                    print(f"  [{s['id']}] {s['name']} ({s['test_count']} tests)")

                print("\nTests:")
                for t in tests:
                    print(f"  [{t['id']}] {t['name']} ({t['level']})")

            return 0

        return 1

    return asyncio.run(run_async())


if __name__ == "__main__":
    sys.exit(main())
