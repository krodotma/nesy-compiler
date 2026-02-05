#!/usr/bin/env python3
"""
framework.py - Testing Framework (Step 243)

PBTSO Phase: VERIFY
A2A Integration: Provides testing capabilities via deploy.testing.run

Provides:
- TestStatus: Test execution status
- TestCase: Test case definition
- TestSuite: Test suite containing test cases
- TestResult: Test execution result
- TestRunner: Test execution engine
- TestReporter: Test result reporting
- TestingFramework: Complete testing framework

Bus Topics:
- deploy.testing.run
- deploy.testing.passed
- deploy.testing.failed
- deploy.testing.report

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union


# ==============================================================================
# Bus Emission Helper with fcntl.flock()
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "testing-framework"
) -> str:
    """Emit an event to the Pluribus bus with file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    SMOKE = "smoke"
    REGRESSION = "regression"
    PERFORMANCE = "performance"
    SECURITY = "security"


class ReportFormat(Enum):
    """Report output formats."""
    TEXT = "text"
    JSON = "json"
    JUNIT = "junit"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class TestCase:
    """
    Test case definition.

    Attributes:
        test_id: Unique test identifier
        name: Test name
        description: Test description
        test_type: Type of test
        test_fn: Test function/coroutine
        setup_fn: Setup function
        teardown_fn: Teardown function
        tags: Test tags
        timeout_s: Test timeout in seconds
        enabled: Whether test is enabled
        depends_on: Tests this depends on
        metadata: Additional metadata
    """
    test_id: str
    name: str
    description: str = ""
    test_type: TestType = TestType.UNIT
    test_fn: Optional[Callable] = None
    setup_fn: Optional[Callable] = None
    teardown_fn: Optional[Callable] = None
    tags: Set[str] = field(default_factory=set)
    timeout_s: int = 60
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "tags": list(self.tags),
            "timeout_s": self.timeout_s,
            "enabled": self.enabled,
            "depends_on": self.depends_on,
            "metadata": self.metadata,
        }


@dataclass
class TestResult:
    """
    Test execution result.

    Attributes:
        test_id: Test identifier
        name: Test name
        status: Execution status
        duration_ms: Execution duration
        message: Status message
        error: Error details if failed
        stack_trace: Stack trace if error
        assertions: Assertion results
        output: Test output/logs
        started_at: Start timestamp
        completed_at: Completion timestamp
    """
    test_id: str
    name: str
    status: TestStatus = TestStatus.PENDING
    duration_ms: float = 0.0
    message: str = ""
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    output: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "error": self.error,
            "stack_trace": self.stack_trace,
            "assertions": self.assertions,
            "output": self.output[:1000],  # Truncate output
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class TestSuite:
    """
    Test suite containing test cases.

    Attributes:
        suite_id: Unique suite identifier
        name: Suite name
        description: Suite description
        tests: List of test cases
        setup_fn: Suite-level setup
        teardown_fn: Suite-level teardown
        tags: Suite tags
        parallel: Run tests in parallel
        stop_on_failure: Stop on first failure
    """
    suite_id: str
    name: str
    description: str = ""
    tests: List[TestCase] = field(default_factory=list)
    setup_fn: Optional[Callable] = None
    teardown_fn: Optional[Callable] = None
    tags: Set[str] = field(default_factory=set)
    parallel: bool = False
    stop_on_failure: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "tests": [t.to_dict() for t in self.tests],
            "tags": list(self.tags),
            "parallel": self.parallel,
            "stop_on_failure": self.stop_on_failure,
        }

    def add_test(self, test: TestCase) -> None:
        """Add a test to the suite."""
        self.tests.append(test)


@dataclass
class SuiteResult:
    """
    Test suite execution result.

    Attributes:
        suite_id: Suite identifier
        name: Suite name
        results: Individual test results
        total: Total tests
        passed: Passed tests
        failed: Failed tests
        skipped: Skipped tests
        errors: Tests with errors
        duration_ms: Total duration
        started_at: Start timestamp
        completed_at: Completion timestamp
    """
    suite_id: str
    name: str
    results: List[TestResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "results": [r.to_dict() for r in self.results],
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success_rate": self.passed / self.total if self.total > 0 else 0,
        }

    @property
    def success(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0


# ==============================================================================
# Test Context
# ==============================================================================

class TestContext:
    """
    Test execution context providing assertions and utilities.
    """

    def __init__(self, test_id: str):
        """Initialize test context."""
        self.test_id = test_id
        self.assertions: List[Dict[str, Any]] = []
        self.output_lines: List[str] = []

    def log(self, message: str) -> None:
        """Log a message."""
        self.output_lines.append(f"[{datetime.now().isoformat()}] {message}")

    def assert_true(self, condition: bool, message: str = "") -> None:
        """Assert that condition is True."""
        self.assertions.append({
            "type": "assert_true",
            "passed": condition,
            "message": message or "Expected True",
        })
        if not condition:
            raise AssertionError(message or "Expected True")

    def assert_false(self, condition: bool, message: str = "") -> None:
        """Assert that condition is False."""
        self.assertions.append({
            "type": "assert_false",
            "passed": not condition,
            "message": message or "Expected False",
        })
        if condition:
            raise AssertionError(message or "Expected False")

    def assert_equal(self, actual: Any, expected: Any, message: str = "") -> None:
        """Assert that actual equals expected."""
        passed = actual == expected
        self.assertions.append({
            "type": "assert_equal",
            "passed": passed,
            "actual": str(actual)[:100],
            "expected": str(expected)[:100],
            "message": message or f"Expected {expected}, got {actual}",
        })
        if not passed:
            raise AssertionError(message or f"Expected {expected}, got {actual}")

    def assert_not_equal(self, actual: Any, expected: Any, message: str = "") -> None:
        """Assert that actual does not equal expected."""
        passed = actual != expected
        self.assertions.append({
            "type": "assert_not_equal",
            "passed": passed,
            "message": message or f"Expected {actual} != {expected}",
        })
        if not passed:
            raise AssertionError(message or f"Expected {actual} != {expected}")

    def assert_in(self, item: Any, container: Any, message: str = "") -> None:
        """Assert that item is in container."""
        passed = item in container
        self.assertions.append({
            "type": "assert_in",
            "passed": passed,
            "message": message or f"Expected {item} in container",
        })
        if not passed:
            raise AssertionError(message or f"Expected {item} in container")

    def assert_raises(self, exception_type: type, callable_fn: Callable, *args, **kwargs) -> None:
        """Assert that callable raises an exception."""
        try:
            callable_fn(*args, **kwargs)
            self.assertions.append({
                "type": "assert_raises",
                "passed": False,
                "message": f"Expected {exception_type.__name__} to be raised",
            })
            raise AssertionError(f"Expected {exception_type.__name__} to be raised")
        except exception_type:
            self.assertions.append({
                "type": "assert_raises",
                "passed": True,
                "message": f"{exception_type.__name__} raised as expected",
            })

    def get_output(self) -> str:
        """Get accumulated output."""
        return "\n".join(self.output_lines)


# ==============================================================================
# Test Runner
# ==============================================================================

class TestRunner:
    """
    Test execution engine.

    Runs tests with proper setup/teardown, timeout handling,
    and dependency resolution.
    """

    def __init__(self, actor_id: str = "test-runner"):
        """Initialize test runner."""
        self.actor_id = actor_id
        self._results: Dict[str, TestResult] = {}

    async def run_test(self, test: TestCase) -> TestResult:
        """
        Run a single test case.

        Args:
            test: Test case to run

        Returns:
            TestResult
        """
        result = TestResult(
            test_id=test.test_id,
            name=test.name,
            started_at=time.time(),
        )

        if not test.enabled:
            result.status = TestStatus.SKIPPED
            result.message = "Test disabled"
            result.completed_at = time.time()
            return result

        # Check dependencies
        for dep_id in test.depends_on:
            dep_result = self._results.get(dep_id)
            if not dep_result or dep_result.status != TestStatus.PASSED:
                result.status = TestStatus.SKIPPED
                result.message = f"Dependency not satisfied: {dep_id}"
                result.completed_at = time.time()
                return result

        result.status = TestStatus.RUNNING
        ctx = TestContext(test.test_id)

        try:
            # Setup
            if test.setup_fn:
                if asyncio.iscoroutinefunction(test.setup_fn):
                    await asyncio.wait_for(test.setup_fn(ctx), timeout=test.timeout_s)
                else:
                    test.setup_fn(ctx)

            # Run test
            if test.test_fn:
                if asyncio.iscoroutinefunction(test.test_fn):
                    await asyncio.wait_for(test.test_fn(ctx), timeout=test.timeout_s)
                else:
                    test.test_fn(ctx)

            result.status = TestStatus.PASSED
            result.message = "Test passed"

        except asyncio.TimeoutError:
            result.status = TestStatus.TIMEOUT
            result.error = f"Test timed out after {test.timeout_s}s"
            result.message = "Timeout"

        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.error = str(e)
            result.message = "Assertion failed"
            result.stack_trace = traceback.format_exc()

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error = str(e)
            result.message = f"Error: {type(e).__name__}"
            result.stack_trace = traceback.format_exc()

        finally:
            # Teardown
            try:
                if test.teardown_fn:
                    if asyncio.iscoroutinefunction(test.teardown_fn):
                        await test.teardown_fn(ctx)
                    else:
                        test.teardown_fn(ctx)
            except Exception as e:
                if result.status == TestStatus.PASSED:
                    result.status = TestStatus.ERROR
                    result.error = f"Teardown error: {e}"

        result.completed_at = time.time()
        result.duration_ms = (result.completed_at - result.started_at) * 1000
        result.assertions = ctx.assertions
        result.output = ctx.get_output()

        self._results[test.test_id] = result
        return result

    async def run_suite(self, suite: TestSuite) -> SuiteResult:
        """
        Run a test suite.

        Args:
            suite: Test suite to run

        Returns:
            SuiteResult
        """
        suite_result = SuiteResult(
            suite_id=suite.suite_id,
            name=suite.name,
            started_at=time.time(),
            total=len(suite.tests),
        )

        # Suite setup
        if suite.setup_fn:
            try:
                if asyncio.iscoroutinefunction(suite.setup_fn):
                    await suite.setup_fn()
                else:
                    suite.setup_fn()
            except Exception as e:
                # Suite setup failed, skip all tests
                for test in suite.tests:
                    result = TestResult(
                        test_id=test.test_id,
                        name=test.name,
                        status=TestStatus.SKIPPED,
                        message=f"Suite setup failed: {e}",
                    )
                    suite_result.results.append(result)
                    suite_result.skipped += 1
                suite_result.completed_at = time.time()
                return suite_result

        # Run tests
        if suite.parallel:
            # Run tests in parallel
            tasks = [self.run_test(test) for test in suite.tests if test.enabled]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    result = TestResult(
                        test_id=suite.tests[i].test_id,
                        name=suite.tests[i].name,
                        status=TestStatus.ERROR,
                        error=str(result),
                    )
                suite_result.results.append(result)
        else:
            # Run tests sequentially
            for test in suite.tests:
                result = await self.run_test(test)
                suite_result.results.append(result)

                if suite.stop_on_failure and result.status in (TestStatus.FAILED, TestStatus.ERROR):
                    # Skip remaining tests
                    break

        # Suite teardown
        if suite.teardown_fn:
            try:
                if asyncio.iscoroutinefunction(suite.teardown_fn):
                    await suite.teardown_fn()
                else:
                    suite.teardown_fn()
            except Exception:
                pass  # Log but don't fail

        # Calculate totals
        for result in suite_result.results:
            if result.status == TestStatus.PASSED:
                suite_result.passed += 1
            elif result.status == TestStatus.FAILED:
                suite_result.failed += 1
            elif result.status == TestStatus.SKIPPED:
                suite_result.skipped += 1
            elif result.status in (TestStatus.ERROR, TestStatus.TIMEOUT):
                suite_result.errors += 1

        suite_result.completed_at = time.time()
        suite_result.duration_ms = (suite_result.completed_at - suite_result.started_at) * 1000

        return suite_result


# ==============================================================================
# Test Reporter
# ==============================================================================

class TestReporter:
    """
    Test result reporting.

    Supports multiple output formats:
    - Text (console)
    - JSON
    - JUnit XML
    - HTML
    - Markdown
    """

    def format_result(
        self,
        result: Union[TestResult, SuiteResult],
        format: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """
        Format a test result.

        Args:
            result: Test or suite result
            format: Output format

        Returns:
            Formatted string
        """
        if format == ReportFormat.JSON:
            return json.dumps(result.to_dict(), indent=2)

        elif format == ReportFormat.JUNIT:
            return self._format_junit(result)

        elif format == ReportFormat.HTML:
            return self._format_html(result)

        elif format == ReportFormat.MARKDOWN:
            return self._format_markdown(result)

        else:  # TEXT
            return self._format_text(result)

    def _format_text(self, result: Union[TestResult, SuiteResult]) -> str:
        """Format as plain text."""
        lines = []

        if isinstance(result, SuiteResult):
            lines.append(f"Test Suite: {result.name}")
            lines.append("=" * 50)
            lines.append(f"Total: {result.total} | Passed: {result.passed} | Failed: {result.failed} | Skipped: {result.skipped}")
            lines.append(f"Duration: {result.duration_ms:.1f}ms")
            lines.append("")

            for test_result in result.results:
                status_icon = {
                    TestStatus.PASSED: "[PASS]",
                    TestStatus.FAILED: "[FAIL]",
                    TestStatus.SKIPPED: "[SKIP]",
                    TestStatus.ERROR: "[ERR ]",
                    TestStatus.TIMEOUT: "[TIME]",
                }.get(test_result.status, "[????]")

                lines.append(f"{status_icon} {test_result.name} ({test_result.duration_ms:.1f}ms)")

                if test_result.error:
                    lines.append(f"        Error: {test_result.error}")

        else:
            status = result.status.value.upper()
            lines.append(f"[{status}] {result.name}")
            lines.append(f"  Duration: {result.duration_ms:.1f}ms")
            if result.error:
                lines.append(f"  Error: {result.error}")

        return "\n".join(lines)

    def _format_junit(self, result: Union[TestResult, SuiteResult]) -> str:
        """Format as JUnit XML."""
        if isinstance(result, SuiteResult):
            xml_lines = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<testsuite name="{result.name}" tests="{result.total}" failures="{result.failed}" errors="{result.errors}" skipped="{result.skipped}" time="{result.duration_ms / 1000:.3f}">',
            ]

            for test_result in result.results:
                xml_lines.append(f'  <testcase name="{test_result.name}" time="{test_result.duration_ms / 1000:.3f}">')

                if test_result.status == TestStatus.FAILED:
                    xml_lines.append(f'    <failure message="{test_result.error or "Test failed"}">')
                    if test_result.stack_trace:
                        xml_lines.append(f'      {test_result.stack_trace}')
                    xml_lines.append('    </failure>')

                elif test_result.status == TestStatus.ERROR:
                    xml_lines.append(f'    <error message="{test_result.error or "Error"}">')
                    if test_result.stack_trace:
                        xml_lines.append(f'      {test_result.stack_trace}')
                    xml_lines.append('    </error>')

                elif test_result.status == TestStatus.SKIPPED:
                    xml_lines.append('    <skipped/>')

                xml_lines.append('  </testcase>')

            xml_lines.append('</testsuite>')
            return "\n".join(xml_lines)

        else:
            return self._format_junit(SuiteResult(
                suite_id="single",
                name="Single Test",
                results=[result],
                total=1,
                passed=1 if result.status == TestStatus.PASSED else 0,
                failed=1 if result.status == TestStatus.FAILED else 0,
            ))

    def _format_html(self, result: Union[TestResult, SuiteResult]) -> str:
        """Format as HTML."""
        if isinstance(result, SuiteResult):
            html = [
                '<!DOCTYPE html>',
                '<html><head><title>Test Results</title>',
                '<style>',
                'body { font-family: sans-serif; margin: 20px; }',
                '.passed { color: green; }',
                '.failed { color: red; }',
                '.skipped { color: gray; }',
                '.error { color: orange; }',
                'table { border-collapse: collapse; width: 100%; }',
                'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                '</style></head><body>',
                f'<h1>{result.name}</h1>',
                f'<p>Total: {result.total} | Passed: {result.passed} | Failed: {result.failed}</p>',
                '<table>',
                '<tr><th>Test</th><th>Status</th><th>Duration</th><th>Message</th></tr>',
            ]

            for test_result in result.results:
                status_class = test_result.status.value
                html.append(f'<tr class="{status_class}">')
                html.append(f'<td>{test_result.name}</td>')
                html.append(f'<td>{test_result.status.value}</td>')
                html.append(f'<td>{test_result.duration_ms:.1f}ms</td>')
                html.append(f'<td>{test_result.error or test_result.message}</td>')
                html.append('</tr>')

            html.extend(['</table>', '</body></html>'])
            return "\n".join(html)

        return self._format_html(SuiteResult(
            suite_id="single",
            name="Test Result",
            results=[result],
            total=1,
        ))

    def _format_markdown(self, result: Union[TestResult, SuiteResult]) -> str:
        """Format as Markdown."""
        lines = []

        if isinstance(result, SuiteResult):
            lines.append(f"# {result.name}")
            lines.append("")
            lines.append(f"**Total:** {result.total} | **Passed:** {result.passed} | **Failed:** {result.failed} | **Skipped:** {result.skipped}")
            lines.append(f"**Duration:** {result.duration_ms:.1f}ms")
            lines.append("")
            lines.append("| Test | Status | Duration | Message |")
            lines.append("|------|--------|----------|---------|")

            for test_result in result.results:
                status_emoji = {
                    TestStatus.PASSED: "PASS",
                    TestStatus.FAILED: "FAIL",
                    TestStatus.SKIPPED: "SKIP",
                    TestStatus.ERROR: "ERR",
                }.get(test_result.status, "?")

                msg = test_result.error or test_result.message
                lines.append(f"| {test_result.name} | {status_emoji} | {test_result.duration_ms:.1f}ms | {msg[:50]} |")

        return "\n".join(lines)


# ==============================================================================
# Testing Framework (Step 243)
# ==============================================================================

class TestingFramework:
    """
    Testing Framework - Complete testing capabilities for deployments.

    PBTSO Phase: VERIFY

    Responsibilities:
    - Define and organize test cases
    - Execute unit and integration tests
    - Report test results
    - Track test coverage metrics

    Example:
        >>> framework = TestingFramework()
        >>> suite = framework.create_suite("api-tests", "API Integration Tests")
        >>>
        >>> async def test_health(ctx):
        ...     ctx.assert_true(True, "Health check passed")
        >>>
        >>> framework.add_test(suite.suite_id, "health-check", test_fn=test_health)
        >>> result = await framework.run_suite(suite.suite_id)
        >>> print(framework.format_report(result))
    """

    BUS_TOPICS = {
        "run": "deploy.testing.run",
        "passed": "deploy.testing.passed",
        "failed": "deploy.testing.failed",
        "report": "deploy.testing.report",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "testing-framework",
    ):
        """
        Initialize the testing framework.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "testing"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Components
        self._runner = TestRunner(actor_id)
        self._reporter = TestReporter()

        # Storage
        self._suites: Dict[str, TestSuite] = {}
        self._results_history: List[SuiteResult] = []

    def create_suite(
        self,
        name: str,
        description: str = "",
        tags: Optional[Set[str]] = None,
        parallel: bool = False,
        stop_on_failure: bool = False,
    ) -> TestSuite:
        """
        Create a new test suite.

        Args:
            name: Suite name
            description: Suite description
            tags: Suite tags
            parallel: Run tests in parallel
            stop_on_failure: Stop on first failure

        Returns:
            Created TestSuite
        """
        suite_id = f"suite-{uuid.uuid4().hex[:12]}"

        suite = TestSuite(
            suite_id=suite_id,
            name=name,
            description=description,
            tags=tags or set(),
            parallel=parallel,
            stop_on_failure=stop_on_failure,
        )

        self._suites[suite_id] = suite

        _emit_bus_event(
            self.BUS_TOPICS["run"],
            {
                "action": "suite_created",
                "suite_id": suite_id,
                "name": name,
            },
            actor=self.actor_id,
        )

        return suite

    def add_test(
        self,
        suite_id: str,
        name: str,
        test_fn: Callable,
        description: str = "",
        test_type: TestType = TestType.UNIT,
        setup_fn: Optional[Callable] = None,
        teardown_fn: Optional[Callable] = None,
        tags: Optional[Set[str]] = None,
        timeout_s: int = 60,
        depends_on: Optional[List[str]] = None,
    ) -> TestCase:
        """
        Add a test case to a suite.

        Args:
            suite_id: Suite ID
            name: Test name
            test_fn: Test function
            description: Test description
            test_type: Type of test
            setup_fn: Setup function
            teardown_fn: Teardown function
            tags: Test tags
            timeout_s: Timeout in seconds
            depends_on: Test dependencies

        Returns:
            Created TestCase
        """
        suite = self._suites.get(suite_id)
        if not suite:
            raise ValueError(f"Suite not found: {suite_id}")

        test_id = f"test-{uuid.uuid4().hex[:12]}"

        test = TestCase(
            test_id=test_id,
            name=name,
            description=description,
            test_type=test_type,
            test_fn=test_fn,
            setup_fn=setup_fn,
            teardown_fn=teardown_fn,
            tags=tags or set(),
            timeout_s=timeout_s,
            depends_on=depends_on or [],
        )

        suite.add_test(test)
        return test

    async def run_suite(self, suite_id: str) -> SuiteResult:
        """
        Run a test suite.

        Args:
            suite_id: Suite ID to run

        Returns:
            SuiteResult
        """
        suite = self._suites.get(suite_id)
        if not suite:
            raise ValueError(f"Suite not found: {suite_id}")

        _emit_bus_event(
            self.BUS_TOPICS["run"],
            {
                "action": "suite_started",
                "suite_id": suite_id,
                "name": suite.name,
                "test_count": len(suite.tests),
            },
            actor=self.actor_id,
        )

        result = await self._runner.run_suite(suite)
        self._results_history.append(result)
        self._save_result(result)

        # Emit completion event
        topic = self.BUS_TOPICS["passed"] if result.success else self.BUS_TOPICS["failed"]
        _emit_bus_event(
            topic,
            {
                "suite_id": suite_id,
                "name": suite.name,
                "passed": result.passed,
                "failed": result.failed,
                "duration_ms": result.duration_ms,
            },
            level="info" if result.success else "error",
            actor=self.actor_id,
        )

        return result

    async def run_test(self, suite_id: str, test_id: str) -> TestResult:
        """
        Run a single test.

        Args:
            suite_id: Suite ID
            test_id: Test ID

        Returns:
            TestResult
        """
        suite = self._suites.get(suite_id)
        if not suite:
            raise ValueError(f"Suite not found: {suite_id}")

        test = next((t for t in suite.tests if t.test_id == test_id), None)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        result = await self._runner.run_test(test)
        return result

    async def run_all(self, tags: Optional[Set[str]] = None) -> List[SuiteResult]:
        """
        Run all test suites.

        Args:
            tags: Filter by tags

        Returns:
            List of SuiteResults
        """
        results = []

        for suite in self._suites.values():
            if tags and not tags.intersection(suite.tags):
                continue

            result = await self.run_suite(suite.suite_id)
            results.append(result)

        return results

    def format_report(
        self,
        result: Union[TestResult, SuiteResult],
        format: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """
        Format a test result as a report.

        Args:
            result: Test result
            format: Report format

        Returns:
            Formatted report string
        """
        report = self._reporter.format_result(result, format)

        _emit_bus_event(
            self.BUS_TOPICS["report"],
            {
                "format": format.value,
                "suite_id": result.suite_id if isinstance(result, SuiteResult) else None,
                "test_id": result.test_id if isinstance(result, TestResult) else None,
            },
            actor=self.actor_id,
        )

        return report

    def get_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Get a test suite by ID."""
        return self._suites.get(suite_id)

    def list_suites(self, tags: Optional[Set[str]] = None) -> List[TestSuite]:
        """List all test suites."""
        suites = list(self._suites.values())
        if tags:
            suites = [s for s in suites if tags.intersection(s.tags)]
        return suites

    def get_results_history(self, limit: int = 100) -> List[SuiteResult]:
        """Get test results history."""
        return self._results_history[-limit:]

    def _save_result(self, result: SuiteResult) -> None:
        """Save result to disk."""
        result_file = self.state_dir / f"result_{result.suite_id}_{int(result.started_at)}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for testing framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Testing Framework (Step 243)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-suite command
    create_parser = subparsers.add_parser("create-suite", help="Create a test suite")
    create_parser.add_argument("name", help="Suite name")
    create_parser.add_argument("--description", "-d", default="", help="Suite description")
    create_parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # run-suite command
    run_parser = subparsers.add_parser("run-suite", help="Run a test suite")
    run_parser.add_argument("suite_id", help="Suite ID")
    run_parser.add_argument("--format", "-f", default="text",
                           choices=["text", "json", "junit", "markdown"])
    run_parser.add_argument("--output", "-o", help="Output file")

    # list-suites command
    list_parser = subparsers.add_parser("list-suites", help="List test suites")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # history command
    history_parser = subparsers.add_parser("history", help="View results history")
    history_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
    history_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    framework = TestingFramework()

    if args.command == "create-suite":
        suite = framework.create_suite(
            name=args.name,
            description=args.description,
            parallel=args.parallel,
        )

        if args.json:
            print(json.dumps(suite.to_dict(), indent=2))
        else:
            print(f"Created suite: {suite.suite_id}")
            print(f"  Name: {suite.name}")

        return 0

    elif args.command == "run-suite":
        result = asyncio.get_event_loop().run_until_complete(
            framework.run_suite(args.suite_id)
        )

        report_format = ReportFormat(args.format)
        report = framework.format_report(result, report_format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report written to {args.output}")
        else:
            print(report)

        return 0 if result.success else 1

    elif args.command == "list-suites":
        suites = framework.list_suites()

        if args.json:
            print(json.dumps([s.to_dict() for s in suites], indent=2))
        else:
            if not suites:
                print("No test suites found")
            else:
                for s in suites:
                    print(f"{s.suite_id} ({s.name}) - {len(s.tests)} tests")

        return 0

    elif args.command == "history":
        history = framework.get_results_history(limit=args.limit)

        if args.json:
            print(json.dumps([r.to_dict() for r in history], indent=2))
        else:
            if not history:
                print("No test results")
            else:
                for r in history:
                    status = "PASS" if r.success else "FAIL"
                    print(f"[{status}] {r.name}: {r.passed}/{r.total} passed ({r.duration_ms:.1f}ms)")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
