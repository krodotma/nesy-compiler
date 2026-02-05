#!/usr/bin/env python3
"""
Monitor Testing Framework - Step 293

Unit and integration testing framework for the Monitor Agent.

PBTSO Phase: VERIFY

Bus Topics:
- monitor.test.suite.start (emitted)
- monitor.test.suite.complete (emitted)
- monitor.test.case.pass (emitted)
- monitor.test.case.fail (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Type


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestType(Enum):
    """Test types."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SMOKE = "smoke"
    REGRESSION = "regression"


@dataclass
class TestAssertion:
    """A test assertion.

    Attributes:
        expression: Assertion expression
        passed: Whether assertion passed
        message: Assertion message
        expected: Expected value
        actual: Actual value
    """
    expression: str
    passed: bool
    message: str = ""
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "expression": self.expression,
            "passed": self.passed,
            "message": self.message,
            "expected": str(self.expected)[:200] if self.expected is not None else None,
            "actual": str(self.actual)[:200] if self.actual is not None else None,
        }


@dataclass
class TestResult:
    """Result of a test execution.

    Attributes:
        test_name: Test name
        status: Test status
        duration_ms: Execution duration
        assertions: Test assertions
        error: Error message if failed
        error_traceback: Error traceback
        output: Test output
        metadata: Additional metadata
    """
    test_name: str
    status: TestStatus
    duration_ms: float = 0.0
    assertions: List[TestAssertion] = field(default_factory=list)
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "assertions": [a.to_dict() for a in self.assertions],
            "assertion_count": len(self.assertions),
            "passed_assertions": sum(1 for a in self.assertions if a.passed),
            "error": self.error,
            "error_traceback": self.error_traceback,
            "output": self.output[:1000] if self.output else "",
            "metadata": self.metadata,
        }


@dataclass
class TestCase:
    """A test case definition.

    Attributes:
        name: Test name
        test_func: Test function
        test_type: Type of test
        tags: Test tags
        timeout_s: Test timeout
        skip: Whether to skip
        skip_reason: Reason for skipping
        setup_func: Setup function
        teardown_func: Teardown function
    """
    name: str
    test_func: Callable[..., Coroutine[Any, Any, None]]
    test_type: TestType = TestType.UNIT
    tags: List[str] = field(default_factory=list)
    timeout_s: int = 30
    skip: bool = False
    skip_reason: str = ""
    setup_func: Optional[Callable[[], Coroutine[Any, Any, None]]] = None
    teardown_func: Optional[Callable[[], Coroutine[Any, Any, None]]] = None


@dataclass
class TestSuiteResult:
    """Result of a test suite execution.

    Attributes:
        suite_name: Suite name
        results: Test results
        total_duration_ms: Total duration
        started_at: Start timestamp
        completed_at: Completion timestamp
    """
    suite_name: str
    results: List[TestResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def passed(self) -> int:
        """Count of passed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        """Count of failed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def skipped(self) -> int:
        """Count of skipped tests."""
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def errors(self) -> int:
        """Count of error tests."""
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def success_rate(self) -> float:
        """Success rate percentage."""
        total = len(self.results) - self.skipped
        return (self.passed / total * 100) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "total": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "success_rate": self.success_rate,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
        }


class TestContext:
    """Context for test execution with assertion helpers."""

    def __init__(self):
        """Initialize test context."""
        self.assertions: List[TestAssertion] = []
        self.output: List[str] = []

    def assert_true(self, condition: bool, message: str = "") -> None:
        """Assert condition is true."""
        assertion = TestAssertion(
            expression="assert_true",
            passed=condition,
            message=message or "Expected True",
            expected=True,
            actual=condition,
        )
        self.assertions.append(assertion)
        if not condition:
            raise AssertionError(message or "Assertion failed: expected True")

    def assert_false(self, condition: bool, message: str = "") -> None:
        """Assert condition is false."""
        assertion = TestAssertion(
            expression="assert_false",
            passed=not condition,
            message=message or "Expected False",
            expected=False,
            actual=condition,
        )
        self.assertions.append(assertion)
        if condition:
            raise AssertionError(message or "Assertion failed: expected False")

    def assert_equal(self, actual: Any, expected: Any, message: str = "") -> None:
        """Assert values are equal."""
        passed = actual == expected
        assertion = TestAssertion(
            expression="assert_equal",
            passed=passed,
            message=message or f"Expected {expected}, got {actual}",
            expected=expected,
            actual=actual,
        )
        self.assertions.append(assertion)
        if not passed:
            raise AssertionError(message or f"Assertion failed: {actual} != {expected}")

    def assert_not_equal(self, actual: Any, expected: Any, message: str = "") -> None:
        """Assert values are not equal."""
        passed = actual != expected
        assertion = TestAssertion(
            expression="assert_not_equal",
            passed=passed,
            message=message or f"Expected {actual} != {expected}",
            expected=expected,
            actual=actual,
        )
        self.assertions.append(assertion)
        if not passed:
            raise AssertionError(message or f"Assertion failed: {actual} == {expected}")

    def assert_none(self, value: Any, message: str = "") -> None:
        """Assert value is None."""
        passed = value is None
        assertion = TestAssertion(
            expression="assert_none",
            passed=passed,
            message=message or "Expected None",
            expected=None,
            actual=value,
        )
        self.assertions.append(assertion)
        if not passed:
            raise AssertionError(message or f"Assertion failed: expected None, got {value}")

    def assert_not_none(self, value: Any, message: str = "") -> None:
        """Assert value is not None."""
        passed = value is not None
        assertion = TestAssertion(
            expression="assert_not_none",
            passed=passed,
            message=message or "Expected not None",
            expected="not None",
            actual=value,
        )
        self.assertions.append(assertion)
        if not passed:
            raise AssertionError(message or "Assertion failed: expected not None")

    def assert_in(self, item: Any, container: Any, message: str = "") -> None:
        """Assert item is in container."""
        passed = item in container
        assertion = TestAssertion(
            expression="assert_in",
            passed=passed,
            message=message or f"Expected {item} in {container}",
            expected=f"in {container}",
            actual=item,
        )
        self.assertions.append(assertion)
        if not passed:
            raise AssertionError(message or f"Assertion failed: {item} not in {container}")

    def assert_raises(
        self,
        exception_type: Type[Exception],
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Assert function raises exception."""
        raised = False
        actual_exception = None
        try:
            func(*args, **kwargs)
        except exception_type:
            raised = True
        except Exception as e:
            actual_exception = e

        assertion = TestAssertion(
            expression="assert_raises",
            passed=raised,
            message=f"Expected {exception_type.__name__}",
            expected=exception_type.__name__,
            actual=type(actual_exception).__name__ if actual_exception else "no exception",
        )
        self.assertions.append(assertion)
        if not raised:
            raise AssertionError(f"Expected {exception_type.__name__} to be raised")

    def log(self, message: str) -> None:
        """Log a message to test output."""
        self.output.append(message)


class MonitorTestingFramework:
    """
    Testing framework for the Monitor Agent.

    Provides:
    - Test case registration
    - Test suite execution
    - Assertion helpers
    - Test result reporting
    - Integration test support

    Example:
        framework = MonitorTestingFramework()

        # Register test
        @framework.test("test_metric_collection")
        async def test_metric_collection(ctx: TestContext):
            collector = MetricCollector()
            metric = collector.collect("test.metric", 42.0)
            ctx.assert_not_none(metric)
            ctx.assert_equal(metric.value, 42.0)

        # Run tests
        result = await framework.run_all()
    """

    BUS_TOPICS = {
        "suite_start": "monitor.test.suite.start",
        "suite_complete": "monitor.test.suite.complete",
        "case_pass": "monitor.test.case.pass",
        "case_fail": "monitor.test.case.fail",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        suite_name: str = "monitor_tests",
        bus_dir: Optional[str] = None,
    ):
        """Initialize testing framework.

        Args:
            suite_name: Test suite name
            bus_dir: Bus directory
        """
        self._suite_name = suite_name
        self._last_heartbeat = time.time()

        # Test registry
        self._tests: Dict[str, TestCase] = {}
        self._results: List[TestSuiteResult] = []

        # Suite-level setup/teardown
        self._suite_setup: Optional[Callable[[], Coroutine[Any, Any, None]]] = None
        self._suite_teardown: Optional[Callable[[], Coroutine[Any, Any, None]]] = None

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register built-in tests
        self._register_builtin_tests()

    def test(
        self,
        name: Optional[str] = None,
        test_type: TestType = TestType.UNIT,
        tags: Optional[List[str]] = None,
        timeout_s: int = 30,
        skip: bool = False,
        skip_reason: str = "",
    ) -> Callable:
        """Decorator to register a test.

        Args:
            name: Test name
            test_type: Type of test
            tags: Test tags
            timeout_s: Timeout
            skip: Whether to skip
            skip_reason: Reason for skipping

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            test_name = name or func.__name__
            test_case = TestCase(
                name=test_name,
                test_func=func,
                test_type=test_type,
                tags=tags or [],
                timeout_s=timeout_s,
                skip=skip,
                skip_reason=skip_reason,
            )
            self._tests[test_name] = test_case
            return func

        return decorator

    def register_test(self, test_case: TestCase) -> None:
        """Register a test case.

        Args:
            test_case: Test case to register
        """
        self._tests[test_case.name] = test_case

    def set_suite_setup(
        self,
        func: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Set suite-level setup function.

        Args:
            func: Setup function
        """
        self._suite_setup = func

    def set_suite_teardown(
        self,
        func: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Set suite-level teardown function.

        Args:
            func: Teardown function
        """
        self._suite_teardown = func

    async def run_test(self, name: str) -> TestResult:
        """Run a single test.

        Args:
            name: Test name

        Returns:
            Test result
        """
        test_case = self._tests.get(name)
        if not test_case:
            return TestResult(
                test_name=name,
                status=TestStatus.ERROR,
                error=f"Test not found: {name}",
            )

        # Check if skipped
        if test_case.skip:
            return TestResult(
                test_name=name,
                status=TestStatus.SKIPPED,
                metadata={"skip_reason": test_case.skip_reason},
            )

        ctx = TestContext()
        start_time = time.time()

        try:
            # Run setup
            if test_case.setup_func:
                await asyncio.wait_for(
                    test_case.setup_func(),
                    timeout=test_case.timeout_s,
                )

            # Run test
            await asyncio.wait_for(
                test_case.test_func(ctx),
                timeout=test_case.timeout_s,
            )

            # All assertions passed
            result = TestResult(
                test_name=name,
                status=TestStatus.PASSED,
                duration_ms=(time.time() - start_time) * 1000,
                assertions=ctx.assertions,
                output="\n".join(ctx.output),
                metadata={"test_type": test_case.test_type.value},
            )

            self._emit_bus_event(
                self.BUS_TOPICS["case_pass"],
                {"test_name": name, "duration_ms": result.duration_ms},
            )

        except asyncio.TimeoutError:
            result = TestResult(
                test_name=name,
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start_time) * 1000,
                assertions=ctx.assertions,
                error=f"Test timed out after {test_case.timeout_s}s",
                output="\n".join(ctx.output),
            )

            self._emit_bus_event(
                self.BUS_TOPICS["case_fail"],
                {"test_name": name, "error": result.error},
                level="warning",
            )

        except AssertionError as e:
            result = TestResult(
                test_name=name,
                status=TestStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                assertions=ctx.assertions,
                error=str(e),
                error_traceback=traceback.format_exc(),
                output="\n".join(ctx.output),
            )

            self._emit_bus_event(
                self.BUS_TOPICS["case_fail"],
                {"test_name": name, "error": str(e)},
                level="warning",
            )

        except Exception as e:
            result = TestResult(
                test_name=name,
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start_time) * 1000,
                assertions=ctx.assertions,
                error=str(e),
                error_traceback=traceback.format_exc(),
                output="\n".join(ctx.output),
            )

            self._emit_bus_event(
                self.BUS_TOPICS["case_fail"],
                {"test_name": name, "error": str(e)},
                level="error",
            )

        finally:
            # Run teardown
            if test_case.teardown_func:
                try:
                    await asyncio.wait_for(
                        test_case.teardown_func(),
                        timeout=test_case.timeout_s,
                    )
                except Exception:
                    pass

        return result

    async def run_all(
        self,
        test_type: Optional[TestType] = None,
        tags: Optional[List[str]] = None,
    ) -> TestSuiteResult:
        """Run all tests.

        Args:
            test_type: Filter by test type
            tags: Filter by tags

        Returns:
            Suite result
        """
        suite_result = TestSuiteResult(
            suite_name=self._suite_name,
            started_at=time.time(),
        )

        self._emit_bus_event(
            self.BUS_TOPICS["suite_start"],
            {
                "suite_name": self._suite_name,
                "test_count": len(self._tests),
            },
        )

        # Run suite setup
        if self._suite_setup:
            try:
                await self._suite_setup()
            except Exception as e:
                suite_result.results.append(TestResult(
                    test_name="suite_setup",
                    status=TestStatus.ERROR,
                    error=str(e),
                ))
                return suite_result

        # Filter tests
        tests_to_run = []
        for name, test_case in self._tests.items():
            if test_type and test_case.test_type != test_type:
                continue
            if tags and not any(t in test_case.tags for t in tags):
                continue
            tests_to_run.append(name)

        # Run tests
        for name in sorted(tests_to_run):
            result = await self.run_test(name)
            suite_result.results.append(result)

        # Run suite teardown
        if self._suite_teardown:
            try:
                await self._suite_teardown()
            except Exception:
                pass

        suite_result.completed_at = time.time()
        suite_result.total_duration_ms = (
            suite_result.completed_at - suite_result.started_at
        ) * 1000

        self._results.append(suite_result)

        self._emit_bus_event(
            self.BUS_TOPICS["suite_complete"],
            {
                "suite_name": self._suite_name,
                "passed": suite_result.passed,
                "failed": suite_result.failed,
                "success_rate": suite_result.success_rate,
                "duration_ms": suite_result.total_duration_ms,
            },
        )

        return suite_result

    async def run_by_tags(self, tags: List[str]) -> TestSuiteResult:
        """Run tests matching tags.

        Args:
            tags: Tags to match

        Returns:
            Suite result
        """
        return await self.run_all(tags=tags)

    async def run_by_type(self, test_type: TestType) -> TestSuiteResult:
        """Run tests of a specific type.

        Args:
            test_type: Test type

        Returns:
            Suite result
        """
        return await self.run_all(test_type=test_type)

    def list_tests(self) -> List[Dict[str, Any]]:
        """List all registered tests.

        Returns:
            Test info list
        """
        return [
            {
                "name": test.name,
                "type": test.test_type.value,
                "tags": test.tags,
                "skip": test.skip,
                "skip_reason": test.skip_reason,
                "timeout_s": test.timeout_s,
            }
            for test in self._tests.values()
        ]

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get test run history.

        Args:
            limit: Maximum results

        Returns:
            History list
        """
        return [
            {
                "suite_name": r.suite_name,
                "passed": r.passed,
                "failed": r.failed,
                "success_rate": r.success_rate,
                "duration_ms": r.total_duration_ms,
                "started_at": r.started_at,
            }
            for r in reversed(self._results[-limit:])
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get test statistics.

        Returns:
            Statistics
        """
        total_runs = len(self._results)
        if total_runs == 0:
            return {
                "total_runs": 0,
                "registered_tests": len(self._tests),
            }

        last_result = self._results[-1]
        avg_success_rate = sum(r.success_rate for r in self._results) / total_runs

        return {
            "total_runs": total_runs,
            "registered_tests": len(self._tests),
            "last_run": {
                "passed": last_result.passed,
                "failed": last_result.failed,
                "success_rate": last_result.success_rate,
            },
            "avg_success_rate": avg_success_rate,
            "by_type": {
                t.value: sum(
                    1 for test in self._tests.values()
                    if test.test_type == t
                )
                for t in TestType
            },
        }

    def _register_builtin_tests(self) -> None:
        """Register built-in tests."""

        @self.test("test_bus_write", test_type=TestType.INTEGRATION, tags=["bus"])
        async def test_bus_write(ctx: TestContext):
            """Test bus write functionality."""
            event_id = self._emit_bus_event(
                "monitor.test.ping",
                {"test": True},
            )
            ctx.assert_not_none(event_id)
            ctx.log(f"Emitted event: {event_id}")

        @self.test("test_context_assertions", test_type=TestType.UNIT, tags=["core"])
        async def test_context_assertions(ctx: TestContext):
            """Test assertion helpers."""
            ctx.assert_true(True, "True should be true")
            ctx.assert_false(False, "False should be false")
            ctx.assert_equal(1, 1, "1 should equal 1")
            ctx.assert_not_equal(1, 2, "1 should not equal 2")
            ctx.assert_none(None, "None should be None")
            ctx.assert_not_none("value", "String should not be None")
            ctx.assert_in("a", ["a", "b", "c"], "a should be in list")

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_testing",
                "status": "healthy",
                "registered_tests": len(self._tests),
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-testing",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_testing: Optional[MonitorTestingFramework] = None


def get_testing() -> MonitorTestingFramework:
    """Get or create the testing framework singleton.

    Returns:
        MonitorTestingFramework instance
    """
    global _testing
    if _testing is None:
        _testing = MonitorTestingFramework()
    return _testing


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Testing Framework (Step 293)")
    parser.add_argument("--run", action="store_true", help="Run all tests")
    parser.add_argument("--test", metavar="NAME", help="Run specific test")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "smoke"], help="Run tests by type")
    parser.add_argument("--tags", nargs="+", help="Run tests by tags")
    parser.add_argument("--list", action="store_true", help="List all tests")
    parser.add_argument("--history", action="store_true", help="Show test history")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    testing = get_testing()

    if args.run:
        result = asyncio.run(testing.run_all())
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Test Suite: {result.suite_name}")
            print(f"  Passed: {result.passed}")
            print(f"  Failed: {result.failed}")
            print(f"  Skipped: {result.skipped}")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Duration: {result.total_duration_ms:.1f}ms")
            for r in result.results:
                icon = "V" if r.status == TestStatus.PASSED else "X" if r.status == TestStatus.FAILED else "-"
                print(f"    {icon} {r.test_name}: {r.status.value}")

    if args.test:
        result = asyncio.run(testing.run_test(args.test))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            icon = "V" if result.status == TestStatus.PASSED else "X"
            print(f"{icon} {result.test_name}: {result.status.value}")
            if result.error:
                print(f"  Error: {result.error}")

    if args.type:
        test_type = TestType(args.type)
        result = asyncio.run(testing.run_by_type(test_type))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Tests by type '{args.type}': {result.passed}/{len(result.results)} passed")

    if args.tags:
        result = asyncio.run(testing.run_by_tags(args.tags))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Tests by tags {args.tags}: {result.passed}/{len(result.results)} passed")

    if args.list:
        tests = testing.list_tests()
        if args.json:
            print(json.dumps(tests, indent=2))
        else:
            print("Registered Tests:")
            for t in tests:
                skip = " [SKIP]" if t["skip"] else ""
                print(f"  {t['name']}: {t['type']} {t['tags']}{skip}")

    if args.history:
        history = testing.get_history()
        if args.json:
            print(json.dumps(history, indent=2))
        else:
            print("Test History:")
            for h in history:
                print(f"  {h['suite_name']}: {h['passed']}/{h['passed']+h['failed']} ({h['success_rate']:.1f}%)")

    if args.stats:
        stats = testing.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Test Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
