#!/usr/bin/env python3
"""
Testing Framework (Step 193)

Comprehensive testing framework for the Review Agent with support for
unit tests, integration tests, and test discovery.

PBTSO Phase: VERIFY
Bus Topics: review.test.run, review.test.result, review.test.coverage

Testing Features:
- Test discovery
- Unit test runner
- Integration test support
- Mock facilities
- Coverage tracking
- Assertion utilities

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import inspect
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union, Awaitable
from functools import wraps

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

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
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SMOKE = "smoke"


class AssertionType(Enum):
    """Assertion types."""
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    TRUE = "true"
    FALSE = "false"
    IS_NONE = "is_none"
    IS_NOT_NONE = "is_not_none"
    RAISES = "raises"
    IN = "in"
    NOT_IN = "not_in"
    GREATER = "greater"
    LESS = "less"


@dataclass
class AssertionError:
    """Assertion failure details."""
    assertion_type: AssertionType
    expected: Any
    actual: Any
    message: str
    line: int = 0
    file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.assertion_type.value,
            "expected": str(self.expected),
            "actual": str(self.actual),
            "message": self.message,
            "line": self.line,
            "file": self.file,
        }


@dataclass
class TestResult:
    """
    Result of a single test.

    Attributes:
        test_id: Test identifier
        name: Test name
        status: Execution status
        duration_ms: Execution time
        error: Error message (if failed)
        traceback: Error traceback
        assertions: Assertion failures
        output: Captured output
        test_type: Type of test
    """
    test_id: str
    name: str
    status: TestStatus = TestStatus.PENDING
    duration_ms: float = 0
    error: Optional[str] = None
    traceback: Optional[str] = None
    assertions: List[AssertionError] = field(default_factory=list)
    output: str = ""
    test_type: TestType = TestType.UNIT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "traceback": self.traceback,
            "assertions": [a.to_dict() for a in self.assertions],
            "test_type": self.test_type.value,
        }


@dataclass
class TestSuiteResult:
    """
    Result of a test suite run.

    Attributes:
        suite_id: Suite identifier
        name: Suite name
        results: Individual test results
        total: Total tests
        passed: Passed tests
        failed: Failed tests
        skipped: Skipped tests
        errors: Tests with errors
        duration_ms: Total execution time
        coverage: Code coverage percentage
    """
    suite_id: str
    name: str
    results: List[TestResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0
    coverage: Optional[float] = None
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "errors": self.errors,
            },
            "duration_ms": round(self.duration_ms, 2),
            "coverage": self.coverage,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "pass_rate": round(self.passed / self.total * 100, 1) if self.total > 0 else 0,
        }

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0


@dataclass
class TestCase:
    """
    Definition of a test case.

    Attributes:
        name: Test name
        func: Test function
        test_type: Type of test
        tags: Test tags for filtering
        timeout: Test timeout in seconds
        skip: Skip this test
        skip_reason: Reason for skipping
        expected_failures: Expected failure count
    """
    name: str
    func: Callable[..., Union[None, Awaitable[None]]]
    test_type: TestType = TestType.UNIT
    tags: List[str] = field(default_factory=list)
    timeout: float = 30.0
    skip: bool = False
    skip_reason: str = ""
    expected_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "test_type": self.test_type.value,
            "tags": self.tags,
            "timeout": self.timeout,
            "skip": self.skip,
        }


@dataclass
class TestSuite:
    """
    A collection of test cases.

    Attributes:
        name: Suite name
        tests: Test cases
        setup: Setup function
        teardown: Teardown function
        fixtures: Shared fixtures
    """
    name: str
    tests: List[TestCase] = field(default_factory=list)
    setup: Optional[Callable[[], Any]] = None
    teardown: Optional[Callable[[], Any]] = None
    fixtures: Dict[str, Any] = field(default_factory=dict)

    def add_test(self, test: TestCase) -> None:
        """Add a test to the suite."""
        self.tests.append(test)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "tests": [t.to_dict() for t in self.tests],
            "test_count": len(self.tests),
            "has_setup": self.setup is not None,
            "has_teardown": self.teardown is not None,
        }


# ============================================================================
# Assertions
# ============================================================================

class Assertions:
    """Assertion utilities for tests."""

    @staticmethod
    def assertEqual(actual: Any, expected: Any, message: str = "") -> None:
        """Assert two values are equal."""
        if actual != expected:
            raise AssertionException(
                AssertionType.EQUAL,
                expected,
                actual,
                message or f"Expected {expected!r}, got {actual!r}",
            )

    @staticmethod
    def assertNotEqual(actual: Any, expected: Any, message: str = "") -> None:
        """Assert two values are not equal."""
        if actual == expected:
            raise AssertionException(
                AssertionType.NOT_EQUAL,
                f"not {expected}",
                actual,
                message or f"Expected not {expected!r}",
            )

    @staticmethod
    def assertTrue(value: Any, message: str = "") -> None:
        """Assert value is truthy."""
        if not value:
            raise AssertionException(
                AssertionType.TRUE,
                True,
                value,
                message or f"Expected truthy value, got {value!r}",
            )

    @staticmethod
    def assertFalse(value: Any, message: str = "") -> None:
        """Assert value is falsy."""
        if value:
            raise AssertionException(
                AssertionType.FALSE,
                False,
                value,
                message or f"Expected falsy value, got {value!r}",
            )

    @staticmethod
    def assertIsNone(value: Any, message: str = "") -> None:
        """Assert value is None."""
        if value is not None:
            raise AssertionException(
                AssertionType.IS_NONE,
                None,
                value,
                message or f"Expected None, got {value!r}",
            )

    @staticmethod
    def assertIsNotNone(value: Any, message: str = "") -> None:
        """Assert value is not None."""
        if value is None:
            raise AssertionException(
                AssertionType.IS_NOT_NONE,
                "not None",
                None,
                message or "Expected not None",
            )

    @staticmethod
    def assertIn(item: Any, container: Any, message: str = "") -> None:
        """Assert item is in container."""
        if item not in container:
            raise AssertionException(
                AssertionType.IN,
                f"{item!r} in container",
                f"{item!r} not found",
                message or f"Expected {item!r} to be in {container!r}",
            )

    @staticmethod
    def assertNotIn(item: Any, container: Any, message: str = "") -> None:
        """Assert item is not in container."""
        if item in container:
            raise AssertionException(
                AssertionType.NOT_IN,
                f"{item!r} not in container",
                f"{item!r} found",
                message or f"Expected {item!r} not to be in {container!r}",
            )

    @staticmethod
    def assertGreater(a: Any, b: Any, message: str = "") -> None:
        """Assert a > b."""
        if not (a > b):
            raise AssertionException(
                AssertionType.GREATER,
                f"> {b}",
                a,
                message or f"Expected {a!r} > {b!r}",
            )

    @staticmethod
    def assertLess(a: Any, b: Any, message: str = "") -> None:
        """Assert a < b."""
        if not (a < b):
            raise AssertionException(
                AssertionType.LESS,
                f"< {b}",
                a,
                message or f"Expected {a!r} < {b!r}",
            )

    @staticmethod
    def assertRaises(
        exception_type: Type[Exception],
        callable_obj: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Assert callable raises exception."""
        try:
            callable_obj(*args, **kwargs)
        except exception_type:
            return  # Success
        except Exception as e:
            raise AssertionException(
                AssertionType.RAISES,
                exception_type.__name__,
                type(e).__name__,
                f"Expected {exception_type.__name__}, got {type(e).__name__}",
            )
        raise AssertionException(
            AssertionType.RAISES,
            exception_type.__name__,
            "no exception",
            f"Expected {exception_type.__name__} to be raised",
        )


class AssertionException(Exception):
    """Exception raised when an assertion fails."""

    def __init__(
        self,
        assertion_type: AssertionType,
        expected: Any,
        actual: Any,
        message: str,
    ):
        super().__init__(message)
        self.assertion_type = assertion_type
        self.expected = expected
        self.actual = actual
        self.message = message

        # Get caller info
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller = frame.f_back.f_back
            self.line = caller.f_lineno
            self.file = caller.f_code.co_filename
        else:
            self.line = 0
            self.file = ""

    def to_assertion_error(self) -> AssertionError:
        """Convert to AssertionError dataclass."""
        return AssertionError(
            assertion_type=self.assertion_type,
            expected=self.expected,
            actual=self.actual,
            message=self.message,
            line=self.line,
            file=self.file,
        )


# ============================================================================
# Mocking
# ============================================================================

class Mock:
    """Simple mock object."""

    def __init__(
        self,
        return_value: Any = None,
        side_effect: Optional[Callable[..., Any]] = None,
    ):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls: List[tuple] = []
        self.call_count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        self.call_count += 1

        if self.side_effect:
            if callable(self.side_effect):
                return self.side_effect(*args, **kwargs)
            raise self.side_effect

        return self.return_value

    def assert_called(self) -> None:
        """Assert mock was called."""
        if self.call_count == 0:
            raise AssertionException(
                AssertionType.TRUE,
                "called",
                "not called",
                "Expected mock to be called",
            )

    def assert_called_with(self, *args: Any, **kwargs: Any) -> None:
        """Assert mock was called with specific arguments."""
        if not self.calls:
            raise AssertionException(
                AssertionType.TRUE,
                f"called with {args}, {kwargs}",
                "not called",
                "Mock was not called",
            )

        last_call = self.calls[-1]
        if last_call != (args, kwargs):
            raise AssertionException(
                AssertionType.EQUAL,
                (args, kwargs),
                last_call,
                f"Expected call with {args}, {kwargs}, got {last_call}",
            )

    def reset(self) -> None:
        """Reset mock state."""
        self.calls.clear()
        self.call_count = 0


class Patch:
    """Context manager for patching objects."""

    def __init__(self, target: str, replacement: Any):
        self.target = target
        self.replacement = replacement
        self._original = None
        self._obj = None
        self._attr = None

    def __enter__(self) -> Any:
        parts = self.target.rsplit(".", 1)
        if len(parts) == 2:
            module_path, attr = parts
            import importlib
            self._obj = importlib.import_module(module_path)
            self._attr = attr
        else:
            raise ValueError(f"Invalid target: {self.target}")

        self._original = getattr(self._obj, self._attr)
        setattr(self._obj, self._attr, self.replacement)
        return self.replacement

    def __exit__(self, *args: Any) -> None:
        if self._obj and self._attr:
            setattr(self._obj, self._attr, self._original)


# ============================================================================
# Test Runner
# ============================================================================

class TestRunner:
    """
    Executes test suites.

    Example:
        runner = TestRunner()

        suite = TestSuite(name="MyTests")
        suite.add_test(TestCase(name="test_example", func=test_func))

        result = await runner.run(suite)
        print(f"Passed: {result.passed}/{result.total}")
    """

    def __init__(
        self,
        bus_path: Optional[Path] = None,
        capture_output: bool = True,
        fail_fast: bool = False,
    ):
        """
        Initialize test runner.

        Args:
            bus_path: Path to event bus file
            capture_output: Capture stdout/stderr
            fail_fast: Stop on first failure
        """
        self.bus_path = bus_path or self._get_bus_path()
        self.capture_output = capture_output
        self.fail_fast = fail_fast

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "test") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "testing-framework",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    async def run(
        self,
        suite: TestSuite,
        filter_tags: Optional[List[str]] = None,
        filter_type: Optional[TestType] = None,
    ) -> TestSuiteResult:
        """
        Run a test suite.

        Args:
            suite: Test suite to run
            filter_tags: Only run tests with these tags
            filter_type: Only run tests of this type

        Returns:
            TestSuiteResult
        """
        suite_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        result = TestSuiteResult(
            suite_id=suite_id,
            name=suite.name,
        )

        self._emit_event("review.test.run", {
            "suite_id": suite_id,
            "suite_name": suite.name,
            "test_count": len(suite.tests),
            "status": "started",
        })

        # Run setup
        if suite.setup:
            try:
                if asyncio.iscoroutinefunction(suite.setup):
                    await suite.setup()
                else:
                    suite.setup()
            except Exception as e:
                self._emit_event("review.test.run", {
                    "suite_id": suite_id,
                    "status": "error",
                    "error": f"Setup failed: {e}",
                })
                result.errors = 1
                return result

        # Filter tests
        tests_to_run = suite.tests
        if filter_tags:
            tests_to_run = [t for t in tests_to_run if any(tag in t.tags for tag in filter_tags)]
        if filter_type:
            tests_to_run = [t for t in tests_to_run if t.test_type == filter_type]

        result.total = len(tests_to_run)

        # Run tests
        for test_case in tests_to_run:
            test_result = await self._run_test(test_case, suite.fixtures)
            result.results.append(test_result)

            if test_result.status == TestStatus.PASSED:
                result.passed += 1
            elif test_result.status == TestStatus.FAILED:
                result.failed += 1
                if self.fail_fast:
                    break
            elif test_result.status == TestStatus.SKIPPED:
                result.skipped += 1
            elif test_result.status == TestStatus.ERROR:
                result.errors += 1
                if self.fail_fast:
                    break

        # Run teardown
        if suite.teardown:
            try:
                if asyncio.iscoroutinefunction(suite.teardown):
                    await suite.teardown()
                else:
                    suite.teardown()
            except Exception:
                pass  # Don't fail suite on teardown error

        result.duration_ms = (time.time() - start_time) * 1000
        result.completed_at = datetime.now(timezone.utc).isoformat() + "Z"

        self._emit_event("review.test.result", {
            "suite_id": suite_id,
            "summary": {
                "total": result.total,
                "passed": result.passed,
                "failed": result.failed,
                "skipped": result.skipped,
                "errors": result.errors,
            },
            "duration_ms": result.duration_ms,
            "all_passed": result.all_passed,
        })

        return result

    async def _run_test(
        self,
        test_case: TestCase,
        fixtures: Dict[str, Any],
    ) -> TestResult:
        """Run a single test case."""
        test_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        result = TestResult(
            test_id=test_id,
            name=test_case.name,
            test_type=test_case.test_type,
        )

        # Check skip
        if test_case.skip:
            result.status = TestStatus.SKIPPED
            result.error = test_case.skip_reason
            return result

        result.status = TestStatus.RUNNING

        try:
            # Run with timeout
            if asyncio.iscoroutinefunction(test_case.func):
                await asyncio.wait_for(
                    test_case.func(**fixtures),
                    timeout=test_case.timeout,
                )
            else:
                test_case.func(**fixtures)

            result.status = TestStatus.PASSED

        except AssertionException as e:
            result.status = TestStatus.FAILED
            result.error = str(e)
            result.assertions.append(e.to_assertion_error())

        except asyncio.TimeoutError:
            result.status = TestStatus.ERROR
            result.error = f"Test timed out after {test_case.timeout}s"

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error = str(e)
            result.traceback = traceback.format_exc()

        result.duration_ms = (time.time() - start_time) * 1000
        return result


# ============================================================================
# Integration Test Support
# ============================================================================

class IntegrationTest:
    """
    Base class for integration tests.

    Example:
        class MyIntegrationTest(IntegrationTest):
            async def setup(self):
                self.client = await create_client()

            async def teardown(self):
                await self.client.close()

            async def test_api_call(self):
                response = await self.client.get("/api")
                self.assertEqual(response.status, 200)
    """

    def __init__(self):
        self._assertions = Assertions()

    async def setup(self) -> None:
        """Override for test setup."""
        pass

    async def teardown(self) -> None:
        """Override for test teardown."""
        pass

    # Expose assertion methods
    def assertEqual(self, actual: Any, expected: Any, message: str = "") -> None:
        Assertions.assertEqual(actual, expected, message)

    def assertNotEqual(self, actual: Any, expected: Any, message: str = "") -> None:
        Assertions.assertNotEqual(actual, expected, message)

    def assertTrue(self, value: Any, message: str = "") -> None:
        Assertions.assertTrue(value, message)

    def assertFalse(self, value: Any, message: str = "") -> None:
        Assertions.assertFalse(value, message)

    def assertIsNone(self, value: Any, message: str = "") -> None:
        Assertions.assertIsNone(value, message)

    def assertIsNotNone(self, value: Any, message: str = "") -> None:
        Assertions.assertIsNotNone(value, message)

    def assertIn(self, item: Any, container: Any, message: str = "") -> None:
        Assertions.assertIn(item, container, message)

    def assertRaises(
        self,
        exception_type: Type[Exception],
        callable_obj: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        Assertions.assertRaises(exception_type, callable_obj, *args, **kwargs)


# ============================================================================
# Testing Framework
# ============================================================================

class TestingFramework:
    """
    Complete testing framework.

    Example:
        framework = TestingFramework()

        # Create suite
        suite = framework.create_suite("ReviewTests")

        # Add tests
        @framework.test(suite, tags=["unit"])
        def test_example():
            assert True

        # Run
        result = await framework.run_suite(suite)
    """

    BUS_TOPICS = {
        "run": "review.test.run",
        "result": "review.test.result",
        "coverage": "review.test.coverage",
    }

    def __init__(
        self,
        bus_path: Optional[Path] = None,
    ):
        """Initialize testing framework."""
        self.bus_path = bus_path or self._get_bus_path()
        self.runner = TestRunner(bus_path=self.bus_path)

        self._suites: Dict[str, TestSuite] = {}
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "test") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "testing-framework",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def create_suite(
        self,
        name: str,
        setup: Optional[Callable[[], Any]] = None,
        teardown: Optional[Callable[[], Any]] = None,
    ) -> TestSuite:
        """Create a test suite."""
        suite = TestSuite(
            name=name,
            setup=setup,
            teardown=teardown,
        )
        self._suites[name] = suite
        return suite

    def test(
        self,
        suite: TestSuite,
        name: Optional[str] = None,
        test_type: TestType = TestType.UNIT,
        tags: Optional[List[str]] = None,
        timeout: float = 30.0,
        skip: bool = False,
        skip_reason: str = "",
    ) -> Callable:
        """
        Decorator to add a test to a suite.

        Args:
            suite: Target test suite
            name: Test name (defaults to function name)
            test_type: Type of test
            tags: Test tags
            timeout: Test timeout
            skip: Skip this test
            skip_reason: Reason for skipping
        """
        def decorator(func: Callable) -> Callable:
            test_name = name or func.__name__
            test_case = TestCase(
                name=test_name,
                func=func,
                test_type=test_type,
                tags=tags or [],
                timeout=timeout,
                skip=skip,
                skip_reason=skip_reason,
            )
            suite.add_test(test_case)
            return func
        return decorator

    async def run_suite(
        self,
        suite: Union[str, TestSuite],
        filter_tags: Optional[List[str]] = None,
        filter_type: Optional[TestType] = None,
    ) -> TestSuiteResult:
        """
        Run a test suite.

        Args:
            suite: Suite name or object
            filter_tags: Filter by tags
            filter_type: Filter by type

        Returns:
            TestSuiteResult
        """
        if isinstance(suite, str):
            suite = self._suites.get(suite)
            if not suite:
                raise ValueError(f"Suite not found: {suite}")

        return await self.runner.run(suite, filter_tags, filter_type)

    async def run_all(
        self,
        filter_tags: Optional[List[str]] = None,
        filter_type: Optional[TestType] = None,
    ) -> List[TestSuiteResult]:
        """Run all registered suites."""
        results = []
        for suite in self._suites.values():
            result = await self.runner.run(suite, filter_tags, filter_type)
            results.append(result)
        return results

    def get_suites(self) -> List[str]:
        """Get all registered suite names."""
        return list(self._suites.keys())

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "testing-framework",
            "healthy": True,
            "suites_registered": len(self._suites),
            "total_tests": sum(len(s.tests) for s in self._suites.values()),
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Testing Framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Testing Framework (Step 193)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run tests")
    run_parser.add_argument("--suite", help="Suite name to run")
    run_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    run_parser.add_argument("--type", choices=["unit", "integration", "functional"],
                            help="Filter by type")

    # List command
    subparsers.add_parser("list", help="List test suites")

    # Demo command
    subparsers.add_parser("demo", help="Run demo tests")

    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")

    args = parser.parse_args()

    framework = TestingFramework()
    framework.runner.fail_fast = args.fail_fast

    if args.command == "demo":
        # Create demo suite
        suite = framework.create_suite("DemoTests")

        @framework.test(suite, tags=["demo"])
        def test_equality():
            Assertions.assertEqual(1 + 1, 2)

        @framework.test(suite, tags=["demo"])
        def test_truthy():
            Assertions.assertTrue(True)

        @framework.test(suite, tags=["demo"], skip=True, skip_reason="Example skip")
        def test_skipped():
            pass

        @framework.test(suite, tags=["demo"])
        def test_list_contains():
            Assertions.assertIn("a", ["a", "b", "c"])

        result = asyncio.run(framework.run_suite(suite))

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Demo Tests: {result.name}")
            print(f"  Total: {result.total}")
            print(f"  Passed: {result.passed}")
            print(f"  Failed: {result.failed}")
            print(f"  Skipped: {result.skipped}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
            print()
            for r in result.results:
                status_symbol = {
                    TestStatus.PASSED: "+",
                    TestStatus.FAILED: "-",
                    TestStatus.SKIPPED: "~",
                    TestStatus.ERROR: "!",
                }.get(r.status, "?")
                print(f"  [{status_symbol}] {r.name}: {r.status.value}")

        return 0 if result.all_passed else 1

    elif args.command == "list":
        suites = framework.get_suites()
        if args.json:
            print(json.dumps({"suites": suites}, indent=2))
        else:
            print(f"Test Suites: {len(suites)}")
            for name in suites:
                print(f"  {name}")

    elif args.command == "run":
        filter_type = TestType[args.type.upper()] if args.type else None

        if args.suite:
            result = asyncio.run(framework.run_suite(
                args.suite,
                filter_tags=args.tags,
                filter_type=filter_type,
            ))
            results = [result]
        else:
            results = asyncio.run(framework.run_all(
                filter_tags=args.tags,
                filter_type=filter_type,
            ))

        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            total_passed = sum(r.passed for r in results)
            total_failed = sum(r.failed for r in results)
            total_tests = sum(r.total for r in results)
            print(f"Tests: {total_passed}/{total_tests} passed, {total_failed} failed")

        return 0 if all(r.all_passed for r in results) else 1

    else:
        # Default: show status
        status = framework.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Testing Framework: {status['suites_registered']} suites, {status['total_tests']} tests")

    return 0


if __name__ == "__main__":
    sys.exit(main())
