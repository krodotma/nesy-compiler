#!/usr/bin/env python3
"""
testing_framework.py - Testing Framework (Step 43)

Comprehensive testing framework for Research Agent with unit tests,
integration tests, mocking, and test fixtures.

PBTSO Phase: VERIFY

Bus Topics:
- a2a.research.test.run
- a2a.research.test.result
- a2a.research.test.coverage
- research.test.suite.register

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import fcntl
import functools
import inspect
import json
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union
)

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
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
    """Type of test."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SMOKE = "smoke"


@dataclass
class TestConfig:
    """Configuration for testing framework."""

    timeout_seconds: int = 30
    parallel: bool = False
    max_parallel: int = 4
    fail_fast: bool = False
    verbose: bool = True
    coverage: bool = True
    output_dir: Optional[str] = None
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"
        if self.output_dir is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.output_dir = f"{pluribus_root}/.pluribus/research/test-results"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class TestCase:
    """A single test case."""

    name: str
    func: Callable
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    test_type: TestType = TestType.UNIT
    timeout: Optional[int] = None
    skip: bool = False
    skip_reason: Optional[str] = None
    fixtures: List[str] = field(default_factory=list)
    expected_exception: Optional[Type[Exception]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "type": self.test_type.value,
            "skip": self.skip,
        }


@dataclass
class TestResult:
    """Result of a test execution."""

    test: TestCase
    status: TestStatus
    duration_ms: float = 0
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    output: Optional[str] = None
    assertions_passed: int = 0
    assertions_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.test.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
        }


@dataclass
class TestSuiteResult:
    """Result of a test suite execution."""

    name: str
    results: List[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total_duration_ms: float = 0
    coverage_percent: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": len(self.results),
            "duration_ms": self.total_duration_ms,
            "coverage_percent": self.coverage_percent,
        }


@dataclass
class CoverageReport:
    """Code coverage report."""

    total_lines: int = 0
    covered_lines: int = 0
    missed_lines: int = 0
    coverage_percent: float = 0
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "missed_lines": self.missed_lines,
            "coverage_percent": self.coverage_percent,
            "file_count": len(self.files),
        }


# ============================================================================
# Assertions
# ============================================================================


class AssertionError(Exception):
    """Test assertion failure."""
    pass


class Assertions:
    """Test assertions collection."""

    def __init__(self):
        self._passed = 0
        self._failed = 0
        self._messages: List[str] = []

    @property
    def passed(self) -> int:
        return self._passed

    @property
    def failed(self) -> int:
        return self._failed

    def reset(self) -> None:
        """Reset assertion counters."""
        self._passed = 0
        self._failed = 0
        self._messages = []

    def assertTrue(self, condition: bool, message: str = "Expected True") -> None:
        """Assert condition is True."""
        if condition:
            self._passed += 1
        else:
            self._failed += 1
            self._messages.append(message)
            raise AssertionError(message)

    def assertFalse(self, condition: bool, message: str = "Expected False") -> None:
        """Assert condition is False."""
        self.assertTrue(not condition, message)

    def assertEqual(self, actual: Any, expected: Any, message: str = "") -> None:
        """Assert two values are equal."""
        if actual == expected:
            self._passed += 1
        else:
            self._failed += 1
            msg = message or f"Expected {expected!r}, got {actual!r}"
            self._messages.append(msg)
            raise AssertionError(msg)

    def assertNotEqual(self, actual: Any, expected: Any, message: str = "") -> None:
        """Assert two values are not equal."""
        if actual != expected:
            self._passed += 1
        else:
            self._failed += 1
            msg = message or f"Values should not be equal: {actual!r}"
            self._messages.append(msg)
            raise AssertionError(msg)

    def assertIsNone(self, value: Any, message: str = "Expected None") -> None:
        """Assert value is None."""
        self.assertTrue(value is None, message)

    def assertIsNotNone(self, value: Any, message: str = "Expected not None") -> None:
        """Assert value is not None."""
        self.assertTrue(value is not None, message)

    def assertIn(self, item: Any, container: Any, message: str = "") -> None:
        """Assert item is in container."""
        if item in container:
            self._passed += 1
        else:
            self._failed += 1
            msg = message or f"Expected {item!r} to be in {container!r}"
            self._messages.append(msg)
            raise AssertionError(msg)

    def assertNotIn(self, item: Any, container: Any, message: str = "") -> None:
        """Assert item is not in container."""
        if item not in container:
            self._passed += 1
        else:
            self._failed += 1
            msg = message or f"Expected {item!r} not to be in {container!r}"
            self._messages.append(msg)
            raise AssertionError(msg)

    def assertIsInstance(self, obj: Any, cls: Type, message: str = "") -> None:
        """Assert object is instance of class."""
        if isinstance(obj, cls):
            self._passed += 1
        else:
            self._failed += 1
            msg = message or f"Expected instance of {cls.__name__}, got {type(obj).__name__}"
            self._messages.append(msg)
            raise AssertionError(msg)

    def assertRaises(self, exception: Type[Exception], func: Callable, *args, **kwargs) -> None:
        """Assert function raises exception."""
        try:
            func(*args, **kwargs)
            self._failed += 1
            msg = f"Expected {exception.__name__} to be raised"
            self._messages.append(msg)
            raise AssertionError(msg)
        except exception:
            self._passed += 1
        except Exception as e:
            self._failed += 1
            msg = f"Expected {exception.__name__}, got {type(e).__name__}"
            self._messages.append(msg)
            raise AssertionError(msg)

    def assertGreater(self, a: Any, b: Any, message: str = "") -> None:
        """Assert a > b."""
        if a > b:
            self._passed += 1
        else:
            self._failed += 1
            msg = message or f"Expected {a!r} > {b!r}"
            self._messages.append(msg)
            raise AssertionError(msg)

    def assertLess(self, a: Any, b: Any, message: str = "") -> None:
        """Assert a < b."""
        if a < b:
            self._passed += 1
        else:
            self._failed += 1
            msg = message or f"Expected {a!r} < {b!r}"
            self._messages.append(msg)
            raise AssertionError(msg)

    def assertAlmostEqual(
        self,
        a: float,
        b: float,
        places: int = 7,
        message: str = "",
    ) -> None:
        """Assert floats are almost equal."""
        if round(abs(a - b), places) == 0:
            self._passed += 1
        else:
            self._failed += 1
            msg = message or f"Expected {a} to be almost equal to {b} (places={places})"
            self._messages.append(msg)
            raise AssertionError(msg)


# Global assertions instance
assert_ = Assertions()


# ============================================================================
# Mocking
# ============================================================================


T = TypeVar("T")


class Mock(Generic[T]):
    """Mock object for testing."""

    def __init__(
        self,
        return_value: Any = None,
        side_effect: Optional[Union[Exception, Callable]] = None,
    ):
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_count = 0
        self.call_args: Optional[Tuple] = None
        self.call_args_list: List[Tuple] = []
        self._calls: List[Dict[str, Any]] = []

    def __call__(self, *args, **kwargs) -> Any:
        self.call_count += 1
        self.call_args = (args, kwargs)
        self.call_args_list.append((args, kwargs))
        self._calls.append({"args": args, "kwargs": kwargs})

        if self.side_effect:
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)

        return self.return_value

    def assert_called(self) -> None:
        """Assert mock was called."""
        if self.call_count == 0:
            raise AssertionError("Expected mock to be called")

    def assert_called_once(self) -> None:
        """Assert mock was called exactly once."""
        if self.call_count != 1:
            raise AssertionError(f"Expected mock to be called once, called {self.call_count} times")

    def assert_called_with(self, *args, **kwargs) -> None:
        """Assert mock was last called with specific arguments."""
        if self.call_args != (args, kwargs):
            raise AssertionError(f"Expected call with {(args, kwargs)}, got {self.call_args}")

    def assert_not_called(self) -> None:
        """Assert mock was not called."""
        if self.call_count > 0:
            raise AssertionError(f"Expected mock not to be called, called {self.call_count} times")

    def reset_mock(self) -> None:
        """Reset mock state."""
        self.call_count = 0
        self.call_args = None
        self.call_args_list = []
        self._calls = []


@contextmanager
def patch(target: Any, attribute: str, mock_value: Any) -> Generator[Mock, None, None]:
    """
    Context manager to patch an attribute.

    Example:
        with patch(module, "function", Mock(return_value=42)) as mock:
            result = module.function()
            mock.assert_called_once()
    """
    original = getattr(target, attribute, None)
    setattr(target, attribute, mock_value)
    try:
        yield mock_value
    finally:
        if original is not None:
            setattr(target, attribute, original)
        else:
            delattr(target, attribute)


# ============================================================================
# Fixtures
# ============================================================================


class Fixture(ABC):
    """Abstract base for test fixtures."""

    @abstractmethod
    def setup(self) -> Any:
        """Set up fixture and return fixture value."""
        pass

    def teardown(self, value: Any) -> None:
        """Tear down fixture."""
        pass


class SimpleFixture(Fixture):
    """Simple fixture with setup function."""

    def __init__(
        self,
        setup_fn: Callable[[], T],
        teardown_fn: Optional[Callable[[T], None]] = None,
    ):
        self._setup_fn = setup_fn
        self._teardown_fn = teardown_fn

    def setup(self) -> T:
        return self._setup_fn()

    def teardown(self, value: T) -> None:
        if self._teardown_fn:
            self._teardown_fn(value)


class TempDirFixture(Fixture):
    """Fixture that creates a temporary directory."""

    def __init__(self, prefix: str = "test_"):
        self.prefix = prefix
        self._path: Optional[Path] = None

    def setup(self) -> Path:
        import tempfile
        self._path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self._path

    def teardown(self, value: Path) -> None:
        import shutil
        if value.exists():
            shutil.rmtree(value)


class MockBusFixture(Fixture):
    """Fixture that provides a mock bus."""

    def setup(self) -> "MockBus":
        return MockBus()

    def teardown(self, value: "MockBus") -> None:
        value.clear()


class MockBus:
    """Mock implementation of AgentBus for testing."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, topic: str, data: Dict[str, Any]) -> str:
        """Emit an event."""
        event_id = str(uuid.uuid4())
        with self._lock:
            self.events.append({
                "id": event_id,
                "topic": topic,
                "data": data,
                "ts": time.time(),
            })
        return event_id

    def get_events(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get events, optionally filtered by topic."""
        with self._lock:
            if topic:
                return [e for e in self.events if e["topic"] == topic]
            return list(self.events)

    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self.events = []


# ============================================================================
# Test Suite
# ============================================================================


class TestSuite:
    """A collection of test cases."""

    def __init__(
        self,
        name: str,
        config: Optional[TestConfig] = None,
    ):
        self.name = name
        self.config = config or TestConfig()
        self._tests: List[TestCase] = []
        self._fixtures: Dict[str, Fixture] = {}
        self._setup: Optional[Callable] = None
        self._teardown: Optional[Callable] = None

    def add_test(self, test: TestCase) -> None:
        """Add a test case."""
        self._tests.append(test)

    def add_fixture(self, name: str, fixture: Fixture) -> None:
        """Add a fixture."""
        self._fixtures[name] = fixture

    def test(
        self,
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        test_type: TestType = TestType.UNIT,
        timeout: Optional[int] = None,
        skip: bool = False,
        skip_reason: Optional[str] = None,
        fixtures: Optional[List[str]] = None,
    ) -> Callable:
        """
        Decorator to register a test function.

        Example:
            @suite.test(tags={"search"})
            def test_search():
                assert_.assertEqual(search("test"), [])
        """
        def decorator(func: Callable) -> Callable:
            test_case = TestCase(
                name=name or func.__name__,
                func=func,
                description=func.__doc__ or "",
                tags=tags or set(),
                test_type=test_type,
                timeout=timeout,
                skip=skip,
                skip_reason=skip_reason,
                fixtures=fixtures or [],
            )
            self._tests.append(test_case)
            return func
        return decorator

    def setup(self, func: Callable) -> Callable:
        """Register setup function."""
        self._setup = func
        return func

    def teardown(self, func: Callable) -> Callable:
        """Register teardown function."""
        self._teardown = func
        return func

    def run(
        self,
        tags: Optional[Set[str]] = None,
        test_type: Optional[TestType] = None,
    ) -> TestSuiteResult:
        """Run all tests in the suite."""
        results: List[TestResult] = []
        start_time = time.time()

        # Filter tests
        tests = self._tests
        if tags:
            tests = [t for t in tests if tags & t.tags]
        if test_type:
            tests = [t for t in tests if t.test_type == test_type]

        # Run setup
        if self._setup:
            try:
                self._setup()
            except Exception as e:
                # Suite setup failed
                return TestSuiteResult(
                    name=self.name,
                    errors=1,
                )

        # Run tests
        for test in tests:
            result = self._run_test(test)
            results.append(result)

            if self.config.fail_fast and result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                break

        # Run teardown
        if self._teardown:
            try:
                self._teardown()
            except Exception:
                pass

        # Compute summary
        total_duration = (time.time() - start_time) * 1000
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)

        return TestSuiteResult(
            name=self.name,
            results=results,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            total_duration_ms=total_duration,
        )

    def _run_test(self, test: TestCase) -> TestResult:
        """Run a single test."""
        if test.skip:
            return TestResult(
                test=test,
                status=TestStatus.SKIPPED,
                output=test.skip_reason,
            )

        # Setup fixtures
        fixture_values: Dict[str, Any] = {}
        try:
            for fixture_name in test.fixtures:
                if fixture_name in self._fixtures:
                    fixture_values[fixture_name] = self._fixtures[fixture_name].setup()
        except Exception as e:
            return TestResult(
                test=test,
                status=TestStatus.ERROR,
                error=f"Fixture setup failed: {e}",
                error_traceback=traceback.format_exc(),
            )

        # Reset assertions
        assert_.reset()

        start_time = time.time()

        try:
            # Run test with timeout
            timeout = test.timeout or self.config.timeout_seconds

            # Pass fixtures as kwargs if function accepts them
            sig = inspect.signature(test.func)
            kwargs = {k: v for k, v in fixture_values.items() if k in sig.parameters}

            # Execute test
            if asyncio.iscoroutinefunction(test.func):
                asyncio.run(asyncio.wait_for(
                    test.func(**kwargs),
                    timeout=timeout
                ))
            else:
                test.func(**kwargs)

            duration = (time.time() - start_time) * 1000

            return TestResult(
                test=test,
                status=TestStatus.PASSED,
                duration_ms=duration,
                assertions_passed=assert_.passed,
                assertions_failed=0,
            )

        except AssertionError as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                test=test,
                status=TestStatus.FAILED,
                duration_ms=duration,
                error=str(e),
                assertions_passed=assert_.passed,
                assertions_failed=assert_.failed,
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                test=test,
                status=TestStatus.ERROR,
                duration_ms=duration,
                error=str(e),
                error_traceback=traceback.format_exc(),
            )

        finally:
            # Teardown fixtures
            for fixture_name, value in fixture_values.items():
                try:
                    self._fixtures[fixture_name].teardown(value)
                except Exception:
                    pass


# ============================================================================
# Test Runner
# ============================================================================


class TestRunner:
    """
    Test runner for Research Agent.

    Features:
    - Multiple test suites
    - Parallel execution
    - Coverage reporting
    - Bus event emission

    PBTSO Phase: VERIFY

    Example:
        runner = TestRunner()

        # Create suite
        suite = runner.create_suite("research")

        @suite.test(tags={"search"})
        def test_search():
            assert_.assertEqual(search("test"), [])

        # Run tests
        results = runner.run_all()
    """

    def __init__(
        self,
        config: Optional[TestConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        self.config = config or TestConfig()
        self.bus = bus or AgentBus()

        self._suites: Dict[str, TestSuite] = {}
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_runs": 0,
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
        }

    def create_suite(self, name: str) -> TestSuite:
        """Create a new test suite."""
        suite = TestSuite(name, self.config)
        with self._lock:
            self._suites[name] = suite
        return suite

    def add_suite(self, suite: TestSuite) -> None:
        """Add an existing test suite."""
        with self._lock:
            self._suites[suite.name] = suite

    def get_suite(self, name: str) -> Optional[TestSuite]:
        """Get a test suite by name."""
        return self._suites.get(name)

    def run_suite(
        self,
        name: str,
        tags: Optional[Set[str]] = None,
        test_type: Optional[TestType] = None,
    ) -> TestSuiteResult:
        """Run a specific test suite."""
        suite = self.get_suite(name)
        if not suite:
            return TestSuiteResult(name=name, errors=1)

        self._emit_event("a2a.research.test.run", {
            "suite": name,
            "tags": list(tags) if tags else None,
            "type": test_type.value if test_type else None,
        })

        result = suite.run(tags, test_type)

        self._stats["total_runs"] += 1
        self._stats["total_tests"] += len(result.results)
        self._stats["total_passed"] += result.passed
        self._stats["total_failed"] += result.failed

        self._emit_event("a2a.research.test.result", result.to_dict())

        return result

    def run_all(
        self,
        tags: Optional[Set[str]] = None,
        test_type: Optional[TestType] = None,
    ) -> Dict[str, TestSuiteResult]:
        """Run all test suites."""
        results: Dict[str, TestSuiteResult] = {}

        for name in self._suites:
            results[name] = self.run_suite(name, tags, test_type)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get test runner statistics."""
        return {
            **self._stats,
            "suites": len(self._suites),
            "pass_rate": (
                self._stats["total_passed"] / self._stats["total_tests"]
                if self._stats["total_tests"] > 0 else 0.0
            ),
        }

    def generate_report(
        self,
        results: Dict[str, TestSuiteResult],
        format: str = "text",
    ) -> str:
        """Generate test report."""
        if format == "json":
            return json.dumps({
                name: result.to_dict()
                for name, result in results.items()
            }, indent=2)

        # Text format
        lines = ["=" * 60, "TEST REPORT", "=" * 60, ""]

        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0

        for name, result in results.items():
            lines.append(f"Suite: {name}")
            lines.append(f"  Passed: {result.passed}")
            lines.append(f"  Failed: {result.failed}")
            lines.append(f"  Skipped: {result.skipped}")
            lines.append(f"  Errors: {result.errors}")
            lines.append(f"  Duration: {result.total_duration_ms:.2f}ms")
            lines.append("")

            # List failed tests
            failed_tests = [r for r in result.results if r.status == TestStatus.FAILED]
            if failed_tests:
                lines.append("  Failed tests:")
                for test_result in failed_tests:
                    lines.append(f"    - {test_result.test.name}: {test_result.error}")
                lines.append("")

            total_passed += result.passed
            total_failed += result.failed
            total_skipped += result.skipped
            total_errors += result.errors

        lines.extend([
            "-" * 60,
            "SUMMARY",
            f"  Total: {total_passed + total_failed + total_skipped + total_errors}",
            f"  Passed: {total_passed}",
            f"  Failed: {total_failed}",
            f"  Skipped: {total_skipped}",
            f"  Errors: {total_errors}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _emit_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
    ) -> str:
        """Emit event with file locking."""
        if not self.config.emit_to_bus:
            return ""

        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "test",
            "level": level,
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Testing Framework."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Testing Framework (Step 43)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run tests")
    run_parser.add_argument("--suite", help="Suite name")
    run_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    run_parser.add_argument("--type", choices=["unit", "integration", "e2e"])
    run_parser.add_argument("--verbose", "-v", action="store_true")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run testing demo")

    args = parser.parse_args()

    runner = TestRunner()

    if args.command == "run":
        tags = set(args.tags) if args.tags else None
        test_type = TestType(args.type) if args.type else None

        if args.suite:
            results = {args.suite: runner.run_suite(args.suite, tags, test_type)}
        else:
            results = runner.run_all(tags, test_type)

        print(runner.generate_report(results))

        # Return non-zero if any failures
        total_failed = sum(r.failed + r.errors for r in results.values())
        return 1 if total_failed > 0 else 0

    elif args.command == "stats":
        stats = runner.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Testing Statistics:")
            print(f"  Total Runs: {stats['total_runs']}")
            print(f"  Total Tests: {stats['total_tests']}")
            print(f"  Total Passed: {stats['total_passed']}")
            print(f"  Total Failed: {stats['total_failed']}")
            print(f"  Pass Rate: {stats['pass_rate']:.1%}")

    elif args.command == "demo":
        print("Running testing framework demo...\n")

        # Create a test suite
        suite = runner.create_suite("demo")

        # Add fixtures
        suite.add_fixture("temp_dir", TempDirFixture())
        suite.add_fixture("mock_bus", MockBusFixture())

        # Register tests
        @suite.test(tags={"basic"})
        def test_assertions():
            """Test basic assertions."""
            assert_.assertEqual(1 + 1, 2)
            assert_.assertTrue(True)
            assert_.assertIn("a", "abc")

        @suite.test(tags={"basic"})
        def test_mock():
            """Test mock functionality."""
            mock_fn = Mock(return_value=42)
            result = mock_fn(1, 2, x=3)
            assert_.assertEqual(result, 42)
            mock_fn.assert_called_once()
            mock_fn.assert_called_with(1, 2, x=3)

        @suite.test(tags={"fixture"}, fixtures=["temp_dir"])
        def test_with_fixture(temp_dir: Path):
            """Test with fixture."""
            assert_.assertTrue(temp_dir.exists())
            test_file = temp_dir / "test.txt"
            test_file.write_text("hello")
            assert_.assertEqual(test_file.read_text(), "hello")

        @suite.test(tags={"failure"})
        def test_failure():
            """Test that fails (expected)."""
            assert_.assertEqual(1, 2, "This should fail")

        @suite.test(skip=True, skip_reason="Demo skip")
        def test_skipped():
            """Skipped test."""
            pass

        # Run tests
        results = runner.run_all()
        print(runner.generate_report(results))

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
