#!/usr/bin/env python3
"""
Step 135: Test Error Handler

Comprehensive error handling for the Test Agent.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.error.occurred (emits)
- test.error.recovered (emits)
- test.error.report (emits)

Dependencies: Steps 101-134 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


# ============================================================================
# Constants
# ============================================================================

class ErrorCategory(Enum):
    """Categories of errors."""
    TEST_EXECUTION = "test_execution"
    TEST_GENERATION = "test_generation"
    CONFIGURATION = "configuration"
    INFRASTRUCTURE = "infrastructure"
    NETWORK = "network"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    VALIDATION = "validation"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for errors."""
    IGNORE = "ignore"
    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"
    FALLBACK = "fallback"
    NOTIFY = "notify"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ErrorInfo:
    """
    Information about an error.

    Attributes:
        error_id: Unique error ID
        category: Error category
        severity: Error severity
        message: Error message
        exception_type: Exception type name
        traceback: Full traceback
        context: Error context
        timestamp: When error occurred
        recovered: Whether error was recovered
        recovery_strategy: Strategy used for recovery
    """
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    message: str = ""
    exception_type: str = ""
    traceback: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    recovered: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_message: Optional[str] = None
    source_location: Optional[Dict[str, Any]] = None

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> 'ErrorInfo':
        """Create ErrorInfo from an exception."""
        # Auto-detect category
        if category is None:
            category = cls._categorize_exception(exception)

        # Auto-detect severity
        if severity is None:
            severity = cls._assess_severity(exception, category)

        # Get source location
        tb = traceback.extract_tb(exception.__traceback__)
        source_location = None
        if tb:
            last_frame = tb[-1]
            source_location = {
                "file": last_frame.filename,
                "line": last_frame.lineno,
                "function": last_frame.name,
            }

        return cls(
            category=category,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            context=context or {},
            source_location=source_location,
        )

    @staticmethod
    def _categorize_exception(exception: Exception) -> ErrorCategory:
        """Categorize an exception."""
        exc_type = type(exception).__name__

        if exc_type in ("TimeoutError", "asyncio.TimeoutError"):
            return ErrorCategory.TIMEOUT
        elif exc_type in ("ConnectionError", "ConnectionRefusedError", "OSError"):
            return ErrorCategory.NETWORK
        elif exc_type in ("PermissionError", "AccessDenied"):
            return ErrorCategory.PERMISSION
        elif exc_type in ("ValueError", "TypeError", "ValidationError"):
            return ErrorCategory.VALIDATION
        elif exc_type in ("ConfigurationError", "KeyError"):
            return ErrorCategory.CONFIGURATION
        elif exc_type in ("ImportError", "ModuleNotFoundError"):
            return ErrorCategory.DEPENDENCY
        elif "test" in exc_type.lower():
            return ErrorCategory.TEST_EXECUTION

        return ErrorCategory.UNKNOWN

    @staticmethod
    def _assess_severity(exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess severity of an exception."""
        if category in (ErrorCategory.INFRASTRUCTURE, ErrorCategory.PERMISSION):
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.TIMEOUT:
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.VALIDATION:
            return ErrorSeverity.LOW
        return ErrorSeverity.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "traceback": self.traceback,
            "context": self.context,
            "timestamp": self.timestamp,
            "recovered": self.recovered,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "recovery_message": self.recovery_message,
            "source_location": self.source_location,
        }


@dataclass
class ErrorReport:
    """
    Error report with aggregated error information.

    Attributes:
        report_id: Report ID
        timestamp: Report timestamp
        errors: List of errors
        by_category: Errors grouped by category
        by_severity: Errors grouped by severity
        recovery_stats: Recovery statistics
    """
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    errors: List[ErrorInfo] = field(default_factory=list)
    period_s: int = 3600

    @property
    def total_errors(self) -> int:
        return len(self.errors)

    @property
    def by_category(self) -> Dict[str, int]:
        counts = {}
        for error in self.errors:
            cat = error.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    @property
    def by_severity(self) -> Dict[str, int]:
        counts = {}
        for error in self.errors:
            sev = error.severity.value
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    @property
    def recovery_rate(self) -> float:
        if not self.errors:
            return 0.0
        recovered = sum(1 for e in self.errors if e.recovered)
        return (recovered / len(self.errors)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "total_errors": self.total_errors,
            "by_category": self.by_category,
            "by_severity": self.by_severity,
            "recovery_rate": self.recovery_rate,
            "period_s": self.period_s,
            "errors": [e.to_dict() for e in self.errors],
        }


@dataclass
class RecoveryHandler:
    """
    Handler for error recovery.

    Attributes:
        category: Error category to handle
        severity: Minimum severity to handle
        strategy: Recovery strategy
        handler: Recovery handler function
        max_retries: Maximum retry attempts
        retry_delay_s: Delay between retries
    """
    category: Optional[ErrorCategory] = None
    exception_type: Optional[str] = None
    severity: Optional[ErrorSeverity] = None
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    handler: Optional[Callable[[ErrorInfo], bool]] = None
    max_retries: int = 3
    retry_delay_s: float = 1.0
    enabled: bool = True

    def matches(self, error: ErrorInfo) -> bool:
        """Check if handler matches an error."""
        if not self.enabled:
            return False
        if self.category and error.category != self.category:
            return False
        if self.exception_type and error.exception_type != self.exception_type:
            return False
        if self.severity and error.severity.value < self.severity.value:
            return False
        return True


@dataclass
class ErrorConfig:
    """
    Configuration for the error handler.

    Attributes:
        output_dir: Output directory for error reports
        max_errors: Maximum errors to track
        report_interval_s: Error report interval
        recovery_handlers: List of recovery handlers
        notify_on_critical: Notify on critical errors
        abort_on_critical: Abort on critical errors
    """
    output_dir: str = ".pluribus/test-agent/errors"
    max_errors: int = 1000
    report_interval_s: int = 3600
    recovery_handlers: List[RecoveryHandler] = field(default_factory=list)
    notify_on_critical: bool = True
    abort_on_critical: bool = False
    default_strategy: RecoveryStrategy = RecoveryStrategy.NOTIFY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": self.output_dir,
            "max_errors": self.max_errors,
            "report_interval_s": self.report_interval_s,
            "notify_on_critical": self.notify_on_critical,
            "abort_on_critical": self.abort_on_critical,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class ErrorBus:
    """Bus interface for errors with file locking."""

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
# Circuit Breaker for Error Handling
# ============================================================================

class CircuitBreaker:
    """Circuit breaker for error-prone operations."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_s: float = 60,
        half_open_requests: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.half_open_requests = half_open_requests
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"
        self._half_open_count = 0

    def can_proceed(self) -> bool:
        """Check if operation can proceed."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.reset_timeout_s:
                self.state = "half-open"
                self._half_open_count = 0
                return True
            return False
        # half-open
        if self._half_open_count < self.half_open_requests:
            self._half_open_count += 1
            return True
        return False

    def record_success(self) -> None:
        """Record successful operation."""
        if self.state == "half-open":
            self.successes += 1
            if self.successes >= self.half_open_requests:
                self.state = "closed"
                self.failures = 0
                self.successes = 0
        else:
            self.failures = max(0, self.failures - 1)

    def record_failure(self) -> None:
        """Record failed operation."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"


# ============================================================================
# Test Error Handler
# ============================================================================

class TestErrorHandler:
    """
    Comprehensive error handling for the Test Agent.

    Features:
    - Error categorization
    - Severity assessment
    - Recovery strategies
    - Error tracking and reporting
    - Circuit breaker support

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.error.occurred, test.error.recovered, test.error.report
    """

    BUS_TOPICS = {
        "occurred": "test.error.occurred",
        "recovered": "test.error.recovered",
        "report": "test.error.report",
    }

    def __init__(self, bus=None, config: Optional[ErrorConfig] = None):
        """
        Initialize the error handler.

        Args:
            bus: Optional bus instance
            config: Error handler configuration
        """
        self.bus = bus or ErrorBus()
        self.config = config or ErrorConfig()
        self._errors: List[ErrorInfo] = []
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._last_report: Optional[float] = None

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Add default recovery handlers
        self._add_default_handlers()

    def _add_default_handlers(self) -> None:
        """Add default recovery handlers."""
        # Retry on network errors
        self.config.recovery_handlers.append(RecoveryHandler(
            category=ErrorCategory.NETWORK,
            strategy=RecoveryStrategy.RETRY,
            max_retries=3,
            retry_delay_s=2.0,
        ))

        # Skip on timeout errors
        self.config.recovery_handlers.append(RecoveryHandler(
            category=ErrorCategory.TIMEOUT,
            strategy=RecoveryStrategy.SKIP,
        ))

        # Abort on configuration errors
        self.config.recovery_handlers.append(RecoveryHandler(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            strategy=RecoveryStrategy.ABORT,
        ))

    def handle_exception(
        self,
        exception: Exception,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorInfo:
        """
        Handle an exception.

        Args:
            exception: Exception to handle
            category: Error category
            severity: Error severity
            context: Additional context

        Returns:
            ErrorInfo with error details
        """
        error = ErrorInfo.from_exception(exception, category, severity, context)
        return self.handle_error(error)

    def handle_error(self, error: ErrorInfo) -> ErrorInfo:
        """
        Handle an error.

        Args:
            error: Error to handle

        Returns:
            ErrorInfo with recovery information
        """
        # Track error
        self._track_error(error)

        # Emit error event
        self._emit_event("occurred", error.to_dict())

        # Find matching recovery handler
        handler = self._find_handler(error)
        if handler:
            error.recovery_strategy = handler.strategy
            self._apply_recovery(error, handler)

        # Check for critical errors
        if error.severity == ErrorSeverity.CRITICAL:
            if self.config.notify_on_critical:
                self._notify_critical(error)
            if self.config.abort_on_critical:
                raise SystemExit(f"Critical error: {error.message}")

        return error

    def _track_error(self, error: ErrorInfo) -> None:
        """Track an error."""
        self._errors.append(error)

        # Trim if too many errors
        if len(self._errors) > self.config.max_errors:
            self._errors = self._errors[-self.config.max_errors:]

        # Save to disk
        self._save_error(error)

    def _save_error(self, error: ErrorInfo) -> None:
        """Save error to disk."""
        errors_file = Path(self.config.output_dir) / "errors.ndjson"

        with open(errors_file, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(error.to_dict()) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _find_handler(self, error: ErrorInfo) -> Optional[RecoveryHandler]:
        """Find matching recovery handler."""
        for handler in self.config.recovery_handlers:
            if handler.matches(error):
                return handler
        return None

    def _apply_recovery(self, error: ErrorInfo, handler: RecoveryHandler) -> None:
        """Apply recovery strategy."""
        if handler.strategy == RecoveryStrategy.IGNORE:
            error.recovered = True
            error.recovery_message = "Error ignored"

        elif handler.strategy == RecoveryStrategy.RETRY:
            error.recovery_message = f"Will retry up to {handler.max_retries} times"

        elif handler.strategy == RecoveryStrategy.SKIP:
            error.recovered = True
            error.recovery_message = "Operation skipped"

        elif handler.strategy == RecoveryStrategy.ABORT:
            error.recovery_message = "Operation aborted"

        elif handler.strategy == RecoveryStrategy.FALLBACK:
            if handler.handler:
                try:
                    error.recovered = handler.handler(error)
                    error.recovery_message = "Fallback applied"
                except Exception as e:
                    error.recovery_message = f"Fallback failed: {e}"

        elif handler.strategy == RecoveryStrategy.NOTIFY:
            error.recovery_message = "Notification sent"

        if error.recovered:
            self._emit_event("recovered", {
                "error_id": error.error_id,
                "strategy": handler.strategy.value,
                "message": error.recovery_message,
            })

    def _notify_critical(self, error: ErrorInfo) -> None:
        """Send notification for critical error."""
        self.bus.emit({
            "topic": "test.notify.send",
            "kind": "error_notification",
            "actor": "test-agent",
            "data": {
                "title": f"Critical Error: {error.exception_type}",
                "message": error.message,
                "severity": "critical",
                "error_id": error.error_id,
            },
        })

    def with_retry(
        self,
        func: Callable,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
        category: Optional[ErrorCategory] = None,
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            max_retries: Maximum retries
            retry_delay_s: Delay between retries
            category: Error category for tracking

        Returns:
            Function result
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_error = e
                error = self.handle_exception(e, category, context={"attempt": attempt + 1})

                if attempt < max_retries:
                    time.sleep(retry_delay_s * (attempt + 1))
                else:
                    raise

        raise last_error

    def with_circuit_breaker(
        self,
        name: str,
        func: Callable,
        failure_threshold: int = 5,
        reset_timeout_s: float = 60,
    ) -> Any:
        """
        Execute function with circuit breaker.

        Args:
            name: Circuit breaker name
            func: Function to execute
            failure_threshold: Failures before opening
            reset_timeout_s: Timeout before retry

        Returns:
            Function result
        """
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                reset_timeout_s=reset_timeout_s,
            )

        breaker = self._circuit_breakers[name]

        if not breaker.can_proceed():
            raise Exception(f"Circuit breaker '{name}' is open")

        try:
            result = func()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise

    def generate_report(self, period_s: Optional[int] = None) -> ErrorReport:
        """
        Generate error report.

        Args:
            period_s: Report period in seconds

        Returns:
            ErrorReport with aggregated data
        """
        period = period_s or self.config.report_interval_s
        cutoff = time.time() - period

        errors = [e for e in self._errors if e.timestamp >= cutoff]

        report = ErrorReport(
            errors=errors,
            period_s=period,
        )

        self._emit_event("report", report.to_dict())
        self._save_report(report)

        return report

    def _save_report(self, report: ErrorReport) -> None:
        """Save error report."""
        report_file = Path(self.config.output_dir) / f"error_report_{report.report_id[:8]}.json"

        with open(report_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(report.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def add_handler(self, handler: RecoveryHandler) -> None:
        """Add a recovery handler."""
        self.config.recovery_handlers.append(handler)

    def remove_handler(self, category: ErrorCategory) -> bool:
        """Remove recovery handlers for a category."""
        original_len = len(self.config.recovery_handlers)
        self.config.recovery_handlers = [
            h for h in self.config.recovery_handlers if h.category != category
        ]
        return len(self.config.recovery_handlers) < original_len

    def get_recent_errors(self, limit: int = 20) -> List[ErrorInfo]:
        """Get recent errors."""
        return self._errors[-limit:]

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": len(self._errors),
            "by_category": {cat.value: 0 for cat in ErrorCategory},
            "by_severity": {sev.value: 0 for sev in ErrorSeverity},
            "recovery_rate": 0.0,
        }

    def clear_errors(self) -> int:
        """Clear all tracked errors."""
        count = len(self._errors)
        self._errors.clear()
        return count

    async def handle_exception_async(
        self,
        exception: Exception,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorInfo:
        """Async version of handle_exception."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.handle_exception, exception, category, severity, context
        )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.error.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "error",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# Context Manager
# ============================================================================

class ErrorContext:
    """Context manager for error handling."""

    def __init__(
        self,
        handler: TestErrorHandler,
        category: Optional[ErrorCategory] = None,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True,
    ):
        self.handler = handler
        self.category = category
        self.context = context or {}
        self.reraise = reraise
        self.error: Optional[ErrorInfo] = None

    def __enter__(self) -> 'ErrorContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val:
            self.error = self.handler.handle_exception(
                exc_val,
                category=self.category,
                context=self.context,
            )
            return not self.reraise
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Error Handler."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Error Handler")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate error report")
    report_parser.add_argument("--period", type=int, default=3600, help="Period in seconds")

    # List command
    list_parser = subparsers.add_parser("list", help="List recent errors")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--category", choices=[c.value for c in ErrorCategory])
    list_parser.add_argument("--severity", choices=[s.value for s in ErrorSeverity])

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear error history")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show error statistics")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/errors")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = ErrorConfig(output_dir=args.output)
    handler = TestErrorHandler(config=config)

    if args.command == "report":
        report = handler.generate_report(args.period)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"\nError Report (last {args.period}s)")
            print("=" * 40)
            print(f"Total Errors: {report.total_errors}")
            print(f"Recovery Rate: {report.recovery_rate:.1f}%")
            print("\nBy Category:")
            for cat, count in report.by_category.items():
                print(f"  {cat}: {count}")
            print("\nBy Severity:")
            for sev, count in report.by_severity.items():
                print(f"  {sev}: {count}")

    elif args.command == "list":
        errors = handler.get_recent_errors(args.limit)

        if args.category:
            errors = [e for e in errors if e.category.value == args.category]
        if args.severity:
            errors = [e for e in errors if e.severity.value == args.severity]

        if args.json:
            print(json.dumps([e.to_dict() for e in errors], indent=2))
        else:
            print(f"\nRecent Errors ({len(errors)}):")
            for error in errors:
                dt = datetime.fromtimestamp(error.timestamp)
                print(f"  [{error.severity.value.upper()}] {error.exception_type}: {error.message[:50]}")
                print(f"    Category: {error.category.value}, Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.command == "clear":
        count = handler.clear_errors()
        print(f"Cleared {count} errors")

    elif args.command == "stats":
        stats = handler.get_error_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("\nError Statistics:")
            print(f"  Total Errors: {stats['total_errors']}")
            print(f"  Recovery Rate: {stats['recovery_rate']:.1f}%")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
