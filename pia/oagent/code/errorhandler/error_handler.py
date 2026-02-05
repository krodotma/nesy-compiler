#!/usr/bin/env python3
"""
error_handler.py - Comprehensive Error Handling (Step 85)

PBTSO Phase: All Phases

Provides:
- Structured error types
- Error categorization and severity
- Recovery strategies
- Error reporting and aggregation
- Circuit breaker pattern

Bus Topics:
- code.error.occurred
- code.error.recovered
- code.error.report

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    IO = "io"
    NETWORK = "network"
    PLUGIN = "plugin"
    OPERATION = "operation"
    INTERNAL = "internal"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    RESOURCE = "resource"


@dataclass
class ErrorContext:
    """Context information for an error."""
    operation: str = ""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    component: str = ""
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    additional: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "component": self.component,
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "additional": self.additional,
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
# Error Classes
# =============================================================================

class CodeAgentError(Exception):
    """
    Base exception for all Code Agent errors.

    Provides structured error information.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = time.time()
        self.error_id = f"err-{uuid.uuid4().hex[:12]}"
        self.stack_trace = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace,
        }


class ValidationError(CodeAgentError):
    """Error in input validation."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs,
        )
        self.field = field
        self.value = value


class ConfigurationError(CodeAgentError):
    """Error in configuration."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs: Any):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs,
        )
        self.config_key = config_key


class OperationError(CodeAgentError):
    """Error during an operation."""

    def __init__(
        self,
        message: str,
        operation: str,
        **kwargs: Any,
    ):
        context = kwargs.pop("context", ErrorContext())
        context.operation = operation
        super().__init__(
            message,
            category=ErrorCategory.OPERATION,
            context=context,
            **kwargs,
        )
        self.operation = operation


class PluginError(CodeAgentError):
    """Error in plugin."""

    def __init__(
        self,
        message: str,
        plugin_id: str,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            category=ErrorCategory.PLUGIN,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        self.plugin_id = plugin_id


# =============================================================================
# Error Report
# =============================================================================

@dataclass
class ErrorReport:
    """Aggregated error report."""
    start_time: float
    end_time: float
    total_errors: int
    errors_by_category: Dict[str, int]
    errors_by_severity: Dict[str, int]
    top_errors: List[Dict[str, Any]]
    recovery_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.end_time - self.start_time,
            "total_errors": self.total_errors,
            "errors_by_category": self.errors_by_category,
            "errors_by_severity": self.errors_by_severity,
            "top_errors": self.top_errors,
            "recovery_rate": self.recovery_rate,
        }


# =============================================================================
# Error Recovery
# =============================================================================

class ErrorRecovery(ABC):
    """Abstract base class for error recovery strategies."""

    @abstractmethod
    def can_recover(self, error: CodeAgentError) -> bool:
        """Check if error can be recovered."""
        pass

    @abstractmethod
    async def recover(self, error: CodeAgentError) -> bool:
        """Attempt to recover from error."""
        pass


class RetryRecovery(ErrorRecovery):
    """Retry-based recovery strategy."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential = exponential
        self._retry_count: Dict[str, int] = {}

    def can_recover(self, error: CodeAgentError) -> bool:
        if not error.recoverable:
            return False
        count = self._retry_count.get(error.error_id, 0)
        return count < self.max_retries

    async def recover(self, error: CodeAgentError) -> bool:
        if not self.can_recover(error):
            return False

        count = self._retry_count.get(error.error_id, 0)
        self._retry_count[error.error_id] = count + 1

        # Calculate delay
        if self.exponential:
            delay = min(self.base_delay * (2 ** count), self.max_delay)
        else:
            delay = self.base_delay

        await asyncio.sleep(delay)
        return True


class CircuitBreakerRecovery(ErrorRecovery):
    """Circuit breaker pattern for recovery."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures: Dict[str, int] = {}
        self._last_failure: Dict[str, float] = {}
        self._open: Dict[str, bool] = {}

    def can_recover(self, error: CodeAgentError) -> bool:
        key = error.context.component or "default"

        # Check if circuit is open
        if self._open.get(key, False):
            last = self._last_failure.get(key, 0)
            if time.time() - last > self.reset_timeout:
                # Reset circuit
                self._open[key] = False
                self._failures[key] = 0
                return True
            return False

        return error.recoverable

    async def recover(self, error: CodeAgentError) -> bool:
        key = error.context.component or "default"

        # Record failure
        self._failures[key] = self._failures.get(key, 0) + 1
        self._last_failure[key] = time.time()

        # Check threshold
        if self._failures[key] >= self.failure_threshold:
            self._open[key] = True
            return False

        return True


# =============================================================================
# Error Handler
# =============================================================================

class ErrorHandler:
    """
    Comprehensive error handling system.

    PBTSO Phase: All Phases

    Features:
    - Error registration and tracking
    - Recovery strategies
    - Error aggregation and reporting
    - Bus event emission

    Usage:
        handler = ErrorHandler()
        try:
            do_work()
        except Exception as e:
            handler.handle(e, context=ErrorContext(operation="do_work"))
    """

    BUS_TOPICS = {
        "occurred": "code.error.occurred",
        "recovered": "code.error.recovered",
        "report": "code.error.report",
    }

    def __init__(
        self,
        bus: Optional[LockedAgentBus] = None,
        recovery_strategies: Optional[List[ErrorRecovery]] = None,
    ):
        self.bus = bus or LockedAgentBus()
        self.recovery_strategies = recovery_strategies or [
            RetryRecovery(),
            CircuitBreakerRecovery(),
        ]
        self._errors: List[CodeAgentError] = []
        self._recovered: int = 0
        self._lock = Lock()

    def handle(
        self,
        error: Union[Exception, CodeAgentError],
        context: Optional[ErrorContext] = None,
        raise_on_failure: bool = True,
    ) -> Optional[CodeAgentError]:
        """
        Handle an error.

        Args:
            error: Exception to handle
            context: Error context
            raise_on_failure: Re-raise if unrecoverable

        Returns:
            CodeAgentError instance
        """
        # Wrap standard exceptions
        if not isinstance(error, CodeAgentError):
            error = CodeAgentError(
                message=str(error),
                context=context or ErrorContext(),
                cause=error,
            )
        elif context:
            error.context = context

        # Record error
        with self._lock:
            self._errors.append(error)

        # Emit event
        self.bus.emit({
            "topic": self.BUS_TOPICS["occurred"],
            "kind": "error",
            "level": error.severity.value,
            "actor": error.context.component or "code-agent",
            "data": error.to_dict(),
        })

        # Attempt recovery
        if error.recoverable:
            for strategy in self.recovery_strategies:
                if strategy.can_recover(error):
                    # Note: actual recovery happens when operation is retried
                    self._recovered += 1
                    self.bus.emit({
                        "topic": self.BUS_TOPICS["recovered"],
                        "kind": "recovery",
                        "actor": "error-handler",
                        "data": {
                            "error_id": error.error_id,
                            "strategy": strategy.__class__.__name__,
                        },
                    })
                    return error

        if raise_on_failure and not error.recoverable:
            raise error

        return error

    async def handle_async(
        self,
        error: Union[Exception, CodeAgentError],
        context: Optional[ErrorContext] = None,
        attempt_recovery: bool = True,
    ) -> tuple[CodeAgentError, bool]:
        """
        Handle an error with async recovery.

        Returns:
            Tuple of (error, recovered)
        """
        wrapped_error = self.handle(error, context, raise_on_failure=False)

        if not wrapped_error:
            wrapped_error = CodeAgentError(str(error), cause=error if isinstance(error, Exception) else None)

        if attempt_recovery and wrapped_error.recoverable:
            for strategy in self.recovery_strategies:
                if await strategy.recover(wrapped_error):
                    return wrapped_error, True

        return wrapped_error, False

    def get_errors(
        self,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        since: Optional[float] = None,
    ) -> List[CodeAgentError]:
        """Get filtered list of errors."""
        errors = self._errors

        if category:
            errors = [e for e in errors if e.category == category]
        if severity:
            errors = [e for e in errors if e.severity == severity]
        if since:
            errors = [e for e in errors if e.timestamp >= since]

        return errors

    def generate_report(
        self,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> ErrorReport:
        """Generate error report."""
        now = time.time()
        since = since or (now - 3600)  # Default: last hour
        until = until or now

        errors = [e for e in self._errors if since <= e.timestamp <= until]

        # Aggregate by category
        by_category: Dict[str, int] = {}
        for e in errors:
            cat = e.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        # Aggregate by severity
        by_severity: Dict[str, int] = {}
        for e in errors:
            sev = e.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # Top errors (by message frequency)
        message_counts: Dict[str, int] = {}
        for e in errors:
            message_counts[e.message] = message_counts.get(e.message, 0) + 1

        top_errors = [
            {"message": msg, "count": cnt}
            for msg, cnt in sorted(message_counts.items(), key=lambda x: -x[1])[:10]
        ]

        # Recovery rate
        recovery_rate = self._recovered / len(errors) if errors else 0.0

        report = ErrorReport(
            start_time=since,
            end_time=until,
            total_errors=len(errors),
            errors_by_category=by_category,
            errors_by_severity=by_severity,
            top_errors=top_errors,
            recovery_rate=recovery_rate,
        )

        # Emit report event
        self.bus.emit({
            "topic": self.BUS_TOPICS["report"],
            "kind": "report",
            "actor": "error-handler",
            "data": report.to_dict(),
        })

        return report

    def clear(self) -> None:
        """Clear error history."""
        with self._lock:
            self._errors = []
            self._recovered = 0


# =============================================================================
# Decorators
# =============================================================================

T = TypeVar("T")


def with_error_handling(
    handler: Optional[ErrorHandler] = None,
    context: Optional[ErrorContext] = None,
    reraise: bool = True,
) -> Callable:
    """Decorator for error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            _handler = handler or ErrorHandler()
            _context = context or ErrorContext(operation=func.__name__)

            try:
                return func(*args, **kwargs)
            except CodeAgentError:
                raise
            except Exception as e:
                _handler.handle(e, _context, raise_on_failure=reraise)
                raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            _handler = handler or ErrorHandler()
            _context = context or ErrorContext(operation=func.__name__)

            try:
                return await func(*args, **kwargs)
            except CodeAgentError:
                raise
            except Exception as e:
                _handler.handle(e, _context, raise_on_failure=reraise)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def recoverable(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator for automatic retry on failure."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        time.sleep(delay * (attempt + 1))
            raise last_error  # type: ignore

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_error  # type: ignore

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Error Handler."""
    import argparse

    parser = argparse.ArgumentParser(description="Error Handler (Step 85)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # report command
    report_parser = subparsers.add_parser("report", help="Generate error report")
    report_parser.add_argument("--hours", type=int, default=1, help="Hours to report")
    report_parser.add_argument("--json", action="store_true", help="JSON output")

    # demo command
    subparsers.add_parser("demo", help="Run error handling demo")

    args = parser.parse_args()
    handler = ErrorHandler()

    if args.command == "report":
        since = time.time() - (args.hours * 3600)
        report = handler.generate_report(since=since)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Error Report (last {args.hours} hours)")
            print(f"  Total errors: {report.total_errors}")
            print(f"  Recovery rate: {report.recovery_rate:.1%}")
            print(f"\nBy category:")
            for cat, cnt in report.errors_by_category.items():
                print(f"    {cat}: {cnt}")
            print(f"\nBy severity:")
            for sev, cnt in report.errors_by_severity.items():
                print(f"    {sev}: {cnt}")

        return 0

    elif args.command == "demo":
        print("Running error handling demo...\n")

        # Test different error types
        try:
            raise ValidationError("Invalid input", field="name", value="")
        except CodeAgentError as e:
            handler.handle(e, raise_on_failure=False)
            print(f"Handled: {e.error_id} - {e.message}")

        try:
            raise OperationError("Operation failed", operation="generate")
        except CodeAgentError as e:
            handler.handle(e, raise_on_failure=False)
            print(f"Handled: {e.error_id} - {e.message}")

        # Generate report
        report = handler.generate_report()
        print(f"\nGenerated report: {report.total_errors} errors")

        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
