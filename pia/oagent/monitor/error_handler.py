#!/usr/bin/env python3
"""
Monitor Error Handler - Step 285

Comprehensive error handling for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.error.occurred (emitted)
- monitor.error.recovered (emitted)
- monitor.error.escalated (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import functools
import json
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, can be ignored
    MEDIUM = "medium"     # Notable issues, should be logged
    HIGH = "high"         # Significant issues, may need attention
    CRITICAL = "critical" # Critical issues, immediate attention


class ErrorCategory(Enum):
    """Error categories."""
    CONFIGURATION = "configuration"
    NETWORK = "network"
    DATABASE = "database"
    PROCESSING = "processing"
    VALIDATION = "validation"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    IGNORE = "ignore"       # Ignore and continue
    RETRY = "retry"         # Retry the operation
    FALLBACK = "fallback"   # Use fallback behavior
    CIRCUIT_BREAK = "circuit_break"  # Stop trying
    ESCALATE = "escalate"   # Escalate to higher level


@dataclass
class ErrorContext:
    """Context information for an error.

    Attributes:
        operation: Operation that failed
        component: Component where error occurred
        input_data: Input data (sanitized)
        timestamp: Error timestamp
        trace_id: Trace ID for correlation
    """
    operation: str
    component: str = "monitor"
    input_data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "component": self.component,
            "input_data": self.input_data,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
        }


@dataclass
class ErrorRecord:
    """A recorded error.

    Attributes:
        error_id: Unique error ID
        error_type: Error type name
        message: Error message
        severity: Error severity
        category: Error category
        context: Error context
        traceback: Stack trace
        recovery_attempts: Number of recovery attempts
        recovered: Whether error was recovered
    """
    error_id: str
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    traceback: Optional[str] = None
    recovery_attempts: int = 0
    recovered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "traceback": self.traceback,
            "recovery_attempts": self.recovery_attempts,
            "recovered": self.recovered,
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state.

    Attributes:
        failures: Number of consecutive failures
        last_failure: Last failure timestamp
        open: Whether circuit is open
        open_until: Time when circuit can be tested
    """
    failures: int = 0
    last_failure: float = 0.0
    open: bool = False
    open_until: float = 0.0

    def is_open(self) -> bool:
        """Check if circuit is open."""
        if not self.open:
            return False
        if time.time() > self.open_until:
            return False  # Half-open state
        return True


class MonitorErrorHandler:
    """
    Comprehensive error handling for the Monitor Agent.

    Provides:
    - Error classification and categorization
    - Automatic recovery strategies
    - Circuit breaker pattern
    - Error aggregation and deduplication
    - Error escalation

    Example:
        handler = MonitorErrorHandler()

        # Handle an error
        try:
            risky_operation()
        except Exception as e:
            handler.handle_error(e, context=ErrorContext(
                operation="risky_operation",
                component="processor",
            ))

        # Use decorator for automatic handling
        @handler.with_error_handling(
            retries=3,
            fallback=lambda: default_value,
        )
        async def process_data(data):
            ...
    """

    BUS_TOPICS = {
        "occurred": "monitor.error.occurred",
        "recovered": "monitor.error.recovered",
        "escalated": "monitor.error.escalated",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    # Error classification mapping
    ERROR_CATEGORIES: Dict[Type[Exception], ErrorCategory] = {
        ConnectionError: ErrorCategory.NETWORK,
        TimeoutError: ErrorCategory.TIMEOUT,
        PermissionError: ErrorCategory.PERMISSION,
        ValueError: ErrorCategory.VALIDATION,
        TypeError: ErrorCategory.VALIDATION,
        MemoryError: ErrorCategory.RESOURCE,
        FileNotFoundError: ErrorCategory.CONFIGURATION,
        KeyError: ErrorCategory.CONFIGURATION,
    }

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
        circuit_threshold: int = 5,
        circuit_timeout_s: float = 60.0,
        error_history_size: int = 1000,
        bus_dir: Optional[str] = None,
    ):
        """Initialize error handler.

        Args:
            max_retries: Maximum retry attempts
            retry_delay_s: Delay between retries
            circuit_threshold: Failures before circuit opens
            circuit_timeout_s: Circuit open duration
            error_history_size: Maximum error history
            bus_dir: Bus directory
        """
        self._max_retries = max_retries
        self._retry_delay = retry_delay_s
        self._circuit_threshold = circuit_threshold
        self._circuit_timeout = circuit_timeout_s
        self._error_history_size = error_history_size

        # Error tracking
        self._errors: List[ErrorRecord] = []
        self._error_counts: Dict[str, int] = {}
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._lock = threading.RLock()
        self._last_heartbeat = time.time()

        # Callbacks
        self._error_callbacks: List[Callable[[ErrorRecord], None]] = []
        self._recovery_callbacks: List[Callable[[ErrorRecord], None]] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
    ) -> ErrorRecord:
        """Handle an error.

        Args:
            error: The exception
            context: Error context
            severity: Override severity
            category: Override category

        Returns:
            Error record
        """
        # Create context if not provided
        if context is None:
            context = ErrorContext(operation="unknown")

        # Classify error
        if category is None:
            category = self._classify_error(error)

        if severity is None:
            severity = self._determine_severity(error, category)

        # Create error record
        record = ErrorRecord(
            error_id=str(uuid.uuid4()),
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            category=category,
            context=context,
            traceback=traceback.format_exc(),
        )

        # Store error
        with self._lock:
            self._errors.append(record)
            if len(self._errors) > self._error_history_size:
                self._errors = self._errors[-self._error_history_size:]

            # Update counts
            key = f"{record.error_type}:{context.operation}"
            self._error_counts[key] = self._error_counts.get(key, 0) + 1

            # Update circuit breaker
            self._update_circuit_breaker(context.operation)

        # Emit event
        self._emit_bus_event(
            self.BUS_TOPICS["occurred"],
            record.to_dict(),
            level="error",
        )

        # Invoke callbacks
        for callback in self._error_callbacks:
            try:
                callback(record)
            except Exception:
                pass

        # Check for escalation
        if severity == ErrorSeverity.CRITICAL:
            self._escalate_error(record)

        return record

    def mark_recovered(self, error_id: str) -> bool:
        """Mark an error as recovered.

        Args:
            error_id: Error ID

        Returns:
            True if marked
        """
        with self._lock:
            for record in self._errors:
                if record.error_id == error_id:
                    record.recovered = True

                    self._emit_bus_event(
                        self.BUS_TOPICS["recovered"],
                        {"error_id": error_id},
                    )

                    for callback in self._recovery_callbacks:
                        try:
                            callback(record)
                        except Exception:
                            pass

                    return True
        return False

    def with_error_handling(
        self,
        operation: Optional[str] = None,
        component: str = "monitor",
        retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        fallback: Optional[Callable[[], Any]] = None,
        circuit_key: Optional[str] = None,
        severity: Optional[ErrorSeverity] = None,
    ) -> Callable:
        """Decorator for automatic error handling.

        Args:
            operation: Operation name
            component: Component name
            retries: Number of retries
            retry_delay: Delay between retries
            fallback: Fallback function
            circuit_key: Circuit breaker key
            severity: Override severity

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation or func.__name__
            max_retries = retries if retries is not None else self._max_retries
            delay = retry_delay if retry_delay is not None else self._retry_delay
            cb_key = circuit_key or op_name

            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    # Check circuit breaker
                    if self._is_circuit_open(cb_key):
                        if fallback:
                            return fallback()
                        raise RuntimeError(f"Circuit breaker open for {cb_key}")

                    context = ErrorContext(
                        operation=op_name,
                        component=component,
                    )

                    last_error: Optional[Exception] = None
                    for attempt in range(max_retries + 1):
                        try:
                            result = await func(*args, **kwargs)
                            # Success - reset circuit breaker
                            self._reset_circuit_breaker(cb_key)
                            return result
                        except Exception as e:
                            last_error = e
                            record = self.handle_error(e, context, severity)
                            record.recovery_attempts = attempt + 1

                            if attempt < max_retries:
                                await asyncio.sleep(delay * (2 ** attempt))
                            else:
                                break

                    # All retries failed
                    if fallback:
                        return fallback()
                    raise last_error

                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    # Check circuit breaker
                    if self._is_circuit_open(cb_key):
                        if fallback:
                            return fallback()
                        raise RuntimeError(f"Circuit breaker open for {cb_key}")

                    context = ErrorContext(
                        operation=op_name,
                        component=component,
                    )

                    last_error: Optional[Exception] = None
                    for attempt in range(max_retries + 1):
                        try:
                            result = func(*args, **kwargs)
                            self._reset_circuit_breaker(cb_key)
                            return result
                        except Exception as e:
                            last_error = e
                            record = self.handle_error(e, context, severity)
                            record.recovery_attempts = attempt + 1

                            if attempt < max_retries:
                                time.sleep(delay * (2 ** attempt))
                            else:
                                break

                    if fallback:
                        return fallback()
                    raise last_error

                return sync_wrapper

        return decorator

    def register_error_callback(
        self,
        callback: Callable[[ErrorRecord], None],
    ) -> None:
        """Register an error callback.

        Args:
            callback: Callback function
        """
        self._error_callbacks.append(callback)

    def register_recovery_callback(
        self,
        callback: Callable[[ErrorRecord], None],
    ) -> None:
        """Register a recovery callback.

        Args:
            callback: Callback function
        """
        self._recovery_callbacks.append(callback)

    def get_errors(
        self,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        operation: Optional[str] = None,
        limit: int = 100,
    ) -> List[ErrorRecord]:
        """Get error history.

        Args:
            severity: Filter by severity
            category: Filter by category
            operation: Filter by operation
            limit: Maximum results

        Returns:
            List of error records
        """
        with self._lock:
            errors = list(self._errors)

        if severity:
            errors = [e for e in errors if e.severity == severity]
        if category:
            errors = [e for e in errors if e.category == category]
        if operation:
            errors = [e for e in errors if e.context.operation == operation]

        return list(reversed(errors[-limit:]))

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.

        Returns:
            Error statistics
        """
        with self._lock:
            by_severity: Dict[str, int] = {}
            by_category: Dict[str, int] = {}
            by_type: Dict[str, int] = {}
            recovered_count = 0

            for error in self._errors:
                by_severity[error.severity.value] = by_severity.get(error.severity.value, 0) + 1
                by_category[error.category.value] = by_category.get(error.category.value, 0) + 1
                by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
                if error.recovered:
                    recovered_count += 1

            return {
                "total_errors": len(self._errors),
                "recovered": recovered_count,
                "by_severity": by_severity,
                "by_category": by_category,
                "by_type": by_type,
                "circuit_breakers": {
                    k: {
                        "open": v.is_open(),
                        "failures": v.failures,
                    }
                    for k, v in self._circuit_breakers.items()
                },
            }

    def clear_errors(self) -> int:
        """Clear error history.

        Returns:
            Number of errors cleared
        """
        with self._lock:
            count = len(self._errors)
            self._errors.clear()
            self._error_counts.clear()
            return count

    def reset_circuit_breaker(self, key: str) -> bool:
        """Manually reset a circuit breaker.

        Args:
            key: Circuit breaker key

        Returns:
            True if reset
        """
        return self._reset_circuit_breaker(key)

    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify an error by category.

        Args:
            error: The exception

        Returns:
            Error category
        """
        for error_type, category in self.ERROR_CATEGORIES.items():
            if isinstance(error, error_type):
                return category
        return ErrorCategory.UNKNOWN

    def _determine_severity(
        self,
        error: Exception,
        category: ErrorCategory,
    ) -> ErrorSeverity:
        """Determine error severity.

        Args:
            error: The exception
            category: Error category

        Returns:
            Error severity
        """
        # Critical errors
        if category in (ErrorCategory.RESOURCE,):
            return ErrorSeverity.CRITICAL

        # High severity
        if category in (ErrorCategory.DATABASE, ErrorCategory.PERMISSION):
            return ErrorSeverity.HIGH

        # Medium severity
        if category in (ErrorCategory.NETWORK, ErrorCategory.TIMEOUT):
            return ErrorSeverity.MEDIUM

        return ErrorSeverity.LOW

    def _is_circuit_open(self, key: str) -> bool:
        """Check if circuit breaker is open.

        Args:
            key: Circuit breaker key

        Returns:
            True if open
        """
        with self._lock:
            state = self._circuit_breakers.get(key)
            if state is None:
                return False
            return state.is_open()

    def _update_circuit_breaker(self, key: str) -> None:
        """Update circuit breaker state.

        Args:
            key: Circuit breaker key
        """
        with self._lock:
            if key not in self._circuit_breakers:
                self._circuit_breakers[key] = CircuitBreakerState()

            state = self._circuit_breakers[key]
            state.failures += 1
            state.last_failure = time.time()

            if state.failures >= self._circuit_threshold:
                state.open = True
                state.open_until = time.time() + self._circuit_timeout

    def _reset_circuit_breaker(self, key: str) -> bool:
        """Reset circuit breaker state.

        Args:
            key: Circuit breaker key

        Returns:
            True if reset
        """
        with self._lock:
            if key in self._circuit_breakers:
                self._circuit_breakers[key] = CircuitBreakerState()
                return True
            return False

    def _escalate_error(self, record: ErrorRecord) -> None:
        """Escalate a critical error.

        Args:
            record: Error record
        """
        self._emit_bus_event(
            self.BUS_TOPICS["escalated"],
            {
                "error_id": record.error_id,
                "error_type": record.error_type,
                "message": record.message,
                "severity": record.severity.value,
                "context": record.context.to_dict(),
            },
            level="critical",
        )

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        stats = self.get_error_stats()

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_error_handler",
                "status": "healthy",
                "total_errors": stats["total_errors"],
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
            "actor": "monitor-error-handler",
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
_handler: Optional[MonitorErrorHandler] = None


def get_error_handler() -> MonitorErrorHandler:
    """Get or create the error handler singleton.

    Returns:
        MonitorErrorHandler instance
    """
    global _handler
    if _handler is None:
        _handler = MonitorErrorHandler()
    return _handler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Error Handler (Step 285)")
    parser.add_argument("--list", action="store_true", help="List recent errors")
    parser.add_argument("--stats", action="store_true", help="Show error statistics")
    parser.add_argument("--clear", action="store_true", help="Clear error history")
    parser.add_argument("--test", action="store_true", help="Test error handling")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    handler = get_error_handler()

    if args.test:
        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            record = handler.handle_error(e, ErrorContext(
                operation="test",
                component="cli",
            ))
            if args.json:
                print(json.dumps(record.to_dict(), indent=2))
            else:
                print(f"Recorded error: {record.error_id}")
                print(f"  Type: {record.error_type}")
                print(f"  Message: {record.message}")
                print(f"  Severity: {record.severity.value}")
                print(f"  Category: {record.category.value}")

    if args.list:
        errors = handler.get_errors(limit=10)
        if args.json:
            print(json.dumps([e.to_dict() for e in errors], indent=2))
        else:
            print("Recent Errors:")
            for e in errors:
                print(f"  [{e.severity.value}] {e.error_type}: {e.message}")

    if args.stats:
        stats = handler.get_error_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Error Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    if args.clear:
        count = handler.clear_errors()
        if args.json:
            print(json.dumps({"cleared": count}))
        else:
            print(f"Cleared {count} errors")
