#!/usr/bin/env python3
"""
error_handler.py - Comprehensive Error Handler (Step 35)

Error handling, recovery, and reporting for Research Agent.
Supports error classification, retry logic, and error aggregation.

PBTSO Phase: STABILIZE

Bus Topics:
- a2a.research.error.handle
- a2a.research.error.recover
- research.error.aggregate

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

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
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from ..bootstrap import AgentBus


# ============================================================================
# Error Codes
# ============================================================================


class ErrorCode(Enum):
    """Error codes for Research Agent."""

    # General errors (1000-1099)
    UNKNOWN = 1000
    INTERNAL = 1001
    TIMEOUT = 1002
    CANCELLED = 1003
    NOT_IMPLEMENTED = 1004

    # Configuration errors (1100-1199)
    CONFIG_INVALID = 1100
    CONFIG_MISSING = 1101
    CONFIG_TYPE_ERROR = 1102

    # Resource errors (1200-1299)
    RESOURCE_NOT_FOUND = 1200
    RESOURCE_UNAVAILABLE = 1201
    RESOURCE_EXHAUSTED = 1202
    RESOURCE_LOCKED = 1203

    # I/O errors (1300-1399)
    IO_READ_ERROR = 1300
    IO_WRITE_ERROR = 1301
    IO_PERMISSION_ERROR = 1302
    IO_NOT_FOUND = 1303

    # Network errors (1400-1499)
    NETWORK_CONNECTION = 1400
    NETWORK_TIMEOUT = 1401
    NETWORK_DNS = 1402

    # Database errors (1500-1599)
    DB_CONNECTION = 1500
    DB_QUERY = 1501
    DB_INTEGRITY = 1502

    # Search errors (1600-1699)
    SEARCH_QUERY_INVALID = 1600
    SEARCH_INDEX_ERROR = 1601
    SEARCH_NO_RESULTS = 1602

    # Analysis errors (1700-1799)
    ANALYSIS_PARSE_ERROR = 1700
    ANALYSIS_SYMBOL_NOT_FOUND = 1701
    ANALYSIS_CYCLE_DETECTED = 1702

    # Plugin errors (1800-1899)
    PLUGIN_LOAD_ERROR = 1800
    PLUGIN_NOT_FOUND = 1801
    PLUGIN_HOOK_ERROR = 1802

    # Rate limit errors (1900-1999)
    RATE_LIMIT_EXCEEDED = 1900
    QUOTA_EXCEEDED = 1901


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    NONE = "none"           # No recovery
    RETRY = "retry"         # Retry operation
    FALLBACK = "fallback"   # Use fallback value
    SKIP = "skip"           # Skip and continue
    ABORT = "abort"         # Abort operation
    ESCALATE = "escalate"   # Escalate to human


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 5000
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[ErrorCode] = field(default_factory=lambda: [
        ErrorCode.TIMEOUT,
        ErrorCode.NETWORK_CONNECTION,
        ErrorCode.NETWORK_TIMEOUT,
        ErrorCode.DB_CONNECTION,
        ErrorCode.RESOURCE_UNAVAILABLE,
    ])


@dataclass
class ErrorConfig:
    """Configuration for error handler."""

    enable_recovery: bool = True
    enable_aggregation: bool = True
    aggregation_window_seconds: int = 60
    max_errors_per_window: int = 100
    log_stack_traces: bool = True
    emit_to_bus: bool = True
    bus_path: Optional[str] = None
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Error Classes
# ============================================================================


@dataclass
class ErrorContext:
    """Context information for an error."""

    operation: str
    component: str = "research"
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "component": self.component,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


class ResearchError(Exception):
    """
    Base exception for Research Agent errors.

    Example:
        raise ResearchError(
            code=ErrorCode.SEARCH_QUERY_INVALID,
            message="Invalid query syntax",
            context=ErrorContext(operation="search"),
        )
    """

    def __init__(
        self,
        code: ErrorCode = ErrorCode.UNKNOWN,
        message: str = "",
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = True,
        recovery_hint: Optional[str] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or ErrorContext(operation="unknown")
        self.cause = cause
        self.severity = severity
        self.recoverable = recoverable
        self.recovery_hint = recovery_hint
        self.timestamp = time.time()
        self.error_id = str(uuid.uuid4())[:8]
        self.stack_trace = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "code": self.code.value,
            "code_name": self.code.name,
            "message": self.message,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "recovery_hint": self.recovery_hint,
            "context": self.context.to_dict() if self.context else None,
            "cause": str(self.cause) if self.cause else None,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return f"[{self.code.name}:{self.error_id}] {self.message}"


# Specific error types
class ConfigError(ResearchError):
    """Configuration error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(code=ErrorCode.CONFIG_INVALID, message=message, **kwargs)


class ResourceError(ResearchError):
    """Resource error."""
    def __init__(self, message: str, code: ErrorCode = ErrorCode.RESOURCE_NOT_FOUND, **kwargs):
        super().__init__(code=code, message=message, **kwargs)


class SearchError(ResearchError):
    """Search error."""
    def __init__(self, message: str, code: ErrorCode = ErrorCode.SEARCH_QUERY_INVALID, **kwargs):
        super().__init__(code=code, message=message, **kwargs)


class AnalysisError(ResearchError):
    """Analysis error."""
    def __init__(self, message: str, code: ErrorCode = ErrorCode.ANALYSIS_PARSE_ERROR, **kwargs):
        super().__init__(code=code, message=message, **kwargs)


class NetworkError(ResearchError):
    """Network error."""
    def __init__(self, message: str, code: ErrorCode = ErrorCode.NETWORK_CONNECTION, **kwargs):
        super().__init__(code=code, message=message, recoverable=True, **kwargs)


# ============================================================================
# Error Handler
# ============================================================================


T = TypeVar("T")


class ErrorHandler:
    """
    Comprehensive error handling with recovery and reporting.

    Features:
    - Error classification and categorization
    - Retry with exponential backoff
    - Error aggregation for rate limiting
    - Bus event emission
    - Recovery strategies

    PBTSO Phase: STABILIZE

    Example:
        handler = ErrorHandler()

        # Handle an error
        handler.handle(error)

        # With recovery
        result = handler.with_retry(
            lambda: risky_operation(),
            context=ErrorContext(operation="search"),
        )

        # As decorator
        @handler.catch(fallback=default_value)
        def my_function():
            ...
    """

    def __init__(
        self,
        config: Optional[ErrorConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the error handler.

        Args:
            config: Error handler configuration
            bus: AgentBus for event emission
        """
        self.config = config or ErrorConfig()
        self.bus = bus or AgentBus()

        # Error aggregation
        self._error_counts: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

        # Recovery callbacks
        self._recovery_handlers: Dict[ErrorCode, Callable] = {}

        # Statistics
        self._stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "unrecovered_errors": 0,
            "by_code": {},
        }

    def handle(
        self,
        error: Union[ResearchError, Exception],
        context: Optional[ErrorContext] = None,
    ) -> None:
        """
        Handle an error.

        Args:
            error: Error to handle
            context: Optional error context
        """
        # Convert to ResearchError if needed
        if not isinstance(error, ResearchError):
            error = ResearchError(
                code=ErrorCode.INTERNAL,
                message=str(error),
                context=context,
                cause=error,
            )

        # Update stats
        self._stats["total_errors"] += 1
        code_name = error.code.name
        self._stats["by_code"][code_name] = self._stats["by_code"].get(code_name, 0) + 1

        # Check aggregation
        if self.config.enable_aggregation:
            if self._is_rate_limited(error):
                return  # Skip if rate limited

        # Emit to bus
        if self.config.emit_to_bus:
            self._emit_error(error)

        # Attempt recovery
        if self.config.enable_recovery and error.recoverable:
            if self._try_recover(error):
                self._stats["recovered_errors"] += 1
                return

        self._stats["unrecovered_errors"] += 1

    def with_retry(
        self,
        func: Callable[[], T],
        context: Optional[ErrorContext] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            context: Error context
            retry_config: Optional retry configuration

        Returns:
            Function result

        Raises:
            ResearchError: If all retries fail
        """
        config = retry_config or self.config.retry_config
        last_error: Optional[Exception] = None

        for attempt in range(config.max_attempts):
            try:
                return func()

            except ResearchError as e:
                last_error = e

                # Check if should retry
                if e.code not in config.retry_on:
                    raise

                if attempt < config.max_attempts - 1:
                    delay = self._calculate_delay(attempt, config)
                    time.sleep(delay / 1000)  # Convert to seconds

            except Exception as e:
                last_error = e

                if attempt < config.max_attempts - 1:
                    delay = self._calculate_delay(attempt, config)
                    time.sleep(delay / 1000)

        # All retries failed
        raise ResearchError(
            code=ErrorCode.TIMEOUT,
            message=f"Operation failed after {config.max_attempts} attempts",
            context=context,
            cause=last_error,
        )

    def catch(
        self,
        fallback: Any = None,
        error_codes: Optional[List[ErrorCode]] = None,
        reraise: bool = False,
    ) -> Callable:
        """
        Decorator to catch and handle errors.

        Args:
            fallback: Fallback value on error
            error_codes: Specific codes to catch
            reraise: Whether to reraise after handling

        Example:
            @handler.catch(fallback=[])
            def search():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except ResearchError as e:
                    if error_codes and e.code not in error_codes:
                        raise

                    self.handle(e)

                    if reraise:
                        raise

                    return fallback

                except Exception as e:
                    research_error = ResearchError(
                        code=ErrorCode.INTERNAL,
                        message=str(e),
                        context=ErrorContext(operation=func.__name__),
                        cause=e,
                    )
                    self.handle(research_error)

                    if reraise:
                        raise research_error

                    return fallback

            return wrapper
        return decorator

    def register_recovery(
        self,
        code: ErrorCode,
        handler: Callable[[ResearchError], bool],
    ) -> None:
        """
        Register a recovery handler for an error code.

        Args:
            code: Error code
            handler: Recovery handler function
        """
        self._recovery_handlers[code] = handler

    def get_stats(self) -> Dict[str, Any]:
        """Get error handler statistics."""
        return {
            **self._stats,
            "recovery_rate": (
                self._stats["recovered_errors"] / self._stats["total_errors"]
                if self._stats["total_errors"] > 0 else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "unrecovered_errors": 0,
            "by_code": {},
        }

    def _is_rate_limited(self, error: ResearchError) -> bool:
        """Check if error should be rate limited."""
        with self._lock:
            key = f"{error.code.name}:{error.context.operation}"
            now = time.time()
            window_start = now - self.config.aggregation_window_seconds

            if key not in self._error_counts:
                self._error_counts[key] = []

            # Remove old entries
            self._error_counts[key] = [
                ts for ts in self._error_counts[key]
                if ts > window_start
            ]

            # Check limit
            if len(self._error_counts[key]) >= self.config.max_errors_per_window:
                return True

            self._error_counts[key].append(now)
            return False

    def _try_recover(self, error: ResearchError) -> bool:
        """Try to recover from an error."""
        handler = self._recovery_handlers.get(error.code)
        if handler:
            try:
                return handler(error)
            except Exception:
                return False
        return False

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = config.initial_delay_ms * (config.exponential_base ** attempt)
        delay = min(delay, config.max_delay_ms)

        if config.jitter:
            import random
            delay = delay * (0.5 + random.random())

        return delay

    def _emit_error(self, error: ResearchError) -> None:
        """Emit error event to bus."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": error.timestamp,
            "iso": datetime.fromtimestamp(error.timestamp, tz=timezone.utc).isoformat() + "Z",
            "topic": "a2a.research.error.handle",
            "kind": "error",
            "level": error.severity.value,
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": error.to_dict(),
        }

        if self.config.log_stack_traces and error.stack_trace:
            event["data"]["stack_trace"] = error.stack_trace

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ============================================================================
# Global Error Handler
# ============================================================================


_default_handler: Optional[ErrorHandler] = None


def get_handler(config: Optional[ErrorConfig] = None) -> ErrorHandler:
    """Get the default error handler."""
    global _default_handler
    if _default_handler is None:
        _default_handler = ErrorHandler(config)
    return _default_handler


def handle(error: Union[ResearchError, Exception], context: Optional[ErrorContext] = None) -> None:
    """Handle an error using the default handler."""
    get_handler().handle(error, context)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Error Handler."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Error Handler (Step 35)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show error statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run error handling demo")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test error handling")
    test_parser.add_argument("--code", default="INTERNAL", help="Error code")
    test_parser.add_argument("--message", default="Test error", help="Error message")

    args = parser.parse_args()

    handler = ErrorHandler()

    if args.command == "stats":
        stats = handler.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Error Handler Statistics:")
            print(f"  Total Errors: {stats['total_errors']}")
            print(f"  Recovered: {stats['recovered_errors']}")
            print(f"  Unrecovered: {stats['unrecovered_errors']}")
            print(f"  Recovery Rate: {stats['recovery_rate']:.1%}")

    elif args.command == "demo":
        print("Running error handling demo...\n")

        # Simple error handling
        error = ResearchError(
            code=ErrorCode.SEARCH_QUERY_INVALID,
            message="Invalid search query",
            context=ErrorContext(operation="search", metadata={"query": "test"}),
        )
        handler.handle(error)
        print(f"Handled error: {error}")

        # With decorator
        @handler.catch(fallback="default")
        def risky_operation():
            raise ValueError("Something went wrong")

        result = risky_operation()
        print(f"Caught error, got fallback: {result}")

        # With retry
        attempt_count = [0]

        @handler.catch(fallback=None)
        def retry_demo():
            return handler.with_retry(
                lambda: attempt_with_failure(attempt_count),
                context=ErrorContext(operation="retry_demo"),
            )

        def attempt_with_failure(count):
            count[0] += 1
            if count[0] < 3:
                raise NetworkError("Connection failed")
            return "Success!"

        result = retry_demo()
        print(f"Retry result: {result}")

        print("\nDemo complete.")

    elif args.command == "test":
        code = ErrorCode[args.code]
        error = ResearchError(
            code=code,
            message=args.message,
            context=ErrorContext(operation="test"),
        )
        handler.handle(error)
        print(f"Error handled: {error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
