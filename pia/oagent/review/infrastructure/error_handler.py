#!/usr/bin/env python3
"""
Review Error Handler (Step 185)

Comprehensive error handling system for the Review Agent with error
classification, recovery strategies, and reporting.

PBTSO Phase: VERIFY, OBSERVE
Bus Topics: review.error.occurred, review.error.recovered, review.error.fatal

Error Features:
- Error classification by category and severity
- Automatic retry with backoff
- Recovery strategies
- Error aggregation and deduplication
- Integration with Omega veto for critical errors

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import functools
import json
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# Types
# ============================================================================

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issue, operation can continue
    MEDIUM = "medium"     # Significant issue, may affect results
    HIGH = "high"         # Serious issue, operation should stop
    CRITICAL = "critical" # Critical failure, requires Omega veto


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"       # Input validation errors
    PARSING = "parsing"             # File/data parsing errors
    ANALYSIS = "analysis"           # Analysis/processing errors
    NETWORK = "network"             # Network/connectivity errors
    TIMEOUT = "timeout"             # Timeout errors
    PERMISSION = "permission"       # Permission/access errors
    RESOURCE = "resource"           # Resource exhaustion
    CONFIGURATION = "configuration" # Configuration errors
    PLUGIN = "plugin"               # Plugin-related errors
    SECURITY = "security"           # Security-related errors
    INTERNAL = "internal"           # Internal/unexpected errors


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"               # Retry the operation
    SKIP = "skip"                 # Skip and continue
    FALLBACK = "fallback"         # Use fallback value/behavior
    ABORT = "abort"               # Abort the operation
    ESCALATE = "escalate"         # Escalate to Omega


@dataclass
class ErrorContext:
    """
    Context for an error occurrence.

    Attributes:
        operation: Operation that failed
        component: Component where error occurred
        review_id: Review ID if applicable
        file_path: File being processed if applicable
        input_data: Input that caused the error
        attempt: Retry attempt number
        timestamp: When the error occurred
    """
    operation: str = ""
    component: str = ""
    review_id: Optional[str] = None
    file_path: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    attempt: int = 1
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ErrorConfig:
    """
    Configuration for error handling.

    Attributes:
        max_retries: Maximum retry attempts
        base_delay_seconds: Base delay for exponential backoff
        max_delay_seconds: Maximum delay between retries
        escalate_on_critical: Auto-escalate critical errors to Omega
        dedupe_window_seconds: Window for error deduplication
        error_threshold: Errors before circuit breaker trips
        circuit_breaker_timeout: Circuit breaker recovery timeout
        capture_input: Capture input data on errors
        redact_sensitive: Redact sensitive data in error reports
    """
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    escalate_on_critical: bool = True
    dedupe_window_seconds: int = 60
    error_threshold: int = 10
    circuit_breaker_timeout: int = 300
    capture_input: bool = True
    redact_sensitive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Review Error Exception
# ============================================================================

class ReviewError(Exception):
    """
    Base exception for review errors.

    Attributes:
        message: Error message
        category: Error category
        severity: Error severity
        context: Error context
        cause: Original exception if wrapped
        recoverable: Whether error is recoverable
        retry_after: Suggested retry delay
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.error_id = str(uuid.uuid4())[:8]
        self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
        }

        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }

        if self.retry_after:
            result["retry_after"] = self.retry_after

        return result


# Specialized error classes
class ValidationError(ReviewError):
    """Validation error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class ParsingError(ReviewError):
    """Parsing error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class AnalysisError(ReviewError):
    """Analysis error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.ANALYSIS,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class TimeoutError(ReviewError):
    """Timeout error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            **kwargs
        )


class SecurityError(ReviewError):
    """Security error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


# ============================================================================
# Error Handler
# ============================================================================

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    traceback: str
    timestamp: float
    recovered: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "traceback": self.traceback,
            "timestamp": self.timestamp,
            "recovered": self.recovered,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
        }


class ErrorHandler:
    """
    Comprehensive error handler for the Review Agent.

    Example:
        handler = ErrorHandler()

        # Handle an error
        try:
            risky_operation()
        except Exception as e:
            result = await handler.handle(e, context=ErrorContext(
                operation="risky_operation",
                component="analysis",
            ))

        # Decorator for automatic handling
        @handler.catch(retries=3)
        async def my_function():
            ...
    """

    BUS_TOPICS = {
        "occurred": "review.error.occurred",
        "recovered": "review.error.recovered",
        "fatal": "review.error.fatal",
    }

    def __init__(
        self,
        config: Optional[ErrorConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the error handler.

        Args:
            config: Error handling configuration
            bus_path: Path to event bus file
        """
        self.config = config or ErrorConfig()
        self.bus_path = bus_path or self._get_bus_path()

        # Error tracking
        self._errors: List[ErrorRecord] = []
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._last_errors: Dict[str, float] = {}

        # Circuit breaker state
        self._circuit_open = False
        self._circuit_opened_at: Optional[float] = None

        # Callbacks
        self._recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "error") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "error-handler",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _should_dedupe(self, error: Exception, context: ErrorContext) -> bool:
        """Check if error should be deduplicated."""
        key = f"{type(error).__name__}:{context.operation}:{context.component}"
        last_time = self._last_errors.get(key, 0)

        if time.time() - last_time < self.config.dedupe_window_seconds:
            return True

        self._last_errors[key] = time.time()
        return False

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operations."""
        if not self._circuit_open:
            return True

        # Check if timeout has passed
        if self._circuit_opened_at:
            if time.time() - self._circuit_opened_at > self.config.circuit_breaker_timeout:
                self._circuit_open = False
                self._circuit_opened_at = None
                return True

        return False

    def _update_circuit_breaker(self) -> None:
        """Update circuit breaker state."""
        recent_errors = sum(
            1 for ts in self._last_errors.values()
            if time.time() - ts < 60
        )

        if recent_errors >= self.config.error_threshold:
            self._circuit_open = True
            self._circuit_opened_at = time.time()

    def register_recovery_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[Exception, ErrorContext], Any],
    ) -> None:
        """
        Register a recovery handler for an error category.

        Args:
            category: Error category
            handler: Recovery handler function
        """
        self._recovery_handlers[category] = handler

    async def handle(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        strategy: Optional[RecoveryStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Handle an error.

        Args:
            error: The error to handle
            context: Error context
            strategy: Recovery strategy override

        Returns:
            Error handling result

        Emits:
            review.error.occurred
            review.error.recovered (if recovered)
            review.error.fatal (if critical)
        """
        ctx = context or ErrorContext()

        # Convert to ReviewError if needed
        if isinstance(error, ReviewError):
            review_error = error
            review_error.context = ctx
        else:
            review_error = ReviewError(
                message=str(error),
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.MEDIUM,
                context=ctx,
                cause=error,
            )

        # Check deduplication
        if self._should_dedupe(error, ctx):
            return {
                "error_id": review_error.error_id,
                "deduplicated": True,
                "recovered": False,
            }

        # Record error
        record = ErrorRecord(
            error_id=review_error.error_id,
            message=review_error.message,
            category=review_error.category,
            severity=review_error.severity,
            context=ctx,
            traceback=traceback.format_exc(),
            timestamp=time.time(),
        )
        self._errors.append(record)
        self._error_counts[review_error.category.value] += 1

        # Emit occurred event
        self._emit_event(self.BUS_TOPICS["occurred"], review_error.to_dict())

        # Determine recovery strategy
        if strategy is None:
            strategy = self._determine_strategy(review_error)

        record.recovery_strategy = strategy

        # Handle critical errors
        if review_error.severity == ErrorSeverity.CRITICAL:
            self._emit_event(self.BUS_TOPICS["fatal"], {
                "error": review_error.to_dict(),
                "escalate": self.config.escalate_on_critical,
            })

            if self.config.escalate_on_critical:
                strategy = RecoveryStrategy.ESCALATE

        # Attempt recovery
        recovered = False
        recovery_result = None

        if strategy == RecoveryStrategy.RETRY:
            # Will be handled by retry decorator
            pass
        elif strategy == RecoveryStrategy.SKIP:
            recovered = True
        elif strategy == RecoveryStrategy.FALLBACK:
            handler = self._recovery_handlers.get(review_error.category)
            if handler:
                try:
                    recovery_result = handler(error, ctx)
                    recovered = True
                except Exception:
                    pass
        elif strategy == RecoveryStrategy.ABORT:
            pass  # Let error propagate
        elif strategy == RecoveryStrategy.ESCALATE:
            # Omega escalation handled externally
            pass

        record.recovered = recovered

        if recovered:
            self._emit_event(self.BUS_TOPICS["recovered"], {
                "error_id": review_error.error_id,
                "strategy": strategy.value,
                "result": recovery_result,
            })

        # Update circuit breaker
        self._update_circuit_breaker()

        return {
            "error_id": review_error.error_id,
            "category": review_error.category.value,
            "severity": review_error.severity.value,
            "strategy": strategy.value,
            "recovered": recovered,
            "recovery_result": recovery_result,
        }

    def _determine_strategy(self, error: ReviewError) -> RecoveryStrategy:
        """Determine recovery strategy for an error."""
        if error.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE

        if not error.recoverable:
            return RecoveryStrategy.ABORT

        if error.category in (ErrorCategory.TIMEOUT, ErrorCategory.NETWORK):
            return RecoveryStrategy.RETRY

        if error.category == ErrorCategory.VALIDATION:
            return RecoveryStrategy.SKIP

        if error.category in self._recovery_handlers:
            return RecoveryStrategy.FALLBACK

        return RecoveryStrategy.ABORT

    def catch(
        self,
        retries: Optional[int] = None,
        exceptions: tuple = (Exception,),
        strategy: Optional[RecoveryStrategy] = None,
        fallback: Optional[Callable[[], T]] = None,
    ) -> Callable[[F], F]:
        """
        Decorator for automatic error handling.

        Args:
            retries: Number of retries (None = use config)
            exceptions: Exception types to catch
            strategy: Recovery strategy
            fallback: Fallback function

        Returns:
            Decorated function
        """
        max_retries = retries if retries is not None else self.config.max_retries

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_error = None

                for attempt in range(max_retries + 1):
                    # Check circuit breaker
                    if not self._check_circuit_breaker():
                        raise ReviewError(
                            "Circuit breaker open",
                            category=ErrorCategory.RESOURCE,
                            severity=ErrorSeverity.HIGH,
                            recoverable=False,
                        )

                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_error = e
                        context = ErrorContext(
                            operation=func.__name__,
                            component=func.__module__,
                            attempt=attempt + 1,
                        )

                        result = await self.handle(e, context=context, strategy=strategy)

                        if result["recovered"]:
                            if fallback:
                                return fallback()
                            return None

                        if attempt < max_retries:
                            delay = min(
                                self.config.base_delay_seconds * (2 ** attempt),
                                self.config.max_delay_seconds,
                            )
                            await asyncio.sleep(delay)

                # All retries exhausted
                if fallback:
                    return fallback()
                raise last_error

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.get_event_loop().run_until_complete(
                    async_wrapper(*args, **kwargs)
                )

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator

    def get_errors(
        self,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        since: Optional[float] = None,
    ) -> List[ErrorRecord]:
        """
        Get recorded errors.

        Args:
            category: Filter by category
            severity: Filter by severity
            since: Filter by timestamp

        Returns:
            List of error records
        """
        errors = self._errors

        if category:
            errors = [e for e in errors if e.category == category]

        if severity:
            errors = [e for e in errors if e.severity == severity]

        if since:
            errors = [e for e in errors if e.timestamp >= since]

        return errors

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        now = time.time()
        recent = [e for e in self._errors if now - e.timestamp < 3600]

        return {
            "total_errors": len(self._errors),
            "recent_errors": len(recent),
            "by_category": dict(self._error_counts),
            "by_severity": {
                s.value: sum(1 for e in self._errors if e.severity == s)
                for s in ErrorSeverity
            },
            "recovered_count": sum(1 for e in self._errors if e.recovered),
            "circuit_breaker_open": self._circuit_open,
        }

    def clear_errors(self, before: Optional[float] = None) -> int:
        """
        Clear error records.

        Args:
            before: Clear errors before this timestamp

        Returns:
            Number of errors cleared
        """
        if before:
            original = len(self._errors)
            self._errors = [e for e in self._errors if e.timestamp >= before]
            return original - len(self._errors)
        else:
            count = len(self._errors)
            self._errors.clear()
            self._error_counts.clear()
            return count

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        stats = self.get_stats()
        status = {
            "agent": "error-handler",
            "healthy": not self._circuit_open,
            "total_errors": stats["total_errors"],
            "recent_errors": stats["recent_errors"],
            "circuit_breaker": "open" if self._circuit_open else "closed",
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
    """CLI entry point for Error Handler."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Error Handler (Step 185)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Stats command
    subparsers.add_parser("stats", help="Show error statistics")

    # List command
    list_parser = subparsers.add_parser("list", help="List recent errors")
    list_parser.add_argument("--category", help="Filter by category")
    list_parser.add_argument("--severity", help="Filter by severity")
    list_parser.add_argument("--limit", type=int, default=10, help="Max errors to show")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear errors")
    clear_parser.add_argument("--before", type=float, help="Clear before timestamp")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    handler = ErrorHandler()

    if args.command == "stats":
        stats = handler.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Error Statistics")
            print(f"  Total Errors: {stats['total_errors']}")
            print(f"  Recent Errors (1h): {stats['recent_errors']}")
            print(f"  Recovered: {stats['recovered_count']}")
            print(f"  Circuit Breaker: {'OPEN' if stats['circuit_breaker_open'] else 'closed'}")
            print("  By Category:")
            for cat, count in stats["by_category"].items():
                print(f"    {cat}: {count}")

    elif args.command == "list":
        category = ErrorCategory(args.category) if args.category else None
        severity = ErrorSeverity(args.severity) if args.severity else None

        errors = handler.get_errors(category=category, severity=severity)
        errors = errors[-args.limit:]

        if args.json:
            print(json.dumps([e.to_dict() for e in errors], indent=2))
        else:
            print(f"Recent Errors: {len(errors)}")
            for e in errors:
                ts = datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  [{ts}] {e.severity.value.upper()} {e.category.value}: {e.message}")

    elif args.command == "clear":
        count = handler.clear_errors(before=args.before)
        print(f"Cleared {count} errors")

    else:
        # Default: show status
        status = handler.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Error Handler: {status['total_errors']} errors, CB: {status['circuit_breaker']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
