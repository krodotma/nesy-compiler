#!/usr/bin/env python3
"""
handler.py - Deploy Error Handler (Step 235)

PBTSO Phase: VERIFY, ITERATE
A2A Integration: Comprehensive error handling via deploy.error.*

Provides:
- ErrorSeverity: Error severity levels
- ErrorCategory: Error categories
- DeployError: Deploy error exception
- ErrorContext: Error context information
- ErrorRecoveryAction: Recovery action definitions
- DeployErrorHandler: Main error handler

Bus Topics:
- deploy.error.occurred
- deploy.error.recovered
- deploy.error.escalated
- deploy.error.circuit.open
- deploy.error.circuit.close

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar


# ==============================================================================
# Bus Emission Helper with File Locking (DKIN v30)
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
    actor: str = "error-handler"
) -> str:
    """Emit an event to the Pluribus bus with fcntl.flock() file locking."""
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

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    BUILD = "build"
    DEPLOYMENT = "deployment"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    DEPENDENCY = "dependency"
    INTERNAL = "internal"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    SKIP = "skip"
    ABORT = "abort"
    MANUAL = "manual"
    CIRCUIT_BREAK = "circuit_break"
    FAILOVER = "failover"
    COMPENSATE = "compensate"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ErrorContext:
    """
    Error context information.

    Attributes:
        error_id: Unique error identifier
        deployment_id: Associated deployment
        service_name: Service name
        environment: Target environment
        operation: Operation that failed
        phase: Deployment phase
        attempt: Attempt number
        max_attempts: Maximum attempts
        trace_id: Distributed trace ID
        user_id: User/actor identifier
        custom: Custom context fields
    """
    error_id: str
    deployment_id: str = ""
    service_name: str = ""
    environment: str = ""
    operation: str = ""
    phase: str = ""
    attempt: int = 1
    max_attempts: int = 3
    trace_id: str = ""
    user_id: str = ""
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "error_id": self.error_id,
            "deployment_id": self.deployment_id,
            "service_name": self.service_name,
            "environment": self.environment,
            "operation": self.operation,
            "phase": self.phase,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "trace_id": self.trace_id,
            "user_id": self.user_id,
        }
        result.update(self.custom)
        return {k: v for k, v in result.items() if v}


@dataclass
class ErrorRecoveryAction:
    """
    Recovery action definitions.

    Attributes:
        action_id: Action identifier
        strategy: Recovery strategy
        description: Action description
        handler: Handler function name
        params: Handler parameters
        timeout_s: Action timeout
        enabled: Whether action is enabled
    """
    action_id: str
    strategy: RecoveryStrategy
    description: str = ""
    handler: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_s: int = 60
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "strategy": self.strategy.value,
            "description": self.description,
            "handler": self.handler,
            "params": self.params,
            "timeout_s": self.timeout_s,
            "enabled": self.enabled,
        }


@dataclass
class ErrorRecord:
    """
    Recorded error information.

    Attributes:
        error_id: Unique error identifier
        timestamp: Error timestamp
        category: Error category
        severity: Error severity
        message: Error message
        exception_type: Exception type name
        exception_message: Exception message
        traceback: Full traceback
        context: Error context
        recovery_action: Applied recovery action
        recovered: Whether error was recovered
        recovery_time_ms: Time to recover
    """
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str = ""
    exception_message: str = ""
    traceback: str = ""
    context: Optional[ErrorContext] = None
    recovery_action: Optional[ErrorRecoveryAction] = None
    recovered: bool = False
    recovery_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat() + "Z",
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "traceback": self.traceback,
            "context": self.context.to_dict() if self.context else None,
            "recovery_action": self.recovery_action.to_dict() if self.recovery_action else None,
            "recovered": self.recovered,
            "recovery_time_ms": self.recovery_time_ms,
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state."""
    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: float = 0.0
    last_success: float = 0.0
    opened_at: float = 0.0
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_s: int = 60

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Deploy Error Exception
# ==============================================================================

class DeployError(Exception):
    """
    Deploy error exception.

    Custom exception for deployment errors with category and severity.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[BaseException] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.cause = cause
        self.recoverable = recoverable
        self.error_id = f"err-{uuid.uuid4().hex[:12]}"
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "context": self.context.to_dict() if self.context else None,
        }


# Specific error types
class BuildError(DeployError):
    """Build phase error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.BUILD, **kwargs)


class ConfigurationError(DeployError):
    """Configuration error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)


class NetworkError(DeployError):
    """Network error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class TimeoutError(DeployError):
    """Timeout error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, **kwargs)


class ValidationError(DeployError):
    """Validation error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class ResourceError(DeployError):
    """Resource error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, **kwargs)


# ==============================================================================
# Deploy Error Handler (Step 235)
# ==============================================================================

E = TypeVar("E", bound=BaseException)


class DeployErrorHandler:
    """
    Deploy Error Handler - comprehensive error handling.

    PBTSO Phase: VERIFY, ITERATE

    Responsibilities:
    - Capture and categorize errors
    - Determine appropriate recovery strategies
    - Execute recovery actions
    - Implement circuit breaker pattern
    - Track error history and patterns

    Example:
        >>> handler = DeployErrorHandler()
        >>> try:
        ...     # deployment code
        ... except Exception as e:
        ...     result = await handler.handle_error(e, context)
        ...     if result.recovered:
        ...         print("Error recovered")
    """

    BUS_TOPICS = {
        "occurred": "deploy.error.occurred",
        "recovered": "deploy.error.recovered",
        "escalated": "deploy.error.escalated",
        "circuit_open": "deploy.error.circuit.open",
        "circuit_close": "deploy.error.circuit.close",
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    # Error categorization rules
    CATEGORY_RULES = {
        "ConnectionError": ErrorCategory.NETWORK,
        "TimeoutError": ErrorCategory.TIMEOUT,
        "asyncio.TimeoutError": ErrorCategory.TIMEOUT,
        "PermissionError": ErrorCategory.AUTHORIZATION,
        "FileNotFoundError": ErrorCategory.RESOURCE,
        "ValueError": ErrorCategory.VALIDATION,
        "KeyError": ErrorCategory.CONFIGURATION,
        "OSError": ErrorCategory.RESOURCE,
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "error-handler",
        max_error_history: int = 1000,
    ):
        """
        Initialize the error handler.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            max_error_history: Maximum errors to keep in history
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "errors"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.max_error_history = max_error_history

        # Error history
        self._errors: List[ErrorRecord] = []

        # Circuit breakers
        self._circuits: Dict[str, CircuitBreakerState] = {}

        # Recovery handlers
        self._recovery_handlers: Dict[str, Callable] = {}

        # Error listeners
        self._listeners: List[Callable[[ErrorRecord], None]] = []

        self._load_state()

    def handle(
        self,
        exception: BaseException,
        context: Optional[ErrorContext] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
    ) -> ErrorRecord:
        """
        Handle an exception synchronously.

        Args:
            exception: The exception to handle
            context: Error context
            category: Override category
            severity: Override severity

        Returns:
            ErrorRecord
        """
        # Determine category
        if category is None:
            category = self._categorize_error(exception)

        # Determine severity
        if severity is None:
            severity = self._determine_severity(exception, category)

        # Create error context if not provided
        if context is None:
            context = ErrorContext(error_id=f"err-{uuid.uuid4().hex[:12]}")

        # Create error record
        record = ErrorRecord(
            error_id=context.error_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback=traceback.format_exc(),
            context=context,
        )

        # Store error
        self._errors.append(record)
        if len(self._errors) > self.max_error_history:
            self._errors = self._errors[-self.max_error_history:]

        # Emit bus event
        _emit_bus_event(
            self.BUS_TOPICS["occurred"],
            {
                "error_id": record.error_id,
                "category": record.category.value,
                "severity": record.severity.value,
                "message": record.message,
                "exception_type": record.exception_type,
                "deployment_id": context.deployment_id if context else "",
                "service_name": context.service_name if context else "",
            },
            level="error",
            actor=self.actor_id,
        )

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(record)
            except Exception:
                pass

        self._save_state()
        return record

    async def handle_async(
        self,
        exception: BaseException,
        context: Optional[ErrorContext] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        auto_recover: bool = True,
    ) -> ErrorRecord:
        """
        Handle an exception asynchronously with optional recovery.

        Args:
            exception: The exception to handle
            context: Error context
            category: Override category
            severity: Override severity
            auto_recover: Attempt automatic recovery

        Returns:
            ErrorRecord
        """
        record = self.handle(exception, context, category, severity)

        if auto_recover:
            record = await self.attempt_recovery(record)

        return record

    async def attempt_recovery(
        self,
        record: ErrorRecord,
        strategy: Optional[RecoveryStrategy] = None,
    ) -> ErrorRecord:
        """
        Attempt to recover from an error.

        Args:
            record: Error record
            strategy: Override recovery strategy

        Returns:
            Updated ErrorRecord
        """
        if strategy is None:
            strategy = self._determine_recovery_strategy(record)

        action = ErrorRecoveryAction(
            action_id=f"action-{uuid.uuid4().hex[:8]}",
            strategy=strategy,
            description=f"Auto-recovery for {record.category.value} error",
        )

        record.recovery_action = action
        start_time = time.time()

        try:
            # Get handler for strategy
            handler_name = f"_recover_{strategy.value}"
            handler = getattr(self, handler_name, None)

            if handler:
                result = await handler(record)
                record.recovered = result
            else:
                # Check custom handlers
                if strategy.value in self._recovery_handlers:
                    custom_handler = self._recovery_handlers[strategy.value]
                    if asyncio.iscoroutinefunction(custom_handler):
                        record.recovered = await custom_handler(record)
                    else:
                        record.recovered = custom_handler(record)
                else:
                    record.recovered = False

            record.recovery_time_ms = (time.time() - start_time) * 1000

            if record.recovered:
                _emit_bus_event(
                    self.BUS_TOPICS["recovered"],
                    {
                        "error_id": record.error_id,
                        "strategy": strategy.value,
                        "recovery_time_ms": record.recovery_time_ms,
                    },
                    actor=self.actor_id,
                )
            else:
                _emit_bus_event(
                    self.BUS_TOPICS["escalated"],
                    {
                        "error_id": record.error_id,
                        "category": record.category.value,
                        "severity": record.severity.value,
                        "strategy_attempted": strategy.value,
                    },
                    level="error",
                    actor=self.actor_id,
                )

        except Exception as e:
            record.recovered = False
            record.recovery_time_ms = (time.time() - start_time) * 1000

        self._save_state()
        return record

    async def _recover_retry(self, record: ErrorRecord) -> bool:
        """Retry recovery strategy."""
        if not record.context:
            return False

        if record.context.attempt >= record.context.max_attempts:
            return False

        # Exponential backoff
        delay = min(30, 2 ** record.context.attempt)
        await asyncio.sleep(delay)

        return True  # Signal retry is possible

    async def _recover_skip(self, record: ErrorRecord) -> bool:
        """Skip recovery strategy."""
        return True  # Skip always succeeds

    async def _recover_abort(self, record: ErrorRecord) -> bool:
        """Abort recovery strategy."""
        return False  # Abort means no recovery

    async def _recover_circuit_break(self, record: ErrorRecord) -> bool:
        """Circuit break recovery strategy."""
        if not record.context:
            return False

        circuit_name = record.context.service_name or "default"
        self.record_circuit_failure(circuit_name)
        return False  # Don't recover, but circuit is now open

    def _categorize_error(self, exception: BaseException) -> ErrorCategory:
        """Categorize an error based on exception type."""
        exc_type = type(exception).__name__

        # Check if it's already a DeployError
        if isinstance(exception, DeployError):
            return exception.category

        # Check rules
        if exc_type in self.CATEGORY_RULES:
            return self.CATEGORY_RULES[exc_type]

        # Check base classes
        for parent in type(exception).__mro__:
            parent_name = parent.__name__
            if parent_name in self.CATEGORY_RULES:
                return self.CATEGORY_RULES[parent_name]

        return ErrorCategory.UNKNOWN

    def _determine_severity(
        self,
        exception: BaseException,
        category: ErrorCategory,
    ) -> ErrorSeverity:
        """Determine error severity."""
        # Check if it's a DeployError
        if isinstance(exception, DeployError):
            return exception.severity

        # Category-based severity
        severity_map = {
            ErrorCategory.AUTHENTICATION: ErrorSeverity.HIGH,
            ErrorCategory.AUTHORIZATION: ErrorSeverity.HIGH,
            ErrorCategory.INTERNAL: ErrorSeverity.CRITICAL,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.VALIDATION: ErrorSeverity.LOW,
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
        }

        return severity_map.get(category, ErrorSeverity.MEDIUM)

    def _determine_recovery_strategy(self, record: ErrorRecord) -> RecoveryStrategy:
        """Determine appropriate recovery strategy."""
        # Category-based default strategies
        strategy_map = {
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorCategory.RESOURCE: RecoveryStrategy.RETRY,
            ErrorCategory.VALIDATION: RecoveryStrategy.ABORT,
            ErrorCategory.CONFIGURATION: RecoveryStrategy.ABORT,
            ErrorCategory.AUTHENTICATION: RecoveryStrategy.ABORT,
            ErrorCategory.AUTHORIZATION: RecoveryStrategy.ABORT,
            ErrorCategory.INTERNAL: RecoveryStrategy.CIRCUIT_BREAK,
            ErrorCategory.EXTERNAL: RecoveryStrategy.CIRCUIT_BREAK,
        }

        # Check if already exceeded retries
        if record.context and record.context.attempt >= record.context.max_attempts:
            return RecoveryStrategy.ABORT

        return strategy_map.get(record.category, RecoveryStrategy.MANUAL)

    # ==================== Circuit Breaker ====================

    def get_circuit(self, name: str) -> CircuitBreakerState:
        """Get or create a circuit breaker."""
        if name not in self._circuits:
            self._circuits[name] = CircuitBreakerState(name=name)
        return self._circuits[name]

    def is_circuit_open(self, name: str) -> bool:
        """Check if circuit is open (blocking calls)."""
        circuit = self.get_circuit(name)

        if circuit.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - circuit.opened_at >= circuit.timeout_s:
                circuit.state = CircuitState.HALF_OPEN
                return False
            return True

        return False

    def record_circuit_failure(self, name: str) -> None:
        """Record a failure for circuit breaker."""
        circuit = self.get_circuit(name)
        circuit.failure_count += 1
        circuit.last_failure = time.time()

        if circuit.state == CircuitState.HALF_OPEN:
            # Fail immediately back to open
            circuit.state = CircuitState.OPEN
            circuit.opened_at = time.time()

            _emit_bus_event(
                self.BUS_TOPICS["circuit_open"],
                {"circuit": name, "failure_count": circuit.failure_count},
                level="warning",
                actor=self.actor_id,
            )

        elif circuit.failure_count >= circuit.failure_threshold:
            circuit.state = CircuitState.OPEN
            circuit.opened_at = time.time()

            _emit_bus_event(
                self.BUS_TOPICS["circuit_open"],
                {"circuit": name, "failure_count": circuit.failure_count},
                level="warning",
                actor=self.actor_id,
            )

    def record_circuit_success(self, name: str) -> None:
        """Record a success for circuit breaker."""
        circuit = self.get_circuit(name)
        circuit.success_count += 1
        circuit.last_success = time.time()

        if circuit.state == CircuitState.HALF_OPEN:
            if circuit.success_count >= circuit.success_threshold:
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                circuit.success_count = 0

                _emit_bus_event(
                    self.BUS_TOPICS["circuit_close"],
                    {"circuit": name},
                    actor=self.actor_id,
                )

        elif circuit.state == CircuitState.CLOSED:
            # Reset failure count on success
            circuit.failure_count = 0

    def reset_circuit(self, name: str) -> None:
        """Reset a circuit breaker."""
        circuit = self.get_circuit(name)
        circuit.state = CircuitState.CLOSED
        circuit.failure_count = 0
        circuit.success_count = 0

        _emit_bus_event(
            self.BUS_TOPICS["circuit_close"],
            {"circuit": name, "reason": "manual_reset"},
            actor=self.actor_id,
        )

    # ==================== Registration ====================

    def register_recovery_handler(
        self,
        strategy: str,
        handler: Callable[[ErrorRecord], bool],
    ) -> None:
        """Register a custom recovery handler."""
        self._recovery_handlers[strategy] = handler

    def register_listener(
        self,
        listener: Callable[[ErrorRecord], None],
    ) -> None:
        """Register an error listener."""
        self._listeners.append(listener)

    # ==================== Query ====================

    def get_errors(
        self,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        service_name: Optional[str] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[ErrorRecord]:
        """Get error history with filters."""
        cutoff = time.time() - (hours * 3600)
        errors = [e for e in self._errors if e.timestamp >= cutoff]

        if category:
            errors = [e for e in errors if e.category == category]

        if severity:
            errors = [e for e in errors if e.severity == severity]

        if service_name:
            errors = [
                e for e in errors
                if e.context and e.context.service_name == service_name
            ]

        return errors[-limit:]

    def get_error(self, error_id: str) -> Optional[ErrorRecord]:
        """Get a specific error by ID."""
        for error in self._errors:
            if error.error_id == error_id:
                return error
        return None

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary statistics."""
        cutoff = time.time() - (hours * 3600)
        errors = [e for e in self._errors if e.timestamp >= cutoff]

        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        recovered = 0
        unrecovered = 0

        for error in errors:
            cat = error.category.value
            sev = error.severity.value

            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

            if error.recovered:
                recovered += 1
            else:
                unrecovered += 1

        return {
            "period_hours": hours,
            "total_errors": len(errors),
            "by_category": by_category,
            "by_severity": by_severity,
            "recovered": recovered,
            "unrecovered": unrecovered,
            "recovery_rate": recovered / len(errors) if errors else 0.0,
        }

    def list_circuits(self) -> List[CircuitBreakerState]:
        """List all circuit breakers."""
        return list(self._circuits.values())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "circuits": {n: c.to_dict() for n, c in self._circuits.items()},
            "recent_errors": [e.to_dict() for e in self._errors[-100:]],
        }
        state_file = self.state_dir / "error_handler_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "error_handler_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for name, data in state.get("circuits", {}).items():
                data["state"] = CircuitState(data.get("state", "closed"))
                self._circuits[name] = CircuitBreakerState(**data)

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# Context Manager for Error Handling
# ==============================================================================

class error_handler_context:
    """Context manager for automatic error handling."""

    def __init__(
        self,
        handler: DeployErrorHandler,
        context: Optional[ErrorContext] = None,
        auto_recover: bool = True,
        reraise: bool = True,
    ):
        self.handler = handler
        self.context = context
        self.auto_recover = auto_recover
        self.reraise = reraise
        self.record: Optional[ErrorRecord] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.record = self.handler.handle(exc_val, self.context)

            if not self.reraise:
                return True  # Suppress exception

        return False


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for error handler."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Error Handler (Step 235)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List errors")
    list_parser.add_argument("--category", "-c", choices=[c.value for c in ErrorCategory])
    list_parser.add_argument("--severity", "-s", choices=[s.value for s in ErrorSeverity])
    list_parser.add_argument("--service", help="Filter by service")
    list_parser.add_argument("--hours", type=int, default=24, help="Time window")
    list_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # summary command
    summary_parser = subparsers.add_parser("summary", help="Error summary")
    summary_parser.add_argument("--hours", type=int, default=24, help="Time window")
    summary_parser.add_argument("--json", action="store_true", help="JSON output")

    # circuits command
    circuits_parser = subparsers.add_parser("circuits", help="List circuit breakers")
    circuits_parser.add_argument("--json", action="store_true", help="JSON output")

    # reset-circuit command
    reset_parser = subparsers.add_parser("reset-circuit", help="Reset circuit breaker")
    reset_parser.add_argument("name", help="Circuit name")

    # simulate command (for testing)
    simulate_parser = subparsers.add_parser("simulate", help="Simulate an error")
    simulate_parser.add_argument("--category", "-c", default="internal")
    simulate_parser.add_argument("--severity", "-s", default="medium")
    simulate_parser.add_argument("--message", "-m", default="Simulated error")

    args = parser.parse_args()
    handler = DeployErrorHandler()

    if args.command == "list":
        category = ErrorCategory(args.category) if args.category else None
        severity = ErrorSeverity(args.severity) if args.severity else None

        errors = handler.get_errors(
            category=category,
            severity=severity,
            service_name=args.service,
            hours=args.hours,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([e.to_dict() for e in errors], indent=2))
        else:
            if not errors:
                print("No errors found")
            else:
                for e in errors:
                    ts = datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    status = "recovered" if e.recovered else "unrecovered"
                    print(f"{ts} [{e.severity.value:8}] [{e.category.value:12}] {e.message[:50]} ({status})")

        return 0

    elif args.command == "summary":
        summary = handler.get_error_summary(hours=args.hours)

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Error Summary ({args.hours}h)")
            print(f"  Total: {summary['total_errors']}")
            print(f"  Recovered: {summary['recovered']}")
            print(f"  Unrecovered: {summary['unrecovered']}")
            print(f"  Recovery Rate: {summary['recovery_rate']:.1%}")
            print("\nBy Category:")
            for cat, count in summary['by_category'].items():
                print(f"  {cat}: {count}")
            print("\nBy Severity:")
            for sev, count in summary['by_severity'].items():
                print(f"  {sev}: {count}")

        return 0

    elif args.command == "circuits":
        circuits = handler.list_circuits()

        if args.json:
            print(json.dumps([c.to_dict() for c in circuits], indent=2))
        else:
            if not circuits:
                print("No circuit breakers")
            else:
                for c in circuits:
                    print(f"{c.name}: {c.state.value} (failures: {c.failure_count})")

        return 0

    elif args.command == "reset-circuit":
        handler.reset_circuit(args.name)
        print(f"Reset circuit: {args.name}")
        return 0

    elif args.command == "simulate":
        context = ErrorContext(
            error_id=f"err-{uuid.uuid4().hex[:12]}",
            service_name="test-service",
        )

        error = DeployError(
            args.message,
            category=ErrorCategory(args.category),
            severity=ErrorSeverity(args.severity),
            context=context,
        )

        record = handler.handle(error, context)
        print(f"Simulated error: {record.error_id}")
        print(f"  Category: {record.category.value}")
        print(f"  Severity: {record.severity.value}")

        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
