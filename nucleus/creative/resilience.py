"""
Resilience Module for Creative Section
=======================================

Provides retry, circuit breaker, timeout, and fallback patterns.
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, ParamSpec
from enum import Enum, auto

T = TypeVar("T")
P = ParamSpec("P")


class ResilienceError(Exception):
    """Base exception for resilience errors."""
    pass


class RetryExhaustedError(ResilienceError):
    """All retry attempts exhausted."""

    def __init__(self, message: str, attempts: int, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class CircuitBreakerOpenError(ResilienceError):
    """Circuit breaker is open, operation not attempted."""

    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)


class TimeoutExceededError(ResilienceError):
    """Operation timed out."""

    def __init__(self, message: str, timeout: float):
        super().__init__(message)
        self.timeout = timeout


class FallbackError(ResilienceError):
    """Fallback also failed."""

    def __init__(self, message: str, primary_error: Exception, fallback_error: Exception):
        super().__init__(message)
        self.primary_error = primary_error
        self.fallback_error = fallback_error


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failures = 0
        elif self.state == CircuitState.CLOSED:
            self._failures = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request is allowed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True
        return False


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error = None
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            raise RetryExhaustedError(
                f"Failed after {max_attempts} attempts",
                max_attempts,
                last_error,
            )

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error = None
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise RetryExhaustedError(
                f"Failed after {max_attempts} attempts",
                max_attempts,
                last_error,
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Circuit breaker decorator."""
    cb = CircuitBreaker(failure_threshold, recovery_timeout)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not cb.allow_request():
                raise CircuitBreakerOpenError()
            try:
                result = await func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure()
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not cb.allow_request():
                raise CircuitBreakerOpenError()
            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def timeout(seconds: float) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Timeout decorator for async functions."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except asyncio.TimeoutError:
                raise TimeoutExceededError(
                    f"Operation timed out after {seconds}s",
                    seconds,
                )

        return wrapper

    return decorator


def fallback(
    fallback_func: Callable[..., T],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Fallback decorator."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as primary_error:
                try:
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    raise FallbackError(
                        "Both primary and fallback failed",
                        primary_error,
                        fallback_error,
                    )

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as primary_error:
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    raise FallbackError(
                        "Both primary and fallback failed",
                        primary_error,
                        fallback_error,
                    )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def resilient(
    max_retries: int = 3,
    timeout_s: float = 30.0,
    circuit_threshold: int = 5,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Combined resilience decorator with retry, timeout, and circuit breaker."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        wrapped = retry(max_retries)(func)
        wrapped = timeout(timeout_s)(wrapped)
        wrapped = circuit_breaker(circuit_threshold)(wrapped)
        return wrapped

    return decorator
