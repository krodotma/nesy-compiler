"""
Observability Package for Creative Section
===========================================

Provides metrics, tracing, and structured logging.
"""

from __future__ import annotations

import time
import functools
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar
from enum import Enum
from contextlib import contextmanager

T = TypeVar("T")


# =============================================================================
# METRICS
# =============================================================================


@dataclass
class Counter:
    """A monotonically increasing counter."""
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    _value: float = field(default=0.0, init=False)

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter."""
        self._value += amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class Gauge:
    """A gauge that can go up and down."""
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    _value: float = field(default=0.0, init=False)

    def set(self, value: float) -> None:
        """Set the gauge value."""
        self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge."""
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge."""
        self._value -= amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class Histogram:
    """A histogram for tracking distributions."""
    name: str
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    labels: Dict[str, str] = field(default_factory=dict)
    _values: List[float] = field(default_factory=list, init=False)

    def observe(self, value: float) -> None:
        """Record an observation."""
        self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def sum(self) -> float:
        return sum(self._values) if self._values else 0.0

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


class MetricsCollector:
    """Collects and manages metrics."""

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

    def counter(self, name: str, **labels) -> Counter:
        """Get or create a counter."""
        key = f"{name}:{labels}"
        if key not in self._counters:
            self._counters[key] = Counter(name, labels)
        return self._counters[key]

    def gauge(self, name: str, **labels) -> Gauge:
        """Get or create a gauge."""
        key = f"{name}:{labels}"
        if key not in self._gauges:
            self._gauges[key] = Gauge(name, labels)
        return self._gauges[key]

    def histogram(self, name: str, **labels) -> Histogram:
        """Get or create a histogram."""
        key = f"{name}:{labels}"
        if key not in self._histograms:
            self._histograms[key] = Histogram(name, labels=labels)
        return self._histograms[key]

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": {k: v.value for k, v in self._counters.items()},
            "gauges": {k: v.value for k, v in self._gauges.items()},
            "histograms": {k: {"count": v.count, "sum": v.sum, "mean": v.mean} for k, v in self._histograms.items()},
        }


_metrics_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def timed(name: str, **labels) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to time a function and record in a histogram."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                get_metrics().histogram(name, **labels).observe(duration)

        return wrapper

    return decorator


# =============================================================================
# TRACING
# =============================================================================


@dataclass
class Span:
    """A trace span."""
    trace_id: str
    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes,
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now(timezone.utc)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000


class Tracer:
    """A simple tracer implementation."""

    def __init__(self, service_name: str = "creative"):
        self.service_name = service_name
        self._spans: List[Span] = []
        self._current_span: Optional[Span] = None

    @contextmanager
    def start_span(self, name: str, **attributes):
        """Start a new span."""
        trace_id = self._current_span.trace_id if self._current_span else str(uuid.uuid4())
        parent_span_id = self._current_span.span_id if self._current_span else None

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:8],
            name=name,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )

        previous_span = self._current_span
        self._current_span = span
        self._spans.append(span)

        try:
            yield span
        finally:
            span.end()
            self._current_span = previous_span


_tracer: Optional[Tracer] = None
_current_trace_id: Optional[str] = None


def get_tracer() -> Tracer:
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    tracer = get_tracer()
    if tracer._current_span:
        return tracer._current_span.trace_id
    return None


def traced(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to trace a function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with get_tracer().start_span(name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    raise

        return wrapper

    return decorator


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================


class LogLevel(Enum):
    """Log levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class StructuredLogger:
    """A structured logger with context."""

    def __init__(self, name: str):
        self.name = name
        self._context: Dict[str, Any] = {}
        self._logger = logging.getLogger(name)

    def bind(self, **context) -> "StructuredLogger":
        """Create a new logger with additional context."""
        new_logger = StructuredLogger(self.name)
        new_logger._context = {**self._context, **context}
        return new_logger

    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal log method."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.name,
            "logger": self.name,
            "message": message,
            **self._context,
            **kwargs,
        }

        trace_id = get_current_trace_id()
        if trace_id:
            record["trace_id"] = trace_id

        self._logger.log(level.value, str(record))

    def debug(self, message: str, **kwargs) -> None:
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log(LogLevel.CRITICAL, message, **kwargs)


_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]
