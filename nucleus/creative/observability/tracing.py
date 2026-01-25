"""
Tracing Module for Observability
================================

Provides distributed tracing primitives: Span, Tracer, and context
propagation utilities for tracking requests across service boundaries.
"""

from __future__ import annotations

import uuid
import functools
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class SpanContext:
    """
    Immutable context for span propagation.

    Contains the trace ID and span ID needed to link spans across
    process boundaries.

    Attributes:
        trace_id: The unique trace identifier.
        span_id: The unique span identifier within the trace.
        baggage: Key-value pairs propagated across spans.
    """
    trace_id: str
    span_id: str
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export span context as a dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "baggage": self.baggage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanContext":
        """Create SpanContext from a dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            baggage=data.get("baggage", {}),
        )

    def to_headers(self) -> Dict[str, str]:
        """
        Export span context as HTTP headers for propagation.

        Uses W3C Trace Context format.
        """
        return {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01",
            "tracestate": ",".join(f"{k}={v}" for k, v in self.baggage.items()),
        }

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["SpanContext"]:
        """
        Extract SpanContext from HTTP headers.

        Args:
            headers: Dictionary of HTTP headers.

        Returns:
            SpanContext if traceparent header exists, None otherwise.
        """
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) >= 3:
                trace_id = parts[1]
                span_id = parts[2]
                baggage = {}

                tracestate = headers.get("tracestate", "")
                if tracestate:
                    for item in tracestate.split(","):
                        if "=" in item:
                            k, v = item.split("=", 1)
                            baggage[k.strip()] = v.strip()

                return cls(trace_id=trace_id, span_id=span_id, baggage=baggage)
        except (ValueError, IndexError):
            pass

        return None


@dataclass
class Span:
    """
    A single span in a distributed trace.

    Represents a unit of work with timing information, attributes,
    and events. Spans can be nested to form a trace tree.

    Attributes:
        trace_id: The trace this span belongs to.
        span_id: Unique identifier for this span.
        name: Human-readable name for this operation.
        parent_span_id: ID of the parent span, if any.
        start_time: When the span started.
        end_time: When the span ended.
        attributes: Key-value metadata about this span.
        events: List of timestamped events within this span.
        status: Status of the span (OK, ERROR).

    Example:
        >>> span = Span(
        ...     trace_id="abc123",
        ...     span_id="def456",
        ...     name="http_request"
        ... )
        >>> span.set_attribute("http.method", "GET")
        >>> span.add_event("request_started")
        >>> span.end()
    """
    trace_id: str
    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "UNSET"
    status_message: Optional[str] = None

    def add_event(self, name: str, **attributes: Any) -> None:
        """
        Add a timestamped event to the span.

        Args:
            name: Name of the event.
            **attributes: Additional event attributes.
        """
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes,
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set a span attribute.

        Args:
            key: Attribute key.
            value: Attribute value.
        """
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Set multiple span attributes.

        Args:
            attributes: Dictionary of attributes to set.
        """
        self.attributes.update(attributes)

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """
        Set the span status.

        Args:
            status: Status string (OK, ERROR).
            message: Optional status message.
        """
        self.status = status
        self.status_message = message

    def record_exception(self, exception: Exception) -> None:
        """
        Record an exception in the span.

        Args:
            exception: The exception to record.
        """
        self.add_event(
            "exception",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
        )
        self.set_status("ERROR", str(exception))

    def end(self) -> None:
        """End the span and record the end time."""
        if self.end_time is None:
            self.end_time = datetime.now(timezone.utc)
            if self.status == "UNSET":
                self.status = "OK"

    @property
    def duration_ms(self) -> float:
        """
        Calculate span duration in milliseconds.

        Returns:
            Duration in milliseconds, or 0 if span not ended.
        """
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def context(self) -> SpanContext:
        """Get the span context for propagation."""
        return SpanContext(trace_id=self.trace_id, span_id=self.span_id)

    def to_dict(self) -> Dict[str, Any]:
        """Export span as a dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "status_message": self.status_message,
        }

    def to_otlp(self) -> Dict[str, Any]:
        """
        Export span in OpenTelemetry Protocol format.

        Returns:
            Dictionary suitable for OTLP export.
        """
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id,
            "name": self.name,
            "startTimeUnixNano": int(self.start_time.timestamp() * 1e9),
            "endTimeUnixNano": int(self.end_time.timestamp() * 1e9) if self.end_time else None,
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in self.attributes.items()
            ],
            "events": [
                {
                    "name": e["name"],
                    "timeUnixNano": int(
                        datetime.fromisoformat(e["timestamp"]).timestamp() * 1e9
                    ),
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in e.get("attributes", {}).items()
                    ],
                }
                for e in self.events
            ],
            "status": {"code": 1 if self.status == "OK" else 2},
        }


class Tracer:
    """
    A tracer for creating and managing spans.

    The Tracer maintains the current span context and provides
    methods for creating new spans, either as root spans or as
    children of the current span.

    Attributes:
        service_name: Name of the service being traced.

    Example:
        >>> tracer = Tracer("my-service")
        >>> with tracer.start_span("operation") as span:
        ...     span.set_attribute("key", "value")
        ...     # do work
    """

    def __init__(self, service_name: str = "creative") -> None:
        """
        Initialize the tracer.

        Args:
            service_name: Name of the service for span metadata.
        """
        self.service_name = service_name
        self._spans: List[Span] = []
        self._current_span: Optional[Span] = None
        self._lock = threading.Lock()
        self._context_stack: List[Span] = []

    def _generate_id(self, length: int = 16) -> str:
        """Generate a random hex ID."""
        return uuid.uuid4().hex[:length]

    @contextmanager
    def start_span(
        self,
        name: str,
        parent_context: Optional[SpanContext] = None,
        **attributes: Any,
    ) -> Generator[Span, None, None]:
        """
        Start a new span as a context manager.

        Args:
            name: Name of the span.
            parent_context: Optional parent context for linking.
            **attributes: Initial span attributes.

        Yields:
            The new Span instance.
        """
        # Determine trace and parent IDs
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        elif self._current_span:
            trace_id = self._current_span.trace_id
            parent_span_id = self._current_span.span_id
        else:
            trace_id = self._generate_id(32)
            parent_span_id = None

        span = Span(
            trace_id=trace_id,
            span_id=self._generate_id(16),
            name=name,
            parent_span_id=parent_span_id,
            attributes={"service.name": self.service_name, **attributes},
        )

        with self._lock:
            previous_span = self._current_span
            self._current_span = span
            self._spans.append(span)

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            with self._lock:
                self._current_span = previous_span

    def start_span_no_context(
        self,
        name: str,
        parent_context: Optional[SpanContext] = None,
        **attributes: Any,
    ) -> Span:
        """
        Start a new span without using context manager.

        The caller is responsible for ending the span.

        Args:
            name: Name of the span.
            parent_context: Optional parent context for linking.
            **attributes: Initial span attributes.

        Returns:
            The new Span instance.
        """
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        elif self._current_span:
            trace_id = self._current_span.trace_id
            parent_span_id = self._current_span.span_id
        else:
            trace_id = self._generate_id(32)
            parent_span_id = None

        span = Span(
            trace_id=trace_id,
            span_id=self._generate_id(16),
            name=name,
            parent_span_id=parent_span_id,
            attributes={"service.name": self.service_name, **attributes},
        )

        with self._lock:
            self._spans.append(span)

        return span

    @property
    def current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._current_span

    def get_all_spans(self) -> List[Span]:
        """Get all recorded spans."""
        return list(self._spans)

    def clear_spans(self) -> None:
        """Clear all recorded spans."""
        with self._lock:
            self._spans.clear()

    def export_spans(self) -> List[Dict[str, Any]]:
        """Export all spans as dictionaries."""
        return [span.to_dict() for span in self._spans]

    def export_spans_otlp(self) -> Dict[str, Any]:
        """
        Export spans in OpenTelemetry Protocol format.

        Returns:
            Dictionary suitable for OTLP export.
        """
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "pluribus.observability"},
                            "spans": [span.to_otlp() for span in self._spans],
                        }
                    ],
                }
            ]
        }


# Global tracer instance
_tracer: Optional[Tracer] = None
_tracer_lock = threading.Lock()


def get_tracer(service_name: str = "creative") -> Tracer:
    """
    Get the global tracer singleton.

    Args:
        service_name: Service name for the tracer (only used on first call).

    Returns:
        The global Tracer instance.
    """
    global _tracer
    with _tracer_lock:
        if _tracer is None:
            _tracer = Tracer(service_name)
    return _tracer


def reset_tracer() -> None:
    """Reset the global tracer."""
    global _tracer
    with _tracer_lock:
        _tracer = None


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID if a span is active.

    Returns:
        The current trace ID or None.
    """
    tracer = get_tracer()
    if tracer.current_span:
        return tracer.current_span.trace_id
    return None


def get_current_span() -> Optional[Span]:
    """
    Get the current active span.

    Returns:
        The current Span or None.
    """
    return get_tracer().current_span


def traced(
    name: Optional[str] = None,
    **default_attributes: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to trace a function.

    Args:
        name: Optional span name (defaults to function name).
        **default_attributes: Default attributes to add to span.

    Returns:
        Decorator function.

    Example:
        >>> @traced("my_operation", component="api")
        ... def my_function(x, y):
        ...     return x + y
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with get_tracer().start_span(span_name, **default_attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def inject_context(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject current trace context into headers.

    Args:
        headers: Headers dictionary to inject into.

    Returns:
        Headers with trace context added.
    """
    span = get_current_span()
    if span:
        headers.update(span.context.to_headers())
    return headers


def extract_context(headers: Dict[str, str]) -> Optional[SpanContext]:
    """
    Extract trace context from headers.

    Args:
        headers: Headers dictionary to extract from.

    Returns:
        SpanContext if found, None otherwise.
    """
    return SpanContext.from_headers(headers)


__all__ = [
    "Span",
    "SpanContext",
    "Tracer",
    "get_tracer",
    "reset_tracer",
    "get_current_trace_id",
    "get_current_span",
    "traced",
    "inject_context",
    "extract_context",
]
