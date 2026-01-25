"""
Bus Event Emitter
=================

BusEmitter class for publishing events to the Pluribus bus.

Events are written to an NDJSON (newline-delimited JSON) file,
which can be consumed by the bus infrastructure for distribution
to interested agents and dashboard components.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import json
import threading
import uuid

from .schemas import EventSchema, EventKind, EventLevel


# Default bus path for the Pluribus project
DEFAULT_BUS_PATH = Path("/pluribus/.pluribus/bus/events.ndjson")


class BusEmitter:
    """
    Emitter for publishing events to the Pluribus bus.

    The BusEmitter writes events to an NDJSON file that serves as
    the primary bus transport for the Creative subsystem. It is
    thread-safe and handles file creation automatically.

    Attributes:
        bus_path: Path to the NDJSON events file.
        agent: Default agent identifier for events.
        default_kind: Default event kind.
        default_level: Default event level.

    Example:
        >>> emitter = BusEmitter()
        >>> emitter.emit("creative.visual.render", {"prompt": "sunset"})
        {'id': '...', 'topic': 'creative.visual.render', ...}

        >>> # With custom bus path
        >>> emitter = BusEmitter(bus_path="/tmp/test_bus.ndjson")
        >>> emitter.emit("creative.test", {"data": 123})
    """

    def __init__(
        self,
        bus_path: Optional[Union[str, Path]] = None,
        agent: str = "creative",
        default_kind: EventKind = "state",
        default_level: EventLevel = "info",
    ) -> None:
        """
        Initialize the BusEmitter.

        Args:
            bus_path: Path to the NDJSON events file (default: DEFAULT_BUS_PATH).
            agent: Default agent identifier for emitted events.
            default_kind: Default event kind.
            default_level: Default event level.
        """
        self.bus_path = Path(bus_path) if bus_path else DEFAULT_BUS_PATH
        self.agent = agent
        self.default_kind = default_kind
        self.default_level = default_level
        self._lock = threading.Lock()
        self._ensure_bus_path()

    def _ensure_bus_path(self) -> None:
        """Ensure the bus directory and file exist."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.bus_path.exists():
            self.bus_path.touch()

    def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        agent: Optional[str] = None,
        kind: Optional[EventKind] = None,
        level: Optional[EventLevel] = None,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Emit an event to the bus.

        Args:
            topic: Event topic (e.g., "creative.visual.render").
            payload: Event-specific data dictionary.
            agent: Agent identifier (default: self.agent).
            kind: Event kind (default: self.default_kind).
            level: Event level (default: self.default_level).
            trace_id: Optional trace ID for distributed tracing.
            parent_id: Optional parent event ID.
            metadata: Optional additional metadata.

        Returns:
            The emitted event as a dictionary.

        Raises:
            IOError: If writing to the bus file fails.

        Example:
            >>> emitter = BusEmitter()
            >>> event = emitter.emit(
            ...     "creative.visual.progress",
            ...     {"job_id": "job-123", "progress": 0.5},
            ...     kind="metric",
            ... )
        """
        event = EventSchema(
            topic=topic,
            payload=payload,
            agent=agent or self.agent,
            kind=kind or self.default_kind,
            level=level or self.default_level,
            trace_id=trace_id,
            parent_id=parent_id,
            metadata=metadata or {},
        )

        return self._write_event(event)

    def emit_event(self, event: EventSchema) -> dict[str, Any]:
        """
        Emit a pre-constructed EventSchema.

        Args:
            event: EventSchema instance to emit.

        Returns:
            The emitted event as a dictionary.

        Example:
            >>> event = EventSchema(
            ...     topic="creative.test",
            ...     payload={"test": True},
            ... )
            >>> emitter.emit_event(event)
        """
        return self._write_event(event)

    def _write_event(self, event: EventSchema) -> dict[str, Any]:
        """
        Write an event to the bus file.

        Thread-safe writing to the NDJSON file.

        Args:
            event: EventSchema to write.

        Returns:
            The event as a dictionary.
        """
        event_dict = event.to_dict()

        with self._lock:
            self._ensure_bus_path()
            with open(self.bus_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_dict, default=str) + "\n")

        return event_dict

    def emit_request(
        self,
        topic: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Emit a request event.

        Convenience method for emitting request-type events.

        Args:
            topic: Event topic.
            payload: Event payload.
            **kwargs: Additional arguments passed to emit().

        Returns:
            The emitted event dictionary.
        """
        return self.emit(topic, payload, kind="request", **kwargs)

    def emit_response(
        self,
        topic: str,
        payload: dict[str, Any],
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Emit a response event.

        Convenience method for emitting response-type events.

        Args:
            topic: Event topic.
            payload: Event payload.
            request_id: Optional ID of the request being responded to.
            **kwargs: Additional arguments passed to emit().

        Returns:
            The emitted event dictionary.
        """
        if request_id:
            kwargs.setdefault("parent_id", request_id)
        return self.emit(topic, payload, kind="response", **kwargs)

    def emit_metric(
        self,
        topic: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Emit a metric event.

        Convenience method for emitting telemetry/metrics.

        Args:
            topic: Event topic.
            payload: Metric data.
            **kwargs: Additional arguments passed to emit().

        Returns:
            The emitted event dictionary.
        """
        return self.emit(topic, payload, kind="metric", **kwargs)

    def emit_alert(
        self,
        topic: str,
        message: str,
        level: EventLevel = "warning",
        payload: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Emit an alert event.

        Convenience method for emitting warnings or errors.

        Args:
            topic: Event topic.
            message: Alert message.
            level: Severity level (default: "warning").
            payload: Optional additional data.
            **kwargs: Additional arguments passed to emit().

        Returns:
            The emitted event dictionary.
        """
        alert_payload = {"message": message, **(payload or {})}
        return self.emit(topic, alert_payload, kind="alert", level=level, **kwargs)

    def emit_state_change(
        self,
        topic: str,
        old_state: Any,
        new_state: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Emit a state change event.

        Convenience method for emitting state transitions.

        Args:
            topic: Event topic.
            old_state: Previous state value.
            new_state: New state value.
            **kwargs: Additional arguments passed to emit().

        Returns:
            The emitted event dictionary.
        """
        payload = {
            "old_state": old_state,
            "new_state": new_state,
            "changed_at": datetime.now(timezone.utc).isoformat(),
        }
        return self.emit(topic, payload, kind="state", **kwargs)

    def create_trace(self) -> str:
        """
        Create a new trace ID for distributed tracing.

        Returns:
            A new UUID string for use as trace_id.
        """
        return str(uuid.uuid4())

    def close(self) -> None:
        """
        Close the emitter.

        Currently a no-op as file handles are not kept open,
        but provided for interface completeness and future use.
        """
        pass

    def __enter__(self) -> "BusEmitter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
