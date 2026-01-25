"""
Bus Event Schemas
=================

Dataclass definitions for bus events with JSON serialization support.

The EventSchema is the canonical structure for all Creative subsystem
bus events, ensuring consistent formatting and validation across agents.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional
import json
import uuid


# Event kind enumeration
EventKind = Literal[
    "request",    # Request for action
    "response",   # Response to request
    "metric",     # Telemetry/metrics data
    "alert",      # Warning or error condition
    "state",      # State change notification
]

# Event severity levels
EventLevel = Literal[
    "debug",      # Detailed debugging info
    "info",       # Informational messages
    "warning",    # Warning conditions
    "error",      # Error conditions
    "critical",   # Critical failures
]


@dataclass
class EventSchema:
    """
    Canonical schema for Creative subsystem bus events.

    This dataclass defines the structure of all events emitted to the
    Pluribus bus from the Creative section.

    Attributes:
        id: Unique event identifier (UUID)
        topic: Event topic (e.g., "creative.visual.render")
        timestamp: ISO 8601 timestamp (UTC)
        agent: Agent identifier that emitted the event
        payload: Event-specific data
        kind: Event kind (request, response, metric, alert, state)
        level: Severity level
        trace_id: Optional trace ID for distributed tracing
        parent_id: Optional parent event ID for causality chains
        metadata: Optional additional metadata

    Example:
        >>> event = EventSchema(
        ...     topic="creative.visual.render",
        ...     payload={"prompt": "sunset over mountains", "steps": 50},
        ... )
        >>> event.id  # Auto-generated UUID
        '550e8400-e29b-41d4-a716-446655440000'
    """

    topic: str
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    agent: str = "creative"
    kind: EventKind = "state"
    level: EventLevel = "info"
    trace_id: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event after initialization."""
        if not self.topic:
            raise ValueError("Event topic cannot be empty")
        if not isinstance(self.payload, dict):
            raise TypeError("Event payload must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert event to dictionary.

        Returns:
            Dictionary representation of the event.
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Serialize event to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventSchema":
        """
        Create an EventSchema from a dictionary.

        Args:
            data: Dictionary with event fields.

        Returns:
            EventSchema instance.

        Raises:
            KeyError: If required fields are missing.
            TypeError: If field types are invalid.
        """
        # Extract required fields
        topic = data["topic"]
        payload = data.get("payload", {})

        # Extract optional fields with defaults
        return cls(
            topic=topic,
            payload=payload,
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat()
            ),
            agent=data.get("agent", "creative"),
            kind=data.get("kind", "state"),
            level=data.get("level", "info"),
            trace_id=data.get("trace_id"),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EventSchema":
        """
        Deserialize event from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            EventSchema instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


def validate_event(event: EventSchema) -> tuple[bool, list[str]]:
    """
    Validate an event schema.

    Args:
        event: EventSchema instance to validate.

    Returns:
        Tuple of (is_valid, list of error messages).

    Example:
        >>> event = EventSchema(topic="creative.test", payload={})
        >>> valid, errors = validate_event(event)
        >>> valid
        True
    """
    errors: list[str] = []

    # Validate topic format
    if not event.topic:
        errors.append("Topic cannot be empty")
    elif not event.topic.startswith("creative."):
        errors.append(f"Topic must start with 'creative.': {event.topic}")

    # Validate ID is UUID-like
    try:
        uuid.UUID(event.id)
    except ValueError:
        errors.append(f"Invalid event ID format: {event.id}")

    # Validate timestamp format
    try:
        datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"Invalid timestamp format: {event.timestamp}")

    # Validate kind
    valid_kinds = {"request", "response", "metric", "alert", "state"}
    if event.kind not in valid_kinds:
        errors.append(f"Invalid event kind: {event.kind}")

    # Validate level
    valid_levels = {"debug", "info", "warning", "error", "critical"}
    if event.level not in valid_levels:
        errors.append(f"Invalid event level: {event.level}")

    # Validate payload is dict
    if not isinstance(event.payload, dict):
        errors.append("Payload must be a dictionary")

    return len(errors) == 0, errors


def event_to_dict(event: EventSchema) -> dict[str, Any]:
    """
    Convert EventSchema to dictionary (convenience function).

    Args:
        event: EventSchema instance.

    Returns:
        Dictionary representation.
    """
    return event.to_dict()


def event_from_dict(data: dict[str, Any]) -> EventSchema:
    """
    Create EventSchema from dictionary (convenience function).

    Args:
        data: Dictionary with event fields.

    Returns:
        EventSchema instance.
    """
    return EventSchema.from_dict(data)
