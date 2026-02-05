#!/usr/bin/env python3
"""
Monitor Event Emitter - Step 290

Event emission system for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.event.* (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventType(Enum):
    """Types of events."""
    METRIC = "metric"           # Metric events
    ALERT = "alert"             # Alert events
    INCIDENT = "incident"       # Incident events
    STATUS = "status"           # Status change events
    HEALTH = "health"           # Health check events
    LIFECYCLE = "lifecycle"     # Lifecycle events
    AUDIT = "audit"             # Audit events
    CUSTOM = "custom"           # Custom events


@dataclass
class Event:
    """An event to be emitted.

    Attributes:
        event_id: Unique event ID
        event_type: Type of event
        topic: Event topic
        data: Event data
        priority: Event priority
        timestamp: Event timestamp
        source: Event source
        correlation_id: Correlation ID for related events
        metadata: Additional metadata
    """
    event_id: str
    event_type: EventType
    topic: str
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: str = "monitor"
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "topic": self.topic,
            "data": self.data,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat() + "Z",
            "source": self.source,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


@dataclass
class EventSubscription:
    """A subscription to events.

    Attributes:
        subscription_id: Unique subscription ID
        topic_pattern: Topic pattern (supports *)
        event_types: Event types to receive
        callback: Callback function
        filter_fn: Optional filter function
        created_at: Subscription creation time
    """
    subscription_id: str
    topic_pattern: str
    event_types: Set[EventType]
    callback: Callable[[Event], Coroutine[Any, Any, None]]
    filter_fn: Optional[Callable[[Event], bool]] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class EventStats:
    """Event emission statistics.

    Attributes:
        total_emitted: Total events emitted
        by_type: Count by event type
        by_topic: Count by topic
        by_priority: Count by priority
    """
    total_emitted: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_topic: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_emitted": self.total_emitted,
            "by_type": self.by_type,
            "by_topic": self.by_topic,
            "by_priority": self.by_priority,
        }


class MonitorEventEmitter:
    """
    Event emission system for the Monitor Agent.

    Provides:
    - Event creation and emission
    - Topic-based subscriptions
    - Event filtering
    - Batch emission
    - Event correlation
    - Bus integration with file locking

    Example:
        emitter = MonitorEventEmitter()

        # Emit an event
        await emitter.emit(
            event_type=EventType.ALERT,
            topic="alert.cpu_high",
            data={"cpu": 95},
            priority=EventPriority.HIGH,
        )

        # Subscribe to events
        async def handle_alert(event: Event):
            print(f"Alert: {event.data}")

        emitter.subscribe(
            topic_pattern="alert.*",
            callback=handle_alert,
        )
    """

    BUS_TOPICS = {
        "emitted": "monitor.event.emitted",
        "error": "monitor.event.error",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        buffer_size: int = 1000,
        flush_interval_s: float = 1.0,
        bus_dir: Optional[str] = None,
    ):
        """Initialize event emitter.

        Args:
            buffer_size: Event buffer size
            flush_interval_s: Buffer flush interval
            bus_dir: Bus directory
        """
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_s
        self._last_heartbeat = time.time()
        self._last_flush = time.time()

        # Event tracking
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._event_buffer: List[Event] = []
        self._stats = EventStats()
        self._lock = threading.RLock()

        # Background tasks
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    async def emit(
        self,
        event_type: EventType,
        topic: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "monitor",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Emit an event.

        Args:
            event_type: Type of event
            topic: Event topic
            data: Event data
            priority: Event priority
            source: Event source
            correlation_id: Correlation ID
            metadata: Additional metadata

        Returns:
            Event ID
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            topic=topic,
            data=data,
            priority=priority,
            source=source,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        # Update statistics
        with self._lock:
            self._stats.total_emitted += 1
            self._stats.by_type[event_type.value] = self._stats.by_type.get(event_type.value, 0) + 1
            self._stats.by_topic[topic] = self._stats.by_topic.get(topic, 0) + 1
            self._stats.by_priority[priority.value] = self._stats.by_priority.get(priority.value, 0) + 1

        # Write to bus
        self._write_to_bus(event)

        # Notify subscribers
        await self._notify_subscribers(event)

        return event.event_id

    async def emit_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """Emit a metric event.

        Args:
            name: Metric name
            value: Metric value
            labels: Metric labels
            **kwargs: Additional emit arguments

        Returns:
            Event ID
        """
        return await self.emit(
            event_type=EventType.METRIC,
            topic=f"metric.{name}",
            data={
                "name": name,
                "value": value,
                "labels": labels or {},
            },
            **kwargs,
        )

    async def emit_alert(
        self,
        name: str,
        severity: str,
        message: str,
        **kwargs: Any,
    ) -> str:
        """Emit an alert event.

        Args:
            name: Alert name
            severity: Alert severity
            message: Alert message
            **kwargs: Additional emit arguments

        Returns:
            Event ID
        """
        priority = {
            "critical": EventPriority.CRITICAL,
            "high": EventPriority.HIGH,
            "warning": EventPriority.NORMAL,
            "low": EventPriority.LOW,
        }.get(severity.lower(), EventPriority.NORMAL)

        return await self.emit(
            event_type=EventType.ALERT,
            topic=f"alert.{name}",
            data={
                "name": name,
                "severity": severity,
                "message": message,
            },
            priority=priority,
            **kwargs,
        )

    async def emit_status(
        self,
        component: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Emit a status event.

        Args:
            component: Component name
            status: Status value
            details: Status details
            **kwargs: Additional emit arguments

        Returns:
            Event ID
        """
        return await self.emit(
            event_type=EventType.STATUS,
            topic=f"status.{component}",
            data={
                "component": component,
                "status": status,
                "details": details or {},
            },
            **kwargs,
        )

    async def emit_lifecycle(
        self,
        event_name: str,
        component: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Emit a lifecycle event.

        Args:
            event_name: Lifecycle event name
            component: Component name
            details: Event details
            **kwargs: Additional emit arguments

        Returns:
            Event ID
        """
        return await self.emit(
            event_type=EventType.LIFECYCLE,
            topic=f"lifecycle.{event_name}",
            data={
                "event": event_name,
                "component": component,
                "details": details or {},
            },
            **kwargs,
        )

    async def emit_batch(
        self,
        events: List[Dict[str, Any]],
    ) -> List[str]:
        """Emit multiple events.

        Args:
            events: List of event specifications

        Returns:
            List of event IDs
        """
        event_ids = []
        for event_spec in events:
            event_id = await self.emit(
                event_type=EventType(event_spec.get("event_type", "custom")),
                topic=event_spec.get("topic", "monitor.event"),
                data=event_spec.get("data", {}),
                priority=EventPriority(event_spec.get("priority", 1)),
                source=event_spec.get("source", "monitor"),
                correlation_id=event_spec.get("correlation_id"),
                metadata=event_spec.get("metadata"),
            )
            event_ids.append(event_id)
        return event_ids

    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[Event], Coroutine[Any, Any, None]],
        event_types: Optional[Set[EventType]] = None,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """Subscribe to events.

        Args:
            topic_pattern: Topic pattern (supports * wildcard)
            callback: Async callback function
            event_types: Event types to receive
            filter_fn: Optional filter function

        Returns:
            Subscription ID
        """
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"

        subscription = EventSubscription(
            subscription_id=sub_id,
            topic_pattern=topic_pattern,
            event_types=event_types or set(EventType),
            callback=callback,
            filter_fn=filter_fn,
        )

        with self._lock:
            self._subscriptions[sub_id] = subscription

        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: Subscription ID

        Returns:
            True if unsubscribed
        """
        with self._lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                return True
            return False

    def list_subscriptions(self) -> List[Dict[str, Any]]:
        """List all subscriptions.

        Returns:
            List of subscription info
        """
        with self._lock:
            return [
                {
                    "subscription_id": sub.subscription_id,
                    "topic_pattern": sub.topic_pattern,
                    "event_types": [t.value for t in sub.event_types],
                    "created_at": sub.created_at,
                }
                for sub in self._subscriptions.values()
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get emission statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                **self._stats.to_dict(),
                "subscriptions": len(self._subscriptions),
                "buffer_size": len(self._event_buffer),
            }

    def create_correlation_id(self) -> str:
        """Create a new correlation ID.

        Returns:
            Correlation ID
        """
        return f"corr-{uuid.uuid4().hex[:12]}"

    async def start_background_tasks(self) -> None:
        """Start background tasks."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._background_flush())

    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

    async def _notify_subscribers(self, event: Event) -> None:
        """Notify subscribers of an event.

        Args:
            event: Event to notify about
        """
        with self._lock:
            subscriptions = list(self._subscriptions.values())

        for sub in subscriptions:
            # Check event type
            if event.event_type not in sub.event_types:
                continue

            # Check topic pattern
            if not self._topic_matches(event.topic, sub.topic_pattern):
                continue

            # Apply filter
            if sub.filter_fn and not sub.filter_fn(event):
                continue

            # Invoke callback
            try:
                await sub.callback(event)
            except Exception:
                pass

    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern.

        Args:
            topic: Event topic
            pattern: Pattern (supports * wildcard)

        Returns:
            True if matches
        """
        if pattern == "*":
            return True

        if "*" not in pattern:
            return topic == pattern

        # Handle patterns like "alert.*" or "*.metric"
        parts = pattern.split("*")
        if len(parts) == 2:
            prefix, suffix = parts
            if prefix and suffix:
                return topic.startswith(prefix) and topic.endswith(suffix)
            elif prefix:
                return topic.startswith(prefix)
            elif suffix:
                return topic.endswith(suffix)

        return False

    def _write_to_bus(self, event: Event) -> None:
        """Write event to bus with file locking.

        Args:
            event: Event to write
        """
        bus_event = {
            "id": event.event_id,
            "ts": event.timestamp,
            "iso": datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat() + "Z",
            "topic": event.topic,
            "kind": event.event_type.value,
            "level": self._priority_to_level(event.priority),
            "actor": event.source,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": event.data,
        }

        if event.correlation_id:
            bus_event["correlation_id"] = event.correlation_id

        if event.metadata:
            bus_event["metadata"] = event.metadata

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(bus_event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

    def _priority_to_level(self, priority: EventPriority) -> str:
        """Convert priority to log level.

        Args:
            priority: Event priority

        Returns:
            Log level string
        """
        return {
            EventPriority.LOW: "debug",
            EventPriority.NORMAL: "info",
            EventPriority.HIGH: "warning",
            EventPriority.CRITICAL: "error",
        }.get(priority, "info")

    async def _background_flush(self) -> None:
        """Background task to flush event buffer."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                # Buffer flush logic would go here if we were batching
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        stats = self.get_stats()

        # Write heartbeat directly
        bus_event = {
            "id": str(uuid.uuid4()),
            "ts": now,
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": "a2a.heartbeat",
            "kind": "heartbeat",
            "level": "info",
            "actor": "monitor-event-emitter",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "component": "monitor_event_emitter",
                "status": "healthy",
                "total_emitted": stats["total_emitted"],
                "subscriptions": stats["subscriptions"],
            },
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(bus_event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return True


# Singleton instance
_emitter: Optional[MonitorEventEmitter] = None


def get_event_emitter() -> MonitorEventEmitter:
    """Get or create the event emitter singleton.

    Returns:
        MonitorEventEmitter instance
    """
    global _emitter
    if _emitter is None:
        _emitter = MonitorEventEmitter()
    return _emitter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Event Emitter (Step 290)")
    parser.add_argument("--emit", metavar="TOPIC", help="Emit an event")
    parser.add_argument("--type", default="custom", help="Event type")
    parser.add_argument("--data", default="{}", help="Event data (JSON)")
    parser.add_argument("--priority", type=int, default=1, help="Event priority (0-3)")
    parser.add_argument("--metric", metavar="NAME=VALUE", help="Emit a metric event")
    parser.add_argument("--alert", metavar="NAME", help="Emit an alert event")
    parser.add_argument("--severity", default="warning", help="Alert severity")
    parser.add_argument("--message", default="Alert", help="Alert message")
    parser.add_argument("--subscriptions", action="store_true", help="List subscriptions")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    emitter = get_event_emitter()

    if args.emit:
        async def emit_event():
            data = json.loads(args.data)
            event_id = await emitter.emit(
                event_type=EventType(args.type),
                topic=args.emit,
                data=data,
                priority=EventPriority(args.priority),
            )
            return event_id

        event_id = asyncio.run(emit_event())
        if args.json:
            print(json.dumps({"event_id": event_id}))
        else:
            print(f"Emitted event: {event_id}")

    if args.metric:
        name, value = args.metric.split("=")
        async def emit_metric():
            return await emitter.emit_metric(name, float(value))
        event_id = asyncio.run(emit_metric())
        if args.json:
            print(json.dumps({"event_id": event_id, "metric": name, "value": float(value)}))
        else:
            print(f"Emitted metric: {name}={value} (event: {event_id})")

    if args.alert:
        async def emit_alert():
            return await emitter.emit_alert(args.alert, args.severity, args.message)
        event_id = asyncio.run(emit_alert())
        if args.json:
            print(json.dumps({"event_id": event_id, "alert": args.alert}))
        else:
            print(f"Emitted alert: {args.alert} [{args.severity}] (event: {event_id})")

    if args.subscriptions:
        subs = emitter.list_subscriptions()
        if args.json:
            print(json.dumps(subs, indent=2))
        else:
            print("Subscriptions:")
            for s in subs:
                print(f"  {s['subscription_id']}: {s['topic_pattern']}")

    if args.stats:
        stats = emitter.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Event Emitter Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
