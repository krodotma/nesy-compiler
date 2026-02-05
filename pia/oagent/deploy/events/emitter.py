#!/usr/bin/env python3
"""
emitter.py - Deploy Event Emitter (Step 240)

PBTSO Phase: SKILL, DISTRIBUTE
A2A Integration: Event emission system for deployments via deploy.events.*

Provides:
- EventPriority: Event priority levels
- EventScope: Event scope levels
- DeployEvent: Deploy event data
- EventSubscription: Event subscription
- EventFilter: Event filtering
- DeployEventEmitter: Main event emitter

Bus Topics:
- deploy.events.emit
- deploy.events.subscribe
- deploy.events.unsubscribe
- deploy.events.broadcast

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import fnmatch
import json
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union


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
    actor: str = "event-emitter"
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

class EventPriority(IntEnum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


class EventScope(Enum):
    """Event scope levels."""
    LOCAL = "local"           # Same process
    SERVICE = "service"       # Same service
    ENVIRONMENT = "environment"  # Same environment
    GLOBAL = "global"         # All deployments


class EventType(Enum):
    """Event types."""
    DEPLOYMENT_START = "deployment.start"
    DEPLOYMENT_COMPLETE = "deployment.complete"
    DEPLOYMENT_FAILED = "deployment.failed"
    DEPLOYMENT_ROLLBACK = "deployment.rollback"
    BUILD_START = "build.start"
    BUILD_COMPLETE = "build.complete"
    BUILD_FAILED = "build.failed"
    HEALTH_CHECK = "health.check"
    HEALTH_DEGRADED = "health.degraded"
    HEALTH_RECOVERED = "health.recovered"
    CONFIG_CHANGE = "config.change"
    SCALE_UP = "scale.up"
    SCALE_DOWN = "scale.down"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    AUDIT = "audit"
    CUSTOM = "custom"


@dataclass
class DeployEvent:
    """
    Deploy event data.

    Attributes:
        event_id: Event identifier
        event_type: Type of event
        topic: Event topic
        timestamp: Event timestamp
        priority: Event priority
        scope: Event scope
        source: Event source
        service_name: Service name
        environment: Environment
        deployment_id: Deployment ID
        data: Event payload
        correlation_id: Correlation ID for tracing
        ttl_s: Time to live in seconds
    """
    event_id: str
    event_type: EventType
    topic: str
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    scope: EventScope = EventScope.LOCAL
    source: str = ""
    service_name: str = ""
    environment: str = ""
    deployment_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    ttl_s: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "topic": self.topic,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat() + "Z",
            "priority": int(self.priority),
            "priority_name": self.priority.name.lower(),
            "scope": self.scope.value,
            "source": self.source,
            "service_name": self.service_name,
            "environment": self.environment,
            "deployment_id": self.deployment_id,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "ttl_s": self.ttl_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeployEvent":
        data = dict(data)
        if "event_type" in data:
            data["event_type"] = EventType(data["event_type"])
        if "priority" in data:
            data["priority"] = EventPriority(data["priority"])
        if "scope" in data:
            data["scope"] = EventScope(data["scope"])
        # Remove computed fields
        data.pop("iso", None)
        data.pop("priority_name", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EventFilter:
    """
    Event filtering criteria.

    Attributes:
        topics: Topic patterns (supports wildcards)
        event_types: Event types to match
        min_priority: Minimum priority
        scopes: Event scopes to match
        services: Service names to match
        environments: Environments to match
    """
    topics: List[str] = field(default_factory=list)
    event_types: List[EventType] = field(default_factory=list)
    min_priority: EventPriority = EventPriority.LOW
    scopes: List[EventScope] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    environments: List[str] = field(default_factory=list)

    def matches(self, event: DeployEvent) -> bool:
        """Check if event matches the filter."""
        # Check topics
        if self.topics:
            topic_match = any(
                fnmatch.fnmatch(event.topic, pattern)
                for pattern in self.topics
            )
            if not topic_match:
                return False

        # Check event types
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check priority
        if event.priority < self.min_priority:
            return False

        # Check scopes
        if self.scopes and event.scope not in self.scopes:
            return False

        # Check services
        if self.services and event.service_name and event.service_name not in self.services:
            return False

        # Check environments
        if self.environments and event.environment and event.environment not in self.environments:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topics": self.topics,
            "event_types": [t.value for t in self.event_types],
            "min_priority": int(self.min_priority),
            "scopes": [s.value for s in self.scopes],
            "services": self.services,
            "environments": self.environments,
        }


@dataclass
class EventSubscription:
    """
    Event subscription.

    Attributes:
        subscription_id: Subscription identifier
        filter: Event filter
        handler: Event handler callback
        async_handler: Whether handler is async
        created_at: Creation timestamp
        last_event_at: Last event timestamp
        event_count: Total events received
        enabled: Whether subscription is enabled
    """
    subscription_id: str
    filter: EventFilter
    handler: Callable[[DeployEvent], None]
    async_handler: bool = False
    created_at: float = field(default_factory=time.time)
    last_event_at: float = 0.0
    event_count: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "filter": self.filter.to_dict(),
            "created_at": self.created_at,
            "last_event_at": self.last_event_at,
            "event_count": self.event_count,
            "enabled": self.enabled,
        }


# ==============================================================================
# Deploy Event Emitter (Step 240)
# ==============================================================================

class DeployEventEmitter:
    """
    Deploy Event Emitter - event emission system for deployments.

    PBTSO Phase: SKILL, DISTRIBUTE

    Responsibilities:
    - Emit deployment events
    - Manage event subscriptions
    - Route events to subscribers
    - Support event filtering
    - Persist events to bus

    A2A heartbeat: 300s interval, 900s timeout (CITIZEN v2)

    Example:
        >>> emitter = DeployEventEmitter()
        >>> def on_deploy(event):
        ...     print(f"Deployment: {event.deployment_id}")
        >>> emitter.subscribe("deployment.*", on_deploy)
        >>> emitter.emit(
        ...     topic="deployment.start",
        ...     data={"version": "v1.0.0"},
        ...     service_name="myapp"
        ... )
    """

    BUS_TOPICS = {
        "emit": "deploy.events.emit",
        "subscribe": "deploy.events.subscribe",
        "unsubscribe": "deploy.events.unsubscribe",
        "broadcast": "deploy.events.broadcast",
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "event-emitter",
        persist_events: bool = True,
        max_event_history: int = 1000,
    ):
        """
        Initialize the event emitter.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            persist_events: Whether to persist events to bus
            max_event_history: Maximum events to keep in history
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "events"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.persist_events = persist_events
        self.max_event_history = max_event_history

        # Subscriptions by topic pattern
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._topic_index: Dict[str, Set[str]] = defaultdict(set)

        # Event history
        self._events: List[DeployEvent] = []

        # Statistics
        self._emit_count = 0
        self._delivery_count = 0

        # Async event queue
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None

    def emit(
        self,
        topic: str,
        data: Optional[Dict[str, Any]] = None,
        event_type: EventType = EventType.CUSTOM,
        priority: EventPriority = EventPriority.NORMAL,
        scope: EventScope = EventScope.LOCAL,
        service_name: str = "",
        environment: str = "",
        deployment_id: str = "",
        correlation_id: str = "",
        source: str = "",
        ttl_s: int = 3600,
    ) -> DeployEvent:
        """
        Emit an event.

        Args:
            topic: Event topic
            data: Event payload
            event_type: Type of event
            priority: Event priority
            scope: Event scope
            service_name: Service name
            environment: Environment
            deployment_id: Deployment ID
            correlation_id: Correlation ID
            source: Event source
            ttl_s: Time to live

        Returns:
            DeployEvent
        """
        event = DeployEvent(
            event_id=f"evt-{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            topic=topic,
            priority=priority,
            scope=scope,
            source=source or self.actor_id,
            service_name=service_name,
            environment=environment,
            deployment_id=deployment_id,
            data=data or {},
            correlation_id=correlation_id,
            ttl_s=ttl_s,
        )

        # Store in history
        self._events.append(event)
        if len(self._events) > self.max_event_history:
            self._events = self._events[-self.max_event_history:]

        self._emit_count += 1

        # Deliver to subscribers
        self._deliver_event(event)

        # Persist to bus
        if self.persist_events:
            _emit_bus_event(
                self.BUS_TOPICS["emit"],
                event.to_dict(),
                kind="event",
                level=self._priority_to_level(priority),
                actor=self.actor_id,
            )

        return event

    def emit_deployment_start(
        self,
        service_name: str,
        version: str,
        environment: str,
        deployment_id: str,
        **extra_data,
    ) -> DeployEvent:
        """Emit deployment start event."""
        return self.emit(
            topic="deployment.start",
            event_type=EventType.DEPLOYMENT_START,
            priority=EventPriority.HIGH,
            service_name=service_name,
            environment=environment,
            deployment_id=deployment_id,
            data={"version": version, **extra_data},
        )

    def emit_deployment_complete(
        self,
        service_name: str,
        version: str,
        environment: str,
        deployment_id: str,
        duration_ms: float = 0.0,
        **extra_data,
    ) -> DeployEvent:
        """Emit deployment complete event."""
        return self.emit(
            topic="deployment.complete",
            event_type=EventType.DEPLOYMENT_COMPLETE,
            priority=EventPriority.HIGH,
            service_name=service_name,
            environment=environment,
            deployment_id=deployment_id,
            data={"version": version, "duration_ms": duration_ms, **extra_data},
        )

    def emit_deployment_failed(
        self,
        service_name: str,
        version: str,
        environment: str,
        deployment_id: str,
        error: str,
        **extra_data,
    ) -> DeployEvent:
        """Emit deployment failed event."""
        return self.emit(
            topic="deployment.failed",
            event_type=EventType.DEPLOYMENT_FAILED,
            priority=EventPriority.CRITICAL,
            service_name=service_name,
            environment=environment,
            deployment_id=deployment_id,
            data={"version": version, "error": error, **extra_data},
        )

    def emit_error(
        self,
        message: str,
        error_type: str = "",
        service_name: str = "",
        **extra_data,
    ) -> DeployEvent:
        """Emit error event."""
        return self.emit(
            topic="error",
            event_type=EventType.ERROR,
            priority=EventPriority.HIGH,
            service_name=service_name,
            data={"message": message, "error_type": error_type, **extra_data},
        )

    def emit_warning(
        self,
        message: str,
        service_name: str = "",
        **extra_data,
    ) -> DeployEvent:
        """Emit warning event."""
        return self.emit(
            topic="warning",
            event_type=EventType.WARNING,
            priority=EventPriority.NORMAL,
            service_name=service_name,
            data={"message": message, **extra_data},
        )

    def emit_audit(
        self,
        action: str,
        user: str,
        resource: str,
        service_name: str = "",
        **extra_data,
    ) -> DeployEvent:
        """Emit audit event."""
        return self.emit(
            topic="audit",
            event_type=EventType.AUDIT,
            priority=EventPriority.HIGH,
            service_name=service_name,
            data={"action": action, "user": user, "resource": resource, **extra_data},
        )

    def subscribe(
        self,
        topic_pattern: str,
        handler: Callable[[DeployEvent], None],
        filter: Optional[EventFilter] = None,
    ) -> str:
        """
        Subscribe to events.

        Args:
            topic_pattern: Topic pattern (supports wildcards)
            handler: Event handler callback
            filter: Additional filter criteria

        Returns:
            Subscription ID
        """
        subscription_id = f"sub-{uuid.uuid4().hex[:8]}"

        event_filter = filter or EventFilter()
        if topic_pattern and topic_pattern not in event_filter.topics:
            event_filter.topics.append(topic_pattern)

        subscription = EventSubscription(
            subscription_id=subscription_id,
            filter=event_filter,
            handler=handler,
            async_handler=asyncio.iscoroutinefunction(handler),
        )

        self._subscriptions[subscription_id] = subscription
        self._topic_index[topic_pattern].add(subscription_id)

        _emit_bus_event(
            self.BUS_TOPICS["subscribe"],
            {
                "subscription_id": subscription_id,
                "topic_pattern": topic_pattern,
            },
            actor=self.actor_id,
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID

        Returns:
            True if unsubscribed
        """
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            return False

        # Remove from topic index
        for topic in subscription.filter.topics:
            if subscription_id in self._topic_index[topic]:
                self._topic_index[topic].remove(subscription_id)

        del self._subscriptions[subscription_id]

        _emit_bus_event(
            self.BUS_TOPICS["unsubscribe"],
            {"subscription_id": subscription_id},
            actor=self.actor_id,
        )

        return True

    def _deliver_event(self, event: DeployEvent) -> int:
        """Deliver event to matching subscribers."""
        delivered = 0

        for subscription in self._subscriptions.values():
            if not subscription.enabled:
                continue

            if subscription.filter.matches(event):
                try:
                    if subscription.async_handler:
                        # Queue for async processing
                        self._event_queue.put_nowait((subscription, event))
                    else:
                        subscription.handler(event)

                    subscription.event_count += 1
                    subscription.last_event_at = time.time()
                    delivered += 1
                    self._delivery_count += 1

                except Exception:
                    pass

        return delivered

    async def broadcast(
        self,
        topic: str,
        data: Dict[str, Any],
        scope: EventScope = EventScope.GLOBAL,
        **kwargs,
    ) -> DeployEvent:
        """
        Broadcast an event with wider scope.

        Args:
            topic: Event topic
            data: Event payload
            scope: Event scope
            **kwargs: Additional event parameters

        Returns:
            DeployEvent
        """
        event = self.emit(
            topic=topic,
            data=data,
            scope=scope,
            **kwargs,
        )

        _emit_bus_event(
            self.BUS_TOPICS["broadcast"],
            {
                "event_id": event.event_id,
                "topic": topic,
                "scope": scope.value,
            },
            actor=self.actor_id,
        )

        return event

    async def start_async_processing(self) -> None:
        """Start async event processing."""
        if self._processing:
            return

        self._processing = True
        self._processor_task = asyncio.create_task(self._process_queue())

    async def stop_async_processing(self) -> None:
        """Stop async event processing."""
        self._processing = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

    async def _process_queue(self) -> None:
        """Process async event queue."""
        while self._processing:
            try:
                subscription, event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )

                if subscription.async_handler:
                    await subscription.handler(event)
                else:
                    subscription.handler(event)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def _priority_to_level(self, priority: EventPriority) -> str:
        """Convert priority to log level."""
        mapping = {
            EventPriority.LOW: "debug",
            EventPriority.NORMAL: "info",
            EventPriority.HIGH: "warn",
            EventPriority.URGENT: "error",
            EventPriority.CRITICAL: "error",
        }
        return mapping.get(priority, "info")

    def get_event(self, event_id: str) -> Optional[DeployEvent]:
        """Get an event by ID."""
        for event in self._events:
            if event.event_id == event_id:
                return event
        return None

    def get_events(
        self,
        topic_pattern: Optional[str] = None,
        event_type: Optional[EventType] = None,
        service_name: Optional[str] = None,
        hours: int = 24,
        limit: int = 100,
    ) -> List[DeployEvent]:
        """Get events with filters."""
        cutoff = time.time() - (hours * 3600)
        events = [e for e in self._events if e.timestamp >= cutoff]

        if topic_pattern:
            events = [e for e in events if fnmatch.fnmatch(e.topic, topic_pattern)]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if service_name:
            events = [e for e in events if e.service_name == service_name]

        return events[-limit:]

    def get_subscription(self, subscription_id: str) -> Optional[EventSubscription]:
        """Get a subscription by ID."""
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(self) -> List[EventSubscription]:
        """List all subscriptions."""
        return list(self._subscriptions.values())

    def enable_subscription(self, subscription_id: str) -> bool:
        """Enable a subscription."""
        subscription = self._subscriptions.get(subscription_id)
        if subscription:
            subscription.enabled = True
            return True
        return False

    def disable_subscription(self, subscription_id: str) -> bool:
        """Disable a subscription."""
        subscription = self._subscriptions.get(subscription_id)
        if subscription:
            subscription.enabled = False
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get event emitter statistics."""
        return {
            "emit_count": self._emit_count,
            "delivery_count": self._delivery_count,
            "subscription_count": len(self._subscriptions),
            "event_history_count": len(self._events),
            "active_subscriptions": sum(1 for s in self._subscriptions.values() if s.enabled),
        }

    def clear_history(self) -> int:
        """Clear event history."""
        count = len(self._events)
        self._events.clear()
        return count


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for event emitter."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Event Emitter (Step 240)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # emit command
    emit_parser = subparsers.add_parser("emit", help="Emit an event")
    emit_parser.add_argument("topic", help="Event topic")
    emit_parser.add_argument("--data", "-d", help="JSON data")
    emit_parser.add_argument("--type", "-t", default="custom",
                            choices=["deployment.start", "deployment.complete", "deployment.failed",
                                    "error", "warning", "info", "audit", "custom"])
    emit_parser.add_argument("--priority", "-p", default="normal",
                            choices=["low", "normal", "high", "urgent", "critical"])
    emit_parser.add_argument("--service", "-s", default="", help="Service name")
    emit_parser.add_argument("--environment", "-e", default="", help="Environment")
    emit_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List events")
    list_parser.add_argument("--topic", "-t", help="Filter by topic pattern")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--hours", type=int, default=24, help="Time window")
    list_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # subscriptions command
    subs_parser = subparsers.add_parser("subscriptions", help="List subscriptions")
    subs_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics")
    stats_parser.add_argument("--json", action="store_true", help="JSON output")

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear event history")
    clear_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    args = parser.parse_args()
    emitter = DeployEventEmitter()

    if args.command == "emit":
        data = json.loads(args.data) if args.data else {}

        # Map type string to EventType
        type_mapping = {
            "deployment.start": EventType.DEPLOYMENT_START,
            "deployment.complete": EventType.DEPLOYMENT_COMPLETE,
            "deployment.failed": EventType.DEPLOYMENT_FAILED,
            "error": EventType.ERROR,
            "warning": EventType.WARNING,
            "info": EventType.INFO,
            "audit": EventType.AUDIT,
            "custom": EventType.CUSTOM,
        }

        priority_mapping = {
            "low": EventPriority.LOW,
            "normal": EventPriority.NORMAL,
            "high": EventPriority.HIGH,
            "urgent": EventPriority.URGENT,
            "critical": EventPriority.CRITICAL,
        }

        event = emitter.emit(
            topic=args.topic,
            data=data,
            event_type=type_mapping.get(args.type, EventType.CUSTOM),
            priority=priority_mapping.get(args.priority, EventPriority.NORMAL),
            service_name=args.service,
            environment=args.environment,
        )

        if args.json:
            print(json.dumps(event.to_dict(), indent=2))
        else:
            print(f"Emitted event: {event.event_id}")
            print(f"  Topic: {event.topic}")
            print(f"  Type: {event.event_type.value}")
            print(f"  Priority: {event.priority.name.lower()}")

        return 0

    elif args.command == "list":
        events = emitter.get_events(
            topic_pattern=args.topic,
            service_name=args.service,
            hours=args.hours,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            if not events:
                print("No events found")
            else:
                for e in events:
                    ts = datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
                    print(f"{ts} [{e.priority.name[:4]}] {e.topic}: {e.event_type.value}")

        return 0

    elif args.command == "subscriptions":
        subscriptions = emitter.list_subscriptions()

        if args.json:
            print(json.dumps([s.to_dict() for s in subscriptions], indent=2))
        else:
            if not subscriptions:
                print("No subscriptions")
            else:
                for s in subscriptions:
                    status = "enabled" if s.enabled else "disabled"
                    print(f"{s.subscription_id} ({s.event_count} events) [{status}]")

        return 0

    elif args.command == "stats":
        stats = emitter.get_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Event Emitter Statistics:")
            print(f"  Events emitted: {stats['emit_count']}")
            print(f"  Events delivered: {stats['delivery_count']}")
            print(f"  Subscriptions: {stats['subscription_count']}")
            print(f"  Active subscriptions: {stats['active_subscriptions']}")
            print(f"  Events in history: {stats['event_history_count']}")

        return 0

    elif args.command == "clear":
        if not args.force:
            confirm = input("Clear event history? [y/N] ")
            if confirm.lower() != "y":
                print("Cancelled")
                return 0

        count = emitter.clear_history()
        print(f"Cleared {count} events from history")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
