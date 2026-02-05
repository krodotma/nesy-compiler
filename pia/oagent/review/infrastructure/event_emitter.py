#!/usr/bin/env python3
"""
Review Event Emitter (Step 190)

Event emission system for the Review Agent with pub/sub patterns,
event filtering, and delivery guarantees.

PBTSO Phase: DISTRIBUTE, OBSERVE
Bus Topics: review.event.emit, review.event.subscribe

Event Features:
- Pub/sub event patterns
- Event filtering and routing
- Priority-based delivery
- Event replay
- Dead letter queue

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Awaitable
import re
import fnmatch

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = 0   # Immediate delivery
    HIGH = 1       # High priority
    NORMAL = 2     # Normal priority
    LOW = 3        # Low priority
    BACKGROUND = 4 # Background/batch processing


class EventDelivery(Enum):
    """Event delivery semantics."""
    AT_MOST_ONCE = "at_most_once"   # Fire and forget
    AT_LEAST_ONCE = "at_least_once" # With retry
    EXACTLY_ONCE = "exactly_once"   # With deduplication


class EventState(Enum):
    """Event processing state."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Event:
    """
    An event in the system.

    Attributes:
        event_id: Unique event identifier
        topic: Event topic/channel
        type: Event type within topic
        payload: Event payload data
        source: Event source identifier
        priority: Event priority
        timestamp: Event creation timestamp
        metadata: Additional metadata
        correlation_id: Correlation ID for tracing
        causation_id: ID of event that caused this one
    """
    event_id: str
    topic: str
    type: str
    payload: Dict[str, Any]
    source: str = "review-agent"
    priority: EventPriority = EventPriority.NORMAL
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "type": self.type,
            "payload": self.payload,
            "source": self.source,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        data = dict(data)
        if "priority" in data:
            data["priority"] = EventPriority(data["priority"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EventConfig:
    """
    Configuration for event system.

    Attributes:
        delivery: Delivery semantics
        max_retries: Maximum delivery retries
        retry_delay_seconds: Delay between retries
        dead_letter_enabled: Enable dead letter queue
        max_queue_size: Maximum event queue size
        batch_size: Batch size for processing
        dedup_window_seconds: Deduplication window
    """
    delivery: EventDelivery = EventDelivery.AT_LEAST_ONCE
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    dead_letter_enabled: bool = True
    max_queue_size: int = 10000
    batch_size: int = 100
    dedup_window_seconds: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "delivery": self.delivery.value,
        }


EventHandler = Callable[[Event], Awaitable[None]]


@dataclass
class EventSubscription:
    """
    An event subscription.

    Attributes:
        subscription_id: Unique subscription ID
        topic_pattern: Topic pattern to match
        type_pattern: Event type pattern to match
        handler: Event handler function
        priority_filter: Minimum priority to receive
        filter_fn: Custom filter function
        created_at: Subscription creation time
    """
    subscription_id: str
    topic_pattern: str
    handler: EventHandler
    type_pattern: str = "*"
    priority_filter: Optional[EventPriority] = None
    filter_fn: Optional[Callable[[Event], bool]] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.subscription_id:
            self.subscription_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def matches(self, event: Event) -> bool:
        """Check if event matches subscription."""
        # Check topic pattern
        if not fnmatch.fnmatch(event.topic, self.topic_pattern):
            return False

        # Check type pattern
        if self.type_pattern != "*" and not fnmatch.fnmatch(event.type, self.type_pattern):
            return False

        # Check priority
        if self.priority_filter and event.priority.value > self.priority_filter.value:
            return False

        # Check custom filter
        if self.filter_fn and not self.filter_fn(event):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subscription_id": self.subscription_id,
            "topic_pattern": self.topic_pattern,
            "type_pattern": self.type_pattern,
            "priority_filter": self.priority_filter.name if self.priority_filter else None,
            "created_at": self.created_at,
        }


@dataclass
class DeliveryRecord:
    """Record of event delivery."""
    event_id: str
    subscription_id: str
    state: EventState
    attempts: int = 0
    last_attempt: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "subscription_id": self.subscription_id,
            "state": self.state.value,
            "attempts": self.attempts,
            "error": self.error,
        }


# ============================================================================
# Event Emitter
# ============================================================================

class EventEmitter:
    """
    Event emission and subscription system.

    Example:
        emitter = EventEmitter()

        # Subscribe to events
        async def handle_review(event: Event):
            print(f"Review: {event.payload}")

        sub_id = emitter.subscribe(
            topic_pattern="review.*",
            handler=handle_review,
        )

        # Emit event
        await emitter.emit(Event(
            event_id="",
            topic="review.completed",
            type="review_done",
            payload={"review_id": "abc123"},
        ))

        # Emit with helper
        await emitter.emit_event(
            topic="review.started",
            type="review_start",
            payload={"files": ["file.py"]},
        )
    """

    BUS_TOPICS = {
        "emit": "review.event.emit",
        "subscribe": "review.event.subscribe",
        "dead_letter": "review.event.dead_letter",
    }

    def __init__(
        self,
        config: Optional[EventConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the event emitter.

        Args:
            config: Event configuration
            bus_path: Path to event bus file
        """
        self.config = config or EventConfig()
        self.bus_path = bus_path or self._get_bus_path()

        # Subscriptions
        self._subscriptions: Dict[str, EventSubscription] = {}

        # Event queue
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self.config.max_queue_size)

        # Delivery tracking
        self._delivery_records: Dict[str, DeliveryRecord] = {}
        self._dead_letter: List[Event] = []

        # Deduplication
        self._seen_events: Dict[str, float] = {}

        # Processing state
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "events_emitted": 0,
            "events_delivered": 0,
            "events_failed": 0,
            "dead_letters": 0,
        }
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_bus_event(self, topic: str, data: Dict[str, Any], kind: str = "event") -> str:
        """Emit event to A2A bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "event-emitter",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _is_duplicate(self, event: Event) -> bool:
        """Check if event is a duplicate."""
        if self.config.delivery != EventDelivery.EXACTLY_ONCE:
            return False

        now = time.time()

        # Clean old entries
        cutoff = now - self.config.dedup_window_seconds
        self._seen_events = {
            k: v for k, v in self._seen_events.items()
            if v > cutoff
        }

        # Check for duplicate
        if event.event_id in self._seen_events:
            return True

        self._seen_events[event.event_id] = now
        return False

    def subscribe(
        self,
        topic_pattern: str,
        handler: EventHandler,
        type_pattern: str = "*",
        priority_filter: Optional[EventPriority] = None,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """
        Subscribe to events.

        Args:
            topic_pattern: Topic pattern (supports wildcards)
            handler: Event handler function
            type_pattern: Event type pattern
            priority_filter: Minimum priority
            filter_fn: Custom filter function

        Returns:
            Subscription ID

        Emits:
            review.event.subscribe
        """
        subscription = EventSubscription(
            subscription_id="",
            topic_pattern=topic_pattern,
            handler=handler,
            type_pattern=type_pattern,
            priority_filter=priority_filter,
            filter_fn=filter_fn,
        )

        self._subscriptions[subscription.subscription_id] = subscription

        self._emit_bus_event(self.BUS_TOPICS["subscribe"], {
            "subscription_id": subscription.subscription_id,
            "topic_pattern": topic_pattern,
            "type_pattern": type_pattern,
        })

        return subscription.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if unsubscribed
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    async def emit(self, event: Event) -> str:
        """
        Emit an event.

        Args:
            event: Event to emit

        Returns:
            Event ID

        Emits:
            review.event.emit
        """
        # Check for duplicates
        if self._is_duplicate(event):
            return event.event_id

        self._stats["events_emitted"] += 1

        # Add to queue or process directly
        if self._running:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                # Process immediately if queue full
                await self._deliver_event(event)
        else:
            await self._deliver_event(event)

        # Emit to bus
        self._emit_bus_event(self.BUS_TOPICS["emit"], {
            "event_id": event.event_id,
            "topic": event.topic,
            "type": event.type,
            "priority": event.priority.name,
        })

        return event.event_id

    async def emit_event(
        self,
        topic: str,
        type: str,
        payload: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Emit an event with convenience parameters.

        Args:
            topic: Event topic
            type: Event type
            payload: Event payload
            priority: Event priority
            correlation_id: Correlation ID
            metadata: Additional metadata

        Returns:
            Event ID
        """
        event = Event(
            event_id="",
            topic=topic,
            type=type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )
        return await self.emit(event)

    async def _deliver_event(self, event: Event) -> None:
        """Deliver event to matching subscribers."""
        matching_subs = [
            sub for sub in self._subscriptions.values()
            if sub.matches(event)
        ]

        # Sort by priority
        matching_subs.sort(key=lambda s: s.created_at)

        for sub in matching_subs:
            record = DeliveryRecord(
                event_id=event.event_id,
                subscription_id=sub.subscription_id,
                state=EventState.PENDING,
            )

            success = await self._attempt_delivery(event, sub, record)

            if success:
                self._stats["events_delivered"] += 1
            else:
                self._stats["events_failed"] += 1
                if self.config.dead_letter_enabled:
                    self._dead_letter.append(event)
                    self._stats["dead_letters"] += 1
                    self._emit_bus_event(self.BUS_TOPICS["dead_letter"], {
                        "event_id": event.event_id,
                        "subscription_id": sub.subscription_id,
                        "error": record.error,
                    })

            self._delivery_records[f"{event.event_id}:{sub.subscription_id}"] = record

    async def _attempt_delivery(
        self,
        event: Event,
        subscription: EventSubscription,
        record: DeliveryRecord,
    ) -> bool:
        """Attempt to deliver event to subscription."""
        for attempt in range(self.config.max_retries + 1):
            record.attempts = attempt + 1
            record.last_attempt = time.time()

            try:
                await subscription.handler(event)
                record.state = EventState.DELIVERED
                return True

            except Exception as e:
                record.error = str(e)

                if self.config.delivery == EventDelivery.AT_MOST_ONCE:
                    record.state = EventState.FAILED
                    return False

                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))

        record.state = EventState.DEAD_LETTER if self.config.dead_letter_enabled else EventState.FAILED
        return False

    async def start(self) -> None:
        """Start the event processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        """Stop the event processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

    async def _process_queue(self) -> None:
        """Process events from queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                await self._deliver_event(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def get_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all subscriptions."""
        return [s.to_dict() for s in self._subscriptions.values()]

    def get_dead_letters(self) -> List[Dict[str, Any]]:
        """Get dead letter events."""
        return [e.to_dict() for e in self._dead_letter]

    def clear_dead_letters(self) -> int:
        """Clear dead letter queue."""
        count = len(self._dead_letter)
        self._dead_letter.clear()
        return count

    async def replay_dead_letters(self) -> int:
        """Replay dead letter events."""
        count = 0
        events = list(self._dead_letter)
        self._dead_letter.clear()

        for event in events:
            await self.emit(event)
            count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        return {
            **self._stats,
            "subscriptions": len(self._subscriptions),
            "queue_size": self._queue.qsize(),
            "dead_letter_count": len(self._dead_letter),
        }

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        stats = self.get_stats()
        status = {
            "agent": "event-emitter",
            "healthy": True,
            "running": self._running,
            "subscriptions": stats["subscriptions"],
            "events_emitted": stats["events_emitted"],
            "events_delivered": stats["events_delivered"],
            "dead_letters": stats["dead_letter_count"],
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_bus_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Event Emitter."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Event Emitter (Step 190)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Emit command
    emit_parser = subparsers.add_parser("emit", help="Emit an event")
    emit_parser.add_argument("topic", help="Event topic")
    emit_parser.add_argument("type", help="Event type")
    emit_parser.add_argument("--payload", default="{}", help="Event payload (JSON)")
    emit_parser.add_argument("--priority", choices=["critical", "high", "normal", "low"],
                             default="normal", help="Priority")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Subscriptions command
    subparsers.add_parser("subs", help="List subscriptions")

    # Dead letters command
    dl_parser = subparsers.add_parser("deadletters", help="Manage dead letters")
    dl_parser.add_argument("--clear", action="store_true", help="Clear dead letters")
    dl_parser.add_argument("--replay", action="store_true", help="Replay dead letters")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo")
    demo_parser.add_argument("--events", type=int, default=5, help="Number of events")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    emitter = EventEmitter()

    if args.command == "emit":
        payload = json.loads(args.payload)
        priority = {
            "critical": EventPriority.CRITICAL,
            "high": EventPriority.HIGH,
            "normal": EventPriority.NORMAL,
            "low": EventPriority.LOW,
        }[args.priority]

        event_id = asyncio.run(emitter.emit_event(
            topic=args.topic,
            type=args.type,
            payload=payload,
            priority=priority,
        ))

        if args.json:
            print(json.dumps({"event_id": event_id}, indent=2))
        else:
            print(f"Emitted event: {event_id}")

    elif args.command == "stats":
        stats = emitter.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Event Emitter Statistics")
            print(f"  Events Emitted: {stats['events_emitted']}")
            print(f"  Events Delivered: {stats['events_delivered']}")
            print(f"  Events Failed: {stats['events_failed']}")
            print(f"  Dead Letters: {stats['dead_letter_count']}")
            print(f"  Subscriptions: {stats['subscriptions']}")

    elif args.command == "subs":
        subs = emitter.get_subscriptions()
        if args.json:
            print(json.dumps(subs, indent=2))
        else:
            print(f"Subscriptions: {len(subs)}")
            for s in subs:
                print(f"  {s['subscription_id']}: {s['topic_pattern']}/{s['type_pattern']}")

    elif args.command == "deadletters":
        if args.clear:
            count = emitter.clear_dead_letters()
            print(f"Cleared {count} dead letters")
        elif args.replay:
            count = asyncio.run(emitter.replay_dead_letters())
            print(f"Replayed {count} dead letters")
        else:
            dls = emitter.get_dead_letters()
            if args.json:
                print(json.dumps(dls, indent=2))
            else:
                print(f"Dead Letters: {len(dls)}")
                for dl in dls[:10]:
                    print(f"  {dl['event_id']}: {dl['topic']}/{dl['type']}")

    elif args.command == "demo":
        # Subscribe to events
        received = []

        async def handler(event: Event):
            received.append(event)
            if not args.json:
                print(f"  Received: {event.topic}/{event.type}")

        emitter.subscribe("demo.*", handler)

        async def run_demo():
            for i in range(args.events):
                await emitter.emit_event(
                    topic="demo.test",
                    type="test_event",
                    payload={"index": i},
                )
            await asyncio.sleep(0.1)  # Allow processing

        if not args.json:
            print(f"Emitting {args.events} demo events...")

        asyncio.run(run_demo())

        if args.json:
            print(json.dumps({
                "emitted": args.events,
                "received": len(received),
            }, indent=2))
        else:
            print(f"\nEmitted: {args.events}, Received: {len(received)}")

    else:
        # Default: show status
        status = emitter.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Event Emitter: {status['events_emitted']} emitted, "
                  f"{status['subscriptions']} subscriptions")

    return 0


if __name__ == "__main__":
    sys.exit(main())
