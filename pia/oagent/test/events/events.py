#!/usr/bin/env python3
"""
Step 140: Test Event Emitter

Event emission system for the Test Agent.

PBTSO Phase: OBSERVE, VERIFY, DISTRIBUTE
Bus Topics:
- test.event.emit (emits)
- test.event.subscribe (emits)
- test.event.* (emits various)

Dependencies: Steps 101-139 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set


# ============================================================================
# Constants
# ============================================================================

class EventType(Enum):
    """Standard event types."""
    # Lifecycle events
    AGENT_START = "agent.start"
    AGENT_STOP = "agent.stop"
    AGENT_HEARTBEAT = "agent.heartbeat"

    # Test events
    TEST_START = "test.start"
    TEST_COMPLETE = "test.complete"
    TEST_PASS = "test.pass"
    TEST_FAIL = "test.fail"
    TEST_SKIP = "test.skip"
    TEST_ERROR = "test.error"

    # Run events
    RUN_START = "run.start"
    RUN_COMPLETE = "run.complete"
    RUN_PROGRESS = "run.progress"

    # Coverage events
    COVERAGE_COLLECT = "coverage.collect"
    COVERAGE_REPORT = "coverage.report"

    # System events
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    # Custom event
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class Event:
    """
    An event to be emitted.

    Attributes:
        event_id: Unique event ID
        event_type: Type of event
        topic: Event topic
        data: Event data
        timestamp: Event timestamp
        source: Event source
        priority: Event priority
        correlation_id: ID for correlating events
        metadata: Additional metadata
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CUSTOM
    topic: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "test-agent"
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "source": self.source,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


@dataclass
class EventSubscription:
    """
    An event subscription.

    Attributes:
        subscription_id: Unique subscription ID
        topic_pattern: Topic pattern to match
        handler: Event handler function
        filter_fn: Optional filter function
        priority: Handler priority
        enabled: Whether subscription is enabled
    """
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic_pattern: str = "*"
    handler: Callable[[Event], None] = None
    filter_fn: Optional[Callable[[Event], bool]] = None
    priority: int = 0
    enabled: bool = True
    _compiled_pattern: Optional[Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        """Compile the topic pattern."""
        # Convert glob pattern to regex
        regex = self.topic_pattern.replace(".", r"\.")
        regex = regex.replace("*", ".*")
        regex = regex.replace("?", ".")
        self._compiled_pattern = re.compile(f"^{regex}$")

    def matches(self, topic: str) -> bool:
        """Check if topic matches the pattern."""
        if self._compiled_pattern:
            return bool(self._compiled_pattern.match(topic))
        return self.topic_pattern == topic or self.topic_pattern == "*"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subscription_id": self.subscription_id,
            "topic_pattern": self.topic_pattern,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class EventFilter:
    """
    Filter for events.

    Attributes:
        event_types: Event types to match
        topics: Topics to match
        sources: Sources to match
        min_priority: Minimum priority
    """
    event_types: Optional[List[EventType]] = None
    topics: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    min_priority: EventPriority = EventPriority.LOW

    def matches(self, event: Event) -> bool:
        """Check if event matches filter."""
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.topics and event.topic not in self.topics:
            return False
        if self.sources and event.source not in self.sources:
            return False
        if event.priority.value < self.min_priority.value:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_types": [e.value for e in self.event_types] if self.event_types else None,
            "topics": self.topics,
            "sources": self.sources,
            "min_priority": self.min_priority.value,
        }


@dataclass
class EventStats:
    """
    Event emission statistics.

    Attributes:
        total_emitted: Total events emitted
        total_delivered: Total events delivered
        total_filtered: Total events filtered out
        by_type: Count by event type
        by_topic: Count by topic
    """
    total_emitted: int = 0
    total_delivered: int = 0
    total_filtered: int = 0
    by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_topic: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    subscriptions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_emitted": self.total_emitted,
            "total_delivered": self.total_delivered,
            "total_filtered": self.total_filtered,
            "by_type": dict(self.by_type),
            "by_topic": dict(self.by_topic),
            "subscriptions": self.subscriptions,
        }


@dataclass
class EventConfig:
    """
    Configuration for the event system.

    Attributes:
        output_dir: Output directory
        enable_persistence: Persist events to disk
        max_queue_size: Maximum event queue size
        async_delivery: Use async delivery
        delivery_timeout_s: Delivery timeout
        enable_bus: Enable A2A bus integration
    """
    output_dir: str = ".pluribus/test-agent/events"
    enable_persistence: bool = True
    max_queue_size: int = 10000
    async_delivery: bool = True
    delivery_timeout_s: float = 5.0
    enable_bus: bool = True
    history_size: int = 1000
    deduplication_window_s: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_persistence": self.enable_persistence,
            "max_queue_size": self.max_queue_size,
            "async_delivery": self.async_delivery,
            "enable_bus": self.enable_bus,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class EventBus:
    """Bus interface for events with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Test Event Emitter
# ============================================================================

class TestEventEmitter:
    """
    Event emission system for the Test Agent.

    Features:
    - Topic-based pub/sub
    - Pattern matching subscriptions
    - Event filtering
    - Async delivery
    - A2A bus integration
    - Event history

    PBTSO Phase: OBSERVE, VERIFY, DISTRIBUTE
    Bus Topics: test.event.emit, test.event.subscribe, test.event.*
    """

    def __init__(self, bus=None, config: Optional[EventConfig] = None):
        """
        Initialize the event emitter.

        Args:
            bus: Optional bus instance
            config: Event configuration
        """
        self.bus = bus or EventBus()
        self.config = config or EventConfig()
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._stats = EventStats()
        self._history: List[Event] = []
        self._recent_events: Set[str] = set()
        self._lock = threading.RLock()
        self._delivery_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Start delivery executor if async
        if self.config.async_delivery:
            import concurrent.futures
            self._delivery_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def emit(
        self,
        event_type: EventType,
        topic: str,
        data: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Emit an event.

        Args:
            event_type: Type of event
            topic: Event topic
            data: Event data
            priority: Event priority
            correlation_id: Correlation ID
            metadata: Additional metadata

        Returns:
            Emitted event
        """
        event = Event(
            event_type=event_type,
            topic=topic,
            data=data or {},
            priority=priority,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        return self.emit_event(event)

    def emit_event(self, event: Event) -> Event:
        """
        Emit an event object.

        Args:
            event: Event to emit

        Returns:
            Emitted event
        """
        # Deduplication check
        dedup_key = f"{event.topic}:{hash(frozenset(event.data.items()))}"
        with self._lock:
            if dedup_key in self._recent_events:
                return event
            self._recent_events.add(dedup_key)

            # Clean old dedup entries (simple approach)
            if len(self._recent_events) > 10000:
                self._recent_events.clear()

        # Update stats
        self._stats.total_emitted += 1
        self._stats.by_type[event.event_type.value] += 1
        self._stats.by_topic[event.topic] = self._stats.by_topic.get(event.topic, 0) + 1

        # Add to history
        with self._lock:
            self._history.append(event)
            if len(self._history) > self.config.history_size:
                self._history = self._history[-self.config.history_size:]

        # Emit to A2A bus
        if self.config.enable_bus:
            self.bus.emit({
                "topic": f"test.event.{event.topic}",
                "kind": "event",
                "actor": event.source,
                "data": event.to_dict(),
            })

        # Persist event
        if self.config.enable_persistence:
            self._persist_event(event)

        # Deliver to subscribers
        self._deliver_event(event)

        return event

    def _deliver_event(self, event: Event) -> None:
        """Deliver event to matching subscribers."""
        matching = []

        with self._lock:
            for subscription in self._subscriptions.values():
                if not subscription.enabled:
                    continue
                if not subscription.matches(event.topic):
                    continue
                if subscription.filter_fn and not subscription.filter_fn(event):
                    self._stats.total_filtered += 1
                    continue
                matching.append(subscription)

        # Sort by priority
        matching.sort(key=lambda s: s.priority, reverse=True)

        # Deliver
        for subscription in matching:
            if self.config.async_delivery and self._delivery_executor:
                self._delivery_executor.submit(
                    self._safe_deliver, subscription, event
                )
            else:
                self._safe_deliver(subscription, event)

    def _safe_deliver(self, subscription: EventSubscription, event: Event) -> None:
        """Safely deliver event to subscription."""
        try:
            if subscription.handler:
                subscription.handler(event)
                self._stats.total_delivered += 1
        except Exception as e:
            self.emit(
                EventType.ERROR,
                "event.delivery.error",
                data={
                    "subscription_id": subscription.subscription_id,
                    "event_id": event.event_id,
                    "error": str(e),
                },
            )

    def subscribe(
        self,
        topic_pattern: str,
        handler: Callable[[Event], None],
        filter_fn: Optional[Callable[[Event], bool]] = None,
        priority: int = 0,
    ) -> str:
        """
        Subscribe to events.

        Args:
            topic_pattern: Topic pattern (supports * wildcard)
            handler: Event handler function
            filter_fn: Optional filter function
            priority: Handler priority

        Returns:
            Subscription ID
        """
        subscription = EventSubscription(
            topic_pattern=topic_pattern,
            handler=handler,
            filter_fn=filter_fn,
            priority=priority,
        )

        with self._lock:
            self._subscriptions[subscription.subscription_id] = subscription
            self._stats.subscriptions = len(self._subscriptions)

        self.bus.emit({
            "topic": "test.event.subscribe",
            "kind": "subscription",
            "actor": "test-agent",
            "data": {
                "subscription_id": subscription.subscription_id,
                "topic_pattern": topic_pattern,
            },
        })

        return subscription.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID

        Returns:
            True if unsubscribed
        """
        with self._lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                self._stats.subscriptions = len(self._subscriptions)
                return True
        return False

    def on(self, topic_pattern: str) -> Callable:
        """
        Decorator for subscribing to events.

        Usage:
            @emitter.on("test.*")
            def handle_test_event(event):
                ...
        """
        def decorator(handler: Callable[[Event], None]) -> Callable:
            self.subscribe(topic_pattern, handler)
            return handler
        return decorator

    def once(self, topic_pattern: str, handler: Callable[[Event], None]) -> str:
        """
        Subscribe for a single event.

        Args:
            topic_pattern: Topic pattern
            handler: Event handler

        Returns:
            Subscription ID
        """
        subscription_id = None

        def one_time_handler(event: Event) -> None:
            try:
                handler(event)
            finally:
                if subscription_id:
                    self.unsubscribe(subscription_id)

        subscription_id = self.subscribe(topic_pattern, one_time_handler)
        return subscription_id

    def get_history(
        self,
        filter: Optional[EventFilter] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get event history.

        Args:
            filter: Optional event filter
            limit: Maximum events to return

        Returns:
            List of events
        """
        with self._lock:
            events = list(self._history)

        if filter:
            events = [e for e in events if filter.matches(e)]

        return events[-limit:]

    def get_stats(self) -> EventStats:
        """Get event statistics."""
        return self._stats

    def list_subscriptions(self) -> List[EventSubscription]:
        """List all subscriptions."""
        with self._lock:
            return list(self._subscriptions.values())

    def clear_history(self) -> int:
        """Clear event history."""
        with self._lock:
            count = len(self._history)
            self._history.clear()
            return count

    def _persist_event(self, event: Event) -> None:
        """Persist event to disk."""
        events_file = Path(self.config.output_dir) / "events.ndjson"

        try:
            with open(events_file, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event.to_dict()) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    async def emit_async(
        self,
        event_type: EventType,
        topic: str,
        data: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> Event:
        """Async version of emit."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.emit, event_type, topic, data, priority, correlation_id, None
        )

    async def wait_for(
        self,
        topic_pattern: str,
        timeout_s: float = 30.0,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> Optional[Event]:
        """
        Wait for an event matching the pattern.

        Args:
            topic_pattern: Topic pattern
            timeout_s: Timeout in seconds
            filter_fn: Optional filter function

        Returns:
            Matching event or None if timeout
        """
        result_event: Optional[Event] = None
        event_received = asyncio.Event()

        def handler(event: Event) -> None:
            nonlocal result_event
            result_event = event
            event_received.set()

        subscription_id = self.subscribe(topic_pattern, handler, filter_fn)

        try:
            await asyncio.wait_for(event_received.wait(), timeout=timeout_s)
            return result_event
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(subscription_id)

    def shutdown(self) -> None:
        """Shutdown the event emitter."""
        if self._delivery_executor:
            self._delivery_executor.shutdown(wait=True)


# ============================================================================
# Convenience functions
# ============================================================================

# Global emitter instance
_global_emitter: Optional[TestEventEmitter] = None


def get_emitter() -> TestEventEmitter:
    """Get or create global emitter."""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = TestEventEmitter()
    return _global_emitter


def emit(
    event_type: EventType,
    topic: str,
    data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Event:
    """Emit event using global emitter."""
    return get_emitter().emit(event_type, topic, data, **kwargs)


def subscribe(
    topic_pattern: str,
    handler: Callable[[Event], None],
    **kwargs,
) -> str:
    """Subscribe using global emitter."""
    return get_emitter().subscribe(topic_pattern, handler, **kwargs)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Event Emitter."""
    import argparse
    import concurrent.futures

    parser = argparse.ArgumentParser(description="Test Event Emitter")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Emit command
    emit_parser = subparsers.add_parser("emit", help="Emit an event")
    emit_parser.add_argument("topic", help="Event topic")
    emit_parser.add_argument("--type", default="custom",
                            choices=["custom", "test.start", "test.complete", "test.pass", "test.fail"])
    emit_parser.add_argument("--data", type=json.loads, default={}, help="Event data (JSON)")
    emit_parser.add_argument("--priority", choices=["low", "normal", "high", "critical"],
                            default="normal")

    # History command
    history_parser = subparsers.add_parser("history", help="Show event history")
    history_parser.add_argument("--limit", type=int, default=20)
    history_parser.add_argument("--topic", help="Filter by topic")
    history_parser.add_argument("--type", help="Filter by event type")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show event statistics")

    # Subscriptions command
    subs_parser = subparsers.add_parser("subscriptions", help="List subscriptions")

    # Listen command
    listen_parser = subparsers.add_parser("listen", help="Listen for events")
    listen_parser.add_argument("pattern", default="*", nargs="?", help="Topic pattern")
    listen_parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/events")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = EventConfig(output_dir=args.output)
    emitter = TestEventEmitter(config=config)

    if args.command == "emit":
        priority_map = {
            "low": EventPriority.LOW,
            "normal": EventPriority.NORMAL,
            "high": EventPriority.HIGH,
            "critical": EventPriority.CRITICAL,
        }

        event_type_map = {
            "custom": EventType.CUSTOM,
            "test.start": EventType.TEST_START,
            "test.complete": EventType.TEST_COMPLETE,
            "test.pass": EventType.TEST_PASS,
            "test.fail": EventType.TEST_FAIL,
        }

        event = emitter.emit(
            event_type=event_type_map.get(args.type, EventType.CUSTOM),
            topic=args.topic,
            data=args.data,
            priority=priority_map.get(args.priority, EventPriority.NORMAL),
        )

        if args.json:
            print(json.dumps(event.to_dict(), indent=2))
        else:
            print(f"Emitted: {event.event_id}")
            print(f"  Topic: {event.topic}")
            print(f"  Type: {event.event_type.value}")

    elif args.command == "history":
        filter_obj = None
        if args.topic or args.type:
            filter_obj = EventFilter(
                topics=[args.topic] if args.topic else None,
                event_types=[EventType(args.type)] if args.type else None,
            )

        events = emitter.get_history(filter_obj, args.limit)

        if args.json:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            print(f"\nEvent History ({len(events)}):")
            for event in events:
                dt = datetime.fromtimestamp(event.timestamp)
                print(f"  [{event.event_type.value}] {event.topic}")
                print(f"    ID: {event.event_id[:8]}..., Time: {dt.strftime('%H:%M:%S.%f')[:-3]}")

    elif args.command == "stats":
        stats = emitter.get_stats()

        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("\nEvent Statistics:")
            print(f"  Total Emitted: {stats.total_emitted}")
            print(f"  Total Delivered: {stats.total_delivered}")
            print(f"  Total Filtered: {stats.total_filtered}")
            print(f"  Subscriptions: {stats.subscriptions}")

            if stats.by_type:
                print("\n  By Type:")
                for event_type, count in stats.by_type.items():
                    print(f"    {event_type}: {count}")

    elif args.command == "subscriptions":
        subs = emitter.list_subscriptions()

        if args.json:
            print(json.dumps([s.to_dict() for s in subs], indent=2))
        else:
            print(f"\nSubscriptions ({len(subs)}):")
            for sub in subs:
                enabled = "[ON]" if sub.enabled else "[OFF]"
                print(f"  {enabled} {sub.subscription_id[:8]}...")
                print(f"      Pattern: {sub.topic_pattern}, Priority: {sub.priority}")

    elif args.command == "listen":
        print(f"Listening for events matching '{args.pattern}'...")
        print("Press Ctrl+C to stop\n")

        def handler(event: Event):
            dt = datetime.fromtimestamp(event.timestamp)
            print(f"[{dt.strftime('%H:%M:%S')}] {event.topic}: {json.dumps(event.data)[:80]}")

        emitter.subscribe(args.pattern, handler)

        try:
            time.sleep(args.timeout)
        except KeyboardInterrupt:
            print("\nStopped")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
