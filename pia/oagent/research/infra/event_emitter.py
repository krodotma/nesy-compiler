#!/usr/bin/env python3
"""
event_emitter.py - Event Emission System (Step 40)

Event emission and subscription system for Research Agent.
Provides pub/sub patterns with filtering and replay support.

PBTSO Phase: COORDINATE

Bus Topics:
- a2a.research.event.emit
- a2a.research.event.subscribe
- research.event.replay

Protocol: DKIN v30, PAIP v16, CITIZEN v2
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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, TypeVar, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class EventConfig:
    """Configuration for event emitter."""

    max_listeners_per_topic: int = 100
    max_replay_events: int = 1000
    enable_replay: bool = True
    enable_wildcards: bool = True
    emit_to_bus: bool = True
    async_emit: bool = True
    bus_path: Optional[str] = None
    ndjson_rotation_mb: int = 10  # Per DKIN protocol

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Event:
    """An event to be emitted."""

    topic: str
    data: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = "research-agent"
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat() + "Z",
            "source": self.source,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            topic=data["topic"],
            data=data.get("data"),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "unknown"),
            priority=EventPriority(data.get("priority", 1)),
            metadata=data.get("metadata", {}),
            correlation_id=data.get("correlation_id"),
        )


EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]


@dataclass
class EventFilter:
    """Filter for event subscriptions."""

    topics: List[str] = field(default_factory=list)  # Wildcard patterns supported
    sources: List[str] = field(default_factory=list)
    min_priority: EventPriority = EventPriority.LOW
    metadata_match: Dict[str, Any] = field(default_factory=dict)

    def matches(self, event: Event) -> bool:
        """Check if event matches this filter."""
        # Topic match
        if self.topics:
            topic_match = any(
                fnmatch(event.topic, pattern) for pattern in self.topics
            )
            if not topic_match:
                return False

        # Source match
        if self.sources and event.source not in self.sources:
            return False

        # Priority match
        if event.priority.value < self.min_priority.value:
            return False

        # Metadata match
        for key, value in self.metadata_match.items():
            if event.metadata.get(key) != value:
                return False

        return True


@dataclass
class Subscription:
    """An event subscription."""

    id: str
    handler: Union[EventHandler, AsyncEventHandler]
    filter: EventFilter
    is_async: bool = False
    once: bool = False  # Unsubscribe after first event
    created_at: float = field(default_factory=time.time)
    event_count: int = 0


# ============================================================================
# Event Emitter
# ============================================================================


class EventEmitter:
    """
    Event emission and subscription system.

    Features:
    - Pub/sub with topic patterns
    - Sync and async handlers
    - Event filtering
    - Event replay from history
    - Bus integration

    PBTSO Phase: COORDINATE

    Example:
        emitter = EventEmitter()

        # Subscribe
        def handler(event):
            print(f"Received: {event.topic}")

        emitter.on("research.*", handler)

        # Emit
        emitter.emit("research.query", {"query": "test"})

        # With async
        @emitter.on_async("research.complete")
        async def async_handler(event):
            await process(event)

        # Replay
        emitter.replay("research.*", since=time.time() - 3600)
    """

    def __init__(
        self,
        config: Optional[EventConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the event emitter.

        Args:
            config: Event emitter configuration
            bus: AgentBus for event emission
        """
        self.config = config or EventConfig()
        self.bus = bus or AgentBus()

        # Subscriptions
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._subscription_lock = threading.RLock()

        # Event history for replay
        self._history: Deque[Event] = deque(maxlen=self.config.max_replay_events)
        self._history_lock = threading.Lock()

        # Statistics
        self._stats = {
            "events_emitted": 0,
            "events_handled": 0,
            "subscriptions_active": 0,
            "handlers_invoked": 0,
        }

        # Async loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def on(
        self,
        topic: str,
        handler: EventHandler,
        filter: Optional[EventFilter] = None,
        once: bool = False,
    ) -> str:
        """
        Subscribe to events.

        Args:
            topic: Topic pattern (supports * wildcard)
            handler: Sync handler function
            filter: Optional event filter
            once: If True, unsubscribe after first event

        Returns:
            Subscription ID
        """
        return self._add_subscription(topic, handler, filter, once, is_async=False)

    def on_async(
        self,
        topic: str,
        handler: Optional[AsyncEventHandler] = None,
        filter: Optional[EventFilter] = None,
        once: bool = False,
    ) -> Union[str, Callable]:
        """
        Subscribe to events with async handler.

        Can be used as decorator or directly.

        Args:
            topic: Topic pattern
            handler: Async handler function
            filter: Optional event filter
            once: If True, unsubscribe after first event

        Returns:
            Subscription ID or decorator
        """
        if handler is None:
            # Used as decorator
            def decorator(fn: AsyncEventHandler) -> AsyncEventHandler:
                self._add_subscription(topic, fn, filter, once, is_async=True)
                return fn
            return decorator

        return self._add_subscription(topic, handler, filter, once, is_async=True)

    def once(self, topic: str, handler: EventHandler) -> str:
        """Subscribe to receive only the first matching event."""
        return self.on(topic, handler, once=True)

    def off(self, subscription_id: str) -> bool:
        """
        Unsubscribe by subscription ID.

        Args:
            subscription_id: ID returned from on()

        Returns:
            True if unsubscribed
        """
        with self._subscription_lock:
            for topic, subs in self._subscriptions.items():
                for sub in subs[:]:
                    if sub.id == subscription_id:
                        subs.remove(sub)
                        self._stats["subscriptions_active"] -= 1
                        return True
        return False

    def off_all(self, topic: Optional[str] = None) -> int:
        """
        Remove all subscriptions for a topic.

        Args:
            topic: Topic to clear (None = all topics)

        Returns:
            Number of subscriptions removed
        """
        with self._subscription_lock:
            if topic is None:
                count = sum(len(subs) for subs in self._subscriptions.values())
                self._subscriptions.clear()
                self._stats["subscriptions_active"] = 0
                return count

            if topic in self._subscriptions:
                count = len(self._subscriptions[topic])
                del self._subscriptions[topic]
                self._stats["subscriptions_active"] -= count
                return count

            return 0

    def emit(
        self,
        topic: str,
        data: Any = None,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Event:
        """
        Emit an event.

        Args:
            topic: Event topic
            data: Event data
            priority: Event priority
            metadata: Optional metadata
            correlation_id: Optional correlation ID

        Returns:
            The emitted event
        """
        event = Event(
            topic=topic,
            data=data,
            priority=priority,
            metadata=metadata or {},
            correlation_id=correlation_id,
        )

        return self.emit_event(event)

    def emit_event(self, event: Event) -> Event:
        """
        Emit a pre-constructed event.

        Args:
            event: Event to emit

        Returns:
            The emitted event
        """
        self._stats["events_emitted"] += 1

        # Store in history
        if self.config.enable_replay:
            with self._history_lock:
                self._history.append(event)

        # Find matching subscriptions
        subscriptions_to_invoke = []
        subscriptions_to_remove = []

        with self._subscription_lock:
            for topic_pattern, subs in self._subscriptions.items():
                if self._matches_topic(event.topic, topic_pattern):
                    for sub in subs:
                        if sub.filter.matches(event):
                            subscriptions_to_invoke.append(sub)
                            if sub.once:
                                subscriptions_to_remove.append((topic_pattern, sub))

        # Invoke handlers
        for sub in subscriptions_to_invoke:
            self._invoke_handler(sub, event)

        # Remove one-time subscriptions
        with self._subscription_lock:
            for topic_pattern, sub in subscriptions_to_remove:
                if topic_pattern in self._subscriptions and sub in self._subscriptions[topic_pattern]:
                    self._subscriptions[topic_pattern].remove(sub)
                    self._stats["subscriptions_active"] -= 1

        # Emit to bus
        if self.config.emit_to_bus:
            self._emit_to_bus(event)

        return event

    def replay(
        self,
        topic_pattern: str,
        since: Optional[float] = None,
        handler: Optional[EventHandler] = None,
    ) -> List[Event]:
        """
        Replay historical events.

        Args:
            topic_pattern: Topic pattern to filter
            since: Only events after this timestamp
            handler: Optional handler to invoke for each event

        Returns:
            List of matching events
        """
        if not self.config.enable_replay:
            return []

        matching_events = []

        with self._history_lock:
            for event in self._history:
                if since and event.timestamp < since:
                    continue

                if self._matches_topic(event.topic, topic_pattern):
                    matching_events.append(event)

                    if handler:
                        try:
                            handler(event)
                        except Exception:
                            pass

        return matching_events

    async def replay_async(
        self,
        topic_pattern: str,
        since: Optional[float] = None,
        handler: Optional[AsyncEventHandler] = None,
    ) -> List[Event]:
        """Async version of replay."""
        if not self.config.enable_replay:
            return []

        matching_events = []

        with self._history_lock:
            events_copy = list(self._history)

        for event in events_copy:
            if since and event.timestamp < since:
                continue

            if self._matches_topic(event.topic, topic_pattern):
                matching_events.append(event)

                if handler:
                    try:
                        result = handler(event)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass

        return matching_events

    def wait_for(
        self,
        topic: str,
        timeout: Optional[float] = None,
        filter: Optional[EventFilter] = None,
    ) -> Optional[Event]:
        """
        Wait for a specific event (blocking).

        Args:
            topic: Topic to wait for
            timeout: Timeout in seconds
            filter: Optional filter

        Returns:
            The event, or None if timeout
        """
        result_event: List[Optional[Event]] = [None]
        event_received = threading.Event()

        def handler(event: Event):
            result_event[0] = event
            event_received.set()

        sub_id = self.on(topic, handler, filter=filter, once=True)

        try:
            event_received.wait(timeout=timeout)
            return result_event[0]
        finally:
            self.off(sub_id)

    async def wait_for_async(
        self,
        topic: str,
        timeout: Optional[float] = None,
        filter: Optional[EventFilter] = None,
    ) -> Optional[Event]:
        """
        Wait for a specific event (async).

        Args:
            topic: Topic to wait for
            timeout: Timeout in seconds
            filter: Optional filter

        Returns:
            The event, or None if timeout
        """
        future: asyncio.Future[Event] = asyncio.get_event_loop().create_future()

        async def handler(event: Event):
            if not future.done():
                future.set_result(event)

        sub_id = self.on_async(topic, handler, filter=filter, once=True)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.off(sub_id)

    def listeners(self, topic: Optional[str] = None) -> int:
        """Get number of listeners for a topic."""
        with self._subscription_lock:
            if topic is None:
                return sum(len(subs) for subs in self._subscriptions.values())

            if topic in self._subscriptions:
                return len(self._subscriptions[topic])

            return 0

    def topics(self) -> List[str]:
        """Get all subscribed topics."""
        with self._subscription_lock:
            return list(self._subscriptions.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get event emitter statistics."""
        with self._subscription_lock:
            self._stats["subscriptions_active"] = sum(
                len(subs) for subs in self._subscriptions.values()
            )

        return {
            **self._stats,
            "history_size": len(self._history) if self.config.enable_replay else 0,
            "topics": len(self._subscriptions),
        }

    def clear_history(self) -> None:
        """Clear event history."""
        with self._history_lock:
            self._history.clear()

    def _add_subscription(
        self,
        topic: str,
        handler: Union[EventHandler, AsyncEventHandler],
        filter: Optional[EventFilter],
        once: bool,
        is_async: bool,
    ) -> str:
        """Add a subscription."""
        sub_id = str(uuid.uuid4())[:8]

        # Create filter
        event_filter = filter or EventFilter(topics=[topic])
        if not event_filter.topics:
            event_filter.topics = [topic]

        subscription = Subscription(
            id=sub_id,
            handler=handler,
            filter=event_filter,
            is_async=is_async,
            once=once,
        )

        with self._subscription_lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = []

            if len(self._subscriptions[topic]) >= self.config.max_listeners_per_topic:
                raise ValueError(f"Max listeners ({self.config.max_listeners_per_topic}) exceeded for topic {topic}")

            self._subscriptions[topic].append(subscription)
            self._stats["subscriptions_active"] += 1

        return sub_id

    def _invoke_handler(self, sub: Subscription, event: Event) -> None:
        """Invoke a subscription handler."""
        self._stats["handlers_invoked"] += 1
        sub.event_count += 1

        try:
            if sub.is_async:
                # Run async handler
                if self.config.async_emit:
                    self._run_async(sub.handler(event))
                else:
                    asyncio.run(sub.handler(event))
            else:
                sub.handler(event)

            self._stats["events_handled"] += 1

        except Exception:
            pass  # Don't let handler errors stop other handlers

    def _run_async(self, coro: Any) -> None:
        """Run a coroutine in the event loop."""
        if asyncio.iscoroutine(coro):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(coro)
                else:
                    loop.run_until_complete(coro)
            except RuntimeError:
                # No event loop, create one
                asyncio.run(coro)

    def _matches_topic(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern."""
        if not self.config.enable_wildcards:
            return topic == pattern

        return fnmatch(topic, pattern)

    def _emit_to_bus(self, event: Event) -> None:
        """Emit event to bus with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        bus_event = {
            "id": event.id,
            "ts": event.timestamp,
            "iso": datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat() + "Z",
            "topic": event.topic,
            "kind": "event",
            "level": "info",
            "actor": event.source,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": event.data,
            "metadata": event.metadata,
        }

        if event.correlation_id:
            bus_event["correlation_id"] = event.correlation_id

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(bus_event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ============================================================================
# Global Event Emitter
# ============================================================================


_default_emitter: Optional[EventEmitter] = None


def get_emitter(config: Optional[EventConfig] = None) -> EventEmitter:
    """Get the default event emitter."""
    global _default_emitter
    if _default_emitter is None:
        _default_emitter = EventEmitter(config)
    return _default_emitter


def emit(topic: str, data: Any = None, **kwargs) -> Event:
    """Emit an event using the default emitter."""
    return get_emitter().emit(topic, data, **kwargs)


def on(topic: str, handler: EventHandler, **kwargs) -> str:
    """Subscribe using the default emitter."""
    return get_emitter().on(topic, handler, **kwargs)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Event Emitter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Event Emitter (Step 40)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Emit command
    emit_parser = subparsers.add_parser("emit", help="Emit an event")
    emit_parser.add_argument("topic", help="Event topic")
    emit_parser.add_argument("--data", help="Event data (JSON)")
    emit_parser.add_argument("--priority", choices=["low", "normal", "high", "critical"], default="normal")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run event emitter demo")

    # Listen command
    listen_parser = subparsers.add_parser("listen", help="Listen for events")
    listen_parser.add_argument("topic", help="Topic pattern")
    listen_parser.add_argument("--timeout", type=int, default=60, help="Timeout seconds")

    args = parser.parse_args()

    emitter = EventEmitter()

    if args.command == "emit":
        data = None
        if args.data:
            data = json.loads(args.data)

        priority = EventPriority[args.priority.upper()]

        event = emitter.emit(args.topic, data, priority=priority)
        print(f"Emitted event: {event.id}")
        print(f"  Topic: {event.topic}")
        print(f"  Timestamp: {datetime.fromtimestamp(event.timestamp)}")

    elif args.command == "stats":
        stats = emitter.get_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Event Emitter Statistics:")
            print(f"  Events Emitted: {stats['events_emitted']}")
            print(f"  Events Handled: {stats['events_handled']}")
            print(f"  Handlers Invoked: {stats['handlers_invoked']}")
            print(f"  Active Subscriptions: {stats['subscriptions_active']}")
            print(f"  History Size: {stats['history_size']}")
            print(f"  Topics: {stats['topics']}")

    elif args.command == "demo":
        print("Running event emitter demo...\n")

        received_events = []

        # Subscribe
        def handler(event: Event):
            received_events.append(event)
            print(f"Received: {event.topic} - {event.data}")

        emitter.on("demo.*", handler)
        emitter.on("research.query", handler)

        # Emit some events
        print("Emitting events...")
        emitter.emit("demo.start", {"message": "Demo started"})
        emitter.emit("research.query", {"query": "test query"})
        emitter.emit("demo.progress", {"percent": 50})
        emitter.emit("demo.complete", {"result": "success"})

        print(f"\nReceived {len(received_events)} events")

        # Replay
        print("\nReplaying events...")
        replayed = emitter.replay("demo.*")
        print(f"Found {len(replayed)} matching events in history")

        print("\nDemo complete.")

    elif args.command == "listen":
        print(f"Listening for events on '{args.topic}' (timeout: {args.timeout}s)...")

        def handler(event: Event):
            print(f"[{datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')}] {event.topic}: {event.data}")

        emitter.on(args.topic, handler)

        # Wait for timeout
        try:
            time.sleep(args.timeout)
        except KeyboardInterrupt:
            print("\nStopped.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
