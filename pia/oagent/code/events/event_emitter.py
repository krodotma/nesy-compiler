#!/usr/bin/env python3
"""
event_emitter.py - Event Emission System (Step 90)

PBTSO Phase: All Phases

Provides:
- Type-safe event emission
- Pattern-based subscription
- Event filtering and routing
- Async event handling
- Event history and replay

Bus Topics:
- code.event.emitted
- code.event.handled
- code.event.error

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Pattern, Set, TypeVar, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class EventPriority(IntEnum):
    """Event handler priority."""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class EventConfig:
    """Configuration for event system."""
    max_listeners: int = 100
    enable_history: bool = True
    history_size: int = 1000
    enable_wildcards: bool = True
    default_priority: EventPriority = EventPriority.NORMAL
    async_handlers: bool = True
    propagation_enabled: bool = True
    bus_enabled: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_listeners": self.max_listeners,
            "enable_history": self.enable_history,
            "history_size": self.history_size,
            "enable_wildcards": self.enable_wildcards,
            "async_handlers": self.async_handlers,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Event Types
# =============================================================================

T = TypeVar("T")


@dataclass
class Event(Generic[T]):
    """
    A typed event with payload.

    Events flow through the system carrying data
    between components.
    """
    id: str
    type: str
    data: T
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    propagation_stopped: bool = False

    def stop_propagation(self) -> None:
        """Stop event from propagating to other handlers."""
        self.propagation_stopped = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data if not callable(getattr(self.data, "to_dict", None)) else self.data.to_dict(),  # type: ignore
            "timestamp": self.timestamp,
            "source": self.source,
            "target": self.target,
            "metadata": self.metadata,
        }


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]
SyncEventHandler = Callable[[Event], None]


@dataclass
class EventSubscription:
    """Subscription to an event type."""
    id: str
    pattern: str
    handler: Union[EventHandler, SyncEventHandler]
    priority: EventPriority
    once: bool = False
    filter_func: Optional[Callable[[Event], bool]] = None
    is_async: bool = True

    def matches(self, event_type: str) -> bool:
        """Check if subscription matches event type."""
        if self.pattern == event_type:
            return True
        if "*" in self.pattern or "?" in self.pattern:
            return fnmatch.fnmatch(event_type, self.pattern)
        return False


@dataclass
class EventFilter:
    """Filter for event subscription."""
    event_types: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    targets: Optional[List[str]] = None
    metadata_match: Optional[Dict[str, Any]] = None
    custom_filter: Optional[Callable[[Event], bool]] = None

    def matches(self, event: Event) -> bool:
        """Check if event matches filter."""
        if self.event_types and event.type not in self.event_types:
            return False
        if self.sources and event.source not in self.sources:
            return False
        if self.targets and event.target not in self.targets:
            return False
        if self.metadata_match:
            for key, value in self.metadata_match.items():
                if event.metadata.get(key) != value:
                    return False
        if self.custom_filter and not self.custom_filter(event):
            return False
        return True


# =============================================================================
# Event Emitter
# =============================================================================

class EventEmitter:
    """
    Event emission and subscription system.

    PBTSO Phase: All Phases

    Features:
    - Type-safe events
    - Pattern-based subscription (wildcards)
    - Priority ordering
    - Async and sync handlers
    - Event history and replay
    - Event filtering

    Usage:
        emitter = EventEmitter()

        @emitter.on("file.changed")
        async def handle_change(event):
            print(f"File changed: {event.data}")

        emitter.emit("file.changed", {"path": "foo.py"})
    """

    BUS_TOPICS = {
        "emitted": "code.event.emitted",
        "handled": "code.event.handled",
        "error": "code.event.error",
    }

    def __init__(
        self,
        config: Optional[EventConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or EventConfig()
        self.bus = bus or LockedAgentBus()

        self._subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._history: List[Event] = []
        self._lock = Lock()

        # Statistics
        self._events_emitted = 0
        self._events_handled = 0
        self._events_failed = 0

    # =========================================================================
    # Subscription
    # =========================================================================

    def on(
        self,
        event_type: str,
        handler: Optional[Union[EventHandler, SyncEventHandler]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[Event], bool]] = None,
    ) -> Union[Callable, EventSubscription]:
        """
        Subscribe to an event type.

        Can be used as decorator or method:
            @emitter.on("event.type")
            async def handler(event): ...

            # or

            emitter.on("event.type", handler)
        """
        def decorator(func: Union[EventHandler, SyncEventHandler]) -> EventSubscription:
            return self._subscribe(event_type, func, priority, False, filter_func)

        if handler is not None:
            return decorator(handler)
        return decorator

    def once(
        self,
        event_type: str,
        handler: Optional[Union[EventHandler, SyncEventHandler]] = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Union[Callable, EventSubscription]:
        """Subscribe to an event type, triggered only once."""
        def decorator(func: Union[EventHandler, SyncEventHandler]) -> EventSubscription:
            return self._subscribe(event_type, func, priority, True, None)

        if handler is not None:
            return decorator(handler)
        return decorator

    def _subscribe(
        self,
        pattern: str,
        handler: Union[EventHandler, SyncEventHandler],
        priority: EventPriority,
        once: bool,
        filter_func: Optional[Callable[[Event], bool]],
    ) -> EventSubscription:
        """Internal subscription method."""
        subscription = EventSubscription(
            id=f"sub-{uuid.uuid4().hex[:8]}",
            pattern=pattern,
            handler=handler,
            priority=priority,
            once=once,
            filter_func=filter_func,
            is_async=asyncio.iscoroutinefunction(handler),
        )

        with self._lock:
            # Check max listeners
            total = sum(len(subs) for subs in self._subscriptions.values())
            if total >= self.config.max_listeners:
                raise ValueError(f"Max listeners ({self.config.max_listeners}) exceeded")

            self._subscriptions[pattern].append(subscription)

            # Sort by priority
            self._subscriptions[pattern].sort(key=lambda s: s.priority)

        return subscription

    def off(
        self,
        event_type: Optional[str] = None,
        subscription_id: Optional[str] = None,
        handler: Optional[Callable] = None,
    ) -> int:
        """
        Unsubscribe from events.

        Can unsubscribe by:
        - event_type: Remove all handlers for type
        - subscription_id: Remove specific subscription
        - handler: Remove all subscriptions with handler
        """
        removed = 0

        with self._lock:
            if subscription_id:
                for pattern in list(self._subscriptions.keys()):
                    before = len(self._subscriptions[pattern])
                    self._subscriptions[pattern] = [
                        s for s in self._subscriptions[pattern]
                        if s.id != subscription_id
                    ]
                    removed += before - len(self._subscriptions[pattern])

            elif event_type:
                if event_type in self._subscriptions:
                    removed = len(self._subscriptions[event_type])
                    del self._subscriptions[event_type]

            elif handler:
                for pattern in list(self._subscriptions.keys()):
                    before = len(self._subscriptions[pattern])
                    self._subscriptions[pattern] = [
                        s for s in self._subscriptions[pattern]
                        if s.handler != handler
                    ]
                    removed += before - len(self._subscriptions[pattern])

        return removed

    # =========================================================================
    # Emission
    # =========================================================================

    def emit(
        self,
        event_type: str,
        data: Any = None,
        source: str = "",
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Emit an event.

        Creates event and dispatches to all matching handlers.
        """
        event = Event(
            id=f"evt-{uuid.uuid4().hex[:12]}",
            type=event_type,
            data=data,
            source=source,
            target=target,
            metadata=metadata or {},
        )

        self._dispatch(event)
        return event

    async def emit_async(
        self,
        event_type: str,
        data: Any = None,
        source: str = "",
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Emit an event and wait for all async handlers."""
        event = Event(
            id=f"evt-{uuid.uuid4().hex[:12]}",
            type=event_type,
            data=data,
            source=source,
            target=target,
            metadata=metadata or {},
        )

        await self._dispatch_async(event)
        return event

    def _dispatch(self, event: Event) -> None:
        """Dispatch event to handlers (sync)."""
        self._events_emitted += 1

        # Add to history
        if self.config.enable_history:
            with self._lock:
                self._history.append(event)
                if len(self._history) > self.config.history_size:
                    self._history = self._history[-self.config.history_size:]

        # Emit to bus
        if self.config.bus_enabled:
            self.bus.emit({
                "topic": self.BUS_TOPICS["emitted"],
                "kind": "event",
                "actor": event.source or "event-emitter",
                "data": event.to_dict(),
            })

        # Find matching subscriptions
        handlers = self._get_matching_handlers(event)

        # Execute handlers
        to_remove: List[EventSubscription] = []

        for subscription in handlers:
            if event.propagation_stopped and self.config.propagation_enabled:
                break

            # Check filter
            if subscription.filter_func and not subscription.filter_func(event):
                continue

            try:
                if subscription.is_async:
                    # Schedule async handler
                    asyncio.create_task(subscription.handler(event))  # type: ignore
                else:
                    subscription.handler(event)  # type: ignore

                self._events_handled += 1

                if subscription.once:
                    to_remove.append(subscription)

            except Exception as e:
                self._events_failed += 1
                self._emit_error(event, subscription, e)

        # Remove once handlers
        for sub in to_remove:
            self.off(subscription_id=sub.id)

    async def _dispatch_async(self, event: Event) -> None:
        """Dispatch event to handlers (async, waits for completion)."""
        self._events_emitted += 1

        # Add to history
        if self.config.enable_history:
            with self._lock:
                self._history.append(event)
                if len(self._history) > self.config.history_size:
                    self._history = self._history[-self.config.history_size:]

        # Emit to bus
        if self.config.bus_enabled:
            self.bus.emit({
                "topic": self.BUS_TOPICS["emitted"],
                "kind": "event",
                "actor": event.source or "event-emitter",
                "data": event.to_dict(),
            })

        # Find matching subscriptions
        handlers = self._get_matching_handlers(event)

        # Execute handlers
        to_remove: List[EventSubscription] = []

        for subscription in handlers:
            if event.propagation_stopped and self.config.propagation_enabled:
                break

            if subscription.filter_func and not subscription.filter_func(event):
                continue

            try:
                if subscription.is_async:
                    await subscription.handler(event)  # type: ignore
                else:
                    subscription.handler(event)  # type: ignore

                self._events_handled += 1

                if subscription.once:
                    to_remove.append(subscription)

            except Exception as e:
                self._events_failed += 1
                self._emit_error(event, subscription, e)

        for sub in to_remove:
            self.off(subscription_id=sub.id)

    def _get_matching_handlers(self, event: Event) -> List[EventSubscription]:
        """Get all handlers matching an event."""
        handlers: List[EventSubscription] = []

        with self._lock:
            for pattern, subscriptions in self._subscriptions.items():
                for sub in subscriptions:
                    if sub.matches(event.type):
                        handlers.append(sub)

        # Sort by priority
        handlers.sort(key=lambda s: s.priority)
        return handlers

    def _emit_error(self, event: Event, subscription: EventSubscription, error: Exception) -> None:
        """Emit error event."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["error"],
            "kind": "error",
            "level": "error",
            "actor": "event-emitter",
            "data": {
                "event_id": event.id,
                "event_type": event.type,
                "subscription_id": subscription.id,
                "error": str(error),
            },
        })

    # =========================================================================
    # History and Replay
    # =========================================================================

    def get_history(
        self,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get event history."""
        with self._lock:
            events = self._history.copy()

        if event_type:
            events = [e for e in events if fnmatch.fnmatch(e.type, event_type)]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    async def replay(
        self,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
    ) -> int:
        """Replay events from history."""
        events = self.get_history(event_type, since, limit=self.config.history_size)

        for event in events:
            await self._dispatch_async(event)

        return len(events)

    # =========================================================================
    # Queries
    # =========================================================================

    def listeners(self, event_type: Optional[str] = None) -> int:
        """Get number of listeners."""
        with self._lock:
            if event_type:
                return len(self._subscriptions.get(event_type, []))
            return sum(len(subs) for subs in self._subscriptions.values())

    def event_types(self) -> List[str]:
        """Get all subscribed event types."""
        with self._lock:
            return list(self._subscriptions.keys())

    def stats(self) -> Dict[str, Any]:
        """Get emitter statistics."""
        return {
            "events_emitted": self._events_emitted,
            "events_handled": self._events_handled,
            "events_failed": self._events_failed,
            "total_subscriptions": self.listeners(),
            "event_types": len(self._subscriptions),
            "history_size": len(self._history),
            "config": self.config.to_dict(),
        }

    def clear(self) -> None:
        """Clear all subscriptions and history."""
        with self._lock:
            self._subscriptions.clear()
            self._history.clear()


# =============================================================================
# Decorator
# =============================================================================

def on_event(
    emitter: EventEmitter,
    event_type: str,
    priority: EventPriority = EventPriority.NORMAL,
) -> Callable:
    """
    Decorator for event handlers.

    Usage:
        @on_event(emitter, "file.changed")
        async def handle_file_change(event):
            print(event.data)
    """
    def decorator(func: Union[EventHandler, SyncEventHandler]) -> Union[EventHandler, SyncEventHandler]:
        emitter.on(event_type, func, priority)
        return func
    return decorator


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Event Emitter."""
    import argparse

    parser = argparse.ArgumentParser(description="Event Emitter (Step 90)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # emit command
    emit_parser = subparsers.add_parser("emit", help="Emit an event")
    emit_parser.add_argument("event_type", help="Event type")
    emit_parser.add_argument("--data", "-d", help="Event data (JSON)")
    emit_parser.add_argument("--source", "-s", help="Event source")

    # history command
    history_parser = subparsers.add_parser("history", help="Show event history")
    history_parser.add_argument("--type", "-t", help="Filter by event type")
    history_parser.add_argument("--limit", "-n", type=int, default=10)
    history_parser.add_argument("--json", action="store_true")

    # stats command
    subparsers.add_parser("stats", help="Show statistics")

    # demo command
    subparsers.add_parser("demo", help="Run event emitter demo")

    args = parser.parse_args()
    emitter = EventEmitter()

    async def run() -> int:
        if args.command == "emit":
            data = None
            if args.data:
                try:
                    data = json.loads(args.data)
                except json.JSONDecodeError:
                    data = args.data

            event = emitter.emit(
                args.event_type,
                data=data,
                source=args.source or "cli",
            )
            print(f"Emitted: {event.id}")
            print(json.dumps(event.to_dict(), indent=2))
            return 0

        elif args.command == "history":
            events = emitter.get_history(
                event_type=args.type,
                limit=args.limit,
            )

            if args.json:
                print(json.dumps([e.to_dict() for e in events], indent=2))
            else:
                print(f"Events: {len(events)}")
                for event in events:
                    print(f"  [{event.type}] {event.id}: {event.data}")

            return 0

        elif args.command == "stats":
            stats = emitter.stats()
            print(json.dumps(stats, indent=2))
            return 0

        elif args.command == "demo":
            print("Running event emitter demo...\n")

            events_received: List[str] = []

            @emitter.on("file.*")
            async def handle_file_events(event: Event):
                events_received.append(f"[file.*] {event.type}: {event.data}")

            @emitter.on("file.changed")
            async def handle_file_changed(event: Event):
                events_received.append(f"[file.changed] {event.data}")

            @emitter.once("app.started")
            async def handle_app_started(event: Event):
                events_received.append(f"[app.started] (once) {event.data}")

            # Emit events
            print("Emitting events...")
            emitter.emit("app.started", {"version": "1.0.0"})
            emitter.emit("file.changed", {"path": "foo.py"})
            emitter.emit("file.created", {"path": "bar.py"})
            emitter.emit("app.started", {"version": "1.0.1"})  # Won't trigger once

            # Wait for async handlers
            await asyncio.sleep(0.1)

            print("\nEvents received:")
            for msg in events_received:
                print(f"  {msg}")

            print("\nStatistics:")
            print(json.dumps(emitter.stats(), indent=2))

            return 0

        return 1

    return asyncio.run(run())


if __name__ == "__main__":
    import sys
    sys.exit(main())
