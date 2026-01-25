#!/usr/bin/env python3
"""
bus_mirror.py - Mirrors and analyzes bus events from the primary trunk.

Part of the pluribus_evolution bridge subsystem.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass
class BusEvent:
    """A bus event from the primary trunk."""
    id: str
    ts: float
    iso: str
    topic: str
    kind: str
    level: str
    actor: str
    data: dict[str, Any] = field(default_factory=dict)


class BusMirror:
    """
    Mirrors and analyzes bus events from the primary trunk.

    The evolution trunk observes all events from the primary trunk
    to detect patterns, anomalies, and optimization opportunities.
    """

    def __init__(
        self,
        primary_bus_path: str = "/pluribus/.pluribus/bus/events.ndjson",
        evolution_bus_path: str = "/pluribus/pluribus_evolution/.bus/events.ndjson"
    ):
        self.primary_bus_path = Path(primary_bus_path)
        self.evolution_bus_path = Path(evolution_bus_path)
        self.evolution_bus_path.parent.mkdir(parents=True, exist_ok=True)

        self.last_position: int = 0
        self.event_handlers: dict[str, list[Callable[[BusEvent], None]]] = {}

    def tail_events(self, since_position: int = 0) -> Iterator[BusEvent]:
        """Tail new events from the primary bus."""
        if not self.primary_bus_path.exists():
            return

        with open(self.primary_bus_path, "r", encoding="utf-8") as f:
            f.seek(since_position)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield BusEvent(
                        id=data.get("id", ""),
                        ts=data.get("ts", 0),
                        iso=data.get("iso", ""),
                        topic=data.get("topic", ""),
                        kind=data.get("kind", ""),
                        level=data.get("level", ""),
                        actor=data.get("actor", ""),
                        data=data.get("data", {})
                    )
                except json.JSONDecodeError:
                    continue
            self.last_position = f.tell()

    def subscribe(self, topic_prefix: str, handler: Callable[[BusEvent], None]) -> None:
        """Subscribe to events matching a topic prefix."""
        if topic_prefix not in self.event_handlers:
            self.event_handlers[topic_prefix] = []
        self.event_handlers[topic_prefix].append(handler)

    def dispatch_event(self, event: BusEvent) -> None:
        """Dispatch an event to matching handlers."""
        for prefix, handlers in self.event_handlers.items():
            if event.topic.startswith(prefix):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        print(f"[BusMirror] Handler error: {e}")

    def emit_evolution_event(self, topic: str, kind: str, level: str, data: dict) -> None:
        """Emit an event to the evolution bus."""
        import uuid

        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": f"evolution.{topic}",
            "kind": kind,
            "level": level,
            "actor": "pluribus_evolution",
            "data": data
        }

        with open(self.evolution_bus_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")

    def analyze_event_stream(self, window_seconds: float = 60.0) -> dict[str, Any]:
        """Analyze recent event stream for patterns."""
        now = time.time()
        cutoff = now - window_seconds

        topic_counts: dict[str, int] = {}
        actor_counts: dict[str, int] = {}
        error_count = 0

        for event in self.tail_events(0):
            if event.ts < cutoff:
                continue

            topic_counts[event.topic] = topic_counts.get(event.topic, 0) + 1
            actor_counts[event.actor] = actor_counts.get(event.actor, 0) + 1

            if event.level == "error":
                error_count += 1

        return {
            "window_seconds": window_seconds,
            "topic_counts": dict(sorted(topic_counts.items(), key=lambda x: -x[1])[:10]),
            "actor_counts": dict(sorted(actor_counts.items(), key=lambda x: -x[1])[:10]),
            "error_count": error_count,
            "total_events": sum(topic_counts.values()),
        }

    def run_continuous(self, poll_interval: float = 1.0) -> None:
        """Run continuous mirroring (blocking)."""
        print(f"[BusMirror] Watching {self.primary_bus_path}")
        print(f"[BusMirror] Writing to {self.evolution_bus_path}")

        while True:
            for event in self.tail_events(self.last_position):
                self.dispatch_event(event)

                # Mirror significant events
                if event.level in ("error", "warn") or event.kind == "artifact":
                    self.emit_evolution_event(
                        topic=f"mirror.{event.topic}",
                        kind="observation",
                        level="info",
                        data={
                            "original_id": event.id,
                            "original_topic": event.topic,
                            "original_actor": event.actor,
                            "original_level": event.level,
                        }
                    )

            time.sleep(poll_interval)


if __name__ == "__main__":
    mirror = BusMirror()

    print("Analyzing primary bus event stream...")
    analysis = mirror.analyze_event_stream(window_seconds=3600)  # Last hour

    print(f"\nEvents in last hour: {analysis['total_events']}")
    print(f"Errors: {analysis['error_count']}")
    print("\nTop topics:")
    for topic, count in list(analysis['topic_counts'].items())[:5]:
        print(f"  {topic}: {count}")
    print("\nTop actors:")
    for actor, count in list(analysis['actor_counts'].items())[:5]:
        print(f"  {actor}: {count}")
