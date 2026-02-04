#!/usr/bin/env python3
"""
Monitor Agent Bootstrap Module - Step 251

Implements the bootstrap and lifecycle management for the Monitor Agent.

PBTSO Phase: SKILL, SEQUESTER

Bus Topics:
- a2a.monitor.bootstrap.start
- a2a.monitor.bootstrap.complete
- telemetry.* (subscribed)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class MonitorState(Enum):
    """Monitor Agent lifecycle states."""
    INITIALIZING = "initializing"
    BOOTSTRAPPING = "bootstrapping"
    RUNNING = "running"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class RingLevel(Enum):
    """Ring security levels for agent operations."""
    RING_0 = 0  # Kernel/Constitutional
    RING_1 = 1  # System agents (Monitor, Deploy)
    RING_2 = 2  # Privileged agents (Code, Test)
    RING_3 = 3  # User agents (Research, Review)


@dataclass
class MonitorAgentConfig:
    """Configuration for Monitor Agent.

    Attributes:
        agent_id: Unique identifier for this agent instance
        ring_level: Security ring level (default Ring 1 for system agent)
        metrics_retention_days: How long to retain metrics
        alert_channels: List of alert delivery channels
        slo_tracking_enabled: Whether to track SLOs
        anomaly_window_size: Window size for anomaly detection
        heartbeat_interval_s: Interval for agent heartbeats
        telemetry_topics: List of telemetry topics to subscribe
        bus_dir: Directory for bus events
    """
    agent_id: str = "monitor-agent"
    ring_level: int = 1
    metrics_retention_days: int = 30
    alert_channels: List[str] = field(default_factory=lambda: ["bus", "slack"])
    slo_tracking_enabled: bool = True
    anomaly_window_size: int = 100
    heartbeat_interval_s: int = 300
    telemetry_topics: List[str] = field(default_factory=lambda: [
        "telemetry.*",
        "a2a.*.complete",
        "a2a.health.*",
        "*.metrics",
        "*.error",
    ])
    bus_dir: Optional[str] = None

    def __post_init__(self):
        if self.bus_dir is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_dir = os.path.join(pluribus_root, ".pluribus", "bus")


@dataclass
class SubscriptionHandle:
    """Handle for a bus subscription."""
    topic_pattern: str
    handler: Callable
    subscription_id: str
    created_at: float = field(default_factory=time.time)


class MonitorAgentBootstrap:
    """
    Bootstrap and lifecycle management for Monitor Agent.

    Responsibilities:
    - Initialize monitor agent with configuration
    - Register with A2A dispatcher
    - Subscribe to telemetry topics
    - Manage agent lifecycle states
    - Emit bootstrap events to bus

    Example:
        config = MonitorAgentConfig(
            agent_id="monitor-agent-01",
            metrics_retention_days=7
        )
        bootstrap = MonitorAgentBootstrap(config)
        bootstrap.start()
    """

    BUS_TOPICS = {
        "bootstrap_start": "a2a.monitor.bootstrap.start",
        "bootstrap_complete": "a2a.monitor.bootstrap.complete",
        "health_check": "a2a.health.check",
        "health_report": "a2a.health.report",
    }

    def __init__(self, config: Optional[MonitorAgentConfig] = None):
        """Initialize Monitor Agent bootstrap.

        Args:
            config: Agent configuration, uses defaults if None
        """
        self.config = config or MonitorAgentConfig()
        self.state = MonitorState.INITIALIZING
        self.subscriptions: Dict[str, SubscriptionHandle] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        self._start_time: Optional[float] = None
        self._last_heartbeat: float = 0.0
        self._event_count: int = 0

        # Ensure bus directory exists
        self._bus_path = Path(self.config.bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> bool:
        """Start the Monitor Agent.

        Returns:
            True if started successfully
        """
        if self.state not in (MonitorState.INITIALIZING, MonitorState.STOPPED):
            return False

        self.state = MonitorState.BOOTSTRAPPING
        self._start_time = time.time()

        # Emit bootstrap start event
        self._emit_bus_event(
            self.BUS_TOPICS["bootstrap_start"],
            {
                "agent_id": self.config.agent_id,
                "ring_level": self.config.ring_level,
                "config": {
                    "metrics_retention_days": self.config.metrics_retention_days,
                    "alert_channels": self.config.alert_channels,
                    "slo_tracking_enabled": self.config.slo_tracking_enabled,
                    "telemetry_topics": self.config.telemetry_topics,
                },
            }
        )

        # Subscribe to telemetry topics
        for topic in self.config.telemetry_topics:
            self._subscribe_topic(topic)

        # Mark as running
        self.state = MonitorState.RUNNING

        # Emit bootstrap complete event
        self._emit_bus_event(
            self.BUS_TOPICS["bootstrap_complete"],
            {
                "agent_id": self.config.agent_id,
                "state": self.state.value,
                "subscriptions": list(self.subscriptions.keys()),
                "startup_duration_ms": int((time.time() - self._start_time) * 1000),
            }
        )

        return True

    def stop(self) -> bool:
        """Stop the Monitor Agent.

        Returns:
            True if stopped successfully
        """
        if self.state == MonitorState.STOPPED:
            return True

        self.state = MonitorState.SHUTTING_DOWN

        # Unsubscribe from all topics
        for sub_id in list(self.subscriptions.keys()):
            self._unsubscribe_topic(sub_id)

        self.state = MonitorState.STOPPED

        self._emit_bus_event(
            "a2a.monitor.shutdown",
            {
                "agent_id": self.config.agent_id,
                "uptime_s": time.time() - (self._start_time or time.time()),
                "events_processed": self._event_count,
            }
        )

        return True

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Monitor Agent.

        Returns:
            Health status dictionary
        """
        uptime = time.time() - (self._start_time or time.time())

        health = {
            "agent_id": self.config.agent_id,
            "state": self.state.value,
            "healthy": self.state == MonitorState.RUNNING,
            "uptime_s": uptime,
            "events_processed": self._event_count,
            "active_subscriptions": len(self.subscriptions),
            "ring_level": self.config.ring_level,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }

        self._emit_bus_event(
            self.BUS_TOPICS["health_report"],
            health
        )

        return health

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type.

        Args:
            event_type: Event type to handle
            handler: Callback function
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def dispatch_event(self, event: Dict[str, Any]) -> bool:
        """Dispatch an event to registered handlers.

        Args:
            event: Event to dispatch

        Returns:
            True if event was handled
        """
        self._event_count += 1
        topic = event.get("topic", "")

        # Find matching handlers
        handled = False
        for pattern, sub in self.subscriptions.items():
            if self._topic_matches(topic, sub.topic_pattern):
                try:
                    sub.handler(event)
                    handled = True
                except Exception as e:
                    self._emit_bus_event(
                        "monitor.handler.error",
                        {
                            "agent_id": self.config.agent_id,
                            "topic": topic,
                            "pattern": sub.topic_pattern,
                            "error": str(e),
                        },
                        level="error"
                    )

        return handled

    def emit_heartbeat(self) -> bool:
        """Emit agent heartbeat.

        Returns:
            True if heartbeat was emitted (rate limited)
        """
        now = time.time()
        if now - self._last_heartbeat < self.config.heartbeat_interval_s - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "agent_id": self.config.agent_id,
                "state": self.state.value,
                "uptime_s": now - (self._start_time or now),
                "events_processed": self._event_count,
            }
        )

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status.

        Returns:
            Status dictionary
        """
        return {
            "agent_id": self.config.agent_id,
            "state": self.state.value,
            "ring_level": self.config.ring_level,
            "start_time": self._start_time,
            "uptime_s": time.time() - (self._start_time or time.time()),
            "event_count": self._event_count,
            "subscriptions": [
                {
                    "pattern": s.topic_pattern,
                    "id": s.subscription_id,
                }
                for s in self.subscriptions.values()
            ],
            "config": asdict(self.config),
        }

    def _subscribe_topic(self, topic_pattern: str) -> str:
        """Subscribe to a topic pattern.

        Args:
            topic_pattern: Topic pattern (supports * wildcard)

        Returns:
            Subscription ID
        """
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"

        def default_handler(event: Dict[str, Any]) -> None:
            """Default event handler that forwards to registered handlers."""
            event_type = event.get("kind", "unknown")
            if event_type in self.handlers:
                for handler in self.handlers[event_type]:
                    handler(event)

        self.subscriptions[sub_id] = SubscriptionHandle(
            topic_pattern=topic_pattern,
            handler=default_handler,
            subscription_id=sub_id,
        )

        return sub_id

    def _unsubscribe_topic(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic.

        Args:
            subscription_id: Subscription to cancel

        Returns:
            True if unsubscribed
        """
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            return True
        return False

    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern.

        Args:
            topic: Actual topic
            pattern: Pattern (supports * wildcard)

        Returns:
            True if matches
        """
        if pattern == "*":
            return True

        parts = pattern.split(".")
        topic_parts = topic.split(".")

        if len(parts) > len(topic_parts) and "*" not in parts:
            return False

        for i, part in enumerate(parts):
            if part == "*":
                continue
            if i >= len(topic_parts):
                return False
            if part != topic_parts[i]:
                return False

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to the Pluribus bus.

        Args:
            topic: Event topic
            data: Event data
            level: Log level
            kind: Event kind

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.config.agent_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id


# Singleton instance
_bootstrap: Optional[MonitorAgentBootstrap] = None


def get_bootstrap(config: Optional[MonitorAgentConfig] = None) -> MonitorAgentBootstrap:
    """Get or create the Monitor Agent bootstrap singleton.

    Args:
        config: Configuration (used only on first call)

    Returns:
        MonitorAgentBootstrap instance
    """
    global _bootstrap
    if _bootstrap is None:
        _bootstrap = MonitorAgentBootstrap(config)
    return _bootstrap


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Agent Bootstrap (Step 251)")
    parser.add_argument("--start", action="store_true", help="Start the agent")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    bootstrap = get_bootstrap()

    if args.start:
        success = bootstrap.start()
        if args.json:
            print(json.dumps({"success": success, "status": bootstrap.get_status()}))
        else:
            print(f"Monitor Agent started: {success}")
            print(f"  State: {bootstrap.state.value}")
            print(f"  Agent ID: {bootstrap.config.agent_id}")

    if args.status:
        status = bootstrap.get_status()
        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print(f"Monitor Agent Status:")
            print(f"  Agent ID: {status['agent_id']}")
            print(f"  State: {status['state']}")
            print(f"  Ring Level: {status['ring_level']}")
            print(f"  Subscriptions: {len(status['subscriptions'])}")

    if args.health:
        health = bootstrap.health_check()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            icon = "V" if health["healthy"] else "X"
            print(f"{icon} Monitor Agent Health: {'healthy' if health['healthy'] else 'unhealthy'}")
            print(f"  Uptime: {health['uptime_s']:.1f}s")
            print(f"  Events: {health['events_processed']}")
