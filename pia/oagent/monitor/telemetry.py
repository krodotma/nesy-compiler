#!/usr/bin/env python3
"""
Monitor Telemetry - Step 297

Meta-telemetry module: monitoring the monitors.

PBTSO Phase: VERIFY

Bus Topics:
- telemetry.monitor.* (emitted to telemetry bucket)
- monitor.meta.telemetry (emitted)
- monitor.meta.health (emitted)

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
from typing import Any, Callable, Dict, List, Optional


class TelemetryLevel(Enum):
    """Telemetry detail levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


class ComponentType(Enum):
    """Monitor component types."""
    COLLECTOR = "collector"
    AGGREGATOR = "aggregator"
    ANALYZER = "analyzer"
    ALERTER = "alerter"
    REPORTER = "reporter"
    API = "api"
    CORE = "core"


@dataclass
class TelemetryPoint:
    """A telemetry data point.

    Attributes:
        name: Metric name
        value: Metric value
        component: Component name
        component_type: Component type
        labels: Metric labels
        timestamp: Metric timestamp
    """
    name: str
    value: float
    component: str
    component_type: ComponentType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "component": self.component,
            "component_type": self.component_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


@dataclass
class ComponentStatus:
    """Status of a monitor component.

    Attributes:
        component: Component name
        component_type: Component type
        healthy: Whether component is healthy
        last_activity: Last activity timestamp
        metrics_processed: Metrics processed count
        errors: Error count
        latency_ms: Average latency
    """
    component: str
    component_type: ComponentType
    healthy: bool = True
    last_activity: float = field(default_factory=time.time)
    metrics_processed: int = 0
    errors: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "type": self.component_type.value,
            "healthy": self.healthy,
            "last_activity": self.last_activity,
            "metrics_processed": self.metrics_processed,
            "errors": self.errors,
            "latency_ms": self.latency_ms,
        }


@dataclass
class TelemetrySummary:
    """Summary of telemetry data.

    Attributes:
        total_points: Total data points
        points_per_second: Collection rate
        components_healthy: Healthy component count
        components_unhealthy: Unhealthy component count
        oldest_data: Oldest data timestamp
        newest_data: Newest data timestamp
    """
    total_points: int = 0
    points_per_second: float = 0.0
    components_healthy: int = 0
    components_unhealthy: int = 0
    oldest_data: float = 0.0
    newest_data: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_points": self.total_points,
            "points_per_second": self.points_per_second,
            "components_healthy": self.components_healthy,
            "components_unhealthy": self.components_unhealthy,
            "oldest_data": self.oldest_data,
            "newest_data": self.newest_data,
        }


class MonitorTelemetry:
    """
    Meta-telemetry for the Monitor Agent.

    Monitors the monitoring system itself:
    - Component health tracking
    - Processing latency
    - Queue depths
    - Error rates
    - Resource usage

    Example:
        telemetry = MonitorTelemetry()

        # Record telemetry
        telemetry.record(TelemetryPoint(
            name="metric_collector.queue_depth",
            value=42,
            component="metric_collector",
            component_type=ComponentType.COLLECTOR,
        ))

        # Get component status
        status = telemetry.get_component_status("metric_collector")

        # Get summary
        summary = telemetry.get_summary()
    """

    BUS_TOPICS = {
        "telemetry": "telemetry.monitor",
        "meta_telemetry": "monitor.meta.telemetry",
        "meta_health": "monitor.meta.health",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    # Telemetry bucket
    TELEMETRY_BUCKET = "telemetry"

    def __init__(
        self,
        level: TelemetryLevel = TelemetryLevel.STANDARD,
        collection_interval: int = 60,
        retention_hours: int = 24,
        bus_dir: Optional[str] = None,
    ):
        """Initialize telemetry module.

        Args:
            level: Telemetry detail level
            collection_interval: Collection interval in seconds
            retention_hours: Hours to retain telemetry
            bus_dir: Bus directory
        """
        self._level = level
        self._collection_interval = collection_interval
        self._retention_hours = retention_hours
        self._last_heartbeat = time.time()
        self._start_time = time.time()

        # Telemetry storage
        self._points: List[TelemetryPoint] = []
        self._components: Dict[str, ComponentStatus] = {}
        self._lock = threading.RLock()

        # Collection state
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._total_collected = 0

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._telemetry_path = Path(self._bus_dir) / "telemetry.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default components
        self._register_default_components()

    def record(self, point: TelemetryPoint) -> None:
        """Record a telemetry point.

        Args:
            point: Telemetry point
        """
        with self._lock:
            self._points.append(point)
            self._total_collected += 1

            # Update component status
            if point.component in self._components:
                comp = self._components[point.component]
                comp.last_activity = time.time()
                comp.metrics_processed += 1

            # Cleanup old points
            self._cleanup_old_points()

        # Write to telemetry bucket
        self._emit_telemetry_event(point)

    def record_error(
        self,
        component: str,
        error: str,
        severity: str = "error",
    ) -> None:
        """Record a component error.

        Args:
            component: Component name
            error: Error message
            severity: Error severity
        """
        with self._lock:
            if component in self._components:
                self._components[component].errors += 1

        self._emit_bus_event(
            f"{self.BUS_TOPICS['telemetry']}.error",
            {
                "component": component,
                "error": error,
                "severity": severity,
            },
            level=severity,
        )

    def record_latency(
        self,
        component: str,
        latency_ms: float,
    ) -> None:
        """Record component latency.

        Args:
            component: Component name
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            if component in self._components:
                # Exponential moving average
                comp = self._components[component]
                alpha = 0.1
                comp.latency_ms = alpha * latency_ms + (1 - alpha) * comp.latency_ms

        self.record(TelemetryPoint(
            name=f"{component}.latency_ms",
            value=latency_ms,
            component=component,
            component_type=self._get_component_type(component),
        ))

    def update_component_health(
        self,
        component: str,
        healthy: bool,
        reason: str = "",
    ) -> None:
        """Update component health status.

        Args:
            component: Component name
            healthy: Health status
            reason: Status reason
        """
        with self._lock:
            if component not in self._components:
                self._components[component] = ComponentStatus(
                    component=component,
                    component_type=ComponentType.CORE,
                )

            self._components[component].healthy = healthy
            self._components[component].last_activity = time.time()

        if not healthy:
            self._emit_bus_event(
                self.BUS_TOPICS["meta_health"],
                {
                    "component": component,
                    "healthy": healthy,
                    "reason": reason,
                },
                level="warning",
            )

    def get_component_status(self, component: str) -> Optional[Dict[str, Any]]:
        """Get component status.

        Args:
            component: Component name

        Returns:
            Status or None
        """
        with self._lock:
            status = self._components.get(component)
            return status.to_dict() if status else None

    def list_components(self) -> List[Dict[str, Any]]:
        """List all component statuses.

        Returns:
            Component status list
        """
        with self._lock:
            return [c.to_dict() for c in self._components.values()]

    def get_summary(self) -> TelemetrySummary:
        """Get telemetry summary.

        Returns:
            Telemetry summary
        """
        with self._lock:
            now = time.time()
            uptime = now - self._start_time

            healthy = sum(1 for c in self._components.values() if c.healthy)
            unhealthy = len(self._components) - healthy

            oldest = min((p.timestamp for p in self._points), default=now)
            newest = max((p.timestamp for p in self._points), default=now)

            return TelemetrySummary(
                total_points=len(self._points),
                points_per_second=self._total_collected / uptime if uptime > 0 else 0,
                components_healthy=healthy,
                components_unhealthy=unhealthy,
                oldest_data=oldest,
                newest_data=newest,
            )

    def get_recent_telemetry(
        self,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent telemetry points.

        Args:
            component: Filter by component
            limit: Maximum results

        Returns:
            Telemetry points
        """
        with self._lock:
            points = self._points
            if component:
                points = [p for p in points if p.component == component]

            return [p.to_dict() for p in reversed(points[-limit:])]

    def get_metrics_by_name(
        self,
        name_pattern: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get metrics by name pattern.

        Args:
            name_pattern: Name pattern (prefix match)
            limit: Maximum results

        Returns:
            Matching metrics
        """
        with self._lock:
            points = [p for p in self._points if p.name.startswith(name_pattern)]
            return [p.to_dict() for p in reversed(points[-limit:])]

    async def start_collection(self) -> None:
        """Start automatic telemetry collection."""
        if self._running:
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())

    async def stop_collection(self) -> None:
        """Stop automatic telemetry collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None

    async def _collection_loop(self) -> None:
        """Background collection loop."""
        while self._running:
            try:
                await self._collect_system_telemetry()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self._collection_interval)

    async def _collect_system_telemetry(self) -> None:
        """Collect system-level telemetry."""
        try:
            import psutil

            process = psutil.Process()

            # Memory
            self.record(TelemetryPoint(
                name="monitor.memory_percent",
                value=process.memory_percent(),
                component="monitor_core",
                component_type=ComponentType.CORE,
            ))

            # CPU
            self.record(TelemetryPoint(
                name="monitor.cpu_percent",
                value=process.cpu_percent(),
                component="monitor_core",
                component_type=ComponentType.CORE,
            ))

            # File descriptors
            self.record(TelemetryPoint(
                name="monitor.open_files",
                value=len(process.open_files()),
                component="monitor_core",
                component_type=ComponentType.CORE,
            ))

            # Thread count
            self.record(TelemetryPoint(
                name="monitor.threads",
                value=process.num_threads(),
                component="monitor_core",
                component_type=ComponentType.CORE,
            ))

        except Exception:
            pass

        # Component telemetry
        with self._lock:
            for comp in self._components.values():
                self.record(TelemetryPoint(
                    name=f"component.{comp.component}.metrics_processed",
                    value=comp.metrics_processed,
                    component=comp.component,
                    component_type=comp.component_type,
                ))
                self.record(TelemetryPoint(
                    name=f"component.{comp.component}.errors",
                    value=comp.errors,
                    component=comp.component,
                    component_type=comp.component_type,
                ))
                self.record(TelemetryPoint(
                    name=f"component.{comp.component}.latency_ms",
                    value=comp.latency_ms,
                    component=comp.component,
                    component_type=comp.component_type,
                ))

    def _register_default_components(self) -> None:
        """Register default monitor components."""
        components = [
            ("metric_collector", ComponentType.COLLECTOR),
            ("metric_aggregator", ComponentType.AGGREGATOR),
            ("log_analyzer", ComponentType.ANALYZER),
            ("anomaly_detector", ComponentType.ANALYZER),
            ("alert_manager", ComponentType.ALERTER),
            ("report_generator", ComponentType.REPORTER),
            ("api", ComponentType.API),
            ("monitor_core", ComponentType.CORE),
        ]

        for name, comp_type in components:
            self._components[name] = ComponentStatus(
                component=name,
                component_type=comp_type,
            )

    def _get_component_type(self, component: str) -> ComponentType:
        """Get component type."""
        with self._lock:
            if component in self._components:
                return self._components[component].component_type
        return ComponentType.CORE

    def _cleanup_old_points(self) -> None:
        """Remove old telemetry points."""
        cutoff = time.time() - (self._retention_hours * 3600)
        self._points = [p for p in self._points if p.timestamp > cutoff]

    def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry statistics.

        Returns:
            Statistics
        """
        summary = self.get_summary()
        return {
            "level": self._level.value,
            "collection_interval": self._collection_interval,
            "retention_hours": self._retention_hours,
            "running": self._running,
            "total_collected": self._total_collected,
            "uptime_s": time.time() - self._start_time,
            "summary": summary.to_dict(),
        }

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        summary = self.get_summary()

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_telemetry",
                "status": "healthy" if summary.components_unhealthy == 0 else "degraded",
                "total_points": summary.total_points,
            },
        )

        return True

    def _emit_telemetry_event(self, point: TelemetryPoint) -> str:
        """Emit telemetry event to telemetry bucket."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": point.timestamp,
            "iso": datetime.fromtimestamp(point.timestamp, timezone.utc).isoformat() + "Z",
            "topic": f"{self.TELEMETRY_BUCKET}.{point.name}",
            "kind": "metric",
            "level": "info",
            "actor": "monitor-telemetry",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": point.to_dict(),
        }

        try:
            with open(self._telemetry_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-telemetry",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_telemetry: Optional[MonitorTelemetry] = None


def get_telemetry() -> MonitorTelemetry:
    """Get or create the telemetry module singleton.

    Returns:
        MonitorTelemetry instance
    """
    global _telemetry
    if _telemetry is None:
        _telemetry = MonitorTelemetry()
    return _telemetry


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Telemetry (Step 297)")
    parser.add_argument("--summary", action="store_true", help="Show telemetry summary")
    parser.add_argument("--components", action="store_true", help="List components")
    parser.add_argument("--recent", action="store_true", help="Show recent telemetry")
    parser.add_argument("--component", metavar="NAME", help="Filter by component")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    telemetry = get_telemetry()

    if args.summary:
        summary = telemetry.get_summary()
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print("Telemetry Summary:")
            print(f"  Total Points: {summary.total_points}")
            print(f"  Rate: {summary.points_per_second:.2f}/s")
            print(f"  Components Healthy: {summary.components_healthy}")
            print(f"  Components Unhealthy: {summary.components_unhealthy}")

    if args.components:
        components = telemetry.list_components()
        if args.json:
            print(json.dumps(components, indent=2))
        else:
            print("Components:")
            for c in components:
                health = "healthy" if c["healthy"] else "unhealthy"
                print(f"  {c['component']}: {health} ({c['metrics_processed']} metrics)")

    if args.recent:
        recent = telemetry.get_recent_telemetry(component=args.component, limit=20)
        if args.json:
            print(json.dumps(recent, indent=2))
        else:
            print("Recent Telemetry:")
            for p in recent:
                print(f"  {p['name']}: {p['value']} ({p['component']})")

    if args.stats:
        stats = telemetry.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Telemetry Statistics:")
            for k, v in stats.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")
