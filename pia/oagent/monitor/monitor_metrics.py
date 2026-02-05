#!/usr/bin/env python3
"""
Monitor Metrics - Step 283

Meta-monitoring metrics for the Monitor Agent itself.

PBTSO Phase: SKILL

Bus Topics:
- monitor.meta.metrics (emitted)
- monitor.meta.health (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import psutil
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class MetaMetricType(Enum):
    """Types of meta-monitoring metrics."""
    THROUGHPUT = "throughput"        # Events processed per second
    LATENCY = "latency"              # Processing latency
    QUEUE_SIZE = "queue_size"        # Pending items
    ERROR_RATE = "error_rate"        # Error percentage
    MEMORY_USAGE = "memory_usage"    # Memory consumption
    CPU_USAGE = "cpu_usage"          # CPU utilization
    UPTIME = "uptime"                # Time since start
    CARDINALITY = "cardinality"      # Unique metric series


@dataclass
class MetaMetricPoint:
    """A meta-monitoring metric point.

    Attributes:
        name: Metric name
        value: Metric value
        metric_type: Type of metric
        timestamp: Collection timestamp
        component: Component name
        labels: Additional labels
    """
    name: str
    value: float
    metric_type: MetaMetricType
    timestamp: float = field(default_factory=time.time)
    component: str = "monitor"
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "timestamp": self.timestamp,
            "component": self.component,
            "labels": self.labels,
        }


@dataclass
class ComponentHealth:
    """Health status of a component.

    Attributes:
        name: Component name
        healthy: Whether component is healthy
        last_check: Last health check timestamp
        error: Error message if unhealthy
        metrics: Component-specific metrics
    """
    name: str
    healthy: bool = True
    last_check: float = field(default_factory=time.time)
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "healthy": self.healthy,
            "last_check": self.last_check,
            "error": self.error,
            "metrics": self.metrics,
        }


class MetaMetricsCollector:
    """
    Collects metrics about the Monitor Agent's own performance.

    Tracks:
    - Throughput and latency
    - Resource usage (CPU, memory)
    - Error rates
    - Component health
    - Queue sizes

    Example:
        collector = MetaMetricsCollector()

        # Record processing time
        with collector.track_latency("process_event"):
            process_event()

        # Record throughput
        collector.record_throughput("events_processed", count=100)

        # Get current metrics
        metrics = collector.collect_all()
    """

    BUS_TOPICS = {
        "metrics": "monitor.meta.metrics",
        "health": "monitor.meta.health",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        collection_interval_s: int = 60,
        history_size: int = 1000,
        bus_dir: Optional[str] = None,
    ):
        """Initialize meta-metrics collector.

        Args:
            collection_interval_s: Collection interval in seconds
            history_size: Number of historical points to keep
            bus_dir: Bus directory
        """
        self._collection_interval = collection_interval_s
        self._history_size = history_size
        self._start_time = time.time()
        self._last_collection = 0.0
        self._last_heartbeat = time.time()

        # Metric storage
        self._metrics: Dict[str, List[MetaMetricPoint]] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._latencies: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

        # Component health
        self._component_health: Dict[str, ComponentHealth] = {}

        # Process reference
        self._process = psutil.Process()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def increment_counter(self, name: str, value: int = 1) -> int:
        """Increment a counter.

        Args:
            name: Counter name
            value: Increment value

        Returns:
            New counter value
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = 0
            self._counters[name] += value
            return self._counters[name]

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value.

        Args:
            name: Gauge name
            value: Gauge value
        """
        with self._lock:
            self._gauges[name] = value

    def record_latency(self, name: str, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            name: Operation name
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            if name not in self._latencies:
                self._latencies[name] = []
            self._latencies[name].append(latency_ms)
            # Keep only recent latencies
            if len(self._latencies[name]) > 1000:
                self._latencies[name] = self._latencies[name][-1000:]

    def track_latency(self, name: str) -> "LatencyTracker":
        """Create a latency tracker context manager.

        Args:
            name: Operation name

        Returns:
            Context manager for tracking latency
        """
        return LatencyTracker(self, name)

    def record_throughput(self, name: str, count: int = 1) -> None:
        """Record throughput.

        Args:
            name: Operation name
            count: Number of items processed
        """
        self.increment_counter(f"{name}_total", count)

    def record_error(self, component: str, error: str) -> None:
        """Record an error.

        Args:
            component: Component name
            error: Error message
        """
        self.increment_counter(f"{component}_errors")
        self._update_component_health(component, healthy=False, error=error)

    def record_success(self, component: str) -> None:
        """Record a success.

        Args:
            component: Component name
        """
        self.increment_counter(f"{component}_success")

    def register_component(self, name: str) -> None:
        """Register a component for health tracking.

        Args:
            name: Component name
        """
        with self._lock:
            if name not in self._component_health:
                self._component_health[name] = ComponentHealth(name=name)

    def _update_component_health(
        self,
        name: str,
        healthy: bool = True,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update component health status.

        Args:
            name: Component name
            healthy: Health status
            error: Error message
            metrics: Component metrics
        """
        with self._lock:
            if name not in self._component_health:
                self._component_health[name] = ComponentHealth(name=name)

            health = self._component_health[name]
            health.healthy = healthy
            health.last_check = time.time()
            health.error = error
            if metrics:
                health.metrics.update(metrics)

    def collect_resource_metrics(self) -> Dict[str, float]:
        """Collect resource usage metrics.

        Returns:
            Resource metrics
        """
        try:
            cpu_percent = self._process.cpu_percent()
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()

            return {
                "cpu_percent": cpu_percent,
                "memory_rss_bytes": memory_info.rss,
                "memory_vms_bytes": memory_info.vms,
                "memory_percent": memory_percent,
                "open_files": len(self._process.open_files()),
                "threads": self._process.num_threads(),
            }
        except Exception:
            return {}

    def collect_latency_stats(self, name: str) -> Dict[str, float]:
        """Collect latency statistics.

        Args:
            name: Operation name

        Returns:
            Latency statistics
        """
        with self._lock:
            latencies = self._latencies.get(name, [])
            if not latencies:
                return {"count": 0}

            sorted_latencies = sorted(latencies)
            count = len(sorted_latencies)

            return {
                "count": count,
                "min_ms": sorted_latencies[0],
                "max_ms": sorted_latencies[-1],
                "avg_ms": sum(sorted_latencies) / count,
                "p50_ms": sorted_latencies[count // 2],
                "p90_ms": sorted_latencies[int(count * 0.9)],
                "p99_ms": sorted_latencies[int(count * 0.99)] if count >= 100 else sorted_latencies[-1],
            }

    def collect_all(self) -> Dict[str, Any]:
        """Collect all meta-metrics.

        Returns:
            All metrics
        """
        now = time.time()
        uptime = now - self._start_time

        with self._lock:
            # Calculate rates
            rates = {}
            for name, count in self._counters.items():
                if uptime > 0:
                    rates[f"{name}_per_sec"] = count / uptime

            # Collect latency stats
            latency_stats = {}
            for name in self._latencies:
                latency_stats[name] = self.collect_latency_stats(name)

            # Resource metrics
            resources = self.collect_resource_metrics()

            # Component health
            components = {
                name: health.to_dict()
                for name, health in self._component_health.items()
            }

            return {
                "timestamp": now,
                "uptime_s": uptime,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "rates": rates,
                "latencies": latency_stats,
                "resources": resources,
                "components": components,
            }

    def emit_metrics(self) -> str:
        """Emit current metrics to bus.

        Returns:
            Event ID
        """
        metrics = self.collect_all()
        return self._emit_bus_event(
            self.BUS_TOPICS["metrics"],
            metrics,
        )

    def emit_health(self) -> str:
        """Emit health status to bus.

        Returns:
            Event ID
        """
        with self._lock:
            all_healthy = all(
                h.healthy for h in self._component_health.values()
            )
            components = {
                name: health.to_dict()
                for name, health in self._component_health.items()
            }

        return self._emit_bus_event(
            self.BUS_TOPICS["health"],
            {
                "healthy": all_healthy,
                "components": components,
                "uptime_s": time.time() - self._start_time,
            },
        )

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary.

        Returns:
            Health summary
        """
        with self._lock:
            healthy_count = sum(1 for h in self._component_health.values() if h.healthy)
            total_count = len(self._component_health)

            return {
                "healthy": healthy_count == total_count,
                "healthy_components": healthy_count,
                "total_components": total_count,
                "uptime_s": time.time() - self._start_time,
            }

    def reset_counters(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._counters.clear()

    def reset_latencies(self) -> None:
        """Reset all latency measurements."""
        with self._lock:
            self._latencies.clear()

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        health = self.get_health_summary()

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_meta_metrics",
                "status": "healthy" if health["healthy"] else "degraded",
                "uptime_s": health["uptime_s"],
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "metric",
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
            "actor": "monitor-meta-metrics",
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


class LatencyTracker:
    """Context manager for tracking operation latency."""

    def __init__(self, collector: MetaMetricsCollector, name: str):
        """Initialize latency tracker.

        Args:
            collector: Meta-metrics collector
            name: Operation name
        """
        self._collector = collector
        self._name = name
        self._start_time: Optional[float] = None

    def __enter__(self) -> "LatencyTracker":
        """Start tracking."""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop tracking and record latency."""
        if self._start_time is not None:
            latency_ms = (time.time() - self._start_time) * 1000
            self._collector.record_latency(self._name, latency_ms)


# Singleton instance
_meta_collector: Optional[MetaMetricsCollector] = None


def get_meta_collector() -> MetaMetricsCollector:
    """Get or create the meta-metrics collector singleton.

    Returns:
        MetaMetricsCollector instance
    """
    global _meta_collector
    if _meta_collector is None:
        _meta_collector = MetaMetricsCollector()
    return _meta_collector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Meta-Metrics (Step 283)")
    parser.add_argument("--collect", action="store_true", help="Collect all metrics")
    parser.add_argument("--health", action="store_true", help="Show health summary")
    parser.add_argument("--resources", action="store_true", help="Show resource metrics")
    parser.add_argument("--emit", action="store_true", help="Emit metrics to bus")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    collector = get_meta_collector()

    # Register some test components
    collector.register_component("metrics_collector")
    collector.register_component("alert_manager")
    collector.register_component("dashboard")

    if args.collect:
        metrics = collector.collect_all()
        if args.json:
            print(json.dumps(metrics, indent=2, default=str))
        else:
            print("Meta-Metrics:")
            print(f"  Uptime: {metrics['uptime_s']:.1f}s")
            print(f"  Counters: {len(metrics['counters'])}")
            print(f"  Gauges: {len(metrics['gauges'])}")
            print(f"  Components: {len(metrics['components'])}")
            print(f"  Resources:")
            for k, v in metrics['resources'].items():
                print(f"    {k}: {v}")

    if args.health:
        health = collector.get_health_summary()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            status = "healthy" if health["healthy"] else "unhealthy"
            print(f"Health: {status}")
            print(f"  Components: {health['healthy_components']}/{health['total_components']}")
            print(f"  Uptime: {health['uptime_s']:.1f}s")

    if args.resources:
        resources = collector.collect_resource_metrics()
        if args.json:
            print(json.dumps(resources, indent=2))
        else:
            print("Resource Metrics:")
            for k, v in resources.items():
                print(f"  {k}: {v}")

    if args.emit:
        event_id = collector.emit_metrics()
        if args.json:
            print(json.dumps({"event_id": event_id}))
        else:
            print(f"Emitted metrics: {event_id}")
