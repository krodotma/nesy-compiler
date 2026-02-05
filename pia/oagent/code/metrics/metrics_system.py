#!/usr/bin/env python3
"""
metrics_system.py - Performance and Usage Metrics (Step 83)

PBTSO Phase: VERIFY, TEST

Provides:
- Metric collection (counters, gauges, histograms)
- Performance timing
- Resource usage tracking
- Metric aggregation and export
- Prometheus-compatible format

Bus Topics:
- code.metrics.record
- code.metrics.export
- code.metrics.alert

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import statistics
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Generator, List, Optional, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class MetricType(Enum):
    """Type of metric."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricConfig:
    """Configuration for metrics system."""
    export_interval_s: int = 60
    retention_hours: int = 24
    enable_prometheus: bool = True
    enable_bus_export: bool = True
    percentiles: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10
    ])
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_interval_s": self.export_interval_s,
            "retention_hours": self.retention_hours,
            "enable_prometheus": self.enable_prometheus,
            "percentiles": self.percentiles,
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
# Metric Classes
# =============================================================================

@dataclass
class MetricLabels:
    """Labels for a metric."""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_label_string(self) -> str:
        """Convert to Prometheus label string."""
        if not self.labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(pairs) + "}"


class Metric(ABC):
    """Abstract base class for metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = MetricLabels(labels or {})
        self.created_at = time.time()
        self._lock = Lock()

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Get metric type."""
        pass

    @abstractmethod
    def value(self) -> Union[float, Dict[str, float]]:
        """Get current value."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric."""
        pass

    def to_prometheus(self) -> str:
        """Export as Prometheus format."""
        val = self.value()
        labels = self.labels.to_label_string()

        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} {self.metric_type.value}",
        ]

        if isinstance(val, dict):
            for suffix, v in val.items():
                lines.append(f"{self.name}_{suffix}{labels} {v}")
        else:
            lines.append(f"{self.name}{labels} {val}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "labels": self.labels.labels,
            "value": self.value(),
            "created_at": self.created_at,
        }


class Counter(Metric):
    """
    Monotonically increasing counter.

    Use for counting events, requests, errors, etc.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, description, labels)
        self._value = 0.0

    @property
    def metric_type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, amount: float = 1.0) -> None:
        """Increment counter."""
        with self._lock:
            self._value += amount

    def value(self) -> float:
        """Get current count."""
        return self._value

    def reset(self) -> None:
        """Reset counter to 0."""
        with self._lock:
            self._value = 0.0


class Gauge(Metric):
    """
    Value that can go up and down.

    Use for current values, resource usage, etc.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, description, labels)
        self._value = 0.0

    @property
    def metric_type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= amount

    def value(self) -> float:
        """Get current value."""
        return self._value

    def reset(self) -> None:
        """Reset gauge to 0."""
        with self._lock:
            self._value = 0.0


class Histogram(Metric):
    """
    Distribution of values in buckets.

    Use for latency, size distributions, etc.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        super().__init__(name, description, labels)
        self.buckets = sorted(buckets or [0.001, 0.01, 0.1, 1, 10, float("inf")])
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._sum = 0.0
        self._count = 0

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._sum += value
            self._count += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1

    def value(self) -> Dict[str, float]:
        """Get histogram values."""
        result = {
            "sum": self._sum,
            "count": float(self._count),
        }
        for bucket, count in self._bucket_counts.items():
            bucket_str = str(bucket) if bucket != float("inf") else "+Inf"
            result[f"bucket_le_{bucket_str}"] = float(count)
        return result

    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            for bucket in self.buckets:
                self._bucket_counts[bucket] = 0
            self._sum = 0.0
            self._count = 0

    def to_prometheus(self) -> str:
        """Export as Prometheus format."""
        labels = self.labels.to_label_string()
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]

        for bucket in self.buckets:
            bucket_str = str(bucket) if bucket != float("inf") else "+Inf"
            count = self._bucket_counts[bucket]
            if labels:
                bucket_labels = labels[:-1] + f',le="{bucket_str}"' + "}"
            else:
                bucket_labels = f'{{le="{bucket_str}"}}'
            lines.append(f"{self.name}_bucket{bucket_labels} {count}")

        lines.append(f"{self.name}_sum{labels} {self._sum}")
        lines.append(f"{self.name}_count{labels} {self._count}")

        return "\n".join(lines)


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer(histogram):
            do_work()
    """

    def __init__(self, metric: Union[Histogram, Gauge]):
        self.metric = metric
        self.start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if isinstance(self.metric, Histogram):
                self.metric.observe(elapsed)
            elif isinstance(self.metric, Gauge):
                self.metric.set(elapsed)


# =============================================================================
# Metric Registry
# =============================================================================

class MetricRegistry:
    """
    Central registry for all metrics.

    Manages metric creation, storage, and export.
    """

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = Lock()

    def register(self, metric: Metric) -> Metric:
        """Register a metric."""
        with self._lock:
            self._metrics[metric.name] = metric
        return metric

    def unregister(self, name: str) -> bool:
        """Unregister a metric."""
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                return True
        return False

    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def list(self) -> List[str]:
        """List all metric names."""
        return list(self._metrics.keys())

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """Create or get a counter."""
        if name in self._metrics:
            metric = self._metrics[name]
            if isinstance(metric, Counter):
                return metric
        counter = Counter(name, description, labels)
        self.register(counter)
        return counter

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """Create or get a gauge."""
        if name in self._metrics:
            metric = self._metrics[name]
            if isinstance(metric, Gauge):
                return metric
        gauge = Gauge(name, description, labels)
        self.register(gauge)
        return gauge

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Create or get a histogram."""
        if name in self._metrics:
            metric = self._metrics[name]
            if isinstance(metric, Histogram):
                return metric
        hist = Histogram(name, description, labels, buckets)
        self.register(hist)
        return hist

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        for metric in self._metrics.values():
            lines.append(metric.to_prometheus())
            lines.append("")
        return "\n".join(lines)

    def export_dict(self) -> Dict[str, Any]:
        """Export all metrics as dictionary."""
        return {name: m.to_dict() for name, m in self._metrics.items()}


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Main metrics collection and export system.

    PBTSO Phase: VERIFY, TEST

    Features:
    - Metric registration and management
    - Automatic collection
    - Prometheus export
    - Bus event export

    Usage:
        collector = MetricsCollector(config)
        counter = collector.counter("requests_total")
        counter.inc()
    """

    BUS_TOPICS = {
        "record": "code.metrics.record",
        "export": "code.metrics.export",
        "alert": "code.metrics.alert",
    }

    # Standard metrics
    STANDARD_METRICS = {
        "code_operations_total": ("counter", "Total code operations"),
        "code_operation_duration_seconds": ("histogram", "Code operation duration"),
        "code_errors_total": ("counter", "Total code errors"),
        "code_lines_processed": ("counter", "Lines of code processed"),
        "code_files_processed": ("counter", "Files processed"),
        "code_cache_hits": ("counter", "Cache hits"),
        "code_cache_misses": ("counter", "Cache misses"),
        "active_operations": ("gauge", "Currently active operations"),
        "memory_usage_bytes": ("gauge", "Memory usage in bytes"),
    }

    def __init__(
        self,
        config: Optional[MetricConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or MetricConfig()
        self.bus = bus or LockedAgentBus()
        self.registry = MetricRegistry()

        # Initialize standard metrics
        self._init_standard_metrics()

        self._export_task: Optional[asyncio.Task] = None

    def _init_standard_metrics(self) -> None:
        """Initialize standard metrics."""
        for name, (metric_type, description) in self.STANDARD_METRICS.items():
            if metric_type == "counter":
                self.registry.counter(name, description)
            elif metric_type == "gauge":
                self.registry.gauge(name, description)
            elif metric_type == "histogram":
                self.registry.histogram(
                    name,
                    description,
                    buckets=self.config.histogram_buckets,
                )

    async def start(self) -> None:
        """Start periodic export."""
        if self.config.enable_bus_export:
            self._export_task = asyncio.create_task(self._export_loop())

    async def stop(self) -> None:
        """Stop periodic export."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

    async def _export_loop(self) -> None:
        """Periodic metric export."""
        while True:
            try:
                await asyncio.sleep(self.config.export_interval_s)
                self._export_to_bus()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def _export_to_bus(self) -> None:
        """Export metrics to bus."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["export"],
            "kind": "metrics",
            "actor": "metrics-collector",
            "data": self.registry.export_dict(),
        })

    # =========================================================================
    # Metric Accessors
    # =========================================================================

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """Get or create a counter."""
        return self.registry.counter(name, description, labels)

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """Get or create a gauge."""
        return self.registry.gauge(name, description, labels)

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        return self.registry.histogram(name, description, labels, buckets)

    @contextmanager
    def timer(self, name: str) -> Generator[Timer, None, None]:
        """Context manager for timing operations."""
        hist = self.histogram(f"{name}_duration_seconds")
        with Timer(hist) as t:
            yield t

    # =========================================================================
    # Recording
    # =========================================================================

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an operation."""
        # Update counters
        self.counter("code_operations_total", labels={"operation": operation}).inc()

        if not success:
            self.counter("code_errors_total", labels={"operation": operation}).inc()

        # Update histogram
        self.histogram("code_operation_duration_seconds").observe(duration)

        # Emit event
        self.bus.emit({
            "topic": self.BUS_TOPICS["record"],
            "kind": "metric",
            "actor": "metrics-collector",
            "data": {
                "operation": operation,
                "duration": duration,
                "success": success,
                "labels": labels or {},
            },
        })

    def record_file_processed(self, file_path: str, lines: int) -> None:
        """Record a file being processed."""
        self.counter("code_files_processed").inc()
        self.counter("code_lines_processed").inc(lines)

    # =========================================================================
    # Export
    # =========================================================================

    def prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return self.registry.export_prometheus()

    def json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(self.registry.export_dict(), indent=2)

    def stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "metrics_count": len(self.registry._metrics),
            "config": self.config.to_dict(),
        }


# =============================================================================
# Decorators
# =============================================================================

def timed(collector: MetricsCollector, name: str) -> Callable:
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                collector.record_operation(name, time.time() - start, True)
                return result
            except Exception as e:
                collector.record_operation(name, time.time() - start, False)
                raise

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                collector.record_operation(name, time.time() - start, True)
                return result
            except Exception as e:
                collector.record_operation(name, time.time() - start, False)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def counted(collector: MetricsCollector, name: str) -> Callable:
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            collector.counter(f"{name}_total").inc()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Metrics System."""
    import argparse

    parser = argparse.ArgumentParser(description="Metrics System (Step 83)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # export command
    export_parser = subparsers.add_parser("export", help="Export metrics")
    export_parser.add_argument("--format", "-f", choices=["prometheus", "json"], default="prometheus")

    # list command
    subparsers.add_parser("list", help="List metrics")

    # stats command
    subparsers.add_parser("stats", help="Show collector statistics")

    # demo command
    subparsers.add_parser("demo", help="Run demo with sample metrics")

    args = parser.parse_args()
    collector = MetricsCollector()

    if args.command == "export":
        if args.format == "prometheus":
            print(collector.prometheus())
        else:
            print(collector.json())
        return 0

    elif args.command == "list":
        metrics = collector.registry.list()
        print(f"Registered metrics ({len(metrics)}):")
        for name in metrics:
            metric = collector.registry.get(name)
            if metric:
                print(f"  {name} ({metric.metric_type.value})")
        return 0

    elif args.command == "stats":
        print(json.dumps(collector.stats(), indent=2))
        return 0

    elif args.command == "demo":
        print("Running metrics demo...")

        # Simulate some operations
        counter = collector.counter("demo_requests_total", "Demo requests")
        gauge = collector.gauge("demo_active_connections", "Active connections")
        hist = collector.histogram("demo_request_duration", "Request duration")

        for i in range(10):
            counter.inc()
            gauge.set(i % 5)
            hist.observe(0.1 * (i + 1))

        print("\nPrometheus format:")
        print(collector.prometheus())
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
