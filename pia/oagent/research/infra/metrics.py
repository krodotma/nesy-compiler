#!/usr/bin/env python3
"""
metrics.py - Performance and Usage Metrics (Step 33)

Comprehensive metrics collection for Research Agent.
Supports counters, gauges, histograms, and timers.

PBTSO Phase: MONITOR

Bus Topics:
- a2a.research.metrics.collect
- a2a.research.metrics.export
- research.metrics.flush

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import statistics
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"      # Monotonically increasing value
    GAUGE = "gauge"          # Value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"          # Duration measurements


@dataclass
class MetricsConfig:
    """Configuration for metrics collector."""

    namespace: str = "research"
    enable_collection: bool = True
    enable_export: bool = True
    export_interval_seconds: int = 60
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    persist_path: Optional[str] = None
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.persist_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.persist_path = f"{pluribus_root}/.pluribus/research/metrics"
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class MetricLabel:
    """Label for metric identification."""

    name: str
    value: str

    def __hash__(self):
        return hash((self.name, self.value))

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value


@dataclass
class Metric:
    """A single metric data point."""

    name: str
    type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "description": self.description,
        }


@dataclass
class HistogramData:
    """Histogram bucket data."""

    buckets: Dict[float, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0
    min: float = float('inf')
    max: float = float('-inf')

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.sum += value
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)

        # Increment appropriate bucket
        for bound in sorted(self.buckets.keys()):
            if value <= bound:
                self.buckets[bound] += 1

    @property
    def mean(self) -> float:
        """Calculate mean value."""
        return self.sum / self.count if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "buckets": {str(k): v for k, v in self.buckets.items()},
            "sum": self.sum,
            "count": self.count,
            "mean": self.mean,
            "min": self.min if self.min != float('inf') else None,
            "max": self.max if self.max != float('-inf') else None,
        }


# ============================================================================
# Metric Types
# ============================================================================


class Counter:
    """
    A counter metric that only increases.

    Example:
        counter = Counter("requests_total", "Total requests processed")
        counter.inc()
        counter.inc(5)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        """Increment counter."""
        if amount < 0:
            raise ValueError("Counter can only increase")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        """Get current value."""
        return self._value

    def to_metric(self) -> Metric:
        """Convert to Metric."""
        return Metric(
            name=self.name,
            type=MetricType.COUNTER,
            value=self._value,
            labels=self.labels,
            description=self.description,
        )


class Gauge:
    """
    A gauge metric that can go up or down.

    Example:
        gauge = Gauge("memory_usage_bytes", "Current memory usage")
        gauge.set(1024000)
        gauge.inc(500)
        gauge.dec(100)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value: float = 0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value

    def to_metric(self) -> Metric:
        """Convert to Metric."""
        return Metric(
            name=self.name,
            type=MetricType.GAUGE,
            value=self._value,
            labels=self.labels,
            description=self.description,
        )


class Histogram:
    """
    A histogram metric for value distributions.

    Example:
        histogram = Histogram("request_duration_seconds", buckets=[0.1, 0.5, 1.0])
        histogram.observe(0.25)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}

        # Initialize buckets
        self._data = HistogramData()
        buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        buckets = sorted(buckets) + [float('inf')]
        for bound in buckets:
            self._data.buckets[bound] = 0

        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._data.observe(value)

    @property
    def count(self) -> int:
        """Get observation count."""
        return self._data.count

    @property
    def sum(self) -> float:
        """Get sum of observations."""
        return self._data.sum

    @property
    def mean(self) -> float:
        """Get mean value."""
        return self._data.mean

    def to_metric(self) -> Metric:
        """Convert to Metric."""
        return Metric(
            name=self.name,
            type=MetricType.HISTOGRAM,
            value=self._data.count,
            labels={**self.labels, "_histogram_data": json.dumps(self._data.to_dict())},
            description=self.description,
        )


class Timer:
    """
    A timer metric for measuring durations.

    Example:
        timer = Timer("query_duration_seconds")

        with timer.time():
            do_something()

        # Or manual
        timer.start()
        do_something()
        timer.stop()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or {}

        self._histogram = Histogram(
            name=f"{name}_seconds",
            description=description,
            labels=labels,
            buckets=buckets,
        )
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the timer."""
        with self._lock:
            self._start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and record duration."""
        with self._lock:
            if self._start_time is None:
                raise RuntimeError("Timer was not started")
            duration = time.time() - self._start_time
            self._histogram.observe(duration)
            self._start_time = None
            return duration

    @contextmanager
    def time(self) -> Generator[None, None, None]:
        """Context manager for timing."""
        self.start()
        try:
            yield
        finally:
            self.stop()

    def record(self, duration: float) -> None:
        """Record a duration directly."""
        self._histogram.observe(duration)

    @property
    def count(self) -> int:
        """Get timing count."""
        return self._histogram.count

    @property
    def mean(self) -> float:
        """Get mean duration."""
        return self._histogram.mean

    def to_metric(self) -> Metric:
        """Convert to Metric."""
        return Metric(
            name=self.name,
            type=MetricType.TIMER,
            value=self._histogram.count,
            labels={**self.labels, "_timer_data": json.dumps(self._histogram._data.to_dict())},
            description=self.description,
        )


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """
    Central metrics collection and management.

    Features:
    - Metric registration and lookup
    - Automatic export to bus
    - Aggregation and reporting
    - Persistence

    PBTSO Phase: MONITOR

    Example:
        collector = MetricsCollector()

        # Create metrics
        requests = collector.counter("requests_total", "Total requests")
        latency = collector.histogram("request_latency", "Request latency")

        # Use metrics
        requests.inc()
        latency.observe(0.5)

        # Export
        collector.export()
    """

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the metrics collector.

        Args:
            config: Metrics configuration
            bus: AgentBus for event emission
        """
        self.config = config or MetricsConfig()
        self.bus = bus or AgentBus()

        # Metric storage
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, Timer] = {}
        self._lock = threading.Lock()

        # Export thread
        self._export_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Register default metrics
        self._register_default_metrics()

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """
        Get or create a counter metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels

        Returns:
            Counter instance
        """
        key = self._make_key(name, labels)

        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(
                    name=f"{self.config.namespace}_{name}",
                    description=description,
                    labels=labels,
                )
            return self._counters[key]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """
        Get or create a gauge metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels

        Returns:
            Gauge instance
        """
        key = self._make_key(name, labels)

        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge(
                    name=f"{self.config.namespace}_{name}",
                    description=description,
                    labels=labels,
                )
            return self._gauges[key]

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """
        Get or create a histogram metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
            buckets: Histogram buckets

        Returns:
            Histogram instance
        """
        key = self._make_key(name, labels)

        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(
                    name=f"{self.config.namespace}_{name}",
                    description=description,
                    labels=labels,
                    buckets=buckets or self.config.histogram_buckets,
                )
            return self._histograms[key]

    def timer(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Timer:
        """
        Get or create a timer metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
            buckets: Timer buckets

        Returns:
            Timer instance
        """
        key = self._make_key(name, labels)

        with self._lock:
            if key not in self._timers:
                self._timers[key] = Timer(
                    name=f"{self.config.namespace}_{name}",
                    description=description,
                    labels=labels,
                    buckets=buckets,
                )
            return self._timers[key]

    def collect(self) -> List[Metric]:
        """
        Collect all current metrics.

        Returns:
            List of all metrics
        """
        metrics = []

        with self._lock:
            for counter in self._counters.values():
                metrics.append(counter.to_metric())
            for gauge in self._gauges.values():
                metrics.append(gauge.to_metric())
            for histogram in self._histograms.values():
                metrics.append(histogram.to_metric())
            for timer in self._timers.values():
                metrics.append(timer.to_metric())

        return metrics

    def export(self) -> None:
        """Export metrics to bus and optionally persist."""
        if not self.config.enable_export:
            return

        metrics = self.collect()

        # Emit to bus
        self._emit_event(
            "a2a.research.metrics.export",
            {
                "metrics_count": len(metrics),
                "metrics": [m.to_dict() for m in metrics],
            }
        )

        # Persist if enabled
        if self.config.persist_path:
            self._persist_metrics(metrics)

    def start_export_thread(self) -> None:
        """Start automatic export thread."""
        if self._export_thread is not None:
            return

        self._stop_event.clear()
        self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self._export_thread.start()

    def stop_export_thread(self) -> None:
        """Stop automatic export thread."""
        self._stop_event.set()
        if self._export_thread:
            self._export_thread.join(timeout=5)
            self._export_thread = None

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "namespace": self.config.namespace,
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "histograms": len(self._histograms),
            "timers": len(self._timers),
            "total_metrics": len(self._counters) + len(self._gauges) + len(self._histograms) + len(self._timers),
            "export_enabled": self.config.enable_export,
            "export_interval": self.config.export_interval_seconds,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._register_default_metrics()

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric identification."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}:{label_str}"
        return name

    def _register_default_metrics(self) -> None:
        """Register default agent metrics."""
        self.gauge("agent_info", "Agent information", {"version": "0.2.0"}).set(1)

    def _export_loop(self) -> None:
        """Background export loop."""
        while not self._stop_event.wait(self.config.export_interval_seconds):
            self.export()

    def _persist_metrics(self, metrics: List[Metric]) -> None:
        """Persist metrics to file."""
        persist_path = Path(self.config.persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = persist_path / f"metrics_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "metrics",
            "level": "info",
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# Decorators
# ============================================================================


def timed(timer: Timer) -> Callable:
    """
    Decorator to time function execution.

    Example:
        timer = Timer("function_duration")

        @timed(timer)
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with timer.time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def counted(counter: Counter) -> Callable:
    """
    Decorator to count function calls.

    Example:
        counter = Counter("function_calls")

        @counted(counter)
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            counter.inc()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Global Collector Instance
# ============================================================================


_default_collector: Optional[MetricsCollector] = None


def get_collector(config: Optional[MetricsConfig] = None) -> MetricsCollector:
    """Get the default metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector(config)
    return _default_collector


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Metrics."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Metrics (Step 33)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show metrics statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export current metrics")
    export_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run metrics demo")

    args = parser.parse_args()

    collector = MetricsCollector()

    if args.command == "stats":
        stats = collector.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Metrics Collector Statistics:")
            print(f"  Namespace: {stats['namespace']}")
            print(f"  Total Metrics: {stats['total_metrics']}")
            print(f"    Counters: {stats['counters']}")
            print(f"    Gauges: {stats['gauges']}")
            print(f"    Histograms: {stats['histograms']}")
            print(f"    Timers: {stats['timers']}")

    elif args.command == "export":
        metrics = collector.collect()
        if args.json:
            print(json.dumps([m.to_dict() for m in metrics], indent=2))
        else:
            print(f"Exported {len(metrics)} metrics")
            for m in metrics:
                print(f"  {m.type.value}: {m.name} = {m.value}")

    elif args.command == "demo":
        print("Running metrics demo...")

        # Create some metrics
        requests = collector.counter("demo_requests", "Demo request counter")
        memory = collector.gauge("demo_memory", "Demo memory gauge")
        latency = collector.histogram("demo_latency", "Demo latency histogram")
        query_timer = collector.timer("demo_query", "Demo query timer")

        # Simulate some activity
        for i in range(10):
            requests.inc()
            memory.set(1000 + i * 100)
            latency.observe(0.1 + i * 0.05)

            with query_timer.time():
                time.sleep(0.01)

        # Export
        collector.export()
        print("Demo complete. Check metrics export.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
