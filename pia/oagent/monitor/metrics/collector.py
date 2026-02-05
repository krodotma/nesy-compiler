#!/usr/bin/env python3
"""
Metric Collector - Step 252

Collects and stores metrics from all agents via telemetry subscriptions.

PBTSO Phase: ITERATE

Bus Topics:
- telemetry.* (subscribed)
- monitor.metrics.collected (emitted)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"       # Monotonically increasing value
    GAUGE = "gauge"          # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"      # Pre-computed percentiles
    TIMER = "timer"          # Duration measurements


@dataclass
class MetricPoint:
    """A single metric data point.

    Attributes:
        name: Metric name (e.g., "agent.requests.total")
        value: Numeric value
        timestamp: Unix timestamp
        labels: Dimension labels (e.g., {"agent": "code", "status": "success"})
        metric_type: Type of metric
        unit: Unit of measurement (optional)
    """
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "metric_type": self.metric_type.value,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricPoint":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            timestamp=data.get("timestamp", time.time()),
            labels=data.get("labels", {}),
            metric_type=MetricType(data.get("metric_type", "gauge")),
            unit=data.get("unit", ""),
        )


@dataclass
class MetricSeries:
    """Time series for a metric name and label set.

    Attributes:
        name: Metric name
        labels: Frozen label set
        points: List of data points
        max_points: Maximum points to retain
    """
    name: str
    labels: Dict[str, str]
    points: List[MetricPoint] = field(default_factory=list)
    max_points: int = 10000

    def add_point(self, point: MetricPoint) -> None:
        """Add a data point."""
        self.points.append(point)
        # Prune oldest points if exceeds max
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points:]

    def get_points_in_window(self, window_s: int) -> List[MetricPoint]:
        """Get points within time window."""
        cutoff = time.time() - window_s
        return [p for p in self.points if p.timestamp >= cutoff]

    def get_label_key(self) -> str:
        """Get unique key for label set."""
        return json.dumps(sorted(self.labels.items()))


class MetricCollector:
    """
    Collect and store metrics from all agents.

    The collector:
    - Receives metrics via telemetry.* bus subscriptions
    - Stores time series data with configurable retention
    - Provides query interface for aggregation
    - Emits collection events to bus

    Example:
        collector = MetricCollector()
        collector.record(MetricPoint(
            name="agent.requests.total",
            value=42,
            timestamp=time.time(),
            labels={"agent": "code"},
            metric_type=MetricType.COUNTER
        ))

        avg = collector.query("agent.requests.total", "avg", window_s=300)
    """

    BUS_TOPICS = {
        "collected": "monitor.metrics.collected",
        "error": "monitor.metrics.error",
    }

    # Standard aggregation functions
    AGGREGATIONS: Dict[str, Callable[[List[MetricPoint]], float]] = {
        "avg": lambda pts: sum(p.value for p in pts) / len(pts) if pts else 0.0,
        "sum": lambda pts: sum(p.value for p in pts),
        "max": lambda pts: max(p.value for p in pts) if pts else 0.0,
        "min": lambda pts: min(p.value for p in pts) if pts else 0.0,
        "count": lambda pts: float(len(pts)),
        "last": lambda pts: pts[-1].value if pts else 0.0,
        "first": lambda pts: pts[0].value if pts else 0.0,
        "rate": lambda pts: (pts[-1].value - pts[0].value) / max(1, pts[-1].timestamp - pts[0].timestamp) if len(pts) >= 2 else 0.0,
    }

    def __init__(
        self,
        retention_hours: int = 24,
        max_series: int = 10000,
        bus_dir: Optional[str] = None
    ):
        """Initialize metric collector.

        Args:
            retention_hours: Hours to retain metrics
            max_series: Maximum number of unique series
            bus_dir: Directory for bus events
        """
        self.retention_hours = retention_hours
        self.max_series = max_series

        # Metrics storage: name -> label_key -> MetricSeries
        self._metrics: Dict[str, Dict[str, MetricSeries]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._record_count: int = 0

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, metric: MetricPoint) -> bool:
        """Record a metric data point.

        Args:
            metric: Metric point to record

        Returns:
            True if recorded successfully
        """
        with self._lock:
            # Get or create series
            label_key = json.dumps(sorted(metric.labels.items()))
            if label_key not in self._metrics[metric.name]:
                # Check series limit
                total_series = sum(len(s) for s in self._metrics.values())
                if total_series >= self.max_series:
                    self._prune_old_series()

                self._metrics[metric.name][label_key] = MetricSeries(
                    name=metric.name,
                    labels=metric.labels,
                )

            series = self._metrics[metric.name][label_key]
            series.add_point(metric)
            self._record_count += 1

            # Prune old metrics periodically
            if self._record_count % 1000 == 0:
                self._prune_old_metrics(metric.name)

        return True

    def record_batch(self, metrics: List[MetricPoint]) -> int:
        """Record multiple metrics at once.

        Args:
            metrics: List of metric points

        Returns:
            Number of metrics recorded
        """
        recorded = 0
        for metric in metrics:
            if self.record(metric):
                recorded += 1
        return recorded

    def query(
        self,
        name: str,
        aggregation: str = "avg",
        window_s: int = 300,
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Query aggregated metric value.

        Args:
            name: Metric name
            aggregation: Aggregation function (avg, sum, max, min, count, last, first, rate)
            window_s: Time window in seconds
            labels: Optional label filter

        Returns:
            Aggregated value
        """
        if aggregation not in self.AGGREGATIONS:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        with self._lock:
            if name not in self._metrics:
                return 0.0

            # Collect all points matching labels
            all_points: List[MetricPoint] = []
            for label_key, series in self._metrics[name].items():
                # Filter by labels if provided
                if labels:
                    match = all(
                        series.labels.get(k) == v
                        for k, v in labels.items()
                    )
                    if not match:
                        continue

                all_points.extend(series.get_points_in_window(window_s))

            if not all_points:
                return 0.0

            # Sort by timestamp
            all_points.sort(key=lambda p: p.timestamp)

            return self.AGGREGATIONS[aggregation](all_points)

    def query_series(
        self,
        name: str,
        window_s: int = 300,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricPoint]:
        """Query raw metric points.

        Args:
            name: Metric name
            window_s: Time window in seconds
            labels: Optional label filter

        Returns:
            List of metric points
        """
        with self._lock:
            if name not in self._metrics:
                return []

            all_points: List[MetricPoint] = []
            for label_key, series in self._metrics[name].items():
                if labels:
                    match = all(
                        series.labels.get(k) == v
                        for k, v in labels.items()
                    )
                    if not match:
                        continue

                all_points.extend(series.get_points_in_window(window_s))

            all_points.sort(key=lambda p: p.timestamp)
            return all_points

    def list_metrics(self) -> List[str]:
        """List all metric names.

        Returns:
            List of metric names
        """
        with self._lock:
            return list(self._metrics.keys())

    def list_labels(self, name: str) -> List[Dict[str, str]]:
        """List all label sets for a metric.

        Args:
            name: Metric name

        Returns:
            List of label dictionaries
        """
        with self._lock:
            if name not in self._metrics:
                return []
            return [series.labels for series in self._metrics[name].values()]

    def get_cardinality(self) -> Dict[str, Any]:
        """Get metric cardinality statistics.

        Returns:
            Cardinality info
        """
        with self._lock:
            total_series = sum(len(s) for s in self._metrics.values())
            total_points = sum(
                len(series.points)
                for series_dict in self._metrics.values()
                for series in series_dict.values()
            )
            return {
                "metric_names": len(self._metrics),
                "total_series": total_series,
                "total_points": total_points,
                "records_received": self._record_count,
            }

    def handle_telemetry_event(self, event: Dict[str, Any]) -> bool:
        """Handle incoming telemetry bus event.

        Args:
            event: Bus event

        Returns:
            True if handled
        """
        data = event.get("data", {})
        topic = event.get("topic", "")

        # Extract metric from event
        metric_name = data.get("metric", data.get("name"))
        if not metric_name:
            # Use topic as metric name
            metric_name = topic.replace(".", "_")

        value = data.get("value")
        if value is None:
            # Try common field names
            for field in ["count", "duration_ms", "latency_ms", "total", "rate"]:
                if field in data:
                    value = data[field]
                    break

        if value is None:
            return False

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        # Build labels from event
        labels = {
            "actor": event.get("actor", "unknown"),
            "host": event.get("host", "unknown"),
        }
        # Add any string fields from data as labels
        for k, v in data.items():
            if isinstance(v, str) and k not in ("metric", "name", "value", "unit"):
                labels[k] = v

        metric = MetricPoint(
            name=metric_name,
            value=value,
            timestamp=event.get("ts", time.time()),
            labels=labels,
            unit=data.get("unit", ""),
        )

        return self.record(metric)

    def emit_collected_event(self, metrics_count: int) -> str:
        """Emit a metrics collected event to bus.

        Args:
            metrics_count: Number of metrics collected

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS["collected"],
            "kind": "metric",
            "level": "info",
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "metrics_count": metrics_count,
                "cardinality": self.get_cardinality(),
            },
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _prune_old_metrics(self, name: str) -> int:
        """Prune metrics older than retention period.

        Args:
            name: Metric name to prune

        Returns:
            Number of points pruned
        """
        cutoff = time.time() - (self.retention_hours * 3600)
        pruned = 0

        with self._lock:
            if name not in self._metrics:
                return 0

            for label_key, series in list(self._metrics[name].items()):
                old_len = len(series.points)
                series.points = [p for p in series.points if p.timestamp >= cutoff]
                pruned += old_len - len(series.points)

                # Remove empty series
                if not series.points:
                    del self._metrics[name][label_key]

            # Remove empty metric names
            if not self._metrics[name]:
                del self._metrics[name]

        return pruned

    def _prune_old_series(self) -> int:
        """Prune oldest series when at capacity.

        Returns:
            Number of series pruned
        """
        # Find series with oldest last point
        series_ages: List[tuple] = []

        with self._lock:
            for name, series_dict in self._metrics.items():
                for label_key, series in series_dict.items():
                    if series.points:
                        last_ts = series.points[-1].timestamp
                    else:
                        last_ts = 0
                    series_ages.append((last_ts, name, label_key))

            # Sort by timestamp (oldest first)
            series_ages.sort()

            # Prune oldest 10%
            to_prune = max(1, len(series_ages) // 10)
            for _, name, label_key in series_ages[:to_prune]:
                if name in self._metrics and label_key in self._metrics[name]:
                    del self._metrics[name][label_key]
                    if not self._metrics[name]:
                        del self._metrics[name]

            return to_prune


# Singleton instance
_collector: Optional[MetricCollector] = None


def get_collector() -> MetricCollector:
    """Get or create the metric collector singleton.

    Returns:
        MetricCollector instance
    """
    global _collector
    if _collector is None:
        _collector = MetricCollector()
    return _collector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metric Collector (Step 252)")
    parser.add_argument("--record", metavar="NAME=VALUE", help="Record a metric")
    parser.add_argument("--query", metavar="NAME", help="Query a metric")
    parser.add_argument("--agg", default="avg", help="Aggregation function")
    parser.add_argument("--window", type=int, default=300, help="Window in seconds")
    parser.add_argument("--list", action="store_true", help="List all metrics")
    parser.add_argument("--cardinality", action="store_true", help="Show cardinality")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    collector = get_collector()

    if args.record:
        name, value = args.record.split("=")
        metric = MetricPoint(
            name=name,
            value=float(value),
            timestamp=time.time(),
        )
        collector.record(metric)
        print(f"Recorded: {name}={value}")

    if args.query:
        result = collector.query(args.query, args.agg, args.window)
        if args.json:
            print(json.dumps({"name": args.query, "value": result, "aggregation": args.agg}))
        else:
            print(f"{args.query}: {result} ({args.agg})")

    if args.list:
        metrics = collector.list_metrics()
        if args.json:
            print(json.dumps(metrics))
        else:
            for m in metrics:
                print(f"  {m}")

    if args.cardinality:
        card = collector.get_cardinality()
        if args.json:
            print(json.dumps(card, indent=2))
        else:
            print(f"Cardinality:")
            print(f"  Metric names: {card['metric_names']}")
            print(f"  Total series: {card['total_series']}")
            print(f"  Total points: {card['total_points']}")
