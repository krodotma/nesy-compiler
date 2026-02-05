#!/usr/bin/env python3
"""
Review Metrics Collector (Step 183)

Performance and usage metrics collection for the Review Agent.

PBTSO Phase: OBSERVE, DISTILL
Bus Topics: review.metrics.record, review.metrics.report

Metric Types:
- Counter: Monotonically increasing values
- Gauge: Point-in-time values
- Histogram: Distribution of values
- Summary: Statistical aggregations

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricUnit(Enum):
    """Units for metrics."""
    COUNT = "count"
    BYTES = "bytes"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    PERCENT = "percent"
    RATIO = "ratio"
    NONE = "none"


class MetricAggregation(Enum):
    """Aggregation methods."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"


class MetricName(Enum):
    """Standard metric names."""
    # Review metrics
    REVIEWS_TOTAL = "review_reviews_total"
    REVIEWS_DURATION_MS = "review_reviews_duration_ms"
    REVIEWS_FILES_COUNT = "review_reviews_files_count"
    REVIEWS_ISSUES_COUNT = "review_reviews_issues_count"

    # Analysis metrics
    ANALYSIS_DURATION_MS = "review_analysis_duration_ms"
    ANALYSIS_FILES_PROCESSED = "review_analysis_files_processed"
    ANALYSIS_ERRORS = "review_analysis_errors"

    # Cache metrics
    CACHE_HITS = "review_cache_hits"
    CACHE_MISSES = "review_cache_misses"
    CACHE_SIZE = "review_cache_size"

    # Plugin metrics
    PLUGIN_EXECUTIONS = "review_plugin_executions"
    PLUGIN_DURATION_MS = "review_plugin_duration_ms"
    PLUGIN_ERRORS = "review_plugin_errors"

    # System metrics
    MEMORY_USED_MB = "review_memory_used_mb"
    CPU_PERCENT = "review_cpu_percent"
    ACTIVE_REQUESTS = "review_active_requests"


@dataclass
class MetricLabels:
    """Labels for metrics (dimensions)."""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_key(self) -> str:
        """Convert to a sortable key string."""
        return ",".join(f"{k}={v}" for k, v in sorted(self.labels.items()))

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return dict(self.labels)


@dataclass
class Metric:
    """
    A metric data point.

    Attributes:
        name: Metric name
        type: Metric type
        value: Metric value
        unit: Metric unit
        labels: Metric labels/dimensions
        timestamp: Collection timestamp
        description: Metric description
    """
    name: str
    type: MetricType
    value: float
    unit: MetricUnit = MetricUnit.NONE
    labels: MetricLabels = field(default_factory=MetricLabels)
    timestamp: float = 0
    description: str = ""

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "unit": self.unit.value,
            "labels": self.labels.to_dict(),
            "timestamp": self.timestamp,
            "description": self.description,
        }

    def to_prometheus(self) -> str:
        """Convert to Prometheus format."""
        label_str = ""
        if self.labels.labels:
            label_parts = [f'{k}="{v}"' for k, v in self.labels.labels.items()]
            label_str = "{" + ",".join(label_parts) + "}"
        return f"{self.name}{label_str} {self.value}"


@dataclass
class MetricConfig:
    """
    Configuration for metrics collection.

    Attributes:
        enabled: Enable metrics collection
        retention_hours: How long to keep metrics
        export_interval_seconds: Export interval
        max_labels_per_metric: Maximum label combinations
        histogram_buckets: Default histogram buckets
        enable_system_metrics: Collect system metrics
    """
    enabled: bool = True
    retention_hours: int = 24
    export_interval_seconds: int = 60
    max_labels_per_metric: int = 100
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10
    ])
    enable_system_metrics: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MetricSeries:
    """A time series of metric values."""
    name: str
    type: MetricType
    unit: MetricUnit
    labels: MetricLabels
    values: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, value)
    description: str = ""

    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add a value to the series."""
        ts = timestamp or time.time()
        self.values.append((ts, value))

    def get_latest(self) -> Optional[float]:
        """Get the latest value."""
        return self.values[-1][1] if self.values else None

    def get_values_since(self, since: float) -> List[Tuple[float, float]]:
        """Get values since timestamp."""
        return [(ts, v) for ts, v in self.values if ts >= since]

    def trim(self, before: float) -> int:
        """Remove values before timestamp."""
        original_len = len(self.values)
        self.values = [(ts, v) for ts, v in self.values if ts >= before]
        return original_len - len(self.values)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "unit": self.unit.value,
            "labels": self.labels.to_dict(),
            "value_count": len(self.values),
            "latest": self.get_latest(),
        }


@dataclass
class MetricReport:
    """A report of metrics."""
    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_prometheus(self) -> str:
        """Convert to Prometheus exposition format."""
        lines = [f"# Report generated at {self.generated_at}"]
        for m in self.metrics:
            labels = m.get("labels", {})
            label_str = ""
            if labels:
                label_parts = [f'{k}="{v}"' for k, v in labels.items()]
                label_str = "{" + ",".join(label_parts) + "}"
            lines.append(f"{m['name']}{label_str} {m['value']}")
        return "\n".join(lines)


# ============================================================================
# Histogram
# ============================================================================

class Histogram:
    """Histogram for tracking value distributions."""

    def __init__(self, buckets: List[float]):
        self.buckets = sorted(buckets)
        self.bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self.bucket_counts[float("inf")] = 0
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self._sum += value
        self._count += 1
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
                break
        else:
            self.bucket_counts[float("inf")] += 1

    @property
    def sum(self) -> float:
        """Get sum of all observations."""
        return self._sum

    @property
    def count(self) -> int:
        """Get count of observations."""
        return self._count

    def get_percentile(self, percentile: float) -> float:
        """Estimate percentile from histogram."""
        if self._count == 0:
            return 0.0

        target = self._count * (percentile / 100)
        cumulative = 0
        prev_bucket = 0.0

        for bucket in self.buckets + [float("inf")]:
            cumulative += self.bucket_counts[bucket]
            if cumulative >= target:
                return bucket if bucket != float("inf") else prev_bucket * 2
            prev_bucket = bucket

        return self.buckets[-1] if self.buckets else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "buckets": {str(k): v for k, v in self.bucket_counts.items()},
            "sum": self._sum,
            "count": self._count,
        }


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """
    Collects and manages metrics.

    Example:
        collector = MetricsCollector()

        # Record counter
        collector.inc("review_reviews_total", labels={"status": "success"})

        # Record gauge
        collector.set("review_active_requests", 5)

        # Record histogram
        collector.observe("review_duration_ms", 150.5)

        # Generate report
        report = await collector.generate_report()
    """

    BUS_TOPICS = {
        "record": "review.metrics.record",
        "report": "review.metrics.report",
    }

    def __init__(
        self,
        config: Optional[MetricConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the metrics collector.

        Args:
            config: Metrics configuration
            bus_path: Path to event bus file
        """
        self.config = config or MetricConfig()
        self.bus_path = bus_path or self._get_bus_path()

        # Metric storage
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, Histogram]] = defaultdict(dict)
        self._series: Dict[str, Dict[str, MetricSeries]] = defaultdict(dict)

        # Metadata
        self._descriptions: Dict[str, str] = {}
        self._units: Dict[str, MetricUnit] = {}
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "metrics") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "metrics-collector",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def register(
        self,
        name: str,
        type: MetricType,
        unit: MetricUnit = MetricUnit.NONE,
        description: str = "",
    ) -> None:
        """
        Register a metric.

        Args:
            name: Metric name
            type: Metric type
            unit: Metric unit
            description: Metric description
        """
        self._descriptions[name] = description
        self._units[name] = unit

    def inc(
        self,
        name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter.

        Args:
            name: Metric name
            value: Increment value
            labels: Metric labels
        """
        label_key = MetricLabels(labels or {}).to_key()
        self._counters[name][label_key] += value

    def dec(
        self,
        name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Decrement a gauge.

        Args:
            name: Metric name
            value: Decrement value
            labels: Metric labels
        """
        label_key = MetricLabels(labels or {}).to_key()
        self._gauges[name][label_key] -= value

    def set(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
        """
        label_key = MetricLabels(labels or {}).to_key()
        self._gauges[name][label_key] = value

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a histogram observation.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        label_key = MetricLabels(labels or {}).to_key()
        if label_key not in self._histograms[name]:
            self._histograms[name][label_key] = Histogram(self.config.histogram_buckets)
        self._histograms[name][label_key].observe(value)

    def record(self, metric: Metric) -> None:
        """
        Record a metric.

        Args:
            metric: Metric to record
        """
        if metric.type == MetricType.COUNTER:
            self.inc(metric.name, metric.value, metric.labels.labels)
        elif metric.type == MetricType.GAUGE:
            self.set(metric.name, metric.value, metric.labels.labels)
        elif metric.type == MetricType.HISTOGRAM:
            self.observe(metric.name, metric.value, metric.labels.labels)

        self._emit_event(self.BUS_TOPICS["record"], metric.to_dict())

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        label_key = MetricLabels(labels or {}).to_key()
        return self._counters.get(name, {}).get(label_key, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        label_key = MetricLabels(labels or {}).to_key()
        return self._gauges.get(name, {}).get(label_key, 0)

    def get_histogram(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Histogram]:
        """Get histogram."""
        label_key = MetricLabels(labels or {}).to_key()
        return self._histograms.get(name, {}).get(label_key)

    def get_all_metrics(self) -> List[Metric]:
        """Get all current metric values."""
        metrics = []

        # Counters
        for name, label_values in self._counters.items():
            for label_key, value in label_values.items():
                labels = dict(kv.split("=") for kv in label_key.split(",") if "=" in kv)
                metrics.append(Metric(
                    name=name,
                    type=MetricType.COUNTER,
                    value=value,
                    unit=self._units.get(name, MetricUnit.COUNT),
                    labels=MetricLabels(labels),
                    description=self._descriptions.get(name, ""),
                ))

        # Gauges
        for name, label_values in self._gauges.items():
            for label_key, value in label_values.items():
                labels = dict(kv.split("=") for kv in label_key.split(",") if "=" in kv)
                metrics.append(Metric(
                    name=name,
                    type=MetricType.GAUGE,
                    value=value,
                    unit=self._units.get(name, MetricUnit.NONE),
                    labels=MetricLabels(labels),
                    description=self._descriptions.get(name, ""),
                ))

        # Histograms (export as multiple metrics)
        for name, label_hists in self._histograms.items():
            for label_key, hist in label_hists.items():
                labels = dict(kv.split("=") for kv in label_key.split(",") if "=" in kv)
                metrics.append(Metric(
                    name=f"{name}_sum",
                    type=MetricType.GAUGE,
                    value=hist.sum,
                    unit=self._units.get(name, MetricUnit.NONE),
                    labels=MetricLabels(labels),
                ))
                metrics.append(Metric(
                    name=f"{name}_count",
                    type=MetricType.COUNTER,
                    value=hist.count,
                    unit=MetricUnit.COUNT,
                    labels=MetricLabels(labels),
                ))

        return metrics

    async def generate_report(
        self,
        period_hours: int = 1,
    ) -> MetricReport:
        """
        Generate a metrics report.

        Args:
            period_hours: Report period in hours

        Returns:
            MetricReport with aggregated metrics

        Emits:
            review.metrics.report
        """
        now = datetime.now(timezone.utc)
        period_start = now - timedelta(hours=period_hours)

        report = MetricReport(
            report_id=str(uuid.uuid4())[:8],
            generated_at=now.isoformat() + "Z",
            period_start=period_start.isoformat() + "Z",
            period_end=now.isoformat() + "Z",
        )

        # Collect all metrics
        metrics = self.get_all_metrics()
        report.metrics = [m.to_dict() for m in metrics]

        # Generate summaries for histograms
        for name, label_hists in self._histograms.items():
            for label_key, hist in label_hists.items():
                key = f"{name}:{label_key}" if label_key else name
                report.summaries[key] = {
                    "count": hist.count,
                    "sum": hist.sum,
                    "avg": hist.sum / hist.count if hist.count > 0 else 0,
                    "p50": hist.get_percentile(50),
                    "p90": hist.get_percentile(90),
                    "p95": hist.get_percentile(95),
                    "p99": hist.get_percentile(99),
                }

        self._emit_event(self.BUS_TOPICS["report"], {
            "report_id": report.report_id,
            "period_hours": period_hours,
            "metric_count": len(report.metrics),
        })

        return report

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset metrics.

        Args:
            name: Specific metric to reset (None = all)
        """
        if name:
            self._counters.pop(name, None)
            self._gauges.pop(name, None)
            self._histograms.pop(name, None)
        else:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        metrics = self.get_all_metrics()
        status = {
            "agent": "metrics-collector",
            "healthy": True,
            "metric_count": len(metrics),
            "counter_count": sum(len(v) for v in self._counters.values()),
            "gauge_count": sum(len(v) for v in self._gauges.values()),
            "histogram_count": sum(len(v) for v in self._histograms.values()),
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# Timer Context Manager
# ============================================================================

class Timer:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.collector = collector
        self.name = name
        self.labels = labels
        self._start: Optional[float] = None
        self._duration: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            self._duration = (time.time() - self._start) * 1000
            self.collector.observe(self.name, self._duration, self.labels)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        return self._duration


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Metrics."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Review Metrics (Step 183)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List all metrics")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--period", type=int, default=1, help="Period in hours")
    report_parser.add_argument("--prometheus", action="store_true", help="Prometheus format")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record a metric")
    record_parser.add_argument("name", help="Metric name")
    record_parser.add_argument("value", type=float, help="Metric value")
    record_parser.add_argument("--type", choices=["counter", "gauge", "histogram"], default="gauge")
    record_parser.add_argument("--labels", help="Labels (key=value,key=value)")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    collector = MetricsCollector()

    if args.command == "list":
        metrics = collector.get_all_metrics()
        if args.json:
            print(json.dumps([m.to_dict() for m in metrics], indent=2))
        else:
            print(f"Metrics: {len(metrics)}")
            for m in metrics:
                labels = ",".join(f"{k}={v}" for k, v in m.labels.labels.items()) or "no labels"
                print(f"  {m.name} ({m.type.value}): {m.value} [{labels}]")

    elif args.command == "report":
        report = asyncio.run(collector.generate_report(period_hours=args.period))
        if args.prometheus:
            print(report.to_prometheus())
        elif args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Report: {report.report_id}")
            print(f"  Period: {report.period_start} to {report.period_end}")
            print(f"  Metrics: {len(report.metrics)}")
            for name, summary in report.summaries.items():
                print(f"  {name}: avg={summary['avg']:.2f}, p95={summary['p95']:.2f}")

    elif args.command == "record":
        labels = {}
        if args.labels:
            for kv in args.labels.split(","):
                k, v = kv.split("=", 1)
                labels[k] = v

        metric_type = {
            "counter": MetricType.COUNTER,
            "gauge": MetricType.GAUGE,
            "histogram": MetricType.HISTOGRAM,
        }[args.type]

        collector.record(Metric(
            name=args.name,
            type=metric_type,
            value=args.value,
            labels=MetricLabels(labels),
        ))
        print(f"Recorded: {args.name} = {args.value}")

    else:
        # Default: show status
        status = collector.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Metrics Collector: {status['metric_count']} metrics")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
