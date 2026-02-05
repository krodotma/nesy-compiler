#!/usr/bin/env python3
"""
Step 133: Test Metrics

Performance and usage metrics for the Test Agent.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.metrics.record (emits)
- test.metrics.report (emits)
- test.metrics.alert (emits)

Dependencies: Steps 101-132 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"        # Monotonically increasing
    GAUGE = "gauge"            # Point-in-time value
    HISTOGRAM = "histogram"    # Distribution of values
    TIMER = "timer"            # Duration measurements
    RATE = "rate"              # Events per time unit


class MetricAggregation(Enum):
    """Aggregation methods for metrics."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    RATE = "rate"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class MetricValue:
    """
    A single metric value.

    Attributes:
        name: Metric name
        value: Metric value
        metric_type: Type of metric
        timestamp: Recording timestamp
        labels: Metric labels/tags
        unit: Unit of measurement
    """
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "unit": self.unit,
        }


@dataclass
class MetricSeries:
    """
    Time series of metric values.

    Attributes:
        name: Metric name
        values: List of values
        metric_type: Type of metric
        labels: Common labels
    """
    name: str
    values: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, value)
    metric_type: MetricType = MetricType.GAUGE
    labels: Dict[str, str] = field(default_factory=dict)
    max_values: int = 1000

    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add a value to the series."""
        ts = timestamp or time.time()
        self.values.append((ts, value))
        # Trim if too many values
        if len(self.values) > self.max_values:
            self.values = self.values[-self.max_values:]

    def get_aggregated(self, aggregation: MetricAggregation, window_s: Optional[float] = None) -> float:
        """Get aggregated value."""
        if not self.values:
            return 0.0

        values = self.values
        if window_s:
            cutoff = time.time() - window_s
            values = [(ts, v) for ts, v in values if ts >= cutoff]

        if not values:
            return 0.0

        nums = [v for _, v in values]

        if aggregation == MetricAggregation.SUM:
            return sum(nums)
        elif aggregation == MetricAggregation.AVG:
            return statistics.mean(nums)
        elif aggregation == MetricAggregation.MIN:
            return min(nums)
        elif aggregation == MetricAggregation.MAX:
            return max(nums)
        elif aggregation == MetricAggregation.COUNT:
            return len(nums)
        elif aggregation == MetricAggregation.P50:
            return self._percentile(nums, 50)
        elif aggregation == MetricAggregation.P90:
            return self._percentile(nums, 90)
        elif aggregation == MetricAggregation.P95:
            return self._percentile(nums, 95)
        elif aggregation == MetricAggregation.P99:
            return self._percentile(nums, 99)
        elif aggregation == MetricAggregation.RATE:
            if len(values) < 2:
                return 0.0
            duration = values[-1][0] - values[0][0]
            return sum(nums) / duration if duration > 0 else 0.0

        return 0.0

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


@dataclass
class MetricsReport:
    """
    Metrics report.

    Attributes:
        report_id: Unique report ID
        timestamp: Report timestamp
        metrics: Metric values
        aggregations: Aggregated metrics
        alerts: Triggered alerts
        period_s: Report period
    """
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    aggregations: Dict[str, float] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    period_s: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "aggregations": self.aggregations,
            "alerts": self.alerts,
            "period_s": self.period_s,
        }


@dataclass
class MetricAlert:
    """
    Metric alert definition.

    Attributes:
        name: Alert name
        metric_name: Metric to watch
        condition: Alert condition (gt, lt, eq)
        threshold: Threshold value
        aggregation: Aggregation method
        window_s: Evaluation window
        enabled: Whether alert is enabled
    """
    name: str
    metric_name: str
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    aggregation: MetricAggregation = MetricAggregation.AVG
    window_s: int = 60
    enabled: bool = True
    cooldown_s: int = 300
    last_triggered: Optional[float] = None

    def evaluate(self, value: float) -> bool:
        """Evaluate if alert should trigger."""
        if not self.enabled:
            return False

        if self.last_triggered and time.time() - self.last_triggered < self.cooldown_s:
            return False

        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return abs(value - self.threshold) < 0.001
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "aggregation": self.aggregation.value,
            "window_s": self.window_s,
            "enabled": self.enabled,
        }


@dataclass
class MetricsConfig:
    """
    Configuration for the metrics system.

    Attributes:
        output_dir: Output directory
        retention_hours: Data retention period
        report_interval_s: Report generation interval
        alerts: Metric alerts
        default_labels: Default labels for all metrics
        enable_histograms: Enable histogram metrics
    """
    output_dir: str = ".pluribus/test-agent/metrics"
    retention_hours: int = 24
    report_interval_s: int = 60
    alerts: List[MetricAlert] = field(default_factory=list)
    default_labels: Dict[str, str] = field(default_factory=dict)
    enable_histograms: bool = True
    max_series: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": self.output_dir,
            "retention_hours": self.retention_hours,
            "report_interval_s": self.report_interval_s,
            "alerts": [a.to_dict() for a in self.alerts],
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class MetricsBus:
    """Bus interface for metrics with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Test Metrics
# ============================================================================

class TestMetrics:
    """
    Performance and usage metrics for the Test Agent.

    Features:
    - Multiple metric types (counter, gauge, histogram, timer)
    - Time series storage
    - Aggregations (sum, avg, percentiles, etc.)
    - Alert support
    - Prometheus-compatible output

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.metrics.record, test.metrics.report, test.metrics.alert
    """

    BUS_TOPICS = {
        "record": "test.metrics.record",
        "report": "test.metrics.report",
        "alert": "test.metrics.alert",
    }

    # Standard test agent metrics
    STANDARD_METRICS = {
        "test.runs.total": MetricType.COUNTER,
        "test.runs.passed": MetricType.COUNTER,
        "test.runs.failed": MetricType.COUNTER,
        "test.runs.skipped": MetricType.COUNTER,
        "test.duration.seconds": MetricType.HISTOGRAM,
        "test.coverage.percent": MetricType.GAUGE,
        "test.cache.hit_rate": MetricType.GAUGE,
        "test.cache.size_bytes": MetricType.GAUGE,
        "test.parallel.workers": MetricType.GAUGE,
        "test.parallel.speedup": MetricType.GAUGE,
        "test.flaky.count": MetricType.GAUGE,
        "test.queue.size": MetricType.GAUGE,
        "test.api.requests.total": MetricType.COUNTER,
        "test.api.latency.seconds": MetricType.HISTOGRAM,
    }

    def __init__(self, bus=None, config: Optional[MetricsConfig] = None):
        """
        Initialize the metrics system.

        Args:
            bus: Optional bus instance
            config: Metrics configuration
        """
        self.bus = bus or MetricsBus()
        self.config = config or MetricsConfig()
        self._series: Dict[str, MetricSeries] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._last_report: Optional[float] = None

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize standard metrics
        for name, metric_type in self.STANDARD_METRICS.items():
            self._get_or_create_series(name, metric_type)

    def _get_or_create_series(
        self,
        name: str,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
    ) -> MetricSeries:
        """Get or create a metric series."""
        key = self._series_key(name, labels or {})
        if key not in self._series:
            if len(self._series) >= self.config.max_series:
                # Evict oldest series
                oldest = min(self._series.keys(),
                            key=lambda k: self._series[k].values[0][0] if self._series[k].values else float('inf'))
                del self._series[oldest]

            self._series[key] = MetricSeries(
                name=name,
                metric_type=metric_type,
                labels={**self.config.default_labels, **(labels or {})},
            )
        return self._series[key]

    def _series_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate a unique key for a series."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        unit: str = "",
    ) -> MetricValue:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Metric labels
            unit: Unit of measurement

        Returns:
            Recorded metric value
        """
        series = self._get_or_create_series(name, metric_type, labels)

        if metric_type == MetricType.COUNTER:
            self._counters[name] += value
            series.add(self._counters[name])
        else:
            series.add(value)

        metric_value = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            unit=unit,
        )

        self._emit_event("record", metric_value.to_dict())

        # Check alerts
        self._check_alerts(name, labels)

        return metric_value

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record(name, value, MetricType.COUNTER, labels)

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, unit: str = "") -> None:
        """Record a gauge metric."""
        self.record(name, value, MetricType.GAUGE, labels, unit)

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, unit: str = "") -> None:
        """Record a histogram metric."""
        self.record(name, value, MetricType.HISTOGRAM, labels, unit)

    def timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> 'Timer':
        """
        Create a timer context manager.

        Usage:
            with metrics.timer("test.duration.seconds"):
                run_test()
        """
        return Timer(self, name, labels)

    def get(
        self,
        name: str,
        aggregation: MetricAggregation = MetricAggregation.AVG,
        labels: Optional[Dict[str, str]] = None,
        window_s: Optional[float] = None,
    ) -> float:
        """
        Get an aggregated metric value.

        Args:
            name: Metric name
            aggregation: Aggregation method
            labels: Metric labels
            window_s: Time window

        Returns:
            Aggregated value
        """
        key = self._series_key(name, labels or {})
        series = self._series.get(key)
        if series is None:
            return 0.0
        return series.get_aggregated(aggregation, window_s)

    def get_series(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        window_s: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Get raw time series data."""
        key = self._series_key(name, labels or {})
        series = self._series.get(key)
        if series is None:
            return []

        values = series.values
        if window_s:
            cutoff = time.time() - window_s
            values = [(ts, v) for ts, v in values if ts >= cutoff]

        return values

    def generate_report(self, period_s: Optional[int] = None) -> MetricsReport:
        """
        Generate a metrics report.

        Args:
            period_s: Report period in seconds

        Returns:
            MetricsReport with aggregated metrics
        """
        period = period_s or self.config.report_interval_s
        report = MetricsReport(period_s=period)

        for key, series in self._series.items():
            metrics_dict = {
                "current": series.get_aggregated(MetricAggregation.AVG, window_s=period),
                "min": series.get_aggregated(MetricAggregation.MIN, window_s=period),
                "max": series.get_aggregated(MetricAggregation.MAX, window_s=period),
                "count": series.get_aggregated(MetricAggregation.COUNT, window_s=period),
            }

            if series.metric_type == MetricType.HISTOGRAM:
                metrics_dict["p50"] = series.get_aggregated(MetricAggregation.P50, window_s=period)
                metrics_dict["p90"] = series.get_aggregated(MetricAggregation.P90, window_s=period)
                metrics_dict["p95"] = series.get_aggregated(MetricAggregation.P95, window_s=period)
                metrics_dict["p99"] = series.get_aggregated(MetricAggregation.P99, window_s=period)

            report.metrics[key] = metrics_dict

        # Standard aggregations
        report.aggregations = {
            "total_tests": self.get("test.runs.total", MetricAggregation.SUM),
            "pass_rate": self._calculate_pass_rate(),
            "avg_duration": self.get("test.duration.seconds", MetricAggregation.AVG, window_s=period),
            "p95_duration": self.get("test.duration.seconds", MetricAggregation.P95, window_s=period),
            "cache_hit_rate": self.get("test.cache.hit_rate", MetricAggregation.AVG, window_s=period),
        }

        self._emit_event("report", report.to_dict())
        self._save_report(report)

        return report

    def _calculate_pass_rate(self) -> float:
        """Calculate test pass rate."""
        total = self.get("test.runs.total", MetricAggregation.SUM)
        passed = self.get("test.runs.passed", MetricAggregation.SUM)
        return (passed / total * 100) if total > 0 else 0.0

    def add_alert(self, alert: MetricAlert) -> None:
        """Add a metric alert."""
        self.config.alerts.append(alert)

    def remove_alert(self, name: str) -> bool:
        """Remove a metric alert."""
        for i, alert in enumerate(self.config.alerts):
            if alert.name == name:
                del self.config.alerts[i]
                return True
        return False

    def _check_alerts(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Check if any alerts should trigger."""
        for alert in self.config.alerts:
            if alert.metric_name != name:
                continue

            value = self.get(name, alert.aggregation, labels, window_s=alert.window_s)

            if alert.evaluate(value):
                alert.last_triggered = time.time()
                self._emit_event("alert", {
                    "alert_name": alert.name,
                    "metric_name": name,
                    "value": value,
                    "threshold": alert.threshold,
                    "condition": alert.condition,
                })

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for key, series in self._series.items():
            # Parse key for name and labels
            name = series.name
            labels = series.labels

            label_str = ""
            if labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"

            # Get current value
            if series.values:
                value = series.values[-1][1]
                lines.append(f"# TYPE {name} {series.metric_type.value}")
                lines.append(f"{name}{label_str} {value}")

                if series.metric_type == MetricType.HISTOGRAM:
                    # Add histogram buckets
                    for bucket in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
                        count = sum(1 for _, v in series.values if v <= bucket)
                        lines.append(f'{name}_bucket{{le="{bucket}"{label_str[1:-1] if label_str else ""}}} {count}')
                    count = len(series.values)
                    lines.append(f'{name}_bucket{{le="+Inf"{label_str[1:-1] if label_str else ""}}} {count}')
                    lines.append(f'{name}_sum{label_str} {sum(v for _, v in series.values)}')
                    lines.append(f'{name}_count{label_str} {count}')

        return "\n".join(lines)

    def _save_report(self, report: MetricsReport) -> None:
        """Save report to disk."""
        output_path = Path(self.config.output_dir)
        report_file = output_path / f"metrics_{report.report_id[:8]}.json"

        with open(report_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(report.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def generate_report_async(self, period_s: Optional[int] = None) -> MetricsReport:
        """Async version of generate_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_report, period_s)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.metrics.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "metrics",
            "actor": "test-agent",
            "data": data,
        })

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        result = {}
        for key, series in self._series.items():
            if series.values:
                result[key] = {
                    "current": series.values[-1][1],
                    "count": len(series.values),
                    "type": series.metric_type.value,
                }
        return result


class Timer:
    """Context manager for timing operations."""

    def __init__(self, metrics: TestMetrics, name: str, labels: Optional[Dict[str, str]] = None):
        self.metrics = metrics
        self.name = name
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self) -> 'Timer':
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record(self.name, duration, MetricType.TIMER, self.labels, unit="seconds")


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Metrics."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Metrics")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record a metric")
    record_parser.add_argument("name", help="Metric name")
    record_parser.add_argument("value", type=float, help="Metric value")
    record_parser.add_argument("--type", choices=["counter", "gauge", "histogram"],
                               default="gauge")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get metric value")
    get_parser.add_argument("name", help="Metric name")
    get_parser.add_argument("--aggregation", choices=["sum", "avg", "min", "max", "p50", "p95", "p99"],
                            default="avg")
    get_parser.add_argument("--window", type=int, help="Time window in seconds")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate metrics report")
    report_parser.add_argument("--period", type=int, default=60, help="Report period in seconds")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export metrics")
    export_parser.add_argument("--format", choices=["json", "prometheus"], default="json")

    # List command
    list_parser = subparsers.add_parser("list", help="List all metrics")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/metrics")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = MetricsConfig(output_dir=args.output)
    metrics = TestMetrics(config=config)

    if args.command == "record":
        metric_type = MetricType(args.type)
        value = metrics.record(args.name, args.value, metric_type)

        if args.json:
            print(json.dumps(value.to_dict(), indent=2))
        else:
            print(f"Recorded: {args.name} = {args.value}")

    elif args.command == "get":
        aggregation = MetricAggregation(args.aggregation)
        value = metrics.get(args.name, aggregation, window_s=args.window)

        if args.json:
            print(json.dumps({"name": args.name, "value": value, "aggregation": args.aggregation}))
        else:
            print(f"{args.name} ({args.aggregation}): {value:.4f}")

    elif args.command == "report":
        report = metrics.generate_report(args.period)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"\nMetrics Report (last {args.period}s)")
            print("=" * 40)
            for name, values in report.aggregations.items():
                print(f"  {name}: {values:.2f}")

    elif args.command == "export":
        if args.format == "prometheus":
            print(metrics.export_prometheus())
        else:
            print(json.dumps(metrics.get_all_metrics(), indent=2))

    elif args.command == "list":
        all_metrics = metrics.get_all_metrics()

        if args.json:
            print(json.dumps(all_metrics, indent=2))
        else:
            print(f"\nMetrics ({len(all_metrics)}):")
            for name, info in all_metrics.items():
                print(f"  {name}: {info['current']:.4f} ({info['type']})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
