#!/usr/bin/env python3
"""
collector.py - Telemetry Collector (Step 247)

PBTSO Phase: OBSERVE
A2A Integration: Usage analytics via deploy.telemetry.collect

Provides:
- MetricType: Types of metrics
- TelemetryEvent: Telemetry event
- UsageMetric: Usage metric
- AnalyticsReport: Analytics report
- TelemetryConfig: Telemetry configuration
- TelemetryCollector: Complete telemetry collection

Bus Topics:
- deploy.telemetry.collect
- deploy.telemetry.metric
- deploy.telemetry.report
- deploy.telemetry.export

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ==============================================================================
# Bus Emission Helper with fcntl.flock()
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "telemetry-collector"
) -> str:
    """Emit an event to the Pluribus bus with file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AggregationType(Enum):
    """Aggregation types."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P99 = "p99"


@dataclass
class TelemetryConfig:
    """
    Telemetry configuration.

    Attributes:
        config_id: Unique config identifier
        enabled: Whether telemetry is enabled
        sample_rate: Sampling rate (0.0-1.0)
        flush_interval_s: Flush interval in seconds
        batch_size: Batch size for sending
        retention_days: Days to retain data
        anonymize: Whether to anonymize data
        exclude_patterns: Patterns to exclude
    """
    config_id: str
    enabled: bool = True
    sample_rate: float = 1.0
    flush_interval_s: int = 60
    batch_size: int = 100
    retention_days: int = 30
    anonymize: bool = False
    exclude_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetryEvent:
    """
    Telemetry event.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        name: Event name
        value: Event value
        tags: Event tags
        timestamp: Event timestamp
        metadata: Additional metadata
    """
    event_id: str
    event_type: str
    name: str
    value: Any = None
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UsageMetric:
    """
    Usage metric.

    Attributes:
        metric_id: Unique metric identifier
        name: Metric name
        metric_type: Type of metric
        value: Current value
        unit: Metric unit
        tags: Metric tags
        timestamp: Metric timestamp
    """
    metric_id: str
    name: str
    metric_type: MetricType = MetricType.GAUGE
    value: float = 0.0
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


@dataclass
class AggregatedMetric:
    """
    Aggregated metric over a time period.

    Attributes:
        name: Metric name
        aggregation: Aggregation type
        value: Aggregated value
        count: Number of samples
        period_start: Period start timestamp
        period_end: Period end timestamp
        tags: Metric tags
    """
    name: str
    aggregation: AggregationType
    value: float
    count: int = 0
    period_start: float = 0.0
    period_end: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aggregation": self.aggregation.value,
            "value": self.value,
            "count": self.count,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "tags": self.tags,
        }


@dataclass
class AnalyticsReport:
    """
    Analytics report.

    Attributes:
        report_id: Unique report identifier
        name: Report name
        period_start: Report period start
        period_end: Report period end
        metrics: Aggregated metrics
        events: Event counts by type
        summary: Report summary
        generated_at: Generation timestamp
    """
    report_id: str
    name: str
    period_start: float
    period_end: float
    metrics: List[AggregatedMetric] = field(default_factory=list)
    events: Dict[str, int] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "name": self.name,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "metrics": [m.to_dict() for m in self.metrics],
            "events": self.events,
            "summary": self.summary,
            "generated_at": self.generated_at,
        }


# ==============================================================================
# Telemetry Collector (Step 247)
# ==============================================================================

class TelemetryCollector:
    """
    Telemetry Collector - Usage analytics and telemetry for deployments.

    PBTSO Phase: OBSERVE

    Responsibilities:
    - Collect telemetry events
    - Track usage metrics
    - Aggregate and analyze data
    - Generate analytics reports
    - Export telemetry data

    Example:
        >>> collector = TelemetryCollector()
        >>> collector.track_event("deployment", "started", {"service": "api"})
        >>> collector.record_metric("deploy_duration_ms", 1523.4, MetricType.HISTOGRAM)
        >>> report = collector.generate_report("daily")
    """

    BUS_TOPICS = {
        "collect": "deploy.telemetry.collect",
        "metric": "deploy.telemetry.metric",
        "report": "deploy.telemetry.report",
        "export": "deploy.telemetry.export",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        config: Optional[TelemetryConfig] = None,
        actor_id: str = "telemetry-collector",
    ):
        """
        Initialize the telemetry collector.

        Args:
            state_dir: Directory for state persistence
            config: Telemetry configuration
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "telemetry"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Configuration
        self.config = config or TelemetryConfig(
            config_id=f"config-{uuid.uuid4().hex[:8]}"
        )

        # Storage
        self._events: List[TelemetryEvent] = []
        self._metrics: Dict[str, List[UsageMetric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)

        # Timing tracking
        self._active_timers: Dict[str, float] = {}

        self._load_state()

    def track_event(
        self,
        event_type: str,
        name: str,
        value: Any = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryEvent:
        """
        Track a telemetry event.

        Args:
            event_type: Type of event
            name: Event name
            value: Event value
            tags: Event tags
            metadata: Additional metadata

        Returns:
            Created TelemetryEvent
        """
        if not self.config.enabled:
            return TelemetryEvent(
                event_id="disabled",
                event_type=event_type,
                name=name,
            )

        # Apply sampling
        import random
        if random.random() > self.config.sample_rate:
            return TelemetryEvent(
                event_id="sampled_out",
                event_type=event_type,
                name=name,
            )

        event_id = f"event-{uuid.uuid4().hex[:12]}"

        event = TelemetryEvent(
            event_id=event_id,
            event_type=event_type,
            name=name,
            value=value,
            tags=tags or {},
            metadata=metadata or {},
        )

        self._events.append(event)

        _emit_bus_event(
            self.BUS_TOPICS["collect"],
            {
                "event_id": event_id,
                "event_type": event_type,
                "name": name,
            },
            kind="telemetry",
            actor=self.actor_id,
        )

        # Auto-flush if batch size reached
        if len(self._events) >= self.config.batch_size:
            self._flush_events()

        return event

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> UsageMetric:
        """
        Record a usage metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            unit: Metric unit
            tags: Metric tags

        Returns:
            Created UsageMetric
        """
        if not self.config.enabled:
            return UsageMetric(
                metric_id="disabled",
                name=name,
                value=value,
            )

        metric_id = f"metric-{uuid.uuid4().hex[:8]}"

        metric = UsageMetric(
            metric_id=metric_id,
            name=name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            tags=tags or {},
        )

        # Store based on type
        if metric_type == MetricType.COUNTER:
            self._counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self._gauges[name] = value
        elif metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
            self._histograms[name].append(value)
            # Keep last 10000 values
            self._histograms[name] = self._histograms[name][-10000:]

        self._metrics[name].append(metric)

        _emit_bus_event(
            self.BUS_TOPICS["metric"],
            {
                "name": name,
                "value": value,
                "type": metric_type.value,
            },
            kind="metric",
            actor=self.actor_id,
        )

        return metric

    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags=tags)

    def gauge(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, unit, tags)

    def histogram(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        self.record_metric(name, value, MetricType.HISTOGRAM, unit, tags)

    def start_timer(self, name: str) -> str:
        """Start a timer."""
        timer_id = f"{name}_{uuid.uuid4().hex[:8]}"
        self._active_timers[timer_id] = time.time()
        return timer_id

    def stop_timer(
        self,
        timer_id: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Stop a timer and record the duration."""
        if timer_id not in self._active_timers:
            return 0.0

        start_time = self._active_timers.pop(timer_id)
        duration_ms = (time.time() - start_time) * 1000

        # Extract metric name from timer_id
        name = timer_id.rsplit("_", 1)[0]
        self.record_metric(
            f"{name}_duration_ms",
            duration_ms,
            MetricType.HISTOGRAM,
            "ms",
            tags,
        )

        return duration_ms

    def generate_report(
        self,
        name: str,
        period_hours: int = 24,
    ) -> AnalyticsReport:
        """
        Generate an analytics report.

        Args:
            name: Report name
            period_hours: Period in hours

        Returns:
            AnalyticsReport
        """
        report_id = f"report-{uuid.uuid4().hex[:8]}"
        period_end = time.time()
        period_start = period_end - (period_hours * 3600)

        # Aggregate metrics
        aggregated_metrics = []

        for metric_name, values in self._histograms.items():
            if not values:
                continue

            # Filter to period
            # (In real implementation, would filter by timestamp)

            aggregated_metrics.extend([
                AggregatedMetric(
                    name=metric_name,
                    aggregation=AggregationType.AVG,
                    value=statistics.mean(values),
                    count=len(values),
                    period_start=period_start,
                    period_end=period_end,
                ),
                AggregatedMetric(
                    name=metric_name,
                    aggregation=AggregationType.P50,
                    value=statistics.median(values),
                    count=len(values),
                    period_start=period_start,
                    period_end=period_end,
                ),
                AggregatedMetric(
                    name=metric_name,
                    aggregation=AggregationType.P99,
                    value=self._percentile(values, 99),
                    count=len(values),
                    period_start=period_start,
                    period_end=period_end,
                ),
                AggregatedMetric(
                    name=metric_name,
                    aggregation=AggregationType.MAX,
                    value=max(values),
                    count=len(values),
                    period_start=period_start,
                    period_end=period_end,
                ),
            ])

        # Add counters
        for counter_name, value in self._counters.items():
            aggregated_metrics.append(AggregatedMetric(
                name=counter_name,
                aggregation=AggregationType.SUM,
                value=value,
                period_start=period_start,
                period_end=period_end,
            ))

        # Add gauges
        for gauge_name, value in self._gauges.items():
            aggregated_metrics.append(AggregatedMetric(
                name=gauge_name,
                aggregation=AggregationType.AVG,
                value=value,
                count=1,
                period_start=period_start,
                period_end=period_end,
            ))

        # Count events by type
        event_counts: Dict[str, int] = defaultdict(int)
        for event in self._events:
            if period_start <= event.timestamp <= period_end:
                event_counts[event.event_type] += 1

        # Generate summary
        summary = {
            "total_events": sum(event_counts.values()),
            "total_metrics": len(aggregated_metrics),
            "unique_metric_names": len(set(m.name for m in aggregated_metrics)),
            "period_hours": period_hours,
        }

        report = AnalyticsReport(
            report_id=report_id,
            name=name,
            period_start=period_start,
            period_end=period_end,
            metrics=aggregated_metrics,
            events=dict(event_counts),
            summary=summary,
        )

        _emit_bus_event(
            self.BUS_TOPICS["report"],
            {
                "report_id": report_id,
                "name": name,
                "metrics_count": len(aggregated_metrics),
                "events_count": sum(event_counts.values()),
            },
            actor=self.actor_id,
        )

        return report

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def export_data(
        self,
        format: str = "json",
        period_hours: int = 24,
    ) -> str:
        """
        Export telemetry data.

        Args:
            format: Export format (json, csv)
            period_hours: Period to export

        Returns:
            Exported data string
        """
        period_end = time.time()
        period_start = period_end - (period_hours * 3600)

        # Filter events
        events = [
            e.to_dict() for e in self._events
            if period_start <= e.timestamp <= period_end
        ]

        # Collect metrics
        metrics = []
        for name, metric_list in self._metrics.items():
            for m in metric_list:
                if period_start <= m.timestamp <= period_end:
                    metrics.append(m.to_dict())

        data = {
            "period_start": period_start,
            "period_end": period_end,
            "events": events,
            "metrics": metrics,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }

        _emit_bus_event(
            self.BUS_TOPICS["export"],
            {
                "format": format,
                "events_count": len(events),
                "metrics_count": len(metrics),
            },
            actor=self.actor_id,
        )

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            # Simple CSV format
            lines = ["timestamp,type,name,value"]
            for event in events:
                lines.append(f"{event['timestamp']},event,{event['name']},{event.get('value', '')}")
            for metric in metrics:
                lines.append(f"{metric['timestamp']},metric,{metric['name']},{metric['value']}")
            return "\n".join(lines)

    def get_metric(self, name: str) -> Optional[float]:
        """Get current metric value."""
        if name in self._gauges:
            return self._gauges[name]
        if name in self._counters:
            return self._counters[name]
        if name in self._histograms and self._histograms[name]:
            return statistics.mean(self._histograms[name])
        return None

    def get_counter(self, name: str) -> float:
        """Get counter value."""
        return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        return self._gauges.get(name)

    def list_metrics(self) -> List[str]:
        """List all metric names."""
        names = set(self._counters.keys())
        names.update(self._gauges.keys())
        names.update(self._histograms.keys())
        return sorted(names)

    def _flush_events(self) -> None:
        """Flush events to storage."""
        if not self._events:
            return

        # Save to disk
        self._save_state()

        # Apply retention
        cutoff = time.time() - (self.config.retention_days * 86400)
        self._events = [e for e in self._events if e.timestamp > cutoff]

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "events": [e.to_dict() for e in self._events[-10000:]],  # Keep last 10000
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {k: v[-1000:] for k, v in self._histograms.items()},
        }
        state_file = self.state_dir / "telemetry_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "telemetry_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for e_data in state.get("events", []):
                self._events.append(TelemetryEvent(**e_data))

            self._counters = defaultdict(float, state.get("counters", {}))
            self._gauges = state.get("gauges", {})
            self._histograms = defaultdict(list, state.get("histograms", {}))

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for telemetry collector."""
    import argparse

    parser = argparse.ArgumentParser(description="Telemetry Collector (Step 247)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # track command
    track_parser = subparsers.add_parser("track", help="Track an event")
    track_parser.add_argument("event_type", help="Event type")
    track_parser.add_argument("name", help="Event name")
    track_parser.add_argument("--value", "-v", help="Event value")
    track_parser.add_argument("--tags", "-t", help="Tags (key=value,...)")

    # metric command
    metric_parser = subparsers.add_parser("metric", help="Record a metric")
    metric_parser.add_argument("name", help="Metric name")
    metric_parser.add_argument("value", type=float, help="Metric value")
    metric_parser.add_argument("--type", "-t", default="gauge",
                              choices=["counter", "gauge", "histogram"])
    metric_parser.add_argument("--unit", "-u", default="", help="Metric unit")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--name", "-n", default="analytics", help="Report name")
    report_parser.add_argument("--period", "-p", type=int, default=24, help="Period in hours")
    report_parser.add_argument("--json", action="store_true", help="JSON output")

    # export command
    export_parser = subparsers.add_parser("export", help="Export telemetry data")
    export_parser.add_argument("--format", "-f", default="json", choices=["json", "csv"])
    export_parser.add_argument("--period", "-p", type=int, default=24, help="Period in hours")
    export_parser.add_argument("--output", "-o", help="Output file")

    # list command
    list_parser = subparsers.add_parser("list", help="List metrics")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    collector = TelemetryCollector()

    if args.command == "track":
        tags = {}
        if args.tags:
            for pair in args.tags.split(","):
                key, _, value = pair.partition("=")
                tags[key] = value

        event = collector.track_event(
            event_type=args.event_type,
            name=args.name,
            value=args.value,
            tags=tags,
        )
        print(f"Tracked event: {event.event_id}")
        return 0

    elif args.command == "metric":
        metric = collector.record_metric(
            name=args.name,
            value=args.value,
            metric_type=MetricType(args.type),
            unit=args.unit,
        )
        print(f"Recorded metric: {metric.metric_id}")
        print(f"  Name: {metric.name}")
        print(f"  Value: {metric.value}")
        return 0

    elif args.command == "report":
        report = collector.generate_report(
            name=args.name,
            period_hours=args.period,
        )

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Report: {report.name}")
            print(f"  ID: {report.report_id}")
            print(f"  Period: {args.period} hours")
            print(f"  Metrics: {len(report.metrics)}")
            print(f"  Events: {sum(report.events.values())}")
            print("")
            print("Event counts:")
            for event_type, count in report.events.items():
                print(f"  {event_type}: {count}")

        return 0

    elif args.command == "export":
        data = collector.export_data(
            format=args.format,
            period_hours=args.period,
        )

        if args.output:
            with open(args.output, "w") as f:
                f.write(data)
            print(f"Exported to {args.output}")
        else:
            print(data)

        return 0

    elif args.command == "list":
        metrics = collector.list_metrics()

        if args.json:
            metric_data = []
            for name in metrics:
                value = collector.get_metric(name)
                metric_data.append({"name": name, "value": value})
            print(json.dumps(metric_data, indent=2))
        else:
            if not metrics:
                print("No metrics found")
            else:
                for name in metrics:
                    value = collector.get_metric(name)
                    print(f"  {name}: {value}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
