#!/usr/bin/env python3
"""
collector.py - Deployment Metrics Collector (Step 221)

PBTSO Phase: VERIFY, ITERATE
A2A Integration: Collects and aggregates deployment metrics via deploy.metrics.*

Provides:
- MetricType: Types of deployment metrics
- MetricAggregation: Aggregation methods
- DeploymentMetric: Individual metric data point
- MetricQuery: Query for retrieving metrics
- MetricSeries: Time series metric data
- DeploymentMetricsCollector: Main collector class

Bus Topics:
- deploy.metrics.record
- deploy.metrics.aggregate
- deploy.metrics.alert
- deploy.metrics.export

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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ==============================================================================
# Bus Emission Helper with File Locking
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
    actor: str = "metrics-collector"
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
    """Types of deployment metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


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


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DeploymentMetric:
    """
    Individual deployment metric data point.

    Attributes:
        metric_id: Unique metric identifier
        name: Metric name
        metric_type: Type of metric
        value: Metric value
        tags: Metric tags/labels
        timestamp: Metric timestamp
        service_name: Associated service
        deployment_id: Associated deployment
        environment: Target environment
    """
    metric_id: str
    name: str
    metric_type: MetricType = MetricType.GAUGE
    value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    service_name: str = ""
    deployment_id: str = ""
    environment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "service_name": self.service_name,
            "deployment_id": self.deployment_id,
            "environment": self.environment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentMetric":
        data = dict(data)
        if "metric_type" in data:
            data["metric_type"] = MetricType(data["metric_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MetricQuery:
    """
    Query for retrieving metrics.

    Attributes:
        name: Metric name pattern (supports wildcards)
        service_name: Filter by service
        deployment_id: Filter by deployment
        environment: Filter by environment
        start_time: Start timestamp
        end_time: End timestamp
        aggregation: Aggregation method
        group_by: Group by tags
        interval_s: Aggregation interval
        limit: Maximum results
    """
    name: str = "*"
    service_name: Optional[str] = None
    deployment_id: Optional[str] = None
    environment: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    aggregation: MetricAggregation = MetricAggregation.AVG
    group_by: List[str] = field(default_factory=list)
    interval_s: int = 60
    limit: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "service_name": self.service_name,
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "aggregation": self.aggregation.value,
            "group_by": self.group_by,
            "interval_s": self.interval_s,
            "limit": self.limit,
        }


@dataclass
class MetricSeries:
    """
    Time series metric data.

    Attributes:
        name: Metric name
        tags: Series tags
        data_points: List of (timestamp, value) tuples
        aggregation: Aggregation used
        start_time: Series start time
        end_time: Series end time
    """
    name: str
    tags: Dict[str, str] = field(default_factory=dict)
    data_points: List[Tuple[float, float]] = field(default_factory=list)
    aggregation: MetricAggregation = MetricAggregation.AVG
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tags": self.tags,
            "data_points": self.data_points,
            "aggregation": self.aggregation.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class AlertRule:
    """
    Alert rule for metrics.

    Attributes:
        rule_id: Unique rule identifier
        name: Rule name
        metric_name: Metric to monitor
        condition: Alert condition (gt, lt, eq)
        threshold: Threshold value
        severity: Alert severity
        duration_s: Duration before alerting
        enabled: Whether rule is enabled
    """
    rule_id: str
    name: str
    metric_name: str
    condition: str = "gt"  # gt, lt, eq, gte, lte
    threshold: float = 0.0
    severity: AlertSeverity = AlertSeverity.WARNING
    duration_s: int = 60
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "duration_s": self.duration_s,
            "enabled": self.enabled,
        }


# ==============================================================================
# Deployment Metrics Collector (Step 221)
# ==============================================================================

class DeploymentMetricsCollector:
    """
    Deployment Metrics Collector - collects and aggregates deployment analytics.

    PBTSO Phase: VERIFY, ITERATE

    Responsibilities:
    - Record deployment metrics (success rate, duration, rollback rate)
    - Aggregate metrics over time windows
    - Generate alerts based on thresholds
    - Export metrics to external systems
    - Track deployment trends

    Example:
        >>> collector = DeploymentMetricsCollector()
        >>> collector.record("deployment.duration", 120.5, tags={"env": "prod"})
        >>> series = collector.query(MetricQuery(name="deployment.*"))
        >>> print(f"Data points: {len(series[0].data_points)}")
    """

    BUS_TOPICS = {
        "record": "deploy.metrics.record",
        "aggregate": "deploy.metrics.aggregate",
        "alert": "deploy.metrics.alert",
        "export": "deploy.metrics.export",
    }

    # Standard deployment metrics
    STANDARD_METRICS = [
        "deployment.count",
        "deployment.duration_ms",
        "deployment.success_rate",
        "deployment.rollback_rate",
        "deployment.failure_rate",
        "deployment.queue_depth",
        "deployment.active_count",
        "build.duration_ms",
        "build.success_rate",
        "container.build_time_ms",
        "container.push_time_ms",
        "health.check_latency_ms",
        "traffic.shift_duration_ms",
    ]

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "metrics-collector",
        retention_hours: int = 168,  # 7 days
    ):
        """
        Initialize the metrics collector.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            retention_hours: Hours to retain metrics
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "metrics"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.retention_hours = retention_hours

        # In-memory storage (recent metrics)
        self._metrics: Dict[str, List[DeploymentMetric]] = {}
        self._counters: Dict[str, float] = {}
        self._alert_rules: Dict[str, AlertRule] = {}
        self._alert_state: Dict[str, Dict[str, Any]] = {}

        # Custom exporters
        self._exporters: Dict[str, Callable[[List[DeploymentMetric]], None]] = {}

        self._load_state()

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        service_name: str = "",
        deployment_id: str = "",
        environment: str = "",
    ) -> DeploymentMetric:
        """
        Record a metric data point.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Additional tags
            service_name: Associated service
            deployment_id: Associated deployment
            environment: Target environment

        Returns:
            Recorded DeploymentMetric
        """
        metric = DeploymentMetric(
            metric_id=f"metric-{uuid.uuid4().hex[:12]}",
            name=name,
            metric_type=metric_type,
            value=value,
            tags=tags or {},
            service_name=service_name,
            deployment_id=deployment_id,
            environment=environment,
        )

        # Store metric
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(metric)

        # Update counter for counter type
        if metric_type == MetricType.COUNTER:
            counter_key = f"{name}:{service_name}:{environment}"
            self._counters[counter_key] = self._counters.get(counter_key, 0) + value

        # Check alert rules
        self._check_alerts(metric)

        # Emit bus event
        _emit_bus_event(
            self.BUS_TOPICS["record"],
            {
                "name": name,
                "value": value,
                "metric_type": metric_type.value,
                "service_name": service_name,
                "deployment_id": deployment_id,
            },
            kind="metric",
            actor=self.actor_id,
        )

        return metric

    def increment(
        self,
        name: str,
        delta: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
        service_name: str = "",
        environment: str = "",
    ) -> float:
        """Increment a counter metric."""
        return self.record(
            name=name,
            value=delta,
            metric_type=MetricType.COUNTER,
            tags=tags,
            service_name=service_name,
            environment=environment,
        ).value

    def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        service_name: str = "",
        environment: str = "",
    ) -> float:
        """Record a gauge metric."""
        return self.record(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags,
            service_name=service_name,
            environment=environment,
        ).value

    def timer(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None,
        service_name: str = "",
        deployment_id: str = "",
    ) -> float:
        """Record a timer/duration metric."""
        return self.record(
            name=name,
            value=duration_ms,
            metric_type=MetricType.TIMER,
            tags=tags,
            service_name=service_name,
            deployment_id=deployment_id,
        ).value

    def query(self, query: MetricQuery) -> List[MetricSeries]:
        """
        Query metrics.

        Args:
            query: Metric query parameters

        Returns:
            List of MetricSeries matching query
        """
        results: Dict[str, List[DeploymentMetric]] = {}

        for name, metrics in self._metrics.items():
            # Check name pattern
            if query.name != "*" and not self._match_pattern(name, query.name):
                continue

            for metric in metrics:
                # Apply filters
                if query.service_name and metric.service_name != query.service_name:
                    continue
                if query.deployment_id and metric.deployment_id != query.deployment_id:
                    continue
                if query.environment and metric.environment != query.environment:
                    continue
                if query.start_time and metric.timestamp < query.start_time:
                    continue
                if query.end_time and metric.timestamp > query.end_time:
                    continue

                # Group by tags
                group_key = self._build_group_key(metric, query.group_by)
                series_key = f"{name}:{group_key}"

                if series_key not in results:
                    results[series_key] = []
                results[series_key].append(metric)

        # Build series
        series_list = []
        for series_key, metrics in results.items():
            if not metrics:
                continue

            name = metrics[0].name
            tags = {t: metrics[0].tags.get(t, "") for t in query.group_by}

            # Aggregate by interval
            data_points = self._aggregate_metrics(metrics, query)

            series = MetricSeries(
                name=name,
                tags=tags,
                data_points=data_points[:query.limit],
                aggregation=query.aggregation,
                start_time=min(m.timestamp for m in metrics),
                end_time=max(m.timestamp for m in metrics),
            )
            series_list.append(series)

        return series_list

    def _match_pattern(self, name: str, pattern: str) -> bool:
        """Match metric name against pattern."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)

    def _build_group_key(self, metric: DeploymentMetric, group_by: List[str]) -> str:
        """Build group key from metric tags."""
        parts = []
        for tag in group_by:
            value = metric.tags.get(tag, "")
            parts.append(f"{tag}={value}")
        return ",".join(parts)

    def _aggregate_metrics(
        self,
        metrics: List[DeploymentMetric],
        query: MetricQuery,
    ) -> List[Tuple[float, float]]:
        """Aggregate metrics into time buckets."""
        if not metrics:
            return []

        # Sort by timestamp
        metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Group into buckets
        buckets: Dict[int, List[float]] = {}
        for metric in metrics:
            bucket = int(metric.timestamp // query.interval_s) * query.interval_s
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(metric.value)

        # Aggregate each bucket
        data_points = []
        for ts, values in sorted(buckets.items()):
            if query.aggregation == MetricAggregation.SUM:
                agg_value = sum(values)
            elif query.aggregation == MetricAggregation.AVG:
                agg_value = statistics.mean(values)
            elif query.aggregation == MetricAggregation.MIN:
                agg_value = min(values)
            elif query.aggregation == MetricAggregation.MAX:
                agg_value = max(values)
            elif query.aggregation == MetricAggregation.COUNT:
                agg_value = len(values)
            elif query.aggregation == MetricAggregation.P50:
                agg_value = statistics.median(values)
            elif query.aggregation == MetricAggregation.P90:
                agg_value = self._percentile(values, 90)
            elif query.aggregation == MetricAggregation.P95:
                agg_value = self._percentile(values, 95)
            elif query.aggregation == MetricAggregation.P99:
                agg_value = self._percentile(values, 99)
            elif query.aggregation == MetricAggregation.RATE:
                agg_value = sum(values) / query.interval_s
            else:
                agg_value = statistics.mean(values)

            data_points.append((float(ts), agg_value))

        return data_points

    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._alert_rules[rule.rule_id] = rule
        self._save_state()

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            self._save_state()
            return True
        return False

    def _check_alerts(self, metric: DeploymentMetric) -> None:
        """Check if metric triggers any alerts."""
        for rule in self._alert_rules.values():
            if not rule.enabled:
                continue
            if not self._match_pattern(metric.name, rule.metric_name):
                continue

            triggered = False
            if rule.condition == "gt" and metric.value > rule.threshold:
                triggered = True
            elif rule.condition == "gte" and metric.value >= rule.threshold:
                triggered = True
            elif rule.condition == "lt" and metric.value < rule.threshold:
                triggered = True
            elif rule.condition == "lte" and metric.value <= rule.threshold:
                triggered = True
            elif rule.condition == "eq" and metric.value == rule.threshold:
                triggered = True

            if triggered:
                self._trigger_alert(rule, metric)

    def _trigger_alert(self, rule: AlertRule, metric: DeploymentMetric) -> None:
        """Trigger an alert."""
        alert_key = f"{rule.rule_id}:{metric.service_name}"
        now = time.time()

        # Check if already in alert state
        if alert_key in self._alert_state:
            state = self._alert_state[alert_key]
            if now - state["first_triggered"] < rule.duration_s:
                return  # Still within duration window
            if state.get("notified"):
                return  # Already notified

        # Record alert state
        self._alert_state[alert_key] = {
            "first_triggered": now,
            "rule_id": rule.rule_id,
            "metric_name": metric.name,
            "value": metric.value,
            "notified": True,
        }

        # Emit alert event
        _emit_bus_event(
            self.BUS_TOPICS["alert"],
            {
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "metric_name": metric.name,
                "value": metric.value,
                "threshold": rule.threshold,
                "condition": rule.condition,
                "severity": rule.severity.value,
                "service_name": metric.service_name,
            },
            kind="alert",
            level=rule.severity.value,
            actor=self.actor_id,
        )

    def register_exporter(
        self,
        name: str,
        exporter: Callable[[List[DeploymentMetric]], None],
    ) -> None:
        """Register a metric exporter."""
        self._exporters[name] = exporter

    async def export(self, exporter_name: Optional[str] = None) -> int:
        """
        Export metrics to registered exporters.

        Args:
            exporter_name: Specific exporter to use, or None for all

        Returns:
            Number of metrics exported
        """
        all_metrics = []
        for metrics in self._metrics.values():
            all_metrics.extend(metrics)

        if not all_metrics:
            return 0

        exporters = (
            {exporter_name: self._exporters[exporter_name]}
            if exporter_name and exporter_name in self._exporters
            else self._exporters
        )

        for name, exporter in exporters.items():
            try:
                if asyncio.iscoroutinefunction(exporter):
                    await exporter(all_metrics)
                else:
                    exporter(all_metrics)
            except Exception as e:
                _emit_bus_event(
                    self.BUS_TOPICS["export"],
                    {"exporter": name, "error": str(e), "count": len(all_metrics)},
                    level="error",
                    actor=self.actor_id,
                )

        _emit_bus_event(
            self.BUS_TOPICS["export"],
            {"count": len(all_metrics), "exporters": list(exporters.keys())},
            actor=self.actor_id,
        )

        return len(all_metrics)

    def get_summary(
        self,
        service_name: Optional[str] = None,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get deployment metrics summary.

        Args:
            service_name: Filter by service
            hours: Time window in hours

        Returns:
            Summary dictionary
        """
        cutoff = time.time() - (hours * 3600)
        summary: Dict[str, Any] = {
            "period_hours": hours,
            "service_name": service_name,
            "metrics": {},
        }

        for name, metrics in self._metrics.items():
            filtered = [
                m for m in metrics
                if m.timestamp >= cutoff
                and (not service_name or m.service_name == service_name)
            ]

            if not filtered:
                continue

            values = [m.value for m in filtered]
            summary["metrics"][name] = {
                "count": len(values),
                "sum": sum(values),
                "avg": statistics.mean(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "p50": statistics.median(values) if values else 0,
                "p95": self._percentile(values, 95),
            }

        return summary

    def prune_old_metrics(self) -> int:
        """Remove metrics older than retention period."""
        cutoff = time.time() - (self.retention_hours * 3600)
        pruned = 0

        for name in list(self._metrics.keys()):
            before = len(self._metrics[name])
            self._metrics[name] = [
                m for m in self._metrics[name] if m.timestamp >= cutoff
            ]
            pruned += before - len(self._metrics[name])

            if not self._metrics[name]:
                del self._metrics[name]

        return pruned

    def list_metrics(self) -> List[str]:
        """List all metric names."""
        return list(self._metrics.keys())

    def list_alert_rules(self) -> List[AlertRule]:
        """List all alert rules."""
        return list(self._alert_rules.values())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "alert_rules": {rid: r.to_dict() for rid, r in self._alert_rules.items()},
            "counters": self._counters,
        }
        state_file = self.state_dir / "metrics_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "metrics_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for rid, data in state.get("alert_rules", {}).items():
                data["severity"] = AlertSeverity(data.get("severity", "warning"))
                self._alert_rules[rid] = AlertRule(**data)

            self._counters = state.get("counters", {})
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for metrics collector."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment Metrics Collector (Step 221)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # record command
    record_parser = subparsers.add_parser("record", help="Record a metric")
    record_parser.add_argument("name", help="Metric name")
    record_parser.add_argument("value", type=float, help="Metric value")
    record_parser.add_argument("--type", "-t", default="gauge",
                               choices=["counter", "gauge", "timer"])
    record_parser.add_argument("--service", "-s", default="", help="Service name")
    record_parser.add_argument("--deployment", "-d", default="", help="Deployment ID")
    record_parser.add_argument("--json", action="store_true", help="JSON output")

    # query command
    query_parser = subparsers.add_parser("query", help="Query metrics")
    query_parser.add_argument("--name", "-n", default="*", help="Metric name pattern")
    query_parser.add_argument("--service", "-s", help="Filter by service")
    query_parser.add_argument("--hours", type=int, default=24, help="Time window")
    query_parser.add_argument("--aggregation", "-a", default="avg",
                              choices=["sum", "avg", "min", "max", "count", "p95"])
    query_parser.add_argument("--json", action="store_true", help="JSON output")

    # summary command
    summary_parser = subparsers.add_parser("summary", help="Get metrics summary")
    summary_parser.add_argument("--service", "-s", help="Filter by service")
    summary_parser.add_argument("--hours", type=int, default=24, help="Time window")
    summary_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List metrics")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # prune command
    prune_parser = subparsers.add_parser("prune", help="Prune old metrics")
    prune_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    collector = DeploymentMetricsCollector()

    if args.command == "record":
        metric = collector.record(
            name=args.name,
            value=args.value,
            metric_type=MetricType(args.type.upper()),
            service_name=args.service,
            deployment_id=args.deployment,
        )

        if args.json:
            print(json.dumps(metric.to_dict(), indent=2))
        else:
            print(f"Recorded: {metric.name} = {metric.value}")

        return 0

    elif args.command == "query":
        start_time = time.time() - (args.hours * 3600)
        query = MetricQuery(
            name=args.name,
            service_name=args.service,
            start_time=start_time,
            aggregation=MetricAggregation(args.aggregation.upper()),
        )

        series = collector.query(query)

        if args.json:
            print(json.dumps([s.to_dict() for s in series], indent=2))
        else:
            for s in series:
                print(f"{s.name}: {len(s.data_points)} data points")
                if s.data_points:
                    latest = s.data_points[-1]
                    print(f"  Latest: {latest[1]:.2f} at {latest[0]}")

        return 0

    elif args.command == "summary":
        summary = collector.get_summary(
            service_name=args.service,
            hours=args.hours,
        )

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Metrics Summary ({args.hours}h)")
            for name, stats in summary.get("metrics", {}).items():
                print(f"  {name}:")
                print(f"    count={stats['count']}, avg={stats['avg']:.2f}, p95={stats['p95']:.2f}")

        return 0

    elif args.command == "list":
        metrics = collector.list_metrics()

        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            for m in metrics:
                print(m)

        return 0

    elif args.command == "prune":
        pruned = collector.prune_old_metrics()

        if args.json:
            print(json.dumps({"pruned": pruned}))
        else:
            print(f"Pruned {pruned} old metrics")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
