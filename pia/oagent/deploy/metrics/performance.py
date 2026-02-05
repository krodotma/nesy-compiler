#!/usr/bin/env python3
"""
performance.py - Deploy Metrics (Step 233)

PBTSO Phase: VERIFY, ITERATE
A2A Integration: Performance and usage metrics via deploy.metrics.performance.*

Provides:
- MetricType: Types of deployment metrics
- MetricSampler: Metric sampling methods
- PerformanceMetric: Performance metric data
- UsageMetric: Usage tracking metric
- MetricAggregation: Aggregated metrics
- DeployMetricsSystem: Main metrics system

Bus Topics:
- deploy.metrics.performance.record
- deploy.metrics.performance.aggregate
- deploy.metrics.usage.record
- deploy.metrics.alert

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
from typing import Any, Callable, Dict, List, Optional, Tuple


# ==============================================================================
# Bus Emission Helper with File Locking (DKIN v30)
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
    actor: str = "metrics-system"
) -> str:
    """Emit an event to the Pluribus bus with fcntl.flock() file locking."""
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
    PERCENTAGE = "percentage"


class MetricSampler(Enum):
    """Metric sampling methods."""
    ALL = "all"
    RANDOM = "random"
    RESERVOIR = "reservoir"
    EXPONENTIAL = "exponential"


class MetricCategory(Enum):
    """Metric categories."""
    DEPLOYMENT = "deployment"
    BUILD = "build"
    CONTAINER = "container"
    HEALTH = "health"
    TRAFFIC = "traffic"
    ROLLBACK = "rollback"
    LATENCY = "latency"
    ERROR = "error"
    RESOURCE = "resource"
    CUSTOM = "custom"


class AggregationType(Enum):
    """Aggregation types."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    RATE_PER_SECOND = "rate_per_second"
    RATE_PER_MINUTE = "rate_per_minute"


@dataclass
class PerformanceMetric:
    """
    Performance metric data point.

    Attributes:
        metric_id: Unique metric identifier
        name: Metric name
        metric_type: Type of metric
        category: Metric category
        value: Metric value
        unit: Measurement unit
        tags: Metric tags/dimensions
        timestamp: Metric timestamp
        service_name: Associated service
        deployment_id: Associated deployment
        environment: Target environment
    """
    metric_id: str
    name: str
    metric_type: MetricType = MetricType.GAUGE
    category: MetricCategory = MetricCategory.CUSTOM
    value: float = 0.0
    unit: str = ""
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
            "category": self.category.value,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "service_name": self.service_name,
            "deployment_id": self.deployment_id,
            "environment": self.environment,
        }


@dataclass
class UsageMetric:
    """
    Usage tracking metric.

    Attributes:
        metric_id: Unique metric identifier
        resource: Resource being tracked
        operation: Operation performed
        count: Operation count
        duration_ms: Total duration in ms
        error_count: Number of errors
        tags: Metric tags
        timestamp: Metric timestamp
        user_id: User/actor identifier
        service_name: Service name
    """
    metric_id: str
    resource: str
    operation: str
    count: int = 1
    duration_ms: float = 0.0
    error_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    service_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "resource": self.resource,
            "operation": self.operation,
            "count": self.count,
            "duration_ms": self.duration_ms,
            "error_count": self.error_count,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "service_name": self.service_name,
        }


@dataclass
class MetricAggregation:
    """
    Aggregated metric data.

    Attributes:
        name: Metric name
        aggregation: Aggregation type used
        value: Aggregated value
        count: Number of samples
        min_value: Minimum value
        max_value: Maximum value
        start_time: Aggregation window start
        end_time: Aggregation window end
        tags: Aggregation dimensions
    """
    name: str
    aggregation: AggregationType
    value: float
    count: int = 0
    min_value: float = 0.0
    max_value: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aggregation": self.aggregation.value,
            "value": self.value,
            "count": self.count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tags": self.tags,
        }


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    name: str
    metric_name: str
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    severity: str = "warning"  # info, warning, critical
    duration_s: int = 60
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Timer Context Manager
# ==============================================================================

class Timer:
    """Context manager for timing operations."""

    def __init__(
        self,
        metrics_system: "DeployMetricsSystem",
        name: str,
        category: MetricCategory = MetricCategory.LATENCY,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.metrics_system = metrics_system
        self.name = name
        self.category = category
        self.tags = tags or {}
        self.start_time = 0.0
        self.duration_ms = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration_ms = (time.time() - self.start_time) * 1000
        self.metrics_system.record_timer(
            name=self.name,
            duration_ms=self.duration_ms,
            category=self.category,
            tags=self.tags,
        )


# ==============================================================================
# Deploy Metrics System (Step 233)
# ==============================================================================

class DeployMetricsSystem:
    """
    Deploy Metrics System - performance and usage metrics.

    PBTSO Phase: VERIFY, ITERATE

    Responsibilities:
    - Record performance metrics (latency, throughput)
    - Track usage metrics (operations, resources)
    - Aggregate metrics over time windows
    - Generate alerts based on thresholds
    - Export metrics to external systems

    Example:
        >>> metrics = DeployMetricsSystem()
        >>> metrics.record_gauge("deployment.active_count", 5)
        >>> metrics.increment("deployment.total_count")
        >>> with metrics.timer("deployment.duration"):
        ...     # perform deployment
        ...     pass
        >>> summary = metrics.get_summary()
    """

    BUS_TOPICS = {
        "perf_record": "deploy.metrics.performance.record",
        "perf_aggregate": "deploy.metrics.performance.aggregate",
        "usage_record": "deploy.metrics.usage.record",
        "alert": "deploy.metrics.alert",
    }

    # Standard performance metrics
    STANDARD_METRICS = {
        "deployment.duration_ms": (MetricType.TIMER, MetricCategory.DEPLOYMENT),
        "deployment.count": (MetricType.COUNTER, MetricCategory.DEPLOYMENT),
        "deployment.success_rate": (MetricType.PERCENTAGE, MetricCategory.DEPLOYMENT),
        "deployment.failure_rate": (MetricType.PERCENTAGE, MetricCategory.DEPLOYMENT),
        "deployment.rollback_rate": (MetricType.PERCENTAGE, MetricCategory.ROLLBACK),
        "build.duration_ms": (MetricType.TIMER, MetricCategory.BUILD),
        "build.success_rate": (MetricType.PERCENTAGE, MetricCategory.BUILD),
        "container.build_time_ms": (MetricType.TIMER, MetricCategory.CONTAINER),
        "health.check_latency_ms": (MetricType.TIMER, MetricCategory.HEALTH),
        "health.success_rate": (MetricType.PERCENTAGE, MetricCategory.HEALTH),
        "traffic.requests_per_second": (MetricType.RATE, MetricCategory.TRAFFIC),
        "error.rate": (MetricType.RATE, MetricCategory.ERROR),
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "metrics-system",
        retention_hours: int = 168,  # 7 days
        aggregation_interval_s: int = 60,
    ):
        """
        Initialize the metrics system.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            retention_hours: Hours to retain metrics
            aggregation_interval_s: Interval for auto-aggregation
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "metrics"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.retention_hours = retention_hours
        self.aggregation_interval_s = aggregation_interval_s

        # In-memory storage
        self._performance_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self._usage_metrics: Dict[str, List[UsageMetric]] = defaultdict(list)
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)

        # Alert thresholds
        self._thresholds: Dict[str, AlertThreshold] = {}
        self._alert_state: Dict[str, Dict[str, Any]] = {}

        # Custom exporters
        self._exporters: List[Callable[[List[PerformanceMetric]], None]] = []

        self._load_state()

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        category: MetricCategory = MetricCategory.CUSTOM,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        service_name: str = "",
        deployment_id: str = "",
        environment: str = "",
    ) -> PerformanceMetric:
        """
        Record a performance metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            category: Metric category
            unit: Measurement unit
            tags: Metric tags/dimensions
            service_name: Associated service
            deployment_id: Associated deployment
            environment: Target environment

        Returns:
            Recorded PerformanceMetric
        """
        metric = PerformanceMetric(
            metric_id=f"metric-{uuid.uuid4().hex[:12]}",
            name=name,
            metric_type=metric_type,
            category=category,
            value=value,
            unit=unit,
            tags=tags or {},
            service_name=service_name,
            deployment_id=deployment_id,
            environment=environment,
        )

        self._performance_metrics[name].append(metric)

        # Update in-memory aggregates
        if metric_type == MetricType.COUNTER:
            self._counters[name] = self._counters.get(name, 0) + value
        elif metric_type == MetricType.GAUGE:
            self._gauges[name] = value
        elif metric_type in (MetricType.HISTOGRAM, MetricType.TIMER):
            self._histograms[name].append(value)

        # Check thresholds
        self._check_thresholds(metric)

        # Emit bus event
        _emit_bus_event(
            self.BUS_TOPICS["perf_record"],
            {
                "name": name,
                "value": value,
                "metric_type": metric_type.value,
                "category": category.value,
                "service_name": service_name,
            },
            kind="metric",
            actor=self.actor_id,
        )

        return metric

    def record_gauge(
        self,
        name: str,
        value: float,
        category: MetricCategory = MetricCategory.CUSTOM,
        **kwargs,
    ) -> PerformanceMetric:
        """Record a gauge metric."""
        return self.record(name, value, MetricType.GAUGE, category, **kwargs)

    def increment(
        self,
        name: str,
        delta: float = 1.0,
        category: MetricCategory = MetricCategory.CUSTOM,
        **kwargs,
    ) -> PerformanceMetric:
        """Increment a counter metric."""
        return self.record(name, delta, MetricType.COUNTER, category, **kwargs)

    def record_timer(
        self,
        name: str,
        duration_ms: float,
        category: MetricCategory = MetricCategory.LATENCY,
        **kwargs,
    ) -> PerformanceMetric:
        """Record a timer/duration metric."""
        return self.record(name, duration_ms, MetricType.TIMER, category, unit="ms", **kwargs)

    def record_histogram(
        self,
        name: str,
        value: float,
        category: MetricCategory = MetricCategory.CUSTOM,
        **kwargs,
    ) -> PerformanceMetric:
        """Record a histogram metric."""
        return self.record(name, value, MetricType.HISTOGRAM, category, **kwargs)

    def record_percentage(
        self,
        name: str,
        value: float,
        category: MetricCategory = MetricCategory.CUSTOM,
        **kwargs,
    ) -> PerformanceMetric:
        """Record a percentage metric (0-100)."""
        return self.record(name, value, MetricType.PERCENTAGE, category, unit="%", **kwargs)

    def timer(
        self,
        name: str,
        category: MetricCategory = MetricCategory.LATENCY,
        tags: Optional[Dict[str, str]] = None,
    ) -> Timer:
        """Create a timer context manager."""
        return Timer(self, name, category, tags)

    def record_usage(
        self,
        resource: str,
        operation: str,
        count: int = 1,
        duration_ms: float = 0.0,
        error_count: int = 0,
        tags: Optional[Dict[str, str]] = None,
        user_id: str = "",
        service_name: str = "",
    ) -> UsageMetric:
        """
        Record a usage metric.

        Args:
            resource: Resource being tracked
            operation: Operation performed
            count: Operation count
            duration_ms: Total duration
            error_count: Number of errors
            tags: Metric tags
            user_id: User/actor identifier
            service_name: Service name

        Returns:
            Recorded UsageMetric
        """
        metric = UsageMetric(
            metric_id=f"usage-{uuid.uuid4().hex[:12]}",
            resource=resource,
            operation=operation,
            count=count,
            duration_ms=duration_ms,
            error_count=error_count,
            tags=tags or {},
            user_id=user_id,
            service_name=service_name,
        )

        key = f"{resource}:{operation}"
        self._usage_metrics[key].append(metric)

        _emit_bus_event(
            self.BUS_TOPICS["usage_record"],
            {
                "resource": resource,
                "operation": operation,
                "count": count,
                "duration_ms": duration_ms,
            },
            kind="metric",
            actor=self.actor_id,
        )

        return metric

    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self._gauges.get(name, 0.0)

    def aggregate(
        self,
        name: str,
        aggregation: AggregationType,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricAggregation:
        """
        Aggregate metrics over a time window.

        Args:
            name: Metric name
            aggregation: Aggregation type
            start_time: Window start time
            end_time: Window end time
            tags: Filter by tags

        Returns:
            MetricAggregation
        """
        metrics = self._performance_metrics.get(name, [])

        # Apply time filter
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]

        # Apply tag filter
        if tags:
            metrics = [
                m for m in metrics
                if all(m.tags.get(k) == v for k, v in tags.items())
            ]

        if not metrics:
            return MetricAggregation(
                name=name,
                aggregation=aggregation,
                value=0.0,
                count=0,
                tags=tags or {},
            )

        values = [m.value for m in metrics]

        if aggregation == AggregationType.SUM:
            agg_value = sum(values)
        elif aggregation == AggregationType.AVG:
            agg_value = statistics.mean(values)
        elif aggregation == AggregationType.MIN:
            agg_value = min(values)
        elif aggregation == AggregationType.MAX:
            agg_value = max(values)
        elif aggregation == AggregationType.COUNT:
            agg_value = float(len(values))
        elif aggregation == AggregationType.P50:
            agg_value = statistics.median(values)
        elif aggregation == AggregationType.P90:
            agg_value = self._percentile(values, 90)
        elif aggregation == AggregationType.P95:
            agg_value = self._percentile(values, 95)
        elif aggregation == AggregationType.P99:
            agg_value = self._percentile(values, 99)
        elif aggregation == AggregationType.RATE_PER_SECOND:
            duration = (end_time or time.time()) - (start_time or metrics[0].timestamp)
            agg_value = sum(values) / duration if duration > 0 else 0.0
        elif aggregation == AggregationType.RATE_PER_MINUTE:
            duration = (end_time or time.time()) - (start_time or metrics[0].timestamp)
            agg_value = (sum(values) / duration) * 60 if duration > 0 else 0.0
        else:
            agg_value = statistics.mean(values)

        result = MetricAggregation(
            name=name,
            aggregation=aggregation,
            value=agg_value,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            start_time=min(m.timestamp for m in metrics),
            end_time=max(m.timestamp for m in metrics),
            tags=tags or {},
        )

        _emit_bus_event(
            self.BUS_TOPICS["perf_aggregate"],
            {
                "name": name,
                "aggregation": aggregation.value,
                "value": agg_value,
                "count": len(values),
            },
            kind="metric",
            actor=self.actor_id,
        )

        return result

    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def add_threshold(self, threshold: AlertThreshold) -> None:
        """Add an alert threshold."""
        self._thresholds[threshold.name] = threshold
        self._save_state()

    def remove_threshold(self, name: str) -> bool:
        """Remove an alert threshold."""
        if name in self._thresholds:
            del self._thresholds[name]
            self._save_state()
            return True
        return False

    def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric triggers any thresholds."""
        for threshold in self._thresholds.values():
            if not threshold.enabled:
                continue
            if not self._matches_pattern(metric.name, threshold.metric_name):
                continue

            triggered = False
            if threshold.condition == "gt" and metric.value > threshold.threshold:
                triggered = True
            elif threshold.condition == "gte" and metric.value >= threshold.threshold:
                triggered = True
            elif threshold.condition == "lt" and metric.value < threshold.threshold:
                triggered = True
            elif threshold.condition == "lte" and metric.value <= threshold.threshold:
                triggered = True
            elif threshold.condition == "eq" and metric.value == threshold.threshold:
                triggered = True

            if triggered:
                self._trigger_alert(threshold, metric)

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Match metric name against pattern."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)

    def _trigger_alert(self, threshold: AlertThreshold, metric: PerformanceMetric) -> None:
        """Trigger an alert."""
        alert_key = f"{threshold.name}:{metric.service_name}"
        now = time.time()

        if alert_key in self._alert_state:
            state = self._alert_state[alert_key]
            if now - state["first_triggered"] < threshold.duration_s:
                return
            if state.get("notified"):
                return

        self._alert_state[alert_key] = {
            "first_triggered": now,
            "threshold_name": threshold.name,
            "metric_name": metric.name,
            "value": metric.value,
            "notified": True,
        }

        _emit_bus_event(
            self.BUS_TOPICS["alert"],
            {
                "threshold_name": threshold.name,
                "metric_name": metric.name,
                "value": metric.value,
                "threshold_value": threshold.threshold,
                "condition": threshold.condition,
                "severity": threshold.severity,
                "service_name": metric.service_name,
            },
            kind="alert",
            level=threshold.severity,
            actor=self.actor_id,
        )

    def get_summary(
        self,
        service_name: Optional[str] = None,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get metrics summary.

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
            "performance": {},
            "usage": {},
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }

        # Performance metrics summary
        for name, metrics in self._performance_metrics.items():
            filtered = [
                m for m in metrics
                if m.timestamp >= cutoff
                and (not service_name or m.service_name == service_name)
            ]

            if not filtered:
                continue

            values = [m.value for m in filtered]
            summary["performance"][name] = {
                "count": len(values),
                "sum": sum(values),
                "avg": statistics.mean(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "p50": statistics.median(values) if values else 0,
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
            }

        # Usage metrics summary
        for key, metrics in self._usage_metrics.items():
            filtered = [
                m for m in metrics
                if m.timestamp >= cutoff
                and (not service_name or m.service_name == service_name)
            ]

            if not filtered:
                continue

            summary["usage"][key] = {
                "total_count": sum(m.count for m in filtered),
                "total_duration_ms": sum(m.duration_ms for m in filtered),
                "total_errors": sum(m.error_count for m in filtered),
                "operations": len(filtered),
            }

        return summary

    def prune_old_metrics(self) -> int:
        """Remove metrics older than retention period."""
        cutoff = time.time() - (self.retention_hours * 3600)
        pruned = 0

        for name in list(self._performance_metrics.keys()):
            before = len(self._performance_metrics[name])
            self._performance_metrics[name] = [
                m for m in self._performance_metrics[name] if m.timestamp >= cutoff
            ]
            pruned += before - len(self._performance_metrics[name])

            if not self._performance_metrics[name]:
                del self._performance_metrics[name]

        for key in list(self._usage_metrics.keys()):
            before = len(self._usage_metrics[key])
            self._usage_metrics[key] = [
                m for m in self._usage_metrics[key] if m.timestamp >= cutoff
            ]
            pruned += before - len(self._usage_metrics[key])

            if not self._usage_metrics[key]:
                del self._usage_metrics[key]

        # Trim histograms
        max_histogram_size = 10000
        for name in self._histograms:
            if len(self._histograms[name]) > max_histogram_size:
                self._histograms[name] = self._histograms[name][-max_histogram_size:]

        return pruned

    def register_exporter(
        self,
        exporter: Callable[[List[PerformanceMetric]], None],
    ) -> None:
        """Register a metric exporter."""
        self._exporters.append(exporter)

    async def export(self) -> int:
        """Export metrics to registered exporters."""
        all_metrics = []
        for metrics in self._performance_metrics.values():
            all_metrics.extend(metrics)

        if not all_metrics:
            return 0

        for exporter in self._exporters:
            try:
                if asyncio.iscoroutinefunction(exporter):
                    await exporter(all_metrics)
                else:
                    exporter(all_metrics)
            except Exception:
                pass

        return len(all_metrics)

    def list_metrics(self) -> List[str]:
        """List all metric names."""
        return list(self._performance_metrics.keys())

    def list_thresholds(self) -> List[AlertThreshold]:
        """List all alert thresholds."""
        return list(self._thresholds.values())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "thresholds": {n: t.to_dict() for n, t in self._thresholds.items()},
            "counters": self._counters,
            "gauges": self._gauges,
        }
        state_file = self.state_dir / "metrics_performance_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "metrics_performance_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for name, data in state.get("thresholds", {}).items():
                self._thresholds[name] = AlertThreshold(**data)

            self._counters = state.get("counters", {})
            self._gauges = state.get("gauges", {})
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for metrics system."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Metrics System (Step 233)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # record command
    record_parser = subparsers.add_parser("record", help="Record a metric")
    record_parser.add_argument("name", help="Metric name")
    record_parser.add_argument("value", type=float, help="Metric value")
    record_parser.add_argument("--type", "-t", default="gauge",
                               choices=["counter", "gauge", "timer", "histogram"])
    record_parser.add_argument("--category", "-c", default="custom")
    record_parser.add_argument("--service", "-s", default="", help="Service name")
    record_parser.add_argument("--json", action="store_true", help="JSON output")

    # summary command
    summary_parser = subparsers.add_parser("summary", help="Get metrics summary")
    summary_parser.add_argument("--service", "-s", help="Filter by service")
    summary_parser.add_argument("--hours", type=int, default=24, help="Time window")
    summary_parser.add_argument("--json", action="store_true", help="JSON output")

    # aggregate command
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate metrics")
    agg_parser.add_argument("name", help="Metric name")
    agg_parser.add_argument("--aggregation", "-a", default="avg",
                            choices=["sum", "avg", "min", "max", "count", "p50", "p95", "p99"])
    agg_parser.add_argument("--hours", type=int, default=24, help="Time window")
    agg_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List metrics")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # threshold command
    threshold_parser = subparsers.add_parser("threshold", help="Manage alert thresholds")
    threshold_parser.add_argument("action", choices=["add", "remove", "list"])
    threshold_parser.add_argument("--name", "-n", help="Threshold name")
    threshold_parser.add_argument("--metric", "-m", help="Metric name pattern")
    threshold_parser.add_argument("--condition", "-c", choices=["gt", "lt", "eq", "gte", "lte"])
    threshold_parser.add_argument("--value", "-v", type=float, help="Threshold value")
    threshold_parser.add_argument("--severity", default="warning", choices=["info", "warning", "critical"])
    threshold_parser.add_argument("--json", action="store_true", help="JSON output")

    # prune command
    prune_parser = subparsers.add_parser("prune", help="Prune old metrics")
    prune_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    metrics = DeployMetricsSystem()

    if args.command == "record":
        metric = metrics.record(
            name=args.name,
            value=args.value,
            metric_type=MetricType(args.type.upper()),
            category=MetricCategory(args.category.upper()) if args.category != "custom" else MetricCategory.CUSTOM,
            service_name=args.service,
        )

        if args.json:
            print(json.dumps(metric.to_dict(), indent=2))
        else:
            print(f"Recorded: {metric.name} = {metric.value}")

        return 0

    elif args.command == "summary":
        summary = metrics.get_summary(
            service_name=args.service,
            hours=args.hours,
        )

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Metrics Summary ({args.hours}h)")
            print("\nPerformance Metrics:")
            for name, stats in summary.get("performance", {}).items():
                print(f"  {name}: count={stats['count']}, avg={stats['avg']:.2f}, p95={stats['p95']:.2f}")
            print("\nCounters:")
            for name, value in summary.get("counters", {}).items():
                print(f"  {name}: {value}")
            print("\nGauges:")
            for name, value in summary.get("gauges", {}).items():
                print(f"  {name}: {value}")

        return 0

    elif args.command == "aggregate":
        start_time = time.time() - (args.hours * 3600)
        agg = metrics.aggregate(
            args.name,
            AggregationType(args.aggregation.upper()),
            start_time=start_time,
        )

        if args.json:
            print(json.dumps(agg.to_dict(), indent=2))
        else:
            print(f"{args.name} ({args.aggregation}): {agg.value:.4f}")
            print(f"  Count: {agg.count}")
            print(f"  Min: {agg.min_value:.4f}, Max: {agg.max_value:.4f}")

        return 0

    elif args.command == "list":
        metric_names = metrics.list_metrics()

        if args.json:
            print(json.dumps(metric_names, indent=2))
        else:
            for name in metric_names:
                print(name)

        return 0

    elif args.command == "threshold":
        if args.action == "add":
            if not all([args.name, args.metric, args.condition, args.value]):
                print("Required: --name, --metric, --condition, --value")
                return 1

            threshold = AlertThreshold(
                name=args.name,
                metric_name=args.metric,
                condition=args.condition,
                threshold=args.value,
                severity=args.severity,
            )
            metrics.add_threshold(threshold)
            print(f"Added threshold: {args.name}")

        elif args.action == "remove":
            if not args.name:
                print("Required: --name")
                return 1
            success = metrics.remove_threshold(args.name)
            if success:
                print(f"Removed threshold: {args.name}")
            else:
                print(f"Threshold not found: {args.name}")
                return 1

        elif args.action == "list":
            thresholds = metrics.list_thresholds()
            if args.json:
                print(json.dumps([t.to_dict() for t in thresholds], indent=2))
            else:
                for t in thresholds:
                    status = "enabled" if t.enabled else "disabled"
                    print(f"{t.name}: {t.metric_name} {t.condition} {t.threshold} [{status}]")

        return 0

    elif args.command == "prune":
        count = metrics.prune_old_metrics()

        if args.json:
            print(json.dumps({"pruned": count}))
        else:
            print(f"Pruned {count} old metrics")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
