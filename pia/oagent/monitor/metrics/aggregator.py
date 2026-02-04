#!/usr/bin/env python3
"""
Metric Aggregator - Step 253

Aggregates metrics with statistical rollups and windowed calculations.

PBTSO Phase: DISTILL

Bus Topics:
- monitor.metrics.aggregate (subscribed)
- monitor.metrics.aggregated (emitted)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .collector import MetricCollector, MetricPoint, get_collector


class AggregationType(Enum):
    """Types of aggregations supported."""
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    STDDEV = "stddev"
    VARIANCE = "variance"


@dataclass
class AggregatedMetric:
    """Result of metric aggregation.

    Attributes:
        name: Original metric name
        aggregation: Type of aggregation performed
        value: Aggregated value
        window_s: Time window in seconds
        labels: Label filter used
        timestamp: Time of aggregation
        sample_count: Number of samples in window
        metadata: Additional metadata
    """
    name: str
    aggregation: AggregationType
    value: float
    window_s: int
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sample_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "aggregation": self.aggregation.value,
            "value": self.value,
            "window_s": self.window_s,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "sample_count": self.sample_count,
            "metadata": self.metadata,
        }


@dataclass
class AggregationRule:
    """Rule for automatic metric aggregation.

    Attributes:
        metric_pattern: Pattern to match metric names (supports * wildcard)
        aggregations: List of aggregations to compute
        window_s: Time window
        interval_s: How often to run aggregation
        labels: Label filter
    """
    metric_pattern: str
    aggregations: List[AggregationType] = field(default_factory=lambda: [AggregationType.AVG])
    window_s: int = 300
    interval_s: int = 60
    labels: Dict[str, str] = field(default_factory=dict)


class MetricAggregator:
    """
    Aggregate metrics with statistical calculations.

    The aggregator:
    - Computes statistical aggregations (avg, sum, percentiles, stddev)
    - Supports multiple time windows
    - Can run periodic rollups
    - Emits aggregated metrics to bus

    Example:
        aggregator = MetricAggregator()

        # Single aggregation
        result = aggregator.aggregate(
            "agent.latency_ms",
            AggregationType.P95,
            window_s=300
        )

        # Multiple aggregations
        results = aggregator.aggregate_multi(
            "agent.latency_ms",
            [AggregationType.AVG, AggregationType.P95, AggregationType.MAX],
            window_s=300
        )
    """

    BUS_TOPICS = {
        "aggregate": "monitor.metrics.aggregate",
        "aggregated": "monitor.metrics.aggregated",
    }

    def __init__(
        self,
        collector: Optional[MetricCollector] = None,
        bus_dir: Optional[str] = None
    ):
        """Initialize metric aggregator.

        Args:
            collector: Metric collector to aggregate from
            bus_dir: Directory for bus events
        """
        self._collector = collector or get_collector()
        self._rules: List[AggregationRule] = []
        self._last_run: Dict[str, float] = {}
        self._aggregation_cache: Dict[str, Tuple[float, AggregatedMetric]] = {}
        self._cache_ttl_s = 10

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def aggregate(
        self,
        metric_name: str,
        aggregation: AggregationType,
        window_s: int = 300,
        labels: Optional[Dict[str, str]] = None,
        use_cache: bool = True
    ) -> AggregatedMetric:
        """Aggregate a single metric.

        Args:
            metric_name: Metric to aggregate
            aggregation: Aggregation type
            window_s: Time window in seconds
            labels: Optional label filter
            use_cache: Whether to use cached results

        Returns:
            Aggregated metric result
        """
        cache_key = f"{metric_name}:{aggregation.value}:{window_s}:{json.dumps(labels or {})}"

        # Check cache
        if use_cache and cache_key in self._aggregation_cache:
            cached_ts, cached_result = self._aggregation_cache[cache_key]
            if time.time() - cached_ts < self._cache_ttl_s:
                return cached_result

        # Get raw points
        points = self._collector.query_series(metric_name, window_s, labels)

        # Compute aggregation
        value = self._compute_aggregation(points, aggregation)

        result = AggregatedMetric(
            name=metric_name,
            aggregation=aggregation,
            value=value,
            window_s=window_s,
            labels=labels or {},
            sample_count=len(points),
            metadata={
                "collector_cardinality": self._collector.get_cardinality(),
            }
        )

        # Cache result
        self._aggregation_cache[cache_key] = (time.time(), result)

        return result

    def aggregate_multi(
        self,
        metric_name: str,
        aggregations: List[AggregationType],
        window_s: int = 300,
        labels: Optional[Dict[str, str]] = None
    ) -> List[AggregatedMetric]:
        """Aggregate a metric with multiple aggregation types.

        Args:
            metric_name: Metric to aggregate
            aggregations: List of aggregation types
            window_s: Time window
            labels: Optional label filter

        Returns:
            List of aggregated metrics
        """
        # Get points once, compute multiple aggregations
        points = self._collector.query_series(metric_name, window_s, labels)

        results = []
        for agg_type in aggregations:
            value = self._compute_aggregation(points, agg_type)
            results.append(AggregatedMetric(
                name=metric_name,
                aggregation=agg_type,
                value=value,
                window_s=window_s,
                labels=labels or {},
                sample_count=len(points),
            ))

        return results

    def compute_percentile(
        self,
        metric_name: str,
        percentile: float,
        window_s: int = 300,
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Compute a specific percentile.

        Args:
            metric_name: Metric name
            percentile: Percentile (0-100)
            window_s: Time window
            labels: Label filter

        Returns:
            Percentile value
        """
        points = self._collector.query_series(metric_name, window_s, labels)
        return self._percentile(points, percentile)

    def compute_statistics(
        self,
        metric_name: str,
        window_s: int = 300,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Compute comprehensive statistics for a metric.

        Args:
            metric_name: Metric name
            window_s: Time window
            labels: Label filter

        Returns:
            Dictionary of statistics
        """
        points = self._collector.query_series(metric_name, window_s, labels)

        if not points:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "sum": 0.0,
                "stddev": 0.0,
                "variance": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        values = [p.value for p in points]
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
        stddev = math.sqrt(variance)

        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "avg": mean,
            "sum": sum(values),
            "stddev": stddev,
            "variance": variance,
            "p50": self._percentile(points, 50),
            "p90": self._percentile(points, 90),
            "p95": self._percentile(points, 95),
            "p99": self._percentile(points, 99),
        }

    def add_rule(self, rule: AggregationRule) -> None:
        """Add an aggregation rule.

        Args:
            rule: Aggregation rule to add
        """
        self._rules.append(rule)

    def run_rules(self) -> List[AggregatedMetric]:
        """Run all aggregation rules.

        Returns:
            List of aggregated metrics
        """
        results: List[AggregatedMetric] = []
        now = time.time()

        for rule in self._rules:
            rule_key = f"{rule.metric_pattern}:{rule.window_s}"
            last_run = self._last_run.get(rule_key, 0)

            # Check if enough time has passed
            if now - last_run < rule.interval_s:
                continue

            # Find matching metrics
            all_metrics = self._collector.list_metrics()
            matching = [m for m in all_metrics if self._pattern_matches(m, rule.metric_pattern)]

            for metric_name in matching:
                for agg_type in rule.aggregations:
                    result = self.aggregate(
                        metric_name,
                        agg_type,
                        rule.window_s,
                        rule.labels,
                        use_cache=False
                    )
                    results.append(result)

            self._last_run[rule_key] = now

        # Emit aggregation event
        if results:
            self._emit_aggregated_event(results)

        return results

    def emit_aggregated_event(self, result: AggregatedMetric) -> str:
        """Emit a single aggregated metric event.

        Args:
            result: Aggregated metric

        Returns:
            Event ID
        """
        return self._emit_aggregated_event([result])

    def handle_aggregate_request(self, event: Dict[str, Any]) -> Optional[AggregatedMetric]:
        """Handle an aggregate request from bus.

        Args:
            event: Bus event with aggregate request

        Returns:
            Aggregated metric or None
        """
        data = event.get("data", {})
        metric_name = data.get("metric", data.get("name"))
        if not metric_name:
            return None

        agg_type = AggregationType(data.get("aggregation", "avg"))
        window_s = data.get("window_s", 300)
        labels = data.get("labels", {})

        result = self.aggregate(metric_name, agg_type, window_s, labels)
        self.emit_aggregated_event(result)

        return result

    def _compute_aggregation(
        self,
        points: List[MetricPoint],
        aggregation: AggregationType
    ) -> float:
        """Compute aggregation on points.

        Args:
            points: Metric points
            aggregation: Aggregation type

        Returns:
            Aggregated value
        """
        if not points:
            return 0.0

        values = [p.value for p in points]

        if aggregation == AggregationType.AVG:
            return sum(values) / len(values)
        elif aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.COUNT:
            return float(len(values))
        elif aggregation == AggregationType.RATE:
            if len(points) < 2:
                return 0.0
            duration = points[-1].timestamp - points[0].timestamp
            if duration <= 0:
                return 0.0
            return (points[-1].value - points[0].value) / duration
        elif aggregation == AggregationType.P50:
            return self._percentile(points, 50)
        elif aggregation == AggregationType.P90:
            return self._percentile(points, 90)
        elif aggregation == AggregationType.P95:
            return self._percentile(points, 95)
        elif aggregation == AggregationType.P99:
            return self._percentile(points, 99)
        elif aggregation == AggregationType.STDDEV:
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return math.sqrt(variance)
        elif aggregation == AggregationType.VARIANCE:
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / len(values)
        else:
            return 0.0

    def _percentile(self, points: List[MetricPoint], percentile: float) -> float:
        """Compute percentile value.

        Args:
            points: Metric points
            percentile: Percentile (0-100)

        Returns:
            Percentile value
        """
        if not points:
            return 0.0

        values = sorted(p.value for p in points)
        n = len(values)

        if n == 1:
            return values[0]

        # Linear interpolation
        k = (n - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return values[int(k)]

        return values[int(f)] * (c - k) + values[int(c)] * (k - f)

    def _pattern_matches(self, metric_name: str, pattern: str) -> bool:
        """Check if metric name matches pattern.

        Args:
            metric_name: Metric name
            pattern: Pattern (supports * wildcard)

        Returns:
            True if matches
        """
        if pattern == "*":
            return True

        parts = pattern.split(".")
        name_parts = metric_name.split(".")

        if len(parts) > len(name_parts) and "*" not in parts:
            return False

        for i, part in enumerate(parts):
            if part == "*":
                continue
            if i >= len(name_parts):
                return False
            if part != name_parts[i]:
                return False

        return True

    def _emit_aggregated_event(self, results: List[AggregatedMetric]) -> str:
        """Emit aggregated metrics event.

        Args:
            results: List of aggregated metrics

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS["aggregated"],
            "kind": "metric",
            "level": "info",
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "metrics": [r.to_dict() for r in results],
                "count": len(results),
            },
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id


# Singleton instance
_aggregator: Optional[MetricAggregator] = None


def get_aggregator() -> MetricAggregator:
    """Get or create the metric aggregator singleton.

    Returns:
        MetricAggregator instance
    """
    global _aggregator
    if _aggregator is None:
        _aggregator = MetricAggregator()
    return _aggregator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metric Aggregator (Step 253)")
    parser.add_argument("--aggregate", metavar="NAME", help="Aggregate a metric")
    parser.add_argument("--agg", default="avg", help="Aggregation type")
    parser.add_argument("--window", type=int, default=300, help="Window in seconds")
    parser.add_argument("--stats", metavar="NAME", help="Get full statistics for metric")
    parser.add_argument("--percentile", type=float, help="Compute specific percentile")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    aggregator = get_aggregator()

    if args.aggregate:
        agg_type = AggregationType(args.agg)
        result = aggregator.aggregate(args.aggregate, agg_type, args.window)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"{result.name} ({result.aggregation.value}): {result.value}")
            print(f"  Samples: {result.sample_count}")
            print(f"  Window: {result.window_s}s")

    if args.stats:
        stats = aggregator.compute_statistics(args.stats, args.window)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Statistics for {args.stats}:")
            for k, v in stats.items():
                print(f"  {k}: {v:.4f}")

    if args.percentile and args.aggregate:
        value = aggregator.compute_percentile(args.aggregate, args.percentile, args.window)
        print(f"P{args.percentile}: {value}")
