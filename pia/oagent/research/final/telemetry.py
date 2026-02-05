#!/usr/bin/env python3
"""
telemetry.py - Telemetry System (Step 47)

Usage analytics and telemetry collection for Research Agent.
Supports event tracking, usage metrics, and analytics aggregation.

PBTSO Phase: MONITOR

Bus Topics:
- a2a.research.telemetry.event
- a2a.research.telemetry.metric
- a2a.research.telemetry.flush
- research.telemetry.aggregate

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import atexit
import fcntl
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class EventCategory(Enum):
    """Telemetry event categories."""
    SEARCH = "search"
    INDEX = "index"
    ANALYSIS = "analysis"
    API = "api"
    CACHE = "cache"
    ERROR = "error"
    PERFORMANCE = "performance"
    USER_ACTION = "user_action"
    SYSTEM = "system"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class TelemetryConfig:
    """Configuration for telemetry system."""

    enabled: bool = True
    flush_interval_seconds: int = 60
    batch_size: int = 100
    sample_rate: float = 1.0  # 0.0 to 1.0
    anonymize_data: bool = True
    storage_dir: str = ""
    retention_days: int = 30
    enable_performance_tracking: bool = True
    enable_error_tracking: bool = True
    enable_user_tracking: bool = False
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        if not self.storage_dir:
            self.storage_dir = f"{pluribus_root}/.pluribus/research/telemetry"
        if self.bus_path is None:
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class TelemetryEvent:
    """A telemetry event."""

    id: str
    timestamp: float
    category: EventCategory
    action: str
    label: Optional[str] = None
    value: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "action": self.action,
            "label": self.label,
            "value": self.value,
            "properties": self.properties,
            "session_id": self.session_id,
            "user_id": self.user_id,
        }


@dataclass
class MetricValue:
    """A metric value."""

    name: str
    type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "unit": self.unit,
        }


@dataclass
class AggregatedMetric:
    """Aggregated metric over time window."""

    name: str
    window_start: float
    window_end: float
    count: int = 0
    sum: float = 0
    min: float = float("inf")
    max: float = float("-inf")
    avg: float = 0
    p50: float = 0
    p95: float = 0
    p99: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "count": self.count,
            "sum": self.sum,
            "min": self.min if self.min != float("inf") else 0,
            "max": self.max if self.max != float("-inf") else 0,
            "avg": self.avg,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass
class UsageReport:
    """Usage report for a time period."""

    start_time: float
    end_time: float
    total_events: int = 0
    events_by_category: Dict[str, int] = field(default_factory=dict)
    total_searches: int = 0
    total_api_calls: int = 0
    total_errors: int = 0
    unique_users: int = 0
    unique_sessions: int = 0
    metrics: List[AggregatedMetric] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_events": self.total_events,
            "events_by_category": self.events_by_category,
            "total_searches": self.total_searches,
            "total_api_calls": self.total_api_calls,
            "total_errors": self.total_errors,
            "unique_users": self.unique_users,
            "unique_sessions": self.unique_sessions,
            "metrics": [m.to_dict() for m in self.metrics],
        }


# ============================================================================
# Telemetry Collector
# ============================================================================


class TelemetryCollector:
    """
    Telemetry collector for Research Agent.

    Features:
    - Event tracking
    - Metric collection
    - Usage analytics
    - Automatic flushing
    - Data aggregation

    PBTSO Phase: MONITOR

    Example:
        telemetry = TelemetryCollector()

        # Track events
        telemetry.track_event(
            category=EventCategory.SEARCH,
            action="query",
            label="code_search",
            value=1,
            properties={"query_length": 50},
        )

        # Record metrics
        telemetry.record_metric("search_latency", 150, MetricType.TIMING)

        # Time operations
        with telemetry.timer("index_operation"):
            perform_indexing()

        # Get usage report
        report = telemetry.get_usage_report()
    """

    def __init__(
        self,
        config: Optional[TelemetryConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the telemetry collector.

        Args:
            config: Telemetry configuration
            bus: AgentBus for event emission
        """
        self.config = config or TelemetryConfig()
        self.bus = bus or AgentBus()

        # Event buffer
        self._events: List[TelemetryEvent] = []
        self._metrics: List[MetricValue] = []
        self._lock = threading.Lock()

        # Aggregation data
        self._metric_values: Dict[str, List[float]] = defaultdict(list)
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._users: Set[str] = set()
        self._sessions: Set[str] = set()

        # Current session
        self._session_id = str(uuid.uuid4())[:8]

        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False

        # Ensure storage directory
        Path(self.config.storage_dir).mkdir(parents=True, exist_ok=True)

        # Statistics
        self._stats = {
            "events_tracked": 0,
            "metrics_recorded": 0,
            "flushes": 0,
            "errors": 0,
        }

        # Start background flushing
        if self.config.enabled:
            self._start_flush_thread()
            atexit.register(self._shutdown)

    def track_event(
        self,
        category: EventCategory,
        action: str,
        label: Optional[str] = None,
        value: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[TelemetryEvent]:
        """
        Track a telemetry event.

        Args:
            category: Event category
            action: Event action
            label: Optional label
            value: Optional numeric value
            properties: Additional properties
            user_id: Optional user identifier

        Returns:
            TelemetryEvent if tracked, None if disabled/sampled out
        """
        if not self.config.enabled:
            return None

        # Apply sampling
        if self.config.sample_rate < 1.0:
            import random
            if random.random() > self.config.sample_rate:
                return None

        # Anonymize user ID if configured
        if user_id and self.config.anonymize_data:
            user_id = self._anonymize(user_id)

        event = TelemetryEvent(
            id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            category=category,
            action=action,
            label=label,
            value=value,
            properties=properties or {},
            session_id=self._session_id,
            user_id=user_id,
        )

        with self._lock:
            self._events.append(event)
            self._event_counts[f"{category.value}.{action}"] += 1

            if user_id:
                self._users.add(user_id)
            self._sessions.add(self._session_id)

            self._stats["events_tracked"] += 1

        # Emit to bus
        if self.config.emit_to_bus:
            self._emit_event("a2a.research.telemetry.event", event.to_dict())

        # Auto-flush if batch is full
        if len(self._events) >= self.config.batch_size:
            self._flush_async()

        return event

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
    ) -> MetricValue:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Metric tags
            unit: Measurement unit

        Returns:
            MetricValue recorded
        """
        metric = MetricValue(
            name=name,
            type=metric_type,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit,
        )

        with self._lock:
            self._metrics.append(metric)
            self._metric_values[name].append(value)
            self._stats["metrics_recorded"] += 1

        # Emit to bus
        if self.config.emit_to_bus:
            self._emit_event("a2a.research.telemetry.metric", metric.to_dict())

        return metric

    def increment(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self.record_metric(name, duration_ms, MetricType.TIMING, tags, unit="ms")

    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.

        Example:
            with telemetry.timer("search_operation"):
                perform_search()
        """
        return _Timer(self, name, tags)

    def track_search(
        self,
        query: str,
        results_count: int,
        latency_ms: float,
        **properties,
    ) -> None:
        """Track a search event."""
        self.track_event(
            category=EventCategory.SEARCH,
            action="query",
            value=results_count,
            properties={
                "query_length": len(query),
                "results_count": results_count,
                "latency_ms": latency_ms,
                **properties,
            },
        )
        self.timing("search_latency", latency_ms)
        self.increment("search_count")

    def track_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        **properties,
    ) -> None:
        """Track an API call."""
        self.track_event(
            category=EventCategory.API,
            action=f"{method}_{endpoint}",
            value=status_code,
            properties={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "latency_ms": latency_ms,
                **properties,
            },
        )
        self.timing("api_latency", latency_ms, {"endpoint": endpoint})
        self.increment("api_calls", tags={"endpoint": endpoint})

    def track_error(
        self,
        error_type: str,
        error_message: str,
        **properties,
    ) -> None:
        """Track an error event."""
        if not self.config.enable_error_tracking:
            return

        self.track_event(
            category=EventCategory.ERROR,
            action=error_type,
            label=error_message[:100],  # Truncate message
            properties={
                "error_type": error_type,
                **properties,
            },
        )
        self.increment("error_count", tags={"type": error_type})

    def get_usage_report(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> UsageReport:
        """
        Generate a usage report.

        Args:
            start_time: Report start time (default: 1 hour ago)
            end_time: Report end time (default: now)

        Returns:
            UsageReport with aggregated data
        """
        now = time.time()
        start = start_time or (now - 3600)
        end = end_time or now

        with self._lock:
            # Filter events by time
            filtered_events = [
                e for e in self._events
                if start <= e.timestamp <= end
            ]

            # Count by category
            events_by_category = defaultdict(int)
            searches = 0
            api_calls = 0
            errors = 0

            for event in filtered_events:
                events_by_category[event.category.value] += 1

                if event.category == EventCategory.SEARCH:
                    searches += 1
                elif event.category == EventCategory.API:
                    api_calls += 1
                elif event.category == EventCategory.ERROR:
                    errors += 1

            # Aggregate metrics
            aggregated_metrics = []
            for name, values in self._metric_values.items():
                if values:
                    aggregated_metrics.append(self._aggregate_metric(name, values, start, end))

            return UsageReport(
                start_time=start,
                end_time=end,
                total_events=len(filtered_events),
                events_by_category=dict(events_by_category),
                total_searches=searches,
                total_api_calls=api_calls,
                total_errors=errors,
                unique_users=len(self._users),
                unique_sessions=len(self._sessions),
                metrics=aggregated_metrics,
            )

    def flush(self) -> int:
        """
        Flush buffered telemetry data to storage.

        Returns:
            Number of events flushed
        """
        with self._lock:
            events_to_flush = self._events.copy()
            metrics_to_flush = self._metrics.copy()
            self._events = []
            self._metrics = []

        if not events_to_flush and not metrics_to_flush:
            return 0

        # Write to storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        storage_path = Path(self.config.storage_dir)

        if events_to_flush:
            events_file = storage_path / f"events_{timestamp}.json"
            with open(events_file, "w") as f:
                json.dump([e.to_dict() for e in events_to_flush], f)

        if metrics_to_flush:
            metrics_file = storage_path / f"metrics_{timestamp}.json"
            with open(metrics_file, "w") as f:
                json.dump([m.to_dict() for m in metrics_to_flush], f)

        self._stats["flushes"] += 1

        # Emit flush event
        if self.config.emit_to_bus:
            self._emit_event("a2a.research.telemetry.flush", {
                "events": len(events_to_flush),
                "metrics": len(metrics_to_flush),
            })

        # Cleanup old files
        self._cleanup_old_files()

        return len(events_to_flush)

    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        with self._lock:
            return {
                **self._stats,
                "buffered_events": len(self._events),
                "buffered_metrics": len(self._metrics),
                "unique_users": len(self._users),
                "unique_sessions": len(self._sessions),
                "session_id": self._session_id,
            }

    def _aggregate_metric(
        self,
        name: str,
        values: List[float],
        start: float,
        end: float,
    ) -> AggregatedMetric:
        """Aggregate metric values."""
        sorted_values = sorted(values)
        count = len(values)

        return AggregatedMetric(
            name=name,
            window_start=start,
            window_end=end,
            count=count,
            sum=sum(values),
            min=min(values),
            max=max(values),
            avg=sum(values) / count if count else 0,
            p50=self._percentile(sorted_values, 50),
            p95=self._percentile(sorted_values, 95),
            p99=self._percentile(sorted_values, 99),
        )

    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not sorted_values:
            return 0
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _anonymize(self, value: str) -> str:
        """Anonymize a value."""
        return hashlib.sha256(value.encode()).hexdigest()[:12]

    def _start_flush_thread(self) -> None:
        """Start background flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            time.sleep(self.config.flush_interval_seconds)
            try:
                self.flush()
            except Exception:
                self._stats["errors"] += 1

    def _flush_async(self) -> None:
        """Trigger async flush."""
        threading.Thread(target=self.flush, daemon=True).start()

    def _shutdown(self) -> None:
        """Shutdown telemetry collector."""
        self._running = False
        self.flush()

    def _cleanup_old_files(self) -> None:
        """Remove old telemetry files."""
        storage_path = Path(self.config.storage_dir)
        cutoff = time.time() - (self.config.retention_days * 86400)

        for file_path in storage_path.glob("*.json"):
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()

    def _emit_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
    ) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "telemetry",
            "level": level,
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


class _Timer:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: TelemetryCollector,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time: Optional[float] = None

    def __enter__(self) -> "_Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.timing(self.name, duration_ms, self.tags)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Telemetry."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Telemetry System (Step 47)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate usage report")
    report_parser.add_argument("--hours", type=int, default=1, help="Hours to include")
    report_parser.add_argument("--json", action="store_true")

    # Flush command
    flush_parser = subparsers.add_parser("flush", help="Flush telemetry data")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run telemetry demo")

    args = parser.parse_args()

    # Disable auto-flush for CLI
    config = TelemetryConfig(flush_interval_seconds=3600)
    telemetry = TelemetryCollector(config)

    if args.command == "report":
        end_time = time.time()
        start_time = end_time - (args.hours * 3600)
        report = telemetry.get_usage_report(start_time, end_time)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Usage Report ({args.hours}h)")
            print(f"  Total Events: {report.total_events}")
            print(f"  Searches: {report.total_searches}")
            print(f"  API Calls: {report.total_api_calls}")
            print(f"  Errors: {report.total_errors}")
            print(f"  Unique Users: {report.unique_users}")
            print(f"  Unique Sessions: {report.unique_sessions}")

            if report.events_by_category:
                print("  Events by Category:")
                for cat, count in report.events_by_category.items():
                    print(f"    {cat}: {count}")

    elif args.command == "flush":
        count = telemetry.flush()
        print(f"Flushed {count} events")

    elif args.command == "stats":
        stats = telemetry.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Telemetry Statistics:")
            print(f"  Events Tracked: {stats['events_tracked']}")
            print(f"  Metrics Recorded: {stats['metrics_recorded']}")
            print(f"  Flushes: {stats['flushes']}")
            print(f"  Buffered Events: {stats['buffered_events']}")
            print(f"  Session ID: {stats['session_id']}")

    elif args.command == "demo":
        print("Running telemetry demo...\n")

        # Track various events
        print("Tracking events...")

        telemetry.track_event(
            category=EventCategory.SEARCH,
            action="query",
            label="code_search",
            value=42,
            properties={"query": "find functions"},
        )
        print("  - Search event tracked")

        telemetry.track_search(
            query="find all classes",
            results_count=15,
            latency_ms=125.5,
        )
        print("  - Search tracking done")

        telemetry.track_api_call(
            endpoint="/api/search",
            method="POST",
            status_code=200,
            latency_ms=150.0,
        )
        print("  - API call tracked")

        telemetry.track_error(
            error_type="IndexError",
            error_message="Symbol not found",
        )
        print("  - Error tracked")

        # Record metrics
        print("\nRecording metrics...")
        telemetry.increment("cache_hits")
        telemetry.gauge("active_connections", 5)
        telemetry.timing("query_time", 50.5)

        # Use timer context manager
        with telemetry.timer("operation_duration"):
            time.sleep(0.1)
        print("  - Timed operation recorded")

        # Get report
        print("\n--- Usage Report ---")
        report = telemetry.get_usage_report()
        print(f"Total Events: {report.total_events}")
        print(f"Searches: {report.total_searches}")
        print(f"API Calls: {report.total_api_calls}")
        print(f"Errors: {report.total_errors}")
        print(f"Unique Sessions: {report.unique_sessions}")

        if report.metrics:
            print("\nMetrics:")
            for metric in report.metrics:
                print(f"  {metric.name}: count={metric.count}, avg={metric.avg:.2f}")

        # Show stats
        print("\n--- Statistics ---")
        stats = telemetry.get_stats()
        print(f"Events Tracked: {stats['events_tracked']}")
        print(f"Metrics Recorded: {stats['metrics_recorded']}")

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
