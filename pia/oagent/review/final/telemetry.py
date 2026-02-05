#!/usr/bin/env python3
"""
Telemetry System (Step 197)

Usage analytics and telemetry collection for the Review Agent with
privacy controls, aggregation, and export capabilities.

PBTSO Phase: OBSERVE, DISTILL
Bus Topics: review.telemetry.event, review.telemetry.metrics, review.telemetry.export

Telemetry Features:
- Event collection
- Metrics aggregation
- Privacy controls
- Export to various formats
- Usage analytics

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class EventType(Enum):
    """Telemetry event types."""
    REVIEW_STARTED = "review_started"
    REVIEW_COMPLETED = "review_completed"
    REVIEW_FAILED = "review_failed"
    SECURITY_SCAN = "security_scan"
    VULNERABILITY_FOUND = "vulnerability_found"
    API_CALL = "api_call"
    ERROR = "error"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class PrivacyLevel(Enum):
    """Privacy levels for telemetry."""
    FULL = "full"           # All data collected
    ANONYMOUS = "anonymous"  # PII removed
    MINIMAL = "minimal"      # Essential only
    NONE = "none"           # No collection


@dataclass
class TelemetryEvent:
    """
    A telemetry event.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        timestamp: Event timestamp
        data: Event data
        session_id: Session identifier
        user_id: User identifier (anonymized)
        privacy_level: Privacy level applied
    """
    event_id: str
    event_type: EventType
    timestamp: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMOUS

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "privacy_level": self.privacy_level.value,
        }

    def anonymize(self) -> "TelemetryEvent":
        """Return anonymized version of event."""
        anonymized_data = {}
        for k, v in self.data.items():
            if k not in ("email", "name", "ip_address", "user_agent"):
                anonymized_data[k] = v

        return TelemetryEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            timestamp=self.timestamp,
            data=anonymized_data,
            session_id=self._hash(self.session_id) if self.session_id else None,
            user_id=self._hash(self.user_id) if self.user_id else None,
            privacy_level=PrivacyLevel.ANONYMOUS,
        )

    def _hash(self, value: str) -> str:
        """Hash a value for anonymization."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]


@dataclass
class Metric:
    """
    A telemetry metric.

    Attributes:
        name: Metric name
        metric_type: Type of metric
        value: Current value
        labels: Metric labels
        timestamp: Measurement timestamp
    """
    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


@dataclass
class UsageStats:
    """
    Aggregated usage statistics.

    Attributes:
        period_start: Period start
        period_end: Period end
        reviews_total: Total reviews
        reviews_passed: Reviews that passed
        reviews_failed: Reviews that failed
        security_scans: Security scans performed
        vulnerabilities_found: Vulnerabilities found
        api_calls: API calls made
        errors: Errors occurred
        active_users: Unique active users
    """
    period_start: str
    period_end: str
    reviews_total: int = 0
    reviews_passed: int = 0
    reviews_failed: int = 0
    security_scans: int = 0
    vulnerabilities_found: int = 0
    api_calls: int = 0
    errors: int = 0
    active_users: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Telemetry Collector
# ============================================================================

class TelemetryCollector:
    """
    Collects telemetry events and metrics.

    Example:
        collector = TelemetryCollector()

        # Track event
        collector.track_event(EventType.REVIEW_COMPLETED, {
            "duration_ms": 1234,
            "issues_found": 5,
        })

        # Track metric
        collector.track_metric("review_duration", MetricType.HISTOGRAM, 1234)

        # Get aggregated data
        stats = collector.get_usage_stats()
    """

    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.ANONYMOUS,
        bus_path: Optional[Path] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize telemetry collector.

        Args:
            privacy_level: Default privacy level
            bus_path: Path to event bus file
            storage_path: Path for telemetry storage
        """
        self.privacy_level = privacy_level
        self.bus_path = bus_path or self._get_bus_path()
        self.storage_path = storage_path or self._get_storage_path()

        self._events: List[TelemetryEvent] = []
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)

        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_storage_path(self) -> Path:
        """Get telemetry storage path."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "telemetry"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "telemetry") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "telemetry-collector",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def track_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> TelemetryEvent:
        """
        Track a telemetry event.

        Args:
            event_type: Type of event
            data: Event data
            session_id: Session identifier
            user_id: User identifier

        Returns:
            Created TelemetryEvent
        """
        if self.privacy_level == PrivacyLevel.NONE:
            return TelemetryEvent(
                event_id="",
                event_type=event_type,
                privacy_level=PrivacyLevel.NONE,
            )

        event = TelemetryEvent(
            event_id="",
            event_type=event_type,
            data=data,
            session_id=session_id,
            user_id=user_id,
            privacy_level=self.privacy_level,
        )

        # Apply privacy level
        if self.privacy_level == PrivacyLevel.ANONYMOUS:
            event = event.anonymize()
        elif self.privacy_level == PrivacyLevel.MINIMAL:
            event = TelemetryEvent(
                event_id=event.event_id,
                event_type=event_type,
                privacy_level=PrivacyLevel.MINIMAL,
            )

        self._events.append(event)

        # Update counters
        self._counters[f"events.{event_type.value}"] += 1

        # Emit to bus
        self._emit_event("review.telemetry.event", {
            "event_type": event_type.value,
            "privacy_level": event.privacy_level.value,
        })

        return event

    def track_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Metric:
        """
        Track a metric.

        Args:
            name: Metric name
            metric_type: Type of metric
            value: Metric value
            labels: Metric labels

        Returns:
            Created Metric
        """
        if self.privacy_level == PrivacyLevel.NONE:
            return Metric(name=name, metric_type=metric_type, value=0)

        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            labels=labels or {},
        )

        self._metrics[name].append(metric)

        # Update counters/gauges
        if metric_type == MetricType.COUNTER:
            self._counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self._counters[name] = value

        return metric

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a counter."""
        self.track_metric(name, MetricType.COUNTER, value)

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self.track_metric(name, MetricType.GAUGE, value)

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> List[TelemetryEvent]:
        """Get collected events."""
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_metrics(
        self,
        name: Optional[str] = None,
        since: Optional[str] = None,
    ) -> Dict[str, List[Metric]]:
        """Get collected metrics."""
        if name:
            metrics = {name: self._metrics.get(name, [])}
        else:
            metrics = dict(self._metrics)

        if since:
            for n, m_list in metrics.items():
                metrics[n] = [m for m in m_list if m.timestamp >= since]

        return metrics

    def get_counter(self, name: str) -> float:
        """Get counter value."""
        return self._counters.get(name, 0)


# ============================================================================
# Usage Analytics
# ============================================================================

class UsageAnalytics:
    """
    Analyzes usage patterns and generates reports.

    Example:
        analytics = UsageAnalytics(collector)

        # Get daily stats
        stats = analytics.get_daily_stats()

        # Get trends
        trends = analytics.get_trends()
    """

    def __init__(self, collector: TelemetryCollector):
        """Initialize usage analytics."""
        self.collector = collector

    def get_usage_stats(
        self,
        period_hours: int = 24,
    ) -> UsageStats:
        """
        Get usage statistics for a period.

        Args:
            period_hours: Period in hours

        Returns:
            UsageStats
        """
        now = datetime.now(timezone.utc)
        period_start = now - timedelta(hours=period_hours)

        events = self.collector.get_events(
            since=period_start.isoformat() + "Z",
            limit=10000,
        )

        stats = UsageStats(
            period_start=period_start.isoformat() + "Z",
            period_end=now.isoformat() + "Z",
        )

        users = set()
        for event in events:
            if event.event_type == EventType.REVIEW_STARTED:
                stats.reviews_total += 1
            elif event.event_type == EventType.REVIEW_COMPLETED:
                stats.reviews_passed += 1
            elif event.event_type == EventType.REVIEW_FAILED:
                stats.reviews_failed += 1
            elif event.event_type == EventType.SECURITY_SCAN:
                stats.security_scans += 1
            elif event.event_type == EventType.VULNERABILITY_FOUND:
                stats.vulnerabilities_found += 1
            elif event.event_type == EventType.API_CALL:
                stats.api_calls += 1
            elif event.event_type == EventType.ERROR:
                stats.errors += 1

            if event.user_id:
                users.add(event.user_id)

        stats.active_users = len(users)
        return stats

    def get_daily_stats(self, days: int = 7) -> List[UsageStats]:
        """Get daily statistics for multiple days."""
        stats_list = []
        for i in range(days):
            # This is simplified - in production would query historical data
            stats = self.get_usage_stats(period_hours=24)
            stats_list.append(stats)
        return stats_list

    def get_top_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common events."""
        event_counts: Dict[str, int] = defaultdict(int)
        for event in self.collector._events:
            event_counts[event.event_type.value] += 1

        sorted_events = sorted(
            event_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"event_type": k, "count": v}
            for k, v in sorted_events[:limit]
        ]


# ============================================================================
# Metrics Exporter
# ============================================================================

class MetricsExporter:
    """
    Exports metrics to various formats.

    Example:
        exporter = MetricsExporter(collector)

        # Export to Prometheus format
        prometheus = exporter.to_prometheus()

        # Export to JSON
        json_data = exporter.to_json()
    """

    def __init__(self, collector: TelemetryCollector):
        """Initialize metrics exporter."""
        self.collector = collector

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append("# HELP review_events_total Total events by type")
        lines.append("# TYPE review_events_total counter")

        for name, value in self.collector._counters.items():
            safe_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"review_{safe_name} {value}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export metrics as JSON."""
        data = {
            "counters": dict(self.collector._counters),
            "metrics": {
                name: [m.to_dict() for m in metrics]
                for name, metrics in self.collector._metrics.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
        return json.dumps(data, indent=2)

    def to_csv(self) -> str:
        """Export metrics as CSV."""
        lines = ["name,type,value,timestamp"]
        for name, metrics in self.collector._metrics.items():
            for m in metrics:
                lines.append(f"{name},{m.metric_type.value},{m.value},{m.timestamp}")
        return "\n".join(lines)


# ============================================================================
# Combined Telemetry
# ============================================================================

class Telemetry:
    """
    Main telemetry interface.

    Example:
        telemetry = Telemetry()

        # Track review
        telemetry.track_review_started(review_id="123")
        telemetry.track_review_completed(review_id="123", issues=5)

        # Get stats
        stats = telemetry.get_stats()
    """

    BUS_TOPICS = {
        "event": "review.telemetry.event",
        "metrics": "review.telemetry.metrics",
        "export": "review.telemetry.export",
    }

    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.ANONYMOUS,
        bus_path: Optional[Path] = None,
    ):
        """Initialize telemetry."""
        self.collector = TelemetryCollector(privacy_level, bus_path)
        self.analytics = UsageAnalytics(self.collector)
        self.exporter = MetricsExporter(self.collector)
        self._last_heartbeat = time.time()

    def track_review_started(
        self,
        review_id: str,
        files: int = 0,
        **extra: Any,
    ) -> None:
        """Track review started."""
        self.collector.track_event(EventType.REVIEW_STARTED, {
            "review_id": review_id,
            "file_count": files,
            **extra,
        })
        self.collector.increment("reviews.started")

    def track_review_completed(
        self,
        review_id: str,
        issues: int = 0,
        duration_ms: float = 0,
        **extra: Any,
    ) -> None:
        """Track review completed."""
        self.collector.track_event(EventType.REVIEW_COMPLETED, {
            "review_id": review_id,
            "issues_found": issues,
            "duration_ms": duration_ms,
            **extra,
        })
        self.collector.increment("reviews.completed")
        self.collector.track_metric("review_duration", MetricType.HISTOGRAM, duration_ms)

    def track_review_failed(
        self,
        review_id: str,
        error: str,
        **extra: Any,
    ) -> None:
        """Track review failed."""
        self.collector.track_event(EventType.REVIEW_FAILED, {
            "review_id": review_id,
            "error": error,
            **extra,
        })
        self.collector.increment("reviews.failed")

    def track_vulnerability(
        self,
        severity: str,
        category: str,
        **extra: Any,
    ) -> None:
        """Track vulnerability found."""
        self.collector.track_event(EventType.VULNERABILITY_FOUND, {
            "severity": severity,
            "category": category,
            **extra,
        })
        self.collector.increment(f"vulnerabilities.{severity}")

    def track_api_call(
        self,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Track API call."""
        self.collector.track_event(EventType.API_CALL, {
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **extra,
        })
        self.collector.increment("api.calls")
        self.collector.track_metric("api_latency", MetricType.HISTOGRAM, duration_ms)

    def track_error(
        self,
        error_type: str,
        message: str,
        **extra: Any,
    ) -> None:
        """Track error."""
        self.collector.track_event(EventType.ERROR, {
            "error_type": error_type,
            "message": message,
            **extra,
        })
        self.collector.increment(f"errors.{error_type}")

    def get_stats(self, period_hours: int = 24) -> UsageStats:
        """Get usage statistics."""
        return self.analytics.get_usage_stats(period_hours)

    def export_prometheus(self) -> str:
        """Export in Prometheus format."""
        return self.exporter.to_prometheus()

    def export_json(self) -> str:
        """Export in JSON format."""
        return self.exporter.to_json()

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "telemetry-collector",
            "healthy": True,
            "events_collected": len(self.collector._events),
            "metrics_tracked": len(self.collector._metrics),
            "privacy_level": self.collector.privacy_level.value,
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self.collector._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Telemetry."""
    import argparse

    parser = argparse.ArgumentParser(description="Telemetry System (Step 197)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show usage stats")
    stats_parser.add_argument("--hours", type=int, default=24, help="Period in hours")

    # Events command
    events_parser = subparsers.add_parser("events", help="Show events")
    events_parser.add_argument("--type", help="Filter by event type")
    events_parser.add_argument("--limit", type=int, default=20, help="Limit results")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export metrics")
    export_parser.add_argument("--format", choices=["prometheus", "json", "csv"],
                               default="json", help="Export format")

    # Demo command
    subparsers.add_parser("demo", help="Generate demo data")

    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--privacy", choices=["full", "anonymous", "minimal", "none"],
                        default="anonymous", help="Privacy level")

    args = parser.parse_args()

    telemetry = Telemetry(privacy_level=PrivacyLevel[args.privacy.upper()])

    if args.command == "demo":
        # Generate demo events
        telemetry.track_review_started("rev-001", files=10)
        telemetry.track_review_completed("rev-001", issues=3, duration_ms=1234)
        telemetry.track_review_started("rev-002", files=5)
        telemetry.track_review_failed("rev-002", error="Timeout")
        telemetry.track_vulnerability("high", "injection")
        telemetry.track_api_call("/api/review", 200, 150)
        telemetry.track_error("validation", "Invalid input")

        print("Demo data generated")

        stats = telemetry.get_stats()
        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print(f"\nUsage Stats:")
            print(f"  Reviews Total: {stats.reviews_total}")
            print(f"  Reviews Passed: {stats.reviews_passed}")
            print(f"  Reviews Failed: {stats.reviews_failed}")
            print(f"  Vulnerabilities: {stats.vulnerabilities_found}")
            print(f"  API Calls: {stats.api_calls}")
            print(f"  Errors: {stats.errors}")

    elif args.command == "stats":
        stats = telemetry.get_stats(period_hours=args.hours)
        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print(f"Usage Stats (last {args.hours} hours)")
            for k, v in stats.to_dict().items():
                print(f"  {k}: {v}")

    elif args.command == "events":
        event_type = EventType[args.type.upper()] if args.type else None
        events = telemetry.collector.get_events(event_type=event_type, limit=args.limit)
        if args.json:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            print(f"Events: {len(events)}")
            for e in events:
                print(f"  [{e.timestamp}] {e.event_type.value}")

    elif args.command == "export":
        if args.format == "prometheus":
            print(telemetry.export_prometheus())
        elif args.format == "json":
            print(telemetry.export_json())
        else:
            print(telemetry.exporter.to_csv())

    else:
        # Default: show status
        status = telemetry.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Telemetry: {status['events_collected']} events, {status['metrics_tracked']} metrics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
