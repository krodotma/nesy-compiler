#!/usr/bin/env python3
"""
Step 147: Test Telemetry Module

Usage analytics and telemetry for the Test Agent.

PBTSO Phase: OBSERVE
Bus Topics:
- telemetry.test.event (emits)
- telemetry.test.metric (emits)
- telemetry.test.coverage (emits)

Dependencies: Steps 101-146 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Constants
# ============================================================================

class TelemetryEventType(Enum):
    """Telemetry event types."""
    # Test events
    TEST_RUN_START = "test.run.start"
    TEST_RUN_COMPLETE = "test.run.complete"
    TEST_PASS = "test.pass"
    TEST_FAIL = "test.fail"
    TEST_SKIP = "test.skip"

    # Coverage events
    COVERAGE_COLLECT = "coverage.collect"
    COVERAGE_REPORT = "coverage.report"

    # Performance events
    PERFORMANCE_SLOW_TEST = "performance.slow_test"
    PERFORMANCE_TIMEOUT = "performance.timeout"

    # System events
    AGENT_START = "agent.start"
    AGENT_STOP = "agent.stop"
    ERROR = "error"

    # Feature usage
    FEATURE_USED = "feature.used"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TelemetryEvent:
    """
    A telemetry event.

    Attributes:
        event_id: Unique event ID
        event_type: Type of event
        timestamp: Event timestamp
        properties: Event properties
        session_id: Session ID
        user_id: User ID (anonymous)
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: TelemetryEventType = TelemetryEventType.FEATURE_USED
    timestamp: float = field(default_factory=time.time)
    properties: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "properties": self.properties,
            "session_id": self.session_id,
            "agent_version": self.agent_version,
        }


@dataclass
class TelemetryMetric:
    """
    A telemetry metric.

    Attributes:
        name: Metric name
        metric_type: Type of metric
        value: Metric value
        tags: Metric tags
        timestamp: Metric timestamp
    """
    name: str
    metric_type: MetricType = MetricType.GAUGE
    value: float = 0
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "unit": self.unit,
        }


@dataclass
class TelemetryReport:
    """
    Aggregated telemetry report.

    Attributes:
        period_start: Report period start
        period_end: Report period end
        events_count: Total events count
        events_by_type: Events grouped by type
        metrics_summary: Metrics summary
        top_features: Top used features
    """
    period_start: float = field(default_factory=time.time)
    period_end: float = field(default_factory=time.time)
    events_count: int = 0
    events_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    metrics_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    test_summary: Dict[str, Any] = field(default_factory=dict)
    coverage_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": datetime.fromtimestamp(self.period_start, tz=timezone.utc).isoformat(),
            "period_end": datetime.fromtimestamp(self.period_end, tz=timezone.utc).isoformat(),
            "events_count": self.events_count,
            "events_by_type": dict(self.events_by_type),
            "metrics_summary": self.metrics_summary,
            "top_features": self.top_features,
            "test_summary": self.test_summary,
            "coverage_summary": self.coverage_summary,
        }


@dataclass
class TelemetryConfig:
    """
    Configuration for telemetry.

    Attributes:
        output_dir: Output directory
        enabled: Whether telemetry is enabled
        sample_rate: Event sampling rate (0-1)
        flush_interval_s: Flush interval
        max_events: Maximum events to buffer
        anonymize: Anonymize data
        collect_performance: Collect performance metrics
    """
    output_dir: str = ".pluribus/test-agent/telemetry"
    enabled: bool = True
    sample_rate: float = 1.0
    flush_interval_s: int = 60
    max_events: int = 10000
    anonymize: bool = True
    collect_performance: bool = True
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "flush_interval_s": self.flush_interval_s,
            "collect_performance": self.collect_performance,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class TelemetryBus:
    """Bus interface for telemetry with file locking."""

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


# ============================================================================
# Test Telemetry
# ============================================================================

class TestTelemetry:
    """
    Telemetry system for the Test Agent.

    Features:
    - Event tracking
    - Metric collection
    - Usage analytics
    - Performance monitoring
    - Privacy-aware collection

    PBTSO Phase: OBSERVE
    Bus Topics: telemetry.test.event, telemetry.test.metric, telemetry.test.coverage
    """

    BUS_TOPICS = {
        "event": "telemetry.test.event",
        "metric": "telemetry.test.metric",
        "coverage": "telemetry.test.coverage",
    }

    def __init__(self, bus=None, config: Optional[TelemetryConfig] = None):
        """
        Initialize the telemetry system.

        Args:
            bus: Optional bus instance
            config: Telemetry configuration
        """
        self.bus = bus or TelemetryBus()
        self.config = config or TelemetryConfig()
        self._events: List[TelemetryEvent] = []
        self._metrics: Dict[str, List[TelemetryMetric]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, float] = {}
        self._feature_usage: Dict[str, int] = defaultdict(int)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def track_event(
        self,
        event_type: TelemetryEventType,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[TelemetryEvent]:
        """
        Track a telemetry event.

        Args:
            event_type: Type of event
            properties: Event properties

        Returns:
            TelemetryEvent if tracked, None if disabled/sampled out
        """
        if not self.config.enabled:
            return None

        # Apply sampling
        import random
        if random.random() > self.config.sample_rate:
            return None

        event = TelemetryEvent(
            event_type=event_type,
            properties=properties or {},
            session_id=self.config.session_id,
        )

        self._events.append(event)

        # Emit to bus
        self._emit_event("event", event.to_dict())

        # Cleanup if buffer is full
        if len(self._events) > self.config.max_events:
            self._flush_events()

        return event

    def track_test_run(
        self,
        total_tests: int,
        passed: int,
        failed: int,
        skipped: int,
        duration_s: float,
    ) -> None:
        """
        Track a test run.

        Args:
            total_tests: Total tests run
            passed: Tests passed
            failed: Tests failed
            skipped: Tests skipped
            duration_s: Run duration
        """
        self.track_event(TelemetryEventType.TEST_RUN_COMPLETE, {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration_s": duration_s,
            "pass_rate": passed / total_tests if total_tests > 0 else 0,
        })

        # Update counters
        self.increment("tests.total", total_tests)
        self.increment("tests.passed", passed)
        self.increment("tests.failed", failed)
        self.increment("tests.skipped", skipped)

        # Update gauges
        self.gauge("tests.pass_rate", passed / total_tests if total_tests > 0 else 0)
        self.gauge("tests.last_duration_s", duration_s)

    def track_coverage(
        self,
        line_coverage: float,
        branch_coverage: float,
        files_covered: int,
        lines_covered: int,
        lines_total: int,
    ) -> None:
        """
        Track coverage metrics.

        Args:
            line_coverage: Line coverage percentage
            branch_coverage: Branch coverage percentage
            files_covered: Number of files covered
            lines_covered: Lines covered
            lines_total: Total lines
        """
        self.track_event(TelemetryEventType.COVERAGE_REPORT, {
            "line_coverage": line_coverage,
            "branch_coverage": branch_coverage,
            "files_covered": files_covered,
            "lines_covered": lines_covered,
            "lines_total": lines_total,
        })

        # Emit coverage event to bus
        self._emit_event("coverage", {
            "line_coverage": line_coverage,
            "branch_coverage": branch_coverage,
        })

        # Update gauges
        self.gauge("coverage.line", line_coverage)
        self.gauge("coverage.branch", branch_coverage)

    def track_feature(self, feature_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track feature usage.

        Args:
            feature_name: Name of feature used
            metadata: Additional metadata
        """
        self._feature_usage[feature_name] += 1

        self.track_event(TelemetryEventType.FEATURE_USED, {
            "feature": feature_name,
            "usage_count": self._feature_usage[feature_name],
            **(metadata or {}),
        })

    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Value to increment by
            tags: Metric tags
        """
        self._counters[name] += value

        metric = TelemetryMetric(
            name=name,
            metric_type=MetricType.COUNTER,
            value=self._counters[name],
            tags=tags or {},
        )
        self._metrics[name].append(metric)

        self._emit_event("metric", metric.to_dict())

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Metric tags
        """
        self._gauges[name] = value

        metric = TelemetryMetric(
            name=name,
            metric_type=MetricType.GAUGE,
            value=value,
            tags=tags or {},
        )
        self._metrics[name].append(metric)

        self._emit_event("metric", metric.to_dict())

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a histogram value.

        Args:
            name: Metric name
            value: Value to record
            tags: Metric tags
        """
        self._histograms[name].append(value)

        metric = TelemetryMetric(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            value=value,
            tags=tags or {},
        )
        self._metrics[name].append(metric)

    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self._timers[name] = time.time()

    def stop_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """
        Stop a timer and record the duration.

        Args:
            name: Timer name
            tags: Metric tags

        Returns:
            Duration in seconds
        """
        start_time = self._timers.pop(name, time.time())
        duration = time.time() - start_time

        metric = TelemetryMetric(
            name=name,
            metric_type=MetricType.TIMER,
            value=duration,
            tags=tags or {},
            unit="seconds",
        )
        self._metrics[name].append(metric)

        return duration

    def generate_report(
        self,
        period_hours: int = 24,
    ) -> TelemetryReport:
        """
        Generate a telemetry report.

        Args:
            period_hours: Report period in hours

        Returns:
            TelemetryReport with aggregated data
        """
        period_start = time.time() - (period_hours * 3600)
        period_end = time.time()

        # Filter events in period
        events_in_period = [
            e for e in self._events
            if e.timestamp >= period_start
        ]

        report = TelemetryReport(
            period_start=period_start,
            period_end=period_end,
            events_count=len(events_in_period),
        )

        # Count events by type
        for event in events_in_period:
            report.events_by_type[event.event_type.value] += 1

        # Summarize metrics
        for name, metrics in self._metrics.items():
            metrics_in_period = [m for m in metrics if m.timestamp >= period_start]
            if metrics_in_period:
                values = [m.value for m in metrics_in_period]
                report.metrics_summary[name] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values),
                }

        # Top features
        report.top_features = [
            {"feature": name, "count": count}
            for name, count in sorted(
                self._feature_usage.items(),
                key=lambda x: -x[1]
            )[:10]
        ]

        # Test summary
        report.test_summary = {
            "total_runs": self._counters.get("tests.total", 0),
            "total_passed": self._counters.get("tests.passed", 0),
            "total_failed": self._counters.get("tests.failed", 0),
            "current_pass_rate": self._gauges.get("tests.pass_rate", 0),
        }

        # Coverage summary
        report.coverage_summary = {
            "line_coverage": self._gauges.get("coverage.line", 0),
            "branch_coverage": self._gauges.get("coverage.branch", 0),
        }

        return report

    def _flush_events(self) -> None:
        """Flush events to disk."""
        if not self._events:
            return

        events_file = Path(self.config.output_dir) / "events.ndjson"

        try:
            with open(events_file, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    for event in self._events:
                        f.write(json.dumps(event.to_dict()) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            self._events.clear()

        except IOError:
            pass

    def save_report(self, report: TelemetryReport) -> Path:
        """Save a telemetry report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(self.config.output_dir) / f"report_{timestamp}.json"

        with open(report_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(report.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return report_path

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0,
                }
                for k, v in self._histograms.items()
            },
        }

    async def track_event_async(
        self,
        event_type: TelemetryEventType,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[TelemetryEvent]:
        """Async version of track_event."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.track_event, event_type, properties
        )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"telemetry.test.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "telemetry",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Telemetry."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Telemetry")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate telemetry report")
    report_parser.add_argument("--hours", type=int, default=24, help="Report period in hours")
    report_parser.add_argument("--save", action="store_true", help="Save report to file")

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Show current metrics")

    # Features command
    features_parser = subparsers.add_parser("features", help="Show feature usage")

    # Track command
    track_parser = subparsers.add_parser("track", help="Track an event")
    track_parser.add_argument("event_type", help="Event type")
    track_parser.add_argument("--props", type=json.loads, default={}, help="Event properties (JSON)")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/telemetry")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = TelemetryConfig(output_dir=args.output)
    telemetry = TestTelemetry(config=config)

    if args.command == "report":
        report = telemetry.generate_report(period_hours=args.hours)

        if args.save:
            path = telemetry.save_report(report)
            print(f"Report saved to: {path}")

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("\nTelemetry Report")
            print(f"Period: {args.hours} hours")
            print(f"\n  Events: {report.events_count}")

            print("\n  Events by Type:")
            for event_type, count in report.events_by_type.items():
                print(f"    {event_type}: {count}")

            if report.test_summary:
                print("\n  Test Summary:")
                for key, value in report.test_summary.items():
                    print(f"    {key}: {value}")

            if report.top_features:
                print("\n  Top Features:")
                for feature in report.top_features[:5]:
                    print(f"    {feature['feature']}: {feature['count']}")

    elif args.command == "metrics":
        metrics = telemetry.get_metrics()

        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print("\nMetrics:")

            if metrics["counters"]:
                print("\n  Counters:")
                for name, value in metrics["counters"].items():
                    print(f"    {name}: {value}")

            if metrics["gauges"]:
                print("\n  Gauges:")
                for name, value in metrics["gauges"].items():
                    print(f"    {name}: {value}")

            if metrics["histograms"]:
                print("\n  Histograms:")
                for name, stats in metrics["histograms"].items():
                    print(f"    {name}: avg={stats['avg']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

    elif args.command == "features":
        if args.json:
            print(json.dumps(dict(telemetry._feature_usage), indent=2))
        else:
            print("\nFeature Usage:")
            for feature, count in sorted(telemetry._feature_usage.items(), key=lambda x: -x[1]):
                print(f"  {feature}: {count}")

    elif args.command == "track":
        try:
            event_type = TelemetryEventType(args.event_type)
        except ValueError:
            print(f"Invalid event type: {args.event_type}")
            print(f"Valid types: {[e.value for e in TelemetryEventType]}")
            exit(1)

        event = telemetry.track_event(event_type, args.props)

        if args.json:
            print(json.dumps(event.to_dict() if event else {"tracked": False}, indent=2))
        else:
            if event:
                print(f"Tracked event: {event.event_id}")
            else:
                print("Event not tracked (disabled or sampled out)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
