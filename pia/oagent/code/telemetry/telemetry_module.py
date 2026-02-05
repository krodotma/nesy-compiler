#!/usr/bin/env python3
"""
telemetry_module.py - Telemetry Module (Step 97)

PBTSO Phase: VERIFY, ITERATE

Provides:
- Usage analytics collection
- Performance metrics
- Distributed tracing
- Custom events
- Metric aggregation

Bus Topics:
- code.telemetry.event
- code.telemetry.metric
- code.telemetry.span
- code.telemetry.report

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import statistics
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class TelemetryLevel(Enum):
    """Telemetry collection level."""
    OFF = "off"
    BASIC = "basic"
    STANDARD = "standard"
    DETAILED = "detailed"


@dataclass
class TelemetryConfig:
    """Configuration for telemetry module."""
    level: TelemetryLevel = TelemetryLevel.STANDARD
    flush_interval_s: float = 30.0
    max_events: int = 10000
    max_spans: int = 1000
    enable_sampling: bool = True
    sample_rate: float = 1.0
    export_enabled: bool = True
    export_path: str = "/pluribus/.pluribus/telemetry"
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "flush_interval_s": self.flush_interval_s,
            "max_events": self.max_events,
            "sample_rate": self.sample_rate,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Telemetry Types
# =============================================================================

@dataclass
class TelemetryEvent:
    """A telemetry event."""
    id: str
    name: str
    timestamp: float
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "properties": self.properties,
            "tags": self.tags,
        }


@dataclass
class TelemetryMetric:
    """A telemetry metric."""
    name: str
    type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "unit": self.unit,
            "tags": self.tags,
        }


@dataclass
class TelemetrySpan:
    """A tracing span for distributed tracing."""
    id: str
    name: str
    trace_id: str
    parent_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[TelemetryEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "event_count": len(self.events),
        }


@dataclass
class MetricAggregation:
    """Aggregated metric statistics."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    stddev: float
    percentiles: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "stddev": self.stddev,
            "percentiles": self.percentiles,
        }


# =============================================================================
# Telemetry Module
# =============================================================================

class TelemetryModule:
    """
    Telemetry module for usage analytics.

    PBTSO Phase: VERIFY, ITERATE

    Features:
    - Event tracking
    - Metric collection
    - Distributed tracing
    - Metric aggregation
    - Export to storage

    Usage:
        telemetry = TelemetryModule()

        # Track event
        telemetry.track_event("user.action", {"action": "click"})

        # Record metric
        telemetry.record_metric("request.latency", 45.2, MetricType.HISTOGRAM)

        # Tracing
        with telemetry.span("operation") as span:
            span.attributes["key"] = "value"
            # do work
    """

    BUS_TOPICS = {
        "event": "code.telemetry.event",
        "metric": "code.telemetry.metric",
        "span": "code.telemetry.span",
        "report": "code.telemetry.report",
    }

    def __init__(
        self,
        config: Optional[TelemetryConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or TelemetryConfig()
        self.bus = bus or LockedAgentBus()

        self._events: List[TelemetryEvent] = []
        self._metrics: Dict[str, List[TelemetryMetric]] = defaultdict(list)
        self._spans: List[TelemetrySpan] = []
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}

        self._lock = Lock()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

        self._export_dir = Path(self.config.export_path)
        self._export_dir.mkdir(parents=True, exist_ok=True)

        # Current trace context
        self._current_trace_id: Optional[str] = None
        self._current_span_id: Optional[str] = None

    # =========================================================================
    # Event Tracking
    # =========================================================================

    def track_event(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> TelemetryEvent:
        """Track a telemetry event."""
        if self.config.level == TelemetryLevel.OFF:
            return TelemetryEvent(
                id="", name=name, timestamp=time.time(),
            )

        # Sampling
        if self.config.enable_sampling:
            import random
            if random.random() > self.config.sample_rate:
                return TelemetryEvent(
                    id="", name=name, timestamp=time.time(),
                )

        event = TelemetryEvent(
            id=f"evt-{uuid.uuid4().hex[:12]}",
            name=name,
            timestamp=time.time(),
            properties=properties or {},
            tags=tags or {},
        )

        with self._lock:
            self._events.append(event)
            # Trim if over limit
            if len(self._events) > self.config.max_events:
                self._events = self._events[-self.config.max_events:]

        self.bus.emit({
            "topic": self.BUS_TOPICS["event"],
            "kind": "telemetry",
            "actor": "telemetry-module",
            "data": event.to_dict(),
        })

        return event

    # =========================================================================
    # Metric Recording
    # =========================================================================

    def record_metric(
        self,
        name: str,
        value: float,
        type: MetricType = MetricType.GAUGE,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> TelemetryMetric:
        """Record a metric value."""
        if self.config.level == TelemetryLevel.OFF:
            return TelemetryMetric(name=name, type=type, value=value)

        metric = TelemetryMetric(
            name=name,
            type=type,
            value=value,
            unit=unit,
            tags=tags or {},
        )

        with self._lock:
            if type == MetricType.COUNTER:
                self._counters[name] += value
            elif type == MetricType.GAUGE:
                self._gauges[name] = value
            else:
                self._metrics[name].append(metric)
                # Trim if over limit
                if len(self._metrics[name]) > 10000:
                    self._metrics[name] = self._metrics[name][-10000:]

        if self.config.level == TelemetryLevel.DETAILED:
            self.bus.emit({
                "topic": self.BUS_TOPICS["metric"],
                "kind": "telemetry",
                "actor": "telemetry-module",
                "data": metric.to_dict(),
            })

        return metric

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        self.record_metric(name, value, MetricType.COUNTER, tags=tags)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value."""
        self.record_metric(name, value, MetricType.GAUGE, tags=tags)

    def histogram(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        self.record_metric(name, value, MetricType.HISTOGRAM, unit=unit, tags=tags)

    def timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer value."""
        self.record_metric(name, duration_ms, MetricType.TIMER, unit="ms", tags=tags)

    # =========================================================================
    # Tracing
    # =========================================================================

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Create a tracing span context manager."""
        if self.config.level == TelemetryLevel.OFF:
            yield None
            return

        # Create or use existing trace
        trace_id = self._current_trace_id or f"trace-{uuid.uuid4().hex[:16]}"
        parent_id = self._current_span_id

        span = TelemetrySpan(
            id=f"span-{uuid.uuid4().hex[:12]}",
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time=time.time(),
            attributes=attributes or {},
        )

        # Set as current
        old_trace_id = self._current_trace_id
        old_span_id = self._current_span_id
        self._current_trace_id = trace_id
        self._current_span_id = span.id

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            span.end_time = time.time()

            # Restore context
            self._current_trace_id = old_trace_id
            self._current_span_id = old_span_id

            with self._lock:
                self._spans.append(span)
                if len(self._spans) > self.config.max_spans:
                    self._spans = self._spans[-self.config.max_spans:]

            self.bus.emit({
                "topic": self.BUS_TOPICS["span"],
                "kind": "telemetry",
                "actor": "telemetry-module",
                "data": span.to_dict(),
            })

            # Record as timer metric
            if span.duration_ms:
                self.timer(f"span.{name}", span.duration_ms)

    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TelemetrySpan:
        """Start a span manually."""
        trace_id = self._current_trace_id or f"trace-{uuid.uuid4().hex[:16]}"
        parent_id = self._current_span_id

        span = TelemetrySpan(
            id=f"span-{uuid.uuid4().hex[:12]}",
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time=time.time(),
            attributes=attributes or {},
        )

        self._current_trace_id = trace_id
        self._current_span_id = span.id

        return span

    def end_span(self, span: TelemetrySpan, status: str = "ok") -> None:
        """End a span manually."""
        span.end_time = time.time()
        span.status = status

        with self._lock:
            self._spans.append(span)

        self.bus.emit({
            "topic": self.BUS_TOPICS["span"],
            "kind": "telemetry",
            "actor": "telemetry-module",
            "data": span.to_dict(),
        })

    # =========================================================================
    # Aggregation
    # =========================================================================

    def aggregate_metric(self, name: str) -> Optional[MetricAggregation]:
        """Get aggregated statistics for a metric."""
        with self._lock:
            metrics = self._metrics.get(name, [])
            if not metrics:
                return None

            values = [m.value for m in metrics]

        if len(values) < 2:
            return MetricAggregation(
                name=name,
                count=len(values),
                sum=sum(values),
                min=min(values) if values else 0,
                max=max(values) if values else 0,
                mean=statistics.mean(values) if values else 0,
                stddev=0,
                percentiles={},
            )

        sorted_values = sorted(values)
        n = len(sorted_values)

        percentiles = {
            "p50": sorted_values[int(n * 0.5)],
            "p90": sorted_values[int(n * 0.9)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        }

        return MetricAggregation(
            name=name,
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            stddev=statistics.stdev(values),
            percentiles=percentiles,
        )

    def get_counters(self) -> Dict[str, float]:
        """Get all counter values."""
        with self._lock:
            return dict(self._counters)

    def get_gauges(self) -> Dict[str, float]:
        """Get all gauge values."""
        with self._lock:
            return dict(self._gauges)

    # =========================================================================
    # Export
    # =========================================================================

    def export(self) -> Dict[str, Any]:
        """Export all telemetry data."""
        with self._lock:
            return {
                "events": [e.to_dict() for e in self._events[-100:]],
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "spans": [s.to_dict() for s in self._spans[-100:]],
                "metrics": {
                    name: self.aggregate_metric(name).to_dict() if self.aggregate_metric(name) else None
                    for name in list(self._metrics.keys())[:20]
                },
                "exported_at": time.time(),
            }

    def flush(self) -> None:
        """Flush telemetry to storage."""
        if not self.config.export_enabled:
            return

        data = self.export()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        export_file = self._export_dir / f"telemetry_{timestamp}.json"
        export_file.write_text(json.dumps(data, indent=2))

        self.bus.emit({
            "topic": self.BUS_TOPICS["report"],
            "kind": "telemetry",
            "actor": "telemetry-module",
            "data": {
                "export_file": str(export_file),
                "events": len(data["events"]),
                "counters": len(data["counters"]),
            },
        })

    async def start(self) -> None:
        """Start background flush task."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop background tasks and flush."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        self.flush()

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval_s)
                self.flush()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    # =========================================================================
    # Utilities
    # =========================================================================

    def clear(self) -> None:
        """Clear all telemetry data."""
        with self._lock:
            self._events.clear()
            self._metrics.clear()
            self._spans.clear()
            self._counters.clear()
            self._gauges.clear()

    def stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        with self._lock:
            return {
                "events": len(self._events),
                "metrics": sum(len(v) for v in self._metrics.values()),
                "spans": len(self._spans),
                "counters": len(self._counters),
                "gauges": len(self._gauges),
                "config": self.config.to_dict(),
            }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Telemetry Module."""
    import argparse

    parser = argparse.ArgumentParser(description="Telemetry Module (Step 97)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats command
    subparsers.add_parser("stats", help="Show telemetry statistics")

    # export command
    export_parser = subparsers.add_parser("export", help="Export telemetry")
    export_parser.add_argument("--output", "-o", help="Output file")

    # counters command
    subparsers.add_parser("counters", help="Show counters")

    # gauges command
    subparsers.add_parser("gauges", help="Show gauges")

    # demo command
    subparsers.add_parser("demo", help="Run telemetry demo")

    args = parser.parse_args()
    telemetry = TelemetryModule()

    if args.command == "stats":
        stats = telemetry.stats()
        print(json.dumps(stats, indent=2))
        return 0

    elif args.command == "export":
        data = telemetry.export()
        if args.output:
            Path(args.output).write_text(json.dumps(data, indent=2))
            print(f"Exported to {args.output}")
        else:
            print(json.dumps(data, indent=2))
        return 0

    elif args.command == "counters":
        counters = telemetry.get_counters()
        for name, value in counters.items():
            print(f"{name}: {value}")
        return 0

    elif args.command == "gauges":
        gauges = telemetry.get_gauges()
        for name, value in gauges.items():
            print(f"{name}: {value}")
        return 0

    elif args.command == "demo":
        print("Telemetry Module Demo\n")

        # Track events
        print("1. Tracking events...")
        telemetry.track_event("app.started", {"version": "1.0.0"})
        telemetry.track_event("user.action", {"action": "click", "element": "button"})
        telemetry.track_event("feature.used", {"feature": "export"})

        # Record metrics
        print("2. Recording metrics...")
        telemetry.increment("requests.total")
        telemetry.increment("requests.total")
        telemetry.increment("requests.total")

        telemetry.gauge("connections.active", 42)
        telemetry.gauge("memory.percent", 65.5)

        for i in range(20):
            telemetry.histogram("request.size", 100 + i * 10, unit="bytes")
            telemetry.timer("request.latency", 10 + i * 2)

        # Tracing
        print("3. Creating spans...")
        with telemetry.span("main_operation", {"user_id": "123"}) as span:
            time.sleep(0.05)
            span.attributes["result"] = "success"

            with telemetry.span("sub_operation") as sub_span:
                time.sleep(0.02)
                sub_span.attributes["items"] = 10

        # Show results
        print("\nResults:")
        print(f"\nCounters: {telemetry.get_counters()}")
        print(f"Gauges: {telemetry.get_gauges()}")

        agg = telemetry.aggregate_metric("request.latency")
        if agg:
            print(f"\nRequest Latency Aggregation:")
            print(f"  Count: {agg.count}")
            print(f"  Mean: {agg.mean:.2f}ms")
            print(f"  Min: {agg.min:.2f}ms")
            print(f"  Max: {agg.max:.2f}ms")
            print(f"  P95: {agg.percentiles.get('p95', 0):.2f}ms")

        print("\nStatistics:")
        print(json.dumps(telemetry.stats(), indent=2))

        # Export
        print("\nExporting...")
        telemetry.flush()
        print("Done!")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
