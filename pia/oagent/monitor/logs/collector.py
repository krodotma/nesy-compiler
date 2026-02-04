#!/usr/bin/env python3
"""
Log Collector - Step 254

Collects logs from all agents via bus event subscriptions.

PBTSO Phase: ITERATE

Bus Topics:
- *.log (subscribed)
- monitor.logs.collected (emitted)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Parse log level from string."""
        level_lower = level.lower()
        mapping = {
            "debug": cls.DEBUG,
            "info": cls.INFO,
            "warn": cls.WARN,
            "warning": cls.WARN,
            "error": cls.ERROR,
            "err": cls.ERROR,
            "fatal": cls.FATAL,
            "critical": cls.FATAL,
            "crit": cls.FATAL,
        }
        return mapping.get(level_lower, cls.INFO)

    def __lt__(self, other: "LogLevel") -> bool:
        order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR, LogLevel.FATAL]
        return order.index(self) < order.index(other)


@dataclass
class LogEntry:
    """A single log entry.

    Attributes:
        message: Log message
        level: Severity level
        timestamp: Unix timestamp
        source: Source agent/service
        trace_id: Distributed trace ID (optional)
        span_id: Span ID (optional)
        labels: Additional labels/tags
        raw: Raw log line if parsed
    """
    message: str
    level: LogLevel
    timestamp: float
    source: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "labels": self.labels,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create from dictionary."""
        return cls(
            message=data.get("message", ""),
            level=LogLevel.from_string(data.get("level", "info")),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "unknown"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            labels=data.get("labels", {}),
            raw=data.get("raw"),
        )

    @classmethod
    def from_bus_event(cls, event: Dict[str, Any]) -> "LogEntry":
        """Create from bus event."""
        data = event.get("data", {})
        return cls(
            message=data.get("message", data.get("msg", str(data))),
            level=LogLevel.from_string(event.get("level", data.get("level", "info"))),
            timestamp=event.get("ts", time.time()),
            source=event.get("actor", "unknown"),
            trace_id=data.get("trace_id", data.get("traceId")),
            span_id=data.get("span_id", data.get("spanId")),
            labels={
                "topic": event.get("topic", ""),
                "host": event.get("host", ""),
                "kind": event.get("kind", ""),
            },
            raw=json.dumps(event),
        )


class LogCollector:
    """
    Collect and store logs from all agents.

    The collector:
    - Receives logs via *.log and *.error bus subscriptions
    - Stores entries in a ring buffer
    - Provides search and filter capabilities
    - Emits collection events

    Example:
        collector = LogCollector(max_entries=10000)

        # Record a log entry
        collector.record(LogEntry(
            message="Request completed",
            level=LogLevel.INFO,
            timestamp=time.time(),
            source="code-agent",
        ))

        # Search logs
        errors = collector.search(level=LogLevel.ERROR)
    """

    BUS_TOPICS = {
        "collected": "monitor.logs.collected",
        "error": "monitor.logs.error",
    }

    def __init__(
        self,
        max_entries: int = 100000,
        retention_hours: int = 24,
        bus_dir: Optional[str] = None
    ):
        """Initialize log collector.

        Args:
            max_entries: Maximum log entries to retain
            retention_hours: Hours to retain logs
            bus_dir: Directory for bus events
        """
        self.max_entries = max_entries
        self.retention_hours = retention_hours

        # Ring buffer for logs
        self._logs: Deque[LogEntry] = deque(maxlen=max_entries)
        self._lock = threading.RLock()
        self._record_count: int = 0

        # Indices for fast lookup
        self._by_source: Dict[str, List[int]] = {}
        self._by_level: Dict[LogLevel, List[int]] = {}
        self._by_trace: Dict[str, List[int]] = {}

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: LogEntry) -> bool:
        """Record a log entry.

        Args:
            entry: Log entry to record

        Returns:
            True if recorded
        """
        with self._lock:
            idx = len(self._logs)
            self._logs.append(entry)
            self._record_count += 1

            # Update indices (note: indices become stale when buffer wraps)
            if entry.source not in self._by_source:
                self._by_source[entry.source] = []
            self._by_source[entry.source].append(idx)

            if entry.level not in self._by_level:
                self._by_level[entry.level] = []
            self._by_level[entry.level].append(idx)

            if entry.trace_id:
                if entry.trace_id not in self._by_trace:
                    self._by_trace[entry.trace_id] = []
                self._by_trace[entry.trace_id].append(idx)

            # Periodically clean up indices
            if self._record_count % 10000 == 0:
                self._cleanup_indices()

        return True

    def record_batch(self, entries: List[LogEntry]) -> int:
        """Record multiple log entries.

        Args:
            entries: Log entries to record

        Returns:
            Number recorded
        """
        recorded = 0
        for entry in entries:
            if self.record(entry):
                recorded += 1
        return recorded

    def search(
        self,
        pattern: Optional[str] = None,
        level: Optional[LogLevel] = None,
        min_level: Optional[LogLevel] = None,
        source: Optional[str] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[LogEntry]:
        """Search log entries.

        Args:
            pattern: Regex pattern to match in message
            level: Exact level match
            min_level: Minimum level (inclusive)
            source: Source filter
            trace_id: Trace ID filter
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching log entries
        """
        with self._lock:
            results: List[LogEntry] = []
            regex = re.compile(pattern, re.IGNORECASE) if pattern else None

            for entry in reversed(list(self._logs)):  # Most recent first
                # Apply filters
                if level and entry.level != level:
                    continue
                if min_level and entry.level < min_level:
                    continue
                if source and entry.source != source:
                    continue
                if trace_id and entry.trace_id != trace_id:
                    continue
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                if regex and not regex.search(entry.message):
                    continue

                results.append(entry)

                if len(results) >= offset + limit:
                    break

            return results[offset:offset + limit]

    def get_by_trace(self, trace_id: str) -> List[LogEntry]:
        """Get all logs for a trace ID.

        Args:
            trace_id: Trace ID

        Returns:
            List of log entries
        """
        return self.search(trace_id=trace_id, limit=1000)

    def get_recent(self, count: int = 100) -> List[LogEntry]:
        """Get most recent logs.

        Args:
            count: Number of logs

        Returns:
            List of log entries
        """
        with self._lock:
            return list(self._logs)[-count:]

    def get_errors(self, hours: float = 1.0) -> List[LogEntry]:
        """Get recent error logs.

        Args:
            hours: Time window in hours

        Returns:
            List of error entries
        """
        start_time = time.time() - (hours * 3600)
        return self.search(min_level=LogLevel.ERROR, start_time=start_time, limit=1000)

    def count_by_level(self, window_s: int = 3600) -> Dict[str, int]:
        """Count logs by level in time window.

        Args:
            window_s: Time window in seconds

        Returns:
            Dictionary of level -> count
        """
        cutoff = time.time() - window_s
        counts: Dict[str, int] = {level.value: 0 for level in LogLevel}

        with self._lock:
            for entry in self._logs:
                if entry.timestamp >= cutoff:
                    counts[entry.level.value] += 1

        return counts

    def count_by_source(self, window_s: int = 3600) -> Dict[str, int]:
        """Count logs by source in time window.

        Args:
            window_s: Time window in seconds

        Returns:
            Dictionary of source -> count
        """
        cutoff = time.time() - window_s
        counts: Dict[str, int] = {}

        with self._lock:
            for entry in self._logs:
                if entry.timestamp >= cutoff:
                    counts[entry.source] = counts.get(entry.source, 0) + 1

        return counts

    def get_sources(self) -> Set[str]:
        """Get all unique sources.

        Returns:
            Set of source names
        """
        with self._lock:
            return set(self._by_source.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "total_entries": len(self._logs),
                "records_received": self._record_count,
                "unique_sources": len(self._by_source),
                "unique_traces": len(self._by_trace),
                "counts_by_level": self.count_by_level(3600),
                "buffer_capacity": self.max_entries,
                "buffer_utilization": len(self._logs) / self.max_entries if self.max_entries else 0,
            }

    def handle_log_event(self, event: Dict[str, Any]) -> bool:
        """Handle incoming log bus event.

        Args:
            event: Bus event

        Returns:
            True if handled
        """
        entry = LogEntry.from_bus_event(event)
        return self.record(entry)

    def emit_collected_event(self, count: int) -> str:
        """Emit logs collected event.

        Args:
            count: Number of logs collected

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS["collected"],
            "kind": "event",
            "level": "info",
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "logs_collected": count,
                "statistics": self.get_statistics(),
            },
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _cleanup_indices(self) -> None:
        """Clean up stale index entries."""
        # Simple cleanup: just clear indices and rely on search filtering
        # More sophisticated implementations could maintain accurate indices
        current_len = len(self._logs)
        for source_indices in self._by_source.values():
            source_indices[:] = [i for i in source_indices if i < current_len]
        for level_indices in self._by_level.values():
            level_indices[:] = [i for i in level_indices if i < current_len]
        for trace_indices in self._by_trace.values():
            trace_indices[:] = [i for i in trace_indices if i < current_len]


# Singleton instance
_collector: Optional[LogCollector] = None


def get_collector() -> LogCollector:
    """Get or create the log collector singleton.

    Returns:
        LogCollector instance
    """
    global _collector
    if _collector is None:
        _collector = LogCollector()
    return _collector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log Collector (Step 254)")
    parser.add_argument("--search", metavar="PATTERN", help="Search logs")
    parser.add_argument("--level", help="Filter by level")
    parser.add_argument("--source", help="Filter by source")
    parser.add_argument("--errors", action="store_true", help="Show recent errors")
    parser.add_argument("--recent", type=int, help="Show N recent logs")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    collector = get_collector()

    if args.search:
        level = LogLevel.from_string(args.level) if args.level else None
        results = collector.search(pattern=args.search, level=level, source=args.source)
        if args.json:
            print(json.dumps([e.to_dict() for e in results], indent=2))
        else:
            for entry in results:
                print(f"[{entry.level.value.upper()}] {entry.source}: {entry.message}")

    if args.errors:
        errors = collector.get_errors()
        if args.json:
            print(json.dumps([e.to_dict() for e in errors], indent=2))
        else:
            for entry in errors:
                print(f"[{entry.level.value.upper()}] {entry.source}: {entry.message}")

    if args.recent:
        recent = collector.get_recent(args.recent)
        if args.json:
            print(json.dumps([e.to_dict() for e in recent], indent=2))
        else:
            for entry in recent:
                print(f"[{entry.level.value.upper()}] {entry.source}: {entry.message}")

    if args.stats:
        stats = collector.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Log Collector Statistics:")
            print(f"  Total entries: {stats['total_entries']}")
            print(f"  Records received: {stats['records_received']}")
            print(f"  Unique sources: {stats['unique_sources']}")
            print(f"  Buffer utilization: {stats['buffer_utilization']:.1%}")
