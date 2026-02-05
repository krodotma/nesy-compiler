#!/usr/bin/env python3
"""
Log Correlator - Step 256

Correlates logs across services using trace IDs, timestamps, and patterns.

PBTSO Phase: RESEARCH

Bus Topics:
- monitor.logs.correlate (subscribed)
- monitor.logs.correlated (emitted)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .collector import LogCollector, LogEntry, LogLevel, get_collector


@dataclass
class CorrelatedEvent:
    """An event correlated across services.

    Attributes:
        event_id: Unique correlation ID
        trace_id: Distributed trace ID
        entries: Correlated log entries
        services: Services involved
        root_cause: Identified root cause (if any)
        timeline: Ordered timeline of events
        correlation_score: Confidence score (0-1)
    """
    event_id: str
    trace_id: Optional[str]
    entries: List[LogEntry] = field(default_factory=list)
    services: Set[str] = field(default_factory=set)
    root_cause: Optional[str] = None
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    correlation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "entry_count": len(self.entries),
            "services": list(self.services),
            "root_cause": self.root_cause,
            "timeline": self.timeline,
            "correlation_score": self.correlation_score,
        }


@dataclass
class CorrelationChain:
    """A chain of correlated events.

    Attributes:
        chain_id: Unique chain ID
        events: Events in the chain
        start_time: Chain start timestamp
        end_time: Chain end timestamp
        services: All services in chain
        error_count: Number of errors in chain
    """
    chain_id: str
    events: List[CorrelatedEvent] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    services: Set[str] = field(default_factory=set)
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain_id,
            "event_count": len(self.events),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (
                int((self.end_time - self.start_time) * 1000)
                if self.start_time and self.end_time else 0
            ),
            "services": list(self.services),
            "error_count": self.error_count,
        }


class LogCorrelator:
    """
    Correlate logs across services.

    The correlator:
    - Groups logs by trace ID
    - Identifies related logs by time proximity
    - Builds event timelines
    - Identifies potential root causes

    Example:
        correlator = LogCorrelator()

        # Correlate by trace ID
        chain = correlator.correlate_by_trace("trace-123")

        # Find related errors
        events = correlator.correlate_errors(window_s=300)
    """

    BUS_TOPICS = {
        "correlate": "monitor.logs.correlate",
        "correlated": "monitor.logs.correlated",
    }

    def __init__(
        self,
        collector: Optional[LogCollector] = None,
        time_threshold_ms: int = 1000,
        bus_dir: Optional[str] = None
    ):
        """Initialize log correlator.

        Args:
            collector: Log collector to use
            time_threshold_ms: Time threshold for correlation
            bus_dir: Directory for bus events
        """
        self._collector = collector or get_collector()
        self._time_threshold_s = time_threshold_ms / 1000.0
        self._correlation_cache: Dict[str, CorrelationChain] = {}

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def correlate_by_trace(self, trace_id: str) -> CorrelationChain:
        """Correlate all logs for a trace ID.

        Args:
            trace_id: Trace ID

        Returns:
            Correlation chain
        """
        logs = self._collector.get_by_trace(trace_id)

        if not logs:
            return CorrelationChain(chain_id=f"chain-{uuid.uuid4().hex[:8]}")

        # Sort by timestamp
        logs.sort(key=lambda e: e.timestamp)

        chain = CorrelationChain(
            chain_id=f"chain-{uuid.uuid4().hex[:8]}",
            start_time=logs[0].timestamp,
            end_time=logs[-1].timestamp,
        )

        # Create correlated event
        event = CorrelatedEvent(
            event_id=f"event-{uuid.uuid4().hex[:8]}",
            trace_id=trace_id,
            entries=logs,
            correlation_score=1.0,  # Perfect correlation by trace ID
        )

        # Build timeline
        for entry in logs:
            event.services.add(entry.source)
            event.timeline.append({
                "timestamp": entry.timestamp,
                "source": entry.source,
                "level": entry.level.value,
                "message": entry.message[:200],
            })

            if entry.level >= LogLevel.ERROR:
                chain.error_count += 1

        chain.events.append(event)
        chain.services = event.services

        # Cache result
        self._correlation_cache[trace_id] = chain

        return chain

    def correlate_errors(
        self,
        window_s: int = 300,
        source: Optional[str] = None
    ) -> List[CorrelatedEvent]:
        """Correlate error events by time proximity.

        Args:
            window_s: Time window
            source: Optional source filter

        Returns:
            List of correlated events
        """
        # Get errors in window
        errors = self._collector.get_errors(hours=window_s / 3600)

        if source:
            errors = [e for e in errors if e.source == source]

        if not errors:
            return []

        # Group by time proximity
        errors.sort(key=lambda e: e.timestamp)
        groups: List[List[LogEntry]] = []
        current_group: List[LogEntry] = []

        for entry in errors:
            if not current_group:
                current_group = [entry]
            elif entry.timestamp - current_group[-1].timestamp <= self._time_threshold_s:
                current_group.append(entry)
            else:
                groups.append(current_group)
                current_group = [entry]

        if current_group:
            groups.append(current_group)

        # Convert groups to correlated events
        events = []
        for group in groups:
            if len(group) >= 2:  # Only correlate if multiple entries
                event = self._create_correlated_event(group)
                events.append(event)

        return events

    def correlate_service_chain(
        self,
        service: str,
        window_s: int = 60
    ) -> List[CorrelationChain]:
        """Find chains of events across services starting from one service.

        Args:
            service: Starting service
            window_s: Time window to search

        Returns:
            List of correlation chains
        """
        start_time = time.time() - window_s
        logs = self._collector.search(
            source=service,
            start_time=start_time,
            limit=1000
        )

        chains = []

        for entry in logs:
            # If log has trace ID, correlate by that
            if entry.trace_id:
                chain = self.correlate_by_trace(entry.trace_id)
                if len(chain.services) > 1:
                    chains.append(chain)
            else:
                # Try to find related logs by time
                related = self._find_related_logs(entry, window_s=5)
                if len(related) > 1:
                    event = self._create_correlated_event(related)
                    chain = CorrelationChain(
                        chain_id=f"chain-{uuid.uuid4().hex[:8]}",
                        events=[event],
                        services=event.services,
                        start_time=min(e.timestamp for e in related),
                        end_time=max(e.timestamp for e in related),
                        error_count=sum(1 for e in related if e.level >= LogLevel.ERROR),
                    )
                    chains.append(chain)

        return chains

    def find_root_cause(
        self,
        chain: CorrelationChain
    ) -> Optional[str]:
        """Attempt to identify root cause in a correlation chain.

        Args:
            chain: Correlation chain

        Returns:
            Root cause description or None
        """
        if not chain.events:
            return None

        # Find first error in chain
        all_entries: List[Tuple[float, LogEntry]] = []
        for event in chain.events:
            for entry in event.entries:
                all_entries.append((entry.timestamp, entry))

        all_entries.sort(key=lambda x: x[0])

        # Look for first error
        for _, entry in all_entries:
            if entry.level >= LogLevel.ERROR:
                return f"First error from {entry.source}: {entry.message[:100]}"

        return None

    def get_service_dependencies(
        self,
        window_s: int = 3600
    ) -> Dict[str, Set[str]]:
        """Infer service dependencies from log correlations.

        Args:
            window_s: Time window

        Returns:
            Dictionary of service -> dependent services
        """
        dependencies: Dict[str, Set[str]] = defaultdict(set)

        # Get all traces in window
        start_time = time.time() - window_s
        logs = self._collector.search(start_time=start_time, limit=100000)

        # Group by trace ID
        by_trace: Dict[str, List[LogEntry]] = defaultdict(list)
        for entry in logs:
            if entry.trace_id:
                by_trace[entry.trace_id].append(entry)

        # Infer dependencies from trace order
        for trace_id, entries in by_trace.items():
            if len(entries) < 2:
                continue

            entries.sort(key=lambda e: e.timestamp)
            sources = [e.source for e in entries]

            # Each subsequent service depends on the previous
            for i in range(1, len(sources)):
                if sources[i] != sources[i-1]:
                    dependencies[sources[i]].add(sources[i-1])

        return dict(dependencies)

    def handle_correlate_request(
        self,
        event: Dict[str, Any]
    ) -> Optional[CorrelationChain]:
        """Handle correlation request from bus.

        Args:
            event: Bus event

        Returns:
            Correlation chain or None
        """
        data = event.get("data", {})
        trace_id = data.get("trace_id")
        service = data.get("service")
        window_s = data.get("window_s", 300)

        if trace_id:
            chain = self.correlate_by_trace(trace_id)
        elif service:
            chains = self.correlate_service_chain(service, window_s)
            chain = chains[0] if chains else None
        else:
            events = self.correlate_errors(window_s)
            if events:
                chain = CorrelationChain(
                    chain_id=f"chain-{uuid.uuid4().hex[:8]}",
                    events=events,
                )
            else:
                chain = None

        if chain:
            self.emit_correlated_event(chain)

        return chain

    def emit_correlated_event(self, chain: CorrelationChain) -> str:
        """Emit correlation result event.

        Args:
            chain: Correlation chain

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS["correlated"],
            "kind": "event",
            "level": "info",
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "chain": chain.to_dict(),
                "events": [e.to_dict() for e in chain.events],
            },
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _create_correlated_event(
        self,
        entries: List[LogEntry]
    ) -> CorrelatedEvent:
        """Create a correlated event from log entries.

        Args:
            entries: Log entries

        Returns:
            Correlated event
        """
        event = CorrelatedEvent(
            event_id=f"event-{uuid.uuid4().hex[:8]}",
            trace_id=entries[0].trace_id if entries else None,
            entries=entries,
        )

        # Calculate correlation score based on factors
        factors = []

        # Same trace ID
        trace_ids = set(e.trace_id for e in entries if e.trace_id)
        if len(trace_ids) == 1:
            factors.append(1.0)
        else:
            factors.append(0.5)

        # Time proximity (within threshold)
        if len(entries) >= 2:
            time_spread = entries[-1].timestamp - entries[0].timestamp
            if time_spread <= self._time_threshold_s:
                factors.append(1.0)
            else:
                factors.append(max(0.3, 1.0 - time_spread / 10.0))

        # Build timeline
        for entry in entries:
            event.services.add(entry.source)
            event.timeline.append({
                "timestamp": entry.timestamp,
                "source": entry.source,
                "level": entry.level.value,
                "message": entry.message[:200],
            })

        event.correlation_score = sum(factors) / len(factors) if factors else 0.5

        return event

    def _find_related_logs(
        self,
        entry: LogEntry,
        window_s: int = 5
    ) -> List[LogEntry]:
        """Find logs related to an entry by time.

        Args:
            entry: Reference entry
            window_s: Time window

        Returns:
            Related log entries
        """
        start_time = entry.timestamp - window_s
        end_time = entry.timestamp + window_s

        related = self._collector.search(
            start_time=start_time,
            end_time=end_time,
            limit=100
        )

        # Filter to only error-level or same source
        return [
            e for e in related
            if e.level >= LogLevel.WARN or e.source == entry.source
        ]


# Singleton instance
_correlator: Optional[LogCorrelator] = None


def get_correlator() -> LogCorrelator:
    """Get or create the log correlator singleton.

    Returns:
        LogCorrelator instance
    """
    global _correlator
    if _correlator is None:
        _correlator = LogCorrelator()
    return _correlator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log Correlator (Step 256)")
    parser.add_argument("--trace", metavar="ID", help="Correlate by trace ID")
    parser.add_argument("--service", help="Find chains for service")
    parser.add_argument("--errors", action="store_true", help="Correlate errors")
    parser.add_argument("--deps", action="store_true", help="Show service dependencies")
    parser.add_argument("--window", type=int, default=300, help="Window in seconds")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    correlator = get_correlator()

    if args.trace:
        chain = correlator.correlate_by_trace(args.trace)
        if args.json:
            print(json.dumps(chain.to_dict(), indent=2))
        else:
            print(f"Correlation Chain: {chain.chain_id}")
            print(f"  Services: {', '.join(chain.services)}")
            print(f"  Events: {len(chain.events)}")
            print(f"  Errors: {chain.error_count}")

    if args.service:
        chains = correlator.correlate_service_chain(args.service, args.window)
        if args.json:
            print(json.dumps([c.to_dict() for c in chains], indent=2))
        else:
            print(f"Found {len(chains)} correlation chains for {args.service}")
            for c in chains[:5]:
                print(f"  - {c.chain_id}: {len(c.services)} services, {c.error_count} errors")

    if args.errors:
        events = correlator.correlate_errors(args.window)
        if args.json:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            print(f"Found {len(events)} correlated error events")
            for e in events[:5]:
                print(f"  - {e.event_id}: {len(e.entries)} entries, score={e.correlation_score:.2f}")

    if args.deps:
        deps = correlator.get_service_dependencies(args.window)
        if args.json:
            print(json.dumps({k: list(v) for k, v in deps.items()}, indent=2))
        else:
            print("Service Dependencies:")
            for service, depends_on in deps.items():
                print(f"  {service} -> {', '.join(depends_on)}")
