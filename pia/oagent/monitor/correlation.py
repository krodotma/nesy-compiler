#!/usr/bin/env python3
"""
Monitor Correlation Engine - Step 274

Event correlation and pattern detection across monitoring data.

PBTSO Phase: RESEARCH

Bus Topics:
- monitor.correlation.detect (subscribed)
- monitor.correlation.found (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class CorrelationType(Enum):
    """Types of correlations."""
    TEMPORAL = "temporal"       # Time-based correlation
    CAUSAL = "causal"          # Cause-effect relationship
    STATISTICAL = "statistical" # Statistical correlation
    SEMANTIC = "semantic"       # Meaning-based correlation
    TOPOLOGICAL = "topological" # Structure-based correlation


class CorrelationStrength(Enum):
    """Correlation strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class Event:
    """A monitoring event for correlation.

    Attributes:
        event_id: Unique event ID
        event_type: Type of event
        source: Event source
        timestamp: Event timestamp
        data: Event data
        labels: Event labels
        severity: Event severity
    """
    event_id: str
    event_type: str
    source: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp,
            "data": self.data,
            "labels": self.labels,
            "severity": self.severity,
        }


@dataclass
class Correlation:
    """A detected correlation between events.

    Attributes:
        correlation_id: Unique correlation ID
        correlation_type: Type of correlation
        events: Correlated events
        strength: Correlation strength
        score: Correlation score (0-1)
        description: Correlation description
        root_event: Potential root cause event
        detected_at: Detection timestamp
        metadata: Additional metadata
    """
    correlation_id: str
    correlation_type: CorrelationType
    events: List[Event] = field(default_factory=list)
    strength: CorrelationStrength = CorrelationStrength.WEAK
    score: float = 0.0
    description: str = ""
    root_event: Optional[Event] = None
    detected_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "correlation_type": self.correlation_type.value,
            "event_count": len(self.events),
            "event_ids": [e.event_id for e in self.events],
            "strength": self.strength.value,
            "score": self.score,
            "description": self.description,
            "root_event_id": self.root_event.event_id if self.root_event else None,
            "detected_at": self.detected_at,
            "metadata": self.metadata,
        }


@dataclass
class CorrelationRule:
    """A rule for detecting correlations.

    Attributes:
        rule_id: Unique rule ID
        name: Rule name
        correlation_type: Type of correlation to detect
        conditions: Rule conditions
        time_window_s: Time window for correlation
        min_events: Minimum events required
        enabled: Whether rule is enabled
    """
    rule_id: str
    name: str
    correlation_type: CorrelationType
    conditions: Dict[str, Any] = field(default_factory=dict)
    time_window_s: int = 300
    min_events: int = 2
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "correlation_type": self.correlation_type.value,
            "conditions": self.conditions,
            "time_window_s": self.time_window_s,
            "min_events": self.min_events,
            "enabled": self.enabled,
        }


class CorrelationEngine:
    """
    Correlate events and detect patterns.

    The engine:
    - Detects temporal correlations between events
    - Identifies causal relationships
    - Calculates statistical correlations
    - Maintains correlation history

    Example:
        engine = CorrelationEngine()

        # Register a correlation rule
        rule = CorrelationRule(
            rule_id="error-cascade",
            name="Error Cascade Detection",
            correlation_type=CorrelationType.CAUSAL,
            conditions={"severity": "error"},
        )
        engine.register_rule(rule)

        # Add events
        engine.add_event(Event(
            event_id="evt-1",
            event_type="error",
            source="service-a",
            timestamp=time.time(),
        ))

        # Detect correlations
        correlations = engine.detect_correlations()
    """

    BUS_TOPICS = {
        "detect": "monitor.correlation.detect",
        "found": "monitor.correlation.found",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        event_buffer_size: int = 10000,
        bus_dir: Optional[str] = None,
    ):
        """Initialize correlation engine.

        Args:
            event_buffer_size: Maximum events to buffer
            bus_dir: Bus directory
        """
        self._event_buffer_size = event_buffer_size
        self._events: List[Event] = []
        self._rules: Dict[str, CorrelationRule] = {}
        self._correlations: Dict[str, Correlation] = {}
        self._correlation_history: List[Correlation] = []
        self._last_heartbeat = time.time()

        # Indexes for fast lookup
        self._events_by_source: Dict[str, List[Event]] = defaultdict(list)
        self._events_by_type: Dict[str, List[Event]] = defaultdict(list)

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default rules
        self._register_default_rules()

    def add_event(self, event: Event) -> None:
        """Add an event to the correlation buffer.

        Args:
            event: Event to add
        """
        self._events.append(event)
        self._events_by_source[event.source].append(event)
        self._events_by_type[event.event_type].append(event)

        # Prune if buffer is full
        if len(self._events) > self._event_buffer_size:
            self._prune_events()

    def add_events(self, events: List[Event]) -> int:
        """Add multiple events.

        Args:
            events: Events to add

        Returns:
            Number of events added
        """
        for event in events:
            self.add_event(event)
        return len(events)

    def register_rule(self, rule: CorrelationRule) -> None:
        """Register a correlation rule.

        Args:
            rule: Rule to register
        """
        self._rules[rule.rule_id] = rule

    def unregister_rule(self, rule_id: str) -> bool:
        """Unregister a rule.

        Args:
            rule_id: Rule ID

        Returns:
            True if removed
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[CorrelationRule]:
        """Get a rule by ID.

        Args:
            rule_id: Rule ID

        Returns:
            Rule or None
        """
        return self._rules.get(rule_id)

    def list_rules(self) -> List[Dict[str, Any]]:
        """List all rules.

        Returns:
            Rule summaries
        """
        return [r.to_dict() for r in self._rules.values()]

    def detect_correlations(
        self,
        time_window_s: Optional[int] = None,
        correlation_type: Optional[CorrelationType] = None,
    ) -> List[Correlation]:
        """Detect correlations in the event buffer.

        Args:
            time_window_s: Time window to analyze
            correlation_type: Filter by correlation type

        Returns:
            Detected correlations
        """
        correlations = []

        # Apply each enabled rule
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if correlation_type and rule.correlation_type != correlation_type:
                continue

            window = time_window_s or rule.time_window_s
            detected = self._apply_rule(rule, window)
            correlations.extend(detected)

        # Store correlations
        for correlation in correlations:
            self._correlations[correlation.correlation_id] = correlation
            self._correlation_history.append(correlation)

            self._emit_bus_event(
                self.BUS_TOPICS["found"],
                correlation.to_dict()
            )

        # Prune history
        if len(self._correlation_history) > 1000:
            self._correlation_history = self._correlation_history[-1000:]

        return correlations

    def detect_temporal_correlation(
        self,
        events: List[Event],
        window_s: int = 60,
    ) -> Optional[Correlation]:
        """Detect temporal correlation between events.

        Args:
            events: Events to correlate
            window_s: Time window

        Returns:
            Correlation or None
        """
        if len(events) < 2:
            return None

        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Check if events are within window
        time_span = sorted_events[-1].timestamp - sorted_events[0].timestamp
        if time_span > window_s:
            return None

        # Calculate correlation score based on temporal proximity
        intervals = []
        for i in range(1, len(sorted_events)):
            intervals.append(sorted_events[i].timestamp - sorted_events[i-1].timestamp)

        avg_interval = sum(intervals) / len(intervals) if intervals else window_s
        regularity = 1.0 - (max(intervals) - min(intervals)) / window_s if intervals and window_s > 0 else 0.0

        score = (1.0 - avg_interval / window_s) * 0.5 + max(0, regularity) * 0.5

        strength = self._score_to_strength(score)

        return Correlation(
            correlation_id=f"corr-{uuid.uuid4().hex[:8]}",
            correlation_type=CorrelationType.TEMPORAL,
            events=sorted_events,
            strength=strength,
            score=score,
            description=f"Temporal correlation: {len(events)} events within {time_span:.1f}s",
            root_event=sorted_events[0],
        )

    def detect_causal_correlation(
        self,
        events: List[Event],
    ) -> Optional[Correlation]:
        """Detect causal correlation between events.

        Args:
            events: Events to correlate

        Returns:
            Correlation or None
        """
        if len(events) < 2:
            return None

        # Sort by timestamp to establish order
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Look for error/warning propagation patterns
        error_events = [e for e in sorted_events if e.severity in ("error", "critical")]
        if not error_events:
            return None

        # First error is potential root cause
        root = error_events[0]

        # Calculate causal score based on severity escalation and timing
        severity_scores = {"info": 1, "warn": 2, "warning": 2, "error": 3, "critical": 4}
        escalation = 0
        for i in range(1, len(sorted_events)):
            prev_score = severity_scores.get(sorted_events[i-1].severity, 1)
            curr_score = severity_scores.get(sorted_events[i].severity, 1)
            if curr_score >= prev_score:
                escalation += 1

        escalation_ratio = escalation / (len(sorted_events) - 1) if len(sorted_events) > 1 else 0

        # Check source diversity (errors spreading across sources)
        sources = set(e.source for e in error_events)
        diversity_score = min(1.0, len(sources) / 3)  # Cap at 3 sources

        score = escalation_ratio * 0.5 + diversity_score * 0.5
        strength = self._score_to_strength(score)

        return Correlation(
            correlation_id=f"corr-{uuid.uuid4().hex[:8]}",
            correlation_type=CorrelationType.CAUSAL,
            events=sorted_events,
            strength=strength,
            score=score,
            description=f"Potential causal chain from {root.source}",
            root_event=root,
            metadata={
                "escalation_ratio": escalation_ratio,
                "source_diversity": len(sources),
            },
        )

    def detect_statistical_correlation(
        self,
        metric_a: str,
        metric_b: str,
        values_a: List[float],
        values_b: List[float],
    ) -> Optional[Correlation]:
        """Detect statistical correlation between metrics.

        Args:
            metric_a: First metric name
            metric_b: Second metric name
            values_a: Values for metric A
            values_b: Values for metric B

        Returns:
            Correlation or None
        """
        if len(values_a) != len(values_b) or len(values_a) < 3:
            return None

        # Calculate Pearson correlation coefficient
        n = len(values_a)
        mean_a = sum(values_a) / n
        mean_b = sum(values_b) / n

        numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))

        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in values_a) / n)
        std_b = math.sqrt(sum((b - mean_b) ** 2 for b in values_b) / n)

        if std_a == 0 or std_b == 0:
            return None

        correlation = numerator / (n * std_a * std_b)

        # Convert to 0-1 score (absolute value)
        score = abs(correlation)
        strength = self._score_to_strength(score)

        direction = "positive" if correlation > 0 else "negative"

        return Correlation(
            correlation_id=f"corr-{uuid.uuid4().hex[:8]}",
            correlation_type=CorrelationType.STATISTICAL,
            strength=strength,
            score=score,
            description=f"Statistical {direction} correlation between {metric_a} and {metric_b}",
            metadata={
                "metric_a": metric_a,
                "metric_b": metric_b,
                "pearson_r": correlation,
                "sample_size": n,
            },
        )

    def get_correlation(self, correlation_id: str) -> Optional[Correlation]:
        """Get a correlation by ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            Correlation or None
        """
        return self._correlations.get(correlation_id)

    def get_correlations_for_event(self, event_id: str) -> List[Correlation]:
        """Get all correlations involving an event.

        Args:
            event_id: Event ID

        Returns:
            Correlations
        """
        return [
            c for c in self._correlations.values()
            if any(e.event_id == event_id for e in c.events)
        ]

    def get_recent_correlations(
        self,
        limit: int = 20,
        min_strength: Optional[CorrelationStrength] = None,
    ) -> List[Correlation]:
        """Get recent correlations.

        Args:
            limit: Maximum results
            min_strength: Minimum strength filter

        Returns:
            Correlations
        """
        correlations = self._correlation_history

        if min_strength:
            strength_order = [
                CorrelationStrength.WEAK,
                CorrelationStrength.MODERATE,
                CorrelationStrength.STRONG,
                CorrelationStrength.VERY_STRONG,
            ]
            min_index = strength_order.index(min_strength)
            correlations = [
                c for c in correlations
                if strength_order.index(c.strength) >= min_index
            ]

        return list(reversed(correlations[-limit:]))

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics
        """
        by_type: Dict[str, int] = {}
        by_strength: Dict[str, int] = {}

        for c in self._correlation_history:
            ctype = c.correlation_type.value
            by_type[ctype] = by_type.get(ctype, 0) + 1

            strength = c.strength.value
            by_strength[strength] = by_strength.get(strength, 0) + 1

        return {
            "events_buffered": len(self._events),
            "rules_registered": len(self._rules),
            "total_correlations": len(self._correlation_history),
            "by_type": by_type,
            "by_strength": by_strength,
            "unique_sources": len(self._events_by_source),
        }

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "correlation_engine",
                "status": "healthy",
                "events": len(self._events),
                "correlations": len(self._correlations),
            }
        )

        return True

    def _apply_rule(
        self,
        rule: CorrelationRule,
        window_s: int,
    ) -> List[Correlation]:
        """Apply a correlation rule.

        Args:
            rule: Rule to apply
            window_s: Time window

        Returns:
            Detected correlations
        """
        correlations = []
        now = time.time()
        cutoff = now - window_s

        # Filter events in window
        events_in_window = [e for e in self._events if e.timestamp >= cutoff]

        # Apply conditions
        filtered_events = self._filter_by_conditions(events_in_window, rule.conditions)

        if len(filtered_events) < rule.min_events:
            return correlations

        # Detect correlation based on type
        if rule.correlation_type == CorrelationType.TEMPORAL:
            correlation = self.detect_temporal_correlation(filtered_events, window_s)
            if correlation:
                correlations.append(correlation)

        elif rule.correlation_type == CorrelationType.CAUSAL:
            correlation = self.detect_causal_correlation(filtered_events)
            if correlation:
                correlations.append(correlation)

        return correlations

    def _filter_by_conditions(
        self,
        events: List[Event],
        conditions: Dict[str, Any],
    ) -> List[Event]:
        """Filter events by conditions.

        Args:
            events: Events to filter
            conditions: Filter conditions

        Returns:
            Filtered events
        """
        if not conditions:
            return events

        filtered = []
        for event in events:
            match = True

            if "event_type" in conditions:
                if event.event_type != conditions["event_type"]:
                    match = False

            if "source" in conditions:
                if event.source != conditions["source"]:
                    match = False

            if "severity" in conditions:
                severities = conditions["severity"]
                if isinstance(severities, str):
                    severities = [severities]
                if event.severity not in severities:
                    match = False

            if "labels" in conditions:
                for k, v in conditions["labels"].items():
                    if event.labels.get(k) != v:
                        match = False

            if match:
                filtered.append(event)

        return filtered

    def _score_to_strength(self, score: float) -> CorrelationStrength:
        """Convert score to strength."""
        if score >= 0.9:
            return CorrelationStrength.VERY_STRONG
        elif score >= 0.7:
            return CorrelationStrength.STRONG
        elif score >= 0.4:
            return CorrelationStrength.MODERATE
        else:
            return CorrelationStrength.WEAK

    def _prune_events(self) -> int:
        """Prune old events from buffer.

        Returns:
            Number of events pruned
        """
        if len(self._events) <= self._event_buffer_size:
            return 0

        to_remove = len(self._events) - self._event_buffer_size
        removed_events = self._events[:to_remove]
        self._events = self._events[to_remove:]

        # Update indexes
        for event in removed_events:
            if event in self._events_by_source[event.source]:
                self._events_by_source[event.source].remove(event)
            if event in self._events_by_type[event.event_type]:
                self._events_by_type[event.event_type].remove(event)

        return to_remove

    def _register_default_rules(self) -> None:
        """Register default correlation rules."""
        self.register_rule(CorrelationRule(
            rule_id="error-cascade",
            name="Error Cascade Detection",
            correlation_type=CorrelationType.CAUSAL,
            conditions={"severity": ["error", "critical"]},
            time_window_s=300,
            min_events=2,
        ))

        self.register_rule(CorrelationRule(
            rule_id="multi-source-error",
            name="Multi-Source Error Correlation",
            correlation_type=CorrelationType.TEMPORAL,
            conditions={"severity": ["error", "critical"]},
            time_window_s=60,
            min_events=3,
        ))

        self.register_rule(CorrelationRule(
            rule_id="alert-burst",
            name="Alert Burst Detection",
            correlation_type=CorrelationType.TEMPORAL,
            conditions={"event_type": "alert"},
            time_window_s=120,
            min_events=5,
        ))

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_engine: Optional[CorrelationEngine] = None


def get_engine() -> CorrelationEngine:
    """Get or create the correlation engine singleton.

    Returns:
        CorrelationEngine instance
    """
    global _engine
    if _engine is None:
        _engine = CorrelationEngine()
    return _engine


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Correlation Engine (Step 274)")
    parser.add_argument("--detect", action="store_true", help="Detect correlations")
    parser.add_argument("--window", type=int, default=300, help="Time window in seconds")
    parser.add_argument("--rules", action="store_true", help="List rules")
    parser.add_argument("--recent", action="store_true", help="Show recent correlations")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    engine = get_engine()

    if args.detect:
        correlations = engine.detect_correlations(time_window_s=args.window)
        if args.json:
            print(json.dumps([c.to_dict() for c in correlations], indent=2))
        else:
            print(f"Detected {len(correlations)} correlations")
            for c in correlations:
                print(f"  [{c.strength.value}] {c.correlation_id}: {c.description}")

    if args.rules:
        rules = engine.list_rules()
        if args.json:
            print(json.dumps(rules, indent=2))
        else:
            print("Correlation Rules:")
            for r in rules:
                enabled = "enabled" if r["enabled"] else "disabled"
                print(f"  [{r['rule_id']}] {r['name']} ({enabled})")

    if args.recent:
        correlations = engine.get_recent_correlations()
        if args.json:
            print(json.dumps([c.to_dict() for c in correlations], indent=2))
        else:
            print("Recent Correlations:")
            for c in correlations:
                print(f"  [{c.strength.value}] {c.correlation_id}: {c.description}")

    if args.stats:
        stats = engine.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Correlation Engine Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
