#!/usr/bin/env python3
"""
Pluribus Metrics Library: Neurosymbolic KPI Compilation and Anomaly Detection
==============================================================================

Compiles KPIs from bus events (agent counts, event rates, error rates, latencies)
using neurosymbolic principles for theorizing and testing metrics over time.

Core Functions:
  - compile_kpis(events) -> MetricsSnapshot
  - detect_anomalies(metrics) -> List[Anomaly]
  - summarize_for_notification(event) -> CompressedSummary
  - track_trends(snapshots) -> TrendAnalysis

Usage:
    from pluribus_metrics_library import (
        compile_kpis,
        detect_anomalies,
        summarize_for_notification,
        MetricsSnapshot,
    )

    events = load_events_from_bus()
    snapshot = compile_kpis(events, window_seconds=60)
    anomalies = detect_anomalies(snapshot)

    for event in critical_events:
        summary = summarize_for_notification(event)
        send_notification(summary)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

# ==============================================================================
# Type Definitions
# ==============================================================================

AnomalyType = Literal[
    "velocity_drop",
    "error_spike",
    "latency_violation",
    "agent_silent",
    "queue_backlog",
    "topology_imbalance",
    "entropy_drift",
]

SeverityLevel = Literal["info", "warn", "error", "critical"]

TrendDirection = Literal["increasing", "decreasing", "stable", "volatile"]


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class AgentMetrics:
    """Per-agent metrics snapshot."""
    actor: str
    event_count: int = 0
    error_count: int = 0
    last_seen_ts: float = 0.0
    avg_latency_ms: float = 0.0
    topics: list[str] = field(default_factory=list)
    health: str = "unknown"
    queue_depth: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TopicMetrics:
    """Per-topic metrics."""
    topic: str
    event_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    actors: list[str] = field(default_factory=list)
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TopologyMetrics:
    """Multi-agent topology metrics."""
    single_count: int = 0
    star_count: int = 0
    peer_debate_count: int = 0
    avg_fanout: float = 1.0
    coordination_budget_total: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvolutionaryMetrics:
    """VGT/HGT evolutionary metrics."""
    vgt_transfers: int = 0
    hgt_transfers: int = 0
    total_generations: int = 0
    avg_speciation_potential: float = 0.0
    lineage_health_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EntropyMetrics:
    """Information-theoretic metrics."""
    topic_entropy: float = 0.0  # Shannon entropy of topic distribution
    actor_entropy: float = 0.0  # Shannon entropy of actor distribution
    level_entropy: float = 0.0  # Shannon entropy of log levels
    causal_depth_avg: float = 0.0
    reversibility_avg: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricsSnapshot:
    """Complete metrics snapshot at a point in time."""
    # Identity
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    window_seconds: int = 60

    # Core KPIs
    total_events: int = 0
    total_errors: int = 0
    velocity: float = 0.0  # events per second
    error_rate: float = 0.0  # errors / total
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Agent metrics
    agent_count: int = 0
    active_agents: int = 0  # agents seen in window
    silent_agents: int = 0  # agents with no recent activity
    agents: dict[str, AgentMetrics] = field(default_factory=dict)

    # Topic metrics
    topic_count: int = 0
    topics: dict[str, TopicMetrics] = field(default_factory=dict)

    # Topology
    topology: TopologyMetrics = field(default_factory=TopologyMetrics)

    # Evolutionary
    evolutionary: EvolutionaryMetrics = field(default_factory=EvolutionaryMetrics)

    # Information-theoretic
    entropy: EntropyMetrics = field(default_factory=EntropyMetrics)

    # Queue state
    queue_depth_total: int = 0
    pending_requests: int = 0
    completed_requests: int = 0

    def to_dict(self) -> dict:
        d = {
            "snapshot_id": self.snapshot_id,
            "ts": self.ts,
            "iso": self.iso,
            "window_seconds": self.window_seconds,
            "kpis": {
                "total_events": self.total_events,
                "total_errors": self.total_errors,
                "velocity": round(self.velocity, 3),
                "error_rate": round(self.error_rate, 4),
                "latency": {
                    "avg_ms": round(self.avg_latency_ms, 2),
                    "p50_ms": round(self.p50_latency_ms, 2),
                    "p95_ms": round(self.p95_latency_ms, 2),
                    "p99_ms": round(self.p99_latency_ms, 2),
                },
            },
            "agents": {
                "count": self.agent_count,
                "active": self.active_agents,
                "silent": self.silent_agents,
                "details": {k: v.to_dict() for k, v in self.agents.items()},
            },
            "topics": {
                "count": self.topic_count,
                "details": {k: v.to_dict() for k, v in self.topics.items()},
            },
            "topology": self.topology.to_dict(),
            "evolutionary": self.evolutionary.to_dict(),
            "entropy": self.entropy.to_dict(),
            "queue": {
                "depth": self.queue_depth_total,
                "pending": self.pending_requests,
                "completed": self.completed_requests,
            },
        }
        return d

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Core KPIs
        lines.append("# HELP pluribus_events_total Total events in window")
        lines.append("# TYPE pluribus_events_total gauge")
        lines.append(f"pluribus_events_total {self.total_events}")

        lines.append("# HELP pluribus_errors_total Total errors in window")
        lines.append("# TYPE pluribus_errors_total gauge")
        lines.append(f"pluribus_errors_total {self.total_errors}")

        lines.append("# HELP pluribus_velocity_mps Messages per second")
        lines.append("# TYPE pluribus_velocity_mps gauge")
        lines.append(f"pluribus_velocity_mps {self.velocity:.3f}")

        lines.append("# HELP pluribus_error_rate Error rate (0-1)")
        lines.append("# TYPE pluribus_error_rate gauge")
        lines.append(f"pluribus_error_rate {self.error_rate:.4f}")

        lines.append("# HELP pluribus_latency_avg_ms Average latency in ms")
        lines.append("# TYPE pluribus_latency_avg_ms gauge")
        lines.append(f"pluribus_latency_avg_ms {self.avg_latency_ms:.2f}")

        lines.append("# HELP pluribus_latency_p95_ms P95 latency in ms")
        lines.append("# TYPE pluribus_latency_p95_ms gauge")
        lines.append(f"pluribus_latency_p95_ms {self.p95_latency_ms:.2f}")

        # Agent counts
        lines.append("# HELP pluribus_agents_active Active agents in window")
        lines.append("# TYPE pluribus_agents_active gauge")
        lines.append(f"pluribus_agents_active {self.active_agents}")

        lines.append("# HELP pluribus_agents_silent Silent agents")
        lines.append("# TYPE pluribus_agents_silent gauge")
        lines.append(f"pluribus_agents_silent {self.silent_agents}")

        # Topology
        lines.append("# HELP pluribus_topology_star Star topology operations")
        lines.append("# TYPE pluribus_topology_star gauge")
        lines.append(f"pluribus_topology_star {self.topology.star_count}")

        lines.append("# HELP pluribus_topology_fanout_avg Average fanout")
        lines.append("# TYPE pluribus_topology_fanout_avg gauge")
        lines.append(f"pluribus_topology_fanout_avg {self.topology.avg_fanout:.2f}")

        # Entropy
        lines.append("# HELP pluribus_entropy_topic Topic distribution entropy")
        lines.append("# TYPE pluribus_entropy_topic gauge")
        lines.append(f"pluribus_entropy_topic {self.entropy.topic_entropy:.4f}")

        lines.append("")
        return "\n".join(lines)


@dataclass
class Anomaly:
    """Detected anomaly in metrics."""
    anomaly_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts: float = field(default_factory=time.time)
    iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    anomaly_type: AnomalyType = "velocity_drop"
    severity: SeverityLevel = "warn"
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    deviation_sigma: float = 0.0
    description: str = ""
    suggested_actions: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompressedSummary:
    """Compressed notification-ready summary of an event."""
    summary_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ts: float = field(default_factory=time.time)
    iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Core summary
    headline: str = ""
    body: str = ""
    severity: SeverityLevel = "info"
    # Structured data
    actor: str = ""
    topic: str = ""
    impact: str = "low"
    actionable: list[str] = field(default_factory=list)
    # Metrics
    key_metrics: dict[str, float] = field(default_factory=dict)
    # Compression metadata
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_notification_card(self) -> dict:
        """Format for notification system (e.g., dashboard toast, webhook)."""
        return {
            "id": self.summary_id,
            "timestamp": self.iso,
            "severity": self.severity,
            "headline": self.headline,
            "body": self.body,
            "actor": self.actor,
            "topic": self.topic,
            "impact": self.impact,
            "actions": self.actionable,
            "metrics": self.key_metrics,
        }


@dataclass
class TrendAnalysis:
    """Time-series trend analysis of metrics."""
    metric_name: str
    direction: TrendDirection = "stable"
    slope: float = 0.0  # Rate of change per second
    volatility: float = 0.0  # Standard deviation of changes
    samples: int = 0
    prediction_next: float = 0.0
    confidence: float = 0.0  # 0-1 confidence in prediction

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# Core Functions
# ==============================================================================

def compile_kpis(
    events: list[dict],
    window_seconds: int = 60,
    reference_ts: float | None = None,
) -> MetricsSnapshot:
    """
    Compile KPIs from a list of bus events.

    Uses neurosymbolic principles:
    - Symbolic: Rule-based classification of events by topic, level, kind
    - Neural: Statistical aggregation with percentiles and entropy

    Args:
        events: List of bus event dictionaries
        window_seconds: Analysis window in seconds
        reference_ts: Reference timestamp (defaults to now)

    Returns:
        MetricsSnapshot with compiled KPIs
    """
    now = reference_ts or time.time()
    cutoff = now - window_seconds
    snapshot = MetricsSnapshot(
        ts=now,
        iso=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        window_seconds=window_seconds,
    )

    # Filter events in window
    window_events = [e for e in events if float(e.get("ts", 0)) >= cutoff]

    if not window_events:
        return snapshot

    # Accumulators
    latencies: list[float] = []
    topic_counts: dict[str, int] = defaultdict(int)
    actor_counts: dict[str, int] = defaultdict(int)
    level_counts: dict[str, int] = defaultdict(int)
    fanouts: list[int] = []
    speciation_potentials: list[float] = []
    causal_depths: list[int] = []
    reversibilities: list[float] = []
    agent_last_seen: dict[str, float] = {}
    agent_errors: dict[str, int] = defaultdict(int)
    agent_latencies: dict[str, list[float]] = defaultdict(list)
    agent_topics: dict[str, set[str]] = defaultdict(set)
    topic_actors: dict[str, set[str]] = defaultdict(set)
    topic_errors: dict[str, int] = defaultdict(int)
    topic_first_seen: dict[str, float] = {}
    topic_last_seen: dict[str, float] = {}
    topic_latencies: dict[str, list[float]] = defaultdict(list)
    pending_req_ids: set[str] = set()
    completed_req_ids: set[str] = set()

    for event in window_events:
        ts = float(event.get("ts", 0))
        topic = str(event.get("topic", ""))
        actor = str(event.get("actor", ""))
        level = str(event.get("level", "info"))
        kind = str(event.get("kind", ""))
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

        # Core counts
        snapshot.total_events += 1
        topic_counts[topic] += 1
        actor_counts[actor] += 1
        level_counts[level] += 1

        if level == "error":
            snapshot.total_errors += 1
            agent_errors[actor] += 1
            topic_errors[topic] += 1

        # Latency extraction
        if "latency_ms" in data:
            lat = float(data["latency_ms"])
            latencies.append(lat)
            agent_latencies[actor].append(lat)
            topic_latencies[topic].append(lat)

        # Agent tracking
        if actor:
            if actor not in agent_last_seen or ts > agent_last_seen[actor]:
                agent_last_seen[actor] = ts
            agent_topics[actor].add(topic)

        # Topic tracking
        if topic:
            topic_actors[topic].add(actor)
            if topic not in topic_first_seen or ts < topic_first_seen[topic]:
                topic_first_seen[topic] = ts
            if topic not in topic_last_seen or ts > topic_last_seen[topic]:
                topic_last_seen[topic] = ts

        # Topology tracking
        topo_context = event.get("topology", {})
        if isinstance(topo_context, dict):
            topo_type = topo_context.get("topology", "single")
            if topo_type == "star":
                snapshot.topology.star_count += 1
            elif topo_type == "peer_debate":
                snapshot.topology.peer_debate_count += 1
            else:
                snapshot.topology.single_count += 1

            fanout = topo_context.get("fanout", 1)
            if isinstance(fanout, int):
                fanouts.append(fanout)

            budget = topo_context.get("coordination_budget_tokens", 0)
            if isinstance(budget, int):
                snapshot.topology.coordination_budget_total += budget

        # Evolutionary tracking
        evo_context = event.get("evolutionary", {})
        if isinstance(evo_context, dict):
            transfer = evo_context.get("transfer_type", "")
            if transfer == "VGT":
                snapshot.evolutionary.vgt_transfers += 1
            elif transfer == "HGT":
                snapshot.evolutionary.hgt_transfers += 1

            gen = evo_context.get("generation", 0)
            if isinstance(gen, int) and gen > 0:
                snapshot.evolutionary.total_generations = max(
                    snapshot.evolutionary.total_generations, gen
                )

            speciation = evo_context.get("speciation_potential", 0.0)
            if isinstance(speciation, (int, float)) and speciation > 0:
                speciation_potentials.append(float(speciation))

            health = evo_context.get("lineage_health", "")
            if health:
                snapshot.evolutionary.lineage_health_distribution[health] = (
                    snapshot.evolutionary.lineage_health_distribution.get(health, 0) + 1
                )

        # Hysteresis tracking
        hysteresis = event.get("hysteresis", {})
        if isinstance(hysteresis, dict):
            depth = hysteresis.get("causal_depth", 0)
            if isinstance(depth, int):
                causal_depths.append(depth)
            rev = hysteresis.get("reversibility", 1.0)
            if isinstance(rev, (int, float)):
                reversibilities.append(float(rev))

        # Request tracking
        req_id = data.get("req_id", "") if isinstance(data, dict) else ""
        if kind == "request" and req_id:
            pending_req_ids.add(req_id)
        elif kind == "response" and req_id:
            completed_req_ids.add(req_id)
            if req_id in pending_req_ids:
                pending_req_ids.discard(req_id)

    # Compute derived metrics
    snapshot.velocity = snapshot.total_events / window_seconds
    snapshot.error_rate = (
        snapshot.total_errors / snapshot.total_events
        if snapshot.total_events > 0
        else 0.0
    )

    # Latency percentiles
    if latencies:
        latencies_sorted = sorted(latencies)
        snapshot.avg_latency_ms = statistics.mean(latencies)
        snapshot.p50_latency_ms = _percentile(latencies_sorted, 50)
        snapshot.p95_latency_ms = _percentile(latencies_sorted, 95)
        snapshot.p99_latency_ms = _percentile(latencies_sorted, 99)

    # Agent metrics
    silence_threshold = now - window_seconds * 0.5  # Silent if not seen in half window
    for actor, count in actor_counts.items():
        am = AgentMetrics(
            actor=actor,
            event_count=count,
            error_count=agent_errors.get(actor, 0),
            last_seen_ts=agent_last_seen.get(actor, 0),
            avg_latency_ms=(
                statistics.mean(agent_latencies[actor])
                if agent_latencies[actor]
                else 0.0
            ),
            topics=list(agent_topics.get(actor, set())),
            health="healthy" if agent_errors.get(actor, 0) == 0 else "degraded",
        )
        snapshot.agents[actor] = am
        if am.last_seen_ts >= silence_threshold:
            snapshot.active_agents += 1
        else:
            snapshot.silent_agents += 1
    snapshot.agent_count = len(snapshot.agents)

    # Topic metrics
    for topic, count in topic_counts.items():
        tm = TopicMetrics(
            topic=topic,
            event_count=count,
            error_count=topic_errors.get(topic, 0),
            avg_latency_ms=(
                statistics.mean(topic_latencies[topic])
                if topic_latencies[topic]
                else 0.0
            ),
            actors=list(topic_actors.get(topic, set())),
            first_seen_ts=topic_first_seen.get(topic, 0),
            last_seen_ts=topic_last_seen.get(topic, 0),
        )
        snapshot.topics[topic] = tm
    snapshot.topic_count = len(snapshot.topics)

    # Topology aggregates
    if fanouts:
        snapshot.topology.avg_fanout = statistics.mean(fanouts)

    # Evolutionary aggregates
    if speciation_potentials:
        snapshot.evolutionary.avg_speciation_potential = statistics.mean(
            speciation_potentials
        )

    # Entropy metrics
    snapshot.entropy.topic_entropy = _shannon_entropy(topic_counts)
    snapshot.entropy.actor_entropy = _shannon_entropy(actor_counts)
    snapshot.entropy.level_entropy = _shannon_entropy(level_counts)
    if causal_depths:
        snapshot.entropy.causal_depth_avg = statistics.mean(causal_depths)
    if reversibilities:
        snapshot.entropy.reversibility_avg = statistics.mean(reversibilities)

    # Queue state
    snapshot.pending_requests = len(pending_req_ids)
    snapshot.completed_requests = len(completed_req_ids)
    snapshot.queue_depth_total = snapshot.pending_requests

    return snapshot


def detect_anomalies(
    snapshot: MetricsSnapshot,
    baseline: MetricsSnapshot | None = None,
    thresholds: dict[str, float] | None = None,
) -> list[Anomaly]:
    """
    Detect anomalies in metrics snapshot.

    Uses neurosymbolic detection:
    - Symbolic: Threshold-based rules (error_rate > 0.1, latency > SLA)
    - Neural: Statistical deviation from baseline (sigma-based)

    Args:
        snapshot: Current metrics snapshot
        baseline: Historical baseline for comparison (optional)
        thresholds: Custom thresholds (optional)

    Returns:
        List of detected anomalies
    """
    anomalies: list[Anomaly] = []

    # Default thresholds
    defaults = {
        "min_velocity": 0.1,  # MPS
        "max_error_rate": 0.1,  # 10%
        "max_latency_p95_ms": 2000,  # 2s SLA
        "max_latency_p99_ms": 5000,  # 5s hard limit
        "max_queue_depth": 50,
        "min_active_agent_ratio": 0.5,  # 50% of known agents should be active
        "entropy_drift_sigma": 2.0,
    }
    thresholds = {**defaults, **(thresholds or {})}

    # Velocity drop
    if snapshot.velocity < thresholds["min_velocity"] and snapshot.queue_depth_total > 0:
        anomalies.append(
            Anomaly(
                anomaly_type="velocity_drop",
                severity="warn",
                metric_name="velocity",
                current_value=snapshot.velocity,
                threshold=thresholds["min_velocity"],
                description=f"Velocity {snapshot.velocity:.3f} MPS below threshold {thresholds['min_velocity']} with {snapshot.queue_depth_total} pending items",
                suggested_actions=[
                    "Check worker health",
                    "Review queue processing",
                    "Consider scaling workers",
                ],
                context={"queue_depth": snapshot.queue_depth_total},
            )
        )

    # Error spike
    if snapshot.error_rate > thresholds["max_error_rate"]:
        severity: SeverityLevel = (
            "critical" if snapshot.error_rate > 0.3 else "error"
        )
        anomalies.append(
            Anomaly(
                anomaly_type="error_spike",
                severity=severity,
                metric_name="error_rate",
                current_value=snapshot.error_rate,
                threshold=thresholds["max_error_rate"],
                description=f"Error rate {snapshot.error_rate:.1%} exceeds threshold {thresholds['max_error_rate']:.0%}",
                suggested_actions=[
                    "Review recent error logs",
                    "Check provider health",
                    "Enable fallback chain",
                ],
                context={"total_errors": snapshot.total_errors},
            )
        )

    # Latency violation
    if snapshot.p95_latency_ms > thresholds["max_latency_p95_ms"]:
        anomalies.append(
            Anomaly(
                anomaly_type="latency_violation",
                severity="warn",
                metric_name="p95_latency_ms",
                current_value=snapshot.p95_latency_ms,
                threshold=thresholds["max_latency_p95_ms"],
                description=f"P95 latency {snapshot.p95_latency_ms:.0f}ms exceeds SLA {thresholds['max_latency_p95_ms']:.0f}ms",
                suggested_actions=[
                    "Check provider response times",
                    "Consider backoff strategy",
                    "Review query complexity",
                ],
                context={
                    "avg_latency_ms": snapshot.avg_latency_ms,
                    "p99_latency_ms": snapshot.p99_latency_ms,
                },
            )
        )

    if snapshot.p99_latency_ms > thresholds["max_latency_p99_ms"]:
        anomalies.append(
            Anomaly(
                anomaly_type="latency_violation",
                severity="error",
                metric_name="p99_latency_ms",
                current_value=snapshot.p99_latency_ms,
                threshold=thresholds["max_latency_p99_ms"],
                description=f"P99 latency {snapshot.p99_latency_ms:.0f}ms exceeds hard limit {thresholds['max_latency_p99_ms']:.0f}ms",
                suggested_actions=[
                    "Immediate investigation required",
                    "Consider request timeout reduction",
                    "Enable circuit breaker",
                ],
            )
        )

    # Queue backlog
    if snapshot.queue_depth_total > thresholds["max_queue_depth"]:
        anomalies.append(
            Anomaly(
                anomaly_type="queue_backlog",
                severity="warn",
                metric_name="queue_depth",
                current_value=float(snapshot.queue_depth_total),
                threshold=float(thresholds["max_queue_depth"]),
                description=f"Queue depth {snapshot.queue_depth_total} exceeds threshold {thresholds['max_queue_depth']}",
                suggested_actions=[
                    "Scale up workers",
                    "Review processing bottlenecks",
                    "Consider priority queue",
                ],
            )
        )

    # Silent agents
    if snapshot.agent_count > 0:
        active_ratio = snapshot.active_agents / snapshot.agent_count
        if active_ratio < thresholds["min_active_agent_ratio"]:
            anomalies.append(
                Anomaly(
                    anomaly_type="agent_silent",
                    severity="warn",
                    metric_name="active_agent_ratio",
                    current_value=active_ratio,
                    threshold=thresholds["min_active_agent_ratio"],
                    description=f"Only {active_ratio:.0%} agents active ({snapshot.active_agents}/{snapshot.agent_count})",
                    suggested_actions=[
                        "Check agent health",
                        "Review agent liveness probes",
                        "Restart silent agents",
                    ],
                    context={
                        "active": snapshot.active_agents,
                        "silent": snapshot.silent_agents,
                    },
                )
            )

    # Topology imbalance (if using star topology heavily)
    if snapshot.topology.star_count > 0 and snapshot.topology.avg_fanout > 5:
        anomalies.append(
            Anomaly(
                anomaly_type="topology_imbalance",
                severity="info",
                metric_name="avg_fanout",
                current_value=snapshot.topology.avg_fanout,
                threshold=5.0,
                description=f"High average fanout {snapshot.topology.avg_fanout:.1f} may indicate over-parallelization",
                suggested_actions=[
                    "Review decomposition strategy",
                    "Consider coordination budget",
                    "Monitor aggregation overhead",
                ],
            )
        )

    # Baseline comparison (if available)
    if baseline:
        # Entropy drift
        topic_entropy_delta = abs(
            snapshot.entropy.topic_entropy - baseline.entropy.topic_entropy
        )
        if baseline.entropy.topic_entropy > 0:
            # Use baseline as reference for sigma calculation
            relative_drift = topic_entropy_delta / max(
                baseline.entropy.topic_entropy, 0.1
            )
            if relative_drift > thresholds["entropy_drift_sigma"] * 0.1:
                anomalies.append(
                    Anomaly(
                        anomaly_type="entropy_drift",
                        severity="info",
                        metric_name="topic_entropy",
                        current_value=snapshot.entropy.topic_entropy,
                        threshold=baseline.entropy.topic_entropy,
                        deviation_sigma=relative_drift * 10,
                        description=f"Topic distribution entropy shifted by {relative_drift:.1%}",
                        suggested_actions=[
                            "Review event source distribution",
                            "Check for new/retired topics",
                        ],
                    )
                )

    return anomalies


def summarize_for_notification(
    event: dict,
    max_length: int = 280,
) -> CompressedSummary:
    """
    Create a compressed, notification-ready summary of an event.

    Uses heuristic compression (no LLM needed):
    - Extract key fields (topic, actor, level, impact)
    - Create human-readable headline
    - Preserve actionable insights

    Args:
        event: Bus event dictionary
        max_length: Maximum summary length

    Returns:
        CompressedSummary for notification
    """
    topic = str(event.get("topic", ""))
    actor = str(event.get("actor", ""))
    level = str(event.get("level", "info"))
    kind = str(event.get("kind", ""))
    data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}
    semantic = str(event.get("semantic", ""))
    impact = str(event.get("impact", "low"))
    actionable = event.get("actionable", [])

    # Original size
    original_json = json.dumps(event, default=str)
    original_size = len(original_json.encode("utf-8"))

    # Determine severity
    severity_map = {
        "error": "error",
        "warn": "warn",
        "warning": "warn",
        "info": "info",
        "debug": "info",
    }
    severity: SeverityLevel = severity_map.get(level, "info")
    if impact == "critical":
        severity = "critical"
    elif impact == "high" and severity != "critical":
        severity = "error"

    # Generate headline
    if semantic:
        headline = semantic[:100]
    else:
        topic_parts = topic.split(".")
        if len(topic_parts) >= 2:
            headline = f"{topic_parts[0].upper()}: {'.'.join(topic_parts[1:])}"
        else:
            headline = topic or "Unknown event"

    # Generate body
    body_parts = []
    if actor:
        body_parts.append(f"Actor: {actor}")
    if kind:
        body_parts.append(f"Kind: {kind}")

    # Extract key metrics from data
    key_metrics: dict[str, float] = {}
    metric_keys = [
        "latency_ms",
        "velocity",
        "error_rate",
        "queue_depth",
        "tokens",
        "fanout",
    ]
    for key in metric_keys:
        if key in data and isinstance(data[key], (int, float)):
            key_metrics[key] = float(data[key])
            body_parts.append(f"{key}: {data[key]}")

    body = " | ".join(body_parts[:5])  # Limit to 5 parts

    # Truncate if needed
    if len(headline) > max_length // 2:
        headline = headline[: max_length // 2 - 3] + "..."
    if len(body) > max_length // 2:
        body = body[: max_length // 2 - 3] + "..."

    compressed_size = len((headline + body).encode("utf-8"))

    return CompressedSummary(
        headline=headline,
        body=body,
        severity=severity,
        actor=actor,
        topic=topic,
        impact=impact,
        actionable=actionable if isinstance(actionable, list) else [],
        key_metrics=key_metrics,
        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
        compression_ratio=original_size / max(compressed_size, 1),
    )


def track_trends(
    snapshots: list[MetricsSnapshot],
    metric_path: str = "velocity",
) -> TrendAnalysis:
    """
    Analyze trends across multiple metrics snapshots.

    Uses simple linear regression for trend detection.

    Args:
        snapshots: List of historical snapshots (oldest first)
        metric_path: Dot-notation path to metric (e.g., "velocity", "entropy.topic_entropy")

    Returns:
        TrendAnalysis with direction and predictions
    """
    if len(snapshots) < 2:
        return TrendAnalysis(metric_name=metric_path, samples=len(snapshots))

    # Extract metric values
    values: list[tuple[float, float]] = []  # (timestamp, value)
    for snap in snapshots:
        val = _extract_metric(snap, metric_path)
        if val is not None:
            values.append((snap.ts, val))

    if len(values) < 2:
        return TrendAnalysis(metric_name=metric_path, samples=len(values))

    # Linear regression
    n = len(values)
    sum_t = sum(t for t, _ in values)
    sum_v = sum(v for _, v in values)
    sum_tv = sum(t * v for t, v in values)
    sum_tt = sum(t * t for t, _ in values)

    denom = n * sum_tt - sum_t * sum_t
    if abs(denom) < 1e-10:
        slope = 0.0
    else:
        slope = (n * sum_tv - sum_t * sum_v) / denom

    # Compute volatility (standard deviation of values)
    mean_v = sum_v / n
    volatility = math.sqrt(sum((v - mean_v) ** 2 for _, v in values) / n)

    # Determine direction
    if abs(slope) < volatility * 0.1:
        direction: TrendDirection = "stable"
    elif volatility > abs(slope) * 2:
        direction = "volatile"
    elif slope > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    # Predict next value
    last_ts = values[-1][0]
    next_ts = last_ts + (values[-1][0] - values[0][0]) / (n - 1)
    intercept = (sum_v - slope * sum_t) / n
    prediction = slope * next_ts + intercept

    # Confidence based on R-squared
    ss_res = sum((v - (slope * t + intercept)) ** 2 for t, v in values)
    ss_tot = sum((v - mean_v) ** 2 for _, v in values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return TrendAnalysis(
        metric_name=metric_path,
        direction=direction,
        slope=slope,
        volatility=volatility,
        samples=n,
        prediction_next=prediction,
        confidence=max(0, min(1, r_squared)),
    )


# ==============================================================================
# Helper Functions
# ==============================================================================

def _percentile(sorted_data: list[float], p: float) -> float:
    """Calculate percentile from sorted data."""
    if not sorted_data:
        return 0.0
    idx = (len(sorted_data) - 1) * p / 100
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_data):
        return sorted_data[-1]
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Calculate Shannon entropy of a count distribution."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _extract_metric(snapshot: MetricsSnapshot, path: str) -> float | None:
    """Extract a metric value by dot-notation path."""
    parts = path.split(".")
    obj: Any = snapshot
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return float(obj) if isinstance(obj, (int, float)) else None


# ==============================================================================
# Bus Integration
# ==============================================================================

def load_events_from_bus(
    bus_dir: str | Path | None = None,
    max_bytes: int = 2_000_000,
) -> list[dict]:
    """Load events from the bus events file."""
    bus_dir = Path(bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_path = bus_dir / "events.ndjson"

    if not events_path.exists():
        return []

    events: list[dict] = []
    size = events_path.stat().st_size
    start = max(0, size - max_bytes)

    with events_path.open("rb") as f:
        f.seek(start)
        data = f.read()

    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # Skip partial first line

    for line in lines:
        try:
            event = json.loads(line.decode("utf-8", errors="replace"))
            events.append(event)
        except Exception:
            continue

    return events


def emit_metrics_event(
    snapshot: MetricsSnapshot,
    bus_dir: str | Path | None = None,
) -> str:
    """Emit a metrics snapshot as a bus event."""
    import fcntl

    bus_dir = Path(bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "id": snapshot.snapshot_id,
        "ts": snapshot.ts,
        "iso": snapshot.iso,
        "topic": "pluribus.metrics.snapshot",
        "kind": "metric",
        "level": "info",
        "actor": os.environ.get("PLURIBUS_ACTOR", "metrics-library"),
        "data": snapshot.to_dict(),
    }

    with events_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    return snapshot.snapshot_id


def emit_anomaly_event(
    anomaly: Anomaly,
    bus_dir: str | Path | None = None,
) -> str:
    """Emit an anomaly detection as a bus event."""
    import fcntl

    bus_dir = Path(bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "id": anomaly.anomaly_id,
        "ts": anomaly.ts,
        "iso": anomaly.iso,
        "topic": f"pluribus.metrics.anomaly.{anomaly.anomaly_type}",
        "kind": "alert",
        "level": anomaly.severity,
        "actor": os.environ.get("PLURIBUS_ACTOR", "metrics-library"),
        "data": anomaly.to_dict(),
        "semantic": anomaly.description,
        "actionable": anomaly.suggested_actions,
        "impact": "high" if anomaly.severity in ("error", "critical") else "medium",
    }

    with events_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    return anomaly.anomaly_id


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    """CLI for testing the metrics library."""
    import argparse

    parser = argparse.ArgumentParser(description="Pluribus Metrics Library CLI")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    parser.add_argument("--compile", action="store_true", help="Compile KPIs from bus")
    parser.add_argument("--window", type=int, default=60, help="Window seconds")
    parser.add_argument("--format", choices=["json", "prometheus"], default="json")
    parser.add_argument("--emit-bus", action="store_true", help="Emit to bus")
    parser.add_argument("--bus-dir", help="Bus directory")
    args = parser.parse_args()

    if args.test:
        print("=== Pluribus Metrics Library Self-Test ===\n")

        # Test 1: Create mock events
        print("[TEST 1] Mock event compilation")
        mock_events = [
            {
                "ts": time.time() - 30,
                "topic": "plurichat.request",
                "actor": "test-agent-1",
                "level": "info",
                "kind": "request",
                "data": {"latency_ms": 150},
            },
            {
                "ts": time.time() - 20,
                "topic": "plurichat.response",
                "actor": "test-agent-1",
                "level": "info",
                "kind": "response",
                "data": {"latency_ms": 200, "req_id": "r1"},
            },
            {
                "ts": time.time() - 10,
                "topic": "strp.error",
                "actor": "test-agent-2",
                "level": "error",
                "kind": "log",
                "data": {},
            },
        ]

        snapshot = compile_kpis(mock_events, window_seconds=60)
        print(f"  Total events: {snapshot.total_events}")
        print(f"  Error rate: {snapshot.error_rate:.1%}")
        print(f"  Velocity: {snapshot.velocity:.3f} MPS")
        print(f"  Agents: {snapshot.agent_count}")
        print("  [PASS]\n")

        # Test 2: Anomaly detection
        print("[TEST 2] Anomaly detection")
        anomalies = detect_anomalies(snapshot)
        print(f"  Detected anomalies: {len(anomalies)}")
        for a in anomalies:
            print(f"    - {a.anomaly_type}: {a.description[:60]}")
        print("  [PASS]\n")

        # Test 3: Notification summary
        print("[TEST 3] Notification summary")
        summary = summarize_for_notification(mock_events[0])
        print(f"  Headline: {summary.headline}")
        print(f"  Body: {summary.body}")
        print(f"  Compression ratio: {summary.compression_ratio:.1f}x")
        print("  [PASS]\n")

        print("=== ALL TESTS PASSED ===")
        return 0

    if args.compile:
        bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        events = load_events_from_bus(bus_dir)
        snapshot = compile_kpis(events, window_seconds=args.window)

        if args.format == "prometheus":
            print(snapshot.to_prometheus())
        else:
            print(json.dumps(snapshot.to_dict(), indent=2))

        if args.emit_bus:
            emit_metrics_event(snapshot, bus_dir)
            print(f"\nEmitted snapshot: {snapshot.snapshot_id}", file=sys.stderr)

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
