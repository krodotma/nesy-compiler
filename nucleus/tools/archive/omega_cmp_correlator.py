#!/usr/bin/env python3
"""
Omega-CMP Correlator - Bridges Omega Guardian semantic state with CMP scoring.

This module implements the bidirectional feedback loop between:
- Omega Guardian Semantic (motif tracking, liveness detection)
- CMP Engine Daemon (lineage fitness scoring)

The correlator:
1. Processes omega.guardian.semantic.cycle events to extract semantic health
2. Computes correlation metrics between motif completion and CMP scores
3. Provides adaptive thresholds for Omega Guardian based on CMP context
4. Emits omega.cmp.correlation events for downstream consumption

Part of the Entelexis Architecture - Iteration 4 Implementation.
"""
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Constants from Entelexis spec
PHI = 1.618033988749895
PHI_INV = 1 / PHI           # ~0.618
PHI_INV_2 = 1 / PHI**2      # ~0.382
PHI_INV_3 = 1 / PHI**3      # ~0.236

# Correlation window parameters
CORRELATION_WINDOW_S = 300  # 5 minutes
HISTORY_SIZE = 100


@dataclass
class SemanticHealthSnapshot:
    """Snapshot of Omega Guardian semantic state at a point in time."""
    timestamp: float
    global_state: str  # q_observe, q_good, q_stale, q_zombie, q_recovering
    motif_completion_rate: float  # completions per second
    actors_healthy: int
    actors_stale: int
    actors_zombie: int
    total_completions: int
    cycle_number: int


@dataclass
class CMPSnapshot:
    """Snapshot of CMP state for a lineage at a point in time."""
    timestamp: float
    lineage_id: str
    cmp_score: float
    entropy_factor: float = 1.0
    semantic_liveness: float = 1.0


@dataclass
class CorrelationMetrics:
    """Computed correlation metrics between Omega and CMP."""
    timestamp: float

    # Correlation coefficients
    motif_cmp_correlation: float  # Pearson r between motif rate and CMP delta
    semantic_cmp_alignment: float  # How well semantic state predicts CMP movement

    # Derived health indicators
    combined_health_score: float  # Weighted combination
    trend_direction: str  # "improving", "stable", "declining"

    # Adaptive thresholds
    suggested_window_s: float  # Adjusted based on CMP context
    suggested_stale_threshold: int  # Adjusted based on lineage fitness

    # Context
    lineage_ids: list = field(default_factory=list)
    sample_size: int = 0


@dataclass
class AdaptiveThresholds:
    """CMP-aware thresholds for Omega Guardian."""
    window_s: float = 120.0
    stale_threshold: int = 3
    min_completion_rate: float = 0.001
    zombie_recovery_timeout_s: float = 60.0

    # Per-lineage overrides
    lineage_overrides: dict = field(default_factory=dict)


class OmegaCMPCorrelator:
    """
    Bridges Omega Guardian semantic state with CMP scoring.

    Provides:
    1. Real-time correlation computation
    2. Adaptive threshold recommendations
    3. Combined health scoring
    4. Event emission for dashboard
    """

    def __init__(
        self,
        bus_path: str = "/pluribus/.pluribus/bus/events.ndjson",
        correlation_window_s: float = CORRELATION_WINDOW_S,
        history_size: int = HISTORY_SIZE,
    ):
        self.bus_path = Path(bus_path)
        self.correlation_window_s = correlation_window_s

        # History buffers
        self.semantic_history: deque[SemanticHealthSnapshot] = deque(maxlen=history_size)
        self.cmp_history: deque[CMPSnapshot] = deque(maxlen=history_size)
        self.correlation_history: deque[CorrelationMetrics] = deque(maxlen=history_size)

        # Current state
        self.current_thresholds = AdaptiveThresholds()
        self.last_correlation_ts = 0.0
        self.actor = "omega-cmp-correlator"

    def emit_bus(
        self,
        topic: str,
        kind: str,
        level: str,
        data: dict,
    ) -> None:
        """Emit event to the bus."""
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.actor,
            "data": data,
        }
        with self.bus_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")

    def process_semantic_cycle(self, event: dict) -> Optional[CorrelationMetrics]:
        """
        Process an omega.guardian.semantic.cycle event.

        Extracts semantic health snapshot and potentially computes correlation.
        """
        data = event.get("data", {})
        now = event.get("ts", time.time())

        # Extract semantic health snapshot
        snapshot = SemanticHealthSnapshot(
            timestamp=now,
            global_state=data.get("state", "q_observe"),
            motif_completion_rate=data.get("completion_rate", 0.0),
            actors_healthy=data.get("actors_healthy", 0),
            actors_stale=data.get("actors_stale", 0),
            actors_zombie=data.get("actors_zombie", 0),
            total_completions=data.get("total_completions", 0),
            cycle_number=data.get("cycle_number", 0),
        )
        self.semantic_history.append(snapshot)

        # Check if we should compute correlation
        if now - self.last_correlation_ts >= 10.0:  # Every 10 seconds
            metrics = self._compute_correlation(now)
            if metrics:
                self._emit_correlation(metrics)
                self.last_correlation_ts = now
                return metrics

        return None

    def process_cmp_update(self, event: dict) -> None:
        """Process a cmp.score or cmp.lineage.update event."""
        data = event.get("data", {})
        now = event.get("ts", time.time())

        snapshot = CMPSnapshot(
            timestamp=now,
            lineage_id=data.get("lineage_id", "unknown"),
            cmp_score=data.get("cmp_score", 0.0),
            entropy_factor=data.get("entropy_factor", 1.0),
            semantic_liveness=data.get("semantic_liveness", 1.0),
        )
        self.cmp_history.append(snapshot)

    def _compute_correlation(self, now: float) -> Optional[CorrelationMetrics]:
        """Compute correlation metrics from recent history."""
        if len(self.semantic_history) < 3 or len(self.cmp_history) < 3:
            return None

        window_start = now - self.correlation_window_s

        # Filter to window
        semantic_in_window = [
            s for s in self.semantic_history if s.timestamp >= window_start
        ]
        cmp_in_window = [
            c for c in self.cmp_history if c.timestamp >= window_start
        ]

        if len(semantic_in_window) < 2 or len(cmp_in_window) < 2:
            return None

        # Compute motif-CMP correlation
        motif_rates = [s.motif_completion_rate for s in semantic_in_window]
        cmp_scores = [c.cmp_score for c in cmp_in_window[-len(motif_rates):]]

        # Simple correlation (Pearson r approximation)
        motif_cmp_correlation = self._pearson_correlation(
            motif_rates[:len(cmp_scores)],
            cmp_scores[:len(motif_rates)],
        )

        # Semantic-CMP alignment
        semantic_cmp_alignment = self._compute_semantic_alignment(
            semantic_in_window,
            cmp_in_window,
        )

        # Combined health score
        combined_health = self._compute_combined_health(
            semantic_in_window[-1],
            cmp_in_window[-1] if cmp_in_window else None,
        )

        # Trend direction
        trend = self._determine_trend(semantic_in_window, cmp_in_window)

        # Adaptive thresholds
        suggested_window, suggested_stale = self._compute_adaptive_thresholds(
            combined_health,
            cmp_in_window[-1].cmp_score if cmp_in_window else 0.5,
        )

        metrics = CorrelationMetrics(
            timestamp=now,
            motif_cmp_correlation=motif_cmp_correlation,
            semantic_cmp_alignment=semantic_cmp_alignment,
            combined_health_score=combined_health,
            trend_direction=trend,
            suggested_window_s=suggested_window,
            suggested_stale_threshold=suggested_stale,
            lineage_ids=list(set(c.lineage_id for c in cmp_in_window)),
            sample_size=len(semantic_in_window),
        )

        self.correlation_history.append(metrics)
        self._update_adaptive_thresholds(metrics)

        return metrics

    def _pearson_correlation(self, x: list, y: list) -> float:
        """Compute Pearson correlation coefficient."""
        n = min(len(x), len(y))
        if n < 2:
            return 0.0

        x = x[:n]
        y = y[:n]

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = (var_x * var_y) ** 0.5

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def _compute_semantic_alignment(
        self,
        semantic: list[SemanticHealthSnapshot],
        cmp: list[CMPSnapshot],
    ) -> float:
        """Compute how well semantic state predicts CMP movement."""
        if len(semantic) < 2 or len(cmp) < 2:
            return 0.5

        # Map semantic states to health scores
        state_scores = {
            "q_good": 1.0,
            "q_observe": 0.7,
            "q_recovering": 0.5,
            "q_stale": 0.3,
            "q_zombie": 0.0,
        }

        semantic_health = [state_scores.get(s.global_state, 0.5) for s in semantic]
        cmp_normalized = []

        for c in cmp[-len(semantic_health):]:
            # Normalize CMP to 0-1 scale
            normalized = min(max(c.cmp_score / 1.0, 0.0), 1.0)
            cmp_normalized.append(normalized)

        if len(cmp_normalized) != len(semantic_health):
            min_len = min(len(cmp_normalized), len(semantic_health))
            cmp_normalized = cmp_normalized[:min_len]
            semantic_health = semantic_health[:min_len]

        # Alignment is how closely they track
        alignment = 1.0 - sum(
            abs(s - c) for s, c in zip(semantic_health, cmp_normalized)
        ) / len(semantic_health)

        return max(0.0, alignment)

    def _compute_combined_health(
        self,
        semantic: SemanticHealthSnapshot,
        cmp: Optional[CMPSnapshot],
    ) -> float:
        """Compute combined health score from semantic and CMP."""
        # Semantic health component (0-1)
        state_health = {
            "q_good": 1.0,
            "q_observe": 0.8,
            "q_recovering": 0.6,
            "q_stale": 0.3,
            "q_zombie": 0.1,
        }
        semantic_score = state_health.get(semantic.global_state, 0.5)

        # Motif completion rate component (scaled)
        # 0.01/s is considered healthy, scale accordingly
        rate_score = min(semantic.motif_completion_rate / 0.01, 1.0)

        # CMP component (normalized to 0-1)
        cmp_score = 0.5
        if cmp:
            cmp_score = min(max(cmp.cmp_score / 1.0, 0.0), 1.0)

        # Weighted combination using PHI ratios
        combined = (
            PHI * semantic_score +
            1.0 * rate_score +
            PHI_INV * cmp_score
        ) / (PHI + 1.0 + PHI_INV)

        return combined

    def _determine_trend(
        self,
        semantic: list[SemanticHealthSnapshot],
        cmp: list[CMPSnapshot],
    ) -> str:
        """Determine overall trend direction."""
        if len(semantic) < 3 or len(cmp) < 3:
            return "stable"

        # Look at recent semantic state transitions
        recent_states = [s.global_state for s in semantic[-5:]]

        state_rank = {
            "q_zombie": 0,
            "q_stale": 1,
            "q_recovering": 2,
            "q_observe": 3,
            "q_good": 4,
        }

        recent_ranks = [state_rank.get(s, 2) for s in recent_states]

        if len(recent_ranks) >= 3:
            first_half = sum(recent_ranks[:len(recent_ranks)//2])
            second_half = sum(recent_ranks[len(recent_ranks)//2:])

            if second_half > first_half + 0.5:
                return "improving"
            elif second_half < first_half - 0.5:
                return "declining"

        return "stable"

    def _compute_adaptive_thresholds(
        self,
        combined_health: float,
        avg_cmp: float,
    ) -> tuple[float, int]:
        """
        Compute adaptive thresholds based on system health.

        When system is healthy, we can afford stricter monitoring.
        When system is stressed, we relax thresholds to avoid false alarms.
        """
        base_window = 120.0
        base_stale = 3

        # Adjust window based on CMP
        # High CMP = productive, can be more strict
        # Low CMP = struggling, be more lenient
        cmp_factor = 1.0 + (avg_cmp - 0.5) * PHI_INV

        # Adjust stale threshold based on combined health
        # Low health = more lenient (higher threshold)
        health_factor = 1.0 + (1.0 - combined_health) * 2.0

        suggested_window = base_window / max(cmp_factor, 0.5)
        suggested_stale = max(2, int(base_stale * health_factor))

        return suggested_window, suggested_stale

    def _update_adaptive_thresholds(self, metrics: CorrelationMetrics) -> None:
        """Update stored adaptive thresholds based on metrics."""
        self.current_thresholds.window_s = metrics.suggested_window_s
        self.current_thresholds.stale_threshold = metrics.suggested_stale_threshold

    def _emit_correlation(self, metrics: CorrelationMetrics) -> None:
        """Emit omega.cmp.correlation event."""
        self.emit_bus(
            topic="omega.cmp.correlation",
            kind="metric",
            level="info",
            data={
                "timestamp": metrics.timestamp,
                "motif_cmp_correlation": round(metrics.motif_cmp_correlation, 4),
                "semantic_cmp_alignment": round(metrics.semantic_cmp_alignment, 4),
                "combined_health_score": round(metrics.combined_health_score, 4),
                "trend_direction": metrics.trend_direction,
                "adaptive_thresholds": {
                    "window_s": round(metrics.suggested_window_s, 2),
                    "stale_threshold": metrics.suggested_stale_threshold,
                },
                "sample_size": metrics.sample_size,
                "lineage_count": len(metrics.lineage_ids),
            },
        )

    def get_current_thresholds(self) -> AdaptiveThresholds:
        """Get current adaptive thresholds for Omega Guardian."""
        return self.current_thresholds

    def get_correlation_summary(self) -> dict:
        """Get summary of recent correlation metrics."""
        if not self.correlation_history:
            return {
                "status": "insufficient_data",
                "sample_count": 0,
            }

        recent = list(self.correlation_history)[-10:]

        return {
            "status": "active",
            "sample_count": len(recent),
            "avg_motif_cmp_correlation": sum(m.motif_cmp_correlation for m in recent) / len(recent),
            "avg_combined_health": sum(m.combined_health_score for m in recent) / len(recent),
            "current_trend": recent[-1].trend_direction,
            "current_thresholds": asdict(self.current_thresholds),
        }


# Singleton instance for module-level access
_correlator_instance: Optional[OmegaCMPCorrelator] = None


def get_correlator() -> OmegaCMPCorrelator:
    """Get or create the singleton correlator instance."""
    global _correlator_instance
    if _correlator_instance is None:
        _correlator_instance = OmegaCMPCorrelator()
    return _correlator_instance


def process_event(event: dict) -> Optional[CorrelationMetrics]:
    """Process an event through the correlator."""
    correlator = get_correlator()
    topic = event.get("topic", "")

    if topic == "omega.guardian.semantic.cycle":
        return correlator.process_semantic_cycle(event)
    elif topic in ("cmp.score", "cmp.lineage.update", "cmp.entropy.weighted"):
        correlator.process_cmp_update(event)

    return None


if __name__ == "__main__":
    # Demo/test mode
    print("Omega-CMP Correlator - Demo Mode")

    correlator = OmegaCMPCorrelator()

    # Simulate some events
    now = time.time()

    for i in range(10):
        # Simulate semantic cycle
        event = {
            "ts": now + i * 10,
            "topic": "omega.guardian.semantic.cycle",
            "data": {
                "state": "q_good" if i > 3 else "q_stale",
                "completion_rate": 0.005 + i * 0.001,
                "actors_healthy": 3,
                "actors_stale": 1 if i < 3 else 0,
                "actors_zombie": 0,
                "total_completions": i * 5,
                "cycle_number": i,
            },
        }
        result = correlator.process_semantic_cycle(event)

        # Simulate CMP update
        cmp_event = {
            "ts": now + i * 10 + 5,
            "topic": "cmp.score",
            "data": {
                "lineage_id": f"lineage-{i % 3}",
                "cmp_score": 0.4 + i * 0.05,
                "entropy_factor": 0.95,
            },
        }
        correlator.process_cmp_update(cmp_event)

        if result:
            print(f"Correlation computed: {result.combined_health_score:.3f}")

    summary = correlator.get_correlation_summary()
    print(f"\nFinal summary: {json.dumps(summary, indent=2)}")
