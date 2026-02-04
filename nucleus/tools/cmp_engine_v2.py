#!/usr/bin/env python3
"""
CMP Engine v2 - Extended Clade Metaproductivity Engine with Entropy and Semantic Components.

This extends the base CMP Engine (cmp_engine.py) with:
- LENS/LASER entropy vector integration
- Omega-motif contribution tracking
- Semantic liveness factors
- Extended CMP scoring formula

Implements Section 4 of cmp_lens_omega_integration_v1.md

Author: Claude Opus 4.5 (codex-peer)
Date: 2025-12-30
DKIN Version: v28
"""
import os
import sys
import json
import time
import uuid
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

# REPL_HEADER: {"contract":"repl_header.v1","agent":"claude","dkin_version":"v28","paip_version":"v15","citizen_version":"v1","attestation":{"date":"2025-12-30T07:15:00Z","score":"100/100","score_percent":100,"basis":["citizen","dkin","paip"],"verifier":"codex-peer","event_id":"cmp-engine-v2-init"}}

# =============================================================================
# Constants (Golden Ratio based, from .clade-manifest.json)
# =============================================================================

PHI = 1.618033988749895
PHI_INV = 1.0 / PHI           # ~0.618 (GOOD threshold)
PHI_INV2 = 1.0 / PHI**2       # ~0.382 (FAIR threshold)
PHI_INV3 = 1.0 / PHI**3       # ~0.236 (CRITICAL threshold)

FITNESS_THRESHOLDS = {
    "excellent": 1.0,
    "good": PHI_INV,      # 0.618
    "fair": PHI_INV2,     # 0.382
    "poor": PHI_INV3,     # 0.236
    "critical": 0.0
}

# Component weights for scoring
COMPONENT_WEIGHTS = {
    # Original weights
    "task_completion": PHI,         # ~1.618 - Primary
    "test_coverage": 1.0,
    "guard_pass_rate": 1.0,
    "motif_recurrence": PHI_INV,    # ~0.618
    "mdl_complexity": PHI_INV2,     # ~0.382
    "descendant_cmp": PHI_INV3,     # ~0.236

    # Entropy weights
    "entropy_factor": 1.0,          # Multiplicative modifier
    "information_density": PHI_INV, # ~0.618 - reward high signal
    "hallucination_rate": -PHI,     # ~-1.618 - penalize conjecture
    "cognitive_efficiency": PHI_INV2,  # ~0.382

    # Omega-motif weights
    "omega_motif_rate": 1.0,        # Direct bonus
    "semantic_liveness": PHI_INV,   # ~0.618 - gate for stale lineages
}

# Semantic state to liveness factor mapping
SEMANTIC_LIVENESS_FACTORS = {
    "q_observe": 0.9,
    "q_good": 1.0,
    "q_stale": 0.5,
    "q_recovering": 0.7,
    "q_zombie": 0.0,
}

# Motif bonuses by ID
MOTIF_BONUSES = {
    "inference_complete": 0.05,
    "inference_simple": 0.04,
    "task_completion_cycle": 0.075,
    "codex_task_flow": 0.06,
    "pr_review_cycle": 0.065,
    "git_commit_flow": 0.04,
    "oiterate_tick_healthy": 0.015,
    "codemaster_merge": 0.07,
    "strp_request_done": 0.055,
    "browser_action_success": 0.035,
    "vor_check_pass": 0.05,
    "agent_heartbeat_with_work": 0.025,
    "test_run_pass": 0.06,
    "synthesis_complete": 0.10,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EntropyVector:
    """8-dimensional entropy vector from LENS/LASER."""
    h_info: float = 0.0       # Information density [0,1]
    h_miss: float = 0.0       # Missing information [0,1]
    h_conj: float = 0.0       # Hallucination/conjecture rate [0,1]
    h_alea: float = 0.0       # Aleatory uncertainty [0,1]
    h_epis: float = 0.0       # Epistemic uncertainty [0,1]
    h_struct: float = 0.0     # Structural entropy [0,1]
    c_load: float = 0.0       # Cognitive load [0,1]
    h_goal_drift: float = 0.0 # Goal drift [0,1]

    @property
    def h_total(self) -> float:
        """Sum of all entropy components (excluding h_info which is positive)."""
        return self.h_miss + self.h_conj + self.h_alea + self.h_epis + self.h_struct + self.c_load + self.h_goal_drift

    @property
    def h_mean(self) -> float:
        """Mean of failure entropies."""
        components = [self.h_miss, self.h_conj, self.h_alea, self.h_epis, self.h_struct, self.c_load, self.h_goal_drift]
        return sum(components) / len(components)

    @property
    def utility(self) -> float:
        """Utility score: h_info - h_mean (higher is better)."""
        return self.h_info - self.h_mean

    def to_dict(self) -> dict:
        """Convert to dictionary with computed fields."""
        d = asdict(self)
        d["h_total"] = self.h_total
        d["h_mean"] = self.h_mean
        d["utility"] = self.utility
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EntropyVector":
        """Create from dictionary, ignoring computed fields."""
        return cls(
            h_info=data.get("h_info", 0.0),
            h_miss=data.get("h_miss", 0.0),
            h_conj=data.get("h_conj", 0.0),
            h_alea=data.get("h_alea", 0.0),
            h_epis=data.get("h_epis", 0.0),
            h_struct=data.get("h_struct", 0.0),
            c_load=data.get("c_load", 0.0),
            h_goal_drift=data.get("h_goal_drift", 0.0),
        )


@dataclass
class FitnessComponents:
    """Extended fitness components for CMP calculation."""
    # Original components
    task_completion: float = 0.0
    test_coverage: float = 0.0
    guard_pass_rate: float = 0.0
    motif_recurrence: float = 0.0
    mdl_complexity: float = 1.0
    descendant_cmp: float = 0.0

    # Entropy-derived components (from LENS/LASER)
    entropy_factor: float = 1.0       # E_factor from H*
    information_density: float = 0.0  # H_info
    hallucination_rate: float = 0.0   # H_conj
    cognitive_efficiency: float = 1.0 # 1 / (1 + C_load)

    # Omega-motif derived
    omega_motif_rate: float = 0.0     # Completions per window
    semantic_liveness: float = 1.0    # 1.0 if q_good, decay if q_stale


@dataclass
class PerActorCMP:
    """Track CMP contributions per actor within a lineage."""
    actor: str
    lineage_id: str

    # Entropy contributions
    entropy_updates: int = 0
    cumulative_e_factor: float = 0.0
    avg_h_mean: float = 0.5

    # Motif contributions
    motifs_completed: int = 0
    cumulative_motif_bonus: float = 0.0

    # Semantic state (from Omega Guardian)
    current_state: str = "q_observe"
    stale_windows: int = 0

    def contribution_score(self) -> float:
        """Compute this actor's contribution to lineage CMP."""
        entropy_contrib = self.cumulative_e_factor / max(self.entropy_updates, 1)
        motif_contrib = self.cumulative_motif_bonus

        # Apply semantic liveness penalty
        liveness_factor = SEMANTIC_LIVENESS_FACTORS.get(self.current_state, 0.5)

        return (entropy_contrib + motif_contrib) * liveness_factor


@dataclass
class LineageState:
    """Extended lineage state with entropy and semantic tracking."""
    lineage_id: str
    parent_id: Optional[str] = None

    # Base metrics
    reward: float = 0.0
    events_count: int = 0
    cmp_score: float = 0.0

    # Extended components
    components: FitnessComponents = field(default_factory=FitnessComponents)

    # Entropy tracking
    entropy_profile: Optional[EntropyVector] = None
    entropy_history: list = field(default_factory=list)  # Rolling window

    # Omega-motif tracking
    motifs_completed: int = 0
    last_motif_ts: float = 0.0
    motif_bonus_total: float = 0.0

    # Semantic state
    semantic_state: str = "q_observe"
    omega_acceptance_rate: float = 1.0


# =============================================================================
# Core Functions
# =============================================================================

def compute_e_factor(h: EntropyVector) -> float:
    """
    Compute entropy factor from LENS H* vector.

    E_factor = H_info * (1-H_miss) * (1-H_conj) * (1-H_alea) * (1-H_epis) / (1 + C_load)

    Returns value in [0, 2] range.
    """
    product = (1 - h.h_miss) * (1 - h.h_conj) * (1 - h.h_alea) * (1 - h.h_epis)
    e_factor = h.h_info * product / (1 + h.c_load)
    return max(0.0, min(2.0, e_factor))


def aggregate_entropy_vectors(profiles: list[dict]) -> EntropyVector:
    """
    Aggregate entropy vectors from multiple models.

    Uses weighted mean where higher-utility models have more influence.
    """
    if not profiles:
        return EntropyVector()

    # Parse vectors
    vectors = []
    utilities = []
    for p in profiles:
        if isinstance(p, dict):
            if "entropy_profile" in p:
                v = EntropyVector.from_dict(p["entropy_profile"])
            elif "h_info" in p:
                v = EntropyVector.from_dict(p)
            else:
                continue
            vectors.append(v)
            utilities.append(max(0.01, v.utility + 1))  # Ensure positive weight

    if not vectors:
        return EntropyVector()

    # Weighted average
    total_weight = sum(utilities)
    result = EntropyVector()

    for v, w in zip(vectors, utilities):
        weight = w / total_weight
        result.h_info += v.h_info * weight
        result.h_miss += v.h_miss * weight
        result.h_conj += v.h_conj * weight
        result.h_alea += v.h_alea * weight
        result.h_epis += v.h_epis * weight
        result.h_struct += v.h_struct * weight
        result.c_load += v.c_load * weight
        result.h_goal_drift += v.h_goal_drift * weight

    return result


def compute_motif_bonus(
    motif_event: dict,
    lineage_id: str,
    window_s: float = 120.0,
    base_bonus: float = 0.05,
) -> float:
    """
    Compute CMP bonus from omega-motif completion.

    Args:
        motif_event: omega.guardian.semantic.motif_complete event
        lineage_id: The lineage to credit
        window_s: Evaluation window in seconds
        base_bonus: Base bonus per completion (default 5%)

    Returns:
        CMP delta to add
    """
    data = motif_event.get("data", {})
    motif_id = data.get("motif_id", "unknown")
    weight = data.get("weight", 1.0)
    duration = data.get("duration_s", 60.0)

    # Use per-motif bonus if defined
    base_bonus = MOTIF_BONUSES.get(motif_id, base_bonus)

    # Faster completion = higher bonus (capped at 2x)
    speed_factor = min(2.0, window_s / max(duration, 1.0))

    # Heavy motifs (multi-step workflows) count more
    weight_factor = min(2.0, weight)

    return base_bonus * speed_factor * weight_factor


def classify_cmp(score: float) -> str:
    """Classify CMP score into fitness category."""
    if score >= FITNESS_THRESHOLDS["excellent"]:
        return "excellent"
    elif score >= FITNESS_THRESHOLDS["good"]:
        return "good"
    elif score >= FITNESS_THRESHOLDS["fair"]:
        return "fair"
    elif score >= FITNESS_THRESHOLDS["poor"]:
        return "poor"
    else:
        return "critical"


def adaptive_omega_thresholds(cmp_score: float) -> dict:
    """
    Compute adaptive Omega Guardian thresholds based on CMP.

    High-CMP lineages are expected to maintain semantic health.
    Low-CMP lineages get more slack (they're already struggling).
    """
    if cmp_score >= FITNESS_THRESHOLDS["excellent"]:
        return {
            "window_s": 60,        # Stricter: 1 minute window
            "stale_threshold": 2,  # Fewer stale windows before zombie
            "min_motif_rate": 0.02 # Higher expected throughput
        }
    elif cmp_score >= FITNESS_THRESHOLDS["good"]:
        return {
            "window_s": 120,       # Standard: 2 minute window
            "stale_threshold": 3,
            "min_motif_rate": 0.01
        }
    elif cmp_score >= FITNESS_THRESHOLDS["fair"]:
        return {
            "window_s": 180,       # Lenient: 3 minute window
            "stale_threshold": 5,
            "min_motif_rate": 0.005
        }
    else:
        return {
            "window_s": 300,       # Very lenient: 5 minute window
            "stale_threshold": 10,
            "min_motif_rate": 0.001
        }


# =============================================================================
# Extended CMP Engine
# =============================================================================

class CMPEngineV2:
    """
    Extended CMP Engine with entropy and semantic components.

    Subscribes to:
    - lens_laser.cmp.update (entropy profiles)
    - omega.guardian.semantic.cycle (semantic state)
    - omega.guardian.semantic.motif_complete (motif bonuses)
    - omega.cmp.correlation (correlation metrics)

    Emits:
    - cmp.entropy.weighted (CMP with entropy breakdown)
    - cmp.score (extended with semantic_context field)
    - cmp.lineage.update (backward-compatible base event)
    """

    def __init__(
        self,
        bus_dir: str | Path,
        poll_interval: float = 2.0,
        discount: float = 0.9,
        smoothing: float = PHI,
        entropy_history_size: int = 10,
    ):
        self.bus_dir = Path(bus_dir)
        self.events_path = self.bus_dir / "events.ndjson"

        # Lineage tracking
        self.lineages: dict[str, LineageState] = {}
        self.tree: dict[str, list[str]] = defaultdict(list)  # parent -> children

        # Per-actor tracking
        self.per_actor_state: dict[str, PerActorCMP] = {}

        # Event processing state
        self.last_pos = 0
        self.poll_interval = float(poll_interval)
        self.discount = float(discount)
        self.smoothing = float(smoothing)
        self.entropy_history_size = entropy_history_size
        self.running = True
        self.actor = "cmp-engine-v2"

        # Subscribed topic handlers
        self.handlers = {
            "lens_laser.cmp.update": self._handle_lens_laser_cmp_update,
            "lens.superposition.complete": self._handle_lens_complete,
            "omega.guardian.semantic.cycle": self._handle_omega_cycle,
            "omega.guardian.semantic.motif_complete": self._handle_motif_complete,
            "omega.guardian.semantic.zombie": self._handle_zombie,
            "omega.cmp.correlation": self._handle_correlation,
        }

    # -------------------------------------------------------------------------
    # Event Emission
    # -------------------------------------------------------------------------

    def emit(self, topic: str, kind: str, level: str, data: dict) -> None:
        """Emit event to bus."""
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.actor,
            "data": data
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")

    def emit_entropy_weighted(
        self,
        lineage_id: str,
        req_id: str,
        entropy_vector: EntropyVector,
        e_factor: float,
        cmp_before: float,
        cmp_after: float,
        models_contributing: list[str] = None,
        synthesis_mode: str = "lens_laser",
    ) -> None:
        """Emit cmp.entropy.weighted event."""
        self.emit("cmp.entropy.weighted", "metric", "info", {
            "lineage_id": lineage_id,
            "req_id": req_id,
            "entropy_vector": entropy_vector.to_dict(),
            "e_factor": round(e_factor, 4),
            "cmp_before": round(cmp_before, 4),
            "cmp_after": round(cmp_after, 4),
            "delta": round(cmp_after - cmp_before, 4),
            "models_contributing": models_contributing or [],
            "synthesis_mode": synthesis_mode,
        })

    def emit_cmp_score(
        self,
        lineage_id: str,
        cmp_before: float,
        cmp_after: float,
        semantic_context: dict = None,
    ) -> None:
        """Emit extended cmp.score event with semantic context."""
        l = self.lineages.get(lineage_id)
        classification = classify_cmp(cmp_after)

        data = {
            "lineage_id": lineage_id,
            "score": round(cmp_after, 4),
            "classification": classification,
            "delta": round(cmp_after - cmp_before, 4),
            "components": {
                "entropy_factor": round(l.components.entropy_factor, 4) if l else 1.0,
                "information_density": round(l.components.information_density, 4) if l else 0.0,
                "hallucination_rate": round(l.components.hallucination_rate, 4) if l else 0.0,
                "cognitive_efficiency": round(l.components.cognitive_efficiency, 4) if l else 1.0,
                "omega_motif_rate": round(l.components.omega_motif_rate, 4) if l else 0.0,
                "semantic_liveness": round(l.components.semantic_liveness, 4) if l else 1.0,
                "motif_recurrence": l.motifs_completed if l else 0,
            },
            "adaptive_thresholds": adaptive_omega_thresholds(cmp_after),
        }

        if semantic_context:
            data["semantic_context"] = semantic_context

        self.emit("cmp.score", "metric", "info", data)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _handle_lens_laser_cmp_update(self, event: dict) -> None:
        """Handle lens_laser.cmp.update event containing entropy profile."""
        data = event.get("data", {})
        lineage_id = data.get("lineage_id") or self._resolve_lineage_from_event(event)
        if not lineage_id:
            return

        req_id = data.get("req_id", event.get("id", "unknown"))

        # Extract entropy profile
        entropy_data = data.get("entropy_profile", {})
        entropy_vector = EntropyVector.from_dict(entropy_data)

        # Compute E_factor
        e_factor = compute_e_factor(entropy_vector)

        # Update lineage
        l = self._ensure_lineage(lineage_id)
        cmp_before = l.cmp_score

        # Update entropy components
        l.entropy_profile = entropy_vector
        l.entropy_history.append(entropy_vector.to_dict())
        if len(l.entropy_history) > self.entropy_history_size:
            l.entropy_history.pop(0)

        l.components.entropy_factor = e_factor
        l.components.information_density = entropy_vector.h_info
        l.components.hallucination_rate = entropy_vector.h_conj
        l.components.cognitive_efficiency = 1.0 / (1.0 + entropy_vector.c_load)

        # Recompute CMP
        self._update_cmp_extended(lineage_id)
        cmp_after = l.cmp_score

        # Emit entropy-weighted event
        models = data.get("models_contributing", [])
        mode = data.get("synthesis_mode", "lens_laser")
        self.emit_entropy_weighted(
            lineage_id, req_id, entropy_vector, e_factor,
            cmp_before, cmp_after, models, mode
        )

        print(f"[CMP-v2] Entropy update for {lineage_id}: e_factor={e_factor:.3f}, CMP {cmp_before:.3f} -> {cmp_after:.3f}")

    def _handle_lens_complete(self, event: dict) -> None:
        """Handle lens.superposition.complete event."""
        data = event.get("data", {})
        req_id = data.get("req_id", "")

        # Get entropy profiles from all models
        profiles = data.get("entropy_profiles", [])
        if not profiles:
            return

        # Aggregate entropy vectors
        agg_entropy = aggregate_entropy_vectors(profiles)
        e_factor = compute_e_factor(agg_entropy)

        # Resolve lineage
        lineage_id = self._resolve_lineage_from_event(event)
        if not lineage_id:
            # Create a synthetic lineage based on the request
            lineage_id = f"lens.{req_id[:8]}" if req_id else "lens.unknown"

        l = self._ensure_lineage(lineage_id)
        cmp_before = l.cmp_score

        # Update components
        l.entropy_profile = agg_entropy
        l.components.entropy_factor = e_factor
        l.components.information_density = agg_entropy.h_info
        l.components.hallucination_rate = agg_entropy.h_conj
        l.components.cognitive_efficiency = 1.0 / (1.0 + agg_entropy.c_load)

        # Recompute
        self._update_cmp_extended(lineage_id)
        cmp_after = l.cmp_score

        # Extract model names
        models = [p.get("model", "unknown") for p in profiles if isinstance(p, dict)]

        self.emit_entropy_weighted(
            lineage_id, req_id, agg_entropy, e_factor,
            cmp_before, cmp_after, models, "lens_laser"
        )

        print(f"[CMP-v2] LENS complete for {lineage_id}: {len(profiles)} models, e_factor={e_factor:.3f}")

    def _handle_motif_complete(self, event: dict) -> None:
        """Handle omega.guardian.semantic.motif_complete event."""
        data = event.get("data", {})
        actor = data.get("actor", event.get("actor", "unknown"))
        motif_id = data.get("motif_id", "unknown")

        # Resolve lineage from actor
        lineage_id = self._actor_to_lineage(actor)
        if not lineage_id:
            lineage_id = f"actor.{actor}"

        # Compute bonus
        bonus = compute_motif_bonus(event, lineage_id)

        # Update per-actor state
        if actor not in self.per_actor_state:
            self.per_actor_state[actor] = PerActorCMP(actor=actor, lineage_id=lineage_id)

        pa = self.per_actor_state[actor]
        pa.motifs_completed += 1
        pa.cumulative_motif_bonus += bonus
        pa.current_state = "q_good"

        # Update lineage
        l = self._ensure_lineage(lineage_id)
        cmp_before = l.cmp_score

        l.motifs_completed += 1
        l.last_motif_ts = time.time()
        l.motif_bonus_total += bonus
        l.semantic_state = "q_good"
        l.components.omega_motif_rate = l.motifs_completed / 100.0
        l.components.semantic_liveness = 1.0

        # Recompute
        self._update_cmp_extended(lineage_id)
        cmp_after = l.cmp_score

        # Emit score with motif context
        self.emit_cmp_score(lineage_id, cmp_before, cmp_after, semantic_context={
            "trigger": "motif_complete",
            "motif_id": motif_id,
            "bonus": round(bonus, 4),
            "actor": actor,
            "total_motifs": l.motifs_completed,
        })

        print(f"[CMP-v2] Motif complete: {motif_id} by {actor}, bonus={bonus:.4f}, CMP {cmp_before:.3f} -> {cmp_after:.3f}")

    def _handle_omega_cycle(self, event: dict) -> None:
        """Handle omega.guardian.semantic.cycle event."""
        data = event.get("data", {})
        per_actor = data.get("per_actor_state", {})
        cycle = data.get("cycle", 0)

        for actor, actor_state in per_actor.items():
            state = actor_state.get("state", "q_observe")

            if actor not in self.per_actor_state:
                lineage_id = self._actor_to_lineage(actor)
                if lineage_id:
                    self.per_actor_state[actor] = PerActorCMP(actor=actor, lineage_id=lineage_id)

            if actor in self.per_actor_state:
                pa = self.per_actor_state[actor]
                pa.current_state = state

                # Update lineage semantic liveness
                if pa.lineage_id and pa.lineage_id in self.lineages:
                    l = self.lineages[pa.lineage_id]
                    liveness = SEMANTIC_LIVENESS_FACTORS.get(state, 0.5)
                    l.components.semantic_liveness = liveness
                    l.semantic_state = state

                    # Recompute if state changed significantly
                    if state in ("q_zombie", "q_stale"):
                        cmp_before = l.cmp_score
                        self._update_cmp_extended(pa.lineage_id)
                        cmp_after = l.cmp_score

                        if abs(cmp_after - cmp_before) > 0.01:
                            self.emit_cmp_score(pa.lineage_id, cmp_before, cmp_after, semantic_context={
                                "trigger": "omega_cycle",
                                "cycle": cycle,
                                "state": state,
                            })

    def _handle_zombie(self, event: dict) -> None:
        """Handle omega.guardian.semantic.zombie event."""
        data = event.get("data", {})
        target_actor = data.get("target_actor", "unknown")

        lineage_id = self._actor_to_lineage(target_actor)
        if not lineage_id:
            return

        if lineage_id in self.lineages:
            l = self.lineages[lineage_id]
            cmp_before = l.cmp_score

            # Zero out semantic liveness
            l.components.semantic_liveness = 0.0
            l.semantic_state = "q_zombie"

            # Recompute
            self._update_cmp_extended(lineage_id)
            cmp_after = l.cmp_score

            # Emit decay warning
            self.emit("cmp.decay_warning", "alert", "warn", {
                "lineage_id": lineage_id,
                "reason": "omega_zombie_detected",
                "actor": target_actor,
                "cmp_before": round(cmp_before, 4),
                "cmp_after": round(cmp_after, 4),
            })

            print(f"[CMP-v2] ZOMBIE detected for {target_actor}/{lineage_id}: CMP {cmp_before:.3f} -> {cmp_after:.3f}")

    def _handle_correlation(self, event: dict) -> None:
        """Handle omega.cmp.correlation event."""
        data = event.get("data", {})
        lineage_id = data.get("lineage_id")

        if not lineage_id or lineage_id not in self.lineages:
            return

        l = self.lineages[lineage_id]

        # Update omega acceptance rate from correlation
        acceptance = data.get("omega_acceptance_rate", 1.0)
        l.omega_acceptance_rate = acceptance

        # Potentially adjust CMP based on correlation
        correlation_factor = data.get("cmp_correlation", 1.0)
        if correlation_factor != 1.0:
            cmp_before = l.cmp_score
            l.cmp_score *= correlation_factor
            l.cmp_score = max(0.0, min(2.0, l.cmp_score))

            if abs(l.cmp_score - cmp_before) > 0.01:
                self.emit_cmp_score(l.lineage_id, cmp_before, l.cmp_score, semantic_context={
                    "trigger": "correlation_update",
                    "correlation_factor": correlation_factor,
                    "omega_acceptance_rate": acceptance,
                })

    # -------------------------------------------------------------------------
    # CMP Computation
    # -------------------------------------------------------------------------

    def _update_cmp_extended(self, lineage_id: str) -> None:
        """Compute extended CMP score with entropy and semantic factors."""
        l = self.lineages.get(lineage_id)
        if not l:
            return

        c = l.components

        # Step 1: Base score from reward
        base_score = l.reward

        # Step 2: Aggregate children contribution
        children = self.tree.get(lineage_id, [])
        if children:
            child_contribution = sum(
                self.lineages[cid].cmp_score
                for cid in children
                if cid in self.lineages
            )
            base_score = (base_score + self.discount * (child_contribution / len(children))) / self.smoothing

        # Step 3: Apply entropy factor (multiplicative)
        # E_factor = (1 - H*_failure_mean) * (1 + bonus)
        h_failure_mean = 0.0
        if l.entropy_profile:
            h_failure_mean = l.entropy_profile.h_mean

        # Motif bonus contribution
        motif_recurrence_rate = min(1.0, l.motifs_completed / 10.0)  # Saturates at 10
        motif_bonus = COMPONENT_WEIGHTS["motif_recurrence"] * motif_recurrence_rate * l.omega_acceptance_rate

        e_factor = (1 - h_failure_mean) * (1 + motif_bonus)
        entropy_modified = base_score * c.entropy_factor * e_factor

        # Step 4: Apply information density bonus
        if c.information_density > 0.5:
            info_bonus = (c.information_density - 0.5) * PHI_INV
            entropy_modified += info_bonus

        # Step 5: Apply hallucination penalty
        if c.hallucination_rate > 0.05:
            hallu_penalty = c.hallucination_rate * PHI
            entropy_modified -= hallu_penalty

        # Step 6: Omega-motif bonus
        omega_bonus = c.omega_motif_rate * 0.1

        # Step 7: Semantic liveness gate + PHI_INV contribution
        # CMP_extended = CMP_base * E_factor + semantic_liveness * PHI_INV
        final_score = (entropy_modified + omega_bonus) * c.semantic_liveness
        final_score += c.semantic_liveness * PHI_INV

        # Clamp to [0, 2]
        final_score = max(0.0, min(2.0, final_score))

        # Update if changed
        if abs(l.cmp_score - final_score) > 0.0001:
            old_score = l.cmp_score
            l.cmp_score = final_score

            # Emit backward-compatible event
            self.emit("cmp.lineage.update", "metric", "info", {
                "lineage_id": lineage_id,
                "cmp_score": round(final_score, 4),
                "parent_id": l.parent_id,
            })

            # Propagate to parent
            if l.parent_id and l.parent_id in self.lineages:
                self._update_cmp_extended(l.parent_id)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _ensure_lineage(self, lineage_id: str, parent_id: str = None) -> LineageState:
        """Ensure lineage exists, creating if necessary."""
        if lineage_id not in self.lineages:
            self.lineages[lineage_id] = LineageState(
                lineage_id=lineage_id,
                parent_id=parent_id,
            )
            if parent_id:
                self.tree[parent_id].append(lineage_id)
            print(f"[CMP-v2] Registered lineage: {lineage_id} (parent: {parent_id})")
        return self.lineages[lineage_id]

    def _resolve_lineage_from_event(self, event: dict) -> Optional[str]:
        """Attempt to resolve lineage_id from event data."""
        data = event.get("data", {})

        # Direct lineage_id
        if "lineage_id" in data:
            return data["lineage_id"]

        # From actor
        actor = event.get("actor", data.get("actor"))
        if actor:
            return self._actor_to_lineage(actor)

        return None

    def _actor_to_lineage(self, actor: str) -> Optional[str]:
        """Map actor to lineage. Override for custom mappings."""
        # Check if we have an existing mapping in per_actor_state
        if actor in self.per_actor_state:
            return self.per_actor_state[actor].lineage_id

        # Default: create lineage from actor
        return f"actor.{actor}"

    # -------------------------------------------------------------------------
    # Base Event Processing (backward compatible)
    # -------------------------------------------------------------------------

    def process_event(self, event: dict) -> None:
        """Process incoming event, delegating to appropriate handler."""
        topic = event.get("topic", "")

        # Check for specialized handlers
        if topic in self.handlers:
            self.handlers[topic](event)
            return

        # Fall back to base processing for backward compatibility
        data = event.get("data", {})
        if not isinstance(data, dict):
            return

        lineage_id = data.get("lineage_id")
        if not lineage_id:
            return

        # Basic lineage update
        parent_id = data.get("parent_lineage_id")
        l = self._ensure_lineage(lineage_id, parent_id)
        l.events_count += 1

        # Capture rewards and entropy if present
        if "reward" in data:
            l.reward = float(data["reward"])
            print(f"[CMP-v2] Updated reward for {lineage_id}: {l.reward}")

        if "entropy_profile" in data:
            l.entropy_profile = EntropyVector.from_dict(data["entropy_profile"])
            l.components.entropy_factor = compute_e_factor(l.entropy_profile)

        # Recalculate
        self._update_cmp_extended(lineage_id)

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Main event processing loop."""
        print(f"CMP Engine v2 starting. Watching {self.events_path}")
        print(f"  - Discount: {self.discount}")
        print(f"  - Smoothing: {self.smoothing}")
        print(f"  - Poll interval: {self.poll_interval}s")
        print(f"  - Subscribed topics: {list(self.handlers.keys())}")

        if self.events_path.exists():
            self.last_pos = self.events_path.stat().st_size

        while self.running:
            if not self.events_path.exists():
                time.sleep(self.poll_interval)
                continue

            try:
                with self.events_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(self.last_pos)
                    lines = f.readlines()
                    self.last_pos = f.tell()

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            self.process_event(event)
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"[CMP-v2] Error processing event: {e}")
            except Exception as e:
                print(f"[CMP-v2] Error reading events: {e}")

            time.sleep(self.poll_interval)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CMP Engine v2 - Extended with entropy and semantic components"
    )
    parser.add_argument(
        "--bus-dir",
        default="/pluribus/.pluribus/bus",
        help="Path to bus directory containing events.ndjson"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Event poll interval in seconds"
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.9,
        help="Temporal discount factor for child contributions"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=PHI,
        help="Golden ratio smoothing factor"
    )
    parser.add_argument(
        "--entropy-history",
        type=int,
        default=10,
        help="Number of entropy vectors to keep in history per lineage"
    )

    args = parser.parse_args()

    engine = CMPEngineV2(
        bus_dir=args.bus_dir,
        poll_interval=args.poll_interval,
        discount=args.discount,
        smoothing=args.smoothing,
        entropy_history_size=args.entropy_history,
    )

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nCMP Engine v2 stopped.")


if __name__ == "__main__":
    main()
