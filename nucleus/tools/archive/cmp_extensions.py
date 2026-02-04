#!/usr/bin/env python3
"""
CMP Extensions - Extracted functions from cmp_engine_v2.py for integration.

These functions extend the base CMP engine with:
- LENS entropy factor computation
- Omega motif bonus calculation
- Adaptive threshold computation

Author: claude (P0 Integration Sprint)
DKIN Version: v28
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# Golden ratio constants
PHI = 1.618033988749895
PHI_INV = 0.6180339887498949  # 1/φ
PHI_INV2 = 0.3819660112501051  # 1/φ²
PHI_INV3 = 0.2360679774997896  # 1/φ³

# Fitness thresholds using phi sequence
FITNESS_THRESHOLDS = {
    "excellent": 1.0,
    "good": PHI_INV,      # 0.618
    "fair": PHI_INV2,     # 0.382
    "poor": PHI_INV3,     # 0.236
    "critical": 0.0
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


@dataclass
class EntropyVectorCompat:
    """
    Compatibility wrapper for entropy vectors.
    Works with both lens_entropy_profiler.EntropyVector and dict formats.
    """
    h_info: float = 0.0
    h_miss: float = 0.0
    h_conj: float = 0.0
    h_alea: float = 0.0
    h_epis: float = 0.0
    h_struct: float = 0.0
    c_load: float = 0.0
    h_goal_drift: float = 0.0

    @property
    def h_total(self) -> float:
        return self.h_miss + self.h_conj + self.h_alea + self.h_epis + self.h_struct + self.c_load + self.h_goal_drift

    @property
    def h_mean(self) -> float:
        components = [self.h_miss, self.h_conj, self.h_alea, self.h_epis, self.h_struct, self.c_load, self.h_goal_drift]
        return sum(components) / len(components)

    @property
    def utility(self) -> float:
        return self.h_info - self.h_mean

    def to_dict(self) -> dict:
        d = asdict(self)
        d["h_total"] = self.h_total
        d["h_mean"] = self.h_mean
        d["utility"] = self.utility
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EntropyVectorCompat":
        return cls(
            h_info=float(data.get("h_info", 0.0)),
            h_miss=float(data.get("h_miss", 0.0)),
            h_conj=float(data.get("h_conj", 0.0)),
            h_alea=float(data.get("h_alea", 0.0)),
            h_epis=float(data.get("h_epis", 0.0)),
            h_struct=float(data.get("h_struct", 0.0)),
            c_load=float(data.get("c_load", 0.0)),
            h_goal_drift=float(data.get("h_goal_drift", 0.0)),
        )


def compute_e_factor(h: EntropyVectorCompat | dict) -> float:
    """
    Compute entropy factor from LENS H* vector.

    E_factor = H_info * (1-H_miss) * (1-H_conj) * (1-H_alea) * (1-H_epis) / (1 + C_load)

    Returns value in [0, 2] range.
    """
    if isinstance(h, dict):
        h = EntropyVectorCompat.from_dict(h)

    product = (1 - h.h_miss) * (1 - h.h_conj) * (1 - h.h_alea) * (1 - h.h_epis)
    e_factor = h.h_info * product / (1 + h.c_load)
    return max(0.0, min(2.0, e_factor))


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


def adaptive_omega_thresholds(cmp_score: float) -> dict:
    """
    Compute adaptive Omega Guardian thresholds based on CMP.

    High-CMP lineages are expected to maintain semantic health.
    Low-CMP lineages get more slack (they're already struggling).
    """
    if cmp_score >= FITNESS_THRESHOLDS["excellent"]:
        return {
            "window_s": 60,
            "stale_threshold": 2,
            "min_motif_rate": 0.02
        }
    elif cmp_score >= FITNESS_THRESHOLDS["good"]:
        return {
            "window_s": 120,
            "stale_threshold": 3,
            "min_motif_rate": 0.01
        }
    elif cmp_score >= FITNESS_THRESHOLDS["fair"]:
        return {
            "window_s": 180,
            "stale_threshold": 4,
            "min_motif_rate": 0.005
        }
    else:
        return {
            "window_s": 300,
            "stale_threshold": 5,
            "min_motif_rate": 0.001
        }


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


def aggregate_entropy_vectors(profiles: list[dict]) -> EntropyVectorCompat:
    """
    Aggregate entropy vectors from multiple models.

    Uses weighted mean where higher-utility models have more influence.
    """
    if not profiles:
        return EntropyVectorCompat()

    vectors = []
    utilities = []
    for p in profiles:
        if isinstance(p, dict):
            if "entropy_profile" in p:
                v = EntropyVectorCompat.from_dict(p["entropy_profile"])
            elif "entropy_vector" in p:
                v = EntropyVectorCompat.from_dict(p["entropy_vector"])
            elif "h_info" in p:
                v = EntropyVectorCompat.from_dict(p)
            else:
                continue
            vectors.append(v)
            utilities.append(max(0.01, v.utility + 1))

    if not vectors:
        return EntropyVectorCompat()

    total_weight = sum(utilities)
    result = EntropyVectorCompat()

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


def semantic_state_to_liveness(state: str) -> float:
    """Convert semantic state to liveness multiplier."""
    return SEMANTIC_LIVENESS_FACTORS.get(state, 0.8)
