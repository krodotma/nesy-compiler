#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _clamp01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.5
    return max(0.0, min(1.0, x))


def approx_tokens_from_text(s: str | None) -> int:
    if not s:
        return 0
    # Conservative heuristic: ~4 chars/token for English-ish text.
    return max(1, int(math.ceil(len(s) / 4.0)))


def _determine_isolation(req: dict, topology: str) -> str:
    """
    Determine isolation level: 'thread' (shared state, fast) or 'process' (isolated, safe).

    Process isolation is used for:
    - Tasks with effects=network (untrusted external access)
    - Tasks with effects=unknown
    - Star/peer_debate topologies (always isolated by design)
    - Explicit isolation_hint=process
    - Deep audit tasks (kind=verify with high tool_density)
    """
    # Explicit hint overrides
    isolation_hint = req.get("isolation_hint")
    if isinstance(isolation_hint, str):
        hint = isolation_hint.strip().lower()
        if hint in {"process", "thread"}:
            return hint

    # Star and peer_debate always use process isolation (spawns multiple workers)
    if topology in {"star", "peer_debate"}:
        return "process"

    # Check effects field
    effects = req.get("effects")
    if isinstance(effects, str):
        effects = effects.strip().lower()
        if effects in {"network", "unknown"}:
            return "process"
    elif isinstance(effects, list):
        for e in effects:
            if isinstance(e, str) and e.strip().lower() in {"network", "unknown"}:
                return "process"

    # Deep audit tasks should be isolated
    kind = req.get("kind")
    if isinstance(kind, str) and kind.strip().lower() == "verify":
        tool_density = _as_float(req.get("tool_density"))
        if tool_density is not None and tool_density >= 0.5:
            return "process"

    # Risky constraints trigger isolation
    constraints = req.get("constraints")
    if isinstance(constraints, dict):
        if constraints.get("untrusted_source") is True:
            return "process"
        if constraints.get("sandbox_required") is True:
            return "process"

    # Default: thread (fast, shared state)
    return "thread"


def choose_topology(req: dict) -> dict:
    """
    Decide multi-agent topology for a STRp request.

    Returns:
      {topology: 'single'|'star'|'peer_debate', fanout: int, isolation: 'thread'|'process', reason: str, inputs: {...}}
    """

    topology_hint = (req.get("topology_hint") or "auto")
    if isinstance(topology_hint, str):
        topology_hint = topology_hint.strip().lower()
    else:
        topology_hint = "auto"

    parallelizable = _as_bool(req.get("parallelizable"))
    tool_density = _as_float(req.get("tool_density"))
    tool_density = _clamp01(tool_density) if tool_density is not None else 0.5
    coord_budget_tokens = _as_int(req.get("coord_budget_tokens"))
    if coord_budget_tokens is not None and coord_budget_tokens <= 0:
        coord_budget_tokens = None

    kind = (req.get("kind") or "distill")
    if isinstance(kind, str):
        kind = kind.strip().lower()
    else:
        kind = "distill"

    effects = req.get("effects")

    # Helper to build result with isolation
    def _result(topo: str, fanout: int, reason: str) -> dict:
        isolation = _determine_isolation(req, topo)
        return {
            "topology": topo,
            "fanout": fanout,
            "isolation": isolation,
            "reason": reason,
            "inputs": {
                "parallelizable": parallelizable,
                "tool_density": tool_density,
                "coord_budget_tokens": coord_budget_tokens,
                "topology_hint": topology_hint,
                "kind": kind,
                "effects": effects,
            },
        }

    # Guards first: tool-heavy tasks and low budgets tend to amplify coordination error.
    if tool_density >= 0.7:
        return _result("single", 1, "tool_dense_fallback")

    if coord_budget_tokens is not None and coord_budget_tokens < 1500:
        return _result("single", 1, "coord_budget_too_low")

    if topology_hint in {"single", "star", "peer_debate"}:
        topo = topology_hint
        fanout = 1 if topo == "single" else 2
        return _result(topo, fanout, "topology_hint")

    # Default policy (v0): do not fan out unless explicitly parallelizable.
    if parallelizable is not True:
        return _result("single", 1, "not_parallelizable_default")

    # For analysis-type work, a small fanout helps; for implement/verify, keep tight by default.
    if kind in {"distill", "hypothesize", "apply"} and tool_density <= 0.4 and (coord_budget_tokens is None or coord_budget_tokens >= 4000):
        return _result("star", 2, "parallelizable_low_tool_density")

    if tool_density <= 0.5 and (coord_budget_tokens is None or coord_budget_tokens >= 6500):
        return _result("peer_debate", 2, "parallelizable_peer_debate_budgeted")

    return _result("single", 1, "parallelizable_but_conservative")

