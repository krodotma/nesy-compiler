#!/usr/bin/env python3
"""
SemanticBuchiRule - Buchi automaton with semantic motif tracking.

This module implements the semantic liveness rule for the Omega Guardian,
extending classical Buchi acceptance with omega-motif tracking to detect
semantically dead (zombie) agents.

State Machine:
    q_observe -> q_good: on first motif completion
    q_good -> q_stale: if no motif in window_s
    q_stale -> q_zombie: if consecutive_stale >= stale_threshold
    q_zombie -> q_recovering: on recovery intervention
    q_recovering -> q_good: on successful recovery
    Any -> q_good: on motif completion (except zombie without intervention)

Reference: nucleus/specs/omega_guardian_semantic_v1.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore


# -----------------------------------------------------------------------------
# Constants (from spec Section 6.2)
# -----------------------------------------------------------------------------
DEFAULT_WINDOW_S = 120
DEFAULT_STALE_THRESHOLD = 3
DEFAULT_MIN_COMPLETION_RATE = 0.001
DEFAULT_RECOVERY_TIMEOUT_S = 60
DEFAULT_COMPLETION_HISTORY_SIZE = 1000
DEFAULT_PARTIAL_MATCH_TTL_S = 600


# -----------------------------------------------------------------------------
# Semantic State Enum
# -----------------------------------------------------------------------------
class SemanticState(str, Enum):
    """Buchi automaton states for semantic liveness."""
    Q_OBSERVE = "q_observe"
    Q_GOOD = "q_good"
    Q_STALE = "q_stale"
    Q_ZOMBIE = "q_zombie"
    Q_RECOVERING = "q_recovering"


# -----------------------------------------------------------------------------
# MotifTracker Class
# -----------------------------------------------------------------------------
@dataclass
class PartialMatch:
    """Tracks a partial motif match in progress."""
    motif_id: str
    started_ts: float
    seen_vertices: set[str] = field(default_factory=set)
    last_ts: float = 0.0
    correlation_value: str = ""


class MotifTracker:
    """
    Track partial and complete motif matches in the event stream.

    A motif is a sequence of events (vertices) connected by temporal edges.
    The tracker maintains partial matches keyed by (motif_id, correlation_value)
    and detects when all vertices have been seen within the max_span.
    """

    def __init__(self, motif_registry: dict, *, completion_history_size: int = DEFAULT_COMPLETION_HISTORY_SIZE):
        """
        Initialize the motif tracker.

        Args:
            motif_registry: Dictionary containing 'motifs' list with motif definitions
            completion_history_size: Maximum number of completed motifs to track
        """
        self.registry = motif_registry
        self.motifs: list[dict] = motif_registry.get("motifs", [])
        self.settings = motif_registry.get("settings", {})

        # Partial matches: key = f"{motif_id}:{correlation_value}"
        self.partial_matches: dict[str, PartialMatch] = {}

        # Completed motifs: (timestamp, motif_id, weight, actor)
        self.completed_motifs: deque[tuple[float, str, float, str]] = deque(maxlen=completion_history_size)

        # Settings with defaults
        self.partial_match_ttl_s = float(self.settings.get("partial_match_ttl_s", DEFAULT_PARTIAL_MATCH_TTL_S))
        self.default_max_span_s = float(self.settings.get("default_max_span_s", 300))
        self.default_weight = float(self.settings.get("default_weight", 1.0))

        # Build vertex-to-motif index for fast lookup
        self._vertex_index: dict[str, list[dict]] = {}
        for motif in self.motifs:
            for vertex in motif.get("vertices", []):
                if vertex not in self._vertex_index:
                    self._vertex_index[vertex] = []
                self._vertex_index[vertex].append(motif)

    def _extract_correlation(self, data: dict, key: str | None) -> str | None:
        """Extract correlation key value from event data."""
        if not key:
            return None
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        return None

    def _is_valid_transition(self, motif: dict, seen_vertices: set[str], new_vertex: str) -> bool:
        """
        Check if transitioning to new_vertex is valid given seen vertices.

        For temporal edges, the source vertex must already be seen.
        """
        edges = motif.get("edges", [])
        vertices = motif.get("vertices", [])

        # If no edges defined (single-vertex motif), allow any vertex
        if not edges:
            return True

        # Find edges that end at new_vertex
        for edge in edges:
            if edge.get("to") == new_vertex:
                from_vertex = edge.get("from")
                edge_type = edge.get("type", "temporal")

                # For temporal edges, the source must be seen
                if edge_type == "temporal":
                    if from_vertex in seen_vertices:
                        return True
                # Optional edges don't require the source to be seen
                elif edge_type == "optional":
                    # Check if any predecessor is seen OR this is the first vertex
                    idx = vertices.index(new_vertex) if new_vertex in vertices else -1
                    if idx <= 0 or from_vertex in seen_vertices:
                        return True

        return False

    def _check_filters(self, motif: dict, data: dict) -> bool:
        """Check if event data passes motif filters."""
        filters = motif.get("filters", {})
        if not filters:
            return True

        for key, allowed_values in filters.items():
            event_value = data.get(key)
            if event_value is not None and event_value not in allowed_values:
                return False

        return True

    def process_event(self, event: dict, now: float) -> list[dict]:
        """
        Process an event and return list of completed motif details.

        Args:
            event: Bus event dictionary with topic, actor, data, etc.
            now: Current timestamp

        Returns:
            List of completed motif info dicts with keys:
            - motif_id: ID of the completed motif
            - correlation_value: The correlation key value
            - duration_s: Time from start to completion
            - weight: Motif weight
            - actor: Actor that triggered completion
        """
        completed: list[dict] = []
        topic = event.get("topic", "")
        actor = event.get("actor", "")
        data = event.get("data", {})
        if not isinstance(data, dict):
            data = {}

        # Only process if this topic matches any motif vertex
        if topic not in self._vertex_index:
            return completed

        # Process each motif that has this topic as a vertex
        for motif in self._vertex_index[topic]:
            motif_id = motif.get("id", "")
            if not motif_id:
                continue

            vertices = motif.get("vertices", [])
            if not vertices:
                continue

            # Check filters
            if not self._check_filters(motif, data):
                continue

            # Extract correlation value
            corr_key = motif.get("correlation_key")
            corr_val = self._extract_correlation(data, corr_key)

            # For actor-based correlation, use the actor field
            if corr_key == "actor":
                corr_val = actor

            # Generate unique partial match key
            partial_key = f"{motif_id}:{corr_val or 'global'}"
            max_span = float(motif.get("max_span_s", self.default_max_span_s))
            weight = float(motif.get("weight", self.default_weight))

            # Check if this is the start of a new motif
            if topic == vertices[0]:
                # Start new partial match (overwrites existing if any)
                self.partial_matches[partial_key] = PartialMatch(
                    motif_id=motif_id,
                    started_ts=now,
                    seen_vertices={topic},
                    last_ts=now,
                    correlation_value=corr_val or "global",
                )

                # Single-vertex motif completes immediately
                if len(vertices) == 1:
                    completed.append({
                        "motif_id": motif_id,
                        "correlation_value": corr_val or "global",
                        "duration_s": 0.0,
                        "weight": weight,
                        "actor": actor,
                    })
                    self.completed_motifs.append((now, motif_id, weight, actor))
                    del self.partial_matches[partial_key]

            elif partial_key in self.partial_matches:
                partial = self.partial_matches[partial_key]

                # Check max_span expiry
                if (now - partial.started_ts) > max_span:
                    del self.partial_matches[partial_key]
                    continue

                # Check edge validity
                if not self._is_valid_transition(motif, partial.seen_vertices, topic):
                    continue

                # Add vertex to seen set
                partial.seen_vertices.add(topic)
                partial.last_ts = now

                # Check completion (all vertices seen)
                if partial.seen_vertices >= set(vertices):
                    duration = now - partial.started_ts
                    completed.append({
                        "motif_id": motif_id,
                        "correlation_value": partial.correlation_value,
                        "duration_s": round(duration, 3),
                        "weight": weight,
                        "actor": actor,
                    })
                    self.completed_motifs.append((now, motif_id, weight, actor))
                    del self.partial_matches[partial_key]

        # Prune expired partial matches
        self._prune_expired(now)

        return completed

    def _prune_expired(self, now: float) -> None:
        """Remove partial matches that have exceeded their TTL."""
        expired_keys = []
        for key, partial in self.partial_matches.items():
            # Find the motif to get its max_span
            motif = next((m for m in self.motifs if m.get("id") == partial.motif_id), None)
            if motif:
                max_span = float(motif.get("max_span_s", self.default_max_span_s))
            else:
                max_span = self.partial_match_ttl_s

            if (now - partial.started_ts) > max_span:
                expired_keys.append(key)

        for key in expired_keys:
            del self.partial_matches[key]

    def last_completion_ts(self, actor: str | None = None) -> float | None:
        """
        Return timestamp of last completed motif.

        Args:
            actor: If specified, return last completion for this actor only

        Returns:
            Timestamp of last completion, or None if no completions
        """
        if not self.completed_motifs:
            return None

        if actor is None:
            return self.completed_motifs[-1][0]

        # Find last completion for specific actor
        for ts, motif_id, weight, ev_actor in reversed(self.completed_motifs):
            if ev_actor == actor:
                return ts

        return None

    def completion_rate(self, window_s: float, now: float, actor: str | None = None) -> float:
        """
        Return motif completions per second in window.

        Args:
            window_s: Time window in seconds
            now: Current timestamp
            actor: If specified, count only completions by this actor

        Returns:
            Completion rate (completions per second)
        """
        if window_s <= 0:
            return 0.0

        cutoff = now - window_s
        count = 0
        for ts, motif_id, weight, ev_actor in self.completed_motifs:
            if ts >= cutoff:
                if actor is None or ev_actor == actor:
                    count += 1

        return count / window_s

    def weighted_throughput(self, window_s: float, now: float, actor: str | None = None) -> float:
        """
        Return weighted motif throughput in window.

        Args:
            window_s: Time window in seconds
            now: Current timestamp
            actor: If specified, count only completions by this actor

        Returns:
            Sum of weights for completed motifs in window
        """
        cutoff = now - window_s
        total = 0.0
        for ts, motif_id, weight, ev_actor in self.completed_motifs:
            if ts >= cutoff:
                if actor is None or ev_actor == actor:
                    total += weight

        return total

    def completions_in_window(self, window_s: float, now: float, actor: str | None = None) -> int:
        """Count motif completions in window."""
        cutoff = now - window_s
        count = 0
        for ts, motif_id, weight, ev_actor in self.completed_motifs:
            if ts >= cutoff:
                if actor is None or ev_actor == actor:
                    count += 1
        return count


# -----------------------------------------------------------------------------
# Per-Actor State Tracking
# -----------------------------------------------------------------------------
@dataclass
class ActorSemanticState:
    """Semantic state for a single actor."""
    state: SemanticState = SemanticState.Q_OBSERVE
    stale_count: int = 0
    last_motif_ts: float = 0.0
    last_evaluation_ts: float = 0.0
    intervention_ts: float = 0.0
    last_transition: str = ""


# -----------------------------------------------------------------------------
# SemanticBuchiRule Class
# -----------------------------------------------------------------------------
class SemanticBuchiRule:
    """
    Buchi rule with semantic motif tracking.

    This class extends the standard Buchi liveness check with omega-motif
    tracking to detect semantically dead agents (zombies). An agent is
    considered a zombie if it continues to send heartbeats but fails to
    complete any semantic motifs within the configured window.

    The state machine tracks each actor independently:
    - q_observe: Initial state, waiting for first motif
    - q_good: Semantic progress confirmed (accepting state)
    - q_stale: Heartbeat present but no recent motif completion
    - q_zombie: Suspected semantic death (alert triggered)
    - q_recovering: Recovery intervention in progress
    """

    def __init__(
        self,
        rule_config: dict,
        *,
        bus_dir: str | None = None,
    ):
        """
        Initialize the semantic Buchi rule.

        Args:
            rule_config: Rule configuration dictionary with keys:
                - motif_registry: Path to motif registry JSON file
                - window_s: Time window for motif completion (default: 120)
                - stale_threshold: Consecutive stale windows before zombie (default: 3)
                - per_actor: Track each actor independently (default: True)
                - actors: List of actors to track (optional)
                - min_completion_rate: Minimum motifs/second (default: 0.001)
                - recovery_timeout_s: Max time in recovering state (default: 60)
            bus_dir: Bus directory for event emission (optional)
        """
        self.rule_id = rule_config.get("id", "semantic-recurrence-buchi")
        self.category = rule_config.get("category", "liveness")

        # Load motif registry
        registry_path = rule_config.get("motif_registry", "nucleus/specs/omega_motifs.json")
        self.motif_registry = self._load_motif_registry(registry_path)
        self.motif_tracker = MotifTracker(self.motif_registry)

        # Configuration parameters (from spec Section 6.2)
        self.window_s = float(rule_config.get("window_s", DEFAULT_WINDOW_S))
        self.stale_threshold = int(rule_config.get("stale_threshold", DEFAULT_STALE_THRESHOLD))
        self.per_actor = bool(rule_config.get("per_actor", True))
        self.tracked_actors: set[str] = set(rule_config.get("actors", []))
        self.min_completion_rate = float(rule_config.get("min_completion_rate", DEFAULT_MIN_COMPLETION_RATE))
        self.recovery_timeout_s = float(rule_config.get("recovery_timeout_s", DEFAULT_RECOVERY_TIMEOUT_S))

        # Recovery action configuration
        self.recovery_action = rule_config.get("recovery_action", {})

        # State tracking
        self.global_state = SemanticState.Q_OBSERVE
        self.per_actor_state: dict[str, ActorSemanticState] = {}
        self.cycle_count = 0

        # Bus configuration
        self.bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")
        self.actor_name = "omega_guardian_semantic"

    def _load_motif_registry(self, registry_path: str) -> dict:
        """Load motif registry from file."""
        path = Path(registry_path)
        if not path.is_absolute():
            path = REPO_ROOT / registry_path

        if not path.exists():
            # Return default empty registry
            return {"motifs": [], "settings": {}}

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"motifs": [], "settings": {}}

    def _emit_bus(self, topic: str, kind: str, level: str, data: dict) -> None:
        """Emit a bus event."""
        if agent_bus is None:
            return
        try:
            paths = agent_bus.resolve_bus_paths(self.bus_dir)
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind=kind,
                level=level,
                actor=self.actor_name,
                data=data,
                trace_id=None,
                run_id=None,
                durable=False,
            )
        except Exception:
            pass

    def _get_actor_state(self, actor: str) -> ActorSemanticState:
        """Get or create actor state."""
        if actor not in self.per_actor_state:
            self.per_actor_state[actor] = ActorSemanticState()
        return self.per_actor_state[actor]

    def _should_track_actor(self, actor: str) -> bool:
        """Check if we should track this actor."""
        if not actor:
            return False
        # If tracked_actors is specified, only track those
        if self.tracked_actors:
            return actor in self.tracked_actors
        # Otherwise track all actors
        return True

    def _transition_state(
        self,
        actor_state: ActorSemanticState,
        new_state: SemanticState,
        actor: str,
        now: float,
        reason: str = "",
    ) -> None:
        """Transition actor to new state with logging."""
        old_state = actor_state.state
        if old_state == new_state:
            return

        actor_state.state = new_state
        actor_state.last_transition = f"{old_state.value}->{new_state.value}: {reason}"

        # Reset stale count on transition to good
        if new_state == SemanticState.Q_GOOD:
            actor_state.stale_count = 0

        # Track intervention time for recovery
        if new_state == SemanticState.Q_RECOVERING:
            actor_state.intervention_ts = now

    def process_event(self, event: dict, now: float) -> list[dict]:
        """
        Process an event and return any triggered alerts/events.

        Args:
            event: Bus event dictionary
            now: Current timestamp

        Returns:
            List of event dictionaries to emit
        """
        alerts: list[dict] = []
        actor = event.get("actor", "system")
        topic = event.get("topic", "")

        # Skip if actor not tracked
        if self.per_actor and not self._should_track_actor(actor):
            return alerts

        # Track motif completions
        completed = self.motif_tracker.process_event(event, now)

        # Process completions
        for completion in completed:
            motif_id = completion["motif_id"]
            completion_actor = completion.get("actor", actor)

            # Update last motif timestamp
            if self.per_actor:
                actor_state = self._get_actor_state(completion_actor)
                actor_state.last_motif_ts = now

                # Handle state transitions on motif completion
                if actor_state.state == SemanticState.Q_OBSERVE:
                    self._transition_state(
                        actor_state, SemanticState.Q_GOOD, completion_actor, now,
                        f"first motif: {motif_id}"
                    )
                elif actor_state.state == SemanticState.Q_STALE:
                    self._transition_state(
                        actor_state, SemanticState.Q_GOOD, completion_actor, now,
                        f"motif after stale: {motif_id}"
                    )
                elif actor_state.state == SemanticState.Q_RECOVERING:
                    self._transition_state(
                        actor_state, SemanticState.Q_GOOD, completion_actor, now,
                        f"recovery success: {motif_id}"
                    )
                    # Emit recovery success event
                    alerts.append({
                        "topic": "omega.guardian.semantic.recovered",
                        "kind": "artifact",
                        "level": "info",
                        "data": {
                            "target_actor": completion_actor,
                            "motif_id": motif_id,
                            "recovery_duration_s": round(now - actor_state.intervention_ts, 2),
                        }
                    })
                # Note: zombie state requires explicit intervention before motif can help

            # Emit motif completion event
            alerts.append({
                "topic": "omega.guardian.semantic.motif_complete",
                "kind": "artifact",
                "level": "info",
                "data": {
                    "motif_id": motif_id,
                    "correlation_key": completion.get("correlation_value", ""),
                    "duration_s": completion.get("duration_s", 0.0),
                    "weight": completion.get("weight", 1.0),
                    "actor": completion_actor,
                }
            })

        # Check for intervention events (recovery trigger)
        if topic == "omega.guardian.semantic.intervention":
            target = event.get("data", {}).get("target_actor", "")
            if target and self.per_actor:
                actor_state = self._get_actor_state(target)
                if actor_state.state == SemanticState.Q_ZOMBIE:
                    self._transition_state(
                        actor_state, SemanticState.Q_RECOVERING, target, now,
                        "intervention received"
                    )

        return alerts

    def evaluate_cycle(self, now: float) -> tuple[str, dict]:
        """
        Per-cycle evaluation of semantic liveness.

        This should be called periodically (e.g., every window_s/2 seconds)
        to check for stale/zombie transitions.

        Args:
            now: Current timestamp

        Returns:
            Tuple of (status, details) where status is "ok", "warn", or "error"
        """
        self.cycle_count += 1
        violations: list[str] = []
        stale_actors: list[str] = []
        per_actor_summary: dict[str, dict] = {}

        if self.per_actor:
            # Evaluate each tracked actor
            for actor, actor_state in list(self.per_actor_state.items()):
                actor_state.last_evaluation_ts = now
                last_ts = actor_state.last_motif_ts
                age = (now - last_ts) if last_ts > 0 else float("inf")

                per_actor_summary[actor] = {
                    "state": actor_state.state.value,
                    "last_motif_age_s": round(age, 2) if age != float("inf") else None,
                    "stale_count": actor_state.stale_count,
                }

                # State transitions based on motif age
                if actor_state.state == SemanticState.Q_GOOD:
                    if age > self.window_s:
                        # Transition to stale
                        actor_state.stale_count += 1
                        self._transition_state(
                            actor_state, SemanticState.Q_STALE, actor, now,
                            f"no motif for {round(age, 1)}s"
                        )
                        stale_actors.append(actor)

                        # Emit stale event
                        self._emit_bus(
                            "omega.guardian.semantic.stale",
                            "metric",
                            "warn",
                            {
                                "target_actor": actor,
                                "last_motif_age_s": round(age, 2),
                                "stale_count": actor_state.stale_count,
                                "threshold": self.stale_threshold,
                            }
                        )

                elif actor_state.state == SemanticState.Q_STALE:
                    if age <= self.window_s:
                        # Recent motif - should have been caught in process_event
                        self._transition_state(
                            actor_state, SemanticState.Q_GOOD, actor, now,
                            "motif age within window"
                        )
                    else:
                        # Still stale, increment count
                        actor_state.stale_count += 1

                        if actor_state.stale_count >= self.stale_threshold:
                            # Transition to zombie
                            self._transition_state(
                                actor_state, SemanticState.Q_ZOMBIE, actor, now,
                                f"stale_count={actor_state.stale_count} >= threshold"
                            )
                            violations.append(actor)

                            # Emit zombie alert
                            self._emit_bus(
                                "omega.guardian.semantic.zombie",
                                "alert",
                                "error",
                                {
                                    "target_actor": actor,
                                    "last_motif_age_s": round(age, 2) if age != float("inf") else None,
                                    "consecutive_stale_windows": actor_state.stale_count,
                                    "heartbeat_present": True,  # Assumed since we're evaluating
                                    "recommended_action": self.recovery_action.get("type", "restart"),
                                    "evidence": {
                                        "last_motif_ts": actor_state.last_motif_ts if actor_state.last_motif_ts > 0 else None,
                                        "stale_count": actor_state.stale_count,
                                        "window_s": self.window_s,
                                    }
                                }
                            )
                        else:
                            stale_actors.append(actor)

                elif actor_state.state == SemanticState.Q_RECOVERING:
                    # Check recovery timeout
                    recovery_age = now - actor_state.intervention_ts
                    if recovery_age > self.recovery_timeout_s:
                        # Recovery failed, back to zombie
                        self._transition_state(
                            actor_state, SemanticState.Q_ZOMBIE, actor, now,
                            f"recovery timeout after {round(recovery_age, 1)}s"
                        )
                        violations.append(actor)

                        self._emit_bus(
                            "omega.guardian.semantic.recovery_failed",
                            "alert",
                            "error",
                            {
                                "target_actor": actor,
                                "recovery_duration_s": round(recovery_age, 2),
                                "timeout_s": self.recovery_timeout_s,
                            }
                        )

                elif actor_state.state == SemanticState.Q_ZOMBIE:
                    # Stay in zombie until intervention
                    violations.append(actor)

                # Update summary with final state
                per_actor_summary[actor]["state"] = actor_state.state.value
                per_actor_summary[actor]["stale_count"] = actor_state.stale_count

        # Compute overall status
        if violations:
            status = "error"
        elif stale_actors:
            status = "warn"
        else:
            status = "ok"

        # Build details
        details = {
            "cycle": self.cycle_count,
            "violations": violations,
            "stale_actors": stale_actors,
            "per_actor_state": per_actor_summary,
            "completion_rate": self.motif_tracker.completion_rate(self.window_s, now),
            "weighted_throughput": round(self.motif_tracker.weighted_throughput(self.window_s, now), 3),
            "motifs_completed_window": self.motif_tracker.completions_in_window(self.window_s, now),
            "partial_matches": len(self.motif_tracker.partial_matches),
            "actors_tracked": list(self.per_actor_state.keys()),
        }

        # Emit cycle event
        self._emit_bus(
            "omega.guardian.semantic.cycle",
            "metric",
            "info",
            details,
        )

        return status, details

    def trigger_intervention(self, target_actor: str, now: float) -> bool:
        """
        Trigger recovery intervention for a zombie actor.

        Args:
            target_actor: Actor to intervene on
            now: Current timestamp

        Returns:
            True if intervention was triggered, False if actor not in zombie state
        """
        if target_actor not in self.per_actor_state:
            return False

        actor_state = self.per_actor_state[target_actor]
        if actor_state.state != SemanticState.Q_ZOMBIE:
            return False

        self._transition_state(
            actor_state, SemanticState.Q_RECOVERING, target_actor, now,
            "manual intervention"
        )

        # Emit recovery request
        self._emit_bus(
            "omega.guardian.semantic.recovery",
            "request",
            "warn",
            {
                "target_actor": target_actor,
                "recovery_action": self.recovery_action,
            }
        )

        return True

    def get_actor_semantic_age(self, actor: str, now: float) -> float:
        """Get time since last motif completion for actor."""
        if actor not in self.per_actor_state:
            return float("inf")

        last_ts = self.per_actor_state[actor].last_motif_ts
        if last_ts <= 0:
            return float("inf")

        return now - last_ts

    def reset_actor(self, actor: str) -> None:
        """Reset actor state to initial."""
        if actor in self.per_actor_state:
            del self.per_actor_state[actor]

    def get_summary(self, now: float) -> dict:
        """Get current rule summary."""
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "window_s": self.window_s,
            "stale_threshold": self.stale_threshold,
            "cycle_count": self.cycle_count,
            "actors_tracked": len(self.per_actor_state),
            "actors_good": sum(1 for s in self.per_actor_state.values() if s.state == SemanticState.Q_GOOD),
            "actors_stale": sum(1 for s in self.per_actor_state.values() if s.state == SemanticState.Q_STALE),
            "actors_zombie": sum(1 for s in self.per_actor_state.values() if s.state == SemanticState.Q_ZOMBIE),
            "actors_recovering": sum(1 for s in self.per_actor_state.values() if s.state == SemanticState.Q_RECOVERING),
            "total_motifs_window": self.motif_tracker.completions_in_window(self.window_s, now),
            "completion_rate": self.motif_tracker.completion_rate(self.window_s, now),
        }


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def load_motif_registry(path: str | Path) -> dict:
    """Load a motif registry from file."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        return {"motifs": [], "settings": {}}

    return json.loads(path.read_text(encoding="utf-8"))


def create_default_rule(bus_dir: str | None = None) -> SemanticBuchiRule:
    """Create a SemanticBuchiRule with default configuration."""
    config = {
        "id": "semantic-recurrence-buchi",
        "type": "semantic_buchi",
        "category": "liveness",
        "motif_registry": "nucleus/specs/omega_motifs.json",
        "window_s": DEFAULT_WINDOW_S,
        "stale_threshold": DEFAULT_STALE_THRESHOLD,
        "per_actor": True,
        "min_completion_rate": DEFAULT_MIN_COMPLETION_RATE,
        "recovery_timeout_s": DEFAULT_RECOVERY_TIMEOUT_S,
    }
    return SemanticBuchiRule(config, bus_dir=bus_dir)


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="SemanticBuchiRule tester")
    parser.add_argument("--config", help="Path to rule config JSON")
    parser.add_argument("--registry", default="nucleus/specs/omega_motifs.json", help="Motif registry path")
    parser.add_argument("--bus-dir", help="Bus directory")
    parser.add_argument("--dump-registry", action="store_true", help="Dump loaded motif registry")
    parser.add_argument("--test-event", help="Process a test event (JSON string)")

    args = parser.parse_args(argv)

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "id": "test-semantic-buchi",
            "motif_registry": args.registry,
            "window_s": DEFAULT_WINDOW_S,
            "stale_threshold": DEFAULT_STALE_THRESHOLD,
        }

    rule = SemanticBuchiRule(config, bus_dir=args.bus_dir)

    if args.dump_registry:
        print(json.dumps(rule.motif_registry, indent=2))
        return 0

    if args.test_event:
        event = json.loads(args.test_event)
        now = time.time()
        alerts = rule.process_event(event, now)
        print(f"Alerts: {json.dumps(alerts, indent=2)}")

        status, details = rule.evaluate_cycle(now)
        print(f"Status: {status}")
        print(f"Details: {json.dumps(details, indent=2)}")
        return 0

    # Default: run a single evaluation cycle
    now = time.time()
    status, details = rule.evaluate_cycle(now)
    print(f"Status: {status}")
    print(f"Summary: {json.dumps(rule.get_summary(now), indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
