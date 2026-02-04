#!/usr/bin/env python3
"""
Omega Motif Tracker - Semantic recurrence detection for the Entelexis system.

This module implements omega-motif tracking as specified in:
- nucleus/specs/omega_guardian_semantic_v1.md (Section 4.2)
- nucleus/specs/omega_motifs.json (motif registry)

An omega-motif is a recurring subgraph in the event bus stream that
characterizes successful task completion. The MotifTracker detects
partial matches and completed motifs in the event stream.

Key concepts:
- Vertices: Event topics that form the motif
- Edges: Temporal/causal relationships between vertices
- Correlation key: Field used to correlate events (e.g., req_id)
- Max span: Maximum time window for motif completion

Buchi acceptance: A motif has Buchi acceptance when it has been
completed at least 3 times (occurs infinitely often in theoretical terms).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
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

# Default paths
DEFAULT_MOTIF_REGISTRY = REPO_ROOT / "nucleus" / "specs" / "omega_motifs.json"
DEFAULT_BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")

# Constants
COMPLETED_MOTIFS_MAXLEN = 1000
BUCHI_ACCEPTANCE_THRESHOLD = 3


def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus(
    bus_dir: str,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict[str, Any],
) -> None:
    """Emit an event to the bus. Silent failure if bus unavailable."""
    if agent_bus is None:
        return
    try:
        paths = agent_bus.resolve_bus_paths(bus_dir)
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception:
        return


@dataclass
class PartialMatch:
    """Represents a partial motif match in progress."""

    motif_id: str
    correlation_key: str
    correlation_value: str
    started_ts: float
    last_ts: float
    seen_vertices: set[str] = field(default_factory=set)
    weight: float = 1.0

    def age(self, now: float) -> float:
        """Return age in seconds since match started."""
        return now - self.started_ts

    def idle_time(self, now: float) -> float:
        """Return seconds since last vertex was seen."""
        return now - self.last_ts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "motif_id": self.motif_id,
            "correlation_key": self.correlation_key,
            "correlation_value": self.correlation_value,
            "started_ts": self.started_ts,
            "last_ts": self.last_ts,
            "seen_vertices": list(self.seen_vertices),
            "weight": self.weight,
        }


@dataclass
class CompletedMotif:
    """Represents a completed motif instance."""

    motif_id: str
    correlation_value: str
    completed_ts: float
    duration_s: float
    weight: float
    actor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "motif_id": self.motif_id,
            "correlation_value": self.correlation_value,
            "completed_ts": self.completed_ts,
            "duration_s": self.duration_s,
            "weight": self.weight,
            "actor": self.actor,
        }


class MotifTracker:
    """
    Track partial motif matches in the event stream.

    This class implements the detection logic from Section 4.2 of
    omega_guardian_semantic_v1.md. It processes events and detects
    when semantic motifs (recurring patterns) complete.

    Attributes:
        registry: The motif registry configuration dict
        partial_matches: In-progress partial matches keyed by partial_key
        completed_motifs: Deque of (timestamp, motif_id) tuples
        completion_counts: Count of completions per motif_id for Buchi acceptance
        bus_dir: Directory for bus event emission
        actor: Actor name for bus events
    """

    def __init__(
        self,
        motif_registry: dict[str, Any],
        *,
        bus_dir: str | None = None,
        actor: str = "omega_motif_tracker",
    ) -> None:
        """
        Initialize the MotifTracker.

        Args:
            motif_registry: Motif definitions loaded from omega_motifs.json
            bus_dir: Bus directory for event emission (optional)
            actor: Actor name for emitted events
        """
        self.registry = motif_registry
        self.bus_dir = bus_dir or DEFAULT_BUS_DIR
        self.actor = actor

        # Settings from registry
        settings = self.registry.get("settings", {})
        self.default_max_span_s = float(settings.get("default_max_span_s", 300))
        self.default_weight = float(settings.get("default_weight", 1.0))
        self.partial_match_ttl_s = float(settings.get("partial_match_ttl_s", 600))

        # State tracking
        self.partial_matches: dict[str, PartialMatch] = {}
        self.completed_motifs: deque[tuple[float, str]] = deque(
            maxlen=COMPLETED_MOTIFS_MAXLEN
        )
        self.completion_counts: dict[str, int] = {}

        # Build lookup structures for efficiency
        self._motif_by_id: dict[str, dict[str, Any]] = {}
        self._motifs_by_vertex: dict[str, list[dict[str, Any]]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build internal lookup indexes for efficient processing."""
        motifs = self.registry.get("motifs", [])
        for motif in motifs:
            motif_id = motif.get("id", "")
            if not motif_id:
                continue

            self._motif_by_id[motif_id] = motif

            # Index by vertex (topic)
            vertices = motif.get("vertices", [])
            for vertex in vertices:
                if vertex not in self._motifs_by_vertex:
                    self._motifs_by_vertex[vertex] = []
                self._motifs_by_vertex[vertex].append(motif)

    def _extract_correlation(
        self,
        data: dict[str, Any],
        correlation_key: str,
    ) -> str | None:
        """
        Extract correlation value from event data.

        Args:
            data: Event data dict
            correlation_key: Key to look for (e.g., "req_id")

        Returns:
            Correlation value if found, None otherwise
        """
        if not data or not correlation_key:
            return None

        value = data.get(correlation_key)
        if isinstance(value, str) and value.strip():
            return value.strip()

        # Also check nested in common locations
        for nested_key in ("meta", "payload", "context"):
            nested = data.get(nested_key)
            if isinstance(nested, dict):
                value = nested.get(correlation_key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    def _is_valid_transition(
        self,
        motif: dict[str, Any],
        seen_vertices: set[str],
        new_vertex: str,
    ) -> bool:
        """
        Check if transitioning to new_vertex is valid given seen vertices.

        The transition is valid if:
        1. There exists an edge from some seen vertex to new_vertex, OR
        2. For linear motifs, the immediate predecessor is in seen_vertices

        Args:
            motif: Motif definition
            seen_vertices: Set of already-seen vertices
            new_vertex: The new vertex to transition to

        Returns:
            True if transition is valid
        """
        vertices = motif.get("vertices", [])
        edges = motif.get("edges", [])

        if new_vertex not in vertices:
            return False

        # Check for explicit edge from a seen vertex to new_vertex
        for edge in edges:
            from_v = edge.get("from", "")
            to_v = edge.get("to", "")

            if to_v == new_vertex and from_v in seen_vertices:
                return True

        # For linear motifs without explicit edges to this vertex,
        # require the immediate predecessor to be present
        new_idx = vertices.index(new_vertex)
        if new_idx == 0:
            # This is a start vertex, should be handled elsewhere
            return False

        # The immediate predecessor must be in seen_vertices
        predecessor = vertices[new_idx - 1]
        if predecessor in seen_vertices:
            return True

        # Check for optional edges that might allow skipping
        for edge in edges:
            if edge.get("type") == "optional":
                from_v = edge.get("from", "")
                to_v = edge.get("to", "")
                # If there's an optional edge from the predecessor to something else,
                # we might be able to skip it
                if from_v == predecessor and to_v != new_vertex:
                    # Check if we can reach new_vertex from some seen vertex
                    for seen in seen_vertices:
                        if seen in vertices:
                            seen_idx = vertices.index(seen)
                            if seen_idx < new_idx - 1:
                                # We skipped some vertices, only allowed with optional edges
                                continue
                            return True

        return False

    def _check_filters(
        self,
        motif: dict[str, Any],
        data: dict[str, Any],
    ) -> bool:
        """
        Check if event data passes motif filters.

        Args:
            motif: Motif definition with optional "filters" field
            data: Event data to check

        Returns:
            True if all filters pass (or no filters defined)
        """
        filters = motif.get("filters", {})
        if not filters:
            return True

        for key, allowed_values in filters.items():
            if not isinstance(allowed_values, list):
                allowed_values = [allowed_values]

            value = data.get(key)
            if value not in allowed_values:
                return False

        return True

    def _prune_expired(self, now: float) -> None:
        """Remove expired partial matches."""
        expired_keys: list[str] = []

        for key, partial in self.partial_matches.items():
            motif = self._motif_by_id.get(partial.motif_id)
            if not motif:
                expired_keys.append(key)
                continue

            max_span = float(motif.get("max_span_s", self.default_max_span_s))
            if partial.age(now) > max_span:
                expired_keys.append(key)
                continue

            # Also expire based on idle time (TTL)
            if partial.idle_time(now) > self.partial_match_ttl_s:
                expired_keys.append(key)

        for key in expired_keys:
            del self.partial_matches[key]

    def _emit_completion_event(
        self,
        motif: dict[str, Any],
        correlation_value: str,
        duration_s: float,
        actor: str | None,
    ) -> None:
        """Emit bus event for motif completion."""
        emit_bus(
            self.bus_dir,
            topic="omega.guardian.semantic.motif_complete",
            kind="artifact",
            level="info",
            actor=self.actor,
            data={
                "motif_id": motif.get("id", ""),
                "correlation_key": motif.get("correlation_key", ""),
                "correlation_value": correlation_value,
                "duration_s": round(duration_s, 3),
                "weight": motif.get("weight", self.default_weight),
                "category": motif.get("category", ""),
                "ts": time.time(),
                "iso": now_iso(),
                "event_actor": actor,
            },
        )

    def process_event(self, event: dict[str, Any], now: float) -> list[str]:
        """
        Process an event and return list of completed motif IDs.

        This is the main entry point for event processing. It:
        1. Checks if the event topic matches any motif vertices
        2. Starts new partial matches if event is a start vertex
        3. Updates existing partial matches if event extends them
        4. Detects and records completed motifs

        Args:
            event: Event dict with "topic", "data", "actor" fields
            now: Current timestamp (allows deterministic testing)

        Returns:
            List of motif IDs that were completed by this event
        """
        completed: list[str] = []
        topic = event.get("topic", "")
        data = event.get("data", {})
        actor = event.get("actor")

        if not isinstance(data, dict):
            data = {}

        # Find motifs that include this topic as a vertex
        matching_motifs = self._motifs_by_vertex.get(topic, [])
        if not matching_motifs:
            self._prune_expired(now)
            return completed

        for motif in matching_motifs:
            motif_id = motif.get("id", "")
            vertices = motif.get("vertices", [])
            correlation_key = motif.get("correlation_key", "")
            weight = float(motif.get("weight", self.default_weight))

            if not motif_id or not vertices:
                continue

            # Check filters
            if not self._check_filters(motif, data):
                continue

            # Extract correlation value
            corr_value = self._extract_correlation(data, correlation_key)
            if not corr_value:
                # For single-vertex motifs (like oiterate_tick), generate synthetic key
                if len(vertices) == 1:
                    corr_value = f"{topic}:{now}"
                else:
                    continue

            partial_key = f"{motif_id}:{corr_value}"

            # Check if this is a start vertex
            is_start_vertex = topic == vertices[0]

            if is_start_vertex:
                # Start new partial match
                self.partial_matches[partial_key] = PartialMatch(
                    motif_id=motif_id,
                    correlation_key=correlation_key,
                    correlation_value=corr_value,
                    started_ts=now,
                    last_ts=now,
                    seen_vertices={topic},
                    weight=weight,
                )

                # For single-vertex motifs, complete immediately
                if len(vertices) == 1:
                    completed.append(motif_id)
                    self.completed_motifs.append((now, motif_id))
                    self.completion_counts[motif_id] = (
                        self.completion_counts.get(motif_id, 0) + 1
                    )
                    self._emit_completion_event(motif, corr_value, 0.0, actor)
                    del self.partial_matches[partial_key]

            elif partial_key in self.partial_matches:
                partial = self.partial_matches[partial_key]

                # Check max_span
                max_span = float(motif.get("max_span_s", self.default_max_span_s))
                if partial.age(now) > max_span:
                    del self.partial_matches[partial_key]
                    continue

                # Check edge validity
                if self._is_valid_transition(motif, partial.seen_vertices, topic):
                    partial.seen_vertices.add(topic)
                    partial.last_ts = now

                    # Check completion
                    if partial.seen_vertices >= set(vertices):
                        duration_s = now - partial.started_ts
                        completed.append(motif_id)
                        self.completed_motifs.append((now, motif_id))
                        self.completion_counts[motif_id] = (
                            self.completion_counts.get(motif_id, 0) + 1
                        )
                        self._emit_completion_event(
                            motif, corr_value, duration_s, actor
                        )
                        del self.partial_matches[partial_key]

        # Prune expired partial matches
        self._prune_expired(now)

        return completed

    def last_completion_ts(self) -> float | None:
        """
        Return timestamp of last completed motif.

        Returns:
            Timestamp of last completion, or None if no completions yet
        """
        if self.completed_motifs:
            return self.completed_motifs[-1][0]
        return None

    def completion_rate(self, window_s: float, now: float) -> float:
        """
        Return motif completions per second in the specified window.

        Args:
            window_s: Time window in seconds to calculate rate
            now: Current timestamp

        Returns:
            Completions per second (float)
        """
        if window_s <= 0:
            return 0.0

        cutoff = now - window_s
        count = sum(1 for ts, _ in self.completed_motifs if ts >= cutoff)
        return count / window_s

    def weighted_completion_rate(self, window_s: float, now: float) -> float:
        """
        Return weighted motif completions per second.

        Weight is determined by the motif's weight field.

        Args:
            window_s: Time window in seconds
            now: Current timestamp

        Returns:
            Weighted completions per second
        """
        if window_s <= 0:
            return 0.0

        cutoff = now - window_s
        weighted_sum = 0.0

        for ts, motif_id in self.completed_motifs:
            if ts >= cutoff:
                motif = self._motif_by_id.get(motif_id, {})
                weight = float(motif.get("weight", self.default_weight))
                weighted_sum += weight

        return weighted_sum / window_s

    def get_partial_matches(self) -> list[dict[str, Any]]:
        """
        Return in-progress partial matches.

        Returns:
            List of partial match dicts with motif info
        """
        now = time.time()
        result: list[dict[str, Any]] = []

        for partial in self.partial_matches.values():
            motif = self._motif_by_id.get(partial.motif_id, {})
            vertices = motif.get("vertices", [])

            result.append({
                **partial.to_dict(),
                "age_s": round(partial.age(now), 2),
                "idle_s": round(partial.idle_time(now), 2),
                "total_vertices": len(vertices),
                "seen_count": len(partial.seen_vertices),
                "progress_pct": (
                    round(100.0 * len(partial.seen_vertices) / len(vertices), 1)
                    if vertices else 0.0
                ),
            })

        return result

    def check_buchi_acceptance(self, motif_id: str) -> bool:
        """
        Check if motif has Buchi acceptance (occurs >= 3 times).

        In omega-automata theory, Buchi acceptance means a state is
        visited infinitely often. We approximate this by requiring
        the motif to have completed at least 3 times.

        Args:
            motif_id: The motif ID to check

        Returns:
            True if motif has achieved Buchi acceptance
        """
        count = self.completion_counts.get(motif_id, 0)
        return count >= BUCHI_ACCEPTANCE_THRESHOLD

    def get_buchi_status(self) -> dict[str, Any]:
        """
        Get Buchi acceptance status for all motifs.

        Returns:
            Dict with per-motif acceptance status and counts
        """
        status: dict[str, Any] = {
            "threshold": BUCHI_ACCEPTANCE_THRESHOLD,
            "motifs": {},
        }

        for motif_id in self._motif_by_id.keys():
            count = self.completion_counts.get(motif_id, 0)
            status["motifs"][motif_id] = {
                "count": count,
                "accepted": count >= BUCHI_ACCEPTANCE_THRESHOLD,
            }

        return status

    def get_completion_history(
        self,
        window_s: float | None = None,
        now: float | None = None,
    ) -> list[tuple[float, str]]:
        """
        Get recent completion history.

        Args:
            window_s: Optional time window to filter (None = all history)
            now: Current timestamp (defaults to time.time())

        Returns:
            List of (timestamp, motif_id) tuples
        """
        if now is None:
            now = time.time()

        if window_s is None:
            return list(self.completed_motifs)

        cutoff = now - window_s
        return [(ts, mid) for ts, mid in self.completed_motifs if ts >= cutoff]

    def get_stats(self, window_s: float = 300.0) -> dict[str, Any]:
        """
        Get comprehensive tracker statistics.

        Args:
            window_s: Time window for rate calculations

        Returns:
            Dict with various statistics
        """
        now = time.time()
        return {
            "total_completions": len(self.completed_motifs),
            "completion_rate": round(self.completion_rate(window_s, now), 4),
            "weighted_rate": round(self.weighted_completion_rate(window_s, now), 4),
            "partial_matches": len(self.partial_matches),
            "motifs_registered": len(self._motif_by_id),
            "buchi_accepted_count": sum(
                1 for mid in self._motif_by_id
                if self.check_buchi_acceptance(mid)
            ),
            "last_completion_ts": self.last_completion_ts(),
            "last_completion_age_s": (
                round(now - self.last_completion_ts(), 2)
                if self.last_completion_ts() else None
            ),
            "window_s": window_s,
        }


def load_motif_registry(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load motif registry from JSON file.

    Args:
        path: Path to registry file (defaults to omega_motifs.json)

    Returns:
        Parsed registry dict
    """
    if path is None:
        path = DEFAULT_MOTIF_REGISTRY

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motif registry not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def create_tracker(
    registry_path: str | Path | None = None,
    bus_dir: str | None = None,
) -> MotifTracker:
    """
    Create a MotifTracker instance with default configuration.

    Args:
        registry_path: Path to motif registry (defaults to omega_motifs.json)
        bus_dir: Bus directory for event emission

    Returns:
        Configured MotifTracker instance
    """
    registry = load_motif_registry(registry_path)
    return MotifTracker(registry, bus_dir=bus_dir)


# -----------------------------------------------------------------------------
# CLI for testing
# -----------------------------------------------------------------------------

def main() -> int:
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Omega Motif Tracker")
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_MOTIF_REGISTRY),
        help="Path to motif registry JSON",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show tracker statistics",
    )
    parser.add_argument(
        "--buchi",
        action="store_true",
        help="Show Buchi acceptance status",
    )
    parser.add_argument(
        "--list-motifs",
        action="store_true",
        help="List all registered motifs",
    )

    args = parser.parse_args()

    try:
        tracker = create_tracker(args.registry)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.list_motifs:
        print("Registered motifs:")
        for motif_id, motif in tracker._motif_by_id.items():
            vertices = " -> ".join(motif.get("vertices", []))
            weight = motif.get("weight", 1.0)
            category = motif.get("category", "")
            print(f"  [{motif_id}] ({category}, w={weight})")
            print(f"    {vertices}")
        return 0

    if args.buchi:
        status = tracker.get_buchi_status()
        print(f"Buchi acceptance threshold: {status['threshold']}")
        print("Motif acceptance status:")
        for mid, info in status["motifs"].items():
            mark = "[OK]" if info["accepted"] else "[--]"
            print(f"  {mark} {mid}: {info['count']} completions")
        return 0

    if args.stats:
        stats = tracker.get_stats()
        print("Tracker statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return 0

    # Default: show summary
    print(f"MotifTracker loaded with {len(tracker._motif_by_id)} motifs")
    print(f"Registry: {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
