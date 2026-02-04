#!/usr/bin/env python3
"""
motif_of_motifs.py - Second-Order Recurrence Tracker

DUALITY-BIND E13: The Ecclesiastes Completion Theorem

From PLURIBUS_BIBLE.md:
> Per Ecclesiastes: "What has been will be again"
> Truth = omega-motifs that satisfy Buchi acceptance (recur infinitely)
> Ephemera = patterns that die out

The catalyst: If truth = recurrence, then META-TRUTH = recurrence of
recurrence-patterns. This module tracks patterns in which motifs appear
TOGETHER. This second-order recurrence is where genuine emergence lives.

Ring: 1 (Operator)
Protocol: DKIN v29
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configuration
META_MOTIF_DIR = Path(os.environ.get("META_MOTIF_DIR", ".pluribus/meta_motifs"))
BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")

# Constants
META_MOTIF_WINDOW_S = 300  # 5 minute window for co-occurrence
BUCHI_THRESHOLD = 3  # Completions for Buchi acceptance
PHI = 1.618033988749895  # Golden ratio

try:
    from nucleus.tools import agent_bus
except ImportError:
    agent_bus = None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_ts() -> float:
    return time.time()


def emit_bus_event(topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    """Emit event to Pluribus bus."""
    if agent_bus is None:
        return
    try:
        paths = agent_bus.resolve_bus_paths(BUS_DIR)
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
        pass


@dataclass
class MotifCompletion:
    """Record of a single motif completion."""
    motif_id: str
    completed_ts: float
    correlation_value: str
    weight: float
    actor: str | None = None


@dataclass
class MetaMotif:
    """
    A meta-motif: a pattern of motifs that co-occur.

    This is second-order structure: not "what events recur" but
    "what patterns of patterns recur".
    """
    meta_id: str
    constituent_motifs: tuple[str, ...]  # Sorted tuple of motif IDs
    signature: str  # Hash of constituent motifs
    first_seen_ts: float
    last_seen_ts: float
    occurrence_count: int = 1
    buchi_accepted: bool = False
    emergence_score: float = 0.0
    # Track which actors are involved in this meta-pattern
    participating_actors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "constituent_motifs": list(self.constituent_motifs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetaMotif":
        data = dict(data)
        data["constituent_motifs"] = tuple(data.get("constituent_motifs", []))
        return cls(**data)


@dataclass
class EmergenceEvent:
    """
    Record of an emergence event: when a meta-motif achieves significance.

    This is the moment when second-order recurrence becomes visible.
    """
    event_id: str
    meta_motif_id: str
    emergence_ts: float
    emergence_type: str  # "buchi_acceptance", "high_emergence_score", "novel_combination"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MotifOfMotifsTracker:
    """
    Track second-order recurrence: patterns of patterns.

    This implements the Ecclesiastes Completion Theorem:
    If truth = recurrence, then meta-truth = recurrence of recurrence-patterns.
    """

    def __init__(
        self,
        state_dir: Path | None = None,
        actor: str = "motif_of_motifs",
    ):
        self.state_dir = state_dir or META_MOTIF_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor = actor

        # Recent completions window
        self.completion_window: deque[MotifCompletion] = deque(maxlen=1000)

        # Meta-motif registry
        self.meta_motifs: dict[str, MetaMotif] = {}

        # Emergence events log
        self.emergence_events: list[EmergenceEvent] = []

        # Co-occurrence matrix (sparse)
        self.cooccurrence: defaultdict[tuple[str, str], int] = defaultdict(int)

        # Load persisted state
        self._load_state()

    def _state_path(self) -> Path:
        return self.state_dir / "meta_motifs.json"

    def _emergence_path(self) -> Path:
        return self.state_dir / "emergence_events.ndjson"

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for m in data.get("meta_motifs", []):
                    mm = MetaMotif.from_dict(m)
                    self.meta_motifs[mm.meta_id] = mm
                # Rebuild co-occurrence matrix
                for co in data.get("cooccurrence", []):
                    key = tuple(co["pair"])
                    self.cooccurrence[key] = co["count"]
            except Exception:
                pass

        # Load emergence events
        epath = self._emergence_path()
        if epath.exists():
            for line in epath.read_text().strip().split("\n"):
                if line:
                    try:
                        self.emergence_events.append(
                            EmergenceEvent(**json.loads(line))
                        )
                    except Exception:
                        continue

    def _save_state(self) -> None:
        """Persist state to disk."""
        path = self._state_path()
        data = {
            "meta_motifs": [m.to_dict() for m in self.meta_motifs.values()],
            "cooccurrence": [
                {"pair": list(k), "count": v}
                for k, v in self.cooccurrence.items()
            ],
            "last_updated": now_iso(),
        }
        with open(path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)

    def _save_emergence(self, event: EmergenceEvent) -> None:
        """Append emergence event to log."""
        path = self._emergence_path()
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(event.to_dict()) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def _compute_signature(self, motif_ids: tuple[str, ...]) -> str:
        """Compute stable signature for a set of motifs."""
        normalized = "|".join(sorted(motif_ids))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _compute_emergence_score(self, meta: MetaMotif, now: float) -> float:
        """
        Compute emergence score for a meta-motif.

        Emergence = f(recurrence, diversity, novelty, recency)

        Higher score = more "emergent" (interesting second-order structure).
        """
        # Recurrence: log-scaled occurrence count
        recurrence = math.log1p(meta.occurrence_count) / math.log1p(BUCHI_THRESHOLD * 5)
        recurrence = min(1.0, recurrence)

        # Diversity: how many different motifs are involved?
        diversity = min(1.0, len(meta.constituent_motifs) / 5)

        # Actor diversity: how many different actors participate?
        actor_diversity = min(1.0, len(set(meta.participating_actors)) / 3)

        # Recency: exponential decay
        age = now - meta.last_seen_ts
        recency = math.exp(-age / META_MOTIF_WINDOW_S)

        # Novelty: inverse of co-occurrence frequency (rare combos are novel)
        avg_cooccurrence = sum(
            self.cooccurrence.get((a, b), 0)
            for i, a in enumerate(meta.constituent_motifs)
            for b in meta.constituent_motifs[i+1:]
        )
        if len(meta.constituent_motifs) > 1:
            pairs = len(meta.constituent_motifs) * (len(meta.constituent_motifs) - 1) / 2
            avg_cooccurrence /= pairs
        novelty = 1.0 / (1.0 + avg_cooccurrence / 10)

        # Golden ratio weighted combination
        score = (
            recurrence * PHI * PHI +
            diversity * PHI +
            actor_diversity +
            recency / PHI +
            novelty / (PHI * PHI)
        ) / (PHI * PHI + PHI + 1 + 1/PHI + 1/(PHI*PHI))

        return round(score, 4)

    def record_completion(self, completion: MotifCompletion) -> None:
        """
        Record a motif completion and detect meta-patterns.

        This is called whenever a first-order motif completes.
        """
        now = completion.completed_ts
        self.completion_window.append(completion)

        # Prune old completions from window
        cutoff = now - META_MOTIF_WINDOW_S
        while self.completion_window and self.completion_window[0].completed_ts < cutoff:
            self.completion_window.popleft()

        # Get recent motif IDs in window
        recent_motifs = [c.motif_id for c in self.completion_window]
        unique_recent = sorted(set(recent_motifs))

        if len(unique_recent) < 2:
            return  # Need at least 2 motifs for meta-pattern

        # Update co-occurrence matrix
        for i, a in enumerate(unique_recent):
            for b in unique_recent[i+1:]:
                key = (a, b) if a < b else (b, a)
                self.cooccurrence[key] += 1

        # Detect meta-motif (set of co-occurring motifs)
        constituent_tuple = tuple(unique_recent)
        signature = self._compute_signature(constituent_tuple)
        meta_id = f"meta-{signature}"

        if meta_id in self.meta_motifs:
            meta = self.meta_motifs[meta_id]
            meta.occurrence_count += 1
            meta.last_seen_ts = now
            if completion.actor and completion.actor not in meta.participating_actors:
                meta.participating_actors.append(completion.actor)
        else:
            actors = [c.actor for c in self.completion_window if c.actor]
            meta = MetaMotif(
                meta_id=meta_id,
                constituent_motifs=constituent_tuple,
                signature=signature,
                first_seen_ts=now,
                last_seen_ts=now,
                occurrence_count=1,
                participating_actors=list(set(actors)),
            )
            self.meta_motifs[meta_id] = meta

        # Compute emergence score
        meta.emergence_score = self._compute_emergence_score(meta, now)

        # Check for Buchi acceptance
        if not meta.buchi_accepted and meta.occurrence_count >= BUCHI_THRESHOLD:
            meta.buchi_accepted = True
            self._record_emergence(meta, "buchi_acceptance", {
                "occurrence_count": meta.occurrence_count,
                "threshold": BUCHI_THRESHOLD,
            })

        # Check for high emergence score
        if meta.emergence_score > 0.7:
            # Only emit once per significant score increase
            existing = [
                e for e in self.emergence_events
                if e.meta_motif_id == meta_id and e.emergence_type == "high_emergence_score"
            ]
            if not existing:
                self._record_emergence(meta, "high_emergence_score", {
                    "score": meta.emergence_score,
                })

        # Emit tracking event
        emit_bus_event(
            topic="omega.meta.completion_recorded",
            kind="metric",
            level="debug",
            actor=self.actor,
            data={
                "motif_id": completion.motif_id,
                "meta_motif_id": meta_id,
                "meta_occurrence": meta.occurrence_count,
                "emergence_score": meta.emergence_score,
                "buchi_accepted": meta.buchi_accepted,
            },
        )

    def _record_emergence(
        self,
        meta: MetaMotif,
        emergence_type: str,
        evidence: dict[str, Any],
    ) -> None:
        """Record an emergence event."""
        event = EmergenceEvent(
            event_id=f"emrg-{uuid.uuid4().hex[:8]}",
            meta_motif_id=meta.meta_id,
            emergence_ts=now_ts(),
            emergence_type=emergence_type,
            evidence={
                **evidence,
                "constituent_motifs": list(meta.constituent_motifs),
                "emergence_score": meta.emergence_score,
            },
        )
        self.emergence_events.append(event)
        self._save_emergence(event)

        # Emit bus event for emergence
        emit_bus_event(
            topic="omega.meta.emergence_detected",
            kind="artifact",
            level="info",
            actor=self.actor,
            data={
                "event_id": event.event_id,
                "meta_motif_id": meta.meta_id,
                "emergence_type": emergence_type,
                "constituent_motifs": list(meta.constituent_motifs),
                "emergence_score": meta.emergence_score,
            },
        )

        self._save_state()

    def get_emergent_meta_motifs(self, min_score: float = 0.5) -> list[MetaMotif]:
        """Get meta-motifs above emergence score threshold."""
        return [
            m for m in self.meta_motifs.values()
            if m.emergence_score >= min_score
        ]

    def get_buchi_accepted(self) -> list[MetaMotif]:
        """Get all meta-motifs that have achieved Buchi acceptance."""
        return [m for m in self.meta_motifs.values() if m.buchi_accepted]

    def find_novel_combinations(self) -> list[tuple[str, str, float]]:
        """
        Find motif pairs that co-occur but are individually rare together.

        Returns: List of (motif_a, motif_b, novelty_score) tuples
        """
        novel = []
        for (a, b), count in self.cooccurrence.items():
            if count >= 2:  # Must co-occur at least twice
                # Novelty = low individual occurrence but high co-occurrence
                novelty = count / (self.cooccurrence.get((a, a), 1) * self.cooccurrence.get((b, b), 1) + 1)
                if novelty > 0.1:
                    novel.append((a, b, round(novelty, 4)))

        return sorted(novel, key=lambda x: -x[2])[:20]

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "total_meta_motifs": len(self.meta_motifs),
            "buchi_accepted": len(self.get_buchi_accepted()),
            "emergence_events": len(self.emergence_events),
            "cooccurrence_pairs": len(self.cooccurrence),
            "completion_window_size": len(self.completion_window),
            "window_s": META_MOTIF_WINDOW_S,
            "buchi_threshold": BUCHI_THRESHOLD,
            "avg_emergence_score": round(
                sum(m.emergence_score for m in self.meta_motifs.values()) / max(1, len(self.meta_motifs)),
                4
            ),
        }

    def render_emergence_report(self) -> str:
        """Render a human-readable emergence report."""
        lines = [
            "# Meta-Motif Emergence Report",
            f"Generated: {now_iso()}",
            "",
            "## Statistics",
        ]

        stats = self.get_stats()
        for k, v in stats.items():
            lines.append(f"- {k}: {v}")

        lines.extend([
            "",
            "## Buchi-Accepted Meta-Motifs (Truth)",
            "",
        ])

        for meta in self.get_buchi_accepted():
            lines.append(f"### {meta.meta_id}")
            lines.append(f"- Constituents: {', '.join(meta.constituent_motifs)}")
            lines.append(f"- Occurrences: {meta.occurrence_count}")
            lines.append(f"- Emergence Score: {meta.emergence_score}")
            lines.append(f"- Actors: {', '.join(meta.participating_actors[:5])}")
            lines.append("")

        lines.extend([
            "## High Emergence Score (Potential Truth)",
            "",
        ])

        for meta in sorted(
            self.get_emergent_meta_motifs(0.6),
            key=lambda m: -m.emergence_score
        )[:10]:
            if not meta.buchi_accepted:
                lines.append(f"- **{meta.meta_id}**: {meta.emergence_score:.3f}")
                lines.append(f"  - Constituents: {', '.join(meta.constituent_motifs)}")
                lines.append(f"  - Occurrences: {meta.occurrence_count} / {BUCHI_THRESHOLD}")
                lines.append("")

        lines.extend([
            "## Novel Combinations (Edge Discovery)",
            "",
        ])

        for a, b, score in self.find_novel_combinations()[:10]:
            lines.append(f"- {a} + {b}: novelty={score}")

        return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Motif of Motifs: Second-order recurrence tracker"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats command
    subparsers.add_parser("stats", help="Show tracker statistics")

    # buchi command
    subparsers.add_parser("buchi", help="List Buchi-accepted meta-motifs")

    # emergent command
    p_emrg = subparsers.add_parser("emergent", help="List emergent meta-motifs")
    p_emrg.add_argument("--min-score", type=float, default=0.5)

    # report command
    subparsers.add_parser("report", help="Generate emergence report")

    # novel command
    subparsers.add_parser("novel", help="Find novel combinations")

    args = parser.parse_args()
    tracker = MotifOfMotifsTracker()

    if args.command == "stats":
        stats = tracker.get_stats()
        print("Motif-of-Motifs Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    elif args.command == "buchi":
        accepted = tracker.get_buchi_accepted()
        if accepted:
            print(f"Buchi-Accepted Meta-Motifs ({len(accepted)}):")
            for m in accepted:
                print(f"  {m.meta_id}:")
                print(f"    constituents: {', '.join(m.constituent_motifs)}")
                print(f"    occurrences: {m.occurrence_count}")
                print(f"    emergence: {m.emergence_score:.3f}")
        else:
            print("No Buchi-accepted meta-motifs yet")
        return 0

    elif args.command == "emergent":
        emergent = tracker.get_emergent_meta_motifs(args.min_score)
        if emergent:
            print(f"Emergent Meta-Motifs (score >= {args.min_score}):")
            for m in sorted(emergent, key=lambda x: -x.emergence_score):
                print(f"  {m.meta_id}: {m.emergence_score:.3f}")
                print(f"    constituents: {', '.join(m.constituent_motifs)}")
        else:
            print(f"No meta-motifs with emergence score >= {args.min_score}")
        return 0

    elif args.command == "report":
        print(tracker.render_emergence_report())
        return 0

    elif args.command == "novel":
        novel = tracker.find_novel_combinations()
        if novel:
            print("Novel Motif Combinations:")
            for a, b, score in novel:
                print(f"  {a} + {b}: novelty={score}")
        else:
            print("No novel combinations detected yet")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
