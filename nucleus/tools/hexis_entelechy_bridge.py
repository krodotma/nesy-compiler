#!/usr/bin/env python3
"""
hexis_entelechy_bridge.py - Bridge from Ephemeral Hexis to Permanent Entelechy

DUALITY-BIND E12: The Ecclesiastes Completion - What recurs becomes truth.

This module bridges the philosophical duality:
- Hexis (ἕξις): disposition, habit, the mutable present state
- Entelechy: the immutable telos, the fixed point

When hexis patterns STABILIZE (recur with consistency), they become
candidates for promotion to the Ω-Entelechy viability manifold.

Ring: 1 (Operator)
Protocol: DKIN v29
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configuration
HEXIS_BUFFER_DIR = Path(os.environ.get("HEXIS_BUFFER_DIR", "/tmp"))
MANIFOLD_DIR = Path(os.environ.get("PLURIBUS_MANIFOLD_DIR", ".pluribus/manifold"))
BRIDGE_STATE_DIR = Path(os.environ.get("HEXIS_BRIDGE_DIR", ".pluribus/bridge"))

# Thresholds for stabilization detection
STABILIZATION_THRESHOLD = 3  # Recurrences needed to consider stable
STABILIZATION_WINDOW_S = 600  # 10 minute window for pattern detection
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold for "same" pattern
PHI = 1.618033988749895  # Golden ratio for weighting

try:
    from nucleus.tools import agent_bus
except ImportError:
    agent_bus = None


def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_ts() -> float:
    """Return current timestamp."""
    return time.time()


def emit_bus_event(
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict[str, Any],
) -> None:
    """Emit event to Pluribus bus."""
    if agent_bus is None:
        return
    try:
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")
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
        pass


@dataclass
class HexisPattern:
    """A detected pattern in the hexis buffer stream."""
    pattern_id: str
    signature: str  # Hash of normalized pattern
    topic_sequence: list[str]
    first_seen_ts: float
    last_seen_ts: float
    occurrence_count: int = 1
    stability_score: float = 0.0
    payload_hash: str = ""
    actors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntelechyCandidate:
    """A candidate for promotion to the Ω-Entelechy manifold."""
    candidate_id: str
    source_pattern_id: str
    proposed_motif_id: str
    vertices: list[str]
    proposed_weight: float
    stability_evidence: float
    promotion_ts: float
    status: str = "pending"  # pending, approved, rejected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HexisEntelechyBridge:
    """
    Bridge between ephemeral hexis buffer and permanent entelechy manifold.

    This implements the third catalyst from the ontological study:
    > Create a hexis_entelechy_bridge.py that:
    > 1. Watches hexis buffer patterns
    > 2. Identifies when hexis patterns stabilize into candidates for entelechy
    > 3. Proposes new ω-motifs from stabilized hexis patterns
    """

    def __init__(
        self,
        hexis_dir: Path | None = None,
        bridge_dir: Path | None = None,
        actor: str = "hexis_entelechy_bridge",
    ):
        self.hexis_dir = hexis_dir or HEXIS_BUFFER_DIR
        self.bridge_dir = bridge_dir or BRIDGE_STATE_DIR
        self.actor = actor

        # Ensure directories exist
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.patterns: dict[str, HexisPattern] = {}
        self.candidates: list[EntelechyCandidate] = []
        self.topic_window: list[tuple[float, str, str]] = []  # (ts, topic, actor)

        # Load persisted state
        self._load_state()

    def _state_path(self) -> Path:
        """Path to persisted bridge state."""
        return self.bridge_dir / "bridge_state.json"

    def _candidates_path(self) -> Path:
        """Path to persisted candidates."""
        return self.bridge_dir / "entelechy_candidates.ndjson"

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        state_path = self._state_path()
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                for p in data.get("patterns", []):
                    pattern = HexisPattern(**p)
                    self.patterns[pattern.pattern_id] = pattern
            except Exception:
                pass

        # Load candidates
        cand_path = self._candidates_path()
        if cand_path.exists():
            for line in cand_path.read_text().strip().split("\n"):
                if line:
                    try:
                        self.candidates.append(EntelechyCandidate(**json.loads(line)))
                    except Exception:
                        continue

    def _save_state(self) -> None:
        """Persist state to disk."""
        state_path = self._state_path()
        data = {
            "patterns": [p.to_dict() for p in self.patterns.values()],
            "last_updated": now_iso(),
        }
        with open(state_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)

    def _save_candidate(self, candidate: EntelechyCandidate) -> None:
        """Append candidate to persistent log."""
        cand_path = self._candidates_path()
        with open(cand_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(candidate.to_dict()) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def _compute_signature(self, topics: list[str]) -> str:
        """Compute a stable signature for a topic sequence."""
        normalized = "|".join(sorted(set(topics)))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _compute_stability(self, pattern: HexisPattern, now: float) -> float:
        """
        Compute stability score for a pattern.

        Stability = f(recurrence, consistency, recency)
        Uses golden ratio weighting for aesthetic balance.
        """
        # Recurrence component (logarithmic)
        import math
        recurrence = math.log1p(pattern.occurrence_count) / math.log1p(STABILIZATION_THRESHOLD * 3)
        recurrence = min(1.0, recurrence)

        # Consistency: how evenly distributed are occurrences?
        time_span = pattern.last_seen_ts - pattern.first_seen_ts
        if time_span > 0 and pattern.occurrence_count > 1:
            expected_interval = STABILIZATION_WINDOW_S / STABILIZATION_THRESHOLD
            actual_interval = time_span / (pattern.occurrence_count - 1)
            consistency = 1.0 - abs(actual_interval - expected_interval) / expected_interval
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 0.5

        # Recency: exponential decay
        age = now - pattern.last_seen_ts
        recency = math.exp(-age / STABILIZATION_WINDOW_S)

        # Golden ratio weighted combination
        stability = (recurrence * PHI + consistency + recency / PHI) / (PHI + 1 + 1/PHI)
        return round(stability, 4)

    def ingest_hexis_message(self, msg: dict[str, Any]) -> HexisPattern | None:
        """
        Ingest a message from the hexis buffer and detect patterns.

        Returns the pattern if it achieved stabilization threshold.
        """
        now = now_ts()
        topic = msg.get("topic", "unknown")
        actor = msg.get("actor", "unknown")
        payload_hash = hashlib.md5(
            json.dumps(msg.get("payload", {}), sort_keys=True).encode()
        ).hexdigest()[:8]

        # Add to sliding window
        self.topic_window.append((now, topic, actor))

        # Prune old entries from window
        cutoff = now - STABILIZATION_WINDOW_S
        self.topic_window = [(ts, t, a) for ts, t, a in self.topic_window if ts >= cutoff]

        # Extract recent topic sequence (last N events)
        recent_topics = [t for _, t, _ in self.topic_window[-10:]]
        if len(recent_topics) < 2:
            return None

        # Compute signature
        signature = self._compute_signature(recent_topics)

        # Find or create pattern
        pattern_id = f"hpat-{signature}"
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.occurrence_count += 1
            pattern.last_seen_ts = now
            if actor not in pattern.actors:
                pattern.actors.append(actor)
        else:
            pattern = HexisPattern(
                pattern_id=pattern_id,
                signature=signature,
                topic_sequence=recent_topics,
                first_seen_ts=now,
                last_seen_ts=now,
                occurrence_count=1,
                payload_hash=payload_hash,
                actors=[actor],
            )
            self.patterns[pattern_id] = pattern

        # Compute stability
        pattern.stability_score = self._compute_stability(pattern, now)

        # Emit tracking event
        emit_bus_event(
            topic="hexis.bridge.pattern_detected",
            kind="metric",
            level="debug",
            actor=self.actor,
            data={
                "pattern_id": pattern_id,
                "occurrence_count": pattern.occurrence_count,
                "stability_score": pattern.stability_score,
                "topic": topic,
            },
        )

        # Check for stabilization threshold
        if (
            pattern.occurrence_count >= STABILIZATION_THRESHOLD
            and pattern.stability_score >= 0.5
        ):
            self._save_state()
            return pattern

        return None

    def propose_motif(self, pattern: HexisPattern) -> EntelechyCandidate | None:
        """
        Propose a stabilized pattern as a new ω-motif candidate.

        This is the key emergence point: ephemeral becomes permanent.
        """
        # Check if already proposed
        for c in self.candidates:
            if c.source_pattern_id == pattern.pattern_id and c.status == "pending":
                return None

        now = now_ts()
        candidate_id = f"ecand-{uuid.uuid4().hex[:8]}"
        proposed_motif_id = f"auto_{pattern.signature}"

        # Deduplicate vertices while preserving order
        seen = set()
        vertices = []
        for v in pattern.topic_sequence:
            if v not in seen:
                vertices.append(v)
                seen.add(v)

        # Compute proposed weight based on stability
        proposed_weight = round(pattern.stability_score * PHI, 3)

        candidate = EntelechyCandidate(
            candidate_id=candidate_id,
            source_pattern_id=pattern.pattern_id,
            proposed_motif_id=proposed_motif_id,
            vertices=vertices,
            proposed_weight=proposed_weight,
            stability_evidence=pattern.stability_score,
            promotion_ts=now,
            status="pending",
        )

        self.candidates.append(candidate)
        self._save_candidate(candidate)

        # Emit promotion event
        emit_bus_event(
            topic="hexis.bridge.motif_proposed",
            kind="artifact",
            level="info",
            actor=self.actor,
            data={
                "candidate_id": candidate_id,
                "proposed_motif_id": proposed_motif_id,
                "vertices": vertices,
                "stability_evidence": pattern.stability_score,
                "occurrence_count": pattern.occurrence_count,
                "source_pattern_id": pattern.pattern_id,
            },
        )

        return candidate

    def scan_hexis_buffers(self) -> list[EntelechyCandidate]:
        """
        Scan all hexis buffer files and propose any stabilized patterns.

        Returns list of new candidates proposed.
        """
        new_candidates = []

        # Find all buffer files
        buffer_files = list(self.hexis_dir.glob("*.buffer"))

        for buffer_path in buffer_files:
            try:
                with open(buffer_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                            pattern = self.ingest_hexis_message(msg)
                            if pattern:
                                candidate = self.propose_motif(pattern)
                                if candidate:
                                    new_candidates.append(candidate)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

        self._save_state()
        return new_candidates

    def get_pending_candidates(self) -> list[EntelechyCandidate]:
        """Get all pending candidates for review."""
        return [c for c in self.candidates if c.status == "pending"]

    def approve_candidate(self, candidate_id: str) -> bool:
        """Approve a candidate for promotion to entelechy."""
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                c.status = "approved"
                emit_bus_event(
                    topic="hexis.bridge.motif_approved",
                    kind="artifact",
                    level="info",
                    actor=self.actor,
                    data={
                        "candidate_id": candidate_id,
                        "motif_id": c.proposed_motif_id,
                    },
                )
                return True
        return False

    def reject_candidate(self, candidate_id: str, reason: str = "") -> bool:
        """Reject a candidate."""
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                c.status = "rejected"
                emit_bus_event(
                    topic="hexis.bridge.motif_rejected",
                    kind="artifact",
                    level="info",
                    actor=self.actor,
                    data={
                        "candidate_id": candidate_id,
                        "reason": reason,
                    },
                )
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics."""
        now = now_ts()
        active_patterns = [
            p for p in self.patterns.values()
            if now - p.last_seen_ts < STABILIZATION_WINDOW_S
        ]

        return {
            "total_patterns": len(self.patterns),
            "active_patterns": len(active_patterns),
            "total_candidates": len(self.candidates),
            "pending_candidates": len([c for c in self.candidates if c.status == "pending"]),
            "approved_candidates": len([c for c in self.candidates if c.status == "approved"]),
            "rejected_candidates": len([c for c in self.candidates if c.status == "rejected"]),
            "window_size": len(self.topic_window),
            "stabilization_threshold": STABILIZATION_THRESHOLD,
            "phi_weight": PHI,
        }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hexis-Entelechy Bridge: From ephemeral to permanent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan command
    p_scan = subparsers.add_parser("scan", help="Scan hexis buffers for stable patterns")
    p_scan.add_argument("--hexis-dir", type=Path, default=HEXIS_BUFFER_DIR)

    # stats command
    subparsers.add_parser("stats", help="Show bridge statistics")

    # candidates command
    subparsers.add_parser("candidates", help="List pending candidates")

    # approve command
    p_approve = subparsers.add_parser("approve", help="Approve a candidate")
    p_approve.add_argument("--id", required=True, help="Candidate ID")

    # reject command
    p_reject = subparsers.add_parser("reject", help="Reject a candidate")
    p_reject.add_argument("--id", required=True, help="Candidate ID")
    p_reject.add_argument("--reason", default="", help="Rejection reason")

    args = parser.parse_args()
    bridge = HexisEntelechyBridge()

    if args.command == "scan":
        candidates = bridge.scan_hexis_buffers()
        if candidates:
            print(f"Proposed {len(candidates)} new motif candidates:")
            for c in candidates:
                print(f"  {c.candidate_id}: {c.proposed_motif_id}")
                print(f"    vertices: {' -> '.join(c.vertices)}")
                print(f"    stability: {c.stability_evidence:.3f}")
        else:
            print("No new candidates detected")
        return 0

    elif args.command == "stats":
        stats = bridge.get_stats()
        print("Hexis-Entelechy Bridge Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    elif args.command == "candidates":
        pending = bridge.get_pending_candidates()
        if pending:
            print(f"Pending candidates ({len(pending)}):")
            for c in pending:
                print(f"  {c.candidate_id}:")
                print(f"    motif: {c.proposed_motif_id}")
                print(f"    vertices: {' -> '.join(c.vertices)}")
                print(f"    stability: {c.stability_evidence:.3f}")
                print(f"    weight: {c.proposed_weight}")
        else:
            print("No pending candidates")
        return 0

    elif args.command == "approve":
        if bridge.approve_candidate(args.id):
            print(f"Approved: {args.id}")
            return 0
        print(f"Candidate not found: {args.id}")
        return 1

    elif args.command == "reject":
        if bridge.reject_candidate(args.id, args.reason):
            print(f"Rejected: {args.id}")
            return 0
        print(f"Candidate not found: {args.id}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
