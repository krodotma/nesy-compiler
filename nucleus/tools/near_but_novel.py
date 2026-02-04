#!/usr/bin/env python3
"""
near_but_novel.py - Edge of Chaos Event Emitter

DUALITY-BIND E14: The Near-But-Novel Gradient

From HGT Protocol v2:
> "HGT when spectral/geometry distance is 'near but novel'"

The catalyst: This "near but novel" principle is itself a teleological
attractor. It defines the EDGE OF CHAOS:
- Too near → stagnation (no learning)
- Too novel → death (catastrophic forgetting)

The system should emit events when it detects near-but-novel relationships
even OUTSIDE HGT context. These events become seeds for exploration.

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configuration
NEAR_NOVEL_DIR = Path(os.environ.get("NEAR_NOVEL_DIR", ".pluribus/near_novel"))
BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")

# The Goldilocks Zone: Near enough to be relevant, novel enough to be interesting
NEAR_THRESHOLD = 0.3  # Below this = too similar (stagnation zone)
NOVEL_THRESHOLD = 0.7  # Above this = too different (death zone)
PHI = 1.618033988749895  # Golden ratio

try:
    from nucleus.tools import agent_bus
except ImportError:
    agent_bus = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


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
class Entity:
    """
    An entity that can be compared for near-but-novel relationships.

    Entities can be lineages, modules, embeddings, or any comparable unit.
    """
    entity_id: str
    entity_type: str  # lineage, module, embedding, motif
    embedding: list[float] | None = None
    signature: str = ""  # Content hash for identity
    metadata: dict[str, Any] = field(default_factory=dict)
    created_ts: float = field(default_factory=now_ts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(**data)


@dataclass
class NearButNovelRelation:
    """
    A detected near-but-novel relationship between two entities.

    This is the Goldilocks zone: not too similar, not too different.
    """
    relation_id: str
    entity_a_id: str
    entity_b_id: str
    distance: float
    near_score: float  # How near (0=identical, 1=completely different)
    novel_score: float  # How novel (interest value)
    in_goldilocks: bool  # True if in the optimal zone
    detected_ts: float
    context: str = ""  # What prompted this detection
    exploration_seeds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExplorationSeed:
    """
    A seed for exploration, generated from near-but-novel detection.

    These become prompts for the system to explore new territory.
    """
    seed_id: str
    source_relation_id: str
    seed_type: str  # hgt_candidate, motif_merge, lineage_cross, exploration_prompt
    description: str
    priority: float
    created_ts: float
    status: str = "pending"  # pending, explored, discarded
    exploration_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NearButNovelDetector:
    """
    Detect near-but-novel relationships and emit exploration seeds.

    This implements the edge-of-chaos principle: the optimal zone for
    learning and adaptation lies between stagnation (too similar) and
    death (too different).
    """

    def __init__(
        self,
        state_dir: Path | None = None,
        actor: str = "near_but_novel",
        near_threshold: float = NEAR_THRESHOLD,
        novel_threshold: float = NOVEL_THRESHOLD,
    ):
        self.state_dir = state_dir or NEAR_NOVEL_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor = actor
        self.near_threshold = near_threshold
        self.novel_threshold = novel_threshold

        # Entity registry
        self.entities: dict[str, Entity] = {}

        # Detected relations
        self.relations: list[NearButNovelRelation] = []

        # Exploration seeds
        self.seeds: list[ExplorationSeed] = []

        # Load state
        self._load_state()

    def _state_path(self) -> Path:
        return self.state_dir / "near_novel_state.json"

    def _seeds_path(self) -> Path:
        return self.state_dir / "exploration_seeds.ndjson"

    def _relations_path(self) -> Path:
        return self.state_dir / "relations.ndjson"

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for e in data.get("entities", []):
                    entity = Entity.from_dict(e)
                    self.entities[entity.entity_id] = entity
            except Exception:
                pass

        # Load relations
        rpath = self._relations_path()
        if rpath.exists():
            for line in rpath.read_text().strip().split("\n"):
                if line:
                    try:
                        self.relations.append(
                            NearButNovelRelation(**json.loads(line))
                        )
                    except Exception:
                        continue

        # Load seeds
        spath = self._seeds_path()
        if spath.exists():
            for line in spath.read_text().strip().split("\n"):
                if line:
                    try:
                        self.seeds.append(ExplorationSeed(**json.loads(line)))
                    except Exception:
                        continue

    def _save_state(self) -> None:
        """Persist state to disk."""
        path = self._state_path()
        data = {
            "entities": [e.to_dict() for e in self.entities.values()],
            "last_updated": now_iso(),
        }
        with open(path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)

    def _save_relation(self, rel: NearButNovelRelation) -> None:
        """Append relation to log."""
        path = self._relations_path()
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(rel.to_dict()) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def _save_seed(self, seed: ExplorationSeed) -> None:
        """Append seed to log."""
        path = self._seeds_path()
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(seed.to_dict()) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def _compute_distance(self, a: Entity, b: Entity) -> float:
        """
        Compute distance between two entities.

        Uses embedding distance if available, else signature similarity.
        """
        if a.embedding and b.embedding and HAS_NUMPY:
            # Cosine distance
            va = np.array(a.embedding)
            vb = np.array(b.embedding)

            # Handle dimension mismatch
            if len(va) != len(vb):
                min_len = min(len(va), len(vb))
                va = va[:min_len]
                vb = vb[:min_len]

            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)

            if norm_a > 0 and norm_b > 0:
                cosine_sim = np.dot(va, vb) / (norm_a * norm_b)
                return 1.0 - float(cosine_sim)  # Distance = 1 - similarity

        # Fallback: signature-based distance
        if a.signature and b.signature:
            # Jaccard distance on character n-grams
            def ngrams(s: str, n: int = 3) -> set:
                return {s[i:i+n] for i in range(len(s) - n + 1)}

            grams_a = ngrams(a.signature)
            grams_b = ngrams(b.signature)

            if grams_a or grams_b:
                intersection = len(grams_a & grams_b)
                union = len(grams_a | grams_b)
                jaccard_sim = intersection / union if union > 0 else 0
                return 1.0 - jaccard_sim

        # Last resort: random-ish distance based on IDs
        hash_a = int(hashlib.md5(a.entity_id.encode()).hexdigest()[:8], 16)
        hash_b = int(hashlib.md5(b.entity_id.encode()).hexdigest()[:8], 16)
        return abs(hash_a - hash_b) / (2 ** 32)

    def _compute_novelty(self, distance: float) -> float:
        """
        Compute novelty score from distance.

        Novelty peaks in the Goldilocks zone and falls off at extremes.
        Uses a beta-like distribution centered on the optimal zone.
        """
        # Optimal distance is the geometric mean of thresholds (golden ratio inspired)
        optimal = math.sqrt(self.near_threshold * self.novel_threshold)

        # Novelty score: peaked at optimal, falling off on both sides
        if distance < self.near_threshold:
            # Stagnation zone: low novelty, increasing toward threshold
            return distance / self.near_threshold * 0.5
        elif distance > self.novel_threshold:
            # Death zone: low novelty, decreasing from threshold
            return (1.0 - distance) / (1.0 - self.novel_threshold) * 0.5
        else:
            # Goldilocks zone: high novelty, peaked at optimal
            # Parabola with peak at optimal
            spread = (self.novel_threshold - self.near_threshold) / 2
            center = (self.near_threshold + self.novel_threshold) / 2
            normalized = abs(distance - center) / spread
            return 0.5 + 0.5 * (1.0 - normalized ** 2)

    def register_entity(self, entity: Entity) -> None:
        """Register an entity for comparison."""
        self.entities[entity.entity_id] = entity
        self._save_state()

    def compare(
        self,
        entity_a: Entity | str,
        entity_b: Entity | str,
        context: str = "",
    ) -> NearButNovelRelation | None:
        """
        Compare two entities and detect if they're in the near-but-novel zone.

        Returns the relation if it's in the Goldilocks zone, None otherwise.
        """
        # Resolve entity references
        if isinstance(entity_a, str):
            entity_a = self.entities.get(entity_a)
        if isinstance(entity_b, str):
            entity_b = self.entities.get(entity_b)

        if not entity_a or not entity_b:
            return None

        if entity_a.entity_id == entity_b.entity_id:
            return None  # Can't compare entity to itself

        distance = self._compute_distance(entity_a, entity_b)
        novelty = self._compute_novelty(distance)
        in_goldilocks = self.near_threshold <= distance <= self.novel_threshold

        relation = NearButNovelRelation(
            relation_id=f"nbn-{uuid.uuid4().hex[:8]}",
            entity_a_id=entity_a.entity_id,
            entity_b_id=entity_b.entity_id,
            distance=round(distance, 4),
            near_score=round(1.0 - distance, 4),
            novel_score=round(novelty, 4),
            in_goldilocks=in_goldilocks,
            detected_ts=now_ts(),
            context=context,
        )

        if in_goldilocks:
            # Generate exploration seeds
            seeds = self._generate_seeds(relation, entity_a, entity_b)
            relation.exploration_seeds = [s.seed_id for s in seeds]

            self.relations.append(relation)
            self._save_relation(relation)

            # Emit bus event
            emit_bus_event(
                topic="entelexis.near_but_novel.detected",
                kind="artifact",
                level="info",
                actor=self.actor,
                data={
                    "relation_id": relation.relation_id,
                    "entity_a": entity_a.entity_id,
                    "entity_b": entity_b.entity_id,
                    "distance": relation.distance,
                    "novelty": relation.novel_score,
                    "context": context,
                    "seeds": len(seeds),
                },
            )

            return relation

        return None

    def _generate_seeds(
        self,
        relation: NearButNovelRelation,
        entity_a: Entity,
        entity_b: Entity,
    ) -> list[ExplorationSeed]:
        """Generate exploration seeds from a near-but-novel relation."""
        seeds = []
        now = now_ts()

        # Seed type depends on entity types
        if entity_a.entity_type == "lineage" and entity_b.entity_type == "lineage":
            # HGT candidate
            seed = ExplorationSeed(
                seed_id=f"seed-{uuid.uuid4().hex[:8]}",
                source_relation_id=relation.relation_id,
                seed_type="hgt_candidate",
                description=f"Consider HGT between {entity_a.entity_id} and {entity_b.entity_id}",
                priority=relation.novel_score * PHI,
                created_ts=now,
            )
            seeds.append(seed)

        elif entity_a.entity_type == "motif" and entity_b.entity_type == "motif":
            # Motif merge candidate
            seed = ExplorationSeed(
                seed_id=f"seed-{uuid.uuid4().hex[:8]}",
                source_relation_id=relation.relation_id,
                seed_type="motif_merge",
                description=f"Consider merging motifs {entity_a.entity_id} and {entity_b.entity_id}",
                priority=relation.novel_score,
                created_ts=now,
            )
            seeds.append(seed)

        # Always generate a generic exploration prompt
        seed = ExplorationSeed(
            seed_id=f"seed-{uuid.uuid4().hex[:8]}",
            source_relation_id=relation.relation_id,
            seed_type="exploration_prompt",
            description=f"Explore connection between {entity_a.entity_id} ({entity_a.entity_type}) and {entity_b.entity_id} ({entity_b.entity_type})",
            priority=relation.novel_score / PHI,
            created_ts=now,
        )
        seeds.append(seed)

        for seed in seeds:
            self.seeds.append(seed)
            self._save_seed(seed)

        return seeds

    def scan_all_pairs(self, entity_type: str | None = None) -> list[NearButNovelRelation]:
        """
        Scan all entity pairs and detect near-but-novel relationships.

        This is expensive for large entity sets; use sparingly.
        """
        entities = list(self.entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]

        found = []
        for i, a in enumerate(entities):
            for b in entities[i+1:]:
                rel = self.compare(a, b, context="batch_scan")
                if rel:
                    found.append(rel)

        return found

    def get_pending_seeds(self) -> list[ExplorationSeed]:
        """Get all pending exploration seeds, sorted by priority."""
        pending = [s for s in self.seeds if s.status == "pending"]
        return sorted(pending, key=lambda s: -s.priority)

    def mark_seed_explored(self, seed_id: str, result: dict[str, Any] = None) -> bool:
        """Mark a seed as explored."""
        for seed in self.seeds:
            if seed.seed_id == seed_id:
                seed.status = "explored"
                seed.exploration_result = result or {}
                return True
        return False

    def mark_seed_discarded(self, seed_id: str) -> bool:
        """Mark a seed as discarded."""
        for seed in self.seeds:
            if seed.seed_id == seed_id:
                seed.status = "discarded"
                return True
        return False

    def get_goldilocks_zone(self) -> dict[str, float]:
        """Get the current Goldilocks zone thresholds."""
        return {
            "near_threshold": self.near_threshold,
            "novel_threshold": self.novel_threshold,
            "optimal_distance": math.sqrt(self.near_threshold * self.novel_threshold),
            "zone_width": self.novel_threshold - self.near_threshold,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        goldilocks_relations = [r for r in self.relations if r.in_goldilocks]
        pending_seeds = [s for s in self.seeds if s.status == "pending"]

        return {
            "total_entities": len(self.entities),
            "entity_types": list(set(e.entity_type for e in self.entities.values())),
            "total_relations": len(self.relations),
            "goldilocks_relations": len(goldilocks_relations),
            "total_seeds": len(self.seeds),
            "pending_seeds": len(pending_seeds),
            "explored_seeds": len([s for s in self.seeds if s.status == "explored"]),
            "discarded_seeds": len([s for s in self.seeds if s.status == "discarded"]),
            "avg_novelty": round(
                sum(r.novel_score for r in goldilocks_relations) / max(1, len(goldilocks_relations)),
                4
            ) if goldilocks_relations else 0,
            **self.get_goldilocks_zone(),
        }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Near-But-Novel: Edge of chaos detector"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats command
    subparsers.add_parser("stats", help="Show detector statistics")

    # zone command
    subparsers.add_parser("zone", help="Show Goldilocks zone parameters")

    # seeds command
    subparsers.add_parser("seeds", help="List pending exploration seeds")

    # compare command
    p_cmp = subparsers.add_parser("compare", help="Compare two entities")
    p_cmp.add_argument("--entity-a", required=True)
    p_cmp.add_argument("--entity-b", required=True)
    p_cmp.add_argument("--context", default="manual_compare")

    # register command
    p_reg = subparsers.add_parser("register", help="Register an entity")
    p_reg.add_argument("--id", required=True)
    p_reg.add_argument("--type", required=True)
    p_reg.add_argument("--signature", default="")
    p_reg.add_argument("--embedding", type=float, nargs="*")

    # scan command
    p_scan = subparsers.add_parser("scan", help="Scan all pairs")
    p_scan.add_argument("--type", default=None, help="Filter by entity type")

    args = parser.parse_args()
    detector = NearButNovelDetector()

    if args.command == "stats":
        stats = detector.get_stats()
        print("Near-But-Novel Detector Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    elif args.command == "zone":
        zone = detector.get_goldilocks_zone()
        print("Goldilocks Zone (Edge of Chaos):")
        print(f"  Stagnation Zone: distance < {zone['near_threshold']}")
        print(f"  Goldilocks Zone: {zone['near_threshold']} <= distance <= {zone['novel_threshold']}")
        print(f"  Death Zone: distance > {zone['novel_threshold']}")
        print(f"  Optimal Distance: {zone['optimal_distance']:.4f}")
        print(f"  Zone Width: {zone['zone_width']:.4f}")
        return 0

    elif args.command == "seeds":
        seeds = detector.get_pending_seeds()
        if seeds:
            print(f"Pending Exploration Seeds ({len(seeds)}):")
            for s in seeds[:20]:  # Limit output
                print(f"  {s.seed_id} [{s.seed_type}] priority={s.priority:.3f}")
                print(f"    {s.description}")
        else:
            print("No pending exploration seeds")
        return 0

    elif args.command == "compare":
        entity_a = detector.entities.get(args.entity_a)
        entity_b = detector.entities.get(args.entity_b)

        if not entity_a:
            print(f"Entity not found: {args.entity_a}")
            return 1
        if not entity_b:
            print(f"Entity not found: {args.entity_b}")
            return 1

        rel = detector.compare(entity_a, entity_b, args.context)
        if rel:
            print(f"Near-But-Novel Relation Detected!")
            print(f"  Distance: {rel.distance}")
            print(f"  Novelty: {rel.novel_score}")
            print(f"  In Goldilocks: {rel.in_goldilocks}")
            print(f"  Seeds Generated: {len(rel.exploration_seeds)}")
        else:
            # Still show the comparison even if not in Goldilocks
            dist = detector._compute_distance(entity_a, entity_b)
            nov = detector._compute_novelty(dist)
            zone = detector.get_goldilocks_zone()
            if dist < zone["near_threshold"]:
                zone_name = "Stagnation Zone (too similar)"
            elif dist > zone["novel_threshold"]:
                zone_name = "Death Zone (too different)"
            else:
                zone_name = "Goldilocks Zone"
            print(f"Comparison Result:")
            print(f"  Distance: {dist:.4f}")
            print(f"  Novelty: {nov:.4f}")
            print(f"  Zone: {zone_name}")
        return 0

    elif args.command == "register":
        entity = Entity(
            entity_id=args.id,
            entity_type=args.type,
            signature=args.signature,
            embedding=args.embedding if args.embedding else None,
        )
        detector.register_entity(entity)
        print(f"Registered entity: {args.id} ({args.type})")
        return 0

    elif args.command == "scan":
        relations = detector.scan_all_pairs(args.type)
        if relations:
            print(f"Found {len(relations)} near-but-novel relations:")
            for r in relations[:20]:
                print(f"  {r.entity_a_id} <-> {r.entity_b_id}")
                print(f"    distance={r.distance:.4f} novelty={r.novel_score:.4f}")
        else:
            print("No near-but-novel relations found")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
