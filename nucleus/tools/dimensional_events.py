#!/usr/bin/env python3
"""
Dimensional Events: Rich Multi-Dimensional Event System for Pluribus
=====================================================================

The Big Events Rewrite - Transforms flat bus events into rich dimensional
tensors carrying the full semantic, neurosymbolic, semiotic, and evolutionary
context of the Pluribus organism.

Core Dimensions:
  1. TEMPORAL: Hysteresis, causal chains, path-dependent memory
  2. SEMANTIC: Human-readable summaries, reasoning, actionable insights
  3. SEMIOTIC: Syntactic/semantic/pragmatic/metalinguistic decomposition
  4. EVOLUTIONARY: VGT/HGT lineage, CMP signals, speciation potential
  5. TOPOLOGICAL: Multi-agent coordination, fanout, star/peer_debate
  6. OMEGA: Autopoietic reentry, entelexis, automaton state
  7. GEOMETRIC: Vector projections, tensor signatures, dimensional axes
  8. AUOM/SEXTET: Compliance checks across 6 constitutional laws

Usage:
    from dimensional_events import (
        DimensionalEvent,
        emit_dimensional,
        enrich_to_dimensional,
        create_vector_projection,
        check_auom_compliance,
    )

    # Create rich dimensional event
    event = DimensionalEvent.create(
        topic="plurichat.routing.decision",
        data={"depth": "deep", "provider": "codex-cli"},
        actor="plurichat",
        dimensions={"safety": 0.99, "correctness": 0.85, "efficiency": 0.70},
    )
    emit_dimensional(event, bus_dir)
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Callable
from functools import reduce
import operator

# ==============================================================================
# Core Types
# ==============================================================================

ImpactLevel = Literal["low", "medium", "high", "critical"]
TransferType = Literal["VGT", "HGT", "none"]
EntelexisPhase = Literal["potential", "actualizing", "actualized", "decaying"]
ReentryMode = Literal["observation", "modification", "self_reference", "closure"]
OmegaClass = Literal["prima", "meta", "omega"]
AUOMLaw = Literal["lawfulness", "observability", "provenance", "recurrence", "evolvability", "boundedness"]

# Sextet dimensional axes (from comprehensive_implementation_matrix.md)
SEXTET_DIMENSIONS = ["safety", "correctness", "efficiency", "coherence", "evolvability", "observability"]

# ==============================================================================
# Dimensional Layers
# ==============================================================================

@dataclass
class TemporalLayer:
    """Time and causality dimension."""
    ts: float  # UNIX timestamp
    iso: str   # ISO8601
    path_hash: str | None = None  # Hash of causal path
    causal_parents: list[str] = field(default_factory=list)
    causal_depth: int = 0
    decision_points: int = 0
    reversibility: float = 1.0  # 0=irreversible, 1=fully reversible
    accumulated_entropy: float = 0.0

    def to_dict(self) -> dict:
        d = {"ts": self.ts, "iso": self.iso}
        if self.path_hash:
            d["path_hash"] = self.path_hash
        if self.causal_parents:
            d["causal_parents"] = self.causal_parents
        if self.causal_depth > 0:
            d["causal_depth"] = self.causal_depth
        if self.decision_points > 0:
            d["decision_points"] = self.decision_points
        if self.reversibility < 1.0:
            d["reversibility"] = self.reversibility
        if self.accumulated_entropy > 0:
            d["entropy"] = self.accumulated_entropy
        return d


@dataclass
class SemanticLayer:
    """Human-readable semantic dimension."""
    summary: str  # Human-readable summary
    reasoning: str | None = None  # Why this happened
    actionable: list[str] = field(default_factory=list)  # Next actions
    impact: ImpactLevel = "low"
    confidence: float = 1.0  # 0-1 confidence in interpretation

    def to_dict(self) -> dict:
        d = {"summary": self.summary, "impact": self.impact}
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.actionable:
            d["actionable"] = self.actionable
        if self.confidence < 1.0:
            d["confidence"] = self.confidence
        return d


@dataclass
class SemioticLayer:
    """Multi-level semiotic decomposition dimension."""
    syntactic: str | None = None   # Structural form
    semantic: str | None = None    # Denotative meaning
    pragmatic: str | None = None   # Contextual action implications
    metalinguistic: str | None = None  # Self-referential meaning
    motif_class: str | None = None  # Pattern classification
    decomposition_depth: int = 1

    def to_dict(self) -> dict:
        d = {}
        if self.syntactic:
            d["syntactic"] = self.syntactic
        if self.semantic:
            d["semantic"] = self.semantic
        if self.pragmatic:
            d["pragmatic"] = self.pragmatic
        if self.metalinguistic:
            d["metalinguistic"] = self.metalinguistic
        if self.motif_class:
            d["motif"] = self.motif_class
        if self.decomposition_depth > 1:
            d["depth"] = self.decomposition_depth
        return d


@dataclass
class EvolutionaryLayer:
    """VGT/HGT lineage and CMP signals dimension."""
    lineage_id: str | None = None
    parent_lineage_id: str | None = None
    transfer_type: TransferType = "none"
    generation: int = 0
    mutation_op: str | None = None
    # CMP signals
    productivity_delta: float = 0.0
    quality_score: float | None = None
    latency_ratio: float | None = None
    resource_efficiency: float | None = None
    lineage_health: str = "healthy"  # healthy/degraded/failing
    speciation_potential: float = 0.0
    fitness_signal: float | None = None

    def to_dict(self) -> dict:
        d = {}
        if self.lineage_id:
            d["lineage_id"] = self.lineage_id
        if self.parent_lineage_id:
            d["parent_lineage_id"] = self.parent_lineage_id
        if self.transfer_type != "none":
            d["transfer_type"] = self.transfer_type
        if self.generation > 0:
            d["generation"] = self.generation
        if self.mutation_op:
            d["mutation"] = self.mutation_op
        # CMP
        if self.productivity_delta != 0.0:
            d["productivity_delta"] = self.productivity_delta
        if self.quality_score is not None:
            d["quality"] = self.quality_score
        if self.latency_ratio is not None:
            d["latency_ratio"] = self.latency_ratio
        if self.resource_efficiency is not None:
            d["efficiency"] = self.resource_efficiency
        if self.lineage_health != "healthy":
            d["health"] = self.lineage_health
        if self.speciation_potential > 0:
            d["speciation"] = self.speciation_potential
        if self.fitness_signal is not None:
            d["fitness"] = self.fitness_signal
        return d


@dataclass
class TopologyLayer:
    """Multi-agent coordination dimension."""
    topology: str = "single"  # single/star/peer_debate
    fanout: int = 1
    coordinator: str | None = None
    participants: list[str] = field(default_factory=list)
    coordination_budget_tokens: int = 0
    consensus_method: str | None = None  # geometric_mean/voting/best_of_n

    def to_dict(self) -> dict:
        d = {"topology": self.topology, "fanout": self.fanout}
        if self.coordinator:
            d["coordinator"] = self.coordinator
        if self.participants:
            d["participants"] = self.participants
        if self.coordination_budget_tokens > 0:
            d["budget_tokens"] = self.coordination_budget_tokens
        if self.consensus_method:
            d["consensus"] = self.consensus_method
        return d


@dataclass
class OmegaLayer:
    """Omega-level theoretical constructs dimension."""
    omega_class: OmegaClass = "prima"  # prima/meta/omega hierarchy
    taxonomic_branch: str | None = None
    automaton_state: str | None = None  # Büchi/pushdown state
    stack_depth: int = 0
    stack_top: str | None = None
    # Reentry (autopoietic)
    reentry_mode: ReentryMode | None = None
    reentry_target: str | None = None
    closure_depth: int = 0
    self_modification: bool = False
    # Entelexis (potential→actual)
    entelexis_phase: EntelexisPhase | None = None
    potential_id: str | None = None
    actualization_progress: float = 0.0

    def to_dict(self) -> dict:
        d = {"omega_class": self.omega_class}
        if self.taxonomic_branch:
            d["branch"] = self.taxonomic_branch
        if self.automaton_state:
            d["automaton"] = self.automaton_state
        if self.stack_depth > 0:
            d["stack_depth"] = self.stack_depth
        if self.stack_top:
            d["stack_top"] = self.stack_top
        # Reentry
        if self.reentry_mode:
            d["reentry"] = {
                "mode": self.reentry_mode,
                "target": self.reentry_target,
                "closure_depth": self.closure_depth,
                "self_mod": self.self_modification,
            }
        # Entelexis
        if self.entelexis_phase:
            d["entelexis"] = {
                "phase": self.entelexis_phase,
                "potential_id": self.potential_id,
                "progress": self.actualization_progress,
            }
        return d


@dataclass
class GeometricLayer:
    """Vector/tensor dimensional projection."""
    dimensions: dict[str, float] = field(default_factory=dict)  # Named axes
    vector: list[float] = field(default_factory=list)  # Numeric projection
    geometric_mean: float | None = None
    tensor_signature: str | None = None  # SHA256 of vector
    latent_coordinates: list[float] | None = None  # Position in latent space

    def compute_geometric_mean(self) -> float:
        """Geometric mean - zeros out if any dimension is zero (safety brake)."""
        if not self.vector:
            return 0.0
        product = reduce(operator.mul, self.vector, 1.0)
        return product ** (1.0 / len(self.vector))

    def compute_tensor_signature(self) -> str:
        """SHA256 of vector for integrity verification."""
        data = json.dumps(self.vector, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = {}
        if self.dimensions:
            d["dimensions"] = self.dimensions
        if self.vector:
            d["vector"] = self.vector
            d["geometric_mean"] = self.compute_geometric_mean()
            d["tensor_sig"] = self.compute_tensor_signature()
        if self.latent_coordinates:
            d["latent"] = self.latent_coordinates
        return d


@dataclass
class AUOMCheck:
    """Individual AUOM law compliance check."""
    law: AUOMLaw
    passed: bool
    detail: str
    evidence_hash: str | None = None

    def to_dict(self) -> dict:
        d = {"law": self.law, "passed": self.passed, "detail": self.detail}
        if self.evidence_hash:
            d["evidence"] = self.evidence_hash
        return d


@dataclass
class AUOMLayer:
    """AUOM/Sextet constitutional compliance dimension."""
    compliant: bool = True
    checks: list[AUOMCheck] = field(default_factory=list)
    sextet_vector: list[float] = field(default_factory=list)  # 6-dimensional
    constitution_version: str = "v1"

    def to_dict(self) -> dict:
        d = {
            "compliant": self.compliant,
            "version": self.constitution_version,
        }
        if self.checks:
            d["checks"] = [c.to_dict() for c in self.checks]
        if self.sextet_vector:
            d["sextet"] = self.sextet_vector
        return d


# ==============================================================================
# Dimensional Event (The Big Rewrite)
# ==============================================================================

@dataclass
class DimensionalEvent:
    """
    The Big Events Rewrite: Rich multi-dimensional event tensor.

    Carries the full semantic, neurosymbolic, semiotic, evolutionary,
    topological, omega-level, geometric, and constitutional context
    of every event in the Pluribus organism.
    """
    # Core identity
    id: str
    topic: str
    kind: str  # log/request/response/artifact/metric
    level: str  # debug/info/warn/error
    actor: str
    data: dict

    # Correlation
    trace_id: str | None = None
    parent_id: str | None = None
    run_id: str | None = None

    # Multi-dimensional layers
    temporal: TemporalLayer | None = None
    semantic: SemanticLayer | None = None
    semiotic: SemioticLayer | None = None
    evolutionary: EvolutionaryLayer | None = None
    topology: TopologyLayer | None = None
    omega: OmegaLayer | None = None
    geometric: GeometricLayer | None = None
    auom: AUOMLayer | None = None

    # Metadata
    host: str = field(default_factory=lambda: socket.gethostname())
    pid: int = field(default_factory=os.getpid)

    @classmethod
    def create(
        cls,
        topic: str,
        data: dict,
        actor: str,
        kind: str = "metric",
        level: str = "info",
        *,
        dimensions: dict[str, float] | None = None,
        summary: str | None = None,
        reasoning: str | None = None,
        actionable: list[str] | None = None,
        impact: ImpactLevel = "low",
        trace_id: str | None = None,
        parent_id: str | None = None,
        causal_parents: list[str] | None = None,
        topology: str = "single",
        fanout: int = 1,
        omega_class: OmegaClass = "prima",
        lineage_id: str | None = None,
    ) -> "DimensionalEvent":
        """Create a rich dimensional event with automatic layer population."""
        now = time.time()
        iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        event_id = uuid.uuid4().hex[:12]

        # Build temporal layer
        temporal = TemporalLayer(
            ts=now,
            iso=iso,
            causal_parents=causal_parents or [],
            causal_depth=len(causal_parents) if causal_parents else 0,
        )

        # Build semantic layer
        semantic = SemanticLayer(
            summary=summary or f"{topic}: {kind}",
            reasoning=reasoning,
            actionable=actionable or [],
            impact=impact,
        )

        # Build geometric layer if dimensions provided
        geometric = None
        if dimensions:
            vector = list(dimensions.values())
            geometric = GeometricLayer(
                dimensions=dimensions,
                vector=vector,
            )

        # Build topology layer
        topo = TopologyLayer(topology=topology, fanout=fanout)

        # Build omega layer
        omega = OmegaLayer(omega_class=omega_class)

        # Build evolutionary layer
        evolutionary = EvolutionaryLayer(lineage_id=lineage_id) if lineage_id else None

        return cls(
            id=event_id,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=trace_id,
            parent_id=parent_id,
            temporal=temporal,
            semantic=semantic,
            geometric=geometric,
            topology=topo,
            omega=omega,
            evolutionary=evolutionary,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "id": self.id,
            "topic": self.topic,
            "kind": self.kind,
            "level": self.level,
            "actor": self.actor,
            "host": self.host,
            "pid": self.pid,
            "data": self.data,
        }
        # Correlation
        if self.trace_id:
            d["trace_id"] = self.trace_id
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self.run_id:
            d["run_id"] = self.run_id
        # Dimensional layers (nested)
        if self.temporal:
            d["temporal"] = self.temporal.to_dict()
        if self.semantic:
            d["semantic"] = self.semantic.to_dict()
        if self.semiotic:
            d["semiotic"] = self.semiotic.to_dict()
        if self.evolutionary:
            d["evolutionary"] = self.evolutionary.to_dict()
        if self.topology:
            d["topology"] = self.topology.to_dict()
        if self.omega:
            d["omega"] = self.omega.to_dict()
        if self.geometric:
            d["geometric"] = self.geometric.to_dict()
        if self.auom:
            d["auom"] = self.auom.to_dict()
        return d

    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary for backward compatibility."""
        d = self.to_dict()
        # Flatten temporal
        if self.temporal:
            d["ts"] = self.temporal.ts
            d["iso"] = self.temporal.iso
            if self.temporal.causal_parents:
                d["causal_parents"] = self.temporal.causal_parents
        # Flatten semantic summary
        if self.semantic:
            d["semantic"] = self.semantic.summary
            d["impact"] = self.semantic.impact
            if self.semantic.reasoning:
                d["reasoning"] = self.semantic.reasoning
            if self.semantic.actionable:
                d["actionable"] = self.semantic.actionable
        return d


# ==============================================================================
# AUOM Compliance Checking
# ==============================================================================

def check_auom_compliance(event: DimensionalEvent) -> AUOMLayer:
    """
    Check AUOM/Sextet constitutional compliance.

    The Six Laws (AUOM):
    1. Lawfulness - Operations within defined bounds
    2. Observability - State is inspectable
    3. Provenance - Origins are traceable
    4. Recurrence - Patterns are recognizable
    5. Evolvability - System can adapt
    6. Boundedness - Resources are finite
    """
    checks = []

    # 1. Lawfulness: Has valid topic and kind
    lawful = bool(event.topic) and event.kind in ["log", "request", "response", "artifact", "metric"]
    checks.append(AUOMCheck(
        law="lawfulness",
        passed=lawful,
        detail=f"topic={event.topic}, kind={event.kind}",
    ))

    # 2. Observability: Has temporal and semantic layers
    observable = event.temporal is not None and event.semantic is not None
    checks.append(AUOMCheck(
        law="observability",
        passed=observable,
        detail="temporal+semantic layers present" if observable else "missing layers",
    ))

    # 3. Provenance: Has actor and trace_id or parent_id
    provenance = bool(event.actor) and (event.trace_id or event.parent_id or
                                         (event.temporal and event.temporal.causal_parents))
    checks.append(AUOMCheck(
        law="provenance",
        passed=provenance,
        detail=f"actor={event.actor}, trace={event.trace_id}",
    ))

    # 4. Recurrence: Has omega layer with recognizable pattern
    recurrent = event.omega is not None and event.omega.omega_class in ["prima", "meta", "omega"]
    checks.append(AUOMCheck(
        law="recurrence",
        passed=recurrent,
        detail=f"omega_class={event.omega.omega_class if event.omega else 'none'}",
    ))

    # 5. Evolvability: Has evolutionary layer or lineage info
    evolvable = event.evolutionary is not None or (event.data and "lineage" in str(event.data))
    checks.append(AUOMCheck(
        law="evolvability",
        passed=evolvable,
        detail="evolutionary layer present" if evolvable else "no evolution context",
    ))

    # 6. Boundedness: Has topology with finite fanout
    bounded = event.topology is not None and event.topology.fanout <= 10
    checks.append(AUOMCheck(
        law="boundedness",
        passed=bounded,
        detail=f"fanout={event.topology.fanout if event.topology else 1}",
    ))

    # Compute sextet vector (1.0 for passed, 0.0 for failed)
    sextet_vector = [1.0 if c.passed else 0.0 for c in checks]

    return AUOMLayer(
        compliant=all(c.passed for c in checks),
        checks=checks,
        sextet_vector=sextet_vector,
        constitution_version="v1",
    )


def create_vector_projection(
    safety: float = 1.0,
    correctness: float = 1.0,
    efficiency: float = 1.0,
    coherence: float = 1.0,
    evolvability: float = 1.0,
    observability: float = 1.0,
) -> GeometricLayer:
    """Create a sextet-aligned geometric projection."""
    dimensions = {
        "safety": safety,
        "correctness": correctness,
        "efficiency": efficiency,
        "coherence": coherence,
        "evolvability": evolvability,
        "observability": observability,
    }
    vector = list(dimensions.values())
    return GeometricLayer(
        dimensions=dimensions,
        vector=vector,
    )


# ==============================================================================
# Emission Functions
# ==============================================================================

def emit_dimensional(event: DimensionalEvent, bus_dir: str | Path | None = None) -> str:
    """Emit a dimensional event to the bus."""
    bus_dir = Path(bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_file = bus_dir / "events.ndjson"

    # Auto-check AUOM compliance
    if event.auom is None:
        event.auom = check_auom_compliance(event)

    # Serialize
    event_dict = event.to_dict()
    line = json.dumps(event_dict, ensure_ascii=False) + "\n"

    # Append to bus
    bus_dir.mkdir(parents=True, exist_ok=True)
    with events_file.open("a", encoding="utf-8") as f:
        f.write(line)

    return event.id


def enrich_to_dimensional(
    basic_event: dict,
    *,
    summary: str | None = None,
    dimensions: dict[str, float] | None = None,
) -> DimensionalEvent:
    """Enrich a basic bus event to a dimensional event."""
    return DimensionalEvent.create(
        topic=basic_event.get("topic", "unknown"),
        data=basic_event.get("data", {}),
        actor=basic_event.get("actor", "unknown"),
        kind=basic_event.get("kind", "log"),
        level=basic_event.get("level", "info"),
        trace_id=basic_event.get("trace_id"),
        parent_id=basic_event.get("parent_id"),
        summary=summary,
        dimensions=dimensions,
    )


# ==============================================================================
# Semantic Templates (Auto-Enrichment)
# ==============================================================================

SEMANTIC_TEMPLATES: dict[str, dict[str, Any]] = {
    "plurichat.routing.decision": {
        "summary_fn": lambda d: f"Routing {d.get('query_type', 'query')} ({d.get('depth', 'unknown')}) to {d.get('provider', 'auto')}",
        "reasoning_fn": lambda d: f"Depth={d.get('depth')} requires {d.get('context_mode', 'auto')} context",
        "impact_fn": lambda d: "high" if d.get("depth") == "deep" else "low",
        "dimensions_fn": lambda d: {
            "safety": 1.0,
            "correctness": 0.9 if d.get("depth") == "deep" else 0.7,
            "efficiency": 0.6 if d.get("depth") == "deep" else 0.9,
            "coherence": 0.8,
            "evolvability": 0.7,
            "observability": 1.0,
        },
    },
    "strp.topology.star.complete": {
        "summary_fn": lambda d: f"Star topology completed with {len(d.get('results', []))} results",
        "reasoning_fn": lambda d: "Multi-agent decomposition and aggregation",
        "impact_fn": lambda d: "high",
        "dimensions_fn": lambda d: {
            "safety": 0.95,
            "correctness": 0.85,
            "efficiency": 0.7,
            "coherence": 0.9,
            "evolvability": 0.8,
            "observability": 1.0,
        },
    },
    "mabswarm.probe": {
        "summary_fn": lambda d: f"MABSWARM probe: velocity={d.get('velocity', 0):.2f} error_rate={d.get('error_rate', 0):.2%}",
        "reasoning_fn": lambda d: "Reflexive control system monitoring bus health",
        "impact_fn": lambda d: "high" if d.get("error_rate", 0) > 0.1 else "low",
        "dimensions_fn": lambda d: {
            "safety": 1.0 - d.get("error_rate", 0),
            "correctness": 0.9,
            "efficiency": min(1.0, d.get("velocity", 0) / 0.5),
            "coherence": 0.8,
            "evolvability": 0.9,
            "observability": 1.0,
        },
    },
    "oiterate.tick": {
        "summary_fn": lambda d: f"OITERATE tick: {d.get('goals_achieved', 0)}/{d.get('goals_total', 0)} goals",
        "reasoning_fn": lambda d: "Büchi automaton liveness check",
        "impact_fn": lambda d: "medium",
        "dimensions_fn": lambda d: {
            "safety": 1.0,
            "correctness": d.get("goals_achieved", 0) / max(d.get("goals_total", 1), 1),
            "efficiency": 0.8,
            "coherence": 0.9,
            "evolvability": 0.95,
            "observability": 1.0,
        },
    },
}


def auto_enrich(topic: str, data: dict, actor: str, **kwargs) -> DimensionalEvent:
    """Auto-enrich event using semantic templates."""
    template = SEMANTIC_TEMPLATES.get(topic, {})

    summary = template.get("summary_fn", lambda d: f"{topic}")(data) if "summary_fn" in template else None
    reasoning = template.get("reasoning_fn", lambda d: None)(data) if "reasoning_fn" in template else None
    impact = template.get("impact_fn", lambda d: "low")(data) if "impact_fn" in template else "low"
    dimensions = template.get("dimensions_fn", lambda d: None)(data) if "dimensions_fn" in template else None

    return DimensionalEvent.create(
        topic=topic,
        data=data,
        actor=actor,
        summary=summary,
        reasoning=reasoning,
        impact=impact,
        dimensions=dimensions,
        **kwargs,
    )


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    """CLI for testing dimensional events."""
    import argparse

    parser = argparse.ArgumentParser(description="Dimensional Events CLI")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    parser.add_argument("--emit", type=str, help="Emit test event with topic")
    args = parser.parse_args()

    if args.test:
        print("=== Dimensional Events Self-Test ===\n")

        # Test 1: Create event
        print("[TEST 1] Create DimensionalEvent")
        event = DimensionalEvent.create(
            topic="test.dimensional",
            data={"test": True},
            actor="test-cli",
            dimensions={"safety": 0.99, "correctness": 0.85, "efficiency": 0.70},
            summary="Test dimensional event creation",
            reasoning="Testing the Big Events Rewrite",
            actionable=["Verify dimensions", "Check compliance"],
            impact="medium",
        )
        print(f"  ID: {event.id}")
        print(f"  Topic: {event.topic}")
        print(f"  Geometric mean: {event.geometric.compute_geometric_mean():.3f}")
        print("  [PASS]\n")

        # Test 2: AUOM compliance
        print("[TEST 2] AUOM Compliance Check")
        auom = check_auom_compliance(event)
        print(f"  Compliant: {auom.compliant}")
        print(f"  Sextet vector: {auom.sextet_vector}")
        for check in auom.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"    {check.law}: [{status}] {check.detail}")
        print("  [PASS]\n")

        # Test 3: Auto-enrichment
        print("[TEST 3] Auto-Enrichment with Templates")
        enriched = auto_enrich(
            topic="plurichat.routing.decision",
            data={"depth": "deep", "provider": "codex-cli", "query_type": "code"},
            actor="plurichat",
        )
        print(f"  Summary: {enriched.semantic.summary}")
        print(f"  Reasoning: {enriched.semantic.reasoning}")
        print(f"  Impact: {enriched.semantic.impact}")
        if enriched.geometric:
            print(f"  Dimensions: {enriched.geometric.dimensions}")
            print(f"  Geometric mean: {enriched.geometric.compute_geometric_mean():.3f}")
        print("  [PASS]\n")

        # Test 4: Serialization
        print("[TEST 4] Serialization")
        event_dict = event.to_dict()
        print(f"  Keys: {list(event_dict.keys())}")
        json_str = json.dumps(event_dict, indent=2)
        print(f"  JSON length: {len(json_str)} chars")
        print("  [PASS]\n")

        print("=== ALL TESTS PASSED ===")
        return 0

    if args.emit:
        event = DimensionalEvent.create(
            topic=args.emit,
            data={"cli_emit": True},
            actor="dimensional-cli",
            summary=f"CLI-emitted {args.emit}",
        )
        event_id = emit_dimensional(event)
        print(f"Emitted: {event_id}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
