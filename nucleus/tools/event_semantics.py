#!/usr/bin/env python3
"""
Event Semantics: Rich Semantic Event Enrichment for Pluribus
=============================================================

Transforms basic bus events into rich, actionable, human-readable events
that capture the evolutionary CMP/VGT/HGT nature of the multi-agent system.

Event Schema Extensions:
- semantic: Human-readable summary of what happened
- reasoning: Why this event/decision was made
- actionable: List of suggested next actions
- lineage: VGT/HGT evolutionary context
- cmp: Clade meta-productivity signals
- topology: Multi-agent coordination context
- impact: Estimated impact level (low/medium/high/critical)

Usage:
    from event_semantics import enrich_event, create_semantic_event

    # Enrich an existing event
    enriched = enrich_event(basic_event)

    # Create a new semantic event
    event = create_semantic_event(
        topic="plurichat.routing.decision",
        data={"depth": "deep", "provider": "codex-cli"},
        semantic="Routing deep query to top-tier agent for project understanding",
        reasoning="Query contains architecture keywords, requires full project context",
        actionable=["Monitor response quality", "Consider caching for similar queries"],
    )
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


# ==============================================================================
# Semantic Event Schema
# ==============================================================================

ImpactLevel = Literal["low", "medium", "high", "critical"]
TransferType = Literal["VGT", "HGT", "none"]
EntelexisPhase = Literal["potential", "actualizing", "actualized", "decaying"]
ReentryMode = Literal["observation", "modification", "self_reference", "closure"]
OmegaClass = Literal["prima", "meta", "omega"]  # Information hierarchy


# ==============================================================================
# Omega-Level Theoretical Constructs
# ==============================================================================
# These embody the deeper theoretical CS and emergent systems dimensions:
# - Autopoiesis/Reentry (Luhmann, Maturana): Self-referential system loops
# - Entelexis (Aristotelian): Actualization of potential
# - Hysteresis: Path-dependent memory in event sequences
# - Semioticalysis: Multi-level semiotic meaning decomposition
# - Omega DNA: Fundamental genetic code of agent evolution
# ==============================================================================

@dataclass
class ReentryMarker:
    """Autopoietic reentry tracking - events that recursively modify the system.

    Based on Luhmann's systems theory: observations that become part of
    what they observe, creating self-referential loops in the agent mesh.
    """
    mode: ReentryMode = "observation"
    references_event_id: str | None = None  # Event this reenters upon
    closure_depth: int = 0  # How many reentry levels deep
    self_modification: bool = False  # Does this event modify the emitter?

    def to_dict(self) -> dict:
        d = {"mode": self.mode}
        if self.references_event_id:
            d["references_event_id"] = self.references_event_id
        if self.closure_depth > 0:
            d["closure_depth"] = self.closure_depth
        if self.self_modification:
            d["self_modification"] = True
        return d


@dataclass
class EntelexisState:
    """Aristotelian entelechy - actualization of potential.

    Tracks the transition from potential (δύναμις) to actuality (ἐνέργεια).
    In agent systems: from request to response, from plan to execution.
    """
    phase: EntelexisPhase = "potential"
    potential_id: str | None = None  # Original potential event
    actualization_progress: float = 0.0  # 0.0 to 1.0
    form_signature: str | None = None  # The "form" being actualized
    material_context: str | None = None  # The "matter" being shaped

    def to_dict(self) -> dict:
        d = {"phase": self.phase}
        if self.potential_id:
            d["potential_id"] = self.potential_id
        if self.actualization_progress > 0:
            d["progress"] = self.actualization_progress
        if self.form_signature:
            d["form"] = self.form_signature
        if self.material_context:
            d["matter"] = self.material_context
        return d


@dataclass
class HysteresisTrace:
    """Path-dependent memory encoding.

    Events don't exist in isolation - they carry the hysteretic trace
    of the path that led to them. This affects interpretation and
    future evolution of the system.
    """
    path_hash: str | None = None  # Hash of event path
    decision_points: int = 0  # Number of branching decisions in path
    reversibility: float = 1.0  # 0.0 = irreversible, 1.0 = fully reversible
    accumulated_entropy: float = 0.0  # Information entropy along path
    causal_chain_length: int = 0  # Depth of causal ancestry

    def to_dict(self) -> dict:
        d = {}
        if self.path_hash:
            d["path_hash"] = self.path_hash
        if self.decision_points > 0:
            d["decision_points"] = self.decision_points
        if self.reversibility < 1.0:
            d["reversibility"] = self.reversibility
        if self.accumulated_entropy > 0:
            d["entropy"] = self.accumulated_entropy
        if self.causal_chain_length > 0:
            d["causal_depth"] = self.causal_chain_length
        return d


@dataclass
class SemioticalysisLayer:
    """Multi-level semiotic meaning decomposition.

    Events carry meaning at multiple levels - syntactic, semantic,
    pragmatic, and metalinguistic. This enables rich interpretation
    by both human operators and autonomous agents.
    """
    syntactic: str | None = None  # Structural form
    semantic: str | None = None  # Denotative meaning
    pragmatic: str | None = None  # Contextual action implications
    metalinguistic: str | None = None  # Self-referential meaning about meaning
    motif_class: str | None = None  # Pattern classification
    decomposition_depth: int = 1  # Levels of analysis performed

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
class OmegaContext:
    """Omega-level fundamental context - the "DNA" of agent evolution.

    Represents the deepest layer of event semantics, encoding the
    generative principles that govern system evolution. Not merely
    emulating neurons, but encoding the omega-code of emergent life.

    Taxonomic hyperclassing: Events exist in a latent evolutionary
    space where they can branch, merge, and speciate.
    """
    omega_class: OmegaClass = "prima"  # prima/meta/omega hierarchy
    taxonomic_branch: str | None = None  # Evolutionary branch ID
    latent_coordinates: list[float] | None = None  # Position in latent space
    speciation_potential: float = 0.0  # Likelihood of evolutionary branching
    fitness_signal: float | None = None  # Evolutionary fitness metric
    automaton_state: str | None = None  # Büchi/pushdown automaton state

    def to_dict(self) -> dict:
        d = {"omega_class": self.omega_class}
        if self.taxonomic_branch:
            d["branch"] = self.taxonomic_branch
        if self.latent_coordinates:
            d["latent"] = self.latent_coordinates
        if self.speciation_potential > 0:
            d["speciation"] = self.speciation_potential
        if self.fitness_signal is not None:
            d["fitness"] = self.fitness_signal
        if self.automaton_state:
            d["automaton"] = self.automaton_state
        return d


@dataclass
class LineageContext:
    """VGT/HGT evolutionary lineage tracking."""
    dag_id: str | None = None
    lineage_id: str | None = None
    parent_lineage_id: str | None = None
    transfer_type: TransferType = "none"
    generation: int = 0
    mutation_op: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != "none"}


@dataclass
class CMPSignal:
    """Clade Meta-Productivity signal."""
    productivity_delta: float = 0.0  # -1.0 to 1.0
    quality_score: float | None = None
    latency_ratio: float | None = None  # actual/expected
    resource_efficiency: float | None = None  # 0.0 to 1.0
    lineage_health: str = "unknown"  # healthy/degraded/failing

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != "unknown"}


@dataclass
class TopologyContext:
    """Multi-agent coordination topology context."""
    topology: str = "single"  # single/star/peer_debate
    fanout: int = 1
    coordinator: str | None = None
    participants: list[str] = field(default_factory=list)
    coordination_budget_tokens: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d.get("participants"):
            del d["participants"]
        if not d.get("coordinator"):
            del d["coordinator"]
        return d


@dataclass
class SemanticEvent:
    """Rich semantic event with human-readable context and actionable insights.

    Omega-level event schema embodying:
    - Autopoietic reentry (self-referential system loops)
    - Entelexis (potential-to-actuality transitions)
    - Hysteresis (path-dependent memory)
    - Semioticalysis (multi-level meaning decomposition)
    - Omega DNA (fundamental evolutionary code)

    Not merely logs but carriers of the generative principles
    governing emergent multi-agent system evolution.
    """
    # Core fields (standard bus event)
    id: str
    ts: float
    iso: str
    topic: str
    kind: str
    level: str
    actor: str
    data: dict

    # Semantic enrichment (human-readable layer)
    semantic: str  # Human-readable summary
    reasoning: str | None = None  # Why this happened
    actionable: list[str] = field(default_factory=list)  # Suggested actions
    impact: ImpactLevel = "low"

    # Evolutionary context (VGT/HGT lineage)
    lineage: LineageContext | None = None
    cmp: CMPSignal | None = None
    topology: TopologyContext | None = None

    # Omega-level theoretical constructs
    reentry: ReentryMarker | None = None  # Autopoietic self-reference
    entelexis: EntelexisState | None = None  # Potential-actuality tracking
    hysteresis: HysteresisTrace | None = None  # Path-dependent memory
    semioticalysis: SemioticalysisLayer | None = None  # Multi-level meaning
    omega: OmegaContext | None = None  # Fundamental evolutionary context

    # Correlation and causality
    trace_id: str | None = None
    parent_id: str | None = None
    causal_parents: list[str] = field(default_factory=list)  # Multiple causal predecessors

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "ts": self.ts,
            "iso": self.iso,
            "topic": self.topic,
            "kind": self.kind,
            "level": self.level,
            "actor": self.actor,
            "data": self.data,
            "semantic": self.semantic,
            "impact": self.impact,
        }
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.actionable:
            d["actionable"] = self.actionable
        if self.lineage:
            d["lineage"] = self.lineage.to_dict()
        if self.cmp:
            d["cmp"] = self.cmp.to_dict()
        if self.topology:
            d["topology"] = self.topology.to_dict()
        # Omega-level constructs
        if self.reentry:
            d["reentry"] = self.reentry.to_dict()
        if self.entelexis:
            d["entelexis"] = self.entelexis.to_dict()
        if self.hysteresis:
            d["hysteresis"] = self.hysteresis.to_dict()
        if self.semioticalysis:
            d["semioticalysis"] = self.semioticalysis.to_dict()
        if self.omega:
            d["omega"] = self.omega.to_dict()
        # Correlation and causality
        if self.trace_id:
            d["trace_id"] = self.trace_id
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self.causal_parents:
            d["causal_parents"] = self.causal_parents
        return d


# ==============================================================================
# Semantic Templates by Topic
# ==============================================================================

SEMANTIC_TEMPLATES = {
    # Plurichat events
    "plurichat.routing.decision": {
        "semantic_template": "Routing {query_type} query ({depth}) to {provider}",
        "reasoning_template": "Depth={depth} requires {context_mode} context via {lane} lane",
        "impact_fn": lambda d: "high" if d.get("depth") == "deep" else "low",
        "actionable_fn": lambda d: [
            f"Monitor {d.get('provider')} response quality",
            "Cache result if reusable" if d.get("depth") == "narrow" else "Consider decomposition for complex queries",
        ],
    },
    "plurichat.strp.dispatch": {
        "semantic_template": "Dispatching to STRp with {topology} topology (fanout={fanout})",
        "reasoning_template": "Query complexity requires multi-agent coordination",
        "impact_fn": lambda d: "high",
        "actionable_fn": lambda d: [
            f"Monitor {d.get('fanout', 1)} parallel workers",
            "Aggregate results when complete",
            f"Check STRp queue depth",
        ],
    },
    "plurichat.request": {
        "semantic_template": "Processing chat request via {provider}",
        "reasoning_template": "User initiated query routing",
        "impact_fn": lambda d: "medium",
        "actionable_fn": lambda d: ["Await response", "Monitor latency"],
    },
    "plurichat.response": {
        "semantic_template": "Chat response {'completed' if d.get('success') else 'failed'} in {latency_ms:.0f}ms",
        "reasoning_template": "Provider {effective_provider} {'succeeded' if d.get('success') else 'failed'}",
        "impact_fn": lambda d: "low" if d.get("success") else "high",
        "actionable_fn": lambda d: [] if d.get("success") else [
            f"Check provider {d.get('provider')} health",
            "Consider fallback chain",
            "Review error: " + str(d.get("error", "unknown"))[:50],
        ],
    },

    # Dialogos events
    "dialogos.submit": {
        "semantic_template": "LLM request submitted to {providers}",
        "reasoning_template": "Streaming inference requested via dialogos lane",
        "impact_fn": lambda d: "medium",
        "actionable_fn": lambda d: ["Monitor streaming output", "Track token usage"],
    },
    "dialogos.cell.start": {
        "semantic_template": "Starting inference cell for request {req_id}",
        "reasoning_template": "Provider selected, beginning generation",
        "impact_fn": lambda d: "low",
        "actionable_fn": lambda d: ["Monitor for first token", "Watch for timeout"],
    },
    "dialogos.cell.output": {
        "semantic_template": "Streaming output chunk ({content_length} chars)",
        "reasoning_template": "Incremental response from LLM",
        "impact_fn": lambda d: "low",
        "actionable_fn": lambda d: [],
    },
    "dialogos.cell.end": {
        "semantic_template": "Inference cell completed ({status})",
        "reasoning_template": "Generation finished, response ready",
        "impact_fn": lambda d: "medium" if d.get("status") == "success" else "high",
        "actionable_fn": lambda d: [
            "Verify response quality",
            "Update VOR metrics",
        ] if d.get("status") == "success" else [
            "Check provider status",
            "Review error logs",
        ],
    },

    # STRp events
    "strp.request.*": {
        "semantic_template": "STRp work request: {kind} - {goal_summary}",
        "reasoning_template": "Task queued for {topology_hint} topology execution",
        "impact_fn": lambda d: "high" if d.get("parallelizable") else "medium",
        "actionable_fn": lambda d: [
            f"Assign to {d.get('topology_hint', 'single')} worker(s)",
            "Monitor coordination budget",
        ],
    },
    "strp.response": {
        "semantic_template": "STRp task completed by {provider_used}",
        "reasoning_template": "Worker finished processing, results available",
        "impact_fn": lambda d: "medium",
        "actionable_fn": lambda d: [
            "Aggregate with other fanout results" if d.get("fanout", 1) > 1 else "Return to requestor",
            "Update CMP metrics",
        ],
    },
    "strp.worker.claim": {
        "semantic_template": "Worker {actor} claiming task {req_id}",
        "reasoning_template": "Task assignment in progress",
        "impact_fn": lambda d: "low",
        "actionable_fn": lambda d: ["Monitor worker progress", "Track claim time"],
    },

    # Git/Evolution events
    "git.evo.branch": {
        "semantic_template": "Created evolutionary branch: {branch_name}",
        "reasoning_template": "VGT: New lineage spawned for experimental work",
        "impact_fn": lambda d: "medium",
        "actionable_fn": lambda d: [
            "Track lineage generation",
            "Monitor branch health",
            "Consider merge criteria",
        ],
    },
    "git.evo.hgt": {
        "semantic_template": "HGT splice requested from {source_commit}",
        "reasoning_template": "Horizontal gene transfer: Cross-lineage module import",
        "impact_fn": lambda d: "high",
        "actionable_fn": lambda d: [
            "Verify type compatibility",
            "Run guard validation",
            "Check for side effects",
            "Monitor lineage health post-splice",
        ],
    },
    "git.commit": {
        "semantic_template": "Committed: {message_summary}",
        "reasoning_template": "VGT: Standard lineage inheritance",
        "impact_fn": lambda d: "low",
        "actionable_fn": lambda d: ["Update lineage.json", "Emit span metrics"],
    },

    # Lens/Collimator events
    "lens.collimator.plan": {
        "semantic_template": "Route planned: {depth}/{lane}/{topology}",
        "reasoning_template": "Query classified, optimal route determined",
        "impact_fn": lambda d: "medium",
        "actionable_fn": lambda d: [
            f"Execute via {d.get('lane')} lane",
            f"Context mode: {d.get('context_mode')}",
        ],
    },
    "lens.collimator.decision": {
        "semantic_template": "Routing decision: {provider} ({depth})",
        "reasoning_template": "Provider selected based on depth and availability",
        "impact_fn": lambda d: "low",
        "actionable_fn": lambda d: [],
    },

    # Service events
    "service.control/start": {
        "semantic_template": "Starting service: {service_id}",
        "reasoning_template": "Service lifecycle management",
        "impact_fn": lambda d: "medium",
        "actionable_fn": lambda d: ["Monitor startup", "Verify health check"],
    },
    "service.control/stop": {
        "semantic_template": "Stopping service: {service_id}",
        "reasoning_template": "Service lifecycle management",
        "impact_fn": lambda d: "medium",
        "actionable_fn": lambda d: ["Verify clean shutdown", "Check dependent services"],
    },
    "service.status": {
        "semantic_template": "Service {service_id}: {status}",
        "reasoning_template": "Health check result",
        "impact_fn": lambda d: "low" if d.get("status") == "running" else "high",
        "actionable_fn": lambda d: [] if d.get("status") == "running" else [
            "Check service logs",
            "Attempt restart",
        ],
    },

    # Agent events
    "omega.heartbeat": {
        "semantic_template": "Agent {actor} heartbeat ({status})",
        "reasoning_template": "Liveness probe",
        "impact_fn": lambda d: "low",
        "actionable_fn": lambda d: [],
    },
    "infer_sync.checkin": {
        "semantic_template": "Infer-sync: {status} (done={done}, errors={errors})",
        "reasoning_template": "Synchronization checkpoint",
        "impact_fn": lambda d: "medium" if d.get("errors", 0) > 0 else "low",
        "actionable_fn": lambda d: [d.get("next", "Continue")] if d.get("next") else [],
    },
}


# ==============================================================================
# Enrichment Functions
# ==============================================================================

def _get_template(topic: str) -> dict | None:
    """Get semantic template for topic, supporting wildcards."""
    if topic in SEMANTIC_TEMPLATES:
        return SEMANTIC_TEMPLATES[topic]

    # Try wildcard matching
    for pattern, template in SEMANTIC_TEMPLATES.items():
        if pattern.endswith("*") and topic.startswith(pattern[:-1]):
            return template

    return None


def _format_template(template: str, data: dict) -> str:
    """Format template with data, handling missing keys gracefully."""
    try:
        # Add computed fields
        enriched = dict(data)
        if "goal" in enriched:
            enriched["goal_summary"] = enriched["goal"][:50] + "..." if len(enriched.get("goal", "")) > 50 else enriched.get("goal", "")
        if "message" in enriched:
            enriched["message_summary"] = enriched["message"][:50] + "..." if len(enriched.get("message", "")) > 50 else enriched.get("message", "")
        if "content" in enriched:
            enriched["content_length"] = len(enriched.get("content", ""))
        if "providers" in enriched and isinstance(enriched["providers"], list):
            enriched["providers"] = ", ".join(enriched["providers"])

        return template.format(**enriched)
    except (KeyError, ValueError):
        return template


def _load_lineage_context() -> LineageContext:
    """Load current lineage context from .pluribus/lineage.json."""
    lineage_path = Path("/pluribus/.pluribus/lineage.json")
    try:
        if lineage_path.exists():
            data = json.loads(lineage_path.read_text())
            return LineageContext(
                dag_id=data.get("dag_id"),
                lineage_id=data.get("lineage_id"),
                parent_lineage_id=data.get("parent_lineage_id"),
                generation=data.get("generation", 0),
            )
    except Exception:
        pass
    return LineageContext()


def enrich_event(
    event: dict,
    include_lineage: bool = True,
    include_cmp: bool = False,
) -> SemanticEvent:
    """Enrich a basic bus event with semantic context."""
    topic = event.get("topic", "")
    data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

    template = _get_template(topic)

    if template:
        semantic = _format_template(template.get("semantic_template", topic), data)
        reasoning = _format_template(template.get("reasoning_template", ""), data) if template.get("reasoning_template") else None

        impact_fn = template.get("impact_fn")
        impact = impact_fn(data) if impact_fn else "low"

        actionable_fn = template.get("actionable_fn")
        actionable = actionable_fn(data) if actionable_fn else []
    else:
        # Fallback for unknown topics
        semantic = f"{topic}: {json.dumps(data)[:100]}"
        reasoning = None
        impact = "low"
        actionable = []

    lineage = _load_lineage_context() if include_lineage else None

    # Extract topology if present
    topology = None
    if "topology" in data or "fanout" in data:
        topology = TopologyContext(
            topology=data.get("topology", "single"),
            fanout=data.get("fanout", 1),
            coordination_budget_tokens=data.get("coord_budget_tokens", 0),
        )

    return SemanticEvent(
        id=event.get("id") or str(uuid.uuid4()),
        ts=event.get("ts", time.time()),
        iso=event.get("iso") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        topic=topic,
        kind=event.get("kind", "event"),
        level=event.get("level", "info"),
        actor=event.get("actor", "unknown"),
        data=data,
        semantic=semantic,
        reasoning=reasoning,
        actionable=actionable,
        impact=impact,
        lineage=lineage,
        topology=topology,
        trace_id=event.get("trace_id"),
        parent_id=event.get("parent_id"),
    )


def _infer_entelexis(topic: str, data: dict) -> EntelexisState | None:
    """Infer entelexis (potential-actuality) state from event topic and data.

    Maps event lifecycle to Aristotelian potential→actuality transitions:
    - request/submit/plan → potential
    - processing/executing → actualizing
    - response/complete/success → actualized
    - failed/timeout/error → decaying
    """
    phase_markers = {
        "potential": ["request", "submit", "plan", "queue", "pending"],
        "actualizing": ["processing", "executing", "running", "streaming", "cell.output"],
        "actualized": ["response", "complete", "success", "done", "end"],
        "decaying": ["failed", "error", "timeout", "cancelled", "rejected"],
    }

    topic_lower = topic.lower()
    status = str(data.get("status", "")).lower()
    success = data.get("success")

    for phase, markers in phase_markers.items():
        if any(m in topic_lower or m in status for m in markers):
            if phase == "actualized" and success is False:
                phase = "decaying"
            return EntelexisState(
                phase=phase,
                potential_id=data.get("req_id") or data.get("parent_id"),
                actualization_progress={"potential": 0.0, "actualizing": 0.5, "actualized": 1.0, "decaying": 0.0}.get(phase, 0.0),
                form_signature=data.get("query_type") or data.get("kind") or topic.split(".")[-1],
            )

    return None


def _infer_semioticalysis(topic: str, data: dict, semantic: str, reasoning: str | None) -> SemioticalysisLayer:
    """Decompose event meaning across semiotic layers.

    - Syntactic: Structural form (topic hierarchy)
    - Semantic: Denotative meaning (the semantic field)
    - Pragmatic: Action implications (from reasoning/actionable)
    - Metalinguistic: Self-referential meaning (for omega-level events)
    """
    topic_parts = topic.split(".")
    syntactic = f"[{topic_parts[0]}]" + ".".join(topic_parts[1:]) if len(topic_parts) > 1 else topic

    # Motif classification based on topic patterns
    motif_map = {
        "routing": "dispatch_decision",
        "request": "initiation",
        "response": "completion",
        "submit": "delegation",
        "cell": "streaming_fragment",
        "claim": "coordination",
        "evo": "evolution",
        "hgt": "horizontal_transfer",
        "commit": "vertical_transfer",
    }
    motif = None
    for key, motif_name in motif_map.items():
        if key in topic.lower():
            motif = motif_name
            break

    return SemioticalysisLayer(
        syntactic=syntactic,
        semantic=semantic[:100] if semantic else None,
        pragmatic=reasoning[:100] if reasoning else None,
        metalinguistic=f"Event self-documents {topic}" if "omega" in topic else None,
        motif_class=motif,
        decomposition_depth=2 if reasoning else 1,
    )


def _infer_omega_class(topic: str, data: dict) -> OmegaContext:
    """Infer omega-level evolutionary context.

    - prima: Base-level events (individual operations)
    - meta: Events about events (monitoring, aggregation)
    - omega: Fundamental evolutionary events (HGT, speciation, fitness)
    """
    topic_lower = topic.lower()

    # Omega-class determination
    if any(k in topic_lower for k in ["hgt", "evo", "speciation", "fitness", "mutation"]):
        omega_class = "omega"
    elif any(k in topic_lower for k in ["routing", "collimator", "lens", "metric", "monitor"]):
        omega_class = "meta"
    else:
        omega_class = "prima"

    # Taxonomic branch from lineage or topic
    branch = data.get("lineage_id") or data.get("branch") or topic.split(".")[0]

    # Speciation potential (likelihood of evolutionary branching)
    speciation = 0.0
    if data.get("depth") == "deep":
        speciation = 0.3
    if data.get("topology") in ("star", "peer_debate"):
        speciation = 0.5
    if "hgt" in topic_lower:
        speciation = 0.8

    # Automaton state based on event phase
    automaton_states = {
        "request": "q_init",
        "processing": "q_active",
        "response": "q_accept",
        "error": "q_reject",
    }
    automaton = None
    for key, state in automaton_states.items():
        if key in topic_lower:
            automaton = state
            break

    return OmegaContext(
        omega_class=omega_class,
        taxonomic_branch=branch,
        speciation_potential=speciation,
        automaton_state=automaton,
    )


def create_semantic_event(
    topic: str,
    data: dict,
    semantic: str,
    *,
    kind: str = "event",
    level: str = "info",
    actor: str | None = None,
    reasoning: str | None = None,
    actionable: list[str] | None = None,
    impact: ImpactLevel = "low",
    lineage: LineageContext | None = None,
    cmp: CMPSignal | None = None,
    topology: TopologyContext | None = None,
    # Omega-level contexts (auto-inferred if not provided)
    reentry: ReentryMarker | None = None,
    entelexis: EntelexisState | None = None,
    hysteresis: HysteresisTrace | None = None,
    semioticalysis: SemioticalysisLayer | None = None,
    omega: OmegaContext | None = None,
    # Correlation
    trace_id: str | None = None,
    parent_id: str | None = None,
    causal_parents: list[str] | None = None,
    # Auto-inference flags
    auto_enrich_omega: bool = True,
) -> SemanticEvent:
    """Create a new semantic event with full omega-level enrichment.

    If auto_enrich_omega is True (default), automatically infers:
    - entelexis: Potential-actuality phase
    - semioticalysis: Multi-level meaning decomposition
    - omega: Evolutionary classification and context
    """
    # Auto-infer omega-level contexts if not provided and auto-enrichment enabled
    if auto_enrich_omega:
        if entelexis is None:
            entelexis = _infer_entelexis(topic, data)
        if semioticalysis is None:
            semioticalysis = _infer_semioticalysis(topic, data, semantic, reasoning)
        if omega is None:
            omega = _infer_omega_class(topic, data)

    return SemanticEvent(
        id=str(uuid.uuid4()),
        ts=time.time(),
        iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        topic=topic,
        kind=kind,
        level=level,
        actor=actor or os.environ.get("PLURIBUS_ACTOR") or "unknown",
        data=data,
        semantic=semantic,
        reasoning=reasoning,
        actionable=actionable or [],
        impact=impact,
        lineage=lineage or _load_lineage_context(),
        cmp=cmp,
        topology=topology,
        reentry=reentry,
        entelexis=entelexis,
        hysteresis=hysteresis,
        semioticalysis=semioticalysis,
        omega=omega,
        trace_id=trace_id,
        parent_id=parent_id,
        causal_parents=causal_parents or [],
    )


def emit_semantic_event(
    bus_dir: Path,
    event: SemanticEvent,
) -> str:
    """Emit a semantic event to the bus."""
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")

    return event.id


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    """CLI for testing semantic event enrichment."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic event enrichment utilities")
    parser.add_argument("--enrich", help="Enrich a JSON event from stdin or file")
    parser.add_argument("--topic", help="Topic for new event")
    parser.add_argument("--data", help="JSON data for new event")
    parser.add_argument("--semantic", help="Semantic summary")
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus", help="Bus directory")

    args = parser.parse_args()

    if args.enrich:
        if args.enrich == "-":
            import sys
            event = json.load(sys.stdin)
        else:
            event = json.loads(args.enrich)

        enriched = enrich_event(event)
        print(json.dumps(enriched.to_dict(), indent=2))

    elif args.topic and args.semantic:
        data = json.loads(args.data) if args.data else {}
        event = create_semantic_event(
            topic=args.topic,
            data=data,
            semantic=args.semantic,
        )
        event_id = emit_semantic_event(Path(args.bus_dir), event)
        print(f"Emitted: {event_id}")

    else:
        # Demo mode - show enrichment examples
        demo_events = [
            {"topic": "plurichat.routing.decision", "data": {"depth": "deep", "lane": "pbpair", "provider": "codex-cli", "query_type": "code", "context_mode": "full"}},
            {"topic": "strp.request.distill", "data": {"goal": "Research the latest advances in transformer architecture", "topology_hint": "star", "parallelizable": True}},
            {"topic": "git.evo.hgt", "data": {"source_commit": "abc123", "span": {"transfer_type": "HGT"}}},
            {"topic": "dialogos.cell.end", "data": {"req_id": "xyz", "status": "success"}},
        ]

        print("=== Semantic Event Enrichment Demo ===\n")
        for evt in demo_events:
            enriched = enrich_event(evt)
            print(f"Topic: {enriched.topic}")
            print(f"  Semantic: {enriched.semantic}")
            print(f"  Reasoning: {enriched.reasoning}")
            print(f"  Impact: {enriched.impact}")
            print(f"  Actionable: {enriched.actionable}")
            print()


if __name__ == "__main__":
    main()
