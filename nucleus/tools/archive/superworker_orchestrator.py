#!/usr/bin/env python3
"""
SUPERWORKER Orchestrator - Python Implementation

Master coordination for transforming naive web chat workers into context-aware agents.
Mirrors the TypeScript architecture in dashboard/src/lib/vision/superworker-orchestrator.ts

This module provides:
  1. Query depth classification using NLP patterns
  2. Gestalt comprehension aggregation
  3. 5-layer context injection (constitutional → idiolect → persona → context → observable)
  4. Golden-ratio quality scoring
  5. Bus event emission for observability

Golden Ratio Philosophy:
  φ = 1.618033988749895
  Quality tiers: excellent≥1.0, good≥0.618, fair≥0.382, poor≥0.236

Usage:
  from superworker_orchestrator import SuperworkerOrchestrator

  orchestrator = SuperworkerOrchestrator(budget_tier="standard")
  result = await orchestrator.process_query("Explain the authentication flow")
  print(result.augmented_query)

@module tools/superworker_orchestrator
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Any
import os

# =============================================================================
# GOLDEN CONSTANTS
# =============================================================================

PHI = 1.618033988749895
"""Golden ratio constant."""

FIBONACCI = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584)
"""Fibonacci sequence for scaling."""

LUCAS = (2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843)
"""Lucas numbers (related to Fibonacci)."""

SILVER_RATIO = 2.414213562373095
"""Silver ratio δ_S = 1 + √2 for secondary optimization."""

PLASTIC_CONSTANT = 1.324717957244746
"""Plastic constant ρ ≈ 1.3247 for ternary scaling."""

# Token budgets using Fibonacci scaling
TOKEN_BUDGETS = {
    "micro": FIBONACCI[4] * 100,      # 300 tokens
    "mini": FIBONACCI[5] * 100,       # 500 tokens
    "lite": FIBONACCI[7] * 100,       # 1,300 tokens
    "standard": FIBONACCI[9] * 100,   # 3,400 tokens
    "full": FIBONACCI[11] * 100,      # 8,900 tokens
    "deep": FIBONACCI[13] * 100,      # 23,300 tokens
    "ultra": FIBONACCI[15] * 100,     # 61,000 tokens
    "maximum": FIBONACCI[17] * 100,   # 158,400 tokens
}

# Attention window sizes (φ-scaled from base 512)
ATTENTION_WINDOWS = {
    "narrow": round(512 / PHI),           # 316 tokens
    "focused": 512,                        # 512 tokens
    "balanced": round(512 * PHI),         # 828 tokens
    "wide": round(512 * PHI * PHI),       # 1,340 tokens
    "panoramic": round(512 * PHI ** 3),   # 2,167 tokens
    "omniscient": round(512 * PHI ** 4),  # 3,505 tokens
}

# Quality thresholds (inverse φ powers)
QUALITY_THRESHOLDS = {
    "excellent": 1.0,
    "good": 1 / PHI,           # ~0.618
    "fair": 1 / (PHI * PHI),   # ~0.382
    "poor": 1 / (PHI ** 3),    # ~0.236
}


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class InjectionLayer(str, Enum):
    """The 5 injection layers that transform naive chat to agentic operation."""
    CONSTITUTIONAL = "constitutional"  # Safety rails, ethical constraints
    IDIOLECT = "idiolect"              # Domain-specific grammar, terminology
    PERSONA = "persona"                # Role expertise, behavioral traits
    CONTEXT = "context"                # Codebase state, file contents
    OBSERVABLE = "observable"          # Live tools, environment state


class DepthLevel(str, Enum):
    """Depth classification for routing queries."""
    MICRO = "micro"
    NARROW = "narrow"
    STANDARD = "standard"
    DEEP = "deep"
    OMNISCIENT = "omniscient"


class EffectsBudget(str, Enum):
    """Effects budget classification (P/E/L/R/Q gates)."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    NETWORK = "network"
    EXECUTE = "execute"
    UNKNOWN = "unknown"


class CoordinationLane(str, Enum):
    """Coordination lanes for multi-agent communication."""
    DIALOGOS = "dialogos"
    PBPAIR = "pbpair"
    STRP = "strp"


class QualityTier(str, Enum):
    """Quality classification tier."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QualityGates:
    """P/E/L/R/Q gates for quality assessment."""
    provenance: float = 1.0    # Source trustworthiness (0-1)
    effects: float = 0.0       # Side-effect risk level (0-1, lower is safer)
    liveness: float = 1.0      # Real-time validity (0-1)
    recovery: float = 1.0      # Rollback capability (0-1)
    quality: float = 1 / PHI   # Overall output quality (0-PHI)


@dataclass
class GestaltState:
    """Holistic codebase understanding state."""
    architecture_summary: str = ""
    key_files: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    active_concerns: list[str] = field(default_factory=list)
    confidence: float = 0.0
    updated_at: float = field(default_factory=time.time)
    golden_score: float = 0.0


@dataclass
class InjectedContext:
    """Injected context ready for message augmentation."""
    layer: InjectionLayer
    weight: float
    token_count: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryAnalysis:
    """Query analysis result for depth classification."""
    depth: DepthLevel
    confidence: float
    intents: list[str]
    entities: list[str]
    suggested_lanes: list[CoordinationLane]
    estimated_tokens: int
    golden_score: float


@dataclass
class ContextSource:
    """Context source for gestalt aggregation."""
    type: str  # "file", "symbol", "dependency", "history", "vision", "bus"
    content: str
    relevance: float  # 0-1
    freshness: float  # 0-1
    confidence: float  # 0-1
    path: Optional[str] = None


@dataclass
class ProcessResult:
    """Result from processing a query."""
    augmented_query: str
    analysis: QueryAnalysis
    injected_layers: list[InjectionLayer]
    tokens_used: int
    golden_score: float


# =============================================================================
# DEPTH PATTERNS - NLP-based Query Analysis
# =============================================================================

DEPTH_PATTERNS = {
    DepthLevel.MICRO: {
        "keywords": [
            "what is", "define", "explain", "meaning of", "tell me about",
            "quick", "simple", "brief", "short", "summary"
        ],
        "max_length": 50,
        "max_entities": 1,
        "weight": 1 / (PHI ** 3),  # ~0.236
    },
    DepthLevel.NARROW: {
        "keywords": [
            "how do i", "where is", "find", "locate", "show me",
            "fix", "change", "update", "modify", "edit",
            "single", "one", "specific", "particular"
        ],
        "max_length": 150,
        "max_entities": 3,
        "weight": 1 / (PHI ** 2),  # ~0.382
    },
    DepthLevel.STANDARD: {
        "keywords": [
            "implement", "create", "build", "add feature", "write",
            "refactor", "improve", "optimize", "debug", "test",
            "several", "multiple", "various", "different"
        ],
        "max_length": 400,
        "max_entities": 7,
        "weight": 1 / PHI,  # ~0.618
    },
    DepthLevel.DEEP: {
        "keywords": [
            "architecture", "design", "system", "integrate", "migrate",
            "comprehensive", "complete", "full", "entire", "all",
            "analyze", "investigate", "research", "understand",
            "pattern", "strategy", "approach"
        ],
        "max_length": 1000,
        "max_entities": 15,
        "weight": 1.0,
    },
    DepthLevel.OMNISCIENT: {
        "keywords": [
            "everything", "entire codebase", "whole project", "all files",
            "holistic", "gestalt", "big picture", "overview",
            "restructure", "rewrite", "overhaul", "transform",
            "cross-cutting", "pervasive", "fundamental"
        ],
        "max_length": float("inf"),
        "max_entities": float("inf"),
        "weight": PHI,  # ~1.618
    },
}

INTENT_PATTERNS = {
    "read": re.compile(r"\b(show|display|get|fetch|read|view|list|find|search|locate|where)\b", re.I),
    "write": re.compile(r"\b(create|write|add|insert|new|generate|make|build)\b", re.I),
    "modify": re.compile(r"\b(change|update|modify|edit|fix|patch|adjust|alter|refactor)\b", re.I),
    "delete": re.compile(r"\b(remove|delete|drop|clear|clean|purge|eliminate)\b", re.I),
    "analyze": re.compile(r"\b(analyze|examine|investigate|debug|understand|explain|why|how)\b", re.I),
    "execute": re.compile(r"\b(run|execute|start|launch|deploy|trigger|invoke|call)\b", re.I),
    "plan": re.compile(r"\b(plan|design|architect|strategy|approach|consider|think)\b", re.I),
}

ENTITY_PATTERNS = {
    "file": re.compile(r"\b[\w-]+\.(ts|tsx|js|jsx|py|md|json|yaml|yml|toml|sh|css|scss)\b", re.I),
    "path": re.compile(r"/?[\w-]+(?:/[\w-]+)+(?:\.\w+)?"),
    "constant": re.compile(r"\b[A-Z][A-Z_0-9]+\b"),
    "camel_case": re.compile(r"\b[a-z]+(?:[A-Z][a-z]+)+\b"),
}

# Layer templates for injection
LAYER_TEMPLATES = {
    InjectionLayer.CONSTITUTIONAL: {
        "prefix": "<constitutional_layer>\n",
        "suffix": "\n</constitutional_layer>",
        "weight": PHI ** 2,
        "max_tokens": FIBONACCI[8] * 10,  # 340 tokens
    },
    InjectionLayer.IDIOLECT: {
        "prefix": '<idiolect_layer domain="pluribus">\n',
        "suffix": "\n</idiolect_layer>",
        "weight": PHI,
        "max_tokens": FIBONACCI[10] * 10,  # 550 tokens
    },
    InjectionLayer.PERSONA: {
        "prefix": "<persona_layer>\n",
        "suffix": "\n</persona_layer>",
        "weight": 1.0,
        "max_tokens": FIBONACCI[11] * 10,  # 890 tokens
    },
    InjectionLayer.CONTEXT: {
        "prefix": "<context_layer>\n",
        "suffix": "\n</context_layer>",
        "weight": 1 / PHI,
        "max_tokens": FIBONACCI[13] * 10,  # 2,330 tokens
    },
    InjectionLayer.OBSERVABLE: {
        "prefix": "<observable_layer>\n",
        "suffix": "\n</observable_layer>",
        "weight": 1 / (PHI * PHI),
        "max_tokens": FIBONACCI[12] * 10,  # 1,440 tokens
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_golden_score(
    resolution: float,
    frame_rate: float,
    stability: float,
    latency: Optional[float] = None
) -> float:
    """
    Calculate golden score from quality metrics.
    Uses geometric mean with φ-weighted factors.
    """
    weights = {
        "resolution": PHI,           # Most important
        "frame_rate": 1.0,
        "stability": 1 / PHI,        # Least variable
        "latency": 1 / (PHI * PHI),  # Often unavailable
    }

    factors = [
        math.pow(resolution, weights["resolution"]),
        math.pow(frame_rate, weights["frame_rate"]),
        math.pow(stability, weights["stability"]),
    ]

    if latency is not None:
        factors.append(math.pow(latency, weights["latency"]))

    # Geometric mean
    product = 1.0
    for f in factors:
        product *= f

    return math.pow(product, 1 / len(factors))


def classify_quality(score: float) -> QualityTier:
    """Classify a golden score into a quality tier."""
    if score >= QUALITY_THRESHOLDS["excellent"]:
        return QualityTier.EXCELLENT
    if score >= QUALITY_THRESHOLDS["good"]:
        return QualityTier.GOOD
    if score >= QUALITY_THRESHOLDS["fair"]:
        return QualityTier.FAIR
    if score >= QUALITY_THRESHOLDS["poor"]:
        return QualityTier.POOR
    return QualityTier.CRITICAL


def analyze_query(query: str) -> QueryAnalysis:
    """
    Analyze a query to determine depth, intents, and entities.
    Uses golden-ratio weighted scoring for classification confidence.
    """
    normalized = query.lower().strip()
    query_length = len(query)

    # Extract entities
    entities: list[str] = []
    for pattern in ENTITY_PATTERNS.values():
        matches = pattern.findall(query)
        entities.extend(matches)
    unique_entities = list(set(entities))

    # Detect intents
    intents: list[str] = []
    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(normalized):
            intents.append(intent)

    # Score each depth level
    depth_scores: dict[DepthLevel, float] = {level: 0.0 for level in DepthLevel}

    for level, config in DEPTH_PATTERNS.items():
        score = 0.0

        # Keyword matching (φ-weighted)
        keyword_matches = sum(1 for kw in config["keywords"] if kw in normalized)
        score += keyword_matches * config["weight"] * PHI

        # Length heuristic
        if query_length <= config["max_length"]:
            score += config["weight"]

        # Entity count heuristic
        if len(unique_entities) <= config["max_entities"]:
            score += config["weight"] / PHI

        depth_scores[level] = score

    # Find best depth match
    best_depth = max(depth_scores, key=lambda k: depth_scores[k])
    best_score = depth_scores[best_depth]

    # Calculate confidence
    total_score = sum(depth_scores.values())
    confidence = best_score / total_score if total_score > 0 else 1 / (PHI * PHI)

    # Suggest coordination lanes
    suggested_lanes: list[CoordinationLane] = []
    if best_depth in (DepthLevel.MICRO, DepthLevel.NARROW):
        suggested_lanes.append(CoordinationLane.DIALOGOS)
    if best_depth in (DepthLevel.STANDARD, DepthLevel.DEEP):
        suggested_lanes.append(CoordinationLane.PBPAIR)
    if best_depth in (DepthLevel.DEEP, DepthLevel.OMNISCIENT):
        suggested_lanes.append(CoordinationLane.STRP)
    if "execute" in intents or "write" in intents:
        if CoordinationLane.PBPAIR not in suggested_lanes:
            suggested_lanes.append(CoordinationLane.PBPAIR)

    # Estimate token requirement
    budget_map = {
        DepthLevel.MICRO: "micro",
        DepthLevel.NARROW: "lite",
        DepthLevel.STANDARD: "standard",
        DepthLevel.DEEP: "deep",
        DepthLevel.OMNISCIENT: "ultra",
    }
    base_tokens = TOKEN_BUDGETS[budget_map[best_depth]]
    entity_multiplier = 1 + (len(unique_entities) * 0.1)
    intent_multiplier = 1 + (len(intents) * 0.05)
    estimated_tokens = round(base_tokens * entity_multiplier * intent_multiplier)

    # Calculate golden score
    golden_score = calculate_golden_score(
        resolution=confidence,
        frame_rate=min(len(intents) / 3, 1),
        stability=min(len(unique_entities) / 10, 1),
        latency=0.8 if query_length > 20 else 0.5,
    )

    return QueryAnalysis(
        depth=best_depth,
        confidence=confidence,
        intents=intents,
        entities=unique_entities,
        suggested_lanes=suggested_lanes,
        estimated_tokens=estimated_tokens,
        golden_score=golden_score,
    )


def aggregate_gestalt(sources: list[ContextSource], max_tokens: int) -> GestaltState:
    """
    Aggregate multiple context sources into gestalt understanding.
    Uses φ-weighted importance scoring for prioritization.
    """
    type_weights = {
        "vision": PHI,
        "file": 1.0,
        "symbol": 1 / PHI,
        "history": 1 / (PHI * PHI),
        "dependency": 1 / (PHI * PHI),
        "bus": 1 / (PHI ** 3),
    }

    # Score each source
    scored_sources = []
    for source in sources:
        type_weight = type_weights.get(source.type, 0.5)

        # Geometric mean with φ-weighting
        score = math.pow(
            math.pow(source.relevance, PHI) *
            math.pow(source.freshness, 1.0) *
            math.pow(source.confidence, 1 / PHI) *
            type_weight,
            1 / 3
        )
        scored_sources.append((source, score))

    # Sort by score descending
    scored_sources.sort(key=lambda x: x[1], reverse=True)

    # Aggregate until token budget exhausted
    key_files: dict[str, str] = {}
    dependencies: dict[str, list[str]] = {}
    active_concerns: list[str] = []
    total_tokens = 0
    total_score = 0.0
    source_count = 0

    for source, score in scored_sources:
        # Estimate tokens (~4 chars per token)
        estimated_tokens = len(source.content) // 4

        if total_tokens + estimated_tokens > max_tokens:
            # Apply diminishing returns with silver ratio
            if total_tokens + estimated_tokens / SILVER_RATIO > max_tokens:
                break

        # Incorporate source
        if source.type == "file" and source.path:
            key_files[source.path] = source.content[:500]
        elif source.type == "dependency" and source.path:
            deps = [d.strip() for d in source.content.split(",")]
            dependencies[source.path] = deps
        elif source.type == "vision":
            active_concerns.append(f"[VISION] {source.content[:200]}")

        total_tokens += estimated_tokens
        total_score += score
        source_count += 1

    # Generate architecture summary
    top_sources = scored_sources[:5]
    architecture_summary = " → ".join(
        s[0].path or s[0].type for s in top_sources
    )

    # Calculate golden score
    avg_score = total_score / source_count if source_count > 0 else 0
    coverage_score = min(source_count / 10, 1)
    golden_score = calculate_golden_score(
        resolution=avg_score,
        frame_rate=coverage_score,
        stability=0.9 if key_files else 0.5,
    )

    return GestaltState(
        architecture_summary=architecture_summary,
        key_files=key_files,
        dependencies=dependencies,
        active_concerns=active_concerns,
        confidence=avg_score,
        updated_at=time.time(),
        golden_score=golden_score,
    )


def build_layer_injection(
    layer: InjectionLayer,
    contents: list[str],
    metadata: Optional[dict[str, Any]] = None
) -> InjectedContext:
    """Build injection content for a specific layer."""
    template = LAYER_TEMPLATES[layer]
    combined = "\n\n".join(contents)
    max_chars = template["max_tokens"] * 4  # ~4 chars/token
    truncated = combined[:max_chars]

    content = f"{template['prefix']}{truncated}{template['suffix']}"
    token_count = len(content) // 4

    return InjectedContext(
        layer=layer,
        weight=template["weight"],
        token_count=token_count,
        content=content,
        metadata={
            **(metadata or {}),
            "original_length": len(combined),
            "truncated": len(combined) > len(truncated),
            "layer_max_tokens": template["max_tokens"],
        },
    )


def assemble_injection(
    contexts: list[InjectedContext],
    max_tokens: int
) -> tuple[str, int, list[InjectionLayer]]:
    """
    Assemble all layers into final injection payload.
    Respects token budget with φ-weighted prioritization.
    """
    # Sort by weight (highest first)
    sorted_contexts = sorted(contexts, key=lambda c: c.weight, reverse=True)

    included: list[InjectedContext] = []
    total_tokens = 0

    for ctx in sorted_contexts:
        if total_tokens + ctx.token_count <= max_tokens:
            included.append(ctx)
            total_tokens += ctx.token_count
        else:
            # Try to fit partial content
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 100:
                ratio = remaining_tokens / ctx.token_count
                truncated_content = ctx.content[:int(len(ctx.content) * ratio)]
                included.append(InjectedContext(
                    layer=ctx.layer,
                    weight=ctx.weight,
                    token_count=remaining_tokens,
                    content=truncated_content + "\n[TRUNCATED]",
                    metadata=ctx.metadata,
                ))
                total_tokens = max_tokens
            break

    # Reorder by layer hierarchy
    layer_order = [
        InjectionLayer.CONSTITUTIONAL,
        InjectionLayer.IDIOLECT,
        InjectionLayer.PERSONA,
        InjectionLayer.CONTEXT,
        InjectionLayer.OBSERVABLE,
    ]
    included.sort(key=lambda c: layer_order.index(c.layer))

    content = "\n\n".join(c.content for c in included)
    layers_included = [c.layer for c in included]

    return content, total_tokens, layers_included


# =============================================================================
# SUPERWORKER ORCHESTRATOR CLASS
# =============================================================================

class SuperworkerOrchestrator:
    """
    Master orchestrator for SUPERWORKER operations.
    Coordinates all subsystems with golden-ratio optimization.
    """

    def __init__(
        self,
        budget_tier: str = "standard",
        attention_window: str = "balanced",
        enabled_layers: Optional[list[InjectionLayer]] = None,
        min_golden_score: float = 1 / (PHI * PHI),  # 0.382 (fair)
        emit_bus_events: bool = True,
        persona_id: Optional[str] = None,
    ):
        self.budget_tier = budget_tier
        self.attention_window = attention_window
        self.enabled_layers = enabled_layers or [
            InjectionLayer.CONSTITUTIONAL,
            InjectionLayer.PERSONA,
            InjectionLayer.CONTEXT,
        ]
        self.min_golden_score = min_golden_score
        self.emit_bus_events = emit_bus_events
        self.persona_id = persona_id

        self._initialize_state()

    def _initialize_state(self):
        """Initialize orchestrator state."""
        import random
        import string

        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        self.session_id = f"sw_{int(time.time())}_{suffix}"
        self.depth_level = DepthLevel.STANDARD
        self.lanes: list[CoordinationLane] = [CoordinationLane.DIALOGOS]
        self.effects_budget: list[EffectsBudget] = [EffectsBudget.NONE, EffectsBudget.READ]
        self.quality_gates = QualityGates()
        self.gestalt = GestaltState()
        self.injected_contexts: dict[InjectionLayer, list[InjectedContext]] = {}
        self.tokens_consumed = 0
        self.token_budget = TOKEN_BUDGETS[self.budget_tier]
        self.golden_score = 1 / PHI
        self.created_at = time.time()
        self.last_activity_at = time.time()
        self.vision_context: Optional[dict] = None

    async def process_query(
        self,
        query: str,
        additional_context: Optional[list[str]] = None
    ) -> ProcessResult:
        """Process a query through the full SUPERWORKER pipeline."""
        # Step 1: Analyze query
        analysis = analyze_query(query)
        self.depth_level = analysis.depth
        self.lanes = analysis.suggested_lanes

        # Step 2: Quality gate check
        if analysis.golden_score < self.min_golden_score:
            self._emit_event("quality.warning", {
                "score": analysis.golden_score,
                "threshold": self.min_golden_score,
                "query": query[:100],
            })

        # Step 3: Build context sources
        sources: list[ContextSource] = []

        if additional_context:
            for ctx in additional_context:
                sources.append(ContextSource(
                    type="file",
                    content=ctx,
                    relevance=0.8,
                    freshness=1.0,
                    confidence=0.9,
                ))

        if self.vision_context:
            sources.append(ContextSource(
                type="vision",
                content=f"[Screen capture: {self.vision_context.get('width', 0)}x{self.vision_context.get('height', 0)}]",
                relevance=0.9,
                freshness=1.0,
                confidence=self.vision_context.get("golden_score", 0.5),
            ))

        # Step 4: Aggregate gestalt
        gestalt_budget = round(self.token_budget * (1 / PHI))
        self.gestalt = aggregate_gestalt(sources, gestalt_budget)

        # Step 5: Build injection layers
        injections: list[InjectedContext] = []

        for layer in self.enabled_layers:
            layer_content = self._build_layer_content(layer, analysis, query)
            if layer_content:
                injections.append(build_layer_injection(
                    layer,
                    layer_content,
                    {"query": query[:50], "depth": analysis.depth.value},
                ))

        # Step 6: Assemble injection
        remaining_budget = self.token_budget - self.tokens_consumed
        content, tokens_used, layers_included = assemble_injection(injections, remaining_budget)

        # Step 7: Augment query
        augmented_query = f"{content}\n\n---\n\n{query}" if content else query

        # Update state
        self.tokens_consumed += tokens_used
        self.last_activity_at = time.time()
        self.golden_score = calculate_golden_score(
            resolution=analysis.confidence,
            frame_rate=len(layers_included) / 5,
            stability=self.gestalt.golden_score,
        )

        # Emit completion event
        self._emit_event("query.processed", {
            "session_id": self.session_id,
            "depth": analysis.depth.value,
            "layers_included": [l.value for l in layers_included],
            "tokens_used": tokens_used,
            "golden_score": self.golden_score,
        })

        return ProcessResult(
            augmented_query=augmented_query,
            analysis=analysis,
            injected_layers=layers_included,
            tokens_used=tokens_used,
            golden_score=self.golden_score,
        )

    def _build_layer_content(
        self,
        layer: InjectionLayer,
        analysis: QueryAnalysis,
        query: str
    ) -> list[str]:
        """Build content for a specific injection layer."""
        contents: list[str] = []

        if layer == InjectionLayer.CONSTITUTIONAL:
            contents.append(f"You are operating as a SUPERWORKER with depth level: {analysis.depth.value}")
            contents.append(f"Quality threshold: {self.min_golden_score:.3f} (golden ratio scaled)")
            contents.append(f"Effects budget: {', '.join(e.value for e in self.effects_budget)}")
            if "execute" in analysis.intents or "write" in analysis.intents:
                contents.append("CAUTION: Write/execute intents detected. Confirm before side effects.")

        elif layer == InjectionLayer.IDIOLECT:
            contents.append("Domain: Pluribus multi-agent orchestration system")
            contents.append("Key terms: bus events (NDJSON), golden scoring (φ), Fibonacci budgets")
            contents.append("Patterns: P/E/L/R/Q gates, 3-lane coordination, 5-layer injection")

        elif layer == InjectionLayer.PERSONA:
            if self.persona_id:
                contents.append(f"Active persona: {self.persona_id}")
            contents.append(f"Coordination lanes: {', '.join(l.value for l in self.lanes)}")
            approach = "comprehensive analysis" if analysis.depth == DepthLevel.DEEP else "focused response"
            contents.append(f"Suggested approach: {approach}")

        elif layer == InjectionLayer.CONTEXT:
            if self.gestalt.architecture_summary:
                contents.append(f"Architecture: {self.gestalt.architecture_summary}")
            if self.gestalt.key_files:
                files = list(self.gestalt.key_files.keys())[:5]
                contents.append(f"Key files: {', '.join(files)}")
            if self.gestalt.active_concerns:
                contents.append(f"Active concerns: {'; '.join(self.gestalt.active_concerns)}")

        elif layer == InjectionLayer.OBSERVABLE:
            contents.append(f"Session: {self.session_id}")
            contents.append(f"Tokens consumed: {self.tokens_consumed}/{self.token_budget}")
            contents.append(f"Golden score: {self.golden_score:.3f}")
            if self.vision_context:
                contents.append(f"Vision active: {self.vision_context.get('provider', 'pending')}")

        return contents

    def set_vision_context(
        self,
        width: int,
        height: int,
        golden_score: float,
        provider: Optional[str] = None
    ):
        """Inject vision context from screen capture."""
        self.vision_context = {
            "width": width,
            "height": height,
            "golden_score": golden_score,
            "provider": provider,
        }
        self.last_activity_at = time.time()

        self._emit_event("vision.injected", {
            "session_id": self.session_id,
            "frame_size": f"{width}x{height}",
            "provider": provider,
            "golden_score": golden_score,
        })

    def get_remaining_budget(self) -> int:
        """Get remaining token budget."""
        return self.token_budget - self.tokens_consumed

    def reset(self):
        """Reset orchestrator for new session."""
        old_session_id = self.session_id
        self._initialize_state()

        self._emit_event("session.reset", {
            "old_session_id": old_session_id,
            "new_session_id": self.session_id,
        })

    def _emit_event(self, topic: str, data: dict[str, Any]):
        """Emit event to bus (if configured)."""
        if not self.emit_bus_events:
            return

        event = {
            "timestamp": int(time.time() * 1000),
            "topic": f"superworker.{topic}",
            "kind": "metric",
            "level": "debug",
            "data": {**data, "phi": PHI},
        }

        # Try to write to bus
        bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
        try:
            bus_dir.mkdir(parents=True, exist_ok=True)
            with open(bus_dir / "events.ndjson", "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass  # Silently ignore bus errors


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "PHI",
    "FIBONACCI",
    "LUCAS",
    "SILVER_RATIO",
    "PLASTIC_CONSTANT",
    "TOKEN_BUDGETS",
    "ATTENTION_WINDOWS",
    "QUALITY_THRESHOLDS",
    # Enums
    "InjectionLayer",
    "DepthLevel",
    "EffectsBudget",
    "CoordinationLane",
    "QualityTier",
    # Data classes
    "QualityGates",
    "GestaltState",
    "InjectedContext",
    "QueryAnalysis",
    "ContextSource",
    "ProcessResult",
    # Functions
    "calculate_golden_score",
    "classify_quality",
    "analyze_query",
    "aggregate_gestalt",
    "build_layer_injection",
    "assemble_injection",
    # Main class
    "SuperworkerOrchestrator",
]


if __name__ == "__main__":
    # Demo usage
    async def main():
        orchestrator = SuperworkerOrchestrator(
            budget_tier="standard",
            persona_id="superworker.context_injector",
        )

        query = "How do I implement a new VLM provider in the vision module?"
        result = await orchestrator.process_query(query)

        print(f"Depth: {result.analysis.depth.value}")
        print(f"Confidence: {result.analysis.confidence:.3f}")
        print(f"Intents: {result.analysis.intents}")
        print(f"Entities: {result.analysis.entities}")
        print(f"Golden Score: {result.golden_score:.3f}")
        print(f"Tokens Used: {result.tokens_used}")
        print(f"Layers: {[l.value for l in result.injected_layers]}")
        print(f"\n--- Augmented Query ---\n{result.augmented_query[:500]}...")

    asyncio.run(main())
