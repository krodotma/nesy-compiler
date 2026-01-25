#!/usr/bin/env python3
"""
LENS/LASER Synthesizer - Multi-Model Entropic Response Synthesis

Implements the LENS/LASER pipeline from lens_laser_synthesizer_v1.md:

LENS (LLM Entropic Natural Superposition):
  - Parallel generation across multiple models
  - Entropy profiling of each response
  - Claim extraction and claim graph construction

LASER (Language-Augmented Superpositional Effective Retrieval):
  - Claim pooling and agreement analysis
  - Evidence retrieval and verification
  - Constrained optimization for final synthesis

The physics metaphor:
  - LENS = Superposition (multiple waveforms coexist)
  - LASER = Measurement/Collapse (verification selects survivors)
  - Entropy interference patterns optimize the synthesis

Author: claude (DKIN v29)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import entropy profiler
try:
    from entropy_profiler import (
        EntropyVector, profile_entropy, compute_utility,
        _tokenize, _extract_claims, _compute_jaccard_similarity
    )
except ImportError:
    # Fallback stubs if profiler not available
    EntropyVector = None
    profile_entropy = None


# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_MODELS = ["claude", "gemini", "gpt"]
DEFAULT_SAMPLES_PER_MODEL = 2
DEFAULT_BUDGETS = {
    "conjecture_max": 0.05,
    "missing_max": 0.10,
    "cogload_max": 0.30,
    "goal_drift_max": 0.10,
    "info_min": 0.50,
}

PROVIDER_ENDPOINTS = {
    "claude": "anthropic",
    "gemini": "google",
    "gpt": "openai",
    "grok": "xai",
    "local": "ollama",
}


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class Claim:
    """Atomic claim extracted from response."""
    id: str
    text: str
    source_model: str
    source_sample: int
    confidence: float = 0.8
    claim_type: str = "factual"  # factual | procedural | evaluative | definitional

    @staticmethod
    def compute_id(text: str) -> str:
        """Compute deterministic claim ID from normalized text."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class ClaimPoolEntry:
    """Claim with aggregated cross-model evidence."""
    claim: Claim
    sources: list[tuple[str, int]]  # (model_id, sample_k)
    agreement_ratio: float = 0.0
    h_alea: float = 0.0  # Aleatoric uncertainty for this claim
    support_score: float = 0.0  # Evidence support
    uncertainty_type: str = "pending"  # verified | aleatoric | epistemic | normative | underdetermined


@dataclass
class ModelResponse:
    """Response from a single model."""
    model_id: str
    sample_k: int
    response: str
    entropy: EntropyVector | None = None
    claims: list[Claim] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class LENSOutput:
    """Output from LENS (superposition) stage."""
    req_id: str
    prompt: str
    responses: list[ModelResponse]
    claim_pool: dict[str, ClaimPoolEntry]  # claim_id -> entry
    total_latency_ms: float = 0.0


@dataclass
class LASEROutput:
    """Output from LASER (verification/synthesis) stage."""
    req_id: str
    verified_claims: list[ClaimPoolEntry]
    selected_claims: list[ClaimPoolEntry]
    synthesized_text: str
    final_entropy: EntropyVector | None = None
    utility: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesizerConfig:
    """Configuration for the LENS/LASER synthesizer."""
    models: list[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    samples_per_model: int = DEFAULT_SAMPLES_PER_MODEL
    budgets: dict[str, float] = field(default_factory=lambda: DEFAULT_BUDGETS.copy())
    task_schema: dict[str, Any] | None = None
    max_latency_ms: float = 30000.0
    min_agreement: float = 0.4  # Minimum agreement ratio to keep claim
    parallel: bool = True
    emit_bus: bool = True
    bus_dir: str = "/pluribus/.pluribus/bus"
    actor: str = "lens-laser-synth"
    # Dual-input configuration
    interference_mode: str = "strict"  # strict | lenient | explore


# ==============================================================================
# Dual-Input Architecture: Repo World Model
# ==============================================================================

@dataclass
class TypeConstraint:
    """Type constraint extracted from source file."""
    source_file: str
    line: int
    constraint_type: str  # parameter | return | property | generic
    expected_type: str
    nullable: bool = False
    required: bool = True


@dataclass
class Invariant:
    """Invariant that must hold in the codebase."""
    id: str
    description: str
    assertion: str  # LTL formula or code assertion
    scope: list[str] = field(default_factory=list)  # Affected files/functions
    verified_by: list[str] = field(default_factory=list)  # Test files


@dataclass
class TestExpectation:
    """Expected behavior from test file analysis."""
    test_file: str
    test_name: str
    target_function: str
    expected_behavior: str
    assertion_type: str  # equality | throws | returns | matches


@dataclass
class IsomorphicBoundary:
    """Boundary between client and server in isomorphic code."""
    shared_modules: list[str] = field(default_factory=list)
    server_only: list[str] = field(default_factory=list)
    client_only: list[str] = field(default_factory=list)


@dataclass
class RepoWorldModel:
    """
    World Model constructed from repository analysis.

    This represents the deterministic layer in the LENS/LASER dual-input
    architecture. The world model provides grounding constraints that
    generated claims must satisfy.

    Per Princeton Web-World-Models architecture:
    - Deterministic layer: Types, tests, invariants from code
    - Generative layer: LLM claims from prompt
    - Interface: Interference zone where claims meet constraints
    """
    repo_root: str
    file_tree: list[str] = field(default_factory=list)
    type_constraints: list[TypeConstraint] = field(default_factory=list)
    invariants: list[Invariant] = field(default_factory=list)
    test_expectations: list[TestExpectation] = field(default_factory=list)
    isomorphic_boundary: IsomorphicBoundary | None = None
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    external_deps: list[str] = field(default_factory=list)

    # Cache for world model hash (for cache invalidation)
    world_model_hash: str = ""

    @staticmethod
    def from_repo(repo_root: str, scope: list[str] | None = None, exclude: list[str] | None = None) -> "RepoWorldModel":
        """Construct world model from repository."""
        import glob
        import ast as python_ast

        wm = RepoWorldModel(repo_root=repo_root)
        root_path = Path(repo_root)

        if not root_path.exists():
            return wm

        # Phase 1: Scan file tree
        scope_patterns = scope or ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"]
        exclude_patterns = exclude or ["**/node_modules/**", "**/.git/**", "**/dist/**", "**/__pycache__/**"]

        for pattern in scope_patterns:
            for file_path in root_path.glob(pattern):
                rel_path = str(file_path.relative_to(root_path))
                # Check exclude patterns
                excluded = False
                for exc in exclude_patterns:
                    if Path(rel_path).match(exc.replace("**/", "")):
                        excluded = True
                        break
                if not excluded:
                    wm.file_tree.append(rel_path)

        # Phase 2: Extract type constraints (Python files)
        for rel_path in wm.file_tree:
            if rel_path.endswith(".py"):
                file_path = root_path / rel_path
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = python_ast.parse(source)
                    wm._extract_python_types(tree, rel_path)
                except Exception:
                    pass  # Skip unparseable files

        # Phase 3: Detect isomorphic patterns (TypeScript/JavaScript)
        wm._detect_isomorphic_boundary()

        # Phase 4: Extract test expectations
        wm._extract_test_expectations()

        # Phase 5: Build dependency graph
        wm._build_dependency_graph()

        # Compute world model hash
        import hashlib
        content = json.dumps({
            "files": sorted(wm.file_tree),
            "types": len(wm.type_constraints),
            "invariants": len(wm.invariants),
            "tests": len(wm.test_expectations),
        }, sort_keys=True)
        wm.world_model_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        return wm

    def _extract_python_types(self, tree, rel_path: str) -> None:
        """Extract type annotations from Python AST."""
        import ast as python_ast

        for node in python_ast.walk(tree):
            if isinstance(node, python_ast.FunctionDef):
                # Return type
                if node.returns:
                    self.type_constraints.append(TypeConstraint(
                        source_file=rel_path,
                        line=node.lineno,
                        constraint_type="return",
                        expected_type=python_ast.unparse(node.returns) if hasattr(python_ast, 'unparse') else str(node.returns),
                        nullable="None" in str(node.returns) or "Optional" in str(node.returns)
                    ))

                # Parameter types
                for arg in node.args.args:
                    if arg.annotation:
                        self.type_constraints.append(TypeConstraint(
                            source_file=rel_path,
                            line=node.lineno,
                            constraint_type="parameter",
                            expected_type=python_ast.unparse(arg.annotation) if hasattr(python_ast, 'unparse') else str(arg.annotation),
                            nullable="None" in str(arg.annotation) or "Optional" in str(arg.annotation)
                        ))

    def _detect_isomorphic_boundary(self) -> None:
        """Detect isomorphic code patterns (Qwik, Next.js, Remix)."""
        shared = []
        server = []
        client = []

        for rel_path in self.file_tree:
            if "server" in rel_path.lower() or "api/" in rel_path:
                server.append(rel_path)
            elif "client" in rel_path.lower() or "components/" in rel_path:
                client.append(rel_path)
            elif "lib/" in rel_path or "utils/" in rel_path or "shared/" in rel_path:
                shared.append(rel_path)

        if shared or (server and client):
            self.isomorphic_boundary = IsomorphicBoundary(
                shared_modules=shared,
                server_only=server,
                client_only=client
            )

    def _extract_test_expectations(self) -> None:
        """Extract test expectations from test files."""
        for rel_path in self.file_tree:
            if "test" in rel_path.lower() or rel_path.endswith("_test.py") or rel_path.endswith(".spec.ts"):
                # Simple pattern matching for test assertions
                file_path = Path(self.repo_root) / rel_path
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Extract function being tested (heuristic)
                    # Look for patterns like test_functionname or describe("functionname"
                    import re
                    test_patterns = re.findall(r'(?:def test_|it\([\'"]|test\([\'"])(\w+)', content)
                    for test_name in test_patterns[:10]:  # Limit per file
                        self.test_expectations.append(TestExpectation(
                            test_file=rel_path,
                            test_name=test_name,
                            target_function=test_name.replace("test_", "").replace("_", ""),
                            expected_behavior=f"Behavior validated by {test_name}",
                            assertion_type="unknown"
                        ))
                except Exception:
                    pass

    def _build_dependency_graph(self) -> None:
        """Build import/require dependency graph."""
        import re

        for rel_path in self.file_tree:
            file_path = Path(self.repo_root) / rel_path
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                deps = []
                # Python imports
                for match in re.finditer(r'^(?:from|import)\s+([\w.]+)', content, re.MULTILINE):
                    deps.append(match.group(1))
                # JS/TS imports
                for match in re.finditer(r'(?:import|require)\s*\(?[\'"]([^"\']+)[\'"]', content):
                    deps.append(match.group(1))

                if deps:
                    self.dependency_graph[rel_path] = deps[:20]  # Limit per file
            except Exception:
                pass

    def check_claim_against_constraints(self, claim: "Claim") -> dict[str, Any]:
        """
        Check if a claim satisfies the world model constraints.

        Returns a dict with:
        - valid: bool
        - violations: list of constraint violations
        - warnings: list of potential issues
        """
        result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }

        # Check type constraints if claim mentions types
        for tc in self.type_constraints:
            if tc.expected_type.lower() in claim.text.lower():
                # Claim mentions a type - check for consistency
                if tc.nullable and "not null" in claim.text.lower():
                    result["warnings"].append(f"Claim may conflict with nullable type at {tc.source_file}:{tc.line}")

        # Check test expectations
        for te in self.test_expectations:
            if te.target_function.lower() in claim.text.lower():
                # Claim mentions a tested function
                result["warnings"].append(f"Claim involves tested function {te.target_function}, verify with {te.test_file}")

        # Check invariants
        for inv in self.invariants:
            if any(scope_item in claim.text for scope_item in inv.scope):
                result["warnings"].append(f"Claim touches invariant scope: {inv.description}")

        return result


@dataclass
class InterferenceZone:
    """Where prompt-generated claims meet repo constraints."""
    claims: list[Claim] = field(default_factory=list)
    interference_type: str = "unknown"  # constructive | destructive | underdetermined
    resultant_amplitude: float = 0.0
    violations: list[dict] = field(default_factory=list)


@dataclass
class ExplorationNode:
    """Node in the tree automata exploration."""
    id: str
    depth: int
    claim: Claim | None = None
    children: list["ExplorationNode"] = field(default_factory=list)
    sources: list[tuple[str, int]] = field(default_factory=list)  # (model_id, sample_k)
    transition_probability: float = 1.0
    acceptance_state: bool = False


@dataclass
class ExplorationTree:
    """Tree automata structure for prompt exploration."""
    root: ExplorationNode
    frontier: list[ExplorationNode] = field(default_factory=list)
    interference_zones: list[InterferenceZone] = field(default_factory=list)
    reentrant_cycles: list[list[str]] = field(default_factory=list)  # Cycles by node ID


# ==============================================================================
# Bus Emission
# ==============================================================================

def emit_bus(config: SynthesizerConfig, topic: str, kind: str, level: str, data: dict) -> None:
    """Emit event to Pluribus bus."""
    if not config.emit_bus:
        return

    bus_path = Path(config.bus_dir) / "events.ndjson"
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": config.actor,
        "data": data,
    }

    try:
        bus_path.parent.mkdir(parents=True, exist_ok=True)
        with bus_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass  # Non-fatal


# ==============================================================================
# Model Providers (Pluggable)
# ==============================================================================

class ModelProvider:
    """Base class for model providers."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response from model."""
        raise NotImplementedError

    def generate_sync(self, prompt: str, temperature: float = 0.7) -> str:
        """Synchronous generation wrapper."""
        return asyncio.run(self.generate(prompt, temperature))


class MockProvider(ModelProvider):
    """Mock provider for testing."""

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Simulate latency
        await asyncio.sleep(0.1 + temperature * 0.2)

        # Generate mock response based on prompt
        words = prompt.lower().split()[:10]
        mock_response = f"Based on the query about {' '.join(words[:3])}, "
        mock_response += f"the {self.model_id} model provides this analysis: "
        mock_response += f"The key concepts involve {' and '.join(words[3:6])}. "
        mock_response += f"This relates to broader principles of information theory and entropy measurement."

        return mock_response


class OllamaProvider(ModelProvider):
    """Ollama local model provider."""

    def __init__(self, model_id: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_id)
        self.base_url = base_url

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        import aiohttp

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", "")
                else:
                    raise Exception(f"Ollama error: {resp.status}")


class RouterProvider(ModelProvider):
    """Provider that delegates to the Pluribus Next router."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.router_path = "/pluribus/pluribus_next/tools/providers/router.py"

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        proc = await asyncio.create_subprocess_exec(
            "python3", self.router_path,
            "--prompt", prompt,
            "--model", self.model_id,
            "--provider", "auto",
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            return f"[Error: Router failed: {stderr.decode()}]"
            
        try:
            result = json.loads(stdout.decode())
            if result.get("ok"):
                return result.get("text", "")
            else:
                return f"[Error: {result.get('error')}]"
        except json.JSONDecodeError:
            return f"[Error: Invalid JSON: {stdout.decode()[:100]}]"


def get_provider(model_id: str) -> ModelProvider:
    """Get appropriate provider for model ID."""
    if model_id.startswith("mock"):
        return MockProvider(model_id)
    elif model_id.startswith("ollama:"):
        return OllamaProvider(model_id.split(":", 1)[1])
    elif any(x in model_id for x in ["gemini", "claude", "gpt", "grok", "codex", "llama", "mistral"]):
        return RouterProvider(model_id)
    else:
        # Default to mock for now - real providers would be added here
        return MockProvider(model_id)


# ==============================================================================
# LENS Stage: Superposition
# ==============================================================================

def extract_claims(response: str, model_id: str, sample_k: int) -> list[Claim]:
    """Extract atomic claims from a response."""
    if _extract_claims is None:
        # Fallback claim extraction
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 20]
        return [
            Claim(
                id=Claim.compute_id(s),
                text=s,
                source_model=model_id,
                source_sample=sample_k
            )
            for s in sentences[:20]  # Limit claims per response
        ]

    raw_claims = _extract_claims(response)
    return [
        Claim(
            id=Claim.compute_id(c),
            text=c,
            source_model=model_id,
            source_sample=sample_k
        )
        for c in raw_claims[:20]
    ]


def generate_single_response(
    prompt: str,
    model_id: str,
    sample_k: int,
    temperature: float = 0.7
) -> ModelResponse:
    """Generate a single response and profile it."""
    provider = get_provider(model_id)

    start = time.time()
    try:
        response_text = provider.generate_sync(prompt, temperature)
    except Exception as e:
        response_text = f"[Error: {e}]"
    latency_ms = (time.time() - start) * 1000

    # Profile entropy
    entropy = None
    if profile_entropy is not None:
        try:
            entropy = profile_entropy(
                Y=response_text,
                X=prompt,
                emit_event=False
            )
        except Exception:
            pass

    # Extract claims
    claims = extract_claims(response_text, model_id, sample_k)

    return ModelResponse(
        model_id=model_id,
        sample_k=sample_k,
        response=response_text,
        entropy=entropy,
        claims=claims,
        latency_ms=latency_ms
    )


def lens_superposition(
    req_id: str,
    prompt: str,
    config: SynthesizerConfig
) -> LENSOutput:
    """
    LENS Stage: Generate superposition of responses across models.

    This is the "multiple waveforms coexisting" phase.
    """
    emit_bus(config, "lens.superposition.started", "metric", "info", {
        "req_id": req_id,
        "models": config.models,
        "samples_per_model": config.samples_per_model
    })

    start_time = time.time()
    responses: list[ModelResponse] = []

    # Generate all responses
    if config.parallel:
        with ThreadPoolExecutor(max_workers=len(config.models) * config.samples_per_model) as executor:
            futures = []
            for model_id in config.models:
                for k in range(config.samples_per_model):
                    # Vary temperature slightly per sample for diversity
                    temp = 0.7 + (k * 0.1)
                    futures.append(
                        executor.submit(generate_single_response, prompt, model_id, k, temp)
                    )

            for future in as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as e:
                    print(f"[LENS] Response generation failed: {e}", file=sys.stderr)
    else:
        for model_id in config.models:
            for k in range(config.samples_per_model):
                temp = 0.7 + (k * 0.1)
                responses.append(generate_single_response(prompt, model_id, k, temp))

    # Build claim pool with agreement analysis
    claim_pool: dict[str, ClaimPoolEntry] = {}

    for resp in responses:
        for claim in resp.claims:
            if claim.id not in claim_pool:
                claim_pool[claim.id] = ClaimPoolEntry(
                    claim=claim,
                    sources=[(resp.model_id, resp.sample_k)]
                )
            else:
                claim_pool[claim.id].sources.append((resp.model_id, resp.sample_k))

    # Compute agreement ratios
    total_sources = len(config.models) * config.samples_per_model
    for entry in claim_pool.values():
        entry.agreement_ratio = len(entry.sources) / total_sources

        # Compute per-claim aleatoric uncertainty
        # High variance in which samples include this claim = high h_alea
        model_presence = {}
        for model_id, sample_k in entry.sources:
            if model_id not in model_presence:
                model_presence[model_id] = []
            model_presence[model_id].append(sample_k)

        # If claim appears in all samples of a model, low aleatoric
        # If claim appears in some samples, high aleatoric
        alea_scores = []
        for model_id in config.models:
            samples_with_claim = len(model_presence.get(model_id, []))
            alea_scores.append(1.0 - samples_with_claim / config.samples_per_model)

        entry.h_alea = sum(alea_scores) / len(alea_scores) if alea_scores else 0.0

    total_latency = (time.time() - start_time) * 1000

    emit_bus(config, "lens.superposition.complete", "metric", "info", {
        "req_id": req_id,
        "response_count": len(responses),
        "claim_count": len(claim_pool),
        "latency_ms": total_latency
    })

    return LENSOutput(
        req_id=req_id,
        prompt=prompt,
        responses=responses,
        claim_pool=claim_pool,
        total_latency_ms=total_latency
    )


# ==============================================================================
# LASER Stage: Verification & Synthesis
# ==============================================================================

def classify_uncertainty(entry: ClaimPoolEntry) -> str:
    """Classify the uncertainty type of a claim."""
    if entry.h_alea > 0.5:
        return "aleatoric"  # Unstable across samples
    elif entry.support_score < 0.3 and entry.agreement_ratio < 0.5:
        return "epistemic"  # Missing evidence
    elif entry.agreement_ratio >= 0.5 and entry.support_score >= 0.5:
        return "verified"
    elif entry.agreement_ratio >= 0.4:
        return "underdetermined"
    else:
        return "epistemic"


def verify_claims(
    claim_pool: dict[str, ClaimPoolEntry],
    config: SynthesizerConfig
) -> list[ClaimPoolEntry]:
    """
    LASER verification: Assign support scores and classify uncertainty.

    In a full implementation, this would query external sources.
    For now, we use cross-model agreement as a proxy for verification.
    """
    verified = []

    for claim_id, entry in claim_pool.items():
        # Use agreement ratio as support score proxy
        # Claims that multiple models agree on are more likely correct
        entry.support_score = entry.agreement_ratio

        # Boost support for claims with low aleatoric uncertainty
        if entry.h_alea < 0.3:
            entry.support_score = min(1.0, entry.support_score * 1.2)

        # Classify uncertainty type
        entry.uncertainty_type = classify_uncertainty(entry)

        verified.append(entry)

    return verified


def select_claims(
    verified: list[ClaimPoolEntry],
    config: SynthesizerConfig
) -> list[ClaimPoolEntry]:
    """
    Select claims that satisfy constraint budgets.

    This is the "wave collapse" phase - selecting which claims survive.
    """
    # Filter by minimum agreement
    candidates = [e for e in verified if e.agreement_ratio >= config.min_agreement]

    # Prefer verified claims
    candidates.sort(key=lambda e: (
        e.uncertainty_type == "verified",  # Verified first
        e.support_score,  # Then by support
        -e.h_alea  # Then by stability (lower aleatoric)
    ), reverse=True)

    # Select top claims while respecting budgets
    selected = []
    total_conj = 0.0

    for entry in candidates:
        # Check conjecture budget
        claim_conj_contrib = 1.0 - entry.support_score
        if total_conj + claim_conj_contrib > config.budgets.get("conjecture_max", 0.05) * len(selected + [entry]):
            continue

        selected.append(entry)
        total_conj += claim_conj_contrib

        # Limit total claims for readability
        if len(selected) >= 25:
            break

    return selected


def synthesize_response(
    selected_claims: list[ClaimPoolEntry],
    prompt: str,
    config: SynthesizerConfig
) -> tuple[str, EntropyVector | None]:
    """
    Compose final synthesized response from selected claims.

    Uses MDL (Minimum Description Length) principle for compression.
    """
    if not selected_claims:
        return "Unable to synthesize response: no verified claims.", None

    # Group claims by semantic similarity (simple heuristic)
    # In a full implementation, this would use proper clustering
    claim_texts = [e.claim.text for e in selected_claims]

    # Remove redundant claims (high overlap)
    unique_claims = []
    seen_tokens = set()

    for entry in selected_claims:
        claim_tokens = set(_tokenize(entry.claim.text)) if _tokenize else set(entry.claim.text.lower().split())

        # Check overlap with already-selected claims
        overlap = len(claim_tokens & seen_tokens) / len(claim_tokens) if claim_tokens else 1.0

        if overlap < 0.5:  # Less than 50% overlap
            unique_claims.append(entry)
            seen_tokens.update(claim_tokens)

    # Compose response
    paragraphs = []
    current_para = []

    for i, entry in enumerate(unique_claims):
        current_para.append(entry.claim.text)

        # Start new paragraph every 3-4 claims
        if len(current_para) >= 3:
            paragraphs.append(" ".join(current_para))
            current_para = []

    if current_para:
        paragraphs.append(" ".join(current_para))

    synthesized = "\n\n".join(paragraphs)

    # Profile final entropy
    final_entropy = None
    if profile_entropy is not None:
        try:
            final_entropy = profile_entropy(
                Y=synthesized,
                X=prompt,
                schema=config.task_schema,
                emit_event=False
            )
        except Exception:
            pass

    return synthesized, final_entropy


def laser_verification_synthesis(
    lens_output: LENSOutput,
    config: SynthesizerConfig
) -> LASEROutput:
    """
    LASER Stage: Verify claims and synthesize optimal response.

    This is the "measurement collapses the wavefunction" phase.
    """
    emit_bus(config, "laser.verification.started", "metric", "info", {
        "req_id": lens_output.req_id,
        "claim_count": len(lens_output.claim_pool)
    })

    # Verify claims
    verified = verify_claims(lens_output.claim_pool, config)

    emit_bus(config, "laser.verification.complete", "metric", "info", {
        "req_id": lens_output.req_id,
        "verified_count": len([v for v in verified if v.uncertainty_type == "verified"]),
        "total_count": len(verified)
    })

    # Select claims under constraints
    selected = select_claims(verified, config)

    emit_bus(config, "laser.selection.complete", "metric", "info", {
        "req_id": lens_output.req_id,
        "selected_count": len(selected)
    })

    # Synthesize final response
    synthesized_text, final_entropy = synthesize_response(
        selected, lens_output.prompt, config
    )

    utility = final_entropy.utility if final_entropy else 0.0

    # Build metadata
    metadata = {
        "version": "1.0.0",
        "req_id": lens_output.req_id,
        "models": config.models,
        "samples_per_model": config.samples_per_model,
        "claim_merge": {"rule": "agreement >= 0.4 OR verified"},
        "budgets": config.budgets,
        "observed": final_entropy.to_dict() if final_entropy else {},
        "U": utility,
        "claims_selected": len(selected),
        "claims_rejected": len(verified) - len(selected),
        "verification_coverage": len([v for v in verified if v.uncertainty_type == "verified"]) / len(verified) if verified else 0
    }

    emit_bus(config, "laser.synthesis.complete", "artifact", "info", {
        "req_id": lens_output.req_id,
        "utility": utility,
        "claims_selected": len(selected),
        "response_length": len(synthesized_text)
    })

    return LASEROutput(
        req_id=lens_output.req_id,
        verified_claims=verified,
        selected_claims=selected,
        synthesized_text=synthesized_text,
        final_entropy=final_entropy,
        utility=utility,
        metadata=metadata
    )


# ==============================================================================
# Main Pipeline
# ==============================================================================

def synthesize(
    prompt: str | None = None,
    config: SynthesizerConfig | None = None,
    req_id: str | None = None,
    *,
    repo_root: str | None = None,
    repo_scope: list[str] | None = None,
    repo_exclude: list[str] | None = None,
) -> LASEROutput:
    """
    Run the full LENS/LASER synthesis pipeline with dual-input support.

    DUAL-INPUT ARCHITECTURE:
    ========================

    This synthesizer supports two fundamentally different input modalities:

    1. PROMPT PATH (Generative):
       - Natural language goal/query
       - Tree automata explore generative possibility space
       - Interference patterns emerge from multi-model agreement

    2. REPO PATH (World Model):
       - Repository/codebase provides grounding constraints
       - Type constraints, test expectations, invariants
       - Deterministic rules that generated claims must satisfy

    The LENS stage generates waveforms (multiple model responses) that
    must satisfy the world model's deterministic rules during LASER
    verification.

    Args:
        prompt: User prompt to synthesize response for (generative path)
        config: Synthesizer configuration
        req_id: Optional request ID (auto-generated if not provided)
        repo_root: Repository root path (world model path)
        repo_scope: Glob patterns to include in world model
        repo_exclude: Glob patterns to exclude from world model

    Returns:
        LASEROutput with synthesized response and metadata
    """
    if config is None:
        config = SynthesizerConfig()

    if req_id is None:
        req_id = f"lens-{uuid.uuid4().hex[:8]}"

    # Require at least one input
    if not prompt and not repo_root:
        raise ValueError("At least one of prompt or repo_root must be provided")

    # Build world model if repo provided
    world_model: RepoWorldModel | None = None
    if repo_root:
        emit_bus(config, "lens_laser.worldmodel.building", "metric", "info", {
            "req_id": req_id,
            "repo_root": repo_root
        })
        world_model = RepoWorldModel.from_repo(
            repo_root=repo_root,
            scope=repo_scope,
            exclude=repo_exclude
        )
        emit_bus(config, "lens_laser.worldmodel.complete", "metric", "info", {
            "req_id": req_id,
            "files": len(world_model.file_tree),
            "type_constraints": len(world_model.type_constraints),
            "test_expectations": len(world_model.test_expectations),
            "world_model_hash": world_model.world_model_hash
        })

    emit_bus(config, "lens_laser.pipeline.started", "metric", "info", {
        "req_id": req_id,
        "prompt_length": len(prompt) if prompt else 0,
        "has_world_model": world_model is not None,
        "dual_input": bool(prompt and repo_root)
    })

    start_time = time.time()

    # LENS: Generate superposition (if prompt provided)
    if prompt:
        lens_output = lens_superposition(req_id, prompt, config)
    else:
        # Repo-only mode: use world model to generate synthetic prompt
        lens_output = LENSOutput(
            req_id=req_id,
            prompt=f"[World model analysis of {repo_root}]",
            responses=[],
            claim_pool={},
            total_latency_ms=0
        )

    # Apply world model constraints to claim pool (interference zone)
    if world_model and lens_output.claim_pool:
        emit_bus(config, "lens_laser.interference.analyzing", "metric", "info", {
            "req_id": req_id,
            "claim_count": len(lens_output.claim_pool)
        })

        interference_zones = []
        for claim_id, entry in lens_output.claim_pool.items():
            constraint_check = world_model.check_claim_against_constraints(entry.claim)

            if constraint_check["violations"]:
                # Create destructive interference zone
                interference_zones.append(InterferenceZone(
                    claims=[entry.claim],
                    interference_type="destructive",
                    resultant_amplitude=0.0,
                    violations=constraint_check["violations"]
                ))

                # In strict mode, mark claim as rejected
                if config.interference_mode == "strict":
                    entry.support_score = 0.0
                    entry.uncertainty_type = "destructive_interference"
                elif config.interference_mode == "lenient":
                    # Just reduce support score
                    entry.support_score *= 0.5

            elif constraint_check["warnings"]:
                # Create underdetermined zone
                interference_zones.append(InterferenceZone(
                    claims=[entry.claim],
                    interference_type="underdetermined",
                    resultant_amplitude=0.7,
                    violations=[]
                ))
            else:
                # Constructive interference - claim satisfies world model
                if entry.agreement_ratio >= config.min_agreement:
                    interference_zones.append(InterferenceZone(
                        claims=[entry.claim],
                        interference_type="constructive",
                        resultant_amplitude=entry.agreement_ratio * 1.2,  # Amplify
                        violations=[]
                    ))
                    # Boost support score for world-model-verified claims
                    entry.support_score = min(1.0, entry.support_score * 1.3)

        emit_bus(config, "lens_laser.interference.complete", "metric", "info", {
            "req_id": req_id,
            "constructive": len([z for z in interference_zones if z.interference_type == "constructive"]),
            "destructive": len([z for z in interference_zones if z.interference_type == "destructive"]),
            "underdetermined": len([z for z in interference_zones if z.interference_type == "underdetermined"])
        })

    # LASER: Verify and synthesize
    laser_output = laser_verification_synthesis(lens_output, config)

    # Add world model info to metadata
    if world_model:
        laser_output.metadata["world_model"] = {
            "repo_root": world_model.repo_root,
            "hash": world_model.world_model_hash,
            "files": len(world_model.file_tree),
            "type_constraints": len(world_model.type_constraints),
            "test_expectations": len(world_model.test_expectations),
            "isomorphic": world_model.isomorphic_boundary is not None
        }

    total_time = (time.time() - start_time) * 1000

    emit_bus(config, "lens_laser.pipeline.complete", "metric", "info", {
        "req_id": req_id,
        "utility": laser_output.utility,
        "total_latency_ms": total_time,
        "dual_input_mode": bool(prompt and repo_root)
    })

    return laser_output


# ==============================================================================
# CLI Interface
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="synthesizer.py",
        description="LENS/LASER Synthesizer - Multi-model entropic response synthesis with dual-input support"
    )
    p.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to synthesize (or read from stdin if not provided)"
    )

    # Dual-input: Repo path arguments
    p.add_argument(
        "--repo", "-r",
        dest="repo_root",
        help="Repository root path for world model construction (enables dual-input mode)"
    )
    p.add_argument(
        "--repo-scope",
        nargs="+",
        help="Glob patterns to include in world model (default: **/*.py **/*.ts **/*.tsx)"
    )
    p.add_argument(
        "--repo-exclude",
        nargs="+",
        help="Glob patterns to exclude from world model (default: node_modules, .git, dist)"
    )
    p.add_argument(
        "--interference-mode",
        choices=["strict", "lenient", "explore"],
        default="strict",
        help="How to handle claims that violate world model (default: strict)"
    )

    # Model configuration
    p.add_argument(
        "--models", "-m",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to use (default: {DEFAULT_MODELS})"
    )
    p.add_argument(
        "--samples", "-k",
        type=int,
        default=DEFAULT_SAMPLES_PER_MODEL,
        help=f"Samples per model (default: {DEFAULT_SAMPLES_PER_MODEL})"
    )
    p.add_argument(
        "--schema",
        help="Task schema JSON file"
    )
    p.add_argument(
        "--bus-dir",
        default="/pluribus/.pluribus/bus",
        help="Bus directory"
    )
    p.add_argument(
        "--actor",
        default="lens-laser-synth",
        help="Actor name for bus events"
    )
    p.add_argument(
        "--no-emit",
        action="store_true",
        help="Do not emit bus events"
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output full JSON result"
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Run model calls sequentially (default: parallel)"
    )
    return p


def main(argv: list[str]) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)

    # Get prompt (optional if repo provided)
    prompt = args.prompt
    if not prompt and not args.repo_root:
        # Try to read from stdin
        import select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            prompt = sys.stdin.read().strip()

    # Require at least one input
    if not prompt and not args.repo_root:
        print("Error: At least one of prompt or --repo must be provided", file=sys.stderr)
        return 1

    # Load schema if provided
    schema = None
    if args.schema:
        try:
            with open(args.schema) as f:
                schema = json.load(f)
        except Exception as e:
            print(f"Error loading schema: {e}", file=sys.stderr)
            return 1

    # Build config
    config = SynthesizerConfig(
        models=args.models,
        samples_per_model=args.samples,
        task_schema=schema,
        parallel=not args.sequential,
        emit_bus=not args.no_emit,
        bus_dir=args.bus_dir,
        actor=args.actor,
        interference_mode=args.interference_mode
    )

    # Run synthesis with dual-input support
    try:
        result = synthesize(
            prompt=prompt,
            config=config,
            repo_root=args.repo_root,
            repo_scope=args.repo_scope,
            repo_exclude=args.repo_exclude
        )
    except Exception as e:
        print(f"Error during synthesis: {e}", file=sys.stderr)
        return 1

    # Output result
    if args.json:
        output = {
            "req_id": result.req_id,
            "synthesized_text": result.synthesized_text,
            "utility": result.utility,
            "claims_selected": len(result.selected_claims),
            "claims_verified": len([v for v in result.verified_claims if v.uncertainty_type == "verified"]),
            "entropy": result.final_entropy.to_dict() if result.final_entropy else None,
            "metadata": result.metadata
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print("LENS/LASER Synthesized Response")
        print("=" * 60)
        print()
        print(result.synthesized_text)
        print()
        print("-" * 60)
        print(f"Utility: {result.utility:.4f}")
        print(f"Claims selected: {len(result.selected_claims)}")
        print(f"Models: {', '.join(config.models)}")
        if result.final_entropy:
            print(f"Quality tier: {result.final_entropy.interpret_quality()}")
            print(f"Clarity: {result.final_entropy.clarity():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
