# NeSy Compiler: Deep Technical README

> **A Verified Program Synthesizer with Learned Heuristics**
> *Implementing AlphaGeometry/AlphaProof Patterns for Code*

---

## Executive Summary

The NeSy Compiler is a **neurosymbolic compilation framework** that combines neural network proposals with symbolic verification, inspired by DeepMind's AlphaGeometry. It implements a sophisticated 5-stage pipeline (PERCEIVE → GROUND → CONSTRAIN → VERIFY → EMIT) for transforming high-level intents into verified, executable code.

**Core Insight:** Code that is spatially well-structured but temporally unstable is fundamentally different from code that is spatially chaotic but temporally stable. The 4D model (X=structure, Y=semantics, Z=trust, T=temporal) captures this distinction.

### Status: Research Prototype (30-40% Complete)

| Dimension | Score | Assessment |
|-----------|-------|------------|
| Architecture | 8/10 | Excellent design, clean boundaries |
| Implementation | 4/10 | Many mock/stub implementations |
| Testing | 5/10 | Good unit tests, missing integration |
| Production Readiness | 3.5/10 | Requires substantial work |

---

## The Denoise Loop: Core Philosophy

```
observation → decompose(aleatoric, epistemic) → denoise_epistemic → accept_aleatoric → evolve
```

**Aleatoric uncertainty** (irreducible randomness) is separated from **epistemic uncertainty** (reducible through learning). The compiler accepts inherent variance in code evolution while systematically reducing knowledge gaps through verification gates.

---

## Architecture Overview

### Package Structure (8 packages, ~15,400 LOC)

```
nesy-compiler/
├── packages/
│   ├── core/          # Primitives (neural, symbolic, bridge) - FUNCTIONAL
│   ├── pipeline/      # 5-stage compilation chain - FUNCTIONAL
│   ├── integration/   # Orchestration & bus - PARTIAL
│   ├── prompt/        # Prompt engineering - FUNCTIONAL
│   ├── learning/      # Meta-learning & RLCF - FUNCTIONAL
│   ├── puzzle/        # SAT/SMT constraint solving - PARTIAL
│   ├── temporal/      # Git archeology, 4D analysis - FUNCTIONAL
│   └── persistence/   # FalkorDB graph storage - STUB
```

### Dependency Graph (Acyclic)

```
                    @nesy/core (foundation)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   @nesy/pipeline    @nesy/prompt    @nesy/temporal
        │                 │                 │
   ┌────┴────┐           │                 │
   │         │           │                 │
@nesy/puzzle │           │                 │
@nesy/learning           │                 │
        │                │                 │
        └────────────────┴─────────────────┘
                        │
                @nesy/integration
                        │
                @nesy/persistence
```

---

## The 5-Stage Pipeline

### Stage 1: PERCEIVE (Neural Feature Extraction)

**File:** `packages/pipeline/src/stages/perceive.ts`

Extracts neural features from input:
- Embeddings (1536-dim vectors)
- Attention weights
- Feature vectors for downstream grounding

**Status:** ⚠️ MOCK - Uses deterministic pseudo-embeddings instead of real model calls

### Stage 2: GROUND (Symbol Extraction)

**File:** `packages/pipeline/src/stages/ground.ts`

Converts neural features to symbolic structures:
- Attention-guided symbol extraction
- Ambiguity detection via cosine similarity
- Candidate scoring mechanism
- Type inference (heuristic-based)

**Status:** ✅ FUNCTIONAL (heuristic-only, no learned model)

### Stage 3: CONSTRAIN (Constraint Satisfaction)

**File:** `packages/pipeline/src/stages/constrain.ts`

Propagates and solves constraints:
- Full constraint propagation loop
- Backtracking search
- Equality, inequality, membership constraints
- Substitution management

**Status:** ✅ FUNCTIONAL (first-order only, no CDCL)

### Stage 4: VERIFY (Sextet Gates)

**File:** `packages/pipeline/src/stages/verify.ts`

The **most complete** implementation. Six verification gates:

| Gate | Purpose | Implementation |
|------|---------|----------------|
| **Provenance** | Hash chain verification, taint tracking | ✅ Complete |
| **Effects** | Side-effect analysis with trust authorization | ✅ Complete |
| **Liveness** | Cycle detection, livelock prevention | ✅ Complete |
| **Recovery** | Checkpoint verification | ✅ Complete |
| **Quality** | Completeness, consistency, groundedness | ✅ Complete |
| **Omega** | Self-referential consistency, paradox detection | ✅ Complete |

### Stage 5: EMIT (Pentad Construction)

**File:** `packages/pipeline/src/stages/emit.ts`

Generates compiled output with:
- **WHY:** Purpose inference from gate results
- **WHERE:** Scope boundaries extraction
- **WHO:** Principal and trust level tracking
- **WHEN:** Temporal metadata
- **WHAT:** Action specification

---

## The 4D Model

### Dimensional Analysis

| Dimension | Meaning | Metrics |
|-----------|---------|---------|
| **X (Structure)** | AST, imports, dependencies | LOC, complexity, export count |
| **Y (Semantics)** | Types, contracts, embeddings | Type completeness, test coverage |
| **Z (Trust)** | Ring levels (0-3), proofs | Proof status, linter score |
| **T (Temporal)** | Git history, evolution | Churn, entelechy, thrash |

### Ring Classification

- **Ring 0:** Proven (healthScore ≥ 85, entelechy = signal)
- **Ring 1:** Trusted (healthScore ≥ 70, no thrash patterns)
- **Ring 2:** Monitored (healthScore ≥ 50)
- **Ring 3:** Experimental (below thresholds)

---

## Formal Methods Components

### SAT Encoder (`packages/puzzle/src/sat-encoder.ts`)

Converts constraints to CNF:
- Tseitin transformation for clause splitting
- DIMACS export format
- Term-to-variable mapping

**Critical Issues:**
- Equality/inequality encoding has semantic mismatch (propositional vs. first-order)
- Incomplete bidirectional equivalence in Tseitin transformation

### SMT Interface (`packages/puzzle/src/smt-interface.ts`)

SMT-LIB2 compatible interface:
- Theory support: Bool, Int, Real, BitVec, Array
- Push/pop incremental solving
- Constraint translation

**Status:** ⛔ STUB - `check()` always returns 'unknown'

### Proof Search (`packages/puzzle/src/proof-search.ts`)

Neural-guided algorithms:
- Best-first search
- Beam search
- MCTS with UCB1 (correctly implemented)

**Issues:**
- Substitution composition uses spread instead of `composeSubstitution`
- MCTS rewards unbounded (no [0,1] normalization)

### Theorem Prover (`packages/puzzle/src/theorem-prover.ts`)

Proof generation with export formats:
- Natural language
- Sequent calculus
- Lean4, Coq (syntax issues)
- LaTeX

**Critical:** All proofs use `sorry`/`admit` placeholders - no actual tactic generation

---

## Learning System

### RLCF (Reinforcement Learning from Compiler Feedback)

**File:** `packages/learning/src/rlcf.ts`

**Status:** Non-standard algorithm, no convergence guarantees

**Issues:**
- Policy update rule is neither policy gradient nor Q-learning
- Missing baseline subtraction for variance reduction
- No entropy regularization
- Fixed epsilon=0.1 (no decay)

### Training Loop

**File:** `packages/learning/src/training-loop.ts`

**Critical:** No actual neural network training occurs - only computes metrics

### Generator (Draft/Refine)

**File:** `packages/learning/src/generator.ts`

- PCFG sampling for structure
- SLM completion for content
- Sequential hole filling (greedy)

**Issue:** No beam search or backtracking for hole filling

### Negative Sampler

**File:** `packages/learning/src/negative-sampler.ts`

10 transform types:
- Dead Code, Phantom Import, Type Mismatch
- Magic Number, Deep Nesting, Long Function
- Hallucinated Import, Incomplete Implementation
- Style Inconsistency, Copy-Paste Error

**Issue:** Limited diversity, no hard negative mining

---

## Temporal Analysis

### Git Walker (`packages/temporal/src/git-walker.ts`)

Fast git history traversal:
- Shell commands for performance
- Rename detection with `-M` flag
- Async generators for memory efficiency

**Issues:**
- No merge commit filtering
- Incomplete rename detection for complex patterns

### Thrash Detector (`packages/temporal/src/thrash-detector.ts`)

Identifies high-churn low-value code:
- Fix-fix pattern detection
- Semantic change ratio (UNIMPLEMENTED)
- Arbitrary thresholds (not empirically calibrated)

### Entelechy Extractor (`packages/temporal/src/entelechy.ts`)

Signal vs. noise identification:
- "Survived refactors" heuristic
- Keyword-based refactor detection (~60% precision)

### 4D Compiler (`packages/temporal/src/four-d-compiler.ts`)

Integrates spatial and temporal analysis:
- Combined health scoring
- Ring classification
- Risk assessment
- Actionable recommendations

---

## Implementation Status by Phase

### Implemented Steps (35 of 150)

| Phase | Steps | Status |
|-------|-------|--------|
| **Phase 1:** LSA Pipeline | 1-4 | ✅ Complete |
| **Phase 1.5:** Sensor Fusion | 11-14 | ✅ Complete |
| **Phase 2:** Temporal | 17-25 | ✅ Complete |
| **Phase 3:** Constraints | 26-30 | ⚠️ Partial (SMT stub) |
| **Phase 4:** Integration | 31-33 | ⚠️ Partial (mock bus) |
| **Phase 5:** Training | 34-50 | ❌ **MISSING** |
| **Phase 6:** Dataset Export | 51-57 | ✅ Complete |
| **Phase 8:** RLCF | 85 | ⚠️ Partial |

### Critical Gap: Phase 5 (Steps 34-50)

The missing phase should bridge Integration (Phase 4) to Dataset Export (Phase 6):
- Model fine-tuning orchestration
- LoRA/prefix-tuning adapters
- Evaluation harness
- Model registry/versioning

---

## Known Issues Summary

### Critical (Blocking Production)

1. **Mock embedding generation** - No real model integration
2. **SMT solver is stub** - Returns placeholder results
3. **FalkorDB client unimplemented** - All TODOs
4. **Bus connection is mock** - In-memory only
5. **Theorem prover produces `sorry`** - No actual proofs

### High Priority

1. **Weak hash function** - Security risk for provenance
2. **SAT encoding semantic mismatch** - Wrong CNF for equality
3. **RLCF non-standard update** - No convergence guarantees
4. **Substitution composition bug** - Breaks unification chains

### Medium Priority

1. **No test coverage for temporal** - 2/9 modules tested
2. **Missing PCFG export** - Not in learning/index.ts
3. **Arbitrary thresholds** - No empirical calibration
4. **No error taxonomy** - Generic Error objects

---

## Theoretical Foundations

### What's Correct

1. **Robinson Unification** - Sound implementation with occurs check
2. **Substitution Application** - Correct recursive resolution
3. **Substitution Composition** - Proper `(s1 . s2)(t) = s2(s1(t))`
4. **UCB1 in MCTS** - Correct formula implementation
5. **IR Transformation Chain** - Type-safe, hash-linked provenance

### What Needs Work

1. **First-order equality** - Needs congruence axioms
2. **SMT translation** - Invalid identifiers, missing sort checking
3. **Lean4/Coq export** - Syntax errors, no actual tactics
4. **Uncertainty decomposition** - Heuristic, not principled
5. **Temporal metrics** - Gaming vulnerabilities, no validation

---

## Expert Audit Ratings

| Expert Domain | Score | Key Finding |
|---------------|-------|-------------|
| **Formal Methods** | 4/10 | SAT/SMT semantically incorrect |
| **ML/Learning** | 3/10 | No actual training occurs |
| **Architecture** | 8/10 | Excellent design, mock implementations |
| **Temporal Analysis** | 6.5/10 | Sound concepts, missing validation |
| **Gap Analysis** | - | Phase 5 (Steps 34-50) missing |

---

## Test Coverage

```
Packages:           8
Test Files:        24
Estimated Coverage: 60-70%

Strong:    @nesy/core, @nesy/learning
Partial:   @nesy/pipeline, @nesy/integration
Weak:      @nesy/puzzle, @nesy/temporal
Minimal:   @nesy/persistence
```

---

## Quick Start

```bash
# Install dependencies
pnpm install

# Build all packages
pnpm build

# Run tests
pnpm test

# Development mode
pnpm dev
```

---

## Critical Files for Implementation

1. **`packages/pipeline/src/stages/perceive.ts`** - Replace mock embeddings with real model calls
2. **`packages/puzzle/src/smt-interface.ts`** - Integrate Z3/CVC4 solver
3. **`packages/persistence/src/falkor-client.ts`** - Implement FalkorDB connection
4. **`packages/integration/src/bus-integration.ts`** - Connect to real event bus
5. **`packages/puzzle/src/proof-search.ts:544`** - Use `composeSubstitution` instead of spread

---

## Roadmap to Production

### Phase 1: Core Functionality (1-3 months)
- [ ] Implement real embedding generation (OpenAI/Anthropic)
- [ ] Complete FalkorDB persistence layer
- [ ] Integrate Z3 SMT solver
- [ ] Connect to Pluribus event bus

### Phase 2: Training Infrastructure (3-6 months)
- [ ] Implement Phase 5 (Steps 34-50)
- [ ] Add model fine-tuning orchestration
- [ ] Create evaluation harness
- [ ] Build model registry

### Phase 3: Production Hardening (6-12 months)
- [ ] Replace weak hash with SHA-256
- [ ] Add comprehensive integration tests
- [ ] Implement circuit breakers
- [ ] Deploy to Kubernetes

---

## Conclusion

The NeSy Compiler represents an **ambitious and theoretically sound** approach to neurosymbolic compilation. The architecture demonstrates excellent design principles with clean package boundaries and sophisticated type safety.

**Current State:** High-quality research prototype requiring 3-6 months of focused development for production readiness.

**Strength:** The foundation is solid - verification gates, IR chain, and 4D model are production-quality designs.

**Challenge:** Implementation completion, not architectural redesign.

---

*Deep README synthesized from 5-expert collaborative audit: Formal Methods, ML/Learning Theory, Software Architecture, Temporal Analysis, and Gap Analysis experts.*

*Generated: 2026-02-04*
