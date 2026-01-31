# Changelog

All notable changes to the NeSy-Compiler project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-31

### Session Context

This release represents the initial creation of the NeSy-Compiler repository, born from discussions about neurosymbolic compilation patterns observed across multiple projects (Inceptional's Knowledge Compiler, Pluribus Intent Compiler).

### Research Phase

#### External Pattern Analysis
The following external research was synthesized to inform the architecture:

1. **Karpathy LLM OS (7 Pillars)**
   - Cognitive core architecture concepts
   - Memory hierarchy principles
   - Tool use integration patterns

2. **AlphaGeometry/AlphaProof Pattern**
   - Neural hypothesis generation
   - Symbolic verification loop
   - Search-guided reasoning

3. **Anthropic Multi-Agent Patterns**
   - Orchestration strategies
   - Agent coordination primitives

4. **Autopoiesis Concepts**
   - Self-maintaining systems
   - Boundary definition
   - Operational closure

5. **seL4/CHERI Security Models**
   - Capability-based security
   - Trust ring hierarchy
   - Formal verification principles

6. **MemGPT Architecture**
   - Memory tiering
   - Context management
   - Retrieval patterns

#### Design Synthesis

Three focused research agents contributed architectural insights:

1. **Agent 1 - Core Synthesis**
   - Defined neurosymbolic compilation as "verified program synthesizer with learned heuristics"
   - Identified neural proposal + symbolic verification as core pattern
   - Mapped trust levels to execution rings

2. **Agent 2 - Contract Design**
   - Established 4 compilation modes: Genesis, Seed, Atom, Constraint
   - Defined CompiledHolon output structure
   - Specified verification guarantees

3. **Agent 3 - Architecture**
   - Proposed monorepo with 6 packages
   - Designed 5-stage pipeline: PERCEIVE → GROUND → CONSTRAIN → VERIFY → EMIT
   - Identified bridge primitives for neural↔symbolic translation

### Implementation Phase

#### Package: @nesy/core (v0.1.0)

**types.ts - Core Type Definitions**
- [x] `TrustLevel` enum aligned with Holon rings (KERNEL=0, PRIVILEGED=1, STANDARD=2, UNTRUSTED=3)
- [x] `CompilationRequest` discriminated union (4 modes)
- [x] `GenesisRequestSchema` - create new holon from specification
- [x] `SeedRequestSchema` - evolve existing holon with mutations
- [x] `AtomRequestSchema` - compile atomic intent
- [x] `ConstraintRequestSchema` - apply constraint refinement
- [x] `CompiledHolon` output type with:
  - `holon` - actualized holon data
  - `ir` - execution plan
  - `proof` - transformation proof
  - `provenance` - lineage tracking
- [x] `Term` type (variable | constant | compound)
- [x] `Constraint` type (equality | inequality | membership | custom)
- [x] `SymbolicStructure` with terms, constraints, metadata
- [x] `Embedding`, `AttentionWeights`, `NeuralFeatures` types
- [x] `GroundingResult`, `LiftingResult` bridge types
- [x] `DiscretizationConfig` for continuous→discrete conversion

**neural.ts - Neural Layer Primitives**
- [x] `createEmbedding()` - construct embedding from vector
- [x] `cosineSimilarity()` - embedding comparison
- [x] `averageEmbeddings()` - embedding aggregation
- [x] `extractAttention()` - attention weight extraction
- [x] `attentionEntropy()` - uncertainty measurement
- [x] `combineFeatures()` - multi-modal feature fusion
- [x] `featureConfidence()` - confidence estimation

**symbolic.ts - Symbolic Layer Primitives**
- [x] Term type guards: `isVariable()`, `isConstant()`, `isCompound()`
- [x] Term constructors: `variable()`, `constant()`, `compound()`
- [x] `termToString()` - term pretty printing
- [x] `emptySubstitution()` - create empty substitution
- [x] `applySubstitution()` - apply substitution to term
- [x] `composeSubstitution()` - combine substitutions
- [x] `occursCheck()` - prevent infinite terms
- [x] `unify()` - Robinson unification algorithm
- [x] `isGround()` - check if term has no variables
- [x] `extractVariables()` - get all variables from term
- [x] Constraint builders: `equalityConstraint()`, `inequalityConstraint()`, `membershipConstraint()`, `customConstraint()`
- [x] `createSymbolicStructure()`, `addTerm()`, `addConstraint()`

**bridge.ts - Neural↔Symbolic Bridge**
- [x] `GroundingStrategy` type for neural→symbolic
- [x] `ThresholdGrounding` - hard discretization
- [x] `SoftGrounding` - probabilistic grounding
- [x] `createGrounding()` - grounding factory
- [x] `groundFeatures()` - main grounding function
- [x] `LiftingStrategy` type for symbolic→neural
- [x] `ConcatLifting` - concatenation-based lifting
- [x] `liftSymbols()` - symbolic to embedding conversion
- [x] `DiscretizationStrategy` for continuous→discrete
- [x] Annealing schedules: linear, exponential, cosine
- [x] `boltzmannDiscretize()` - temperature-based discretization

**context.ts - Compilation Context**
- [x] `CompilationContext` interface:
  - Model configuration (model, temperature)
  - Trust tracking (trustLevel, taintVector)
  - Discretization config
  - Embedding cache
  - Trace entries
  - Limits (maxIterations, timeoutMs)
- [x] `createContext()` - context factory with defaults
- [x] `addTrace()` - add stage trace entry
- [x] `elevateContext()` - trust level elevation
- [x] `taintContext()` - add taint to context

#### Package: @nesy/pipeline (v0.1.0)

**ir.ts - Intermediate Representation**
- [x] `IRNodeKind` enum: neural, grounded, constrained, verified, compiled
- [x] `IRNode` base interface with id, hashes, timestamp, metadata
- [x] Stage-specific IR types:
  - `PerceiveIR` - neural features + modality
  - `GroundIR` - symbols + confidence + ambiguities
  - `ConstrainIR` - satisfied/unsatisfied constraints + search steps
  - `VerifyIR` - proof + gates passed/failed
  - `CompiledIR` - artifact + provenance chain
- [x] `VerificationProof` with Sextet gates:
  - `provenance` - lineage verification
  - `effects` - side-effect analysis
  - `liveness` - deadlock/livelock detection
  - `recovery` - rollback capability
  - `quality` - output quality metrics
  - `omega` - self-referential consistency
- [x] `GateResult` with status, evidence, timestamp
- [x] `CompiledArtifact` with Pentad (5 Ws):
  - `why` - purpose and motivation
  - `where` - spatial/contextual scope
  - `who` - actors and principals
  - `when` - temporal constraints
  - `what` - actions and outputs
- [x] `ExecutionStep` for execution plan
- [x] `ProvenanceChain` and `ProvenanceEntry` for audit trail
- [x] `createIRNode()`, `hashIR()` utilities

**stages/perceive.ts - PERCEIVE Stage**
- [x] Input: Raw input (text, image, audio)
- [x] Output: NeuralFeatures (embedding + attention + confidence)
- [x] `perceive()` main function
- [x] `detectModality()` - text/image/audio/multimodal detection
- [x] `extractSourceText()` - source text extraction
- [x] `hashInput()` - cache key generation
- [x] `hashFeatures()` - feature fingerprinting
- [x] `generateEmbedding()` - mock embedding generation (1536-dim)
- [x] `extractAttention()` - mock attention extraction (12 heads)
- [x] `computeConfidence()` - entropy-based confidence
- [x] Embedding cache integration

**stages/ground.ts - GROUND Stage**
- [x] Input: NeuralFeatures from PERCEIVE
- [x] Output: SymbolicStructure (grounded symbols)
- [x] `ground()` main function
- [x] `extractSymbolCandidates()` - propose symbols from embeddings
- [x] `findSalientRegions()` - attention-guided region detection
- [x] `proposeSymbol()` - symbol type inference from embedding
- [x] `scoreGroundingCandidates()` - candidate evaluation
- [x] `computeGroundingScore()` - grounding quality metric
- [x] `computeAmbiguityScore()` - ambiguity detection
- [x] `buildSymbolicStructure()` - term construction
- [x] `buildTerm()` - individual term building
- [x] `computeGroundingConfidence()` - overall confidence

**stages/constrain.ts - CONSTRAIN Stage**
- [x] Input: SymbolicStructure from GROUND + external constraints
- [x] Output: Satisfied constraints + search trace
- [x] `constrain()` main function
- [x] `SearchState` management
- [x] `propagateConstraints()` - iterative propagation
- [x] `tryConstraint()` - individual constraint evaluation
- [x] `tryEqualityConstraint()` - equality handling
- [x] `tryInequalityConstraint()` - inequality handling
- [x] `tryMembershipConstraint()` - set membership
- [x] `tryCustomConstraint()` - extensible constraints
- [x] `backtrack()` - backtracking search
- [x] `resolveTermWithSubst()` - substitution application
- [x] `hasUnboundVariables()` - groundness check
- [x] `extractVariables()` - variable extraction from constraints

**stages/verify.ts - VERIFY Stage (Sextet Gates)**
- [x] Input: Constrained structure
- [x] Output: VerificationProof
- [x] `verify()` main function orchestrating all gates
- [x] **Gate 1: Provenance**
  - Hash chain integrity verification
  - Taint tracking
- [x] **Gate 2: Effects**
  - Side-effect term detection
  - Effect classification (io, network, state, system)
  - Trust-level based effect authorization
- [x] **Gate 3: Liveness**
  - Dependency graph construction
  - Cycle detection (deadlock)
  - Unbounded loop detection (livelock)
- [x] **Gate 4: Recovery**
  - Checkpoint availability check
  - State snapshot verification
  - Recoverability assessment
- [x] **Gate 5: Quality**
  - Completeness metric
  - Consistency metric (contradiction detection)
  - Groundedness metric
- [x] **Gate 6: Omega (Self-referential)**
  - Meta-level term detection
  - Paradox detection
  - Fixed-point consistency check

**stages/emit.ts - EMIT Stage**
- [x] Input: Verified IR
- [x] Output: CompiledArtifact with Pentad
- [x] `emit()` main function
- [x] `buildPentad()` - construct 5 Ws
  - `buildWhy()` - purpose inference
  - `buildWhere()` - scope boundaries
  - `buildWho()` - principal authorization
  - `buildWhen()` - temporal constraints
  - `buildWhat()` - action specification
- [x] `generateExecutionPlan()` - step generation
- [x] `determineTrustLevel()` - final trust assignment
- [x] `buildProvenanceChain()` - audit trail construction

**compiler.ts - Main NeSyCompiler Class**
- [x] `CompilerConfig` with:
  - enableCache, timeout, maxIterations, trace, model
- [x] `CompilationResult` with:
  - ir, holon, stages, metrics
- [x] `CompilationMetrics`:
  - totalDurationMs, stageDurations, cacheHits
  - constraintIterations, verificationGates passed/failed
- [x] `NeSyCompiler` class
  - `compile()` - main entry point
  - Mode dispatch based on request type
  - Stage orchestration with timing
  - Metrics collection
- [x] `createCompiler()` factory
- [x] `compileText()` quick compile
- [x] `compileAndVerify()` with verification check

### Infrastructure

#### Monorepo Setup
- [x] `pnpm-workspace.yaml` - workspace configuration
- [x] `turbo.json` - build orchestration
- [x] `tsconfig.base.json` - shared TypeScript config
- [x] Root `package.json` with scripts
- [x] `.gitignore` - standard ignores
- [x] Package-level `tsconfig.json` files

#### Dependencies
- `zod` - runtime type validation
- `typescript` - type system
- `vitest` - testing framework
- `turbo` - build system

### Git History

```
db80416 feat: Initialize neurosymbolic compiler pipeline
        - 24 files, 3852 insertions
        - Created on krodotma/nesy-compiler
```

### Architecture Decisions

1. **5-Stage Pipeline** chosen over 3-stage for finer granularity:
   - PERCEIVE: Modal input processing
   - GROUND: Neural→Symbolic translation
   - CONSTRAIN: Constraint satisfaction
   - VERIFY: Multi-gate verification
   - EMIT: Artifact generation

2. **Sextet Verification** implements 6 gates:
   - Maps to Holon verification framework
   - Each gate produces evidence trail
   - Gates can be passed/failed/skipped

3. **Pentad Output** (5 Ws):
   - Structured intent representation
   - Compatible with Holon specification
   - Enables downstream reasoning

4. **Trust Level Alignment**:
   - KERNEL (0) - highest trust
   - PRIVILEGED (1) - elevated
   - STANDARD (2) - normal
   - UNTRUSTED (3) - unverified
   - Maps to execution ring model

5. **IR-Centric Design**:
   - Each stage produces typed IR node
   - Hash-linked provenance chain
   - Enables caching and replay

### Known Limitations

- [ ] Mock embedding generation (needs real model integration)
- [ ] Mock attention extraction (needs model internals access)
- [ ] Simplified constraint solver (needs full Prolog-style search)
- [ ] No persistent caching (in-memory only)
- [ ] Missing packages: prompt, learning, puzzle, integration

### Future Work (Planned)

- [ ] `@nesy/prompt` - Prompt engineering primitives
- [ ] `@nesy/learning` - Online learning from feedback
- [ ] `@nesy/puzzle` - Puzzle embedding and solving
- [ ] `@nesy/integration` - Holon ecosystem integration
- [ ] Real embedding model integration (via API)
- [ ] Persistent IR cache
- [ ] Incremental compilation
- [ ] Parallel constraint solving

---

## Session Timeline

| Time | Action |
|------|--------|
| T+0 | Observed pattern: "Knowledge Compiler" (Inceptional) + "Intent Compiler" (Pluribus) |
| T+1 | User clarified: observation not directive - recognized compiler pattern emergence |
| T+2 | User requested: dedicated neurosymbolic compiler repo |
| T+3 | Launched 3 research agents for architecture synthesis |
| T+4 | Agent 1: Core synthesis - "verified program synthesizer with learned heuristics" |
| T+5 | Agent 2: Contract design - 4 modes, CompiledHolon output |
| T+6 | Agent 3: Architecture - 6 packages, 5-stage pipeline |
| T+7 | Created /tmp/nesy-compiler directory structure |
| T+8 | Implemented @nesy/core/types.ts |
| T+9 | Implemented @nesy/core/neural.ts |
| T+10 | Implemented @nesy/core/symbolic.ts |
| T+11 | Implemented @nesy/core/bridge.ts |
| T+12 | Implemented @nesy/core/context.ts |
| T+13 | Implemented @nesy/pipeline/ir.ts |
| T+14 | Implemented @nesy/pipeline/stages/perceive.ts |
| T+15 | Implemented @nesy/pipeline/stages/ground.ts |
| T+16 | Implemented @nesy/pipeline/stages/constrain.ts |
| T+17 | Implemented @nesy/pipeline/stages/verify.ts |
| T+18 | Implemented @nesy/pipeline/stages/emit.ts |
| T+19 | Implemented @nesy/pipeline/compiler.ts |
| T+20 | Created monorepo config (pnpm, turbo, tsconfig) |
| T+21 | Initialized git, committed |
| T+22 | Created GitHub repo krodotma/nesy-compiler |
| T+23 | Pushed to GitHub |
| T+24 | Created this CHANGELOG |

---

*This CHANGELOG was generated to capture the complete development arc of the NeSy-Compiler initial release.*
