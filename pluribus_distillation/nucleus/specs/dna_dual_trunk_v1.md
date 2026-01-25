# DNA: Dual Neurosymbolic Automata Architecture

**Version:** 1.0
**Status:** Draft
**Author:** Claude Opus 4.5
**Date:** 2026-01-01
**Protocol:** DKIN v28

---

## 1. Executive Summary

DNA (Dual Neurosymbolic Automata) defines the fundamental architecture of Pluribus as a **supersymmetric app pair**: two trunks that observe, refine, and evolve each other in perpetual balance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DNA: DUAL NEUROSYMBOLIC AUTOMATA                    │
│                                                                             │
│    TRUNK A: pluribus                    TRUNK B: pluribus_evolution         │
│    ════════════════════                 ════════════════════════════        │
│    Execution / Runtime                  Refinement / Evolution              │
│    Operational protocols                Retroactive analysis                │
│    Bus infrastructure                   Vector/manifold optimization        │
│    Dashboard / PORTAL                   Neurosymbolic synthesis             │
│                                                                             │
│              ◄────────── SUPERSYMMETRY ──────────►                          │
│              (No client/server distinction in production)                   │
│                                                                             │
│    isogit ─────► rhizome ─────► evolutionary protocols ─────► auom/axioms   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Principles

### 2.1 Supersymmetry

In particle physics, supersymmetry relates bosons and fermions. In Pluribus:

| Concept | Trunk A (pluribus) | Trunk B (pluribus_evolution) |
|---------|-------------------|------------------------------|
| **Role** | Execution (Force) | Observation (Matter) |
| **Phenotype** | Server-like | Client-like |
| **State** | Mutable, operational | Analytical, reflective |
| **Time** | Present execution | Past/future refinement |

**The Supersymmetric Principle:** In production, there is no distinction. Both trunks are isomorphic—they can run the same code, access the same bus, and produce equivalent outcomes.

### 2.2 The Evolutionary Chain

```
isogit ──► rhizome ──► evolutionary protocols ──► auom/sextet/axioms
   │           │                │                        │
   │           │                │                        └─► Constitutional law
   │           │                └─► CMP, Clade-Weave, DNA
   │           └─► Content-addressed storage, provenance
   └─► Isomorphic git (browser ≡ server)
```

### 2.3 Dense Subtrees

Each vertical-purpose component should be a **focused, dense subtree** rather than scattered across a vast monorepo. Benefits:

- **Graspability**: ~5-10 files to understand, not 500
- **Iterability**: Changes are localized and testable
- **Portability**: Subtrees can be reused in other projects
- **CMP-fitness**: Clades compete on focused metrics

### 2.4 Primary Mode Lattice (Cross-Trunk)

Both trunks implement the same mode lattice as a cross-cutting overlay. This keeps execution and evolution semantics aligned and prevents drift between subtrees.

- **Ingest:** capture signals with context governance (curation, dedupe, window control)
- **Inception:** canonicalize ingress and assign lineage/ingress ids
- **Delineate:** Sextet + omega gating (AM/SM/decay)
- **Orchestrate:** topology selection and handoffs
- **Align:** etymon/entelechy checks across intent and outcomes
- **Operationalize:** holon/commit/service materialization + verification

Context governance is an Ingest property, not a separate mode.

---

## 3. Subtree Inventory

### 3.1 Current Subtrees (2)

| Subtree | Type | Source |
|---------|------|--------|
| `membrane/graphiti` | submodule | github.com/getzep/graphiti |
| `membrane/mem0-fork` | subtree | mem0ai/mem0 (squashed) |

### 3.2 Candidate Subtrees (New)

| Subtree | Purpose | Current Location | Files |
|---------|---------|------------------|-------|
| **LASER/** | Entropy synthesis, multi-model fusion | `nucleus/tools/lens_*` | ~5 |
| **portal/** | Declarative A2UI renderer | `nucleus/dashboard/src/components/portal/` | ~4 |
| **WUA/** | Web User Agent (iframe/WebRTC) | `local_wua-plan.md` (unimplemented) | 0 |
| **SMOKE/** | WASM x86 emulator substrate | Planned vendor | 0 |
| **AURALUX/** | Voice pipeline | `nucleus/auralux/` | ~10 |

### 3.3 Trunk Subtrees

| Subtree | Purpose |
|---------|---------|
| **pluribus/** | Primary execution trunk |
| **pluribus_evolution/** | Secondary refinement trunk (autoclaude integrated) |

---

## 4. LASER Subtree Specification

### 4.1 Extraction Plan

**Current:**
```
nucleus/tools/lens_laser_synth.py       (1200 lines)
nucleus/tools/lens_collimator.py        (800 lines)
nucleus/tools/lens_entropy_profiler.py  (1000 lines)
nucleus/tools/topology_policy.py        (200 lines)
nucleus/specs/lens_laser_synthesizer_v1.md (1300 lines)
```

**Target:**
```
laser/
├── __init__.py
├── synthesizer.py          # LENS/LASER core pipeline
├── entropy_profiler.py     # H* 8-dimensional vector
├── world_model.py          # RepoWorldModel class
├── collimator.py           # Query routing
├── topology.py             # Multi-agent topology selection
├── tree_automata.py        # Generative exploration
├── interference.py         # Prompt ↔ Repo collision
├── specs/
│   ├── synthesizer.md
│   ├── entropy_schema.json
│   └── dual_input.md
├── tests/
│   ├── test_synthesizer.py
│   └── test_entropy.py
├── pyproject.toml
└── README.md
```

### 4.2 Interface Contract

```python
# laser/__init__.py
from .synthesizer import synthesize, SynthesizerConfig
from .entropy_profiler import EntropyVector, profile_entropy
from .world_model import RepoWorldModel
from .collimator import route_query, RoutePlan

__all__ = [
    "synthesize",
    "SynthesizerConfig",
    "EntropyVector",
    "profile_entropy",
    "RepoWorldModel",
    "route_query",
    "RoutePlan",
]
```

### 4.3 Bus Integration

LASER emits to the shared bus via topic prefix `laser.*`:

```
laser.pipeline.started
laser.worldmodel.complete
laser.interference.analyzing
laser.synthesis.complete
```

---

## 5. PORTAL Subtree Specification

### 5.1 Extraction Plan

**Current:**
```
nucleus/dashboard/src/components/portal/
├── PortalRenderer.tsx
├── portal.css
├── portal-types.ts
└── portal-actions.ts
```

**Target:**
```
portal/
├── src/
│   ├── PortalRenderer.tsx
│   ├── portal.css
│   ├── types.ts
│   └── actions.ts
├── specs/
│   └── a2ui_protocol.md
├── tests/
│   └── portal.spec.ts
├── package.json
└── README.md
```

### 5.2 A2UI Protocol

PORTAL interprets declarative UI messages:

```typescript
interface PortalMessage {
  type: "card" | "form" | "list" | "chart" | "markdown";
  id: string;
  content: unknown;
  actions?: PortalAction[];
}

interface PortalAction {
  id: string;
  label: string;
  topic: string;  // Bus topic to emit on click
  payload: unknown;
}
```

---

## 6. WUA Subtree Specification

### 6.1 Purpose

WUA (Web User Agent) provides browser automation via iframe/WebRTC for:
- Visual debugging and screenshot analysis
- E2E testing without Playwright overhead
- Agent-driven UI exploration
- Live debugging of Pluribus dashboard

### 6.2 Architecture (from local_wua-plan.md)

```
┌─────────────────────────────────────────────────────────────────┐
│                         WUA ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Capture   │───►│  Perception │───►│   Planner   │         │
│  │  (iframe)   │    │  (VLM/OCR)  │    │  (Agent)    │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│         ▲                                      │                │
│         │           ┌─────────────┐            │                │
│         └───────────│   Actions   │◄───────────┘                │
│                     │ (Synthetic) │                             │
│                     └─────────────┘                             │
│                                                                 │
│  Evidence Store: logs, screenshots, DOM snapshots, diffs       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Target Structure

```
wua/
├── src/
│   ├── capture.ts          # Frame pump, getDisplayMedia
│   ├── perception.ts       # VLM/OCR tiered analysis
│   ├── planner.ts          # Action selection
│   ├── actions.ts          # Synthetic input (click, type)
│   ├── evidence.ts         # Artifact storage
│   └── grammar.ts          # WUA action DSL
├── specs/
│   └── wua_protocol.md
├── tests/
│   └── wua.spec.ts
├── package.json
└── README.md
```

---

## 7. pluribus_evolution Trunk

### 7.1 Purpose

The secondary trunk that observes and refines the primary:

| Capability | Description |
|------------|-------------|
| **Retroactive** | Analyze past code, find patterns, suggest refactors |
| **Current** | Optimize active inference, tune parameters |
| **Future** | Evolve code/manifold coupling, predict drift |

### 7.2 Integration: autoclaude

`pluribus_evolution` integrates with the autoclaude pattern for continuous refinement:

```
pluribus_evolution/
├── observer/               # Watches pluribus primary
│   ├── code_analyzer.py
│   ├── vector_profiler.py
│   └── drift_detector.py
├── refiner/                # Proposes improvements
│   ├── refactor_planner.py
│   ├── manifold_optimizer.py
│   └── axiom_evolver.py
├── synthesizer/            # Generates refined code
│   ├── code_generator.py
│   └── test_generator.py
├── bridge/                 # Connects to primary trunk
│   ├── bus_mirror.py
│   └── rhizome_sync.py
└── autoclaude/             # Integrated autoclaude
    ├── agent.py
    └── config.json
```

### 7.3 Supersymmetric Deployment

In production, both trunks run simultaneously:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        SUPERSYMMETRIC DEPLOYMENT                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   pluribus (primary)              pluribus_evolution (secondary)           │
│   ══════════════════              ══════════════════════════════           │
│         │                                    │                             │
│         │     ┌──────────────────────────┐   │                             │
│         └────►│     Shared Bus           │◄──┘                             │
│               │  (events.ndjson / WS)    │                                 │
│               └──────────────────────────┘                                 │
│                          │                                                 │
│               ┌──────────┴──────────┐                                      │
│               │      Rhizome        │                                      │
│               │ (Content-Addressed) │                                      │
│               └─────────────────────┘                                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Git Subtree Operations

### 8.1 Extracting a Subtree

```bash
# Extract LASER as subtree
git subtree split --prefix=nucleus/tools --branch=laser-extract

# Create new repo
cd /path/to/laser
git init
git pull /pluribus laser-extract

# Add back as subtree
cd /pluribus
git subtree add --prefix=laser /path/to/laser main --squash
```

### 8.2 Updating a Subtree

```bash
# Pull updates from subtree repo
git subtree pull --prefix=laser /path/to/laser main --squash

# Push changes to subtree repo
git subtree push --prefix=laser /path/to/laser main
```

### 8.3 CMP Governance

The Clade-Weave protocol governs subtree evolution:

1. **Clade Isolation**: Changes happen in `clade/<agent>/<task>` branches
2. **Fitness Evaluation**: CMP scores determine merge priority
3. **Neurosymbolic Weave**: Conflicts resolved via synthesis, not overwrite
4. **Genotype Protection**: Core axioms require Constitutional Review

---

## 9. Implementation Roadmap

### Phase 1: LASER Extraction
- [ ] Create `laser/` subtree structure
- [ ] Move lens_* files with git filter-branch
- [ ] Add pyproject.toml, tests, README
- [ ] Verify bus integration still works

### Phase 2: PORTAL Extraction
- [ ] Create `portal/` subtree structure
- [ ] Extract from dashboard/src/components/portal
- [ ] Add package.json, tests
- [ ] Verify A2UI protocol

### Phase 3: WUA Implementation
- [ ] Create `wua/` subtree from local_wua-plan.md
- [ ] Implement capture, perception, planner
- [ ] Integrate with PORTAL for UI automation

### Phase 4: pluribus_evolution Trunk
- [ ] Create secondary trunk structure
- [ ] Implement observer, refiner, synthesizer
- [ ] Integrate autoclaude
- [ ] Establish supersymmetric deployment

---

## 10. Axiom Integration

This architecture derives from core axioms:

```
AXIOM append_only_evidence {
  ∀e ∈ Event: once_emitted(e) → immutable(e)
}

AXIOM supersymmetry {
  ∀f ∈ Function: executable(f, backend) ↔ executable(f, edge)
}

AXIOM dual_trunk {
  ∃ trunk_a, trunk_b:
    observes(trunk_b, trunk_a) ∧
    refines(trunk_b, trunk_a) ∧
    isomorphic(trunk_a, trunk_b)
}

AXIOM dense_subtree {
  ∀s ∈ Subtree: |files(s)| ≤ 15 ∧ focused(s) ∧ portable(s)
}
```

---

## Appendix A: Supersymmetric Capability Matrix

| Capability | Backend (VPS) | Edge (Browser) |
|------------|---------------|----------------|
| **Mind** | Gemini/Claude API | WebLLM (WebGPU) |
| **Memory** | Local FS + Git | IsoGit (PQC Signed) |
| **Nervous System** | events.ndjson (File) | BusClient (WS/RTC) |
| **Body/Computer** | Linux Kernel | Smoke (WASM x86) |
| **UI** | Dashboard (SSR) | Dashboard (CSR) |
| **Synthesis** | LASER (Python) | LASER (Pyodide) |

---

**Document Status**

| Field | Value |
|-------|-------|
| Created | 2026-01-01 |
| Protocol | DKIN v28 |
| Author | Claude Opus 4.5 |
| Verified | Pending |
