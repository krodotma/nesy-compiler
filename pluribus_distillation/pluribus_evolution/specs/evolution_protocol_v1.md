# Evolution Protocol v1

**Status:** Draft
**Part of:** DNA (Dual Neurosymbolic Automata)

---

## 1. Overview

The Evolution Protocol defines how the secondary trunk (`pluribus_evolution`) observes, analyzes, and refines the primary trunk (`pluribus`).

## 2. Observation Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EVOLUTION OBSERVATION LOOP                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. OBSERVE                                                          │
│     ├─ Bus events via BusMirror                                      │
│     ├─ Code changes via git diff                                     │
│     └─ Vector drift via embedding analysis                           │
│                                                                      │
│  2. ANALYZE                                                          │
│     ├─ Pattern detection (CodeAnalyzer)                              │
│     ├─ Antipattern identification                                    │
│     └─ Drift measurement                                             │
│                                                                      │
│  3. PROPOSE                                                          │
│     ├─ Refactoring suggestions                                       │
│     ├─ Optimization opportunities                                    │
│     └─ Axiom refinements                                             │
│                                                                      │
│  4. SYNTHESIZE                                                       │
│     ├─ Generate code patches                                         │
│     ├─ Generate tests                                                │
│     └─ Emit to evolution bus                                         │
│                                                                      │
│  5. INTEGRATE (via Clade-Weave)                                      │
│     ├─ Create evolution clade                                        │
│     ├─ CMP evaluation                                                │
│     └─ Neurosymbolic merge                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Bus Topics

Evolution trunk emits to `evolution.*` namespace:

| Topic | Kind | Description |
|-------|------|-------------|
| `evolution.observer.analysis` | artifact | Code analysis results |
| `evolution.observer.drift` | metric | Drift detection alerts |
| `evolution.mirror.*` | observation | Mirrored primary events |
| `evolution.refiner.proposal` | proposal | Refactoring proposals |
| `evolution.synthesizer.patch` | artifact | Generated code patches |

## 4. Temporal Modes

| Mode | Description |
|------|-------------|
| **Retroactive** | Analyze past code and commits for patterns |
| **Current** | Optimize active inference and parameters |
| **Predictive** | Forecast drift and suggest preemptive changes |

## 5. Integration with LASER

Evolution trunk uses LASER for:
- Entropy profiling of code changes
- Multi-model synthesis of refactoring proposals
- World model construction for constraint verification

```python
from laser import synthesize, RepoWorldModel

# Build world model from primary trunk
world_model = RepoWorldModel.from_repo("/pluribus")

# Synthesize refactoring proposal
result = synthesize(
    prompt="Refactor this function to reduce cyclomatic complexity",
    repo_root="/pluribus",
    config=SynthesizerConfig(interference_mode="lenient")
)
```

## 6. Supersymmetric Execution

The evolution trunk can run:
- **Server-side**: As a daemon on VPS
- **Edge-side**: In browser via Pyodide + Smoke
- **Hybrid**: Observation on edge, synthesis on server

This is the supersymmetric principle in action.

---

*Protocol version: 1.0*
*See also: `nucleus/specs/dna_dual_trunk_v1.md`*
