# DNA.md - Dual Neurosymbolic Automata

**Version:** v2.0 | **Protocol Stack:** DKIN v28 | PAIP v15 | Citizen v1

> **DNA** = **D**ual **N**eurosymbolic **A**utomata
>
> The paradigm backbone of Pluribus: web code provides structured latent state (symbolic),
> while LLMs generate context, narrative, and decisions (neural).

---

## Axiom Bindings

These are the gravitational forces that bind disembodied agents with energy tokens flowing through them:

### 1. Entelecheia (ἐντελέχεια)
**The critical gravity of purpose.**

Every agent, episode, and lineage has an intrinsic *telos* (end, purpose).
Evolution is not random drift—it's movement toward entelecheia: the state where the organism fulfills its inherent potential.

Observable signals:
- `entelecheia_delta.telos_alignment` — closeness to purpose
- `entelecheia_delta.semantic_coherence` — internal consistency
- `entelecheia_delta.human_resonance` — "yes, that's what I meant"

### 2. Inertia
**Resistance to purposeless change.**

Systems at rest tend to stay at rest. Systems in motion toward telos continue unless deflected.
Inertia prevents:
- Churn without progress
- Refactoring that doesn't serve purpose
- Drift from semantic coherence

### 3. Witness
**Every mutation must have a witness.**

*Replaces the former VOR (Verification, Observability, Reproducibility) pattern.*

Witnesses produce **Attestations** — the only admissible evidence of entelecheia.
- Verification witness: saw the action succeed/fail
- Observation witness: can report what happened
- Reproduction witness: can repeat the action

### 4. Hysteresis
**Memory of past states influences present behavior.**

The system doesn't respond purely to current input—it carries traces of its evolutionary history.
Lineage DAG, CMP history, and attestation ledger are hysteresis mechanisms.

### 5. Infinity (Ω-logic)
**Omega acceptance for infinite traces.**

Evolution is unbounded. The system must remain live (ω-gate) and safe (Ω-gate) across infinite time horizons.
Büchi acceptance ensures that good states are visited infinitely often.

---

## SemOps Scope Policy

Evolution is a **tabula rasa** experiment. The synced `semops.json` (30 operators from Pluribus) exists for reference but has scope restrictions:

| Agent Context | SemOps Access | Rationale |
|---------------|---------------|-----------|
| **Agents INSIDE evolution** | 🚫 IGNORE | Tabula rasa: build fresh omega-centric vocabulary |
| **Agents working ON evolution** | ✅ USE | Pluribus orchestrators (us) can leverage existing ops |

**Inside evolution:** Agents spawned within the DNA experiment should not reference Pluribus SemOps. They develop their own meta-language for the dual neurosymbolic automata.

**On evolution:** External orchestrators (Pluribus, Antigravity) coordinating the evolution experiment may use the full SemOps registry.

This distinction preserves the experimental integrity while allowing orchestration.

## Energy Token Flow

Energy (attention, compute, human guidance) flows through the organism:

```
Human Intent (telos seed)
       │
       ▼
┌──────────────────────┐
│  PERCEIVE            │ ← Ingest priors, SOTA, user reqs
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  ENCODE              │ ← Genotype → Phenotype mapping
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  LOOP                │ ← Iterate with CMP fitness, Witness attestations
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  REFINE              │ ← Selection pressure, prune failures (Inertia)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  QUERY               │ ← Verification against invariants (Witness)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Ω-gate + ω-gate     │ ← Safety + Liveness (Infinity)
└──────────────────────┘
           │
           ▼
    Entelecheia achieved? ← (Human resonance signal)
```

---

## Taxon: Genetic Information Sharing

Genetically useful information flows across taxonomic levels:

| Level | Scope | Transfer Mechanism |
|-------|-------|-------------------|
| **Clone** | Single PAIP instance | In-memory state |
| **Agent** | Individual CAGENT | Bus events, ledgers |
| **Clade** | Cooperating agents | CMP aggregation |
| **Species** | Shared lineage | VGT (Vertical Gene Transfer) |
| **Family** | Cross-lineage | HGT (Horizontal Gene Transfer) |
| **Class** | Cross-project | Archive/Fossil record |

### HGT Guard Ladder (G1-G6)
Every horizontal transfer passes through:
- **G1** Type Compatibility
- **G2** Timing Compatibility
- **G3** Effect Boundary (Ring 0 protection)
- **G4** Omega Acceptance (lineage compatibility)
- **G5** MDL Penalty (complexity cost)
- **G6** Spectral Stability (PQC signatures)

---

## Invocation Modes

Two modes for engaging the DNA organism:

### Mode A: Prompt as Weights
```
Input: Single prompt or instruction
Process: LLM generates according to DNA axioms
Output: Episode with entelecheia_delta
```
The prompt *biases* generation toward specific telos.

### Mode B: Repo as Substrate
```
Input: Messy human/machine collaboration (entire repository)
Process: Evolutionary observation, transformation, purification
Output: Organism moving toward coherent entelecheia
```
The repo *is* the substrate upon which DNA evolves.

---

## WWM Principles (Web World Models)

> "World state in web code for logical consistency + LLMs for narrative/decisions"

| Principle | Implementation |
|-----------|----------------|
| Code-defined rules | iso_git.mjs, guards, typed interfaces |
| Model-driven imagination | Dialogos, LASER/LENS superposition |
| Typed web interfaces | NDJSON ledgers, bus events, lineage DAG |
| Deterministic generation | HGT guards, CMP scoring |

### LASER / LENS
- **LASER**: Language Augmented Superposition Effective Retrieval
- **LENS**: LLM Entropic Natural Superposition

---

## Ring Hierarchy

| Ring | Zone | Access | Components |
|------|------|--------|------------|
| 0 | KERNEL | Operator-Only | DNA.md, CITIZEN.md, ring_guard.py |
| 1 | OPERATOR | Elevated | agent_bus.py, witness.py, cmp_engine_v2.py |
| 2 | APPLICATION | Standard | Dashboard, tools, iso_git.mjs |
| 3 | EPHEMERAL | Scoped | PAIP clones, episodes |

---

## Planning Artifacts (RAK)

| File | Purpose |
|------|---------|
| `kanban.md` | Active task/episode board |
| `archive.md` | Completed episodes with attestations |
| `PROMPT.md` | Ralph loop instructions |
| `@fix_plan.md` | Sprint acceptance criteria |
| `AI_WORKFLOW.md` | Agent behavior guidelines |

---

## Immutable Principles

1. **Sovereignty First** — Dialogos owns agent identity
2. **Protocol Compliance** — REPL Headers (DKIN v28)
3. **Witness Covenant** — Attestations for every mutation
4. **Ring Compartmentalization** — Access via Ring 0-3
5. **Lossless Ledger** — No work shall vanish
6. **Evidence Emission** — All actions produce bus events
7. **Golden Ratio Threshold** — Φ-score ≥ 0.618 for citizenship
8. **Horizontal Gene Transfer** — HGT Guard Ladder (G1-G6)
9. **Clade Productivity** — CMP over individual metrics
10. **Graceful Degradation** — Amber preservation on failure
11. **Entelecheia Orientation** — Purpose over completion
12. **Inertial Stability** — Resist purposeless change

---

## Evolution Records

| Date | Event | Version | Author |
|------|-------|---------|--------|
| 2025-12-30 | Phase 0 Foundation | v1.0 | Multi-Agent Swarm |
| 2025-12-31 | DNA Axiom Rewrite | v2.0 | Antigravity + User |
