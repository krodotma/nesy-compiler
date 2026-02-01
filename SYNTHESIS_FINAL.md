# NeSy Compiler — Synthesis Report

## Architecture Overview

The NeSy Compiler implements a 5-stage neurosymbolic compilation pipeline:

```
PERCEIVE → GROUND → CONSTRAIN → VERIFY → EMIT
```

Each stage transforms an Intermediate Representation (IR) node, with the full pipeline converting raw input (text/image/audio) into a compiled Holon artifact with cryptographic provenance.

## Package Structure (6 packages)

| Package | Role | Dependencies |
|---------|------|-------------|
| `@nesy/core` | Type definitions, symbolic operations, neural primitives, bridge layer | zod |
| `@nesy/pipeline` | 5-stage compilation pipeline, IR nodes, NeSyCompiler | core |
| `@nesy/integration` | High-level API: HolonCompiler, batch processing, result analysis | core, pipeline |
| `@nesy/prompt` | Prompt construction, templates, constraint adaptation, few-shot | core |
| `@nesy/learning` | Feedback collection, experience replay, adaptation strategies | core, pipeline |
| `@nesy/puzzle` | AlphaGeometry-inspired puzzle solving (neural propose, symbolic verify) | core, pipeline |

## Type Flow

### Core Types
- **Term**: `variable | constant | compound` (discriminated on `type`)
- **Constraint**: `equality | inequality | membership | custom` (discriminated on `type`)
- **SymbolicStructure**: `{ terms: Term[], constraints: Constraint[], metadata?: Record<string, unknown> }`
- **TrustLevel**: Ring 0 (KERNEL) → Ring 3 (UNTRUSTED)

### IR Flow
```
PerceiveIR (neural) → GroundIR (grounded) → ConstrainIR (constrained) → VerifyIR (verified) → CompiledIR (compiled)
```

### Sextet Gates (PELRQΩ)
| Gate | Purpose |
|------|---------|
| **P** Provenance | Lineage verification |
| **E** Effects | Side-effect analysis |
| **L** Liveness | Deadlock/livelock detection |
| **R** Recovery | Rollback capability |
| **Q** Quality | Output quality metrics |
| **Ω** Omega | Self-referential consistency |

### Pentad Output (5 Ws)
The EMIT stage constructs: **WHY**, **WHERE**, **WHO**, **WHEN**, **WHAT**

## Key Design Patterns

### AlphaGeometry Pattern (`@nesy/puzzle`)
Neural network proposes construction steps; symbolic engine verifies each step via unification against known terms and constraints.

### Bridge Layer (`@nesy/core`)
- **Grounding** (Neural → Symbolic): ThresholdGrounding, SoftGrounding (Boltzmann)
- **Lifting** (Symbolic → Neural): ConcatLifting with loss estimation
- **Discretization**: Temperature-scaled sigmoid with annealing (linear/exponential/cosine)

### Compilation Modes
- **Genesis**: Create new holon from specification
- **Seed**: Evolve existing holon via mutations
- **Atom**: Compile atomic operation from intent string
- **Constraint**: Apply external constraint refinement

## Fixes Applied

1. **`bridge.ts`**: Corrected Term discriminator from `kind` to `type`, `derivation` field to `metadata`, and Constraint serialization to handle all 4 union variants.
2. **`compiler.ts`**: Fixed `buildHolon` to use `sextetReceipt` (matching Zod schema), construct proper `ProvenanceChain` with `trust/taint/lineage`, and cast external constraints.
3. **`tsconfig.base.json`**: Added `paths` mapping for all `@nesy/*` packages to enable workspace resolution without symlinks.

## Build Status

All 6 packages compile with **zero type errors** using TypeScript 5.x strict mode (ES2022, NodeNext).
