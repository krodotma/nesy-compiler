# nesy-compiler

> A neurosymbolic compilation library for bridging neural perception with symbolic reasoning.

**NeSy Compiler** transforms heterogeneous representations—continuous neural embeddings and discrete symbolic structures—into verified, executable artifacts through a bidirectional translation pipeline.

## What is Neurosymbolic Compilation?

Unlike traditional compilers (static source-to-target transformation) or pure ML systems (differentiable loss optimization), a neurosymbolic compiler mediates between stochastic pattern recognition and deterministic logical reasoning.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NESY COMPILER PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │ PERCEIVE │ →  │  GROUND  │ →  │CONSTRAIN │ →  │  VERIFY  │ →  │ EMIT  │ │
│  │          │    │          │    │          │    │          │    │       │ │
│  │ Neural   │    │ Symbol   │    │ SAT/SMT  │    │ Proof    │    │ IR +  │ │
│  │ Encoding │    │ Binding  │    │ Solving  │    │ Check    │    │ Holon │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│       ↓               ↓               ↓               ↓               ↓     │
│   Embedding       Grounded        Satisfied       Verified       Compiled  │
│   + Attention     Symbols         Constraints     Derivation     Artifact  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Insight

The compiler is a **verified program synthesizer with learned heuristics**:
- **Neural proposal**: LLM generates candidate structures
- **Symbolic verification**: Formal methods certify correctness
- **Bidirectional gradient**: Learning improves both sides

This is the AlphaGeometry/AlphaProof pattern generalized: neural intuition + symbolic rigor.

## The Compilation Contract

```typescript
import { NeSyCompiler } from '@nesy/core';

const compiler = new NeSyCompiler();

// Four input modes
const result = await compiler.compile(request);

// Genesis: Raw NL → full verification
compiler.compile({ nl: "Create a file processor", context });

// Seed: Partial Pentad → completion + verification
compiler.compile({ pentad: { why, what }, context });

// Atom: Verified pair → transformation only
compiler.compile({ atom: verifiedPentadSextet });

// Constraint: Sextet constraints → find satisfying Pentad
compiler.compile({ sextet: constraints, goal: "..." });
```

### Output: `CompiledHolon`

```typescript
{
  holon: ActualizedHolon,      // Living unit, ready for bus emission
  ir: ExecutionPlan,           // Intermediate representation
  proof: TransformationProof,  // Sextet passage receipt + derivation
  provenance: Lineage          // Taint vector, trust score
}
```

### Guarantees

1. **Sextet Closure**: No output without complete gate passage
2. **Pentad Completeness**: All 5Ws resolved in output
3. **Taint Propagation**: Trust never silently upgraded
4. **Deterministic Proof**: Reproducible transformation chain

## Packages

| Package | Description |
|---------|-------------|
| `@nesy/core` | Core primitives: neural, symbolic, bridge |
| `@nesy/pipeline` | Compilation stages: perceive → ground → constrain → verify → emit |
| `@nesy/prompt` | Prompt engineering: templates, CoT, structured generation |
| `@nesy/learning` | Meta-learning: MAML variants, experience replay, bandit selection |
| `@nesy/puzzle` | Constraint solving: SAT embedding, planning, propagation |
| `@nesy/integration` | External hooks: Holon, Pluribus, bus events |

## Installation

```bash
pnpm add @nesy/core @nesy/pipeline
```

## Quick Start

```typescript
import { perceive, ground, constrain, verify, emit } from '@nesy/pipeline';
import { createContext } from '@nesy/core';

const context = createContext({ model: 'opus-4.5' });

// 5-stage compilation
const embedding = await perceive("Summarize the codebase", context);
const symbols = await ground(embedding);
const satisfied = await constrain(symbols, sextet);
const proof = await verify(satisfied);
const holon = await emit(proof);

console.log(holon.pentad); // Complete 5Ws
console.log(holon.proof);  // Verification receipt
```

## Architecture

```
nesy-compiler/
├── packages/
│   ├── core/           # Neural, symbolic, bridge primitives
│   ├── pipeline/       # Compilation stages + IR
│   ├── prompt/         # Prompt engineering toolkit
│   ├── learning/       # Meta-learning and adaptation
│   ├── puzzle/         # SAT/SMT, planning, constraints
│   └── integration/    # Holon, Pluribus, bus hooks
├── examples/
│   ├── arc-reasoning/  # ARC puzzle compilation
│   └── code-synthesis/ # Program synthesis demo
└── docs/
    ├── architecture.md
    ├── ir-spec.md
    └── integration-guide.md
```

## Key Primitives

### Neural Layer
- **Embedding**: Map discrete tokens to continuous vectors
- **Attention**: Compute relevance-weighted aggregations
- **Adapter**: LoRA, prefix-tuning for efficient fine-tuning

### Symbolic Layer
- **Unification**: Match symbolic terms
- **Constraint Propagation**: Reduce solution spaces via logical implications
- **Proof Search**: Navigate derivation trees

### Bridge Layer
- **Grounding**: Neural → Symbolic (map embeddings to terms)
- **Lifting**: Symbolic → Neural (abstract structures to embeddings)
- **Discretization**: Continuous → Discrete (Boltzmann with annealing)

## Learning in Compilers

Unlike static compilers, nesy-compiler improves through experience:

1. **Library Induction**: Discover reusable symbolic primitives from solved problems
2. **Neural Module Transfer**: Pre-train banks of composable neural components
3. **Reinforcement from Verification**: Use formal verification as reward signal

## Integration with Holon

The compiler integrates with the Holon ecosystem:

```typescript
import { toHolon, fromPentad } from '@nesy/integration/holon';

// Compile NL to verified Holon
const holon = await compiler.compile({ nl: intent, context });

// Emit to Pluribus bus
await bus.emit('holon.actualized', toHolon(holon));
```

## Research Foundations

- DeepMind AlphaGeometry/AlphaProof: Neural hypothesis + symbolic verification
- Scallop: Differentiable Datalog for neurosymbolic programming
- Logic Tensor Networks: First-order logic as fuzzy tensor operations
- SATNet: Differentiable SAT solving as neural layer
- MemGPT: Virtual context for long-horizon reasoning

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and conventions.

## License

MIT

---

*Neural intuition proposes. Symbolic rigor verifies. Compilation unifies.*
