# ARK Swarm Synthesis: Codex 5.2 XHigh

**Agent**: Codex 5.2 XHigh (Implementation Engineer)
**Ring**: 1 (Operator)
**Focus**: isogit internals, AST manipulation, CLI implementation

---

## 1. Isogit Foundation

### 1.1 Package Selection

```bash
npm install isomorphic-git @isomorphic-git/lightning-fs
```

**Why isomorphic-git:**
- Pure JavaScript, no native bindings
- Works in Node, Browser, Electron, React Native
- Same API everywhere = isomorphic parity
- Pluggable filesystem = can run in WASM

### 1.2 ArkRepository Wrapper

```typescript
// nucleus/ark/core/repository.ts
import * as git from 'isomorphic-git';
import http from 'isomorphic-git/http/node';
import * as fs from 'fs';

export class ArkRepository {
  private dir: string;
  private fs: typeof fs;
  
  constructor(dir: string) {
    this.dir = dir;
    this.fs = fs;
  }
  
  async init(): Promise<void> {
    await git.init({ fs: this.fs, dir: this.dir });
    await this.initRhizom();
    await this.initRibosome();
  }
  
  async commit(message: string, context: ArkCommitContext): Promise<string> {
    // Pre-commit: Run Cell Cycle
    const g1 = await this.runG1(context);
    if (!g1.pass) {
      throw new Error(`G1 SLEEP: ${g1.reason}`);
    }
    
    const synthesis = await this.runS(context);
    const g2 = await this.runG2(context, synthesis);
    if (!g2.pass) {
      throw new Error(`G2 ABORT: ${g2.reason}`);
    }
    
    // M-phase: Actual commit
    const sha = await git.commit({
      fs: this.fs,
      dir: this.dir,
      message: this.encodeArkMessage(message, context),
      author: context.author,
    });
    
    // Post-commit: Update Rhizom
    await this.rhizom.insert(sha, context);
    
    return sha;
  }
  
  private encodeArkMessage(message: string, ctx: ArkCommitContext): string {
    // Embed DNA metadata in commit message
    return `${message}

---ARK-METADATA---
cmp: ${ctx.cmp}
entropy: ${JSON.stringify(ctx.entropy)}
witness: ${ctx.witness?.id ?? 'none'}
spec_ref: ${ctx.specRef ?? 'none'}
`;
  }
}
```

### 1.3 Custom Object Types

Extend git objects with ARK-specific types:

```typescript
// Gene object (stored as blob with metadata)
interface GeneObject {
  type: 'gene';
  content: Uint8Array;
  metadata: {
    etymology: string;
    complexity: number;
    usage_count: number;
  };
}

// Phenotype object (derived state)
interface PhenotypeObject {
  type: 'phenotype';
  genes: string[];  // Gene OIDs
  traits: Record<string, number>;
  generation: number;
}
```

---

## 2. CLI Implementation

### 2.1 Command Structure

```bash
ark <command> [options]

Commands:
  init              Initialize ARK repository
  clone <url>       Clone with DNA validation
  status            Show DNA-aware status
  commit            DNA-gated commit
  log               Show log with CMP/entropy
  
  ingest <source>   Distill entropic source
  crystallize       Promote to ISO layer
  purify            Reduce entropy
  
  clade list        List clades
  clade create      Create new clade
  clade merge       LTL-verified merge
  
  spec add          Add LTL spec
  spec verify       Verify current state
  
  swarm init        Initialize multi-agent mode
  health            Show system health
  
  migrate <repo>    Migrate standard git repo
  doctor            Diagnostics
```

### 2.2 Entry Point

```python
#!/usr/bin/env python3
# nucleus/ark/cli.py

import argparse
import sys
from pathlib import Path

from nucleus.ark.commands import (
    init, clone, status, commit, log,
    ingest, crystallize, purify,
    clade, spec, swarm, health, migrate, doctor
)

def main():
    parser = argparse.ArgumentParser(
        prog='ark',
        description='ARK: Autonomous Reactive Kernel - Negentropic VCS'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Register commands
    init.register(subparsers)
    clone.register(subparsers)
    status.register(subparsers)
    commit.register(subparsers)
    # ... etc
    
    args = parser.parse_args()
    return args.func(args)

if __name__ == '__main__':
    sys.exit(main())
```

---

## 3. AST Manipulation for Neural Gate

### 3.1 Feature Extraction

```python
# nucleus/ark/neural/features.py
import ast
from dataclasses import dataclass
from typing import List

@dataclass
class ASTFeatures:
    complexity: float      # McCabe
    depth: float           # Max nesting
    node_count: int        # Total AST nodes
    function_count: int
    class_count: int
    anti_patterns: List[str]

def extract_features(code: str) -> ASTFeatures:
    tree = ast.parse(code)
    
    complexity = _calculate_mccabe(tree)
    depth = _calculate_depth(tree)
    node_count = sum(1 for _ in ast.walk(tree))
    
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    
    patterns = _detect_anti_patterns(tree)
    
    return ASTFeatures(
        complexity=complexity / 20.0,  # Normalize
        depth=depth / 10.0,
        node_count=node_count,
        function_count=len(functions),
        class_count=len(classes),
        anti_patterns=patterns
    )
```

### 3.2 Grammar Filter Integration

```python
# nucleus/ark/synthesis/grammar.py
ALLOWED_NODES = {
    'Module', 'FunctionDef', 'ClassDef', 'Return',
    'If', 'For', 'While', 'With',
    'Assign', 'AugAssign', 'AnnAssign',
    'Import', 'ImportFrom',
    'Name', 'Constant', 'List', 'Dict', 'Tuple',
    # ... whitelist
}

FORBIDDEN_PATTERNS = [
    'AbstractFactoryFactory',
    'MultipleInheritance>2',
    'FunctionLines>100',
    'GlobalState',
]

def validate_grammar(tree: ast.AST) -> tuple[bool, str]:
    for node in ast.walk(tree):
        if type(node).__name__ not in ALLOWED_NODES:
            return False, f"Forbidden node: {type(node).__name__}"
    
    for pattern in FORBIDDEN_PATTERNS:
        if _matches_pattern(tree, pattern):
            return False, f"Forbidden pattern: {pattern}"
    
    return True, "OK"
```

---

## 4. Rhizom DAG Storage

### 4.1 Data Model

```python
# nucleus/ark/rhizom/dag.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class RhizomNode:
    sha: str
    etymology: str              # Semantic origin
    cmp: float                  # Fitness score
    entropy: Dict[str, float]   # H* vector
    parents: List[str]          # Parent SHAs
    lineage_tags: List[str]     # Clade memberships
    timestamp: float
    witness_id: Optional[str]

class RhizomDAG:
    def __init__(self, storage_path: Path):
        self.path = storage_path / '.ark' / 'rhizom.json'
        self.nodes: Dict[str, RhizomNode] = {}
        self._load()
    
    def insert(self, sha: str, context: ArkCommitContext) -> None:
        node = RhizomNode(
            sha=sha,
            etymology=context.etymology,
            cmp=context.cmp,
            entropy=context.entropy,
            parents=[context.parent] if context.parent else [],
            lineage_tags=context.lineage_tags,
            timestamp=time.time(),
            witness_id=context.witness.id if context.witness else None
        )
        self.nodes[sha] = node
        self._save()
    
    def query_by_etymology(self, term: str) -> List[RhizomNode]:
        return [n for n in self.nodes.values() if term in n.etymology]
    
    def ancestry(self, sha: str, depth: int = 10) -> List[RhizomNode]:
        result = []
        current = self.nodes.get(sha)
        for _ in range(depth):
            if not current or not current.parents:
                break
            parent_sha = current.parents[0]
            current = self.nodes.get(parent_sha)
            if current:
                result.append(current)
        return result
```

---

## 5. File Structure

```
nucleus/ark/
├── __init__.py
├── cli.py                    # Entry point
├── core/
│   ├── repository.py         # ArkRepository
│   ├── commit.py             # Commit context
│   └── config.py             # ARK configuration
├── gates/
│   ├── inertia.py
│   ├── entelecheia.py
│   └── homeostasis.py
├── neural/
│   ├── adapter.py
│   ├── features.py
│   └── thrash_detector.py
├── rhizom/
│   ├── dag.py
│   ├── node.py
│   └── query.py
├── ribosome/
│   ├── gene.py
│   ├── clade.py
│   └── genome.py
├── synthesis/
│   ├── ltl_spec.py
│   ├── grammar.py
│   └── reactive.py
├── portal/
│   ├── ingest.py
│   ├── distill.py
│   └── crystallize.py
├── commands/
│   ├── init.py
│   ├── commit.py
│   └── ... (one per command)
└── tests/
    └── ...
```

---

## 6. Recommendations

1. **TypeScript for isogit layer**: Better type safety for git internals
2. **Python for CLI/gates**: Consistency with existing nucleus tools
3. **Bridge via subprocess**: Python CLI calls TypeScript core
4. **Tests first**: Write tests before implementation (Steps 25, 50, 75, 100, 125, 147)

---

*Logged: 2026-01-23 | Agent: Codex 5.2 XHigh | Protocol: PBTSO-ARK-SWARM-003*
