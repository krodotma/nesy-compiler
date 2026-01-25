
# Agent Log: Codex (Synthesizer)
## Mission: Program Synthesis & Logic
**Identity**: Ring 3 Synthesizer | **Mode**: CodeGen | **Session**: PBTSO-001

### 1. Concrete SyGuS Implementation
Based on Architect's "Grammar Filter", here is the Python implementation for the `GrammarFilter` and `InertiaRank`.

### 2. `grammar_filter.py` Stub
```python
import ast

class GrammarFilter:
    """
    Mark Santolucito's SyGuS concept applied to Python AST.
    Restricts mutations to a 'Safe Grammar'.
    """
    
    ALLOWED_TRANSFORMS = {
        "extract_method",
        "intro_variable",
        "add_type_hint",
        "clean_imports"
    }

    def check(self, source_code: str, mutation_type: str) -> bool:
        if mutation_type not in self.ALLOWED_TRANSFORMS:
            return False
            
        tree = ast.parse(source_code)
        # SyGuS Constraint: Logic Preservation
        # Ensure no logical side-effects (heuristic)
        if self._detects_logic_shift(tree):
            return False
            
        return True

    def _detects_logic_shift(self, tree: ast.AST) -> bool:
        # Placeholder for SMT solver / verification
        # For now, simplistic check: did we lose a return statement?
        return False
```

### 3. `inertia_rank.py` Stub
```python
import networkx as nx
from pathlib import Path

class InertiaRank:
    """
    PageRank-based Inertia Calculator.
    High Rank = High Inertia (DO NOT TOUCH).
    """
    
    def __init__(self, root: str):
        self.root = Path(root)
        self.graph = nx.DiGraph()

    def build_graph(self):
        # Walk files, parse imports
        for path in self.root.rglob("*.py"):
            self.graph.add_node(str(path))
            imports = self._parse_imports(path)
            for imp in imports:
                # Edge: Importer -> Importee
                # Wait! Dependency flows A -> B.
                # If A imports B, A depends on B. 
                # B supports A.
                # Inertia means "How many depend on me?"
                # So we want In-Degree (or PageRank on Dependency Graph).
                # Edge direction: A imports B => Edge A->B?
                # Standard PR: Vote flows with link.
                # If A links to B, A votes for B.
                # So yes, A -> B. B gets "score" from A.
                self.graph.add_edge(str(path), imp)
    
    def calculate(self):
        return nx.pagerank(self.graph, alpha=0.85)
```

### 4. Integration into `ohmyopencode.py`
```python
# In mutator loop:
inertia = InertiaRank(root).calculate()
filter = GrammarFilter()

for file in files:
    if inertia[file] > 0.05:
        print(f"Skipping {file}: Inertia Too High ({inertia[file]})")
        continue
        
    mutation = propose(file)
    if not filter.check(file, mutation.type):
        print(f"Skipping {file}: Mutation violates Grammar")
        continue

    # Apply
```
