#!/usr/bin/env python3
"""
ohmyopencode.py - The Mutator Daemon (Phase 3.1)
Part of the CMP (Cosmic Microwave Background) Evolution Engine.

Purpose:
- Monitors the code for "fitness" (via cmp_large.py)
- Proposes valid mutations (refactors, optimizations)
- Applies HGT (Horizontal Gene Transfer) between compatible components

Protocol: DKIN v28 / PAIP v15
Ring: 1 (Operator)
"""

import os
import sys
import time
import json
import random
import ast
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configuration
MUTATION_RATE = 0.05
FITNESS_THRESHOLD = 0.85
MAX_GENERATIONS = 100

class CodeMutator:
    """
    The engine of evolution. Parses Python ASTs and proposes
    semantically valid transformations.
    """
    
    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir)
        self.ledger_path = self.target_dir / ".pluribus" / "evolution" / "mutation_ledger.json"
        
    def scan_clade(self) -> List[Path]:
        """Scan directory for mutable DNA (Python files)."""
        return list(self.target_dir.rglob("*.py"))
    
    def calculate_complexity(self, content: str) -> int:
        """Rough cyclomatic complexity estimator."""
        try:
            tree = ast.parse(content)
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                    complexity += 1
            return complexity
        except SyntaxError:
            return 999  # Penalize broken code

    def propose_mutation(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Analyze a file and propose a beneficial mutation.
        Current strategies:
        1. Docstring injection (MutationType: DOCUMENT)
        2. Complexity reduction (MutationType: SIMPLIFY) - *Placeholder*
        3. Type hint addition (MutationType: TYPING) - *Placeholder*
        """
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Simple fitness check: Missing docstrings?
            tree = ast.parse(content)
            
            # Check for module docstring
            if not ast.get_docstring(tree):
                return {
                    "type": "DOCUMENT",
                    "target": str(file_path),
                    "reason": "Missing module docstring",
                    "suggestion": '"""\nAuto-generated module docstring.\nTimestamp: {}\n"""\n'.format(time.time())
                }
            
            # Check complexity
            complexity = self.calculate_complexity(content)
            if complexity > 10:
                return {
                    "type": "SIMPLIFY",
                    "target": str(file_path),
                    "reason": f"High cyclomatic complexity ({complexity})",
                    "suggestion": "# TODO: Refactor this high-complexity module"
                }

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
        return None

    def record_mutation(self, mutation: Dict[str, Any]):
        """Log the proposed mutation to the ledger."""
        if not self.ledger_path.parent.exists():
            self.ledger_path.parent.mkdir(parents=True)
        
        ledger = []
        if self.ledger_path.exists():
            with open(self.ledger_path, "r") as f:
                try:
                    ledger = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        mutation["status"] = "PROPOSED"
        mutation["timestamp"] = time.time()
        ledger.append(mutation)
        
        with open(self.ledger_path, "w") as f:
            json.dump(ledger, f, indent=2)
            
    def run_cycle(self):
        """Run one evolutionary cycle."""
        targets = self.scan_clade()
        print(f"[Mutator] Scanning {len(targets)} eligible hosts...")
        
        mutations = []
        for target in targets:
            mutation = self.propose_mutation(target)
            if mutation:
                print(f"[Mutator] Proposed {mutation['type']} for {target.name}")
                mutations.append(mutation)
                self.record_mutation(mutation)
        
        print(f"[Mutator] Cycle complete. {len(mutations)} mutations proposed.")

if __name__ == "__main__":
    print("Initializing OhMyOpenCode Mutator Daemon v1.0...")
    target = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    mutator = CodeMutator(target)
    mutator.run_cycle()
