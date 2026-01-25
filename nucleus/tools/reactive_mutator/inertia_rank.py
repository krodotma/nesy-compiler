
# inertia_rank.py - PageRank-based Stability Metric
# Part of Reactive Evolution v1
# Implements: DNA Axiom "Inertia"

import networkx as nx
import ast
import os
from pathlib import Path
from typing import Dict

class InertiaRank:
    """
    Calculates the 'Inertia' of codebase modules using PageRank.
    High Rank = High Dependencies = High Inertia (Do Not Touch).
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.graph = nx.DiGraph()
        self._build_dependency_graph()
        self.ranks = self._calculate_ranks()

    def get_rank(self, filepath: str) -> float:
        """
        Returns the normalized Inertia Rank [0.0, 1.0].
        """
        # Normalize path to relative string key
        try:
            rel_path = str(Path(filepath).relative_to(self.root_dir))
        except ValueError:
            rel_path = filepath # Fallback
            
        return self.ranks.get(rel_path, 0.0)

    def _build_dependency_graph(self):
        """
        Walks the codebase and builds the import graph.
        Edge A -> B means 'A imports B'.
        Wait, PageRank means 'Importance flows from A to B'.
        If A imports B, A depends on B. B is critical for A.
        So A 'votes' for B's importance.
        Edge Direction: A -> B.
        """
        for py_file in self.root_dir.rglob("*.py"):
            rel_path = str(py_file.relative_to(self.root_dir))
            self.graph.add_node(rel_path)
            
            imports = self._extract_imports(py_file)
            for imp in imports:
                # Naive resolution: map import name to potential file path
                # In full implementation, we need robust module resolution
                target_path = self._resolve_import(imp)
                if target_path:
                    self.graph.add_edge(rel_path, target_path)

    def _extract_imports(self, filepath: Path) -> set:
        try:
            with open(filepath, "r") as f:
                tree = ast.parse(f.read())
        except Exception:
            return set()
            
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        return imports

    def _resolve_import(self, module_name: str) -> str:
        # Very simple heuristic resolution for MVP
        possible_path = module_name.replace(".", "/") + ".py"
        full_path = self.root_dir / possible_path
        if full_path.exists():
            return str(possible_path)
        return None

    def _calculate_ranks(self) -> Dict[str, float]:
        if not self.graph.nodes:
            return {}
        try:
            return nx.pagerank(self.graph, alpha=0.85)
        except Exception:
            # Fallback for empty/disconnected graphs
            return {n: 0.0 for n in self.graph.nodes}
