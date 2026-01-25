"""
Theia EGGP — Evolving Graph Grammar Programs.

Port of EGGP concepts for graph-based program evolution.

From graph_evolutionary_programming_distillation.md:
    "EGGP for Refactoring — Graph programs = refactoring rules"

Key concepts:
    - Graph programs as AST transformation rules
    - Evolution of refactoring patterns
    - Multi-language support via typed ASTs
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import random


@dataclass
class GraphNode:
    """Node in a graph program."""
    id: str
    node_type: str  # "literal", "variable", "operation", "pattern"
    value: Any = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge in a graph program."""
    source: str  # Node ID
    target: str  # Node ID
    edge_type: str  # "data", "control", "matches"
    label: str = ""


@dataclass
class GraphProgram:
    """
    A graph program representing a transformation rule.
    
    Structure:
        LHS (pattern to match) --transform--> RHS (replacement)
    """
    id: str
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    lhs_node_ids: List[str] = field(default_factory=list)
    rhs_node_ids: List[str] = field(default_factory=list)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def add_node(self, node: GraphNode) -> None:
        """Add node to program."""
        self.nodes.append(node)
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add edge to program."""
        self.edges.append(edge)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "nodes": [
                {"id": n.id, "type": n.node_type, "value": n.value}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "type": e.edge_type}
                for e in self.edges
            ],
            "lhs": self.lhs_node_ids,
            "rhs": self.rhs_node_ids,
        }


# =============================================================================
# GRAPH GRAMMAR
# =============================================================================

class GraphGrammar:
    """
    Grammar for graph programs.
    
    Defines the space of valid graph transformations.
    """
    
    def __init__(self):
        self.node_types = [
            "literal", "variable", "operation",
            "pattern", "wildcard", "sequence",
        ]
        self.edge_types = [
            "data", "control", "matches", "produces",
        ]
        self.operations = [
            "add", "mul", "sub", "div",
            "and", "or", "not",
            "eq", "lt", "gt",
            "call", "return",
        ]
    
    def random_node(self, prefix: str = "n") -> GraphNode:
        """Generate random valid node."""
        node_type = random.choice(self.node_types)
        
        if node_type == "literal":
            value = random.choice([0, 1, 2, -1, True, False, "x"])
        elif node_type == "operation":
            value = random.choice(self.operations)
        else:
            value = None
        
        return GraphNode(
            id=f"{prefix}_{random.randint(0, 9999):04d}",
            node_type=node_type,
            value=value,
        )
    
    def random_edge(self, source: str, target: str) -> GraphEdge:
        """Generate random valid edge."""
        return GraphEdge(
            source=source,
            target=target,
            edge_type=random.choice(self.edge_types),
        )
    
    def random_program(self, n_nodes: int = 5) -> GraphProgram:
        """Generate random graph program."""
        program = GraphProgram(id=f"gp_{random.randint(0, 9999):04d}")
        
        # Create nodes
        for i in range(n_nodes):
            node = self.random_node(f"n{i}")
            program.add_node(node)
        
        # Create edges (random DAG-like structure)
        for i in range(n_nodes - 1):
            j = random.randint(i + 1, n_nodes - 1)
            edge = self.random_edge(
                program.nodes[i].id,
                program.nodes[j].id,
            )
            program.add_edge(edge)
        
        # Assign LHS/RHS
        mid = n_nodes // 2
        program.lhs_node_ids = [n.id for n in program.nodes[:mid]]
        program.rhs_node_ids = [n.id for n in program.nodes[mid:]]
        
        return program


# =============================================================================
# EGGP EVOLUTION
# =============================================================================

class EGGPEvolver:
    """
    Evolving Graph Grammar Programs.
    
    Uses evolutionary operators to evolve refactoring rules.
    """
    
    def __init__(self, grammar: Optional[GraphGrammar] = None):
        self.grammar = grammar or GraphGrammar()
        self.population: List[GraphProgram] = []
        self.generation = 0
    
    def initialize(self, pop_size: int = 20) -> None:
        """Initialize random population."""
        self.population = [
            self.grammar.random_program()
            for _ in range(pop_size)
        ]
    
    def mutate(self, program: GraphProgram) -> GraphProgram:
        """Mutate a graph program."""
        import copy
        mutant = copy.deepcopy(program)
        mutant.id = f"gp_{random.randint(0, 9999):04d}"
        
        mutation_type = random.choice([
            "add_node", "remove_node", "change_node",
            "add_edge", "remove_edge",
        ])
        
        if mutation_type == "add_node":
            new_node = self.grammar.random_node()
            mutant.add_node(new_node)
            if mutant.nodes and random.random() < 0.5:
                # Connect to existing node
                existing = random.choice(mutant.nodes[:-1])
                edge = self.grammar.random_edge(existing.id, new_node.id)
                mutant.add_edge(edge)
        
        elif mutation_type == "remove_node" and len(mutant.nodes) > 2:
            removed = mutant.nodes.pop(random.randint(0, len(mutant.nodes) - 1))
            # Remove associated edges
            mutant.edges = [
                e for e in mutant.edges
                if e.source != removed.id and e.target != removed.id
            ]
        
        elif mutation_type == "change_node" and mutant.nodes:
            node = random.choice(mutant.nodes)
            node.node_type = random.choice(self.grammar.node_types)
            if node.node_type == "operation":
                node.value = random.choice(self.grammar.operations)
        
        elif mutation_type == "add_edge" and len(mutant.nodes) >= 2:
            src = random.choice(mutant.nodes)
            tgt = random.choice([n for n in mutant.nodes if n.id != src.id])
            edge = self.grammar.random_edge(src.id, tgt.id)
            mutant.add_edge(edge)
        
        elif mutation_type == "remove_edge" and mutant.edges:
            mutant.edges.pop(random.randint(0, len(mutant.edges) - 1))
        
        return mutant
    
    def crossover(
        self,
        parent1: GraphProgram,
        parent2: GraphProgram,
    ) -> GraphProgram:
        """Crossover two graph programs."""
        import copy
        
        child = GraphProgram(id=f"gp_{random.randint(0, 9999):04d}")
        
        # Take nodes from both parents
        mid1 = len(parent1.nodes) // 2
        mid2 = len(parent2.nodes) // 2
        
        for node in parent1.nodes[:mid1]:
            child.add_node(copy.deepcopy(node))
        for node in parent2.nodes[mid2:]:
            child.add_node(copy.deepcopy(node))
        
        # Create new edges connecting them
        if len(child.nodes) >= 2:
            src = child.nodes[0]
            tgt = child.nodes[-1]
            child.add_edge(self.grammar.random_edge(src.id, tgt.id))
        
        return child
    
    def evaluate(
        self,
        program: GraphProgram,
        fitness_fn: Optional[Callable] = None,
    ) -> float:
        """Evaluate fitness of a graph program."""
        if fitness_fn:
            return fitness_fn(program)
        
        # Default: size-based fitness (prefer smaller programs)
        return 1.0 / (1.0 + len(program.nodes) + len(program.edges))
    
    def evolve_generation(
        self,
        fitness_fn: Optional[Callable] = None,
        tournament_size: int = 3,
    ) -> GraphProgram:
        """Evolve one generation. Returns best individual."""
        # Evaluate fitness
        fitness_scores = [
            (self.evaluate(p, fitness_fn), p)
            for p in self.population
        ]
        fitness_scores.sort(key=lambda x: -x[0])
        
        # Tournament selection + mutation/crossover
        new_pop = []
        
        # Elitism: keep best
        new_pop.append(fitness_scores[0][1])
        
        while len(new_pop) < len(self.population):
            # Tournament selection
            tournament = random.sample(fitness_scores, tournament_size)
            winner = max(tournament, key=lambda x: x[0])[1]
            
            if random.random() < 0.5:
                # Mutation
                new_pop.append(self.mutate(winner))
            else:
                # Crossover
                tournament2 = random.sample(fitness_scores, tournament_size)
                parent2 = max(tournament2, key=lambda x: x[0])[1]
                new_pop.append(self.crossover(winner, parent2))
        
        self.population = new_pop
        self.generation += 1
        
        return fitness_scores[0][1]


# =============================================================================
# METAGRAMMAR (Stub for multi-language support)
# =============================================================================

@dataclass
class Metagrammar:
    """
    Metagrammar for multi-language AST transformation.
    
    Stub for future MAGE (Multi-lang AST Grammar Evolution) support.
    """
    name: str = "default"
    source_languages: List[str] = field(default_factory=lambda: ["python"])
    target_languages: List[str] = field(default_factory=lambda: ["python"])
    type_mappings: Dict[str, str] = field(default_factory=dict)
    
    def can_transform(self, source_lang: str, target_lang: str) -> bool:
        """Check if transformation is supported."""
        return (
            source_lang in self.source_languages and
            target_lang in self.target_languages
        )


__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphProgram",
    "GraphGrammar",
    "EGGPEvolver",
    "Metagrammar",
]
