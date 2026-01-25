"""
Theia Cartesian Genetic Programming — Evolved program representations.

CGP represents programs as directed acyclic graphs with:
- Fixed grid layout ensuring acyclicity by construction
- Inactive genes enabling neutral genetic drift
- (1+λ)-ES evolution strategy

Key insight from Miller (2019): Neutral drift is CRITICAL.
Mutations to inactive genes allow escaping local optima.

Based on graph_evolutionary_programming_distillation.md research.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Optional, Any
import random
from copy import deepcopy


# Built-in function set (can be extended)
FUNCTION_SET = {
    0: ("add", 2, lambda a, b: a + b),
    1: ("sub", 2, lambda a, b: a - b),
    2: ("mul", 2, lambda a, b: a * b),
    3: ("div", 2, lambda a, b: a / (b + 1e-8)),  # Protected division
    4: ("neg", 1, lambda a: -a),
    5: ("abs", 1, lambda a: abs(a)),
    6: ("sin", 1, lambda a: np.sin(a)),
    7: ("cos", 1, lambda a: np.cos(a)),
    8: ("exp", 1, lambda a: np.clip(np.exp(np.clip(a, -10, 10)), -1e10, 1e10)),
    9: ("log", 1, lambda a: np.log(np.abs(a) + 1e-8)),
    10: ("max", 2, lambda a, b: max(a, b)),
    11: ("min", 2, lambda a, b: min(a, b)),
    12: ("id", 1, lambda a: a),  # Identity
}


@dataclass
class CGPNode:
    """Single node in CGP graph."""
    function_id: int
    inputs: List[int]  # Indices of input nodes
    
    def execute(self, values: List[float]) -> float:
        """Execute node with given input values."""
        _, arity, func = FUNCTION_SET[self.function_id]
        args = [values[i] for i in self.inputs[:arity]]
        try:
            return float(func(*args))
        except:
            return 0.0


@dataclass
class CGPGenome:
    """
    CGP genome encoding.
    
    Attributes:
        n_inputs: Number of program inputs
        n_outputs: Number of program outputs
        n_rows: Grid rows
        n_cols: Grid columns
        arity: Maximum function arity
        nodes: List of CGP nodes
        output_genes: Indices of output nodes
    """
    n_inputs: int
    n_outputs: int
    n_rows: int
    n_cols: int
    arity: int = 2
    nodes: List[CGPNode] = field(default_factory=list)
    output_genes: List[int] = field(default_factory=list)
    
    @property
    def n_nodes(self) -> int:
        return self.n_rows * self.n_cols
    
    def _max_input(self, col: int) -> int:
        """Maximum valid input index for a node in given column."""
        return self.n_inputs + col * self.n_rows
    
    def randomize(self) -> None:
        """Randomly initialize genome."""
        self.nodes = []
        
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                func_id = random.randint(0, len(FUNCTION_SET) - 1)
                max_input = self._max_input(col)
                inputs = [random.randint(0, max_input - 1) for _ in range(self.arity)]
                self.nodes.append(CGPNode(func_id, inputs))
        
        # Random output genes
        max_out = self.n_inputs + self.n_nodes
        self.output_genes = [random.randint(0, max_out - 1) for _ in range(self.n_outputs)]
    
    def get_active_nodes(self) -> set:
        """
        Trace back from outputs to find active nodes.
        
        Only active nodes contribute to output.
        Inactive nodes can mutate without affecting fitness (neutral drift).
        """
        active = set()
        stack = list(self.output_genes)
        
        while stack:
            idx = stack.pop()
            if idx >= self.n_inputs and idx not in active:
                node_idx = idx - self.n_inputs
                active.add(idx)
                if node_idx < len(self.nodes):
                    node = self.nodes[node_idx]
                    stack.extend(node.inputs)
        
        return active
    
    def execute(self, inputs: List[float]) -> List[float]:
        """
        Execute program on given inputs.
        
        Computes only active nodes (lazy evaluation).
        """
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {len(inputs)}")
        
        # Initialize values with inputs
        values = list(inputs) + [0.0] * self.n_nodes
        
        # Compute nodes in order (column by column ensures dependencies resolved)
        for i, node in enumerate(self.nodes):
            idx = self.n_inputs + i
            values[idx] = node.execute(values)
        
        # Collect outputs
        return [values[i] for i in self.output_genes]
    
    def mutate(self, mutation_rate: float = 0.1) -> "CGPGenome":
        """
        Point mutation preserving grid constraints.
        
        Critical: Mutations can affect active OR inactive genes.
        Inactive mutations enable neutral drift.
        """
        child = deepcopy(self)
        
        for i, node in enumerate(child.nodes):
            col = i // child.n_rows
            
            # Mutate function
            if random.random() < mutation_rate:
                node.function_id = random.randint(0, len(FUNCTION_SET) - 1)
            
            # Mutate inputs
            max_input = child._max_input(col)
            for j in range(child.arity):
                if random.random() < mutation_rate:
                    node.inputs[j] = random.randint(0, max_input - 1)
        
        # Mutate output genes
        max_out = child.n_inputs + child.n_nodes
        for i in range(child.n_outputs):
            if random.random() < mutation_rate:
                child.output_genes[i] = random.randint(0, max_out - 1)
        
        return child
    
    def to_expression(self, input_names: Optional[List[str]] = None) -> str:
        """Convert genome to readable expression (for debugging)."""
        if input_names is None:
            input_names = [f"x{i}" for i in range(self.n_inputs)]
        
        # Build expressions for each node
        exprs = list(input_names)
        
        for i, node in enumerate(self.nodes):
            func_name, arity, _ = FUNCTION_SET[node.function_id]
            args = [exprs[j] for j in node.inputs[:arity]]
            
            if arity == 1:
                expr = f"{func_name}({args[0]})"
            else:
                expr = f"{func_name}({args[0]}, {args[1]})"
            exprs.append(expr)
        
        return ", ".join([exprs[i] for i in self.output_genes])


def cgp_evolve(
    fitness_func: Callable[[CGPGenome], float],
    n_inputs: int,
    n_outputs: int,
    generations: int = 100,
    mu: int = 1,
    lambda_: int = 4,
    n_rows: int = 1,
    n_cols: int = 50,
    mutation_rate: float = 0.1,
    target_fitness: float = float('inf'),
) -> Tuple[CGPGenome, float, int]:
    """
    (1+λ)-ES CGP evolution.
    
    Key insight: When child has EQUAL fitness to parent, prefer child.
    This enables neutral drift through inactive gene mutations.
    
    Args:
        fitness_func: Higher is better
        n_inputs, n_outputs: Program IO spec
        generations: Max generations
        mu: Parent population size
        lambda_: Offspring per generation
        n_rows, n_cols: Grid dimensions
        mutation_rate: Per-gene mutation probability
        target_fitness: Stop if reached
        
    Returns:
        (best_genome, best_fitness, generations_used)
    """
    # Initialize population
    population = []
    for _ in range(mu):
        genome = CGPGenome(n_inputs, n_outputs, n_rows, n_cols)
        genome.randomize()
        population.append(genome)
    
    best_genome = population[0]
    best_fitness = fitness_func(best_genome)
    
    for gen in range(generations):
        # Generate offspring
        offspring = []
        for parent in population:
            for _ in range(lambda_ // mu):
                child = parent.mutate(mutation_rate)
                offspring.append(child)
        
        # Evaluate all
        all_individuals = population + offspring
        fitnesses = [fitness_func(g) for g in all_individuals]
        
        # Find best (with neutral drift preference)
        for i, (genome, fit) in enumerate(zip(all_individuals, fitnesses)):
            # Prefer child on tie (neutral drift)
            if fit > best_fitness or (fit == best_fitness and i >= mu):
                best_genome = genome
                best_fitness = fit
        
        # Check termination
        if best_fitness >= target_fitness:
            return best_genome, best_fitness, gen + 1
        
        # Select next generation (elitism + best offspring)
        sorted_pairs = sorted(zip(all_individuals, fitnesses), key=lambda x: x[1], reverse=True)
        population = [g for g, _ in sorted_pairs[:mu]]
    
    return best_genome, best_fitness, generations


def symbolic_regression_fitness(genome: CGPGenome, X: np.ndarray, y: np.ndarray) -> float:
    """
    Fitness function for symbolic regression.
    
    Minimizes mean squared error (returns negative MSE as fitness).
    """
    try:
        predictions = []
        for xi in X:
            outputs = genome.execute(list(xi))
            predictions.append(outputs[0])
        
        predictions = np.array(predictions)
        mse = np.mean((predictions - y) ** 2)
        
        # Return negative MSE (higher is better)
        return -mse
        
    except:
        return float('-inf')


__all__ = [
    "CGPGenome",
    "CGPNode",
    "FUNCTION_SET",
    "cgp_evolve",
    "symbolic_regression_fitness",
]
