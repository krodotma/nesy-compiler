"""
Cartesian Genetic Programming (CGP) Implementation
===================================================

This module provides a complete CGP implementation for evolving
mathematical expressions and programs represented as directed acyclic graphs.

CGP represents programs as a grid of nodes where each node has a function
and connections to previous nodes or inputs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class CGPFunction:
    """
    Represents a function in the CGP function set.

    Attributes:
        name: Human-readable name of the function.
        arity: Number of inputs the function takes.
        func: The actual Python function to execute.
        symbol: Optional mathematical symbol for display.
    """
    name: str
    arity: int
    func: Callable[..., float]
    symbol: Optional[str] = None

    def __call__(self, *args: float) -> float:
        """Execute the function with given arguments."""
        try:
            result = self.func(*args[:self.arity])
            # Handle edge cases
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0


def _protected_div(a: float, b: float) -> float:
    """Protected division that returns 1.0 when dividing by zero."""
    if abs(b) < 1e-10:
        return 1.0
    return a / b


def _protected_log(x: float) -> float:
    """Protected logarithm that handles non-positive values."""
    if x <= 0:
        return 0.0
    return math.log(x)


def _protected_sqrt(x: float) -> float:
    """Protected square root that handles negative values."""
    return math.sqrt(abs(x))


def _protected_exp(x: float) -> float:
    """Protected exponential that clips extreme values."""
    x = max(-100, min(100, x))
    return math.exp(x)


# Standard function set for CGP
FUNCTION_SET: list[CGPFunction] = [
    CGPFunction("add", 2, lambda a, b: a + b, "+"),
    CGPFunction("sub", 2, lambda a, b: a - b, "-"),
    CGPFunction("mul", 2, lambda a, b: a * b, "*"),
    CGPFunction("div", 2, _protected_div, "/"),
    CGPFunction("sin", 1, math.sin, "sin"),
    CGPFunction("cos", 1, math.cos, "cos"),
    CGPFunction("exp", 1, _protected_exp, "exp"),
    CGPFunction("log", 1, _protected_log, "log"),
    CGPFunction("sqrt", 1, _protected_sqrt, "sqrt"),
    CGPFunction("abs", 1, abs, "abs"),
    CGPFunction("neg", 1, lambda x: -x, "-"),
    CGPFunction("pow2", 1, lambda x: x * x, "^2"),
    CGPFunction("max", 2, max, "max"),
    CGPFunction("min", 2, min, "min"),
    CGPFunction("tanh", 1, math.tanh, "tanh"),
]


@dataclass
class CGPNode:
    """
    A node in the CGP graph.

    Attributes:
        function_id: Index into the function set.
        inputs: List of indices pointing to previous nodes or inputs.
    """
    function_id: int
    inputs: list[int]

    def copy(self) -> CGPNode:
        """Create a deep copy of this node."""
        return CGPNode(
            function_id=self.function_id,
            inputs=self.inputs.copy()
        )


@dataclass
class CGPGenome:
    """
    A complete CGP genome representing a program.

    The genome consists of:
    - n_inputs: Number of input terminals
    - n_outputs: Number of output terminals
    - A grid of n_rows x n_cols nodes
    - Output genes pointing to nodes that produce outputs

    Nodes are indexed as:
    - 0 to n_inputs-1: Input terminals
    - n_inputs to n_inputs + n_rows*n_cols - 1: Computational nodes

    Attributes:
        n_inputs: Number of inputs to the program.
        n_outputs: Number of outputs from the program.
        n_rows: Number of rows in the node grid.
        n_cols: Number of columns in the node grid.
        nodes: List of computational nodes.
        output_genes: Indices of nodes connected to outputs.
        function_set: The set of functions available to nodes.
        levels_back: How many columns back a node can connect to (None = unlimited).
    """
    n_inputs: int
    n_outputs: int
    n_rows: int
    n_cols: int
    nodes: list[CGPNode] = field(default_factory=list)
    output_genes: list[int] = field(default_factory=list)
    function_set: list[CGPFunction] = field(default_factory=lambda: FUNCTION_SET.copy())
    levels_back: Optional[int] = None
    _fitness: Optional[float] = field(default=None, repr=False)
    _active_nodes: Optional[set[int]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize nodes if not provided."""
        if not self.nodes:
            self._initialize_random()

    def _initialize_random(self) -> None:
        """Initialize genome with random nodes and connections."""
        self.nodes = []

        for col in range(self.n_cols):
            for row in range(self.n_rows):
                node = self._create_random_node(col)
                self.nodes.append(node)

        # Initialize output genes
        max_gene = self.n_inputs + len(self.nodes) - 1
        self.output_genes = [
            random.randint(0, max_gene) for _ in range(self.n_outputs)
        ]

    def _create_random_node(self, col: int) -> CGPNode:
        """Create a random node for a given column."""
        func_id = random.randint(0, len(self.function_set) - 1)
        func = self.function_set[func_id]

        # Determine valid connection range
        if self.levels_back is not None:
            min_col = max(0, col - self.levels_back)
        else:
            min_col = 0

        # Calculate valid node indices for connections
        min_idx = 0
        max_idx = self.n_inputs + min_col * self.n_rows - 1
        if max_idx < 0:
            max_idx = self.n_inputs - 1

        # Create connections based on function arity
        inputs = []
        for _ in range(func.arity):
            if max_idx >= 0:
                inputs.append(random.randint(min_idx, max_idx))
            else:
                inputs.append(random.randint(0, self.n_inputs - 1))

        return CGPNode(function_id=func_id, inputs=inputs)

    def get_node_index(self, row: int, col: int) -> int:
        """Get the absolute index of a node at (row, col)."""
        return self.n_inputs + col * self.n_rows + row

    def get_node_position(self, idx: int) -> tuple[int, int]:
        """Get the (row, col) position of a node given its index."""
        if idx < self.n_inputs:
            raise ValueError(f"Index {idx} is an input, not a node")
        node_idx = idx - self.n_inputs
        col = node_idx // self.n_rows
        row = node_idx % self.n_rows
        return (row, col)

    def evaluate(self, inputs: list[float]) -> list[float]:
        """
        Evaluate the genome with given inputs.

        Args:
            inputs: List of input values.

        Returns:
            List of output values.
        """
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {len(inputs)}")

        # Build value cache
        values: dict[int, float] = {}
        for i, val in enumerate(inputs):
            values[i] = val

        # Evaluate nodes in topological order (column by column)
        for idx, node in enumerate(self.nodes):
            node_idx = self.n_inputs + idx
            func = self.function_set[node.function_id]
            args = [values.get(inp, 0.0) for inp in node.inputs[:func.arity]]
            values[node_idx] = func(*args)

        # Collect outputs
        return [values.get(out_idx, 0.0) for out_idx in self.output_genes]

    def get_active_nodes(self) -> set[int]:
        """
        Find all active nodes (nodes that contribute to outputs).

        Returns:
            Set of active node indices.
        """
        if self._active_nodes is not None:
            return self._active_nodes

        active: set[int] = set()
        to_process = list(self.output_genes)

        while to_process:
            idx = to_process.pop()
            if idx in active or idx < self.n_inputs:
                continue

            active.add(idx)
            node = self.nodes[idx - self.n_inputs]
            func = self.function_set[node.function_id]
            to_process.extend(node.inputs[:func.arity])

        self._active_nodes = active
        return active

    def copy(self) -> CGPGenome:
        """Create a deep copy of this genome."""
        new_genome = CGPGenome(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            nodes=[node.copy() for node in self.nodes],
            output_genes=self.output_genes.copy(),
            function_set=self.function_set,
            levels_back=self.levels_back
        )
        return new_genome

    def mutate(self, mutation_rate: float = 0.1) -> CGPGenome:
        """
        Create a mutated copy of this genome.

        Uses point mutation on genes (function IDs, connections, outputs).

        Args:
            mutation_rate: Probability of mutating each gene.

        Returns:
            A new mutated genome.
        """
        child = self.copy()
        child._fitness = None
        child._active_nodes = None

        # Mutate nodes
        for idx, node in enumerate(child.nodes):
            col = idx // self.n_rows

            # Mutate function
            if random.random() < mutation_rate:
                node.function_id = random.randint(0, len(self.function_set) - 1)

            # Mutate connections
            func = self.function_set[node.function_id]

            # Ensure inputs list has correct arity
            while len(node.inputs) < func.arity:
                node.inputs.append(0)

            for i in range(func.arity):
                if random.random() < mutation_rate:
                    # Calculate valid connection range
                    if self.levels_back is not None:
                        min_col = max(0, col - self.levels_back)
                    else:
                        min_col = 0
                    max_idx = self.n_inputs + min_col * self.n_rows - 1
                    if max_idx < 0:
                        max_idx = self.n_inputs - 1
                    node.inputs[i] = random.randint(0, max_idx)

        # Mutate output genes
        max_gene = self.n_inputs + len(self.nodes) - 1
        for i in range(len(child.output_genes)):
            if random.random() < mutation_rate:
                child.output_genes[i] = random.randint(0, max_gene)

        return child

    def to_expression(self, output_idx: int = 0) -> str:
        """
        Convert the genome to a human-readable expression string.

        Args:
            output_idx: Which output to convert.

        Returns:
            String representation of the expression.
        """
        def node_to_str(idx: int) -> str:
            if idx < self.n_inputs:
                return f"x{idx}"

            node = self.nodes[idx - self.n_inputs]
            func = self.function_set[node.function_id]
            args = [node_to_str(inp) for inp in node.inputs[:func.arity]]

            symbol = func.symbol or func.name
            if func.arity == 1:
                return f"{symbol}({args[0]})"
            elif func.arity == 2:
                if symbol in "+-*/":
                    return f"({args[0]} {symbol} {args[1]})"
                return f"{symbol}({args[0]}, {args[1]})"
            return f"{symbol}({', '.join(args)})"

        return node_to_str(self.output_genes[output_idx])

    @property
    def fitness(self) -> Optional[float]:
        """Get the fitness value."""
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        """Set the fitness value."""
        self._fitness = value


def cgp_evolve(
    fitness_func: Callable[[CGPGenome], float],
    n_inputs: int = 1,
    n_outputs: int = 1,
    n_rows: int = 1,
    n_cols: int = 10,
    population_size: int = 5,
    n_generations: int = 100,
    mutation_rate: float = 0.1,
    function_set: Optional[list[CGPFunction]] = None,
    levels_back: Optional[int] = None,
    elitism: int = 1,
    verbose: bool = False
) -> tuple[CGPGenome, list[float]]:
    """
    Evolve CGP genomes using (1+lambda) evolutionary strategy.

    Args:
        fitness_func: Function that evaluates genome fitness (higher is better).
        n_inputs: Number of inputs.
        n_outputs: Number of outputs.
        n_rows: Number of rows in the grid.
        n_cols: Number of columns in the grid.
        population_size: Number of children per generation.
        n_generations: Number of generations to evolve.
        mutation_rate: Probability of mutating each gene.
        function_set: Custom function set (uses default if None).
        levels_back: Connection locality constraint.
        elitism: Number of best individuals to preserve.
        verbose: Print progress information.

    Returns:
        Tuple of (best genome, fitness history).
    """
    if function_set is None:
        function_set = FUNCTION_SET.copy()

    # Initialize population
    population: list[CGPGenome] = []
    for _ in range(population_size):
        genome = CGPGenome(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_rows=n_rows,
            n_cols=n_cols,
            function_set=function_set,
            levels_back=levels_back
        )
        genome.fitness = fitness_func(genome)
        population.append(genome)

    # Sort by fitness
    population.sort(key=lambda g: g.fitness or float('-inf'), reverse=True)

    best_genome = population[0].copy()
    best_genome.fitness = population[0].fitness
    fitness_history: list[float] = [best_genome.fitness or 0.0]

    for gen in range(n_generations):
        # Create next generation
        next_pop: list[CGPGenome] = []

        # Elitism: keep best individuals
        for i in range(min(elitism, len(population))):
            elite = population[i].copy()
            elite.fitness = population[i].fitness
            next_pop.append(elite)

        # Generate children through mutation
        parent_idx = 0
        while len(next_pop) < population_size:
            parent = population[parent_idx % len(population)]
            child = parent.mutate(mutation_rate)
            child.fitness = fitness_func(child)
            next_pop.append(child)
            parent_idx += 1

        population = next_pop
        population.sort(key=lambda g: g.fitness or float('-inf'), reverse=True)

        # Update best
        if (population[0].fitness or 0) > (best_genome.fitness or 0):
            best_genome = population[0].copy()
            best_genome.fitness = population[0].fitness

        fitness_history.append(best_genome.fitness or 0.0)

        if verbose and (gen + 1) % 10 == 0:
            print(f"Generation {gen + 1}: Best fitness = {best_genome.fitness:.6f}")

    return best_genome, fitness_history
