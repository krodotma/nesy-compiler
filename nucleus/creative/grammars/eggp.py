"""
Ephemeral Graph Genetic Programming (EGGP) Implementation
==========================================================

This module provides EGGP, an extension of genetic programming that uses
ephemeral graphs - dynamically constructed expression graphs that can
represent programs with shared subexpressions and cycles.

EGGP is particularly suited for:
- Evolving mathematical expressions with shared structure
- Discovering reusable program components
- Compact representation of complex programs
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Union


class NodeType(Enum):
    """Types of nodes in an EGGP graph."""
    INPUT = auto()      # Input terminal
    CONSTANT = auto()   # Ephemeral constant
    FUNCTION = auto()   # Function application
    OUTPUT = auto()     # Output node


@dataclass
class GraphNode:
    """
    A node in an EGGP expression graph.

    Attributes:
        node_id: Unique identifier for this node.
        node_type: Type of the node (input, constant, function, output).
        value: For constants, the numeric value. For functions, the function name.
        inputs: List of node IDs that feed into this node.
        arity: Number of inputs required (for function nodes).
        label: Optional human-readable label.
    """
    node_id: int
    node_type: NodeType
    value: Union[float, str, None] = None
    inputs: list[int] = field(default_factory=list)
    arity: int = 0
    label: Optional[str] = None

    def copy(self) -> GraphNode:
        """Create a deep copy of this node."""
        return GraphNode(
            node_id=self.node_id,
            node_type=self.node_type,
            value=self.value,
            inputs=self.inputs.copy(),
            arity=self.arity,
            label=self.label
        )

    def is_terminal(self) -> bool:
        """Check if this node is a terminal (input or constant)."""
        return self.node_type in (NodeType.INPUT, NodeType.CONSTANT)


# Built-in function definitions
EGGP_FUNCTIONS: dict[str, tuple[int, Callable[..., float]]] = {
    "add": (2, lambda a, b: a + b),
    "sub": (2, lambda a, b: a - b),
    "mul": (2, lambda a, b: a * b),
    "div": (2, lambda a, b: a / b if abs(b) > 1e-10 else 1.0),
    "sin": (1, math.sin),
    "cos": (1, math.cos),
    "exp": (1, lambda x: math.exp(max(-100, min(100, x)))),
    "log": (1, lambda x: math.log(x) if x > 0 else 0.0),
    "sqrt": (1, lambda x: math.sqrt(abs(x))),
    "pow": (2, lambda a, b: math.pow(abs(a), b) if abs(b) < 10 else a),
    "neg": (1, lambda x: -x),
    "abs": (1, abs),
    "max": (2, max),
    "min": (2, min),
    "tanh": (1, math.tanh),
    "if_pos": (3, lambda cond, a, b: a if cond >= 0 else b),
}


@dataclass
class GraphProgram:
    """
    A complete EGGP program represented as a directed graph.

    The graph consists of nodes connected by edges. Each node can have
    multiple inputs (edges from other nodes) and can be referenced by
    multiple other nodes (shared subexpressions).

    Attributes:
        nodes: Dictionary mapping node IDs to GraphNode objects.
        input_ids: IDs of input nodes.
        output_ids: IDs of output nodes.
        functions: Available function definitions.
        next_id: Counter for generating unique node IDs.
    """
    nodes: dict[int, GraphNode] = field(default_factory=dict)
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    functions: dict[str, tuple[int, Callable[..., float]]] = field(
        default_factory=lambda: EGGP_FUNCTIONS.copy()
    )
    next_id: int = 0
    _fitness: Optional[float] = field(default=None, repr=False)

    def add_node(self, node_type: NodeType, value: Any = None,
                 inputs: Optional[list[int]] = None, label: Optional[str] = None) -> int:
        """
        Add a new node to the graph.

        Args:
            node_type: Type of node to create.
            value: Value for constant nodes or function name for function nodes.
            inputs: List of input node IDs.
            label: Optional label for the node.

        Returns:
            ID of the newly created node.
        """
        node_id = self.next_id
        self.next_id += 1

        arity = 0
        if node_type == NodeType.FUNCTION and isinstance(value, str):
            if value in self.functions:
                arity = self.functions[value][0]

        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            value=value,
            inputs=inputs or [],
            arity=arity,
            label=label
        )

        self.nodes[node_id] = node

        if node_type == NodeType.INPUT:
            self.input_ids.append(node_id)
        elif node_type == NodeType.OUTPUT:
            self.output_ids.append(node_id)

        return node_id

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the graph."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            if node_id in self.input_ids:
                self.input_ids.remove(node_id)
            if node_id in self.output_ids:
                self.output_ids.remove(node_id)

    def connect(self, from_id: int, to_id: int, input_idx: int = 0) -> bool:
        """
        Connect one node to another.

        Args:
            from_id: Source node ID.
            to_id: Target node ID.
            input_idx: Which input slot to use on the target.

        Returns:
            True if connection was made, False otherwise.
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return False

        target = self.nodes[to_id]

        # Extend inputs list if needed
        while len(target.inputs) <= input_idx:
            target.inputs.append(-1)

        target.inputs[input_idx] = from_id
        return True

    def evaluate(self, inputs: list[float]) -> list[float]:
        """
        Evaluate the program with given inputs.

        Args:
            inputs: List of input values.

        Returns:
            List of output values.
        """
        if len(inputs) != len(self.input_ids):
            raise ValueError(
                f"Expected {len(self.input_ids)} inputs, got {len(inputs)}"
            )

        # Build value cache
        cache: dict[int, float] = {}

        # Set input values
        for idx, node_id in enumerate(self.input_ids):
            cache[node_id] = inputs[idx]

        def eval_node(node_id: int, visited: set[int]) -> float:
            """Recursively evaluate a node."""
            if node_id in cache:
                return cache[node_id]

            if node_id in visited:
                # Cycle detected, return 0
                return 0.0

            if node_id not in self.nodes:
                return 0.0

            visited.add(node_id)
            node = self.nodes[node_id]

            if node.node_type == NodeType.CONSTANT:
                result = float(node.value) if node.value is not None else 0.0
            elif node.node_type == NodeType.FUNCTION:
                func_name = str(node.value)
                if func_name not in self.functions:
                    result = 0.0
                else:
                    arity, func = self.functions[func_name]
                    args = []
                    for i in range(arity):
                        if i < len(node.inputs) and node.inputs[i] >= 0:
                            args.append(eval_node(node.inputs[i], visited.copy()))
                        else:
                            args.append(0.0)
                    try:
                        result = func(*args)
                        if math.isnan(result) or math.isinf(result):
                            result = 0.0
                    except Exception:
                        result = 0.0
            elif node.node_type == NodeType.OUTPUT:
                if node.inputs and node.inputs[0] >= 0:
                    result = eval_node(node.inputs[0], visited)
                else:
                    result = 0.0
            else:
                result = 0.0

            cache[node_id] = result
            visited.discard(node_id)
            return result

        # Evaluate outputs
        outputs = []
        for out_id in self.output_ids:
            outputs.append(eval_node(out_id, set()))

        return outputs

    def copy(self) -> GraphProgram:
        """Create a deep copy of this program."""
        new_program = GraphProgram(
            nodes={nid: node.copy() for nid, node in self.nodes.items()},
            input_ids=self.input_ids.copy(),
            output_ids=self.output_ids.copy(),
            functions=self.functions,
            next_id=self.next_id
        )
        return new_program

    def get_active_nodes(self) -> set[int]:
        """Find all nodes that contribute to outputs."""
        active: set[int] = set()
        to_process = self.output_ids.copy()

        while to_process:
            node_id = to_process.pop()
            if node_id in active or node_id not in self.nodes:
                continue

            active.add(node_id)
            node = self.nodes[node_id]
            for inp in node.inputs:
                if inp >= 0:
                    to_process.append(inp)

        return active

    def simplify(self) -> GraphProgram:
        """Remove inactive nodes and return simplified program."""
        active = self.get_active_nodes()
        new_program = GraphProgram(
            functions=self.functions,
            next_id=self.next_id
        )

        for node_id in active:
            node = self.nodes[node_id].copy()
            new_program.nodes[node_id] = node
            if node.node_type == NodeType.INPUT:
                new_program.input_ids.append(node_id)
            elif node.node_type == NodeType.OUTPUT:
                new_program.output_ids.append(node_id)

        return new_program

    def to_expression(self, output_idx: int = 0) -> str:
        """Convert to human-readable expression string."""
        if output_idx >= len(self.output_ids):
            return ""

        def node_to_str(node_id: int, visited: set[int]) -> str:
            if node_id in visited or node_id not in self.nodes:
                return "..."

            visited.add(node_id)
            node = self.nodes[node_id]

            if node.node_type == NodeType.INPUT:
                idx = self.input_ids.index(node_id) if node_id in self.input_ids else 0
                return f"x{idx}"
            elif node.node_type == NodeType.CONSTANT:
                return str(node.value)
            elif node.node_type == NodeType.FUNCTION:
                args = [node_to_str(inp, visited.copy())
                        for inp in node.inputs if inp >= 0]
                return f"{node.value}({', '.join(args)})"
            elif node.node_type == NodeType.OUTPUT:
                if node.inputs and node.inputs[0] >= 0:
                    return node_to_str(node.inputs[0], visited)
                return "?"
            return "?"

        out_id = self.output_ids[output_idx]
        return node_to_str(out_id, set())

    @property
    def fitness(self) -> Optional[float]:
        """Get the fitness value."""
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        """Set the fitness value."""
        self._fitness = value

    @staticmethod
    def create_random(
        n_inputs: int,
        n_outputs: int,
        n_nodes: int,
        functions: Optional[dict[str, tuple[int, Callable[..., float]]]] = None,
        constant_range: tuple[float, float] = (-5.0, 5.0)
    ) -> GraphProgram:
        """
        Create a random EGGP program.

        Args:
            n_inputs: Number of input terminals.
            n_outputs: Number of outputs.
            n_nodes: Number of internal function nodes.
            functions: Function set to use.
            constant_range: Range for random constants.

        Returns:
            A randomly initialized GraphProgram.
        """
        program = GraphProgram()
        if functions:
            program.functions = functions

        # Create input nodes
        for i in range(n_inputs):
            program.add_node(NodeType.INPUT, label=f"x{i}")

        # Create some ephemeral constants
        n_constants = max(1, n_nodes // 4)
        constant_ids = []
        for _ in range(n_constants):
            val = random.uniform(*constant_range)
            cid = program.add_node(NodeType.CONSTANT, value=round(val, 4))
            constant_ids.append(cid)

        # Create function nodes
        func_names = list(program.functions.keys())
        available_ids = program.input_ids.copy() + constant_ids

        for _ in range(n_nodes):
            func_name = random.choice(func_names)
            arity = program.functions[func_name][0]

            # Select random inputs
            inputs = []
            for _ in range(arity):
                if available_ids:
                    inputs.append(random.choice(available_ids))

            node_id = program.add_node(NodeType.FUNCTION, value=func_name, inputs=inputs)
            available_ids.append(node_id)

        # Create output nodes
        for i in range(n_outputs):
            out_id = program.add_node(NodeType.OUTPUT, label=f"out{i}")
            if available_ids:
                program.connect(random.choice(available_ids), out_id)

        return program


@dataclass
class EGGPConfig:
    """
    Configuration for EGGP evolution.

    Attributes:
        n_inputs: Number of input terminals.
        n_outputs: Number of outputs.
        n_nodes: Number of internal nodes.
        population_size: Size of the population.
        n_generations: Number of generations to evolve.
        mutation_rate: Probability of mutating each element.
        crossover_rate: Probability of crossover.
        tournament_size: Size for tournament selection.
        elitism: Number of best individuals to preserve.
        constant_range: Range for ephemeral constants.
        functions: Available function set.
    """
    n_inputs: int = 1
    n_outputs: int = 1
    n_nodes: int = 20
    population_size: int = 100
    n_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    tournament_size: int = 5
    elitism: int = 2
    constant_range: tuple[float, float] = (-5.0, 5.0)
    functions: dict[str, tuple[int, Callable[..., float]]] = field(
        default_factory=lambda: EGGP_FUNCTIONS.copy()
    )


class EGGPEvolver:
    """
    Evolutionary algorithm for EGGP programs.

    Implements tournament selection, subtree crossover, and various mutation
    operators for evolving graph programs.
    """

    def __init__(self, config: EGGPConfig):
        """
        Initialize the evolver with given configuration.

        Args:
            config: Evolution configuration.
        """
        self.config = config
        self.population: list[GraphProgram] = []
        self.best_individual: Optional[GraphProgram] = None
        self.fitness_history: list[float] = []
        self.generation: int = 0

    def initialize_population(self) -> None:
        """Create initial random population."""
        self.population = []
        for _ in range(self.config.population_size):
            program = GraphProgram.create_random(
                n_inputs=self.config.n_inputs,
                n_outputs=self.config.n_outputs,
                n_nodes=self.config.n_nodes,
                functions=self.config.functions,
                constant_range=self.config.constant_range
            )
            self.population.append(program)

    def evaluate_population(self, fitness_func: Callable[[GraphProgram], float]) -> None:
        """Evaluate fitness of all individuals."""
        for program in self.population:
            if program.fitness is None:
                program.fitness = fitness_func(program)

        # Update best
        self.population.sort(key=lambda p: p.fitness or float('-inf'), reverse=True)
        if self.population:
            current_best = self.population[0]
            if (self.best_individual is None or
                (current_best.fitness or 0) > (self.best_individual.fitness or 0)):
                self.best_individual = current_best.copy()
                self.best_individual.fitness = current_best.fitness

    def tournament_select(self) -> GraphProgram:
        """Select an individual using tournament selection."""
        candidates = random.sample(
            self.population,
            min(self.config.tournament_size, len(self.population))
        )
        return max(candidates, key=lambda p: p.fitness or float('-inf'))

    def mutate(self, program: GraphProgram) -> GraphProgram:
        """
        Apply mutation operators to a program.

        Mutation types:
        - Point mutation: Change a node's function or constant value
        - Structural mutation: Add/remove nodes
        - Connection mutation: Rewire edges
        """
        mutant = program.copy()
        mutant._fitness = None

        func_names = list(mutant.functions.keys())

        for node_id, node in list(mutant.nodes.items()):
            if random.random() >= self.config.mutation_rate:
                continue

            if node.node_type == NodeType.CONSTANT:
                # Mutate constant value
                node.value = node.value + random.gauss(0, 1)

            elif node.node_type == NodeType.FUNCTION:
                mutation_type = random.choice(["function", "connection"])

                if mutation_type == "function":
                    # Change function
                    new_func = random.choice(func_names)
                    node.value = new_func
                    node.arity = mutant.functions[new_func][0]
                    # Adjust inputs
                    while len(node.inputs) < node.arity:
                        valid_ids = [nid for nid in mutant.nodes.keys()
                                    if nid != node_id]
                        if valid_ids:
                            node.inputs.append(random.choice(valid_ids))
                        else:
                            break

                elif mutation_type == "connection":
                    # Rewire a connection
                    if node.inputs:
                        idx = random.randint(0, len(node.inputs) - 1)
                        valid_ids = [nid for nid in mutant.nodes.keys()
                                    if nid != node_id]
                        if valid_ids:
                            node.inputs[idx] = random.choice(valid_ids)

        # Occasionally add a new node
        if random.random() < self.config.mutation_rate * 0.5:
            func_name = random.choice(func_names)
            arity = mutant.functions[func_name][0]
            valid_ids = list(mutant.nodes.keys())
            inputs = [random.choice(valid_ids) for _ in range(arity)] if valid_ids else []
            mutant.add_node(NodeType.FUNCTION, value=func_name, inputs=inputs)

        return mutant

    def crossover(self, parent1: GraphProgram, parent2: GraphProgram) -> GraphProgram:
        """
        Perform crossover between two parents.

        Uses subgraph swapping - selects a random subgraph from one parent
        and grafts it into the other.
        """
        if random.random() > self.config.crossover_rate:
            return parent1.copy()

        child = parent1.copy()
        child._fitness = None

        # Select random nodes from parent2 to copy
        p2_nodes = list(parent2.nodes.keys())
        if not p2_nodes:
            return child

        n_copy = random.randint(1, min(5, len(p2_nodes)))
        nodes_to_copy = random.sample(p2_nodes, n_copy)

        # Copy nodes with remapped IDs
        id_map: dict[int, int] = {}
        for old_id in nodes_to_copy:
            node = parent2.nodes[old_id].copy()
            new_id = child.next_id
            child.next_id += 1
            node.node_id = new_id
            id_map[old_id] = new_id
            child.nodes[new_id] = node

        # Update connections in copied nodes
        for new_id in id_map.values():
            node = child.nodes[new_id]
            new_inputs = []
            for inp in node.inputs:
                if inp in id_map:
                    new_inputs.append(id_map[inp])
                elif inp in child.nodes:
                    new_inputs.append(inp)
                elif child.nodes:
                    new_inputs.append(random.choice(list(child.nodes.keys())))
            node.inputs = new_inputs

        return child

    def evolve(
        self,
        fitness_func: Callable[[GraphProgram], float],
        verbose: bool = False
    ) -> tuple[GraphProgram, list[float]]:
        """
        Run the evolutionary algorithm.

        Args:
            fitness_func: Function to evaluate program fitness (higher is better).
            verbose: Print progress information.

        Returns:
            Tuple of (best program, fitness history).
        """
        self.initialize_population()
        self.evaluate_population(fitness_func)
        self.fitness_history = [self.best_individual.fitness or 0.0]

        for gen in range(self.config.n_generations):
            self.generation = gen + 1

            # Create next generation
            next_pop: list[GraphProgram] = []

            # Elitism
            for i in range(min(self.config.elitism, len(self.population))):
                elite = self.population[i].copy()
                elite.fitness = self.population[i].fitness
                next_pop.append(elite)

            # Generate rest through selection, crossover, mutation
            while len(next_pop) < self.config.population_size:
                parent1 = self.tournament_select()
                parent2 = self.tournament_select()

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_pop.append(child)

            self.population = next_pop
            self.evaluate_population(fitness_func)
            self.fitness_history.append(self.best_individual.fitness or 0.0)

            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}: "
                      f"Best fitness = {self.best_individual.fitness:.6f}")

        return self.best_individual, self.fitness_history
