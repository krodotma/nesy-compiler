"""
Grammars Subsystem Benchmarks
=============================

Benchmarks for CGP evolution, EGGP, and grammar parsing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Any, Optional

from .bench_runner import BenchmarkSuite, BenchmarkResult


@dataclass
class CGPNode:
    """Mock CGP node for benchmarking."""
    function_id: int
    inputs: list[int]
    output: int

    def evaluate(self, values: list[float]) -> float:
        """Evaluate node with given input values."""
        if not self.inputs:
            return values[0] if values else 0.0

        x = values[self.inputs[0] % len(values)] if values else 0.0
        y = values[self.inputs[1] % len(values)] if len(self.inputs) > 1 and values else 0.0

        ops = [
            lambda a, b: a + b,
            lambda a, b: a * b,
            lambda a, b: a - b,
            lambda a, b: a / b if b != 0 else 0.0,
            lambda a, b: max(a, b),
            lambda a, b: min(a, b),
            lambda a, b: abs(a),
            lambda a, b: a ** 2,
        ]

        return ops[self.function_id % len(ops)](x, y)


@dataclass
class CGPGenome:
    """Mock CGP genome for benchmarking."""
    nodes: list[CGPNode]
    outputs: list[int]
    n_inputs: int

    @classmethod
    def random(cls, n_inputs: int, n_nodes: int, n_outputs: int) -> "CGPGenome":
        """Create a random genome."""
        nodes = []
        for i in range(n_nodes):
            inputs = [random.randint(0, n_inputs + i - 1) for _ in range(2)]
            nodes.append(CGPNode(
                function_id=random.randint(0, 7),
                inputs=inputs,
                output=n_inputs + i,
            ))

        outputs = [random.randint(0, n_inputs + n_nodes - 1) for _ in range(n_outputs)]

        return cls(nodes=nodes, outputs=outputs, n_inputs=n_inputs)

    def evaluate(self, inputs: list[float]) -> list[float]:
        """Evaluate genome with given inputs."""
        values = inputs.copy()

        for node in self.nodes:
            result = node.evaluate(values)
            values.append(result)

        return [values[i] for i in self.outputs]

    def mutate(self, rate: float = 0.1) -> "CGPGenome":
        """Create mutated copy."""
        new_nodes = []
        for node in self.nodes:
            if random.random() < rate:
                new_node = CGPNode(
                    function_id=random.randint(0, 7),
                    inputs=[random.randint(0, self.n_inputs + len(new_nodes) - 1) for _ in range(2)],
                    output=node.output,
                )
                new_nodes.append(new_node)
            else:
                new_nodes.append(node)

        return CGPGenome(nodes=new_nodes, outputs=self.outputs.copy(), n_inputs=self.n_inputs)


def cgp_evolve(
    fitness_func: Callable[[CGPGenome], float],
    n_inputs: int = 4,
    n_nodes: int = 50,
    n_outputs: int = 1,
    population_size: int = 5,
    generations: int = 100,
) -> tuple[CGPGenome, float]:
    """
    Run CGP evolution.

    Args:
        fitness_func: Function to evaluate genome fitness
        n_inputs: Number of input nodes
        n_nodes: Number of computational nodes
        n_outputs: Number of output nodes
        population_size: Population size (1+lambda)
        generations: Number of generations

    Returns:
        Tuple of (best_genome, best_fitness)
    """
    # Initialize parent
    parent = CGPGenome.random(n_inputs, n_nodes, n_outputs)
    parent_fitness = fitness_func(parent)

    for _ in range(generations):
        # Generate offspring
        offspring = [parent.mutate() for _ in range(population_size)]
        offspring_fitness = [fitness_func(g) for g in offspring]

        # Select best
        best_idx = max(range(len(offspring)), key=lambda i: offspring_fitness[i])

        if offspring_fitness[best_idx] >= parent_fitness:
            parent = offspring[best_idx]
            parent_fitness = offspring_fitness[best_idx]

    return parent, parent_fitness


@dataclass
class GraphNode:
    """Mock EGGP graph node."""
    node_id: int
    node_type: str
    edges: list[int]
    weight: float = 1.0


@dataclass
class GraphProgram:
    """Mock EGGP graph program."""
    nodes: list[GraphNode]
    input_nodes: list[int]
    output_nodes: list[int]

    @classmethod
    def random(cls, n_nodes: int, density: float = 0.3) -> "GraphProgram":
        """Create random graph program."""
        nodes = []
        node_types = ["add", "mul", "sub", "div", "max", "min", "abs", "neg"]

        for i in range(n_nodes):
            n_edges = max(1, int(n_nodes * density))
            edges = random.sample(range(n_nodes), min(n_edges, n_nodes))
            nodes.append(GraphNode(
                node_id=i,
                node_type=random.choice(node_types),
                edges=edges,
                weight=random.random(),
            ))

        n_io = max(1, n_nodes // 10)
        input_nodes = list(range(n_io))
        output_nodes = list(range(n_nodes - n_io, n_nodes))

        return cls(nodes=nodes, input_nodes=input_nodes, output_nodes=output_nodes)

    def crossover(self, other: "GraphProgram") -> "GraphProgram":
        """Create offspring via crossover."""
        # Simple uniform crossover
        new_nodes = []
        for i in range(min(len(self.nodes), len(other.nodes))):
            if random.random() < 0.5:
                new_nodes.append(self.nodes[i])
            else:
                new_nodes.append(other.nodes[i])

        return GraphProgram(
            nodes=new_nodes,
            input_nodes=self.input_nodes.copy(),
            output_nodes=self.output_nodes.copy(),
        )


@dataclass
class ASTNode:
    """Mock AST node for parsing benchmarks."""
    node_type: str
    value: Optional[str] = None
    children: list["ASTNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @classmethod
    def random_tree(cls, depth: int = 5, branching: int = 3) -> "ASTNode":
        """Generate random AST tree."""
        node_types = ["expr", "term", "factor", "literal", "identifier", "operator"]

        node = cls(
            node_type=random.choice(node_types),
            value=str(random.randint(0, 100)) if random.random() < 0.3 else None,
        )

        if depth > 0:
            n_children = random.randint(0, branching)
            node.children = [cls.random_tree(depth - 1, branching) for _ in range(n_children)]

        return node

    def count_nodes(self) -> int:
        """Count total nodes in tree."""
        return 1 + sum(child.count_nodes() for child in self.children)


class GrammarParser:
    """Mock grammar parser."""

    def __init__(self, grammar: str):
        self.grammar = grammar
        self.rules = self._parse_grammar(grammar)

    def _parse_grammar(self, grammar: str) -> dict[str, list[list[str]]]:
        """Parse BNF-like grammar."""
        rules: dict[str, list[list[str]]] = {}

        for line in grammar.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "::=" in line:
                name, prods = line.split("::=", 1)
                name = name.strip()
                prods = [p.strip().split() for p in prods.split("|")]
                rules[name] = prods

        return rules

    def parse(self, text: str) -> Optional[ASTNode]:
        """Parse text according to grammar."""
        # Simplified recursive descent parser
        tokens = text.split()

        if not tokens:
            return None

        return ASTNode(
            node_type="program",
            children=[ASTNode(node_type="token", value=t) for t in tokens],
        )

    def generate(self, start: str = "start", max_depth: int = 10) -> str:
        """Generate random string from grammar."""
        if max_depth <= 0 or start not in self.rules:
            return start

        production = random.choice(self.rules[start])
        result = []

        for symbol in production:
            if symbol in self.rules:
                result.append(self.generate(symbol, max_depth - 1))
            else:
                result.append(symbol)

        return " ".join(result)


SAMPLE_GRAMMAR = """
start ::= expr
expr ::= term | expr + term | expr - term
term ::= factor | term * factor | term / factor
factor ::= number | ( expr ) | identifier
number ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
identifier ::= x | y | z | a | b | c
"""


class GrammarsBenchmark(BenchmarkSuite):
    """Benchmark suite for grammars subsystem."""

    @property
    def name(self) -> str:
        return "grammars"

    @property
    def description(self) -> str:
        return "CGP/EGGP synthesis, parsing benchmarks"

    def __init__(self):
        self._parser: Optional[GrammarParser] = None
        self._genomes: list[CGPGenome] = []
        self._graphs: list[GraphProgram] = []

    def setup(self) -> None:
        """Setup test data."""
        self._parser = GrammarParser(SAMPLE_GRAMMAR)
        self._genomes = [CGPGenome.random(4, 50, 1) for _ in range(10)]
        self._graphs = [GraphProgram.random(100) for _ in range(10)]

    def get_benchmarks(self) -> list[tuple[str, Callable[[], Any]]]:
        """Get all grammar benchmarks."""
        return [
            ("cgp_genome_create_small", self._cgp_genome_create_small),
            ("cgp_genome_create_medium", self._cgp_genome_create_medium),
            ("cgp_genome_create_large", self._cgp_genome_create_large),
            ("cgp_genome_evaluate", self._cgp_genome_evaluate),
            ("cgp_genome_mutate", self._cgp_genome_mutate),
            ("cgp_evolution_short", self._cgp_evolution_short),
            ("cgp_evolution_medium", self._cgp_evolution_medium),
            ("eggp_graph_create", self._eggp_graph_create),
            ("eggp_graph_crossover", self._eggp_crossover),
            ("parser_parse_simple", self._parser_parse_simple),
            ("parser_parse_complex", self._parser_parse_complex),
            ("parser_generate", self._parser_generate),
            ("ast_create_shallow", self._ast_create_shallow),
            ("ast_create_deep", self._ast_create_deep),
            ("ast_count_nodes", self._ast_count_nodes),
        ]

    def _cgp_genome_create_small(self) -> CGPGenome:
        """Create small CGP genome (4 inputs, 20 nodes)."""
        return CGPGenome.random(4, 20, 1)

    def _cgp_genome_create_medium(self) -> CGPGenome:
        """Create medium CGP genome (8 inputs, 100 nodes)."""
        return CGPGenome.random(8, 100, 2)

    def _cgp_genome_create_large(self) -> CGPGenome:
        """Create large CGP genome (16 inputs, 500 nodes)."""
        return CGPGenome.random(16, 500, 4)

    def _cgp_genome_evaluate(self) -> list[float]:
        """Evaluate CGP genome."""
        genome = self._genomes[0]
        inputs = [random.random() for _ in range(genome.n_inputs)]
        return genome.evaluate(inputs)

    def _cgp_genome_mutate(self) -> CGPGenome:
        """Mutate CGP genome."""
        return self._genomes[0].mutate(0.1)

    def _cgp_evolution_short(self) -> tuple[CGPGenome, float]:
        """Run short CGP evolution (10 generations)."""
        def fitness(g: CGPGenome) -> float:
            result = g.evaluate([1.0, 2.0, 3.0, 4.0])
            return -abs(result[0] - 10.0) if result else -100.0

        return cgp_evolve(fitness, generations=10)

    def _cgp_evolution_medium(self) -> tuple[CGPGenome, float]:
        """Run medium CGP evolution (50 generations)."""
        def fitness(g: CGPGenome) -> float:
            result = g.evaluate([1.0, 2.0, 3.0, 4.0])
            return -abs(result[0] - 10.0) if result else -100.0

        return cgp_evolve(fitness, generations=50)

    def _eggp_graph_create(self) -> GraphProgram:
        """Create EGGP graph program."""
        return GraphProgram.random(100)

    def _eggp_crossover(self) -> GraphProgram:
        """Perform EGGP crossover."""
        return self._graphs[0].crossover(self._graphs[1])

    def _parser_parse_simple(self) -> Optional[ASTNode]:
        """Parse simple expression."""
        return self._parser.parse("x + 1")

    def _parser_parse_complex(self) -> Optional[ASTNode]:
        """Parse complex expression."""
        return self._parser.parse("( x + y ) * ( a - b ) / ( c + 1 )")

    def _parser_generate(self) -> str:
        """Generate string from grammar."""
        return self._parser.generate("start", max_depth=5)

    def _ast_create_shallow(self) -> ASTNode:
        """Create shallow AST tree."""
        return ASTNode.random_tree(depth=3, branching=2)

    def _ast_create_deep(self) -> ASTNode:
        """Create deep AST tree."""
        return ASTNode.random_tree(depth=8, branching=3)

    def _ast_count_nodes(self) -> int:
        """Count nodes in AST tree."""
        tree = ASTNode.random_tree(depth=5, branching=3)
        return tree.count_nodes()
