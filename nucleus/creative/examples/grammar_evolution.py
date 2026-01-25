#!/usr/bin/env python3
"""
Grammar Evolution Example
=========================

Demonstrates the Grammars subsystem for CGP/EGGP synthesis,
metagrammar transformations, and grammar parsing.
"""

import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# =============================================================================
# MOCK IMPLEMENTATIONS
# =============================================================================

@dataclass
class MockASTNode:
    """Mock AST node."""
    type: str
    value: any = None
    children: list = field(default_factory=list)

    def pretty_print(self, indent: int = 0) -> str:
        """Pretty print the AST."""
        result = "  " * indent + f"{self.type}"
        if self.value is not None:
            result += f": {self.value}"
        result += "\n"
        for child in self.children:
            result += child.pretty_print(indent + 1)
        return result


class MockGrammarParser:
    """Mock Earley parser for grammars."""

    def __init__(self, grammar_name: str = "arithmetic"):
        self.grammar_name = grammar_name

    def parse(self, text: str) -> MockASTNode:
        """Parse input text."""
        print(f"   Parsing: '{text}'")

        # Mock: create simple AST based on operators
        tokens = text.replace('(', ' ( ').replace(')', ' ) ').split()

        return self._build_ast(tokens)

    def _build_ast(self, tokens: list) -> MockASTNode:
        """Build mock AST from tokens."""
        if not tokens:
            return MockASTNode(type="empty")

        # Simplified mock parsing
        root = MockASTNode(type="expression")

        for token in tokens:
            if token.isdigit():
                root.children.append(MockASTNode(type="number", value=int(token)))
            elif token in "+-*/":
                root.children.append(MockASTNode(type="operator", value=token))
            elif token == '(':
                root.children.append(MockASTNode(type="lparen"))
            elif token == ')':
                root.children.append(MockASTNode(type="rparen"))

        return root


@dataclass
class MockCGPNode:
    """Mock CGP node."""
    function: str
    inputs: list
    active: bool = True


@dataclass
class MockCGPGenome:
    """Mock CGP genome."""
    nodes: list
    n_inputs: int
    n_outputs: int
    output_nodes: list

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate genome on inputs."""
        # Mock evaluation - just return sum of inputs
        return np.sum(inputs, axis=1, keepdims=True)

    def mutate(self, rate: float = 0.05) -> "MockCGPGenome":
        """Create mutated copy."""
        new_nodes = []
        for node in self.nodes:
            if np.random.random() < rate:
                # Mutate function
                new_func = np.random.choice(["+", "-", "*", "/", "sin", "cos"])
                new_nodes.append(MockCGPNode(
                    function=new_func,
                    inputs=node.inputs,
                    active=node.active,
                ))
            else:
                new_nodes.append(node)

        return MockCGPGenome(
            nodes=new_nodes,
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            output_nodes=self.output_nodes,
        )

    def to_expression(self, var_names: list[str]) -> str:
        """Convert to human-readable expression."""
        # Mock: return simple expression
        return f"{var_names[0]} + {var_names[1]}"


@dataclass
class MockEvolutionResult:
    """Mock CGP evolution result."""
    best_genome: MockCGPGenome
    best_fitness: float
    generations: int
    fitness_history: list


def mock_cgp_evolve(
    fitness_fn: Callable,
    n_inputs: int,
    n_outputs: int,
    generations: int = 100,
    population_size: int = 50,
    mutation_rate: float = 0.05,
) -> MockEvolutionResult:
    """Run mock CGP evolution."""
    print(f"   Starting CGP evolution...")
    print(f"   Population: {population_size}, Generations: {generations}")
    print(f"   Inputs: {n_inputs}, Outputs: {n_outputs}")

    # Create initial population
    population = []
    for _ in range(population_size):
        nodes = [
            MockCGPNode(function=np.random.choice(["+", "-", "*"]), inputs=[0, 1])
            for _ in range(10)
        ]
        genome = MockCGPGenome(
            nodes=nodes,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            output_nodes=[9],
        )
        population.append(genome)

    fitness_history = []
    best_fitness = 0.0
    best_genome = population[0]

    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [fitness_fn(g) for g in population]

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_genome = population[gen_best_idx]

        fitness_history.append(best_fitness)

        # Selection and mutation
        new_population = [best_genome]  # Elitism
        for _ in range(population_size - 1):
            parent = population[np.random.randint(len(population))]
            child = parent.mutate(mutation_rate)
            new_population.append(child)

        population = new_population

        if gen % 20 == 0:
            print(f"   Gen {gen}: best_fitness = {best_fitness:.4f}")

    print(f"   Evolution complete! Final fitness: {best_fitness:.4f}")

    return MockEvolutionResult(
        best_genome=best_genome,
        best_fitness=best_fitness,
        generations=generations,
        fitness_history=fitness_history,
    )


@dataclass
class MockGraphNode:
    """Mock EGGP graph node."""
    id: int
    type: str
    value: any = None


@dataclass
class MockGraphEdge:
    """Mock EGGP graph edge."""
    source: int
    target: int
    label: str = ""


@dataclass
class MockGraphProgram:
    """Mock EGGP graph program."""
    nodes: list
    edges: list


class MockEGGPEvolver:
    """Mock EGGP evolver."""

    def __init__(self, grammar=None):
        self.grammar = grammar
        self.population = []
        self._initialized = False

    def initialize(self, pop_size: int = 30):
        """Initialize population."""
        print(f"   Initializing EGGP population (size={pop_size})...")

        self.population = []
        for i in range(pop_size):
            nodes = [
                MockGraphNode(id=j, type=np.random.choice(["expr", "term", "factor"]))
                for j in range(5)
            ]
            edges = [
                MockGraphEdge(source=j, target=j + 1)
                for j in range(4)
            ]
            self.population.append(MockGraphProgram(nodes=nodes, edges=edges))

        self._initialized = True
        print(f"   Initialized {len(self.population)} programs")

    def evolve(
        self,
        generations: int = 50,
        target_fitness: float = 1.0,
    ) -> tuple:
        """Run EGGP evolution."""
        if not self._initialized:
            self.initialize()

        print(f"   Running EGGP evolution for {generations} generations...")

        best_fitness = 0.0
        best_program = self.population[0]

        for gen in range(generations):
            # Mock fitness evaluation
            fitnesses = [0.5 + 0.5 * np.random.random() for _ in self.population]

            # Track best
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_program = self.population[gen_best_idx]

            if best_fitness >= target_fitness:
                print(f"   Converged at generation {gen}!")
                return best_program, best_fitness, gen

            if gen % 10 == 0:
                print(f"   Gen {gen}: best_fitness = {best_fitness:.4f}")

        print(f"   Evolution complete! Final fitness: {best_fitness:.4f}")
        return best_program, best_fitness, generations


class MockMetagrammarRegistry:
    """Mock metagrammar transformation registry."""

    def __init__(self):
        self.rules = []

    def transform(
        self,
        code: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Transform code between languages."""
        print(f"   Transforming {source_lang} -> {target_lang}")
        print(f"   Input: '{code}'")

        # Mock transformations
        transformations = {
            ("python", "javascript"): {
                "for i in range(10):": "for (let i = 0; i < 10; i++) {",
                "def ": "function ",
                "print(": "console.log(",
                "True": "true",
                "False": "false",
                "None": "null",
            },
            ("javascript", "python"): {
                "for (let i = 0;": "for i in range(",
                "function ": "def ",
                "console.log(": "print(",
                "true": "True",
                "false": "False",
                "null": "None",
            },
        }

        result = code
        rules = transformations.get((source_lang, target_lang), {})
        for pattern, replacement in rules.items():
            result = result.replace(pattern, replacement)

        print(f"   Output: '{result}'")
        return result


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

async def main():
    """Run grammar evolution example."""

    from nucleus.creative import emit_bus_event
    from nucleus.creative.bus.topics import GrammarTopics

    print("=" * 60)
    print("Grammars Subsystem - Program Synthesis Example")
    print("=" * 60)

    # 1. Grammar Parsing
    print("\n1. Grammar Parsing (Earley Algorithm)")
    print("-" * 40)

    parser = MockGrammarParser(grammar_name="arithmetic")

    expressions = [
        "(1 + 2) * 3",
        "4 / 2 - 1",
        "5 * (3 + 2)",
    ]

    for expr in expressions:
        ast = parser.parse(expr)
        print(f"\n   Expression: {expr}")
        print(f"   AST nodes: {len(ast.children)}")
        print("   AST structure:")
        print(ast.pretty_print().rstrip())

    # 2. CGP Evolution
    print("\n2. Cartesian Genetic Programming (CGP)")
    print("-" * 40)

    # Define training data for symbolic regression
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
    ], dtype=np.float32)

    y = X[:, 0] + X[:, 1]  # Target: a + b

    def fitness_fn(genome: MockCGPGenome) -> float:
        """Calculate fitness based on prediction error."""
        predictions = genome.evaluate(X)
        error = np.mean((predictions.flatten() - y) ** 2)
        return 1.0 / (1.0 + error)

    result = mock_cgp_evolve(
        fitness_fn=fitness_fn,
        n_inputs=2,
        n_outputs=1,
        generations=50,
        population_size=20,
        mutation_rate=0.1,
    )

    print(f"\n   Results:")
    print(f"   - Best fitness: {result.best_fitness:.4f}")
    print(f"   - Generations: {result.generations}")
    print(f"   - Expression: {result.best_genome.to_expression(['a', 'b'])}")

    # Emit bus event
    emit_bus_event(
        topic=GrammarTopics.CGP_EVOLUTION_COMPLETE,
        payload={
            "grammar_type": "cgp",
            "final_fitness": float(result.best_fitness),
            "generations_run": result.generations,
            "converged": bool(result.best_fitness > 0.95),
        },
    )

    # 3. EGGP Evolution
    print("\n3. Evolving Graph Grammar Programs (EGGP)")
    print("-" * 40)

    evolver = MockEGGPEvolver()
    evolver.initialize(pop_size=30)

    best_program, fitness, gens = evolver.evolve(
        generations=30,
        target_fitness=0.95,
    )

    print(f"\n   Results:")
    print(f"   - Best fitness: {fitness:.4f}")
    print(f"   - Generations: {gens}")
    print(f"   - Program nodes: {len(best_program.nodes)}")
    print(f"   - Program edges: {len(best_program.edges)}")

    # 4. Metagrammar Transformations
    print("\n4. Cross-Language Transformations")
    print("-" * 40)

    registry = MockMetagrammarRegistry()

    # Python to JavaScript
    python_code = "for i in range(10):"
    js_code = registry.transform(python_code, "python", "javascript")
    print()

    # JavaScript to Python
    js_code2 = "function add(a, b) { return a + b; }"
    py_code2 = registry.transform(js_code2, "javascript", "python")
    print()

    # More examples
    examples = [
        ("print('hello')", "python", "javascript"),
        ("console.log(true)", "javascript", "python"),
    ]

    for code, src, tgt in examples:
        result = registry.transform(code, src, tgt)
        print()

    # 5. Fitness Functions
    print("\n5. Available Fitness Functions")
    print("-" * 40)

    fitness_functions = [
        ("symbolic_regression", "Minimize prediction error on numeric data"),
        ("classification", "Maximize classification accuracy"),
        ("boolean", "Match boolean function truth table"),
        ("transformation_coverage", "Maximize code transformation coverage"),
        ("refactoring", "Optimize code quality metrics"),
    ]

    for name, desc in fitness_functions:
        print(f"   {name}: {desc}")

    # 6. Grammar Specification
    print("\n6. Grammar Specification (GIF Format)")
    print("-" * 40)

    from nucleus.creative.grammars import GrammarSpec, Terminal, Production

    spec = GrammarSpec(
        id="arithmetic_v1",
        name="Arithmetic Grammar",
        version="1.0.0",
        start="expr",
    )

    # Add terminals
    spec.terminals["NUMBER"] = Terminal(
        name="NUMBER",
        pattern=r"\d+",
        precedence=0,
    )
    spec.terminals["PLUS"] = Terminal(
        name="PLUS",
        pattern=r"\+",
        precedence=1,
    )
    spec.terminals["TIMES"] = Terminal(
        name="TIMES",
        pattern=r"\*",
        precedence=2,
    )

    # Add productions
    spec.productions["expr"] = [
        Production(id="add", symbols=["expr", {"terminal": "PLUS"}, "term"]),
        Production(id="term_only", symbols=["term"]),
    ]
    spec.productions["term"] = [
        Production(id="mul", symbols=["term", {"terminal": "TIMES"}, "factor"]),
        Production(id="factor_only", symbols=["factor"]),
    ]

    print(f"   Grammar ID: {spec.id}")
    print(f"   Start symbol: {spec.start}")
    print(f"   Terminals: {list(spec.terminals.keys())}")
    print(f"   Non-terminals: {list(spec.productions.keys())}")

    # 7. Function Set
    print("\n7. CGP Function Set")
    print("-" * 40)

    function_set = [
        ("+", "Addition", 2),
        ("-", "Subtraction", 2),
        ("*", "Multiplication", 2),
        ("/", "Protected division", 2),
        ("sin", "Sine", 1),
        ("cos", "Cosine", 1),
        ("exp", "Exponential", 1),
        ("log", "Protected log", 1),
        ("sqrt", "Protected sqrt", 1),
        ("pow", "Power", 2),
        ("max", "Maximum", 2),
        ("min", "Minimum", 2),
        ("if", "Conditional", 3),
    ]

    for symbol, name, arity in function_set:
        print(f"   {symbol:6s} ({arity}): {name}")

    print("\n" + "=" * 60)
    print("Grammar evolution example complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
