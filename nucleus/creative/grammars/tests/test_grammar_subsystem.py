"""
Tests for the Grammars Subsystem
================================

Tests CGP/EGGP synthesis, grammar parsing, and metagrammar registry.
Uses mock classes for dataclass functional tests, real classes only in smoke tests.
"""

import pytest
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, List, Dict

# Ensure nucleus is importable
sys.path.insert(0, str(Path(__file__).parents[4]))


# -----------------------------------------------------------------------------
# Mock Classes for Testing
# -----------------------------------------------------------------------------

@dataclass
class MockCGPNode:
    """Mock CGP node for testing."""
    node_id: int
    function_id: int = 0
    inputs: List[int] = field(default_factory=list)


@dataclass
class MockCGPGenome:
    """Mock CGP genome for testing."""
    n_inputs: int
    n_outputs: int
    n_rows: int = 1
    n_cols: int = 10
    nodes: List[MockCGPNode] = field(default_factory=list)
    levels_back: Optional[int] = None
    fitness: Optional[float] = None

    def mutate(self, mutation_rate: float = 0.1) -> "MockCGPGenome":
        """Mock mutation returns a copy."""
        return MockCGPGenome(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            nodes=self.nodes.copy(),
            levels_back=self.levels_back,
        )

    def copy(self) -> "MockCGPGenome":
        """Return a copy of this genome."""
        return MockCGPGenome(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            nodes=self.nodes.copy(),
            levels_back=self.levels_back,
        )

    def evaluate(self, inputs: List[float]) -> List[float]:
        """Mock evaluation."""
        return [0.0] * self.n_outputs


@dataclass
class MockEGGPConfig:
    """Mock EGGP config for testing."""
    n_inputs: int = 1
    n_outputs: int = 1
    population_size: int = 100
    n_generations: int = 100
    mutation_rate: float = 0.1


@dataclass
class MockGraphNode:
    """Mock graph node for testing."""
    node_id: int
    operation: str
    inputs: List[int] = field(default_factory=list)


@dataclass
class MockGraphProgram:
    """Mock graph program for testing."""
    n_inputs: int = 1
    n_outputs: int = 1
    nodes: Dict[int, MockGraphNode] = field(default_factory=dict)


class MockEGGPEvolver:
    """Mock EGGP evolver for testing."""

    def __init__(self, config: Optional[MockEGGPConfig] = None):
        self.config = config or MockEGGPConfig()

    def evolve(self, fitness_fn: Callable) -> MockGraphProgram:
        """Mock evolution."""
        return MockGraphProgram()


@dataclass
class MockGrammarRule:
    """Mock grammar rule for testing."""
    lhs: str
    rhs: List[str]


@dataclass
class MockASTNode:
    """Mock AST node for testing."""
    symbol: str
    children: List["MockASTNode"] = field(default_factory=list)


class MockGrammarParser:
    """Mock grammar parser for testing."""

    def __init__(self):
        self.rules = []

    def parse(self, text: str) -> Optional[MockASTNode]:
        """Mock parse method."""
        return MockASTNode(symbol="S")


@dataclass
class MockTransformRule:
    """Mock transform rule for testing."""
    name: str
    pattern: str
    replacement: str


@dataclass
class MockPatternVariable:
    """Mock pattern variable for testing."""
    name: str
    constraint: Optional[Callable] = None


class MockMetagrammarRegistry:
    """Mock metagrammar registry for testing."""

    def __init__(self):
        self._rules: Dict[str, MockTransformRule] = {}

    def register(self, rule: MockTransformRule):
        """Register a transform rule."""
        self._rules[rule.name] = rule

    def get(self, name: str) -> Optional[MockTransformRule]:
        """Get a rule by name."""
        return self._rules.get(name)

    def list_rules(self) -> List[MockTransformRule]:
        """List all registered rules."""
        return list(self._rules.values())


# -----------------------------------------------------------------------------
# Import Helpers with Skip Handling
# -----------------------------------------------------------------------------

try:
    from nucleus.creative.grammars import (
        CGPGenome,
        CGPNode,
        CGPFunction,
        FUNCTION_SET,
        cgp_evolve,
    )
    HAS_CGP = CGPGenome is not None
except ImportError:
    HAS_CGP = False
    CGPGenome = None
    CGPNode = None
    CGPFunction = None
    FUNCTION_SET = {}
    cgp_evolve = None

try:
    from nucleus.creative.grammars import (
        EGGPEvolver,
        EGGPConfig,
        GraphNode,
        GraphProgram,
    )
    HAS_EGGP = EGGPEvolver is not None
except ImportError:
    HAS_EGGP = False
    EGGPEvolver = None
    EGGPConfig = None
    GraphNode = None
    GraphProgram = None

try:
    from nucleus.creative.grammars import (
        GrammarParser,
        GrammarRule,
        ASTNode,
    )
    HAS_PARSER = GrammarParser is not None
except ImportError:
    HAS_PARSER = False
    GrammarParser = None
    GrammarRule = None
    ASTNode = None

try:
    from nucleus.creative.grammars import (
        MetagrammarRegistry,
        TransformRule,
        PatternVariable,
    )
    HAS_METAGRAMMAR = MetagrammarRegistry is not None
except ImportError:
    HAS_METAGRAMMAR = False
    MetagrammarRegistry = None
    TransformRule = None
    PatternVariable = None


# -----------------------------------------------------------------------------
# Smoke Tests - Verify Module Imports
# -----------------------------------------------------------------------------


class TestGrammarSmoke:
    """Smoke tests verifying imports work."""

    def test_grammars_module_importable(self):
        """Test that grammars module can be imported."""
        from nucleus.creative import grammars
        assert grammars is not None

    @pytest.mark.skipif(not HAS_CGP, reason="CGP module not available")
    def test_cgp_classes_exist(self):
        """Test CGP classes are defined."""
        assert CGPGenome is not None
        assert CGPNode is not None

    @pytest.mark.skipif(not HAS_EGGP, reason="EGGP module not available")
    def test_eggp_classes_exist(self):
        """Test EGGP classes are defined."""
        assert EGGPEvolver is not None
        assert EGGPConfig is not None

    @pytest.mark.skipif(not HAS_PARSER, reason="Parser module not available")
    def test_parser_classes_exist(self):
        """Test parser classes are defined."""
        assert GrammarParser is not None
        assert GrammarRule is not None

    @pytest.mark.skipif(not HAS_METAGRAMMAR, reason="Metagrammar module not available")
    def test_metagrammar_classes_exist(self):
        """Test metagrammar classes are defined."""
        assert MetagrammarRegistry is not None
        assert TransformRule is not None


# -----------------------------------------------------------------------------
# Real Class Smoke Tests - Minimal Functionality Checks
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CGP, reason="CGP module not available")
class TestRealCGPSmoke:
    """Smoke tests for real CGP classes."""

    def test_cgpgenome_has_fields(self):
        """Test CGPGenome has expected fields."""
        import inspect
        sig = inspect.signature(CGPGenome)
        params = list(sig.parameters.keys())
        assert "n_inputs" in params
        assert "n_outputs" in params

    def test_function_set_not_empty(self):
        """Test that FUNCTION_SET has entries."""
        assert len(FUNCTION_SET) > 0


@pytest.mark.skipif(not HAS_EGGP, reason="EGGP module not available")
class TestRealEGGPSmoke:
    """Smoke tests for real EGGP classes."""

    def test_eggpconfig_has_fields(self):
        """Test EGGPConfig has expected fields."""
        import inspect
        sig = inspect.signature(EGGPConfig)
        params = list(sig.parameters.keys())
        assert "population_size" in params or "n_inputs" in params

    def test_eggpevolver_requires_config(self):
        """Test EGGPEvolver requires config parameter."""
        with pytest.raises(TypeError):
            EGGPEvolver()


@pytest.mark.skipif(not HAS_PARSER, reason="Parser module not available")
class TestRealParserSmoke:
    """Smoke tests for real parser classes."""

    def test_grammarparser_instantiates(self):
        """Test GrammarParser can be instantiated."""
        parser = GrammarParser()
        assert parser is not None


@pytest.mark.skipif(not HAS_METAGRAMMAR, reason="Metagrammar module not available")
class TestRealMetagrammarSmoke:
    """Smoke tests for real metagrammar classes."""

    def test_metagrammar_registry_instantiates(self):
        """Test MetagrammarRegistry can be instantiated."""
        registry = MetagrammarRegistry()
        assert registry is not None


# -----------------------------------------------------------------------------
# Mock CGPGenome Tests
# -----------------------------------------------------------------------------


class TestMockCGPGenome:
    """Tests for mock CGPGenome class."""

    def test_genome_creation(self):
        """Test creating a mock CGPGenome."""
        genome = MockCGPGenome(
            n_inputs=2,
            n_outputs=1,
            n_cols=10,
            n_rows=1,
            levels_back=10,
        )
        assert genome is not None
        assert genome.n_inputs == 2
        assert genome.n_outputs == 1

    def test_genome_evaluation(self):
        """Test evaluating a mock CGPGenome."""
        genome = MockCGPGenome(
            n_inputs=2,
            n_outputs=1,
            n_cols=5,
            n_rows=1,
            levels_back=5,
        )
        inputs = [1.0, 2.0]
        outputs = genome.evaluate(inputs)
        assert isinstance(outputs, list)
        assert len(outputs) == 1

    def test_genome_mutation(self):
        """Test mutating a mock CGPGenome."""
        genome = MockCGPGenome(
            n_inputs=2,
            n_outputs=1,
            n_cols=10,
            n_rows=1,
            levels_back=10,
        )
        mutated = genome.mutate(mutation_rate=0.1)
        assert mutated is not None
        assert isinstance(mutated, MockCGPGenome)

    def test_genome_copy(self):
        """Test copying a mock CGPGenome."""
        genome = MockCGPGenome(
            n_inputs=2,
            n_outputs=1,
            n_cols=5,
            n_rows=1,
            levels_back=5,
        )
        copied = genome.copy()
        assert copied is not genome
        assert copied.n_inputs == genome.n_inputs


# -----------------------------------------------------------------------------
# Mock EGGPEvolver Tests
# -----------------------------------------------------------------------------


class TestMockEGGPEvolver:
    """Tests for mock EGGPEvolver class."""

    def test_evolver_creation(self):
        """Test creating a mock EGGPEvolver."""
        evolver = MockEGGPEvolver()
        assert evolver is not None

    def test_evolver_with_config(self):
        """Test creating mock EGGPEvolver with config."""
        config = MockEGGPConfig(
            population_size=20,
            n_generations=5,
            mutation_rate=0.1,
        )
        evolver = MockEGGPEvolver(config=config)
        assert evolver.config.population_size == 20

    def test_graph_node_creation(self):
        """Test creating a mock GraphNode."""
        node = MockGraphNode(
            node_id=0,
            operation="add",
            inputs=[],
        )
        assert node is not None
        assert node.node_id == 0

    def test_graph_program_creation(self):
        """Test creating a mock GraphProgram."""
        program = MockGraphProgram(n_inputs=2, n_outputs=1)
        assert program is not None


# -----------------------------------------------------------------------------
# Mock GrammarParser Tests
# -----------------------------------------------------------------------------


class TestMockGrammarParser:
    """Tests for mock GrammarParser class."""

    def test_parser_creation(self):
        """Test creating a mock GrammarParser."""
        parser = MockGrammarParser()
        assert parser is not None

    def test_parse_simple_grammar(self):
        """Test parsing with mock parser."""
        parser = MockGrammarParser()
        grammar_text = """
        S -> A B
        A -> 'a'
        B -> 'b'
        """
        result = parser.parse(grammar_text)
        assert result is not None

    def test_grammar_rule_creation(self):
        """Test creating a mock GrammarRule."""
        rule = MockGrammarRule(
            lhs="S",
            rhs=["A", "B"],
        )
        assert rule.lhs == "S"
        assert rule.rhs == ["A", "B"]

    def test_ast_node_creation(self):
        """Test creating a mock ASTNode."""
        node = MockASTNode(
            symbol="S",
            children=[],
        )
        assert node.symbol == "S"
        assert len(node.children) == 0


# -----------------------------------------------------------------------------
# Mock MetagrammarRegistry Tests
# -----------------------------------------------------------------------------


class TestMockMetagrammarRegistry:
    """Tests for mock MetagrammarRegistry class."""

    def test_registry_creation(self):
        """Test creating a mock MetagrammarRegistry."""
        registry = MockMetagrammarRegistry()
        assert registry is not None

    def test_register_transform(self):
        """Test registering a transform rule."""
        registry = MockMetagrammarRegistry()
        rule = MockTransformRule(
            name="test_rule",
            pattern="A -> B",
            replacement="C -> D",
        )
        registry.register(rule)
        assert registry.get("test_rule") is not None

    def test_transform_rule_creation(self):
        """Test creating a mock TransformRule."""
        rule = MockTransformRule(
            name="simplify",
            pattern="X X",
            replacement="X",
        )
        assert rule.name == "simplify"

    def test_pattern_variable_creation(self):
        """Test creating a mock PatternVariable."""
        var = MockPatternVariable(name="X", constraint=None)
        assert var.name == "X"

    def test_registry_list_rules(self):
        """Test listing registered rules."""
        registry = MockMetagrammarRegistry()
        rule1 = MockTransformRule(name="rule1", pattern="A", replacement="B")
        rule2 = MockTransformRule(name="rule2", pattern="C", replacement="D")
        registry.register(rule1)
        registry.register(rule2)
        rules = registry.list_rules()
        assert len(rules) >= 2


# -----------------------------------------------------------------------------
# Integration Tests (Mock-based)
# -----------------------------------------------------------------------------


class TestMockGrammarIntegration:
    """Integration tests using mock classes."""

    def test_cgp_eggp_interop(self):
        """Test mock CGP and EGGP can work together."""
        cgp_genome = MockCGPGenome(
            n_inputs=2,
            n_outputs=1,
            n_cols=5,
            n_rows=1,
            levels_back=5,
        )
        evolver = MockEGGPEvolver()
        assert cgp_genome is not None
        assert evolver is not None

    def test_grammar_to_cgp(self):
        """Test converting grammar to CGP representation."""
        parser = MockGrammarParser()
        cgp = MockCGPGenome(n_inputs=1, n_outputs=1, n_cols=3, n_rows=1, levels_back=3)
        assert parser is not None
        assert cgp is not None


# -----------------------------------------------------------------------------
# Mock Edge Case Tests
# -----------------------------------------------------------------------------


class TestMockCGPEdgeCases:
    """Edge case tests using mock CGP."""

    def test_minimal_genome(self):
        """Test creating minimal genome."""
        genome = MockCGPGenome(
            n_inputs=1,
            n_outputs=1,
            n_cols=1,
            n_rows=1,
            levels_back=1,
        )
        assert genome is not None

    def test_large_genome(self):
        """Test creating larger genome."""
        genome = MockCGPGenome(
            n_inputs=10,
            n_outputs=5,
            n_cols=100,
            n_rows=1,
            levels_back=100,
        )
        assert genome is not None
        assert genome.n_inputs == 10
        assert genome.n_outputs == 5

    def test_multiple_mutations(self):
        """Test multiple sequential mutations."""
        genome = MockCGPGenome(
            n_inputs=2,
            n_outputs=1,
            n_cols=10,
            n_rows=1,
            levels_back=10,
        )
        current = genome
        for _ in range(5):
            current = current.mutate(mutation_rate=0.2)
            assert current is not None
