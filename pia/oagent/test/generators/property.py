#!/usr/bin/env python3
"""
Step 105: Property-Based Test Generator

Generates property-based tests using Hypothesis patterns.

PBTSO Phase: SKILL
Bus Topics:
- test.property.generate (subscribes)
- test.property.generated (emits)

Dependencies: Step 102 (Unit Test Generator)
"""
from __future__ import annotations

import ast
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class PropertySpec:
    """Specification for a property to test."""
    name: str
    description: str
    property_type: str  # invariant, commutative, idempotent, inverse, oracle
    function: str
    input_types: List[str]
    output_type: Optional[str] = None
    hypothesis_strategy: str = ""
    constraints: List[str] = field(default_factory=list)


@dataclass
class PropertyTestCase:
    """Represents a property-based test case."""
    id: str
    name: str
    target_module: str
    target_function: str
    property_spec: PropertySpec
    test_code: str
    strategies: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    priority: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "target_module": self.target_module,
            "target_function": self.target_function,
            "property_spec": {
                "name": self.property_spec.name,
                "description": self.property_spec.description,
                "property_type": self.property_spec.property_type,
                "function": self.property_spec.function,
                "input_types": self.property_spec.input_types,
                "output_type": self.property_spec.output_type,
                "hypothesis_strategy": self.property_spec.hypothesis_strategy,
                "constraints": self.property_spec.constraints,
            },
            "test_code": self.test_code,
            "strategies": self.strategies,
            "settings": self.settings,
            "tags": list(self.tags),
            "priority": self.priority,
        }


@dataclass
class PropertyRequest:
    """Request to generate property-based tests."""
    source_file: str
    target_functions: Optional[List[str]] = None
    property_types: Optional[List[str]] = None  # Types of properties to check
    max_examples: int = 100
    deadline_ms: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class PropertyResult:
    """Result of property-based test generation."""
    request_id: str
    source_file: str
    tests_generated: List[PropertyTestCase]
    properties_discovered: int
    generation_time_s: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "source_file": self.source_file,
            "tests_generated": [t.to_dict() for t in self.tests_generated],
            "properties_discovered": self.properties_discovered,
            "generation_time_s": self.generation_time_s,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ============================================================================
# Type to Strategy Mapping
# ============================================================================

TYPE_STRATEGIES = {
    "int": "st.integers()",
    "float": "st.floats(allow_nan=False)",
    "str": "st.text(min_size=0, max_size=100)",
    "bool": "st.booleans()",
    "bytes": "st.binary()",
    "list": "st.lists(st.integers())",
    "dict": "st.dictionaries(st.text(), st.integers())",
    "set": "st.frozensets(st.integers())",
    "tuple": "st.tuples(st.integers(), st.integers())",
    "Optional": "st.none() | {inner}",
    "List": "st.lists({inner})",
    "Dict": "st.dictionaries(st.text(), {inner})",
    "Set": "st.frozensets({inner})",
    "Tuple": "st.tuples({inner})",
    "Any": "st.from_type(type).flatmap(st.from_type)",
    "Path": "st.text().map(Path)",
}

# Property patterns to look for
PROPERTY_PATTERNS = {
    "invariant": {
        "description": "Output satisfies an invariant property",
        "patterns": ["len", "size", "count", "is_", "has_"],
        "template": """
@given({strategies})
def test_{func_name}_invariant(self, {params}):
    \"\"\"Property: {description}\"\"\"
    result = {call}
    assert {invariant}
""",
    },
    "commutative": {
        "description": "Operation is commutative: f(a, b) == f(b, a)",
        "patterns": ["add", "sum", "merge", "combine", "union", "intersect"],
        "template": """
@given({strategies})
def test_{func_name}_commutative(self, {params}):
    \"\"\"Property: {func_name}(a, b) == {func_name}(b, a)\"\"\"
    result1 = {call1}
    result2 = {call2}
    assert result1 == result2
""",
    },
    "idempotent": {
        "description": "Applying operation twice gives same result: f(f(x)) == f(x)",
        "patterns": ["normalize", "clean", "trim", "sort", "unique", "dedupe"],
        "template": """
@given({strategies})
def test_{func_name}_idempotent(self, {params}):
    \"\"\"Property: {func_name}({func_name}(x)) == {func_name}(x)\"\"\"
    result1 = {call}
    result2 = {func_name}(result1)
    assert result1 == result2
""",
    },
    "inverse": {
        "description": "Inverse operation restores original: g(f(x)) == x",
        "patterns": [
            ("encode", "decode"),
            ("serialize", "deserialize"),
            ("compress", "decompress"),
            ("encrypt", "decrypt"),
            ("to_json", "from_json"),
            ("to_dict", "from_dict"),
        ],
        "template": """
@given({strategies})
def test_{func_name}_inverse(self, {params}):
    \"\"\"Property: {inverse_func}({func_name}(x)) == x\"\"\"
    encoded = {call}
    decoded = {inverse_call}
    assert decoded == {original}
""",
    },
    "associative": {
        "description": "Operation is associative: f(f(a, b), c) == f(a, f(b, c))",
        "patterns": ["concat", "append", "chain", "compose"],
        "template": """
@given({strategies})
def test_{func_name}_associative(self, {params}):
    \"\"\"Property: {func_name}({func_name}(a, b), c) == {func_name}(a, {func_name}(b, c))\"\"\"
    result1 = {func_name}({func_name}(a, b), c)
    result2 = {func_name}(a, {func_name}(b, c))
    assert result1 == result2
""",
    },
    "roundtrip": {
        "description": "Roundtrip through transformation preserves data",
        "patterns": ["parse", "format", "convert", "transform"],
        "template": """
@given({strategies})
def test_{func_name}_roundtrip(self, {params}):
    \"\"\"Property: Roundtrip preserves data\"\"\"
    original = {original}
    transformed = {call}
    restored = {inverse_call}
    assert restored == original or transformed == original
""",
    },
}


# ============================================================================
# Property-Based Test Generator
# ============================================================================

class PropertyTestGenerator:
    """
    Generates property-based tests using Hypothesis.

    PBTSO Phase: SKILL
    Bus Topics: test.property.generate, test.property.generated
    """

    BUS_TOPICS = {
        "generate": "test.property.generate",
        "generated": "test.property.generated",
    }

    def __init__(self, bus=None):
        """
        Initialize the property-based test generator.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus

    def generate(self, request: Dict[str, Any]) -> PropertyResult:
        """
        Generate property-based tests for a source file.

        Args:
            request: Generation request parameters

        Returns:
            PropertyResult with generated tests
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Parse request
        gen_request = self._parse_request(request)

        # Emit generation start event
        self._emit_event("generate", {
            "request_id": request_id,
            "source_file": gen_request.source_file,
            "status": "started",
        })

        tests = []
        errors = []
        warnings = []
        properties_discovered = 0

        try:
            # Read and parse source file
            source_path = Path(gen_request.source_file)
            if not source_path.exists():
                errors.append(f"Source file not found: {gen_request.source_file}")
            else:
                source_code = source_path.read_text()
                tree = ast.parse(source_code)

                # Extract functions
                functions = self._extract_functions(tree)

                # Filter by target functions
                if gen_request.target_functions:
                    functions = [
                        f for f in functions
                        if f["name"] in gen_request.target_functions
                    ]

                # Discover properties for each function
                for func_info in functions:
                    properties = self._discover_properties(
                        func_info,
                        gen_request.property_types,
                    )
                    properties_discovered += len(properties)

                    # Generate tests for each property
                    for prop in properties:
                        test = self._generate_property_test(
                            func_info,
                            prop,
                            source_path.stem,
                            gen_request,
                        )
                        if test:
                            tests.append(test)

        except SyntaxError as e:
            errors.append(f"Syntax error in source file: {e}")
        except Exception as e:
            errors.append(f"Error during generation: {e}")

        generation_time = time.time() - start_time

        result = PropertyResult(
            request_id=request_id,
            source_file=gen_request.source_file,
            tests_generated=tests,
            properties_discovered=properties_discovered,
            generation_time_s=generation_time,
            errors=errors,
            warnings=warnings,
        )

        # Emit generation complete event
        self._emit_event("generated", {
            "request_id": request_id,
            "source_file": gen_request.source_file,
            "tests_count": len(tests),
            "properties_discovered": properties_discovered,
            "status": "completed" if not errors else "completed_with_errors",
        })

        return result

    def _parse_request(self, request: Dict[str, Any]) -> PropertyRequest:
        """Parse generation request from dictionary."""
        return PropertyRequest(
            source_file=request.get("source_file", ""),
            target_functions=request.get("target_functions"),
            property_types=request.get("property_types"),
            max_examples=request.get("max_examples", 100),
            deadline_ms=request.get("deadline_ms"),
            seed=request.get("seed"),
        )

    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function information from AST."""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("_"):
                    continue

                args = []
                for arg in node.args.args:
                    if arg.arg in ("self", "cls"):
                        continue
                    arg_info = {"name": arg.arg}
                    if arg.annotation:
                        arg_info["type"] = self._get_annotation(arg.annotation)
                    else:
                        arg_info["type"] = "Any"
                    args.append(arg_info)

                return_type = None
                if node.returns:
                    return_type = self._get_annotation(node.returns)

                functions.append({
                    "name": node.name,
                    "args": args,
                    "return_type": return_type,
                    "docstring": ast.get_docstring(node),
                })

        return functions

    def _get_annotation(self, node) -> str:
        """Get type annotation as string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                return f"{node.value.id}[{self._get_annotation(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        return "Any"

    def _discover_properties(
        self,
        func_info: Dict[str, Any],
        property_types: Optional[List[str]],
    ) -> List[PropertySpec]:
        """Discover testable properties for a function."""
        properties = []
        func_name = func_info["name"]
        func_name_lower = func_name.lower()

        types_to_check = property_types or list(PROPERTY_PATTERNS.keys())

        for prop_type in types_to_check:
            if prop_type not in PROPERTY_PATTERNS:
                continue

            pattern_info = PROPERTY_PATTERNS[prop_type]
            patterns = pattern_info["patterns"]

            # Check if function name matches any pattern
            matches = False
            if prop_type == "inverse":
                # Special handling for inverse pairs
                for forward, backward in patterns:
                    if forward in func_name_lower:
                        matches = True
                        break
            else:
                matches = any(p in func_name_lower for p in patterns)

            if matches:
                prop = PropertySpec(
                    name=f"{func_name}_{prop_type}",
                    description=pattern_info["description"],
                    property_type=prop_type,
                    function=func_name,
                    input_types=[a["type"] for a in func_info["args"]],
                    output_type=func_info.get("return_type"),
                )
                properties.append(prop)

        # Always add basic invariant tests
        if not properties:
            prop = PropertySpec(
                name=f"{func_name}_doesnt_crash",
                description="Function handles arbitrary valid input without crashing",
                property_type="invariant",
                function=func_name,
                input_types=[a["type"] for a in func_info["args"]],
                output_type=func_info.get("return_type"),
            )
            properties.append(prop)

        return properties

    def _generate_property_test(
        self,
        func_info: Dict[str, Any],
        prop: PropertySpec,
        module_name: str,
        request: PropertyRequest,
    ) -> Optional[PropertyTestCase]:
        """Generate a property test case."""
        func_name = func_info["name"]
        args = func_info["args"]

        # Generate strategies for each argument
        strategies = {}
        strategy_strs = []
        param_names = []

        for arg in args:
            arg_name = arg["name"]
            arg_type = arg["type"]
            strategy = self._type_to_strategy(arg_type)
            strategies[arg_name] = strategy
            strategy_strs.append(f"{arg_name}={strategy}")
            param_names.append(arg_name)

        # Generate test code based on property type
        test_code = self._render_property_test(
            func_name,
            prop,
            strategies,
            param_names,
            request,
        )

        settings = {
            "max_examples": request.max_examples,
        }
        if request.deadline_ms:
            settings["deadline"] = request.deadline_ms
        if request.seed:
            settings["database"] = None  # Deterministic with seed

        return PropertyTestCase(
            id=str(uuid.uuid4()),
            name=f"test_{prop.name}",
            target_module=module_name,
            target_function=func_name,
            property_spec=prop,
            test_code=test_code,
            strategies=strategies,
            settings=settings,
            tags={"property", prop.property_type, "auto_generated"},
            priority=2,
        )

    def _type_to_strategy(self, type_hint: str) -> str:
        """Convert a type hint to a Hypothesis strategy."""
        type_lower = type_hint.lower()

        # Check exact matches
        if type_hint in TYPE_STRATEGIES:
            return TYPE_STRATEGIES[type_hint]

        # Check partial matches
        for type_name, strategy in TYPE_STRATEGIES.items():
            if type_name.lower() in type_lower:
                return strategy

        # Handle generic types
        if "[" in type_hint:
            outer = type_hint.split("[")[0]
            inner = type_hint.split("[")[1].rstrip("]")
            inner_strategy = self._type_to_strategy(inner)

            if outer in TYPE_STRATEGIES:
                template = TYPE_STRATEGIES[outer]
                if "{inner}" in template:
                    return template.format(inner=inner_strategy)

        # Default to text
        return "st.text(min_size=0, max_size=50)"

    def _render_property_test(
        self,
        func_name: str,
        prop: PropertySpec,
        strategies: Dict[str, str],
        param_names: List[str],
        request: PropertyRequest,
    ) -> str:
        """Render test code for a property."""
        strategy_str = ", ".join(f"{k}={v}" for k, v in strategies.items())
        params_str = ", ".join(param_names)
        call_str = f"{func_name}({params_str})"

        if prop.property_type == "invariant":
            return f'''
@settings(max_examples={request.max_examples})
@given({strategy_str})
def test_{prop.name}({params_str}):
    """Property: {prop.description}"""
    try:
        result = {call_str}
        # Function completed without error
        assert True
    except (ValueError, TypeError) as e:
        # Expected errors are acceptable
        pass
'''

        elif prop.property_type == "idempotent":
            return f'''
@settings(max_examples={request.max_examples})
@given({strategy_str})
def test_{prop.name}({params_str}):
    """Property: {func_name}({func_name}(x)) == {func_name}(x)"""
    result1 = {call_str}
    result2 = {func_name}(result1)
    assert result1 == result2
'''

        elif prop.property_type == "commutative":
            if len(param_names) >= 2:
                a, b = param_names[0], param_names[1]
                return f'''
@settings(max_examples={request.max_examples})
@given({strategy_str})
def test_{prop.name}({params_str}):
    """Property: {func_name}(a, b) == {func_name}(b, a)"""
    result1 = {func_name}({a}, {b})
    result2 = {func_name}({b}, {a})
    assert result1 == result2
'''

        elif prop.property_type == "inverse":
            # Try to find inverse function
            inverse_name = self._find_inverse_function(func_name)
            if inverse_name:
                return f'''
@settings(max_examples={request.max_examples})
@given({strategy_str})
def test_{prop.name}({params_str}):
    """Property: {inverse_name}({func_name}(x)) == x"""
    original = {param_names[0] if param_names else "x"}
    encoded = {call_str}
    decoded = {inverse_name}(encoded)
    assert decoded == original
'''

        # Default template
        return f'''
@settings(max_examples={request.max_examples})
@given({strategy_str})
def test_{prop.name}({params_str}):
    """Property: {prop.description}"""
    result = {call_str}
    assert result is not None or result == {{}}  # Basic sanity check
'''

    def _find_inverse_function(self, func_name: str) -> Optional[str]:
        """Find the inverse function name if it exists."""
        inverse_pairs = [
            ("encode", "decode"),
            ("serialize", "deserialize"),
            ("compress", "decompress"),
            ("encrypt", "decrypt"),
            ("to_json", "from_json"),
            ("to_dict", "from_dict"),
            ("pack", "unpack"),
            ("marshal", "unmarshal"),
        ]

        func_lower = func_name.lower()
        for forward, backward in inverse_pairs:
            if forward in func_lower:
                return func_name.replace(forward, backward)
            if backward in func_lower:
                return func_name.replace(backward, forward)

        return None

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        if self.bus:
            topic = self.BUS_TOPICS.get(event_type, f"test.property.{event_type}")
            self.bus.emit({
                "topic": topic,
                "kind": "test_generation",
                "actor": "test-agent",
                "data": data,
            })

    def render_test_file(self, result: PropertyResult) -> str:
        """
        Render generated tests as a pytest test file.

        Args:
            result: Generation result with test cases

        Returns:
            Python source code for test file
        """
        lines = [
            '"""',
            f'Auto-generated property-based tests for {result.source_file}',
            f'Generated by Test Agent (request_id: {result.request_id})',
            '"""',
            'import pytest',
            'from hypothesis import given, settings, strategies as st',
            f'from {Path(result.source_file).stem} import *',
            '',
            '',
        ]

        for test in result.tests_generated:
            lines.append(test.test_code)
            lines.append('')

        return '\n'.join(lines)
