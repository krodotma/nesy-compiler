#!/usr/bin/env python3
"""
Step 102: Unit Test Generator

Generates unit tests for individual functions and classes.

PBTSO Phase: SKILL
Bus Topics:
- test.unit.generate (subscribes)
- test.unit.generated (emits)

Dependencies: Step 101 (Bootstrap)
"""
from __future__ import annotations

import ast
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TestCase:
    """Represents a generated test case."""
    id: str
    name: str
    target_module: str
    target_function: str
    test_type: str = "unit"
    inputs: List[Any] = field(default_factory=list)
    expected_output: Optional[Any] = None
    expected_exception: Optional[str] = None
    assertions: List[str] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    test_code: str = ""
    tags: Set[str] = field(default_factory=set)
    priority: int = 1
    timeout_s: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "target_module": self.target_module,
            "target_function": self.target_function,
            "test_type": self.test_type,
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "expected_exception": self.expected_exception,
            "assertions": self.assertions,
            "setup_code": self.setup_code,
            "teardown_code": self.teardown_code,
            "test_code": self.test_code,
            "tags": list(self.tags),
            "priority": self.priority,
            "timeout_s": self.timeout_s,
        }


@dataclass
class GenerationRequest:
    """Request to generate unit tests."""
    source_file: str
    target_functions: Optional[List[str]] = None  # None = all functions
    target_classes: Optional[List[str]] = None
    coverage_target: float = 0.8
    include_edge_cases: bool = True
    include_error_cases: bool = True
    max_tests_per_function: int = 5
    framework: str = "pytest"


@dataclass
class GenerationResult:
    """Result of test generation."""
    request_id: str
    source_file: str
    tests_generated: List[TestCase]
    coverage_estimate: float
    generation_time_s: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "source_file": self.source_file,
            "tests_generated": [t.to_dict() for t in self.tests_generated],
            "coverage_estimate": self.coverage_estimate,
            "generation_time_s": self.generation_time_s,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ============================================================================
# AST Analysis Helpers
# ============================================================================

class FunctionAnalyzer(ast.NodeVisitor):
    """Analyzes Python AST to extract function information."""

    def __init__(self):
        self.functions: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self._current_class: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        func_info = self._extract_function_info(node)
        if self._current_class:
            func_info["class"] = self._current_class
        self.functions.append(func_info)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        func_info = self._extract_function_info(node)
        func_info["async"] = True
        if self._current_class:
            func_info["class"] = self._current_class
        self.functions.append(func_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        class_info = {
            "name": node.name,
            "bases": [self._get_name(b) for b in node.bases],
            "methods": [],
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "lineno": node.lineno,
        }
        self.classes.append(class_info)

        # Track current class for method association
        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def _extract_function_info(self, node) -> Dict[str, Any]:
        """Extract information from a function node."""
        args = []
        defaults = []

        for arg in node.args.args:
            arg_info = {"name": arg.arg}
            if arg.annotation:
                arg_info["type"] = self._get_annotation(arg.annotation)
            args.append(arg_info)

        for default in node.args.defaults:
            defaults.append(self._get_default_value(default))

        return_type = None
        if node.returns:
            return_type = self._get_annotation(node.returns)

        docstring = ast.get_docstring(node)

        return {
            "name": node.name,
            "args": args,
            "defaults": defaults,
            "return_type": return_type,
            "docstring": docstring,
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "lineno": node.lineno,
            "async": False,
            "class": None,
        }

    def _get_name(self, node) -> str:
        """Get name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def _get_annotation(self, node) -> str:
        """Get type annotation as string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_annotation(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        return "Any"

    def _get_default_value(self, node) -> Any:
        """Get default value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._get_default_value(e) for e in node.elts]
        elif isinstance(node, ast.Dict):
            return {}
        elif isinstance(node, ast.Name):
            if node.id in ("None", "True", "False"):
                return eval(node.id)
            return f"<{node.id}>"
        return None

    def _get_decorator_name(self, node) -> str:
        """Get decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"


# ============================================================================
# Unit Test Generator
# ============================================================================

class UnitTestGenerator:
    """
    Generates unit tests for Python source files.

    PBTSO Phase: SKILL
    Bus Topics: test.unit.generate, test.unit.generated
    """

    BUS_TOPICS = {
        "generate": "test.unit.generate",
        "generated": "test.unit.generated",
    }

    def __init__(self, bus=None):
        """
        Initialize the unit test generator.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus
        self._test_templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load test code templates."""
        return {
            "pytest_function": '''
def test_{function_name}_{case_name}():
    """Test {function_name}: {description}"""
    {setup}
    result = {call_expression}
    {assertions}
''',
            "pytest_class": '''
class Test{class_name}:
    """Tests for {class_name}"""

    def setup_method(self):
        """Set up test fixtures."""
        {setup}

    def teardown_method(self):
        """Tear down test fixtures."""
        {teardown}

{test_methods}
''',
            "pytest_exception": '''
def test_{function_name}_raises_{exception_name}():
    """Test {function_name} raises {exception_name}"""
    {setup}
    with pytest.raises({exception_name}):
        {call_expression}
''',
            "pytest_parametrized": '''
@pytest.mark.parametrize("{param_names}", {param_values})
def test_{function_name}_parametrized({param_names}):
    """Test {function_name} with parametrized inputs"""
    result = {call_expression}
    {assertions}
''',
        }

    def generate(self, request: Dict[str, Any]) -> GenerationResult:
        """
        Generate unit tests for a source file.

        Args:
            request: Generation request parameters

        Returns:
            GenerationResult with generated tests
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

        try:
            # Read and parse source file
            source_path = Path(gen_request.source_file)
            if not source_path.exists():
                errors.append(f"Source file not found: {gen_request.source_file}")
            else:
                source_code = source_path.read_text()
                tree = ast.parse(source_code)

                # Analyze AST
                analyzer = FunctionAnalyzer()
                analyzer.visit(tree)

                # Generate tests for each function
                for func_info in analyzer.functions:
                    # Skip private/dunder functions unless explicitly requested
                    if func_info["name"].startswith("_") and not func_info["name"].startswith("__"):
                        if gen_request.target_functions is None:
                            continue

                    # Filter by target functions if specified
                    if gen_request.target_functions:
                        if func_info["name"] not in gen_request.target_functions:
                            continue

                    # Generate tests for this function
                    func_tests = self._generate_function_tests(
                        func_info,
                        gen_request,
                        source_path.stem,
                    )
                    tests.extend(func_tests)

        except SyntaxError as e:
            errors.append(f"Syntax error in source file: {e}")
        except Exception as e:
            errors.append(f"Error during generation: {e}")

        # Calculate coverage estimate
        coverage_estimate = self._estimate_coverage(tests, gen_request)

        generation_time = time.time() - start_time

        result = GenerationResult(
            request_id=request_id,
            source_file=gen_request.source_file,
            tests_generated=tests,
            coverage_estimate=coverage_estimate,
            generation_time_s=generation_time,
            errors=errors,
            warnings=warnings,
        )

        # Emit generation complete event
        self._emit_event("generated", {
            "request_id": request_id,
            "source_file": gen_request.source_file,
            "tests_count": len(tests),
            "coverage_estimate": coverage_estimate,
            "status": "completed" if not errors else "completed_with_errors",
        })

        return result

    def _parse_request(self, request: Dict[str, Any]) -> GenerationRequest:
        """Parse generation request from dictionary."""
        return GenerationRequest(
            source_file=request.get("source_file", ""),
            target_functions=request.get("target_functions"),
            target_classes=request.get("target_classes"),
            coverage_target=request.get("coverage_target", 0.8),
            include_edge_cases=request.get("include_edge_cases", True),
            include_error_cases=request.get("include_error_cases", True),
            max_tests_per_function=request.get("max_tests_per_function", 5),
            framework=request.get("framework", "pytest"),
        )

    def _generate_function_tests(
        self,
        func_info: Dict[str, Any],
        request: GenerationRequest,
        module_name: str,
    ) -> List[TestCase]:
        """Generate tests for a single function."""
        tests = []
        func_name = func_info["name"]

        # 1. Basic happy path test
        tests.append(self._generate_happy_path_test(func_info, module_name))

        # 2. Edge case tests
        if request.include_edge_cases:
            edge_tests = self._generate_edge_case_tests(func_info, module_name)
            tests.extend(edge_tests[:request.max_tests_per_function - 1])

        # 3. Error case tests
        if request.include_error_cases:
            error_tests = self._generate_error_case_tests(func_info, module_name)
            tests.extend(error_tests[:2])  # Limit error tests

        return tests[:request.max_tests_per_function]

    def _generate_happy_path_test(
        self,
        func_info: Dict[str, Any],
        module_name: str,
    ) -> TestCase:
        """Generate a happy path test case."""
        func_name = func_info["name"]
        args = func_info.get("args", [])

        # Generate sample inputs based on type hints
        inputs = []
        input_code = []
        for i, arg in enumerate(args):
            if arg["name"] in ("self", "cls"):
                continue
            arg_type = arg.get("type", "Any")
            sample_value = self._sample_value_for_type(arg_type, arg["name"])
            inputs.append(sample_value)
            input_code.append(f"{arg['name']}={repr(sample_value)}")

        # Build call expression
        call_expr = f"{func_name}({', '.join(input_code)})"

        # Generate test code
        test_code = self._test_templates["pytest_function"].format(
            function_name=func_name,
            case_name="happy_path",
            description="basic functionality",
            setup="",
            call_expression=call_expr,
            assertions="assert result is not None  # TODO: Add specific assertion",
        )

        return TestCase(
            id=str(uuid.uuid4()),
            name=f"test_{func_name}_happy_path",
            target_module=module_name,
            target_function=func_name,
            test_type="unit",
            inputs=inputs,
            assertions=["assert result is not None"],
            test_code=test_code,
            tags={"happy_path", "auto_generated"},
            priority=1,
        )

    def _generate_edge_case_tests(
        self,
        func_info: Dict[str, Any],
        module_name: str,
    ) -> List[TestCase]:
        """Generate edge case test cases."""
        tests = []
        func_name = func_info["name"]
        args = func_info.get("args", [])

        edge_cases = [
            ("empty_input", "", []),
            ("none_input", None, None),
            ("zero_input", 0, 0),
            ("negative_input", -1, -1),
            ("large_input", 10**9, 10**9),
        ]

        for case_name, str_val, num_val in edge_cases:
            for arg in args:
                if arg["name"] in ("self", "cls"):
                    continue

                arg_type = arg.get("type", "Any")

                # Select appropriate edge value
                if "str" in arg_type.lower():
                    edge_value = str_val
                elif any(t in arg_type.lower() for t in ["int", "float", "number"]):
                    edge_value = num_val
                elif "list" in arg_type.lower():
                    edge_value = []
                else:
                    continue

                test = TestCase(
                    id=str(uuid.uuid4()),
                    name=f"test_{func_name}_{case_name}",
                    target_module=module_name,
                    target_function=func_name,
                    test_type="unit",
                    inputs=[edge_value],
                    tags={"edge_case", "auto_generated"},
                    priority=2,
                )
                tests.append(test)
                break  # One edge case per type

        return tests

    def _generate_error_case_tests(
        self,
        func_info: Dict[str, Any],
        module_name: str,
    ) -> List[TestCase]:
        """Generate error/exception test cases."""
        tests = []
        func_name = func_info["name"]
        args = func_info.get("args", [])

        # Common error scenarios
        error_scenarios = [
            ("invalid_type", "invalid_type", "TypeError"),
            ("invalid_value", "invalid_value", "ValueError"),
        ]

        for scenario_name, _, expected_exception in error_scenarios:
            test = TestCase(
                id=str(uuid.uuid4()),
                name=f"test_{func_name}_{scenario_name}",
                target_module=module_name,
                target_function=func_name,
                test_type="unit",
                expected_exception=expected_exception,
                tags={"error_case", "auto_generated"},
                priority=3,
            )
            tests.append(test)

        return tests

    def _sample_value_for_type(self, type_hint: str, arg_name: str) -> Any:
        """Generate a sample value based on type hint."""
        type_lower = type_hint.lower()

        if "str" in type_lower:
            return f"test_{arg_name}"
        elif "int" in type_lower:
            return 42
        elif "float" in type_lower:
            return 3.14
        elif "bool" in type_lower:
            return True
        elif "list" in type_lower:
            return [1, 2, 3]
        elif "dict" in type_lower:
            return {"key": "value"}
        elif "path" in type_lower:
            return "/tmp/test"
        elif "optional" in type_lower:
            return None
        else:
            return "test_value"

    def _estimate_coverage(
        self,
        tests: List[TestCase],
        request: GenerationRequest,
    ) -> float:
        """Estimate code coverage from generated tests."""
        if not tests:
            return 0.0

        # Rough estimation based on test count and types
        base_coverage = min(len(tests) * 0.1, 0.6)

        # Bonus for edge cases and error cases
        edge_count = sum(1 for t in tests if "edge_case" in t.tags)
        error_count = sum(1 for t in tests if "error_case" in t.tags)

        bonus = (edge_count * 0.05) + (error_count * 0.05)

        return min(base_coverage + bonus, 0.95)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        if self.bus:
            topic = self.BUS_TOPICS.get(event_type, f"test.unit.{event_type}")
            self.bus.emit({
                "topic": topic,
                "kind": "test_generation",
                "actor": "test-agent",
                "data": data,
            })

    def render_test_file(self, result: GenerationResult) -> str:
        """
        Render generated tests as a pytest test file.

        Args:
            result: Generation result with test cases

        Returns:
            Python source code for test file
        """
        lines = [
            '"""',
            f'Auto-generated unit tests for {result.source_file}',
            f'Generated by Test Agent (request_id: {result.request_id})',
            '"""',
            'import pytest',
            f'from {Path(result.source_file).stem} import *',
            '',
            '',
        ]

        for test in result.tests_generated:
            if test.test_code:
                lines.append(test.test_code)
            else:
                # Generate minimal test code
                lines.append(f'def {test.name}():')
                lines.append(f'    """Test {test.target_function}"""')
                lines.append(f'    # TODO: Implement test')
                lines.append(f'    pass')
            lines.append('')

        return '\n'.join(lines)
