#!/usr/bin/env python3
"""
Step 103: Integration Test Generator

Generates integration tests for module interactions and API endpoints.

PBTSO Phase: SKILL
Bus Topics:
- test.integration.generate (subscribes)
- test.integration.generated (emits)

Dependencies: Step 102 (Unit Test Generator)
"""
from __future__ import annotations

import ast
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ModuleInterface:
    """Describes a module's public interface."""
    module_path: str
    module_name: str
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    api_endpoints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IntegrationTestCase:
    """Represents an integration test case."""
    id: str
    name: str
    description: str
    source_module: str
    target_module: str
    interaction_type: str  # import, call, api, event
    setup_modules: List[str] = field(default_factory=list)
    setup_code: str = ""
    test_code: str = ""
    teardown_code: str = ""
    assertions: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    mocks: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    priority: int = 2
    timeout_s: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_module": self.source_module,
            "target_module": self.target_module,
            "interaction_type": self.interaction_type,
            "setup_modules": self.setup_modules,
            "setup_code": self.setup_code,
            "test_code": self.test_code,
            "teardown_code": self.teardown_code,
            "assertions": self.assertions,
            "fixtures": self.fixtures,
            "mocks": self.mocks,
            "tags": list(self.tags),
            "priority": self.priority,
            "timeout_s": self.timeout_s,
        }


@dataclass
class IntegrationRequest:
    """Request to generate integration tests."""
    source_files: List[str]
    dependency_graph: Optional[Dict[str, List[str]]] = None
    focus_interactions: Optional[List[Tuple[str, str]]] = None
    include_api_tests: bool = True
    include_event_tests: bool = True
    max_depth: int = 2
    framework: str = "pytest"


@dataclass
class IntegrationResult:
    """Result of integration test generation."""
    request_id: str
    modules_analyzed: List[str]
    tests_generated: List[IntegrationTestCase]
    dependency_graph: Dict[str, List[str]]
    interaction_count: int
    generation_time_s: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "modules_analyzed": self.modules_analyzed,
            "tests_generated": [t.to_dict() for t in self.tests_generated],
            "dependency_graph": self.dependency_graph,
            "interaction_count": self.interaction_count,
            "generation_time_s": self.generation_time_s,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ============================================================================
# Import Analysis
# ============================================================================

class ImportAnalyzer(ast.NodeVisitor):
    """Analyzes Python imports and module dependencies."""

    def __init__(self):
        self.imports: List[str] = []
        self.from_imports: List[Tuple[str, List[str]]] = []
        self.function_calls: List[Tuple[str, str]] = []  # (module, function)
        self._current_imports: Dict[str, str] = {}

    def visit_Import(self, node: ast.Import):
        """Visit import statement."""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports.append(alias.name)
            self._current_imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from-import statement."""
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.from_imports.append((module, names))
        for alias in node.names:
            name = alias.asname or alias.name
            self._current_imports[name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Visit function call to track cross-module calls."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module = node.func.value.id
                func = node.func.attr
                if module in self._current_imports:
                    self.function_calls.append((self._current_imports[module], func))
        self.generic_visit(node)

    def get_dependencies(self) -> List[str]:
        """Get list of module dependencies."""
        deps = set(self.imports)
        for module, _ in self.from_imports:
            if module:
                deps.add(module.split(".")[0])
        return list(deps)


# ============================================================================
# Integration Test Generator
# ============================================================================

class IntegrationTestGenerator:
    """
    Generates integration tests for module interactions.

    PBTSO Phase: SKILL
    Bus Topics: test.integration.generate, test.integration.generated
    """

    BUS_TOPICS = {
        "generate": "test.integration.generate",
        "generated": "test.integration.generated",
    }

    # Common integration patterns
    INTERACTION_PATTERNS = {
        "database": ["connect", "query", "execute", "commit", "rollback"],
        "http": ["get", "post", "put", "delete", "request", "fetch"],
        "file": ["read", "write", "open", "close", "save", "load"],
        "cache": ["get", "set", "delete", "clear", "invalidate"],
        "queue": ["publish", "subscribe", "consume", "send", "receive"],
        "event": ["emit", "on", "subscribe", "dispatch", "handle"],
    }

    def __init__(self, bus=None):
        """
        Initialize the integration test generator.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus
        self._dependency_cache: Dict[str, List[str]] = {}

    def generate(self, request: Dict[str, Any]) -> IntegrationResult:
        """
        Generate integration tests for a set of modules.

        Args:
            request: Generation request parameters

        Returns:
            IntegrationResult with generated tests
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Parse request
        gen_request = self._parse_request(request)

        # Emit generation start event
        self._emit_event("generate", {
            "request_id": request_id,
            "source_files": gen_request.source_files,
            "status": "started",
        })

        tests = []
        errors = []
        warnings = []
        modules_analyzed = []
        dependency_graph: Dict[str, List[str]] = {}

        try:
            # Analyze each source file
            module_interfaces = []
            for source_file in gen_request.source_files:
                source_path = Path(source_file)
                if not source_path.exists():
                    errors.append(f"Source file not found: {source_file}")
                    continue

                interface = self._analyze_module(source_path)
                module_interfaces.append(interface)
                modules_analyzed.append(interface.module_name)
                dependency_graph[interface.module_name] = interface.dependencies

            # Build or use provided dependency graph
            if gen_request.dependency_graph:
                dependency_graph.update(gen_request.dependency_graph)

            # Find interaction points
            interactions = self._find_interactions(
                module_interfaces,
                dependency_graph,
                gen_request.focus_interactions,
            )

            # Generate tests for each interaction
            for source_mod, target_mod, interaction_type in interactions:
                test = self._generate_interaction_test(
                    source_mod,
                    target_mod,
                    interaction_type,
                    gen_request,
                )
                if test:
                    tests.append(test)

            # Generate API tests if enabled
            if gen_request.include_api_tests:
                for interface in module_interfaces:
                    api_tests = self._generate_api_tests(interface)
                    tests.extend(api_tests)

            # Generate event tests if enabled
            if gen_request.include_event_tests:
                event_tests = self._generate_event_tests(module_interfaces)
                tests.extend(event_tests)

        except Exception as e:
            errors.append(f"Error during generation: {e}")

        generation_time = time.time() - start_time

        result = IntegrationResult(
            request_id=request_id,
            modules_analyzed=modules_analyzed,
            tests_generated=tests,
            dependency_graph=dependency_graph,
            interaction_count=len(tests),
            generation_time_s=generation_time,
            errors=errors,
            warnings=warnings,
        )

        # Emit generation complete event
        self._emit_event("generated", {
            "request_id": request_id,
            "modules_analyzed": modules_analyzed,
            "tests_count": len(tests),
            "status": "completed" if not errors else "completed_with_errors",
        })

        return result

    def _parse_request(self, request: Dict[str, Any]) -> IntegrationRequest:
        """Parse generation request from dictionary."""
        return IntegrationRequest(
            source_files=request.get("source_files", []),
            dependency_graph=request.get("dependency_graph"),
            focus_interactions=request.get("focus_interactions"),
            include_api_tests=request.get("include_api_tests", True),
            include_event_tests=request.get("include_event_tests", True),
            max_depth=request.get("max_depth", 2),
            framework=request.get("framework", "pytest"),
        )

    def _analyze_module(self, source_path: Path) -> ModuleInterface:
        """Analyze a module's interface and dependencies."""
        source_code = source_path.read_text()
        tree = ast.parse(source_code)

        analyzer = ImportAnalyzer()
        analyzer.visit(tree)

        # Extract exports (__all__ or public names)
        exports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            exports = [
                                elt.value for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]

        # If no __all__, use public names
        if not exports:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not node.name.startswith("_"):
                        exports.append(node.name)

        # Detect API endpoints (FastAPI/Flask patterns)
        api_endpoints = self._detect_api_endpoints(tree)

        return ModuleInterface(
            module_path=str(source_path),
            module_name=source_path.stem,
            imports=analyzer.imports,
            exports=exports,
            dependencies=analyzer.get_dependencies(),
            entry_points=[f for f in exports if f.startswith("main") or f.endswith("_main")],
            api_endpoints=api_endpoints,
        )

    def _detect_api_endpoints(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect API endpoints from decorator patterns."""
        endpoints = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    # FastAPI/Flask route decorators
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            method = decorator.func.attr
                            if method in ("get", "post", "put", "delete", "patch"):
                                path = ""
                                if decorator.args:
                                    if isinstance(decorator.args[0], ast.Constant):
                                        path = decorator.args[0].value
                                endpoints.append({
                                    "method": method.upper(),
                                    "path": path,
                                    "handler": node.name,
                                })

        return endpoints

    def _find_interactions(
        self,
        interfaces: List[ModuleInterface],
        dep_graph: Dict[str, List[str]],
        focus: Optional[List[Tuple[str, str]]],
    ) -> List[Tuple[str, str, str]]:
        """Find interaction points between modules."""
        interactions = []

        # Use focus interactions if provided
        if focus:
            for source, target in focus:
                interactions.append((source, target, "explicit"))
            return interactions

        # Find interactions from dependency graph
        module_names = {i.module_name for i in interfaces}

        for interface in interfaces:
            for dep in interface.dependencies:
                # Only consider internal dependencies
                if dep in module_names or dep.split(".")[0] in module_names:
                    interaction_type = self._classify_interaction(interface, dep)
                    interactions.append((interface.module_name, dep, interaction_type))

        return interactions

    def _classify_interaction(self, interface: ModuleInterface, dependency: str) -> str:
        """Classify the type of interaction between modules."""
        dep_lower = dependency.lower()

        for pattern_type, keywords in self.INTERACTION_PATTERNS.items():
            if any(kw in dep_lower for kw in keywords):
                return pattern_type

        # Check for common module types
        if "api" in dep_lower or "router" in dep_lower:
            return "http"
        if "db" in dep_lower or "model" in dep_lower:
            return "database"
        if "cache" in dep_lower or "redis" in dep_lower:
            return "cache"

        return "import"

    def _generate_interaction_test(
        self,
        source_mod: str,
        target_mod: str,
        interaction_type: str,
        request: IntegrationRequest,
    ) -> Optional[IntegrationTestCase]:
        """Generate a test for a specific module interaction."""
        test_name = f"test_{source_mod}_integrates_with_{target_mod}"

        # Generate appropriate setup/test code based on interaction type
        setup_code, test_code, mocks = self._generate_interaction_code(
            source_mod,
            target_mod,
            interaction_type,
        )

        return IntegrationTestCase(
            id=str(uuid.uuid4()),
            name=test_name,
            description=f"Integration test: {source_mod} -> {target_mod} ({interaction_type})",
            source_module=source_mod,
            target_module=target_mod,
            interaction_type=interaction_type,
            setup_modules=[source_mod, target_mod],
            setup_code=setup_code,
            test_code=test_code,
            mocks=mocks,
            assertions=[
                f"assert {source_mod} can interact with {target_mod}",
            ],
            tags={"integration", interaction_type, "auto_generated"},
            priority=2,
        )

    def _generate_interaction_code(
        self,
        source_mod: str,
        target_mod: str,
        interaction_type: str,
    ) -> Tuple[str, str, Dict[str, str]]:
        """Generate code for interaction test."""
        mocks: Dict[str, str] = {}

        if interaction_type == "database":
            setup_code = f"""
    # Set up test database
    test_db = setup_test_database()
"""
            test_code = f"""
    # Test {source_mod} database interaction with {target_mod}
    from {source_mod} import *
    # Execute database operation
    result = execute_with_db(test_db)
    assert result is not None
"""
            mocks = {target_mod: "MagicMock()"}

        elif interaction_type == "http":
            setup_code = f"""
    # Set up test client
    client = TestClient(app)
"""
            test_code = f"""
    # Test {source_mod} HTTP interaction
    response = client.get("/test-endpoint")
    assert response.status_code == 200
"""
            mocks = {}

        elif interaction_type == "event":
            setup_code = f"""
    # Set up event bus mock
    mock_bus = MagicMock()
"""
            test_code = f"""
    # Test {source_mod} event emission to {target_mod}
    from {source_mod} import *
    # Trigger event
    emit_event("test.event", {{"data": "test"}})
    mock_bus.emit.assert_called_once()
"""
            mocks = {"bus": "mock_bus"}

        else:  # import/generic
            setup_code = ""
            test_code = f"""
    # Test {source_mod} import of {target_mod}
    from {source_mod} import *
    from {target_mod} import *
    # Verify modules can be imported together
    assert True
"""
            mocks = {}

        return setup_code, test_code, mocks

    def _generate_api_tests(
        self,
        interface: ModuleInterface,
    ) -> List[IntegrationTestCase]:
        """Generate tests for API endpoints."""
        tests = []

        for endpoint in interface.api_endpoints:
            method = endpoint["method"]
            path = endpoint["path"]
            handler = endpoint["handler"]

            test = IntegrationTestCase(
                id=str(uuid.uuid4()),
                name=f"test_api_{handler}_{method.lower()}",
                description=f"API test: {method} {path}",
                source_module=interface.module_name,
                target_module="api",
                interaction_type="http",
                test_code=f"""
    response = client.{method.lower()}("{path}")
    assert response.status_code in [200, 201, 204]
""",
                assertions=[f"assert response.status_code in [200, 201, 204]"],
                fixtures=["client"],
                tags={"integration", "api", "auto_generated"},
                priority=2,
            )
            tests.append(test)

        return tests

    def _generate_event_tests(
        self,
        interfaces: List[ModuleInterface],
    ) -> List[IntegrationTestCase]:
        """Generate tests for event-based interactions."""
        tests = []

        # Look for event patterns in module names/exports
        for interface in interfaces:
            if any("event" in e.lower() or "bus" in e.lower() for e in interface.exports):
                test = IntegrationTestCase(
                    id=str(uuid.uuid4()),
                    name=f"test_events_{interface.module_name}",
                    description=f"Event integration test for {interface.module_name}",
                    source_module=interface.module_name,
                    target_module="event_bus",
                    interaction_type="event",
                    test_code=f"""
    # Verify event emission and handling
    with event_capture() as events:
        trigger_action()
        assert len(events) > 0
""",
                    assertions=["assert len(events) > 0"],
                    tags={"integration", "event", "auto_generated"},
                    priority=3,
                )
                tests.append(test)

        return tests

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        if self.bus:
            topic = self.BUS_TOPICS.get(event_type, f"test.integration.{event_type}")
            self.bus.emit({
                "topic": topic,
                "kind": "test_generation",
                "actor": "test-agent",
                "data": data,
            })

    def render_test_file(self, result: IntegrationResult) -> str:
        """
        Render generated tests as a pytest test file.

        Args:
            result: Generation result with test cases

        Returns:
            Python source code for test file
        """
        lines = [
            '"""',
            'Auto-generated integration tests',
            f'Generated by Test Agent (request_id: {result.request_id})',
            f'Modules: {", ".join(result.modules_analyzed)}',
            '"""',
            'import pytest',
            'from unittest.mock import MagicMock, patch',
            '',
            '',
            '@pytest.fixture',
            'def mock_bus():',
            '    """Mock event bus fixture."""',
            '    return MagicMock()',
            '',
            '',
        ]

        for test in result.tests_generated:
            lines.append(f'def {test.name}():')
            lines.append(f'    """{test.description}"""')
            if test.setup_code:
                lines.append(test.setup_code)
            if test.test_code:
                lines.append(test.test_code)
            else:
                lines.append('    pass  # TODO: Implement test')
            lines.append('')

        return '\n'.join(lines)
