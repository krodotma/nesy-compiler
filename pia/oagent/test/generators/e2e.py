#!/usr/bin/env python3
"""
Step 104: E2E Test Generator

Generates end-to-end tests for full system workflows.

PBTSO Phase: SKILL
Bus Topics:
- test.e2e.generate (subscribes)
- test.e2e.generated (emits)

Dependencies: Step 103 (Integration Test Generator)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class WorkflowStep:
    """A single step in an E2E workflow."""
    id: str
    name: str
    action: str  # navigate, click, input, assert, wait, api_call
    target: str  # selector, URL, API endpoint
    value: Optional[str] = None
    expected: Optional[Any] = None
    timeout_ms: int = 5000
    screenshot: bool = False


@dataclass
class E2EWorkflow:
    """Describes an end-to-end test workflow."""
    id: str
    name: str
    description: str
    entry_point: str  # URL or API endpoint
    steps: List[WorkflowStep] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    cleanup: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "entry_point": self.entry_point,
            "steps": [
                {
                    "id": s.id,
                    "name": s.name,
                    "action": s.action,
                    "target": s.target,
                    "value": s.value,
                    "expected": s.expected,
                    "timeout_ms": s.timeout_ms,
                    "screenshot": s.screenshot,
                }
                for s in self.steps
            ],
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "cleanup": self.cleanup,
            "tags": list(self.tags),
        }


@dataclass
class E2ETestCase:
    """Represents an E2E test case."""
    id: str
    name: str
    workflow: E2EWorkflow
    framework: str = "playwright"  # playwright, cypress, selenium
    test_code: str = ""
    fixtures: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    priority: int = 3
    timeout_s: int = 120
    retry_count: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "workflow": self.workflow.to_dict(),
            "framework": self.framework,
            "test_code": self.test_code,
            "fixtures": self.fixtures,
            "environment": self.environment,
            "priority": self.priority,
            "timeout_s": self.timeout_s,
            "retry_count": self.retry_count,
        }


@dataclass
class E2ERequest:
    """Request to generate E2E tests."""
    spec_file: Optional[str] = None  # OpenAPI/workflow spec
    base_url: str = "http://localhost:3000"
    user_journeys: Optional[List[str]] = None  # Named journeys to generate
    include_auth_flow: bool = True
    include_error_scenarios: bool = True
    framework: str = "playwright"
    headless: bool = True


@dataclass
class E2EResult:
    """Result of E2E test generation."""
    request_id: str
    workflows_generated: List[E2EWorkflow]
    tests_generated: List[E2ETestCase]
    generation_time_s: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "workflows_generated": [w.to_dict() for w in self.workflows_generated],
            "tests_generated": [t.to_dict() for t in self.tests_generated],
            "generation_time_s": self.generation_time_s,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ============================================================================
# Common User Journeys
# ============================================================================

COMMON_JOURNEYS = {
    "login": {
        "name": "User Login",
        "description": "Standard user authentication flow",
        "steps": [
            {"action": "navigate", "target": "/login"},
            {"action": "input", "target": "#email", "value": "test@example.com"},
            {"action": "input", "target": "#password", "value": "test_password"},
            {"action": "click", "target": "button[type=submit]"},
            {"action": "assert", "target": ".dashboard", "expected": "visible"},
        ],
    },
    "signup": {
        "name": "User Registration",
        "description": "New user signup flow",
        "steps": [
            {"action": "navigate", "target": "/signup"},
            {"action": "input", "target": "#email", "value": "new@example.com"},
            {"action": "input", "target": "#password", "value": "new_password"},
            {"action": "input", "target": "#confirm-password", "value": "new_password"},
            {"action": "click", "target": "button[type=submit]"},
            {"action": "assert", "target": ".welcome-message", "expected": "visible"},
        ],
    },
    "crud": {
        "name": "CRUD Operations",
        "description": "Create, Read, Update, Delete workflow",
        "steps": [
            {"action": "navigate", "target": "/items"},
            {"action": "click", "target": ".create-btn"},
            {"action": "input", "target": "#name", "value": "Test Item"},
            {"action": "click", "target": "button[type=submit]"},
            {"action": "assert", "target": ".item-list", "expected": "contains:Test Item"},
            {"action": "click", "target": ".edit-btn"},
            {"action": "input", "target": "#name", "value": "Updated Item"},
            {"action": "click", "target": "button[type=submit]"},
            {"action": "click", "target": ".delete-btn"},
            {"action": "assert", "target": ".item-list", "expected": "not_contains:Updated Item"},
        ],
    },
    "search": {
        "name": "Search Flow",
        "description": "Search and filter functionality",
        "steps": [
            {"action": "navigate", "target": "/search"},
            {"action": "input", "target": "#search-input", "value": "test query"},
            {"action": "click", "target": ".search-btn"},
            {"action": "wait", "target": ".results", "value": "5000"},
            {"action": "assert", "target": ".results-count", "expected": "gt:0"},
        ],
    },
    "checkout": {
        "name": "Checkout Flow",
        "description": "E-commerce checkout process",
        "steps": [
            {"action": "navigate", "target": "/products"},
            {"action": "click", "target": ".add-to-cart"},
            {"action": "navigate", "target": "/cart"},
            {"action": "click", "target": ".checkout-btn"},
            {"action": "input", "target": "#card-number", "value": "4242424242424242"},
            {"action": "input", "target": "#expiry", "value": "12/25"},
            {"action": "input", "target": "#cvv", "value": "123"},
            {"action": "click", "target": ".pay-btn"},
            {"action": "assert", "target": ".confirmation", "expected": "visible"},
        ],
    },
}


# ============================================================================
# E2E Test Generator
# ============================================================================

class E2ETestGenerator:
    """
    Generates end-to-end tests for full system workflows.

    PBTSO Phase: SKILL
    Bus Topics: test.e2e.generate, test.e2e.generated
    """

    BUS_TOPICS = {
        "generate": "test.e2e.generate",
        "generated": "test.e2e.generated",
    }

    def __init__(self, bus=None):
        """
        Initialize the E2E test generator.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus
        self._templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load E2E test code templates."""
        return {
            "playwright_test": '''
import {{ test, expect }} from '@playwright/test';

test.describe('{workflow_name}', () => {{
    test.beforeEach(async ({{ page }}) => {{
        {setup_code}
    }});

    test('{test_name}', async ({{ page }}) => {{
        {test_steps}
    }});

    test.afterEach(async ({{ page }}) => {{
        {cleanup_code}
    }});
}});
''',
            "pytest_playwright": '''
import pytest
from playwright.sync_api import Page, expect

class Test{workflow_class}:
    """E2E tests for {workflow_name}"""

    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Set up test fixtures."""
        {setup_code}
        yield
        {cleanup_code}

    def test_{test_name}(self, page: Page):
        """{test_description}"""
        {test_steps}
''',
            "cypress_test": '''
describe('{workflow_name}', () => {{
    beforeEach(() => {{
        {setup_code}
    }});

    it('{test_name}', () => {{
        {test_steps}
    }});

    afterEach(() => {{
        {cleanup_code}
    }});
}});
''',
        }

    def generate(self, request: Dict[str, Any]) -> E2EResult:
        """
        Generate E2E tests based on request parameters.

        Args:
            request: Generation request parameters

        Returns:
            E2EResult with generated tests
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Parse request
        gen_request = self._parse_request(request)

        # Emit generation start event
        self._emit_event("generate", {
            "request_id": request_id,
            "base_url": gen_request.base_url,
            "status": "started",
        })

        workflows = []
        tests = []
        errors = []
        warnings = []

        try:
            # Load spec file if provided
            if gen_request.spec_file:
                spec_workflows = self._parse_spec_file(gen_request.spec_file)
                workflows.extend(spec_workflows)

            # Generate from named journeys
            if gen_request.user_journeys:
                for journey_name in gen_request.user_journeys:
                    if journey_name in COMMON_JOURNEYS:
                        workflow = self._create_workflow_from_journey(
                            journey_name,
                            gen_request.base_url,
                        )
                        workflows.append(workflow)
                    else:
                        warnings.append(f"Unknown journey: {journey_name}")

            # Default: generate common workflows
            if not workflows:
                for journey_name in ["login", "crud", "search"]:
                    workflow = self._create_workflow_from_journey(
                        journey_name,
                        gen_request.base_url,
                    )
                    workflows.append(workflow)

            # Add auth flow if requested
            if gen_request.include_auth_flow:
                auth_workflow = self._create_workflow_from_journey("login", gen_request.base_url)
                if auth_workflow not in workflows:
                    workflows.append(auth_workflow)

            # Generate test cases for each workflow
            for workflow in workflows:
                test = self._generate_test_from_workflow(
                    workflow,
                    gen_request.framework,
                    gen_request.headless,
                )
                tests.append(test)

                # Generate error scenario tests
                if gen_request.include_error_scenarios:
                    error_test = self._generate_error_scenario_test(
                        workflow,
                        gen_request.framework,
                    )
                    if error_test:
                        tests.append(error_test)

        except Exception as e:
            errors.append(f"Error during generation: {e}")

        generation_time = time.time() - start_time

        result = E2EResult(
            request_id=request_id,
            workflows_generated=workflows,
            tests_generated=tests,
            generation_time_s=generation_time,
            errors=errors,
            warnings=warnings,
        )

        # Emit generation complete event
        self._emit_event("generated", {
            "request_id": request_id,
            "workflows_count": len(workflows),
            "tests_count": len(tests),
            "status": "completed" if not errors else "completed_with_errors",
        })

        return result

    def _parse_request(self, request: Dict[str, Any]) -> E2ERequest:
        """Parse generation request from dictionary."""
        return E2ERequest(
            spec_file=request.get("spec_file"),
            base_url=request.get("base_url", "http://localhost:3000"),
            user_journeys=request.get("user_journeys"),
            include_auth_flow=request.get("include_auth_flow", True),
            include_error_scenarios=request.get("include_error_scenarios", True),
            framework=request.get("framework", "playwright"),
            headless=request.get("headless", True),
        )

    def _parse_spec_file(self, spec_file: str) -> List[E2EWorkflow]:
        """Parse OpenAPI or workflow spec file."""
        workflows = []
        spec_path = Path(spec_file)

        if not spec_path.exists():
            return workflows

        try:
            with open(spec_path) as f:
                spec = json.load(f)

            # Handle OpenAPI spec
            if "paths" in spec:
                workflows.extend(self._workflows_from_openapi(spec))

            # Handle custom workflow spec
            if "workflows" in spec:
                for w in spec["workflows"]:
                    workflow = E2EWorkflow(
                        id=w.get("id", str(uuid.uuid4())),
                        name=w.get("name", "Unnamed Workflow"),
                        description=w.get("description", ""),
                        entry_point=w.get("entry_point", "/"),
                        steps=[
                            WorkflowStep(
                                id=str(uuid.uuid4()),
                                name=s.get("name", f"Step {i}"),
                                action=s.get("action", "navigate"),
                                target=s.get("target", ""),
                                value=s.get("value"),
                                expected=s.get("expected"),
                            )
                            for i, s in enumerate(w.get("steps", []))
                        ],
                        tags=set(w.get("tags", [])),
                    )
                    workflows.append(workflow)

        except (json.JSONDecodeError, KeyError) as e:
            pass

        return workflows

    def _workflows_from_openapi(self, spec: Dict[str, Any]) -> List[E2EWorkflow]:
        """Generate workflows from OpenAPI spec paths."""
        workflows = []

        for path, methods in spec.get("paths", {}).items():
            for method, details in methods.items():
                if method not in ("get", "post", "put", "delete", "patch"):
                    continue

                workflow = E2EWorkflow(
                    id=str(uuid.uuid4()),
                    name=f"API {method.upper()} {path}",
                    description=details.get("summary", f"Test {method.upper()} {path}"),
                    entry_point=path,
                    steps=[
                        WorkflowStep(
                            id=str(uuid.uuid4()),
                            name=f"{method.upper()} request",
                            action="api_call",
                            target=path,
                            value=method.upper(),
                        ),
                    ],
                    tags={"api", method},
                )
                workflows.append(workflow)

        return workflows

    def _create_workflow_from_journey(
        self,
        journey_name: str,
        base_url: str,
    ) -> E2EWorkflow:
        """Create a workflow from a named journey template."""
        journey = COMMON_JOURNEYS.get(journey_name, {})

        steps = []
        for i, step_data in enumerate(journey.get("steps", [])):
            step = WorkflowStep(
                id=str(uuid.uuid4()),
                name=f"Step {i + 1}: {step_data.get('action', 'unknown')}",
                action=step_data.get("action", "navigate"),
                target=step_data.get("target", ""),
                value=step_data.get("value"),
                expected=step_data.get("expected"),
            )
            steps.append(step)

        return E2EWorkflow(
            id=str(uuid.uuid4()),
            name=journey.get("name", journey_name),
            description=journey.get("description", ""),
            entry_point=base_url + (steps[0].target if steps else "/"),
            steps=steps,
            tags={journey_name, "user_journey"},
        )

    def _generate_test_from_workflow(
        self,
        workflow: E2EWorkflow,
        framework: str,
        headless: bool,
    ) -> E2ETestCase:
        """Generate a test case from a workflow."""
        test_code = self._render_test_code(workflow, framework)

        return E2ETestCase(
            id=str(uuid.uuid4()),
            name=f"test_{self._sanitize_name(workflow.name)}",
            workflow=workflow,
            framework=framework,
            test_code=test_code,
            environment={"HEADLESS": str(headless).lower()},
            priority=3,
        )

    def _generate_error_scenario_test(
        self,
        workflow: E2EWorkflow,
        framework: str,
    ) -> Optional[E2ETestCase]:
        """Generate an error scenario test from a workflow."""
        # Create error version of workflow
        error_steps = []
        for step in workflow.steps:
            if step.action == "input":
                # Use invalid input
                error_step = WorkflowStep(
                    id=str(uuid.uuid4()),
                    name=f"Invalid {step.name}",
                    action=step.action,
                    target=step.target,
                    value="",  # Empty/invalid
                    expected="error_visible",
                )
                error_steps.append(error_step)
                break  # One error per scenario

        if not error_steps:
            return None

        error_workflow = E2EWorkflow(
            id=str(uuid.uuid4()),
            name=f"{workflow.name} - Error Scenario",
            description=f"Error handling test for {workflow.name}",
            entry_point=workflow.entry_point,
            steps=workflow.steps[:1] + error_steps,  # Navigate + error input
            tags=workflow.tags | {"error_scenario"},
        )

        test_code = self._render_test_code(error_workflow, framework)

        return E2ETestCase(
            id=str(uuid.uuid4()),
            name=f"test_{self._sanitize_name(workflow.name)}_error_handling",
            workflow=error_workflow,
            framework=framework,
            test_code=test_code,
            priority=4,
        )

    def _render_test_code(self, workflow: E2EWorkflow, framework: str) -> str:
        """Render test code for a workflow."""
        if framework == "playwright":
            return self._render_playwright_test(workflow)
        elif framework == "cypress":
            return self._render_cypress_test(workflow)
        else:
            return self._render_pytest_playwright_test(workflow)

    def _render_playwright_test(self, workflow: E2EWorkflow) -> str:
        """Render Playwright test code."""
        steps = []
        for step in workflow.steps:
            if step.action == "navigate":
                steps.append(f"        await page.goto('{step.target}');")
            elif step.action == "click":
                steps.append(f"        await page.click('{step.target}');")
            elif step.action == "input":
                steps.append(f"        await page.fill('{step.target}', '{step.value}');")
            elif step.action == "assert":
                if step.expected == "visible":
                    steps.append(f"        await expect(page.locator('{step.target}')).toBeVisible();")
                else:
                    steps.append(f"        await expect(page.locator('{step.target}')).toContainText('{step.expected}');")
            elif step.action == "wait":
                steps.append(f"        await page.waitForSelector('{step.target}', {{ timeout: {step.value or 5000} }});")
            elif step.action == "api_call":
                steps.append(f"        // API call: {step.value} {step.target}")

        test_steps = "\n".join(steps)

        return self._templates["playwright_test"].format(
            workflow_name=workflow.name,
            test_name=self._sanitize_name(workflow.name),
            setup_code="// Setup code here",
            test_steps=test_steps,
            cleanup_code="// Cleanup code here",
        )

    def _render_cypress_test(self, workflow: E2EWorkflow) -> str:
        """Render Cypress test code."""
        steps = []
        for step in workflow.steps:
            if step.action == "navigate":
                steps.append(f"        cy.visit('{step.target}');")
            elif step.action == "click":
                steps.append(f"        cy.get('{step.target}').click();")
            elif step.action == "input":
                steps.append(f"        cy.get('{step.target}').type('{step.value}');")
            elif step.action == "assert":
                if step.expected == "visible":
                    steps.append(f"        cy.get('{step.target}').should('be.visible');")
                else:
                    steps.append(f"        cy.get('{step.target}').should('contain', '{step.expected}');")
            elif step.action == "wait":
                steps.append(f"        cy.get('{step.target}', {{ timeout: {step.value or 5000} }});")

        test_steps = "\n".join(steps)

        return self._templates["cypress_test"].format(
            workflow_name=workflow.name,
            test_name=self._sanitize_name(workflow.name),
            setup_code="// Setup code here",
            test_steps=test_steps,
            cleanup_code="// Cleanup code here",
        )

    def _render_pytest_playwright_test(self, workflow: E2EWorkflow) -> str:
        """Render pytest-playwright test code."""
        steps = []
        for step in workflow.steps:
            if step.action == "navigate":
                steps.append(f"        page.goto('{step.target}')")
            elif step.action == "click":
                steps.append(f"        page.click('{step.target}')")
            elif step.action == "input":
                steps.append(f"        page.fill('{step.target}', '{step.value}')")
            elif step.action == "assert":
                if step.expected == "visible":
                    steps.append(f"        expect(page.locator('{step.target}')).to_be_visible()")
                else:
                    steps.append(f"        expect(page.locator('{step.target}')).to_contain_text('{step.expected}')")
            elif step.action == "wait":
                steps.append(f"        page.wait_for_selector('{step.target}', timeout={step.value or 5000})")

        test_steps = "\n".join(steps)

        return self._templates["pytest_playwright"].format(
            workflow_class=self._to_class_name(workflow.name),
            workflow_name=workflow.name,
            test_name=self._sanitize_name(workflow.name),
            test_description=workflow.description,
            setup_code="pass",
            test_steps=test_steps,
            cleanup_code="pass",
        )

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use as a test function name."""
        return name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")

    def _to_class_name(self, name: str) -> str:
        """Convert a name to a class name."""
        return "".join(word.title() for word in name.split())

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        if self.bus:
            topic = self.BUS_TOPICS.get(event_type, f"test.e2e.{event_type}")
            self.bus.emit({
                "topic": topic,
                "kind": "test_generation",
                "actor": "test-agent",
                "data": data,
            })
