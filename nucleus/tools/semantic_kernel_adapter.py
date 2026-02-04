#!/usr/bin/env python3
"""
Semantic Kernel Adapter for Pluribus
=====================================

Bridges Microsoft Semantic Kernel planners and tool orchestration to Pluribus.
Aligns SK planners with Lens/Collimator `effects` typing and bridges SK tool
schemas to MCP tool format.

Integration points:
- SK Planner -> Lens route planning
- SK Functions -> MCP tools
- SK Memory -> Rhizome artifacts
- SK Kernel -> PluriChat provider chain

Usage:
    from nucleus.tools.semantic_kernel_adapter import (
        SemanticKernelAdapter,
        PluribusTool,
        create_kernel
    )

    adapter = SemanticKernelAdapter()
    kernel = adapter.create_kernel()

    # Register Pluribus tools as SK functions
    adapter.register_tool(my_mcp_tool)

    # Execute a plan with bus tracing
    result = adapter.execute_plan(goal="Summarize the latest research papers")

Service Registry Entry:
    id: semantic-kernel-adapter
    kind: process
    entry_point: nucleus/tools/semantic_kernel_adapter.py
    tags: [semantic-kernel, microsoft, planner, sota]
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

sys.dont_write_bytecode = True

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from agent_bus import resolve_bus_paths, emit_event, default_actor  # type: ignore


# Type definitions aligned with Lens/Collimator
EffectsType = str  # "none" | "file" | "network" | "unknown"
DepthType = str    # "narrow" | "deep"
LaneType = str     # "dialogos" | "pbpair"


class SKPlanStep(TypedDict, total=False):
    """A single step in a Semantic Kernel plan."""
    step_id: str
    function_name: str
    plugin_name: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    effects: EffectsType


class SKPlanEvent(TypedDict, total=False):
    """Bus event for SK plan execution."""
    req_id: str
    trace_id: str
    plan_id: str
    goal: str
    steps: List[SKPlanStep]
    status: str
    depth: DepthType
    lane: LaneType
    effects: EffectsType


class SKToolSchema(TypedDict, total=False):
    """Tool schema compatible with both MCP and SK."""
    name: str
    description: str
    parameters: Dict[str, Any]
    effects: EffectsType
    plugin_name: str
    is_async: bool


class MCPToolSchema(TypedDict, total=False):
    """MCP-format tool schema."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class PluribusTool:
    """
    A tool definition that bridges SK functions and MCP tools.

    This provides a unified interface that can be:
    1. Registered with Semantic Kernel as a native function
    2. Exported as an MCP tool schema
    3. Called with Pluribus bus tracing
    """
    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    effects: EffectsType = "none"
    plugin_name: str = "pluribus"
    is_async: bool = False
    tags: List[str] = field(default_factory=list)

    def to_sk_schema(self) -> SKToolSchema:
        """Export as Semantic Kernel function schema."""
        return SKToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            effects=self.effects,
            plugin_name=self.plugin_name,
            is_async=self.is_async,
        )

    def to_mcp_schema(self) -> MCPToolSchema:
        """Export as MCP tool schema."""
        # Convert parameters to JSON Schema format
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param_info in self.parameters.items():
            param_type = param_info.get("type", "string")
            properties[param_name] = {
                "type": param_type,
                "description": param_info.get("description", ""),
            }
            if param_info.get("required", False):
                required.append(param_name)

        return MCPToolSchema(
            name=self.name,
            description=self.description,
            inputSchema={
                "type": "object",
                "properties": properties,
                "required": required,
            }
        )

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool function."""
        return self.func(**kwargs)


@dataclass
class PlanResult:
    """Result of executing a Semantic Kernel plan."""
    req_id: str
    trace_id: str
    plan_id: str
    goal: str
    steps_executed: int
    steps_total: int
    final_result: Any
    status: str  # "success" | "partial" | "error"
    execution_time_ms: float
    effects_observed: List[EffectsType]
    errors: List[str] = field(default_factory=list)


def _default_bus_dir() -> str:
    """Resolve the default bus directory."""
    env = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if env:
        return env
    repo_bus = Path("/pluribus/.pluribus/bus")
    if repo_bus.exists():
        return str(repo_bus)
    return str(Path.home() / ".local" / "state" / "nucleus" / "bus")


def now_iso_utc() -> str:
    """Return current UTC timestamp in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def classify_effects_from_goal(goal: str) -> EffectsType:
    """
    Classify expected effects from a goal string.

    Aligns with Lens/Collimator _normalize_effects.
    """
    goal_lower = goal.lower()

    # Network effects
    if any(w in goal_lower for w in ["fetch", "download", "api", "http", "request", "url", "web"]):
        return "network"

    # File effects
    if any(w in goal_lower for w in ["write", "save", "create file", "edit", "modify", "delete"]):
        return "file"

    # Pure/read-only operations
    if any(w in goal_lower for w in ["read", "analyze", "summarize", "explain", "list", "search"]):
        return "none"

    return "unknown"


def classify_depth_from_goal(goal: str) -> DepthType:
    """
    Classify depth (narrow/deep) from a goal string.

    Aligns with Lens/Collimator _classify_depth.
    """
    goal_lower = goal.lower()

    # Deep indicators
    deep_keywords = [
        "architecture", "design", "spec", "research", "theory",
        "protocol", "schema", "dsl", "neurosymbolic", "collimator",
        "lens", "comprehensive", "full", "complete", "detailed"
    ]

    if any(w in goal_lower for w in deep_keywords):
        return "deep"

    if len(goal) > 240:
        return "deep"

    return "narrow"


class SemanticKernelAdapter:
    """
    Adapter bridging Semantic Kernel to Pluribus bus and MCP tools.

    Key responsibilities:
    1. Wrap SK Kernel with bus event emission
    2. Register Pluribus tools as SK native functions
    3. Map SK planner output to Lens route plans
    4. Convert SK function schemas to MCP tool format
    5. Execute plans with full bus tracing
    """

    def __init__(
        self,
        bus_dir: Optional[str] = None,
        actor: Optional[str] = None,
        use_sk: bool = True
    ):
        self.bus_dir = bus_dir or _default_bus_dir()
        self.actor = actor or os.environ.get("PLURIBUS_ACTOR") or default_actor()
        self._bus_paths = resolve_bus_paths(self.bus_dir)

        # Current execution context
        self.current_req_id: str = ""
        self.current_trace_id: str = ""
        self.current_plan_id: str = ""

        # Tool registry
        self._tools: Dict[str, PluribusTool] = {}
        self._plugins: Dict[str, List[str]] = {}  # plugin_name -> [tool_names]

        # Kernel instance
        self._kernel: Any = None
        self._sk_available = False
        self._sk = None

        if use_sk:
            try:
                import semantic_kernel  # type: ignore
                self._sk = semantic_kernel
                self._sk_available = True
            except ImportError:
                pass

    @property
    def sk_available(self) -> bool:
        """Check if Semantic Kernel is available."""
        return self._sk_available

    def emit_bus_event(
        self,
        topic: str,
        kind: str,
        level: str,
        data: Dict[str, Any]
    ) -> str:
        """Emit an event to the Pluribus bus."""
        return emit_event(
            self._bus_paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=self.actor,
            data=data,
            trace_id=self.current_trace_id or None,
            run_id=self.current_req_id or None,
            durable=False
        )

    def create_kernel(
        self,
        service_id: str = "pluribus-sk",
        ai_service: Optional[Any] = None
    ) -> Any:
        """
        Create a Semantic Kernel instance with bus integration.

        If SK is not available, returns a mock kernel for testing.
        """
        if self._sk_available and self._sk:
            try:
                kernel = self._sk.Kernel()

                # Register AI service if provided
                if ai_service:
                    kernel.add_service(ai_service)

                self._kernel = kernel

                self.emit_bus_event(
                    topic="sk.kernel.created",
                    kind="metric",
                    level="info",
                    data={
                        "service_id": service_id,
                        "sk_version": getattr(self._sk, "__version__", "unknown"),
                        "plugins_loaded": 0,
                    }
                )

                return kernel
            except Exception as e:
                self.emit_bus_event(
                    topic="sk.kernel.error",
                    kind="log",
                    level="error",
                    data={"error": str(e), "fallback": "mock"}
                )

        # Mock kernel for testing
        self._kernel = MockKernel(self)
        return self._kernel

    def register_tool(
        self,
        tool: PluribusTool,
        register_with_sk: bool = True
    ) -> None:
        """
        Register a tool with the adapter.

        Args:
            tool: PluribusTool to register
            register_with_sk: Whether to also register with SK kernel
        """
        self._tools[tool.name] = tool

        # Track plugin membership
        if tool.plugin_name not in self._plugins:
            self._plugins[tool.plugin_name] = []
        if tool.name not in self._plugins[tool.plugin_name]:
            self._plugins[tool.plugin_name].append(tool.name)

        # Register with SK kernel if available
        if register_with_sk and self._kernel and self._sk_available:
            self._register_tool_with_sk(tool)

        self.emit_bus_event(
            topic="sk.tool.registered",
            kind="metric",
            level="info",
            data={
                "name": tool.name,
                "plugin": tool.plugin_name,
                "effects": tool.effects,
                "sk_registered": register_with_sk and self._sk_available,
            }
        )

    def _register_tool_with_sk(self, tool: PluribusTool) -> None:
        """Register a tool as an SK native function."""
        if not self._sk or not self._kernel:
            return

        try:
            # Create SK function decorator
            from semantic_kernel.functions import kernel_function  # type: ignore

            @kernel_function(name=tool.name, description=tool.description)
            def sk_wrapper(**kwargs: Any) -> Any:
                return self._execute_tool_with_tracing(tool, kwargs)

            # Add to kernel
            self._kernel.add_function(
                plugin_name=tool.plugin_name,
                function=sk_wrapper
            )
        except Exception as e:
            self.emit_bus_event(
                topic="sk.tool.register.error",
                kind="log",
                level="warn",
                data={"tool": tool.name, "error": str(e)}
            )

    def _execute_tool_with_tracing(
        self,
        tool: PluribusTool,
        kwargs: Dict[str, Any]
    ) -> Any:
        """Execute a tool with bus tracing."""
        step_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        self.emit_bus_event(
            topic="sk.tool.execute.start",
            kind="log",
            level="info",
            data={
                "step_id": step_id,
                "tool": tool.name,
                "plugin": tool.plugin_name,
                "effects": tool.effects,
                "parameters": {k: str(v)[:100] for k, v in kwargs.items()},
            }
        )

        try:
            result = tool(**kwargs)
            elapsed_ms = (time.time() - start_time) * 1000

            self.emit_bus_event(
                topic="sk.tool.execute.end",
                kind="metric",
                level="info",
                data={
                    "step_id": step_id,
                    "tool": tool.name,
                    "status": "success",
                    "elapsed_ms": elapsed_ms,
                    "result_type": type(result).__name__,
                }
            )

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            self.emit_bus_event(
                topic="sk.tool.execute.error",
                kind="log",
                level="error",
                data={
                    "step_id": step_id,
                    "tool": tool.name,
                    "error": str(e),
                    "elapsed_ms": elapsed_ms,
                }
            )
            raise

    def create_tool_from_function(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        plugin_name: str = "pluribus",
        effects: EffectsType = "none",
        tags: Optional[List[str]] = None
    ) -> PluribusTool:
        """
        Create a PluribusTool from a Python function.

        Automatically extracts parameters from function signature.
        """
        func_name = name or func.__name__
        func_desc = description or (func.__doc__ or "").strip() or f"Function {func_name}"

        # Extract parameters from signature
        sig = inspect.signature(func)
        parameters: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                ann = param.annotation
                if ann in (int, float):
                    param_type = "number"
                elif ann == bool:
                    param_type = "boolean"
                elif ann == list or (hasattr(ann, "__origin__") and ann.__origin__ is list):
                    param_type = "array"
                elif ann == dict or (hasattr(ann, "__origin__") and ann.__origin__ is dict):
                    param_type = "object"

            required = param.default == inspect.Parameter.empty

            parameters[param_name] = {
                "type": param_type,
                "required": required,
                "description": f"Parameter {param_name}",
            }

        tool = PluribusTool(
            name=func_name,
            description=func_desc,
            func=func,
            parameters=parameters,
            effects=effects,
            plugin_name=plugin_name,
            is_async=inspect.iscoroutinefunction(func),
            tags=tags or [],
        )

        return tool

    def export_mcp_tools(self) -> List[MCPToolSchema]:
        """Export all registered tools as MCP schemas."""
        return [tool.to_mcp_schema() for tool in self._tools.values()]

    def export_sk_tools(self) -> List[SKToolSchema]:
        """Export all registered tools as SK schemas."""
        return [tool.to_sk_schema() for tool in self._tools.values()]

    async def create_plan(
        self,
        goal: str,
        req_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> List[SKPlanStep]:
        """
        Create an execution plan for a goal.

        Uses SK planner if available, otherwise generates a mock plan.
        """
        self.current_req_id = req_id or str(uuid.uuid4())
        self.current_trace_id = trace_id or str(uuid.uuid4())
        self.current_plan_id = f"plan_{str(uuid.uuid4())[:8]}"

        effects = classify_effects_from_goal(goal)
        depth = classify_depth_from_goal(goal)

        self.emit_bus_event(
            topic="sk.plan.create.start",
            kind="metric",
            level="info",
            data={
                "req_id": self.current_req_id,
                "plan_id": self.current_plan_id,
                "goal": goal[:500],
                "effects": effects,
                "depth": depth,
            }
        )

        steps: List[SKPlanStep] = []

        if self._sk_available and self._kernel:
            try:
                steps = await self._create_sk_plan(goal)
            except Exception as e:
                self.emit_bus_event(
                    topic="sk.plan.create.error",
                    kind="log",
                    level="warn",
                    data={"error": str(e), "fallback": "mock"}
                )
                steps = self._create_mock_plan(goal)
        else:
            steps = self._create_mock_plan(goal)

        self.emit_bus_event(
            topic="sk.plan.create.end",
            kind="metric",
            level="info",
            data={
                "plan_id": self.current_plan_id,
                "steps_count": len(steps),
                "effects": effects,
                "depth": depth,
            }
        )

        return steps

    async def _create_sk_plan(self, goal: str) -> List[SKPlanStep]:
        """Create a plan using Semantic Kernel planner."""
        if not self._sk or not self._kernel:
            return []

        try:
            from semantic_kernel.planners import SequentialPlanner  # type: ignore

            planner = SequentialPlanner(self._kernel)
            plan = await planner.create_plan(goal)

            steps: List[SKPlanStep] = []
            for i, step in enumerate(plan.steps):
                steps.append(SKPlanStep(
                    step_id=f"step_{i}",
                    function_name=step.name,
                    plugin_name=step.plugin_name,
                    parameters=step.parameters,
                    dependencies=[],
                    effects=self._tools.get(step.name, PluribusTool(
                        name=step.name, description="", func=lambda: None
                    )).effects,
                ))

            return steps

        except Exception as e:
            self.emit_bus_event(
                topic="sk.planner.error",
                kind="log",
                level="warn",
                data={"error": str(e)}
            )
            return []

    def _create_mock_plan(self, goal: str) -> List[SKPlanStep]:
        """Create a mock plan for testing."""
        # Generate simple plan based on goal keywords
        steps: List[SKPlanStep] = []

        goal_lower = goal.lower()

        # Add steps based on goal content
        if "search" in goal_lower or "find" in goal_lower:
            steps.append(SKPlanStep(
                step_id="step_0",
                function_name="search",
                plugin_name="pluribus",
                parameters={"query": goal},
                dependencies=[],
                effects="network",
            ))

        if "analyze" in goal_lower or "summarize" in goal_lower:
            steps.append(SKPlanStep(
                step_id=f"step_{len(steps)}",
                function_name="analyze",
                plugin_name="pluribus",
                parameters={"input": "${previous_result}"},
                dependencies=[steps[-1]["step_id"]] if steps else [],
                effects="none",
            ))

        if "write" in goal_lower or "create" in goal_lower:
            steps.append(SKPlanStep(
                step_id=f"step_{len(steps)}",
                function_name="generate",
                plugin_name="pluribus",
                parameters={"context": "${previous_result}"},
                dependencies=[steps[-1]["step_id"]] if steps else [],
                effects="file" if "file" in goal_lower else "none",
            ))

        # Default step if no matches
        if not steps:
            steps.append(SKPlanStep(
                step_id="step_0",
                function_name="process",
                plugin_name="pluribus",
                parameters={"goal": goal},
                dependencies=[],
                effects="unknown",
            ))

        return steps

    async def execute_plan(
        self,
        goal: str,
        req_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> PlanResult:
        """
        Create and execute a plan for a goal.

        Returns a PlanResult with full execution details.
        """
        start_time = time.time()

        steps = await self.create_plan(goal, req_id, trace_id)

        self.emit_bus_event(
            topic="sk.plan.execute.start",
            kind="metric",
            level="info",
            data={
                "plan_id": self.current_plan_id,
                "goal": goal[:500],
                "steps_count": len(steps),
            }
        )

        executed_steps = 0
        errors: List[str] = []
        effects_observed: List[EffectsType] = []
        final_result: Any = None

        for step in steps:
            step_start = time.time()

            self.emit_bus_event(
                topic="sk.step.start",
                kind="log",
                level="info",
                data={
                    "plan_id": self.current_plan_id,
                    "step_id": step["step_id"],
                    "function": step["function_name"],
                    "plugin": step["plugin_name"],
                }
            )

            try:
                # Try to find and execute the tool
                tool = self._tools.get(step["function_name"])
                if tool:
                    final_result = tool(**step.get("parameters", {}))
                    effects_observed.append(tool.effects)
                else:
                    # Mock execution
                    final_result = f"[Mock result for {step['function_name']}]"
                    effects_observed.append(step.get("effects", "unknown"))

                executed_steps += 1

                self.emit_bus_event(
                    topic="sk.step.end",
                    kind="metric",
                    level="info",
                    data={
                        "plan_id": self.current_plan_id,
                        "step_id": step["step_id"],
                        "status": "success",
                        "elapsed_ms": (time.time() - step_start) * 1000,
                    }
                )

            except Exception as e:
                errors.append(f"{step['function_name']}: {e}")

                self.emit_bus_event(
                    topic="sk.step.error",
                    kind="log",
                    level="error",
                    data={
                        "plan_id": self.current_plan_id,
                        "step_id": step["step_id"],
                        "error": str(e),
                    }
                )

        # Determine status
        if executed_steps == len(steps) and not errors:
            status = "success"
        elif executed_steps > 0:
            status = "partial"
        else:
            status = "error"

        execution_time_ms = (time.time() - start_time) * 1000

        self.emit_bus_event(
            topic="sk.plan.execute.end",
            kind="metric",
            level="info",
            data={
                "plan_id": self.current_plan_id,
                "status": status,
                "steps_executed": executed_steps,
                "steps_total": len(steps),
                "execution_time_ms": execution_time_ms,
                "effects_observed": list(set(effects_observed)),
                "errors_count": len(errors),
            }
        )

        return PlanResult(
            req_id=self.current_req_id,
            trace_id=self.current_trace_id,
            plan_id=self.current_plan_id,
            goal=goal,
            steps_executed=executed_steps,
            steps_total=len(steps),
            final_result=final_result,
            status=status,
            execution_time_ms=execution_time_ms,
            effects_observed=list(set(effects_observed)),
            errors=errors,
        )

    def get_tool(self, name: str) -> Optional[PluribusTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_plugins(self) -> Dict[str, List[str]]:
        """List all plugins and their tools."""
        return dict(self._plugins)


class MockKernel:
    """Mock Semantic Kernel for testing without SK installed."""

    def __init__(self, adapter: SemanticKernelAdapter):
        self.adapter = adapter
        self._functions: Dict[str, Callable] = {}
        self._services: List[Any] = []

    def add_function(self, plugin_name: str, function: Callable) -> None:
        """Add a function to the mock kernel."""
        func_name = getattr(function, "__name__", "unknown")
        self._functions[f"{plugin_name}.{func_name}"] = function

    def add_service(self, service: Any) -> None:
        """Add a service to the mock kernel."""
        self._services.append(service)

    def invoke(self, function_name: str, **kwargs: Any) -> Any:
        """Invoke a function on the mock kernel."""
        func = self._functions.get(function_name)
        if func:
            return func(**kwargs)
        return f"[Mock result for {function_name}]"


# Service Registry Entry (for BUILTIN_SERVICES)
SK_SERVICE_DEF = {
    "id": "semantic-kernel-adapter",
    "name": "Semantic Kernel Adapter",
    "kind": "process",
    "entry_point": "nucleus/tools/semantic_kernel_adapter.py",
    "description": "Bridges Microsoft Semantic Kernel planners and tools to Pluribus",
    "tags": ["semantic-kernel", "microsoft", "planner", "sota", "orchestration"],
    "lineage": "sota.microsoft",
    "omega_motif": False,
    "gates": {"E": "plan_execution", "P": "step_provenance"},
}


async def cmd_demo(args: argparse.Namespace) -> int:
    """Run a demo plan execution."""
    adapter = SemanticKernelAdapter(bus_dir=args.bus_dir)

    print(f"Semantic Kernel available: {adapter.sk_available}")
    print("Creating kernel...")

    kernel = adapter.create_kernel()

    # Register some example tools
    def search_papers(query: str) -> Dict[str, Any]:
        """Search for research papers."""
        return {"results": [f"Paper about {query}"], "count": 1}

    def summarize(text: str) -> str:
        """Summarize text."""
        return f"Summary of: {text[:50]}..."

    def write_report(content: str, filename: str = "report.md") -> Dict[str, Any]:
        """Write a report file."""
        return {"filename": filename, "size": len(content)}

    # Create and register tools
    search_tool = adapter.create_tool_from_function(
        search_papers, effects="network", tags=["search", "research"]
    )
    summarize_tool = adapter.create_tool_from_function(
        summarize, effects="none", tags=["nlp", "summarize"]
    )
    write_tool = adapter.create_tool_from_function(
        write_report, effects="file", tags=["io", "write"]
    )

    adapter.register_tool(search_tool)
    adapter.register_tool(summarize_tool)
    adapter.register_tool(write_tool)

    print(f"Tools registered: {adapter.list_tools()}")
    print(f"Plugins: {adapter.list_plugins()}")

    # Execute plan
    goal = args.goal or "Search for papers about transformers, summarize them, and write a report"
    print(f"\nExecuting plan for goal: {goal}")

    result = await adapter.execute_plan(goal=goal)

    print(f"\nPlan execution completed:")
    print(f"  Status: {result.status}")
    print(f"  Steps executed: {result.steps_executed}/{result.steps_total}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")
    print(f"  Effects observed: {result.effects_observed}")
    print(f"  Plan ID: {result.plan_id}")
    print(f"  Trace ID: {result.trace_id}")

    if result.errors:
        print(f"  Errors: {result.errors}")

    # Export MCP schemas
    print("\nMCP Tool Schemas:")
    for schema in adapter.export_mcp_tools():
        print(f"  - {schema['name']}: {schema.get('description', '')[:50]}...")

    return 0


def cmd_export_mcp(args: argparse.Namespace) -> int:
    """Export registered tools as MCP schemas."""
    adapter = SemanticKernelAdapter(bus_dir=args.bus_dir, use_sk=False)

    # Register example tools for export demo
    def example_tool(input: str) -> str:
        """An example tool."""
        return input

    tool = adapter.create_tool_from_function(example_tool)
    adapter.register_tool(tool)

    schemas = adapter.export_mcp_tools()
    print(json.dumps(schemas, indent=2, ensure_ascii=False))

    return 0


def cmd_list_tools(args: argparse.Namespace) -> int:
    """List tool conversion capabilities."""
    print("Semantic Kernel <-> MCP Tool Mapping:")
    print("  SK native_function -> PluribusTool -> MCP tool")
    print("  SK plugin -> MCP server namespace")
    print("  SK parameters -> JSON Schema inputSchema")
    print("\nEffects typing alignment:")
    print("  SK function -> effects: none | file | network | unknown")
    print("  Lens/Collimator depth: narrow | deep")
    print("  Lens/Collimator lane: dialogos | pbpair")
    return 0


def main(argv: List[str]) -> int:
    import asyncio

    parser = argparse.ArgumentParser(
        prog="semantic_kernel_adapter.py",
        description="Semantic Kernel Adapter for Pluribus - Planner and tool orchestration"
    )
    parser.add_argument("--bus-dir", default=None, help="Bus directory")

    sub = parser.add_subparsers(dest="cmd", required=True)

    demo_p = sub.add_parser("demo", help="Run a demo plan execution")
    demo_p.add_argument("--goal", default=None, help="Goal to plan for")
    demo_p.set_defaults(func=lambda args: asyncio.run(cmd_demo(args)))

    export_p = sub.add_parser("export-mcp", help="Export tools as MCP schemas")
    export_p.set_defaults(func=cmd_export_mcp)

    list_p = sub.add_parser("list-tools", help="List tool conversion info")
    list_p.set_defaults(func=cmd_list_tools)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
