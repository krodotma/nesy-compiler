#!/usr/bin/env python3
"""
Tests for Semantic Kernel Adapter
=================================

Tests the Semantic Kernel <-> Pluribus integration.
Works in mock mode when Semantic Kernel is not installed.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add tools directory to path
TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from semantic_kernel_adapter import (
    SemanticKernelAdapter,
    PluribusTool,
    PlanResult,
    MockKernel,
    classify_effects_from_goal,
    classify_depth_from_goal,
    SK_SERVICE_DEF,
)


@pytest.fixture
def temp_bus_dir():
    """Create a temporary bus directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bus_dir = Path(tmpdir) / "bus"
        bus_dir.mkdir(parents=True, exist_ok=True)
        yield str(bus_dir)


@pytest.fixture
def adapter(temp_bus_dir: str) -> SemanticKernelAdapter:
    """Create a SemanticKernelAdapter for testing."""
    return SemanticKernelAdapter(bus_dir=temp_bus_dir, use_sk=False)


def read_bus_events(bus_dir: str) -> List[Dict[str, Any]]:
    """Read all events from the bus."""
    events_path = Path(bus_dir) / "events.ndjson"
    if not events_path.exists():
        return []
    events = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


class TestEffectsClassification:
    """Test effects classification from goals."""

    def test_network_effects(self):
        """Test network effect classification."""
        assert classify_effects_from_goal("fetch data from API") == "network"
        assert classify_effects_from_goal("download the file") == "network"
        assert classify_effects_from_goal("make HTTP request") == "network"
        assert classify_effects_from_goal("get URL content") == "network"

    def test_file_effects(self):
        """Test file effect classification."""
        assert classify_effects_from_goal("write to file") == "file"
        assert classify_effects_from_goal("save the document") == "file"
        assert classify_effects_from_goal("create file output.txt") == "file"
        assert classify_effects_from_goal("edit the configuration") == "file"

    def test_no_effects(self):
        """Test no effect classification."""
        assert classify_effects_from_goal("read the document") == "none"
        assert classify_effects_from_goal("analyze this data") == "none"
        assert classify_effects_from_goal("summarize the text") == "none"
        assert classify_effects_from_goal("explain quantum computing") == "none"

    def test_unknown_effects(self):
        """Test unknown effect classification."""
        assert classify_effects_from_goal("do something mysterious") == "unknown"
        assert classify_effects_from_goal("process the thing") == "unknown"


class TestDepthClassification:
    """Test depth classification from goals."""

    def test_deep_goals(self):
        """Test deep goal classification."""
        assert classify_depth_from_goal("design the architecture") == "deep"
        assert classify_depth_from_goal("create a detailed spec") == "deep"
        assert classify_depth_from_goal("research the topic thoroughly") == "deep"
        assert classify_depth_from_goal("define the protocol schema") == "deep"

    def test_narrow_goals(self):
        """Test narrow goal classification."""
        assert classify_depth_from_goal("add a button") == "narrow"
        assert classify_depth_from_goal("fix the bug") == "narrow"
        assert classify_depth_from_goal("update the text") == "narrow"

    def test_long_goals_are_deep(self):
        """Test that long goals are classified as deep."""
        long_goal = "This is a very long goal that contains many words and should be classified as deep because it exceeds the character threshold and likely represents a complex task that requires careful consideration. " * 2
        assert len(long_goal) > 240
        assert classify_depth_from_goal(long_goal) == "deep"


class TestSemanticKernelAdapter:
    """Test suite for SemanticKernelAdapter."""

    def test_create_adapter(self, temp_bus_dir: str):
        """Test creating an adapter instance."""
        adapter = SemanticKernelAdapter(bus_dir=temp_bus_dir)
        assert adapter.bus_dir == temp_bus_dir
        assert adapter.actor is not None

    def test_create_kernel(self, adapter: SemanticKernelAdapter, temp_bus_dir: str):
        """Test creating a kernel."""
        kernel = adapter.create_kernel()
        assert kernel is not None
        # In mock mode, we get a MockKernel instance
        assert isinstance(kernel, MockKernel)

    def test_create_tool_from_function(self, adapter: SemanticKernelAdapter):
        """Test creating a tool from a function."""
        def sample_function(text: str, count: int = 1) -> str:
            """A sample function."""
            return text * count

        tool = adapter.create_tool_from_function(
            sample_function,
            effects="none",
            tags=["test"]
        )

        assert tool.name == "sample_function"
        assert tool.description == "A sample function."
        assert "text" in tool.parameters
        assert "count" in tool.parameters
        assert tool.parameters["text"]["required"] is True
        assert tool.parameters["count"]["required"] is False

    def test_register_tool(self, adapter: SemanticKernelAdapter, temp_bus_dir: str):
        """Test registering a tool."""
        def test_func(x: str) -> str:
            return x

        tool = adapter.create_tool_from_function(test_func)
        adapter.register_tool(tool)

        assert "test_func" in adapter.list_tools()

        # Should emit registration event
        events = read_bus_events(temp_bus_dir)
        reg_events = [e for e in events if e.get("topic") == "sk.tool.registered"]
        assert len(reg_events) >= 1
        assert reg_events[-1]["data"]["name"] == "test_func"

    def test_export_mcp_tools(self, adapter: SemanticKernelAdapter):
        """Test exporting tools as MCP schemas."""
        def mcp_test(input: str) -> str:
            """Test tool for MCP export."""
            return input

        tool = adapter.create_tool_from_function(mcp_test)
        adapter.register_tool(tool)

        schemas = adapter.export_mcp_tools()
        assert len(schemas) >= 1

        schema = schemas[0]
        assert "name" in schema
        assert "description" in schema
        assert "inputSchema" in schema
        assert schema["inputSchema"]["type"] == "object"

    def test_export_sk_tools(self, adapter: SemanticKernelAdapter):
        """Test exporting tools as SK schemas."""
        def sk_test(input: str) -> str:
            """Test tool for SK export."""
            return input

        tool = adapter.create_tool_from_function(sk_test)
        adapter.register_tool(tool)

        schemas = adapter.export_sk_tools()
        assert len(schemas) >= 1

        schema = schemas[0]
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert "effects" in schema

    def test_create_plan(self, adapter: SemanticKernelAdapter, temp_bus_dir: str):
        """Test creating a plan."""
        adapter.create_kernel()

        steps = asyncio.run(adapter.create_plan(
            goal="Search for papers and summarize them"
        ))

        assert isinstance(steps, list)
        assert len(steps) > 0

        # Check step structure
        step = steps[0]
        assert "step_id" in step
        assert "function_name" in step
        assert "plugin_name" in step

        # Should emit plan creation events
        events = read_bus_events(temp_bus_dir)
        start_events = [e for e in events if e.get("topic") == "sk.plan.create.start"]
        end_events = [e for e in events if e.get("topic") == "sk.plan.create.end"]
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    def test_execute_plan(self, adapter: SemanticKernelAdapter, temp_bus_dir: str):
        """Test executing a plan."""
        adapter.create_kernel()

        # Register a test tool
        def process(goal: str) -> str:
            return f"Processed: {goal}"

        tool = adapter.create_tool_from_function(process)
        adapter.register_tool(tool)

        result = asyncio.run(adapter.execute_plan(
            goal="Process some data"
        ))

        assert isinstance(result, PlanResult)
        assert result.status in ("success", "partial", "error")
        assert result.req_id != ""
        assert result.trace_id != ""
        assert result.plan_id.startswith("plan_")

        # Should emit execution events
        events = read_bus_events(temp_bus_dir)
        exec_start = [e for e in events if e.get("topic") == "sk.plan.execute.start"]
        exec_end = [e for e in events if e.get("topic") == "sk.plan.execute.end"]
        assert len(exec_start) >= 1
        assert len(exec_end) >= 1

    def test_get_tool(self, adapter: SemanticKernelAdapter):
        """Test getting a tool by name."""
        def findable(x: str) -> str:
            return x

        tool = adapter.create_tool_from_function(findable)
        adapter.register_tool(tool)

        found = adapter.get_tool("findable")
        assert found is not None
        assert found.name == "findable"

        not_found = adapter.get_tool("nonexistent")
        assert not_found is None

    def test_list_plugins(self, adapter: SemanticKernelAdapter):
        """Test listing plugins."""
        def plugin_test(x: str) -> str:
            return x

        tool = adapter.create_tool_from_function(
            plugin_test, plugin_name="test_plugin"
        )
        adapter.register_tool(tool)

        plugins = adapter.list_plugins()
        assert "test_plugin" in plugins
        assert "plugin_test" in plugins["test_plugin"]


class TestPluribusTool:
    """Test suite for PluribusTool."""

    def test_tool_creation(self):
        """Test creating a PluribusTool."""
        def sample(x: str) -> str:
            return x

        tool = PluribusTool(
            name="sample",
            description="A sample tool",
            func=sample,
            parameters={"x": {"type": "string", "required": True}},
            effects="none",
            plugin_name="test"
        )

        assert tool.name == "sample"
        assert tool.description == "A sample tool"
        assert tool.effects == "none"
        assert tool.plugin_name == "test"

    def test_tool_call(self):
        """Test calling a tool."""
        def add(a: int, b: int) -> int:
            return a + b

        tool = PluribusTool(
            name="add",
            description="Add two numbers",
            func=add,
            parameters={
                "a": {"type": "number", "required": True},
                "b": {"type": "number", "required": True}
            }
        )

        result = tool(a=2, b=3)
        assert result == 5

    def test_to_sk_schema(self):
        """Test SK schema export."""
        tool = PluribusTool(
            name="test",
            description="Test tool",
            func=lambda: None,
            effects="file",
            plugin_name="my_plugin"
        )

        schema = tool.to_sk_schema()
        assert schema["name"] == "test"
        assert schema["description"] == "Test tool"
        assert schema["effects"] == "file"
        assert schema["plugin_name"] == "my_plugin"

    def test_to_mcp_schema(self):
        """Test MCP schema export."""
        tool = PluribusTool(
            name="mcp_tool",
            description="MCP tool",
            func=lambda: None,
            parameters={
                "query": {"type": "string", "required": True, "description": "Search query"},
                "limit": {"type": "number", "required": False, "description": "Result limit"}
            }
        )

        schema = tool.to_mcp_schema()
        assert schema["name"] == "mcp_tool"
        assert schema["description"] == "MCP tool"
        assert schema["inputSchema"]["type"] == "object"
        assert "query" in schema["inputSchema"]["properties"]
        assert "limit" in schema["inputSchema"]["properties"]
        assert "query" in schema["inputSchema"]["required"]
        assert "limit" not in schema["inputSchema"]["required"]


class TestMockKernel:
    """Test suite for MockKernel."""

    def test_mock_kernel_creation(self, adapter: SemanticKernelAdapter):
        """Test creating a mock kernel."""
        kernel = MockKernel(adapter)
        assert kernel.adapter == adapter
        assert isinstance(kernel._functions, dict)

    def test_add_function(self, adapter: SemanticKernelAdapter):
        """Test adding a function to mock kernel."""
        kernel = MockKernel(adapter)

        def test_func():
            return "test"

        kernel.add_function("plugin", test_func)
        assert "plugin.test_func" in kernel._functions

    def test_invoke(self, adapter: SemanticKernelAdapter):
        """Test invoking a function on mock kernel."""
        kernel = MockKernel(adapter)

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        kernel.add_function("greeting", greet)
        result = kernel.invoke("greeting.greet", name="World")
        assert result == "Hello, World!"

    def test_invoke_missing(self, adapter: SemanticKernelAdapter):
        """Test invoking missing function."""
        kernel = MockKernel(adapter)
        result = kernel.invoke("missing.function")
        assert "Mock result" in result


class TestPlanResult:
    """Test suite for PlanResult."""

    def test_plan_result_fields(self):
        """Test PlanResult dataclass."""
        result = PlanResult(
            req_id="req123",
            trace_id="trace456",
            plan_id="plan789",
            goal="Test goal",
            steps_executed=3,
            steps_total=5,
            final_result={"status": "ok"},
            status="partial",
            execution_time_ms=150.5,
            effects_observed=["none", "file"]
        )

        assert result.req_id == "req123"
        assert result.trace_id == "trace456"
        assert result.plan_id == "plan789"
        assert result.steps_executed == 3
        assert result.steps_total == 5
        assert result.status == "partial"
        assert result.execution_time_ms == 150.5
        assert "file" in result.effects_observed


class TestServiceDef:
    """Test service registry definition."""

    def test_service_def_structure(self):
        """Test SK_SERVICE_DEF has required fields."""
        assert "id" in SK_SERVICE_DEF
        assert "name" in SK_SERVICE_DEF
        assert "kind" in SK_SERVICE_DEF
        assert "entry_point" in SK_SERVICE_DEF
        assert "tags" in SK_SERVICE_DEF

        assert SK_SERVICE_DEF["id"] == "semantic-kernel-adapter"
        assert SK_SERVICE_DEF["kind"] == "process"
        assert "semantic-kernel" in SK_SERVICE_DEF["tags"]
        assert "microsoft" in SK_SERVICE_DEF["tags"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
