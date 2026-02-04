#!/usr/bin/env python3
"""
Tests for AutoGen Bridge
========================

Tests the AutoGen <-> Pluribus bus integration.
Works in mock mode when AutoGen is not installed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add tools directory to path
TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from autogen_bridge import (
    AutoGenBridge,
    PluribusAgent,
    AgentConfig,
    ConversationResult,
    AUTOGEN_SERVICE_DEF,
)


@pytest.fixture
def temp_bus_dir():
    """Create a temporary bus directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bus_dir = Path(tmpdir) / "bus"
        bus_dir.mkdir(parents=True, exist_ok=True)
        yield str(bus_dir)


@pytest.fixture
def bridge(temp_bus_dir: str) -> AutoGenBridge:
    """Create an AutoGenBridge for testing."""
    return AutoGenBridge(bus_dir=temp_bus_dir, use_autogen=False)


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


class TestAutogenBridge:
    """Test suite for AutoGenBridge."""

    def test_create_bridge(self, temp_bus_dir: str):
        """Test creating a bridge instance."""
        bridge = AutoGenBridge(bus_dir=temp_bus_dir)
        assert bridge.bus_dir == temp_bus_dir
        assert bridge.actor is not None
        assert isinstance(bridge.list_agents(), list)

    def test_create_agent(self, bridge: AutoGenBridge):
        """Test creating an agent."""
        agent = bridge.create_agent(
            name="test_agent",
            system_message="You are a test agent.",
            model="auto",
            effects="none"
        )

        assert agent.name == "test_agent"
        assert agent.config.system_message == "You are a test agent."
        assert agent.config.effects == "none"
        assert "test_agent" in bridge.list_agents()

    def test_create_assistant(self, bridge: AutoGenBridge):
        """Test creating an assistant agent."""
        assistant = bridge.create_assistant(
            name="assistant",
            system_message="You are helpful."
        )

        assert assistant.name == "assistant"
        assert assistant.config.human_input_mode == "NEVER"

    def test_create_user_proxy(self, bridge: AutoGenBridge):
        """Test creating a user proxy agent."""
        proxy = bridge.create_user_proxy(
            name="proxy",
            code_execution_config={"use_docker": False}
        )

        assert proxy.name == "proxy"
        assert proxy.config.effects == "file"

    def test_agent_created_event(self, bridge: AutoGenBridge, temp_bus_dir: str):
        """Test that agent creation emits bus event."""
        bridge.create_agent(name="event_test_agent", system_message="test")

        events = read_bus_events(temp_bus_dir)
        agent_events = [e for e in events if e.get("topic") == "autogen.agent.created"]

        assert len(agent_events) >= 1
        assert agent_events[-1]["data"]["name"] == "event_test_agent"

    def test_run_conversation_mock(self, bridge: AutoGenBridge, temp_bus_dir: str):
        """Test running a mock conversation."""
        agent1 = bridge.create_agent(name="agent1", system_message="First agent")
        agent2 = bridge.create_agent(name="agent2", system_message="Second agent")

        result = bridge.run_conversation(
            agents=[agent1, agent2],
            initial_message="Hello, let's discuss.",
            max_rounds=3
        )

        assert isinstance(result, ConversationResult)
        assert result.status in ("success", "error")
        assert result.req_id != ""
        assert result.trace_id != ""
        assert result.group_id.startswith("group_")
        assert len(result.messages) > 0

    def test_conversation_events(self, bridge: AutoGenBridge, temp_bus_dir: str):
        """Test that conversation emits proper bus events."""
        agent1 = bridge.create_agent(name="conv_agent1", system_message="Agent 1")
        agent2 = bridge.create_agent(name="conv_agent2", system_message="Agent 2")

        result = bridge.run_conversation(
            agents=[agent1, agent2],
            initial_message="Test message",
            max_rounds=2
        )

        events = read_bus_events(temp_bus_dir)
        topics = [e.get("topic") for e in events]

        # Should have start and end events
        assert "autogen.conversation.start" in topics
        assert "autogen.conversation.end" in topics

        # Check conversation end event
        end_events = [e for e in events if e.get("topic") == "autogen.conversation.end"]
        assert len(end_events) >= 1
        assert end_events[-1]["data"]["group_id"] == result.group_id

    def test_conversation_with_trace_ids(self, bridge: AutoGenBridge):
        """Test conversation with custom trace IDs."""
        agent = bridge.create_agent(name="trace_agent", system_message="test")

        req_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())

        result = bridge.run_conversation(
            agents=[agent],
            initial_message="Test",
            max_rounds=1,
            req_id=req_id,
            trace_id=trace_id
        )

        assert result.req_id == req_id
        assert result.trace_id == trace_id

    def test_get_agent(self, bridge: AutoGenBridge):
        """Test getting an agent by name."""
        bridge.create_agent(name="findable_agent", system_message="test")

        found = bridge.get_agent("findable_agent")
        assert found is not None
        assert found.name == "findable_agent"

        not_found = bridge.get_agent("nonexistent")
        assert not_found is None

    def test_reset_all(self, bridge: AutoGenBridge):
        """Test resetting all agents."""
        agent1 = bridge.create_agent(name="reset_agent1", system_message="test")
        agent2 = bridge.create_agent(name="reset_agent2", system_message="test")

        # Run conversation to populate history
        bridge.run_conversation(
            agents=[agent1, agent2],
            initial_message="Test",
            max_rounds=2
        )

        # Reset
        bridge.reset_all()

        # Agents should still exist but be reset
        assert "reset_agent1" in bridge.list_agents()
        assert "reset_agent2" in bridge.list_agents()


class TestPluribusAgent:
    """Test suite for PluribusAgent."""

    def test_agent_config(self):
        """Test AgentConfig dataclass."""
        config = AgentConfig(
            name="test",
            system_message="You are a test.",
            model="gpt-4",
            effects="file"
        )

        assert config.name == "test"
        assert config.model == "gpt-4"
        assert config.effects == "file"
        assert config.human_input_mode == "NEVER"

    def test_agent_name_property(self, bridge: AutoGenBridge):
        """Test agent name property."""
        agent = bridge.create_agent(name="name_test", system_message="test")
        assert agent.name == "name_test"

    def test_agent_reset(self, bridge: AutoGenBridge):
        """Test agent reset."""
        agent = bridge.create_agent(name="reset_test", system_message="test")
        agent._message_history.append({"content": "test"})

        agent.reset()
        assert len(agent._message_history) == 0


class TestConversationResult:
    """Test suite for ConversationResult."""

    def test_conversation_result_fields(self):
        """Test ConversationResult dataclass."""
        result = ConversationResult(
            req_id="req123",
            trace_id="trace456",
            group_id="group789",
            messages=[{"content": "test"}],
            final_message="Final message",
            total_rounds=3,
            status="success"
        )

        assert result.req_id == "req123"
        assert result.trace_id == "trace456"
        assert result.group_id == "group789"
        assert len(result.messages) == 1
        assert result.total_rounds == 3
        assert result.status == "success"


class TestServiceDef:
    """Test service registry definition."""

    def test_service_def_structure(self):
        """Test AUTOGEN_SERVICE_DEF has required fields."""
        assert "id" in AUTOGEN_SERVICE_DEF
        assert "name" in AUTOGEN_SERVICE_DEF
        assert "kind" in AUTOGEN_SERVICE_DEF
        assert "entry_point" in AUTOGEN_SERVICE_DEF
        assert "tags" in AUTOGEN_SERVICE_DEF

        assert AUTOGEN_SERVICE_DEF["id"] == "autogen-bridge"
        assert AUTOGEN_SERVICE_DEF["kind"] == "process"
        assert "autogen" in AUTOGEN_SERVICE_DEF["tags"]
        assert "microsoft" in AUTOGEN_SERVICE_DEF["tags"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
