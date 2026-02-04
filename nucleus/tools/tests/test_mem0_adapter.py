#!/usr/bin/env python3
"""Tests for mem0_adapter - Agent Long-Term Memory."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mem0_adapter import (
    Mem0Service,
    Memory,
    simple_hash_embed,
    cosine_similarity,
    extract_memories_from_bus_event,
)


@pytest.fixture
def temp_root():
    """Create a temporary root directory."""
    with tempfile.TemporaryDirectory(prefix="mem0_test_") as tmpdir:
        root = Path(tmpdir)
        (root / ".pluribus" / "memory").mkdir(parents=True)
        yield root


class TestSimpleHashEmbed:
    """Tests for the hash-based embedding function."""

    def test_deterministic(self):
        """Same text produces same embedding."""
        text = "Hello world"
        emb1 = simple_hash_embed(text)
        emb2 = simple_hash_embed(text)
        assert emb1 == emb2

    def test_dimension(self):
        """Embedding has correct dimension."""
        text = "Test text"
        emb = simple_hash_embed(text, dim=64)
        assert len(emb) == 64

        emb2 = simple_hash_embed(text, dim=128)
        assert len(emb2) == 128

    def test_different_text_different_embedding(self):
        """Different text produces different embeddings."""
        emb1 = simple_hash_embed("Hello")
        emb2 = simple_hash_embed("World")
        assert emb1 != emb2

    def test_values_normalized(self):
        """Embedding values are in [0, 1] range."""
        emb = simple_hash_embed("Test normalization")
        assert all(0 <= v <= 1 for v in emb)


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        vec = [0.5, 0.5, 0.5]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0."""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        assert abs(cosine_similarity(vec1, vec2)) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        vec1 = [1, 0, 0]
        vec2 = [-1, 0, 0]
        assert abs(cosine_similarity(vec1, vec2) + 1.0) < 1e-6

    def test_empty_vectors(self):
        """Empty vectors return 0."""
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_length(self):
        """Mismatched lengths return 0."""
        assert cosine_similarity([1, 2], [1, 2, 3]) == 0.0


class TestMemory:
    """Tests for Memory dataclass."""

    def test_to_dict_round_trip(self):
        """Memory can round-trip through dict."""
        mem = Memory(
            id="test-id",
            agent_id="agent-1",
            content="Test content",
            metadata={"key": "value"},
            embedding=[0.1, 0.2, 0.3],
            created_at="2024-01-01T00:00:00Z",
            accessed_at="2024-01-01T00:00:00Z",
            access_count=5,
            session_id="session-1",
            tags=["tag1", "tag2"],
        )

        d = mem.to_dict()
        assert d["kind"] == "mem0_memory"
        assert d["id"] == "test-id"
        assert d["agent_id"] == "agent-1"
        assert d["content"] == "Test content"

        mem2 = Memory.from_dict(d)
        assert mem2.id == mem.id
        assert mem2.agent_id == mem.agent_id
        assert mem2.content == mem.content
        assert mem2.tags == mem.tags


class TestMem0Service:
    """Tests for Mem0Service."""

    def test_store_and_retrieve(self, temp_root):
        """Can store and retrieve a memory."""
        service = Mem0Service(temp_root)

        # Store a memory
        result = service.store(
            agent_id="test-agent",
            content="I learned that Python is great for data science.",
            tags=["python", "learning"],
        )

        assert "error" not in result
        assert "id" in result
        assert result["status"] == "stored"

        # Retrieve it
        retrieve_result = service.retrieve(
            query="Python data science",
            agent_id="test-agent",
        )

        assert len(retrieve_result["memories"]) > 0
        mem = retrieve_result["memories"][0]
        assert "python" in mem["content"].lower()

    def test_store_requires_agent_id(self, temp_root):
        """Store fails without agent_id."""
        service = Mem0Service(temp_root)
        result = service.store(agent_id="", content="Test")
        assert "error" in result
        assert "agent_id" in result["error"]

    def test_store_requires_content(self, temp_root):
        """Store fails without content."""
        service = Mem0Service(temp_root)
        result = service.store(agent_id="test", content="")
        assert "error" in result
        assert "content" in result["error"]

    def test_retrieve_with_threshold(self, temp_root):
        """Retrieve respects similarity threshold."""
        service = Mem0Service(temp_root)

        # Store some memories
        service.store(agent_id="agent", content="The quick brown fox jumps")
        service.store(agent_id="agent", content="Python programming language")

        # Query with high threshold
        result = service.retrieve(
            query="completely unrelated topic xyz123",
            threshold=0.99,  # Very high threshold
        )

        assert result["count"] == 0

    def test_retrieve_with_limit(self, temp_root):
        """Retrieve respects limit."""
        service = Mem0Service(temp_root)

        # Store multiple memories
        for i in range(5):
            service.store(agent_id="agent", content=f"Memory content {i}")

        result = service.retrieve(query="Memory content", limit=2)
        assert len(result["memories"]) <= 2

    def test_update_memory(self, temp_root):
        """Can update an existing memory."""
        service = Mem0Service(temp_root)

        # Store
        store_result = service.store(
            agent_id="agent",
            content="Original content",
            tags=["original"],
        )
        memory_id = store_result["id"]

        # Update
        update_result = service.update(
            memory_id=memory_id,
            content="Updated content",
            tags=["updated"],
        )

        assert "error" not in update_result
        assert update_result["status"] == "updated"

    def test_update_nonexistent_memory(self, temp_root):
        """Update fails for nonexistent memory."""
        service = Mem0Service(temp_root)
        result = service.update(memory_id="nonexistent-id")
        assert "error" in result
        assert "not found" in result["error"]

    def test_delete_memory(self, temp_root):
        """Can delete a memory."""
        service = Mem0Service(temp_root)

        # Store
        store_result = service.store(agent_id="agent", content="To be deleted")
        memory_id = store_result["id"]

        # Delete
        delete_result = service.delete(memory_id=memory_id)
        assert "error" not in delete_result
        assert delete_result["status"] == "deleted"

        # List should not include deleted memory
        list_result = service.list_memories(agent_id="agent")
        ids = [m["id"] for m in list_result["memories"]]
        assert memory_id not in ids

    def test_list_memories(self, temp_root):
        """Can list memories with filtering."""
        service = Mem0Service(temp_root)

        # Store memories for different agents
        service.store(agent_id="agent-1", content="Agent 1 memory")
        service.store(agent_id="agent-2", content="Agent 2 memory")
        service.store(agent_id="agent-1", content="Another agent 1 memory")

        # List for agent-1
        result = service.list_memories(agent_id="agent-1")
        assert result["total"] == 2
        assert all(m["id"] for m in result["memories"])

    def test_list_with_pagination(self, temp_root):
        """List supports pagination."""
        service = Mem0Service(temp_root)

        # Store several memories
        for i in range(10):
            service.store(agent_id="agent", content=f"Memory {i}")

        # Get first page
        page1 = service.list_memories(limit=3, offset=0)
        assert len(page1["memories"]) == 3

        # Get second page
        page2 = service.list_memories(limit=3, offset=3)
        assert len(page2["memories"]) == 3

        # Pages should have different memories
        ids1 = {m["id"] for m in page1["memories"]}
        ids2 = {m["id"] for m in page2["memories"]}
        assert not ids1.intersection(ids2)

    def test_tools_list(self, temp_root):
        """Tools list returns expected tools."""
        service = Mem0Service(temp_root)
        result = service.tools_list()

        assert "tools" in result
        tool_names = {t["name"] for t in result["tools"]}
        expected = {"mem0_store", "mem0_retrieve", "mem0_update", "mem0_delete", "mem0_list"}
        assert expected.issubset(tool_names)

    def test_tools_call_store(self, temp_root):
        """Tools call dispatches store correctly."""
        service = Mem0Service(temp_root)
        result = service.tools_call("mem0_store", {
            "agent_id": "test-agent",
            "content": "Test content via MCP",
        })
        assert "error" not in result
        assert "id" in result

    def test_tools_call_unknown(self, temp_root):
        """Tools call returns error for unknown tool."""
        service = Mem0Service(temp_root)
        result = service.tools_call("unknown_tool", {})
        assert "error" in result
        assert "unknown tool" in result["error"]


class TestExtractMemoriesFromBusEvent:
    """Tests for bus event memory extraction."""

    def test_extract_from_dialogos_output(self):
        """Extracts memory from dialogos output."""
        event = {
            "id": "evt-1",
            "topic": "dialogos.cell.output",
            "kind": "response",
            "actor": "dialogosd",
            "data": {
                "content": "This is a substantial LLM response about Python programming and its applications in data science and machine learning.",
                "provider": "anthropic",
            },
        }

        candidates = extract_memories_from_bus_event(event)
        assert len(candidates) > 0
        assert "dialogos" in candidates[0]["tags"]
        assert candidates[0]["metadata"]["source"] == "dialogos"

    def test_extract_from_task_completion(self):
        """Extracts memory from task completion."""
        event = {
            "id": "evt-2",
            "topic": "codex.task.complete",
            "kind": "response",
            "actor": "codex-peer",
            "data": {
                "summary": "Successfully implemented the new feature with proper error handling and tests.",
                "status": "success",
            },
        }

        candidates = extract_memories_from_bus_event(event)
        assert len(candidates) > 0
        assert "task" in candidates[0]["tags"]

    def test_skip_metric_events(self):
        """Skips metric events."""
        event = {
            "id": "evt-3",
            "topic": "system.metrics",
            "kind": "metric",
            "actor": "monitor",
            "data": {"cpu": 50, "memory": 75},
        }

        candidates = extract_memories_from_bus_event(event)
        assert len(candidates) == 0

    def test_skip_short_content(self):
        """Skips content that is too short."""
        event = {
            "id": "evt-4",
            "topic": "dialogos.cell.output",
            "kind": "response",
            "actor": "dialogosd",
            "data": {
                "content": "OK",  # Too short
                "provider": "test",
            },
        }

        candidates = extract_memories_from_bus_event(event)
        assert len(candidates) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
