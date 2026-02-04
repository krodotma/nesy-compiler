#!/usr/bin/env python3
"""Tests for graphiti_bridge - Temporal Knowledge Graph."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from graphiti_bridge import (
    GraphitiService,
    Entity,
    Fact,
    parse_iso_ts,
    extract_facts_from_bus_event,
)


@pytest.fixture
def temp_root():
    """Create a temporary root directory."""
    with tempfile.TemporaryDirectory(prefix="graphiti_test_") as tmpdir:
        root = Path(tmpdir)
        (root / ".pluribus" / "kg").mkdir(parents=True)
        yield root


class TestParseIsoTs:
    """Tests for ISO timestamp parsing."""

    def test_parse_z_suffix(self):
        """Parses timestamp with Z suffix."""
        ts = parse_iso_ts("2024-01-15T10:30:00Z")
        assert ts > 0

    def test_parse_invalid(self):
        """Returns 0 for invalid timestamp."""
        ts = parse_iso_ts("invalid")
        assert ts == 0.0

    def test_parse_empty(self):
        """Returns 0 for empty string."""
        ts = parse_iso_ts("")
        assert ts == 0.0


class TestEntity:
    """Tests for Entity dataclass."""

    def test_to_dict_round_trip(self):
        """Entity can round-trip through dict."""
        ent = Entity(
            id="ent-1",
            name="Python",
            entity_type="programming_language",
            properties={"year": 1991},
            aliases=["py", "python3"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        d = ent.to_dict()
        assert d["kind"] == "graphiti_entity"
        assert d["name"] == "Python"

        ent2 = Entity.from_dict(d)
        assert ent2.id == ent.id
        assert ent2.name == ent.name
        assert ent2.aliases == ent.aliases


class TestFact:
    """Tests for Fact dataclass."""

    def test_to_dict_round_trip(self):
        """Fact can round-trip through dict."""
        fact = Fact(
            id="fact-1",
            subject_id="ent-1",
            predicate="created_by",
            object_id="ent-2",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            confidence=0.95,
            source="wikipedia",
            properties={"verified": True},
            created_at="2024-01-01T00:00:00Z",
        )

        d = fact.to_dict()
        assert d["kind"] == "graphiti_fact"
        assert d["predicate"] == "created_by"

        fact2 = Fact.from_dict(d)
        assert fact2.id == fact.id
        assert fact2.confidence == fact.confidence

    def test_is_valid_at(self):
        """Tests temporal validity check."""
        fact = Fact(
            id="fact-1",
            subject_id="s",
            predicate="p",
            object_id="o",
            valid_from="2024-01-01T00:00:00Z",
            valid_to="2024-12-31T23:59:59Z",
            confidence=1.0,
            source="test",
        )

        # Within range
        mid_ts = parse_iso_ts("2024-06-15T00:00:00Z")
        assert fact.is_valid_at(mid_ts)

        # Before range
        before_ts = parse_iso_ts("2023-06-15T00:00:00Z")
        assert not fact.is_valid_at(before_ts)

        # After range
        after_ts = parse_iso_ts("2025-06-15T00:00:00Z")
        assert not fact.is_valid_at(after_ts)

    def test_is_valid_at_open_ended(self):
        """Open-ended facts (no valid_to) are valid indefinitely."""
        fact = Fact(
            id="fact-1",
            subject_id="s",
            predicate="p",
            object_id="o",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,  # Still valid
            confidence=1.0,
            source="test",
        )

        # Way in the future
        future_ts = parse_iso_ts("2100-01-01T00:00:00Z")
        assert fact.is_valid_at(future_ts)


class TestGraphitiService:
    """Tests for GraphitiService."""

    def test_add_entity(self, temp_root):
        """Can add an entity."""
        service = GraphitiService(temp_root)

        result = service.add_entity(
            name="Claude",
            entity_type="ai_model",
            properties={"company": "Anthropic"},
            aliases=["claude-3"],
        )

        assert "error" not in result
        assert "id" in result
        assert result["name"] == "Claude"
        assert result["status"] == "created"

    def test_add_entity_updates_existing(self, temp_root):
        """Adding entity with same name updates it."""
        service = GraphitiService(temp_root)

        # Add first
        result1 = service.add_entity(name="Python", entity_type="language")
        assert result1["status"] == "created"

        # Add again - should update
        result2 = service.add_entity(
            name="Python",
            entity_type="programming_language",
            aliases=["py"],
        )
        assert result2["status"] == "updated"
        assert result2["id"] == result1["id"]

    def test_add_entity_requires_name(self, temp_root):
        """Add entity fails without name."""
        service = GraphitiService(temp_root)
        result = service.add_entity(name="")
        assert "error" in result
        assert "name" in result["error"]

    def test_add_fact(self, temp_root):
        """Can add a fact."""
        service = GraphitiService(temp_root)

        result = service.add_fact(
            subject="Python",
            predicate="is_a",
            object_value="programming language",
            confidence=1.0,
            source="knowledge",
        )

        assert "error" not in result
        assert "id" in result
        assert result["predicate"] == "is_a"
        assert result["status"] == "created"

    def test_add_fact_creates_entities(self, temp_root):
        """Adding fact auto-creates missing entities."""
        service = GraphitiService(temp_root)

        result = service.add_fact(
            subject="NewEntity1",
            predicate="related_to",
            object_value="NewEntity2",
        )

        assert "error" not in result

        # Entities should exist
        ent1 = service.get_entity(name="NewEntity1")
        ent2 = service.get_entity(name="NewEntity2")
        assert ent1["found"]
        assert ent2["found"]

    def test_add_fact_requires_triple(self, temp_root):
        """Add fact fails without complete triple."""
        service = GraphitiService(temp_root)

        result = service.add_fact(subject="", predicate="p", object_value="o")
        assert "error" in result

        result = service.add_fact(subject="s", predicate="", object_value="o")
        assert "error" in result

        result = service.add_fact(subject="s", predicate="p", object_value="")
        assert "error" in result

    def test_query_facts_by_subject(self, temp_root):
        """Can query facts by subject."""
        service = GraphitiService(temp_root)

        # Add some facts
        service.add_fact(subject="Python", predicate="is_a", object_value="language")
        service.add_fact(subject="Python", predicate="created_by", object_value="Guido")
        service.add_fact(subject="Java", predicate="is_a", object_value="language")

        # Query by subject
        result = service.query_facts(subject="Python")
        assert result["count"] == 2
        predicates = {f["predicate"] for f in result["facts"]}
        assert "is_a" in predicates
        assert "created_by" in predicates

    def test_query_facts_by_predicate(self, temp_root):
        """Can query facts by predicate."""
        service = GraphitiService(temp_root)

        service.add_fact(subject="Python", predicate="is_a", object_value="language")
        service.add_fact(subject="Java", predicate="is_a", object_value="language")
        service.add_fact(subject="Python", predicate="version", object_value="3.12")

        result = service.query_facts(predicate="is_a")
        assert result["count"] == 2

    def test_query_facts_at_time(self, temp_root):
        """Can query facts valid at a specific time."""
        service = GraphitiService(temp_root)

        # Add fact with time bounds
        service.add_fact(
            subject="Company",
            predicate="ceo",
            object_value="Person1",
            valid_from="2020-01-01T00:00:00Z",
            valid_to="2023-01-01T00:00:00Z",
        )

        # Add current fact (no end date)
        service.add_fact(
            subject="Company",
            predicate="ceo",
            object_value="Person2",
            valid_from="2023-01-01T00:00:00Z",
        )

        # Query at time when Person1 was CEO
        result1 = service.query_facts(
            subject="Company",
            predicate="ceo",
            at_time="2022-06-01T00:00:00Z",
        )
        assert result1["count"] == 1
        assert result1["facts"][0]["object"]["name"] == "Person1"

        # Query at current time
        result2 = service.query_facts(
            subject="Company",
            predicate="ceo",
            at_time="2024-06-01T00:00:00Z",
        )
        assert result2["count"] == 1
        assert result2["facts"][0]["object"]["name"] == "Person2"

    def test_invalidate_fact(self, temp_root):
        """Can invalidate a fact."""
        service = GraphitiService(temp_root)

        # Add fact
        result = service.add_fact(
            subject="Product",
            predicate="price",
            object_value="$100",
        )
        fact_id = result["id"]

        # Invalidate
        inv_result = service.invalidate_fact(fact_id=fact_id, reason="price changed")
        assert "error" not in inv_result
        assert inv_result["status"] == "invalidated"
        assert inv_result["valid_to"] is not None

        # Query at current time should not return invalidated fact
        # (valid_to was just set to now, so fact is no longer valid)
        query_result = service.query_facts(
            subject="Product",
            predicate="price",
            at_time="2100-01-01T00:00:00Z",  # Future time
        )
        assert query_result["count"] == 0

    def test_get_entity_by_id(self, temp_root):
        """Can get entity by ID."""
        service = GraphitiService(temp_root)

        add_result = service.add_entity(name="TestEntity")
        entity_id = add_result["id"]

        get_result = service.get_entity(entity_id=entity_id)
        assert get_result["found"]
        assert get_result["entity"]["name"] == "TestEntity"

    def test_get_entity_by_name(self, temp_root):
        """Can get entity by name."""
        service = GraphitiService(temp_root)

        service.add_entity(name="TestEntity", aliases=["alias1"])

        # By exact name
        result1 = service.get_entity(name="TestEntity")
        assert result1["found"]

        # By alias
        result2 = service.get_entity(name="alias1")
        assert result2["found"]

    def test_list_entities(self, temp_root):
        """Can list entities with filtering."""
        service = GraphitiService(temp_root)

        service.add_entity(name="Python", entity_type="language")
        service.add_entity(name="Java", entity_type="language")
        service.add_entity(name="Anthropic", entity_type="company")

        # List all
        result = service.list_entities()
        assert result["count"] == 3

        # List by type
        result2 = service.list_entities(entity_type="language")
        assert result2["count"] == 2

    def test_tools_list(self, temp_root):
        """Tools list returns expected tools."""
        service = GraphitiService(temp_root)
        result = service.tools_list()

        assert "tools" in result
        tool_names = {t["name"] for t in result["tools"]}
        expected = {
            "graphiti_add_entity",
            "graphiti_add_fact",
            "graphiti_query",
            "graphiti_get_entity",
            "graphiti_list_entities",
            "graphiti_invalidate",
        }
        assert expected.issubset(tool_names)

    def test_tools_call_add_fact(self, temp_root):
        """Tools call dispatches add_fact correctly."""
        service = GraphitiService(temp_root)
        result = service.tools_call("graphiti_add_fact", {
            "subject": "A",
            "predicate": "relates_to",
            "object": "B",
        })
        assert "error" not in result
        assert "id" in result


class TestExtractFactsFromBusEvent:
    """Tests for bus event fact extraction."""

    def test_extract_from_task_event(self):
        """Extracts fact from task event."""
        event = {
            "id": "evt-1",
            "topic": "codex.task.complete",
            "kind": "response",
            "actor": "codex-peer",
            "iso": "2024-01-15T10:00:00Z",
            "data": {
                "task_id": "task-123",
                "status": "success",
            },
        }

        facts = extract_facts_from_bus_event(event)
        assert len(facts) > 0
        assert facts[0]["subject"] == "task:task-123"
        assert facts[0]["predicate"] == "has_status"
        assert facts[0]["object"] == "success"

    def test_extract_from_request_event(self):
        """Extracts fact from request event."""
        event = {
            "id": "evt-2",
            "topic": "dialogos.request",
            "kind": "request",
            "actor": "user-agent",
            "iso": "2024-01-15T10:00:00Z",
            "data": {
                "req_id": "req-456",
            },
        }

        facts = extract_facts_from_bus_event(event)
        assert len(facts) > 0
        assert "sent_request" in facts[0]["predicate"]

    def test_skip_metric_events(self):
        """Skips metric events."""
        event = {
            "id": "evt-3",
            "topic": "system.metrics",
            "kind": "metric",
            "actor": "monitor",
            "data": {"cpu": 50},
        }

        facts = extract_facts_from_bus_event(event)
        assert len(facts) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
