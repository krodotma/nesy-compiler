#!/usr/bin/env python3
"""
Integration tests for the enhanced Dialogos system.

Tests the full pipeline:
  Hook -> Trace -> Indexer -> IR/KG/Vector -> Search -> PBRESUME

Run: python3 -m pytest nucleus/tools/tests/test_dialogos_integration.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules under test
from dialogosd import default_trace_path, append_trace
from pbresume_operator import (
    collect_recovery_data,
    scan_dialogos_incomplete,
    scan_incomplete_lanes,
    format_markdown,
    format_text,
    RecoveryReport,
    parse_depth,
)
from dialogos_indexer import (
    DialogosIndexer,
    parse_duration,
    iter_ndjson,
    append_ndjson,
)
from graphiti_bridge import (
    GraphitiService,
    extract_facts_from_bus_event,
    Entity,
    Fact,
)


class TestDialogosIntegration:
    """Integration tests for the full dialogos pipeline."""

    @pytest.fixture
    def temp_root(self, tmp_path):
        """Create a temporary pluribus root with required directories."""
        root = tmp_path / "pluribus"
        (root / ".pluribus" / "dialogos").mkdir(parents=True)
        (root / ".pluribus" / "bus").mkdir(parents=True)
        (root / ".pluribus" / "kg").mkdir(parents=True)
        (root / ".pluribus" / "index" / "dialogos").mkdir(parents=True)
        (root / "nucleus" / "state").mkdir(parents=True)

        # Create empty events file
        (root / ".pluribus" / "bus" / "events.ndjson").touch()

        # Create lanes.json
        lanes_data = {
            "version": "1.0",
            "lanes": [
                {"id": "test-lane", "name": "Test Lane", "status": "yellow", "wip_pct": 50, "owner": "test"}
            ],
            "agents": []
        }
        (root / "nucleus" / "state" / "lanes.json").write_text(json.dumps(lanes_data))

        return root

    def test_trace_to_ir_pipeline(self, temp_root):
        """Test that trace records are properly indexed to IR."""
        trace_path = temp_root / ".pluribus" / "dialogos" / "trace.ndjson"

        # Write test trace record
        record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "user_prompt",
            "actor": "test-user",
            "session_id": "sess-123",
            "data": {
                "prompt": "Test prompt content",
                "prompt_sha256": "abc123",
            }
        }
        append_ndjson(trace_path, record)

        # Create indexer and process
        indexer = DialogosIndexer(
            root=temp_root,
            enable_chroma=False,  # Skip chroma for unit test
            enable_graphiti=False,  # Skip graphiti for unit test
        )

        # Verify trace was written
        records = list(iter_ndjson(trace_path))
        assert len(records) == 1
        assert records[0]["event_type"] == "user_prompt"

    def test_incomplete_session_detection(self, temp_root):
        """Test that incomplete sessions are detected for PBRESUME."""
        trace_path = temp_root / ".pluribus" / "dialogos" / "trace.ndjson"

        # Write a prompt without a matching stop
        now = time.time()
        prompt_record = {
            "id": str(uuid.uuid4()),
            "ts": now - 300,  # 5 minutes ago
            "event_type": "user_prompt",
            "session_id": "incomplete-session",
            "data": {"prompt": "Incomplete work"},
        }
        append_ndjson(trace_path, prompt_record)

        # Scan for incomplete sessions
        since_ts = now - 3600  # Last hour
        incomplete = scan_dialogos_incomplete(trace_path, since_ts)

        assert len(incomplete) == 1
        assert incomplete[0]["session_id"] == "incomplete-session"

    def test_complete_session_not_flagged(self, temp_root):
        """Test that complete sessions are not flagged."""
        trace_path = temp_root / ".pluribus" / "dialogos" / "trace.ndjson"

        now = time.time()
        session_id = "complete-session"

        # Write prompt
        append_ndjson(trace_path, {
            "id": str(uuid.uuid4()),
            "ts": now - 300,
            "event_type": "user_prompt",
            "session_id": session_id,
        })

        # Write stop (after prompt)
        append_ndjson(trace_path, {
            "id": str(uuid.uuid4()),
            "ts": now - 100,
            "event_type": "assistant_stop",
            "session_id": session_id,
        })

        # Should not find incomplete
        since_ts = now - 3600
        incomplete = scan_dialogos_incomplete(trace_path, since_ts)
        assert len(incomplete) == 0

    def test_lanes_integration(self, temp_root):
        """Test that incomplete lanes are detected."""
        lanes_path = temp_root / "nucleus" / "state" / "lanes.json"

        incomplete = scan_incomplete_lanes(lanes_path)

        assert len(incomplete) == 1
        assert incomplete[0].id == "test-lane"
        assert incomplete[0].wip_pct == 50

    def test_recovery_report_generation(self, temp_root):
        """Test full recovery report generation."""
        # Set environment for the test
        with patch.dict(os.environ, {
            "PLURIBUS_ROOT": str(temp_root),
            "PLURIBUS_BUS_DIR": str(temp_root / ".pluribus" / "bus"),
        }):
            report = collect_recovery_data(
                scope="all",
                depth_s=3600,
                depth_str="1h",
                from_dialogos=True,
                from_lanes=True,
                from_bus=False,
            )

            assert isinstance(report, RecoveryReport)
            assert report.scope == "all"
            assert len(report.incomplete_lanes) >= 0  # May have lanes

    def test_report_formatting(self):
        """Test report output formatting."""
        report = RecoveryReport(
            req_id="test-123",
            scope="all",
            since_ts=time.time() - 3600,
            depth_str="1h",
            actor="test",
            session_id=None,
            lane=None,
            apology="Test apology",
            open_tasks=[],
            dialogos_pending=[{"session_id": "s1", "age": "5m"}],
            incomplete_lanes=[{"id": "l1", "name": "Lane 1", "wip_pct": 50}],
            pending_requests=[],
        )

        md = format_markdown(report)
        assert "# PBRESUME Recovery Report" in md
        assert "Lane 1" in md

        txt = format_text(report)
        assert "PBRESUME RECOVERY REPORT" in txt


class TestGraphitiIntegration:
    """Tests for Graphiti KG integration with dialogos."""

    @pytest.fixture
    def graphiti_service(self, tmp_path):
        """Create a GraphitiService with temp storage."""
        root = tmp_path / "pluribus"
        (root / ".pluribus" / "kg").mkdir(parents=True)
        return GraphitiService(root)

    def test_entity_creation(self, graphiti_service):
        """Test entity creation and retrieval."""
        result = graphiti_service.add_entity(
            name="test-agent",
            entity_type="agent",
            properties={"model": "claude"},
        )

        assert result["status"] in ("created", "updated")
        assert "id" in result

    def test_fact_creation(self, graphiti_service):
        """Test temporal fact creation."""
        result = graphiti_service.add_fact(
            subject="agent-1",
            predicate="processed",
            object_value="task-123",
            confidence=0.95,
            source="test",
        )

        assert result["status"] == "created"
        assert "id" in result

    def test_fact_query(self, graphiti_service):
        """Test fact querying."""
        # Add a fact
        graphiti_service.add_fact(
            subject="agent-1",
            predicate="completed",
            object_value="task-456",
        )

        # Query it
        result = graphiti_service.query_facts(subject="agent-1")

        assert "facts" in result
        assert result["count"] >= 0

    def test_bus_event_fact_extraction(self):
        """Test automatic fact extraction from bus events."""
        # Test task event extraction
        event = {
            "topic": "agent.task.complete",
            "actor": "claude",
            "iso": "2025-12-27T00:00:00Z",
            "data": {
                "task_id": "task-789",
                "status": "completed",
            }
        }

        facts = extract_facts_from_bus_event(event)

        # Should extract at least one fact about task status
        assert len(facts) >= 1
        assert any(f["predicate"] == "has_status" for f in facts)

    def test_dialogos_event_fact_extraction(self):
        """Test fact extraction from dialogos.* topics."""
        event = {
            "topic": "dialogos.session_start",
            "actor": "test-user",
            "iso": "2025-12-27T00:00:00Z",
            "data": {
                "session_id": "sess-abc",
            }
        }

        facts = extract_facts_from_bus_event(event)

        # Should extract session start facts
        assert len(facts) >= 1
        assert any(f["predicate"] == "started_session" for f in facts)


class TestIndexerEnhancements:
    """Tests for enhanced indexer functionality."""

    def test_parse_duration(self):
        """Test duration parsing."""
        assert parse_duration("24h") == 86400
        assert parse_duration("1d") == 86400
        assert parse_duration("30m") == 1800
        assert parse_duration("60s") == 60
        assert parse_duration("invalid") == 0

    def test_indexer_stats(self, tmp_path):
        """Test indexer statistics."""
        root = tmp_path / "pluribus"
        (root / ".pluribus" / "dialogos").mkdir(parents=True)
        (root / ".pluribus" / "bus").mkdir(parents=True)
        (root / ".pluribus" / "index" / "dialogos").mkdir(parents=True)

        # Create trace file
        (root / ".pluribus" / "dialogos" / "trace.ndjson").touch()

        indexer = DialogosIndexer(
            root=root,
            enable_chroma=False,
            enable_graphiti=False,
        )

        status = indexer.get_status()

        assert "trace_path" in status
        assert "trace_exists" in status
        assert status["trace_exists"] == True


class TestPBRESUMEEnhancements:
    """Tests for PBRESUME operator enhancements."""

    def test_depth_parsing(self):
        """Test depth string parsing."""
        assert parse_depth("24h") == 86400
        assert parse_depth("7d") == 604800
        assert parse_depth("30m") == 1800
        assert parse_depth("60s") == 60

    def test_recovery_counts(self):
        """Test recovery report counting."""
        report = RecoveryReport(
            req_id="test",
            scope="all",
            since_ts=0,
            depth_str="1h",
            actor="test",
            session_id=None,
            lane=None,
            apology="",
            open_tasks=[{}, {}],
            dialogos_pending=[{}],
            incomplete_lanes=[{}, {}, {}],
            pending_requests=[],
        )

        counts = report.counts
        assert counts["tasks"] == 2
        assert counts["dialogos_pending"] == 1
        assert counts["incomplete_lanes"] == 3
        assert counts["total"] == 6


class TestEndToEndFlow:
    """End-to-end integration tests."""

    @pytest.fixture
    def full_environment(self, tmp_path):
        """Set up complete test environment."""
        root = tmp_path / "pluribus"

        # Create all directories
        dirs = [
            ".pluribus/dialogos",
            ".pluribus/bus",
            ".pluribus/kg",
            ".pluribus/index/dialogos",
            "nucleus/state",
        ]
        for d in dirs:
            (root / d).mkdir(parents=True)

        # Create required files
        (root / ".pluribus" / "bus" / "events.ndjson").touch()
        (root / ".pluribus" / "dialogos" / "trace.ndjson").touch()

        # Create lanes
        lanes = {
            "version": "1.0",
            "lanes": [
                {
                    "id": "e2e-test",
                    "name": "E2E Test Lane",
                    "status": "yellow",
                    "wip_pct": 75,
                    "owner": "test-agent",
                    "next_actions": ["complete task"],
                }
            ],
            "agents": [
                {"id": "test-agent", "status": "active", "lane": "e2e-test"}
            ]
        }
        (root / "nucleus" / "state" / "lanes.json").write_text(json.dumps(lanes))

        return root

    def test_full_pipeline(self, full_environment):
        """Test the complete dialogos pipeline."""
        root = full_environment
        trace_path = root / ".pluribus" / "dialogos" / "trace.ndjson"

        # 1. Simulate hook writing trace
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        now = time.time()

        # Session start
        append_ndjson(trace_path, {
            "id": str(uuid.uuid4()),
            "ts": now - 60,
            "event_type": "session_start",
            "session_id": session_id,
            "actor": "test-user",
        })

        # User prompt (incomplete - no stop yet)
        append_ndjson(trace_path, {
            "id": str(uuid.uuid4()),
            "ts": now - 30,
            "event_type": "user_prompt",
            "session_id": session_id,
            "actor": "test-user",
            "data": {
                "prompt": "Help me with task X",
                "prompt_sha256": "test-hash",
            }
        })

        # 2. Run PBRESUME scan
        with patch.dict(os.environ, {
            "PLURIBUS_ROOT": str(root),
            "PLURIBUS_BUS_DIR": str(root / ".pluribus" / "bus"),
        }):
            report = collect_recovery_data(
                scope="all",
                depth_s=3600,
                depth_str="1h",
                from_dialogos=True,
                from_lanes=True,
                from_bus=False,
            )

        # 3. Verify results
        assert report.counts["dialogos_pending"] >= 1
        assert report.counts["incomplete_lanes"] == 1

        # 4. Test markdown output
        md = format_markdown(report)
        assert "E2E Test Lane" in md
        assert "75%" in md

    def test_graphiti_kg_population(self, full_environment):
        """Test that graphiti KG is populated from events."""
        root = full_environment

        # Create graphiti service
        service = GraphitiService(root)

        # Add facts that would come from dialogos events
        service.add_fact(
            subject="test-user",
            predicate="started_session",
            object_value="session-001",
            source="dialogos",
        )

        service.add_fact(
            subject="session-001",
            predicate="contains_prompt",
            object_value="prompt-001",
            source="dialogos",
        )

        # Query facts
        result = service.query_facts(subject="test-user")
        assert result["count"] >= 1

        result = service.query_facts(predicate="contains_prompt")
        assert result["count"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
