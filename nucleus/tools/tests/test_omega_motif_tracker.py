#!/usr/bin/env python3
"""
Tests for omega_motif_tracker.py

Tests the MotifTracker class for semantic recurrence detection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure nucleus/tools is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools.omega_motif_tracker import (
    BUCHI_ACCEPTANCE_THRESHOLD,
    CompletedMotif,
    MotifTracker,
    PartialMatch,
    load_motif_registry,
)


@pytest.fixture
def sample_registry() -> dict:
    """Create a minimal test registry."""
    return {
        "schema_version": 1,
        "motifs": [
            {
                "id": "test_flow",
                "description": "Test request-response flow",
                "vertices": ["test.request", "test.started", "test.complete"],
                "edges": [
                    {"from": "test.request", "to": "test.started", "type": "temporal"},
                    {"from": "test.started", "to": "test.complete", "type": "temporal"},
                ],
                "correlation_key": "req_id",
                "max_span_s": 60,
                "weight": 1.0,
                "category": "test",
            },
            {
                "id": "simple_pair",
                "description": "Simple request-response",
                "vertices": ["simple.request", "simple.response"],
                "edges": [
                    {"from": "simple.request", "to": "simple.response", "type": "temporal"},
                ],
                "correlation_key": "id",
                "max_span_s": 30,
                "weight": 0.5,
                "category": "test",
            },
            {
                "id": "single_vertex",
                "description": "Single-vertex motif (tick)",
                "vertices": ["tick.event"],
                "edges": [],
                "correlation_key": "session_id",
                "max_span_s": 10,
                "weight": 0.1,
                "category": "heartbeat",
            },
            {
                "id": "filtered_motif",
                "description": "Motif with filters",
                "vertices": ["filtered.event"],
                "edges": [],
                "correlation_key": "id",
                "max_span_s": 30,
                "weight": 1.0,
                "category": "filtered",
                "filters": {
                    "state": ["RUNNING", "OK"],
                },
            },
        ],
        "settings": {
            "default_max_span_s": 300,
            "default_weight": 1.0,
            "partial_match_ttl_s": 120,
            "completion_history_size": 1000,
        },
    }


@pytest.fixture
def tracker(sample_registry: dict) -> MotifTracker:
    """Create a MotifTracker with the sample registry."""
    return MotifTracker(sample_registry, bus_dir="/tmp/test_bus")


class TestPartialMatch:
    """Tests for PartialMatch dataclass."""

    def test_age_calculation(self):
        """Test age calculation."""
        pm = PartialMatch(
            motif_id="test",
            correlation_key="req_id",
            correlation_value="123",
            started_ts=100.0,
            last_ts=110.0,
            seen_vertices={"a", "b"},
        )
        assert pm.age(120.0) == 20.0

    def test_idle_time_calculation(self):
        """Test idle time calculation."""
        pm = PartialMatch(
            motif_id="test",
            correlation_key="req_id",
            correlation_value="123",
            started_ts=100.0,
            last_ts=110.0,
            seen_vertices={"a", "b"},
        )
        assert pm.idle_time(120.0) == 10.0

    def test_to_dict(self):
        """Test serialization to dict."""
        pm = PartialMatch(
            motif_id="test",
            correlation_key="req_id",
            correlation_value="123",
            started_ts=100.0,
            last_ts=110.0,
            seen_vertices={"a", "b"},
            weight=1.5,
        )
        d = pm.to_dict()
        assert d["motif_id"] == "test"
        assert d["correlation_value"] == "123"
        assert set(d["seen_vertices"]) == {"a", "b"}
        assert d["weight"] == 1.5


class TestCompletedMotif:
    """Tests for CompletedMotif dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        cm = CompletedMotif(
            motif_id="test",
            correlation_value="123",
            completed_ts=200.0,
            duration_s=10.0,
            weight=1.0,
            actor="test_actor",
        )
        d = cm.to_dict()
        assert d["motif_id"] == "test"
        assert d["completed_ts"] == 200.0
        assert d["actor"] == "test_actor"


class TestMotifTracker:
    """Tests for MotifTracker class."""

    def test_init_builds_indexes(self, tracker: MotifTracker):
        """Test that indexes are built on init."""
        assert "test_flow" in tracker._motif_by_id
        assert "simple_pair" in tracker._motif_by_id
        assert "test.request" in tracker._motifs_by_vertex
        assert "test.started" in tracker._motifs_by_vertex

    def test_process_event_starts_partial_match(self, tracker: MotifTracker):
        """Test that start vertex creates partial match."""
        event = {
            "topic": "test.request",
            "data": {"req_id": "abc123"},
            "actor": "test",
        }
        completed = tracker.process_event(event, 100.0)

        assert completed == []
        assert "test_flow:abc123" in tracker.partial_matches
        pm = tracker.partial_matches["test_flow:abc123"]
        assert pm.seen_vertices == {"test.request"}
        assert pm.started_ts == 100.0

    def test_process_event_advances_partial_match(self, tracker: MotifTracker):
        """Test that subsequent vertices advance partial match."""
        # Start
        tracker.process_event(
            {"topic": "test.request", "data": {"req_id": "abc"}},
            100.0,
        )
        # Advance
        completed = tracker.process_event(
            {"topic": "test.started", "data": {"req_id": "abc"}},
            105.0,
        )

        assert completed == []
        pm = tracker.partial_matches["test_flow:abc"]
        assert pm.seen_vertices == {"test.request", "test.started"}
        assert pm.last_ts == 105.0

    def test_process_event_completes_motif(self, tracker: MotifTracker):
        """Test that final vertex completes motif."""
        tracker.process_event(
            {"topic": "test.request", "data": {"req_id": "xyz"}},
            100.0,
        )
        tracker.process_event(
            {"topic": "test.started", "data": {"req_id": "xyz"}},
            105.0,
        )
        completed = tracker.process_event(
            {"topic": "test.complete", "data": {"req_id": "xyz"}},
            110.0,
        )

        assert completed == ["test_flow"]
        assert "test_flow:xyz" not in tracker.partial_matches
        assert len(tracker.completed_motifs) == 1
        assert tracker.completed_motifs[0] == (110.0, "test_flow")

    def test_simple_pair_completion(self, tracker: MotifTracker):
        """Test simple two-vertex motif."""
        tracker.process_event(
            {"topic": "simple.request", "data": {"id": "s1"}},
            100.0,
        )
        completed = tracker.process_event(
            {"topic": "simple.response", "data": {"id": "s1"}},
            102.0,
        )

        assert completed == ["simple_pair"]

    def test_single_vertex_motif_completes_immediately(self, tracker: MotifTracker):
        """Test single-vertex motif completes on first event."""
        completed = tracker.process_event(
            {"topic": "tick.event", "data": {"session_id": "sess1"}},
            100.0,
        )

        assert completed == ["single_vertex"]

    def test_expired_partial_match_is_pruned(self, tracker: MotifTracker):
        """Test that partial matches exceeding max_span are removed."""
        tracker.process_event(
            {"topic": "test.request", "data": {"req_id": "old"}},
            100.0,
        )
        # Process event after max_span (60s)
        tracker.process_event(
            {"topic": "test.started", "data": {"req_id": "old"}},
            200.0,  # 100s later, exceeds 60s max_span
        )

        # Should be pruned
        assert "test_flow:old" not in tracker.partial_matches

    def test_correlation_from_nested_data(self, tracker: MotifTracker):
        """Test correlation extraction from nested data."""
        event = {
            "topic": "test.request",
            "data": {"meta": {"req_id": "nested123"}},
        }
        completed = tracker.process_event(event, 100.0)

        assert "test_flow:nested123" in tracker.partial_matches

    def test_missing_correlation_skips_event(self, tracker: MotifTracker):
        """Test that events without correlation key are skipped for multi-vertex motifs."""
        event = {
            "topic": "test.request",
            "data": {"other_key": "value"},
        }
        tracker.process_event(event, 100.0)

        # No partial match should be created
        assert len(tracker.partial_matches) == 0

    def test_filtered_motif_accepts_matching(self, tracker: MotifTracker):
        """Test that filtered motif accepts matching events."""
        event = {
            "topic": "filtered.event",
            "data": {"id": "f1", "state": "RUNNING"},
        }
        completed = tracker.process_event(event, 100.0)

        assert completed == ["filtered_motif"]

    def test_filtered_motif_rejects_non_matching(self, tracker: MotifTracker):
        """Test that filtered motif rejects non-matching events."""
        event = {
            "topic": "filtered.event",
            "data": {"id": "f2", "state": "FAILED"},
        }
        completed = tracker.process_event(event, 100.0)

        assert completed == []

    def test_completion_rate(self, tracker: MotifTracker):
        """Test completion rate calculation."""
        # Complete 3 motifs in 30 seconds
        for i in range(3):
            tracker.process_event(
                {"topic": "simple.request", "data": {"id": f"r{i}"}},
                100.0 + i * 10,
            )
            tracker.process_event(
                {"topic": "simple.response", "data": {"id": f"r{i}"}},
                102.0 + i * 10,
            )

        # Rate in 60s window
        rate = tracker.completion_rate(60.0, 130.0)
        assert rate == pytest.approx(3 / 60.0)

    def test_weighted_completion_rate(self, tracker: MotifTracker):
        """Test weighted completion rate."""
        # simple_pair has weight 0.5
        tracker.process_event(
            {"topic": "simple.request", "data": {"id": "w1"}},
            100.0,
        )
        tracker.process_event(
            {"topic": "simple.response", "data": {"id": "w1"}},
            101.0,
        )

        rate = tracker.weighted_completion_rate(60.0, 110.0)
        assert rate == pytest.approx(0.5 / 60.0)

    def test_last_completion_ts(self, tracker: MotifTracker):
        """Test last completion timestamp."""
        assert tracker.last_completion_ts() is None

        tracker.process_event(
            {"topic": "simple.request", "data": {"id": "l1"}},
            100.0,
        )
        tracker.process_event(
            {"topic": "simple.response", "data": {"id": "l1"}},
            105.0,
        )

        assert tracker.last_completion_ts() == 105.0

    def test_get_partial_matches(self, tracker: MotifTracker):
        """Test get_partial_matches returns correct info."""
        tracker.process_event(
            {"topic": "test.request", "data": {"req_id": "pm1"}},
            100.0,
        )
        tracker.process_event(
            {"topic": "test.started", "data": {"req_id": "pm1"}},
            102.0,
        )

        partials = tracker.get_partial_matches()
        assert len(partials) == 1
        p = partials[0]
        assert p["motif_id"] == "test_flow"
        assert p["seen_count"] == 2
        assert p["total_vertices"] == 3
        assert p["progress_pct"] == pytest.approx(66.7, rel=0.1)

    def test_buchi_acceptance_requires_threshold(self, tracker: MotifTracker):
        """Test Buchi acceptance requires BUCHI_ACCEPTANCE_THRESHOLD completions."""
        assert not tracker.check_buchi_acceptance("simple_pair")

        # Complete the motif THRESHOLD times
        for i in range(BUCHI_ACCEPTANCE_THRESHOLD):
            tracker.process_event(
                {"topic": "simple.request", "data": {"id": f"b{i}"}},
                100.0 + i,
            )
            tracker.process_event(
                {"topic": "simple.response", "data": {"id": f"b{i}"}},
                101.0 + i,
            )

        assert tracker.check_buchi_acceptance("simple_pair")

    def test_buchi_status(self, tracker: MotifTracker):
        """Test get_buchi_status returns all motifs."""
        status = tracker.get_buchi_status()

        assert status["threshold"] == BUCHI_ACCEPTANCE_THRESHOLD
        assert "test_flow" in status["motifs"]
        assert "simple_pair" in status["motifs"]
        assert status["motifs"]["test_flow"]["count"] == 0
        assert status["motifs"]["test_flow"]["accepted"] is False

    def test_get_completion_history(self, tracker: MotifTracker):
        """Test completion history retrieval."""
        # Complete two motifs at different times
        tracker.process_event(
            {"topic": "simple.request", "data": {"id": "h1"}},
            100.0,
        )
        tracker.process_event(
            {"topic": "simple.response", "data": {"id": "h1"}},
            101.0,
        )
        tracker.process_event(
            {"topic": "simple.request", "data": {"id": "h2"}},
            200.0,
        )
        tracker.process_event(
            {"topic": "simple.response", "data": {"id": "h2"}},
            201.0,
        )

        # All history
        history = tracker.get_completion_history()
        assert len(history) == 2

        # Windowed history
        recent = tracker.get_completion_history(window_s=50.0, now=210.0)
        assert len(recent) == 1
        assert recent[0][0] == 201.0

    def test_get_stats(self, tracker: MotifTracker):
        """Test stats aggregation."""
        tracker.process_event(
            {"topic": "simple.request", "data": {"id": "s1"}},
            100.0,
        )
        tracker.process_event(
            {"topic": "simple.response", "data": {"id": "s1"}},
            101.0,
        )

        stats = tracker.get_stats(window_s=60.0)

        assert stats["total_completions"] == 1
        assert stats["partial_matches"] == 0
        assert stats["motifs_registered"] == 4
        assert stats["last_completion_ts"] == 101.0

    def test_multiple_motifs_same_topic(self, sample_registry: dict):
        """Test handling when one topic belongs to multiple motifs."""
        # Add a second motif that uses test.request
        sample_registry["motifs"].append({
            "id": "alt_flow",
            "vertices": ["test.request", "alt.complete"],
            "edges": [
                {"from": "test.request", "to": "alt.complete", "type": "temporal"},
            ],
            "correlation_key": "req_id",
            "max_span_s": 60,
            "weight": 1.0,
        })

        tracker = MotifTracker(sample_registry, bus_dir="/tmp/test")

        # Start event creates partial for both motifs
        tracker.process_event(
            {"topic": "test.request", "data": {"req_id": "multi1"}},
            100.0,
        )

        assert "test_flow:multi1" in tracker.partial_matches
        assert "alt_flow:multi1" in tracker.partial_matches

    def test_out_of_order_events_rejected(self, tracker: MotifTracker):
        """Test that out-of-order events don't advance partial match."""
        tracker.process_event(
            {"topic": "test.request", "data": {"req_id": "ooo"}},
            100.0,
        )
        # Skip test.started and try test.complete
        completed = tracker.process_event(
            {"topic": "test.complete", "data": {"req_id": "ooo"}},
            102.0,
        )

        # Should not complete
        assert completed == []
        pm = tracker.partial_matches.get("test_flow:ooo")
        assert pm is not None
        # Should not have added test.complete
        assert "test.complete" not in pm.seen_vertices


class TestLoadRegistry:
    """Tests for load_motif_registry function."""

    def test_load_default_registry(self):
        """Test loading the default registry file."""
        registry = load_motif_registry()

        assert "motifs" in registry
        assert "settings" in registry
        assert len(registry["motifs"]) > 0

    def test_load_nonexistent_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_motif_registry("/nonexistent/path.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
