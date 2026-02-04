#!/usr/bin/env python3
"""
Tests for SemanticBuchiRule - Buchi automaton with semantic motif tracking.

These tests verify:
1. Motif tracking and completion detection
2. State machine transitions (observe -> good -> stale -> zombie)
3. Per-actor isolation
4. Recovery intervention flow
5. Bus event emission
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from nucleus.tools.semantic_buchi_rule import (
    DEFAULT_STALE_THRESHOLD,
    DEFAULT_WINDOW_S,
    MotifTracker,
    SemanticBuchiRule,
    SemanticState,
    create_default_rule,
    load_motif_registry,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def simple_registry() -> dict:
    """Simple motif registry for testing."""
    return {
        "motifs": [
            {
                "id": "test_request_response",
                "vertices": ["test.request", "test.response"],
                "edges": [
                    {"from": "test.request", "to": "test.response", "type": "temporal"}
                ],
                "correlation_key": "req_id",
                "max_span_s": 60,
                "weight": 1.0,
            },
            {
                "id": "test_single_vertex",
                "vertices": ["test.heartbeat"],
                "edges": [],
                "correlation_key": "actor",
                "max_span_s": 30,
                "weight": 0.3,
            },
        ],
        "settings": {
            "default_max_span_s": 300,
            "default_weight": 1.0,
        },
    }


@pytest.fixture
def motif_tracker(simple_registry: dict) -> MotifTracker:
    """Create a MotifTracker with simple registry."""
    return MotifTracker(simple_registry)


@pytest.fixture
def rule_config() -> dict:
    """Default rule configuration for testing."""
    return {
        "id": "test-semantic-buchi",
        "motif_registry": "nucleus/specs/omega_motifs.json",
        "window_s": 10,  # Short window for testing
        "stale_threshold": 2,
        "per_actor": True,
    }


@pytest.fixture
def semantic_rule(rule_config: dict) -> SemanticBuchiRule:
    """Create a SemanticBuchiRule for testing."""
    return SemanticBuchiRule(rule_config)


# -----------------------------------------------------------------------------
# MotifTracker Tests
# -----------------------------------------------------------------------------
class TestMotifTracker:
    """Tests for MotifTracker class."""

    def test_create_tracker(self, simple_registry: dict) -> None:
        """Test tracker creation."""
        tracker = MotifTracker(simple_registry)
        assert len(tracker.motifs) == 2
        assert "test.request" in tracker._vertex_index
        assert "test.response" in tracker._vertex_index
        assert "test.heartbeat" in tracker._vertex_index

    def test_single_vertex_motif(self, motif_tracker: MotifTracker) -> None:
        """Test single-vertex motif completes immediately."""
        now = time.time()
        event = {
            "topic": "test.heartbeat",
            "actor": "test-actor",
            "data": {},
        }
        completed = motif_tracker.process_event(event, now)
        assert len(completed) == 1
        assert completed[0]["motif_id"] == "test_single_vertex"
        assert completed[0]["duration_s"] == 0.0

    def test_two_vertex_motif_completion(self, motif_tracker: MotifTracker) -> None:
        """Test two-vertex motif requires both events."""
        now = time.time()

        # First event: request
        event1 = {
            "topic": "test.request",
            "actor": "test-actor",
            "data": {"req_id": "abc123"},
        }
        completed1 = motif_tracker.process_event(event1, now)
        assert len(completed1) == 0
        assert len(motif_tracker.partial_matches) == 1

        # Second event: response
        event2 = {
            "topic": "test.response",
            "actor": "test-actor",
            "data": {"req_id": "abc123"},
        }
        completed2 = motif_tracker.process_event(event2, now + 1)
        assert len(completed2) == 1
        assert completed2[0]["motif_id"] == "test_request_response"
        assert completed2[0]["correlation_value"] == "abc123"
        assert completed2[0]["duration_s"] == 1.0
        assert len(motif_tracker.partial_matches) == 0

    def test_motif_expiration(self, motif_tracker: MotifTracker) -> None:
        """Test motif partial match expires after max_span."""
        now = time.time()

        # Start a request
        event1 = {
            "topic": "test.request",
            "actor": "test-actor",
            "data": {"req_id": "expired123"},
        }
        motif_tracker.process_event(event1, now)
        assert len(motif_tracker.partial_matches) == 1

        # Response after max_span (60s) should not complete
        event2 = {
            "topic": "test.response",
            "actor": "test-actor",
            "data": {"req_id": "expired123"},
        }
        completed = motif_tracker.process_event(event2, now + 70)
        assert len(completed) == 0
        # Partial match should be pruned
        assert len(motif_tracker.partial_matches) == 0

    def test_correlation_isolation(self, motif_tracker: MotifTracker) -> None:
        """Test that different correlation values are isolated."""
        now = time.time()

        # Start two requests
        event1 = {"topic": "test.request", "actor": "a", "data": {"req_id": "r1"}}
        event2 = {"topic": "test.request", "actor": "a", "data": {"req_id": "r2"}}
        motif_tracker.process_event(event1, now)
        motif_tracker.process_event(event2, now + 0.1)
        assert len(motif_tracker.partial_matches) == 2

        # Complete only r1
        event3 = {"topic": "test.response", "actor": "a", "data": {"req_id": "r1"}}
        completed = motif_tracker.process_event(event3, now + 1)
        assert len(completed) == 1
        assert completed[0]["correlation_value"] == "r1"
        assert len(motif_tracker.partial_matches) == 1  # r2 still pending

    def test_completion_rate(self, motif_tracker: MotifTracker) -> None:
        """Test completion rate calculation."""
        now = time.time()

        # Complete 5 heartbeats
        for i in range(5):
            event = {"topic": "test.heartbeat", "actor": "test", "data": {}}
            motif_tracker.process_event(event, now + i)

        # Rate should be 5 completions in 10 seconds = 0.5/s
        rate = motif_tracker.completion_rate(10.0, now + 5)
        assert 0.49 <= rate <= 0.51

    def test_weighted_throughput(self, motif_tracker: MotifTracker) -> None:
        """Test weighted throughput calculation."""
        now = time.time()

        # Complete a request-response (weight 1.0)
        motif_tracker.process_event(
            {"topic": "test.request", "actor": "a", "data": {"req_id": "1"}},
            now
        )
        motif_tracker.process_event(
            {"topic": "test.response", "actor": "a", "data": {"req_id": "1"}},
            now + 0.1
        )

        # Complete a heartbeat (weight 0.3)
        motif_tracker.process_event(
            {"topic": "test.heartbeat", "actor": "a", "data": {}},
            now + 0.2
        )

        throughput = motif_tracker.weighted_throughput(10.0, now + 1)
        assert 1.29 <= throughput <= 1.31  # 1.0 + 0.3


# -----------------------------------------------------------------------------
# SemanticBuchiRule Tests
# -----------------------------------------------------------------------------
class TestSemanticBuchiRule:
    """Tests for SemanticBuchiRule class."""

    def test_create_rule(self, rule_config: dict) -> None:
        """Test rule creation."""
        rule = SemanticBuchiRule(rule_config)
        assert rule.rule_id == "test-semantic-buchi"
        assert rule.window_s == 10
        assert rule.stale_threshold == 2

    def test_initial_state(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test initial state is q_observe."""
        assert semantic_rule.global_state == SemanticState.Q_OBSERVE
        assert len(semantic_rule.per_actor_state) == 0

    def test_transition_observe_to_good(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test transition from q_observe to q_good on motif completion."""
        now = time.time()

        # Simulate inference request/response
        event1 = {
            "topic": "infer_sync.request",
            "actor": "claude",
            "data": {"req_id": "test-1"},
        }
        semantic_rule.process_event(event1, now)

        event2 = {
            "topic": "infer_sync.response",
            "actor": "claude",
            "data": {"req_id": "test-1"},
        }
        alerts = semantic_rule.process_event(event2, now + 1)

        # Should have motif_complete alert
        assert any(a["topic"] == "omega.guardian.semantic.motif_complete" for a in alerts)

        # Actor should be in q_good state
        assert "claude" in semantic_rule.per_actor_state
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_GOOD

    def test_transition_good_to_stale(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test transition from q_good to q_stale on window timeout."""
        now = time.time()

        # Get actor to q_good state
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "1"}},
            now
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "1"}},
            now + 0.1
        )
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_GOOD

        # Evaluate after window expires (window_s = 10)
        status, details = semantic_rule.evaluate_cycle(now + 15)
        assert status == "warn"
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_STALE
        assert semantic_rule.per_actor_state["claude"].stale_count == 1

    def test_transition_stale_to_zombie(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test transition from q_stale to q_zombie after threshold."""
        now = time.time()

        # Get actor to q_good state
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "1"}},
            now
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "1"}},
            now + 0.1
        )

        # First stale window
        semantic_rule.evaluate_cycle(now + 15)
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_STALE
        assert semantic_rule.per_actor_state["claude"].stale_count == 1

        # Second stale window (reaches threshold of 2)
        status, details = semantic_rule.evaluate_cycle(now + 30)
        assert status == "error"
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_ZOMBIE
        assert "claude" in details["violations"]

    def test_stale_to_good_on_motif(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test transition from q_stale back to q_good on motif completion."""
        now = time.time()

        # Get actor to stale state
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "1"}},
            now
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "1"}},
            now + 0.1
        )
        semantic_rule.evaluate_cycle(now + 15)
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_STALE

        # Complete a new motif
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "2"}},
            now + 16
        )
        alerts = semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "2"}},
            now + 17
        )

        # Should be back to q_good
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_GOOD
        assert semantic_rule.per_actor_state["claude"].stale_count == 0

    def test_per_actor_isolation(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test that actors are tracked independently."""
        now = time.time()

        # Actor A: gets to q_good
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "actorA", "data": {"req_id": "a1"}},
            now
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "actorA", "data": {"req_id": "a1"}},
            now + 0.1
        )

        # Actor B: gets to q_good
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "actorB", "data": {"req_id": "b1"}},
            now + 0.2
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "actorB", "data": {"req_id": "b1"}},
            now + 0.3
        )

        # Both in q_good
        assert semantic_rule.per_actor_state["actorA"].state == SemanticState.Q_GOOD
        assert semantic_rule.per_actor_state["actorB"].state == SemanticState.Q_GOOD

        # Stale window - both go stale
        semantic_rule.evaluate_cycle(now + 15)

        # Actor B recovers
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "actorB", "data": {"req_id": "b2"}},
            now + 16
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "actorB", "data": {"req_id": "b2"}},
            now + 17
        )

        # Actor A is stale, Actor B is good
        assert semantic_rule.per_actor_state["actorA"].state == SemanticState.Q_STALE
        assert semantic_rule.per_actor_state["actorB"].state == SemanticState.Q_GOOD

    def test_intervention_recovery(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test intervention triggers recovery flow."""
        now = time.time()

        # Get actor to zombie state
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "1"}},
            now
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "1"}},
            now + 0.1
        )
        semantic_rule.evaluate_cycle(now + 15)  # stale_count = 1
        semantic_rule.evaluate_cycle(now + 30)  # stale_count = 2 -> zombie
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_ZOMBIE

        # Trigger intervention
        success = semantic_rule.trigger_intervention("claude", now + 31)
        assert success
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_RECOVERING

        # Complete a motif during recovery
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "2"}},
            now + 32
        )
        alerts = semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "2"}},
            now + 33
        )

        # Should be back to q_good
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_GOOD
        # Should have recovery success alert
        assert any(a["topic"] == "omega.guardian.semantic.recovered" for a in alerts)

    def test_recovery_timeout(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test recovery times out and returns to zombie."""
        now = time.time()
        semantic_rule.recovery_timeout_s = 5  # Short timeout for testing

        # Get actor to recovering state
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "1"}},
            now
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "1"}},
            now + 0.1
        )
        semantic_rule.evaluate_cycle(now + 15)
        semantic_rule.evaluate_cycle(now + 30)
        semantic_rule.trigger_intervention("claude", now + 31)
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_RECOVERING

        # Evaluate after recovery timeout
        status, details = semantic_rule.evaluate_cycle(now + 40)  # 9s after intervention
        assert semantic_rule.per_actor_state["claude"].state == SemanticState.Q_ZOMBIE

    def test_summary(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test summary generation."""
        now = time.time()

        # Add some state
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "1"}},
            now
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "1"}},
            now + 0.1
        )
        semantic_rule.evaluate_cycle(now + 1)

        summary = semantic_rule.get_summary(now + 2)
        assert summary["rule_id"] == "test-semantic-buchi"
        assert summary["actors_tracked"] == 1
        assert summary["actors_good"] == 1
        assert summary["total_motifs_window"] >= 1

    def test_reset_actor(self, semantic_rule: SemanticBuchiRule) -> None:
        """Test actor state reset."""
        now = time.time()

        # Add actor state
        semantic_rule.process_event(
            {"topic": "infer_sync.request", "actor": "claude", "data": {"req_id": "1"}},
            now
        )
        semantic_rule.process_event(
            {"topic": "infer_sync.response", "actor": "claude", "data": {"req_id": "1"}},
            now + 0.1
        )
        assert "claude" in semantic_rule.per_actor_state

        # Reset
        semantic_rule.reset_actor("claude")
        assert "claude" not in semantic_rule.per_actor_state


# -----------------------------------------------------------------------------
# Utility Function Tests
# -----------------------------------------------------------------------------
class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_default_rule(self) -> None:
        """Test default rule creation."""
        rule = create_default_rule()
        assert rule.window_s == DEFAULT_WINDOW_S
        assert rule.stale_threshold == DEFAULT_STALE_THRESHOLD

    def test_load_motif_registry_missing(self) -> None:
        """Test loading non-existent registry returns empty."""
        registry = load_motif_registry("/nonexistent/path.json")
        assert registry == {"motifs": [], "settings": {}}


# -----------------------------------------------------------------------------
# SemanticState Enum Tests
# -----------------------------------------------------------------------------
class TestSemanticState:
    """Tests for SemanticState enum."""

    def test_all_states_defined(self) -> None:
        """Test all required states are defined."""
        assert SemanticState.Q_OBSERVE.value == "q_observe"
        assert SemanticState.Q_GOOD.value == "q_good"
        assert SemanticState.Q_STALE.value == "q_stale"
        assert SemanticState.Q_ZOMBIE.value == "q_zombie"
        assert SemanticState.Q_RECOVERING.value == "q_recovering"

    def test_state_count(self) -> None:
        """Test exactly 5 states are defined."""
        assert len(SemanticState) == 5


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
