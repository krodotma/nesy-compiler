#!/usr/bin/env python3
"""
Integration tests for CMP Engine LENS/Omega wiring.

Tests that:
1. entropy.profile.computed events are processed
2. omega.guardian.semantic.motif_complete events are processed
3. cmp_extensions functions work correctly
"""
import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cmp_extensions import (
    compute_e_factor,
    compute_motif_bonus,
    adaptive_omega_thresholds,
    EntropyVectorCompat,
    classify_cmp,
    PHI_INV,
)
from cmp_engine import CMPEngine


def test_entropy_vector_compat():
    """Test EntropyVectorCompat dataclass."""
    v = EntropyVectorCompat(
        h_info=0.8,
        h_miss=0.1,
        h_conj=0.05,
        h_alea=0.1,
        h_epis=0.1,
        h_struct=0.05,
        c_load=0.1,
        h_goal_drift=0.05
    )
    assert 0 < v.h_total < 1, f"h_total should be in (0,1), got {v.h_total}"
    assert 0 < v.h_mean < 1, f"h_mean should be in (0,1), got {v.h_mean}"
    assert v.utility > 0, f"utility should be positive for good vector, got {v.utility}"
    print(f"[PASS] EntropyVectorCompat: h_total={v.h_total:.3f}, h_mean={v.h_mean:.3f}, utility={v.utility:.3f}")


def test_entropy_vector_from_dict():
    """Test EntropyVectorCompat.from_dict()."""
    d = {"h_info": 0.7, "h_miss": 0.2, "h_conj": 0.1}
    v = EntropyVectorCompat.from_dict(d)
    assert v.h_info == 0.7
    assert v.h_miss == 0.2
    assert v.h_conj == 0.1
    assert v.h_alea == 0.0  # Default
    print("[PASS] EntropyVectorCompat.from_dict()")


def test_compute_e_factor():
    """Test E-factor computation."""
    # High quality vector
    good = EntropyVectorCompat(h_info=0.9, h_miss=0.05, h_conj=0.02, h_alea=0.05, h_epis=0.05, c_load=0.1)
    e_good = compute_e_factor(good)
    assert 0.5 < e_good < 2.0, f"E-factor for good vector should be high, got {e_good}"

    # Low quality vector
    bad = EntropyVectorCompat(h_info=0.2, h_miss=0.5, h_conj=0.3, h_alea=0.4, h_epis=0.3, c_load=0.5)
    e_bad = compute_e_factor(bad)
    assert e_bad < e_good, f"Bad vector E-factor should be lower: {e_bad} >= {e_good}"

    print(f"[PASS] compute_e_factor: good={e_good:.3f}, bad={e_bad:.3f}")


def test_compute_e_factor_dict():
    """Test E-factor computation from dict."""
    d = {"h_info": 0.8, "h_miss": 0.1, "h_conj": 0.1, "h_alea": 0.1, "h_epis": 0.1, "c_load": 0.1}
    e = compute_e_factor(d)
    assert 0 < e < 2, f"E-factor from dict should be in (0,2), got {e}"
    print(f"[PASS] compute_e_factor(dict): {e:.3f}")


def test_compute_motif_bonus():
    """Test motif bonus computation."""
    event = {
        "data": {
            "motif_id": "inference_complete",
            "weight": 1.5,
            "duration_s": 30.0
        }
    }
    bonus = compute_motif_bonus(event, "test-lineage")
    assert bonus > 0, f"Bonus should be positive, got {bonus}"
    assert bonus < 0.5, f"Bonus should be reasonable, got {bonus}"
    print(f"[PASS] compute_motif_bonus: {bonus:.4f}")


def test_adaptive_thresholds():
    """Test adaptive threshold computation."""
    # Excellent CMP
    t_exc = adaptive_omega_thresholds(1.0)
    assert t_exc["window_s"] == 60

    # Poor CMP
    t_poor = adaptive_omega_thresholds(0.2)
    assert t_poor["window_s"] == 300
    assert t_poor["stale_threshold"] > t_exc["stale_threshold"]

    print(f"[PASS] adaptive_omega_thresholds: excellent={t_exc}, poor={t_poor}")


def test_classify_cmp():
    """Test CMP classification."""
    assert classify_cmp(1.0) == "excellent"
    assert classify_cmp(0.65) == "good"  # Clearly above 0.618
    assert classify_cmp(0.4) == "fair"   # Between 0.382 and 0.618
    assert classify_cmp(0.25) == "poor"  # Between 0.236 and 0.382
    assert classify_cmp(0.1) == "critical"  # Below 0.236
    print("[PASS] classify_cmp")


def test_cmp_engine_entropy_handler():
    """Test CMP engine handles entropy.profile.computed events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bus_dir = Path(tmpdir)
        (bus_dir / "events.ndjson").touch()

        engine = CMPEngine(bus_dir)
        assert engine.global_entropy == 0.5  # Initial

        # Simulate LENS entropy event
        event = {
            "topic": "entropy.profile.computed",
            "data": {
                "entropy_vector": {
                    "h_info": 0.8,
                    "h_miss": 0.1,
                    "h_conj": 0.05,
                    "h_alea": 0.1,
                    "h_epis": 0.1,
                    "h_struct": 0.05,
                    "c_load": 0.1,
                    "h_goal_drift": 0.05,
                    "h_mean": 0.0714  # (0.1+0.05+0.1+0.1+0.05+0.1+0.05)/7
                }
            }
        }
        engine.process_event(event)

        # Global entropy should update to h_mean
        assert abs(engine.global_entropy - 0.0714) < 0.01, f"Expected ~0.0714, got {engine.global_entropy}"
        print(f"[PASS] CMP engine entropy handler: global_entropy={engine.global_entropy:.4f}")


def test_cmp_engine_motif_handler():
    """Test CMP engine handles omega.guardian.semantic.motif_complete events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bus_dir = Path(tmpdir)
        (bus_dir / "events.ndjson").touch()

        engine = CMPEngine(bus_dir)
        initial_liveness = engine.global_liveness
        initial_motifs = engine.motif_completions

        # Simulate motif completion event
        event = {
            "topic": "omega.guardian.semantic.motif_complete",
            "actor": "omega-tracker",
            "data": {
                "motif_id": "test_motif",
                "event_actor": "claude",
                "weight": 1.0,
                "duration_s": 45.0
            }
        }
        engine.process_event(event)

        # Should increment motif count and boost liveness
        assert engine.motif_completions == initial_motifs + 1
        assert engine.global_liveness >= initial_liveness
        print(f"[PASS] CMP engine motif handler: motifs={engine.motif_completions}, liveness={engine.global_liveness:.4f}")


def test_cmp_engine_full_integration():
    """Test full integration: LENS -> CMP -> Omega flow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bus_dir = Path(tmpdir)
        (bus_dir / "events.ndjson").touch()

        engine = CMPEngine(bus_dir)

        # 1. Register a lineage
        lineage_event = {
            "topic": "lineage.created",
            "data": {
                "lineage_id": "test-lineage-001",
                "parent_lineage_id": None,
                "reward": 0.5
            }
        }
        engine.process_event(lineage_event)
        assert "test-lineage-001" in engine.lineages

        # 2. Process LENS entropy
        entropy_event = {
            "topic": "entropy.profile.computed",
            "data": {
                "entropy_vector": {
                    "h_info": 0.75,
                    "h_mean": 0.1
                }
            }
        }
        engine.process_event(entropy_event)
        assert engine.global_entropy == 0.1

        # 3. Process motif completion for actor associated with lineage
        # First, update lineage with actor info
        engine.lineages["actor.claude"] = {
            "reward": 0.6,
            "parent": None,
            "cmp_score": 0.0,
            "events_count": 0,
            "entropy_profile": {},
            "quality_functional": 0.0,
            "liveness_penalty": 0.0,
            "hgt_count": 0,
            "motif_bonus": 0.0,
            "motifs_completed": 0,
        }

        motif_event = {
            "topic": "omega.guardian.semantic.motif_complete",
            "actor": "omega-tracker",
            "data": {
                "motif_id": "inference_complete",
                "event_actor": "claude",
                "weight": 1.0,
                "duration_s": 30.0
            }
        }
        engine.process_event(motif_event)

        # Verify lineage got motif bonus
        l = engine.lineages.get("actor.claude", {})
        assert l.get("motifs_completed", 0) >= 1
        assert l.get("motif_bonus", 0) > 0

        print(f"[PASS] Full integration: lineage motif_bonus={l.get('motif_bonus', 0):.4f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CMP Integration Tests")
    print("=" * 60)

    tests = [
        test_entropy_vector_compat,
        test_entropy_vector_from_dict,
        test_compute_e_factor,
        test_compute_e_factor_dict,
        test_compute_motif_bonus,
        test_adaptive_thresholds,
        test_classify_cmp,
        test_cmp_engine_entropy_handler,
        test_cmp_engine_motif_handler,
        test_cmp_engine_full_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
