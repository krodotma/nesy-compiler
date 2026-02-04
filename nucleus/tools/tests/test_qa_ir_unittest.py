import json

from nucleus.tools import qa_ir


def test_normalize_event_minimum_fields():
    raw = {
        "id": "evt-1",
        "ts": 1.0,
        "iso": "2025-01-01T00:00:00Z",
        "topic": "dashboard.test",
        "actor": "tester",
        "kind": "log",
        "level": "error",
        "data": {"msg": "boom"},
        "trace_id": "trace-1",
        "run_id": "run-1",
    }
    ir = qa_ir.normalize_event(raw, source="bus")
    for key in ("id", "ts", "iso", "source", "actor", "topic", "kind", "level"):
        assert key in ir
    assert ir["source"] == "bus"
    assert ir["severity"] >= 3
    assert 0 <= ir["entropy_score"] <= 1
    assert len(ir["fingerprint"]) == 64


def test_entropy_score_handles_none():
    assert qa_ir.compute_entropy_score(None) == 0.0


def test_fingerprint_is_stable():
    fp1 = qa_ir.fingerprint_event("topic", "actor", {"a": 1, "b": 2})
    fp2 = qa_ir.fingerprint_event("topic", "actor", {"b": 2, "a": 1})
    assert fp1 == fp2

