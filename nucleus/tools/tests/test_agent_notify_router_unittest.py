#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from nucleus.tools import agent_notify_router as router


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def test_notify_router_mirrors_to_hexis():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bus_dir = tmp_path / "bus"
        buf_dir = tmp_path / "buffers"
        bus_dir.mkdir(parents=True, exist_ok=True)
        buf_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        os.environ["PLURIBUS_BUS_DIR"] = str(bus_dir)
        os.environ["HEXIS_BUFFER_DIR"] = str(buf_dir)
        os.environ["PLURIBUS_ACTOR"] = "router-test"

        try:
            event = {
                "id": "evt-1",
                "ts": 0,
                "iso": "2025-01-01T00:00:00Z",
                "topic": "agent.notify.request",
                "kind": "request",
                "actor": "tester",
                "data": {
                    "req_id": "req-1",
                    "trace_id": "trace-1",
                    "message": "Ping from router test",
                    "severity": "info",
                    "targets": ["gemini"],
                    "broadcast": False,
                    "hexis_delivery": "auto",
                    "payload": {"source": "test"},
                },
            }
            events_path = bus_dir / "events.ndjson"
            events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

            processed = router.run_once(
                bus_dir=bus_dir,
                actor="router-test",
                topics=router.DEFAULT_TOPICS,
                class_map={},
                hexis_topic=router.DEFAULT_HEXIS_TOPIC,
                hexis_effects=router.DEFAULT_EFFECTS,
                hexis_lane=router.DEFAULT_LANE,
                hexis_topology=router.DEFAULT_TOPOLOGY,
                hexis_agent_type="worker",
                bootstrap_bytes=router.DEFAULT_BOOTSTRAP_BYTES,
            )
            assert processed == 1

            buffer_path = buf_dir / "gemini.buffer"
            assert buffer_path.exists()
            lines = [line for line in buffer_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert lines
            msg = json.loads(lines[-1])
            assert msg["payload"]["message"] == "Ping from router test"

            events = _read_events(events_path)
            dispatch = [e for e in events if e.get("topic") == "agent.notify.dispatch"]
            assert dispatch, "agent.notify.dispatch not emitted"
        finally:
            os.environ.clear()
            os.environ.update(env)
