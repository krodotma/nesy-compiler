import json
import pathlib
import sys
import tempfile
import time
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import mabswarm  # noqa: E402


def _write_event(f, *, ts, topic, kind="log", level="info", actor="test", data=None):
    evt = {
        "id": "x",
        "ts": float(ts),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts))),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data or {},
    }
    f.write(json.dumps(evt) + "\n")


class TestMabSwarm(unittest.TestCase):
    def test_nudge_requires_queue_depth(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as tmp:
            bus_dir = pathlib.Path(tmp)
            events = bus_dir / "events.ndjson"
            now = time.time()
            with events.open("w", encoding="utf-8") as f:
                _write_event(f, ts=now - 2, topic="infer_sync.request", kind="request", data={"req_id": "r1"})

            metrics = mabswarm.analyze_window(bus_dir, window=60)
            actions = mabswarm.decide_action(metrics)
            self.assertTrue(any(a["type"] == "NUDGE" for a in actions), actions)

    def test_reflect_on_error_rate(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as tmp:
            bus_dir = pathlib.Path(tmp)
            events = bus_dir / "events.ndjson"
            now = time.time()
            with events.open("w", encoding="utf-8") as f:
                for i in range(20):
                    lvl = "error" if i < 3 else "info"
                    _write_event(f, ts=now - 1, topic="x.y", level=lvl)

            metrics = mabswarm.analyze_window(bus_dir, window=60)
            actions = mabswarm.decide_action(metrics)
            self.assertTrue(any(a["type"] == "REFLECT" for a in actions), actions)

    def test_backoff_on_latency(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as tmp:
            bus_dir = pathlib.Path(tmp)
            events = bus_dir / "events.ndjson"
            now = time.time()
            with events.open("w", encoding="utf-8") as f:
                for _ in range(5):
                    _write_event(f, ts=now - 1, topic="plurichat.response", data={"latency_ms": 5000})

            metrics = mabswarm.analyze_window(bus_dir, window=60)
            actions = mabswarm.decide_action(metrics)
            self.assertTrue(any(a["type"] == "BACKOFF" for a in actions), actions)


if __name__ == "__main__":
    unittest.main()

