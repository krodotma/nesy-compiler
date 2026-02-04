import json
import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import omega_heartbeat  # noqa: E402


class TestOmegaHeartbeatA2A(unittest.TestCase):
    def test_counts_pending_a2a_by_topic(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events_path = bus_dir / "events.ndjson"
            events_path.write_text("", encoding="utf-8")

            def append(obj):
                with events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(obj) + "\n")

            append({"topic": "sky.signaling.research", "kind": "request", "data": {"req_id": "r1"}})
            append({"topic": "sky.signaling.research", "kind": "request", "data": {"req_id": "r2"}})
            append({"topic": "sky.signaling.research", "kind": "response", "data": {"req_id": "r2"}})
            append({"topic": "lens.collimator.router.research", "kind": "request", "data": {"req_id": "r3"}})
            append({"topic": "other.topic", "kind": "request", "data": {"req_id": "r4"}})

            out = omega_heartbeat.count_pending_a2a(
                str(bus_dir),
                topics=["sky.signaling.research", "lens.collimator.router.research"],
            )
            self.assertEqual(out["total"], 2)
            self.assertEqual(out["by_topic"]["sky.signaling.research"], 1)  # r1 pending
            self.assertEqual(out["by_topic"]["lens.collimator.router.research"], 1)  # r3 pending


if __name__ == "__main__":
    unittest.main()

