import json
import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import a2a_bridge  # noqa: E402
import dialogosd  # noqa: E402


class TestA2ABridge(unittest.TestCase):
    def test_forwards_research_request_to_dialogos(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events_path = bus_dir / "events.ndjson"
            events_path.write_text("", encoding="utf-8")

            req = {
                "id": "e1",
                "ts": 0.0,
                "iso": "1970-01-01T00:00:00Z",
                "topic": "sky.signaling.research",
                "kind": "request",
                "level": "info",
                "actor": "tester",
                "data": {"req_id": "r1", "ask": ["Do the thing"], "constraints": ["no code cowboy"]},
            }
            events_path.write_text(json.dumps(req) + "\n", encoding="utf-8")

            processed = a2a_bridge.process_events_once(
                bus_dir=bus_dir,
                actor="a2a-bridge-test",
                topics=["sky.signaling.research"],
                providers=["mock"],
                mode="llm",
            )
            self.assertEqual(processed, 1)

            # Idempotent: second pass should not re-forward.
            processed2 = a2a_bridge.process_events_once(
                bus_dir=bus_dir,
                actor="a2a-bridge-test",
                topics=["sky.signaling.research"],
                providers=["mock"],
                mode="llm",
            )
            self.assertEqual(processed2, 0)

            # Ensure dialogos.submit exists for r1.
            lines = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            submits = [l for l in lines if l.get("topic") == "dialogos.submit" and (l.get("data") or {}).get("req_id") == "r1"]
            self.assertEqual(len(submits), 1)

            # Execute it via dialogosd (mock provider).
            dialogosd.process_events_once(bus_dir=bus_dir, actor="dialogosd-test", emit_infer_sync=False)
            a2a_bridge.process_responses_once(bus_dir=bus_dir, actor="a2a-bridge-test")
            lines2 = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            topics = [l.get("topic") for l in lines2]
            self.assertIn("dialogos.cell.start", topics)
            self.assertIn("dialogos.cell.output", topics)
            self.assertIn("dialogos.cell.end", topics)

            # Ensure a2a response is emitted on the original topic.
            a2a_resps = [
                l
                for l in lines2
                if l.get("topic") == "sky.signaling.research"
                and l.get("kind") == "response"
                and (l.get("data") or {}).get("req_id") == "r1"
            ]
            self.assertEqual(len(a2a_resps), 1)


if __name__ == "__main__":
    unittest.main()
