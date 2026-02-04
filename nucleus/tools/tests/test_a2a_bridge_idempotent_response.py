import json
import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import a2a_bridge  # noqa: E402
import dialogosd  # noqa: E402


class TestA2ABridgeIdempotentResponse(unittest.TestCase):
    def test_response_emitted_once(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events_path = bus_dir / "events.ndjson"
            events_path.write_text("", encoding="utf-8")

            req = {
                "id": "e1",
                "ts": 0.0,
                "iso": "1970-01-01T00:00:00Z",
                "topic": "lens.collimator.router.research",
                "kind": "request",
                "level": "info",
                "actor": "tester",
                "data": {"req_id": "rX", "ask": ["Plan"], "deliverable": {"format": "json"}},
            }
            events_path.write_text(json.dumps(req) + "\n", encoding="utf-8")

            _ = a2a_bridge.process_events_once(
                bus_dir=bus_dir,
                actor="a2a-bridge-test",
                topics=["lens.collimator.router.research"],
                providers=["mock"],
                mode="llm",
            )
            dialogosd.process_events_once(bus_dir=bus_dir, actor="dialogosd-test", emit_infer_sync=False)

            n1 = a2a_bridge.process_responses_once(bus_dir=bus_dir, actor="a2a-bridge-test")
            n2 = a2a_bridge.process_responses_once(bus_dir=bus_dir, actor="a2a-bridge-test")
            self.assertEqual(n1, 1)
            self.assertEqual(n2, 0)

            lines = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            resps = [
                l
                for l in lines
                if l.get("topic") == "lens.collimator.router.research"
                and l.get("kind") == "response"
                and (l.get("data") or {}).get("req_id") == "rX"
            ]
            self.assertEqual(len(resps), 1)


if __name__ == "__main__":
    unittest.main()

