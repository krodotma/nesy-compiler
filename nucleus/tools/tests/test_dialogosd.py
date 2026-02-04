import json
import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import dialogosd  # noqa: E402


class TestDialogosD(unittest.TestCase):
    def test_llm_mode_mock_emits_cells(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events_path = bus_dir / "events.ndjson"
            events_path.write_text("", encoding="utf-8")

            submit = {
                "id": "e1",
                "ts": 0.0,
                "iso": "1970-01-01T00:00:00Z",
                "topic": "dialogos.submit",
                "kind": "request",
                "level": "info",
                "actor": "test",
                "data": {"req_id": "r1", "mode": "llm", "providers": ["mock"], "prompt": "hello"},
            }
            events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

            processed = dialogosd.process_events_once(bus_dir=bus_dir, actor="dialogosd-test")
            self.assertEqual(processed, 1)

            lines = events_path.read_text(encoding="utf-8").splitlines()
            topics = [json.loads(l).get("topic") for l in lines if l.strip()]
            self.assertIn("dialogos.cell.start", topics)
            self.assertIn("dialogos.cell.output", topics)
            self.assertIn("dialogos.cell.end", topics)

            outputs = [json.loads(l) for l in lines if json.loads(l).get("topic") == "dialogos.cell.output"]
            self.assertTrue(any("hello" in json.dumps(o) for o in outputs))

    def test_multi_provider_emits_multiple_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events_path = bus_dir / "events.ndjson"
            events_path.write_text("", encoding="utf-8")

            submit = {
                "id": "e1",
                "ts": 0.0,
                "iso": "1970-01-01T00:00:00Z",
                "topic": "dialogos.submit",
                "kind": "request",
                "level": "info",
                "actor": "test",
                "data": {"req_id": "r2", "mode": "llm", "providers": ["mock", "mock"], "prompt": "hi"},
            }
            events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

            _ = dialogosd.process_events_once(bus_dir=bus_dir, actor="dialogosd-test")

            lines = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            outputs = [l for l in lines if l.get("topic") == "dialogos.cell.output" and (l.get("data") or {}).get("req_id") == "r2"]
            self.assertGreaterEqual(len(outputs), 2)


if __name__ == "__main__":
    unittest.main()

