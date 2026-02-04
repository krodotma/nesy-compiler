import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestPBFLUSHOperator(unittest.TestCase):
    def test_pbflush_operator_emits_request_and_infer_sync(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        tool = tools_dir / "pbflush_operator.py"

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-pbflush",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            p = subprocess.run(
                [os.environ.get("PYTHON", "python3"), str(tool), "--bus-dir", str(bus_dir), "--actor", "tester-pbflush", "--message", "test"],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            req_id = (p.stdout or "").strip()
            self.assertTrue(req_id)

            events_path = bus_dir / "events.ndjson"
            self.assertTrue(events_path.exists())
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            topics = [e.get("topic") for e in events]
            self.assertIn("operator.pbflush.request", topics)
            self.assertIn("infer_sync.request", topics)

            reqs = [e for e in events if e.get("topic") == "operator.pbflush.request"]
            mirrors = [e for e in events if e.get("topic") == "infer_sync.request"]
            self.assertGreaterEqual(len(reqs), 1)
            self.assertGreaterEqual(len(mirrors), 1)

            self.assertEqual(reqs[-1]["data"]["req_id"], req_id)
            self.assertEqual(reqs[-1]["data"]["intent"], "pbflush")
            self.assertEqual(mirrors[-1]["data"]["req_id"], req_id)
            self.assertEqual(mirrors[-1]["data"]["intent"], "pbflush")


if __name__ == "__main__":
    unittest.main()

