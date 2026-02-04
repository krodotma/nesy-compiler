import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestPbassimilateOperator(unittest.TestCase):
    def test_operator_emits_request_and_screening(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        tool = tools_dir / "pbassimilate_operator.py"

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-pbassimilate",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            p = subprocess.run(
                [
                    os.environ.get("PYTHON", "python3"),
                    str(tool),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "tester-pbassimilate",
                    "--target",
                    "https://github.com/example/pbassimilate-demo.git",
                    "--purpose",
                    "Test screening output",
                ],
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
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            topics = [e.get("topic") for e in events]
            self.assertIn("operator.pbassimilate.request", topics)
            self.assertIn("operator.pbassimilate.screening", topics)
            self.assertIn("infer_sync.request", topics)

            req_evt = [e for e in events if e.get("topic") == "operator.pbassimilate.request"][-1]
            payload = req_evt.get("data") or {}
            self.assertEqual(payload.get("req_id"), req_id)
            target = payload.get("target") or {}
            self.assertEqual(target.get("kind"), "git")
            self.assertTrue(target.get("name"))
            screening = payload.get("screening") or {}
            self.assertIn(screening.get("recommendation"), {"proceed", "review", "reject"})


if __name__ == "__main__":
    unittest.main()
