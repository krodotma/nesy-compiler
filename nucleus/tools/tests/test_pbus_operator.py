import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestPBUSOperator(unittest.TestCase):
    def test_pbus_operator_emits_request_and_report(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        tool = tools_dir / "pbus_operator.py"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bus_dir = tmp / "bus"
            report_dir = tmp / "reports"

            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-pbus",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            p = subprocess.run(
                [
                    os.environ.get("PYTHON", "python3"),
                    str(tool),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "tester-pbus",
                    "--report-dir",
                    str(report_dir),
                    "--window-s",
                    "60",
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
            self.assertTrue(events_path.exists())
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            topics = [e.get("topic") for e in events]
            self.assertIn("operator.pbus.request", topics)
            self.assertIn("infer_sync.request", topics)
            self.assertIn("operator.pbus.report", topics)

            reports = [e for e in events if e.get("topic") == "operator.pbus.report"]
            self.assertGreaterEqual(len(reports), 1)
            report_data = reports[-1]["data"]
            self.assertEqual(report_data.get("req_id"), req_id)

            report_path = Path(report_data.get("report_path") or "")
            if not report_path.is_absolute():
                report_path = (tools_dir.parents[2] / report_path).resolve()
            self.assertTrue(report_path.exists())

            text = report_path.read_text(encoding="utf-8", errors="replace")
            self.assertIn("PBUS_REPORT v1", text)


if __name__ == "__main__":
    unittest.main()
