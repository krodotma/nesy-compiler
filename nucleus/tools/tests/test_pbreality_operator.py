import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestPBREALITYOperator(unittest.TestCase):
    def test_pbreality_operator_emits_request_and_report(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        tool = tools_dir / "pbreality_operator.py"

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td) / "bus"
            report_dir = Path(td) / "reports"
            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-pbreality",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            p = subprocess.run(
                [
                    os.environ.get("PYTHON", "python3"),
                    str(tool),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "tester-pbreality",
                    "--report-dir",
                    str(report_dir),
                    "--scope",
                    "test",
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
            self.assertIn("operator.pbreality.request", topics)
            self.assertIn("operator.pbreality.report", topics)
            self.assertIn("infer_sync.request", topics)

            reports = [e for e in events if e.get("topic") == "operator.pbreality.report"]
            self.assertGreaterEqual(len(reports), 1)
            report_data = reports[-1]["data"]
            self.assertEqual(report_data.get("req_id"), req_id)

            report_path = Path(report_data.get("report_path") or "")
            if not report_path.is_absolute():
                report_path = (tools_dir.parents[2] / report_path).resolve()
            self.assertTrue(report_path.exists())

            text = report_path.read_text(encoding="utf-8", errors="replace")
            self.assertIn("PBREALITY_REPORT v1", text)


if __name__ == "__main__":
    unittest.main()
