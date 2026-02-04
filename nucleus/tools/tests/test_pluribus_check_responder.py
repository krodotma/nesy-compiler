import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class TestPluribusCheckResponder(unittest.TestCase):
    def test_responder_emits_report_on_trigger(self):
        tools_dir = Path(__file__).resolve().parents[1]
        responder = tools_dir / "pluribus_check_responder.py"
        report_tool = tools_dir / "pluribus_check.py"
        self.assertTrue(report_tool.exists(), f"pluribus_check.py not found at {report_tool}")

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            events = bus_dir / "events.ndjson"
            events.write_text("", encoding="utf-8")

            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-responder",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            proc = subprocess.Popen(
                [os.environ.get("PYTHON", "python3"), str(responder), "--bus-dir", str(bus_dir), "--run-for-s", "1.5", "--poll", "0.05", "--since-ts", "0"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Give responder time to start tailing.
            time.sleep(0.2)

            trigger = {
                "id": "t-1",
                "ts": time.time(),
                "iso": "x",
                "topic": "pluribus.check.trigger",
                "kind": "request",
                "level": "warn",
                "actor": "operator",
                "data": {"message": "test"},
            }
            with events.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trigger) + "\n")

            proc.wait(timeout=5)
            stderr = proc.stderr.read() if proc.stderr else ""

            # Confirm a report was appended.
            lines = events.read_text(encoding="utf-8", errors="replace").splitlines()
            reports = []
            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("topic") == "pluribus.check.report" and obj.get("actor") == "tester-responder":
                    reports.append(obj)
            
            if len(reports) < 1:
                self.fail(f"No reports found. Responder stderr: {stderr}")

            self.assertGreaterEqual(len(reports), 1)


if __name__ == "__main__":
    unittest.main()

