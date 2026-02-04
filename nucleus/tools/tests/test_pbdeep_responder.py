import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class TestPBDEEPResponder(unittest.TestCase):
    def test_responder_emits_report_on_trigger(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        responder = tools_dir / "pbdeep_responder.py"

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events = bus_dir / "events.ndjson"
            events.write_text("", encoding="utf-8")

            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-pbdeep",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            proc = subprocess.Popen(
                [
                    os.environ.get("PYTHON", "python3"),
                    str(responder),
                    "--bus-dir",
                    str(bus_dir),
                    "--root",
                    td,
                    "--scan-mode",
                    "noop",
                    "--run-for-s",
                    "1.2",
                    "--poll",
                    "0.05",
                    "--since-ts",
                    "0",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            time.sleep(0.2)

            trigger_req_id = "pbdeep-req-1"
            trigger = {
                "id": "t-1",
                "ts": time.time(),
                "iso": "x",
                "topic": "operator.pbdeep.request",
                "kind": "request",
                "level": "info",
                "actor": "operator",
                "data": {"req_id": trigger_req_id, "instruction": "noop", "intent": "pbdeep"},
            }
            with events.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trigger) + "\n")

            proc.wait(timeout=5)

            lines = events.read_text(encoding="utf-8", errors="replace").splitlines()
            reports = []
            progress = []
            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("topic") == "operator.pbdeep.report":
                    reports.append(obj)
                if obj.get("topic") == "operator.pbdeep.progress":
                    progress.append(obj)

            self.assertGreaterEqual(len(reports), 1)
            self.assertEqual(str(reports[-1].get("data", {}).get("req_id")), trigger_req_id)
            self.assertGreaterEqual(len(progress), 1)


if __name__ == "__main__":
    unittest.main()
