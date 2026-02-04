import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class TestSemopsInvokeResponder(unittest.TestCase):
    def test_responder_emits_response_on_invoke_request(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        responder = tools_dir / "semops_invoke_responder.py"

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            events = bus_dir / "events.ndjson"
            events.write_text("", encoding="utf-8")

            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-semops",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            proc = subprocess.Popen(
                [
                    os.environ.get("PYTHON", "python3"),
                    str(responder),
                    "--bus-dir",
                    str(bus_dir),
                    "--run-for-s",
                    "1.5",
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

            req_id = "req-semops-1"
            trigger = {
                "id": "t-1",
                "ts": time.time(),
                "iso": "x",
                "topic": "semops.invoke.request",
                "kind": "request",
                "level": "info",
                "actor": "dashboard",
                "data": {
                    "req_id": req_id,
                    "operator_key": "MYOP",
                    "mode": "auto",
                    "effects": "file",
                    "operator": {"name": "MYOP", "domain": "execution", "category": "tool", "tool": "nucleus/tools/myop_operator.py", "effects": "file"},
                    "payload": {"input": "x"},
                },
            }
            with events.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trigger) + "\n")

            proc.wait(timeout=5)

            lines = events.read_text(encoding="utf-8", errors="replace").splitlines()
            responses = []
            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("topic") != "semops.invoke.response":
                    continue
                data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
                if str(data.get("req_id") or "") != req_id:
                    continue
                responses.append(obj)

            self.assertGreaterEqual(len(responses), 1)
            data = responses[-1].get("data", {})
            self.assertEqual(str(data.get("operator_key")), "MYOP")
            self.assertEqual(str(data.get("effects")), "file")
            self.assertIn(str(data.get("mode_resolved")), {"tool", "bus", "policy", "evolution", "ui", "auto"})


if __name__ == "__main__":
    unittest.main()

