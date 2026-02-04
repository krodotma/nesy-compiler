import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class TestPBFLUSHResponder(unittest.TestCase):
    def test_responder_emits_ack_on_trigger(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        responder = tools_dir / "pbflush_responder.py"

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
                stderr=subprocess.DEVNULL,
            )

            time.sleep(0.2)

            trigger_req_id = "req-123"
            trigger = {
                "id": "t-1",
                "ts": time.time(),
                "iso": "x",
                "topic": "operator.pbflush.request",
                "kind": "request",
                "level": "warn",
                "actor": "operator",
                "data": {"req_id": trigger_req_id, "intent": "pbflush"},
            }
            with events.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trigger) + "\n")

            proc.wait(timeout=5)

            lines = events.read_text(encoding="utf-8", errors="replace").splitlines()
            acks = []
            mirror_resps = []
            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("topic") == "operator.pbflush.ack":
                    acks.append(obj)
                if obj.get("topic") == "infer_sync.response":
                    mirror_resps.append(obj)

            self.assertGreaterEqual(len(acks), 1)
            self.assertEqual(str(acks[-1].get("data", {}).get("req_id")), trigger_req_id)
            self.assertGreaterEqual(len(mirror_resps), 1)
            self.assertEqual(str(mirror_resps[-1].get("data", {}).get("req_id")), trigger_req_id)


if __name__ == "__main__":
    unittest.main()

