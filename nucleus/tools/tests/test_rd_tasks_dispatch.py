import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class TestRdTasksDispatch(unittest.TestCase):
    def test_dispatch_emits_request_and_infer_sync(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        tool = tools_dir / "rd_tasks_dispatch.py"

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-rd",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            p = subprocess.run(
                [os.environ.get("PYTHON", "python3"), str(tool), "--bus-dir", str(bus_dir), "--actor", "tester-rd", "--targets", "claude,codex"],
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
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            topics = [e.get("topic") for e in events]
            self.assertIn("rd.tasks.dispatch", topics)
            self.assertIn("infer_sync.request", topics)

            rd_evt = [e for e in events if e.get("topic") == "rd.tasks.dispatch"][-1]
            self.assertEqual(rd_evt.get("kind"), "request")
            payload = rd_evt.get("data") or {}
            self.assertEqual(payload.get("req_id"), req_id)
            self.assertEqual(payload.get("task_id"), "REALAGENTS_upgrade")

    def test_responder_acks_dispatch(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        dispatch = tools_dir / "rd_tasks_dispatch.py"
        responder = tools_dir / "rd_tasks_responder.py"

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

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

            time.sleep(0.15)
            p = subprocess.run(
                [os.environ.get("PYTHON", "python3"), str(dispatch), "--bus-dir", str(bus_dir), "--actor", "tester-rd", "--targets", "tester-responder"],
                env={**env, "PLURIBUS_ACTOR": "tester-rd"},
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            req_id = (p.stdout or "").strip()

            proc.wait(timeout=5)
            stderr = proc.stderr.read() if proc.stderr else ""
            if proc.returncode != 0:
                print(f"Responder stderr: {stderr}")

            lines = (bus_dir / "events.ndjson").read_text(encoding="utf-8", errors="replace").splitlines()
            acks = []
            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("topic") == "rd.tasks.ack":
                    acks.append(obj)
            
            if len(acks) < 1:
                self.fail(f"No acks found. Responder stderr: {stderr}")
            
            self.assertGreaterEqual(len(acks), 1)
            self.assertEqual(str(acks[-1].get("data", {}).get("req_id")), req_id)


if __name__ == "__main__":
    unittest.main()

