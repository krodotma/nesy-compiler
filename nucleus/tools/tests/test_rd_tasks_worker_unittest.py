import json
import tempfile
import time
import unittest
from pathlib import Path


def _read_events(bus_dir: Path) -> list[dict]:
    p = bus_dir / "events.ndjson"
    if not p.exists():
        return []
    out: list[dict] = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


class TestRDTasksWorker(unittest.TestCase):
    def test_handle_dispatch_emits_done(self):
        from nucleus.tools import rd_tasks_worker

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            trig = {
                "id": "evt-1",
                "ts": time.time(),
                "iso": "2025-12-15T00:00:00Z",
                "topic": "rd.tasks.dispatch",
                "kind": "request",
                "level": "info",
                "actor": "tester",
                "data": {
                    "req_id": "req-1",
                    "targets": ["rd-tasks-worker"],
                    "constraints": {"execute": True},
                    "tasks": [
                        {
                            "id": "T0",
                            "title": "smoke",
                            "depends_on": [],
                            "deliverables": ["noop"],
                            "acceptance": ["noop"],
                            "actions": [
                                {
                                    "id": "A0",
                                    "kind": "python",
                                    "argv": ["python3", "nucleus/tools/pbeport.py", "--help"],
                                    "cwd": "/pluribus",
                                    "timeout_s": 10,
                                }
                            ],
                        }
                    ],
                },
            }

            processed = rd_tasks_worker.handle_dispatch(bus_dir=str(bus_dir), actor="rd-tasks-worker", trig=trig)
            self.assertTrue(processed)

            events = _read_events(bus_dir)
            topics = [e.get("topic") for e in events]
            self.assertIn("rd.tasks.done", topics)
            done = [e for e in events if e.get("topic") == "rd.tasks.done"][-1]
            self.assertEqual((done.get("data") or {}).get("req_id"), "req-1")

