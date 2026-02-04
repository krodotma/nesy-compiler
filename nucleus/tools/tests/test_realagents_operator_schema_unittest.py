import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


def _read_events(bus_dir: pathlib.Path) -> list[dict]:
    p = bus_dir / "events.ndjson"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


class TestRealagentsOperatorSchema(unittest.TestCase):
    def test_realagents_operator_emits_schema_like_payload(self):
        tool = pathlib.Path(__file__).resolve().parents[1] / "realagents_operator.py"
        self.assertTrue(tool.exists())

        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            bus_dir = pathlib.Path(td)
            p = subprocess.run(
                [sys.executable, str(tool), "--bus-dir", str(bus_dir), "--intent", "Deepen MCP/A2A/ADK", "--targets", "codex"],
                check=False,
                capture_output=True,
                text=True,
                env={**dict(**__import__("os").environ), "PYTHONDONTWRITEBYTECODE": "1"},
                timeout=10.0,
            )
            self.assertEqual(p.returncode, 0, msg=(p.stderr or p.stdout))
            rid = (p.stdout or "").strip()
            self.assertTrue(rid)

            events = _read_events(bus_dir)
            dispatch = None
            for e in events:
                if e.get("topic") == "rd.tasks.dispatch" and e.get("kind") == "request":
                    d = e.get("data")
                    if isinstance(d, dict) and d.get("req_id") == rid:
                        dispatch = d
                        break
            self.assertIsNotNone(dispatch)
            # Required fields per rd_tasks_dispatch.schema.json
            for key in ("req_id", "task_id", "intent", "iso", "spec_ref", "targets", "tasks"):
                self.assertIn(key, dispatch)


if __name__ == "__main__":
    unittest.main()

