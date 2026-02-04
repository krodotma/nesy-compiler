import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


class TestPBOSNAPOperator(unittest.TestCase):
    def test_pbosnap_omega_sandwich_section(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        tool = tools_dir / "pbosnap_operator.py"

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            bus_dir = base / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            lanes_path = base / "lanes.json"
            lanes_path.write_text(json.dumps({"lanes": [], "agents": []}), encoding="utf-8")

            now = time.time()
            events = [
                {
                    "ts": now - 5,
                    "iso": "2025-12-26T00:00:05Z",
                    "topic": "omega.heartbeat",
                    "kind": "metric",
                    "level": "info",
                    "actor": "omega",
                    "data": {"cycle": 5, "uptime_s": 50},
                },
                {
                    "ts": now - 4,
                    "iso": "2025-12-26T00:00:06Z",
                    "topic": "omega.queue.depth",
                    "kind": "metric",
                    "level": "info",
                    "actor": "omega",
                    "data": {"pending_requests": 3, "total_events": 123},
                },
                {
                    "ts": now - 3,
                    "iso": "2025-12-26T00:00:07Z",
                    "topic": "omega.pending.pairs",
                    "kind": "metric",
                    "level": "info",
                    "actor": "omega",
                    "data": {
                        "total": 1,
                        "by_pair": {"pluribuscheck": {"pending": 1, "oldest_age_s": 12.0}},
                    },
                },
                {
                    "ts": now - 2,
                    "iso": "2025-12-26T00:00:08Z",
                    "topic": "omega.providers.scan",
                    "kind": "metric",
                    "level": "info",
                    "actor": "omega",
                    "data": {"providers": {"claude": True, "codex": False}},
                },
                {
                    "ts": now - 1,
                    "iso": "2025-12-26T00:00:09Z",
                    "topic": "strp.worker.start",
                    "kind": "metric",
                    "level": "info",
                    "actor": "tester",
                    "data": {},
                },
                {
                    "ts": now,
                    "iso": "2025-12-26T00:00:10Z",
                    "topic": "test.error",
                    "kind": "metric",
                    "level": "error",
                    "actor": "tester",
                    "data": {},
                },
            ]

            events_path = bus_dir / "events.ndjson"
            events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "tester-pbosnap",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            p = subprocess.run(
                [
                    os.environ.get("PYTHON", "python3"),
                    str(tool),
                    "--bus-dir",
                    str(bus_dir),
                    "--lanes-path",
                    str(lanes_path),
                    "--window",
                    "3600",
                    "--max-bytes",
                    "1000000",
                    "--max-lines",
                    "1000",
                    "--limit",
                    "5",
                    "--no-report",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            out = p.stdout or ""
            self.assertIn("OMEGA SANDWICH", out)
            self.assertIn("cycle=5", out)
            self.assertIn("uptime_s=50", out)
            self.assertIn("pending_requests=3", out)
            self.assertIn("total_events=123", out)
            self.assertIn("pluribuscheck=1", out)
            self.assertIn("providers: claude", out)
            self.assertIn("last_error: test.error", out)


if __name__ == "__main__":
    unittest.main()
