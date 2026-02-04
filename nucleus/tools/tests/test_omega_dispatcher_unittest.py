import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "omega_dispatcher.py"


class TestOmegaDispatcher(unittest.TestCase):
    def test_emit_on_empty_produces_tick(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td) / "bus"
            bus_dir.mkdir()
            state_path = Path(td) / "state.json"

            cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                "--bus-dir",
                str(bus_dir),
                "--state-file",
                str(state_path),
                "--run-for-s",
                "0.2",
                "--tick-s",
                "0.01",
                "--poll",
                "0.01",
                "--emit-on-empty",
            ]
            subprocess.run(cmd, check=True, timeout=3)

            events_path = bus_dir / "events.ndjson"
            self.assertTrue(events_path.exists())
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            topics = {event.get("topic") for event in events}
            self.assertIn("omega.dispatcher.ready", topics)
            self.assertIn("omega.dispatch.tick", topics)

            ticks = [event for event in events if event.get("topic") == "omega.dispatch.tick"]
            self.assertTrue(ticks)
            self.assertEqual(ticks[0].get("data", {}).get("pending_total"), 0)


if __name__ == "__main__":
    unittest.main()
