import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


class TestPluriChatIterateOperator(unittest.TestCase):
    def test_oneshot_iterate_emits_bus_requests(self):
        tool = pathlib.Path(__file__).resolve().parents[1] / "plurichat.py"
        self.assertTrue(tool.exists())

        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PLURIBUS_ACTOR": "test-actor"}
            p = subprocess.run(
                [sys.executable, str(tool), "--mode", "oneshot", "--ask", "iterate", "--json-output"],
                check=False,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20.0,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            out = json.loads(p.stdout.strip())
            self.assertEqual(out.get("operator"), "iterate")
            self.assertTrue(out.get("req_id"))

            events = [json.loads(l) for l in (bus_dir / "events.ndjson").read_text(encoding="utf-8").splitlines() if l.strip()]
            topics = [e.get("topic") for e in events]
            self.assertIn("infer_sync.request", topics)
            self.assertIn("operator.iterate.request", topics)


if __name__ == "__main__":
    unittest.main()

