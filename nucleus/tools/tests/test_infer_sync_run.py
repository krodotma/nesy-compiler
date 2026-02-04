import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


class TestInferSyncRun(unittest.TestCase):
    def test_emits_start_and_end(self):
        tool = pathlib.Path(__file__).resolve().parents[1] / "infer_sync_run.py"
        self.assertTrue(tool.exists())

        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PLURIBUS_ACTOR": "test-infer-sync-run"}
            cmd = [
                sys.executable,
                str(tool),
                "--subproject",
                "infercell",
                "--",
                sys.executable,
                "-c",
                "print('ok')",
            ]
            p = subprocess.run(cmd, check=False, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

            lines = (bus_dir / "events.ndjson").read_text(encoding="utf-8").splitlines()
            topics = [json.loads(l).get("topic") for l in lines if l.strip()]
            self.assertIn("infer_sync.test.start", topics)
            self.assertIn("infer_sync.test.end", topics)
            self.assertIn("infer_sync.checkin", topics)


if __name__ == "__main__":
    unittest.main()

