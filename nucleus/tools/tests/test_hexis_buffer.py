#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
HEXIS = TOOLS_DIR / "hexis_buffer.py"


def run_hexis(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(HEXIS), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


class TestHexisBuffer(unittest.TestCase):
    def test_pub_pull_ack_mirrors_to_bus(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            buf_dir = tmp_path / "buffers"
            bus_dir = tmp_path / "bus"
            buf_dir.mkdir(parents=True, exist_ok=True)
            bus_dir.mkdir(parents=True, exist_ok=True)

            env = {
                **os.environ,
                "HEXIS_BUFFER_DIR": str(buf_dir),
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_ACTOR": "test",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            pub = run_hexis(
                env,
                "pub",
                "--agent",
                "unit",
                "--topic",
                "infercell.uiux.design.request",
                "--kind",
                "request",
                "--effects",
                "none",
                "--lane",
                "pbpair",
                "--topology",
                "star",
                "--json",
                '{"hello":"world"}',
            )
            self.assertEqual(pub.returncode, 0, pub.stderr)
            msg_id = pub.stdout.strip()
            self.assertTrue(msg_id)

            status = run_hexis(env, "status", "--agent", "unit")
            self.assertEqual(status.returncode, 0, status.stderr)
            status_obj = json.loads(status.stdout)
            self.assertEqual(status_obj["unit"]["pending"], 1)

            pull = run_hexis(env, "pull", "--agent", "unit", "--max", "1")
            self.assertEqual(pull.returncode, 0, pull.stderr)
            pulled = json.loads(pull.stdout.splitlines()[0])
            self.assertEqual(pulled["msg_id"], msg_id)
            self.assertEqual(pulled["topic"], "infercell.uiux.design.request")
            self.assertIn("req_id", pulled)
            self.assertIn("trace_id", pulled)

            ack = run_hexis(env, "ack", "--agent", "unit", "--msg-id", msg_id)
            self.assertEqual(ack.returncode, 0, ack.stderr)

            status2 = run_hexis(env, "status", "--agent", "unit")
            self.assertEqual(status2.returncode, 0, status2.stderr)
            status2_obj = json.loads(status2.stdout)
            # File may remain present but should be empty (pending=0) after ack.
            if status2_obj.get("unit") is not None:
                self.assertEqual(status2_obj["unit"]["pending"], 0)

            events_path = bus_dir / "events.ndjson"
            self.assertTrue(events_path.exists())
            lines = [l for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            consumed = [json.loads(l) for l in lines if '"topic": "hexis.buffer.consumed"' in l]
            self.assertGreaterEqual(len(consumed), 1)
            last = consumed[-1]["data"]
            self.assertEqual(last["msg_id"], msg_id)
            self.assertEqual(last["agent"], "unit")
            self.assertEqual(last["topic"], "infercell.uiux.design.request")
            self.assertEqual(last["effects"], "none")
            self.assertEqual(last["lane"], "pbpair")
            self.assertEqual(last["topology"], "star")
            self.assertTrue(last["req_id"])
            self.assertTrue(last["trace_id"])


if __name__ == "__main__":
    unittest.main()
