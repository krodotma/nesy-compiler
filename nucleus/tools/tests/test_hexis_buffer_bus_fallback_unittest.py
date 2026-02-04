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


class TestHexisBufferBusFallback(unittest.TestCase):
    def test_ack_uses_fallback_bus_when_primary_not_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            buf_dir = tmp_path / "buffers"
            # Force the primary bus dir candidate to be non-usable by making it a file, not a directory.
            primary_bus_dir = tmp_path / "primary_bus_file"
            fallback_bus_dir = tmp_path / "fallback_bus"
            buf_dir.mkdir(parents=True, exist_ok=True)
            fallback_bus_dir.mkdir(parents=True, exist_ok=True)
            primary_bus_dir.write_text("not a directory", encoding="utf-8")

            env = {
                **os.environ,
                "HEXIS_BUFFER_DIR": str(buf_dir),
                "PLURIBUS_BUS_DIR": str(primary_bus_dir),
                "PLURIBUS_FALLBACK_BUS_DIR": str(fallback_bus_dir),
                "PLURIBUS_ACTOR": "test",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            pub = run_hexis(
                env,
                "pub",
                "--agent",
                "unit",
                "--topic",
                "infercell.bus.fallback.request",
                "--kind",
                "request",
                "--effects",
                "none",
                "--lane",
                "pbpair",
                "--topology",
                "single",
                "--json",
                '{"hello":"fallback"}',
            )
            self.assertEqual(pub.returncode, 0, pub.stderr)
            msg_id = pub.stdout.strip()
            self.assertTrue(msg_id)

            ack = run_hexis(env, "ack", "--agent", "unit", "--msg-id", msg_id)
            self.assertEqual(ack.returncode, 0, ack.stderr)

            # Primary should remain empty (or absent); fallback must contain hexis.buffer.consumed.
            primary_events = primary_bus_dir / "events.ndjson"
            fallback_events = fallback_bus_dir / "events.ndjson"
            self.assertTrue(fallback_events.exists())
            lines = [l for l in fallback_events.read_text(encoding="utf-8").splitlines() if l.strip()]
            consumed = [json.loads(l) for l in lines if '"topic": "hexis.buffer.consumed"' in l]
            self.assertGreaterEqual(len(consumed), 1)
            last = consumed[-1]["data"]
            self.assertEqual(last["msg_id"], msg_id)
            self.assertEqual(last["agent"], "unit")
            self.assertEqual(last["topic"], "infercell.bus.fallback.request")

            if primary_events.exists():
                prim_lines = [l for l in primary_events.read_text(encoding="utf-8").splitlines() if l.strip()]
                self.assertFalse(any(msg_id in l for l in prim_lines))


if __name__ == "__main__":
    unittest.main()
