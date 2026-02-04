import json
import pathlib
import sys
import tempfile
import time
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import mbad  # noqa: E402


class TestMbad(unittest.TestCase):
    def test_snapshot_includes_hexis_pending(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as tmp_bus, tempfile.TemporaryDirectory(prefix="pluribus_hexis_") as tmp_hex:
            bus_dir = pathlib.Path(tmp_bus)
            hexis_dir = pathlib.Path(tmp_hex)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")
            (hexis_dir / "gemini.buffer").write_text('{"iso":"x","topic":"a"}\n{"iso":"y","topic":"b"}\n', encoding="utf-8")

            snap = mbad.build_snapshot(bus_dir=bus_dir, window_s=900, width=24, hexis_dir=hexis_dir)
            self.assertEqual(snap["hexis_pending_total"], 2)
            self.assertIn("gemini", snap["hexis_by_agent"])

    def test_snapshot_counts_recent_events(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as tmp_bus, tempfile.TemporaryDirectory(prefix="pluribus_hexis_") as tmp_hex:
            bus_dir = pathlib.Path(tmp_bus)
            hexis_dir = pathlib.Path(tmp_hex)
            now = time.time()
            (hexis_dir / "codex.buffer").write_text("", encoding="utf-8")
            events = [
                {"id": "1", "ts": now - 1, "iso": "x", "topic": "a.b", "kind": "metric", "level": "info", "actor": "t", "data": {}},
                {"id": "2", "ts": now - 2, "iso": "x", "topic": "infer_sync.request", "kind": "request", "level": "info", "actor": "t", "data": {"req_id": "r1"}},
            ]
            (bus_dir / "events.ndjson").write_text("\n".join([json.dumps(e) for e in events]) + "\n", encoding="utf-8")

            snap = mbad.build_snapshot(bus_dir=bus_dir, window_s=60, width=12, hexis_dir=hexis_dir)
            self.assertGreaterEqual(snap["events"], 2)
            self.assertEqual(snap["infer_sync_pending"], 1)


if __name__ == "__main__":
    unittest.main()
