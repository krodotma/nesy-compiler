#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import bus_mirror_daemon  # noqa: E402


def mk_event(event_id: str, topic: str) -> bytes:
    obj = {"id": event_id, "ts": 0, "iso": "2025-01-01T00:00:00Z", "topic": topic, "kind": "log", "level": "info", "actor": "t", "data": {}}
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


class TestBusMirrorDaemon(unittest.TestCase):
    def test_mirror_once_appends_new_lines_and_skips_recent_dupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_bus = tmp_path / "src"
            dst_bus = tmp_path / "dst"
            src_bus.mkdir(parents=True, exist_ok=True)
            dst_bus.mkdir(parents=True, exist_ok=True)
            src_events = src_bus / "events.ndjson"
            dst_events = dst_bus / "events.ndjson"

            # Destination already contains event "a".
            dst_events.write_bytes(mk_event("a", "existing"))

            # Source contains "a" (dup) then "b" (new).
            src_events.write_bytes(mk_event("a", "dup") + mk_event("b", "new"))

            state = bus_mirror_daemon.MirrorState(offset=0, src_inode=None)
            state, stats = bus_mirror_daemon.mirror_once(
                src_events=src_events,
                dest_events=dst_events,
                state=state,
                recent_bytes_back=1024 * 1024,
                max_recent_ids=10_000,
            )

            self.assertEqual(stats["mirrored"], 1)
            self.assertGreaterEqual(stats["skipped"], 1)
            self.assertEqual(state.offset, src_events.stat().st_size)

            lines = [l for l in dst_events.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 2)
            ids = [json.loads(l)["id"] for l in lines]
            self.assertEqual(ids, ["a", "b"])

            # Second run should be a no-op.
            state, stats = bus_mirror_daemon.mirror_once(
                src_events=src_events,
                dest_events=dst_events,
                state=state,
                recent_bytes_back=1024 * 1024,
                max_recent_ids=10_000,
            )
            self.assertEqual(stats["mirrored"], 0)

    def test_start_at_end_skips_backlog_on_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_bus = tmp_path / "src"
            dst_bus = tmp_path / "dst"
            src_bus.mkdir(parents=True, exist_ok=True)
            dst_bus.mkdir(parents=True, exist_ok=True)
            src_events = src_bus / "events.ndjson"
            dst_events = dst_bus / "events.ndjson"

            # Source starts with two historical lines.
            src_events.write_bytes(mk_event("a", "old") + mk_event("b", "old"))

            rc = bus_mirror_daemon.main(
                [
                    "--from-bus-dir",
                    str(src_bus),
                    "--to-bus-dir",
                    str(dst_bus),
                    "--once",
                    "--start-at-end",
                    "--recent-bytes-back",
                    "1024",
                    "--max-recent-ids",
                    "1000",
                ]
            )
            self.assertEqual(rc, 0)

            # First run should skip backlog and not write anything to dest.
            if dst_events.exists():
                self.assertEqual(dst_events.read_text(encoding="utf-8").strip(), "")

            # Append a new line and run again; only the new line should be mirrored.
            with src_events.open("ab") as f:
                f.write(mk_event("c", "new"))
            rc = bus_mirror_daemon.main(
                [
                    "--from-bus-dir",
                    str(src_bus),
                    "--to-bus-dir",
                    str(dst_bus),
                    "--once",
                    "--start-at-end",
                    "--recent-bytes-back",
                    "1024",
                    "--max-recent-ids",
                    "1000",
                ]
            )
            self.assertEqual(rc, 0)

            lines = [l for l in dst_events.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["id"], "c")


if __name__ == "__main__":
    unittest.main()
