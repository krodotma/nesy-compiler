#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import bus_doctor  # noqa: E402


class TestBusDoctor(unittest.TestCase):
    def test_inspect_reports_last_timestamp_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            bus_dir = Path(tmp)
            events = bus_dir / "events.ndjson"
            lines = [
                json.dumps({"ts": 1700000000, "iso": "2023-11-14T22:13:20Z", "topic": "alpha"}),
                json.dumps({"ts": 1700000001, "iso": "2023-11-14T22:13:21Z", "topic": "beta"}),
            ]
            events.write_text("\n".join(lines) + "\n", encoding="utf-8")

            report = bus_doctor.inspect_bus(
                bus_dir=bus_dir,
                events_path=None,
                max_bytes=10_000,
                max_age_hours=None,
                tail_max_lines=1000,
                now=1700000002,
            )

            self.assertEqual(report.json_errors, 0)
            self.assertEqual(report.json_valid, 2)
            self.assertAlmostEqual(report.last_event_ts or 0, 1700000001, places=3)
            self.assertFalse(report.needs_rotation)

    def test_rotation_moves_log_and_creates_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            bus_dir = Path(tmp) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events = bus_dir / "events.ndjson"
            events.write_text('{"ts": 1700000000, "topic": "alpha"}\n', encoding="utf-8")

            report = bus_doctor.inspect_bus(
                bus_dir=bus_dir,
                events_path=None,
                max_bytes=1,
                max_age_hours=None,
                tail_max_lines=1000,
                now=1700000002,
            )
            archive_dir = Path(tmp) / "archive"
            archive_path, note_path = bus_doctor.rotate_events(report, archive_dir=archive_dir, dry_run=False)

            self.assertTrue(archive_path)
            self.assertTrue(note_path)
            self.assertTrue(Path(archive_path).exists())
            self.assertTrue(Path(note_path).exists())
            self.assertTrue(events.exists())
            self.assertEqual(events.stat().st_size, 0)

    def test_json_errors_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            bus_dir = Path(tmp)
            events = bus_dir / "events.ndjson"
            events.write_text('{"ts": 1700000000, "topic": "alpha"}\n{not-json}\n', encoding="utf-8")

            report = bus_doctor.inspect_bus(
                bus_dir=bus_dir,
                events_path=None,
                max_bytes=10_000,
                max_age_hours=None,
                tail_max_lines=1000,
                now=1700000002,
            )

            self.assertGreater(report.json_errors, 0)


if __name__ == "__main__":
    unittest.main()
