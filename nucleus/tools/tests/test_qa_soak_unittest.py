import os
import tempfile
import unittest

from nucleus.tools.qa_soak import run_soak


class TestQaSoak(unittest.TestCase):
    def test_run_soak_emits_events(self):
        with tempfile.TemporaryDirectory() as td:
            summary = run_soak(
                bus_dir=td,
                topic="qa.soak.event",
                total_events=3,
                rate=0.0,
                payload_bytes=0,
                emit_bus_summary=False,
            )
            self.assertEqual(summary["events"], 3)
            events_path = os.path.join(td, "events.ndjson")
            with open(events_path, "r", encoding="utf-8") as handle:
                lines = [line for line in handle.readlines() if line.strip()]
            self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
