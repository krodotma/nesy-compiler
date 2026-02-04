import os
import tempfile
import time
import unittest

from nucleus.tools.qa_observer import (
    RecurrenceTracker,
    _in_scope,
    _memory_pressure,
    _parse_meminfo,
    rotate_bus_file,
    should_rotate,
)


class TestQaObserverRotation(unittest.TestCase):
    def test_should_rotate_by_size(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.ndjson")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("x" * 20)
            stats = os.stat(path)
            self.assertTrue(should_rotate(stats, now=time.time(), max_bytes=10, max_age_s=0))

    def test_rotate_bus_file_creates_archive(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.ndjson")
            archive_dir = os.path.join(td, "archive")
            content = "line1\nline2\n"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            result = rotate_bus_file(path, archive_dir)
            self.assertIsNotNone(result)
            archive_path = result["archive_path"]
            self.assertTrue(os.path.exists(archive_path))
            with open(archive_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), content)
            with open(path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "")

    def test_rotate_bus_file_skips_empty(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.ndjson")
            archive_dir = os.path.join(td, "archive")
            with open(path, "w", encoding="utf-8"):
                pass
            result = rotate_bus_file(path, archive_dir)
            self.assertIsNone(result)


class TestQaObserverRecurrence(unittest.TestCase):
    def test_recurrence_tracker_counts_and_span(self):
        tracker = RecurrenceTracker(window_s=10)
        count, span = tracker.add("fp", 0.0)
        self.assertEqual(count, 1)
        self.assertEqual(span, 0.0)
        count, span = tracker.add("fp", 4.0)
        self.assertEqual(count, 2)
        self.assertEqual(span, 4.0)
        count, span = tracker.add("fp", 12.0)
        self.assertEqual(count, 2)
        self.assertEqual(span, 8.0)

    def test_in_scope_prefixes(self):
        self.assertTrue(_in_scope("qa.action.review.request", ["qa."], []))
        self.assertFalse(_in_scope("telemetry.client.error", ["qa."], []))
        self.assertFalse(_in_scope("qa.action.review.request", ["qa."], ["qa."]))


class TestQaObserverMeminfo(unittest.TestCase):
    def test_parse_meminfo(self):
        text = "MemTotal:       8000000 kB\nMemAvailable:    500000 kB\n"
        info = _parse_meminfo(text)
        self.assertEqual(info["MemTotal"], 8000000)
        self.assertEqual(info["MemAvailable"], 500000)

    def test_memory_pressure_thresholds(self):
        info = {"MemTotal": 1000, "MemAvailable": 80}
        self.assertTrue(_memory_pressure(info, min_available_kb=100, min_available_ratio=0.0))
        self.assertTrue(_memory_pressure(info, min_available_kb=0, min_available_ratio=0.2))
        self.assertFalse(_memory_pressure(info, min_available_kb=50, min_available_ratio=0.0))


if __name__ == "__main__":
    unittest.main()
