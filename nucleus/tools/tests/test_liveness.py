import unittest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liveness import TimeBoundMonitor  # noqa: E402

class TestTimeBoundMonitor(unittest.TestCase):
    def test_healthy_within_limit(self):
        monitor = TimeBoundMonitor(max_seconds=1.0)
        self.assertTrue(monitor.is_healthy())

    def test_unhealthy_after_limit(self):
        monitor = TimeBoundMonitor(max_seconds=0.1)
        time.sleep(0.2)
        self.assertFalse(monitor.is_healthy())

    def test_diagnostics_structure(self):
        monitor = TimeBoundMonitor(max_seconds=10.0)
        diag = monitor.diagnostics()
        self.assertIn("elapsed", diag)
        self.assertIn("limit", diag)
        self.assertEqual(diag["limit"], 10.0)

if __name__ == "__main__":
    unittest.main()
