import json
import tempfile
import time
import unittest
from pathlib import Path


class TestPluribusCheck(unittest.TestCase):
    def test_expected_actor_parser(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import pluribus_check  # noqa: E402

        self.assertEqual(pluribus_check.parse_expected_actors(None), [])
        self.assertEqual(pluribus_check.parse_expected_actors(""), [])
        self.assertEqual(pluribus_check.parse_expected_actors("a,b,c"), ["a", "b", "c"])
        self.assertEqual(pluribus_check.parse_expected_actors(" a ,  b "), ["a", "b"])

    def test_trigger_result_watches_reports_nonblocking(self):
        import sys
        import threading

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import pluribus_check  # noqa: E402

        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            events = bus_dir / "events.ndjson"
            events.write_text("", encoding="utf-8")

            def writer():
                time.sleep(0.2)
                report = {
                    "id": "x",
                    "ts": time.time(),
                    "iso": "x",
                    "topic": "pluribus.check.report",
                    "kind": "metric",
                    "level": "info",
                    "actor": "tester",
                    "host": "x",
                    "pid": 1,
                    "trace_id": None,
                    "run_id": None,
                    "data": {"status": "idle"},
                }
                with events.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(report) + "\n")

            t = threading.Thread(target=writer, daemon=True)
            t.start()

            args = type(
                "Args",
                (),
                {
                    "bus_dir": str(bus_dir),
                    "message": "t",
                    "no_watch": False,
                    "timeout_s": "5",
                    "poll": "0.05",
                    "follow": False,
                    "expect_actors": "tester",
                    "min_reports": 0,
                    "strict": True,
                    "quiet": True,
                },
            )()
            # Should return 0 because expected actor appears.
            rc = pluribus_check.cmd_trigger(args)
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
