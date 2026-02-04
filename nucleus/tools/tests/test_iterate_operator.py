import json
import os
import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import iterate_operator  # noqa: E402


class TestIterateOperator(unittest.TestCase):
    def test_publish_iterate_emits_requests(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            rid = iterate_operator.publish_iterate(
                bus_dir=bus_dir,
                actor="test-agent",
                req_id="r-1",
                subproject="beam_10x",
                intent="iterate",
                response_topic="infer_sync.response",
                window_s=60,
            )
            self.assertEqual(rid, "r-1")

            events = [json.loads(l) for l in (bus_dir / "events.ndjson").read_text(encoding="utf-8").splitlines() if l.strip()]
            topics = [e.get("topic") for e in events]
            self.assertIn("infer_sync.request", topics)
            self.assertIn("operator.iterate.request", topics)
            reqs = [e for e in events if e.get("topic") == "infer_sync.request"]
            self.assertEqual(reqs[0]["kind"], "request")
            self.assertEqual(reqs[0]["data"]["req_id"], "r-1")


if __name__ == "__main__":
    unittest.main()

