import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class TestPBEPORT(unittest.TestCase):
    def test_snapshot_renders_sections(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            events = bus_dir / "events.ndjson"
            now = time.time()
            sample = [
                {
                    "id": "e1",
                    "ts": now - 5,
                    "iso": "2025-12-15T00:00:01Z",
                    "topic": "infer_sync.checkin",
                    "kind": "metric",
                    "level": "info",
                    "actor": "agent-a",
                    "data": {"status": "working", "done": 1, "open": 2, "blocked": 0, "errors": 0, "next": "x", "subproject": "infercell"},
                },
                {
                    "id": "e2",
                    "ts": now - 4,
                    "iso": "2025-12-15T00:00:02Z",
                    "topic": "providers.incident",
                    "kind": "metric",
                    "level": "warn",
                    "actor": "router",
                    "data": {"provider": "claude-cli", "blocker": "auth", "exit_code": 2, "cooldown_s": 0},
                },
                {
                    "id": "e3",
                    "ts": now - 3,
                    "iso": "2025-12-15T00:00:03Z",
                    "topic": "infer_sync.request",
                    "kind": "request",
                    "level": "info",
                    "actor": "agent-b",
                    "data": {"req_id": "r1", "intent": "do thing", "subproject": "plurichat"},
                },
            ]
            events.write_text("\n".join(json.dumps(x) for x in sample) + "\n", encoding="utf-8")

            os.environ["PLURIBUS_BUS_DIR"] = str(bus_dir)
            from nucleus.tools import pbeport

            snap = pbeport.build_snapshot(bus_dir=bus_dir, window_s=60, width=12)
            out = pbeport.render_snapshot(snap)
            self.assertIn("PBEPORT snapshot", out)
            self.assertIn("PRESENT", out)
            self.assertIn("PAST", out)
            self.assertIn("FUTURE", out)
            self.assertIn("providers.incident", out)
            self.assertIn("pending infer_sync req_ids", out)

    def test_cli_runs(self):
        with tempfile.TemporaryDirectory() as td:
            bus_dir = Path(td)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")
            os.environ["PLURIBUS_BUS_DIR"] = str(bus_dir)
            from nucleus.tools import pbeport

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = pbeport.main(["--bus-dir", str(bus_dir), "--window", "60", "--width", "12"])
            self.assertEqual(rc, 0)
            self.assertIn("PBEPORT snapshot", buf.getvalue())

    def test_counts_helpers(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            beam = base / "beam.md"
            golden = base / "golden.md"
            beam.write_text("# x\n## Entry 1\nx\n## Entry 2\n", encoding="utf-8")
            golden.write_text("a\nb\nc\n", encoding="utf-8")

            from nucleus.tools import pbeport

            self.assertEqual(pbeport.count_beam_entries(beam), 2)
            self.assertEqual(pbeport.count_lines(golden), 3)


if __name__ == "__main__":
    unittest.main()
