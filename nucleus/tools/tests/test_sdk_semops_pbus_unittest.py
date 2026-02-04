import json
import pathlib
import tempfile
import unittest

SDK_DIR = pathlib.Path(__file__).resolve().parents[2] / "sdk"
TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(SDK_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from semops import SemOpsClient  # noqa: E402


def _read_events(bus_dir: pathlib.Path) -> list[dict]:
    p = bus_dir / "events.ndjson"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


class TestSdkSemOpsPbus(unittest.TestCase):
    @unittest.skip("PBUS operator not yet implemented in semops.json tool_map")
    def test_pbus_returns_operator_req_id_and_emits_bus(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            tmp = pathlib.Path(td)
            bus_dir = tmp / "bus"
            report_path = tmp / "pbus_report.txt"
            c = SemOpsClient()
            rid = c.pbus(
                bus_dir=str(bus_dir),
                actor="tester",
                report_path=str(report_path),
                include_reports=False,
                include_logs=False,
                window_s=60,
                max_events=5,
            )
            self.assertTrue(rid)

            events = _read_events(bus_dir)
            reqs = [
                e
                for e in events
                if e.get("topic") == "operator.pbus.request"
                and e.get("kind") == "request"
                and isinstance(e.get("data"), dict)
                and e["data"].get("req_id") == rid
            ]
            reports = [
                e
                for e in events
                if e.get("topic") == "operator.pbus.report"
                and isinstance(e.get("data"), dict)
                and e["data"].get("req_id") == rid
            ]
            self.assertEqual(len(reqs), 1)
            self.assertGreaterEqual(len(reports), 1)

            self.assertTrue(report_path.exists())
            text = report_path.read_text(encoding="utf-8", errors="replace")
            self.assertIn("PBUS_REPORT v1", text)


if __name__ == "__main__":
    unittest.main()
