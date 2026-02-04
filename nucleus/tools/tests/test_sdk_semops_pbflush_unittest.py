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


class TestSdkSemOpsPbflush(unittest.TestCase):
    def test_pbflush_returns_operator_req_id_and_emits_bus(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            bus_dir = pathlib.Path(td)
            c = SemOpsClient()
            rid = c.pbflush(message="wrap + await next ckin", bus_dir=str(bus_dir), actor="tester")
            self.assertTrue(rid)

            events = _read_events(bus_dir)
            reqs = [
                e
                for e in events
                if e.get("topic") == "operator.pbflush.request"
                and e.get("kind") == "request"
                and isinstance(e.get("data"), dict)
                and e["data"].get("req_id") == rid
            ]
            self.assertEqual(len(reqs), 1)


if __name__ == "__main__":
    unittest.main()

