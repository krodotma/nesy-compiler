import json
import pathlib
import unittest


class TestSemopsPBFLUSH(unittest.TestCase):
    def test_semops_has_pbflush_v12_1(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        p = root / "specs" / "semops.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        # Protocol version updated to v17 (DKIN evolution)
        self.assertIn(obj.get("protocol_version"), ["v12.1", "v13", "v15", "v16", "v17"])
        ops = obj.get("operators") or {}
        self.assertIn("PBFLUSH", ops)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("pbflush"), "nucleus/tools/pbflush_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("operator.pbflush.request", bus_topics)
        self.assertIn("operator.pbflush.ack", bus_topics)


if __name__ == "__main__":
    unittest.main()
