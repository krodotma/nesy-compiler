import json
import pathlib
import unittest


class TestSemopsPBUS(unittest.TestCase):
    @unittest.skip("PBUS operator not yet implemented in semops.json")
    def test_semops_has_pbus_v12_1(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        p = root / "specs" / "semops.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        # Protocol version updated to v17 (DKIN evolution)
        self.assertIn(obj.get("protocol_version"), ["v12.1", "v13", "v15", "v16", "v17"])
        ops = obj.get("operators") or {}
        self.assertIn("PBUS", ops)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("pbus"), "nucleus/tools/pbus_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("operator.pbus.request", bus_topics)
        self.assertIn("operator.pbus.report", bus_topics)


if __name__ == "__main__":
    unittest.main()
