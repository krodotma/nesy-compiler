import json
import pathlib
import unittest


class TestSemopsPBDEEP(unittest.TestCase):
    def test_semops_has_pbdeep(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        p = root / "specs" / "semops.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        ops = obj.get("operators") or {}
        self.assertIn("PBDEEP", ops)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("pbdeep"), "nucleus/tools/pbdeep_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("operator.pbdeep.request", bus_topics)
        self.assertIn("operator.pbdeep.progress", bus_topics)
        self.assertIn("operator.pbdeep.index.updated", bus_topics)
        self.assertIn("operator.pbdeep.report", bus_topics)


if __name__ == "__main__":
    unittest.main()
