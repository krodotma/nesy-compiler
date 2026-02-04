import json
import pathlib
import unittest


class TestSemopsPBOSNAP(unittest.TestCase):
    def test_semops_has_pbosnap(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        semops_path = root / "specs" / "semops.json"
        obj = json.loads(semops_path.read_text(encoding="utf-8"))
        ops = obj.get("operators") or {}
        self.assertIn("PBOSNAP", ops)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("pbosnap"), "nucleus/tools/pbosnap_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("operator.pbosnap.report", bus_topics)
        grammar = obj.get("grammar") or {}
        operator_pattern = grammar.get("operator_pattern") or ""
        self.assertIn("pbosnap", operator_pattern)


if __name__ == "__main__":
    unittest.main()
