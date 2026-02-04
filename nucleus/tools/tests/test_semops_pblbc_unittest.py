import json
import pathlib
import unittest


class TestSemopsPBLBC(unittest.TestCase):
    def test_semops_has_pblbc(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        semops_path = root / "specs" / "semops.json"
        obj = json.loads(semops_path.read_text(encoding="utf-8"))
        ops = obj.get("operators") or {}
        self.assertIn("PBLBC", ops)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("pblbc"), "nucleus/tools/pblbc_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("operator.pblbc.report", bus_topics)
        grammar = obj.get("grammar") or {}
        operator_pattern = grammar.get("operator_pattern") or ""
        self.assertIn("pblbc", operator_pattern)

    def test_semops_has_pblbcpurge(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        semops_path = root / "specs" / "semops.json"
        obj = json.loads(semops_path.read_text(encoding="utf-8"))
        ops = obj.get("operators") or {}
        self.assertIn("PBLBCPURGE", ops)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("pblbcpurge"), "nucleus/tools/pblbc_purge_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("operator.pblbc.purge", bus_topics)
        grammar = obj.get("grammar") or {}
        operator_pattern = grammar.get("operator_pattern") or ""
        self.assertIn("pblbcpurge", operator_pattern)


if __name__ == "__main__":
    unittest.main()
