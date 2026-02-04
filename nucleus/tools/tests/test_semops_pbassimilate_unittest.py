import json
import pathlib
import unittest


class TestSemopsPBASSIMILATE(unittest.TestCase):
    def test_semops_has_pbassimilate(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        semops_path = root / "specs" / "semops.json"
        obj = json.loads(semops_path.read_text(encoding="utf-8"))
        ops = obj.get("operators") or {}
        self.assertIn("PBASSIMILATE", ops)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("pbassimilate"), "nucleus/tools/pbassimilate_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("operator.pbassimilate.request", bus_topics)
        self.assertIn("operator.pbassimilate.screening", bus_topics)
        self.assertIn("operator.pbassimilate.consensus", bus_topics)
        self.assertIn("operator.pbassimilate.plan", bus_topics)
        grammar = obj.get("grammar") or {}
        operator_pattern = grammar.get("operator_pattern") or ""
        slash_pattern = grammar.get("slash_command_pattern") or ""
        self.assertIn("pbassimilate", operator_pattern)
        self.assertIn("pbassimilate", slash_pattern)


if __name__ == "__main__":
    unittest.main()
