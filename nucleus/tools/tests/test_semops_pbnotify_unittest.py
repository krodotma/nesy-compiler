import json
import pathlib
import unittest


class TestSemopsPBNOTIFY(unittest.TestCase):
    def test_semops_has_pbnotify(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        semops_path = root / "specs" / "semops.json"
        obj = json.loads(semops_path.read_text(encoding="utf-8"))
        ops = obj.get("operators") or {}
        self.assertIn("PBNOTIFY", ops)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("pbnotify"), "nucleus/tools/pbnotify_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("agent.notify.request", bus_topics)
        grammar = obj.get("grammar") or {}
        operator_pattern = grammar.get("operator_pattern") or ""
        self.assertIn("pbnotify", operator_pattern)
        slash_pattern = grammar.get("slash_command_pattern") or ""
        self.assertIn("pbnotify", slash_pattern)


if __name__ == "__main__":
    unittest.main()
