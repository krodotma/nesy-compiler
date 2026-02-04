import json
import pathlib
import unittest


class TestSemopsSpeakOperator(unittest.TestCase):
    def test_semops_has_speak(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        p = root / "specs" / "semops.json"
        obj = json.loads(p.read_text(encoding="utf-8"))
        ops = obj.get("operators") or {}
        self.assertIn("SPEAK", ops)
        speak = ops.get("SPEAK") or {}
        aliases = [str(a).lower() for a in (speak.get("aliases") or [])]
        self.assertIn("speak", aliases)
        self.assertIn("speak dump to this endpoint file content requested spoken", aliases)
        tool_map = obj.get("tool_map") or {}
        self.assertEqual(tool_map.get("speak"), "nucleus/tools/speak_operator.py")
        bus_topics = obj.get("bus_topics") or {}
        self.assertIn("speaker.bus.write", bus_topics)


if __name__ == "__main__":
    unittest.main()
