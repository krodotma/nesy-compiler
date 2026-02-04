import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from pluribus_directive import detect_pluribus_directive  # noqa: E402


class TestPluribusDirective(unittest.TestCase):
    def test_prefix_parses_kind_effects_goal(self):
        text = "user: PLURIBUS(kind=apply,effects=file): implement X"
        d = detect_pluribus_directive(text)
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.kind, "apply")
        self.assertEqual(d.effects, "file")
        self.assertEqual(d.goal, "implement X")
        bus = d.to_bus_dict()
        self.assertIn("goal_sha256", bus)
        self.assertNotIn("goal", bus)

    def test_inline_parses_goal(self):
        text = "please PLURIBUS: do the thing"
        d = detect_pluribus_directive(text)
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.kind, "other")
        self.assertEqual(d.effects, "unknown")
        self.assertEqual(d.goal, "do the thing")


if __name__ == "__main__":
    unittest.main()

