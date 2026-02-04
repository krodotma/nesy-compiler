import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from plurichat_tui import Picker  # noqa: E402


class TestPicker(unittest.TestCase):
    def test_picker_wraps(self):
        p = Picker("provider", ["a", "b", "c"])
        self.assertEqual(p.current(), "a")
        p.move(1)
        self.assertEqual(p.current(), "b")
        p.move(10)
        self.assertEqual(p.current(), "c")
        p.move(1)
        self.assertEqual(p.current(), "a")

    def test_toggle(self):
        p = Picker("provider", ["auto"])
        self.assertFalse(p.active)
        p.toggle()
        self.assertTrue(p.active)


if __name__ == "__main__":
    unittest.main()
