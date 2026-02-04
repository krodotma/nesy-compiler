import json
import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
ROOT = TOOLS_DIR.parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import idiolect  # noqa: E402


class TestIdiolect(unittest.TestCase):
    def test_idiolect_validates(self):
        path = ROOT / "nucleus" / "specs" / "idiolect.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        ok, errs = idiolect.validate_idiolect(obj)
        self.assertTrue(ok, errs)


if __name__ == "__main__":
    unittest.main()

