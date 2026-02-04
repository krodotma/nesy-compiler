#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import code_executor  # noqa: E402


class TestCodeExecutorRing0(unittest.TestCase):
    def test_detects_simple_ring0_write(self):
        code = "open('AGENTS.md','w').write('oops')"
        self.assertEqual(code_executor.check_ring0_violation(code), "AGENTS.md")

    def test_allows_non_ring0_code(self):
        code = "print('hello')"
        self.assertIsNone(code_executor.check_ring0_violation(code))


if __name__ == "__main__":
    unittest.main()

