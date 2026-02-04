#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import repl_header_audit  # noqa: E402


class TestReplHeaderAudit(unittest.TestCase):
    def test_repl_header_audit_passes(self):
        results = repl_header_audit.run_audit()
        self.assertEqual(results["verdict"], "PASS")
        self.assertEqual(results["score_percent"], 100)


if __name__ == "__main__":
    unittest.main()
