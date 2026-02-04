#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import rd_ingest  # noqa: E402


class TestRDIngestUnstructuredFlag(unittest.TestCase):
    def test_parser_accepts_unstructured_flags(self):
        parser = rd_ingest.build_parser()
        args = parser.parse_args(["ingest", "/tmp/x", "--unstructured"])
        self.assertTrue(args.unstructured)
        self.assertEqual(args.unstructured_strategy, "fast")

        args2 = parser.parse_args(["ingest", "/tmp/x", "--unstructured", "--unstructured-strategy", "hi_res"])
        self.assertTrue(args2.unstructured)
        self.assertEqual(args2.unstructured_strategy, "hi_res")


if __name__ == "__main__":
    unittest.main()

