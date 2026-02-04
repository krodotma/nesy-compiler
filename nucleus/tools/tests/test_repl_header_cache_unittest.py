#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(TOOLS_DIR))

import repl_header_cache  # noqa: E402


class TestReplHeaderCache(unittest.TestCase):
    def test_put_get_expiry(self):
        header = {
            "contract": "repl_header.v1",
            "agent": "codex",
            "dkin_version": "v28",
            "paip_version": "v15",
            "citizen_version": "v1",
            "attestation": {
                "date": "2025-12-28T23:30:46Z",
                "score": "100/100"
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            os.environ["REPL_HEADER_CACHE_PATH"] = str(cache_path)
            current_versions = repl_header_cache.load_protocol_versions()
            cache = repl_header_cache.load_cache(cache_path, current_versions)

            repl_header_cache.put_entry(cache, agent="codex", header=header, ttl=2, now_ts=1000)
            repl_header_cache.save_cache(cache_path, cache)

            cache2 = repl_header_cache.load_cache(cache_path, current_versions)
            entry = repl_header_cache.get_entry(cache2, agent="codex", now_ts=1001)
            self.assertIsNotNone(entry)

            expired = repl_header_cache.get_entry(cache2, agent="codex", now_ts=1005)
            self.assertIsNone(expired)

        os.environ.pop("REPL_HEADER_CACHE_PATH", None)


if __name__ == "__main__":
    unittest.main()
