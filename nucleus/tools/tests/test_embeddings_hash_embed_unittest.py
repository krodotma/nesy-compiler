#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import embeddings  # noqa: E402


class TestEmbeddingsHashEmbed(unittest.TestCase):
    def test_hash_embed_is_deterministic_and_normalized(self):
        v1 = embeddings.hash_embed("hello world", dim=32)
        v2 = embeddings.hash_embed("hello world", dim=32)
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), 32)

        # Normalization: norm ~ 1 (or 0 if empty text)
        norm = sum(x * x for x in v1) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_embed_text_hash_mode_override(self):
        prev = os.environ.get("PLURIBUS_EMBED_MODE")
        try:
            os.environ["PLURIBUS_EMBED_MODE"] = "hash"
            vec, meta = embeddings.embed_text("a test document", dim=64)
            self.assertIsNotNone(vec)
            self.assertEqual(len(vec or []), 64)
            self.assertEqual(meta.get("mode"), "hash")
        finally:
            if prev is None:
                os.environ.pop("PLURIBUS_EMBED_MODE", None)
            else:
                os.environ["PLURIBUS_EMBED_MODE"] = prev


if __name__ == "__main__":
    unittest.main()

