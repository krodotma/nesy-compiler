#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import rag_index  # noqa: E402


class TestRagIndexSemanticQuery(unittest.TestCase):
    def test_semantic_query_prefers_token_overlap_under_hash_embed(self):
        prev = os.environ.get("PLURIBUS_EMBED_MODE")
        try:
            os.environ["PLURIBUS_EMBED_MODE"] = "hash"
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                db_path = rag_index.db_path_for_root(root)
                con = rag_index.connect(db_path)
                rag_index.init_db(con)

                doc1 = rag_index.upsert_text(
                    con,
                    title="doc1",
                    source="unit",
                    text="hello world",
                    meta={"kind": "unit"},
                    actor="unit",
                )
                doc2 = rag_index.upsert_text(
                    con,
                    title="doc2",
                    source="unit",
                    text="photons entanglement measurement",
                    meta={"kind": "unit"},
                    actor="unit",
                )
                con.close()

                args = type(
                    "Args",
                    (),
                    {
                        "root": str(root),
                        "bus_dir": None,
                        "query": "photons",
                        "k": 2,
                        "semantic": True,
                    },
                )()

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = rag_index.cmd_query(args)
                self.assertEqual(rc, 0)
                lines = [l for l in buf.getvalue().splitlines() if l.strip()]
                self.assertGreaterEqual(len(lines), 1)

                # First result should be doc2 (token overlap).
                import json

                first = json.loads(lines[0])
                self.assertEqual(first["method"], "semantic")
                self.assertEqual(first["doc_id"], doc2)
                self.assertNotEqual(first["doc_id"], doc1)
        finally:
            if prev is None:
                os.environ.pop("PLURIBUS_EMBED_MODE", None)
            else:
                os.environ["PLURIBUS_EMBED_MODE"] = prev


if __name__ == "__main__":
    unittest.main()

