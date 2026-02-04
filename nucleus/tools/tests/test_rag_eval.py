import pathlib
import sqlite3
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import rag_index  # noqa: E402


class TestRagEval(unittest.TestCase):
    def _fts5_available(self) -> bool:
        try:
            con = sqlite3.connect(":memory:")
            con.execute("CREATE VIRTUAL TABLE t USING fts5(x);")
            con.close()
            return True
        except Exception:
            return False

    def test_basic_retrieval_returns_expected_doc(self):
        if not self._fts5_available():
            self.skipTest("sqlite3 FTS5 not available in this runtime")

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            db_path = rag_index.db_path_for_root(root)
            con = rag_index.connect(db_path)
            rag_index.init_db(con)

            doc_a = rag_index.upsert_text(
                con,
                title="A",
                source="unit",
                text="alpha unique_token_alpha zzz",
                meta={"kind": "test"},
                actor="test",
            )
            doc_b = rag_index.upsert_text(
                con,
                title="B",
                source="unit",
                text="beta unique_token_beta yyy",
                meta={"kind": "test"},
                actor="test",
            )

            rows = con.execute(
                """
                SELECT d.doc_id
                FROM docs_fts
                JOIN docs d ON d.doc_id = docs_fts.doc_id
                WHERE docs_fts MATCH ?
                ORDER BY bm25(docs_fts)
                LIMIT 1
                """,
                ("unique_token_beta",),
            ).fetchall()
            self.assertEqual(rows[0][0], doc_b)
            self.assertNotEqual(doc_a, doc_b)


if __name__ == "__main__":
    unittest.main()

