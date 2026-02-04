import json
import pathlib
import sys
import tempfile
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from grounding import verify_grounded_output  # noqa: E402
import rag_index  # noqa: E402


class TestGroundingVerifier(unittest.TestCase):
    def test_requires_citations_when_enabled(self):
        ok, issues = verify_grounded_output({"summary": "x"}, require_citations=True)
        self.assertFalse(ok)
        self.assertIn("missing_citations", issues)

    def test_accepts_string_citations(self):
        ok, issues = verify_grounded_output(
            {"summary": "x", "citations": ["rag:doc123", "https://example.com/paper"]},
            require_citations=True,
        )
        self.assertTrue(ok, issues)

    def test_accepts_object_citations(self):
        ok, issues = verify_grounded_output(
            {"summary": "x", "citations": [{"ref": "rag:doc123"}, {"url": "https://example.com"}]},
            require_citations=True,
        )
        self.assertTrue(ok, issues)

    def test_rejects_unknown_citation_types(self):
        ok, issues = verify_grounded_output(
            {"summary": "x", "citations": [123]},
            require_citations=True,
        )
        self.assertFalse(ok)
        self.assertIn("invalid_citation_entry", issues)

    def test_json_string_input_supported(self):
        payload = json.dumps({"summary": "x", "citations": ["rag:doc123"]})
        ok, issues = verify_grounded_output(payload, require_citations=True)
        self.assertTrue(ok, issues)

    def test_validate_rag_doc_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            db_path = rag_index.db_path_for_root(root)
            con = rag_index.connect(db_path)
            rag_index.init_db(con)
            doc_id = rag_index.upsert_text(
                con,
                title="t",
                source="unit",
                text="alpha",
                meta={"kind": "test"},
                actor="test",
            )
            ok, issues = verify_grounded_output(
                {"summary": "x", "citations": [f"rag:{doc_id}"]},
                require_citations=True,
                validate_refs=True,
                root=str(root),
            )
            self.assertTrue(ok, issues)

            ok2, issues2 = verify_grounded_output(
                {"summary": "x", "citations": ["rag:not_a_real_doc"]},
                require_citations=True,
                validate_refs=True,
                root=str(root),
            )
            self.assertFalse(ok2)
            self.assertIn("citation_not_found_rag", issues2)

    def test_validate_rhizome_object_sha(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            obj_dir = root / ".pluribus" / "objects"
            obj_dir.mkdir(parents=True, exist_ok=True)
            sha = "a" * 64
            (obj_dir / sha).write_bytes(b"x")

            ok, issues = verify_grounded_output(
                {"summary": "x", "citations": [f"sha256:{sha}"]},
                require_citations=True,
                validate_refs=True,
                root=str(root),
            )
            self.assertTrue(ok, issues)

            ok2, issues2 = verify_grounded_output(
                {"summary": "x", "citations": [f"sha256:{'b'*64}"]},
                require_citations=True,
                validate_refs=True,
                root=str(root),
            )
            self.assertFalse(ok2)
            self.assertIn("citation_not_found_object", issues2)


if __name__ == "__main__":
    unittest.main()
