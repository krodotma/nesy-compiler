import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sota_kg import append_sota_kg_node, build_sota_kg_node, kg_nodes_path  # noqa: E402


class TestSotaKg(unittest.TestCase):
    def test_build_and_append(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus" / "index").mkdir(parents=True, exist_ok=True)
            node = build_sota_kg_node(
                sota_item={"id": "item-1", "title": "X", "url": "https://example.com", "tags": ["memory", "rag"]},
                ref="https://example.com",
                actor="tester",
                context="test",
                extra_tags=["from:test"],
            )
            append_sota_kg_node(root=root, node=node)
            path = kg_nodes_path(root)
            self.assertTrue(path.exists())
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("sota_item_id:item-1", lines[0])


if __name__ == "__main__":
    unittest.main()

