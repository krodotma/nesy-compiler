import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sota_distillations import artifact_path_for, materialize_sota_distillation  # noqa: E402


class TestSotaDistillations(unittest.TestCase):
    def test_materialize_writes_markdown_and_appends_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus" / "index").mkdir(parents=True, exist_ok=True)
            resp = {
                "id": "r1",
                "kind": "strp_response",
                "req_id": "req-1",
                "provider": "auto",
                "model": None,
                "exit_code": 0,
                "iso": "2025-01-01T00:00:00Z",
                "output": "# Hello\\n\\nworld",
                "stderr": "",
                "ts": 0.0,
            }

            rec = materialize_sota_distillation(root=root, sota_item_id="item-1", response=resp)
            path = Path(rec["path"])
            self.assertTrue(path.exists())
            self.assertEqual(path, artifact_path_for(root=root, sota_item_id="item-1", req_id="req-1"))

            artifacts = (root / ".pluribus" / "index" / "artifacts.ndjson").read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(artifacts), 1)
            obj = json.loads(artifacts[-1])
            self.assertEqual(obj["type"], "sota_distillation")
            self.assertEqual(obj["sota_item_id"], "item-1")
            self.assertEqual(obj["req_id"], "req-1")


if __name__ == "__main__":
    unittest.main()

