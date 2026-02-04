import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import sota  # noqa: E402


class TestSotaDistillRequest(unittest.TestCase):
    def test_build_sota_distill_request_has_required_fields(self):
        root = pathlib.Path("/tmp")
        payload = sota.build_sota_distill_request(
            root=root,
            actor="tester",
            provider="auto",
            item={"id": "item-1", "url": "https://example.com", "title": "Example"},
        )
        self.assertEqual(payload["kind"], "distill")
        self.assertEqual(payload["provider_hint"], "auto")
        self.assertEqual(payload["inputs"]["sota_item_id"], "item-1")
        self.assertIn("gates", payload["constraints"])
        self.assertIn("P", payload["constraints"]["gates"])


if __name__ == "__main__":
    unittest.main()

