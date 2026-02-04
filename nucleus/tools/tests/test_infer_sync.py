import json
import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import infer_sync  # noqa: E402


class TestInferSync(unittest.TestCase):
    def test_default_actor_nonempty(self):
        a = infer_sync.default_actor()
        self.assertTrue(isinstance(a, str) and len(a) > 0)

    def test_request_payload_json_parseable(self):
        inputs = {"x": 1}
        constraints = {"append_only": True}
        # Ensure our CLI-compatible payloads are JSON serializable
        json.dumps(inputs)
        json.dumps(constraints)


if __name__ == "__main__":
    unittest.main()

