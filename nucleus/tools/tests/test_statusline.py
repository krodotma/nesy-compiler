import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statusline


class TestStatusline(unittest.TestCase):
    def test_classify_failure_text(self):
        self.assertEqual(statusline.classify_failure_text("Invalid API key · Please run /login"), "blocked_auth")
        self.assertEqual(statusline.classify_failure_text("http error: 429 RESOURCE_EXHAUSTED"), "blocked_quota")
        self.assertEqual(statusline.classify_failure_text("no provider configured"), "blocked_config")
        self.assertEqual(statusline.classify_failure_text("something else broke"), "error")

    def test_compute_summary_counts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")
            idx = root / ".pluribus" / "index"
            idx.mkdir(parents=True, exist_ok=True)

            now = time.time()
            req1 = {"req_id": "r1", "ts": now - 5, "iso": "x", "goal": "done"}
            req2 = {"req_id": "r2", "ts": now - 5, "iso": "x", "goal": "blocked"}
            req3 = {"req_id": "r3", "ts": now - 5, "iso": "x", "goal": "open"}
            (idx / "requests.ndjson").write_text(
                "\n".join(json.dumps(x) for x in [req1, req2, req3]) + "\n", encoding="utf-8"
            )

            resp1 = {"req_id": "r1", "ts": now - 4, "iso": "x", "exit_code": 0, "output": "{}"}
            resp2 = {"req_id": "r2", "ts": now - 4, "iso": "x", "exit_code": 1, "stderr": "Please run /login"}
            (idx / "responses.ndjson").write_text(
                "\n".join(json.dumps(x) for x in [resp1, resp2]) + "\n", encoding="utf-8"
            )

            s = statusline.compute_summary(root=root, bus_dir=None, stuck_after_s=3600)
            self.assertEqual(s["counts"]["requests"], 3)
            self.assertEqual(s["counts"]["done"], 1)
            self.assertEqual(s["counts"]["open"], 2)
            self.assertEqual(s["counts"]["blocked"], 1)
            self.assertEqual(s["counts"]["errors"], 0)
            self.assertEqual(s["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
