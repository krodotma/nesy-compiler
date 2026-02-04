import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


class TestStrpWorkerOnlyReqId(unittest.TestCase):
    def test_only_req_id_processes_single_item(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")
            idx = root / ".pluribus" / "index"
            idx.mkdir(parents=True, exist_ok=True)

            reqs = idx / "requests.ndjson"
            resps = idx / "responses.ndjson"
            req1 = {"req_id": "r1", "goal": "Say hello", "kind": "distill", "provider_hint": "mock", "parallelizable": False}
            req2 = {"req_id": "r2", "goal": "Say hello again", "kind": "distill", "provider_hint": "mock", "parallelizable": False}
            reqs.write_text(json.dumps(req1) + "\n" + json.dumps(req2) + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PLURIBUS_BUS_DIR"] = str(root / ".pluribus" / "bus")
            env["PLURIBUS_ALLOW_MOCK"] = "1"
            pathlib.Path(env["PLURIBUS_BUS_DIR"]).mkdir(parents=True, exist_ok=True)

            tool = pathlib.Path("/pluribus/nucleus/tools/strp_worker.py")
            p = subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "--root",
                    str(root),
                    "--provider",
                    "mock",
                    "--only-req-id",
                    "r2",
                    "--once",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(p.returncode, 0, p.stderr)

            self.assertTrue(resps.exists(), "responses.ndjson not created")
            lines = [json.loads(x) for x in resps.read_text(encoding="utf-8").splitlines() if x.strip()]
            got = [x.get("req_id") for x in lines]
            self.assertIn("r2", got)
            self.assertNotIn("r1", got)


if __name__ == "__main__":
    unittest.main()
