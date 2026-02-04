import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import uuid


class TestStrpWorkerInferCellBinding(unittest.TestCase):
    def test_trace_id_binds_to_infercell_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")
            idx = root / ".pluribus" / "index"
            idx.mkdir(parents=True, exist_ok=True)

            trace_id = str(uuid.uuid4())
            reqs = idx / "requests.ndjson"
            resps = idx / "responses.ndjson"
            req = {"req_id": "r1", "trace_id": trace_id, "goal": "Say hello", "kind": "distill", "provider_hint": "mock"}
            reqs.write_text(json.dumps(req) + "\n", encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PLURIBUS_BUS_DIR"] = str(root / ".pluribus" / "bus")
            env["PLURIBUS_ALLOW_MOCK"] = "1"
            pathlib.Path(env["PLURIBUS_BUS_DIR"]).mkdir(parents=True, exist_ok=True)

            tool = pathlib.Path("/pluribus/nucleus/tools/strp_worker.py")
            p = subprocess.run(
                [sys.executable, str(tool), "--root", str(root), "--provider", "mock", "--only-req-id", "r1", "--once"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(resps.exists(), "responses.ndjson not created")

            # Verify infercell mapping exists for this trace_id.
            index_path = root / ".pluribus" / "infercells" / "index.json"
            self.assertTrue(index_path.exists(), "infercells/index.json not created")
            mapping = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertIn(trace_id, mapping)
            cell_id = mapping[trace_id]
            self.assertTrue(isinstance(cell_id, str) and len(cell_id) > 0)

            # Verify workspace exists.
            workspace = root / ".pluribus" / "infercells" / cell_id / "workspace"
            self.assertTrue(workspace.exists() and workspace.is_dir(), f"workspace not created: {workspace}")

            # Verify binding evidence exists on the bus.
            events_path = root / ".pluribus" / "bus" / "events.ndjson"
            self.assertTrue(events_path.exists(), "events.ndjson not created")
            found = False
            for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("topic") == "infercell.exec.bound":
                    data = obj.get("data") or {}
                    if data.get("req_id") == "r1" and data.get("trace_id") == trace_id and data.get("cell_id") == cell_id:
                        found = True
                        break
            self.assertTrue(found, "infercell.exec.bound evidence not found")


if __name__ == "__main__":
    unittest.main()
