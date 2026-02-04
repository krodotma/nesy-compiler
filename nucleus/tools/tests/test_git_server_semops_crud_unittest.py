import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen


def _free_port() -> int:
    try:
        s = socket.socket()
    except PermissionError:
        raise unittest.SkipTest("socket creation not permitted in this environment")
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_http_ok(url: str, *, deadline_s: float = 3.0) -> None:
    deadline = time.time() + deadline_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=0.5) as resp:
                body = resp.read(200).decode("utf-8", errors="replace")
            if body:
                return
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {url} ({last_err})")


def _post_json(url: str, payload: dict) -> dict:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestGitServerSemopsCRUD(unittest.TestCase):
    def test_semops_define_and_undefine(self):
        tools_dir = Path(__file__).resolve().parents[1]
        server = tools_dir / "git_server.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            # Minimal built-in semops registry so we can test override prevention.
            semops_path = root / "nucleus" / "specs" / "semops.json"
            semops_path.parent.mkdir(parents=True, exist_ok=True)
            semops_path.write_text(
                json.dumps(
                    {
                        "operators": {
                            "CKIN": {
                                "id": "ckin",
                                "name": "CKIN",
                                "domain": "observability",
                                "category": "status",
                            "description": "check-in",
                            "aliases": ["ckin", "CKIN"],
                            "tool": "nucleus/tools/ckin_report.py",
                            "bus_topic": "ckin.report",
                            "bus_kind": "metric",
                            "effects": "none",
                        }
                    },
                    "grammar": {"slash_command_pattern": "^/(help|ckin)(\\s+.*)?$"},
                },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            bus_dir = root / ".pluribus" / "bus"
            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}

            proc = subprocess.Popen(
                ["python3", str(server), "--root", str(root), "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                with urlopen(f"http://127.0.0.1:{port}/semops", timeout=2) as resp:
                    schema = json.loads(resp.read().decode("utf-8"))
                self.assertIn("operators", schema)
                self.assertIn("CKIN", schema["operators"])
                self.assertFalse(schema["operators"]["CKIN"].get("user_defined", True))
                self.assertEqual(schema["operators"]["CKIN"].get("effects"), "none")

                # Define MYOP.
                out = _post_json(
                    f"http://127.0.0.1:{port}/semops/user_ops/define",
                    {
                        "actor": "test",
                        "req_id": "req-test-1",
                        "operator": {
                            "key": "MYOP",
                            "id": "myop",
                            "name": "MYOP",
                            "domain": "user",
                            "category": "custom",
                            "effects": "file",
                            "description": "my op",
                            "aliases": ["myop", "MYOP"],
                            "tool": "nucleus/tools/myop_operator.py",
                            "bus_topic": "operator.myop.request",
                            "bus_kind": "request",
                            "ui": {"route": "/#semops", "component": "SemopsEditor"},
                            "agents": ["codex", "gemini"],
                            "apps": ["dashboard"],
                        },
                    },
                )
                self.assertTrue(out.get("success"))
                self.assertEqual(out.get("operator_key"), "MYOP")

                with urlopen(f"http://127.0.0.1:{port}/semops", timeout=2) as resp:
                    schema2 = json.loads(resp.read().decode("utf-8"))
                self.assertIn("MYOP", schema2["operators"])
                self.assertTrue(schema2["operators"]["MYOP"].get("user_defined"))
                self.assertEqual(schema2["operators"]["MYOP"].get("bus_kind"), "request")
                self.assertEqual(schema2["operators"]["MYOP"].get("effects"), "file")

                # Invoke MYOP (non-executing) and verify the bus event.
                inv = _post_json(
                    f"http://127.0.0.1:{port}/semops/invoke",
                    {
                        "actor": "test",
                        "req_id": "req-invoke-1",
                        "operator_key": "MYOP",
                        "mode": "tool",
                        "payload": {"input": "x"},
                    },
                )
                self.assertTrue(inv.get("success"))

                events_path = bus_dir / "events.ndjson"
                deadline = time.time() + 2.0
                matched = None
                while time.time() < deadline and matched is None:
                    if events_path.exists():
                        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if obj.get("topic") != "semops.invoke.request":
                                continue
                            data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
                            if data.get("req_id") != "req-invoke-1":
                                continue
                            matched = obj
                            break
                    if matched is None:
                        time.sleep(0.05)

                self.assertIsNotNone(matched, "missing semops.invoke.request for req-invoke-1")
                data = matched.get("data")  # type: ignore[union-attr]
                self.assertEqual(data.get("operator_key"), "MYOP")
                self.assertEqual(data.get("mode"), "tool")
                self.assertEqual(data.get("effects"), "file")
                self.assertEqual((data.get("payload") or {}).get("input"), "x")
                op_snapshot = data.get("operator")
                self.assertIsInstance(op_snapshot, dict)
                self.assertEqual(op_snapshot.get("effects"), "file")

                # Cannot override built-in CKIN.
                out2 = _post_json(
                    f"http://127.0.0.1:{port}/semops/user_ops/define",
                    {"actor": "test", "operator": {"key": "CKIN", "description": "override attempt"}},
                )
                self.assertFalse(out2.get("success"))
                self.assertIn("Cannot override", out2.get("error", ""))

                # Undefine MYOP.
                out3 = _post_json(
                    f"http://127.0.0.1:{port}/semops/user_ops/undefine",
                    {"actor": "test", "operator_key": "MYOP"},
                )
                self.assertTrue(out3.get("success"))

                with urlopen(f"http://127.0.0.1:{port}/semops", timeout=2) as resp:
                    schema3 = json.loads(resp.read().decode("utf-8"))
                self.assertNotIn("MYOP", schema3["operators"])
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()


if __name__ == "__main__":
    unittest.main()
