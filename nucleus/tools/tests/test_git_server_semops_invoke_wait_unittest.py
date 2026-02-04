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
    with urlopen(req, timeout=4) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestGitServerSemopsInvokeWait(unittest.TestCase):
    def test_semops_invoke_wait_returns_response(self) -> None:
        tools_dir = Path(__file__).resolve().parents[1]
        server = tools_dir / "git_server.py"
        responder = tools_dir / "semops_invoke_responder.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            semops_path = root / "nucleus" / "specs" / "semops.json"
            semops_path.parent.mkdir(parents=True, exist_ok=True)
            semops_path.write_text(json.dumps({"operators": {}, "grammar": {}}, indent=2) + "\n", encoding="utf-8")

            bus_dir = root / ".pluribus" / "bus"
            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}

            responder_proc = subprocess.Popen(
                [
                    os.environ.get("PYTHON", "python3"),
                    str(responder),
                    "--bus-dir",
                    str(bus_dir),
                    "--run-for-s",
                    "2.0",
                    "--poll",
                    "0.05",
                    "--since-ts",
                    "0",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )

            server_proc = subprocess.Popen(
                ["python3", str(server), "--root", str(root), "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                out = _post_json(
                    f"http://127.0.0.1:{port}/semops/user_ops/define",
                    {
                        "actor": "test",
                        "req_id": "req-define-1",
                        "operator": {
                            "key": "MYOP",
                            "id": "myop",
                            "name": "MYOP",
                            "domain": "execution",
                            "category": "tool",
                            "effects": "file",
                            "description": "my op",
                            "aliases": ["myop", "MYOP"],
                            "tool": "nucleus/tools/myop_operator.py",
                            "bus_topic": "operator.myop.request",
                            "bus_kind": "request",
                        },
                    },
                )
                self.assertTrue(out.get("success"))

                inv = _post_json(
                    f"http://127.0.0.1:{port}/semops/invoke",
                    {
                        "actor": "test",
                        "req_id": "req-invoke-wait-1",
                        "operator_key": "MYOP",
                        "mode": "auto",
                        "payload": {"input": "x"},
                        "wait": True,
                        "timeout_s": 2.0,
                    },
                )
                self.assertTrue(inv.get("success"))
                self.assertEqual(inv.get("req_id"), "req-invoke-wait-1")
                resp = inv.get("response")
                self.assertIsInstance(resp, dict)
                self.assertEqual(resp.get("req_id"), "req-invoke-wait-1")
                self.assertEqual(resp.get("operator_key"), "MYOP")
            finally:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=2)
                except Exception:
                    server_proc.kill()
                try:
                    responder_proc.wait(timeout=3)
                except Exception:
                    responder_proc.kill()


if __name__ == "__main__":
    unittest.main()

