import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import urlopen


def _free_port() -> int:
    try:
        s = socket.socket()
    except PermissionError:
        raise unittest.SkipTest("socket creation not permitted in this environment")
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


class TestGitServerHealth(unittest.TestCase):
    def test_health_endpoint(self):
        tools_dir = Path(__file__).resolve().parents[1]
        server = tools_dir / "git_server.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proc = subprocess.Popen(
                ["python3", str(server), "--root", str(root), "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.time() + 3.0
                ok = False
                while time.time() < deadline:
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as resp:
                            body = resp.read(200).decode("utf-8", errors="replace")
                        if '"ok"' in body:
                            ok = True
                            break
                    except Exception:
                        time.sleep(0.1)
                self.assertTrue(ok)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
