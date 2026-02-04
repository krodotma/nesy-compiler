import json
import os
import socket
import subprocess
import tempfile
import threading
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
    with urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestGitServerBrowserVNC(unittest.TestCase):
    def test_vnc_focus_tab_roundtrip_via_bus(self):
        tools_dir = Path(__file__).resolve().parents[1]
        server = tools_dir / "git_server.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / ".pluribus"
            bus_dir = state_dir / "bus"
            events_path = bus_dir / "events.ndjson"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events_path.write_text("", encoding="utf-8")

            # Pretend the browser daemon is running so the handler will accept requests.
            # Use a tiny stub process whose cmdline contains "browser_session_daemon.py" so
            # git_server's liveness check passes without requiring Playwright.
            stub = root / "browser_session_daemon.py"
            stub.write_text("import time\ntry:\n  time.sleep(60)\nexcept KeyboardInterrupt:\n  pass\n", encoding="utf-8")
            stub_proc = subprocess.Popen(["python3", str(stub)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            (state_dir / "browser_daemon.json").write_text(
                json.dumps({"running": True, "pid": stub_proc.pid}) + "\n",
                encoding="utf-8",
            )

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(
                ["python3", str(server), "--root", str(root), "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                req_id = "req-test-focus-1"

                def _emit_response():
                    time.sleep(0.25)
                    ev = {
                        "id": "resp-1",
                        "ts": time.time(),
                        "topic": "browser.vnc.focus_tab.response",
                        "actor": "browser_daemon",
                        "data": {
                            "req_id": req_id,
                            "success": True,
                            "provider": "gemini-web",
                            "current_url": "https://aistudio.google.com/",
                            "title": "Gemini",
                            "message": "Focused gemini-web",
                        },
                    }
                    with events_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n")

                t = threading.Thread(target=_emit_response, daemon=True)
                t.start()

                out = _post_json(
                    f"http://127.0.0.1:{port}/browser/vnc/focus_tab",
                    {"provider": "gemini-web", "req_id": req_id, "timeout_s": 2},
                )
                self.assertTrue(out.get("success"))
                self.assertEqual(out.get("provider"), "gemini-web")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
                stub_proc.terminate()
                try:
                    stub_proc.wait(timeout=2)
                except Exception:
                    stub_proc.kill()

    def test_vnc_enable_roundtrip_via_bus(self):
        tools_dir = Path(__file__).resolve().parents[1]
        server = tools_dir / "git_server.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / ".pluribus"
            bus_dir = state_dir / "bus"
            events_path = bus_dir / "events.ndjson"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events_path.write_text("", encoding="utf-8")

            stub = root / "browser_session_daemon.py"
            stub.write_text("import time\ntry:\n  time.sleep(60)\nexcept KeyboardInterrupt:\n  pass\n", encoding="utf-8")
            stub_proc = subprocess.Popen(["python3", str(stub)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            (state_dir / "browser_daemon.json").write_text(
                json.dumps({"running": True, "pid": stub_proc.pid}) + "\n",
                encoding="utf-8",
            )

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(
                ["python3", str(server), "--root", str(root), "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                req_id = "req-test-vnc-enable-1"

                def _emit_response():
                    time.sleep(0.25)
                    ev = {
                        "id": "resp-enable-1",
                        "ts": time.time(),
                        "topic": "browser.vnc.enable.response",
                        "actor": "browser_daemon",
                        "data": {
                            "req_id": req_id,
                            "success": True,
                            "display": ":1",
                            "vnc_detected": True,
                            "message": "VNC mode enabled",
                        },
                    }
                    with events_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n")

                t = threading.Thread(target=_emit_response, daemon=True)
                t.start()

                out = _post_json(
                    f"http://127.0.0.1:{port}/browser/vnc/enable",
                    {"req_id": req_id, "timeout_s": 2},
                )
                self.assertTrue(out.get("success"))
                self.assertEqual(out.get("display"), ":1")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
                stub_proc.terminate()
                try:
                    stub_proc.wait(timeout=2)
                except Exception:
                    stub_proc.kill()


if __name__ == "__main__":
    unittest.main()
