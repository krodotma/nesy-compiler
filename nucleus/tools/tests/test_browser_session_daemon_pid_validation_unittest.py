import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestBrowserSessionDaemonPidValidation(unittest.TestCase):
    def test_load_state_clears_stale_pid_reuse(self) -> None:
        from nucleus.tools.browser_session_daemon import BrowserSessionDaemon

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / ".pluribus" / "browser_daemon.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "running": True,
                        "pid": os.getpid(),  # points at pytest, not the daemon
                        "started_at": "2025-01-01T00:00:00Z",
                        "tabs": {},
                    }
                ),
                encoding="utf-8",
            )

            daemon = BrowserSessionDaemon(root, root / ".pluribus" / "bus")
            state = daemon.load_state()

            self.assertFalse(state.running)
            self.assertEqual(state.pid, 0)

    def test_is_browser_session_daemon_pid_true_for_stub(self) -> None:
        from nucleus.tools import browser_session_daemon as mod

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stub = root / "browser_session_daemon.py"
            stub.write_text("import time\ntry:\n  time.sleep(60)\nexcept KeyboardInterrupt:\n  pass\n", encoding="utf-8")
            proc = subprocess.Popen(["python3", str(stub)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                self.assertTrue(mod._is_browser_session_daemon_pid(proc.pid))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()


if __name__ == "__main__":
    unittest.main()
