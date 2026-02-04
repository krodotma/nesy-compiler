import os
import types
import unittest
from pathlib import Path
from unittest import mock


class TestGitServerBrowserBootstrapNoVNC(unittest.TestCase):
    def test_bootstrap_autostarts_novnc_when_ports_closed(self):
        # Import within test so patching works against the module object.
        from nucleus.tools import git_server as mod

        tools_dir = Path(__file__).resolve().parents[1]
        self.assertTrue((tools_dir / "novnc_start.sh").exists())
        self.assertTrue((tools_dir / "browser_session_daemon.py").exists())

        class FakeHandler:
            def __init__(self):
                self.tools_dir = tools_dir
                self.root = Path("/tmp/pluribus_test_bootstrap_root")
                (self.root / ".pluribus" / "logs").mkdir(parents=True, exist_ok=True)
                self._bus_reqs: list[tuple[str, str, dict]] = []
                self._emits: list[tuple[str, dict]] = []
                self._sent: dict | None = None

            def _browser_daemon_running(self) -> bool:
                return True

            def _append_bus_request(self, topic: str, actor: str, data: dict):
                self._bus_reqs.append((topic, actor, data))
                return "req-1"

            def _emit_bus(self, *, topic: str, kind: str, level: str, actor: str, data: dict):
                self._emits.append((topic, data))

            def send_json(self, data: dict):
                self._sent = data

        fake = FakeHandler()

        class FakeSocket:
            def settimeout(self, _t):
                return None

            def connect_ex(self, _addr):
                return 1  # port closed

            def close(self):
                return None

        class FakeProc:
            pid = 12345

        with (
            mock.patch.dict(os.environ, {"PLURIBUS_NOVNC_AUTOSTART": "1"}, clear=False),
            mock.patch.object(mod.socket, "socket", return_value=FakeSocket()),
            mock.patch.object(mod.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="", stderr="")),
            mock.patch.object(mod.subprocess, "Popen", return_value=FakeProc()) as popen,
        ):
            mod.GitFSHandler.handle_browser_bootstrap(fake, {"actor": "test", "providers": ["chatgpt-web"]})

        self.assertIsNotNone(fake._sent)
        self.assertTrue(fake._sent.get("success"))
        self.assertTrue(fake._sent.get("vnc_autostart_enabled"))
        self.assertTrue(fake._sent.get("vnc_started"))
        self.assertEqual(fake._sent.get("vnc_autostart_pid"), 12345)
        self.assertEqual((fake._sent.get("queued") or [])[0].get("provider"), "chatgpt-web")

        self.assertTrue(popen.called)
        all_calls = [" ".join(map(str, c.args[0])) for c in popen.call_args_list if c.args]
        self.assertTrue(any("novnc_start.sh" in c for c in all_calls), f"expected novnc_start.sh in popen calls, got: {all_calls}")


if __name__ == "__main__":
    unittest.main()
