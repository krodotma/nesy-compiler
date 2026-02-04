import importlib.util
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch


TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
ROUTER_PATH = TOOLS_DIR / "providers" / "router.py"


def _load_router_module():
    spec = importlib.util.spec_from_file_location("provider_router", ROUTER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


class TestProviderRouterAutostartDaemon(unittest.TestCase):
    def test_autostarts_browser_daemon_when_missing(self):
        router = _load_router_module()
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            # sentinel so _find_pluribus_root() accepts this as a repo root
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")
            env = {
                "PLURIBUS_PROVIDER_PROFILE": "web-only",
                "PLURIBUS_ROUTER_AUTOSTART_BROWSER_DAEMON": "1",
                "PLURIBUS_BUS_DIR": str(root / ".pluribus" / "bus"),
            }

            calls = []

            def fake_popen(*args, **kwargs):
                calls.append((args, kwargs))

                class P:
                    pid = 12345

                return P()

            with patch.dict(os.environ, env, clear=False):
                with patch.object(router, "_find_pluribus_root", return_value=root):
                    with patch.object(router, "emit_bus_event", return_value=None):
                        with patch.object(router.subprocess, "Popen", side_effect=fake_popen):
                            router._DAEMON_AUTOSTART_ATTEMPTED = False
                            router._ensure_browser_daemon_running()

            self.assertTrue(calls, "expected autostart to spawn browser_session_daemon.py")
            argv = calls[0][0][0]
            self.assertIn("browser_session_daemon.py", " ".join(argv))
            self.assertIn("start", argv)

    def test_autostart_respects_disable_flag(self):
        router = _load_router_module()
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")
            env = {
                "PLURIBUS_PROVIDER_PROFILE": "web-only",
                "PLURIBUS_ROUTER_AUTOSTART_BROWSER_DAEMON": "0",
                "PLURIBUS_BUS_DIR": str(root / ".pluribus" / "bus"),
            }
            with patch.dict(os.environ, env, clear=False):
                with patch.object(router, "_find_pluribus_root", return_value=root):
                    with patch.object(router.subprocess, "Popen") as popen:
                        router._DAEMON_AUTOSTART_ATTEMPTED = False
                        router._ensure_browser_daemon_running()
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
