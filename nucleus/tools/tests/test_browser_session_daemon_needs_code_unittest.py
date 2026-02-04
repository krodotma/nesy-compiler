import pathlib
import tempfile
import unittest


class TestBrowserSessionDaemonNeedsCode(unittest.TestCase):
    def test_needs_code_counts_as_needs_action_in_vnc_status(self):
        tools_dir = pathlib.Path(__file__).resolve().parents[1]
        daemon_path = tools_dir / "browser_session_daemon.py"

        # Import via spec to avoid importing playwright at test collection time.
        import importlib.util

        spec = importlib.util.spec_from_file_location("browser_session_daemon", daemon_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            bus = root / ".pluribus" / "bus"
            bus.mkdir(parents=True, exist_ok=True)

            daemon = mod.BrowserSessionDaemon(root=root, bus_dir=bus)
            daemon.state.tabs = {
                "chatgpt-web": mod.TabSession(provider_id="chatgpt-web", tab_id="t1", url="https://chat.openai.com/", status="needs_code"),
                "claude-web": mod.TabSession(provider_id="claude-web", tab_id="t2", url="https://claude.ai/new", status="needs_login"),
                "gemini-web": mod.TabSession(provider_id="gemini-web", tab_id="t3", url="https://aistudio.google.com/", status="ready"),
            }

            st = daemon.get_vnc_status()
            providers = st.get("providers_status") or {}
            self.assertTrue(providers["chatgpt-web"]["needs_action"])
            self.assertTrue(providers["claude-web"]["needs_action"])
            self.assertFalse(providers["gemini-web"]["needs_action"])


if __name__ == "__main__":
    unittest.main()
