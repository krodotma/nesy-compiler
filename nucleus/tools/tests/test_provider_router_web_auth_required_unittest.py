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


class TestProviderRouterWebAuthRequired(unittest.TestCase):
    def test_emits_bus_request_when_no_web_sessions_ready(self):
        router = _load_router_module()

        calls = []

        def capture(topic: str, *, level: str, data: dict) -> None:
            calls.append((topic, level, data))

        with tempfile.TemporaryDirectory() as td:
            bus_dir = pathlib.Path(td)
            env = {
                "PLURIBUS_PROVIDER_PROFILE": "web-only",
                "PLURIBUS_BUS_DIR": str(bus_dir),
            }
            with patch.dict(os.environ, env, clear=False):
                with patch.object(router, "_browser_tab_ready", return_value=False):
                    with patch.object(router, "emit_bus_request", side_effect=capture):
                        code = router.main(["--provider", "auto", "--prompt", "hi", "--format", "json"])

        self.assertEqual(code, 2)
        self.assertTrue(calls, "expected router to emit a bus request when web sessions are unavailable")
        topics = [c[0] for c in calls]
        self.assertIn("providers.web.auth.required", topics)
        self.assertIn("plurichat.web_session.auth.required", topics)
        levels = [c[1] for c in calls]
        self.assertTrue(all(l == "warn" for l in levels))
        self.assertTrue(any("providers" in (c[2] or {}) for c in calls))


if __name__ == "__main__":
    unittest.main()
