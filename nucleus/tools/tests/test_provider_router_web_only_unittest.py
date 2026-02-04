import importlib.util
import os
import pathlib
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


class TestProviderRouterWebOnly(unittest.TestCase):
    def test_pick_provider_uses_other_web_sessions_before_cli(self):
        router = _load_router_module()
        with patch.dict(os.environ, {"PLURIBUS_PROVIDER_PROFILE": "web-only"}, clear=False):
            with patch.object(router, "_vps_active_fallback", return_value=None):
                with patch.object(router, "_browser_tab_ready", side_effect=lambda pid: pid == "claude-web"):
                    with patch.object(router, "have_codex_cli", return_value=True):
                        picked = router.pick_provider("auto")
        self.assertEqual(picked, "claude-web")

    def test_candidate_providers_web_only_excludes_cli_and_mock(self):
        router = _load_router_module()
        with patch.dict(os.environ, {"PLURIBUS_PROVIDER_PROFILE": "web-only"}, clear=False):
            with patch.object(router, "_browser_tab_ready", side_effect=lambda pid: pid == "gemini-web"):
                with patch.object(router, "have_codex_cli", return_value=True):
                    with patch.object(router, "have_gemini", return_value=True):
                        with patch.object(router, "have_vertex_gemini", return_value=True):
                            with patch.object(router, "have_claude", return_value=True):
                                with patch.object(router, "have_claude_cli", return_value=True):
                                    candidates = router.candidate_providers("auto", model=None)
        self.assertTrue(candidates, "expected at least one candidate when a web tab is ready")
        self.assertTrue(all(c in {"chatgpt-web", "claude-web", "gemini-web"} for c in candidates))
        self.assertNotIn("mock", candidates)
        self.assertNotIn("codex-cli", candidates)

    def test_allowlist_filters_out_non_allowed_providers(self):
        router = _load_router_module()
        env = {
            "PLURIBUS_PROVIDER_PROFILE": "full",
            "PLURIBUS_ALLOWED_PROVIDERS": "chatgpt-web,claude-web,gemini-web,vertex-gemini",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch.object(router, "_browser_tab_ready", side_effect=lambda pid: pid == "chatgpt-web"):
                with patch.object(router, "have_vllm", return_value=True):
                    with patch.object(router, "have_ollama", return_value=True):
                        with patch.object(router, "have_tensorzero", return_value=True):
                            with patch.object(router, "have_codex_cli", return_value=True):
                                with patch.object(router, "have_claude", return_value=True):
                                    with patch.object(router, "have_claude_cli", return_value=True):
                                        with patch.object(router, "have_gemini", return_value=True):
                                            with patch.object(router, "have_vertex_gemini", return_value=True):
                                                candidates = router.candidate_providers("auto", model="gemini-3-pro")
        self.assertTrue(candidates, "expected candidates when allowlist and providers are available")
        self.assertTrue(all(c in {"chatgpt-web", "claude-web", "gemini-web", "vertex-gemini"} for c in candidates))
        self.assertNotIn("vllm-local", candidates)
        self.assertNotIn("ollama-local", candidates)
        self.assertNotIn("tensorzero", candidates)
        self.assertNotIn("codex-cli", candidates)

    def test_verified_profile_defaults_to_web_plus_vertex(self):
        router = _load_router_module()
        with patch.dict(os.environ, {"PLURIBUS_PROVIDER_PROFILE": "verified"}, clear=False):
            with patch.object(router, "_browser_tab_ready", return_value=False):
                with patch.object(router, "have_vertex_gemini", return_value=True):
                    candidates = router.candidate_providers("auto", model="gemini-3-pro")
        self.assertIn("vertex-gemini", candidates)
        self.assertTrue(all(c in {"chatgpt-web", "claude-web", "gemini-web", "vertex-gemini", "vertex-gemini-curl"} for c in candidates))


if __name__ == "__main__":
    unittest.main()
