import os
import unittest
from unittest.mock import patch


class TestRouterModelAware(unittest.TestCase):
    def test_gemini3_prefers_vertex_when_full_profile(self):
        from nucleus.tools.providers import router

        def _which(name: str):
            return "/usr/bin/curl" if name == "curl" else None

        with patch.dict(os.environ, {"PLURIBUS_PROVIDER_PROFILE": "full"}, clear=False):
            with (
                patch.object(router, "_vps_active_fallback", return_value=None),
                patch.object(router, "_browser_tab_ready", return_value=False),
                patch.object(router, "_vllm_ready", return_value=False),
                patch.object(router, "have_vertex_gemini", return_value=True),
                patch.object(router, "have_codex_cli", return_value=True),
                patch.object(router.shutil, "which", side_effect=_which),
            ):
                got = router.pick_provider("auto", model="gemini-3-pro-preview")
                self.assertEqual(got, "vertex-gemini-curl")

    def test_gemini3_overrides_vps_codex_fallback(self):
        from nucleus.tools.providers import router

        def _which(name: str):
            return "/usr/bin/curl" if name == "curl" else None

        with patch.dict(os.environ, {"PLURIBUS_PROVIDER_PROFILE": "full"}, clear=False):
            with (
                patch.object(router, "_vps_active_fallback", return_value="codex-cli"),
                patch.object(router, "_map_fallback_to_provider", return_value="codex-cli"),
                patch.object(router, "_browser_tab_ready", return_value=False),
                patch.object(router, "_vllm_ready", return_value=False),
                patch.object(router, "have_vertex_gemini", return_value=True),
                patch.object(router, "have_codex_cli", return_value=True),
                patch.object(router.shutil, "which", side_effect=_which),
            ):
                got = router.pick_provider("auto", model="gemini-3-pro-preview")
                self.assertEqual(got, "vertex-gemini-curl")

    def test_gemini3_prefers_gemini_web_when_ready_in_web_only(self):
        from nucleus.tools.providers import router

        def tab_ready(provider_id: str) -> bool:
            return provider_id == "gemini-web"

        with patch.dict(os.environ, {"PLURIBUS_PROVIDER_PROFILE": "web-only"}, clear=False):
            with (
                patch.object(router, "_vps_active_fallback", return_value=None),
                patch.object(router, "_browser_tab_ready", side_effect=tab_ready),
                patch.object(router, "_vllm_ready", return_value=False),
            ):
                got = router.pick_provider("auto", model="gemini-3-pro-preview")
                self.assertEqual(got, "gemini-web")

    def test_require_local_only_limits_candidates(self):
        from nucleus.tools.providers import router

        with patch.dict(os.environ, {"PLURIBUS_ROUTER_REQUIRE_LOCAL": "1"}, clear=False):
            with (
                patch.object(router, "have_vllm", return_value=True),
                patch.object(router, "have_ollama", return_value=False),
                patch.object(router, "_allow_mock", return_value=False),
                # Must disable allowlist filtering to let vllm-local through
                patch.object(router, "_allowed_providers", return_value=None),
            ):
                got = router.candidate_providers("auto", model="gpt-5.2")
                self.assertEqual(got, ["vllm-local"])


if __name__ == "__main__":
    unittest.main()
