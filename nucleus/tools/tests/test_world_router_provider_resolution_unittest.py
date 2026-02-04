import unittest


class TestWorldRouterProviderResolution(unittest.TestCase):
    def test_resolve_provider_includes_vertex_models(self):
        from nucleus.tools.world_router import _resolve_provider

        self.assertEqual(_resolve_provider("vertex-gemini"), "vertex-gemini")
        self.assertEqual(_resolve_provider("vertex-gemini-curl"), "vertex-gemini-curl")

    def test_resolve_provider_gemini_3_defers_to_router(self):
        from nucleus.tools.world_router import _resolve_provider

        self.assertEqual(_resolve_provider("gemini-3-pro"), "auto")

    def test_circuit_auto_does_not_preselect_provider(self):
        from nucleus.tools.world_router import WorldRouterConfig, WorldRouterHandlers

        handlers = WorldRouterHandlers(WorldRouterConfig())
        ok, provider = handlers._check_circuit_and_route("auto")
        self.assertTrue(ok)
        self.assertEqual(provider, "auto")

    def test_circuit_auto_blocks_when_all_open(self):
        from nucleus.tools.world_router import WorldRouterConfig, WorldRouterHandlers

        handlers = WorldRouterHandlers(WorldRouterConfig())
        for p in ["chatgpt-web", "claude-web", "gemini-web", "vertex-gemini", "vertex-gemini-curl"]:
            handlers.circuit_breaker.record_failure(p)
            handlers.circuit_breaker.record_failure(p)
            handlers.circuit_breaker.record_failure(p)
        ok, provider = handlers._check_circuit_and_route("auto")
        self.assertFalse(ok)
        self.assertEqual(provider, "auto")

