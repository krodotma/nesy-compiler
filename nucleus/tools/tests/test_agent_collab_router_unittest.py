from nucleus.tools import agent_collab_router as router


def test_normalize_providers_prefers_target():
    assert router.normalize_providers("claude", ["auto"]) == ["claude"]
    assert router.normalize_providers(["claude", "gemini"], ["auto"]) == ["claude", "gemini"]
    assert router.normalize_providers("auto", ["gemini"]) == ["gemini"]


def test_build_prompt_includes_intent_and_req_id():
    event = {
        "topic": "agent.collab.request",
        "actor": "codex",
        "iso": "2025-01-01T00:00:00Z",
        "data": {"req_id": "req_test_1", "intent": "Find repo"},
    }
    prompt = router.build_prompt(event)
    assert "req_test_1" in prompt
    assert "Find repo" in prompt
