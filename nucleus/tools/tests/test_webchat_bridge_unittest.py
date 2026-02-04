import json
from pathlib import Path

import pytest

from nucleus.tools import webchat_bridge


def test_webchat_readiness_populates_provider_statuses(tmp_path: Path) -> None:
    (tmp_path / ".pluribus").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".pluribus" / "browser_daemon.json").write_text(
        json.dumps(
            {
                "running": True,
                "tabs": {
                    "chatgpt-web": {"status": "ready"},
                    "claude-web": {"status": "needs_login"},
                },
            }
        ),
        encoding="utf-8",
    )

    readiness = webchat_bridge.webchat_readiness(root=tmp_path)

    assert readiness.daemon_running is True
    assert readiness.tab_status["chatgpt-web"] == "ready"
    assert readiness.tab_status["claude-web"] == "needs_login"
    assert readiness.tab_status["gemini-web"] == "missing"
    assert readiness.missing_tabs == ["gemini-web"]
    assert readiness.not_ready_tabs == ["claude-web"]


def test_send_webchat_prompt_wraps_plurichat_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeResp:
        success = True
        text = "OK\n"
        req_id = "req-123"
        latency_ms = 12.5

    def fake_exec(prompt: str, provider: str, bus_dir: Path, actor: str, timeout: float):
        assert prompt == "ping"
        assert provider == "chatgpt-web"
        assert actor == "tester"
        assert timeout == 1.0
        assert bus_dir == tmp_path
        return FakeResp()

    monkeypatch.setattr(webchat_bridge, "execute_web_session_inference", fake_exec)

    result = webchat_bridge.send_webchat_prompt(
        prompt="ping",
        provider="chatgpt-web",
        bus_dir=tmp_path,
        actor="tester",
        timeout_s=1.0,
    )

    assert result.ok is True
    assert result.req_id == "req-123"
    assert result.response_preview == "OK"
    assert result.error is None


def test_send_webchat_prompt_reports_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeResp:
        success = False
        text = ""
        req_id = "req-err"
        latency_ms = 0.0
        error = "timeout"

    def fake_exec(prompt: str, provider: str, bus_dir: Path, actor: str, timeout: float):
        return FakeResp()

    monkeypatch.setattr(webchat_bridge, "execute_web_session_inference", fake_exec)

    result = webchat_bridge.send_webchat_prompt(
        prompt="ping",
        provider="gemini-web",
        bus_dir=tmp_path,
        actor="tester",
        timeout_s=1.0,
    )

    assert result.ok is False
    assert result.req_id == "req-err"
    assert result.response_preview == ""
    assert result.error == "timeout"

