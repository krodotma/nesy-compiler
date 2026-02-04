#!/usr/bin/env python3
"""
PluriChat Test Suite (unittest-only)
====================================

This repository runs in environments where installing extra Python deps may be unavailable.
Keep tests runnable with the standard library only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import plurichat  # noqa: E402


@contextmanager
def _temp_bus_dir() -> Path:
    with tempfile.TemporaryDirectory(prefix="plurichat_test_bus_") as td:
        bus_dir = Path(td) / "bus"
        bus_dir.mkdir(parents=True, exist_ok=True)
        (bus_dir / "events.ndjson").write_text("", encoding="utf-8")
        yield bus_dir


class TestQueryClassification(unittest.TestCase):
    def test_classify_query(self) -> None:
        self.assertEqual(plurichat.classify_query("Write a function to sort a list"), "code")
        self.assertEqual(plurichat.classify_query("Explain quantum computing"), "research")
        self.assertEqual(plurichat.classify_query("Write a poem about nature"), "creative")
        self.assertEqual(plurichat.classify_query("Analyze this dataset"), "analysis")
        self.assertEqual(plurichat.classify_query("Solve this equation"), "math")
        self.assertEqual(plurichat.classify_query("Hello"), "general")


class TestProviderSelection(unittest.TestCase):
    def test_auto_selection_returns_web_provider(self) -> None:
        checked_at = "2025-01-01T00:00:00Z"
        available = {
            "chatgpt-web": plurichat.ProviderStatus(name="chatgpt-web", available=True, model="gpt-5.2-turbo", checked_at=checked_at),
            "claude-web": plurichat.ProviderStatus(name="claude-web", available=True, model="claude-opus-4-5", checked_at=checked_at),
            "gemini-web": plurichat.ProviderStatus(name="gemini-web", available=True, model="gemini-3-pro", checked_at=checked_at),
        }
        route = plurichat.select_provider_for_query("Hello", "auto", available, include_lens=False)
        self.assertIn(route.provider, {"chatgpt-web", "claude-web", "gemini-web"})

    def test_explicit_web_provider_override(self) -> None:
        available = {
            "chatgpt-web": plurichat.ProviderStatus(name="chatgpt-web", available=True, model="gpt-5.2-turbo"),
            "claude-web": plurichat.ProviderStatus(name="claude-web", available=True, model="claude-opus-4-5"),
            "gemini-web": plurichat.ProviderStatus(name="gemini-web", available=True, model="gemini-3-pro"),
        }
        route = plurichat.select_provider_for_query("Hello", "claude-web", available, include_lens=False)
        self.assertEqual(route.provider, "claude-web")

    def test_no_providers_does_not_return_mock(self) -> None:
        available = {
            "chatgpt-web": plurichat.ProviderStatus(name="chatgpt-web", available=False, model="gpt-5.2-turbo", error="needs_login"),
            "claude-web": plurichat.ProviderStatus(name="claude-web", available=False, model="claude-opus-4-5", error="needs_login"),
            "gemini-web": plurichat.ProviderStatus(name="gemini-web", available=False, model="gemini-3-pro", error="needs_login"),
        }
        route = plurichat.select_provider_for_query("Hello", "auto", available, include_lens=False)
        self.assertIn(route.provider, {"chatgpt-web", "claude-web", "gemini-web"})


class TestBusEvents(unittest.TestCase):
    def test_emit_bus_appends_ndjson(self) -> None:
        with _temp_bus_dir() as bus_dir:
            evt_id = plurichat.emit_bus(bus_dir, topic="test.topic", kind="metric", level="info", actor="test", data={"k": "v"})
            lines = (bus_dir / "events.ndjson").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            event = json.loads(lines[0])
            self.assertEqual(event["id"], evt_id)
            self.assertEqual(event["topic"], "test.topic")
            self.assertEqual(event["kind"], "metric")
            self.assertEqual(event["level"], "info")
            self.assertEqual(event["actor"], "test")
            self.assertEqual(event["data"]["k"], "v")
            self.assertIn("ts", event)
            self.assertIn("iso", event)


class TestWebSessionInference(unittest.TestCase):
    def test_web_session_inference_never_falls_back_to_cli(self) -> None:
        with _temp_bus_dir() as bus_dir:
            with patch.object(plurichat, "check_browser_daemon_status", return_value={"running": False, "tabs": {}}):
                with patch.object(plurichat.subprocess, "run", side_effect=AssertionError("subprocess.run should not be called")):
                    resp = plurichat.execute_web_session_inference("Say hello", "chatgpt-web", bus_dir, "test", timeout=0.1)
            self.assertFalse(resp.success)
            self.assertNotIn("→", resp.provider)
            self.assertTrue(resp.error or resp.text)

    def test_web_session_routes_when_tab_exists(self) -> None:
        with _temp_bus_dir() as bus_dir:
            status = {"running": True, "tabs": {"chatgpt-web": {"status": "error"}}}
            with patch.object(plurichat, "check_browser_daemon_status", return_value=status):
                with patch.object(plurichat.time, "sleep", return_value=None):
                    resp = plurichat.execute_web_session_inference("Say hello", "chatgpt-web", bus_dir, "test", timeout=0.01)
            self.assertFalse(resp.success)
            events = [json.loads(line) for line in (bus_dir / "events.ndjson").read_text(encoding="utf-8").splitlines()]
            request_events = [e for e in events if e.get("topic") == "plurichat.web_session.request"]
            self.assertTrue(request_events)
            last_request = request_events[-1]
            self.assertEqual(last_request["data"]["routing"], "browser_daemon")
            self.assertFalse(last_request["data"]["tab_ready"])


class TestChatResponse(unittest.TestCase):
    def test_chat_response_dataclass(self) -> None:
        resp = plurichat.ChatResponse(
            text="Hello",
            provider="chatgpt-web",
            model="gpt-5.2-turbo",
            latency_ms=1.0,
            req_id="r1",
            success=True,
            error=None,
        )
        self.assertEqual(resp.text, "Hello")
        self.assertEqual(resp.provider, "chatgpt-web")
        self.assertTrue(resp.success)


class TestDirectExecution(unittest.TestCase):
    @patch("subprocess.run")
    def test_direct_execution_success(self, mock_run: MagicMock) -> None:
        with _temp_bus_dir() as bus_dir:
            mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
            resp = plurichat.execute_chat_direct("Say hello", "auto", bus_dir, "test", timeout=1.0)
            self.assertTrue(resp.success)
            self.assertEqual(resp.text, "OK")


class TestCommandHandling(unittest.TestCase):
    def test_provider_command_sets_web_provider(self) -> None:
        with _temp_bus_dir() as bus_dir:
            state = plurichat.ChatState(bus_dir=bus_dir, actor="test", provider="auto")
            plurichat.handle_command("/provider chatgpt-web", state)
            self.assertEqual(state.provider, "chatgpt-web")


class TestLiveness(unittest.TestCase):
    def test_get_all_provider_status_includes_web_providers(self) -> None:
        with patch.object(
            plurichat,
            "check_browser_daemon_status",
            return_value={
                "running": True,
                "tabs": {
                    "chatgpt-web": {"status": "ready", "last_health_check": "2025-01-01T00:00:00Z"},
                    "claude-web": {"status": "needs_login", "error": "login", "last_health_check": "2025-01-01T00:00:00Z"},
                    "gemini-web": {"status": "ready", "last_health_check": "2025-01-01T00:00:00Z"},
                },
            },
        ):
            statuses = plurichat.get_all_provider_status()
        self.assertTrue({"chatgpt-web", "claude-web", "gemini-web"}.issubset(set(statuses.keys())))
        self.assertTrue(statuses["chatgpt-web"].available)
        self.assertFalse(statuses["claude-web"].available)
        self.assertTrue(statuses["gemini-web"].available)


if __name__ == "__main__":
    unittest.main()
