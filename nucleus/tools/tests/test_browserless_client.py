#!/usr/bin/env python3
"""Tests for browserless_client.py - HTTP Chrome API wrapper."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from browserless_client import (
    BrowserlessClient,
    BrowserlessConfig,
    ScrapeResult,
    emit_bus_event,
    screenshot_sync,
    scrape_sync,
)


class TestBrowserlessConfig:
    """Test BrowserlessConfig dataclass."""

    def test_default_config(self):
        """Default configuration should have sensible values."""
        config = BrowserlessConfig()
        assert config.base_url == "http://localhost:3000"
        assert config.token is None
        assert config.timeout == 30.0
        assert config.default_viewport == {"width": 1920, "height": 1080}
        assert config.default_wait_for is None
        assert config.user_agent is None

    def test_custom_config(self):
        """Custom configuration should be applied."""
        config = BrowserlessConfig(
            base_url="http://custom:8080",
            token="test-token",
            timeout=60.0,
            default_viewport={"width": 800, "height": 600},
        )
        assert config.base_url == "http://custom:8080"
        assert config.token == "test-token"
        assert config.timeout == 60.0
        assert config.default_viewport == {"width": 800, "height": 600}


class TestScrapeResult:
    """Test ScrapeResult dataclass."""

    def test_default_result(self):
        """Default result should have url and empty fields."""
        result = ScrapeResult(url="https://example.com")
        assert result.url == "https://example.com"
        assert result.html is None
        assert result.text is None
        assert result.elements == {}
        assert result.screenshot_b64 is None
        assert result.pdf_b64 is None
        assert result.error is None
        assert result.duration_ms == 0.0

    def test_result_with_data(self):
        """Result should store provided data."""
        result = ScrapeResult(
            url="https://example.com",
            html="<html><body>Test</body></html>",
            text="Test",
            elements={"h1": ["Header"]},
            duration_ms=150.5,
        )
        assert result.html == "<html><body>Test</body></html>"
        assert result.text == "Test"
        assert result.elements == {"h1": ["Header"]}
        assert result.duration_ms == 150.5


class TestBrowserlessClient:
    """Test BrowserlessClient class."""

    def test_init_with_string(self):
        """Initialize client with URL string."""
        client = BrowserlessClient("http://custom:9000")
        assert client.config.base_url == "http://custom:9000"

    def test_init_with_config(self):
        """Initialize client with config object."""
        config = BrowserlessConfig(base_url="http://test:3000", token="abc123")
        client = BrowserlessClient(config)
        assert client.config.base_url == "http://test:3000"
        assert client.config.token == "abc123"

    def test_init_with_none(self):
        """Initialize client with None uses defaults."""
        client = BrowserlessClient(None)
        assert client.config.base_url == "http://localhost:3000"


class TestBusEventEmission:
    """Test bus event emission."""

    def test_emit_bus_event_no_crash(self, temp_bus_dir):
        """Emit bus event should not crash even if bus is unavailable."""
        # This should not raise even if bus emission fails
        emit_bus_event(
            "browserless.test",
            {"url": "https://example.com", "success": True},
        )
        # Just verify no exception was raised


@pytest.mark.asyncio
class TestBrowserlessClientAsync:
    """Async tests for BrowserlessClient."""

    async def test_screenshot_mocked(self):
        """Screenshot should call correct endpoint with payload."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not available")

        mock_response = mock.MagicMock()
        mock_response.content = b"\x89PNG\r\n\x1a\n..."  # Fake PNG header
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_client = mock.AsyncMock()
            mock_client.post = mock.AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = BrowserlessClient()
            client._client = mock_client

            result = await client.screenshot("https://example.com")

            assert result == b"\x89PNG\r\n\x1a\n..."
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "/screenshot"
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["options"]["type"] == "png"

    async def test_pdf_mocked(self):
        """PDF should call correct endpoint with payload."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not available")

        mock_response = mock.MagicMock()
        mock_response.content = b"%PDF-1.4..."  # Fake PDF header
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_client = mock.AsyncMock()
            mock_client.post = mock.AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = BrowserlessClient()
            client._client = mock_client

            result = await client.pdf("https://example.com", format="Letter")

            assert result == b"%PDF-1.4..."
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "/pdf"
            payload = call_args[1]["json"]
            assert payload["url"] == "https://example.com"
            assert payload["options"]["format"] == "Letter"

    async def test_scrape_mocked(self):
        """Scrape should call content endpoint and return result."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not available")

        mock_response = mock.MagicMock()
        mock_response.text = "<html><body><h1>Test</h1></body></html>"
        mock_response.raise_for_status = mock.MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.json = mock.MagicMock(side_effect=json.JSONDecodeError("", "", 0))

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_client = mock.AsyncMock()
            mock_client.post = mock.AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = BrowserlessClient()
            client._client = mock_client

            result = await client.scrape(
                "https://example.com",
                include_html=True,
            )

            assert isinstance(result, ScrapeResult)
            assert result.url == "https://example.com"
            assert result.html == "<html><body><h1>Test</h1></body></html>"
            assert result.error is None

    async def test_health_check_healthy(self):
        """Health check should return healthy status."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not available")

        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"healthy": True, "version": "1.0"}
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = BrowserlessClient()
            client._client = mock_client

            result = await client.health_check()

            assert result["status"] == "healthy"
            assert result["data"] == {"healthy": True, "version": "1.0"}

    async def test_health_check_unhealthy(self):
        """Health check should return unhealthy on error."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not available")

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(side_effect=Exception("Connection refused"))
            MockClient.return_value = mock_client

            client = BrowserlessClient()
            client._client = mock_client

            result = await client.health_check()

            assert result["status"] == "unhealthy"
            assert "Connection refused" in result["error"]

    async def test_close(self):
        """Close should close the HTTP client."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not available")

        mock_client = mock.AsyncMock()
        mock_client.aclose = mock.AsyncMock()

        client = BrowserlessClient()
        client._client = mock_client

        await client.close()

        mock_client.aclose.assert_called_once()
        assert client._client is None


class TestCLIFunctions:
    """Test CLI command functions."""

    def test_build_parser(self):
        """Parser should have all expected subcommands."""
        from browserless_client import build_parser

        parser = build_parser()
        # Parser should not raise on help
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

    def test_screenshot_subcommand_parse(self):
        """Screenshot subcommand should parse arguments."""
        from browserless_client import build_parser

        parser = build_parser()
        args = parser.parse_args(["screenshot", "https://example.com", "-o", "out.png"])
        assert args.url == "https://example.com"
        assert args.output == "out.png"
        assert args.cmd == "screenshot"

    def test_scrape_subcommand_parse(self):
        """Scrape subcommand should parse arguments."""
        from browserless_client import build_parser

        parser = build_parser()
        args = parser.parse_args(["scrape", "https://example.com", "-e", "h1,p", "--json"])
        assert args.url == "https://example.com"
        assert args.elements == "h1,p"
        assert args.json is True
        assert args.cmd == "scrape"

    def test_health_subcommand_parse(self):
        """Health subcommand should parse arguments."""
        from browserless_client import build_parser

        parser = build_parser()
        args = parser.parse_args(["health"])
        assert args.cmd == "health"


class TestServiceRegistryEntry:
    """Test that browserless is registered in service registry."""

    def test_browserless_in_registry(self):
        """Browserless should be in BUILTIN_SERVICES."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from service_registry import BUILTIN_SERVICES

        browserless_entries = [s for s in BUILTIN_SERVICES if s["id"] == "browserless"]
        assert len(browserless_entries) == 1

        entry = browserless_entries[0]
        assert entry["name"] == "Browserless Chrome API"
        assert entry["kind"] == "port"
        assert entry["port"] == 3000
        assert "browser" in entry["tags"]
        assert "scraping" in entry["tags"]
        assert entry["lineage"] == "core.browser"

    def test_pyodide_in_registry(self):
        """Pyodide should be in BUILTIN_SERVICES."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from service_registry import BUILTIN_SERVICES

        pyodide_entries = [s for s in BUILTIN_SERVICES if s["id"] == "pyodide-runtime"]
        assert len(pyodide_entries) == 1

        entry = pyodide_entries[0]
        assert entry["name"] == "Pyodide WASM Runtime"
        assert "pyodide" in entry["tags"]
        assert "wasm" in entry["tags"]
        assert entry["lineage"] == "core.browser"
