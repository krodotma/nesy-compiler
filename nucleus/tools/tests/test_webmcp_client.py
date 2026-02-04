#!/usr/bin/env python3
"""Tests for webmcp_client - WebMCP Bridge."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from webmcp_client import (
    WebMCPClient,
    WebMCPServer,
    WebMCPTransport,
    CachedToolDef,
)


@pytest.fixture
def temp_root():
    """Create a temporary root directory."""
    with tempfile.TemporaryDirectory(prefix="webmcp_test_") as tmpdir:
        root = Path(tmpdir)
        (root / ".pluribus" / "mcp").mkdir(parents=True)
        yield root


class TestWebMCPServer:
    """Tests for WebMCPServer dataclass."""

    def test_to_dict(self):
        """Server can be converted to dict safely (no token exposure)."""
        server = WebMCPServer(
            name="test-server",
            url="http://localhost:8000",
            transport="http",
            auth_type="bearer",
            auth_token="secret-token-123",
            headers={"X-Custom": "value"},
            timeout_s=60.0,
            enabled=True,
        )

        d = server.to_dict()
        assert d["name"] == "test-server"
        assert d["url"] == "http://localhost:8000"
        # Token should not be exposed
        assert d["auth_token"] is True  # Just indicates presence
        # Headers should only show keys
        assert "X-Custom" in d["headers"]

    def test_from_dict(self):
        """Server can be created from dict."""
        d = {
            "name": "server1",
            "url": "http://example.com",
            "transport": "websocket",
            "auth_type": "api_key",
            "auth_token": "key123",
            "headers": {"Accept": "application/json"},
            "timeout_s": 45.0,
            "enabled": False,
        }

        server = WebMCPServer.from_dict(d)
        assert server.name == "server1"
        assert server.url == "http://example.com"
        assert server.transport == "websocket"
        assert server.auth_type == "api_key"
        assert server.auth_token == "key123"
        assert server.timeout_s == 45.0
        assert server.enabled is False


class TestCachedToolDef:
    """Tests for CachedToolDef dataclass."""

    def test_to_dict(self):
        """Cached tool def can be converted to dict."""
        cached = CachedToolDef(
            server_name="server1",
            tool={"name": "test_tool", "description": "A test tool"},
            cached_at="2024-01-15T10:00:00Z",
            expires_at="2024-01-16T10:00:00Z",
        )

        d = cached.to_dict()
        assert d["kind"] == "webmcp_cached_tool"
        assert d["server_name"] == "server1"
        assert d["tool"]["name"] == "test_tool"


class TestWebMCPTransport:
    """Tests for WebMCPTransport."""

    def test_build_headers_bearer(self):
        """Builds headers with bearer auth."""
        server = WebMCPServer(
            name="test",
            url="http://localhost",
            auth_type="bearer",
            auth_token="token123",
        )
        transport = WebMCPTransport(server)
        headers = transport._build_headers()

        assert headers["Authorization"] == "Bearer token123"
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_api_key(self):
        """Builds headers with API key auth."""
        server = WebMCPServer(
            name="test",
            url="http://localhost",
            auth_type="api_key",
            auth_token="apikey456",
        )
        transport = WebMCPTransport(server)
        headers = transport._build_headers()

        assert headers["X-API-Key"] == "apikey456"

    def test_build_headers_custom(self):
        """Builds headers with custom headers."""
        server = WebMCPServer(
            name="test",
            url="http://localhost",
            headers={"X-Custom-Header": "custom-value"},
        )
        transport = WebMCPTransport(server)
        headers = transport._build_headers()

        assert headers["X-Custom-Header"] == "custom-value"


class TestWebMCPClient:
    """Tests for WebMCPClient."""

    def test_register_server(self, temp_root):
        """Can register a server."""
        client = WebMCPClient(temp_root)

        result = client.register_server(
            name="test-server",
            url="http://localhost:8000/mcp",
            transport="http",
            timeout_s=30.0,
        )

        assert "error" not in result
        assert result["name"] == "test-server"
        assert result["status"] == "registered"

        # Server should be in list
        servers = client.list_servers()
        assert servers["count"] == 1
        assert servers["servers"][0]["name"] == "test-server"

    def test_register_server_requires_name(self, temp_root):
        """Register fails without name."""
        client = WebMCPClient(temp_root)
        result = client.register_server(name="", url="http://localhost")
        assert "error" in result
        assert "name" in result["error"]

    def test_register_server_requires_url(self, temp_root):
        """Register fails without URL."""
        client = WebMCPClient(temp_root)
        result = client.register_server(name="test", url="")
        assert "error" in result
        assert "url" in result["error"]

    def test_unregister_server(self, temp_root):
        """Can unregister a server."""
        client = WebMCPClient(temp_root)

        # Register first
        client.register_server(name="to-remove", url="http://localhost")

        # Unregister
        result = client.unregister_server(name="to-remove")
        assert "error" not in result
        assert result["status"] == "unregistered"

        # Should not be in list
        servers = client.list_servers()
        assert servers["count"] == 0

    def test_unregister_nonexistent(self, temp_root):
        """Unregister fails for nonexistent server."""
        client = WebMCPClient(temp_root)
        result = client.unregister_server(name="nonexistent")
        assert "error" in result
        assert "not found" in result["error"]

    def test_list_servers_empty(self, temp_root):
        """List servers returns empty list when none registered."""
        client = WebMCPClient(temp_root)
        result = client.list_servers()
        assert result["count"] == 0
        assert result["servers"] == []

    def test_list_servers_multiple(self, temp_root):
        """List servers returns all registered servers."""
        client = WebMCPClient(temp_root)

        client.register_server(name="server1", url="http://localhost:8001")
        client.register_server(name="server2", url="http://localhost:8002")
        client.register_server(name="server3", url="http://localhost:8003")

        result = client.list_servers()
        assert result["count"] == 3
        names = {s["name"] for s in result["servers"]}
        assert names == {"server1", "server2", "server3"}

    def test_call_tool_unregistered_server(self, temp_root):
        """Call tool fails for unregistered server."""
        client = WebMCPClient(temp_root)
        result = client.call_tool(
            server_name="unknown",
            tool_name="some_tool",
        )
        assert "error" in result
        assert "not registered" in result["error"]

    def test_call_tool_requires_server_name(self, temp_root):
        """Call tool fails without server name."""
        client = WebMCPClient(temp_root)
        result = client.call_tool(server_name="", tool_name="tool")
        assert "error" in result
        assert "server_name" in result["error"]

    def test_call_tool_requires_tool_name(self, temp_root):
        """Call tool fails without tool name."""
        client = WebMCPClient(temp_root)
        client.register_server(name="server", url="http://localhost")
        result = client.call_tool(server_name="server", tool_name="")
        assert "error" in result
        assert "tool_name" in result["error"]

    @patch("webmcp_client.urllib.request.urlopen")
    def test_call_tool_success(self, mock_urlopen, temp_root):
        """Call tool returns result on success."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {"output": "success"},
        }).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = WebMCPClient(temp_root)
        client.register_server(name="server", url="http://localhost:8000")

        # Patch the request ID for matching
        with patch("webmcp_client.uuid.uuid4", return_value=MagicMock(hex="test-id", __str__=lambda s: "test-id")):
            result = client.call_tool(
                server_name="server",
                tool_name="test_tool",
                arguments={"arg1": "value1"},
            )

        assert "error" not in result
        assert "result" in result
        assert result["server"] == "server"
        assert result["tool"] == "test_tool"

    def test_health_check_unregistered(self, temp_root):
        """Health check reports unknown for unregistered server."""
        client = WebMCPClient(temp_root)
        result = client.health_check(server_name="unknown")

        assert result["count"] == 1
        assert result["health"][0]["status"] == "unknown"
        assert "not registered" in result["health"][0]["error"]

    def test_tools_list(self, temp_root):
        """Tools list returns expected tools."""
        client = WebMCPClient(temp_root)
        result = client.tools_list()

        assert "tools" in result
        tool_names = {t["name"] for t in result["tools"]}
        expected = {
            "webmcp_register",
            "webmcp_unregister",
            "webmcp_list_servers",
            "webmcp_list_tools",
            "webmcp_call",
            "webmcp_health",
        }
        assert expected.issubset(tool_names)

    def test_tools_call_register(self, temp_root):
        """Tools call dispatches register correctly."""
        client = WebMCPClient(temp_root)
        result = client.tools_call("webmcp_register", {
            "name": "via-mcp",
            "url": "http://localhost:9000",
        })
        assert "error" not in result
        assert result["status"] == "registered"

    def test_tools_call_unknown(self, temp_root):
        """Tools call returns error for unknown tool."""
        client = WebMCPClient(temp_root)
        result = client.tools_call("unknown_tool", {})
        assert "error" in result
        assert "unknown tool" in result["error"]

    def test_persistence(self, temp_root):
        """Server registrations persist across client instances."""
        # Register with first client
        client1 = WebMCPClient(temp_root)
        client1.register_server(name="persistent", url="http://localhost:8000")

        # Create new client instance
        client2 = WebMCPClient(temp_root)
        servers = client2.list_servers()

        assert servers["count"] == 1
        assert servers["servers"][0]["name"] == "persistent"


class TestWebMCPIntegration:
    """Integration tests for WebMCP components."""

    @patch("webmcp_client.urllib.request.urlopen")
    def test_list_tools_and_call(self, mock_urlopen, temp_root):
        """Can list tools and call them."""
        # Mock tools/list response
        tools_response = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "tools": [
                    {"name": "tool1", "description": "First tool"},
                    {"name": "tool2", "description": "Second tool"},
                ]
            },
        }

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(tools_response).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = WebMCPClient(temp_root)
        client.register_server(name="test-server", url="http://localhost:8000")

        # List tools
        with patch("webmcp_client.uuid.uuid4", return_value=MagicMock(hex="test-id", __str__=lambda s: "test-id")):
            result = client.list_tools(server_name="test-server", refresh=True)

        assert result["count"] == 2
        tool_names = {t["tool"]["name"] for t in result["tools"]}
        assert "tool1" in tool_names
        assert "tool2" in tool_names

    def test_disabled_server_not_called(self, temp_root):
        """Disabled servers are not called."""
        client = WebMCPClient(temp_root)

        # Register and then disable
        client.register_server(name="disabled", url="http://localhost:8000")
        client._servers["disabled"].enabled = False

        result = client.call_tool(
            server_name="disabled",
            tool_name="tool",
        )

        assert "error" in result
        assert "disabled" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
