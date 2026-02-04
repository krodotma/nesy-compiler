#!/usr/bin/env python3
"""WebMCP Client for Pluribus - Bridge to Browser-Based MCP Tools.

Provides MCP client functionality for web-based tools:
- Connect to WebMCP servers via WebSocket or HTTP
- Bridge web tools to Pluribus MCP ecosystem
- Cache and proxy tool definitions
- Handle authentication and sessions

Supports:
- HTTP JSON-RPC transport
- WebSocket transport (for real-time)
- Server-Sent Events for streaming

Effects: R(net), W(net), R(file), W(file)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


@dataclass
class WebMCPServer:
    """Configuration for a WebMCP server."""
    name: str
    url: str
    transport: str = "http"  # http, websocket, sse
    auth_type: str | None = None  # bearer, basic, api_key
    auth_token: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 30.0
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "kind": "webmcp_server",
            "name": self.name,
            "url": self.url,
            "transport": self.transport,
            "auth_type": self.auth_type,
            "auth_token": self.auth_token is not None,  # Don't expose token
            "headers": list(self.headers.keys()),  # Just header names
            "timeout_s": self.timeout_s,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WebMCPServer":
        return cls(
            name=str(d.get("name") or ""),
            url=str(d.get("url") or ""),
            transport=str(d.get("transport") or "http"),
            auth_type=d.get("auth_type") if isinstance(d.get("auth_type"), str) else None,
            auth_token=d.get("auth_token") if isinstance(d.get("auth_token"), str) else None,
            headers=d.get("headers") if isinstance(d.get("headers"), dict) else {},
            timeout_s=float(d.get("timeout_s") or 30.0),
            enabled=bool(d.get("enabled", True)),
        )


@dataclass
class CachedToolDef:
    """Cached tool definition from a WebMCP server."""
    server_name: str
    tool: dict
    cached_at: str
    expires_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "kind": "webmcp_cached_tool",
            "server_name": self.server_name,
            "tool": self.tool,
            "cached_at": self.cached_at,
            "expires_at": self.expires_at,
        }


class WebMCPTransport:
    """HTTP transport for WebMCP."""

    def __init__(self, server: WebMCPServer):
        self.server = server

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.server.headers,
        }
        if self.server.auth_type == "bearer" and self.server.auth_token:
            headers["Authorization"] = f"Bearer {self.server.auth_token}"
        elif self.server.auth_type == "basic" and self.server.auth_token:
            encoded = base64.b64encode(self.server.auth_token.encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"
        elif self.server.auth_type == "api_key" and self.server.auth_token:
            headers["X-API-Key"] = self.server.auth_token
        return headers

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        """Send a JSON-RPC request to the WebMCP server."""
        timeout = timeout_s or self.server.timeout_s

        req_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        body = json.dumps(payload).encode("utf-8")
        headers = self._build_headers()

        req = urllib.request.Request(
            self.server.url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                result = json.loads(resp_body)

                if result.get("id") != req_id:
                    return {"error": "response id mismatch"}
                if "error" in result:
                    return {"error": result["error"]}
                return result.get("result", {})

        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except urllib.error.URLError as e:
            return {"error": f"URL error: {e.reason}"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {e}"}
        except Exception as e:
            return {"error": str(e)}

    def health_check(self) -> dict:
        """Check if the server is healthy."""
        try:
            headers = self._build_headers()
            req = urllib.request.Request(
                self.server.url.rstrip("/") + "/health",
                headers=headers,
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return {"status": "healthy", "code": resp.status}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class WebMCPClient:
    """Client for interacting with WebMCP servers."""

    def __init__(self, root: Path | str):
        self.root = Path(root) if isinstance(root, str) else root
        self._servers: dict[str, WebMCPServer] = {}
        self._transports: dict[str, WebMCPTransport] = {}
        self._load_servers()

    @property
    def servers_path(self) -> Path:
        return self.root / ".pluribus" / "mcp" / "webmcp_servers.json"

    @property
    def cache_path(self) -> Path:
        return self.root / ".pluribus" / "mcp" / "webmcp_cache.ndjson"

    @property
    def log_path(self) -> Path:
        return self.root / ".pluribus" / "mcp" / "webmcp_log.ndjson"

    def _load_servers(self) -> None:
        """Load server configurations from file."""
        if not self.servers_path.exists():
            return
        try:
            data = json.loads(self.servers_path.read_text(encoding="utf-8"))
            for name, cfg in data.get("servers", {}).items():
                cfg["name"] = name
                self._servers[name] = WebMCPServer.from_dict(cfg)
        except Exception:
            pass

    def _save_servers(self) -> None:
        """Save server configurations to file."""
        ensure_dir(self.servers_path.parent)
        data = {"servers": {}}
        for name, server in self._servers.items():
            data["servers"][name] = {
                "url": server.url,
                "transport": server.transport,
                "auth_type": server.auth_type,
                "auth_token": server.auth_token,
                "headers": server.headers,
                "timeout_s": server.timeout_s,
                "enabled": server.enabled,
            }
        self.servers_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _get_transport(self, server_name: str) -> WebMCPTransport | None:
        """Get or create transport for a server."""
        if server_name not in self._servers:
            return None
        if server_name not in self._transports:
            self._transports[server_name] = WebMCPTransport(self._servers[server_name])
        return self._transports[server_name]

    def _log_call(self, server: str, method: str, success: bool, duration_ms: float) -> None:
        """Log a call for analytics."""
        append_ndjson(self.log_path, {
            "kind": "webmcp_call",
            "server": server,
            "method": method,
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "ts": time.time(),
            "iso": now_iso_utc(),
        })

    def register_server(
        self,
        *,
        name: str,
        url: str,
        transport: str = "http",
        auth_type: str | None = None,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict:
        """Register a new WebMCP server.

        Args:
            name: Unique server name
            url: Server URL (http/https endpoint)
            transport: Transport type (http, websocket, sse)
            auth_type: Authentication type (bearer, basic, api_key)
            auth_token: Authentication token/credentials
            headers: Additional headers
            timeout_s: Request timeout

        Returns:
            Dict with registration status
        """
        if not name:
            return {"error": "name is required"}
        if not url:
            return {"error": "url is required"}

        server = WebMCPServer(
            name=name,
            url=url,
            transport=transport,
            auth_type=auth_type,
            auth_token=auth_token,
            headers=headers or {},
            timeout_s=timeout_s,
            enabled=True,
        )

        self._servers[name] = server
        self._transports.pop(name, None)  # Clear cached transport
        self._save_servers()

        return {"name": name, "url": url, "status": "registered"}

    def unregister_server(self, *, name: str) -> dict:
        """Remove a WebMCP server registration."""
        if not name:
            return {"error": "name is required"}
        if name not in self._servers:
            return {"error": f"server not found: {name}"}

        del self._servers[name]
        self._transports.pop(name, None)
        self._save_servers()

        return {"name": name, "status": "unregistered"}

    def list_servers(self) -> dict:
        """List all registered WebMCP servers."""
        servers = []
        for server in self._servers.values():
            servers.append(server.to_dict())
        return {"servers": servers, "count": len(servers)}

    def list_tools(self, *, server_name: str | None = None, refresh: bool = False) -> dict:
        """List tools from WebMCP servers.

        Args:
            server_name: Specific server to query (or all if None)
            refresh: Force refresh from servers (bypass cache)

        Returns:
            Dict with tools list
        """
        now = now_iso_utc()

        # Check cache first if not refreshing
        cached_tools: list[dict] = []
        if not refresh:
            for obj in iter_ndjson(self.cache_path):
                if obj.get("kind") != "webmcp_cached_tool":
                    continue
                if server_name and obj.get("server_name") != server_name:
                    continue
                # Check expiration
                expires = obj.get("expires_at")
                if expires and expires < now:
                    continue
                cached_tools.append({
                    "server": obj.get("server_name"),
                    "tool": obj.get("tool"),
                    "cached": True,
                })

            if cached_tools:
                return {"tools": cached_tools, "count": len(cached_tools), "from_cache": True}

        # Query servers
        all_tools: list[dict] = []
        errors: list[dict] = []

        servers_to_query = [server_name] if server_name else list(self._servers.keys())

        for name in servers_to_query:
            if name not in self._servers:
                errors.append({"server": name, "error": "not registered"})
                continue

            server = self._servers[name]
            if not server.enabled:
                continue

            transport = self._get_transport(name)
            if not transport:
                errors.append({"server": name, "error": "no transport"})
                continue

            start = time.time()
            result = transport.request("tools/list")
            duration_ms = (time.time() - start) * 1000

            if "error" in result:
                self._log_call(name, "tools/list", False, duration_ms)
                errors.append({"server": name, "error": result["error"]})
                continue

            self._log_call(name, "tools/list", True, duration_ms)

            tools = result.get("tools", [])
            for tool in tools:
                all_tools.append({
                    "server": name,
                    "tool": tool,
                    "cached": False,
                })
                # Cache the tool
                cache_entry = CachedToolDef(
                    server_name=name,
                    tool=tool,
                    cached_at=now,
                    expires_at=None,  # No expiration for now
                )
                append_ndjson(self.cache_path, cache_entry.to_dict())

        return {
            "tools": all_tools,
            "count": len(all_tools),
            "from_cache": False,
            "errors": errors if errors else None,
        }

    def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict | None = None,
        timeout_s: float | None = None,
    ) -> dict:
        """Call a tool on a WebMCP server.

        Args:
            server_name: Server to call
            tool_name: Tool name
            arguments: Tool arguments
            timeout_s: Optional timeout override

        Returns:
            Tool result
        """
        if not server_name:
            return {"error": "server_name is required"}
        if not tool_name:
            return {"error": "tool_name is required"}

        if server_name not in self._servers:
            return {"error": f"server not registered: {server_name}"}

        server = self._servers[server_name]
        if not server.enabled:
            return {"error": f"server disabled: {server_name}"}

        transport = self._get_transport(server_name)
        if not transport:
            return {"error": "no transport available"}

        start = time.time()
        result = transport.request(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
            timeout_s=timeout_s,
        )
        duration_ms = (time.time() - start) * 1000

        success = "error" not in result
        self._log_call(server_name, f"tools/call:{tool_name}", success, duration_ms)

        if success:
            return {
                "result": result,
                "server": server_name,
                "tool": tool_name,
                "duration_ms": round(duration_ms, 2),
            }
        return result

    def health_check(self, *, server_name: str | None = None) -> dict:
        """Check health of WebMCP servers.

        Args:
            server_name: Specific server to check (or all if None)

        Returns:
            Dict with health status
        """
        results: list[dict] = []
        servers_to_check = [server_name] if server_name else list(self._servers.keys())

        for name in servers_to_check:
            if name not in self._servers:
                results.append({"server": name, "status": "unknown", "error": "not registered"})
                continue

            transport = self._get_transport(name)
            if not transport:
                results.append({"server": name, "status": "error", "error": "no transport"})
                continue

            health = transport.health_check()
            results.append({"server": name, **health})

        return {"health": results, "count": len(results)}

    def tools_list(self) -> dict:
        """Return MCP tools list for the WebMCP client itself."""
        return {
            "tools": [
                {
                    "name": "webmcp_register",
                    "description": "Register a WebMCP server",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "url": {"type": "string"},
                            "transport": {"type": "string", "default": "http"},
                            "auth_type": {"type": "string"},
                            "auth_token": {"type": "string"},
                            "timeout_s": {"type": "number", "default": 30.0},
                        },
                        "required": ["name", "url"],
                    },
                },
                {
                    "name": "webmcp_unregister",
                    "description": "Remove a WebMCP server registration",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
                {
                    "name": "webmcp_list_servers",
                    "description": "List registered WebMCP servers",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "webmcp_list_tools",
                    "description": "List tools from WebMCP servers",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "server_name": {"type": "string"},
                            "refresh": {"type": "boolean", "default": False},
                        },
                    },
                },
                {
                    "name": "webmcp_call",
                    "description": "Call a tool on a WebMCP server",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "server_name": {"type": "string"},
                            "tool_name": {"type": "string"},
                            "arguments": {"type": "object"},
                            "timeout_s": {"type": "number"},
                        },
                        "required": ["server_name", "tool_name"],
                    },
                },
                {
                    "name": "webmcp_health",
                    "description": "Check health of WebMCP servers",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"server_name": {"type": "string"}},
                    },
                },
            ]
        }

    def tools_call(self, name: str, args: dict) -> dict:
        """Dispatch an MCP tool call."""
        if name == "webmcp_register":
            return self.register_server(
                name=str(args.get("name") or ""),
                url=str(args.get("url") or ""),
                transport=str(args.get("transport") or "http"),
                auth_type=args.get("auth_type") if isinstance(args.get("auth_type"), str) else None,
                auth_token=args.get("auth_token") if isinstance(args.get("auth_token"), str) else None,
                headers=args.get("headers") if isinstance(args.get("headers"), dict) else None,
                timeout_s=float(args.get("timeout_s") or 30.0),
            )
        if name == "webmcp_unregister":
            return self.unregister_server(name=str(args.get("name") or ""))
        if name == "webmcp_list_servers":
            return self.list_servers()
        if name == "webmcp_list_tools":
            return self.list_tools(
                server_name=args.get("server_name") if isinstance(args.get("server_name"), str) else None,
                refresh=bool(args.get("refresh")),
            )
        if name == "webmcp_call":
            return self.call_tool(
                server_name=str(args.get("server_name") or ""),
                tool_name=str(args.get("tool_name") or ""),
                arguments=args.get("arguments") if isinstance(args.get("arguments"), dict) else None,
                timeout_s=float(args.get("timeout_s")) if args.get("timeout_s") else None,
            )
        if name == "webmcp_health":
            return self.health_check(
                server_name=args.get("server_name") if isinstance(args.get("server_name"), str) else None,
            )
        return {"error": f"unknown tool: {name}"}


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def default_bus_dir() -> Path:
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()


def emit_bus_event(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    """Emit an event to the bus."""
    events_path = bus_dir / "events.ndjson"
    ensure_dir(events_path.parent)
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(events_path, evt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="webmcp_client.py", description="WebMCP client for Pluribus MCP bridge.")
    p.add_argument("--root", help="Pluribus root directory")
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    register = sub.add_parser("register", help="Register a WebMCP server")
    register.add_argument("--name", required=True)
    register.add_argument("--url", required=True)
    register.add_argument("--transport", default="http")
    register.add_argument("--auth-type", default=None, choices=["bearer", "basic", "api_key"])
    register.add_argument("--auth-token", default=None)
    register.add_argument("--timeout", type=float, default=30.0)

    unregister = sub.add_parser("unregister", help="Remove a server registration")
    unregister.add_argument("--name", required=True)

    list_servers = sub.add_parser("list-servers", help="List registered servers")

    list_tools = sub.add_parser("list-tools", help="List tools from servers")
    list_tools.add_argument("--server", default=None)
    list_tools.add_argument("--refresh", action="store_true")

    call = sub.add_parser("call", help="Call a tool")
    call.add_argument("--server", required=True)
    call.add_argument("--tool", required=True)
    call.add_argument("--arguments", default="{}", help="JSON arguments")
    call.add_argument("--timeout", type=float, default=None)

    health = sub.add_parser("health", help="Check server health")
    health.add_argument("--server", default=None)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    client = WebMCPClient(root)

    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else default_bus_dir()

    if args.cmd == "register":
        result = client.register_server(
            name=args.name,
            url=args.url,
            transport=args.transport,
            auth_type=args.auth_type,
            auth_token=args.auth_token,
            timeout_s=args.timeout,
        )
        print(json.dumps(result, indent=2))
        if "error" not in result:
            emit_bus_event(
                bus_dir,
                topic="webmcp.server.registered",
                kind="metric",
                level="info",
                actor="webmcp-client",
                data={"name": args.name, "url": args.url},
            )
        return 0 if "error" not in result else 1

    if args.cmd == "unregister":
        result = client.unregister_server(name=args.name)
        print(json.dumps(result, indent=2))
        return 0 if "error" not in result else 1

    if args.cmd == "list-servers":
        result = client.list_servers()
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "list-tools":
        result = client.list_tools(server_name=args.server, refresh=args.refresh)
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "call":
        try:
            arguments = json.loads(args.arguments)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid JSON arguments"}))
            return 1

        result = client.call_tool(
            server_name=args.server,
            tool_name=args.tool,
            arguments=arguments,
            timeout_s=args.timeout,
        )
        print(json.dumps(result, indent=2))

        # Emit bus event for tool calls
        emit_bus_event(
            bus_dir,
            topic="webmcp.tool.called",
            kind="metric",
            level="info" if "error" not in result else "warn",
            actor="webmcp-client",
            data={
                "server": args.server,
                "tool": args.tool,
                "success": "error" not in result,
            },
        )

        return 0 if "error" not in result else 1

    if args.cmd == "health":
        result = client.health_check(server_name=args.server)
        print(json.dumps(result, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
