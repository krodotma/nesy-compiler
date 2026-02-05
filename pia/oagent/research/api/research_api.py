#!/usr/bin/env python3
"""
research_api.py - Research API (Step 29)

REST API for research queries and operations.
Provides HTTP endpoints for all research functionality.

PBTSO Phase: INTERFACE

Bus Topics:
- a2a.research.api.request
- a2a.research.api.response
- research.api.error

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class APIConfig:
    """Configuration for Research API."""

    host: str = "127.0.0.1"
    port: int = 8080
    enable_cors: bool = True
    api_key: Optional[str] = None  # Optional API key for auth
    rate_limit: int = 100  # Requests per minute
    timeout_seconds: int = 30
    log_requests: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Request/Response Models
# ============================================================================


@dataclass
class APIRequest:
    """API request context."""

    method: str
    path: str
    params: Dict[str, Any]
    body: Optional[Dict[str, Any]]
    headers: Dict[str, str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "path": self.path,
            "params": self.params,
            "timestamp": self.timestamp,
        }


@dataclass
class APIResponse:
    """API response."""

    status: int
    data: Any
    error: Optional[str] = None
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        if self.error:
            return {"error": self.error, "status": self.status}
        return {"data": self.data, "status": self.status}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# ============================================================================
# Research API
# ============================================================================


class ResearchAPI:
    """
    REST API server for research operations.

    Endpoints:
    - GET /health - Health check
    - POST /search - Execute a search query
    - GET /symbol/{name} - Find symbol definition
    - GET /usages/{name} - Find symbol usages
    - GET /context - Get assembled context
    - POST /feedback - Submit feedback
    - GET /stats - Get research statistics

    PBTSO Phase: INTERFACE

    Example:
        api = ResearchAPI()
        await api.start()

        # Or use programmatically
        response = await api.handle_search({"query": "UserService"})
    """

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the Research API.

        Args:
            config: API configuration
            bus: AgentBus for event emission
        """
        self.config = config or APIConfig()
        self.bus = bus or AgentBus()

        # Route handlers
        self._routes: Dict[str, Callable] = {}
        self._register_routes()

        # Rate limiting
        self._request_counts: Dict[str, List[float]] = {}

        # Server state
        self._server = None
        self._running = False

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time_ms": 0,
        }

    async def start(self) -> None:
        """Start the API server."""
        try:
            from aiohttp import web

            app = web.Application()

            # Add routes
            app.router.add_get("/health", self._handle_health)
            app.router.add_post("/search", self._handle_search)
            app.router.add_get("/symbol/{name}", self._handle_symbol)
            app.router.add_get("/usages/{name}", self._handle_usages)
            app.router.add_get("/context", self._handle_context)
            app.router.add_post("/feedback", self._handle_feedback)
            app.router.add_get("/stats", self._handle_stats)

            # Add CORS middleware if enabled
            if self.config.enable_cors:
                app.middlewares.append(self._cors_middleware)

            # Add auth middleware if API key configured
            if self.config.api_key:
                app.middlewares.append(self._auth_middleware)

            # Add logging middleware
            if self.config.log_requests:
                app.middlewares.append(self._logging_middleware)

            runner = web.AppRunner(app)
            await runner.setup()

            self._server = web.TCPSite(runner, self.config.host, self.config.port)
            await self._server.start()

            self._running = True
            print(f"Research API started on http://{self.config.host}:{self.config.port}")

        except ImportError:
            # Fall back to basic HTTP server
            print("aiohttp not available, using basic server")
            await self._start_basic_server()

    async def stop(self) -> None:
        """Stop the API server."""
        self._running = False
        if self._server:
            await self._server.stop()

    async def handle_request(self, request: APIRequest) -> APIResponse:
        """
        Handle an API request programmatically.

        Args:
            request: API request

        Returns:
            API response
        """
        start_time = time.time()

        self._emit_with_lock({
            "topic": "a2a.research.api.request",
            "kind": "api",
            "data": request.to_dict()
        })

        self._stats["total_requests"] += 1

        # Check rate limit
        if not self._check_rate_limit("default"):
            return APIResponse(status=429, error="Rate limit exceeded")

        # Find handler
        handler = self._routes.get(request.path)
        if not handler:
            self._stats["failed_requests"] += 1
            return APIResponse(status=404, error="Not found")

        try:
            result = await handler(request)
            duration = (time.time() - start_time) * 1000

            response = APIResponse(
                status=200,
                data=result,
                duration_ms=duration,
            )

            self._stats["successful_requests"] += 1
            self._update_avg_response_time(duration)

        except Exception as e:
            self._stats["failed_requests"] += 1
            response = APIResponse(
                status=500,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

            self._emit_with_lock({
                "topic": "research.api.error",
                "kind": "error",
                "level": "error",
                "data": {"error": str(e), "path": request.path}
            })

        self._emit_with_lock({
            "topic": "a2a.research.api.response",
            "kind": "api",
            "data": {
                "status": response.status,
                "duration_ms": response.duration_ms,
            }
        })

        return response

    # ========================================================================
    # Route Registration
    # ========================================================================

    def _register_routes(self) -> None:
        """Register API route handlers."""
        self._routes["/health"] = self._route_health
        self._routes["/search"] = self._route_search
        self._routes["/symbol"] = self._route_symbol
        self._routes["/usages"] = self._route_usages
        self._routes["/context"] = self._route_context
        self._routes["/feedback"] = self._route_feedback
        self._routes["/stats"] = self._route_stats

    # ========================================================================
    # Route Handlers (Programmatic)
    # ========================================================================

    async def _route_health(self, request: APIRequest) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._stats.get("start_time", time.time()),
        }

    async def _route_search(self, request: APIRequest) -> Dict[str, Any]:
        """Search endpoint."""
        body = request.body or {}
        query = body.get("query", request.params.get("query", ""))

        if not query:
            raise ValueError("Query parameter required")

        # This would integrate with the orchestrator
        # For now, return mock results
        return {
            "query": query,
            "results": [],
            "total": 0,
            "message": "Search not yet integrated - use orchestrator directly",
        }

    async def _route_symbol(self, request: APIRequest) -> Dict[str, Any]:
        """Symbol lookup endpoint."""
        name = request.params.get("name", "")

        if not name:
            raise ValueError("Symbol name required")

        return {
            "symbol": name,
            "results": [],
            "message": "Symbol search not yet integrated",
        }

    async def _route_usages(self, request: APIRequest) -> Dict[str, Any]:
        """Symbol usages endpoint."""
        name = request.params.get("name", "")

        if not name:
            raise ValueError("Symbol name required")

        return {
            "symbol": name,
            "usages": [],
            "message": "Usage search not yet integrated",
        }

    async def _route_context(self, request: APIRequest) -> Dict[str, Any]:
        """Context assembly endpoint."""
        params = request.params
        paths = params.get("paths", [])

        return {
            "paths": paths,
            "context": "",
            "tokens": 0,
            "message": "Context assembly not yet integrated",
        }

    async def _route_feedback(self, request: APIRequest) -> Dict[str, Any]:
        """Feedback submission endpoint."""
        body = request.body or {}

        return {
            "received": True,
            "feedback_id": "pending",
            "message": "Feedback processing not yet integrated",
        }

    async def _route_stats(self, request: APIRequest) -> Dict[str, Any]:
        """Statistics endpoint."""
        return {
            **self._stats,
            "timestamp": time.time(),
        }

    # ========================================================================
    # HTTP Handlers (aiohttp)
    # ========================================================================

    async def _handle_health(self, request) -> "web.Response":
        """HTTP handler for health check."""
        from aiohttp import web

        api_request = APIRequest(
            method="GET",
            path="/health",
            params={},
            body=None,
            headers=dict(request.headers),
        )
        response = await self.handle_request(api_request)
        return web.json_response(response.to_dict(), status=response.status)

    async def _handle_search(self, request) -> "web.Response":
        """HTTP handler for search."""
        from aiohttp import web

        body = await request.json() if request.body_exists else {}
        api_request = APIRequest(
            method="POST",
            path="/search",
            params=dict(request.query),
            body=body,
            headers=dict(request.headers),
        )
        response = await self.handle_request(api_request)
        return web.json_response(response.to_dict(), status=response.status)

    async def _handle_symbol(self, request) -> "web.Response":
        """HTTP handler for symbol lookup."""
        from aiohttp import web

        api_request = APIRequest(
            method="GET",
            path="/symbol",
            params={"name": request.match_info.get("name", "")},
            body=None,
            headers=dict(request.headers),
        )
        response = await self.handle_request(api_request)
        return web.json_response(response.to_dict(), status=response.status)

    async def _handle_usages(self, request) -> "web.Response":
        """HTTP handler for usages."""
        from aiohttp import web

        api_request = APIRequest(
            method="GET",
            path="/usages",
            params={"name": request.match_info.get("name", "")},
            body=None,
            headers=dict(request.headers),
        )
        response = await self.handle_request(api_request)
        return web.json_response(response.to_dict(), status=response.status)

    async def _handle_context(self, request) -> "web.Response":
        """HTTP handler for context."""
        from aiohttp import web

        api_request = APIRequest(
            method="GET",
            path="/context",
            params=dict(request.query),
            body=None,
            headers=dict(request.headers),
        )
        response = await self.handle_request(api_request)
        return web.json_response(response.to_dict(), status=response.status)

    async def _handle_feedback(self, request) -> "web.Response":
        """HTTP handler for feedback."""
        from aiohttp import web

        body = await request.json() if request.body_exists else {}
        api_request = APIRequest(
            method="POST",
            path="/feedback",
            params={},
            body=body,
            headers=dict(request.headers),
        )
        response = await self.handle_request(api_request)
        return web.json_response(response.to_dict(), status=response.status)

    async def _handle_stats(self, request) -> "web.Response":
        """HTTP handler for stats."""
        from aiohttp import web

        api_request = APIRequest(
            method="GET",
            path="/stats",
            params={},
            body=None,
            headers=dict(request.headers),
        )
        response = await self.handle_request(api_request)
        return web.json_response(response.to_dict(), status=response.status)

    # ========================================================================
    # Middleware
    # ========================================================================

    @staticmethod
    async def _cors_middleware(app, handler):
        """CORS middleware."""
        async def middleware_handler(request):
            from aiohttp import web

            if request.method == "OPTIONS":
                response = web.Response()
            else:
                response = await handler(request)

            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response

        return middleware_handler

    async def _auth_middleware(self, app, handler):
        """Authentication middleware."""
        async def middleware_handler(request):
            from aiohttp import web

            if request.path == "/health":
                return await handler(request)

            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                if token == self.config.api_key:
                    return await handler(request)

            return web.json_response(
                {"error": "Unauthorized"},
                status=401,
            )

        return middleware_handler

    async def _logging_middleware(self, app, handler):
        """Request logging middleware."""
        async def middleware_handler(request):
            start = time.time()
            response = await handler(request)
            duration = (time.time() - start) * 1000

            self._emit_with_lock({
                "topic": "a2a.research.api.request",
                "kind": "api",
                "data": {
                    "method": request.method,
                    "path": request.path,
                    "status": response.status,
                    "duration_ms": duration,
                }
            })

            return response

        return middleware_handler

    # ========================================================================
    # Helpers
    # ========================================================================

    async def _start_basic_server(self) -> None:
        """Start a basic HTTP server without aiohttp."""
        import http.server
        import socketserver

        class Handler(http.server.BaseHTTPRequestHandler):
            api = self

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "healthy", "note": "basic server"}')

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"message": "Use aiohttp for full API"}')

        with socketserver.TCPServer((self.config.host, self.config.port), Handler) as httpd:
            self._running = True
            httpd.serve_forever()

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        window_start = now - 60  # 1 minute window

        if client_id not in self._request_counts:
            self._request_counts[client_id] = []

        # Remove old requests
        self._request_counts[client_id] = [
            ts for ts in self._request_counts[client_id]
            if ts > window_start
        ]

        if len(self._request_counts[client_id]) >= self.config.rate_limit:
            return False

        self._request_counts[client_id].append(now)
        return True

    def _update_avg_response_time(self, duration_ms: float) -> None:
        """Update average response time."""
        total = self._stats["total_requests"]
        current_avg = self._stats["avg_response_time_ms"]

        # Exponential moving average
        if total == 1:
            self._stats["avg_response_time_ms"] = duration_ms
        else:
            self._stats["avg_response_time_ms"] = (
                0.9 * current_avg + 0.1 * duration_ms
            )

    def _emit_with_lock(self, event: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        import socket
        import uuid

        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": event.get("topic", "unknown"),
            "kind": event.get("kind", "event"),
            "level": event.get("level", "info"),
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": event.get("data", {}),
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(full_event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Research API."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Research API (Step 29)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    serve_parser.add_argument("--api-key", help="API key for authentication")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test API endpoints")
    test_parser.add_argument("--host", default="127.0.0.1", help="API host")
    test_parser.add_argument("--port", type=int, default=8080, help="API port")

    args = parser.parse_args()

    if args.command == "serve":
        config = APIConfig(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        api = ResearchAPI(config)

        async def run():
            await api.start()
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
            await api.stop()

        asyncio.run(run())

    elif args.command == "test":
        print(f"Testing API at http://{args.host}:{args.port}")

        api = ResearchAPI()

        async def run_tests():
            # Test health
            request = APIRequest(
                method="GET",
                path="/health",
                params={},
                body=None,
                headers={},
            )
            response = await api.handle_request(request)
            print(f"Health: {response.to_json()}")

            # Test stats
            request = APIRequest(
                method="GET",
                path="/stats",
                params={},
                body=None,
                headers={},
            )
            response = await api.handle_request(request)
            print(f"Stats: {response.to_json()}")

        asyncio.run(run_tests())

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
