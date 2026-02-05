#!/usr/bin/env python3
"""
Step 129: Test API

Provides REST API for test operations.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.api.request (subscribes)
- test.api.response (emits)

Dependencies: Steps 101-128 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import threading


# ============================================================================
# Data Types
# ============================================================================

class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIResponse:
    """API response."""
    status_code: int = 200
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        if self.error:
            return json.dumps({"error": self.error, "status": self.status_code})
        return json.dumps(self.body) if self.body is not None else ""


@dataclass
class APIRequest:
    """API request."""
    method: HTTPMethod
    path: str
    query_params: Dict[str, List[str]] = field(default_factory=dict)
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)

    def get_param(self, name: str, default: str = "") -> str:
        """Get a query parameter."""
        values = self.query_params.get(name, [])
        return values[0] if values else default


@dataclass
class APIConfig:
    """
    Configuration for the test API.

    Attributes:
        host: Server host
        port: Server port
        enable_cors: Enable CORS headers
        api_prefix: API path prefix
        rate_limit: Rate limit per minute
        auth_token: Optional authentication token
        output_dir: Output directory
    """
    host: str = "127.0.0.1"
    port: int = 8080
    enable_cors: bool = True
    api_prefix: str = "/api/v1"
    rate_limit: int = 100
    auth_token: Optional[str] = None
    output_dir: str = ".pluribus/test-agent/api"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "enable_cors": self.enable_cors,
            "api_prefix": self.api_prefix,
            "rate_limit": self.rate_limit,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class APIBus:
    """Bus interface for API with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            if client_id not in self._requests:
                self._requests[client_id] = []

            # Remove old requests
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > cutoff
            ]

            if len(self._requests[client_id]) >= self.max_requests:
                return False

            self._requests[client_id].append(now)
            return True


# ============================================================================
# Test API
# ============================================================================

class TestAPI:
    """
    REST API for test operations.

    Endpoints:
    - GET /api/v1/tests - List tests
    - POST /api/v1/tests/run - Run tests
    - GET /api/v1/tests/{id}/status - Get test status
    - GET /api/v1/runs - List test runs
    - GET /api/v1/runs/{id} - Get run details
    - GET /api/v1/coverage - Get coverage data
    - GET /api/v1/flaky - Get flaky tests
    - GET /api/v1/stats - Get statistics
    - GET /api/v1/health - Health check

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.api.request, test.api.response
    """

    BUS_TOPICS = {
        "request": "test.api.request",
        "response": "test.api.response",
    }

    def __init__(self, bus=None, config: Optional[APIConfig] = None):
        """
        Initialize the test API.

        Args:
            bus: Optional bus instance
            config: API configuration
        """
        self.bus = bus or APIBus()
        self.config = config or APIConfig()
        self._routes: Dict[str, Dict[HTTPMethod, Callable]] = {}
        self._rate_limiter = RateLimiter(max_requests=self.config.rate_limit)
        self._server: Optional[HTTPServer] = None
        self._running = False

        # Test agent components (lazy loaded)
        self._runner = None
        self._history = None
        self._cache = None
        self._dashboard = None

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Register routes
        self._register_routes()

    def _register_routes(self) -> None:
        """Register API routes."""
        prefix = self.config.api_prefix

        # Health
        self.route(f"{prefix}/health", HTTPMethod.GET, self._handle_health)

        # Tests
        self.route(f"{prefix}/tests", HTTPMethod.GET, self._handle_list_tests)
        self.route(f"{prefix}/tests/run", HTTPMethod.POST, self._handle_run_tests)

        # Runs
        self.route(f"{prefix}/runs", HTTPMethod.GET, self._handle_list_runs)
        self.route(f"{prefix}/runs/{id}", HTTPMethod.GET, self._handle_get_run)

        # Coverage
        self.route(f"{prefix}/coverage", HTTPMethod.GET, self._handle_coverage)

        # Flaky
        self.route(f"{prefix}/flaky", HTTPMethod.GET, self._handle_flaky)

        # Stats
        self.route(f"{prefix}/stats", HTTPMethod.GET, self._handle_stats)

        # Cache
        self.route(f"{prefix}/cache", HTTPMethod.GET, self._handle_cache_stats)
        self.route(f"{prefix}/cache/clear", HTTPMethod.POST, self._handle_cache_clear)

        # Dashboard
        self.route(f"{prefix}/dashboard", HTTPMethod.GET, self._handle_dashboard)

    def route(self, path: str, method: HTTPMethod, handler: Callable) -> None:
        """Register a route handler."""
        if path not in self._routes:
            self._routes[path] = {}
        self._routes[path][method] = handler

    def handle_request(self, request: APIRequest) -> APIResponse:
        """
        Handle an API request.

        Args:
            request: API request

        Returns:
            API response
        """
        # Emit request event
        self._emit_event("request", {
            "method": request.method.value,
            "path": request.path,
        })

        # Check rate limit
        client_id = request.headers.get("X-Client-ID", "default")
        if not self._rate_limiter.is_allowed(client_id):
            return APIResponse(
                status_code=429,
                error="Rate limit exceeded",
            )

        # Check authentication if configured
        if self.config.auth_token:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer ") or \
               auth_header[7:] != self.config.auth_token:
                return APIResponse(
                    status_code=401,
                    error="Unauthorized",
                )

        # Find handler
        handler = None
        route_params = {}

        # Check exact match first
        if request.path in self._routes:
            if request.method in self._routes[request.path]:
                handler = self._routes[request.path][request.method]

        # Check pattern matches
        if handler is None:
            for route_path, methods in self._routes.items():
                if "{" in route_path:
                    # Simple pattern matching
                    pattern_parts = route_path.split("/")
                    path_parts = request.path.split("/")

                    if len(pattern_parts) == len(path_parts):
                        match = True
                        for pp, rp in zip(path_parts, pattern_parts):
                            if rp.startswith("{") and rp.endswith("}"):
                                param_name = rp[1:-1]
                                route_params[param_name] = pp
                            elif pp != rp:
                                match = False
                                break

                        if match and request.method in methods:
                            handler = methods[request.method]
                            break

        if handler is None:
            return APIResponse(
                status_code=404,
                error=f"Not found: {request.path}",
            )

        try:
            response = handler(request, route_params)

            # Emit response event
            self._emit_event("response", {
                "method": request.method.value,
                "path": request.path,
                "status_code": response.status_code,
            })

            return response

        except Exception as e:
            return APIResponse(
                status_code=500,
                error=str(e),
            )

    # ========================================================================
    # Route Handlers
    # ========================================================================

    def _handle_health(self, request: APIRequest, params: Dict) -> APIResponse:
        """Health check endpoint."""
        return APIResponse(
            status_code=200,
            body={
                "status": "healthy",
                "timestamp": time.time(),
                "version": "1.0.0",
            },
        )

    def _handle_list_tests(self, request: APIRequest, params: Dict) -> APIResponse:
        """List available tests."""
        test_dir = request.get_param("dir", "tests/")
        pattern = request.get_param("pattern", "test_*.py")

        tests = []
        test_path = Path(test_dir)

        if test_path.exists():
            for test_file in test_path.rglob(pattern):
                tests.append({
                    "path": str(test_file),
                    "name": test_file.stem,
                    "modified": test_file.stat().st_mtime,
                })

        return APIResponse(
            status_code=200,
            body={
                "tests": tests,
                "count": len(tests),
            },
        )

    def _handle_run_tests(self, request: APIRequest, params: Dict) -> APIResponse:
        """Trigger test run."""
        body = request.body or {}
        test_paths = body.get("tests", ["tests/"])
        parallel = body.get("parallel", True)
        workers = body.get("workers", 4)

        run_id = str(uuid.uuid4())

        # Emit run request to bus
        self.bus.emit({
            "topic": "test.run.request",
            "kind": "api_request",
            "actor": "test-api",
            "data": {
                "run_id": run_id,
                "tests": test_paths,
                "parallel": parallel,
                "workers": workers,
            },
        })

        return APIResponse(
            status_code=202,
            body={
                "run_id": run_id,
                "status": "queued",
                "tests": test_paths,
            },
        )

    def _handle_list_runs(self, request: APIRequest, params: Dict) -> APIResponse:
        """List test runs."""
        limit = int(request.get_param("limit", "20"))

        # Load from history
        runs = []
        runs_dir = Path(self.config.output_dir).parent / "orchestration"

        if runs_dir.exists():
            for run_file in sorted(runs_dir.glob("orchestration_*.json"), reverse=True)[:limit]:
                try:
                    with open(run_file) as f:
                        data = json.load(f)
                        runs.append({
                            "run_id": data.get("run_id"),
                            "status": data.get("status"),
                            "started_at": data.get("started_at"),
                            "duration_s": data.get("duration_s"),
                        })
                except (json.JSONDecodeError, IOError):
                    pass

        return APIResponse(
            status_code=200,
            body={
                "runs": runs,
                "count": len(runs),
            },
        )

    def _handle_get_run(self, request: APIRequest, params: Dict) -> APIResponse:
        """Get run details."""
        run_id = params.get("id", "")

        # Find run file
        runs_dir = Path(self.config.output_dir).parent / "orchestration"
        run_file = runs_dir / f"orchestration_{run_id}.json"

        if not run_file.exists():
            return APIResponse(
                status_code=404,
                error=f"Run not found: {run_id}",
            )

        try:
            with open(run_file) as f:
                data = json.load(f)

            return APIResponse(
                status_code=200,
                body=data,
            )
        except (json.JSONDecodeError, IOError) as e:
            return APIResponse(
                status_code=500,
                error=str(e),
            )

    def _handle_coverage(self, request: APIRequest, params: Dict) -> APIResponse:
        """Get coverage data."""
        coverage_dir = Path(self.config.output_dir).parent / "coverage"

        if not coverage_dir.exists():
            return APIResponse(
                status_code=200,
                body={"coverage": None, "message": "No coverage data available"},
            )

        # Find latest coverage report
        coverage_files = sorted(coverage_dir.glob("coverage_*.json"), reverse=True)

        if not coverage_files:
            return APIResponse(
                status_code=200,
                body={"coverage": None, "message": "No coverage data available"},
            )

        try:
            with open(coverage_files[0]) as f:
                data = json.load(f)

            return APIResponse(
                status_code=200,
                body=data,
            )
        except (json.JSONDecodeError, IOError) as e:
            return APIResponse(
                status_code=500,
                error=str(e),
            )

    def _handle_flaky(self, request: APIRequest, params: Dict) -> APIResponse:
        """Get flaky tests."""
        flaky_file = Path(self.config.output_dir).parent / "flaky" / "flaky_tests.json"

        if not flaky_file.exists():
            return APIResponse(
                status_code=200,
                body={"flaky_tests": [], "count": 0},
            )

        try:
            with open(flaky_file) as f:
                data = json.load(f)

            flaky_tests = [
                {"name": name, **info}
                for name, info in data.items()
            ]

            return APIResponse(
                status_code=200,
                body={
                    "flaky_tests": flaky_tests,
                    "count": len(flaky_tests),
                },
            )
        except (json.JSONDecodeError, IOError) as e:
            return APIResponse(
                status_code=500,
                error=str(e),
            )

    def _handle_stats(self, request: APIRequest, params: Dict) -> APIResponse:
        """Get test statistics."""
        days = int(request.get_param("days", "7"))

        stats = {
            "period_days": days,
            "total_runs": 0,
            "total_tests": 0,
            "pass_rate": 0,
        }

        # Calculate from history if available
        history_file = Path(self.config.output_dir).parent / "history" / "test_history.db"

        if history_file.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(history_file))
                cursor = conn.cursor()

                cutoff = time.time() - (days * 24 * 3600)

                cursor.execute("""
                    SELECT COUNT(*), COUNT(DISTINCT run_id),
                           SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END)
                    FROM test_results WHERE timestamp >= ?
                """, (cutoff,))

                row = cursor.fetchone()
                if row:
                    stats["total_tests"] = row[0] or 0
                    stats["total_runs"] = row[1] or 0
                    if stats["total_tests"] > 0:
                        stats["pass_rate"] = (row[2] or 0) / stats["total_tests"] * 100

                conn.close()
            except Exception:
                pass

        return APIResponse(
            status_code=200,
            body=stats,
        )

    def _handle_cache_stats(self, request: APIRequest, params: Dict) -> APIResponse:
        """Get cache statistics."""
        cache_index = Path(self.config.output_dir).parent / "cache" / "cache_index.json"

        if not cache_index.exists():
            return APIResponse(
                status_code=200,
                body={"entries": 0, "hits": 0, "misses": 0, "hit_rate": 0},
            )

        try:
            with open(cache_index) as f:
                data = json.load(f)

            stats = data.get("stats", {})
            entries = len(data.get("entries", {}))
            hits = stats.get("hits", 0)
            misses = stats.get("misses", 0)
            hit_rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0

            return APIResponse(
                status_code=200,
                body={
                    "entries": entries,
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": hit_rate,
                },
            )
        except (json.JSONDecodeError, IOError) as e:
            return APIResponse(
                status_code=500,
                error=str(e),
            )

    def _handle_cache_clear(self, request: APIRequest, params: Dict) -> APIResponse:
        """Clear cache."""
        cache_index = Path(self.config.output_dir).parent / "cache" / "cache_index.json"

        if cache_index.exists():
            try:
                with open(cache_index, "w") as f:
                    json.dump({"entries": {}, "stats": {"hits": 0, "misses": 0}}, f)

                return APIResponse(
                    status_code=200,
                    body={"message": "Cache cleared"},
                )
            except IOError as e:
                return APIResponse(
                    status_code=500,
                    error=str(e),
                )

        return APIResponse(
            status_code=200,
            body={"message": "No cache to clear"},
        )

    def _handle_dashboard(self, request: APIRequest, params: Dict) -> APIResponse:
        """Get dashboard state."""
        dashboard_file = Path(self.config.output_dir).parent / "dashboard" / "dashboard_state.json"

        if not dashboard_file.exists():
            return APIResponse(
                status_code=200,
                body={"status": "idle", "metrics": {}},
            )

        try:
            with open(dashboard_file) as f:
                data = json.load(f)

            return APIResponse(
                status_code=200,
                body=data,
            )
        except (json.JSONDecodeError, IOError) as e:
            return APIResponse(
                status_code=500,
                error=str(e),
            )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.api.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "api",
            "actor": "test-agent",
            "data": data,
        })

    def start(self) -> None:
        """Start the API server."""
        api = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self._handle_request(HTTPMethod.GET)

            def do_POST(self):
                self._handle_request(HTTPMethod.POST)

            def do_PUT(self):
                self._handle_request(HTTPMethod.PUT)

            def do_DELETE(self):
                self._handle_request(HTTPMethod.DELETE)

            def _handle_request(self, method: HTTPMethod):
                # Parse URL
                parsed = urlparse(self.path)
                query_params = parse_qs(parsed.query)

                # Read body
                body = None
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length > 0:
                    raw_body = self.rfile.read(content_length)
                    try:
                        body = json.loads(raw_body)
                    except json.JSONDecodeError:
                        body = raw_body.decode()

                # Build request
                request = APIRequest(
                    method=method,
                    path=parsed.path,
                    query_params=query_params,
                    body=body,
                    headers=dict(self.headers),
                )

                # Handle request
                response = api.handle_request(request)

                # Send response
                self.send_response(response.status_code)
                self.send_header("Content-Type", "application/json")

                if api.config.enable_cors:
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

                for key, value in response.headers.items():
                    self.send_header(key, value)

                self.end_headers()

                response_body = response.to_json()
                if response_body:
                    self.wfile.write(response_body.encode())

            def log_message(self, format, *args):
                pass  # Suppress default logging

        self._server = HTTPServer(
            (self.config.host, self.config.port),
            RequestHandler,
        )

        self._running = True
        print(f"Test API server running at http://{self.config.host}:{self.config.port}")
        self._server.serve_forever()

    def stop(self) -> None:
        """Stop the API server."""
        self._running = False
        if self._server:
            self._server.shutdown()


def create_app(config: Optional[APIConfig] = None) -> TestAPI:
    """Create a TestAPI application."""
    return TestAPI(config=config)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test API."""
    import argparse

    parser = argparse.ArgumentParser(description="Test API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--no-cors", action="store_true", help="Disable CORS")
    parser.add_argument("--auth-token", help="Authentication token")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/api")

    args = parser.parse_args()

    config = APIConfig(
        host=args.host,
        port=args.port,
        enable_cors=not args.no_cors,
        auth_token=args.auth_token,
        output_dir=args.output,
    )

    api = TestAPI(config=config)

    try:
        api.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api.stop()


if __name__ == "__main__":
    main()
