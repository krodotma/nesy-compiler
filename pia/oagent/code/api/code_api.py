#!/usr/bin/env python3
"""
code_api.py - REST API for Code Operations (Step 79)

PBTSO Phase: All Phases

Provides:
- REST API endpoints for code operations
- Generate, format, lint, check types
- Async request handling
- Rate limiting and authentication
- OpenAPI documentation

Bus Topics:
- code.api.request
- code.api.response
- code.api.error

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for the Code API."""
    host: str = "0.0.0.0"
    port: int = 8079
    debug: bool = False
    enable_cors: bool = True
    rate_limit: int = 100  # requests per minute
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    timeout_s: int = 60
    api_key: Optional[str] = None  # Optional API key authentication
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "enable_cors": self.enable_cors,
            "rate_limit": self.rate_limit,
            "max_request_size": self.max_request_size,
            "timeout_s": self.timeout_s,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class APIRequest:
    """Incoming API request."""
    id: str
    method: str
    path: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "method": self.method,
            "path": self.path,
            "timestamp": self.timestamp,
        }


@dataclass
class APIResponse:
    """Outgoing API response."""
    status: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "body": self.body,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Simple rate limiter."""

    def __init__(self, limit: int = 100, window: int = 60):
        self.limit = limit
        self.window = window
        self._requests: Dict[str, List[float]] = {}

    def allow(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        cutoff = now - self.window

        if client_id not in self._requests:
            self._requests[client_id] = []

        # Remove old requests
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > cutoff
        ]

        # Check limit
        if len(self._requests[client_id]) >= self.limit:
            return False

        # Record request
        self._requests[client_id].append(now)
        return True


# =============================================================================
# Code API
# =============================================================================

class CodeAPI:
    """
    REST API for code operations.

    PBTSO Phase: All Phases

    Endpoints:
    - POST /api/v1/generate - Generate code
    - POST /api/v1/format - Format code
    - POST /api/v1/lint - Lint code
    - POST /api/v1/typecheck - Type check code
    - POST /api/v1/refactor - Refactor code
    - GET /api/v1/templates - List templates
    - GET /api/v1/snippets - Search snippets
    - GET /api/v1/health - Health check

    Usage:
        api = CodeAPI(config)
        await api.start()
    """

    BUS_TOPICS = {
        "request": "code.api.request",
        "response": "code.api.response",
        "error": "code.api.error",
        "heartbeat": "code.api.heartbeat",
    }

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or APIConfig()
        self.bus = bus or LockedAgentBus()
        self._rate_limiter = RateLimiter(self.config.rate_limit)
        self._routes: Dict[str, Dict[str, Callable]] = {}
        self._running = False

        # Components (lazy loaded)
        self._generator = None
        self._formatter = None
        self._linter = None
        self._type_checker = None
        self._refactorer = None
        self._template_engine = None
        self._snippet_manager = None

        self._register_routes()

    def _register_routes(self) -> None:
        """Register API routes."""
        self._routes = {
            "GET": {
                "/api/v1/health": self._handle_health,
                "/api/v1/templates": self._handle_list_templates,
                "/api/v1/snippets": self._handle_search_snippets,
                "/api/v1/stats": self._handle_stats,
            },
            "POST": {
                "/api/v1/generate": self._handle_generate,
                "/api/v1/format": self._handle_format,
                "/api/v1/lint": self._handle_lint,
                "/api/v1/typecheck": self._handle_typecheck,
                "/api/v1/refactor": self._handle_refactor,
                "/api/v1/docs": self._handle_docs,
            },
        }

    async def _load_components(self) -> None:
        """Lazy load components."""
        from ..generator import CodeGenerator
        from ..formatter import CodeFormatter
        from ..linter import UnifiedLinter
        from ..typechecker import TypeChecker
        from ..refactor import RefactoringEngine
        from ..template import TemplateEngine
        from ..snippet import SnippetManager

        self._generator = CodeGenerator()
        self._formatter = CodeFormatter()
        self._linter = UnifiedLinter()
        self._type_checker = TypeChecker()
        self._refactorer = RefactoringEngine()
        self._template_engine = TemplateEngine()
        self._snippet_manager = SnippetManager()

    async def handle_request(self, request: APIRequest) -> APIResponse:
        """
        Handle an API request.

        Args:
            request: Incoming request

        Returns:
            API response
        """
        start_time = time.time()

        # Emit request event
        self.bus.emit({
            "topic": self.BUS_TOPICS["request"],
            "kind": "api",
            "actor": "code-api",
            "data": request.to_dict(),
        })

        try:
            # Check rate limit
            client_id = request.headers.get("X-Client-ID", "anonymous")
            if not self._rate_limiter.allow(client_id):
                return APIResponse(
                    status=429,
                    body={"error": "Rate limit exceeded"},
                )

            # Check API key if configured
            if self.config.api_key:
                auth = request.headers.get("Authorization", "")
                if auth != f"Bearer {self.config.api_key}":
                    return APIResponse(
                        status=401,
                        body={"error": "Unauthorized"},
                    )

            # Find handler
            handlers = self._routes.get(request.method, {})
            handler = handlers.get(request.path)

            if not handler:
                return APIResponse(
                    status=404,
                    body={"error": f"Not found: {request.path}"},
                )

            # Execute handler
            response = await handler(request)

            # Emit response event
            elapsed_ms = (time.time() - start_time) * 1000
            self.bus.emit({
                "topic": self.BUS_TOPICS["response"],
                "kind": "api",
                "actor": "code-api",
                "data": {
                    "request_id": request.id,
                    "status": response.status,
                    "elapsed_ms": elapsed_ms,
                },
            })

            return response

        except Exception as e:
            # Emit error event
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "error",
                "actor": "code-api",
                "data": {
                    "request_id": request.id,
                    "error": str(e),
                },
            })

            return APIResponse(
                status=500,
                body={"error": str(e)},
            )

    # =========================================================================
    # Route Handlers
    # =========================================================================

    async def _handle_health(self, request: APIRequest) -> APIResponse:
        """Health check endpoint."""
        return APIResponse(
            status=200,
            body={
                "status": "healthy",
                "timestamp": time.time(),
                "version": "0.1.0",
            },
        )

    async def _handle_stats(self, request: APIRequest) -> APIResponse:
        """Get API statistics."""
        return APIResponse(
            status=200,
            body={
                "config": self.config.to_dict(),
                "routes": {
                    method: list(routes.keys())
                    for method, routes in self._routes.items()
                },
            },
        )

    async def _handle_generate(self, request: APIRequest) -> APIResponse:
        """Generate code endpoint."""
        if not self._generator:
            await self._load_components()

        body = request.body
        prompt = body.get("prompt", "")
        language = body.get("language", "python")
        pattern = body.get("pattern")

        from ..generator import GenerationRequest, CodePattern

        gen_request = GenerationRequest(
            id=request.id,
            prompt=prompt,
            language=language,
            pattern=CodePattern(pattern) if pattern else None,
            context=body.get("context", {}),
        )

        result = await self._generator.generate(gen_request)

        return APIResponse(
            status=200 if result.success else 400,
            body=result.to_dict(),
        )

    async def _handle_format(self, request: APIRequest) -> APIResponse:
        """Format code endpoint."""
        if not self._formatter:
            await self._load_components()

        body = request.body
        code = body.get("code", "")
        language = body.get("language", "python")
        file_path = body.get("file_path")

        if file_path:
            path = Path(file_path)
            result = self._formatter.format_file(path)
            return APIResponse(
                status=200 if result.success else 400,
                body=result.to_dict(),
            )
        else:
            # Format in-memory code
            import tempfile
            ext_map = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
            }
            ext = ext_map.get(language, ".py")

            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                f.write(code)
                temp_path = Path(f.name)

            try:
                result = self._formatter.format_file(temp_path)
                formatted_code = temp_path.read_text() if result.success else code
                return APIResponse(
                    status=200,
                    body={
                        "success": result.success,
                        "formatted_code": formatted_code,
                        "was_modified": result.was_modified,
                    },
                )
            finally:
                temp_path.unlink(missing_ok=True)

    async def _handle_lint(self, request: APIRequest) -> APIResponse:
        """Lint code endpoint."""
        if not self._linter:
            await self._load_components()

        body = request.body
        file_path = body.get("file_path")
        fix = body.get("fix", False)

        if not file_path:
            return APIResponse(
                status=400,
                body={"error": "file_path required"},
            )

        path = Path(file_path)

        if fix:
            result = self._linter.fix_file(path)
            return APIResponse(
                status=200,
                body=result.to_dict(),
            )
        else:
            result = self._linter.lint_file(path)
            return APIResponse(
                status=200,
                body=result.to_dict(),
            )

    async def _handle_typecheck(self, request: APIRequest) -> APIResponse:
        """Type check endpoint."""
        if not self._type_checker:
            await self._load_components()

        body = request.body
        file_path = body.get("file_path")

        if not file_path:
            return APIResponse(
                status=400,
                body={"error": "file_path required"},
            )

        path = Path(file_path)
        result = self._type_checker.check_file(path)

        return APIResponse(
            status=200,
            body=result.to_dict(),
        )

    async def _handle_refactor(self, request: APIRequest) -> APIResponse:
        """Refactor code endpoint."""
        if not self._refactorer:
            await self._load_components()

        body = request.body
        refactor_type = body.get("type", "rename")
        target = body.get("target", "")
        options = body.get("options", {})

        from ..refactor import RefactoringOperation, RefactoringType

        try:
            op_type = RefactoringType(refactor_type)
        except ValueError:
            return APIResponse(
                status=400,
                body={"error": f"Invalid refactoring type: {refactor_type}"},
            )

        operation = RefactoringOperation(
            id=request.id,
            refactor_type=op_type,
            target=target,
            options=options,
            scope=body.get("scope", []),
        )

        result = self._refactorer.refactor(operation)

        return APIResponse(
            status=200 if result.success else 400,
            body=result.to_dict(),
        )

    async def _handle_docs(self, request: APIRequest) -> APIResponse:
        """Documentation generation endpoint."""
        body = request.body
        file_path = body.get("file_path")
        output_format = body.get("format", "markdown")

        if not file_path:
            return APIResponse(
                status=400,
                body={"error": "file_path required"},
            )

        from ..docs import DocumentationGenerator, DocFormat

        generator = DocumentationGenerator()
        format_map = {
            "markdown": DocFormat.MARKDOWN,
            "json": DocFormat.JSON,
            "rst": DocFormat.RST,
        }

        result = generator.generate(
            Path(file_path),
            format_map.get(output_format, DocFormat.MARKDOWN),
        )

        return APIResponse(
            status=200 if result.success else 400,
            body=result.to_dict(),
        )

    async def _handle_list_templates(self, request: APIRequest) -> APIResponse:
        """List templates endpoint."""
        if not self._template_engine:
            await self._load_components()

        language = request.body.get("language")
        category = request.body.get("category")

        templates = self._template_engine.list_templates(
            language=language,
            category=category,
        )

        return APIResponse(
            status=200,
            body={
                "templates": [t.to_dict() for t in templates],
            },
        )

    async def _handle_search_snippets(self, request: APIRequest) -> APIResponse:
        """Search snippets endpoint."""
        if not self._snippet_manager:
            await self._load_components()

        query = request.body.get("query", "")
        language = request.body.get("language")
        limit = request.body.get("limit", 10)

        results = self._snippet_manager.search(
            query,
            language=language,
            limit=limit,
        )

        return APIResponse(
            status=200,
            body={
                "results": [r.to_dict() for r in results],
            },
        )

    def get_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Code Agent API",
                "version": "0.1.0",
                "description": "REST API for code operations",
            },
            "servers": [
                {"url": f"http://{self.config.host}:{self.config.port}"},
            ],
            "paths": {
                "/api/v1/health": {
                    "get": {
                        "summary": "Health check",
                        "responses": {
                            "200": {"description": "Healthy"},
                        },
                    },
                },
                "/api/v1/generate": {
                    "post": {
                        "summary": "Generate code",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prompt": {"type": "string"},
                                            "language": {"type": "string"},
                                            "pattern": {"type": "string"},
                                        },
                                        "required": ["prompt"],
                                    },
                                },
                            },
                        },
                        "responses": {
                            "200": {"description": "Generated code"},
                        },
                    },
                },
                "/api/v1/format": {
                    "post": {
                        "summary": "Format code",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "code": {"type": "string"},
                                            "language": {"type": "string"},
                                            "file_path": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {
                            "200": {"description": "Formatted code"},
                        },
                    },
                },
                "/api/v1/lint": {
                    "post": {
                        "summary": "Lint code",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "file_path": {"type": "string"},
                                            "fix": {"type": "boolean"},
                                        },
                                        "required": ["file_path"],
                                    },
                                },
                            },
                        },
                        "responses": {
                            "200": {"description": "Lint results"},
                        },
                    },
                },
                "/api/v1/typecheck": {
                    "post": {
                        "summary": "Type check code",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "file_path": {"type": "string"},
                                        },
                                        "required": ["file_path"],
                                    },
                                },
                            },
                        },
                        "responses": {
                            "200": {"description": "Type check results"},
                        },
                    },
                },
                "/api/v1/refactor": {
                    "post": {
                        "summary": "Refactor code",
                        "responses": {
                            "200": {"description": "Refactoring results"},
                        },
                    },
                },
                "/api/v1/templates": {
                    "get": {
                        "summary": "List templates",
                        "responses": {
                            "200": {"description": "Template list"},
                        },
                    },
                },
                "/api/v1/snippets": {
                    "get": {
                        "summary": "Search snippets",
                        "responses": {
                            "200": {"description": "Snippet search results"},
                        },
                    },
                },
            },
        }


def create_app(config: Optional[APIConfig] = None) -> CodeAPI:
    """Create a Code API instance."""
    return CodeAPI(config)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Code API."""
    import argparse

    parser = argparse.ArgumentParser(description="Code API Server (Step 79)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8079, help="Port to bind")
    serve_parser.add_argument("--debug", action="store_true", help="Debug mode")

    # openapi command
    subparsers.add_parser("openapi", help="Print OpenAPI spec")

    # test command
    test_parser = subparsers.add_parser("test", help="Test API endpoint")
    test_parser.add_argument("endpoint", help="Endpoint path (e.g., /api/v1/health)")
    test_parser.add_argument("--method", "-m", default="GET", help="HTTP method")
    test_parser.add_argument("--data", "-d", help="Request body (JSON)")

    args = parser.parse_args()

    config = APIConfig(
        host=getattr(args, "host", "0.0.0.0"),
        port=getattr(args, "port", 8079),
        debug=getattr(args, "debug", False),
    )
    api = CodeAPI(config)

    if args.command == "serve":
        print(f"Code API would serve on http://{config.host}:{config.port}")
        print("Note: Actual HTTP server requires aiohttp/fastapi dependency")
        print("\nAvailable endpoints:")
        for method, routes in api._routes.items():
            for path in routes:
                print(f"  {method} {path}")
        return 0

    elif args.command == "openapi":
        spec = api.get_openapi_spec()
        print(json.dumps(spec, indent=2))
        return 0

    elif args.command == "test":
        # Simulate request
        body = {}
        if args.data:
            try:
                body = json.loads(args.data)
            except json.JSONDecodeError:
                print(f"Invalid JSON: {args.data}")
                return 1

        request = APIRequest(
            id=f"test-{uuid.uuid4().hex[:8]}",
            method=args.method.upper(),
            path=args.endpoint,
            headers={},
            body=body,
        )

        async def test():
            await api._load_components()
            return await api.handle_request(request)

        response = asyncio.run(test())
        print(json.dumps(response.to_dict(), indent=2))
        return 0 if response.status == 200 else 1

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
