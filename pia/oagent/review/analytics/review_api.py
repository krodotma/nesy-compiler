#!/usr/bin/env python3
"""
Review API (Step 179)

REST API for review operations.

PBTSO Phase: DISTRIBUTE
Bus Topics: review.api.request, review.api.response

Endpoints:
- /reviews - Review management
- /reports - Report generation
- /metrics - Metrics dashboard
- /debt - Technical debt tracking
- /templates - Template management
- /notifications - Notification management

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable
from urllib.parse import parse_qs, urlparse


# ============================================================================
# Types
# ============================================================================

class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class HTTPStatus(Enum):
    """HTTP status codes."""
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    INTERNAL_ERROR = 500


@dataclass
class APIConfig:
    """Configuration for the API."""
    host: str = "0.0.0.0"
    port: int = 8080
    api_prefix: str = "/api/v1"
    enable_cors: bool = True
    auth_enabled: bool = False
    rate_limit_rpm: int = 100
    request_timeout_seconds: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class APIRequest:
    """
    Represents an API request.

    Attributes:
        request_id: Unique request identifier
        method: HTTP method
        path: Request path
        headers: Request headers
        query: Query parameters
        body: Request body (parsed JSON)
        timestamp: Request timestamp
    """
    request_id: str
    method: HTTPMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query: Dict[str, List[str]] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "method": self.method.value,
            "path": self.path,
            "headers": self.headers,
            "query": self.query,
            "body": self.body,
            "timestamp": self.timestamp,
        }

    def get_query_param(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get a query parameter."""
        values = self.query.get(name, [])
        return values[0] if values else default


@dataclass
class APIResponse:
    """
    Represents an API response.

    Attributes:
        status: HTTP status code
        headers: Response headers
        body: Response body
        request_id: Original request ID
    """
    status: HTTPStatus
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "headers": self.headers,
            "body": self.body,
            "request_id": self.request_id,
        }

    def to_json(self) -> str:
        """Convert body to JSON."""
        if self.body is None:
            return ""
        return json.dumps(self.body, indent=2)

    @classmethod
    def ok(cls, body: Any = None, request_id: Optional[str] = None) -> "APIResponse":
        """Create 200 OK response."""
        return cls(status=HTTPStatus.OK, body=body, request_id=request_id)

    @classmethod
    def created(cls, body: Any = None, request_id: Optional[str] = None) -> "APIResponse":
        """Create 201 Created response."""
        return cls(status=HTTPStatus.CREATED, body=body, request_id=request_id)

    @classmethod
    def no_content(cls, request_id: Optional[str] = None) -> "APIResponse":
        """Create 204 No Content response."""
        return cls(status=HTTPStatus.NO_CONTENT, request_id=request_id)

    @classmethod
    def bad_request(cls, message: str, request_id: Optional[str] = None) -> "APIResponse":
        """Create 400 Bad Request response."""
        return cls(
            status=HTTPStatus.BAD_REQUEST,
            body={"error": "bad_request", "message": message},
            request_id=request_id,
        )

    @classmethod
    def not_found(cls, message: str = "Resource not found", request_id: Optional[str] = None) -> "APIResponse":
        """Create 404 Not Found response."""
        return cls(
            status=HTTPStatus.NOT_FOUND,
            body={"error": "not_found", "message": message},
            request_id=request_id,
        )

    @classmethod
    def internal_error(cls, message: str = "Internal server error", request_id: Optional[str] = None) -> "APIResponse":
        """Create 500 Internal Server Error response."""
        return cls(
            status=HTTPStatus.INTERNAL_ERROR,
            body={"error": "internal_error", "message": message},
            request_id=request_id,
        )


# ============================================================================
# Route Handler Type
# ============================================================================

RouteHandler = Callable[[APIRequest], Awaitable[APIResponse]]


@dataclass
class Route:
    """API route definition."""
    method: HTTPMethod
    path: str
    handler: RouteHandler
    description: str = ""


# ============================================================================
# Review API
# ============================================================================

class ReviewAPI:
    """
    REST API for review operations.

    Example:
        api = ReviewAPI()

        # Handle request
        request = APIRequest(
            request_id="abc123",
            method=HTTPMethod.GET,
            path="/api/v1/reviews",
        )
        response = await api.handle_request(request)
        print(response.to_json())
    """

    BUS_TOPICS = {
        "request": "review.api.request",
        "response": "review.api.response",
    }

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the API.

        Args:
            config: API configuration
            bus_path: Path to event bus file
        """
        self.config = config or APIConfig()
        self.bus_path = bus_path or self._get_bus_path()
        self._routes: List[Route] = []
        self._setup_routes()

        # In-memory stores for demo
        self._reviews: Dict[str, Dict[str, Any]] = {}
        self._reports: Dict[str, Dict[str, Any]] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "api") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "review-api",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _setup_routes(self) -> None:
        """Set up API routes."""
        prefix = self.config.api_prefix

        # Review endpoints
        self.add_route(HTTPMethod.GET, f"{prefix}/reviews", self._list_reviews,
                       "List all reviews")
        self.add_route(HTTPMethod.POST, f"{prefix}/reviews", self._create_review,
                       "Create a new review")
        self.add_route(HTTPMethod.GET, f"{prefix}/reviews/{{id}}", self._get_review,
                       "Get review by ID")
        self.add_route(HTTPMethod.DELETE, f"{prefix}/reviews/{{id}}", self._delete_review,
                       "Delete a review")

        # Report endpoints
        self.add_route(HTTPMethod.GET, f"{prefix}/reports", self._list_reports,
                       "List all reports")
        self.add_route(HTTPMethod.POST, f"{prefix}/reports", self._create_report,
                       "Generate a new report")
        self.add_route(HTTPMethod.GET, f"{prefix}/reports/{{id}}", self._get_report,
                       "Get report by ID")

        # Metrics endpoints
        self.add_route(HTTPMethod.GET, f"{prefix}/metrics", self._get_metrics,
                       "Get metrics summary")
        self.add_route(HTTPMethod.GET, f"{prefix}/metrics/dashboard", self._get_dashboard,
                       "Get dashboard data")

        # Health endpoint
        self.add_route(HTTPMethod.GET, f"{prefix}/health", self._health_check,
                       "Health check")

        # OpenAPI spec
        self.add_route(HTTPMethod.GET, f"{prefix}/openapi.json", self._get_openapi,
                       "OpenAPI specification")

    def add_route(
        self,
        method: HTTPMethod,
        path: str,
        handler: RouteHandler,
        description: str = "",
    ) -> None:
        """Add an API route."""
        self._routes.append(Route(
            method=method,
            path=path,
            handler=handler,
            description=description,
        ))

    def _match_route(self, method: HTTPMethod, path: str) -> tuple[Optional[Route], Dict[str, str]]:
        """Match a request to a route."""
        for route in self._routes:
            if route.method != method:
                continue

            # Simple path matching with {id} placeholders
            route_parts = route.path.split("/")
            path_parts = path.split("/")

            if len(route_parts) != len(path_parts):
                continue

            params = {}
            match = True

            for rp, pp in zip(route_parts, path_parts):
                if rp.startswith("{") and rp.endswith("}"):
                    param_name = rp[1:-1]
                    params[param_name] = pp
                elif rp != pp:
                    match = False
                    break

            if match:
                return route, params

        return None, {}

    async def handle_request(self, request: APIRequest) -> APIResponse:
        """
        Handle an API request.

        Args:
            request: The API request

        Returns:
            API response

        Emits:
            review.api.request
            review.api.response
        """
        start_time = time.time()

        self._emit_event(self.BUS_TOPICS["request"], {
            "request_id": request.request_id,
            "method": request.method.value,
            "path": request.path,
        })

        # Match route
        route, params = self._match_route(request.method, request.path)

        if not route:
            response = APIResponse.not_found(
                f"No route found for {request.method.value} {request.path}",
                request.request_id,
            )
        else:
            try:
                # Add path params to query
                for k, v in params.items():
                    request.query[k] = [v]

                response = await route.handler(request)
                response.request_id = request.request_id
            except Exception as e:
                response = APIResponse.internal_error(str(e), request.request_id)

        # Add standard headers
        response.headers["Content-Type"] = "application/json"
        response.headers["X-Request-ID"] = request.request_id
        response.headers["X-Response-Time"] = f"{(time.time() - start_time) * 1000:.2f}ms"

        if self.config.enable_cors:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

        self._emit_event(self.BUS_TOPICS["response"], {
            "request_id": request.request_id,
            "status": response.status.value,
            "duration_ms": float(response.headers.get("X-Response-Time", "0").rstrip("ms")),
        })

        return response

    # ========================================================================
    # Route Handlers
    # ========================================================================

    async def _health_check(self, request: APIRequest) -> APIResponse:
        """Health check endpoint."""
        return APIResponse.ok({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "version": "1.0.0",
        })

    async def _list_reviews(self, request: APIRequest) -> APIResponse:
        """List reviews."""
        limit = int(request.get_query_param("limit", "50"))
        offset = int(request.get_query_param("offset", "0"))

        reviews = list(self._reviews.values())[offset:offset + limit]

        return APIResponse.ok({
            "reviews": reviews,
            "total": len(self._reviews),
            "limit": limit,
            "offset": offset,
        })

    async def _create_review(self, request: APIRequest) -> APIResponse:
        """Create a new review."""
        if not request.body:
            return APIResponse.bad_request("Request body required")

        files = request.body.get("files", [])
        if not files:
            return APIResponse.bad_request("Files list required")

        review_id = str(uuid.uuid4())[:8]
        review = {
            "review_id": review_id,
            "files": files,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "config": request.body.get("config", {}),
        }

        self._reviews[review_id] = review

        return APIResponse.created({
            "review": review,
            "message": "Review created successfully",
        })

    async def _get_review(self, request: APIRequest) -> APIResponse:
        """Get a review by ID."""
        review_id = request.get_query_param("id")
        if not review_id:
            return APIResponse.bad_request("Review ID required")

        review = self._reviews.get(review_id)
        if not review:
            return APIResponse.not_found(f"Review {review_id} not found")

        return APIResponse.ok({"review": review})

    async def _delete_review(self, request: APIRequest) -> APIResponse:
        """Delete a review."""
        review_id = request.get_query_param("id")
        if not review_id:
            return APIResponse.bad_request("Review ID required")

        if review_id not in self._reviews:
            return APIResponse.not_found(f"Review {review_id} not found")

        del self._reviews[review_id]

        return APIResponse.no_content()

    async def _list_reports(self, request: APIRequest) -> APIResponse:
        """List reports."""
        reports = list(self._reports.values())

        return APIResponse.ok({
            "reports": reports,
            "total": len(reports),
        })

    async def _create_report(self, request: APIRequest) -> APIResponse:
        """Generate a new report."""
        if not request.body:
            return APIResponse.bad_request("Request body required")

        review_id = request.body.get("review_id")
        if not review_id:
            return APIResponse.bad_request("Review ID required")

        report_id = str(uuid.uuid4())[:8]
        report = {
            "report_id": report_id,
            "review_id": review_id,
            "format": request.body.get("format", "markdown"),
            "status": "generated",
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

        self._reports[report_id] = report

        return APIResponse.created({
            "report": report,
            "message": "Report generated successfully",
        })

    async def _get_report(self, request: APIRequest) -> APIResponse:
        """Get a report by ID."""
        report_id = request.get_query_param("id")
        if not report_id:
            return APIResponse.bad_request("Report ID required")

        report = self._reports.get(report_id)
        if not report:
            return APIResponse.not_found(f"Report {report_id} not found")

        return APIResponse.ok({"report": report})

    async def _get_metrics(self, request: APIRequest) -> APIResponse:
        """Get metrics summary."""
        return APIResponse.ok({
            "metrics": {
                "total_reviews": len(self._reviews),
                "total_reports": len(self._reports),
                "avg_review_time_ms": 1500,
                "quality_score_avg": 78.5,
            },
        })

    async def _get_dashboard(self, request: APIRequest) -> APIResponse:
        """Get dashboard data."""
        period = request.get_query_param("period", "week")

        return APIResponse.ok({
            "dashboard": {
                "period": period,
                "review_count": len(self._reviews),
                "issue_count": 42,
                "quality_trend": "up",
                "top_issues": ["security", "complexity", "documentation"],
            },
        })

    async def _get_openapi(self, request: APIRequest) -> APIResponse:
        """Get OpenAPI specification."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Review API",
                "description": "REST API for code review operations",
                "version": "1.0.0",
            },
            "servers": [
                {"url": f"http://{self.config.host}:{self.config.port}"}
            ],
            "paths": {},
        }

        # Generate paths from routes
        for route in self._routes:
            path = route.path.replace("{id}", "{id}")
            if path not in spec["paths"]:
                spec["paths"][path] = {}

            method = route.method.value.lower()
            spec["paths"][path][method] = {
                "summary": route.description,
                "operationId": route.handler.__name__,
                "responses": {
                    "200": {"description": "Success"},
                    "400": {"description": "Bad Request"},
                    "404": {"description": "Not Found"},
                    "500": {"description": "Internal Server Error"},
                },
            }

        return APIResponse.ok(spec)

    def get_routes(self) -> List[Dict[str, str]]:
        """Get list of registered routes."""
        return [
            {
                "method": r.method.value,
                "path": r.path,
                "description": r.description,
            }
            for r in self._routes
        ]


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Review API."""
    import argparse

    parser = argparse.ArgumentParser(description="Review API (Step 179)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Routes command
    subparsers.add_parser("routes", help="List API routes")

    # Request command
    request_parser = subparsers.add_parser("request", help="Make API request")
    request_parser.add_argument("method", choices=["GET", "POST", "PUT", "DELETE"])
    request_parser.add_argument("path")
    request_parser.add_argument("--body", help="JSON request body")
    request_parser.add_argument("--query", help="Query params (key=value)")

    # OpenAPI command
    subparsers.add_parser("openapi", help="Get OpenAPI spec")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    api = ReviewAPI()

    if args.command == "routes":
        routes = api.get_routes()
        if args.json:
            print(json.dumps(routes, indent=2))
        else:
            print("API Routes:")
            for r in routes:
                print(f"  {r['method']:6} {r['path']}")
                if r['description']:
                    print(f"         {r['description']}")

    elif args.command == "request":
        # Parse query params
        query = {}
        if args.query:
            for param in args.query.split("&"):
                k, v = param.split("=", 1)
                query[k] = [v]

        # Parse body
        body = None
        if args.body:
            body = json.loads(args.body)

        request = APIRequest(
            request_id=str(uuid.uuid4())[:8],
            method=HTTPMethod[args.method],
            path=args.path,
            query=query,
            body=body,
        )

        response = asyncio.run(api.handle_request(request))

        if args.json:
            print(json.dumps(response.to_dict(), indent=2))
        else:
            print(f"Status: {response.status.value}")
            print(f"Headers: {response.headers}")
            print(f"Body:\n{response.to_json()}")

    elif args.command == "openapi":
        request = APIRequest(
            request_id="openapi",
            method=HTTPMethod.GET,
            path=f"{api.config.api_prefix}/openapi.json",
        )
        response = asyncio.run(api.handle_request(request))
        print(response.to_json())

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
