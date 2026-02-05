#!/usr/bin/env python3
"""
Monitor API - Step 279

REST API for monitor operations.

PBTSO Phase: SKILL

Bus Topics:
- monitor.api.request (subscribed)
- monitor.api.response (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIStatus(Enum):
    """API status codes."""
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    INTERNAL_ERROR = 500


@dataclass
class APIRequest:
    """An API request.

    Attributes:
        request_id: Unique request ID
        method: HTTP method
        path: Request path
        query_params: Query parameters
        body: Request body
        headers: Request headers
        timestamp: Request timestamp
    """
    request_id: str
    method: HTTPMethod
    path: str
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "method": self.method.value,
            "path": self.path,
            "query_params": self.query_params,
            "body": self.body,
            "headers": {k: v for k, v in self.headers.items() if k.lower() != "authorization"},
            "timestamp": self.timestamp,
        }


@dataclass
class APIResponse:
    """An API response.

    Attributes:
        request_id: Request ID
        status: Status code
        body: Response body
        headers: Response headers
        duration_ms: Request duration
    """
    request_id: str
    status: APIStatus
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "body": self.body,
            "headers": self.headers,
            "duration_ms": self.duration_ms,
        }


@dataclass
class APIEndpoint:
    """An API endpoint.

    Attributes:
        path: Endpoint path
        method: HTTP method
        handler: Handler function
        description: Endpoint description
        auth_required: Whether auth is required
    """
    path: str
    method: HTTPMethod
    handler: Callable[..., Coroutine[Any, Any, APIResponse]]
    description: str = ""
    auth_required: bool = False


class MonitorAPI:
    """
    REST API for monitor operations.

    The API provides:
    - Metrics querying
    - Alert management
    - Dashboard operations
    - Report generation
    - Health checks

    Example:
        api = MonitorAPI()

        # Handle a request
        request = APIRequest(
            request_id="req-123",
            method=HTTPMethod.GET,
            path="/api/v1/metrics",
        )

        response = await api.handle_request(request)
    """

    BUS_TOPICS = {
        "request": "monitor.api.request",
        "response": "monitor.api.response",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    API_VERSION = "v1"
    BASE_PATH = f"/api/{API_VERSION}"

    def __init__(
        self,
        bus_dir: Optional[str] = None,
    ):
        """Initialize monitor API.

        Args:
            bus_dir: Bus directory
        """
        self._endpoints: Dict[Tuple[str, HTTPMethod], APIEndpoint] = {}
        self._request_history: List[Tuple[APIRequest, APIResponse]] = []
        self._last_heartbeat = time.time()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register endpoints
        self._register_endpoints()

    def register_endpoint(
        self,
        path: str,
        method: HTTPMethod,
        handler: Callable[..., Coroutine[Any, Any, APIResponse]],
        description: str = "",
        auth_required: bool = False,
    ) -> None:
        """Register an API endpoint.

        Args:
            path: Endpoint path
            method: HTTP method
            handler: Handler function
            description: Endpoint description
            auth_required: Whether auth is required
        """
        full_path = f"{self.BASE_PATH}{path}"
        endpoint = APIEndpoint(
            path=full_path,
            method=method,
            handler=handler,
            description=description,
            auth_required=auth_required,
        )
        self._endpoints[(full_path, method)] = endpoint

    async def handle_request(self, request: APIRequest) -> APIResponse:
        """Handle an API request.

        Args:
            request: API request

        Returns:
            API response
        """
        start_time = time.time()

        self._emit_bus_event(
            self.BUS_TOPICS["request"],
            request.to_dict()
        )

        # Find endpoint
        endpoint = self._endpoints.get((request.path, request.method))

        if not endpoint:
            # Try matching with path parameters
            endpoint = self._match_endpoint(request.path, request.method)

        if not endpoint:
            # Check if path exists with different method
            path_exists = any(
                p == request.path
                for (p, _) in self._endpoints.keys()
            )

            if path_exists:
                response = APIResponse(
                    request_id=request.request_id,
                    status=APIStatus.METHOD_NOT_ALLOWED,
                    body={"error": f"Method {request.method.value} not allowed"},
                )
            else:
                response = APIResponse(
                    request_id=request.request_id,
                    status=APIStatus.NOT_FOUND,
                    body={"error": f"Endpoint not found: {request.path}"},
                )
        else:
            try:
                response = await endpoint.handler(request)
            except Exception as e:
                response = APIResponse(
                    request_id=request.request_id,
                    status=APIStatus.INTERNAL_ERROR,
                    body={"error": str(e)},
                )

        response.duration_ms = (time.time() - start_time) * 1000

        self._request_history.append((request, response))
        if len(self._request_history) > 1000:
            self._request_history = self._request_history[-1000:]

        self._emit_bus_event(
            self.BUS_TOPICS["response"],
            {
                "request_id": request.request_id,
                "status": response.status.value,
                "duration_ms": response.duration_ms,
            }
        )

        return response

    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all registered endpoints.

        Returns:
            Endpoint summaries
        """
        return [
            {
                "path": ep.path,
                "method": ep.method.value,
                "description": ep.description,
                "auth_required": ep.auth_required,
            }
            for ep in self._endpoints.values()
        ]

    def get_request_history(
        self,
        limit: int = 50,
        path_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get request history.

        Args:
            limit: Maximum results
            path_filter: Filter by path

        Returns:
            Request history
        """
        history = self._request_history

        if path_filter:
            history = [
                (req, resp) for req, resp in history
                if path_filter in req.path
            ]

        return [
            {
                "request": req.to_dict(),
                "response": {
                    "status": resp.status.value,
                    "duration_ms": resp.duration_ms,
                },
            }
            for req, resp in list(reversed(history[-limit:]))
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get API statistics.

        Returns:
            Statistics
        """
        total_requests = len(self._request_history)

        by_status: Dict[int, int] = {}
        by_method: Dict[str, int] = {}
        by_path: Dict[str, int] = {}
        total_duration = 0.0

        for req, resp in self._request_history:
            status = resp.status.value
            by_status[status] = by_status.get(status, 0) + 1

            method = req.method.value
            by_method[method] = by_method.get(method, 0) + 1

            path = req.path
            by_path[path] = by_path.get(path, 0) + 1

            total_duration += resp.duration_ms

        avg_duration = total_duration / total_requests if total_requests else 0.0

        # Get top paths
        top_paths = sorted(by_path.items(), key=lambda x: -x[1])[:10]

        return {
            "total_requests": total_requests,
            "endpoints_registered": len(self._endpoints),
            "by_status": by_status,
            "by_method": by_method,
            "top_paths": dict(top_paths),
            "avg_duration_ms": avg_duration,
        }

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_api",
                "status": "healthy",
                "endpoints": len(self._endpoints),
            }
        )

        return True

    def _match_endpoint(
        self,
        path: str,
        method: HTTPMethod,
    ) -> Optional[APIEndpoint]:
        """Match endpoint with path parameters.

        Args:
            path: Request path
            method: HTTP method

        Returns:
            Matching endpoint or None
        """
        for (ep_path, ep_method), endpoint in self._endpoints.items():
            if ep_method != method:
                continue

            # Simple path parameter matching
            ep_parts = ep_path.split("/")
            path_parts = path.split("/")

            if len(ep_parts) != len(path_parts):
                continue

            match = True
            for ep_part, path_part in zip(ep_parts, path_parts):
                if ep_part.startswith("{") and ep_part.endswith("}"):
                    continue  # Path parameter, matches anything
                if ep_part != path_part:
                    match = False
                    break

            if match:
                return endpoint

        return None

    def _register_endpoints(self) -> None:
        """Register default API endpoints."""

        # Health endpoints
        async def health_check(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "status": "healthy",
                    "timestamp": time.time(),
                },
            )

        self.register_endpoint(
            "/health",
            HTTPMethod.GET,
            health_check,
            "Health check endpoint",
        )

        # Metrics endpoints
        async def list_metrics(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "metrics": [],
                    "count": 0,
                },
            )

        async def query_metric(request: APIRequest) -> APIResponse:
            metric_name = request.query_params.get("name", "")
            aggregation = request.query_params.get("agg", "avg")
            window = int(request.query_params.get("window", "300"))

            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "metric": metric_name,
                    "aggregation": aggregation,
                    "window_s": window,
                    "value": 0.0,
                },
            )

        self.register_endpoint(
            "/metrics",
            HTTPMethod.GET,
            list_metrics,
            "List all metrics",
        )

        self.register_endpoint(
            "/metrics/query",
            HTTPMethod.GET,
            query_metric,
            "Query metric value",
        )

        # Alert endpoints
        async def list_alerts(request: APIRequest) -> APIResponse:
            state = request.query_params.get("state", "firing")
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "alerts": [],
                    "state_filter": state,
                    "count": 0,
                },
            )

        async def get_alert(request: APIRequest) -> APIResponse:
            alert_id = request.path.split("/")[-1]
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "alert_id": alert_id,
                    "state": "firing",
                },
            )

        async def acknowledge_alert(request: APIRequest) -> APIResponse:
            alert_id = request.path.split("/")[-2]
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "alert_id": alert_id,
                    "acknowledged": True,
                },
            )

        async def resolve_alert(request: APIRequest) -> APIResponse:
            alert_id = request.path.split("/")[-2]
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "alert_id": alert_id,
                    "resolved": True,
                },
            )

        self.register_endpoint(
            "/alerts",
            HTTPMethod.GET,
            list_alerts,
            "List alerts",
        )

        self.register_endpoint(
            "/alerts/{id}",
            HTTPMethod.GET,
            get_alert,
            "Get alert by ID",
        )

        self.register_endpoint(
            "/alerts/{id}/acknowledge",
            HTTPMethod.POST,
            acknowledge_alert,
            "Acknowledge alert",
        )

        self.register_endpoint(
            "/alerts/{id}/resolve",
            HTTPMethod.POST,
            resolve_alert,
            "Resolve alert",
        )

        # Dashboard endpoints
        async def list_dashboards(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "dashboards": [],
                    "count": 0,
                },
            )

        async def get_dashboard(request: APIRequest) -> APIResponse:
            dashboard_id = request.path.split("/")[-1]
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "dashboard_id": dashboard_id,
                    "name": "Dashboard",
                    "panels": [],
                },
            )

        self.register_endpoint(
            "/dashboards",
            HTTPMethod.GET,
            list_dashboards,
            "List dashboards",
        )

        self.register_endpoint(
            "/dashboards/{id}",
            HTTPMethod.GET,
            get_dashboard,
            "Get dashboard by ID",
        )

        # Report endpoints
        async def list_reports(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "reports": [],
                    "count": 0,
                },
            )

        async def generate_report(request: APIRequest) -> APIResponse:
            report_type = request.body.get("type", "system_health") if request.body else "system_health"
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.CREATED,
                body={
                    "report_id": f"report-{uuid.uuid4().hex[:8]}",
                    "type": report_type,
                    "status": "generating",
                },
            )

        self.register_endpoint(
            "/reports",
            HTTPMethod.GET,
            list_reports,
            "List reports",
        )

        self.register_endpoint(
            "/reports",
            HTTPMethod.POST,
            generate_report,
            "Generate a report",
        )

        # SLO endpoints
        async def list_slos(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "slos": [],
                    "count": 0,
                },
            )

        self.register_endpoint(
            "/slos",
            HTTPMethod.GET,
            list_slos,
            "List SLOs",
        )

        # Incident endpoints
        async def list_incidents(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "incidents": [],
                    "count": 0,
                },
            )

        self.register_endpoint(
            "/incidents",
            HTTPMethod.GET,
            list_incidents,
            "List incidents",
        )

        # API docs endpoint
        async def get_api_docs(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status=APIStatus.OK,
                body={
                    "version": self.API_VERSION,
                    "endpoints": self.list_endpoints(),
                },
            )

        self.register_endpoint(
            "/docs",
            HTTPMethod.GET,
            get_api_docs,
            "API documentation",
        )

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_api: Optional[MonitorAPI] = None


def get_api() -> MonitorAPI:
    """Get or create the monitor API singleton.

    Returns:
        MonitorAPI instance
    """
    global _api
    if _api is None:
        _api = MonitorAPI()
    return _api


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor API (Step 279)")
    parser.add_argument("--endpoints", action="store_true", help="List endpoints")
    parser.add_argument("--request", metavar="PATH", help="Make a request")
    parser.add_argument("--method", default="GET", help="HTTP method")
    parser.add_argument("--body", help="Request body (JSON)")
    parser.add_argument("--history", action="store_true", help="Show request history")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    api = get_api()

    if args.endpoints:
        endpoints = api.list_endpoints()
        if args.json:
            print(json.dumps(endpoints, indent=2))
        else:
            print("API Endpoints:")
            for ep in endpoints:
                print(f"  {ep['method']:6} {ep['path']}")
                if ep['description']:
                    print(f"         {ep['description']}")

    if args.request:
        async def run():
            body = json.loads(args.body) if args.body else None
            request = APIRequest(
                request_id=f"req-{uuid.uuid4().hex[:8]}",
                method=HTTPMethod(args.method),
                path=args.request,
                body=body,
            )
            return await api.handle_request(request)

        response = asyncio.run(run())
        if args.json:
            print(json.dumps(response.to_dict(), indent=2))
        else:
            print(f"Response [{response.status.value}]:")
            print(f"  Duration: {response.duration_ms:.1f}ms")
            print(f"  Body: {json.dumps(response.body, indent=2)}")

    if args.history:
        history = api.get_request_history()
        if args.json:
            print(json.dumps(history, indent=2))
        else:
            print("Request History:")
            for h in history:
                req = h["request"]
                resp = h["response"]
                print(f"  [{resp['status']}] {req['method']} {req['path']} ({resp['duration_ms']:.1f}ms)")

    if args.stats:
        stats = api.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("API Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
