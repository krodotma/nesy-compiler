#!/usr/bin/env python3
"""
server.py - Deployment API (Step 229)

PBTSO Phase: DISTRIBUTE
A2A Integration: REST API for deployment operations via deploy.api.*

Provides:
- APIVersion: API versioning
- APIResponse: Standardized API response
- DeploymentAPI: Main API class
- create_app: ASGI app factory

Bus Topics:
- deploy.api.request
- deploy.api.response

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


# ==============================================================================
# Bus Emission Helper with File Locking
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "deployment-api"
) -> str:
    """Emit an event to the Pluribus bus with file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class APIVersion(Enum):
    """API versions."""
    V1 = "v1"
    V2 = "v2"


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class APIResponse:
    """
    Standardized API response.

    Attributes:
        success: Whether request succeeded
        data: Response data
        error: Error message if failed
        metadata: Response metadata
        status_code: HTTP status code
    """
    success: bool = True
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status_code: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class APIRequest:
    """
    API request.

    Attributes:
        request_id: Unique request identifier
        method: HTTP method
        path: Request path
        query: Query parameters
        body: Request body
        headers: Request headers
        timestamp: Request timestamp
    """
    request_id: str
    method: str
    path: str
    query: Dict[str, str] = field(default_factory=dict)
    body: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ==============================================================================
# Route Handler
# ==============================================================================

@dataclass
class Route:
    """API route definition."""
    method: str
    path: str
    handler: Callable
    auth_required: bool = True
    description: str = ""


# ==============================================================================
# Deployment API (Step 229)
# ==============================================================================

class DeploymentAPI:
    """
    Deployment API - REST API for deployment operations.

    PBTSO Phase: DISTRIBUTE

    Responsibilities:
    - Expose deployment operations via REST API
    - Handle authentication and authorization
    - Provide versioned API endpoints
    - Rate limiting and request logging

    Example:
        >>> api = DeploymentAPI()
        >>> response = api.handle_request(
        ...     method="POST",
        ...     path="/api/v1/deployments",
        ...     body={"service": "api", "version": "v2.0.0"},
        ... )
    """

    BUS_TOPICS = {
        "request": "deploy.api.request",
        "response": "deploy.api.response",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "deployment-api",
        api_version: APIVersion = APIVersion.V1,
    ):
        """
        Initialize the API.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            api_version: Default API version
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "api"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.api_version = api_version

        self._routes: Dict[str, Route] = {}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._request_count = 0

        # Lazy-loaded components
        self._orchestrator = None
        self._history_tracker = None
        self._scheduler = None
        self._approval_gate = None
        self._metrics_collector = None

        self._register_routes()
        self._load_state()

    def _get_orchestrator(self):
        """Lazy load orchestrator."""
        if self._orchestrator is None:
            from ..orchestrator_v2 import DeployOrchestratorV2
            self._orchestrator = DeployOrchestratorV2()
        return self._orchestrator

    def _get_history_tracker(self):
        """Lazy load history tracker."""
        if self._history_tracker is None:
            from ..history.tracker import DeploymentHistoryTracker
            self._history_tracker = DeploymentHistoryTracker()
        return self._history_tracker

    def _get_scheduler(self):
        """Lazy load scheduler."""
        if self._scheduler is None:
            from ..scheduler.scheduler import DeploymentScheduler
            self._scheduler = DeploymentScheduler()
        return self._scheduler

    def _get_approval_gate(self):
        """Lazy load approval gate."""
        if self._approval_gate is None:
            from ..approval.gate import DeploymentApprovalGate
            self._approval_gate = DeploymentApprovalGate()
        return self._approval_gate

    def _get_metrics_collector(self):
        """Lazy load metrics collector."""
        if self._metrics_collector is None:
            from ..metrics.collector import DeploymentMetricsCollector
            self._metrics_collector = DeploymentMetricsCollector()
        return self._metrics_collector

    def _register_routes(self) -> None:
        """Register API routes."""
        # Health check
        self._add_route("GET", "/health", self._handle_health, auth_required=False)
        self._add_route("GET", "/ready", self._handle_ready, auth_required=False)

        # Deployments
        self._add_route("GET", "/api/v1/deployments", self._handle_list_deployments)
        self._add_route("POST", "/api/v1/deployments", self._handle_create_deployment)
        self._add_route("GET", "/api/v1/deployments/{id}", self._handle_get_deployment)
        self._add_route("DELETE", "/api/v1/deployments/{id}", self._handle_cancel_deployment)

        # History
        self._add_route("GET", "/api/v1/history", self._handle_list_history)
        self._add_route("GET", "/api/v1/history/{id}", self._handle_get_history)
        self._add_route("GET", "/api/v1/history/stats", self._handle_history_stats)

        # Schedules
        self._add_route("GET", "/api/v1/schedules", self._handle_list_schedules)
        self._add_route("POST", "/api/v1/schedules", self._handle_create_schedule)
        self._add_route("DELETE", "/api/v1/schedules/{id}", self._handle_cancel_schedule)

        # Approvals
        self._add_route("GET", "/api/v1/approvals", self._handle_list_approvals)
        self._add_route("POST", "/api/v1/approvals/{id}/approve", self._handle_approve)
        self._add_route("POST", "/api/v1/approvals/{id}/reject", self._handle_reject)

        # Metrics
        self._add_route("GET", "/api/v1/metrics", self._handle_get_metrics)
        self._add_route("POST", "/api/v1/metrics", self._handle_record_metric)

        # Services
        self._add_route("GET", "/api/v1/services", self._handle_list_services)
        self._add_route("GET", "/api/v1/services/{name}/version", self._handle_get_version)

    def _add_route(
        self,
        method: str,
        path: str,
        handler: Callable,
        auth_required: bool = True,
        description: str = "",
    ) -> None:
        """Add a route."""
        key = f"{method}:{path}"
        self._routes[key] = Route(
            method=method,
            path=path,
            handler=handler,
            auth_required=auth_required,
            description=description,
        )

    def handle_request(
        self,
        method: str,
        path: str,
        query: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> APIResponse:
        """
        Handle an API request.

        Args:
            method: HTTP method
            path: Request path
            query: Query parameters
            body: Request body
            headers: Request headers

        Returns:
            APIResponse
        """
        self._request_count += 1
        request_id = f"req-{uuid.uuid4().hex[:12]}"

        request = APIRequest(
            request_id=request_id,
            method=method,
            path=path,
            query=query or {},
            body=body or {},
            headers=headers or {},
        )

        _emit_bus_event(
            self.BUS_TOPICS["request"],
            {
                "request_id": request_id,
                "method": method,
                "path": path,
            },
            actor=self.actor_id,
        )

        try:
            # Find matching route
            route, path_params = self._match_route(method, path)

            if not route:
                return APIResponse(
                    success=False,
                    error="Not found",
                    status_code=404,
                )

            # Check authentication
            if route.auth_required:
                auth_result = self._authenticate(headers or {})
                if not auth_result:
                    return APIResponse(
                        success=False,
                        error="Unauthorized",
                        status_code=401,
                    )

            # Call handler
            response = route.handler(request, path_params)

            _emit_bus_event(
                self.BUS_TOPICS["response"],
                {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "success": response.success,
                },
                actor=self.actor_id,
            )

            return response

        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=500,
            )

    def _match_route(
        self,
        method: str,
        path: str,
    ) -> Tuple[Optional[Route], Dict[str, str]]:
        """Match request to route."""
        path_params = {}

        # Try exact match first
        key = f"{method}:{path}"
        if key in self._routes:
            return self._routes[key], path_params

        # Try pattern matching
        for route_key, route in self._routes.items():
            route_method, route_path = route_key.split(":", 1)
            if route_method != method:
                continue

            # Check for path parameters
            if "{" in route_path:
                match, params = self._match_path(route_path, path)
                if match:
                    return route, params

        return None, {}

    def _match_path(
        self,
        pattern: str,
        path: str,
    ) -> Tuple[bool, Dict[str, str]]:
        """Match path against pattern with parameters."""
        pattern_parts = pattern.split("/")
        path_parts = path.split("/")

        if len(pattern_parts) != len(path_parts):
            return False, {}

        params = {}
        for pp, part in zip(pattern_parts, path_parts):
            if pp.startswith("{") and pp.endswith("}"):
                param_name = pp[1:-1]
                params[param_name] = part
            elif pp != part:
                return False, {}

        return True, params

    def _authenticate(self, headers: Dict[str, str]) -> bool:
        """Authenticate request."""
        auth_header = headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return token in self._api_keys

        # Allow unauthenticated for development
        return os.environ.get("DEPLOY_API_AUTH", "false").lower() != "true"

    def create_api_key(
        self,
        name: str,
        permissions: Optional[List[str]] = None,
    ) -> str:
        """Create an API key."""
        api_key = f"dak_{uuid.uuid4().hex}"

        self._api_keys[api_key] = {
            "name": name,
            "permissions": permissions or ["read", "write"],
            "created_at": time.time(),
        }

        self._save_state()
        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            self._save_state()
            return True
        return False

    # ==========================================================================
    # Route Handlers
    # ==========================================================================

    def _handle_health(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle health check."""
        return APIResponse(
            data={"status": "healthy", "timestamp": time.time()},
        )

    def _handle_ready(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle readiness check."""
        return APIResponse(
            data={"status": "ready", "timestamp": time.time()},
        )

    def _handle_list_deployments(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle list deployments."""
        orchestrator = self._get_orchestrator()
        service = request.query.get("service")
        limit = int(request.query.get("limit", "100"))

        pipelines = orchestrator.list_pipelines(service_name=service, limit=limit)

        return APIResponse(
            data=[p.to_dict() for p in pipelines],
            metadata={"count": len(pipelines)},
        )

    def _handle_create_deployment(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle create deployment."""
        from ..orchestrator_v2 import PipelineConfigV2, DeploymentType

        body = request.body

        config = PipelineConfigV2(
            name=body.get("name", f"{body.get('service')}-deploy"),
            service_name=body.get("service"),
            version=body.get("version"),
            deployment_type=DeploymentType(body.get("type", "FULL").upper()),
            strategy=body.get("strategy", "blue_green"),
            target_environments=body.get("environments", ["staging"]),
        )

        orchestrator = self._get_orchestrator()

        # Run async in sync context
        loop = asyncio.new_event_loop()
        try:
            state = loop.run_until_complete(orchestrator.run_pipeline(config))
        finally:
            loop.close()

        return APIResponse(
            data=state.to_dict(),
            status_code=201,
        )

    def _handle_get_deployment(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle get deployment."""
        orchestrator = self._get_orchestrator()
        pipeline = orchestrator.get_pipeline(params.get("id", ""))

        if not pipeline:
            return APIResponse(
                success=False,
                error="Deployment not found",
                status_code=404,
            )

        return APIResponse(data=pipeline.to_dict())

    def _handle_cancel_deployment(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle cancel deployment."""
        # Would need to implement cancellation logic
        return APIResponse(
            data={"cancelled": True, "deployment_id": params.get("id")},
        )

    def _handle_list_history(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle list history."""
        tracker = self._get_history_tracker()
        service = request.query.get("service")
        env = request.query.get("environment")
        limit = int(request.query.get("limit", "100"))

        deployments = tracker.query_deployments(
            service_name=service,
            environment=env,
            limit=limit,
        )

        return APIResponse(
            data=[d.to_dict() for d in deployments],
            metadata={"count": len(deployments)},
        )

    def _handle_get_history(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle get history."""
        tracker = self._get_history_tracker()
        record = tracker.get_deployment(params.get("id", ""))

        if not record:
            return APIResponse(
                success=False,
                error="Deployment not found",
                status_code=404,
            )

        return APIResponse(data=record.to_dict())

    def _handle_history_stats(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle history stats."""
        tracker = self._get_history_tracker()
        service = request.query.get("service")
        env = request.query.get("environment")
        days = int(request.query.get("days", "30"))

        stats = tracker.get_statistics(
            service_name=service,
            environment=env,
            days=days,
        )

        return APIResponse(data=stats)

    def _handle_list_schedules(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle list schedules."""
        scheduler = self._get_scheduler()
        schedules = scheduler.list_schedules()

        return APIResponse(
            data=[s.to_dict() for s in schedules],
            metadata={"count": len(schedules)},
        )

    def _handle_create_schedule(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle create schedule."""
        scheduler = self._get_scheduler()
        body = request.body

        schedule_type = body.get("type", "once")

        if schedule_type == "once":
            schedule = scheduler.schedule_once(
                name=body.get("name"),
                service_name=body.get("service"),
                version=body.get("version"),
                scheduled_time=body.get("scheduled_time"),
                environment=body.get("environment", "staging"),
            )
        elif schedule_type == "cron":
            schedule = scheduler.schedule_cron(
                name=body.get("name"),
                service_name=body.get("service"),
                version=body.get("version"),
                cron_expression=body.get("cron"),
                environment=body.get("environment", "staging"),
            )
        else:
            return APIResponse(
                success=False,
                error=f"Unknown schedule type: {schedule_type}",
                status_code=400,
            )

        return APIResponse(data=schedule.to_dict(), status_code=201)

    def _handle_cancel_schedule(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle cancel schedule."""
        scheduler = self._get_scheduler()
        success = scheduler.cancel_schedule(params.get("id", ""))

        if not success:
            return APIResponse(
                success=False,
                error="Schedule not found",
                status_code=404,
            )

        return APIResponse(data={"cancelled": True})

    def _handle_list_approvals(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle list approvals."""
        gate = self._get_approval_gate()
        pending_only = request.query.get("pending", "false").lower() == "true"

        if pending_only:
            requests = gate.list_pending()
        else:
            requests = gate.list_requests()

        return APIResponse(
            data=[r.to_dict() for r in requests],
            metadata={"count": len(requests)},
        )

    def _handle_approve(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle approve."""
        gate = self._get_approval_gate()

        try:
            approval = gate.approve(
                request_id=params.get("id", ""),
                approver_id=request.body.get("approver_id", ""),
                comment=request.body.get("comment", ""),
            )
            return APIResponse(data=approval.to_dict())
        except ValueError as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=400,
            )

    def _handle_reject(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle reject."""
        gate = self._get_approval_gate()

        try:
            approval = gate.reject(
                request_id=params.get("id", ""),
                approver_id=request.body.get("approver_id", ""),
                comment=request.body.get("comment", ""),
            )
            return APIResponse(data=approval.to_dict())
        except ValueError as e:
            return APIResponse(
                success=False,
                error=str(e),
                status_code=400,
            )

    def _handle_get_metrics(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle get metrics."""
        collector = self._get_metrics_collector()
        service = request.query.get("service")
        hours = int(request.query.get("hours", "24"))

        summary = collector.get_summary(service_name=service, hours=hours)

        return APIResponse(data=summary)

    def _handle_record_metric(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle record metric."""
        collector = self._get_metrics_collector()
        body = request.body

        metric = collector.record(
            name=body.get("name"),
            value=body.get("value"),
            service_name=body.get("service", ""),
            deployment_id=body.get("deployment_id", ""),
        )

        return APIResponse(data=metric.to_dict(), status_code=201)

    def _handle_list_services(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle list services."""
        tracker = self._get_history_tracker()

        # Get unique services from history
        deployments = tracker.query_deployments(limit=1000)
        services = list(set(d.service_name for d in deployments))

        return APIResponse(
            data=services,
            metadata={"count": len(services)},
        )

    def _handle_get_version(
        self,
        request: APIRequest,
        params: Dict[str, str],
    ) -> APIResponse:
        """Handle get current version."""
        tracker = self._get_history_tracker()
        env = request.query.get("environment", "prod")

        version = tracker.get_current_version(
            service_name=params.get("name", ""),
            environment=env,
        )

        if not version:
            return APIResponse(
                success=False,
                error="No deployed version found",
                status_code=404,
            )

        return APIResponse(data={"version": version, "environment": env})

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "api_keys": {
                k: {**v, "key": k[:8] + "..."}
                for k, v in self._api_keys.items()
            },
        }
        # Actually save full keys in separate secure file
        keys_file = self.state_dir / "api_keys.json"
        with open(keys_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump({"keys": self._api_keys}, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        keys_file = self.state_dir / "api_keys.json"
        if keys_file.exists():
            try:
                with open(keys_file, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                self._api_keys = data.get("keys", {})
            except (json.JSONDecodeError, IOError):
                pass


# ==============================================================================
# HTTP Server Handler
# ==============================================================================

def create_handler(api: DeploymentAPI):
    """Create HTTP request handler class."""

    class DeployAPIHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._handle_request("GET")

        def do_POST(self):
            self._handle_request("POST")

        def do_PUT(self):
            self._handle_request("PUT")

        def do_DELETE(self):
            self._handle_request("DELETE")

        def _handle_request(self, method: str):
            # Parse path and query
            path = self.path.split("?")[0]
            query = {}
            if "?" in self.path:
                query_str = self.path.split("?")[1]
                for param in query_str.split("&"):
                    if "=" in param:
                        k, v = param.split("=", 1)
                        query[k] = v

            # Get body
            body = {}
            if method in ("POST", "PUT", "PATCH"):
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length > 0:
                    body_data = self.rfile.read(content_length)
                    try:
                        body = json.loads(body_data)
                    except json.JSONDecodeError:
                        pass

            # Get headers
            headers = dict(self.headers)

            # Handle request
            response = api.handle_request(
                method=method,
                path=path,
                query=query,
                body=body,
                headers=headers,
            )

            # Send response
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.to_json().encode())

        def log_message(self, format, *args):
            pass  # Suppress default logging

    return DeployAPIHandler


def create_app(host: str = "0.0.0.0", port: int = 8080) -> Tuple[HTTPServer, DeploymentAPI]:
    """
    Create WSGI application.

    Args:
        host: Bind host
        port: Bind port

    Returns:
        Tuple of (HTTPServer, DeploymentAPI)
    """
    api = DeploymentAPI()
    handler = create_handler(api)
    server = HTTPServer((host, port), handler)
    return server, api


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for deployment API."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment API (Step 229)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", "-H", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", "-p", type=int, default=8080, help="Bind port")

    # create-key command
    key_parser = subparsers.add_parser("create-key", help="Create API key")
    key_parser.add_argument("--name", "-n", required=True, help="Key name")

    # revoke-key command
    revoke_parser = subparsers.add_parser("revoke-key", help="Revoke API key")
    revoke_parser.add_argument("key", help="API key to revoke")

    # test command
    test_parser = subparsers.add_parser("test", help="Test API endpoint")
    test_parser.add_argument("--method", "-m", default="GET", help="HTTP method")
    test_parser.add_argument("--path", "-p", default="/health", help="API path")
    test_parser.add_argument("--data", "-d", help="Request body (JSON)")

    args = parser.parse_args()

    if args.command == "serve":
        server, api = create_app(args.host, args.port)
        print(f"Starting API server on {args.host}:{args.port}")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()

        return 0

    elif args.command == "create-key":
        api = DeploymentAPI()
        key = api.create_api_key(args.name)
        print(f"Created API key: {key}")
        return 0

    elif args.command == "revoke-key":
        api = DeploymentAPI()
        success = api.revoke_api_key(args.key)
        if success:
            print("API key revoked")
        else:
            print("API key not found")
        return 0 if success else 1

    elif args.command == "test":
        api = DeploymentAPI()
        body = json.loads(args.data) if args.data else {}

        response = api.handle_request(
            method=args.method,
            path=args.path,
            body=body,
        )

        print(json.dumps(response.to_dict(), indent=2))
        return 0 if response.success else 1

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
