#!/usr/bin/env python3
"""
hub.py - Deployment Integration Hub (Step 228)

PBTSO Phase: PLAN, DISTRIBUTE
A2A Integration: Integrates with CI/CD systems via deploy.integration.*

Provides:
- IntegrationType: Types of integrations
- IntegrationStatus: Integration status
- IntegrationConfig: Integration configuration
- IntegrationEvent: Integration event
- DeploymentIntegrationHub: Main hub class

Bus Topics:
- deploy.integration.trigger
- deploy.integration.webhook
- deploy.integration.sync

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
from typing import Any, Callable, Dict, List, Optional
import urllib.request
import urllib.error


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
    actor: str = "integration-hub"
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

class IntegrationType(Enum):
    """Types of CI/CD integrations."""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    AZURE_DEVOPS = "azure_devops"
    BITBUCKET = "bitbucket"
    ARGOCD = "argocd"
    SPINNAKER = "spinnaker"
    TEKTON = "tekton"
    WEBHOOK = "webhook"
    CUSTOM = "custom"


class IntegrationStatus(Enum):
    """Integration status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class TriggerType(Enum):
    """Trigger types."""
    PUSH = "push"
    PR = "pr"
    TAG = "tag"
    MANUAL = "manual"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    API = "api"


@dataclass
class IntegrationConfig:
    """
    Integration configuration.

    Attributes:
        integration_id: Unique integration identifier
        name: Integration name
        integration_type: Type of integration
        endpoint_url: Integration endpoint URL
        auth_type: Authentication type
        auth_config: Authentication configuration
        services: Services to integrate (empty = all)
        environments: Environments to integrate
        triggers: Enabled trigger types
        status: Integration status
        created_at: Creation timestamp
        last_sync: Last sync timestamp
        metadata: Additional metadata
    """
    integration_id: str
    name: str
    integration_type: IntegrationType
    endpoint_url: str = ""
    auth_type: str = "token"  # token, basic, oauth, none
    auth_config: Dict[str, str] = field(default_factory=dict)
    services: List[str] = field(default_factory=list)
    environments: List[str] = field(default_factory=list)
    triggers: List[TriggerType] = field(default_factory=list)
    status: IntegrationStatus = IntegrationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    last_sync: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "integration_id": self.integration_id,
            "name": self.name,
            "integration_type": self.integration_type.value,
            "endpoint_url": self.endpoint_url,
            "auth_type": self.auth_type,
            "auth_config": {k: "***" for k in self.auth_config},  # Hide secrets
            "services": self.services,
            "environments": self.environments,
            "triggers": [t.value for t in self.triggers],
            "status": self.status.value,
            "created_at": self.created_at,
            "last_sync": self.last_sync,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationConfig":
        data = dict(data)
        if "integration_type" in data:
            data["integration_type"] = IntegrationType(data["integration_type"])
        if "status" in data:
            data["status"] = IntegrationStatus(data["status"])
        if "triggers" in data:
            data["triggers"] = [TriggerType(t) for t in data["triggers"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class IntegrationEvent:
    """
    Integration event.

    Attributes:
        event_id: Unique event identifier
        integration_id: Associated integration
        event_type: Event type
        trigger_type: Trigger type
        payload: Event payload
        status: Event status
        created_at: Creation timestamp
        processed_at: Processing timestamp
        result: Processing result
    """
    event_id: str
    integration_id: str
    event_type: str
    trigger_type: TriggerType = TriggerType.WEBHOOK
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, processing, completed, failed
    created_at: float = field(default_factory=time.time)
    processed_at: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "integration_id": self.integration_id,
            "event_type": self.event_type,
            "trigger_type": self.trigger_type.value,
            "payload": self.payload,
            "status": self.status,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
            "result": self.result,
        }


# ==============================================================================
# Circuit Breaker
# ==============================================================================

class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures: Dict[str, int] = {}
        self._last_failure: Dict[str, float] = {}
        self._open: Dict[str, bool] = {}

    def is_open(self, service_id: str) -> bool:
        """Check if circuit is open."""
        if not self._open.get(service_id, False):
            return False

        # Check if reset timeout has passed
        if time.time() - self._last_failure.get(service_id, 0) > self.reset_timeout:
            self._open[service_id] = False
            self._failures[service_id] = 0
            return False

        return True

    def record_success(self, service_id: str) -> None:
        """Record successful call."""
        self._failures[service_id] = 0
        self._open[service_id] = False

    def record_failure(self, service_id: str) -> None:
        """Record failed call."""
        self._failures[service_id] = self._failures.get(service_id, 0) + 1
        self._last_failure[service_id] = time.time()

        if self._failures[service_id] >= self.failure_threshold:
            self._open[service_id] = True


# ==============================================================================
# Deployment Integration Hub (Step 228)
# ==============================================================================

class DeploymentIntegrationHub:
    """
    Deployment Integration Hub - integrates with CI/CD systems.

    PBTSO Phase: PLAN, DISTRIBUTE

    Responsibilities:
    - Connect to CI/CD platforms
    - Receive and process webhooks
    - Trigger external pipelines
    - Sync deployment status
    - Handle authentication

    Example:
        >>> hub = DeploymentIntegrationHub()
        >>> config = hub.register_integration(
        ...     name="github-actions",
        ...     integration_type=IntegrationType.GITHUB_ACTIONS,
        ...     endpoint_url="https://api.github.com",
        ...     auth_config={"token": "ghp_xxx"},
        ... )
        >>> await hub.trigger_pipeline(config.integration_id, ...)
    """

    BUS_TOPICS = {
        "trigger": "deploy.integration.trigger",
        "webhook": "deploy.integration.webhook",
        "sync": "deploy.integration.sync",
        "error": "deploy.integration.error",
    }

    # A2A heartbeat configuration
    HEARTBEAT_INTERVAL = 300  # 5 minutes
    HEARTBEAT_TIMEOUT = 900   # 15 minutes

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "integration-hub",
    ):
        """
        Initialize the integration hub.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "integrations"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._integrations: Dict[str, IntegrationConfig] = {}
        self._events: Dict[str, IntegrationEvent] = {}
        self._handlers: Dict[IntegrationType, Callable] = {}
        self._circuit_breaker = CircuitBreaker()

        self._load_state()

    def register_integration(
        self,
        name: str,
        integration_type: IntegrationType,
        endpoint_url: str = "",
        auth_type: str = "token",
        auth_config: Optional[Dict[str, str]] = None,
        services: Optional[List[str]] = None,
        environments: Optional[List[str]] = None,
        triggers: Optional[List[TriggerType]] = None,
    ) -> IntegrationConfig:
        """
        Register a CI/CD integration.

        Args:
            name: Integration name
            integration_type: Type of integration
            endpoint_url: Integration endpoint URL
            auth_type: Authentication type
            auth_config: Authentication configuration
            services: Services to integrate
            environments: Environments to integrate
            triggers: Enabled trigger types

        Returns:
            Created IntegrationConfig
        """
        integration_id = f"integration-{uuid.uuid4().hex[:12]}"

        config = IntegrationConfig(
            integration_id=integration_id,
            name=name,
            integration_type=integration_type,
            endpoint_url=endpoint_url,
            auth_type=auth_type,
            auth_config=auth_config or {},
            services=services or [],
            environments=environments or [],
            triggers=triggers or [TriggerType.WEBHOOK],
            status=IntegrationStatus.ACTIVE,
        )

        self._integrations[integration_id] = config
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["sync"],
            {
                "integration_id": integration_id,
                "name": name,
                "type": integration_type.value,
                "action": "registered",
            },
            actor=self.actor_id,
        )

        return config

    def register_handler(
        self,
        integration_type: IntegrationType,
        handler: Callable[[IntegrationConfig, Dict[str, Any]], Any],
    ) -> None:
        """Register a handler for an integration type."""
        self._handlers[integration_type] = handler

    async def trigger_pipeline(
        self,
        integration_id: str,
        service_name: str,
        version: str,
        environment: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> IntegrationEvent:
        """
        Trigger an external CI/CD pipeline.

        Args:
            integration_id: Integration to use
            service_name: Service to deploy
            version: Version to deploy
            environment: Target environment
            parameters: Additional parameters

        Returns:
            Created IntegrationEvent
        """
        config = self._integrations.get(integration_id)
        if not config:
            raise ValueError(f"Integration not found: {integration_id}")

        if config.status != IntegrationStatus.ACTIVE:
            raise ValueError(f"Integration not active: {config.status.value}")

        # Check circuit breaker
        if self._circuit_breaker.is_open(integration_id):
            raise RuntimeError(f"Circuit breaker open for: {integration_id}")

        event_id = f"event-{uuid.uuid4().hex[:12]}"
        event = IntegrationEvent(
            event_id=event_id,
            integration_id=integration_id,
            event_type="pipeline_trigger",
            trigger_type=TriggerType.API,
            payload={
                "service_name": service_name,
                "version": version,
                "environment": environment,
                "parameters": parameters or {},
            },
            status="processing",
        )

        _emit_bus_event(
            self.BUS_TOPICS["trigger"],
            {
                "event_id": event_id,
                "integration_id": integration_id,
                "service_name": service_name,
                "version": version,
            },
            actor=self.actor_id,
        )

        try:
            result = await self._execute_trigger(config, event)
            event.status = "completed"
            event.result = result
            event.processed_at = time.time()
            self._circuit_breaker.record_success(integration_id)

        except Exception as e:
            event.status = "failed"
            event.result = {"error": str(e)}
            event.processed_at = time.time()
            self._circuit_breaker.record_failure(integration_id)

            _emit_bus_event(
                self.BUS_TOPICS["error"],
                {
                    "event_id": event_id,
                    "integration_id": integration_id,
                    "error": str(e),
                },
                level="error",
                actor=self.actor_id,
            )

        self._events[event_id] = event
        config.last_sync = time.time()
        self._save_state()

        return event

    async def _execute_trigger(
        self,
        config: IntegrationConfig,
        event: IntegrationEvent,
    ) -> Dict[str, Any]:
        """Execute pipeline trigger."""
        # Check for custom handler
        handler = self._handlers.get(config.integration_type)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                return await handler(config, event.payload)
            return handler(config, event.payload)

        # Default: HTTP webhook trigger
        if config.endpoint_url:
            return await self._http_trigger(config, event.payload)

        return {"status": "simulated", "message": "No endpoint configured"}

    async def _http_trigger(
        self,
        config: IntegrationConfig,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Trigger via HTTP webhook."""
        headers = {"Content-Type": "application/json"}

        # Add authentication
        if config.auth_type == "token" and "token" in config.auth_config:
            headers["Authorization"] = f"Bearer {config.auth_config['token']}"
        elif config.auth_type == "basic":
            import base64
            creds = f"{config.auth_config.get('username', '')}:{config.auth_config.get('password', '')}"
            headers["Authorization"] = f"Basic {base64.b64encode(creds.encode()).decode()}"

        try:
            request = urllib.request.Request(
                config.endpoint_url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(request, timeout=30)
            )

            return {
                "status_code": response.getcode(),
                "body": response.read().decode()[:1000],
            }

        except urllib.error.HTTPError as e:
            return {
                "status_code": e.code,
                "error": str(e.reason),
            }
        except urllib.error.URLError as e:
            raise RuntimeError(f"URL error: {e.reason}")

    async def process_webhook(
        self,
        integration_id: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> IntegrationEvent:
        """
        Process incoming webhook.

        Args:
            integration_id: Integration receiving webhook
            payload: Webhook payload
            headers: Request headers

        Returns:
            Created IntegrationEvent
        """
        config = self._integrations.get(integration_id)
        if not config:
            raise ValueError(f"Integration not found: {integration_id}")

        event_id = f"event-{uuid.uuid4().hex[:12]}"
        event = IntegrationEvent(
            event_id=event_id,
            integration_id=integration_id,
            event_type="webhook",
            trigger_type=TriggerType.WEBHOOK,
            payload=payload,
            status="processing",
        )

        _emit_bus_event(
            self.BUS_TOPICS["webhook"],
            {
                "event_id": event_id,
                "integration_id": integration_id,
                "event_type": payload.get("action", "unknown"),
            },
            actor=self.actor_id,
        )

        try:
            # Process based on integration type
            result = await self._process_webhook_payload(config, payload, headers)
            event.status = "completed"
            event.result = result

        except Exception as e:
            event.status = "failed"
            event.result = {"error": str(e)}

        event.processed_at = time.time()
        self._events[event_id] = event
        self._save_state()

        return event

    async def _process_webhook_payload(
        self,
        config: IntegrationConfig,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Process webhook payload based on integration type."""
        if config.integration_type == IntegrationType.GITHUB_ACTIONS:
            return self._process_github_webhook(payload, headers)
        elif config.integration_type == IntegrationType.GITLAB_CI:
            return self._process_gitlab_webhook(payload, headers)
        else:
            return {"processed": True, "payload_keys": list(payload.keys())}

    def _process_github_webhook(
        self,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Process GitHub webhook."""
        event_type = headers.get("X-GitHub-Event", "") if headers else ""

        return {
            "event_type": event_type,
            "action": payload.get("action", ""),
            "repository": payload.get("repository", {}).get("full_name", ""),
            "sender": payload.get("sender", {}).get("login", ""),
        }

    def _process_gitlab_webhook(
        self,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Process GitLab webhook."""
        return {
            "event_type": payload.get("object_kind", ""),
            "project": payload.get("project", {}).get("path_with_namespace", ""),
            "user": payload.get("user", {}).get("username", ""),
        }

    async def sync_status(
        self,
        integration_id: str,
    ) -> IntegrationConfig:
        """
        Sync integration status.

        Args:
            integration_id: Integration to sync

        Returns:
            Updated IntegrationConfig
        """
        config = self._integrations.get(integration_id)
        if not config:
            raise ValueError(f"Integration not found: {integration_id}")

        # Check circuit breaker state
        if self._circuit_breaker.is_open(integration_id):
            config.status = IntegrationStatus.ERROR
        else:
            # Try to reach endpoint
            if config.endpoint_url:
                try:
                    request = urllib.request.Request(
                        config.endpoint_url,
                        method="HEAD",
                    )
                    urllib.request.urlopen(request, timeout=10)
                    config.status = IntegrationStatus.ACTIVE
                except:
                    config.status = IntegrationStatus.ERROR
            else:
                config.status = IntegrationStatus.ACTIVE

        config.last_sync = time.time()
        self._save_state()

        return config

    def get_integration(self, integration_id: str) -> Optional[IntegrationConfig]:
        """Get an integration by ID."""
        return self._integrations.get(integration_id)

    def list_integrations(
        self,
        integration_type: Optional[IntegrationType] = None,
        status: Optional[IntegrationStatus] = None,
    ) -> List[IntegrationConfig]:
        """List integrations."""
        integrations = list(self._integrations.values())

        if integration_type:
            integrations = [i for i in integrations if i.integration_type == integration_type]
        if status:
            integrations = [i for i in integrations if i.status == status]

        return integrations

    def list_events(
        self,
        integration_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[IntegrationEvent]:
        """List integration events."""
        events = list(self._events.values())

        if integration_id:
            events = [e for e in events if e.integration_id == integration_id]

        events.sort(key=lambda e: e.created_at, reverse=True)
        return events[:limit]

    def update_integration(
        self,
        integration_id: str,
        **updates,
    ) -> Optional[IntegrationConfig]:
        """Update an integration."""
        config = self._integrations.get(integration_id)
        if not config:
            return None

        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self._save_state()
        return config

    def delete_integration(self, integration_id: str) -> bool:
        """Delete an integration."""
        if integration_id not in self._integrations:
            return False

        del self._integrations[integration_id]
        self._save_state()
        return True

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "integrations": {
                iid: {**i.to_dict(), "auth_config": i.auth_config}
                for iid, i in self._integrations.items()
            },
            "events": {
                eid: e.to_dict()
                for eid, e in list(self._events.items())[-500:]
            },
        }
        state_file = self.state_dir / "integration_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "integration_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for iid, data in state.get("integrations", {}).items():
                self._integrations[iid] = IntegrationConfig.from_dict(data)

            for eid, data in state.get("events", {}).items():
                data["trigger_type"] = TriggerType(data.get("trigger_type", "webhook"))
                self._events[eid] = IntegrationEvent(**{
                    k: v for k, v in data.items() if k in IntegrationEvent.__dataclass_fields__
                })
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for integration hub."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment Integration Hub (Step 228)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register command
    register_parser = subparsers.add_parser("register", help="Register integration")
    register_parser.add_argument("--name", "-n", required=True, help="Integration name")
    register_parser.add_argument("--type", "-t", required=True,
                                  choices=[t.value for t in IntegrationType])
    register_parser.add_argument("--url", "-u", default="", help="Endpoint URL")
    register_parser.add_argument("--token", help="Auth token")
    register_parser.add_argument("--json", action="store_true", help="JSON output")

    # trigger command
    trigger_parser = subparsers.add_parser("trigger", help="Trigger pipeline")
    trigger_parser.add_argument("integration_id", help="Integration ID")
    trigger_parser.add_argument("--service", "-s", required=True, help="Service name")
    trigger_parser.add_argument("--version", "-v", required=True, help="Version")
    trigger_parser.add_argument("--env", "-e", default="staging", help="Environment")
    trigger_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List integrations")
    list_parser.add_argument("--type", "-t", help="Filter by type")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # events command
    events_parser = subparsers.add_parser("events", help="List events")
    events_parser.add_argument("--integration", "-i", help="Filter by integration")
    events_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit")
    events_parser.add_argument("--json", action="store_true", help="JSON output")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync integration status")
    sync_parser.add_argument("integration_id", help="Integration ID")
    sync_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    hub = DeploymentIntegrationHub()

    if args.command == "register":
        auth_config = {}
        if args.token:
            auth_config["token"] = args.token

        config = hub.register_integration(
            name=args.name,
            integration_type=IntegrationType(args.type),
            endpoint_url=args.url,
            auth_config=auth_config,
        )

        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print(f"Registered: {config.integration_id}")
            print(f"  Name: {config.name}")
            print(f"  Type: {config.integration_type.value}")
            print(f"  Status: {config.status.value}")

        return 0

    elif args.command == "trigger":
        try:
            event = asyncio.get_event_loop().run_until_complete(
                hub.trigger_pipeline(
                    integration_id=args.integration_id,
                    service_name=args.service,
                    version=args.version,
                    environment=args.env,
                )
            )

            if args.json:
                print(json.dumps(event.to_dict(), indent=2))
            else:
                print(f"Triggered: {event.event_id}")
                print(f"  Status: {event.status}")

            return 0 if event.status == "completed" else 1
        except Exception as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "list":
        itype = IntegrationType(args.type) if args.type else None
        status = IntegrationStatus(args.status) if args.status else None
        integrations = hub.list_integrations(integration_type=itype, status=status)

        if args.json:
            print(json.dumps([i.to_dict() for i in integrations], indent=2))
        else:
            for i in integrations:
                print(f"{i.integration_id} ({i.name}) - {i.integration_type.value} [{i.status.value}]")

        return 0

    elif args.command == "events":
        events = hub.list_events(
            integration_id=args.integration,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            for e in events:
                print(f"{e.event_id} ({e.event_type}) - {e.status}")

        return 0

    elif args.command == "sync":
        try:
            config = asyncio.get_event_loop().run_until_complete(
                hub.sync_status(args.integration_id)
            )

            if args.json:
                print(json.dumps(config.to_dict(), indent=2))
            else:
                print(f"Synced: {config.integration_id}")
                print(f"  Status: {config.status.value}")

            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
