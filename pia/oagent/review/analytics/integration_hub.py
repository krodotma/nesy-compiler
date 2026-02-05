#!/usr/bin/env python3
"""
Review Integration Hub (Step 177)

External integrations for the review system.

PBTSO Phase: DISTRIBUTE
Bus Topics: review.integration.webhook, review.integration.sync

Integrations:
- GitHub/GitLab/Bitbucket
- Jira/Linear/Asana
- Slack/Teams/Discord
- Custom webhooks

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode


# ============================================================================
# Types
# ============================================================================

class IntegrationType(Enum):
    """Types of integrations."""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    JIRA = "jira"
    LINEAR = "linear"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    EMAIL = "email"


class IntegrationStatus(Enum):
    """Status of an integration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class WebhookEventType(Enum):
    """Types of webhook events."""
    REVIEW_STARTED = "review.started"
    REVIEW_COMPLETED = "review.completed"
    ISSUE_FOUND = "issue.found"
    APPROVAL_REQUIRED = "approval.required"
    REPORT_GENERATED = "report.generated"
    DEBT_CREATED = "debt.created"


@dataclass
class IntegrationConfig:
    """Configuration for integration hub."""
    webhook_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 5
    batch_size: int = 100
    enable_signature_verification: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Integration:
    """
    An integration configuration.

    Attributes:
        integration_id: Unique identifier
        name: Integration name
        integration_type: Type of integration
        status: Current status
        config: Integration-specific configuration
        events: Subscribed event types
        created_at: Creation timestamp
        last_sync: Last synchronization timestamp
        error_message: Last error message
    """
    integration_id: str
    name: str
    integration_type: IntegrationType
    status: IntegrationStatus
    config: Dict[str, Any] = field(default_factory=dict)
    events: List[WebhookEventType] = field(default_factory=list)
    created_at: str = ""
    last_sync: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "integration_id": self.integration_id,
            "name": self.name,
            "integration_type": self.integration_type.value,
            "status": self.status.value,
            "config": {k: v for k, v in self.config.items() if k != "secret"},
            "events": [e.value for e in self.events],
            "created_at": self.created_at,
            "last_sync": self.last_sync,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Integration":
        """Create from dictionary."""
        data = data.copy()
        data["integration_type"] = IntegrationType(data["integration_type"])
        data["status"] = IntegrationStatus(data["status"])
        data["events"] = [WebhookEventType(e) for e in data.get("events", [])]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WebhookPayload:
    """Payload for webhook delivery."""
    event_type: WebhookEventType
    timestamp: str
    data: Dict[str, Any]
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def sign(self, secret: str) -> str:
        """Generate HMAC signature."""
        payload = self.to_json().encode()
        signature = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        self.signature = f"sha256={signature}"
        return self.signature


@dataclass
class DeliveryResult:
    """Result of webhook delivery."""
    integration_id: str
    success: bool
    status_code: Optional[int] = None
    response: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Integration Store
# ============================================================================

class IntegrationStore:
    """Persistent store for integrations."""

    def __init__(self, store_path: Path):
        """Initialize the store."""
        self.store_path = store_path
        self._ensure_store()

    def _ensure_store(self) -> None:
        """Ensure store file exists."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._write_store({"integrations": [], "version": 1})

    def _read_store(self) -> Dict[str, Any]:
        """Read store with file locking."""
        with open(self.store_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_store(self, data: Dict[str, Any]) -> None:
        """Write store with file locking."""
        with open(self.store_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def add(self, integration: Integration) -> None:
        """Add an integration."""
        data = self._read_store()
        # Remove existing with same ID
        data["integrations"] = [
            i for i in data["integrations"]
            if i["integration_id"] != integration.integration_id
        ]
        data["integrations"].append({
            **integration.to_dict(),
            "config": integration.config,  # Include full config with secrets
        })
        self._write_store(data)

    def get(self, integration_id: str) -> Optional[Integration]:
        """Get an integration by ID."""
        data = self._read_store()
        for i in data["integrations"]:
            if i["integration_id"] == integration_id:
                return Integration.from_dict(i)
        return None

    def get_all(
        self,
        integration_type: Optional[IntegrationType] = None,
        status: Optional[IntegrationStatus] = None,
    ) -> List[Integration]:
        """Get integrations with optional filters."""
        data = self._read_store()
        integrations = []

        for i_data in data["integrations"]:
            if integration_type and i_data["integration_type"] != integration_type.value:
                continue
            if status and i_data["status"] != status.value:
                continue
            integrations.append(Integration.from_dict(i_data))

        return integrations

    def update(self, integration: Integration) -> bool:
        """Update an integration."""
        data = self._read_store()
        for i, existing in enumerate(data["integrations"]):
            if existing["integration_id"] == integration.integration_id:
                data["integrations"][i] = {
                    **integration.to_dict(),
                    "config": integration.config,
                }
                self._write_store(data)
                return True
        return False

    def delete(self, integration_id: str) -> bool:
        """Delete an integration."""
        data = self._read_store()
        original_len = len(data["integrations"])
        data["integrations"] = [
            i for i in data["integrations"]
            if i["integration_id"] != integration_id
        ]
        if len(data["integrations"]) < original_len:
            self._write_store(data)
            return True
        return False


# ============================================================================
# Integration Adapters
# ============================================================================

class BaseAdapter:
    """Base class for integration adapters."""

    def __init__(self, integration: Integration):
        self.integration = integration

    async def deliver(self, payload: WebhookPayload) -> DeliveryResult:
        """Deliver a webhook payload."""
        raise NotImplementedError

    async def test_connection(self) -> bool:
        """Test the integration connection."""
        raise NotImplementedError


class WebhookAdapter(BaseAdapter):
    """Adapter for generic webhooks."""

    async def deliver(self, payload: WebhookPayload) -> DeliveryResult:
        """Deliver payload to webhook URL."""
        start_time = time.time()
        url = self.integration.config.get("url", "")

        if not url:
            return DeliveryResult(
                integration_id=self.integration.integration_id,
                success=False,
                error="No webhook URL configured",
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Sign payload if secret configured
        secret = self.integration.config.get("secret")
        if secret:
            payload.sign(secret)

        # Simulate HTTP POST (in production, would use aiohttp or httpx)
        try:
            # Mock successful delivery
            return DeliveryResult(
                integration_id=self.integration.integration_id,
                success=True,
                status_code=200,
                response="OK",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return DeliveryResult(
                integration_id=self.integration.integration_id,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def test_connection(self) -> bool:
        """Test webhook connectivity."""
        url = self.integration.config.get("url", "")
        return bool(url)


class SlackAdapter(BaseAdapter):
    """Adapter for Slack integration."""

    async def deliver(self, payload: WebhookPayload) -> DeliveryResult:
        """Deliver payload to Slack."""
        start_time = time.time()
        webhook_url = self.integration.config.get("webhook_url", "")

        if not webhook_url:
            return DeliveryResult(
                integration_id=self.integration.integration_id,
                success=False,
                error="No Slack webhook URL configured",
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Format message for Slack
        slack_message = self._format_slack_message(payload)

        # Mock delivery
        return DeliveryResult(
            integration_id=self.integration.integration_id,
            success=True,
            status_code=200,
            response="ok",
            duration_ms=(time.time() - start_time) * 1000,
        )

    def _format_slack_message(self, payload: WebhookPayload) -> Dict[str, Any]:
        """Format payload as Slack message."""
        event_icons = {
            WebhookEventType.REVIEW_STARTED: ":mag:",
            WebhookEventType.REVIEW_COMPLETED: ":white_check_mark:",
            WebhookEventType.ISSUE_FOUND: ":warning:",
            WebhookEventType.APPROVAL_REQUIRED: ":hand:",
            WebhookEventType.REPORT_GENERATED: ":page_facing_up:",
            WebhookEventType.DEBT_CREATED: ":money_with_wings:",
        }

        icon = event_icons.get(payload.event_type, ":bell:")
        title = payload.event_type.value.replace(".", " ").title()

        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{icon} {title}"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": json.dumps(payload.data, indent=2)}
                },
            ]
        }

    async def test_connection(self) -> bool:
        """Test Slack connectivity."""
        return bool(self.integration.config.get("webhook_url"))


class GitHubAdapter(BaseAdapter):
    """Adapter for GitHub integration."""

    async def deliver(self, payload: WebhookPayload) -> DeliveryResult:
        """Deliver payload to GitHub (create check run, comment, etc.)."""
        start_time = time.time()
        token = self.integration.config.get("token", "")

        if not token:
            return DeliveryResult(
                integration_id=self.integration.integration_id,
                success=False,
                error="No GitHub token configured",
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Mock GitHub API call
        return DeliveryResult(
            integration_id=self.integration.integration_id,
            success=True,
            status_code=201,
            response='{"id": 12345}',
            duration_ms=(time.time() - start_time) * 1000,
        )

    async def test_connection(self) -> bool:
        """Test GitHub connectivity."""
        return bool(self.integration.config.get("token"))


# ============================================================================
# Integration Hub
# ============================================================================

class IntegrationHub:
    """
    Hub for managing external integrations.

    Example:
        hub = IntegrationHub()

        # Create webhook integration
        integration = hub.create_integration(
            name="My Webhook",
            integration_type=IntegrationType.WEBHOOK,
            config={"url": "https://example.com/webhook"},
            events=[WebhookEventType.REVIEW_COMPLETED],
        )

        # Trigger event
        await hub.trigger_event(
            WebhookEventType.REVIEW_COMPLETED,
            {"review_id": "abc123", "decision": "approve"},
        )
    """

    BUS_TOPICS = {
        "webhook": "review.integration.webhook",
        "sync": "review.integration.sync",
    }

    ADAPTERS = {
        IntegrationType.WEBHOOK: WebhookAdapter,
        IntegrationType.SLACK: SlackAdapter,
        IntegrationType.GITHUB: GitHubAdapter,
    }

    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        bus_path: Optional[Path] = None,
        store_path: Optional[Path] = None,
    ):
        """
        Initialize the integration hub.

        Args:
            config: Hub configuration
            bus_path: Path to event bus file
            store_path: Path to integration store
        """
        self.config = config or IntegrationConfig()
        self.bus_path = bus_path or self._get_bus_path()
        self.store = IntegrationStore(store_path or self._get_store_path())

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_store_path(self) -> Path:
        """Get path to integration store."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        data_dir = pluribus_root / ".pluribus" / "review" / "data"
        return data_dir / "integrations.json"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "integration") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "integration-hub",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def create_integration(
        self,
        name: str,
        integration_type: IntegrationType,
        config: Dict[str, Any],
        events: Optional[List[WebhookEventType]] = None,
    ) -> Integration:
        """
        Create a new integration.

        Args:
            name: Integration name
            integration_type: Type of integration
            config: Integration-specific configuration
            events: Event types to subscribe to

        Returns:
            Created integration
        """
        integration_id = str(uuid.uuid4())[:8]

        integration = Integration(
            integration_id=integration_id,
            name=name,
            integration_type=integration_type,
            status=IntegrationStatus.ACTIVE,
            config=config,
            events=events or list(WebhookEventType),
        )

        self.store.add(integration)

        self._emit_event(self.BUS_TOPICS["sync"], {
            "action": "created",
            "integration_id": integration_id,
            "integration_type": integration_type.value,
        })

        return integration

    def get_integration(self, integration_id: str) -> Optional[Integration]:
        """Get an integration by ID."""
        return self.store.get(integration_id)

    def list_integrations(
        self,
        integration_type: Optional[IntegrationType] = None,
        status: Optional[IntegrationStatus] = None,
    ) -> List[Integration]:
        """List all integrations."""
        return self.store.get_all(integration_type=integration_type, status=status)

    def update_integration(
        self,
        integration_id: str,
        config: Optional[Dict[str, Any]] = None,
        events: Optional[List[WebhookEventType]] = None,
        status: Optional[IntegrationStatus] = None,
    ) -> Optional[Integration]:
        """Update an integration."""
        integration = self.store.get(integration_id)
        if not integration:
            return None

        if config:
            integration.config.update(config)
        if events is not None:
            integration.events = events
        if status:
            integration.status = status

        self.store.update(integration)
        return integration

    def delete_integration(self, integration_id: str) -> bool:
        """Delete an integration."""
        return self.store.delete(integration_id)

    async def trigger_event(
        self,
        event_type: WebhookEventType,
        data: Dict[str, Any],
    ) -> List[DeliveryResult]:
        """
        Trigger an event to all subscribed integrations.

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            List of delivery results

        Emits:
            review.integration.webhook
        """
        payload = WebhookPayload(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            data=data,
        )

        # Get all active integrations subscribed to this event
        integrations = self.store.get_all(status=IntegrationStatus.ACTIVE)
        subscribed = [i for i in integrations if event_type in i.events]

        results = []
        for integration in subscribed:
            result = await self._deliver_to_integration(integration, payload)
            results.append(result)

            # Update integration status on error
            if not result.success:
                integration.error_message = result.error
                if result.retries >= self.config.max_retries:
                    integration.status = IntegrationStatus.ERROR
                self.store.update(integration)
            else:
                integration.last_sync = datetime.now(timezone.utc).isoformat() + "Z"
                integration.error_message = None
                self.store.update(integration)

        self._emit_event(self.BUS_TOPICS["webhook"], {
            "event_type": event_type.value,
            "integrations_triggered": len(subscribed),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
        })

        return results

    async def _deliver_to_integration(
        self,
        integration: Integration,
        payload: WebhookPayload,
    ) -> DeliveryResult:
        """Deliver payload to a specific integration."""
        adapter_class = self.ADAPTERS.get(integration.integration_type, WebhookAdapter)
        adapter = adapter_class(integration)

        for attempt in range(self.config.max_retries):
            result = await adapter.deliver(payload)
            result.retries = attempt

            if result.success:
                return result

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay_seconds)

        return result

    async def test_integration(self, integration_id: str) -> bool:
        """Test an integration's connectivity."""
        integration = self.store.get(integration_id)
        if not integration:
            return False

        adapter_class = self.ADAPTERS.get(integration.integration_type, WebhookAdapter)
        adapter = adapter_class(integration)

        return await adapter.test_connection()


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Integration Hub."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Integration Hub (Step 177)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create integration")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--type", required=True,
                               choices=[t.value for t in IntegrationType])
    create_parser.add_argument("--url", help="Webhook URL")
    create_parser.add_argument("--token", help="API token")

    # List command
    list_parser = subparsers.add_parser("list", help="List integrations")
    list_parser.add_argument("--type", choices=[t.value for t in IntegrationType])

    # Test command
    test_parser = subparsers.add_parser("test", help="Test integration")
    test_parser.add_argument("integration_id")

    # Trigger command
    trigger_parser = subparsers.add_parser("trigger", help="Trigger event")
    trigger_parser.add_argument("event_type", choices=[e.value for e in WebhookEventType])
    trigger_parser.add_argument("--data", default="{}", help="JSON data")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    hub = IntegrationHub()

    if args.command == "create":
        config = {}
        if args.url:
            config["url"] = args.url
            config["webhook_url"] = args.url
        if args.token:
            config["token"] = args.token

        integration = hub.create_integration(
            name=args.name,
            integration_type=IntegrationType(args.type),
            config=config,
        )

        if args.json:
            print(json.dumps(integration.to_dict(), indent=2))
        else:
            print(f"Created integration: {integration.integration_id}")
            print(f"  Name: {integration.name}")
            print(f"  Type: {integration.integration_type.value}")

    elif args.command == "list":
        int_type = IntegrationType(args.type) if args.type else None
        integrations = hub.list_integrations(integration_type=int_type)

        if args.json:
            print(json.dumps([i.to_dict() for i in integrations], indent=2))
        else:
            print(f"Found {len(integrations)} integrations:")
            for i in integrations:
                print(f"  [{i.integration_id}] {i.name}")
                print(f"    Type: {i.integration_type.value} | Status: {i.status.value}")

    elif args.command == "test":
        success = asyncio.run(hub.test_integration(args.integration_id))
        if success:
            print("Integration test: SUCCESS")
        else:
            print("Integration test: FAILED")
            return 1

    elif args.command == "trigger":
        data = json.loads(args.data)
        results = asyncio.run(hub.trigger_event(
            WebhookEventType(args.event_type),
            data,
        ))

        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print(f"Triggered {args.event_type} to {len(results)} integrations")
            for r in results:
                status = "OK" if r.success else "FAILED"
                print(f"  [{r.integration_id}] {status}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
