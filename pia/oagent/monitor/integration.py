#!/usr/bin/env python3
"""
Monitor Integration Hub - Step 277

External integrations for monitoring data and alerts.

PBTSO Phase: DISTRIBUTE

Bus Topics:
- monitor.integration.send (subscribed)
- monitor.integration.receive (emitted)
- monitor.integration.status (emitted)

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
from typing import Any, Callable, Coroutine, Dict, List, Optional


class IntegrationType(Enum):
    """Types of integrations."""
    WEBHOOK = "webhook"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    EMAIL = "email"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    CUSTOM = "custom"


class IntegrationStatus(Enum):
    """Integration status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class IntegrationConfig:
    """Configuration for an integration.

    Attributes:
        integration_id: Unique integration ID
        name: Integration name
        integration_type: Type of integration
        config: Integration-specific configuration
        enabled: Whether integration is enabled
        rate_limit: Rate limit (requests per minute)
        retry_count: Number of retries
        timeout_s: Request timeout
        created_at: Creation timestamp
    """
    integration_id: str
    name: str
    integration_type: IntegrationType
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    rate_limit: int = 60
    retry_count: int = 3
    timeout_s: int = 30
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "integration_id": self.integration_id,
            "name": self.name,
            "integration_type": self.integration_type.value,
            "config": {k: v for k, v in self.config.items() if k != "secret"},
            "enabled": self.enabled,
            "rate_limit": self.rate_limit,
            "retry_count": self.retry_count,
            "timeout_s": self.timeout_s,
            "created_at": self.created_at,
        }


@dataclass
class IntegrationMessage:
    """A message to send via integration.

    Attributes:
        message_id: Unique message ID
        integration_id: Target integration ID
        message_type: Type of message
        payload: Message payload
        priority: Message priority
        created_at: Creation timestamp
    """
    message_id: str
    integration_id: str
    message_type: str
    payload: Dict[str, Any]
    priority: str = "normal"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "integration_id": self.integration_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "priority": self.priority,
            "created_at": self.created_at,
        }


@dataclass
class DeliveryResult:
    """Result of message delivery.

    Attributes:
        message_id: Message ID
        integration_id: Integration ID
        success: Whether delivery succeeded
        status_code: HTTP status code (if applicable)
        response: Response data
        error: Error message if failed
        duration_ms: Delivery duration
        retry_count: Number of retries used
    """
    message_id: str
    integration_id: str
    success: bool
    status_code: Optional[int] = None
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "integration_id": self.integration_id,
            "success": self.success,
            "status_code": self.status_code,
            "response": self.response,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
        }


class IntegrationHub:
    """
    Integration hub for external services.

    The hub:
    - Manages integration configurations
    - Sends messages to external services
    - Handles rate limiting and retries
    - Tracks delivery status

    Example:
        hub = IntegrationHub()

        # Configure Slack integration
        hub.register_integration(IntegrationConfig(
            integration_id="slack-alerts",
            name="Slack Alerts",
            integration_type=IntegrationType.SLACK,
            config={"webhook_url": "https://hooks.slack.com/..."},
        ))

        # Send alert
        result = await hub.send_message(
            integration_id="slack-alerts",
            message_type="alert",
            payload={"text": "Alert: High CPU usage"},
        )
    """

    BUS_TOPICS = {
        "send": "monitor.integration.send",
        "receive": "monitor.integration.receive",
        "status": "monitor.integration.status",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        bus_dir: Optional[str] = None,
    ):
        """Initialize integration hub.

        Args:
            bus_dir: Bus directory
        """
        self._integrations: Dict[str, IntegrationConfig] = {}
        self._handlers: Dict[IntegrationType, Callable] = {}
        self._status: Dict[str, IntegrationStatus] = {}
        self._delivery_history: List[DeliveryResult] = []
        self._rate_counters: Dict[str, List[float]] = {}
        self._last_heartbeat = time.time()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default handlers
        self._register_default_handlers()

    def register_integration(self, config: IntegrationConfig) -> None:
        """Register an integration.

        Args:
            config: Integration configuration
        """
        self._integrations[config.integration_id] = config
        self._status[config.integration_id] = (
            IntegrationStatus.ACTIVE if config.enabled
            else IntegrationStatus.INACTIVE
        )

        self._emit_bus_event(
            self.BUS_TOPICS["status"],
            {
                "integration_id": config.integration_id,
                "action": "registered",
                "type": config.integration_type.value,
            }
        )

    def unregister_integration(self, integration_id: str) -> bool:
        """Unregister an integration.

        Args:
            integration_id: Integration ID

        Returns:
            True if removed
        """
        if integration_id in self._integrations:
            del self._integrations[integration_id]
            del self._status[integration_id]
            return True
        return False

    def get_integration(self, integration_id: str) -> Optional[IntegrationConfig]:
        """Get an integration by ID.

        Args:
            integration_id: Integration ID

        Returns:
            Integration config or None
        """
        return self._integrations.get(integration_id)

    def list_integrations(self) -> List[Dict[str, Any]]:
        """List all integrations.

        Returns:
            Integration summaries
        """
        return [
            {
                **config.to_dict(),
                "status": self._status.get(config.integration_id, IntegrationStatus.INACTIVE).value,
            }
            for config in self._integrations.values()
        ]

    def enable_integration(self, integration_id: str) -> bool:
        """Enable an integration.

        Args:
            integration_id: Integration ID

        Returns:
            True if enabled
        """
        config = self._integrations.get(integration_id)
        if config:
            config.enabled = True
            self._status[integration_id] = IntegrationStatus.ACTIVE
            return True
        return False

    def disable_integration(self, integration_id: str) -> bool:
        """Disable an integration.

        Args:
            integration_id: Integration ID

        Returns:
            True if disabled
        """
        config = self._integrations.get(integration_id)
        if config:
            config.enabled = False
            self._status[integration_id] = IntegrationStatus.INACTIVE
            return True
        return False

    def register_handler(
        self,
        integration_type: IntegrationType,
        handler: Callable[..., Coroutine[Any, Any, DeliveryResult]],
    ) -> None:
        """Register a handler for an integration type.

        Args:
            integration_type: Type of integration
            handler: Async handler function
        """
        self._handlers[integration_type] = handler

    async def send_message(
        self,
        integration_id: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: str = "normal",
    ) -> DeliveryResult:
        """Send a message via an integration.

        Args:
            integration_id: Integration ID
            message_type: Message type
            payload: Message payload
            priority: Message priority

        Returns:
            Delivery result
        """
        config = self._integrations.get(integration_id)
        if not config:
            return DeliveryResult(
                message_id=f"msg-{uuid.uuid4().hex[:8]}",
                integration_id=integration_id,
                success=False,
                error="Integration not found",
            )

        if not config.enabled:
            return DeliveryResult(
                message_id=f"msg-{uuid.uuid4().hex[:8]}",
                integration_id=integration_id,
                success=False,
                error="Integration is disabled",
            )

        # Check rate limit
        if not self._check_rate_limit(integration_id, config.rate_limit):
            self._status[integration_id] = IntegrationStatus.RATE_LIMITED
            return DeliveryResult(
                message_id=f"msg-{uuid.uuid4().hex[:8]}",
                integration_id=integration_id,
                success=False,
                error="Rate limit exceeded",
            )

        message = IntegrationMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            integration_id=integration_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
        )

        # Get handler
        handler = self._handlers.get(config.integration_type)
        if not handler:
            return DeliveryResult(
                message_id=message.message_id,
                integration_id=integration_id,
                success=False,
                error=f"No handler for {config.integration_type.value}",
            )

        # Execute with retries
        result = await self._execute_with_retry(handler, message, config)

        # Update status
        if result.success:
            self._status[integration_id] = IntegrationStatus.ACTIVE
        else:
            self._status[integration_id] = IntegrationStatus.ERROR

        # Store in history
        self._delivery_history.append(result)
        if len(self._delivery_history) > 1000:
            self._delivery_history = self._delivery_history[-1000:]

        self._emit_bus_event(
            self.BUS_TOPICS["send"],
            {
                "message_id": message.message_id,
                "integration_id": integration_id,
                "success": result.success,
            }
        )

        return result

    async def send_to_all(
        self,
        message_type: str,
        payload: Dict[str, Any],
        filter_types: Optional[List[IntegrationType]] = None,
    ) -> List[DeliveryResult]:
        """Send message to all enabled integrations.

        Args:
            message_type: Message type
            payload: Message payload
            filter_types: Filter by integration types

        Returns:
            Delivery results
        """
        results = []

        for config in self._integrations.values():
            if not config.enabled:
                continue

            if filter_types and config.integration_type not in filter_types:
                continue

            result = await self.send_message(
                config.integration_id,
                message_type,
                payload,
            )
            results.append(result)

        return results

    def get_delivery_history(
        self,
        integration_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[DeliveryResult]:
        """Get delivery history.

        Args:
            integration_id: Filter by integration
            limit: Maximum results

        Returns:
            Delivery history
        """
        history = self._delivery_history

        if integration_id:
            history = [r for r in history if r.integration_id == integration_id]

        return list(reversed(history[-limit:]))

    def get_status(self, integration_id: str) -> Optional[IntegrationStatus]:
        """Get integration status.

        Args:
            integration_id: Integration ID

        Returns:
            Status or None
        """
        return self._status.get(integration_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get hub statistics.

        Returns:
            Statistics
        """
        by_type: Dict[str, int] = {}
        for config in self._integrations.values():
            itype = config.integration_type.value
            by_type[itype] = by_type.get(itype, 0) + 1

        recent = self._delivery_history[-100:]
        success_count = sum(1 for r in recent if r.success)
        success_rate = success_count / len(recent) if recent else 0.0

        return {
            "total_integrations": len(self._integrations),
            "active_integrations": sum(
                1 for s in self._status.values()
                if s == IntegrationStatus.ACTIVE
            ),
            "by_type": by_type,
            "total_deliveries": len(self._delivery_history),
            "recent_success_rate": success_rate,
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
                "component": "integration_hub",
                "status": "healthy",
                "integrations": len(self._integrations),
            }
        )

        return True

    def _check_rate_limit(
        self,
        integration_id: str,
        limit: int,
    ) -> bool:
        """Check if rate limit allows request.

        Args:
            integration_id: Integration ID
            limit: Rate limit per minute

        Returns:
            True if allowed
        """
        now = time.time()
        minute_ago = now - 60

        # Get recent requests
        if integration_id not in self._rate_counters:
            self._rate_counters[integration_id] = []

        # Prune old entries
        self._rate_counters[integration_id] = [
            t for t in self._rate_counters[integration_id]
            if t >= minute_ago
        ]

        # Check limit
        if len(self._rate_counters[integration_id]) >= limit:
            return False

        # Record request
        self._rate_counters[integration_id].append(now)
        return True

    async def _execute_with_retry(
        self,
        handler: Callable,
        message: IntegrationMessage,
        config: IntegrationConfig,
    ) -> DeliveryResult:
        """Execute handler with retries.

        Args:
            handler: Handler function
            message: Message to send
            config: Integration config

        Returns:
            Delivery result
        """
        start_time = time.time()
        retry = 0

        while retry <= config.retry_count:
            try:
                result = await asyncio.wait_for(
                    handler(message, config),
                    timeout=config.timeout_s
                )

                result.retry_count = retry
                result.duration_ms = (time.time() - start_time) * 1000
                return result

            except asyncio.TimeoutError:
                error = f"Request timed out after {config.timeout_s}s"
            except Exception as e:
                error = str(e)

            retry += 1
            if retry <= config.retry_count:
                await asyncio.sleep(min(5 * retry, 30))

        return DeliveryResult(
            message_id=message.message_id,
            integration_id=message.integration_id,
            success=False,
            error=error,
            duration_ms=(time.time() - start_time) * 1000,
            retry_count=retry - 1,
        )

    def _register_default_handlers(self) -> None:
        """Register default integration handlers."""

        async def webhook_handler(
            message: IntegrationMessage,
            config: IntegrationConfig,
        ) -> DeliveryResult:
            """Handle webhook integration."""
            # Simulate webhook delivery
            return DeliveryResult(
                message_id=message.message_id,
                integration_id=message.integration_id,
                success=True,
                status_code=200,
                response={"status": "delivered"},
            )

        async def slack_handler(
            message: IntegrationMessage,
            config: IntegrationConfig,
        ) -> DeliveryResult:
            """Handle Slack integration."""
            # Simulate Slack delivery
            return DeliveryResult(
                message_id=message.message_id,
                integration_id=message.integration_id,
                success=True,
                status_code=200,
                response={"ok": True},
            )

        async def email_handler(
            message: IntegrationMessage,
            config: IntegrationConfig,
        ) -> DeliveryResult:
            """Handle email integration."""
            # Simulate email delivery
            return DeliveryResult(
                message_id=message.message_id,
                integration_id=message.integration_id,
                success=True,
                response={"queued": True},
            )

        async def pagerduty_handler(
            message: IntegrationMessage,
            config: IntegrationConfig,
        ) -> DeliveryResult:
            """Handle PagerDuty integration."""
            # Simulate PagerDuty delivery
            return DeliveryResult(
                message_id=message.message_id,
                integration_id=message.integration_id,
                success=True,
                status_code=202,
                response={"status": "accepted"},
            )

        async def custom_handler(
            message: IntegrationMessage,
            config: IntegrationConfig,
        ) -> DeliveryResult:
            """Handle custom integration."""
            return DeliveryResult(
                message_id=message.message_id,
                integration_id=message.integration_id,
                success=True,
                response={"processed": True},
            )

        self._handlers[IntegrationType.WEBHOOK] = webhook_handler
        self._handlers[IntegrationType.SLACK] = slack_handler
        self._handlers[IntegrationType.EMAIL] = email_handler
        self._handlers[IntegrationType.PAGERDUTY] = pagerduty_handler
        self._handlers[IntegrationType.CUSTOM] = custom_handler

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
_hub: Optional[IntegrationHub] = None


def get_hub() -> IntegrationHub:
    """Get or create the integration hub singleton.

    Returns:
        IntegrationHub instance
    """
    global _hub
    if _hub is None:
        _hub = IntegrationHub()
    return _hub


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Integration Hub (Step 277)")
    parser.add_argument("--list", action="store_true", help="List integrations")
    parser.add_argument("--add", metavar="NAME", help="Add integration")
    parser.add_argument("--type", default="webhook", help="Integration type")
    parser.add_argument("--enable", metavar="ID", help="Enable integration")
    parser.add_argument("--disable", metavar="ID", help="Disable integration")
    parser.add_argument("--send", metavar="ID", help="Send test message")
    parser.add_argument("--history", action="store_true", help="Show delivery history")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    hub = get_hub()

    if args.add:
        config = IntegrationConfig(
            integration_id=f"int-{uuid.uuid4().hex[:8]}",
            name=args.add,
            integration_type=IntegrationType(args.type),
        )
        hub.register_integration(config)
        print(f"Added integration: {config.integration_id}")

    if args.list:
        integrations = hub.list_integrations()
        if args.json:
            print(json.dumps(integrations, indent=2))
        else:
            print("Integrations:")
            for i in integrations:
                status = i.get("status", "unknown")
                print(f"  [{i['integration_id']}] {i['name']} ({i['integration_type']}) - {status}")

    if args.enable:
        success = hub.enable_integration(args.enable)
        print(f"Enabled: {success}")

    if args.disable:
        success = hub.disable_integration(args.disable)
        print(f"Disabled: {success}")

    if args.send:
        async def run():
            return await hub.send_message(
                args.send,
                "test",
                {"message": "Test message"},
            )

        result = asyncio.run(run())
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Delivery: {'SUCCESS' if result.success else 'FAILED'}")
            if result.error:
                print(f"  Error: {result.error}")

    if args.history:
        history = hub.get_delivery_history()
        if args.json:
            print(json.dumps([r.to_dict() for r in history], indent=2))
        else:
            print("Delivery History:")
            for r in history:
                status = "OK" if r.success else "FAIL"
                print(f"  [{status}] {r.message_id} -> {r.integration_id}")

    if args.stats:
        stats = hub.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Integration Hub Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
