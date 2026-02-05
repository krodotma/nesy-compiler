#!/usr/bin/env python3
"""
blue_green.py - Blue-Green Deployment Manager (Step 206)

PBTSO Phase: ITERATE
A2A Integration: Manages blue-green via deploy.bluegreen.switch

Provides:
- SlotType: Blue or Green slot
- BlueGreenState: State of blue-green deployment
- BlueGreenDeploymentManager: Manages blue-green deployments

Bus Topics:
- deploy.bluegreen.switch
- deploy.bluegreen.rollback
- deploy.bluegreen.deploy
- deploy.bluegreen.status

Protocol: DKIN v30
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==============================================================================
# Bus Emission Helper
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
    actor: str = "bluegreen-manager"
) -> str:
    """Emit an event to the Pluribus bus."""
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
            f.write(json.dumps(event) + "\n")
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class SlotType(Enum):
    """Blue-green slot types."""
    BLUE = "blue"
    GREEN = "green"


@dataclass
class SlotState:
    """
    State of a deployment slot.

    Attributes:
        slot: Slot type (blue or green)
        artifact_id: Deployed artifact ID
        version: Deployed version
        deployed_at: Timestamp when deployed
        health_status: Health check status
        endpoint: Service endpoint
        replicas: Number of replicas
    """
    slot: SlotType
    artifact_id: str = ""
    version: str = ""
    deployed_at: float = 0.0
    health_status: str = "unknown"
    endpoint: str = ""
    replicas: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot.value,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "deployed_at": self.deployed_at,
            "health_status": self.health_status,
            "endpoint": self.endpoint,
            "replicas": self.replicas,
        }


@dataclass
class BlueGreenState:
    """
    State of a blue-green deployment.

    Attributes:
        deployment_id: Unique deployment identifier
        service_name: Name of the service
        active: Currently active slot
        standby: Standby slot
        blue: Blue slot state
        green: Green slot state
        last_switch_ts: Timestamp of last traffic switch
        switch_count: Number of switches performed
        created_at: Timestamp when created
    """
    deployment_id: str
    service_name: str
    active: SlotType = SlotType.BLUE
    standby: SlotType = SlotType.GREEN
    blue: SlotState = field(default_factory=lambda: SlotState(slot=SlotType.BLUE))
    green: SlotState = field(default_factory=lambda: SlotState(slot=SlotType.GREEN))
    last_switch_ts: float = 0.0
    switch_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "service_name": self.service_name,
            "active": self.active.value,
            "standby": self.standby.value,
            "blue": self.blue.to_dict(),
            "green": self.green.to_dict(),
            "last_switch_ts": self.last_switch_ts,
            "switch_count": self.switch_count,
            "created_at": self.created_at,
        }


# ==============================================================================
# Blue-Green Deployment Manager (Step 206)
# ==============================================================================

class BlueGreenDeploymentManager:
    """
    Blue-Green Deployment Manager - manages blue-green deployment strategy.

    PBTSO Phase: ITERATE

    The blue-green strategy maintains two identical environments:
    - Blue (active): Serving live traffic
    - Green (standby): Ready for new deployments

    Workflow:
    1. Deploy new version to standby slot
    2. Run health checks on standby
    3. Switch traffic from active to standby
    4. Old active becomes new standby (ready for rollback)

    Example:
        >>> manager = BlueGreenDeploymentManager()
        >>> state = manager.create_deployment("myservice")
        >>> await manager.deploy_to_standby(state, artifact_id="artifact-123", version="v2.0")
        >>> if await manager.verify_standby(state):
        ...     await manager.switch_traffic(state)
    """

    BUS_TOPICS = {
        "deploy": "deploy.bluegreen.deploy",
        "switch": "deploy.bluegreen.switch",
        "rollback": "deploy.bluegreen.rollback",
        "status": "deploy.bluegreen.status",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "bluegreen-manager",
    ):
        """
        Initialize the blue-green deployment manager.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "bluegreen"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self._deployments: Dict[str, BlueGreenState] = {}
        self._load_deployments()

    def create_deployment(self, service_name: str) -> BlueGreenState:
        """
        Create a new blue-green deployment for a service.

        Args:
            service_name: Name of the service

        Returns:
            BlueGreenState for the deployment
        """
        deployment_id = f"bg-{uuid.uuid4().hex[:12]}"

        state = BlueGreenState(
            deployment_id=deployment_id,
            service_name=service_name,
        )

        self._deployments[deployment_id] = state
        self._save_deployment(state)

        _emit_bus_event(
            "deploy.bluegreen.created",
            {
                "deployment_id": deployment_id,
                "service_name": service_name,
            },
            actor=self.actor_id,
        )

        return state

    async def deploy_to_standby(
        self,
        state: BlueGreenState,
        artifact_id: str,
        version: str,
        replicas: int = 1,
    ) -> bool:
        """
        Deploy artifact to the standby slot.

        Args:
            state: BlueGreenState
            artifact_id: Artifact ID to deploy
            version: Version string
            replicas: Number of replicas

        Returns:
            True if deployment successful
        """
        standby_slot = state.standby
        slot_state = state.blue if standby_slot == SlotType.BLUE else state.green

        _emit_bus_event(
            self.BUS_TOPICS["deploy"],
            {
                "deployment_id": state.deployment_id,
                "service_name": state.service_name,
                "slot": standby_slot.value,
                "artifact_id": artifact_id,
                "version": version,
            },
            actor=self.actor_id,
        )

        try:
            # Simulate deployment
            await asyncio.sleep(0.1)

            slot_state.artifact_id = artifact_id
            slot_state.version = version
            slot_state.deployed_at = time.time()
            slot_state.replicas = replicas
            slot_state.health_status = "pending"
            slot_state.endpoint = f"https://{state.service_name}-{standby_slot.value}.local"

            self._save_deployment(state)
            return True

        except Exception as e:
            slot_state.health_status = "failed"
            self._save_deployment(state)
            return False

    async def verify_standby(
        self,
        state: BlueGreenState,
        timeout_s: int = 60,
    ) -> bool:
        """
        Verify the standby slot is healthy.

        Args:
            state: BlueGreenState
            timeout_s: Timeout for health checks

        Returns:
            True if standby is healthy
        """
        standby_slot = state.standby
        slot_state = state.blue if standby_slot == SlotType.BLUE else state.green

        start = time.time()
        while time.time() - start < timeout_s:
            # Simulate health check
            await asyncio.sleep(0.1)
            slot_state.health_status = "healthy"
            self._save_deployment(state)
            return True

        slot_state.health_status = "unhealthy"
        self._save_deployment(state)
        return False

    async def switch_traffic(self, state: BlueGreenState) -> bool:
        """
        Switch traffic from active to standby.

        Args:
            state: BlueGreenState

        Returns:
            True if switch successful
        """
        old_active = state.active
        new_active = state.standby

        _emit_bus_event(
            self.BUS_TOPICS["switch"],
            {
                "deployment_id": state.deployment_id,
                "service_name": state.service_name,
                "from_slot": old_active.value,
                "to_slot": new_active.value,
            },
            actor=self.actor_id,
        )

        try:
            # Simulate traffic switch
            await asyncio.sleep(0.05)

            # Swap active/standby
            state.active = new_active
            state.standby = old_active
            state.last_switch_ts = time.time()
            state.switch_count += 1

            self._save_deployment(state)
            return True

        except Exception:
            return False

    async def rollback(self, state: BlueGreenState) -> bool:
        """
        Rollback by switching back to previous active.

        Args:
            state: BlueGreenState

        Returns:
            True if rollback successful
        """
        _emit_bus_event(
            self.BUS_TOPICS["rollback"],
            {
                "deployment_id": state.deployment_id,
                "service_name": state.service_name,
                "from_slot": state.active.value,
                "to_slot": state.standby.value,
            },
            actor=self.actor_id,
        )

        return await self.switch_traffic(state)

    def get_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get deployment status.

        Args:
            deployment_id: Deployment ID

        Returns:
            Status dictionary
        """
        state = self._deployments.get(deployment_id)
        if not state:
            return None

        active_slot = state.blue if state.active == SlotType.BLUE else state.green

        status = {
            "deployment_id": deployment_id,
            "service_name": state.service_name,
            "active_slot": state.active.value,
            "standby_slot": state.standby.value,
            "active_version": active_slot.version,
            "active_health": active_slot.health_status,
            "switch_count": state.switch_count,
            "last_switch_ts": state.last_switch_ts,
        }

        _emit_bus_event(
            self.BUS_TOPICS["status"],
            status,
            kind="metric",
            actor=self.actor_id,
        )

        return status

    def get_deployment(self, deployment_id: str) -> Optional[BlueGreenState]:
        """Get a deployment by ID."""
        return self._deployments.get(deployment_id)

    def list_deployments(self) -> List[BlueGreenState]:
        """List all deployments."""
        return list(self._deployments.values())

    def _save_deployment(self, state: BlueGreenState) -> None:
        """Save deployment state to disk."""
        state_file = self.state_dir / f"{state.deployment_id}.json"
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _load_deployments(self) -> None:
        """Load deployments from disk."""
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)

                    blue = SlotState(
                        slot=SlotType.BLUE,
                        **{k: v for k, v in data.get("blue", {}).items() if k != "slot"},
                    )
                    green = SlotState(
                        slot=SlotType.GREEN,
                        **{k: v for k, v in data.get("green", {}).items() if k != "slot"},
                    )

                    state = BlueGreenState(
                        deployment_id=data["deployment_id"],
                        service_name=data["service_name"],
                        active=SlotType(data["active"]),
                        standby=SlotType(data["standby"]),
                        blue=blue,
                        green=green,
                        last_switch_ts=data.get("last_switch_ts", 0.0),
                        switch_count=data.get("switch_count", 0),
                        created_at=data.get("created_at", time.time()),
                    )

                    self._deployments[state.deployment_id] = state
            except (json.JSONDecodeError, KeyError, IOError):
                continue


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for blue-green deployment manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Blue-Green Deployment Manager (Step 206)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a deployment")
    create_parser.add_argument("service_name", help="Service name")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to standby")
    deploy_parser.add_argument("deployment_id", help="Deployment ID")
    deploy_parser.add_argument("--artifact", "-a", required=True, help="Artifact ID")
    deploy_parser.add_argument("--version", "-v", required=True, help="Version")
    deploy_parser.add_argument("--json", action="store_true", help="JSON output")

    # switch command
    switch_parser = subparsers.add_parser("switch", help="Switch traffic")
    switch_parser.add_argument("deployment_id", help="Deployment ID")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous")
    rollback_parser.add_argument("deployment_id", help="Deployment ID")

    # status command
    status_parser = subparsers.add_parser("status", help="Get status")
    status_parser.add_argument("deployment_id", help="Deployment ID")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List deployments")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    manager = BlueGreenDeploymentManager()

    if args.command == "create":
        state = manager.create_deployment(args.service_name)
        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            print(f"Created deployment: {state.deployment_id}")
            print(f"  Service: {state.service_name}")
            print(f"  Active: {state.active.value}")
        return 0

    elif args.command == "deploy":
        state = manager.get_deployment(args.deployment_id)
        if not state:
            print(f"Deployment not found: {args.deployment_id}")
            return 1

        success = asyncio.get_event_loop().run_until_complete(
            manager.deploy_to_standby(state, args.artifact, args.version)
        )

        if args.json:
            print(json.dumps({"success": success, "deployment": state.to_dict()}, indent=2))
        else:
            if success:
                print(f"Deployed to {state.standby.value}")
            else:
                print("Deployment failed")

        return 0 if success else 1

    elif args.command == "switch":
        state = manager.get_deployment(args.deployment_id)
        if not state:
            print(f"Deployment not found: {args.deployment_id}")
            return 1

        success = asyncio.get_event_loop().run_until_complete(
            manager.switch_traffic(state)
        )

        if success:
            print(f"Traffic switched to {state.active.value}")
        else:
            print("Switch failed")

        return 0 if success else 1

    elif args.command == "rollback":
        state = manager.get_deployment(args.deployment_id)
        if not state:
            print(f"Deployment not found: {args.deployment_id}")
            return 1

        success = asyncio.get_event_loop().run_until_complete(
            manager.rollback(state)
        )

        if success:
            print(f"Rolled back to {state.active.value}")
        else:
            print("Rollback failed")

        return 0 if success else 1

    elif args.command == "status":
        status = manager.get_status(args.deployment_id)
        if not status:
            print(f"Deployment not found: {args.deployment_id}")
            return 1

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Deployment: {status['deployment_id']}")
            print(f"  Service: {status['service_name']}")
            print(f"  Active: {status['active_slot']} ({status['active_version']})")
            print(f"  Standby: {status['standby_slot']}")
            print(f"  Switches: {status['switch_count']}")

        return 0

    elif args.command == "list":
        deployments = manager.list_deployments()
        if args.json:
            print(json.dumps([d.to_dict() for d in deployments], indent=2))
        else:
            if not deployments:
                print("No deployments found")
            else:
                for d in deployments:
                    print(f"{d.deployment_id} ({d.service_name}) - active: {d.active.value}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
