#!/usr/bin/env python3
"""
manager.py - Feature Flag Manager (Step 209)

PBTSO Phase: ITERATE
A2A Integration: Manages flags via deploy.flag.toggle

Provides:
- FlagType: Types of feature flags
- FlagState: State of a feature flag
- FeatureFlag: Feature flag definition
- FeatureFlagManager: Manages feature flags

Bus Topics:
- deploy.flag.toggle
- deploy.flag.status
- deploy.flag.created
- deploy.flag.deleted

Protocol: DKIN v30
"""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
    actor: str = "flag-manager"
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

class FlagType(Enum):
    """Types of feature flags."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ENVIRONMENT = "environment"
    CUSTOM = "custom"


class FlagState(Enum):
    """State of a feature flag."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PARTIAL = "partial"  # For percentage or user list rollouts


@dataclass
class FeatureFlag:
    """
    Feature flag definition.

    Attributes:
        flag_id: Unique flag identifier
        name: Human-readable flag name
        description: Flag description
        flag_type: Type of flag
        state: Current state
        value: Current value (depends on type)
        environments: Environments where flag is active
        targeting_rules: Rules for flag evaluation
        metadata: Additional metadata
        created_at: Timestamp when created
        updated_at: Timestamp when last updated
        expires_at: Optional expiration timestamp
    """
    flag_id: str
    name: str
    description: str = ""
    flag_type: FlagType = FlagType.BOOLEAN
    state: FlagState = FlagState.DISABLED
    value: Any = False
    environments: List[str] = field(default_factory=lambda: ["dev", "staging", "prod"])
    targeting_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flag_id": self.flag_id,
            "name": self.name,
            "description": self.description,
            "flag_type": self.flag_type.value,
            "state": self.state.value,
            "value": self.value,
            "environments": self.environments,
            "targeting_rules": self.targeting_rules,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureFlag":
        data = dict(data)
        if "flag_type" in data:
            data["flag_type"] = FlagType(data["flag_type"])
        if "state" in data:
            data["state"] = FlagState(data["state"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Feature Flag Manager (Step 209)
# ==============================================================================

class FeatureFlagManager:
    """
    Feature Flag Manager - manages feature flags for deployment.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Create and manage feature flags
    - Toggle flags on/off
    - Evaluate flags for users/environments
    - Track flag state changes
    - Emit flag events to A2A bus

    Example:
        >>> manager = FeatureFlagManager()
        >>> flag = manager.create_flag(
        ...     name="new_checkout_flow",
        ...     flag_type=FlagType.PERCENTAGE,
        ...     value=10  # 10% rollout
        ... )
        >>> if manager.evaluate(flag.flag_id, environment="prod"):
        ...     # Show new checkout flow
        ...     pass
    """

    BUS_TOPICS = {
        "toggle": "deploy.flag.toggle",
        "status": "deploy.flag.status",
        "created": "deploy.flag.created",
        "deleted": "deploy.flag.deleted",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "flag-manager",
    ):
        """
        Initialize the feature flag manager.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "flags"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._flags: Dict[str, FeatureFlag] = {}
        self._load_flags()

    def create_flag(
        self,
        name: str,
        flag_type: Union[FlagType, str] = FlagType.BOOLEAN,
        description: str = "",
        value: Any = False,
        environments: Optional[List[str]] = None,
        targeting_rules: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeatureFlag:
        """
        Create a new feature flag.

        Args:
            name: Human-readable flag name
            flag_type: Type of flag
            description: Flag description
            value: Initial value
            environments: Active environments
            targeting_rules: Targeting rules
            metadata: Additional metadata

        Returns:
            Created FeatureFlag
        """
        if isinstance(flag_type, str):
            flag_type = FlagType(flag_type)

        flag_id = f"flag-{uuid.uuid4().hex[:12]}"

        flag = FeatureFlag(
            flag_id=flag_id,
            name=name,
            description=description,
            flag_type=flag_type,
            value=value,
            environments=environments or ["dev", "staging", "prod"],
            targeting_rules=targeting_rules or {},
            metadata=metadata or {},
        )

        # Determine initial state
        if flag_type == FlagType.BOOLEAN:
            flag.state = FlagState.ENABLED if value else FlagState.DISABLED
        elif flag_type == FlagType.PERCENTAGE:
            if value == 0:
                flag.state = FlagState.DISABLED
            elif value == 100:
                flag.state = FlagState.ENABLED
            else:
                flag.state = FlagState.PARTIAL
        else:
            flag.state = FlagState.PARTIAL if value else FlagState.DISABLED

        self._flags[flag_id] = flag
        self._save_flag(flag)

        _emit_bus_event(
            self.BUS_TOPICS["created"],
            {
                "flag_id": flag_id,
                "name": name,
                "flag_type": flag_type.value,
                "state": flag.state.value,
                "value": value,
            },
            actor=self.actor_id,
        )

        return flag

    def toggle(
        self,
        flag_id: str,
        enabled: Optional[bool] = None,
        value: Any = None,
    ) -> Optional[FeatureFlag]:
        """
        Toggle a feature flag.

        Args:
            flag_id: Flag ID
            enabled: If provided, set enabled state
            value: If provided, set new value

        Returns:
            Updated FeatureFlag or None if not found
        """
        flag = self._flags.get(flag_id)
        if not flag:
            return None

        old_state = flag.state
        old_value = flag.value

        if enabled is not None:
            flag.state = FlagState.ENABLED if enabled else FlagState.DISABLED
            if flag.flag_type == FlagType.BOOLEAN:
                flag.value = enabled
            elif flag.flag_type == FlagType.PERCENTAGE:
                flag.value = 100 if enabled else 0

        if value is not None:
            flag.value = value
            # Update state based on new value
            if flag.flag_type == FlagType.BOOLEAN:
                flag.state = FlagState.ENABLED if value else FlagState.DISABLED
            elif flag.flag_type == FlagType.PERCENTAGE:
                if value == 0:
                    flag.state = FlagState.DISABLED
                elif value == 100:
                    flag.state = FlagState.ENABLED
                else:
                    flag.state = FlagState.PARTIAL

        flag.updated_at = time.time()
        self._save_flag(flag)

        _emit_bus_event(
            self.BUS_TOPICS["toggle"],
            {
                "flag_id": flag_id,
                "name": flag.name,
                "old_state": old_state.value,
                "new_state": flag.state.value,
                "old_value": old_value,
                "new_value": flag.value,
            },
            actor=self.actor_id,
        )

        return flag

    def evaluate(
        self,
        flag_id: str,
        environment: str = "prod",
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate a feature flag.

        Args:
            flag_id: Flag ID
            environment: Environment to evaluate for
            user_id: Optional user ID for targeting
            context: Optional context for evaluation

        Returns:
            True if flag is enabled for the given context
        """
        flag = self._flags.get(flag_id)
        if not flag:
            return False

        # Check expiration
        if flag.expires_at and time.time() > flag.expires_at:
            return False

        # Check environment
        if environment not in flag.environments:
            return False

        # Check state
        if flag.state == FlagState.DISABLED:
            return False

        if flag.state == FlagState.ENABLED:
            return True

        # Handle partial rollout
        if flag.flag_type == FlagType.PERCENTAGE:
            # Use user_id or random for percentage check
            if user_id:
                # Deterministic hash-based percentage
                hash_val = hash(f"{flag_id}:{user_id}") % 100
                return hash_val < flag.value
            else:
                # Random percentage
                import random
                return random.random() * 100 < flag.value

        if flag.flag_type == FlagType.USER_LIST:
            # Check if user is in the list
            user_list = flag.value if isinstance(flag.value, list) else []
            return user_id in user_list

        if flag.flag_type == FlagType.ENVIRONMENT:
            # Value is dict of environment -> enabled
            env_values = flag.value if isinstance(flag.value, dict) else {}
            return env_values.get(environment, False)

        # Default: return based on value truthiness
        return bool(flag.value)

    def get_flag(self, flag_id: str) -> Optional[FeatureFlag]:
        """Get a flag by ID."""
        return self._flags.get(flag_id)

    def get_flag_by_name(self, name: str) -> Optional[FeatureFlag]:
        """Get a flag by name."""
        for flag in self._flags.values():
            if flag.name == name:
                return flag
        return None

    def list_flags(
        self,
        environment: Optional[str] = None,
        state: Optional[FlagState] = None,
    ) -> List[FeatureFlag]:
        """List all flags."""
        flags = list(self._flags.values())

        if environment:
            flags = [f for f in flags if environment in f.environments]

        if state:
            flags = [f for f in flags if f.state == state]

        return flags

    def delete_flag(self, flag_id: str) -> bool:
        """Delete a flag."""
        flag = self._flags.get(flag_id)
        if not flag:
            return False

        del self._flags[flag_id]

        # Delete state file
        state_file = self.state_dir / f"{flag_id}.json"
        if state_file.exists():
            state_file.unlink()

        _emit_bus_event(
            self.BUS_TOPICS["deleted"],
            {
                "flag_id": flag_id,
                "name": flag.name,
            },
            actor=self.actor_id,
        )

        return True

    def get_status(self, flag_id: str) -> Optional[Dict[str, Any]]:
        """Get flag status."""
        flag = self._flags.get(flag_id)
        if not flag:
            return None

        status = {
            "flag_id": flag_id,
            "name": flag.name,
            "state": flag.state.value,
            "value": flag.value,
            "type": flag.flag_type.value,
            "environments": flag.environments,
            "expired": flag.expires_at is not None and time.time() > flag.expires_at,
        }

        _emit_bus_event(
            self.BUS_TOPICS["status"],
            status,
            kind="metric",
            actor=self.actor_id,
        )

        return status

    def _save_flag(self, flag: FeatureFlag) -> None:
        """Save flag to disk."""
        state_file = self.state_dir / f"{flag.flag_id}.json"
        with open(state_file, "w") as f:
            json.dump(flag.to_dict(), f, indent=2)

    def _load_flags(self) -> None:
        """Load flags from disk."""
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    flag = FeatureFlag.from_dict(data)
                    self._flags[flag.flag_id] = flag
            except (json.JSONDecodeError, KeyError, IOError):
                continue


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for feature flag manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature Flag Manager (Step 209)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a flag")
    create_parser.add_argument("name", help="Flag name")
    create_parser.add_argument("--type", "-t", default="boolean", choices=["boolean", "percentage", "user_list", "environment"])
    create_parser.add_argument("--value", "-v", help="Initial value")
    create_parser.add_argument("--description", "-d", default="", help="Description")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # toggle command
    toggle_parser = subparsers.add_parser("toggle", help="Toggle a flag")
    toggle_parser.add_argument("flag_id", help="Flag ID")
    toggle_parser.add_argument("--enable", action="store_true", help="Enable flag")
    toggle_parser.add_argument("--disable", action="store_true", help="Disable flag")
    toggle_parser.add_argument("--value", "-v", help="Set value")
    toggle_parser.add_argument("--json", action="store_true", help="JSON output")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a flag")
    eval_parser.add_argument("flag_id", help="Flag ID")
    eval_parser.add_argument("--env", "-e", default="prod", help="Environment")
    eval_parser.add_argument("--user", "-u", help="User ID")

    # list command
    list_parser = subparsers.add_parser("list", help="List flags")
    list_parser.add_argument("--env", "-e", help="Filter by environment")
    list_parser.add_argument("--state", "-s", choices=["enabled", "disabled", "partial"])
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a flag")
    delete_parser.add_argument("flag_id", help="Flag ID")

    args = parser.parse_args()
    manager = FeatureFlagManager()

    if args.command == "create":
        # Parse value based on type
        value = args.value
        if args.type == "boolean":
            value = args.value.lower() in ("true", "1", "yes") if args.value else False
        elif args.type == "percentage":
            value = int(args.value) if args.value else 0
        elif args.type == "user_list":
            value = args.value.split(",") if args.value else []

        flag = manager.create_flag(
            name=args.name,
            flag_type=args.type,
            description=args.description,
            value=value,
        )

        if args.json:
            print(json.dumps(flag.to_dict(), indent=2))
        else:
            print(f"Created flag: {flag.flag_id}")
            print(f"  Name: {flag.name}")
            print(f"  Type: {flag.flag_type.value}")
            print(f"  State: {flag.state.value}")
            print(f"  Value: {flag.value}")

        return 0

    elif args.command == "toggle":
        enabled = None
        if args.enable:
            enabled = True
        elif args.disable:
            enabled = False

        value = None
        if args.value:
            # Try to parse as int or bool
            try:
                value = int(args.value)
            except ValueError:
                value = args.value.lower() in ("true", "1", "yes")

        flag = manager.toggle(args.flag_id, enabled=enabled, value=value)

        if not flag:
            print(f"Flag not found: {args.flag_id}")
            return 1

        if args.json:
            print(json.dumps(flag.to_dict(), indent=2))
        else:
            print(f"Toggled flag: {flag.flag_id}")
            print(f"  State: {flag.state.value}")
            print(f"  Value: {flag.value}")

        return 0

    elif args.command == "evaluate":
        result = manager.evaluate(
            args.flag_id,
            environment=args.env,
            user_id=args.user,
        )

        print(f"Flag {args.flag_id} = {result}")
        return 0 if result else 1

    elif args.command == "list":
        state = FlagState(args.state) if args.state else None
        flags = manager.list_flags(environment=args.env, state=state)

        if args.json:
            print(json.dumps([f.to_dict() for f in flags], indent=2))
        else:
            if not flags:
                print("No flags found")
            else:
                for f in flags:
                    print(f"{f.flag_id} ({f.name}) - {f.state.value}")

        return 0

    elif args.command == "delete":
        success = manager.delete_flag(args.flag_id)
        if success:
            print(f"Deleted flag: {args.flag_id}")
        else:
            print(f"Flag not found: {args.flag_id}")
        return 0 if success else 1

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
