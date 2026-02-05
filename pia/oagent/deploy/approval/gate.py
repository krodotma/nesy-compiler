#!/usr/bin/env python3
"""
gate.py - Deployment Approval Gate (Step 224)

PBTSO Phase: PLAN, VERIFY
A2A Integration: Manages deployment approvals via deploy.approval.*

Provides:
- ApprovalStatus: Approval status enum
- ApprovalType: Types of approval
- ApprovalLevel: Approval levels
- Approver: Approver information
- ApprovalRequest: Approval request
- ApprovalPolicy: Approval policy configuration
- DeploymentApprovalGate: Main approval gate class

Bus Topics:
- deploy.approval.request
- deploy.approval.approve
- deploy.approval.reject
- deploy.approval.timeout

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
    actor: str = "approval-gate"
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

class ApprovalStatus(Enum):
    """Approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    AUTO_APPROVED = "auto_approved"


class ApprovalType(Enum):
    """Types of approval."""
    MANUAL = "manual"           # Requires human approval
    AUTO = "auto"               # Auto-approve based on conditions
    REVIEW = "review"           # Approval with mandatory review
    EMERGENCY = "emergency"     # Emergency bypass
    SCHEDULED = "scheduled"     # Auto-approve at scheduled time


class ApprovalLevel(Enum):
    """Approval levels."""
    NONE = "none"              # No approval required
    TEAM = "team"              # Team lead approval
    MANAGER = "manager"        # Manager approval
    DIRECTOR = "director"      # Director approval
    EXECUTIVE = "executive"    # Executive approval


@dataclass
class Approver:
    """
    Approver information.

    Attributes:
        approver_id: Unique approver identifier
        name: Approver name
        email: Approver email
        role: Approver role
        level: Approval level
        teams: Teams the approver belongs to
        active: Whether approver is active
    """
    approver_id: str
    name: str
    email: str = ""
    role: str = ""
    level: ApprovalLevel = ApprovalLevel.TEAM
    teams: List[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approver_id": self.approver_id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "level": self.level.value,
            "teams": self.teams,
            "active": self.active,
        }


@dataclass
class ApprovalDecision:
    """
    Individual approval decision.

    Attributes:
        approver_id: Approver who made decision
        decision: Approved or rejected
        timestamp: Decision timestamp
        comment: Optional comment
    """
    approver_id: str
    decision: ApprovalStatus
    timestamp: float = field(default_factory=time.time)
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approver_id": self.approver_id,
            "decision": self.decision.value,
            "timestamp": self.timestamp,
            "comment": self.comment,
        }


@dataclass
class ApprovalRequest:
    """
    Approval request.

    Attributes:
        request_id: Unique request identifier
        deployment_id: Associated deployment
        service_name: Service being deployed
        version: Version to deploy
        environment: Target environment
        approval_type: Type of approval
        required_level: Required approval level
        required_approvals: Number of approvals needed
        status: Current status
        requestor: Requestor identifier
        decisions: Approval decisions
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        metadata: Additional metadata
    """
    request_id: str
    deployment_id: str
    service_name: str
    version: str
    environment: str = "prod"
    approval_type: ApprovalType = ApprovalType.MANUAL
    required_level: ApprovalLevel = ApprovalLevel.TEAM
    required_approvals: int = 1
    status: ApprovalStatus = ApprovalStatus.PENDING
    requestor: str = ""
    decisions: List[ApprovalDecision] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "deployment_id": self.deployment_id,
            "service_name": self.service_name,
            "version": self.version,
            "environment": self.environment,
            "approval_type": self.approval_type.value,
            "required_level": self.required_level.value,
            "required_approvals": self.required_approvals,
            "status": self.status.value,
            "requestor": self.requestor,
            "decisions": [d.to_dict() for d in self.decisions],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRequest":
        data = dict(data)
        if "approval_type" in data:
            data["approval_type"] = ApprovalType(data["approval_type"])
        if "required_level" in data:
            data["required_level"] = ApprovalLevel(data["required_level"])
        if "status" in data:
            data["status"] = ApprovalStatus(data["status"])
        if "decisions" in data:
            data["decisions"] = [
                ApprovalDecision(
                    approver_id=d["approver_id"],
                    decision=ApprovalStatus(d["decision"]),
                    timestamp=d.get("timestamp", 0),
                    comment=d.get("comment", ""),
                )
                for d in data["decisions"]
            ]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def approval_count(self) -> int:
        """Count of approvals."""
        return sum(
            1 for d in self.decisions
            if d.decision == ApprovalStatus.APPROVED
        )

    @property
    def is_approved(self) -> bool:
        """Check if request has enough approvals."""
        return self.approval_count >= self.required_approvals


@dataclass
class ApprovalPolicy:
    """
    Approval policy configuration.

    Attributes:
        policy_id: Unique policy identifier
        name: Policy name
        environments: Environments policy applies to
        services: Services policy applies to (empty = all)
        required_level: Required approval level
        required_approvals: Number of approvals needed
        approval_type: Default approval type
        timeout_hours: Hours before expiration
        auto_approve_conditions: Conditions for auto-approval
        enabled: Whether policy is enabled
    """
    policy_id: str
    name: str
    environments: List[str] = field(default_factory=lambda: ["prod"])
    services: List[str] = field(default_factory=list)
    required_level: ApprovalLevel = ApprovalLevel.TEAM
    required_approvals: int = 1
    approval_type: ApprovalType = ApprovalType.MANUAL
    timeout_hours: int = 24
    auto_approve_conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "environments": self.environments,
            "services": self.services,
            "required_level": self.required_level.value,
            "required_approvals": self.required_approvals,
            "approval_type": self.approval_type.value,
            "timeout_hours": self.timeout_hours,
            "auto_approve_conditions": self.auto_approve_conditions,
            "enabled": self.enabled,
        }

    def matches(self, service_name: str, environment: str) -> bool:
        """Check if policy matches service and environment."""
        if not self.enabled:
            return False
        if environment not in self.environments:
            return False
        if self.services and service_name not in self.services:
            return False
        return True


# ==============================================================================
# Deployment Approval Gate (Step 224)
# ==============================================================================

class DeploymentApprovalGate:
    """
    Deployment Approval Gate - manages deployment approvals and workflows.

    PBTSO Phase: PLAN, VERIFY

    Responsibilities:
    - Create approval requests for deployments
    - Manage approver registry
    - Enforce approval policies
    - Track approval decisions
    - Handle approval timeouts

    Example:
        >>> gate = DeploymentApprovalGate()
        >>> request = gate.create_request(
        ...     deployment_id="deploy-123",
        ...     service_name="api",
        ...     version="v2.0.0",
        ...     environment="prod",
        ... )
        >>> gate.approve(request.request_id, approver_id="user-1")
        >>> print(f"Approved: {request.is_approved}")
    """

    BUS_TOPICS = {
        "request": "deploy.approval.request",
        "approve": "deploy.approval.approve",
        "reject": "deploy.approval.reject",
        "timeout": "deploy.approval.timeout",
        "cancelled": "deploy.approval.cancelled",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "approval-gate",
        default_timeout_hours: int = 24,
    ):
        """
        Initialize the approval gate.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            default_timeout_hours: Default approval timeout
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "approvals"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.default_timeout_hours = default_timeout_hours

        self._requests: Dict[str, ApprovalRequest] = {}
        self._approvers: Dict[str, Approver] = {}
        self._policies: Dict[str, ApprovalPolicy] = {}
        self._notification_callback: Optional[Callable] = None

        self._load_state()

    def register_approver(
        self,
        name: str,
        email: str = "",
        role: str = "",
        level: ApprovalLevel = ApprovalLevel.TEAM,
        teams: Optional[List[str]] = None,
    ) -> Approver:
        """
        Register an approver.

        Args:
            name: Approver name
            email: Approver email
            role: Approver role
            level: Approval level
            teams: Teams the approver belongs to

        Returns:
            Created Approver
        """
        approver_id = f"approver-{uuid.uuid4().hex[:12]}"

        approver = Approver(
            approver_id=approver_id,
            name=name,
            email=email,
            role=role,
            level=level,
            teams=teams or [],
        )

        self._approvers[approver_id] = approver
        self._save_state()

        return approver

    def create_policy(
        self,
        name: str,
        environments: Optional[List[str]] = None,
        services: Optional[List[str]] = None,
        required_level: ApprovalLevel = ApprovalLevel.TEAM,
        required_approvals: int = 1,
        approval_type: ApprovalType = ApprovalType.MANUAL,
        timeout_hours: int = 24,
    ) -> ApprovalPolicy:
        """
        Create an approval policy.

        Args:
            name: Policy name
            environments: Target environments
            services: Target services (empty = all)
            required_level: Required approval level
            required_approvals: Number of approvals needed
            approval_type: Default approval type
            timeout_hours: Hours before expiration

        Returns:
            Created ApprovalPolicy
        """
        policy_id = f"policy-{uuid.uuid4().hex[:12]}"

        policy = ApprovalPolicy(
            policy_id=policy_id,
            name=name,
            environments=environments or ["prod"],
            services=services or [],
            required_level=required_level,
            required_approvals=required_approvals,
            approval_type=approval_type,
            timeout_hours=timeout_hours,
        )

        self._policies[policy_id] = policy
        self._save_state()

        return policy

    def create_request(
        self,
        deployment_id: str,
        service_name: str,
        version: str,
        environment: str,
        requestor: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        Create an approval request for a deployment.

        Args:
            deployment_id: Deployment identifier
            service_name: Service being deployed
            version: Version to deploy
            environment: Target environment
            requestor: Requestor identifier
            metadata: Additional metadata

        Returns:
            Created ApprovalRequest
        """
        request_id = f"approval-{uuid.uuid4().hex[:12]}"

        # Find matching policy
        policy = self._find_policy(service_name, environment)

        if policy:
            approval_type = policy.approval_type
            required_level = policy.required_level
            required_approvals = policy.required_approvals
            timeout_hours = policy.timeout_hours
        else:
            approval_type = ApprovalType.MANUAL
            required_level = ApprovalLevel.TEAM
            required_approvals = 1
            timeout_hours = self.default_timeout_hours

        expires_at = time.time() + (timeout_hours * 3600)

        request = ApprovalRequest(
            request_id=request_id,
            deployment_id=deployment_id,
            service_name=service_name,
            version=version,
            environment=environment,
            approval_type=approval_type,
            required_level=required_level,
            required_approvals=required_approvals,
            requestor=requestor,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        # Check for auto-approval
        if approval_type == ApprovalType.AUTO:
            if self._check_auto_approve(request, policy):
                request.status = ApprovalStatus.AUTO_APPROVED

        self._requests[request_id] = request
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["request"],
            {
                "request_id": request_id,
                "deployment_id": deployment_id,
                "service_name": service_name,
                "version": version,
                "environment": environment,
                "required_approvals": required_approvals,
            },
            actor=self.actor_id,
        )

        # Send notifications
        if self._notification_callback and request.status == ApprovalStatus.PENDING:
            self._notify_approvers(request)

        return request

    def _find_policy(
        self,
        service_name: str,
        environment: str,
    ) -> Optional[ApprovalPolicy]:
        """Find matching policy for service and environment."""
        for policy in self._policies.values():
            if policy.matches(service_name, environment):
                return policy
        return None

    def _check_auto_approve(
        self,
        request: ApprovalRequest,
        policy: Optional[ApprovalPolicy],
    ) -> bool:
        """Check if request should be auto-approved."""
        if not policy or not policy.auto_approve_conditions:
            return False

        conditions = policy.auto_approve_conditions

        # Check for staging auto-approve
        if conditions.get("staging_first"):
            # Check if staging was deployed successfully
            # This would need integration with deployment history
            pass

        # Check for same-version auto-approve
        if conditions.get("same_version"):
            # Check if same version is already in environment
            pass

        return False

    def approve(
        self,
        request_id: str,
        approver_id: str,
        comment: str = "",
    ) -> ApprovalRequest:
        """
        Approve a deployment request.

        Args:
            request_id: Request to approve
            approver_id: Approver identifier
            comment: Optional comment

        Returns:
            Updated ApprovalRequest
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request not pending: {request.status.value}")

        # Verify approver
        approver = self._approvers.get(approver_id)
        if approver and approver.level.value < request.required_level.value:
            # This would need proper level comparison logic
            pass

        # Check for existing decision
        for decision in request.decisions:
            if decision.approver_id == approver_id:
                raise ValueError("Approver already made a decision")

        # Add decision
        decision = ApprovalDecision(
            approver_id=approver_id,
            decision=ApprovalStatus.APPROVED,
            comment=comment,
        )
        request.decisions.append(decision)

        # Check if fully approved
        if request.is_approved:
            request.status = ApprovalStatus.APPROVED

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["approve"],
            {
                "request_id": request_id,
                "deployment_id": request.deployment_id,
                "approver_id": approver_id,
                "approval_count": request.approval_count,
                "fully_approved": request.is_approved,
            },
            actor=self.actor_id,
        )

        return request

    def reject(
        self,
        request_id: str,
        approver_id: str,
        comment: str = "",
    ) -> ApprovalRequest:
        """
        Reject a deployment request.

        Args:
            request_id: Request to reject
            approver_id: Approver identifier
            comment: Reason for rejection

        Returns:
            Updated ApprovalRequest
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request not pending: {request.status.value}")

        # Add decision
        decision = ApprovalDecision(
            approver_id=approver_id,
            decision=ApprovalStatus.REJECTED,
            comment=comment,
        )
        request.decisions.append(decision)
        request.status = ApprovalStatus.REJECTED

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["reject"],
            {
                "request_id": request_id,
                "deployment_id": request.deployment_id,
                "approver_id": approver_id,
                "comment": comment,
            },
            level="warn",
            actor=self.actor_id,
        )

        return request

    def cancel(self, request_id: str) -> ApprovalRequest:
        """Cancel an approval request."""
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request not pending: {request.status.value}")

        request.status = ApprovalStatus.CANCELLED
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["cancelled"],
            {
                "request_id": request_id,
                "deployment_id": request.deployment_id,
            },
            actor=self.actor_id,
        )

        return request

    async def check_expired(self) -> List[ApprovalRequest]:
        """Check for and expire timed-out requests."""
        now = time.time()
        expired = []

        for request in self._requests.values():
            if request.status != ApprovalStatus.PENDING:
                continue

            if request.expires_at and request.expires_at <= now:
                request.status = ApprovalStatus.EXPIRED
                expired.append(request)

                _emit_bus_event(
                    self.BUS_TOPICS["timeout"],
                    {
                        "request_id": request.request_id,
                        "deployment_id": request.deployment_id,
                    },
                    level="warn",
                    actor=self.actor_id,
                )

        if expired:
            self._save_state()

        return expired

    def set_notification_callback(
        self,
        callback: Callable[[ApprovalRequest, List[Approver]], None],
    ) -> None:
        """Set callback for approval notifications."""
        self._notification_callback = callback

    def _notify_approvers(self, request: ApprovalRequest) -> None:
        """Notify relevant approvers of a new request."""
        if not self._notification_callback:
            return

        # Get eligible approvers
        approvers = [
            a for a in self._approvers.values()
            if a.active and self._is_eligible_approver(a, request)
        ]

        try:
            self._notification_callback(request, approvers)
        except Exception:
            pass

    def _is_eligible_approver(
        self,
        approver: Approver,
        request: ApprovalRequest,
    ) -> bool:
        """Check if approver is eligible for request."""
        # Simple level check
        level_order = [
            ApprovalLevel.NONE,
            ApprovalLevel.TEAM,
            ApprovalLevel.MANAGER,
            ApprovalLevel.DIRECTOR,
            ApprovalLevel.EXECUTIVE,
        ]

        approver_idx = level_order.index(approver.level)
        required_idx = level_order.index(request.required_level)

        return approver_idx >= required_idx

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a request by ID."""
        return self._requests.get(request_id)

    def get_deployment_request(
        self,
        deployment_id: str,
    ) -> Optional[ApprovalRequest]:
        """Get request for a deployment."""
        for request in self._requests.values():
            if request.deployment_id == deployment_id:
                return request
        return None

    def list_requests(
        self,
        status: Optional[ApprovalStatus] = None,
        environment: Optional[str] = None,
    ) -> List[ApprovalRequest]:
        """List approval requests."""
        requests = list(self._requests.values())

        if status:
            requests = [r for r in requests if r.status == status]
        if environment:
            requests = [r for r in requests if r.environment == environment]

        return sorted(requests, key=lambda r: r.created_at, reverse=True)

    def list_pending(self, approver_id: Optional[str] = None) -> List[ApprovalRequest]:
        """List pending requests for an approver."""
        pending = [
            r for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]

        if approver_id:
            approver = self._approvers.get(approver_id)
            if approver:
                pending = [
                    r for r in pending
                    if self._is_eligible_approver(approver, r)
                ]

        return sorted(pending, key=lambda r: r.created_at)

    def list_approvers(
        self,
        level: Optional[ApprovalLevel] = None,
    ) -> List[Approver]:
        """List approvers."""
        approvers = list(self._approvers.values())

        if level:
            approvers = [a for a in approvers if a.level == level]

        return approvers

    def list_policies(self) -> List[ApprovalPolicy]:
        """List approval policies."""
        return list(self._policies.values())

    def delete_request(self, request_id: str) -> bool:
        """Delete an approval request."""
        if request_id not in self._requests:
            return False

        del self._requests[request_id]
        self._save_state()
        return True

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "requests": {rid: r.to_dict() for rid, r in self._requests.items()},
            "approvers": {aid: a.to_dict() for aid, a in self._approvers.items()},
            "policies": {pid: p.to_dict() for pid, p in self._policies.items()},
        }
        state_file = self.state_dir / "approval_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "approval_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for rid, data in state.get("requests", {}).items():
                self._requests[rid] = ApprovalRequest.from_dict(data)

            for aid, data in state.get("approvers", {}).items():
                data["level"] = ApprovalLevel(data.get("level", "team"))
                self._approvers[aid] = Approver(**{
                    k: v for k, v in data.items() if k in Approver.__dataclass_fields__
                })

            for pid, data in state.get("policies", {}).items():
                data["required_level"] = ApprovalLevel(data.get("required_level", "team"))
                data["approval_type"] = ApprovalType(data.get("approval_type", "manual"))
                self._policies[pid] = ApprovalPolicy(**{
                    k: v for k, v in data.items() if k in ApprovalPolicy.__dataclass_fields__
                })
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for approval gate."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment Approval Gate (Step 224)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # request command
    request_parser = subparsers.add_parser("request", help="Create approval request")
    request_parser.add_argument("deployment_id", help="Deployment ID")
    request_parser.add_argument("--service", "-s", required=True, help="Service name")
    request_parser.add_argument("--version", "-v", required=True, help="Version")
    request_parser.add_argument("--env", "-e", default="prod", help="Environment")
    request_parser.add_argument("--requestor", "-r", default="", help="Requestor")
    request_parser.add_argument("--json", action="store_true", help="JSON output")

    # approve command
    approve_parser = subparsers.add_parser("approve", help="Approve a request")
    approve_parser.add_argument("request_id", help="Request ID")
    approve_parser.add_argument("--approver", "-a", required=True, help="Approver ID")
    approve_parser.add_argument("--comment", "-c", default="", help="Comment")
    approve_parser.add_argument("--json", action="store_true", help="JSON output")

    # reject command
    reject_parser = subparsers.add_parser("reject", help="Reject a request")
    reject_parser.add_argument("request_id", help="Request ID")
    reject_parser.add_argument("--approver", "-a", required=True, help="Approver ID")
    reject_parser.add_argument("--comment", "-c", required=True, help="Rejection reason")
    reject_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List requests")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--env", "-e", help="Filter by environment")
    list_parser.add_argument("--pending", action="store_true", help="Show pending only")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # approver command
    approver_parser = subparsers.add_parser("approver", help="Register approver")
    approver_parser.add_argument("--name", "-n", required=True, help="Approver name")
    approver_parser.add_argument("--email", "-e", default="", help="Email")
    approver_parser.add_argument("--level", "-l", default="team",
                                  choices=["team", "manager", "director", "executive"])
    approver_parser.add_argument("--json", action="store_true", help="JSON output")

    # policy command
    policy_parser = subparsers.add_parser("policy", help="Create policy")
    policy_parser.add_argument("--name", "-n", required=True, help="Policy name")
    policy_parser.add_argument("--env", "-e", default="prod", help="Environment")
    policy_parser.add_argument("--level", "-l", default="team",
                                choices=["team", "manager", "director", "executive"])
    policy_parser.add_argument("--approvals", "-a", type=int, default=1, help="Required approvals")
    policy_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    gate = DeploymentApprovalGate()

    if args.command == "request":
        request = gate.create_request(
            deployment_id=args.deployment_id,
            service_name=args.service,
            version=args.version,
            environment=args.env,
            requestor=args.requestor,
        )

        if args.json:
            print(json.dumps(request.to_dict(), indent=2))
        else:
            print(f"Created request: {request.request_id}")
            print(f"  Status: {request.status.value}")
            print(f"  Required approvals: {request.required_approvals}")

        return 0

    elif args.command == "approve":
        try:
            request = gate.approve(
                request_id=args.request_id,
                approver_id=args.approver,
                comment=args.comment,
            )

            if args.json:
                print(json.dumps(request.to_dict(), indent=2))
            else:
                print(f"Approved: {request.request_id}")
                print(f"  Approvals: {request.approval_count}/{request.required_approvals}")
                print(f"  Status: {request.status.value}")

            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "reject":
        try:
            request = gate.reject(
                request_id=args.request_id,
                approver_id=args.approver,
                comment=args.comment,
            )

            if args.json:
                print(json.dumps(request.to_dict(), indent=2))
            else:
                print(f"Rejected: {request.request_id}")

            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "list":
        if args.pending:
            requests = gate.list_pending()
        else:
            status = ApprovalStatus(args.status) if args.status else None
            requests = gate.list_requests(status=status, environment=args.env)

        if args.json:
            print(json.dumps([r.to_dict() for r in requests], indent=2))
        else:
            for r in requests:
                print(f"{r.request_id} ({r.service_name}:{r.version}) - {r.status.value}")

        return 0

    elif args.command == "approver":
        approver = gate.register_approver(
            name=args.name,
            email=args.email,
            level=ApprovalLevel(args.level.upper()),
        )

        if args.json:
            print(json.dumps(approver.to_dict(), indent=2))
        else:
            print(f"Registered: {approver.approver_id}")
            print(f"  Name: {approver.name}")
            print(f"  Level: {approver.level.value}")

        return 0

    elif args.command == "policy":
        policy = gate.create_policy(
            name=args.name,
            environments=[args.env],
            required_level=ApprovalLevel(args.level.upper()),
            required_approvals=args.approvals,
        )

        if args.json:
            print(json.dumps(policy.to_dict(), indent=2))
        else:
            print(f"Created policy: {policy.policy_id}")
            print(f"  Name: {policy.name}")
            print(f"  Level: {policy.required_level.value}")
            print(f"  Approvals: {policy.required_approvals}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
