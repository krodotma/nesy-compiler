#!/usr/bin/env python3
"""
Omega Veto Integration (Step 161)

Handles integration with the Omega veto system for constitutional-level
review decisions. When critical security or architectural issues are found,
this module can escalate to Omega for final veto authority.

PBTSO Phase: VERIFY, SEQUESTER
Bus Topics: omega.veto.request, omega.veto.response, omega.veto.audit

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Types
# ============================================================================

class VetoReason(Enum):
    """Reasons for requesting Omega veto."""
    SECURITY_CRITICAL = "security_critical"
    ARCHITECTURE_VIOLATION = "architecture_violation"
    COMPLIANCE_FAILURE = "compliance_failure"
    DEPENDENCY_VULNERABILITY = "dependency_vulnerability"
    DATA_INTEGRITY_RISK = "data_integrity_risk"
    ACCESS_CONTROL_BREACH = "access_control_breach"
    CRYPTOGRAPHIC_WEAKNESS = "cryptographic_weakness"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"


class VetoSeverity(Enum):
    """Severity levels for veto requests."""
    CRITICAL = "critical"  # Immediate block required
    HIGH = "high"          # Should block, pending review
    ELEVATED = "elevated"  # Needs attention, may block


class VetoDecision(Enum):
    """Omega's decision on a veto request."""
    APPROVED = "approved"      # Veto approved - block merge
    DENIED = "denied"          # Veto denied - allow proceed
    DEFERRED = "deferred"      # Needs more information
    ESCALATED = "escalated"    # Escalated to higher authority
    EXPIRED = "expired"        # Request timed out


class RingLevel(Enum):
    """Security ring levels."""
    RING_0 = 0  # Constitutional - Omega authority
    RING_1 = 1  # Infrastructure - Core systems
    RING_2 = 2  # Application - Standard agents
    RING_3 = 3  # User - External interactions


@dataclass
class VetoRequest:
    """
    A request for Omega veto authority.

    Attributes:
        request_id: Unique identifier for this request
        reason: Category of veto reason
        severity: Severity level
        description: Human-readable description
        evidence: Supporting evidence (file locations, findings)
        source_agent: Agent requesting the veto
        target_resource: Resource being vetoed (PR ID, file, etc.)
        context: Additional context data
        ring_level: Security ring level of the request
        timeout_seconds: How long to wait for response
        created_at: Timestamp of request creation
    """
    request_id: str
    reason: VetoReason
    severity: VetoSeverity
    description: str
    evidence: List[Dict[str, Any]]
    source_agent: str
    target_resource: str
    context: Dict[str, Any] = field(default_factory=dict)
    ring_level: RingLevel = RingLevel.RING_0
    timeout_seconds: int = 300
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["reason"] = self.reason.value
        result["severity"] = self.severity.value
        result["ring_level"] = self.ring_level.value
        return result


@dataclass
class VetoResult:
    """
    Result of an Omega veto request.

    Attributes:
        request_id: Original request ID
        decision: Omega's decision
        rationale: Explanation for the decision
        conditions: Any conditions attached to the decision
        decided_by: Authority that made the decision
        decided_at: Timestamp of decision
        valid_until: Expiration of decision validity
        enforcement_actions: Actions to take based on decision
    """
    request_id: str
    decision: VetoDecision
    rationale: str
    conditions: List[str] = field(default_factory=list)
    decided_by: str = "omega"
    decided_at: str = ""
    valid_until: Optional[str] = None
    enforcement_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.decided_at:
            self.decided_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["decision"] = self.decision.value
        return result

    @property
    def is_blocking(self) -> bool:
        """Check if this result blocks the operation."""
        return self.decision in (VetoDecision.APPROVED, VetoDecision.DEFERRED)


# ============================================================================
# Omega Veto Integration
# ============================================================================

class OmegaVetoIntegration:
    """
    Integration with the Omega veto system.

    Handles requesting, tracking, and enforcing Omega veto decisions.
    The Omega is the constitutional authority that can override any
    agent decision for security or compliance reasons.

    Example:
        integration = OmegaVetoIntegration(agent_id="review-agent")

        # Request veto for critical security issue
        request = integration.create_request(
            reason=VetoReason.SECURITY_CRITICAL,
            severity=VetoSeverity.CRITICAL,
            description="SQL injection vulnerability in user input handler",
            evidence=[{"file": "auth/login.py", "line": 42}],
            target_resource="PR-123",
        )

        # Submit and wait for response
        result = await integration.submit_and_wait(request)

        if result.is_blocking:
            print(f"Merge blocked: {result.rationale}")
    """

    BUS_TOPICS = {
        "request": "omega.veto.request",
        "response": "omega.veto.response",
        "audit": "omega.veto.audit",
        "enforcement": "omega.veto.enforcement",
    }

    def __init__(
        self,
        agent_id: str = "review-agent",
        bus_path: Optional[Path] = None,
        omega_endpoint: Optional[str] = None,
    ):
        """
        Initialize Omega veto integration.

        Args:
            agent_id: ID of the requesting agent
            bus_path: Path to event bus file
            omega_endpoint: Endpoint for Omega service (optional)
        """
        self.agent_id = agent_id
        self.bus_path = bus_path or self._get_bus_path()
        self.omega_endpoint = omega_endpoint or os.environ.get("OMEGA_ENDPOINT", "")
        self._pending_requests: Dict[str, VetoRequest] = {}
        self._results: Dict[str, VetoResult] = {}
        self._callbacks: Dict[str, Callable[[VetoResult], None]] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "veto") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": self.agent_id,
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def create_request(
        self,
        reason: VetoReason,
        severity: VetoSeverity,
        description: str,
        evidence: List[Dict[str, Any]],
        target_resource: str,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300,
    ) -> VetoRequest:
        """
        Create a new veto request.

        Args:
            reason: Category of veto reason
            severity: Severity level
            description: Human-readable description
            evidence: Supporting evidence
            target_resource: Resource being vetoed
            context: Additional context
            timeout_seconds: Request timeout

        Returns:
            VetoRequest object
        """
        return VetoRequest(
            request_id=str(uuid.uuid4())[:8],
            reason=reason,
            severity=severity,
            description=description,
            evidence=evidence,
            source_agent=self.agent_id,
            target_resource=target_resource,
            context=context or {},
            ring_level=RingLevel.RING_0,
            timeout_seconds=timeout_seconds,
        )

    def submit(
        self,
        request: VetoRequest,
        on_result: Optional[Callable[[VetoResult], None]] = None,
    ) -> str:
        """
        Submit a veto request to Omega.

        Args:
            request: The veto request
            on_result: Callback for when result is received

        Returns:
            Request ID

        Emits:
            omega.veto.request
        """
        self._pending_requests[request.request_id] = request

        if on_result:
            self._callbacks[request.request_id] = on_result

        # Emit request event
        self._emit_event(self.BUS_TOPICS["request"], {
            "request": request.to_dict(),
            "status": "submitted",
        })

        # Emit audit event
        self._emit_event(self.BUS_TOPICS["audit"], {
            "action": "request_submitted",
            "request_id": request.request_id,
            "reason": request.reason.value,
            "severity": request.severity.value,
            "target": request.target_resource,
        }, kind="audit")

        return request.request_id

    async def submit_and_wait(
        self,
        request: VetoRequest,
        poll_interval: float = 1.0,
    ) -> VetoResult:
        """
        Submit a veto request and wait for response.

        Args:
            request: The veto request
            poll_interval: How often to check for response

        Returns:
            VetoResult from Omega

        Raises:
            TimeoutError: If request times out
        """
        self.submit(request)

        start_time = time.time()
        while time.time() - start_time < request.timeout_seconds:
            # Check for response
            if request.request_id in self._results:
                return self._results[request.request_id]

            # In production, this would poll Omega service
            # For now, simulate auto-approval for critical issues
            if self._should_auto_approve(request):
                result = self._create_auto_result(request, VetoDecision.APPROVED)
                self._handle_result(result)
                return result

            await asyncio.sleep(poll_interval)

        # Timeout - create expired result
        result = VetoResult(
            request_id=request.request_id,
            decision=VetoDecision.EXPIRED,
            rationale=f"Request timed out after {request.timeout_seconds}s",
            decided_by="system",
        )
        self._handle_result(result)
        return result

    def _should_auto_approve(self, request: VetoRequest) -> bool:
        """Determine if request should be auto-approved."""
        # Auto-approve critical security issues
        if request.severity == VetoSeverity.CRITICAL:
            if request.reason in (
                VetoReason.SECURITY_CRITICAL,
                VetoReason.ACCESS_CONTROL_BREACH,
                VetoReason.CRYPTOGRAPHIC_WEAKNESS,
            ):
                return True
        return False

    def _create_auto_result(
        self,
        request: VetoRequest,
        decision: VetoDecision,
    ) -> VetoResult:
        """Create an automated result."""
        rationale_map = {
            VetoDecision.APPROVED: f"Auto-approved veto for {request.reason.value}",
            VetoDecision.DENIED: "No blocking issues found",
        }

        enforcement_map = {
            VetoDecision.APPROVED: ["block_merge", "notify_author", "log_security_event"],
            VetoDecision.DENIED: [],
        }

        return VetoResult(
            request_id=request.request_id,
            decision=decision,
            rationale=rationale_map.get(decision, "Automated decision"),
            conditions=[],
            decided_by="omega-auto",
            enforcement_actions=enforcement_map.get(decision, []),
        )

    def _handle_result(self, result: VetoResult) -> None:
        """Handle a veto result."""
        self._results[result.request_id] = result

        # Remove from pending
        if result.request_id in self._pending_requests:
            del self._pending_requests[result.request_id]

        # Emit response event
        self._emit_event(self.BUS_TOPICS["response"], {
            "result": result.to_dict(),
            "status": "resolved",
        })

        # Emit audit event
        self._emit_event(self.BUS_TOPICS["audit"], {
            "action": "decision_received",
            "request_id": result.request_id,
            "decision": result.decision.value,
            "decided_by": result.decided_by,
        }, kind="audit")

        # Emit enforcement if blocking
        if result.is_blocking:
            self._emit_event(self.BUS_TOPICS["enforcement"], {
                "request_id": result.request_id,
                "actions": result.enforcement_actions,
                "blocking": True,
            }, kind="enforcement")

        # Call callback if registered
        if result.request_id in self._callbacks:
            callback = self._callbacks.pop(result.request_id)
            callback(result)

    def receive_result(self, result_data: Dict[str, Any]) -> None:
        """
        Receive a veto result from Omega.

        Args:
            result_data: Result data dictionary
        """
        result = VetoResult(
            request_id=result_data["request_id"],
            decision=VetoDecision(result_data["decision"]),
            rationale=result_data.get("rationale", ""),
            conditions=result_data.get("conditions", []),
            decided_by=result_data.get("decided_by", "omega"),
            decided_at=result_data.get("decided_at", ""),
            valid_until=result_data.get("valid_until"),
            enforcement_actions=result_data.get("enforcement_actions", []),
        )
        self._handle_result(result)

    def get_pending_requests(self) -> List[VetoRequest]:
        """Get all pending veto requests."""
        return list(self._pending_requests.values())

    def get_result(self, request_id: str) -> Optional[VetoResult]:
        """Get result for a specific request."""
        return self._results.get(request_id)

    def is_resource_blocked(self, resource: str) -> bool:
        """Check if a resource is currently blocked by veto."""
        for result in self._results.values():
            if result.is_blocking:
                # Check if there's a pending request for this resource
                for req in self._pending_requests.values():
                    if req.target_resource == resource:
                        return True
        return False


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Omega Veto Integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Omega Veto Integration (Step 161)")
    parser.add_argument("--request", action="store_true", help="Create a veto request")
    parser.add_argument("--reason", choices=[r.value for r in VetoReason],
                        default="security_critical", help="Veto reason")
    parser.add_argument("--severity", choices=[s.value for s in VetoSeverity],
                        default="critical", help="Severity level")
    parser.add_argument("--target", required=True, help="Target resource (PR ID, etc.)")
    parser.add_argument("--description", required=True, help="Description of the issue")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    integration = OmegaVetoIntegration()

    if args.request:
        request = integration.create_request(
            reason=VetoReason(args.reason),
            severity=VetoSeverity(args.severity),
            description=args.description,
            evidence=[],
            target_resource=args.target,
        )

        request_id = integration.submit(request)

        if args.json:
            print(json.dumps({
                "request_id": request_id,
                "status": "submitted",
                "request": request.to_dict(),
            }, indent=2))
        else:
            print(f"Veto Request Submitted")
            print(f"  Request ID: {request_id}")
            print(f"  Reason: {request.reason.value}")
            print(f"  Severity: {request.severity.value}")
            print(f"  Target: {request.target_resource}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
