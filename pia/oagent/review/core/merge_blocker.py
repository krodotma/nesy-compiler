#!/usr/bin/env python3
"""
Review Merge Blocker (Step 169)

Enforces merge blocking policies based on review findings,
approval status, and quality gates.

PBTSO Phase: VERIFY, SEQUESTER
Bus Topics: review.merge.block, review.merge.allow, review.merge.override

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# Types
# ============================================================================

class BlockSeverity(Enum):
    """Severity levels for merge blocks."""
    HARD = "hard"        # Cannot be overridden
    SOFT = "soft"        # Can be overridden with justification
    WARNING = "warning"  # Advisory only


class BlockCategory(Enum):
    """Categories of merge blocks."""
    SECURITY = "security"
    QUALITY = "quality"
    APPROVAL = "approval"
    TEST = "test"
    COMPLIANCE = "compliance"
    POLICY = "policy"
    OMEGA_VETO = "omega_veto"


class MergeStatus(Enum):
    """Status of merge eligibility."""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    PENDING = "pending"
    OVERRIDDEN = "overridden"


@dataclass
class BlockReason:
    """
    A reason for blocking merge.

    Attributes:
        reason_id: Unique ID
        category: Block category
        severity: Block severity
        title: Short title
        description: Detailed description
        source: Source of the block (analyzer, policy, etc.)
        details: Additional details
        can_override: Whether this can be overridden
        override_requires: What's needed to override
    """
    reason_id: str
    category: BlockCategory
    severity: BlockSeverity
    title: str
    description: str
    source: str = "review-agent"
    details: Dict[str, Any] = field(default_factory=dict)
    can_override: bool = False
    override_requires: Optional[str] = None

    def __post_init__(self):
        if not self.reason_id:
            self.reason_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reason_id": self.reason_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "details": self.details,
            "can_override": self.can_override,
            "override_requires": self.override_requires,
        }


@dataclass
class Override:
    """
    An override of a merge block.

    Attributes:
        override_id: Unique ID
        reason_id: ID of overridden reason
        overridden_by: Who overrode the block
        justification: Reason for override
        approved_by: Who approved the override
        expires_at: When override expires
        created_at: Override timestamp
    """
    override_id: str
    reason_id: str
    overridden_by: str
    justification: str
    approved_by: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.override_id:
            self.override_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MergeDecision:
    """
    Final merge decision.

    Attributes:
        decision_id: Unique ID
        pr_id: Pull request ID
        status: Merge status
        reasons: List of block reasons
        overrides: Applied overrides
        approvals: Review approvals received
        required_approvals: Required approvals
        quality_gate_passed: Whether quality gate passed
        can_merge: Final merge eligibility
        message: Human-readable message
        decided_at: Decision timestamp
    """
    decision_id: str
    pr_id: str
    status: MergeStatus
    reasons: List[BlockReason] = field(default_factory=list)
    overrides: List[Override] = field(default_factory=list)
    approvals: int = 0
    required_approvals: int = 1
    quality_gate_passed: bool = True
    can_merge: bool = False
    message: str = ""
    decided_at: str = ""

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = str(uuid.uuid4())[:8]
        if not self.decided_at:
            self.decided_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "pr_id": self.pr_id,
            "status": self.status.value,
            "reasons": [r.to_dict() for r in self.reasons],
            "overrides": [o.to_dict() for o in self.overrides],
            "approvals": self.approvals,
            "required_approvals": self.required_approvals,
            "quality_gate_passed": self.quality_gate_passed,
            "can_merge": self.can_merge,
            "message": self.message,
            "decided_at": self.decided_at,
        }

    @property
    def hard_blocks(self) -> List[BlockReason]:
        """Get hard blocks that cannot be overridden."""
        return [r for r in self.reasons if r.severity == BlockSeverity.HARD]

    @property
    def soft_blocks(self) -> List[BlockReason]:
        """Get soft blocks that can be overridden."""
        return [r for r in self.reasons if r.severity == BlockSeverity.SOFT]


@dataclass
class MergePolicy:
    """
    Policy configuration for merge blocking.

    Attributes:
        require_approvals: Minimum approvals required
        require_security_review: Require security reviewer approval
        block_on_critical_security: Block on critical security issues
        block_on_failing_tests: Block if tests fail
        min_quality_score: Minimum quality score to pass
        allow_override: Allow soft block overrides
        override_approvers: Who can approve overrides
    """
    require_approvals: int = 1
    require_security_review: bool = True
    block_on_critical_security: bool = True
    block_on_failing_tests: bool = True
    min_quality_score: Optional[float] = 70.0
    allow_override: bool = True
    override_approvers: List[str] = field(default_factory=lambda: ["admin"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Merge Blocker
# ============================================================================

class MergeBlocker:
    """
    Enforces merge blocking policies.

    Evaluates review findings and policies to determine merge eligibility.

    Example:
        blocker = MergeBlocker()

        # Evaluate merge eligibility
        decision = blocker.evaluate(
            pr_id="123",
            security_issues=2,
            critical_issues=0,
            approvals=1,
            required_approvals=1,
        )

        if decision.can_merge:
            print("PR can be merged")
        else:
            for reason in decision.reasons:
                print(f"Blocked: {reason.title}")
    """

    BUS_TOPICS = {
        "block": "review.merge.block",
        "allow": "review.merge.allow",
        "override": "review.merge.override",
        "decision": "review.merge.decision",
    }

    def __init__(
        self,
        policy: Optional[MergePolicy] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the merge blocker.

        Args:
            policy: Merge policy configuration
            bus_path: Path to event bus file
        """
        self.policy = policy or MergePolicy()
        self.bus_path = bus_path or self._get_bus_path()
        self._overrides: Dict[str, List[Override]] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "merge") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "merge-blocker",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def evaluate(
        self,
        pr_id: str,
        security_issues: int = 0,
        critical_issues: int = 0,
        high_issues: int = 0,
        approvals: int = 0,
        required_approvals: Optional[int] = None,
        tests_passed: bool = True,
        quality_score: Optional[float] = None,
        has_security_review: bool = False,
        omega_veto: bool = False,
        omega_veto_reason: Optional[str] = None,
    ) -> MergeDecision:
        """
        Evaluate merge eligibility for a PR.

        Args:
            pr_id: Pull request ID
            security_issues: Number of security issues
            critical_issues: Number of critical issues
            high_issues: Number of high issues
            approvals: Number of approvals received
            required_approvals: Required approvals (uses policy default)
            tests_passed: Whether tests passed
            quality_score: Quality score if computed
            has_security_review: Whether security review completed
            omega_veto: Whether Omega has vetoed
            omega_veto_reason: Reason for Omega veto

        Returns:
            MergeDecision with status and reasons

        Emits:
            review.merge.block or review.merge.allow
            review.merge.decision
        """
        required = required_approvals if required_approvals is not None else self.policy.require_approvals
        reasons: List[BlockReason] = []
        overrides = self._overrides.get(pr_id, [])
        overridden_reasons = {o.reason_id for o in overrides}

        # Check Omega veto (cannot be overridden)
        if omega_veto:
            reasons.append(BlockReason(
                reason_id="omega_veto",
                category=BlockCategory.OMEGA_VETO,
                severity=BlockSeverity.HARD,
                title="Omega Veto",
                description=omega_veto_reason or "Omega has vetoed this merge",
                source="omega",
                can_override=False,
            ))

        # Check security issues
        if security_issues > 0 and self.policy.block_on_critical_security:
            reason = BlockReason(
                reason_id="security_issues",
                category=BlockCategory.SECURITY,
                severity=BlockSeverity.HARD if security_issues >= 1 else BlockSeverity.SOFT,
                title="Security Issues Found",
                description=f"Found {security_issues} security issues that must be resolved",
                details={"count": security_issues},
                can_override=False,
            )
            reasons.append(reason)

        # Check critical issues
        if critical_issues > 0:
            reason = BlockReason(
                reason_id="critical_issues",
                category=BlockCategory.QUALITY,
                severity=BlockSeverity.HARD,
                title="Critical Issues Found",
                description=f"Found {critical_issues} critical issues that must be resolved",
                details={"count": critical_issues},
                can_override=False,
            )
            reasons.append(reason)

        # Check high issues (soft block)
        if high_issues > 5:
            reason = BlockReason(
                reason_id="high_issues",
                category=BlockCategory.QUALITY,
                severity=BlockSeverity.SOFT,
                title="Many High-Severity Issues",
                description=f"Found {high_issues} high-severity issues. Consider addressing them.",
                details={"count": high_issues},
                can_override=True,
                override_requires="maintainer approval",
            )
            if reason.reason_id not in overridden_reasons:
                reasons.append(reason)

        # Check approvals
        if approvals < required:
            reason = BlockReason(
                reason_id="insufficient_approvals",
                category=BlockCategory.APPROVAL,
                severity=BlockSeverity.SOFT,
                title="Insufficient Approvals",
                description=f"Requires {required} approvals, has {approvals}",
                details={"required": required, "current": approvals},
                can_override=True,
                override_requires="admin approval",
            )
            if reason.reason_id not in overridden_reasons:
                reasons.append(reason)

        # Check tests
        if not tests_passed and self.policy.block_on_failing_tests:
            reason = BlockReason(
                reason_id="tests_failed",
                category=BlockCategory.TEST,
                severity=BlockSeverity.SOFT,
                title="Tests Failed",
                description="CI tests have failed",
                can_override=True,
                override_requires="maintainer approval with justification",
            )
            if reason.reason_id not in overridden_reasons:
                reasons.append(reason)

        # Check quality score
        if quality_score is not None and self.policy.min_quality_score:
            if quality_score < self.policy.min_quality_score:
                reason = BlockReason(
                    reason_id="quality_gate",
                    category=BlockCategory.QUALITY,
                    severity=BlockSeverity.SOFT,
                    title="Quality Gate Failed",
                    description=f"Quality score {quality_score:.1f} below threshold {self.policy.min_quality_score}",
                    details={"score": quality_score, "threshold": self.policy.min_quality_score},
                    can_override=True,
                    override_requires="tech lead approval",
                )
                if reason.reason_id not in overridden_reasons:
                    reasons.append(reason)

        # Check security review requirement
        if self.policy.require_security_review and security_issues > 0 and not has_security_review:
            reason = BlockReason(
                reason_id="security_review_required",
                category=BlockCategory.SECURITY,
                severity=BlockSeverity.SOFT,
                title="Security Review Required",
                description="Changes require security team review",
                can_override=True,
                override_requires="security team approval",
            )
            if reason.reason_id not in overridden_reasons:
                reasons.append(reason)

        # Determine status
        hard_blocks = [r for r in reasons if r.severity == BlockSeverity.HARD]
        soft_blocks = [r for r in reasons if r.severity == BlockSeverity.SOFT]

        if hard_blocks:
            status = MergeStatus.BLOCKED
            can_merge = False
            message = f"Merge blocked by {len(hard_blocks)} critical issue(s)"
        elif soft_blocks:
            if all(r.reason_id in overridden_reasons for r in soft_blocks):
                status = MergeStatus.OVERRIDDEN
                can_merge = True
                message = "Merge allowed with overrides"
            else:
                status = MergeStatus.BLOCKED
                can_merge = False
                message = f"Merge blocked by {len(soft_blocks)} issue(s) (can be overridden)"
        else:
            status = MergeStatus.ALLOWED
            can_merge = True
            message = "Merge allowed"

        decision = MergeDecision(
            decision_id=str(uuid.uuid4())[:8],
            pr_id=pr_id,
            status=status,
            reasons=reasons,
            overrides=overrides,
            approvals=approvals,
            required_approvals=required,
            quality_gate_passed=quality_score is None or quality_score >= (self.policy.min_quality_score or 0),
            can_merge=can_merge,
            message=message,
        )

        # Emit events
        if can_merge:
            self._emit_event(self.BUS_TOPICS["allow"], {
                "pr_id": pr_id,
                "decision_id": decision.decision_id,
                "status": status.value,
            })
        else:
            self._emit_event(self.BUS_TOPICS["block"], {
                "pr_id": pr_id,
                "decision_id": decision.decision_id,
                "reasons": [r.to_dict() for r in reasons],
            })

        self._emit_event(self.BUS_TOPICS["decision"], {
            "decision": decision.to_dict(),
        })

        return decision

    def request_override(
        self,
        pr_id: str,
        reason_id: str,
        requested_by: str,
        justification: str,
    ) -> Optional[Override]:
        """
        Request an override for a soft block.

        Args:
            pr_id: Pull request ID
            reason_id: ID of the reason to override
            requested_by: Who is requesting
            justification: Why override is needed

        Returns:
            Override if successful, None if not allowed

        Emits:
            review.merge.override
        """
        if not self.policy.allow_override:
            return None

        override = Override(
            override_id=str(uuid.uuid4())[:8],
            reason_id=reason_id,
            overridden_by=requested_by,
            justification=justification,
        )

        if pr_id not in self._overrides:
            self._overrides[pr_id] = []
        self._overrides[pr_id].append(override)

        self._emit_event(self.BUS_TOPICS["override"], {
            "pr_id": pr_id,
            "override": override.to_dict(),
        })

        return override

    def approve_override(
        self,
        pr_id: str,
        override_id: str,
        approved_by: str,
    ) -> bool:
        """
        Approve an override request.

        Args:
            pr_id: Pull request ID
            override_id: Override ID
            approved_by: Who is approving

        Returns:
            True if approved
        """
        if approved_by not in self.policy.override_approvers:
            return False

        overrides = self._overrides.get(pr_id, [])
        for override in overrides:
            if override.override_id == override_id:
                override.approved_by = approved_by
                return True

        return False


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Merge Blocker."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Merge Blocker (Step 169)")
    parser.add_argument("--pr", required=True, help="PR ID")
    parser.add_argument("--security", type=int, default=0, help="Security issues")
    parser.add_argument("--critical", type=int, default=0, help="Critical issues")
    parser.add_argument("--approvals", type=int, default=0, help="Approvals received")
    parser.add_argument("--required", type=int, default=1, help="Required approvals")
    parser.add_argument("--quality", type=float, help="Quality score")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    blocker = MergeBlocker()

    decision = blocker.evaluate(
        pr_id=args.pr,
        security_issues=args.security,
        critical_issues=args.critical,
        approvals=args.approvals,
        required_approvals=args.required,
        quality_score=args.quality,
    )

    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
    else:
        status_icon = "[OK]" if decision.can_merge else "[X]"
        print(f"{status_icon} Merge Decision for PR {decision.pr_id}")
        print(f"  Status: {decision.status.value}")
        print(f"  Can Merge: {decision.can_merge}")
        print(f"  Message: {decision.message}")
        print(f"  Approvals: {decision.approvals}/{decision.required_approvals}")

        if decision.reasons:
            print("\nBlock Reasons:")
            for reason in decision.reasons:
                override_note = " (overridable)" if reason.can_override else ""
                print(f"  [{reason.severity.value}] {reason.title}{override_note}")
                print(f"      {reason.description}")

    return 0 if decision.can_merge else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
