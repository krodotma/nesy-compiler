#!/usr/bin/env python3
"""
Review Approval Manager (Step 170)

Manages the approval workflow for code reviews, tracking approvals,
rejections, and coordination of multi-reviewer scenarios.

PBTSO Phase: VERIFY, DISTRIBUTE
Bus Topics: review.approval.manage, review.approval.submit, review.approval.status

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
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Types
# ============================================================================

class ApprovalAction(Enum):
    """Actions in the approval workflow."""
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    COMMENT = "comment"
    DISMISS = "dismiss"
    ABSTAIN = "abstain"


class ApprovalState(Enum):
    """State of approval workflow."""
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class ReviewerStatus(Enum):
    """Status of individual reviewer."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMMENTED = "commented"
    DISMISSED = "dismissed"


@dataclass
class ReviewerApproval:
    """
    Approval status for a single reviewer.

    Attributes:
        reviewer_id: Reviewer ID
        status: Current status
        action: Action taken
        comment: Review comment
        submitted_at: Submission timestamp
        valid: Whether approval is still valid
    """
    reviewer_id: str
    status: ReviewerStatus
    action: Optional[ApprovalAction] = None
    comment: str = ""
    submitted_at: str = ""
    valid: bool = True

    def __post_init__(self):
        if not self.submitted_at and self.action:
            self.submitted_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reviewer_id": self.reviewer_id,
            "status": self.status.value,
            "action": self.action.value if self.action else None,
            "comment": self.comment,
            "submitted_at": self.submitted_at,
            "valid": self.valid,
        }


@dataclass
class ApprovalRequirement:
    """
    Requirements for approval.

    Attributes:
        min_approvals: Minimum approvals needed
        required_reviewers: Specific reviewers required
        require_codeowner: Require codeowner approval
        dismiss_stale_reviews: Dismiss reviews after push
        require_fresh_review: Require review after last push
    """
    min_approvals: int = 1
    required_reviewers: List[str] = field(default_factory=list)
    require_codeowner: bool = False
    dismiss_stale_reviews: bool = True
    require_fresh_review: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ApprovalWorkflow:
    """
    Complete approval workflow for a PR.

    Attributes:
        workflow_id: Unique workflow ID
        pr_id: Pull request ID
        repository: Repository name
        state: Current workflow state
        requirements: Approval requirements
        reviewers: Status of each reviewer
        head_sha: Current head commit SHA
        approvals_count: Number of valid approvals
        rejections_count: Number of rejections
        created_at: Workflow creation timestamp
        updated_at: Last update timestamp
    """
    workflow_id: str
    pr_id: str
    repository: str
    state: ApprovalState
    requirements: ApprovalRequirement
    reviewers: Dict[str, ReviewerApproval] = field(default_factory=dict)
    head_sha: str = ""
    approvals_count: int = 0
    rejections_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.workflow_id:
            self.workflow_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "pr_id": self.pr_id,
            "repository": self.repository,
            "state": self.state.value,
            "requirements": self.requirements.to_dict(),
            "reviewers": {k: v.to_dict() for k, v in self.reviewers.items()},
            "head_sha": self.head_sha,
            "approvals_count": self.approvals_count,
            "rejections_count": self.rejections_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @property
    def is_approved(self) -> bool:
        """Check if workflow is approved."""
        return self.state == ApprovalState.APPROVED

    @property
    def has_required_approvals(self) -> bool:
        """Check if minimum approvals are met."""
        return self.approvals_count >= self.requirements.min_approvals

    @property
    def missing_required_reviewers(self) -> List[str]:
        """Get list of required reviewers who haven't approved."""
        missing = []
        for reviewer_id in self.requirements.required_reviewers:
            reviewer = self.reviewers.get(reviewer_id)
            if not reviewer or reviewer.status != ReviewerStatus.APPROVED:
                missing.append(reviewer_id)
        return missing


# ============================================================================
# Approval Manager
# ============================================================================

class ApprovalManager:
    """
    Manages approval workflows for code reviews.

    Tracks approvals, enforces requirements, and coordinates reviewers.

    Example:
        manager = ApprovalManager()

        # Create workflow
        workflow = manager.create_workflow(
            pr_id="123",
            repository="owner/repo",
            requirements=ApprovalRequirement(min_approvals=2),
        )

        # Submit approval
        manager.submit_approval(
            workflow_id=workflow.workflow_id,
            reviewer_id="alice",
            action=ApprovalAction.APPROVE,
        )

        # Check status
        if workflow.is_approved:
            print("PR approved!")
    """

    BUS_TOPICS = {
        "manage": "review.approval.manage",
        "submit": "review.approval.submit",
        "status": "review.approval.status",
        "invalidate": "review.approval.invalidate",
    }

    def __init__(
        self,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the approval manager.

        Args:
            bus_path: Path to event bus file
        """
        self.bus_path = bus_path or self._get_bus_path()
        self._workflows: Dict[str, ApprovalWorkflow] = {}
        self._pr_to_workflow: Dict[str, str] = {}
        self._callbacks: Dict[str, List[Callable[[ApprovalWorkflow], None]]] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "approval") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "approval-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def create_workflow(
        self,
        pr_id: str,
        repository: str,
        requirements: Optional[ApprovalRequirement] = None,
        head_sha: str = "",
        requested_reviewers: Optional[List[str]] = None,
    ) -> ApprovalWorkflow:
        """
        Create a new approval workflow.

        Args:
            pr_id: Pull request ID
            repository: Repository name
            requirements: Approval requirements
            head_sha: Current head commit SHA
            requested_reviewers: Initial reviewers to request

        Returns:
            Created ApprovalWorkflow

        Emits:
            review.approval.manage
        """
        requirements = requirements or ApprovalRequirement()

        workflow = ApprovalWorkflow(
            workflow_id=str(uuid.uuid4())[:8],
            pr_id=pr_id,
            repository=repository,
            state=ApprovalState.PENDING,
            requirements=requirements,
            head_sha=head_sha,
        )

        # Add requested reviewers
        if requested_reviewers:
            for reviewer_id in requested_reviewers:
                workflow.reviewers[reviewer_id] = ReviewerApproval(
                    reviewer_id=reviewer_id,
                    status=ReviewerStatus.PENDING,
                )

        self._workflows[workflow.workflow_id] = workflow
        self._pr_to_workflow[pr_id] = workflow.workflow_id

        self._emit_event(self.BUS_TOPICS["manage"], {
            "workflow_id": workflow.workflow_id,
            "pr_id": pr_id,
            "action": "created",
            "requirements": requirements.to_dict(),
        })

        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[ApprovalWorkflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def get_workflow_by_pr(self, pr_id: str) -> Optional[ApprovalWorkflow]:
        """Get workflow for a PR."""
        workflow_id = self._pr_to_workflow.get(pr_id)
        return self._workflows.get(workflow_id) if workflow_id else None

    def submit_approval(
        self,
        workflow_id: str,
        reviewer_id: str,
        action: ApprovalAction,
        comment: str = "",
    ) -> Optional[ApprovalWorkflow]:
        """
        Submit an approval action.

        Args:
            workflow_id: Workflow ID
            reviewer_id: Reviewer submitting
            action: Approval action
            comment: Review comment

        Returns:
            Updated workflow or None

        Emits:
            review.approval.submit
            review.approval.status
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        # Map action to status
        status_map = {
            ApprovalAction.APPROVE: ReviewerStatus.APPROVED,
            ApprovalAction.REQUEST_CHANGES: ReviewerStatus.REJECTED,
            ApprovalAction.COMMENT: ReviewerStatus.COMMENTED,
            ApprovalAction.DISMISS: ReviewerStatus.DISMISSED,
            ApprovalAction.ABSTAIN: ReviewerStatus.PENDING,
        }

        reviewer_approval = ReviewerApproval(
            reviewer_id=reviewer_id,
            status=status_map.get(action, ReviewerStatus.PENDING),
            action=action,
            comment=comment,
            valid=True,
        )

        workflow.reviewers[reviewer_id] = reviewer_approval
        workflow.updated_at = datetime.now(timezone.utc).isoformat() + "Z"

        # Recalculate counts
        self._recalculate_workflow(workflow)

        self._emit_event(self.BUS_TOPICS["submit"], {
            "workflow_id": workflow_id,
            "pr_id": workflow.pr_id,
            "reviewer_id": reviewer_id,
            "action": action.value,
        })

        self._emit_event(self.BUS_TOPICS["status"], {
            "workflow_id": workflow_id,
            "state": workflow.state.value,
            "approvals": workflow.approvals_count,
            "rejections": workflow.rejections_count,
        })

        # Trigger callbacks
        self._notify_callbacks(workflow)

        return workflow

    def _recalculate_workflow(self, workflow: ApprovalWorkflow) -> None:
        """Recalculate workflow state based on reviewer statuses."""
        approvals = 0
        rejections = 0

        for reviewer in workflow.reviewers.values():
            if not reviewer.valid:
                continue
            if reviewer.status == ReviewerStatus.APPROVED:
                approvals += 1
            elif reviewer.status == ReviewerStatus.REJECTED:
                rejections += 1

        workflow.approvals_count = approvals
        workflow.rejections_count = rejections

        # Determine state
        if rejections > 0:
            workflow.state = ApprovalState.CHANGES_REQUESTED
        elif approvals >= workflow.requirements.min_approvals:
            # Check required reviewers
            missing = workflow.missing_required_reviewers
            if not missing:
                workflow.state = ApprovalState.APPROVED
            else:
                workflow.state = ApprovalState.PENDING
        else:
            workflow.state = ApprovalState.PENDING

    def invalidate_reviews(
        self,
        workflow_id: str,
        new_head_sha: str,
        keep_approvals: bool = False,
    ) -> Optional[ApprovalWorkflow]:
        """
        Invalidate reviews after new commits.

        Args:
            workflow_id: Workflow ID
            new_head_sha: New head commit SHA
            keep_approvals: Whether to keep existing approvals

        Returns:
            Updated workflow or None

        Emits:
            review.approval.invalidate
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        if not workflow.requirements.dismiss_stale_reviews:
            # Just update SHA, don't invalidate
            workflow.head_sha = new_head_sha
            return workflow

        # Invalidate reviews
        invalidated_reviewers = []
        for reviewer_id, reviewer in workflow.reviewers.items():
            if reviewer.valid and reviewer.status != ReviewerStatus.PENDING:
                if not keep_approvals or reviewer.status != ReviewerStatus.APPROVED:
                    reviewer.valid = False
                    invalidated_reviewers.append(reviewer_id)

        workflow.head_sha = new_head_sha
        workflow.updated_at = datetime.now(timezone.utc).isoformat() + "Z"

        self._recalculate_workflow(workflow)

        self._emit_event(self.BUS_TOPICS["invalidate"], {
            "workflow_id": workflow_id,
            "pr_id": workflow.pr_id,
            "new_head_sha": new_head_sha,
            "invalidated_reviewers": invalidated_reviewers,
        })

        return workflow

    def dismiss_review(
        self,
        workflow_id: str,
        reviewer_id: str,
        reason: str,
        dismissed_by: str,
    ) -> Optional[ApprovalWorkflow]:
        """
        Dismiss a specific review.

        Args:
            workflow_id: Workflow ID
            reviewer_id: Reviewer whose review is dismissed
            reason: Reason for dismissal
            dismissed_by: Who is dismissing

        Returns:
            Updated workflow or None
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        if reviewer_id in workflow.reviewers:
            workflow.reviewers[reviewer_id].status = ReviewerStatus.DISMISSED
            workflow.reviewers[reviewer_id].valid = False

        workflow.updated_at = datetime.now(timezone.utc).isoformat() + "Z"
        self._recalculate_workflow(workflow)

        self._emit_event(self.BUS_TOPICS["manage"], {
            "workflow_id": workflow_id,
            "action": "review_dismissed",
            "reviewer_id": reviewer_id,
            "dismissed_by": dismissed_by,
            "reason": reason,
        })

        return workflow

    def request_reviewer(
        self,
        workflow_id: str,
        reviewer_id: str,
    ) -> Optional[ApprovalWorkflow]:
        """
        Request a review from a specific reviewer.

        Args:
            workflow_id: Workflow ID
            reviewer_id: Reviewer to request

        Returns:
            Updated workflow or None
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        if reviewer_id not in workflow.reviewers:
            workflow.reviewers[reviewer_id] = ReviewerApproval(
                reviewer_id=reviewer_id,
                status=ReviewerStatus.PENDING,
            )

        self._emit_event(self.BUS_TOPICS["manage"], {
            "workflow_id": workflow_id,
            "action": "reviewer_requested",
            "reviewer_id": reviewer_id,
        })

        return workflow

    def on_state_change(
        self,
        workflow_id: str,
        callback: Callable[[ApprovalWorkflow], None],
    ) -> None:
        """Register callback for state changes."""
        if workflow_id not in self._callbacks:
            self._callbacks[workflow_id] = []
        self._callbacks[workflow_id].append(callback)

    def _notify_callbacks(self, workflow: ApprovalWorkflow) -> None:
        """Notify registered callbacks."""
        callbacks = self._callbacks.get(workflow.workflow_id, [])
        for callback in callbacks:
            try:
                callback(workflow)
            except Exception:
                pass

    def get_pending_workflows(
        self,
        repository: Optional[str] = None,
    ) -> List[ApprovalWorkflow]:
        """Get all pending workflows."""
        pending = []
        for workflow in self._workflows.values():
            if workflow.state == ApprovalState.PENDING:
                if repository is None or workflow.repository == repository:
                    pending.append(workflow)
        return pending

    def get_status_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get a summary of workflow status."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {}

        return {
            "workflow_id": workflow.workflow_id,
            "pr_id": workflow.pr_id,
            "state": workflow.state.value,
            "is_approved": workflow.is_approved,
            "approvals": workflow.approvals_count,
            "required": workflow.requirements.min_approvals,
            "rejections": workflow.rejections_count,
            "missing_required": workflow.missing_required_reviewers,
            "pending_reviewers": [
                r.reviewer_id for r in workflow.reviewers.values()
                if r.status == ReviewerStatus.PENDING
            ],
        }


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Approval Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Approval Manager (Step 170)")
    parser.add_argument("--pr", required=True, help="PR ID")
    parser.add_argument("--repo", default="owner/repo", help="Repository")
    parser.add_argument("--create", action="store_true", help="Create workflow")
    parser.add_argument("--approve", help="Reviewer to approve")
    parser.add_argument("--reject", help="Reviewer to reject")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--required", type=int, default=1, help="Required approvals")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manager = ApprovalManager()

    # Get or create workflow
    workflow = manager.get_workflow_by_pr(args.pr)

    if args.create or not workflow:
        workflow = manager.create_workflow(
            pr_id=args.pr,
            repository=args.repo,
            requirements=ApprovalRequirement(min_approvals=args.required),
        )

    if args.approve and workflow:
        manager.submit_approval(
            workflow_id=workflow.workflow_id,
            reviewer_id=args.approve,
            action=ApprovalAction.APPROVE,
        )

    if args.reject and workflow:
        manager.submit_approval(
            workflow_id=workflow.workflow_id,
            reviewer_id=args.reject,
            action=ApprovalAction.REQUEST_CHANGES,
        )

    # Show status
    if workflow:
        summary = manager.get_status_summary(workflow.workflow_id)

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            state_icon = "[OK]" if workflow.is_approved else "[  ]"
            print(f"{state_icon} Approval Workflow for PR {workflow.pr_id}")
            print(f"  State: {workflow.state.value}")
            print(f"  Approvals: {workflow.approvals_count}/{workflow.requirements.min_approvals}")

            if workflow.rejections_count > 0:
                print(f"  Rejections: {workflow.rejections_count}")

            print("\nReviewers:")
            for reviewer_id, reviewer in workflow.reviewers.items():
                valid_mark = "" if reviewer.valid else " (stale)"
                print(f"  - {reviewer_id}: {reviewer.status.value}{valid_mark}")

            if summary.get("missing_required"):
                print(f"\nMissing Required: {', '.join(summary['missing_required'])}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
