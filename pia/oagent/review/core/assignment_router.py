#!/usr/bin/env python3
"""
Review Assignment Router (Step 168)

Routes review assignments to appropriate reviewers based on
expertise, availability, and workload balancing.

PBTSO Phase: PLAN, DISTRIBUTE
Bus Topics: review.assignment.route, review.assignment.complete

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
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# Types
# ============================================================================

class ReviewerRole(Enum):
    """Roles for reviewers."""
    CODEOWNER = "codeowner"
    DOMAIN_EXPERT = "domain_expert"
    SECURITY_REVIEWER = "security_reviewer"
    ARCHITECT = "architect"
    GENERAL = "general"
    BOT = "bot"


class AssignmentReason(Enum):
    """Reasons for assignment."""
    CODEOWNER_MATCH = "codeowner_match"
    EXPERTISE_MATCH = "expertise_match"
    SECURITY_REQUIRED = "security_required"
    ARCHITECTURE_REVIEW = "architecture_review"
    LOAD_BALANCE = "load_balance"
    FALLBACK = "fallback"
    MANUAL = "manual"


class AssignmentStatus(Enum):
    """Status of an assignment."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class Reviewer:
    """
    A reviewer with their attributes.

    Attributes:
        reviewer_id: Unique ID (username)
        name: Display name
        roles: Reviewer roles
        expertise: Areas of expertise (file patterns, languages)
        capacity: Current review capacity (0-1)
        active_reviews: Number of active reviews
        max_reviews: Maximum concurrent reviews
        timezone: Reviewer's timezone
        available: Whether reviewer is available
    """
    reviewer_id: str
    name: str
    roles: List[ReviewerRole] = field(default_factory=list)
    expertise: List[str] = field(default_factory=list)
    capacity: float = 1.0
    active_reviews: int = 0
    max_reviews: int = 5
    timezone: str = "UTC"
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reviewer_id": self.reviewer_id,
            "name": self.name,
            "roles": [r.value for r in self.roles],
            "expertise": self.expertise,
            "capacity": self.capacity,
            "active_reviews": self.active_reviews,
            "max_reviews": self.max_reviews,
            "timezone": self.timezone,
            "available": self.available,
        }

    @property
    def has_capacity(self) -> bool:
        """Check if reviewer has capacity for more reviews."""
        return self.available and self.active_reviews < self.max_reviews


@dataclass
class AssignmentRule:
    """
    A rule for automatic assignment.

    Attributes:
        rule_id: Unique rule ID
        name: Rule name
        pattern: File pattern to match (glob)
        reviewers: Reviewers to assign when matched
        min_approvals: Minimum required approvals
        priority: Rule priority (higher = checked first)
        enabled: Whether rule is active
    """
    rule_id: str
    name: str
    pattern: str
    reviewers: List[str]
    min_approvals: int = 1
    priority: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def matches(self, file_path: str) -> bool:
        """Check if file matches the pattern."""
        import fnmatch
        return fnmatch.fnmatch(file_path, self.pattern)


@dataclass
class ReviewAssignment:
    """
    An assignment of reviewers to a PR.

    Attributes:
        assignment_id: Unique ID
        pr_id: Pull request ID
        repository: Repository name
        assignees: Assigned reviewers with reasons
        required_approvals: Number of required approvals
        rules_applied: Rules that triggered assignments
        status: Assignment status
        created_at: Assignment timestamp
        completed_at: Completion timestamp
    """
    assignment_id: str
    pr_id: str
    repository: str
    assignees: List[Dict[str, Any]] = field(default_factory=list)
    required_approvals: int = 1
    rules_applied: List[str] = field(default_factory=list)
    status: AssignmentStatus = AssignmentStatus.PENDING
    created_at: str = ""
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.assignment_id:
            self.assignment_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assignment_id": self.assignment_id,
            "pr_id": self.pr_id,
            "repository": self.repository,
            "assignees": self.assignees,
            "required_approvals": self.required_approvals,
            "rules_applied": self.rules_applied,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @property
    def reviewer_ids(self) -> List[str]:
        """Get list of reviewer IDs."""
        return [a["reviewer_id"] for a in self.assignees]


# ============================================================================
# Assignment Router
# ============================================================================

class AssignmentRouter:
    """
    Routes review assignments to appropriate reviewers.

    Uses CODEOWNERS patterns, expertise matching, and load balancing
    to determine optimal reviewer assignments.

    Example:
        router = AssignmentRouter()

        # Add reviewers
        router.add_reviewer(Reviewer(
            reviewer_id="alice",
            name="Alice",
            roles=[ReviewerRole.CODEOWNER],
            expertise=["**/auth/**", "python"],
        ))

        # Route assignment
        assignment = router.route(
            pr_id="123",
            repository="owner/repo",
            files_changed=["auth/login.py"],
        )

        print(f"Assigned to: {assignment.reviewer_ids}")
    """

    BUS_TOPICS = {
        "route": "review.assignment.route",
        "complete": "review.assignment.complete",
        "update": "review.assignment.update",
    }

    def __init__(
        self,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the assignment router.

        Args:
            bus_path: Path to event bus file
        """
        self.bus_path = bus_path or self._get_bus_path()
        self._reviewers: Dict[str, Reviewer] = {}
        self._rules: List[AssignmentRule] = []
        self._assignments: Dict[str, ReviewAssignment] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "assignment") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "assignment-router",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def add_reviewer(self, reviewer: Reviewer) -> None:
        """Add a reviewer to the pool."""
        self._reviewers[reviewer.reviewer_id] = reviewer

    def remove_reviewer(self, reviewer_id: str) -> None:
        """Remove a reviewer from the pool."""
        if reviewer_id in self._reviewers:
            del self._reviewers[reviewer_id]

    def add_rule(self, rule: AssignmentRule) -> None:
        """Add an assignment rule."""
        self._rules.append(rule)
        # Sort by priority (higher first)
        self._rules.sort(key=lambda r: -r.priority)

    def route(
        self,
        pr_id: str,
        repository: str,
        files_changed: List[str],
        author: Optional[str] = None,
        labels: Optional[List[str]] = None,
        min_reviewers: int = 1,
        max_reviewers: int = 3,
    ) -> ReviewAssignment:
        """
        Route a review to appropriate reviewers.

        Args:
            pr_id: Pull request ID
            repository: Repository name
            files_changed: List of changed files
            author: PR author (excluded from assignment)
            labels: PR labels for routing decisions
            min_reviewers: Minimum reviewers to assign
            max_reviewers: Maximum reviewers to assign

        Returns:
            ReviewAssignment with selected reviewers

        Emits:
            review.assignment.route
        """
        self._emit_event(self.BUS_TOPICS["route"], {
            "pr_id": pr_id,
            "repository": repository,
            "files_count": len(files_changed),
            "status": "started",
        })

        labels = labels or []
        assignees: List[Dict[str, Any]] = []
        rules_applied: List[str] = []
        assigned_ids: Set[str] = set()

        # Exclude author
        if author:
            assigned_ids.add(author)

        # Apply rules
        for rule in self._rules:
            if not rule.enabled:
                continue

            # Check if any file matches the rule
            matching_files = [f for f in files_changed if rule.matches(f)]
            if not matching_files:
                continue

            # Add rule's reviewers
            for reviewer_id in rule.reviewers:
                if reviewer_id in assigned_ids:
                    continue
                if len(assignees) >= max_reviewers:
                    break

                reviewer = self._reviewers.get(reviewer_id)
                if reviewer and reviewer.has_capacity:
                    assignees.append({
                        "reviewer_id": reviewer_id,
                        "reason": AssignmentReason.CODEOWNER_MATCH.value,
                        "matched_files": matching_files[:5],
                        "rule_id": rule.rule_id,
                    })
                    assigned_ids.add(reviewer_id)
                    rules_applied.append(rule.rule_id)

        # If not enough reviewers, try expertise matching
        if len(assignees) < min_reviewers:
            for reviewer in self._reviewers.values():
                if reviewer.reviewer_id in assigned_ids:
                    continue
                if not reviewer.has_capacity:
                    continue
                if len(assignees) >= max_reviewers:
                    break

                # Check expertise match
                for pattern in reviewer.expertise:
                    matching = [f for f in files_changed if self._matches_expertise(f, pattern)]
                    if matching:
                        assignees.append({
                            "reviewer_id": reviewer.reviewer_id,
                            "reason": AssignmentReason.EXPERTISE_MATCH.value,
                            "matched_files": matching[:5],
                        })
                        assigned_ids.add(reviewer.reviewer_id)
                        break

        # Check for security-related changes
        security_labels = {"security", "vulnerability", "auth"}
        if any(l.lower() in security_labels for l in labels):
            for reviewer in self._reviewers.values():
                if ReviewerRole.SECURITY_REVIEWER in reviewer.roles:
                    if reviewer.reviewer_id not in assigned_ids and reviewer.has_capacity:
                        assignees.append({
                            "reviewer_id": reviewer.reviewer_id,
                            "reason": AssignmentReason.SECURITY_REQUIRED.value,
                        })
                        assigned_ids.add(reviewer.reviewer_id)
                        break

        # Fallback to load-balanced assignment
        if len(assignees) < min_reviewers:
            available = [
                r for r in self._reviewers.values()
                if r.reviewer_id not in assigned_ids and r.has_capacity
            ]
            # Sort by capacity (most available first)
            available.sort(key=lambda r: r.active_reviews)

            for reviewer in available:
                if len(assignees) >= min_reviewers:
                    break
                assignees.append({
                    "reviewer_id": reviewer.reviewer_id,
                    "reason": AssignmentReason.LOAD_BALANCE.value,
                })
                assigned_ids.add(reviewer.reviewer_id)

        assignment = ReviewAssignment(
            assignment_id=str(uuid.uuid4())[:8],
            pr_id=pr_id,
            repository=repository,
            assignees=assignees,
            required_approvals=min_reviewers,
            rules_applied=rules_applied,
            status=AssignmentStatus.PENDING,
        )

        self._assignments[assignment.assignment_id] = assignment

        # Update reviewer active counts
        for assignee in assignees:
            reviewer_id = assignee["reviewer_id"]
            if reviewer_id in self._reviewers:
                self._reviewers[reviewer_id].active_reviews += 1

        self._emit_event(self.BUS_TOPICS["route"], {
            "pr_id": pr_id,
            "assignment_id": assignment.assignment_id,
            "assignees": [a["reviewer_id"] for a in assignees],
            "rules_applied": rules_applied,
            "status": "completed",
        })

        return assignment

    def _matches_expertise(self, file_path: str, pattern: str) -> bool:
        """Check if file matches expertise pattern."""
        import fnmatch

        # Check glob pattern
        if fnmatch.fnmatch(file_path, pattern):
            return True

        # Check language
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".go": "go",
            ".rs": "rust",
        }
        ext = Path(file_path).suffix.lower()
        if ext_map.get(ext) == pattern.lower():
            return True

        # Check directory
        if pattern in file_path.lower():
            return True

        return False

    def complete_assignment(
        self,
        assignment_id: str,
        approved_by: List[str],
    ) -> Optional[ReviewAssignment]:
        """
        Complete an assignment.

        Args:
            assignment_id: Assignment ID
            approved_by: List of reviewers who approved

        Returns:
            Updated assignment or None
        """
        assignment = self._assignments.get(assignment_id)
        if not assignment:
            return None

        assignment.status = AssignmentStatus.COMPLETED
        assignment.completed_at = datetime.now(timezone.utc).isoformat() + "Z"

        # Update reviewer active counts
        for assignee in assignment.assignees:
            reviewer_id = assignee["reviewer_id"]
            if reviewer_id in self._reviewers:
                self._reviewers[reviewer_id].active_reviews = max(
                    0, self._reviewers[reviewer_id].active_reviews - 1
                )

        self._emit_event(self.BUS_TOPICS["complete"], {
            "assignment_id": assignment_id,
            "pr_id": assignment.pr_id,
            "approved_by": approved_by,
        })

        return assignment

    def get_assignment(self, assignment_id: str) -> Optional[ReviewAssignment]:
        """Get an assignment by ID."""
        return self._assignments.get(assignment_id)

    def get_reviewer_workload(self) -> Dict[str, int]:
        """Get current workload for all reviewers."""
        return {
            r.reviewer_id: r.active_reviews
            for r in self._reviewers.values()
        }


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Assignment Router."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Assignment Router (Step 168)")
    parser.add_argument("--pr", required=True, help="PR ID")
    parser.add_argument("--repo", required=True, help="Repository")
    parser.add_argument("--files", nargs="*", default=[], help="Changed files")
    parser.add_argument("--author", help="PR author")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    router = AssignmentRouter()

    # Add some demo reviewers
    router.add_reviewer(Reviewer(
        reviewer_id="alice",
        name="Alice",
        roles=[ReviewerRole.CODEOWNER],
        expertise=["**/auth/**", "python"],
    ))
    router.add_reviewer(Reviewer(
        reviewer_id="bob",
        name="Bob",
        roles=[ReviewerRole.DOMAIN_EXPERT],
        expertise=["**/api/**", "typescript"],
    ))
    router.add_reviewer(Reviewer(
        reviewer_id="review-bot",
        name="Review Bot",
        roles=[ReviewerRole.BOT],
        expertise=["*"],
    ))

    # Add some rules
    router.add_rule(AssignmentRule(
        rule_id="auth",
        name="Auth Codeowners",
        pattern="**/auth/**",
        reviewers=["alice"],
        priority=10,
    ))

    assignment = router.route(
        pr_id=args.pr,
        repository=args.repo,
        files_changed=args.files,
        author=args.author,
    )

    if args.json:
        print(json.dumps(assignment.to_dict(), indent=2))
    else:
        print(f"Assignment for PR {assignment.pr_id}")
        print(f"  ID: {assignment.assignment_id}")
        print(f"  Status: {assignment.status.value}")
        print(f"  Required Approvals: {assignment.required_approvals}")
        print("\nAssignees:")
        for assignee in assignment.assignees:
            print(f"  - {assignee['reviewer_id']} ({assignee['reason']})")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
