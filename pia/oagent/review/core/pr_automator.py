#!/usr/bin/env python3
"""
PR Review Automator (Step 162)

Automates pull request review workflows, including fetching PR data,
triggering reviews, and posting review comments back to the PR.

PBTSO Phase: PLAN, BUILD, VERIFY
Bus Topics: review.pr.automate, review.pr.fetch, review.pr.post

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Types
# ============================================================================

class PRState(Enum):
    """Pull request states."""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    DRAFT = "draft"


class PRProvider(Enum):
    """Supported PR providers."""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    LOCAL = "local"


class ReviewAction(Enum):
    """Actions to take on a PR."""
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    COMMENT = "comment"


@dataclass
class PRFile:
    """A file changed in a PR."""
    path: str
    status: str  # added, modified, deleted, renamed
    additions: int
    deletions: int
    patch: Optional[str] = None
    previous_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PRInfo:
    """
    Information about a pull request.

    Attributes:
        pr_id: Pull request ID/number
        title: PR title
        description: PR description/body
        author: Author username
        state: Current state
        base_branch: Target branch
        head_branch: Source branch
        base_sha: Base commit SHA
        head_sha: Head commit SHA
        files: List of changed files
        labels: PR labels
        reviewers: Requested reviewers
        created_at: Creation timestamp
        updated_at: Last update timestamp
        url: PR URL
        provider: PR provider
    """
    pr_id: str
    title: str
    description: str
    author: str
    state: PRState
    base_branch: str
    head_branch: str
    base_sha: str
    head_sha: str
    files: List[PRFile] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    url: str = ""
    provider: PRProvider = PRProvider.GITHUB

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["state"] = self.state.value
        result["provider"] = self.provider.value
        result["files"] = [f.to_dict() if isinstance(f, PRFile) else f for f in self.files]
        return result

    @property
    def total_additions(self) -> int:
        """Total lines added."""
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        """Total lines deleted."""
        return sum(f.deletions for f in self.files)

    @property
    def file_paths(self) -> List[str]:
        """Get list of file paths."""
        return [f.path for f in self.files]


@dataclass
class ReviewComment:
    """A comment to post on the PR."""
    body: str
    path: Optional[str] = None
    line: Optional[int] = None
    side: str = "RIGHT"  # LEFT or RIGHT for diff
    start_line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AutomatedReview:
    """
    Result of an automated PR review.

    Attributes:
        pr_id: Pull request ID
        action: Review action to take
        body: Review body/summary
        comments: Inline comments
        files_reviewed: Number of files reviewed
        issues_found: Total issues found
        blocking_issues: Number of blocking issues
        duration_ms: Review duration
        review_id: Unique review ID
        submitted_at: Submission timestamp
    """
    pr_id: str
    action: ReviewAction
    body: str
    comments: List[ReviewComment] = field(default_factory=list)
    files_reviewed: int = 0
    issues_found: int = 0
    blocking_issues: int = 0
    duration_ms: float = 0
    review_id: str = ""
    submitted_at: str = ""

    def __post_init__(self):
        if not self.review_id:
            self.review_id = str(uuid.uuid4())[:8]
        if not self.submitted_at:
            self.submitted_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["action"] = self.action.value
        result["comments"] = [c.to_dict() if isinstance(c, ReviewComment) else c for c in self.comments]
        return result


# ============================================================================
# PR Review Automator
# ============================================================================

class PRReviewAutomator:
    """
    Automates pull request review workflows.

    Handles fetching PR data, coordinating reviews, and posting results.

    Example:
        automator = PRReviewAutomator()

        # Fetch PR info
        pr_info = automator.fetch_pr("owner/repo", "123")

        # Run automated review (integrates with orchestrator)
        review = await automator.run_review(pr_info)

        # Post review back to PR
        automator.post_review(pr_info, review)
    """

    BUS_TOPICS = {
        "automate": "review.pr.automate",
        "fetch": "review.pr.fetch",
        "post": "review.pr.post",
        "status": "review.pr.status",
    }

    def __init__(
        self,
        bus_path: Optional[Path] = None,
        github_token: Optional[str] = None,
    ):
        """
        Initialize the PR automator.

        Args:
            bus_path: Path to event bus file
            github_token: GitHub API token (optional)
        """
        self.bus_path = bus_path or self._get_bus_path()
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self._review_callbacks: Dict[str, Any] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "pr") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "pr-automator",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _run_gh_command(self, args: List[str]) -> Tuple[int, str, str]:
        """Run a gh CLI command."""
        try:
            env = os.environ.copy()
            if self.github_token:
                env["GITHUB_TOKEN"] = self.github_token

            result = subprocess.run(
                ["gh"] + args,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            return -1, "", "gh CLI not found"
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"

    def fetch_pr(
        self,
        repo: str,
        pr_number: str,
        provider: PRProvider = PRProvider.GITHUB,
    ) -> Optional[PRInfo]:
        """
        Fetch pull request information.

        Args:
            repo: Repository in owner/repo format
            pr_number: PR number
            provider: PR provider

        Returns:
            PRInfo or None if fetch failed

        Emits:
            review.pr.fetch
        """
        self._emit_event(self.BUS_TOPICS["fetch"], {
            "repo": repo,
            "pr_number": pr_number,
            "provider": provider.value,
            "status": "started",
        })

        if provider == PRProvider.GITHUB:
            return self._fetch_github_pr(repo, pr_number)
        elif provider == PRProvider.LOCAL:
            return self._fetch_local_pr(pr_number)
        else:
            return None

    def _fetch_github_pr(self, repo: str, pr_number: str) -> Optional[PRInfo]:
        """Fetch PR from GitHub."""
        # Get PR info
        returncode, stdout, stderr = self._run_gh_command([
            "pr", "view", pr_number,
            "--repo", repo,
            "--json", "number,title,body,author,state,baseRefName,headRefName,files,labels,reviewRequests,createdAt,updatedAt,url"
        ])

        if returncode != 0:
            self._emit_event(self.BUS_TOPICS["fetch"], {
                "repo": repo,
                "pr_number": pr_number,
                "status": "failed",
                "error": stderr,
            })
            return None

        try:
            data = json.loads(stdout)

            # Parse state
            state_map = {
                "OPEN": PRState.OPEN,
                "CLOSED": PRState.CLOSED,
                "MERGED": PRState.MERGED,
            }
            state = state_map.get(data.get("state", "OPEN"), PRState.OPEN)

            # Parse files
            files = []
            for f in data.get("files", []):
                files.append(PRFile(
                    path=f.get("path", ""),
                    status=f.get("status", "modified").lower(),
                    additions=f.get("additions", 0),
                    deletions=f.get("deletions", 0),
                    patch=f.get("patch"),
                ))

            # Get SHAs
            returncode2, stdout2, _ = self._run_gh_command([
                "pr", "view", pr_number,
                "--repo", repo,
                "--json", "baseRefOid,headRefOid"
            ])
            sha_data = json.loads(stdout2) if returncode2 == 0 else {}

            pr_info = PRInfo(
                pr_id=str(data.get("number", pr_number)),
                title=data.get("title", ""),
                description=data.get("body", "") or "",
                author=data.get("author", {}).get("login", ""),
                state=state,
                base_branch=data.get("baseRefName", "main"),
                head_branch=data.get("headRefName", ""),
                base_sha=sha_data.get("baseRefOid", ""),
                head_sha=sha_data.get("headRefOid", ""),
                files=files,
                labels=[l.get("name", "") for l in data.get("labels", [])],
                reviewers=[r.get("login", "") for r in data.get("reviewRequests", [])],
                created_at=data.get("createdAt", ""),
                updated_at=data.get("updatedAt", ""),
                url=data.get("url", ""),
                provider=PRProvider.GITHUB,
            )

            self._emit_event(self.BUS_TOPICS["fetch"], {
                "repo": repo,
                "pr_number": pr_number,
                "status": "completed",
                "files_count": len(files),
            })

            return pr_info

        except json.JSONDecodeError:
            self._emit_event(self.BUS_TOPICS["fetch"], {
                "repo": repo,
                "pr_number": pr_number,
                "status": "failed",
                "error": "Failed to parse PR data",
            })
            return None

    def _fetch_local_pr(self, branch: str) -> Optional[PRInfo]:
        """Fetch PR info from local git repo."""
        try:
            # Get diff against main
            result = subprocess.run(
                ["git", "diff", "--name-status", "main...HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            files = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                status = parts[0]
                path = parts[1] if len(parts) > 1 else ""

                status_map = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}
                files.append(PRFile(
                    path=path,
                    status=status_map.get(status[0], "modified"),
                    additions=0,
                    deletions=0,
                ))

            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
            )
            head_branch = result.stdout.strip() if result.returncode == 0 else branch

            return PRInfo(
                pr_id=f"local-{head_branch}",
                title=f"Local changes on {head_branch}",
                description="",
                author="local",
                state=PRState.OPEN,
                base_branch="main",
                head_branch=head_branch,
                base_sha="",
                head_sha="",
                files=files,
                provider=PRProvider.LOCAL,
            )

        except subprocess.TimeoutExpired:
            return None

    async def run_review(
        self,
        pr_info: PRInfo,
        review_orchestrator: Optional[Any] = None,
    ) -> AutomatedReview:
        """
        Run automated review on a PR.

        Args:
            pr_info: PR information
            review_orchestrator: Optional orchestrator instance

        Returns:
            AutomatedReview result

        Emits:
            review.pr.automate
        """
        start_time = time.time()

        self._emit_event(self.BUS_TOPICS["automate"], {
            "pr_id": pr_info.pr_id,
            "files_count": len(pr_info.files),
            "status": "started",
        })

        # If orchestrator provided, use it
        if review_orchestrator:
            result = review_orchestrator.run(pr_info.file_paths)
            action = self._determine_action(result)
            body = result.generated_review.to_markdown() if result.generated_review else ""
            comments = self._convert_comments(result)
            issues = result.total_issues
            blocking = result.blocking_issues
        else:
            # Simple review without orchestrator
            action = ReviewAction.COMMENT
            body = self._generate_simple_review(pr_info)
            comments = []
            issues = 0
            blocking = 0

        review = AutomatedReview(
            pr_id=pr_info.pr_id,
            action=action,
            body=body,
            comments=comments,
            files_reviewed=len(pr_info.files),
            issues_found=issues,
            blocking_issues=blocking,
            duration_ms=(time.time() - start_time) * 1000,
        )

        self._emit_event(self.BUS_TOPICS["automate"], {
            "pr_id": pr_info.pr_id,
            "action": action.value,
            "issues_found": issues,
            "blocking_issues": blocking,
            "status": "completed",
        })

        return review

    def _determine_action(self, result: Any) -> ReviewAction:
        """Determine review action from orchestrator result."""
        if result.blocking_issues > 0:
            return ReviewAction.REQUEST_CHANGES
        elif result.total_issues > 0:
            return ReviewAction.COMMENT
        else:
            return ReviewAction.APPROVE

    def _convert_comments(self, result: Any) -> List[ReviewComment]:
        """Convert orchestrator comments to review comments."""
        comments = []
        if result.generated_review:
            for c in result.generated_review.comments[:50]:  # Limit to 50
                comments.append(ReviewComment(
                    body=f"**{c.title}**\n\n{c.body}",
                    path=c.file,
                    line=c.line,
                ))
        return comments

    def _generate_simple_review(self, pr_info: PRInfo) -> str:
        """Generate a simple review summary."""
        lines = [
            f"## PR Review: {pr_info.title}",
            "",
            f"**Files Changed:** {len(pr_info.files)}",
            f"**Lines Added:** {pr_info.total_additions}",
            f"**Lines Deleted:** {pr_info.total_deletions}",
            "",
            "### Changed Files",
            "",
        ]

        for f in pr_info.files[:20]:
            lines.append(f"- `{f.path}` ({f.status})")

        if len(pr_info.files) > 20:
            lines.append(f"- ... and {len(pr_info.files) - 20} more files")

        lines.extend([
            "",
            "_Automated review by Review Agent_",
        ])

        return "\n".join(lines)

    def post_review(
        self,
        pr_info: PRInfo,
        review: AutomatedReview,
    ) -> bool:
        """
        Post review to PR.

        Args:
            pr_info: PR information
            review: Review to post

        Returns:
            True if successful

        Emits:
            review.pr.post
        """
        self._emit_event(self.BUS_TOPICS["post"], {
            "pr_id": pr_info.pr_id,
            "action": review.action.value,
            "comments_count": len(review.comments),
            "status": "started",
        })

        if pr_info.provider == PRProvider.GITHUB:
            success = self._post_github_review(pr_info, review)
        else:
            # Local or unsupported - just log
            print(f"Review for {pr_info.pr_id}:")
            print(review.body)
            success = True

        self._emit_event(self.BUS_TOPICS["post"], {
            "pr_id": pr_info.pr_id,
            "status": "completed" if success else "failed",
        })

        return success

    def _post_github_review(self, pr_info: PRInfo, review: AutomatedReview) -> bool:
        """Post review to GitHub PR."""
        # Map action to GitHub review event
        event_map = {
            ReviewAction.APPROVE: "APPROVE",
            ReviewAction.REQUEST_CHANGES: "REQUEST_CHANGES",
            ReviewAction.COMMENT: "COMMENT",
        }
        event = event_map[review.action]

        # Extract repo from URL
        match = re.search(r"github\.com/([^/]+/[^/]+)", pr_info.url)
        if not match:
            return False
        repo = match.group(1)

        # Post review
        returncode, _, stderr = self._run_gh_command([
            "pr", "review", pr_info.pr_id,
            "--repo", repo,
            "--body", review.body,
            "--" + event.lower().replace("_", "-"),
        ])

        if returncode != 0:
            return False

        # Post inline comments
        for comment in review.comments[:10]:  # Limit inline comments
            if comment.path and comment.line:
                self._run_gh_command([
                    "pr", "comment", pr_info.pr_id,
                    "--repo", repo,
                    "--body", f"**{comment.path}:{comment.line}**\n\n{comment.body}",
                ])

        return True


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for PR Review Automator."""
    import argparse

    parser = argparse.ArgumentParser(description="PR Review Automator (Step 162)")
    parser.add_argument("--repo", help="Repository (owner/repo)")
    parser.add_argument("--pr", help="PR number")
    parser.add_argument("--local", action="store_true", help="Review local changes")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    automator = PRReviewAutomator()

    if args.local:
        pr_info = automator._fetch_local_pr("HEAD")
    elif args.repo and args.pr:
        pr_info = automator.fetch_pr(args.repo, args.pr)
    else:
        parser.print_help()
        return 1

    if not pr_info:
        print("Failed to fetch PR info")
        return 1

    if args.json:
        print(json.dumps(pr_info.to_dict(), indent=2))
    else:
        print(f"PR: {pr_info.title}")
        print(f"  Author: {pr_info.author}")
        print(f"  State: {pr_info.state.value}")
        print(f"  Files: {len(pr_info.files)}")
        print(f"  +{pr_info.total_additions} -{pr_info.total_deletions}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
