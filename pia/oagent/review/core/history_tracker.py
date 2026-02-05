#!/usr/bin/env python3
"""
Review History Tracker (Step 164)

Tracks review decisions, outcomes, and patterns over time.
Enables learning from past reviews and trend analysis.

PBTSO Phase: OBSERVE, DISTILL
Bus Topics: review.history.track, review.history.query, review.history.trend

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator


# ============================================================================
# Types
# ============================================================================

class ReviewOutcome(Enum):
    """Outcomes of reviews."""
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    MERGED = "merged"


class ReviewerType(Enum):
    """Types of reviewers."""
    AUTOMATED = "automated"
    HUMAN = "human"
    HYBRID = "hybrid"


@dataclass
class ReviewRecord:
    """
    A record of a single review.

    Attributes:
        record_id: Unique record ID
        review_id: Review ID from the review system
        pr_id: Pull request ID
        repository: Repository name
        author: PR author
        reviewer: Reviewer ID (agent or human)
        reviewer_type: Type of reviewer
        outcome: Review outcome
        issues_found: Number of issues found
        blocking_issues: Number of blocking issues
        quality_score: Quality score if computed
        duration_ms: Review duration
        files_reviewed: Files reviewed
        lines_changed: Lines changed
        comments_posted: Comments posted
        created_at: Review timestamp
        tags: Categorization tags
        metadata: Additional metadata
    """
    record_id: str
    review_id: str
    pr_id: str
    repository: str
    author: str
    reviewer: str
    reviewer_type: ReviewerType
    outcome: ReviewOutcome
    issues_found: int
    blocking_issues: int
    quality_score: Optional[float] = None
    duration_ms: float = 0
    files_reviewed: int = 0
    lines_changed: int = 0
    comments_posted: int = 0
    created_at: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["reviewer_type"] = self.reviewer_type.value
        result["outcome"] = self.outcome.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewRecord":
        """Create from dictionary."""
        data = data.copy()
        data["reviewer_type"] = ReviewerType(data.get("reviewer_type", "automated"))
        data["outcome"] = ReviewOutcome(data.get("outcome", "approved"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReviewTrend:
    """
    Trend analysis of reviews.

    Attributes:
        period: Time period (e.g., "7d", "30d")
        total_reviews: Total reviews in period
        approval_rate: Percentage approved
        avg_issues: Average issues per review
        avg_duration_ms: Average review duration
        top_issue_types: Most common issue types
        quality_trend: Quality score trend
        by_outcome: Count by outcome
        by_repository: Count by repository
    """
    period: str
    total_reviews: int
    approval_rate: float
    avg_issues: float
    avg_duration_ms: float
    top_issue_types: List[str] = field(default_factory=list)
    quality_trend: List[float] = field(default_factory=list)
    by_outcome: Dict[str, int] = field(default_factory=dict)
    by_repository: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ReviewHistory:
    """
    Collection of review history with query support.

    Attributes:
        records: List of review records
        total_count: Total records in history
        query_params: Parameters used for filtering
    """
    records: List[ReviewRecord] = field(default_factory=list)
    total_count: int = 0
    query_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "records": [r.to_dict() for r in self.records],
            "total_count": self.total_count,
            "query_params": self.query_params,
        }


# ============================================================================
# Review History Tracker
# ============================================================================

class ReviewHistoryTracker:
    """
    Tracks and queries review history.

    Stores review records and provides trend analysis.

    Example:
        tracker = ReviewHistoryTracker()

        # Record a review
        record = tracker.record_review(
            review_id="rev-123",
            pr_id="PR-456",
            repository="owner/repo",
            author="dev",
            outcome=ReviewOutcome.APPROVED,
            issues_found=5,
        )

        # Query history
        history = tracker.query(
            repository="owner/repo",
            start_date=datetime.now() - timedelta(days=7),
        )

        # Get trends
        trend = tracker.compute_trend(period="7d")
    """

    BUS_TOPICS = {
        "track": "review.history.track",
        "query": "review.history.query",
        "trend": "review.history.trend",
    }

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the history tracker.

        Args:
            storage_path: Path to store history
            bus_path: Path to event bus file
        """
        self.bus_path = bus_path or self._get_bus_path()
        self.storage_path = storage_path or self._get_storage_path()
        self._cache: Dict[str, ReviewRecord] = {}

        # Load existing records
        self._load_history()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_storage_path(self) -> Path:
        """Get path to history storage."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "review_history.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "history") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "history-tracker",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _load_history(self) -> None:
        """Load history from storage."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        record = ReviewRecord.from_dict(data)
                        self._cache[record.record_id] = record
        except (IOError, json.JSONDecodeError):
            pass

    def _save_record(self, record: ReviewRecord) -> None:
        """Save a record to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.storage_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def record_review(
        self,
        review_id: str,
        pr_id: str,
        repository: str,
        author: str,
        outcome: ReviewOutcome,
        issues_found: int,
        blocking_issues: int = 0,
        reviewer: str = "review-agent",
        reviewer_type: ReviewerType = ReviewerType.AUTOMATED,
        quality_score: Optional[float] = None,
        duration_ms: float = 0,
        files_reviewed: int = 0,
        lines_changed: int = 0,
        comments_posted: int = 0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReviewRecord:
        """
        Record a new review.

        Args:
            review_id: Review ID
            pr_id: Pull request ID
            repository: Repository name
            author: PR author
            outcome: Review outcome
            issues_found: Number of issues
            blocking_issues: Number of blocking issues
            reviewer: Reviewer ID
            reviewer_type: Type of reviewer
            quality_score: Quality score if computed
            duration_ms: Review duration
            files_reviewed: Files reviewed
            lines_changed: Lines changed
            comments_posted: Comments posted
            tags: Categorization tags
            metadata: Additional metadata

        Returns:
            The created ReviewRecord

        Emits:
            review.history.track
        """
        record = ReviewRecord(
            record_id=str(uuid.uuid4())[:8],
            review_id=review_id,
            pr_id=pr_id,
            repository=repository,
            author=author,
            reviewer=reviewer,
            reviewer_type=reviewer_type,
            outcome=outcome,
            issues_found=issues_found,
            blocking_issues=blocking_issues,
            quality_score=quality_score,
            duration_ms=duration_ms,
            files_reviewed=files_reviewed,
            lines_changed=lines_changed,
            comments_posted=comments_posted,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Store record
        self._cache[record.record_id] = record
        self._save_record(record)

        # Emit tracking event
        self._emit_event(self.BUS_TOPICS["track"], {
            "record_id": record.record_id,
            "review_id": record.review_id,
            "repository": record.repository,
            "outcome": record.outcome.value,
            "issues_found": record.issues_found,
        })

        return record

    def get_record(self, record_id: str) -> Optional[ReviewRecord]:
        """Get a specific record by ID."""
        return self._cache.get(record_id)

    def query(
        self,
        repository: Optional[str] = None,
        author: Optional[str] = None,
        reviewer: Optional[str] = None,
        outcome: Optional[ReviewOutcome] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ReviewHistory:
        """
        Query review history.

        Args:
            repository: Filter by repository
            author: Filter by author
            reviewer: Filter by reviewer
            outcome: Filter by outcome
            start_date: Start of date range
            end_date: End of date range
            tags: Filter by tags (any match)
            limit: Maximum records to return
            offset: Offset for pagination

        Returns:
            ReviewHistory with matching records

        Emits:
            review.history.query
        """
        query_params = {
            "repository": repository,
            "author": author,
            "reviewer": reviewer,
            "outcome": outcome.value if outcome else None,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "tags": tags,
            "limit": limit,
            "offset": offset,
        }

        self._emit_event(self.BUS_TOPICS["query"], {
            "params": {k: v for k, v in query_params.items() if v is not None},
            "status": "started",
        })

        # Filter records
        matching = []
        for record in self._cache.values():
            if repository and record.repository != repository:
                continue
            if author and record.author != author:
                continue
            if reviewer and record.reviewer != reviewer:
                continue
            if outcome and record.outcome != outcome:
                continue
            if start_date:
                record_date = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
                if record_date < start_date:
                    continue
            if end_date:
                record_date = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
                if record_date > end_date:
                    continue
            if tags and not any(t in record.tags for t in tags):
                continue

            matching.append(record)

        # Sort by date descending
        matching.sort(key=lambda r: r.created_at, reverse=True)

        # Apply pagination
        total = len(matching)
        matching = matching[offset:offset + limit]

        history = ReviewHistory(
            records=matching,
            total_count=total,
            query_params=query_params,
        )

        self._emit_event(self.BUS_TOPICS["query"], {
            "params": {k: v for k, v in query_params.items() if v is not None},
            "results": len(matching),
            "total": total,
            "status": "completed",
        })

        return history

    def compute_trend(
        self,
        period: str = "7d",
        repository: Optional[str] = None,
    ) -> ReviewTrend:
        """
        Compute trend analysis.

        Args:
            period: Time period ("7d", "30d", "90d")
            repository: Optional repository filter

        Returns:
            ReviewTrend with analysis

        Emits:
            review.history.trend
        """
        # Parse period
        days = int(period.rstrip("d"))
        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Query records in period
        history = self.query(
            repository=repository,
            start_date=start_date,
            limit=10000,
        )

        records = history.records

        if not records:
            return ReviewTrend(
                period=period,
                total_reviews=0,
                approval_rate=0.0,
                avg_issues=0.0,
                avg_duration_ms=0.0,
            )

        # Calculate metrics
        total = len(records)
        approved = sum(1 for r in records if r.outcome == ReviewOutcome.APPROVED)
        approval_rate = (approved / total * 100) if total > 0 else 0.0

        avg_issues = sum(r.issues_found for r in records) / total
        avg_duration = sum(r.duration_ms for r in records) / total

        # Count by outcome
        by_outcome: Dict[str, int] = {}
        for r in records:
            by_outcome[r.outcome.value] = by_outcome.get(r.outcome.value, 0) + 1

        # Count by repository
        by_repository: Dict[str, int] = {}
        for r in records:
            by_repository[r.repository] = by_repository.get(r.repository, 0) + 1

        # Quality trend (daily averages)
        quality_scores = [r.quality_score for r in records if r.quality_score is not None]
        quality_trend = quality_scores[:10] if quality_scores else []

        trend = ReviewTrend(
            period=period,
            total_reviews=total,
            approval_rate=approval_rate,
            avg_issues=avg_issues,
            avg_duration_ms=avg_duration,
            quality_trend=quality_trend,
            by_outcome=by_outcome,
            by_repository=by_repository,
        )

        self._emit_event(self.BUS_TOPICS["trend"], {
            "period": period,
            "total_reviews": total,
            "approval_rate": approval_rate,
            "repository": repository,
        })

        return trend

    def get_author_stats(self, author: str) -> Dict[str, Any]:
        """Get statistics for a specific author."""
        history = self.query(author=author, limit=1000)
        records = history.records

        if not records:
            return {"author": author, "reviews": 0}

        return {
            "author": author,
            "reviews": len(records),
            "approval_rate": sum(1 for r in records if r.outcome == ReviewOutcome.APPROVED) / len(records) * 100,
            "avg_issues": sum(r.issues_found for r in records) / len(records),
            "avg_quality": sum(r.quality_score or 0 for r in records) / len(records) if any(r.quality_score for r in records) else None,
        }

    def iter_records(self) -> Iterator[ReviewRecord]:
        """Iterate over all records."""
        yield from self._cache.values()


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Review History Tracker."""
    import argparse

    parser = argparse.ArgumentParser(description="Review History Tracker (Step 164)")
    parser.add_argument("--query", action="store_true", help="Query history")
    parser.add_argument("--trend", action="store_true", help="Show trend analysis")
    parser.add_argument("--repo", help="Repository filter")
    parser.add_argument("--period", default="7d", help="Trend period (7d, 30d, 90d)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    tracker = ReviewHistoryTracker()

    if args.trend:
        trend = tracker.compute_trend(period=args.period, repository=args.repo)

        if args.json:
            print(json.dumps(trend.to_dict(), indent=2))
        else:
            print(f"Review Trend ({trend.period})")
            print(f"  Total Reviews: {trend.total_reviews}")
            print(f"  Approval Rate: {trend.approval_rate:.1f}%")
            print(f"  Avg Issues: {trend.avg_issues:.1f}")
            print(f"  Avg Duration: {trend.avg_duration_ms:.0f}ms")
            if trend.by_outcome:
                print("\nBy Outcome:")
                for outcome, count in trend.by_outcome.items():
                    print(f"  {outcome}: {count}")

    elif args.query:
        history = tracker.query(repository=args.repo, limit=20)

        if args.json:
            print(json.dumps(history.to_dict(), indent=2))
        else:
            print(f"Review History ({history.total_count} total)")
            for record in history.records:
                print(f"  {record.record_id}: {record.pr_id} - {record.outcome.value}")

    else:
        # Show stats
        total = len(list(tracker.iter_records()))
        print(f"Review History Tracker")
        print(f"  Total Records: {total}")
        print(f"  Storage: {tracker.storage_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
