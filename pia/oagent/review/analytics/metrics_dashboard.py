#!/usr/bin/env python3
"""
Review Metrics Dashboard (Step 171)

Provides analytics and visualization data for code review metrics.

PBTSO Phase: OBSERVE, DISTILL
Bus Topics: review.metrics.dashboard, review.metrics.trend

Tracks:
- Review completion rates
- Issue distribution
- Quality trends over time
- Reviewer performance
- Code hotspots

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300  # 5 minutes
A2A_HEARTBEAT_TIMEOUT = 900   # 15 minutes


# ============================================================================
# Types
# ============================================================================

class MetricType(Enum):
    """Types of review metrics."""
    REVIEW_COUNT = "review_count"
    ISSUE_COUNT = "issue_count"
    BLOCKING_RATE = "blocking_rate"
    APPROVAL_RATE = "approval_rate"
    AVG_REVIEW_TIME = "avg_review_time"
    FILES_REVIEWED = "files_reviewed"
    LINES_REVIEWED = "lines_reviewed"
    QUALITY_SCORE = "quality_score"
    DEBT_HOURS = "debt_hours"
    HOTSPOT_COUNT = "hotspot_count"


class TrendDirection(Enum):
    """Direction of metric trend."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class TimePeriod(Enum):
    """Time periods for aggregation."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class MetricTrend:
    """
    Trend information for a metric.

    Attributes:
        metric_type: Type of metric
        current_value: Current metric value
        previous_value: Previous period value
        change_percent: Percentage change
        direction: Trend direction
        data_points: Historical data points
    """
    metric_type: MetricType
    current_value: float
    previous_value: float
    change_percent: float
    direction: TrendDirection
    data_points: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "current_value": round(self.current_value, 2),
            "previous_value": round(self.previous_value, 2),
            "change_percent": round(self.change_percent, 2),
            "direction": self.direction.value,
            "data_points": self.data_points,
        }


@dataclass
class MetricSummary:
    """
    Summary statistics for a metric.

    Attributes:
        metric_type: Type of metric
        total: Total/sum value
        average: Average value
        minimum: Minimum value
        maximum: Maximum value
        median: Median value
        std_dev: Standard deviation
        count: Number of data points
    """
    metric_type: MetricType
    total: float
    average: float
    minimum: float
    maximum: float
    median: float
    std_dev: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "total": round(self.total, 2),
            "average": round(self.average, 2),
            "minimum": round(self.minimum, 2),
            "maximum": round(self.maximum, 2),
            "median": round(self.median, 2),
            "std_dev": round(self.std_dev, 2),
            "count": self.count,
        }


@dataclass
class FileHotspot:
    """Represents a code hotspot (frequently changed/problematic file)."""
    file_path: str
    review_count: int
    issue_count: int
    avg_issues_per_review: float
    last_reviewed: str
    change_frequency: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ReviewerStats:
    """Statistics for a reviewer."""
    reviewer_id: str
    reviews_completed: int
    issues_found: int
    avg_review_time_min: float
    approval_rate: float
    blocking_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "avg_review_time_min": round(self.avg_review_time_min, 1),
            "approval_rate": round(self.approval_rate, 2),
            "blocking_rate": round(self.blocking_rate, 2),
        }


@dataclass
class DashboardData:
    """
    Complete dashboard data.

    Attributes:
        dashboard_id: Unique dashboard instance ID
        generated_at: Generation timestamp
        period: Time period covered
        summaries: Metric summaries
        trends: Metric trends
        hotspots: Code hotspots
        reviewer_stats: Per-reviewer statistics
        issue_distribution: Issues by category
        quality_by_path: Quality scores by path
    """
    dashboard_id: str
    generated_at: str
    period: TimePeriod
    summaries: List[MetricSummary] = field(default_factory=list)
    trends: List[MetricTrend] = field(default_factory=list)
    hotspots: List[FileHotspot] = field(default_factory=list)
    reviewer_stats: List[ReviewerStats] = field(default_factory=list)
    issue_distribution: Dict[str, int] = field(default_factory=dict)
    quality_by_path: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dashboard_id": self.dashboard_id,
            "generated_at": self.generated_at,
            "period": self.period.value,
            "summaries": [s.to_dict() for s in self.summaries],
            "trends": [t.to_dict() for t in self.trends],
            "hotspots": [h.to_dict() for h in self.hotspots],
            "reviewer_stats": [r.to_dict() for r in self.reviewer_stats],
            "issue_distribution": self.issue_distribution,
            "quality_by_path": {k: round(v, 2) for k, v in self.quality_by_path.items()},
        }

    def to_markdown(self) -> str:
        """Convert to markdown report."""
        lines = [
            "# Review Metrics Dashboard",
            "",
            f"**Generated:** {self.generated_at}",
            f"**Period:** {self.period.value.title()}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Total | Average | Min | Max |",
            "|--------|-------|---------|-----|-----|",
        ]

        for s in self.summaries:
            lines.append(
                f"| {s.metric_type.value.replace('_', ' ').title()} | "
                f"{s.total:.1f} | {s.average:.2f} | {s.minimum:.1f} | {s.maximum:.1f} |"
            )

        if self.trends:
            lines.extend([
                "",
                "## Trends",
                "",
            ])
            for t in self.trends:
                arrow = {"up": "^", "down": "v", "stable": "-"}.get(t.direction.value, "-")
                lines.append(
                    f"- **{t.metric_type.value.replace('_', ' ').title()}**: "
                    f"{t.current_value:.1f} ({arrow} {t.change_percent:+.1f}%)"
                )

        if self.hotspots:
            lines.extend([
                "",
                "## Code Hotspots",
                "",
                "| File | Reviews | Issues | Avg Issues |",
                "|------|---------|--------|------------|",
            ])
            for h in self.hotspots[:10]:
                lines.append(
                    f"| `{h.file_path}` | {h.review_count} | "
                    f"{h.issue_count} | {h.avg_issues_per_review:.1f} |"
                )

        if self.issue_distribution:
            lines.extend([
                "",
                "## Issue Distribution",
                "",
            ])
            for category, count in sorted(self.issue_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"- **{category.replace('_', ' ').title()}**: {count}")

        lines.extend([
            "",
            "_Generated by Review Metrics Dashboard_",
        ])

        return "\n".join(lines)


@dataclass
class DashboardConfig:
    """Configuration for the metrics dashboard."""
    data_retention_days: int = 90
    hotspot_threshold: int = 5
    trend_periods: int = 7
    aggregation_period: TimePeriod = TimePeriod.DAY
    include_reviewer_stats: bool = True
    include_file_hotspots: bool = True
    max_hotspots: int = 20

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "aggregation_period": self.aggregation_period.value,
        }


# ============================================================================
# Review Data Store
# ============================================================================

@dataclass
class ReviewRecord:
    """Record of a single review."""
    review_id: str
    files: List[str]
    issue_count: int
    blocking_count: int
    decision: str
    duration_ms: float
    quality_score: float
    reviewer_id: Optional[str]
    created_at: str
    categories: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewRecord":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ReviewDataStore:
    """Persistent store for review data."""

    def __init__(self, store_path: Path):
        """
        Initialize the data store.

        Args:
            store_path: Path to the data file
        """
        self.store_path = store_path
        self._ensure_store()

    def _ensure_store(self) -> None:
        """Ensure store file exists."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._write_store({"reviews": [], "version": 1})

    def _read_store(self) -> Dict[str, Any]:
        """Read store with file locking."""
        with open(self.store_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_store(self, data: Dict[str, Any]) -> None:
        """Write store with file locking."""
        with open(self.store_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def add_review(self, record: ReviewRecord) -> None:
        """Add a review record."""
        data = self._read_store()
        data["reviews"].append(record.to_dict())
        self._write_store(data)

    def get_reviews(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[ReviewRecord]:
        """Get reviews within a time range."""
        data = self._read_store()
        reviews = []

        for r in data.get("reviews", []):
            record = ReviewRecord.from_dict(r)
            created = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))

            if since and created < since:
                continue
            if until and created > until:
                continue

            reviews.append(record)

        return reviews

    def cleanup_old_records(self, days: int) -> int:
        """Remove records older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        data = self._read_store()

        original_count = len(data.get("reviews", []))
        data["reviews"] = [
            r for r in data.get("reviews", [])
            if datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) >= cutoff
        ]

        removed = original_count - len(data["reviews"])
        if removed > 0:
            self._write_store(data)

        return removed


# ============================================================================
# Metrics Dashboard
# ============================================================================

class MetricsDashboard:
    """
    Review metrics dashboard for analytics.

    Provides aggregated metrics, trends, and insights from code reviews.

    Example:
        dashboard = MetricsDashboard()

        # Record a review
        dashboard.record_review(
            review_id="abc123",
            files=["file.py"],
            issue_count=5,
            blocking_count=1,
            decision="request_changes",
            duration_ms=1500,
            quality_score=75.0,
        )

        # Generate dashboard
        data = await dashboard.generate(period=TimePeriod.WEEK)
        print(data.to_markdown())
    """

    BUS_TOPICS = {
        "dashboard": "review.metrics.dashboard",
        "trend": "review.metrics.trend",
        "record": "review.metrics.record",
    }

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        bus_path: Optional[Path] = None,
        store_path: Optional[Path] = None,
    ):
        """
        Initialize the metrics dashboard.

        Args:
            config: Dashboard configuration
            bus_path: Path to event bus file
            store_path: Path to review data store
        """
        self.config = config or DashboardConfig()
        self.bus_path = bus_path or self._get_bus_path()
        self.store = ReviewDataStore(store_path or self._get_store_path())
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_store_path(self) -> Path:
        """Get path to review data store."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        data_dir = pluribus_root / ".pluribus" / "review" / "data"
        return data_dir / "review_metrics.json"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "metrics") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "metrics-dashboard",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def record_review(
        self,
        review_id: str,
        files: List[str],
        issue_count: int,
        blocking_count: int,
        decision: str,
        duration_ms: float,
        quality_score: float,
        reviewer_id: Optional[str] = None,
        categories: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Record a completed review.

        Args:
            review_id: Unique review ID
            files: Files reviewed
            issue_count: Total issues found
            blocking_count: Blocking issues found
            decision: Review decision (approve, request_changes, comment)
            duration_ms: Review duration in milliseconds
            quality_score: Quality score (0-100)
            reviewer_id: Optional reviewer identifier
            categories: Issue counts by category
        """
        record = ReviewRecord(
            review_id=review_id,
            files=files,
            issue_count=issue_count,
            blocking_count=blocking_count,
            decision=decision,
            duration_ms=duration_ms,
            quality_score=quality_score,
            reviewer_id=reviewer_id,
            created_at=datetime.now(timezone.utc).isoformat() + "Z",
            categories=categories or {},
        )

        self.store.add_review(record)

        self._emit_event(self.BUS_TOPICS["record"], {
            "review_id": review_id,
            "issue_count": issue_count,
            "decision": decision,
        })

    async def generate(
        self,
        period: Optional[TimePeriod] = None,
    ) -> DashboardData:
        """
        Generate dashboard data.

        Args:
            period: Time period to aggregate

        Returns:
            DashboardData with all metrics

        Emits:
            review.metrics.dashboard
        """
        period = period or self.config.aggregation_period
        dashboard_id = str(uuid.uuid4())[:8]

        self._emit_event(self.BUS_TOPICS["dashboard"], {
            "dashboard_id": dashboard_id,
            "period": period.value,
            "status": "generating",
        })

        # Get time range
        now = datetime.now(timezone.utc)
        since = self._get_period_start(now, period)
        previous_since = self._get_period_start(since, period)

        # Fetch reviews
        current_reviews = self.store.get_reviews(since=since, until=now)
        previous_reviews = self.store.get_reviews(since=previous_since, until=since)

        # Calculate summaries
        summaries = self._calculate_summaries(current_reviews)

        # Calculate trends
        trends = self._calculate_trends(current_reviews, previous_reviews)

        # Find hotspots
        hotspots = []
        if self.config.include_file_hotspots:
            hotspots = self._find_hotspots(current_reviews)

        # Calculate reviewer stats
        reviewer_stats = []
        if self.config.include_reviewer_stats:
            reviewer_stats = self._calculate_reviewer_stats(current_reviews)

        # Calculate issue distribution
        issue_distribution = self._calculate_issue_distribution(current_reviews)

        # Calculate quality by path
        quality_by_path = self._calculate_quality_by_path(current_reviews)

        dashboard = DashboardData(
            dashboard_id=dashboard_id,
            generated_at=datetime.now(timezone.utc).isoformat() + "Z",
            period=period,
            summaries=summaries,
            trends=trends,
            hotspots=hotspots,
            reviewer_stats=reviewer_stats,
            issue_distribution=issue_distribution,
            quality_by_path=quality_by_path,
        )

        self._emit_event(self.BUS_TOPICS["dashboard"], {
            "dashboard_id": dashboard_id,
            "period": period.value,
            "review_count": len(current_reviews),
            "status": "completed",
        })

        return dashboard

    def _get_period_start(self, end: datetime, period: TimePeriod) -> datetime:
        """Get the start of a time period."""
        if period == TimePeriod.HOUR:
            return end - timedelta(hours=1)
        elif period == TimePeriod.DAY:
            return end - timedelta(days=1)
        elif period == TimePeriod.WEEK:
            return end - timedelta(weeks=1)
        elif period == TimePeriod.MONTH:
            return end - timedelta(days=30)
        elif period == TimePeriod.QUARTER:
            return end - timedelta(days=90)
        elif period == TimePeriod.YEAR:
            return end - timedelta(days=365)
        return end - timedelta(days=7)

    def _calculate_summaries(self, reviews: List[ReviewRecord]) -> List[MetricSummary]:
        """Calculate metric summaries."""
        summaries = []

        if not reviews:
            return summaries

        # Review count
        summaries.append(MetricSummary(
            metric_type=MetricType.REVIEW_COUNT,
            total=len(reviews),
            average=len(reviews),
            minimum=0,
            maximum=len(reviews),
            median=len(reviews),
            std_dev=0,
            count=1,
        ))

        # Issue count
        issue_counts = [r.issue_count for r in reviews]
        if issue_counts:
            summaries.append(MetricSummary(
                metric_type=MetricType.ISSUE_COUNT,
                total=sum(issue_counts),
                average=statistics.mean(issue_counts),
                minimum=min(issue_counts),
                maximum=max(issue_counts),
                median=statistics.median(issue_counts),
                std_dev=statistics.stdev(issue_counts) if len(issue_counts) > 1 else 0,
                count=len(issue_counts),
            ))

        # Quality score
        quality_scores = [r.quality_score for r in reviews if r.quality_score > 0]
        if quality_scores:
            summaries.append(MetricSummary(
                metric_type=MetricType.QUALITY_SCORE,
                total=sum(quality_scores),
                average=statistics.mean(quality_scores),
                minimum=min(quality_scores),
                maximum=max(quality_scores),
                median=statistics.median(quality_scores),
                std_dev=statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                count=len(quality_scores),
            ))

        # Review time
        durations = [r.duration_ms / 1000 / 60 for r in reviews]  # Convert to minutes
        if durations:
            summaries.append(MetricSummary(
                metric_type=MetricType.AVG_REVIEW_TIME,
                total=sum(durations),
                average=statistics.mean(durations),
                minimum=min(durations),
                maximum=max(durations),
                median=statistics.median(durations),
                std_dev=statistics.stdev(durations) if len(durations) > 1 else 0,
                count=len(durations),
            ))

        return summaries

    def _calculate_trends(
        self,
        current: List[ReviewRecord],
        previous: List[ReviewRecord],
    ) -> List[MetricTrend]:
        """Calculate metric trends."""
        trends = []

        def calc_trend(current_val: float, previous_val: float, metric: MetricType) -> MetricTrend:
            if previous_val == 0:
                change = 100.0 if current_val > 0 else 0.0
            else:
                change = ((current_val - previous_val) / previous_val) * 100

            if abs(change) < 5:
                direction = TrendDirection.STABLE
            elif change > 0:
                direction = TrendDirection.UP
            else:
                direction = TrendDirection.DOWN

            return MetricTrend(
                metric_type=metric,
                current_value=current_val,
                previous_value=previous_val,
                change_percent=change,
                direction=direction,
            )

        # Review count trend
        trends.append(calc_trend(len(current), len(previous), MetricType.REVIEW_COUNT))

        # Issue count trend
        current_issues = sum(r.issue_count for r in current)
        previous_issues = sum(r.issue_count for r in previous)
        trends.append(calc_trend(current_issues, previous_issues, MetricType.ISSUE_COUNT))

        # Quality score trend
        current_quality = statistics.mean([r.quality_score for r in current]) if current else 0
        previous_quality = statistics.mean([r.quality_score for r in previous]) if previous else 0
        trends.append(calc_trend(current_quality, previous_quality, MetricType.QUALITY_SCORE))

        # Blocking rate trend
        current_blocking = sum(1 for r in current if r.blocking_count > 0) / len(current) * 100 if current else 0
        previous_blocking = sum(1 for r in previous if r.blocking_count > 0) / len(previous) * 100 if previous else 0
        trends.append(calc_trend(current_blocking, previous_blocking, MetricType.BLOCKING_RATE))

        return trends

    def _find_hotspots(self, reviews: List[ReviewRecord]) -> List[FileHotspot]:
        """Find code hotspots."""
        file_stats: Dict[str, Dict[str, Any]] = {}

        for review in reviews:
            for file_path in review.files:
                if file_path not in file_stats:
                    file_stats[file_path] = {
                        "review_count": 0,
                        "issue_count": 0,
                        "last_reviewed": review.created_at,
                    }
                file_stats[file_path]["review_count"] += 1
                file_stats[file_path]["issue_count"] += review.issue_count // max(1, len(review.files))
                if review.created_at > file_stats[file_path]["last_reviewed"]:
                    file_stats[file_path]["last_reviewed"] = review.created_at

        hotspots = []
        for file_path, stats in file_stats.items():
            if stats["review_count"] >= self.config.hotspot_threshold:
                hotspots.append(FileHotspot(
                    file_path=file_path,
                    review_count=stats["review_count"],
                    issue_count=stats["issue_count"],
                    avg_issues_per_review=stats["issue_count"] / stats["review_count"],
                    last_reviewed=stats["last_reviewed"],
                    change_frequency=stats["review_count"] / 7,  # Per day over week
                ))

        return sorted(hotspots, key=lambda h: -h.avg_issues_per_review)[:self.config.max_hotspots]

    def _calculate_reviewer_stats(self, reviews: List[ReviewRecord]) -> List[ReviewerStats]:
        """Calculate per-reviewer statistics."""
        reviewer_data: Dict[str, Dict[str, Any]] = {}

        for review in reviews:
            reviewer = review.reviewer_id or "unknown"
            if reviewer not in reviewer_data:
                reviewer_data[reviewer] = {
                    "reviews": 0,
                    "issues": 0,
                    "duration_sum": 0,
                    "approvals": 0,
                    "blocks": 0,
                }
            reviewer_data[reviewer]["reviews"] += 1
            reviewer_data[reviewer]["issues"] += review.issue_count
            reviewer_data[reviewer]["duration_sum"] += review.duration_ms
            if review.decision == "approve":
                reviewer_data[reviewer]["approvals"] += 1
            if review.blocking_count > 0:
                reviewer_data[reviewer]["blocks"] += 1

        stats = []
        for reviewer, data in reviewer_data.items():
            if data["reviews"] > 0:
                stats.append(ReviewerStats(
                    reviewer_id=reviewer,
                    reviews_completed=data["reviews"],
                    issues_found=data["issues"],
                    avg_review_time_min=data["duration_sum"] / data["reviews"] / 1000 / 60,
                    approval_rate=data["approvals"] / data["reviews"] * 100,
                    blocking_rate=data["blocks"] / data["reviews"] * 100,
                ))

        return sorted(stats, key=lambda s: -s.reviews_completed)

    def _calculate_issue_distribution(self, reviews: List[ReviewRecord]) -> Dict[str, int]:
        """Calculate issue distribution by category."""
        distribution: Dict[str, int] = {}

        for review in reviews:
            for category, count in review.categories.items():
                distribution[category] = distribution.get(category, 0) + count

        return dict(sorted(distribution.items(), key=lambda x: -x[1]))

    def _calculate_quality_by_path(self, reviews: List[ReviewRecord]) -> Dict[str, float]:
        """Calculate average quality score by path prefix."""
        path_scores: Dict[str, List[float]] = {}

        for review in reviews:
            if review.quality_score <= 0:
                continue
            for file_path in review.files:
                # Get top-level directory
                parts = file_path.split("/")
                prefix = parts[0] if len(parts) > 1 else "root"
                if prefix not in path_scores:
                    path_scores[prefix] = []
                path_scores[prefix].append(review.quality_score)

        return {
            path: statistics.mean(scores)
            for path, scores in path_scores.items()
            if scores
        }

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "metrics-dashboard",
            "healthy": True,
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Metrics Dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Metrics Dashboard (Step 171)")
    parser.add_argument("--period", choices=["hour", "day", "week", "month"],
                        default="week", help="Aggregation period")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    parser.add_argument("--record", action="store_true", help="Record a demo review")
    parser.add_argument("--cleanup", type=int, help="Cleanup records older than N days")

    args = parser.parse_args()

    dashboard = MetricsDashboard()

    if args.record:
        # Record a demo review
        dashboard.record_review(
            review_id=str(uuid.uuid4())[:8],
            files=["example/file.py"],
            issue_count=5,
            blocking_count=1,
            decision="request_changes",
            duration_ms=30000,
            quality_score=75.0,
            categories={"security": 2, "style": 3},
        )
        print("Demo review recorded.")
        return 0

    if args.cleanup:
        removed = dashboard.store.cleanup_old_records(args.cleanup)
        print(f"Removed {removed} old records.")
        return 0

    # Generate dashboard
    period = TimePeriod[args.period.upper()]
    data = asyncio.run(dashboard.generate(period=period))

    if args.json:
        print(json.dumps(data.to_dict(), indent=2))
    elif args.markdown:
        print(data.to_markdown())
    else:
        print(f"Dashboard ID: {data.dashboard_id}")
        print(f"Period: {data.period.value}")
        print(f"Generated: {data.generated_at}")
        print(f"\nSummaries: {len(data.summaries)}")
        for s in data.summaries:
            print(f"  {s.metric_type.value}: avg={s.average:.2f}, total={s.total:.1f}")
        print(f"\nTrends: {len(data.trends)}")
        for t in data.trends:
            print(f"  {t.metric_type.value}: {t.direction.value} ({t.change_percent:+.1f}%)")
        print(f"\nHotspots: {len(data.hotspots)}")
        print(f"Reviewers: {len(data.reviewer_stats)}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
