#!/usr/bin/env python3
"""
Technical Debt Tracker (Step 174)

Tracks and manages technical debt identification and remediation.

PBTSO Phase: OBSERVE, DISTILL
Bus Topics: review.debt.track, review.debt.report

Tracks:
- Code debt (complexity, duplication)
- Design debt (architecture violations)
- Test debt (low coverage)
- Documentation debt
- Dependency debt

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# Types
# ============================================================================

class DebtCategory(Enum):
    """Categories of technical debt."""
    CODE = "code"           # Complexity, smells, style
    DESIGN = "design"       # Architecture violations
    TEST = "test"           # Coverage, missing tests
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"  # Outdated, vulnerable deps
    BUILD = "build"         # CI/CD issues
    SECURITY = "security"   # Security vulnerabilities


class DebtPriority(Enum):
    """Priority levels for debt items."""
    CRITICAL = "critical"   # Must fix immediately
    HIGH = "high"           # Fix within sprint
    MEDIUM = "medium"       # Fix within quarter
    LOW = "low"             # Fix when possible
    ACCEPTED = "accepted"   # Acknowledged but not planned


class DebtStatus(Enum):
    """Status of debt items."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"
    WONT_FIX = "wont_fix"


@dataclass
class DebtConfig:
    """Configuration for debt tracking."""
    hours_per_complexity_point: float = 0.5
    hours_per_coverage_percent: float = 2.0
    hours_per_smell: float = 1.0
    hours_per_security_issue: float = 4.0
    hours_per_doc_issue: float = 0.5
    interest_rate_percent: float = 10.0  # Annual interest rate
    data_retention_days: int = 365

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DebtItem:
    """
    A single technical debt item.

    Attributes:
        debt_id: Unique identifier
        category: Type of debt
        priority: Priority level
        status: Current status
        title: Short title
        description: Detailed description
        file: Related file (if applicable)
        line: Related line (if applicable)
        estimated_hours: Estimated hours to fix
        actual_hours: Actual hours spent (if resolved)
        interest_hours: Accumulated interest
        created_at: When debt was identified
        updated_at: Last update time
        resolved_at: When resolved (if applicable)
        tags: Associated tags
        assignee: Assigned person/team
    """
    debt_id: str
    category: DebtCategory
    priority: DebtPriority
    status: DebtStatus
    title: str
    description: str
    file: Optional[str] = None
    line: Optional[int] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    interest_hours: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    resolved_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    assignee: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at

    @property
    def total_hours(self) -> float:
        """Total hours including interest."""
        return self.estimated_hours + self.interest_hours

    @property
    def age_days(self) -> int:
        """Days since creation."""
        created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - created).days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "debt_id": self.debt_id,
            "category": self.category.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "file": self.file,
            "line": self.line,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "interest_hours": round(self.interest_hours, 2),
            "total_hours": round(self.total_hours, 2),
            "age_days": self.age_days,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "tags": self.tags,
            "assignee": self.assignee,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebtItem":
        """Create from dictionary."""
        data = data.copy()
        data["category"] = DebtCategory(data["category"])
        data["priority"] = DebtPriority(data["priority"])
        data["status"] = DebtStatus(data["status"])
        # Remove computed fields
        data.pop("total_hours", None)
        data.pop("age_days", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DebtSummary:
    """Summary of debt by category."""
    category: DebtCategory
    total_items: int
    open_items: int
    total_hours: float
    interest_hours: float
    avg_age_days: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "total_items": self.total_items,
            "open_items": self.open_items,
            "total_hours": round(self.total_hours, 1),
            "interest_hours": round(self.interest_hours, 1),
            "avg_age_days": round(self.avg_age_days, 1),
        }


@dataclass
class DebtReport:
    """
    Comprehensive debt report.

    Attributes:
        report_id: Unique report ID
        generated_at: Report generation time
        total_items: Total debt items
        open_items: Open debt items
        total_hours: Total estimated hours
        interest_hours: Total accumulated interest
        summaries: Summaries by category
        top_items: Highest priority items
        trends: Trend data
    """
    report_id: str
    generated_at: str
    total_items: int
    open_items: int
    total_hours: float
    interest_hours: float
    summaries: List[DebtSummary]
    top_items: List[DebtItem]
    trends: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "total_items": self.total_items,
            "open_items": self.open_items,
            "total_hours": round(self.total_hours, 1),
            "interest_hours": round(self.interest_hours, 1),
            "summaries": [s.to_dict() for s in self.summaries],
            "top_items": [i.to_dict() for i in self.top_items],
            "trends": self.trends,
        }

    def to_markdown(self) -> str:
        """Convert to markdown report."""
        lines = [
            "# Technical Debt Report",
            "",
            f"**Generated:** {self.generated_at}",
            "",
            "## Summary",
            "",
            f"- **Total Debt Items:** {self.total_items}",
            f"- **Open Items:** {self.open_items}",
            f"- **Total Estimated Hours:** {self.total_hours:.1f}",
            f"- **Accumulated Interest:** {self.interest_hours:.1f} hours",
            "",
            "## By Category",
            "",
            "| Category | Items | Open | Hours | Interest | Avg Age |",
            "|----------|------:|-----:|------:|---------:|--------:|",
        ]

        for s in sorted(self.summaries, key=lambda x: -x.total_hours):
            lines.append(
                f"| {s.category.value.title()} | {s.total_items} | {s.open_items} | "
                f"{s.total_hours:.1f} | {s.interest_hours:.1f} | {s.avg_age_days:.0f}d |"
            )

        if self.top_items:
            lines.extend([
                "",
                "## Top Priority Items",
                "",
            ])
            for item in self.top_items[:10]:
                status_icon = "[X]" if item.status == DebtStatus.OPEN else "[-]"
                lines.append(
                    f"{status_icon} **{item.title}** ({item.priority.value}) - "
                    f"{item.estimated_hours:.1f}h + {item.interest_hours:.1f}h interest"
                )
                if item.file:
                    lines.append(f"   `{item.file}:{item.line or ''}`")
                lines.append("")

        lines.extend([
            "",
            "_Generated by Technical Debt Tracker_",
        ])

        return "\n".join(lines)


# ============================================================================
# Debt Store
# ============================================================================

class DebtStore:
    """Persistent store for debt items."""

    def __init__(self, store_path: Path):
        """
        Initialize the debt store.

        Args:
            store_path: Path to the data file
        """
        self.store_path = store_path
        self._ensure_store()

    def _ensure_store(self) -> None:
        """Ensure store file exists."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._write_store({"items": [], "version": 1})

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

    def add(self, item: DebtItem) -> None:
        """Add a debt item."""
        data = self._read_store()
        data["items"].append(item.to_dict())
        self._write_store(data)

    def update(self, item: DebtItem) -> bool:
        """Update an existing debt item."""
        data = self._read_store()
        for i, existing in enumerate(data["items"]):
            if existing["debt_id"] == item.debt_id:
                item.updated_at = datetime.now(timezone.utc).isoformat() + "Z"
                data["items"][i] = item.to_dict()
                self._write_store(data)
                return True
        return False

    def get(self, debt_id: str) -> Optional[DebtItem]:
        """Get a debt item by ID."""
        data = self._read_store()
        for item_data in data["items"]:
            if item_data["debt_id"] == debt_id:
                return DebtItem.from_dict(item_data)
        return None

    def get_all(
        self,
        category: Optional[DebtCategory] = None,
        status: Optional[DebtStatus] = None,
        priority: Optional[DebtPriority] = None,
    ) -> List[DebtItem]:
        """Get debt items with optional filters."""
        data = self._read_store()
        items = []

        for item_data in data["items"]:
            if category and item_data["category"] != category.value:
                continue
            if status and item_data["status"] != status.value:
                continue
            if priority and item_data["priority"] != priority.value:
                continue
            items.append(DebtItem.from_dict(item_data))

        return items

    def delete(self, debt_id: str) -> bool:
        """Delete a debt item."""
        data = self._read_store()
        original_len = len(data["items"])
        data["items"] = [i for i in data["items"] if i["debt_id"] != debt_id]
        if len(data["items"]) < original_len:
            self._write_store(data)
            return True
        return False


# ============================================================================
# Technical Debt Tracker
# ============================================================================

class TechnicalDebtTracker:
    """
    Tracks and manages technical debt.

    Example:
        tracker = TechnicalDebtTracker()

        # Add debt item
        item = tracker.create_debt(
            category=DebtCategory.CODE,
            priority=DebtPriority.HIGH,
            title="Complex function needs refactoring",
            description="Function has CC of 25",
            file="src/main.py",
            line=100,
            estimated_hours=4.0,
        )

        # Generate report
        report = await tracker.generate_report()
        print(report.to_markdown())
    """

    BUS_TOPICS = {
        "track": "review.debt.track",
        "report": "review.debt.report",
        "update": "review.debt.update",
    }

    def __init__(
        self,
        config: Optional[DebtConfig] = None,
        bus_path: Optional[Path] = None,
        store_path: Optional[Path] = None,
    ):
        """
        Initialize the debt tracker.

        Args:
            config: Tracker configuration
            bus_path: Path to event bus file
            store_path: Path to debt store
        """
        self.config = config or DebtConfig()
        self.bus_path = bus_path or self._get_bus_path()
        self.store = DebtStore(store_path or self._get_store_path())

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_store_path(self) -> Path:
        """Get path to debt store."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        data_dir = pluribus_root / ".pluribus" / "review" / "data"
        return data_dir / "technical_debt.json"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "debt") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "debt-tracker",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def create_debt(
        self,
        category: DebtCategory,
        priority: DebtPriority,
        title: str,
        description: str,
        file: Optional[str] = None,
        line: Optional[int] = None,
        estimated_hours: Optional[float] = None,
        tags: Optional[List[str]] = None,
        assignee: Optional[str] = None,
    ) -> DebtItem:
        """
        Create a new debt item.

        Args:
            category: Debt category
            priority: Priority level
            title: Short title
            description: Detailed description
            file: Related file
            line: Related line
            estimated_hours: Estimated hours to fix
            tags: Associated tags
            assignee: Assigned person

        Returns:
            Created DebtItem

        Emits:
            review.debt.track
        """
        # Auto-estimate if not provided
        if estimated_hours is None:
            estimated_hours = self._estimate_hours(category)

        debt_id = str(uuid.uuid4())[:8]
        item = DebtItem(
            debt_id=debt_id,
            category=category,
            priority=priority,
            status=DebtStatus.OPEN,
            title=title,
            description=description,
            file=file,
            line=line,
            estimated_hours=estimated_hours,
            tags=tags or [],
            assignee=assignee,
        )

        self.store.add(item)

        self._emit_event(self.BUS_TOPICS["track"], {
            "action": "created",
            "debt_id": debt_id,
            "category": category.value,
            "priority": priority.value,
            "estimated_hours": estimated_hours,
        })

        return item

    def _estimate_hours(self, category: DebtCategory) -> float:
        """Estimate hours based on category."""
        estimates = {
            DebtCategory.CODE: self.config.hours_per_smell,
            DebtCategory.DESIGN: self.config.hours_per_smell * 2,
            DebtCategory.TEST: self.config.hours_per_coverage_percent,
            DebtCategory.DOCUMENTATION: self.config.hours_per_doc_issue,
            DebtCategory.DEPENDENCY: self.config.hours_per_smell,
            DebtCategory.BUILD: self.config.hours_per_smell,
            DebtCategory.SECURITY: self.config.hours_per_security_issue,
        }
        return estimates.get(category, 1.0)

    def update_status(
        self,
        debt_id: str,
        status: DebtStatus,
        actual_hours: Optional[float] = None,
    ) -> Optional[DebtItem]:
        """
        Update debt item status.

        Args:
            debt_id: Debt item ID
            status: New status
            actual_hours: Actual hours spent (for resolved items)

        Returns:
            Updated item or None if not found

        Emits:
            review.debt.update
        """
        item = self.store.get(debt_id)
        if not item:
            return None

        old_status = item.status
        item.status = status

        if actual_hours is not None:
            item.actual_hours = actual_hours

        if status == DebtStatus.RESOLVED:
            item.resolved_at = datetime.now(timezone.utc).isoformat() + "Z"

        self.store.update(item)

        self._emit_event(self.BUS_TOPICS["update"], {
            "debt_id": debt_id,
            "old_status": old_status.value,
            "new_status": status.value,
            "actual_hours": actual_hours,
        })

        return item

    def calculate_interest(self) -> int:
        """
        Calculate and apply interest to open debt items.

        Interest accrues based on age and configuration.

        Returns:
            Number of items updated
        """
        items = self.store.get_all(status=DebtStatus.OPEN)
        updated = 0

        daily_rate = self.config.interest_rate_percent / 365 / 100

        for item in items:
            # Interest accrues daily
            interest = item.estimated_hours * daily_rate * item.age_days
            if interest != item.interest_hours:
                item.interest_hours = interest
                self.store.update(item)
                updated += 1

        return updated

    async def generate_report(self) -> DebtReport:
        """
        Generate a comprehensive debt report.

        Returns:
            DebtReport with summaries and trends

        Emits:
            review.debt.report
        """
        report_id = str(uuid.uuid4())[:8]

        self._emit_event(self.BUS_TOPICS["report"], {
            "report_id": report_id,
            "status": "generating",
        })

        # Calculate interest first
        self.calculate_interest()

        # Get all items
        all_items = self.store.get_all()
        open_items = [i for i in all_items if i.status == DebtStatus.OPEN]

        # Calculate summaries by category
        summaries: List[DebtSummary] = []
        for category in DebtCategory:
            cat_items = [i for i in all_items if i.category == category]
            cat_open = [i for i in cat_items if i.status == DebtStatus.OPEN]

            if cat_items:
                summaries.append(DebtSummary(
                    category=category,
                    total_items=len(cat_items),
                    open_items=len(cat_open),
                    total_hours=sum(i.total_hours for i in cat_items),
                    interest_hours=sum(i.interest_hours for i in cat_items),
                    avg_age_days=sum(i.age_days for i in cat_items) / len(cat_items),
                ))

        # Get top priority items
        priority_order = {
            DebtPriority.CRITICAL: 4,
            DebtPriority.HIGH: 3,
            DebtPriority.MEDIUM: 2,
            DebtPriority.LOW: 1,
            DebtPriority.ACCEPTED: 0,
        }
        top_items = sorted(
            open_items,
            key=lambda i: (-priority_order[i.priority], -i.total_hours),
        )[:20]

        report = DebtReport(
            report_id=report_id,
            generated_at=datetime.now(timezone.utc).isoformat() + "Z",
            total_items=len(all_items),
            open_items=len(open_items),
            total_hours=sum(i.total_hours for i in all_items),
            interest_hours=sum(i.interest_hours for i in all_items),
            summaries=summaries,
            top_items=top_items,
        )

        self._emit_event(self.BUS_TOPICS["report"], {
            "report_id": report_id,
            "total_items": report.total_items,
            "total_hours": report.total_hours,
            "status": "completed",
        })

        return report

    def import_from_review(
        self,
        review_id: str,
        findings: List[Dict[str, Any]],
    ) -> List[DebtItem]:
        """
        Import debt items from review findings.

        Args:
            review_id: Source review ID
            findings: List of finding dictionaries

        Returns:
            List of created debt items
        """
        items = []

        category_map = {
            "security": DebtCategory.SECURITY,
            "architecture": DebtCategory.DESIGN,
            "documentation": DebtCategory.DOCUMENTATION,
            "maintainability": DebtCategory.CODE,
            "style": DebtCategory.CODE,
            "test": DebtCategory.TEST,
        }

        severity_map = {
            "blocker": DebtPriority.CRITICAL,
            "critical": DebtPriority.CRITICAL,
            "major": DebtPriority.HIGH,
            "minor": DebtPriority.MEDIUM,
            "suggestion": DebtPriority.LOW,
        }

        for finding in findings:
            category = category_map.get(
                finding.get("category", "").lower(),
                DebtCategory.CODE,
            )
            priority = severity_map.get(
                finding.get("severity", "").lower(),
                DebtPriority.MEDIUM,
            )

            item = self.create_debt(
                category=category,
                priority=priority,
                title=finding.get("title", "Unknown issue"),
                description=finding.get("description", ""),
                file=finding.get("file"),
                line=finding.get("line"),
                tags=[f"review:{review_id}"],
            )
            items.append(item)

        return items


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Technical Debt Tracker."""
    import argparse

    parser = argparse.ArgumentParser(description="Technical Debt Tracker (Step 174)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add debt item")
    add_parser.add_argument("--category", required=True,
                            choices=[c.value for c in DebtCategory])
    add_parser.add_argument("--priority", required=True,
                            choices=[p.value for p in DebtPriority])
    add_parser.add_argument("--title", required=True)
    add_parser.add_argument("--description", default="")
    add_parser.add_argument("--file")
    add_parser.add_argument("--hours", type=float)

    # List command
    list_parser = subparsers.add_parser("list", help="List debt items")
    list_parser.add_argument("--category", choices=[c.value for c in DebtCategory])
    list_parser.add_argument("--status", choices=[s.value for s in DebtStatus])

    # Report command
    subparsers.add_parser("report", help="Generate report")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update status")
    update_parser.add_argument("debt_id")
    update_parser.add_argument("status", choices=[s.value for s in DebtStatus])
    update_parser.add_argument("--hours", type=float)

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    tracker = TechnicalDebtTracker()

    if args.command == "add":
        item = tracker.create_debt(
            category=DebtCategory(args.category),
            priority=DebtPriority(args.priority),
            title=args.title,
            description=args.description,
            file=args.file,
            estimated_hours=args.hours,
        )
        if args.json:
            print(json.dumps(item.to_dict(), indent=2))
        else:
            print(f"Created debt item: {item.debt_id}")
            print(f"  Title: {item.title}")
            print(f"  Estimated: {item.estimated_hours:.1f}h")

    elif args.command == "list":
        category = DebtCategory(args.category) if args.category else None
        status = DebtStatus(args.status) if args.status else None
        items = tracker.store.get_all(category=category, status=status)

        if args.json:
            print(json.dumps([i.to_dict() for i in items], indent=2))
        else:
            print(f"Found {len(items)} debt items:")
            for item in items:
                print(f"  [{item.debt_id}] {item.title}")
                print(f"    {item.category.value} | {item.priority.value} | {item.status.value}")
                print(f"    {item.estimated_hours:.1f}h + {item.interest_hours:.1f}h interest")

    elif args.command == "report":
        report = asyncio.run(tracker.generate_report())
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.to_markdown())

    elif args.command == "update":
        item = tracker.update_status(
            args.debt_id,
            DebtStatus(args.status),
            actual_hours=args.hours,
        )
        if item:
            print(f"Updated {item.debt_id} to {item.status.value}")
        else:
            print(f"Item {args.debt_id} not found")
            return 1

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
