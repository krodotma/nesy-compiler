#!/usr/bin/env python3
"""
Budget Monitor - Step 265

Tracks cost and resource budget usage.

PBTSO Phase: ITERATE

Bus Topics:
- monitor.budget.track (emitted)
- monitor.budget.alert (emitted)
- monitor.cost.track (emitted)

Protocol: DKIN v30, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class BudgetType(Enum):
    """Types of budgets."""
    COST = "cost"
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALLS = "api_calls"
    TOKENS = "tokens"
    CUSTOM = "custom"


class BudgetPeriod(Enum):
    """Budget periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class AlertLevel(Enum):
    """Budget alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Budget:
    """Budget definition.

    Attributes:
        name: Budget name
        budget_type: Type of budget
        limit: Budget limit
        period: Budget period
        unit: Unit of measurement
        department: Department or team
        description: Budget description
        alerts: Alert thresholds (percent)
        created_at: Creation timestamp
    """
    name: str
    budget_type: BudgetType
    limit: float
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    unit: str = "USD"
    department: str = ""
    description: str = ""
    alerts: List[float] = field(default_factory=lambda: [50.0, 75.0, 90.0])
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "budget_type": self.budget_type.value,
            "limit": self.limit,
            "period": self.period.value,
            "unit": self.unit,
            "department": self.department,
            "description": self.description,
            "alerts": self.alerts,
            "created_at": self.created_at,
        }


@dataclass
class CostAllocation:
    """Cost allocation entry.

    Attributes:
        budget_name: Associated budget
        resource: Resource name/ID
        amount: Cost amount
        category: Cost category
        tags: Cost tags
        timestamp: Allocation timestamp
    """
    budget_name: str
    resource: str
    amount: float
    category: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BudgetUsage:
    """Budget usage summary.

    Attributes:
        budget_name: Budget name
        current_usage: Current usage amount
        limit: Budget limit
        usage_percent: Usage as percentage
        remaining: Remaining budget
        burn_rate: Current burn rate
        projected_usage: Projected end-of-period usage
        period_start: Period start timestamp
        period_end: Period end timestamp
        timestamp: Calculation timestamp
    """
    budget_name: str
    current_usage: float
    limit: float
    usage_percent: float
    remaining: float
    burn_rate: float
    projected_usage: float
    period_start: float
    period_end: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def will_exceed(self) -> bool:
        """Check if projected to exceed budget."""
        return self.projected_usage > self.limit

    @property
    def days_remaining(self) -> float:
        """Days remaining in period."""
        return max(0, (self.period_end - time.time()) / 86400)


@dataclass
class BudgetAlert:
    """Budget alert.

    Attributes:
        budget_name: Budget name
        level: Alert level
        message: Alert message
        usage_percent: Current usage percentage
        threshold: Threshold that triggered alert
        timestamp: Alert timestamp
        alert_id: Unique alert ID
    """
    budget_name: str
    level: AlertLevel
    message: str
    usage_percent: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "budget_name": self.budget_name,
            "level": self.level.value,
            "message": self.message,
            "usage_percent": self.usage_percent,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


@dataclass
class CostBreakdown:
    """Cost breakdown by category.

    Attributes:
        budget_name: Budget name
        by_category: Costs by category
        by_resource: Costs by resource
        by_tag: Costs by tag
        total: Total cost
        period: Breakdown period
    """
    budget_name: str
    by_category: Dict[str, float] = field(default_factory=dict)
    by_resource: Dict[str, float] = field(default_factory=dict)
    by_tag: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total: float = 0.0
    period: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BudgetMonitor:
    """
    Monitor cost and resource budgets.

    The monitor:
    - Tracks budget definitions
    - Records cost allocations
    - Calculates usage and projections
    - Generates alerts on thresholds
    - Provides cost breakdowns

    Example:
        monitor = BudgetMonitor()
        monitor.register_budget(Budget(
            name="compute-budget",
            budget_type=BudgetType.COMPUTE,
            limit=10000.0,
            period=BudgetPeriod.MONTHLY,
        ))

        monitor.record_cost("compute-budget", CostAllocation(
            budget_name="compute-budget",
            resource="instance-001",
            amount=150.0,
            category="ec2",
        ))

        usage = monitor.get_usage("compute-budget")
        print(f"Usage: {usage.usage_percent}%")
    """

    BUS_TOPICS = {
        "track": "monitor.budget.track",
        "alert": "monitor.budget.alert",
        "cost": "monitor.cost.track",
    }

    def __init__(
        self,
        history_days: int = 90,
        bus_dir: Optional[str] = None,
    ):
        """Initialize budget monitor.

        Args:
            history_days: Days of cost history to retain
            bus_dir: Bus directory
        """
        self.history_days = history_days

        # Budget registry
        self._budgets: Dict[str, Budget] = {}
        self._costs: Dict[str, List[CostAllocation]] = {}
        self._triggered_alerts: Dict[str, set] = {}

        # State
        self._alert_callbacks: List[Callable[[BudgetAlert], None]] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def register_budget(self, budget: Budget) -> None:
        """Register a budget.

        Args:
            budget: Budget definition
        """
        self._budgets[budget.name] = budget
        if budget.name not in self._costs:
            self._costs[budget.name] = []
        if budget.name not in self._triggered_alerts:
            self._triggered_alerts[budget.name] = set()

        self._emit_bus_event(
            "monitor.budget.registered",
            budget.to_dict()
        )

    def unregister_budget(self, name: str) -> bool:
        """Unregister a budget.

        Args:
            name: Budget name

        Returns:
            True if unregistered
        """
        if name not in self._budgets:
            return False

        del self._budgets[name]
        if name in self._costs:
            del self._costs[name]
        if name in self._triggered_alerts:
            del self._triggered_alerts[name]

        return True

    def record_cost(self, budget_name: str, cost: CostAllocation) -> bool:
        """Record a cost allocation.

        Args:
            budget_name: Budget name
            cost: Cost allocation

        Returns:
            True if recorded
        """
        if budget_name not in self._budgets:
            return False

        self._costs[budget_name].append(cost)

        # Prune old history
        cutoff = time.time() - (self.history_days * 86400)
        self._costs[budget_name] = [
            c for c in self._costs[budget_name]
            if c.timestamp >= cutoff
        ]

        # Emit event
        self._emit_bus_event(
            self.BUS_TOPICS["cost"],
            cost.to_dict()
        )

        # Check alerts
        self._check_alerts(budget_name)

        return True

    def record_usage(
        self,
        budget_name: str,
        resource: str,
        amount: float,
        category: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Record usage/cost.

        Args:
            budget_name: Budget name
            resource: Resource identifier
            amount: Amount
            category: Category
            tags: Optional tags

        Returns:
            True if recorded
        """
        cost = CostAllocation(
            budget_name=budget_name,
            resource=resource,
            amount=amount,
            category=category,
            tags=tags or {},
        )
        return self.record_cost(budget_name, cost)

    def get_usage(self, budget_name: str) -> Optional[BudgetUsage]:
        """Get budget usage.

        Args:
            budget_name: Budget name

        Returns:
            Budget usage or None
        """
        if budget_name not in self._budgets:
            return None

        budget = self._budgets[budget_name]
        period_start, period_end = self._get_period_bounds(budget.period)

        # Calculate current usage
        period_costs = [
            c for c in self._costs.get(budget_name, [])
            if period_start <= c.timestamp < period_end
        ]

        current_usage = sum(c.amount for c in period_costs)
        usage_percent = 100.0 * current_usage / budget.limit if budget.limit > 0 else 0.0
        remaining = max(0, budget.limit - current_usage)

        # Calculate burn rate (daily)
        period_elapsed = (time.time() - period_start) / 86400
        burn_rate = current_usage / period_elapsed if period_elapsed > 0 else 0.0

        # Project end-of-period usage
        period_days = (period_end - period_start) / 86400
        projected_usage = burn_rate * period_days

        return BudgetUsage(
            budget_name=budget_name,
            current_usage=current_usage,
            limit=budget.limit,
            usage_percent=usage_percent,
            remaining=remaining,
            burn_rate=burn_rate,
            projected_usage=projected_usage,
            period_start=period_start,
            period_end=period_end,
        )

    def get_all_usage(self) -> Dict[str, BudgetUsage]:
        """Get usage for all budgets.

        Returns:
            Dictionary of budget usages
        """
        result = {}
        for name in self._budgets:
            usage = self.get_usage(name)
            if usage:
                result[name] = usage
        return result

    def get_breakdown(self, budget_name: str) -> Optional[CostBreakdown]:
        """Get cost breakdown for a budget.

        Args:
            budget_name: Budget name

        Returns:
            Cost breakdown or None
        """
        if budget_name not in self._budgets:
            return None

        budget = self._budgets[budget_name]
        period_start, period_end = self._get_period_bounds(budget.period)

        period_costs = [
            c for c in self._costs.get(budget_name, [])
            if period_start <= c.timestamp < period_end
        ]

        by_category: Dict[str, float] = {}
        by_resource: Dict[str, float] = {}
        by_tag: Dict[str, Dict[str, float]] = {}

        for cost in period_costs:
            # By category
            if cost.category:
                by_category[cost.category] = by_category.get(cost.category, 0) + cost.amount

            # By resource
            by_resource[cost.resource] = by_resource.get(cost.resource, 0) + cost.amount

            # By tag
            for tag_key, tag_value in cost.tags.items():
                if tag_key not in by_tag:
                    by_tag[tag_key] = {}
                by_tag[tag_key][tag_value] = by_tag[tag_key].get(tag_value, 0) + cost.amount

        total = sum(c.amount for c in period_costs)

        return CostBreakdown(
            budget_name=budget_name,
            by_category=by_category,
            by_resource=by_resource,
            by_tag=by_tag,
            total=total,
            period=budget.period.value,
        )

    def get_cost_history(
        self,
        budget_name: str,
        window_days: int = 30
    ) -> List[CostAllocation]:
        """Get cost history.

        Args:
            budget_name: Budget name
            window_days: Time window

        Returns:
            Cost history
        """
        if budget_name not in self._costs:
            return []

        cutoff = time.time() - (window_days * 86400)
        return [
            c for c in self._costs[budget_name]
            if c.timestamp >= cutoff
        ]

    def get_daily_costs(
        self,
        budget_name: str,
        window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get daily cost aggregation.

        Args:
            budget_name: Budget name
            window_days: Time window

        Returns:
            Daily costs
        """
        history = self.get_cost_history(budget_name, window_days)

        # Aggregate by day
        daily: Dict[str, float] = {}
        for cost in history:
            date = datetime.fromtimestamp(cost.timestamp).strftime("%Y-%m-%d")
            daily[date] = daily.get(date, 0) + cost.amount

        return [
            {"date": date, "amount": amount}
            for date, amount in sorted(daily.items())
        ]

    def forecast(
        self,
        budget_name: str,
        days_ahead: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Forecast future usage.

        Args:
            budget_name: Budget name
            days_ahead: Days to forecast

        Returns:
            Forecast or None
        """
        usage = self.get_usage(budget_name)
        if not usage:
            return None

        budget = self._budgets[budget_name]
        current_rate = usage.burn_rate

        # Simple linear forecast
        forecast_total = usage.current_usage + (current_rate * days_ahead)
        will_exceed = forecast_total > budget.limit
        days_to_exceed = None

        if current_rate > 0 and usage.remaining > 0:
            days_to_exceed = usage.remaining / current_rate

        return {
            "budget_name": budget_name,
            "current_usage": usage.current_usage,
            "burn_rate": current_rate,
            "forecast_days": days_ahead,
            "forecast_total": forecast_total,
            "limit": budget.limit,
            "will_exceed": will_exceed,
            "days_to_exceed": days_to_exceed,
        }

    def register_alert_callback(
        self,
        callback: Callable[[BudgetAlert], None]
    ) -> None:
        """Register alert callback.

        Args:
            callback: Callback function
        """
        self._alert_callbacks.append(callback)

    def reset_period(self, budget_name: str) -> bool:
        """Reset budget period (clears triggered alerts).

        Args:
            budget_name: Budget name

        Returns:
            True if reset
        """
        if budget_name not in self._budgets:
            return False

        self._triggered_alerts[budget_name] = set()
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get budget summary.

        Returns:
            Summary dictionary
        """
        total = len(self._budgets)
        over_budget = 0
        at_risk = 0
        total_limit = 0.0
        total_usage = 0.0

        for name in self._budgets:
            usage = self.get_usage(name)
            if usage:
                total_limit += usage.limit
                total_usage += usage.current_usage

                if usage.usage_percent >= 100:
                    over_budget += 1
                elif usage.usage_percent >= 75 or usage.will_exceed:
                    at_risk += 1

        return {
            "total_budgets": total,
            "over_budget": over_budget,
            "at_risk": at_risk,
            "healthy": total - over_budget - at_risk,
            "total_limit": total_limit,
            "total_usage": total_usage,
            "overall_percent": 100.0 * total_usage / total_limit if total_limit > 0 else 0.0,
        }

    def _get_period_bounds(self, period: BudgetPeriod) -> Tuple[float, float]:
        """Get period start and end timestamps.

        Args:
            period: Budget period

        Returns:
            (start, end) timestamps
        """
        now = datetime.now()

        if period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif period == BudgetPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        elif period == BudgetPeriod.QUARTERLY:
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            start = now.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_month = quarter_start_month + 3
            if end_month > 12:
                end = start.replace(year=now.year + 1, month=end_month - 12)
            else:
                end = start.replace(month=end_month)
        elif period == BudgetPeriod.ANNUAL:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(year=now.year + 1)
        else:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=30)

        return start.timestamp(), end.timestamp()

    def _check_alerts(self, budget_name: str) -> None:
        """Check and generate alerts.

        Args:
            budget_name: Budget name
        """
        usage = self.get_usage(budget_name)
        if not usage:
            return

        budget = self._budgets[budget_name]
        triggered = self._triggered_alerts.get(budget_name, set())

        for threshold in sorted(budget.alerts):
            if usage.usage_percent >= threshold and threshold not in triggered:
                # Determine alert level
                if threshold >= 90:
                    level = AlertLevel.CRITICAL
                elif threshold >= 75:
                    level = AlertLevel.WARNING
                else:
                    level = AlertLevel.INFO

                alert = BudgetAlert(
                    budget_name=budget_name,
                    level=level,
                    message=f"Budget {budget_name} has reached {usage.usage_percent:.1f}% of limit ({threshold}% threshold)",
                    usage_percent=usage.usage_percent,
                    threshold=threshold,
                )

                triggered.add(threshold)
                self._triggered_alerts[budget_name] = triggered

                self._emit_bus_event(
                    self.BUS_TOPICS["alert"],
                    alert.to_dict(),
                    level=level.value
                )

                for callback in self._alert_callbacks:
                    callback(alert)

        # Check for projection alerts
        if usage.will_exceed and "projected" not in triggered:
            alert = BudgetAlert(
                budget_name=budget_name,
                level=AlertLevel.WARNING,
                message=f"Budget {budget_name} projected to exceed limit: {usage.projected_usage:.2f} > {usage.limit}",
                usage_percent=usage.usage_percent,
                threshold=100.0,
            )

            triggered.add("projected")
            self._triggered_alerts[budget_name] = triggered

            self._emit_bus_event(
                self.BUS_TOPICS["alert"],
                alert.to_dict(),
                level="warning"
            )

            for callback in self._alert_callbacks:
                callback(alert)

        # Emit usage event
        self._emit_bus_event(
            self.BUS_TOPICS["track"],
            usage.to_dict()
        )

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

        return event_id


# Singleton
_monitor: Optional[BudgetMonitor] = None


def get_budget_monitor() -> BudgetMonitor:
    """Get or create the budget monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = BudgetMonitor()
    return _monitor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Budget Monitor (Step 265)")
    parser.add_argument("--summary", action="store_true", help="Show budget summary")
    parser.add_argument("--usage", metavar="NAME", help="Show usage for budget")
    parser.add_argument("--breakdown", metavar="NAME", help="Show cost breakdown")
    parser.add_argument("--forecast", metavar="NAME", help="Show forecast for budget")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    monitor = get_budget_monitor()

    if args.summary:
        summary = monitor.get_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("Budget Summary:")
            print(f"  Total Budgets: {summary['total_budgets']}")
            print(f"  Over Budget: {summary['over_budget']}")
            print(f"  At Risk: {summary['at_risk']}")
            print(f"  Healthy: {summary['healthy']}")
            print(f"  Overall Usage: {summary['overall_percent']:.1f}%")

    if args.usage:
        usage = monitor.get_usage(args.usage)
        if usage:
            if args.json:
                print(json.dumps(usage.to_dict(), indent=2))
            else:
                print(f"Budget Usage: {args.usage}")
                print(f"  Current: {usage.current_usage:.2f} / {usage.limit:.2f}")
                print(f"  Usage: {usage.usage_percent:.1f}%")
                print(f"  Remaining: {usage.remaining:.2f}")
                print(f"  Burn Rate: {usage.burn_rate:.2f}/day")
                print(f"  Projected: {usage.projected_usage:.2f}")
                print(f"  Will Exceed: {usage.will_exceed}")
        else:
            print(f"Budget not found: {args.usage}")

    if args.breakdown:
        breakdown = monitor.get_breakdown(args.breakdown)
        if breakdown:
            if args.json:
                print(json.dumps(breakdown.to_dict(), indent=2))
            else:
                print(f"Cost Breakdown: {args.breakdown}")
                print(f"  Total: {breakdown.total:.2f}")
                print(f"  By Category:")
                for cat, amount in sorted(breakdown.by_category.items(), key=lambda x: -x[1]):
                    print(f"    - {cat}: {amount:.2f}")
                print(f"  By Resource:")
                for res, amount in sorted(breakdown.by_resource.items(), key=lambda x: -x[1])[:10]:
                    print(f"    - {res}: {amount:.2f}")
        else:
            print(f"Budget not found: {args.breakdown}")

    if args.forecast:
        forecast = monitor.forecast(args.forecast)
        if forecast:
            if args.json:
                print(json.dumps(forecast, indent=2))
            else:
                print(f"Forecast: {args.forecast}")
                print(f"  Current: {forecast['current_usage']:.2f}")
                print(f"  Burn Rate: {forecast['burn_rate']:.2f}/day")
                print(f"  30-day Forecast: {forecast['forecast_total']:.2f}")
                print(f"  Limit: {forecast['limit']:.2f}")
                print(f"  Will Exceed: {forecast['will_exceed']}")
                if forecast['days_to_exceed']:
                    print(f"  Days to Exceed: {forecast['days_to_exceed']:.1f}")
        else:
            print(f"Budget not found: {args.forecast}")
