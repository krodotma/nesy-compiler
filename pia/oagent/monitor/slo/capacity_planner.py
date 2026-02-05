#!/usr/bin/env python3
"""
Capacity Planner - Step 266

Provides capacity planning and forecasting for resources.

PBTSO Phase: PLAN

Bus Topics:
- monitor.capacity.plan (emitted)
- monitor.capacity.forecast (emitted)
- monitor.capacity.alert (emitted)

Protocol: DKIN v30, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class ResourceCategory(Enum):
    """Resource categories for planning."""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CUSTOM = "custom"


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ForecastConfidence(Enum):
    """Forecast confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CapacityThreshold:
    """Capacity threshold configuration.

    Attributes:
        resource: Resource name
        category: Resource category
        current_capacity: Current capacity
        warning_threshold: Warning threshold (%)
        critical_threshold: Critical threshold (%)
        scale_up_threshold: Scale up threshold (%)
        scale_down_threshold: Scale down threshold (%)
        unit: Capacity unit
    """
    resource: str
    category: ResourceCategory
    current_capacity: float
    warning_threshold: float = 75.0
    critical_threshold: float = 90.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    unit: str = "units"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource": self.resource,
            "category": self.category.value,
            "current_capacity": self.current_capacity,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "unit": self.unit,
        }


@dataclass
class ResourceUsagePoint:
    """Resource usage data point.

    Attributes:
        resource: Resource name
        usage: Current usage
        capacity: Current capacity
        timestamp: Measurement timestamp
    """
    resource: str
    usage: float
    capacity: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def utilization(self) -> float:
        """Utilization percentage."""
        if self.capacity == 0:
            return 0.0
        return 100.0 * self.usage / self.capacity


@dataclass
class ResourceProjection:
    """Resource usage projection.

    Attributes:
        resource: Resource name
        current_usage: Current usage
        current_capacity: Current capacity
        projected_usage: Projected usage
        projection_date: Projection target date
        days_ahead: Days into future
        growth_rate: Daily growth rate
        confidence: Projection confidence
    """
    resource: str
    current_usage: float
    current_capacity: float
    projected_usage: float
    projection_date: float
    days_ahead: int
    growth_rate: float
    confidence: ForecastConfidence = ForecastConfidence.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource": self.resource,
            "current_usage": self.current_usage,
            "current_capacity": self.current_capacity,
            "projected_usage": self.projected_usage,
            "projected_utilization": self.projected_utilization,
            "projection_date": self.projection_date,
            "days_ahead": self.days_ahead,
            "growth_rate": self.growth_rate,
            "confidence": self.confidence.value,
        }

    @property
    def projected_utilization(self) -> float:
        """Projected utilization percentage."""
        if self.current_capacity == 0:
            return 0.0
        return 100.0 * self.projected_usage / self.current_capacity

    @property
    def will_exceed(self) -> bool:
        """Check if projected to exceed capacity."""
        return self.projected_usage > self.current_capacity


@dataclass
class CapacityForecast:
    """Capacity forecast with multiple projections.

    Attributes:
        resource: Resource name
        current_usage: Current usage
        current_capacity: Current capacity
        projections: List of projections at different time horizons
        days_to_capacity: Estimated days until capacity reached
        recommended_capacity: Recommended capacity
        timestamp: Forecast timestamp
    """
    resource: str
    current_usage: float
    current_capacity: float
    projections: List[ResourceProjection] = field(default_factory=list)
    days_to_capacity: Optional[float] = None
    recommended_capacity: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource": self.resource,
            "current_usage": self.current_usage,
            "current_capacity": self.current_capacity,
            "current_utilization": self.current_utilization,
            "projections": [p.to_dict() for p in self.projections],
            "days_to_capacity": self.days_to_capacity,
            "recommended_capacity": self.recommended_capacity,
            "timestamp": self.timestamp,
        }

    @property
    def current_utilization(self) -> float:
        """Current utilization percentage."""
        if self.current_capacity == 0:
            return 0.0
        return 100.0 * self.current_usage / self.current_capacity


@dataclass
class ScalingRecommendation:
    """Scaling recommendation.

    Attributes:
        resource: Resource name
        direction: Scaling direction
        current_capacity: Current capacity
        recommended_capacity: Recommended capacity
        reason: Recommendation reason
        urgency: Urgency level (1-5)
        estimated_cost_change: Estimated cost change
        timestamp: Recommendation timestamp
    """
    resource: str
    direction: ScalingDirection
    current_capacity: float
    recommended_capacity: float
    reason: str
    urgency: int = 3
    estimated_cost_change: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource": self.resource,
            "direction": self.direction.value,
            "current_capacity": self.current_capacity,
            "recommended_capacity": self.recommended_capacity,
            "capacity_change": self.capacity_change,
            "change_percent": self.change_percent,
            "reason": self.reason,
            "urgency": self.urgency,
            "estimated_cost_change": self.estimated_cost_change,
            "timestamp": self.timestamp,
        }

    @property
    def capacity_change(self) -> float:
        """Capacity change amount."""
        return self.recommended_capacity - self.current_capacity

    @property
    def change_percent(self) -> float:
        """Capacity change percentage."""
        if self.current_capacity == 0:
            return 0.0
        return 100.0 * self.capacity_change / self.current_capacity


class CapacityPlanner:
    """
    Plan capacity and forecast resource needs.

    The planner:
    - Tracks resource usage over time
    - Forecasts future capacity needs
    - Generates scaling recommendations
    - Alerts on capacity issues

    Example:
        planner = CapacityPlanner()
        planner.register_resource(CapacityThreshold(
            resource="db-cluster",
            category=ResourceCategory.DATABASE,
            current_capacity=1000,
        ))

        planner.record_usage("db-cluster", ResourceUsagePoint(
            resource="db-cluster",
            usage=750,
            capacity=1000,
        ))

        forecast = planner.forecast("db-cluster", days_ahead=30)
        print(f"30-day projection: {forecast.projections[0].projected_utilization}%")
    """

    BUS_TOPICS = {
        "plan": "monitor.capacity.plan",
        "forecast": "monitor.capacity.forecast",
        "alert": "monitor.capacity.alert",
    }

    # Default forecast horizons in days
    DEFAULT_HORIZONS = [7, 14, 30, 60, 90]

    def __init__(
        self,
        history_days: int = 90,
        bus_dir: Optional[str] = None,
    ):
        """Initialize capacity planner.

        Args:
            history_days: Days of usage history to retain
            bus_dir: Bus directory
        """
        self.history_days = history_days

        # Resource registry
        self._thresholds: Dict[str, CapacityThreshold] = {}
        self._usage_history: Dict[str, List[ResourceUsagePoint]] = {}

        # State
        self._alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def register_resource(self, threshold: CapacityThreshold) -> None:
        """Register a resource for capacity planning.

        Args:
            threshold: Resource threshold configuration
        """
        self._thresholds[threshold.resource] = threshold
        if threshold.resource not in self._usage_history:
            self._usage_history[threshold.resource] = []

        self._emit_bus_event(
            "monitor.capacity.registered",
            threshold.to_dict()
        )

    def unregister_resource(self, resource: str) -> bool:
        """Unregister a resource.

        Args:
            resource: Resource name

        Returns:
            True if unregistered
        """
        if resource not in self._thresholds:
            return False

        del self._thresholds[resource]
        if resource in self._usage_history:
            del self._usage_history[resource]

        return True

    def record_usage(self, resource: str, point: ResourceUsagePoint) -> bool:
        """Record a usage data point.

        Args:
            resource: Resource name
            point: Usage data point

        Returns:
            True if recorded
        """
        if resource not in self._thresholds:
            return False

        self._usage_history[resource].append(point)

        # Update capacity if changed
        if point.capacity != self._thresholds[resource].current_capacity:
            self._thresholds[resource].current_capacity = point.capacity

        # Prune old history
        cutoff = time.time() - (self.history_days * 86400)
        self._usage_history[resource] = [
            p for p in self._usage_history[resource]
            if p.timestamp >= cutoff
        ]

        # Check thresholds
        self._check_thresholds(resource, point)

        return True

    def forecast(
        self,
        resource: str,
        days_ahead: Optional[List[int]] = None
    ) -> Optional[CapacityForecast]:
        """Generate capacity forecast.

        Args:
            resource: Resource name
            days_ahead: Forecast horizons (days)

        Returns:
            Capacity forecast or None
        """
        if resource not in self._thresholds:
            return None

        threshold = self._thresholds[resource]
        history = self._usage_history.get(resource, [])

        if not history:
            return None

        horizons = days_ahead or self.DEFAULT_HORIZONS
        current_usage = history[-1].usage
        current_capacity = threshold.current_capacity

        # Calculate growth rate using linear regression
        growth_rate, confidence = self._calculate_growth_rate(history)

        projections = []
        for days in horizons:
            projected_usage = current_usage + (growth_rate * days)
            projection_date = time.time() + (days * 86400)

            projections.append(ResourceProjection(
                resource=resource,
                current_usage=current_usage,
                current_capacity=current_capacity,
                projected_usage=projected_usage,
                projection_date=projection_date,
                days_ahead=days,
                growth_rate=growth_rate,
                confidence=confidence,
            ))

        # Calculate days to capacity
        days_to_capacity = None
        if growth_rate > 0:
            remaining = current_capacity - current_usage
            days_to_capacity = remaining / growth_rate

        # Calculate recommended capacity (20% buffer)
        max_projected = max(p.projected_usage for p in projections)
        recommended_capacity = max_projected * 1.2

        forecast = CapacityForecast(
            resource=resource,
            current_usage=current_usage,
            current_capacity=current_capacity,
            projections=projections,
            days_to_capacity=days_to_capacity,
            recommended_capacity=recommended_capacity,
        )

        # Emit event
        self._emit_bus_event(
            self.BUS_TOPICS["forecast"],
            forecast.to_dict()
        )

        return forecast

    def get_recommendation(self, resource: str) -> Optional[ScalingRecommendation]:
        """Get scaling recommendation for a resource.

        Args:
            resource: Resource name

        Returns:
            Scaling recommendation or None
        """
        if resource not in self._thresholds:
            return None

        threshold = self._thresholds[resource]
        history = self._usage_history.get(resource, [])

        if not history:
            return None

        current_usage = history[-1].usage
        current_capacity = threshold.current_capacity
        utilization = 100.0 * current_usage / current_capacity if current_capacity > 0 else 0.0

        # Get forecast
        forecast = self.forecast(resource, [30])

        # Determine scaling direction
        direction = ScalingDirection.NONE
        recommended_capacity = current_capacity
        reason = ""
        urgency = 1

        if utilization >= threshold.scale_up_threshold:
            direction = ScalingDirection.UP
            # Scale up by 50% or to accommodate 30-day forecast
            if forecast and forecast.projections:
                recommended_capacity = max(
                    current_capacity * 1.5,
                    forecast.projections[0].projected_usage * 1.2
                )
            else:
                recommended_capacity = current_capacity * 1.5

            if utilization >= threshold.critical_threshold:
                urgency = 5
                reason = f"Critical utilization ({utilization:.1f}%) - immediate scaling required"
            else:
                urgency = 3
                reason = f"High utilization ({utilization:.1f}%) - scaling recommended"

        elif utilization <= threshold.scale_down_threshold:
            direction = ScalingDirection.DOWN
            # Scale down to 60% utilization target
            recommended_capacity = current_usage / 0.6
            urgency = 1
            reason = f"Low utilization ({utilization:.1f}%) - cost optimization opportunity"

        elif forecast and forecast.days_to_capacity and forecast.days_to_capacity < 30:
            direction = ScalingDirection.UP
            recommended_capacity = forecast.recommended_capacity or current_capacity * 1.5
            urgency = 3
            reason = f"Projected to reach capacity in {forecast.days_to_capacity:.0f} days"

        if direction == ScalingDirection.NONE:
            return None

        recommendation = ScalingRecommendation(
            resource=resource,
            direction=direction,
            current_capacity=current_capacity,
            recommended_capacity=recommended_capacity,
            reason=reason,
            urgency=urgency,
        )

        # Emit event
        self._emit_bus_event(
            self.BUS_TOPICS["plan"],
            recommendation.to_dict()
        )

        return recommendation

    def get_all_recommendations(self) -> List[ScalingRecommendation]:
        """Get recommendations for all resources.

        Returns:
            List of scaling recommendations
        """
        recommendations = []
        for resource in self._thresholds:
            rec = self.get_recommendation(resource)
            if rec:
                recommendations.append(rec)

        # Sort by urgency
        recommendations.sort(key=lambda r: -r.urgency)
        return recommendations

    def get_usage_history(
        self,
        resource: str,
        window_days: int = 30
    ) -> List[ResourceUsagePoint]:
        """Get usage history.

        Args:
            resource: Resource name
            window_days: Time window

        Returns:
            Usage history
        """
        if resource not in self._usage_history:
            return []

        cutoff = time.time() - (window_days * 86400)
        return [
            p for p in self._usage_history[resource]
            if p.timestamp >= cutoff
        ]

    def get_daily_stats(
        self,
        resource: str,
        window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get daily usage statistics.

        Args:
            resource: Resource name
            window_days: Time window

        Returns:
            Daily statistics
        """
        history = self.get_usage_history(resource, window_days)

        # Aggregate by day
        daily: Dict[str, List[float]] = {}
        for point in history:
            date = datetime.fromtimestamp(point.timestamp).strftime("%Y-%m-%d")
            if date not in daily:
                daily[date] = []
            daily[date].append(point.utilization)

        stats = []
        for date, values in sorted(daily.items()):
            stats.append({
                "date": date,
                "avg_utilization": sum(values) / len(values),
                "max_utilization": max(values),
                "min_utilization": min(values),
                "samples": len(values),
            })

        return stats

    def register_alert_callback(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register alert callback.

        Args:
            callback: Callback function
        """
        self._alert_callbacks.append(callback)

    def get_summary(self) -> Dict[str, Any]:
        """Get capacity planning summary.

        Returns:
            Summary dictionary
        """
        total = len(self._thresholds)
        critical = 0
        warning = 0
        scaling_needed = 0

        for resource in self._thresholds:
            history = self._usage_history.get(resource, [])
            if not history:
                continue

            utilization = history[-1].utilization
            threshold = self._thresholds[resource]

            if utilization >= threshold.critical_threshold:
                critical += 1
            elif utilization >= threshold.warning_threshold:
                warning += 1

            rec = self.get_recommendation(resource)
            if rec and rec.direction != ScalingDirection.NONE:
                scaling_needed += 1

        return {
            "total_resources": total,
            "critical": critical,
            "warning": warning,
            "healthy": total - critical - warning,
            "scaling_needed": scaling_needed,
        }

    def _calculate_growth_rate(
        self,
        history: List[ResourceUsagePoint]
    ) -> Tuple[float, ForecastConfidence]:
        """Calculate growth rate from history.

        Args:
            history: Usage history

        Returns:
            (growth_rate per day, confidence)
        """
        if len(history) < 2:
            return 0.0, ForecastConfidence.LOW

        # Use simple linear regression
        n = len(history)
        x_values = [(p.timestamp - history[0].timestamp) / 86400 for p in history]  # Days
        y_values = [p.usage for p in history]

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0, ForecastConfidence.LOW

        slope = numerator / denominator

        # Calculate R-squared for confidence
        y_pred = [y_mean + slope * (x - x_mean) for x in x_values]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_values, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if r_squared > 0.8:
            confidence = ForecastConfidence.HIGH
        elif r_squared > 0.5:
            confidence = ForecastConfidence.MEDIUM
        else:
            confidence = ForecastConfidence.LOW

        return slope, confidence

    def _check_thresholds(
        self,
        resource: str,
        point: ResourceUsagePoint
    ) -> None:
        """Check thresholds and emit alerts.

        Args:
            resource: Resource name
            point: Usage point
        """
        threshold = self._thresholds[resource]
        utilization = point.utilization

        if utilization >= threshold.critical_threshold:
            alert = {
                "resource": resource,
                "level": "critical",
                "message": f"Resource {resource} at critical utilization: {utilization:.1f}%",
                "utilization": utilization,
                "threshold": threshold.critical_threshold,
            }
            self._emit_bus_event(
                self.BUS_TOPICS["alert"],
                alert,
                level="error"
            )
            for callback in self._alert_callbacks:
                callback(alert)

        elif utilization >= threshold.warning_threshold:
            alert = {
                "resource": resource,
                "level": "warning",
                "message": f"Resource {resource} at warning utilization: {utilization:.1f}%",
                "utilization": utilization,
                "threshold": threshold.warning_threshold,
            }
            self._emit_bus_event(
                self.BUS_TOPICS["alert"],
                alert,
                level="warning"
            )
            for callback in self._alert_callbacks:
                callback(alert)

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
_planner: Optional[CapacityPlanner] = None


def get_capacity_planner() -> CapacityPlanner:
    """Get or create the capacity planner singleton."""
    global _planner
    if _planner is None:
        _planner = CapacityPlanner()
    return _planner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capacity Planner (Step 266)")
    parser.add_argument("--summary", action="store_true", help="Show capacity summary")
    parser.add_argument("--forecast", metavar="NAME", help="Show forecast for resource")
    parser.add_argument("--recommend", action="store_true", help="Show all recommendations")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    planner = get_capacity_planner()

    if args.summary:
        summary = planner.get_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("Capacity Planning Summary:")
            print(f"  Total Resources: {summary['total_resources']}")
            print(f"  Critical: {summary['critical']}")
            print(f"  Warning: {summary['warning']}")
            print(f"  Healthy: {summary['healthy']}")
            print(f"  Scaling Needed: {summary['scaling_needed']}")

    if args.forecast:
        forecast = planner.forecast(args.forecast)
        if forecast:
            if args.json:
                print(json.dumps(forecast.to_dict(), indent=2))
            else:
                print(f"Forecast: {args.forecast}")
                print(f"  Current: {forecast.current_usage:.1f} / {forecast.current_capacity:.1f} ({forecast.current_utilization:.1f}%)")
                print(f"  Projections:")
                for p in forecast.projections:
                    print(f"    {p.days_ahead}d: {p.projected_usage:.1f} ({p.projected_utilization:.1f}%)")
                if forecast.days_to_capacity:
                    print(f"  Days to Capacity: {forecast.days_to_capacity:.0f}")
                if forecast.recommended_capacity:
                    print(f"  Recommended Capacity: {forecast.recommended_capacity:.1f}")
        else:
            print(f"Resource not found: {args.forecast}")

    if args.recommend:
        recommendations = planner.get_all_recommendations()
        if args.json:
            print(json.dumps([r.to_dict() for r in recommendations], indent=2))
        else:
            if recommendations:
                print("Scaling Recommendations:")
                for rec in recommendations:
                    print(f"  [{rec.urgency}] {rec.resource}: {rec.direction.value}")
                    print(f"      {rec.current_capacity:.1f} -> {rec.recommended_capacity:.1f} ({rec.change_percent:+.1f}%)")
                    print(f"      Reason: {rec.reason}")
            else:
                print("No scaling recommendations at this time.")
