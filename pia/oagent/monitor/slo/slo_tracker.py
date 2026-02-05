#!/usr/bin/env python3
"""
SLO Tracker - Step 264

Tracks Service Level Objectives (SLOs) and Service Level Indicators (SLIs).

PBTSO Phase: VERIFY

Bus Topics:
- monitor.slo.track (emitted)
- monitor.slo.breach (emitted)
- monitor.sli.calculate (emitted)

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
from typing import Any, Callable, Dict, List, Optional, Tuple


class SLOType(Enum):
    """Types of SLOs."""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    SATURATION = "saturation"
    CUSTOM = "custom"


class ComplianceState(Enum):
    """SLO compliance states."""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class ComplianceWindow(Enum):
    """Compliance calculation windows."""
    ROLLING_7D = "rolling_7d"
    ROLLING_30D = "rolling_30d"
    CALENDAR_MONTH = "calendar_month"
    CALENDAR_QUARTER = "calendar_quarter"


@dataclass
class SLI:
    """Service Level Indicator.

    Attributes:
        name: SLI name
        value: Current SLI value
        unit: Unit of measurement
        good_events: Count of good events
        total_events: Count of total events
        timestamp: Measurement timestamp
    """
    name: str
    value: float
    unit: str = "%"
    good_events: int = 0
    total_events: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def ratio(self) -> float:
        """Get good/total ratio."""
        if self.total_events == 0:
            return 1.0
        return self.good_events / self.total_events


@dataclass
class SLOTarget:
    """SLO target definition.

    Attributes:
        target_value: Target value (e.g., 99.9 for 99.9% availability)
        operator: Comparison operator (>=, <=, ==)
        window: Compliance window
    """
    target_value: float
    operator: str = ">="
    window: ComplianceWindow = ComplianceWindow.ROLLING_30D

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_value": self.target_value,
            "operator": self.operator,
            "window": self.window.value,
        }

    def is_met(self, value: float) -> bool:
        """Check if value meets target.

        Args:
            value: Current value

        Returns:
            True if target is met
        """
        if self.operator == ">=":
            return value >= self.target_value
        elif self.operator == "<=":
            return value <= self.target_value
        elif self.operator == "==":
            return abs(value - self.target_value) < 0.0001
        elif self.operator == ">":
            return value > self.target_value
        elif self.operator == "<":
            return value < self.target_value
        return False


@dataclass
class SLO:
    """Service Level Objective.

    Attributes:
        name: SLO name
        slo_type: Type of SLO
        target: Target specification
        description: SLO description
        service: Service name
        sli_name: Associated SLI name
        owner: SLO owner
        created_at: Creation timestamp
    """
    name: str
    slo_type: SLOType
    target: SLOTarget
    description: str = ""
    service: str = ""
    sli_name: str = ""
    owner: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "slo_type": self.slo_type.value,
            "target": self.target.to_dict(),
            "description": self.description,
            "service": self.service,
            "sli_name": self.sli_name,
            "owner": self.owner,
            "created_at": self.created_at,
        }


@dataclass
class SLOCompliance:
    """SLO compliance status.

    Attributes:
        slo_name: SLO name
        state: Compliance state
        current_value: Current SLI value
        target_value: Target value
        budget_remaining: Error budget remaining (%)
        budget_consumed: Error budget consumed (%)
        time_to_breach: Estimated time to breach (hours)
        window_start: Window start timestamp
        window_end: Window end timestamp
        timestamp: Calculation timestamp
    """
    slo_name: str
    state: ComplianceState
    current_value: float
    target_value: float
    budget_remaining: float
    budget_consumed: float
    time_to_breach: Optional[float] = None
    window_start: float = 0.0
    window_end: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slo_name": self.slo_name,
            "state": self.state.value,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "budget_remaining": self.budget_remaining,
            "budget_consumed": self.budget_consumed,
            "time_to_breach": self.time_to_breach,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "timestamp": self.timestamp,
        }

    @property
    def is_healthy(self) -> bool:
        """Check if SLO is healthy (compliant with margin)."""
        return self.state == ComplianceState.COMPLIANT and self.budget_remaining > 25.0


@dataclass
class SLOBreach:
    """SLO breach event.

    Attributes:
        slo_name: SLO name
        breach_value: Value at breach
        target_value: Target value
        budget_exhausted: Error budget exhausted
        duration_s: Breach duration in seconds
        timestamp: Breach timestamp
        breach_id: Unique breach ID
    """
    slo_name: str
    breach_value: float
    target_value: float
    budget_exhausted: float
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)
    breach_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ErrorBudget:
    """Error budget calculation.

    Attributes:
        slo_name: SLO name
        total_budget: Total error budget (in minutes or events)
        consumed: Budget consumed
        remaining: Budget remaining
        burn_rate: Current burn rate
        window_days: Window in days
    """
    slo_name: str
    total_budget: float
    consumed: float
    remaining: float
    burn_rate: float
    window_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def remaining_percent(self) -> float:
        """Remaining budget as percentage."""
        if self.total_budget == 0:
            return 0.0
        return 100.0 * self.remaining / self.total_budget

    @property
    def time_to_exhaustion(self) -> Optional[float]:
        """Time to budget exhaustion in hours."""
        if self.burn_rate <= 0:
            return None
        return self.remaining / self.burn_rate


class SLOTracker:
    """
    Track Service Level Objectives and calculate compliance.

    The tracker:
    - Manages SLO definitions
    - Collects SLI measurements
    - Calculates compliance status
    - Tracks error budgets
    - Detects and alerts on breaches

    Example:
        tracker = SLOTracker()
        tracker.register_slo(SLO(
            name="api-availability",
            slo_type=SLOType.AVAILABILITY,
            target=SLOTarget(target_value=99.9),
            service="api-server",
        ))

        tracker.record_sli("api-availability", SLI(
            name="availability",
            value=99.95,
            good_events=9995,
            total_events=10000,
        ))

        compliance = tracker.get_compliance("api-availability")
        print(f"Compliance: {compliance.state.value}")
    """

    BUS_TOPICS = {
        "track": "monitor.slo.track",
        "breach": "monitor.slo.breach",
        "sli": "monitor.sli.calculate",
    }

    def __init__(
        self,
        history_days: int = 90,
        bus_dir: Optional[str] = None,
    ):
        """Initialize SLO tracker.

        Args:
            history_days: Days of SLI history to retain
            bus_dir: Bus directory
        """
        self.history_days = history_days

        # SLO registry
        self._slos: Dict[str, SLO] = {}
        self._sli_history: Dict[str, List[SLI]] = {}
        self._compliance_cache: Dict[str, SLOCompliance] = {}
        self._breaches: Dict[str, List[SLOBreach]] = {}

        # State
        self._breach_callbacks: List[Callable[[SLOBreach], None]] = []
        self._running = False
        self._evaluation_task: Optional[asyncio.Task] = None

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def register_slo(self, slo: SLO) -> None:
        """Register an SLO.

        Args:
            slo: SLO definition
        """
        self._slos[slo.name] = slo
        if slo.name not in self._sli_history:
            self._sli_history[slo.name] = []
        if slo.name not in self._breaches:
            self._breaches[slo.name] = []

        self._emit_bus_event(
            "monitor.slo.registered",
            slo.to_dict()
        )

    def unregister_slo(self, name: str) -> bool:
        """Unregister an SLO.

        Args:
            name: SLO name

        Returns:
            True if unregistered
        """
        if name not in self._slos:
            return False

        del self._slos[name]
        if name in self._sli_history:
            del self._sli_history[name]
        if name in self._compliance_cache:
            del self._compliance_cache[name]

        return True

    def record_sli(self, slo_name: str, sli: SLI) -> bool:
        """Record an SLI measurement.

        Args:
            slo_name: Associated SLO name
            sli: SLI measurement

        Returns:
            True if recorded
        """
        if slo_name not in self._slos:
            return False

        self._sli_history[slo_name].append(sli)

        # Prune old history
        cutoff = time.time() - (self.history_days * 86400)
        self._sli_history[slo_name] = [
            s for s in self._sli_history[slo_name]
            if s.timestamp >= cutoff
        ]

        # Emit event
        self._emit_bus_event(
            self.BUS_TOPICS["sli"],
            {
                "slo_name": slo_name,
                "sli": sli.to_dict(),
            }
        )

        # Update compliance
        self._update_compliance(slo_name)

        return True

    def record_events(
        self,
        slo_name: str,
        good_events: int,
        total_events: int,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Record good/total events for an SLO.

        Args:
            slo_name: SLO name
            good_events: Count of good events
            total_events: Count of total events
            timestamp: Optional timestamp

        Returns:
            True if recorded
        """
        if slo_name not in self._slos:
            return False

        if total_events == 0:
            return False

        value = 100.0 * good_events / total_events

        sli = SLI(
            name=self._slos[slo_name].sli_name or slo_name,
            value=value,
            good_events=good_events,
            total_events=total_events,
            timestamp=timestamp or time.time(),
        )

        return self.record_sli(slo_name, sli)

    def get_compliance(
        self,
        slo_name: str,
        window: Optional[ComplianceWindow] = None
    ) -> Optional[SLOCompliance]:
        """Get SLO compliance status.

        Args:
            slo_name: SLO name
            window: Override window

        Returns:
            Compliance status or None
        """
        if slo_name not in self._slos:
            return None

        slo = self._slos[slo_name]
        effective_window = window or slo.target.window

        return self._calculate_compliance(slo_name, effective_window)

    def get_all_compliance(self) -> Dict[str, SLOCompliance]:
        """Get compliance for all SLOs.

        Returns:
            Dictionary of compliance statuses
        """
        result = {}
        for name in self._slos:
            compliance = self.get_compliance(name)
            if compliance:
                result[name] = compliance
        return result

    def get_error_budget(self, slo_name: str) -> Optional[ErrorBudget]:
        """Calculate error budget for an SLO.

        Args:
            slo_name: SLO name

        Returns:
            Error budget or None
        """
        if slo_name not in self._slos:
            return None

        slo = self._slos[slo_name]
        window_days = self._get_window_days(slo.target.window)

        # Calculate total budget
        # For availability SLO, budget is (100 - target) percent of window
        error_budget_percent = 100.0 - slo.target.target_value
        total_budget_minutes = (window_days * 24 * 60) * (error_budget_percent / 100.0)

        # Calculate consumed budget from SLI history
        window_start = time.time() - (window_days * 86400)
        recent_slis = [
            s for s in self._sli_history.get(slo_name, [])
            if s.timestamp >= window_start
        ]

        if not recent_slis:
            return ErrorBudget(
                slo_name=slo_name,
                total_budget=total_budget_minutes,
                consumed=0.0,
                remaining=total_budget_minutes,
                burn_rate=0.0,
                window_days=window_days,
            )

        # Calculate consumed from bad events ratio
        total_good = sum(s.good_events for s in recent_slis)
        total_all = sum(s.total_events for s in recent_slis)

        if total_all == 0:
            consumed = 0.0
        else:
            error_rate = 1.0 - (total_good / total_all)
            consumed = total_budget_minutes * (error_rate / (error_budget_percent / 100.0))

        remaining = max(0, total_budget_minutes - consumed)

        # Calculate burn rate (errors per hour over last 24h)
        recent_window = time.time() - 86400
        recent_24h = [s for s in recent_slis if s.timestamp >= recent_window]
        if recent_24h:
            errors_24h = sum(s.total_events - s.good_events for s in recent_24h)
            burn_rate = errors_24h / 24.0
        else:
            burn_rate = 0.0

        return ErrorBudget(
            slo_name=slo_name,
            total_budget=total_budget_minutes,
            consumed=consumed,
            remaining=remaining,
            burn_rate=burn_rate,
            window_days=window_days,
        )

    def get_breaches(
        self,
        slo_name: str,
        window_days: int = 30
    ) -> List[SLOBreach]:
        """Get SLO breaches.

        Args:
            slo_name: SLO name
            window_days: Time window

        Returns:
            List of breaches
        """
        if slo_name not in self._breaches:
            return []

        cutoff = time.time() - (window_days * 86400)
        return [
            b for b in self._breaches[slo_name]
            if b.timestamp >= cutoff
        ]

    def get_sli_history(
        self,
        slo_name: str,
        window_days: int = 7
    ) -> List[SLI]:
        """Get SLI history.

        Args:
            slo_name: SLO name
            window_days: Time window

        Returns:
            SLI history
        """
        if slo_name not in self._sli_history:
            return []

        cutoff = time.time() - (window_days * 86400)
        return [
            s for s in self._sli_history[slo_name]
            if s.timestamp >= cutoff
        ]

    def register_breach_callback(
        self,
        callback: Callable[[SLOBreach], None]
    ) -> None:
        """Register breach callback.

        Args:
            callback: Callback function
        """
        self._breach_callbacks.append(callback)

    async def start(self) -> bool:
        """Start periodic evaluation.

        Returns:
            True if started
        """
        if self._running:
            return False

        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

        return True

    async def stop(self) -> bool:
        """Stop periodic evaluation.

        Returns:
            True if stopped
        """
        if not self._running:
            return False

        self._running = False
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get SLO summary.

        Returns:
            Summary dictionary
        """
        total = len(self._slos)
        compliant = 0
        at_risk = 0
        breached = 0

        for name in self._slos:
            compliance = self.get_compliance(name)
            if compliance:
                if compliance.state == ComplianceState.COMPLIANT:
                    compliant += 1
                elif compliance.state == ComplianceState.AT_RISK:
                    at_risk += 1
                elif compliance.state == ComplianceState.BREACHED:
                    breached += 1

        return {
            "total_slos": total,
            "compliant": compliant,
            "at_risk": at_risk,
            "breached": breached,
            "compliance_rate": 100.0 * compliant / total if total > 0 else 0.0,
        }

    def _calculate_compliance(
        self,
        slo_name: str,
        window: ComplianceWindow
    ) -> SLOCompliance:
        """Calculate compliance for an SLO.

        Args:
            slo_name: SLO name
            window: Compliance window

        Returns:
            Compliance status
        """
        slo = self._slos[slo_name]
        window_days = self._get_window_days(window)
        window_start = time.time() - (window_days * 86400)
        window_end = time.time()

        # Get SLIs in window
        slis = [
            s for s in self._sli_history.get(slo_name, [])
            if s.timestamp >= window_start
        ]

        if not slis:
            return SLOCompliance(
                slo_name=slo_name,
                state=ComplianceState.UNKNOWN,
                current_value=0.0,
                target_value=slo.target.target_value,
                budget_remaining=100.0,
                budget_consumed=0.0,
                window_start=window_start,
                window_end=window_end,
            )

        # Calculate aggregate SLI
        total_good = sum(s.good_events for s in slis)
        total_all = sum(s.total_events for s in slis)

        if total_all == 0:
            current_value = 100.0
        else:
            current_value = 100.0 * total_good / total_all

        # Calculate error budget
        error_budget = self.get_error_budget(slo_name)
        budget_remaining = error_budget.remaining_percent if error_budget else 100.0
        budget_consumed = 100.0 - budget_remaining

        # Determine state
        if slo.target.is_met(current_value):
            if budget_remaining < 25.0:
                state = ComplianceState.AT_RISK
            else:
                state = ComplianceState.COMPLIANT
        else:
            state = ComplianceState.BREACHED

        # Estimate time to breach
        time_to_breach = None
        if error_budget and error_budget.burn_rate > 0:
            time_to_breach = error_budget.remaining / error_budget.burn_rate

        return SLOCompliance(
            slo_name=slo_name,
            state=state,
            current_value=current_value,
            target_value=slo.target.target_value,
            budget_remaining=budget_remaining,
            budget_consumed=budget_consumed,
            time_to_breach=time_to_breach,
            window_start=window_start,
            window_end=window_end,
        )

    def _update_compliance(self, slo_name: str) -> None:
        """Update compliance and check for breaches.

        Args:
            slo_name: SLO name
        """
        compliance = self.get_compliance(slo_name)
        if not compliance:
            return

        prev_compliance = self._compliance_cache.get(slo_name)
        self._compliance_cache[slo_name] = compliance

        # Emit compliance event
        self._emit_bus_event(
            self.BUS_TOPICS["track"],
            compliance.to_dict()
        )

        # Check for new breach
        if compliance.state == ComplianceState.BREACHED:
            if not prev_compliance or prev_compliance.state != ComplianceState.BREACHED:
                breach = SLOBreach(
                    slo_name=slo_name,
                    breach_value=compliance.current_value,
                    target_value=compliance.target_value,
                    budget_exhausted=compliance.budget_consumed,
                )
                self._breaches[slo_name].append(breach)

                self._emit_bus_event(
                    self.BUS_TOPICS["breach"],
                    breach.to_dict(),
                    level="error"
                )

                for callback in self._breach_callbacks:
                    callback(breach)

    def _get_window_days(self, window: ComplianceWindow) -> int:
        """Get window duration in days.

        Args:
            window: Compliance window

        Returns:
            Days
        """
        if window == ComplianceWindow.ROLLING_7D:
            return 7
        elif window == ComplianceWindow.ROLLING_30D:
            return 30
        elif window == ComplianceWindow.CALENDAR_MONTH:
            return 30
        elif window == ComplianceWindow.CALENDAR_QUARTER:
            return 90
        return 30

    async def _evaluation_loop(self) -> None:
        """Periodic evaluation loop."""
        while self._running:
            try:
                for slo_name in self._slos:
                    self._update_compliance(slo_name)
            except Exception as e:
                self._emit_bus_event(
                    "monitor.slo.error",
                    {"error": str(e)},
                    level="error"
                )

            await asyncio.sleep(60)  # Evaluate every minute

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
_tracker: Optional[SLOTracker] = None


def get_slo_tracker() -> SLOTracker:
    """Get or create the SLO tracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = SLOTracker()
    return _tracker


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLO Tracker (Step 264)")
    parser.add_argument("--summary", action="store_true", help="Show SLO summary")
    parser.add_argument("--list", action="store_true", help="List all SLOs")
    parser.add_argument("--compliance", metavar="NAME", help="Show compliance for SLO")
    parser.add_argument("--budget", metavar="NAME", help="Show error budget for SLO")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    tracker = get_slo_tracker()

    if args.summary:
        summary = tracker.get_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("SLO Summary:")
            print(f"  Total SLOs: {summary['total_slos']}")
            print(f"  Compliant: {summary['compliant']}")
            print(f"  At Risk: {summary['at_risk']}")
            print(f"  Breached: {summary['breached']}")
            print(f"  Compliance Rate: {summary['compliance_rate']:.1f}%")

    if args.list:
        slos = list(tracker._slos.values())
        if args.json:
            print(json.dumps([s.to_dict() for s in slos], indent=2))
        else:
            print("Registered SLOs:")
            for slo in slos:
                print(f"  - {slo.name}: {slo.slo_type.value} (target: {slo.target.target_value}%)")

    if args.compliance:
        compliance = tracker.get_compliance(args.compliance)
        if compliance:
            if args.json:
                print(json.dumps(compliance.to_dict(), indent=2))
            else:
                print(f"Compliance for {args.compliance}:")
                print(f"  State: {compliance.state.value}")
                print(f"  Current: {compliance.current_value:.2f}%")
                print(f"  Target: {compliance.target_value}%")
                print(f"  Budget Remaining: {compliance.budget_remaining:.1f}%")
        else:
            print(f"SLO not found: {args.compliance}")

    if args.budget:
        budget = tracker.get_error_budget(args.budget)
        if budget:
            if args.json:
                print(json.dumps(budget.to_dict(), indent=2))
            else:
                print(f"Error Budget for {args.budget}:")
                print(f"  Total: {budget.total_budget:.1f} minutes")
                print(f"  Consumed: {budget.consumed:.1f} minutes")
                print(f"  Remaining: {budget.remaining:.1f} minutes ({budget.remaining_percent:.1f}%)")
                print(f"  Burn Rate: {budget.burn_rate:.2f}/hour")
                if budget.time_to_exhaustion:
                    print(f"  Time to Exhaustion: {budget.time_to_exhaustion:.1f} hours")
        else:
            print(f"SLO not found: {args.budget}")
