#!/usr/bin/env python3
"""
quota_manager.py - Central Quota Enforcement for Multi-Provider Orchestration

Version: 1.0.0
Ring: 1 (Services)
Protocol: Quota Protocol v1 / DKIN v30 / CITIZEN v2

This module provides:
  1. Provider-specific quota tracking (Claude, Codex, Gemini, Grok)
  2. Budget normalization across metering shapes
  3. Gating logic for wait/go decisions
  4. Cost vector prediction for operations
  5. Bus event emission for observability
  6. Empirical cost model fitting from observed deltas

Key Design Principles (from GPT research):
  - Gemini: Request-counted - maximize work per request
  - Claude/Codex: Compute-metered - minimize context + iterations
  - Universal heuristic: Optimize for "work completed per quota unit"

Usage:
    from quota_manager import QuotaManager, Provider

    qm = QuotaManager()

    # Check before spawning
    can_proceed, reason = qm.can_proceed(Provider.CLAUDE, tokens_in=5000)

    # Record consumption
    qm.record_consumption(Provider.CLAUDE, tokens_in=5000, tokens_out=2000)

    # Get optimal provider
    provider = qm.select_optimal_provider(task_complexity="medium")

Semops:
    PBQUOTA: Quota management operations

@module tools/quota_manager
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
from threading import Lock
from typing import Optional, Any, Callable

# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = "1.1.0"

# Bus path
BUS_PATH = Path(os.environ.get("PLURIBUS_BUS_PATH", ".pluribus/bus/events.ndjson"))

# State persistence
QUOTA_STATE_PATH = Path(os.environ.get("PLURIBUS_QUOTA_STATE", ".pluribus/state/quota_state.json"))

# Cost model path
COST_MODEL_PATH = Path(__file__).parent.parent / "specs" / "provider_cost_model_v1.json"

# Gating thresholds
CONSERVATION_THRESHOLD_PCT = 15  # Switch to conservation mode at 15%
CRITICAL_THRESHOLD_PCT = 5       # Critical mode at 5%
GEMINI_STOP_MULTI_STEP = 100     # Stop multi-step when <100 daily requests
GEMINI_QUEUE_RPM = 10            # Queue when <10 RPM

# Rolling window durations
ROLLING_5H_SECONDS = 5 * 60 * 60  # 5 hours in seconds
ROLLING_1M_SECONDS = 60           # 1 minute for RPM


# =============================================================================
# ENUMS
# =============================================================================

class Provider(str, Enum):
    """Supported providers."""
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"
    GROK = "grok"
    LOCAL = "local"


class MeteringShape(str, Enum):
    """How providers meter usage."""
    COMPUTE_METERED = "compute_metered"   # Based on context/complexity
    REQUEST_COUNTED = "request_counted"   # Based on request count


class BudgetType(str, Enum):
    """Budget categories."""
    HARD = "hard"           # Context window limit
    BURST = "burst"         # RPM limit
    SESSION = "session"     # Rolling 5h limit
    LONG = "long"           # Daily/weekly limit


class GatingDecision(str, Enum):
    """Gating decisions for operations."""
    PROCEED = "proceed"
    QUEUE = "queue"
    DOWNGRADE = "downgrade"  # Use cheaper model
    REJECT = "reject"


class TaskComplexity(str, Enum):
    """Task complexity classification."""
    TRIVIAL = "trivial"      # Single file, few lines
    SIMPLE = "simple"        # Few files, straightforward
    MEDIUM = "medium"        # Multiple files, coordination
    COMPLEX = "complex"      # Many files, architecture
    MASSIVE = "massive"      # Full codebase analysis


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CostVector:
    """Predicted cost for an operation."""
    delta_burst: float = 0.0      # RPM impact
    delta_session: float = 0.0    # 5h budget impact
    delta_long: float = 0.0       # Daily/weekly impact
    tokens_in: int = 0            # Input tokens
    tokens_out_est: int = 0       # Estimated output tokens

    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out_est


@dataclass
class BudgetState:
    """Current budget state for a provider."""
    remaining_burst: float = float('inf')      # Remaining RPM
    remaining_session: float = float('inf')    # Remaining 5h allowance
    remaining_long: float = float('inf')       # Remaining daily/weekly
    remaining_session_pct: float = 100.0
    remaining_long_pct: float = 100.0
    context_window: int = 200000
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsumptionRecord:
    """Record of a single consumption event."""
    provider: Provider
    timestamp: float
    tokens_in: int
    tokens_out: int
    files_included: int = 0
    tool_calls: int = 0
    reasoning_level: str = "medium"
    task_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class GatingResult:
    """Result of a gating decision."""
    decision: GatingDecision
    reason: str
    recommended_provider: Optional[Provider] = None
    recommended_model: Optional[str] = None
    wait_seconds: Optional[int] = None


# =============================================================================
# QUOTA MANAGER
# =============================================================================

class QuotaManager:
    """Central quota management for multi-provider orchestration."""

    def __init__(self, tier: str = "pro", working_dir: Optional[str] = None):
        self.tier = tier
        self.working_dir = working_dir or os.getcwd()
        self._lock = Lock()

        # Load cost model
        self.cost_model = self._load_cost_model()

        # Initialize budget states per provider
        self.budget_states: dict[Provider, BudgetState] = {}
        for provider in Provider:
            self.budget_states[provider] = self._init_budget_state(provider)

        # Consumption history for empirical fitting
        self.consumption_history: list[ConsumptionRecord] = []

        # Queued operations
        self.operation_queue: list[dict] = []

        # Load persisted state
        self._load_state()

        # Callbacks for status updates
        self._status_callbacks: list[Callable] = []

    def _load_cost_model(self) -> dict:
        """Load the provider cost model."""
        if COST_MODEL_PATH.exists():
            with open(COST_MODEL_PATH) as f:
                return json.load(f)
        return {"providers": {}}

    def _init_budget_state(self, provider: Provider) -> BudgetState:
        """Initialize budget state for a provider based on tier."""
        provider_config = self.cost_model.get("providers", {}).get(provider.value, {})
        tier_config = provider_config.get("budgets", {}).get(self.tier, {})

        state = BudgetState()
        state.context_window = provider_config.get("context_window", 200000)

        # Set initial budgets based on tier
        if provider == Provider.GEMINI:
            daily = tier_config.get("per_day", {}).get("requests", 1000)
            state.remaining_long = daily
            state.remaining_burst = tier_config.get("per_minute", {}).get("requests", 60)
        elif provider in (Provider.CLAUDE, Provider.CODEX):
            session = tier_config.get("rolling_5h", {})
            if "prompts_claude_code" in session:
                state.remaining_session = session["prompts_claude_code"].get("max", 40)
            elif "local_messages" in session:
                state.remaining_session = session["local_messages"].get("max", 225)

        return state

    def _load_state(self) -> None:
        """Load persisted quota state."""
        state_path = Path(self.working_dir) / QUOTA_STATE_PATH
        if state_path.exists():
            try:
                with open(state_path) as f:
                    data = json.load(f)
                for provider_name, state_data in data.get("budget_states", {}).items():
                    try:
                        provider = Provider(provider_name)
                        self.budget_states[provider] = BudgetState(**state_data)
                    except (ValueError, TypeError):
                        pass
                self.consumption_history = [
                    ConsumptionRecord(**r) for r in data.get("consumption_history", [])[-1000:]
                ]
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self) -> None:
        """Persist quota state."""
        state_path = Path(self.working_dir) / QUOTA_STATE_PATH
        state_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "budget_states": {
                p.value: s.to_dict() for p, s in self.budget_states.items()
            },
            "consumption_history": [asdict(r) for r in self.consumption_history[-1000:]],
            "tier": self.tier,
            "updated_at": time.time()
        }

        with open(state_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _emit_bus_event(self, topic: str, data: dict, level: str = "info") -> str:
        """Emit event to the Pluribus bus."""
        bus_path = Path(self.working_dir) / BUS_PATH
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = uuid.uuid4().hex
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "kind": "event",
            "level": level,
            "actor": "quota_manager",
            "host": os.uname().nodename if hasattr(os, 'uname') else "localhost",
            "pid": os.getpid(),
            "data": data
        }

        with open(bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_metering_shape(self, provider: Provider) -> MeteringShape:
        """Get the metering shape for a provider."""
        provider_config = self.cost_model.get("providers", {}).get(provider.value, {})
        shape = provider_config.get("metering_shape", "compute_metered")
        return MeteringShape(shape)

    def get_budget_state(self, provider: Provider) -> BudgetState:
        """Get current budget state for a provider."""
        return self.budget_states.get(provider, BudgetState())

    def predict_cost(
        self,
        provider: Provider,
        tokens_in: int,
        files_included: int = 0,
        tool_calls: int = 0,
        reasoning_level: str = "medium"
    ) -> CostVector:
        """
        Predict the cost of an operation.

        Uses empirical model when available, otherwise heuristics.
        """
        shape = self.get_metering_shape(provider)

        cost = CostVector(tokens_in=tokens_in)

        if shape == MeteringShape.REQUEST_COUNTED:
            # Gemini: Each prompt can be 1+ requests depending on agent loops
            estimated_requests = 1 + (tool_calls * 0.5)  # Tool calls may trigger additional requests
            cost.delta_burst = 1
            cost.delta_long = estimated_requests
        else:
            # Compute-metered: Based on context size and complexity
            # Empirical coefficients (to be refined from observations)
            a = 0.1  # Base cost
            b = 0.00001  # Per token
            c = 0.0001  # Per file byte
            d = 0.05  # Per tool call
            e = {"low": 0.5, "medium": 1.0, "high": 2.0}.get(reasoning_level, 1.0)

            session_cost = a + (b * tokens_in) + (c * files_included) + (d * tool_calls) * e
            cost.delta_session = session_cost
            cost.delta_long = session_cost * 0.1  # Weekly is accumulated

        # Estimate output tokens (heuristic: 0.5x input for code tasks)
        cost.tokens_out_est = int(tokens_in * 0.5)

        return cost

    def can_proceed(
        self,
        provider: Provider,
        tokens_in: int = 0,
        task_complexity: TaskComplexity = TaskComplexity.MEDIUM,
        **kwargs
    ) -> GatingResult:
        """
        Determine if an operation can proceed on the given provider.

        Returns a GatingResult with decision and recommendations.
        """
        with self._lock:
            state = self.budget_states[provider]
            shape = self.get_metering_shape(provider)
            cost = self.predict_cost(provider, tokens_in, **kwargs)

            # Check context window limit (hard constraint)
            if tokens_in > state.context_window:
                return GatingResult(
                    decision=GatingDecision.REJECT,
                    reason=f"tokens_in ({tokens_in}) exceeds context_window ({state.context_window})"
                )

            # Large context gate (Rule A)
            if tokens_in > 0.35 * state.context_window:
                if task_complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE):
                    return GatingResult(
                        decision=GatingDecision.REJECT,
                        reason=f"Large context ({tokens_in}) not justified for {task_complexity.value} task"
                    )

            # Provider-specific gating
            if shape == MeteringShape.REQUEST_COUNTED:
                return self._gate_request_counted(provider, state, cost)
            else:
                return self._gate_compute_metered(provider, state, cost)

    def _gate_request_counted(
        self,
        provider: Provider,
        state: BudgetState,
        cost: CostVector
    ) -> GatingResult:
        """Gating logic for request-counted providers (Gemini, Grok)."""
        # Check daily budget
        if state.remaining_long < GEMINI_STOP_MULTI_STEP:
            if cost.delta_long > 1:
                return GatingResult(
                    decision=GatingDecision.DOWNGRADE,
                    reason=f"Daily requests low ({state.remaining_long}), single-shot only",
                    recommended_provider=Provider.LOCAL
                )

        # Check RPM
        if state.remaining_burst < GEMINI_QUEUE_RPM:
            return GatingResult(
                decision=GatingDecision.QUEUE,
                reason=f"RPM exhausted ({state.remaining_burst}), queueing",
                wait_seconds=60
            )

        # Check if we have enough
        if state.remaining_long < cost.delta_long:
            return GatingResult(
                decision=GatingDecision.REJECT,
                reason=f"Insufficient daily budget ({state.remaining_long} < {cost.delta_long})"
            )

        return GatingResult(
            decision=GatingDecision.PROCEED,
            reason="Budget available"
        )

    def _gate_compute_metered(
        self,
        provider: Provider,
        state: BudgetState,
        cost: CostVector
    ) -> GatingResult:
        """Gating logic for compute-metered providers (Claude, Codex)."""
        # Conservation mode at 15%
        if state.remaining_session_pct < CONSERVATION_THRESHOLD_PCT:
            # Switch to cheaper model
            cheaper_model = "haiku" if provider == Provider.CLAUDE else "gpt-5.1-codex-mini"
            return GatingResult(
                decision=GatingDecision.DOWNGRADE,
                reason=f"Session budget low ({state.remaining_session_pct:.1f}%), conservation mode",
                recommended_model=cheaper_model
            )

        # Critical mode at 5%
        if state.remaining_session_pct < CRITICAL_THRESHOLD_PCT:
            return GatingResult(
                decision=GatingDecision.REJECT,
                reason=f"Session budget critical ({state.remaining_session_pct:.1f}%)"
            )

        # Weekly conservation
        if state.remaining_long_pct < CONSERVATION_THRESHOLD_PCT:
            return GatingResult(
                decision=GatingDecision.DOWNGRADE,
                reason=f"Weekly budget low ({state.remaining_long_pct:.1f}%), high-ROI only",
                recommended_provider=Provider.LOCAL
            )

        return GatingResult(
            decision=GatingDecision.PROCEED,
            reason="Budget available"
        )

    def record_consumption(
        self,
        provider: Provider,
        tokens_in: int,
        tokens_out: int,
        files_included: int = 0,
        tool_calls: int = 0,
        reasoning_level: str = "medium",
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        delta_from_status: Optional[dict] = None
    ) -> None:
        """
        Record consumption for a completed operation.

        If delta_from_status is provided (from /status command), use actual values.
        Otherwise, use predicted values.
        """
        with self._lock:
            # Create record
            record = ConsumptionRecord(
                provider=provider,
                timestamp=time.time(),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                files_included=files_included,
                tool_calls=tool_calls,
                reasoning_level=reasoning_level,
                task_id=task_id,
                agent_id=agent_id
            )
            self.consumption_history.append(record)

            # Update budget state
            state = self.budget_states[provider]

            if delta_from_status:
                # Use actual deltas from /status
                state.remaining_session -= delta_from_status.get("delta_5h", 0)
                state.remaining_long -= delta_from_status.get("delta_weekly", 0)
            else:
                # Use predicted cost
                cost = self.predict_cost(
                    provider, tokens_in, files_included, tool_calls, reasoning_level
                )
                state.remaining_burst -= cost.delta_burst
                state.remaining_session -= cost.delta_session
                state.remaining_long -= cost.delta_long

            # Recalculate percentages
            initial = self._init_budget_state(provider)
            if initial.remaining_session > 0:
                state.remaining_session_pct = (state.remaining_session / initial.remaining_session) * 100
            if initial.remaining_long > 0:
                state.remaining_long_pct = (state.remaining_long / initial.remaining_long) * 100

            state.last_updated = time.time()

            # Emit bus event
            self._emit_bus_event("quota.consumed", {
                "provider": provider.value,
                "tier": self.tier,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "remaining_session": state.remaining_session,
                "remaining_session_pct": state.remaining_session_pct,
                "remaining_long": state.remaining_long,
                "remaining_long_pct": state.remaining_long_pct,
                "agent_id": agent_id,
                "task_id": task_id
            })

            # Check thresholds
            self._check_thresholds(provider, state)

            # Persist
            self._save_state()

    def _check_thresholds(self, provider: Provider, state: BudgetState) -> None:
        """Check and emit threshold crossing events."""
        thresholds = [50, 25, 15, 5]

        for threshold in thresholds:
            if state.remaining_session_pct <= threshold < state.remaining_session_pct + 5:
                self._emit_bus_event("quota.threshold", {
                    "provider": provider.value,
                    "budget_type": "session",
                    "threshold_pct": threshold,
                    "remaining_pct": state.remaining_session_pct
                }, level="warning" if threshold <= 15 else "info")

            if state.remaining_long_pct <= threshold < state.remaining_long_pct + 5:
                self._emit_bus_event("quota.threshold", {
                    "provider": provider.value,
                    "budget_type": "long",
                    "threshold_pct": threshold,
                    "remaining_pct": state.remaining_long_pct
                }, level="warning" if threshold <= 15 else "info")

    def update_from_status(self, provider: Provider, status_output: str) -> None:
        """
        Update budget state from /status command output.

        Parses the status output to extract remaining limits.
        """
        with self._lock:
            state = self.budget_states[provider]

            # Parse status output (format varies by provider)
            # Example patterns to match:
            # "Remaining: 35/40 prompts this 5-hour period"
            # "Daily requests: 850/1000"

            import re

            # Pattern for "X/Y" format
            remaining_pattern = r'(\d+)\s*/\s*(\d+)'
            matches = re.findall(remaining_pattern, status_output)

            if matches:
                # Assume first match is session, second is daily/weekly
                if len(matches) >= 1:
                    remaining, total = int(matches[0][0]), int(matches[0][1])
                    state.remaining_session = remaining
                    state.remaining_session_pct = (remaining / total) * 100 if total > 0 else 0

                if len(matches) >= 2:
                    remaining, total = int(matches[1][0]), int(matches[1][1])
                    state.remaining_long = remaining
                    state.remaining_long_pct = (remaining / total) * 100 if total > 0 else 0

            state.last_updated = time.time()
            self._save_state()

            self._emit_bus_event("quota.remaining", {
                "provider": provider.value,
                "remaining_session": state.remaining_session,
                "remaining_session_pct": state.remaining_session_pct,
                "remaining_long": state.remaining_long,
                "remaining_long_pct": state.remaining_long_pct
            })

    def select_optimal_provider(
        self,
        task_complexity: TaskComplexity = TaskComplexity.MEDIUM,
        tokens_in: int = 0,
        prefer_providers: Optional[list[Provider]] = None,
        exclude_providers: Optional[list[Provider]] = None
    ) -> tuple[Provider, GatingResult]:
        """
        Select the optimal provider based on current budgets and task.

        Returns the best provider and the gating result.
        """
        candidates = prefer_providers or [Provider.CLAUDE, Provider.CODEX, Provider.GEMINI]
        if exclude_providers:
            candidates = [p for p in candidates if p not in exclude_providers]

        best_provider = None
        best_result = None
        best_score = -1

        for provider in candidates:
            result = self.can_proceed(provider, tokens_in, task_complexity)

            if result.decision == GatingDecision.PROCEED:
                state = self.budget_states[provider]
                # Score based on remaining budget
                score = state.remaining_session_pct + state.remaining_long_pct

                if score > best_score:
                    best_score = score
                    best_provider = provider
                    best_result = result

        if best_provider:
            return best_provider, best_result

        # If no provider can proceed, return the one with best downgrade option
        for provider in candidates:
            result = self.can_proceed(provider, tokens_in, task_complexity)
            if result.decision == GatingDecision.DOWNGRADE:
                return provider, result

        # Last resort: queue on first candidate
        return candidates[0], GatingResult(
            decision=GatingDecision.QUEUE,
            reason="All providers at capacity"
        )

    def get_optimization_rules(self, provider: Provider) -> list[str]:
        """Get optimization rules for a provider."""
        provider_config = self.cost_model.get("providers", {}).get(provider.value, {})
        return provider_config.get("optimization_rules", [])

    def format_status_report(self) -> str:
        """Generate a human-readable status report."""
        lines = [
            "=" * 60,
            f"QUOTA STATUS REPORT - Tier: {self.tier}",
            "=" * 60,
            ""
        ]

        for provider in Provider:
            state = self.budget_states[provider]
            shape = self.get_metering_shape(provider)

            lines.append(f"[{provider.value.upper()}] ({shape.value})")
            lines.append(f"  Context Window: {state.context_window:,} tokens")

            if shape == MeteringShape.REQUEST_COUNTED:
                lines.append(f"  Burst (RPM):    {state.remaining_burst:.0f} remaining")
                lines.append(f"  Daily:          {state.remaining_long:.0f} requests ({state.remaining_long_pct:.1f}%)")
            else:
                lines.append(f"  Session (5h):   {state.remaining_session:.1f} ({state.remaining_session_pct:.1f}%)")
                lines.append(f"  Weekly:         {state.remaining_long:.1f} ({state.remaining_long_pct:.1f}%)")

            # WIP meter
            pct = state.remaining_session_pct if shape == MeteringShape.COMPUTE_METERED else state.remaining_long_pct
            filled = int(pct / 10)
            meter = "[" + "#" * filled + "-" * (10 - filled) + "]"
            lines.append(f"  Budget Meter:   {meter} {pct:.1f}%")
            lines.append("")

        return "\n".join(lines)

    def reset_session_budgets(self) -> None:
        """Reset session budgets (call on 5h window reset)."""
        with self._lock:
            for provider in Provider:
                initial = self._init_budget_state(provider)
                self.budget_states[provider].remaining_session = initial.remaining_session
                self.budget_states[provider].remaining_session_pct = 100.0
            self._save_state()
            self._emit_bus_event("quota.reset", {"type": "session"})

    def reset_long_budgets(self) -> None:
        """Reset daily/weekly budgets."""
        with self._lock:
            for provider in Provider:
                initial = self._init_budget_state(provider)
                self.budget_states[provider].remaining_long = initial.remaining_long
                self.budget_states[provider].remaining_long_pct = 100.0
            self._save_state()
            self._emit_bus_event("quota.reset", {"type": "long"})


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for PBQUOTA operations."""
    import sys

    if len(sys.argv) < 2:
        print(f"QUOTA MANAGER v{VERSION}")
        print("\nUsage:")
        print("  python3 quota_manager.py status              # Show quota status")
        print("  python3 quota_manager.py check <provider>    # Check if can proceed")
        print("  python3 quota_manager.py select              # Select optimal provider")
        print("  python3 quota_manager.py reset-session       # Reset 5h budgets")
        print("  python3 quota_manager.py reset-long          # Reset daily/weekly budgets")
        print("\nSemops: PBQUOTA")
        sys.exit(1)

    cmd = sys.argv[1]
    qm = QuotaManager()

    if cmd == "status":
        print(qm.format_status_report())

    elif cmd == "check":
        if len(sys.argv) < 3:
            print("Usage: quota_manager.py check <provider>")
            sys.exit(1)
        provider = Provider(sys.argv[2])
        result = qm.can_proceed(provider)
        print(f"Provider: {provider.value}")
        print(f"Decision: {result.decision.value}")
        print(f"Reason: {result.reason}")
        if result.recommended_model:
            print(f"Recommended Model: {result.recommended_model}")
        if result.recommended_provider:
            print(f"Recommended Provider: {result.recommended_provider.value}")

    elif cmd == "select":
        provider, result = qm.select_optimal_provider()
        print(f"Optimal Provider: {provider.value}")
        print(f"Decision: {result.decision.value}")
        print(f"Reason: {result.reason}")

    elif cmd == "reset-session":
        qm.reset_session_budgets()
        print("Session budgets reset.")

    elif cmd == "reset-long":
        qm.reset_long_budgets()
        print("Daily/weekly budgets reset.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
