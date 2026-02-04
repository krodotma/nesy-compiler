#!/usr/bin/env python3
"""
Agent Lightning Trainer Integration
====================================

RL training framework for agents with bus-first evidence and omega-liveness gates.
Integrates Microsoft's Agent Lightning patterns with Pluribus observability.

Key Features:
- Episode-based RL training with bus event rewards
- Omega-liveness gates for training episode validation
- Experience replay from bus evidence
- Policy gradient optimization with observable metrics

Reference: Microsoft Agent Lightning / RL training frameworks

Usage:
    # Run training episode
    python3 agent_lightning_trainer.py train --episodes 10 --goal "improve code review"

    # Evaluate policy
    python3 agent_lightning_trainer.py eval --policy-path /path/to/policy.json

    # Export experience replay
    python3 agent_lightning_trainer.py export-replay --output replay.ndjson

    # Daemon mode (continuous training)
    python3 agent_lightning_trainer.py daemon --emit-bus
"""
from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

sys.dont_write_bytecode = True

# CMP/RGMA constants (from rhizome_godel_alpha.md)
PHI = 1.618033988749895
CMP_DISCOUNT = 1 / PHI  # ~0.618 per generation
GLOBAL_CMP_FLOOR = 0.236  # 1/PHI^3 - below this, lineage goes extinct


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict, trace_id: str | None = None) -> None:
    """Emit event to Pluribus bus."""
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    cmd = [
        sys.executable,
        str(tool),
        "--bus-dir",
        bus_dir,
        "pub",
        "--topic",
        topic,
        "--kind",
        kind,
        "--level",
        level,
        "--actor",
        actor,
        "--data",
        json.dumps(data, ensure_ascii=False),
    ]
    if trace_id:
        cmd.extend(["--trace-id", trace_id])
    subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


# ============================================================================
# Liveness Monitoring (Omega Gates)
# ============================================================================

class LivenessMonitor(ABC):
    """Abstract base for omega-liveness monitoring."""

    @abstractmethod
    def heartbeat(self, state: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        pass

    @abstractmethod
    def diagnostics(self) -> dict[str, Any]:
        pass


class TrainingLivenessMonitor(LivenessMonitor):
    """Liveness monitor for RL training episodes."""

    def __init__(
        self,
        max_seconds: float = 300.0,
        min_progress_per_minute: float = 0.01,
        max_consecutive_failures: int = 5,
    ):
        self.max_seconds = max_seconds
        self.min_progress_per_minute = min_progress_per_minute
        self.max_consecutive_failures = max_consecutive_failures

        self.start_time = time.time()
        self.last_beat = self.start_time
        self.progress_history: list[tuple[float, float]] = []  # (timestamp, progress)
        self.consecutive_failures = 0
        self.total_failures = 0

    def heartbeat(self, state: dict[str, Any]) -> None:
        now = time.time()
        self.last_beat = now

        progress = state.get("progress", 0.0)
        self.progress_history.append((now, progress))

        # Trim old history (keep last 5 minutes)
        cutoff = now - 300
        self.progress_history = [(t, p) for t, p in self.progress_history if t > cutoff]

        # Track failures
        if state.get("failure"):
            self.consecutive_failures += 1
            self.total_failures += 1
        else:
            self.consecutive_failures = 0

    def is_healthy(self) -> bool:
        now = time.time()
        elapsed = now - self.start_time

        # Time bound check
        if elapsed > self.max_seconds:
            return False

        # Consecutive failure check
        if self.consecutive_failures >= self.max_consecutive_failures:
            return False

        # Progress check (after warmup period)
        if elapsed > 60 and len(self.progress_history) >= 2:
            first_ts, first_prog = self.progress_history[0]
            last_ts, last_prog = self.progress_history[-1]
            minutes = (last_ts - first_ts) / 60
            if minutes > 0:
                progress_rate = (last_prog - first_prog) / minutes
                if progress_rate < self.min_progress_per_minute:
                    return False

        return True

    def diagnostics(self) -> dict[str, Any]:
        now = time.time()
        return {
            "elapsed_s": now - self.start_time,
            "since_last_beat_s": now - self.last_beat,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "progress_points": len(self.progress_history),
            "limit_s": self.max_seconds,
        }


# ============================================================================
# Experience / Trajectory Structures
# ============================================================================

@dataclass
class Transition:
    """Single state-action-reward transition."""
    state: dict[str, Any]
    action: str
    reward: float
    next_state: dict[str, Any]
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Complete training episode."""
    episode_id: str
    trace_id: str | None
    goal: str
    transitions: list[Transition]
    total_reward: float
    success: bool
    latency_s: float
    liveness_healthy: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CMPAwareEpisode(Episode):
    """
    Extended episode with CMP lineage tracking for RGMA integration.

    Tracks:
    - clade_id: identifier for the lineage/clade this episode belongs to
    - parent_cmp: CMP score of the parent lineage node before this episode
    - generation: generation number within the lineage tree
    - cmp_delta: change in CMP after this episode (computed post-hoc)
    """
    clade_id: str = ""
    parent_cmp: float = 0.5
    generation: int = 0
    cmp_delta: float = 0.0
    fitness_components: dict[str, float] = field(default_factory=dict)


@dataclass
class Policy:
    """Agent policy representation."""
    policy_id: str
    version: int
    weights: dict[str, float]
    learning_rate: float
    discount_factor: float
    entropy_coef: float
    created_iso: str
    performance_metrics: dict[str, float] = field(default_factory=dict)


# ============================================================================
# Reward Functions (Bus-First Evidence)
# ============================================================================

class RewardFunction:
    """Base reward function using bus events as evidence."""

    def __init__(self, bus_dir: str | None = None):
        self.bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    def compute_reward(self, transition: Transition, episode_context: dict) -> float:
        """Compute reward for a transition. Override in subclasses."""
        return 0.0

    def get_bus_evidence(self, trace_id: str, topic_filter: str | None = None) -> list[dict]:
        """Extract relevant bus events for reward computation."""
        if not self.bus_dir:
            return []

        events_path = Path(self.bus_dir) / "events.ndjson"
        if not events_path.exists():
            return []

        evidence = []
        for event in iter_ndjson(events_path):
            if event.get("trace_id") != trace_id:
                continue
            if topic_filter and not event.get("topic", "").startswith(topic_filter):
                continue
            evidence.append(event)

        return evidence


class TaskCompletionReward(RewardFunction):
    """Reward based on task completion signals from bus."""

    def __init__(self, bus_dir: str | None = None):
        super().__init__(bus_dir)
        self.completion_reward = 1.0
        self.error_penalty = -0.5
        self.latency_penalty_per_sec = -0.01

    def compute_reward(self, transition: Transition, episode_context: dict) -> float:
        trace_id = episode_context.get("trace_id")
        if not trace_id:
            return 0.0

        evidence = self.get_bus_evidence(trace_id, "strp.")

        reward = 0.0
        for event in evidence:
            topic = event.get("topic", "")
            level = event.get("level", "")

            if topic == "strp.worker.item" and level == "info":
                data = event.get("data", {})
                if data.get("exit_code") == 0:
                    reward += self.completion_reward
                else:
                    reward += self.error_penalty

        # Latency penalty
        latency = transition.info.get("latency_s", 0)
        if latency > 10:
            reward += (latency - 10) * self.latency_penalty_per_sec

        return reward


class QualityReward(RewardFunction):
    """Reward based on output quality signals."""

    def __init__(self, bus_dir: str | None = None):
        super().__init__(bus_dir)
        self.grounding_bonus = 0.3
        self.citation_bonus = 0.1
        self.verification_bonus = 0.5

    def compute_reward(self, transition: Transition, episode_context: dict) -> float:
        trace_id = episode_context.get("trace_id")
        if not trace_id:
            return 0.0

        evidence = self.get_bus_evidence(trace_id)

        reward = 0.0
        for event in evidence:
            topic = event.get("topic", "")
            data = event.get("data", {})

            if topic == "strp.output.grounding":
                if data.get("ok"):
                    reward += self.grounding_bonus
                    citations = len(data.get("citations", []))
                    reward += citations * self.citation_bonus

            if topic.endswith(".verified"):
                if data.get("passed"):
                    reward += self.verification_bonus

        return reward


class CompositeReward(RewardFunction):
    """Combines multiple reward functions."""

    def __init__(self, rewards: list[tuple[RewardFunction, float]], bus_dir: str | None = None):
        super().__init__(bus_dir)
        self.rewards = rewards  # (reward_fn, weight) pairs

    def compute_reward(self, transition: Transition, episode_context: dict) -> float:
        total = 0.0
        for reward_fn, weight in self.rewards:
            total += weight * reward_fn.compute_reward(transition, episode_context)
        return total


# ============================================================================
# CMP-Aware Rewards (RGMA Integration)
# ============================================================================

class CMPReward(RewardFunction):
    """
    Reward based on lineage fitness improvement (CMP delta).

    Per RGMA spec section 3, CMP aggregates descendant returns with
    temporal discount (PHI-based) and spectral smoothing.

    This reward function:
    - Rewards positive CMP delta (lineage improvement)
    - Penalizes CMP regression below floor
    - Uses PHI-weighted fitness components
    """

    def __init__(
        self,
        bus_dir: str | None = None,
        task_completion_weight: float = PHI,  # ~1.618 (primary signal)
        test_coverage_weight: float = 1.0,
        guard_pass_weight: float = 1.0,
        motif_recurrence_weight: float = 1 / PHI,  # ~0.618
        mdl_complexity_weight: float = 1 / (PHI ** 2),  # ~0.382
    ):
        super().__init__(bus_dir)
        self.weights = {
            "task_completion": task_completion_weight,
            "test_coverage": test_coverage_weight,
            "guard_pass_rate": guard_pass_weight,
            "motif_recurrence": motif_recurrence_weight,
            "mdl_complexity": mdl_complexity_weight,
        }
        self._lineage_cache: dict[str, float] = {}  # clade_id -> last known CMP

    def compute_fitness_components(
        self,
        transition: Transition,
        episode_context: dict,
    ) -> dict[str, float]:
        """
        Compute individual fitness components for CMP calculation.

        Returns dict of component_name -> score (0.0-1.0 each)
        """
        components: dict[str, float] = {}

        # Task completion from bus evidence
        trace_id = episode_context.get("trace_id")
        if trace_id:
            evidence = self.get_bus_evidence(trace_id)
            completions = sum(
                1 for e in evidence
                if e.get("topic", "").endswith(".complete") and e.get("level") == "info"
            )
            errors = sum(
                1 for e in evidence
                if e.get("level") == "error"
            )
            components["task_completion"] = min(1.0, completions / max(1, completions + errors))
        else:
            # Infer from transition
            result = transition.info.get("result", "")
            components["task_completion"] = 1.0 if "complete" in result.lower() else 0.5

        # Test coverage (inferred from verification events)
        if trace_id:
            evidence = self.get_bus_evidence(trace_id, "test.")
            test_passes = sum(1 for e in evidence if "pass" in e.get("data", {}).get("status", ""))
            test_fails = sum(1 for e in evidence if "fail" in e.get("data", {}).get("status", ""))
            if test_passes + test_fails > 0:
                components["test_coverage"] = test_passes / (test_passes + test_fails)
            else:
                components["test_coverage"] = 0.5  # neutral when no tests
        else:
            components["test_coverage"] = 0.5

        # Guard pass rate (from guard ladder events)
        if trace_id:
            evidence = self.get_bus_evidence(trace_id, "guard.")
            guard_passes = sum(1 for e in evidence if e.get("data", {}).get("passed", False))
            guard_checks = len(evidence)
            components["guard_pass_rate"] = guard_passes / max(1, guard_checks)
        else:
            components["guard_pass_rate"] = 1.0  # assume pass if no guards checked

        # Motif recurrence (placeholder - requires motif tracking)
        components["motif_recurrence"] = episode_context.get("motif_recurrence", 0.5)

        # MDL complexity (lower is better, invert for reward)
        action_count = len(transition.state.get("history", []))
        # Normalize: fewer actions = simpler = higher score
        components["mdl_complexity"] = 1.0 / (1.0 + action_count * 0.1)

        return components

    def compute_cmp_score(self, components: dict[str, float]) -> float:
        """
        Compute CMP score from fitness components using PHI-weighted aggregation.

        Formula: CMP = sum(weight_i * component_i) / sum(weights)
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for component_name, weight in self.weights.items():
            if component_name in components:
                weighted_sum += weight * components[component_name]
                weight_total += weight

        if weight_total == 0:
            return 0.5  # neutral

        return weighted_sum / weight_total

    def compute_reward(self, transition: Transition, episode_context: dict) -> float:
        """
        Compute reward based on CMP delta (improvement over parent lineage).
        """
        components = self.compute_fitness_components(transition, episode_context)
        current_cmp = self.compute_cmp_score(components)

        # Get parent CMP from context
        parent_cmp = episode_context.get("parent_cmp", 0.5)
        clade_id = episode_context.get("clade_id")

        # Use cached value if available
        if clade_id and clade_id in self._lineage_cache:
            parent_cmp = self._lineage_cache[clade_id]

        # Compute CMP delta
        cmp_delta = current_cmp - parent_cmp

        # Update cache
        if clade_id:
            self._lineage_cache[clade_id] = current_cmp

        # Store components in episode context for later use
        episode_context["_cmp_components"] = components
        episode_context["_cmp_current"] = current_cmp
        episode_context["_cmp_delta"] = cmp_delta

        # Reward is scaled CMP delta with floor protection
        reward = cmp_delta * PHI  # Scale by PHI for emphasis

        # Additional penalty for falling below global floor
        if current_cmp < GLOBAL_CMP_FLOOR:
            reward -= (GLOBAL_CMP_FLOOR - current_cmp) * 2.0

        return reward

    def get_lineage_cmp(self, clade_id: str) -> float:
        """Get cached CMP for a lineage."""
        return self._lineage_cache.get(clade_id, 0.5)


# ============================================================================
# CMP-Adaptive Policy (Thompson Sampling)
# ============================================================================

@dataclass
class BetaPrior:
    """Beta distribution prior for Thompson sampling."""
    alpha: float = 1.0  # successes + 1
    beta: float = 1.0   # failures + 1

    def sample(self) -> float:
        """Sample from Beta(alpha, beta) distribution."""
        # Use inverse transform sampling for Beta distribution
        # Approximation using ratio of gamma random variables
        import random as _random
        # Generate gamma variates
        def gamma_sample(shape: float) -> float:
            if shape < 1:
                return gamma_sample(1 + shape) * (_random.random() ** (1 / shape))
            d = shape - 1/3
            c = 1 / math.sqrt(9 * d)
            while True:
                x = _random.gauss(0, 1)
                v = (1 + c * x) ** 3
                if v > 0:
                    u = _random.random()
                    if u < 1 - 0.0331 * (x ** 2) ** 2:
                        return d * v
                    if math.log(u) < 0.5 * x ** 2 + d * (1 - v + math.log(v)):
                        return d * v

        x = gamma_sample(self.alpha)
        y = gamma_sample(self.beta)
        return x / (x + y) if (x + y) > 0 else 0.5

    def update(self, success: bool, magnitude: float = 1.0) -> None:
        """Update prior based on observation."""
        if success:
            self.alpha += magnitude
        else:
            self.beta += magnitude


class CMPAdaptivePolicy:
    """
    Thompson-sampled action selection for CMP-aware training.

    Per RGMA spec section 3.3, uses multi-armed bandit over lineages
    with Beta priors updated by CMP outcomes.

    This policy:
    - Maintains Beta priors per action
    - Samples from posteriors to select actions (exploration)
    - Updates priors based on CMP improvement (exploitation)
    - Adapts exploration rate based on lineage promise
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
    ):
        self.actions = actions or [
            "analyze_requirements",
            "generate_code",
            "verify_output",
            "refine_solution",
            "complete_task",
            "request_feedback",
            "explore_alternative",
        ]
        self.priors: dict[str, BetaPrior] = {
            action: BetaPrior(initial_alpha, initial_beta)
            for action in self.actions
        }
        self.action_counts: dict[str, int] = {action: 0 for action in self.actions}
        self.total_selections = 0

    def select_action(
        self,
        state: dict[str, Any],
        clade_promise: float = 0.5,
    ) -> str:
        """
        Select action using Thompson sampling.

        Args:
            state: Current state dict
            clade_promise: Current lineage's promise score (0-1).
                          Higher promise -> more exploitation.
        """
        self.total_selections += 1

        # Sample from each action's posterior
        samples = {
            action: prior.sample()
            for action, prior in self.priors.items()
        }

        # Apply state-based modifiers
        step = state.get("step", 0)
        goal = state.get("goal", "").lower()

        # Boost certain actions based on state
        if step == 0:
            samples["analyze_requirements"] *= 1.5
        elif step < 3:
            samples["generate_code"] *= 1.3
        elif step < 5:
            samples["verify_output"] *= 1.2
        elif step > 7:
            samples["complete_task"] *= 1.5

        # High-promise lineages: prefer exploitation (reliable actions)
        if clade_promise > 0.7:
            for action in ["verify_output", "complete_task"]:
                samples[action] *= 1.2

        # Low-promise lineages: prefer exploration (novel actions)
        if clade_promise < 0.3:
            samples["explore_alternative"] *= 1.5
            samples["request_feedback"] *= 1.3

        # Select action with highest sampled value
        selected = max(samples, key=samples.get)
        self.action_counts[selected] += 1

        return selected

    def update(
        self,
        action: str,
        cmp_delta: float,
        success: bool = True,
    ) -> None:
        """
        Update action prior based on CMP outcome.

        Args:
            action: The action that was taken
            cmp_delta: Change in CMP after the action
            success: Whether the action was successful
        """
        if action not in self.priors:
            self.priors[action] = BetaPrior()
            self.action_counts[action] = 0

        # Magnitude based on CMP delta
        magnitude = abs(cmp_delta) * PHI

        if cmp_delta > 0:
            # Positive CMP delta = success
            self.priors[action].update(success=True, magnitude=magnitude)
        else:
            # Negative CMP delta = failure
            self.priors[action].update(success=False, magnitude=magnitude)

    def get_exploration_rate(self) -> float:
        """
        Compute current exploration rate based on action distribution.

        Returns value 0-1: higher = more exploration happening.
        """
        if self.total_selections == 0:
            return 1.0

        # Entropy-based exploration measure
        probs = [
            count / self.total_selections
            for count in self.action_counts.values()
            if count > 0
        ]
        if not probs:
            return 1.0

        # Normalized entropy
        max_entropy = math.log(len(self.actions))
        entropy = -sum(p * math.log(p) for p in probs)

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic info about the policy."""
        return {
            "total_selections": self.total_selections,
            "action_counts": dict(self.action_counts),
            "exploration_rate": self.get_exploration_rate(),
            "priors": {
                action: {"alpha": p.alpha, "beta": p.beta}
                for action, p in self.priors.items()
            },
        }


# ============================================================================
# Training Control Functions
# ============================================================================

def should_pause_training(
    episodes: list[Episode],
    *,
    min_episodes: int = 10,
    cmp_decay_threshold: float = 0.1,
    decay_window: int = 5,
    floor_breaches: int = 3,
) -> tuple[bool, str]:
    """
    Determine if training should pause based on CMP decay patterns.

    Per RGMA spec, training should pause when:
    - CMP consistently decaying over window
    - CMP below global floor for multiple episodes
    - No improvement after sufficient exploration

    Args:
        episodes: List of completed episodes
        min_episodes: Minimum episodes before considering pause
        cmp_decay_threshold: Max acceptable CMP decay rate
        decay_window: Number of recent episodes to consider
        floor_breaches: Max consecutive floor breaches before pause

    Returns:
        (should_pause, reason)
    """
    if len(episodes) < min_episodes:
        return False, "insufficient_episodes"

    # Extract CMP info from recent episodes
    recent = episodes[-decay_window:]
    cmp_values = []

    for ep in recent:
        if isinstance(ep, CMPAwareEpisode):
            # Get CMP from metadata or compute from fitness
            cmp = ep.parent_cmp + ep.cmp_delta
        else:
            # Estimate from reward
            cmp = 0.5 + ep.total_reward * 0.1
        cmp_values.append(max(0.0, min(1.0, cmp)))

    if len(cmp_values) < 2:
        return False, "insufficient_cmp_data"

    # Check for consistent decay
    decay_count = sum(
        1 for i in range(1, len(cmp_values))
        if cmp_values[i] < cmp_values[i-1]
    )
    decay_rate = decay_count / (len(cmp_values) - 1)

    if decay_rate > (1 - cmp_decay_threshold):
        return True, f"consistent_cmp_decay_{decay_rate:.2f}"

    # Check for floor breaches
    floor_breach_count = sum(
        1 for cmp in cmp_values
        if cmp < GLOBAL_CMP_FLOOR
    )

    if floor_breach_count >= floor_breaches:
        return True, f"floor_breaches_{floor_breach_count}"

    # Check for stagnation (no improvement)
    if len(cmp_values) >= 5:
        improvement = cmp_values[-1] - cmp_values[0]
        if improvement < -cmp_decay_threshold:
            return True, f"net_regression_{improvement:.3f}"

    return False, "healthy"


# ============================================================================
# Policy Gradient Trainer
# ============================================================================

class AgentLightningTrainer:
    """
    RL trainer for agents using policy gradient methods.

    Integrates with:
    - Bus-first evidence for rewards
    - Omega-liveness gates for episode health
    - Experience replay buffer
    - CMP-LARGE lineage tracking (RGMA integration)
    """

    def __init__(
        self,
        bus_dir: str | None = None,
        storage_dir: Path | None = None,
        reward_fn: RewardFunction | None = None,
        *,
        cmp_enabled: bool = False,
        clade_id: str | None = None,
    ):
        self.bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        self.actor = default_actor()

        root = Path("/pluribus")
        self.storage_dir = storage_dir or (root / ".pluribus" / "agent_lightning")
        ensure_dir(self.storage_dir)

        self.replay_path = self.storage_dir / "replay.ndjson"
        self.policy_path = self.storage_dir / "policy.json"
        self.metrics_path = self.storage_dir / "metrics.ndjson"

        self.reward_fn = reward_fn or CompositeReward([
            (TaskCompletionReward(self.bus_dir), 0.6),
            (QualityReward(self.bus_dir), 0.4),
        ], self.bus_dir)

        self.policy = self._load_or_init_policy()
        self.episodes: list[Episode] = []
        self.training_metrics: list[dict] = []

        # CMP-aware extensions
        self.cmp_enabled = cmp_enabled
        self.clade_id = clade_id or str(uuid.uuid4())
        self.generation = 0
        self.cmp_reward: Optional[CMPReward] = None
        self.adaptive_policy: Optional[CMPAdaptivePolicy] = None

        if cmp_enabled:
            self._init_cmp_components()

    def _init_cmp_components(self) -> None:
        """Initialize CMP-aware components for RGMA integration."""
        self.cmp_reward = CMPReward(self.bus_dir)
        self.adaptive_policy = CMPAdaptivePolicy()

        # Add CMP reward to composite if using default
        if isinstance(self.reward_fn, CompositeReward):
            self.reward_fn.rewards.append((self.cmp_reward, 0.4))

        self._emit(
            "lightning.cmp.initialized",
            kind="log",
            level="info",
            data={
                "clade_id": self.clade_id,
                "generation": self.generation,
            },
        )

    def _record_to_cmp_large(
        self,
        episode: Episode,
        cmp_score: float,
        cmp_delta: float,
        fitness_components: dict[str, float],
    ) -> Optional[str]:
        """
        Record episode to CMP-LARGE lineage ledger.

        Calls cmp_large.record_episode() to persist lineage data
        and emits rgma.lineage.cmp_updated bus event.
        """
        if not self.cmp_enabled:
            return None

        try:
            # Import cmp_large from same directory
            from nucleus.tools import cmp_large

            record_id = cmp_large.record_episode(
                episode_id=episode.episode_id,
                clade_id=self.clade_id,
                generation=self.generation,
                cmp_score=cmp_score,
                cmp_delta=cmp_delta,
                fitness_components=fitness_components,
                bus_dir=self.bus_dir,
                actor=self.actor,
                metadata={
                    "goal": episode.goal[:200],
                    "success": episode.success,
                    "total_reward": episode.total_reward,
                    "policy_version": self.policy.version,
                },
            )

            return record_id

        except ImportError:
            # Fallback: emit bus event directly
            self._emit(
                "rgma.lineage.cmp_updated",
                kind="metric",
                level="info",
                data={
                    "episode_id": episode.episode_id,
                    "clade_id": self.clade_id,
                    "generation": self.generation,
                    "cmp_score": cmp_score,
                    "cmp_delta": cmp_delta,
                    "fitness_components": fitness_components,
                },
                trace_id=episode.trace_id,
            )
            return None

        except Exception as e:
            self._emit(
                "lightning.cmp.record_error",
                kind="log",
                level="warn",
                data={
                    "episode_id": episode.episode_id,
                    "error": str(e),
                },
                trace_id=episode.trace_id,
            )
            return None

    def run_cmp_episode(
        self,
        goal: str,
        *,
        max_steps: int = 20,
        trace_id: str | None = None,
        executor: Callable[[str, dict], tuple[str, dict]] | None = None,
    ) -> CMPAwareEpisode:
        """
        Run a CMP-aware training episode with lineage tracking.

        This method extends run_episode with:
        - CMP fitness component tracking
        - Lineage-aware context
        - Recording to CMP-LARGE ledger
        - Adaptive policy updates

        Args:
            goal: The task goal for this episode
            max_steps: Maximum transitions in episode
            trace_id: Trace ID for correlation
            executor: Function(action, state) -> (result, new_state)

        Returns:
            CMPAwareEpisode with lineage metadata
        """
        if not self.cmp_enabled:
            self._init_cmp_components()
            self.cmp_enabled = True

        episode_id = str(uuid.uuid4())
        trace_id = trace_id or episode_id
        t0 = time.perf_counter()

        # Get parent CMP from lineage
        parent_cmp = 0.5
        if self.cmp_reward:
            parent_cmp = self.cmp_reward.get_lineage_cmp(self.clade_id)

        self._emit(
            "lightning.cmp_episode.start",
            kind="metric",
            level="info",
            data={
                "episode_id": episode_id,
                "goal": goal[:200],
                "clade_id": self.clade_id,
                "generation": self.generation,
                "parent_cmp": parent_cmp,
                "policy_version": self.policy.version,
            },
            trace_id=trace_id,
        )

        # Initialize liveness monitor
        monitor = TrainingLivenessMonitor(
            max_seconds=300.0,
            min_progress_per_minute=0.01,
            max_consecutive_failures=5,
        )

        transitions: list[Transition] = []
        state = {"goal": goal, "step": 0, "history": []}
        episode_context = {
            "trace_id": trace_id,
            "episode_id": episode_id,
            "clade_id": self.clade_id,
            "parent_cmp": parent_cmp,
            "generation": self.generation,
        }
        total_reward = 0.0
        success = False
        fitness_components: dict[str, float] = {}

        for step in range(max_steps):
            if not monitor.is_healthy():
                self._emit(
                    "lightning.cmp_episode.liveness_fail",
                    kind="metric",
                    level="warn",
                    data={
                        "episode_id": episode_id,
                        "step": step,
                        "diagnostics": monitor.diagnostics(),
                    },
                    trace_id=trace_id,
                )
                break

            # Select action using adaptive policy if available
            if self.adaptive_policy:
                action = self.adaptive_policy.select_action(
                    state,
                    clade_promise=parent_cmp,
                )
            else:
                action = self._select_action(state)

            # Execute action
            if executor:
                result, next_state = executor(action, state)
            else:
                result = f"executed_{action}"
                next_state = {
                    **state,
                    "step": step + 1,
                    "history": state.get("history", []) + [action],
                }

            done = step >= max_steps - 1 or "complete" in result.lower()

            transition = Transition(
                state=state,
                action=action,
                reward=0.0,
                next_state=next_state,
                done=done,
                info={"result": result, "step": step},
            )

            # Compute reward (CMP reward tracks components)
            transition.reward = self.reward_fn.compute_reward(transition, episode_context)
            total_reward += transition.reward

            # Extract CMP components from context
            if "_cmp_components" in episode_context:
                fitness_components = episode_context["_cmp_components"]

            transitions.append(transition)

            # Update adaptive policy
            if self.adaptive_policy:
                cmp_delta = episode_context.get("_cmp_delta", 0.0)
                self.adaptive_policy.update(action, cmp_delta, success=transition.reward > 0)

            monitor.heartbeat({
                "progress": (step + 1) / max_steps,
                "reward": transition.reward,
                "failure": transition.reward < 0,
            })

            state = next_state
            if done:
                success = "error" not in result.lower()
                break

        latency_s = time.perf_counter() - t0
        liveness_healthy = monitor.is_healthy()

        # Compute final CMP
        cmp_current = episode_context.get("_cmp_current", parent_cmp)
        cmp_delta = episode_context.get("_cmp_delta", 0.0)

        # Create CMP-aware episode
        episode = CMPAwareEpisode(
            episode_id=episode_id,
            trace_id=trace_id,
            goal=goal,
            transitions=transitions,
            total_reward=total_reward,
            success=success,
            latency_s=latency_s,
            liveness_healthy=liveness_healthy,
            metadata={
                "policy_version": self.policy.version,
                "steps": len(transitions),
                "liveness_diagnostics": monitor.diagnostics(),
            },
            clade_id=self.clade_id,
            parent_cmp=parent_cmp,
            generation=self.generation,
            cmp_delta=cmp_delta,
            fitness_components=fitness_components,
        )

        self.episodes.append(episode)
        self._save_episode_to_replay(episode)

        # Record to CMP-LARGE ledger
        record_id = self._record_to_cmp_large(
            episode,
            cmp_score=cmp_current,
            cmp_delta=cmp_delta,
            fitness_components=fitness_components,
        )

        # Increment generation for next episode
        self.generation += 1

        self._emit(
            "lightning.cmp_episode.complete",
            kind="metric",
            level="info" if success else "warn",
            data={
                "episode_id": episode_id,
                "goal": goal[:200],
                "clade_id": self.clade_id,
                "generation": self.generation - 1,
                "cmp_score": cmp_current,
                "cmp_delta": cmp_delta,
                "total_reward": total_reward,
                "success": success,
                "record_id": record_id,
            },
            trace_id=trace_id,
        )

        return episode

    def check_should_pause(self) -> tuple[bool, str]:
        """
        Check if training should pause based on CMP patterns.

        Returns:
            (should_pause, reason)
        """
        return should_pause_training(self.episodes)

    def _load_or_init_policy(self) -> Policy:
        """Load existing policy or initialize new one."""
        if self.policy_path.exists():
            try:
                data = json.loads(self.policy_path.read_text(encoding="utf-8"))
                return Policy(
                    policy_id=data["policy_id"],
                    version=data["version"],
                    weights=data["weights"],
                    learning_rate=data["learning_rate"],
                    discount_factor=data["discount_factor"],
                    entropy_coef=data["entropy_coef"],
                    created_iso=data["created_iso"],
                    performance_metrics=data.get("performance_metrics", {}),
                )
            except Exception:
                pass

        return Policy(
            policy_id=str(uuid.uuid4()),
            version=1,
            weights={
                "task_completion": 1.0,
                "latency": 0.5,
                "quality": 0.8,
                "exploration": 0.1,
            },
            learning_rate=0.001,
            discount_factor=0.99,
            entropy_coef=0.01,
            created_iso=now_iso_utc(),
        )

    def _save_policy(self) -> None:
        """Save policy to disk."""
        self.policy_path.write_text(
            json.dumps(asdict(self.policy), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _emit(self, topic: str, kind: str, level: str, data: dict, trace_id: str | None = None) -> None:
        """Emit training event to bus."""
        if self.bus_dir:
            emit_bus(self.bus_dir, topic=topic, kind=kind, level=level, actor=self.actor, data=data, trace_id=trace_id)

    def run_episode(
        self,
        goal: str,
        *,
        max_steps: int = 20,
        trace_id: str | None = None,
        executor: Callable[[str, dict], tuple[str, dict]] | None = None,
    ) -> Episode:
        """
        Run a single training episode.

        Args:
            goal: The task goal for this episode
            max_steps: Maximum transitions in episode
            trace_id: Trace ID for correlation
            executor: Function(action, state) -> (result, new_state)
        """
        episode_id = str(uuid.uuid4())
        trace_id = trace_id or episode_id
        t0 = time.perf_counter()

        self._emit(
            "lightning.episode.start",
            kind="metric",
            level="info",
            data={
                "episode_id": episode_id,
                "goal": goal[:200],
                "max_steps": max_steps,
                "policy_version": self.policy.version,
            },
            trace_id=trace_id,
        )

        # Initialize liveness monitor
        monitor = TrainingLivenessMonitor(
            max_seconds=300.0,
            min_progress_per_minute=0.01,
            max_consecutive_failures=5,
        )

        transitions: list[Transition] = []
        state = {"goal": goal, "step": 0, "history": []}
        episode_context = {"trace_id": trace_id, "episode_id": episode_id}
        total_reward = 0.0
        success = False

        for step in range(max_steps):
            if not monitor.is_healthy():
                self._emit(
                    "lightning.episode.liveness_fail",
                    kind="metric",
                    level="warn",
                    data={
                        "episode_id": episode_id,
                        "step": step,
                        "diagnostics": monitor.diagnostics(),
                    },
                    trace_id=trace_id,
                )
                break

            # Select action using policy
            action = self._select_action(state)

            # Execute action
            if executor:
                result, next_state = executor(action, state)
            else:
                # Default mock executor
                result = f"executed_{action}"
                next_state = {
                    **state,
                    "step": step + 1,
                    "history": state.get("history", []) + [action],
                }

            # Determine if episode is done
            done = step >= max_steps - 1 or "complete" in result.lower()

            # Create transition
            transition = Transition(
                state=state,
                action=action,
                reward=0.0,  # Will be computed
                next_state=next_state,
                done=done,
                info={"result": result, "step": step},
            )

            # Compute reward
            transition.reward = self.reward_fn.compute_reward(transition, episode_context)
            total_reward += transition.reward

            transitions.append(transition)

            # Update liveness
            monitor.heartbeat({
                "progress": (step + 1) / max_steps,
                "reward": transition.reward,
                "failure": transition.reward < 0,
            })

            self._emit(
                "lightning.step.complete",
                kind="metric",
                level="debug",
                data={
                    "episode_id": episode_id,
                    "step": step,
                    "action": action[:100],
                    "reward": transition.reward,
                    "total_reward": total_reward,
                },
                trace_id=trace_id,
            )

            state = next_state
            if done:
                success = "error" not in result.lower()
                break

        latency_s = time.perf_counter() - t0
        liveness_healthy = monitor.is_healthy()

        episode = Episode(
            episode_id=episode_id,
            trace_id=trace_id,
            goal=goal,
            transitions=transitions,
            total_reward=total_reward,
            success=success,
            latency_s=latency_s,
            liveness_healthy=liveness_healthy,
            metadata={
                "policy_version": self.policy.version,
                "steps": len(transitions),
                "liveness_diagnostics": monitor.diagnostics(),
            },
        )

        self.episodes.append(episode)
        self._save_episode_to_replay(episode)

        self._emit(
            "lightning.episode.complete",
            kind="metric",
            level="info" if success else "warn",
            data={
                "episode_id": episode_id,
                "goal": goal[:200],
                "total_reward": total_reward,
                "steps": len(transitions),
                "success": success,
                "latency_s": latency_s,
                "liveness_healthy": liveness_healthy,
            },
            trace_id=trace_id,
        )

        return episode

    def _select_action(self, state: dict) -> str:
        """Select action using policy (softmax over action weights)."""
        goal = state.get("goal", "")
        step = state.get("step", 0)

        # Simplified action space for demonstration
        actions = [
            "analyze_requirements",
            "generate_code",
            "verify_output",
            "refine_solution",
            "complete_task",
        ]

        # Compute action probabilities using policy weights
        exploration = self.policy.weights.get("exploration", 0.1)

        # Epsilon-greedy with policy-based preferences
        if random.random() < exploration:
            return random.choice(actions)

        # Deterministic selection based on state
        if step == 0:
            return "analyze_requirements"
        elif step < 3:
            return "generate_code"
        elif step < 5:
            return "verify_output"
        else:
            return "complete_task"

    def _save_episode_to_replay(self, episode: Episode) -> None:
        """Save episode to replay buffer."""
        record = {
            "episode_id": episode.episode_id,
            "trace_id": episode.trace_id,
            "goal": episode.goal,
            "total_reward": episode.total_reward,
            "success": episode.success,
            "latency_s": episode.latency_s,
            "liveness_healthy": episode.liveness_healthy,
            "transitions": [asdict(t) for t in episode.transitions],
            "metadata": episode.metadata,
            "ts": time.time(),
            "iso": now_iso_utc(),
        }
        append_ndjson(self.replay_path, record)

    def update_policy(self, batch_size: int = 32) -> dict:
        """
        Update policy using collected episodes.

        Returns metrics about the update.
        """
        if len(self.episodes) < batch_size:
            return {"skipped": True, "reason": f"need {batch_size} episodes, have {len(self.episodes)}"}

        # Sample batch
        batch = random.sample(self.episodes, min(batch_size, len(self.episodes)))

        # Compute advantages
        returns = []
        for episode in batch:
            episode_return = sum(t.reward for t in episode.transitions)
            returns.append(episode_return)

        mean_return = sum(returns) / len(returns) if returns else 0
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns)) if len(returns) > 1 else 1

        # Normalize advantages
        advantages = [(r - mean_return) / (std_return + 1e-8) for r in returns]

        # Simple policy gradient update (demonstration)
        lr = self.policy.learning_rate
        for i, episode in enumerate(batch):
            advantage = advantages[i]

            # Update weights based on episode outcome
            if episode.success:
                self.policy.weights["task_completion"] += lr * advantage
            if episode.liveness_healthy:
                self.policy.weights["quality"] += lr * advantage * 0.5

        # Clip weights
        for key in self.policy.weights:
            self.policy.weights[key] = max(0.01, min(2.0, self.policy.weights[key]))

        # Update version
        self.policy.version += 1
        self.policy.performance_metrics = {
            "mean_return": mean_return,
            "std_return": std_return,
            "batch_size": len(batch),
            "success_rate": sum(1 for e in batch if e.success) / len(batch),
        }

        self._save_policy()

        metrics = {
            "policy_version": self.policy.version,
            "mean_return": mean_return,
            "std_return": std_return,
            "batch_size": len(batch),
            "success_rate": self.policy.performance_metrics["success_rate"],
        }

        self._emit(
            "lightning.policy.update",
            kind="metric",
            level="info",
            data=metrics,
        )

        self.training_metrics.append({"ts": time.time(), **metrics})
        append_ndjson(self.metrics_path, {"kind": "policy_update", "ts": time.time(), **metrics})

        return metrics

    def evaluate_policy(self, goals: list[str], episodes_per_goal: int = 3) -> dict:
        """Evaluate current policy on test goals."""
        results = []

        self._emit(
            "lightning.eval.start",
            kind="metric",
            level="info",
            data={
                "goals_count": len(goals),
                "episodes_per_goal": episodes_per_goal,
                "policy_version": self.policy.version,
            },
        )

        for goal in goals:
            goal_results = []
            for _ in range(episodes_per_goal):
                episode = self.run_episode(goal, max_steps=10)
                goal_results.append({
                    "success": episode.success,
                    "reward": episode.total_reward,
                    "steps": len(episode.transitions),
                })
            results.append({
                "goal": goal[:100],
                "success_rate": sum(1 for r in goal_results if r["success"]) / len(goal_results),
                "mean_reward": sum(r["reward"] for r in goal_results) / len(goal_results),
            })

        eval_summary = {
            "policy_version": self.policy.version,
            "overall_success_rate": sum(r["success_rate"] for r in results) / len(results) if results else 0,
            "overall_mean_reward": sum(r["mean_reward"] for r in results) / len(results) if results else 0,
            "goals_evaluated": len(goals),
        }

        self._emit(
            "lightning.eval.complete",
            kind="metric",
            level="info",
            data=eval_summary,
        )

        return {"summary": eval_summary, "per_goal": results}


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Run training episodes."""
    trainer = AgentLightningTrainer(bus_dir=args.bus_dir)

    emit_bus(
        args.bus_dir,
        topic="lightning.train.start",
        kind="log",
        level="info",
        actor=default_actor(),
        data={
            "episodes": args.episodes,
            "goal": args.goal[:200],
            "update_interval": args.update_interval,
        },
    )

    for i in range(args.episodes):
        episode = trainer.run_episode(args.goal, max_steps=args.max_steps)
        sys.stdout.write(f"Episode {i+1}/{args.episodes}: reward={episode.total_reward:.3f} success={episode.success}\n")

        if (i + 1) % args.update_interval == 0:
            metrics = trainer.update_policy()
            if not metrics.get("skipped"):
                sys.stdout.write(f"  Policy update: v{metrics['policy_version']} mean_return={metrics['mean_return']:.3f}\n")

    emit_bus(
        args.bus_dir,
        topic="lightning.train.complete",
        kind="response",
        level="info",
        actor=default_actor(),
        data={
            "episodes_completed": args.episodes,
            "policy_version": trainer.policy.version,
            "final_weights": trainer.policy.weights,
        },
    )

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate policy."""
    trainer = AgentLightningTrainer(bus_dir=args.bus_dir)

    if args.policy_path:
        # Load external policy
        try:
            data = json.loads(Path(args.policy_path).read_text(encoding="utf-8"))
            trainer.policy = Policy(**data)
        except Exception as e:
            sys.stderr.write(f"Failed to load policy: {e}\n")
            return 1

    goals = args.goals.split(",") if args.goals else ["summarize content", "generate code", "review PR"]

    results = trainer.evaluate_policy(goals, episodes_per_goal=args.episodes_per_goal)

    if args.json:
        sys.stdout.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    else:
        summary = results["summary"]
        sys.stdout.write(f"Policy v{summary['policy_version']} Evaluation:\n")
        sys.stdout.write(f"  Success Rate: {summary['overall_success_rate']:.2%}\n")
        sys.stdout.write(f"  Mean Reward: {summary['overall_mean_reward']:.3f}\n")

    return 0


def cmd_export_replay(args: argparse.Namespace) -> int:
    """Export experience replay buffer."""
    trainer = AgentLightningTrainer(bus_dir=args.bus_dir)

    if not trainer.replay_path.exists():
        sys.stderr.write("No replay buffer found\n")
        return 1

    output = Path(args.output)
    count = 0

    with output.open("w", encoding="utf-8") as f:
        for record in iter_ndjson(trainer.replay_path):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    sys.stdout.write(f"Exported {count} episodes to {output}\n")
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Run training daemon."""
    trainer = AgentLightningTrainer(bus_dir=args.bus_dir)

    emit_bus(
        args.bus_dir,
        topic="lightning.daemon.start",
        kind="log",
        level="info",
        actor=default_actor(),
        data={
            "interval_s": args.interval,
            "policy_version": trainer.policy.version,
        },
    )

    sys.stdout.write(f"Agent Lightning daemon started (interval={args.interval}s)\n")

    while True:
        try:
            # Run exploration episode
            goals = [
                "improve code quality",
                "reduce response latency",
                "enhance documentation",
                "fix potential bugs",
            ]
            goal = random.choice(goals)

            episode = trainer.run_episode(goal, max_steps=10)

            # Periodic policy updates
            if len(trainer.episodes) >= 10 and len(trainer.episodes) % 10 == 0:
                trainer.update_policy()

            emit_bus(
                args.bus_dir,
                topic="lightning.daemon.heartbeat",
                kind="metric",
                level="info",
                actor=default_actor(),
                data={
                    "episodes_total": len(trainer.episodes),
                    "policy_version": trainer.policy.version,
                    "last_reward": episode.total_reward,
                    "last_success": episode.success,
                },
            )

        except Exception as e:
            emit_bus(
                args.bus_dir,
                topic="lightning.daemon.error",
                kind="log",
                level="error",
                actor=default_actor(),
                data={"error": str(e)},
            )

        time.sleep(args.interval)


def cmd_status(args: argparse.Namespace) -> int:
    """Show trainer status."""
    trainer = AgentLightningTrainer(bus_dir=args.bus_dir)

    status = {
        "policy_id": trainer.policy.policy_id,
        "policy_version": trainer.policy.version,
        "weights": trainer.policy.weights,
        "performance": trainer.policy.performance_metrics,
        "replay_exists": trainer.replay_path.exists(),
        "storage_dir": str(trainer.storage_dir),
    }

    if args.json:
        sys.stdout.write(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"Policy: {status['policy_id'][:8]} v{status['policy_version']}\n")
        sys.stdout.write(f"Weights: {json.dumps(status['weights'])}\n")
        if status["performance"]:
            sys.stdout.write(f"Performance: {json.dumps(status['performance'])}\n")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_lightning_trainer.py",
        description="RL training framework for agents with bus-first evidence",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory (or PLURIBUS_BUS_DIR)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    train_p = sub.add_parser("train", help="Run training episodes")
    train_p.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    train_p.add_argument("--goal", required=True, help="Training goal/task")
    train_p.add_argument("--max-steps", type=int, default=20, help="Max steps per episode")
    train_p.add_argument("--update-interval", type=int, default=5, help="Episodes between policy updates")
    train_p.set_defaults(func=cmd_train)

    # eval
    eval_p = sub.add_parser("eval", help="Evaluate policy")
    eval_p.add_argument("--policy-path", default=None, help="Path to policy JSON")
    eval_p.add_argument("--goals", default=None, help="Comma-separated test goals")
    eval_p.add_argument("--episodes-per-goal", type=int, default=3, help="Episodes per goal")
    eval_p.add_argument("--json", action="store_true", help="Output as JSON")
    eval_p.set_defaults(func=cmd_eval)

    # export-replay
    export_p = sub.add_parser("export-replay", help="Export experience replay")
    export_p.add_argument("--output", required=True, help="Output file path")
    export_p.set_defaults(func=cmd_export_replay)

    # daemon
    daemon_p = sub.add_parser("daemon", help="Run training daemon")
    daemon_p.add_argument("--interval", type=float, default=60.0, help="Interval between episodes")
    daemon_p.set_defaults(func=cmd_daemon)

    # status
    status_p = sub.add_parser("status", help="Show trainer status")
    status_p.add_argument("--json", action="store_true", help="Output as JSON")
    status_p.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
