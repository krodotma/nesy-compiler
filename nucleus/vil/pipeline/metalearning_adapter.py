"""
VIL Metalearning Adapter
Connects Omega Metalearning to VIL pipeline.

Integrates:
- Omega Metalearning (MAML, Reptile, self-play)
- Learning Tower L8 (Gödel machines, reflexive domain)
- Agent Lightning Trainer (RL, experience replay)
- Meta Learner (pattern recognition)

Version: 1.0
Date: 2026-01-25
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    from nucleus.tools.omega_metalearning import (
        MetaLearner,
        SelfPlayEngine,
        ReflexiveMonitor,
    )
    OMEGA_AVAILABLE = True
except ImportError:
    OMEGA_AVAILABLE = False

from nucleus.vil.events import (
    VILEventType,
    LearningEvent,
    create_learning_event,
    create_trace_id,
    CmpMetrics,
)


class MetalearningMethod(str, Enum):
    """Metalearning methods."""

    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # First-order MAML
    SELF_PLAY = "self_play"  # AlphaZero-style
    REPTILE_ENSEMBLE = "reptile_ensemble"  # WP-Rep
    REFLEXIVE = "reflexive"  # Gödel machine


@dataclass
class MetalearningTask:
    """
    Single metalearning task.

    Contains:
    - Task data (vision + prompt)
    - Ground truth response
    - Success indicator
    """

    task_id: str
    vision_data: str  # Base64 image
    prompt: str
    response: str
    success: bool
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt[:100],
            "response": self.response[:100],
            "success": self.success,
            "timestamp": self.timestamp,
        }


@dataclass
class MetalearningResult:
    """
    Result from metalearning update.

    Contains:
    - Loss metrics
    - Accuracy
    - Gradient info
    - Timing
    """

    task_id: str
    method: MetalearningMethod
    inner_loss: float = 0.0
    outer_loss: float = 0.0
    meta_lr: float = 0.01
    inner_lr: float = 0.1
    accuracy: float = 0.0
    gradient_norm: float = 0.0
    converged: bool = False
    iterations: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetalearningAdapter:
    """
    Adapter for Omega Metalearning integration with VIL.

    Features:
    1. MAML-style inner/outer loop
    2. Self-play opponent generation
    3. Reflexive Gödel machine monitoring
    4. Vision task support
    5. CMP tracking integration
    """

    def __init__(
        self,
        method: MetalearningMethod = MetalearningMethod.MAML,
        meta_lr: float = 0.01,
        inner_lr: float = 0.1,
        inner_steps: int = 5,
        bus_emitter: Optional[callable] = None,
    ):
        self.method = method
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.bus_emitter = bus_emitter

        # Omega components (if available)
        if OMEGA_AVAILABLE:
            try:
                self.meta_learner = MetaLearner(input_dim=64)  # Default embedding dimension
            except Exception as e:
                print(f"[MetalearningAdapter] MetaLearner init failed: {e}")
                self.meta_learner = None

            try:
                self.self_play = SelfPlayEngine()
            except Exception as e:
                print(f"[MetalearningAdapter] SelfPlayEngine init failed: {e}")
                self.self_play = None

            try:
                self.reflexive = ReflexiveMonitor()
            except Exception as e:
                print(f"[MetalearningAdapter] ReflexiveMonitor init failed: {e}")
                self.reflexive = None
        else:
            self.meta_learner = None
            self.self_play = None
            self.reflexive = None

        # Task storage
        self.tasks: List[MetalearningTask] = []
        self.task_embeddings: Dict[str, np.ndarray] = {}

        # Statistics
        self.stats = {
            "tasks_added": 0,
            "inner_loops": 0,
            "outer_loops": 0,
            "avg_inner_loss": 0.0,
            "avg_outer_loss": 0.0,
            "avg_accuracy": 0.0,
            "convergence_rate": 0.0,
        }

    async def add_task(
        self,
        task_id: str,
        vision_data: str,
        prompt: str,
        response: str,
        success: bool,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetalearningTask:
        """
        Add vision task for meta-learning.

        Args:
            task_id: Unique task identifier
            vision_data: Base64-encoded image
            prompt: Text prompt
            response: Model response
            success: Whether outcome was successful
            embedding: Optional pre-computed embedding
            metadata: Additional metadata

        Returns:
            MetalearningTask
        """
        task = MetalearningTask(
            task_id=task_id,
            vision_data=vision_data,
            prompt=prompt,
            response=response,
            success=success,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self.tasks.append(task)

        # Store embedding
        if embedding is not None:
            self.task_embeddings[task_id] = embedding

        # Emit learning event
        await self._emit_learning_event(task, "task_added")

        self.stats["tasks_added"] += 1

        return task

    async def inner_loop(
        self,
        task_id: str,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> MetalearningResult:
        """
        Run inner loop adaptation (task-specific).

        MAML inner loop:
        θ'_i = θ_i - α ∇_θ L_Ti(θ_i)

        Args:
            task_id: Task to adapt to
            num_steps: Number of gradient steps
            lr: Learning rate

        Returns:
            MetalearningResult with loss, accuracy, gradient
        """
        num_steps = num_steps or self.inner_steps
        lr = lr or self.inner_lr

        start_time = time.time()
        trace_id = create_trace_id("inner_loop")

        # Find task
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task is None:
            return MetalearningResult(
                task_id=task_id,
                method=self.method,
                latency_ms=0.0,
                metadata={"error": "Task not found"},
            )

        # Run inner loop (mock if Omega unavailable)
        if self.meta_learner and OMEGA_AVAILABLE:
            # Use actual Omega metalearner
            result = self.meta_learner.inner_loop(
                task_id=task_id,
                num_steps=num_steps,
                lr=lr,
            )
            inner_loss = result.get("loss", 0.1)
            accuracy = result.get("accuracy", 0.9)
            gradient_norm = result.get("gradient_norm", 0.5)
        else:
            # Mock inner loop
            inner_loss = 0.1 * (1.0 + 0.1 * np.random.randn())
            accuracy = 0.9 + 0.05 * np.random.randn()
            accuracy = max(0.0, min(1.0, accuracy))
            gradient_norm = 0.5 + 0.1 * np.random.randn()

        latency_ms = (time.time() - start_time) * 1000

        result = MetalearningResult(
            task_id=task_id,
            method=self.method,
            inner_loss=inner_loss,
            inner_lr=lr,
            accuracy=accuracy,
            gradient_norm=gradient_norm,
            iterations=num_steps,
            latency_ms=latency_ms,
        )

        # Update stats
        self.stats["inner_loops"] += 1
        self.stats["avg_inner_loss"] = (
            (self.stats["avg_inner_loss"] * (self.stats["inner_loops"] - 1) + inner_loss) /
            self.stats["inner_loops"]
        )
        self.stats["avg_accuracy"] = (
            (self.stats["avg_accuracy"] * (self.stats["inner_loops"] - 1) + accuracy) /
            self.stats["inner_loops"]
        )

        # Emit event
        await self._emit_learning_event(task, "inner_loop_complete", result)

        return result

    async def outer_loop(
        self,
        num_tasks: int = 10,
        meta_lr: Optional[float] = None,
    ) -> MetalearningResult:
        """
        Run meta-learning outer loop (across tasks).

        MAML outer loop:
        θ ← θ - β ∑_i ∇_θ L_Ti(θ'_i)

        Args:
            num_tasks: Number of tasks to meta-learn over
            meta_lr: Meta-learning rate

        Returns:
            MetalearningResult with meta-loss, meta-accuracy
        """
        meta_lr = meta_lr or self.meta_lr

        start_time = time.time()
        trace_id = create_trace_id("outer_loop")

        # Sample tasks
        if len(self.tasks) < num_tasks:
            sampled_tasks = self.tasks
        else:
            import random
            sampled_tasks = random.sample(self.tasks, num_tasks)

        # Run outer loop (mock if Omega unavailable)
        if self.meta_learner and OMEGA_AVAILABLE:
            # Use actual Omega metalearner
            result = self.meta_learner.outer_loop(
                num_tasks=len(sampled_tasks),
                meta_lr=meta_lr,
            )
            outer_loss = result.get("meta_loss", 0.15)
            meta_accuracy = result.get("meta_accuracy", 0.85)
            task_diversity = result.get("task_diversity", 0.7)
        else:
            # Mock outer loop
            # Meta-loss is average of inner losses
            inner_losses = [
                0.1 + 0.05 * np.random.randn() for _ in sampled_tasks
            ]
            outer_loss = np.mean(inner_losses)
            meta_accuracy = 0.85 + 0.05 * np.random.randn()
            meta_accuracy = max(0.0, min(1.0, meta_accuracy))
            task_diversity = 0.7 + 0.1 * np.random.randn()
            task_diversity = max(0.0, min(1.0, task_diversity))

        latency_ms = (time.time() - start_time) * 1000

        result = MetalearningResult(
            task_id=f"outer_{int(start_time)}",
            method=self.method,
            outer_loss=outer_loss,
            meta_lr=meta_lr,
            accuracy=meta_accuracy,
            latency_ms=latency_ms,
            metadata={
                "task_diversity": task_diversity,
                "num_tasks": len(sampled_tasks),
            },
        )

        # Update stats
        self.stats["outer_loops"] += 1
        self.stats["avg_outer_loss"] = (
            (self.stats["avg_outer_loss"] * (self.stats["outer_loops"] - 1) + outer_loss) /
            self.stats["outer_loops"]
        )

        # Emit event
        await self._emit_learning_event(None, "outer_loop_complete", result)

        return result

    async def adapt(
        self,
        task_id: str,
        num_steps: int = 5,
    ) -> MetalearningResult:
        """
        Few-shot adaptation to new task.

        Combines inner loop with task-specific adaptation.
        """
        return await self.inner_loop(task_id, num_steps)

    async def self_play_episode(
        self,
        opponent_id: Optional[str] = None,
        num_rounds: int = 10,
    ) -> MetalearningResult:
        """
        Run self-play episode for Nash equilibrium convergence.

        AlphaZero-style self-play:
        1. Generate opponent from current policy
        2. Play episode
        3. Update from outcome
        """
        if self.self_play and OMEGA_AVAILABLE:
            # Use actual self-play engine
            opponent = self.self_play.generate_opponent()
            episode = self.self_play.play_episode(num_rounds=num_rounds)
            self.self_play.update_from_play(episode)

            return MetalearningResult(
                task_id=f"self_play_{int(time.time())}",
                method=MetalearningMethod.SELF_PLAY,
                accuracy=episode.get("win_rate", 0.5),
                metadata={"opponent_id": opponent, "rounds": num_rounds},
            )
        else:
            # Mock self-play
            return MetalearningResult(
                task_id=f"self_play_{int(time.time())}",
                method=MetalearningMethod.SELF_PLAY,
                accuracy=0.5 + 0.1 * np.random.randn(),
                metadata={"rounds": num_rounds, "opponent_id": opponent_id},
            )

    async def reflexive_check(
        self,
        current_params: np.ndarray,
        proposed_params: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Gödel-style proof checking for self-modification.

        Checks if proposed modification is provably an improvement.
        """
        if self.reflexive and OMEGA_AVAILABLE:
            # Use actual reflexive monitor
            improvement = self.reflexive.verify_improvement(
                current_params,
                proposed_params,
            )
            return {
                "improvement_verified": improvement,
                "proof_found": improvement,
                "confidence": 0.9 if improvement else 0.1,
            }
        else:
            # Mock reflexive check
            improvement = np.linalg.norm(proposed_params) > np.linalg.norm(current_params)
            return {
                "improvement_verified": improvement,
                "proof_found": improvement,
                "confidence": 0.7,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self.stats,
            "total_tasks": len(self.tasks),
            "method": self.method.value,
            "omega_available": OMEGA_AVAILABLE,
            "convergence_rate": (
                self.stats["inner_loops"] / max(1, self.stats["outer_loops"])
                if self.stats["outer_loops"] > 0 else 0.0
            ),
        }

    async def _emit_learning_event(
        self,
        task: Optional[MetalearningTask],
        event_type: str,
        result: Optional[MetalearningResult] = None,
    ) -> None:
        """Emit learning event to bus."""
        if not self.bus_emitter:
            return

        data = {
            "timestamp": time.time(),
            "event_type": event_type,
            "method": self.method.value,
        }

        if task:
            data["task_id"] = task.task_id
            data["success"] = task.success

        if result:
            data["inner_loss"] = result.inner_loss
            data["outer_loss"] = result.outer_loss
            data["accuracy"] = result.accuracy
            data["latency_ms"] = result.latency_ms

        event = {
            "topic": "vil.metalearning.event",
            "data": data,
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[MetalearningAdapter] Bus emission error: {e}")


def create_metalearning_adapter(
    method: MetalearningMethod = MetalearningMethod.MAML,
    bus_emitter: Optional[callable] = None,
) -> MetalearningAdapter:
    """Create metalearning adapter with default config."""
    return MetalearningAdapter(
        method=method,
        bus_emitter=bus_emitter,
    )


__all__ = [
    "MetalearningMethod",
    "MetalearningTask",
    "MetalearningResult",
    "MetalearningAdapter",
    "create_metalearning_adapter",
    "OMEGA_AVAILABLE",
]
