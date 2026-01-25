#!/usr/bin/env python3
"""
curriculum.py - Curriculum Learning for ARK Gates

P2-031: Implement curriculum learning for gates
P2-032: Add difficulty progression logic
P2-030: Create learning rate scheduler

Implements:
- Difficulty-based ordering of training examples
- Progressive hardening of gate thresholds
- Competence-based advancement
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import logging
import math

logger = logging.getLogger("ARK.CL.Curriculum")


@dataclass
class DifficultyLevel:
    """A difficulty level in the curriculum."""
    level: int
    name: str
    entropy_max: float  # Max entropy for this level
    cmp_threshold: float  # Min CMP to pass this level
    gate_strictness: float  # Gate threshold multiplier
    min_successes: int  # Successes needed to advance
    
    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "name": self.name,
            "entropy_max": self.entropy_max,
            "cmp_threshold": self.cmp_threshold,
            "gate_strictness": self.gate_strictness,
            "min_successes": self.min_successes
        }


# Default curriculum levels
DEFAULT_CURRICULUM = [
    DifficultyLevel(0, "Beginner", 0.9, 0.2, 0.5, 5),
    DifficultyLevel(1, "Novice", 0.8, 0.3, 0.6, 10),
    DifficultyLevel(2, "Intermediate", 0.7, 0.4, 0.7, 15),
    DifficultyLevel(3, "Advanced", 0.6, 0.5, 0.8, 20),
    DifficultyLevel(4, "Expert", 0.5, 0.6, 0.9, 30),
    DifficultyLevel(5, "Master", 0.4, 0.7, 1.0, 50),
]


class DifficultyScheduler:
    """
    Schedules learning rate and difficulty progression.
    
    P2-030: Learning rate scheduler
    P2-032: Difficulty progression logic
    """
    
    def __init__(
        self,
        initial_lr: float = 0.01,
        min_lr: float = 0.0001,
        decay_steps: int = 1000,
        warmup_steps: int = 100
    ):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self) -> None:
        """Advance scheduler by one step."""
        self.current_step += 1
    
    def get_lr(self) -> float:
        """Get current learning rate with warmup and decay."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (self.current_step / self.warmup_steps)
        
        # Cosine annealing after warmup
        progress = (self.current_step - self.warmup_steps) / self.decay_steps
        progress = min(1.0, progress)
        
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        return lr
    
    def get_difficulty_multiplier(self) -> float:
        """
        Get difficulty multiplier based on training progress.
        
        Starts easy (0.5) and progresses to full difficulty (1.0).
        """
        if self.current_step < self.warmup_steps:
            return 0.5
        
        progress = min(1.0, (self.current_step - self.warmup_steps) / self.decay_steps)
        return 0.5 + 0.5 * progress


class ThompsonSampler:
    """
    Thompson Sampling for contextual clade selection.
    
    Gemini 3 Pro R&D insight: "Explore contextual bandits where 
    the context is the current entropy state. This enables 
    state-dependent selection."
    """
    
    def __init__(self, n_arms: int = 6):
        self.n_arms = n_arms
        # Beta distribution parameters for each arm
        self.alpha = [1.0] * n_arms  # Successes + 1
        self.beta = [1.0] * n_arms   # Failures + 1
        self.pulls = [0] * n_arms
        self.context_history: List[Dict] = []
    
    def sample(self, context: Optional[Dict[str, float]] = None) -> int:
        """
        Sample best arm using Thompson Sampling.
        
        Args:
            context: Optional entropy context for contextual bandits
        """
        import random
        
        # If context provided, adjust priors based on context
        adjusted_alpha = self.alpha.copy()
        adjusted_beta = self.beta.copy()
        
        if context:
            # Contextual adjustment based on entropy
            h_total = sum(context.values()) / max(len(context), 1)
            
            # Lower entropy → favor exploitation (higher arms already successful)
            # Higher entropy → favor exploration (try different arms)
            exploration_bonus = h_total * 0.5
            
            for i in range(self.n_arms):
                # Add exploration bonus to less-pulled arms
                if self.pulls[i] < sum(self.pulls) / self.n_arms:
                    adjusted_alpha[i] += exploration_bonus
        
        # Sample from Beta distribution for each arm
        samples = []
        for i in range(self.n_arms):
            # Use simple approximation if no scipy
            a, b = adjusted_alpha[i], adjusted_beta[i]
            sample = random.betavariate(a, b)
            samples.append(sample)
        
        return samples.index(max(samples))
    
    def update(self, arm: int, reward: float) -> None:
        """Update arm based on reward (0 or 1)."""
        self.pulls[arm] += 1
        if reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        means = [a / (a + b) for a, b in zip(self.alpha, self.beta)]
        return {
            "n_arms": self.n_arms,
            "total_pulls": sum(self.pulls),
            "arm_means": means,
            "best_arm": means.index(max(means)),
            "pulls_per_arm": self.pulls
        }


class CurriculumLearning:
    """
    Curriculum learning manager for ARK gates.
    
    P2-031: Curriculum learning for gates
    
    Trains gates on progressively harder commits:
    1. Start with easy commits (low entropy, clear purpose)
    2. Track success rate at each level
    3. Advance when competent
    4. Adjust gate thresholds based on level
    """
    
    def __init__(
        self,
        levels: Optional[List[DifficultyLevel]] = None,
        storage_path: Optional[str] = None
    ):
        self.levels = levels or DEFAULT_CURRICULUM.copy()
        self.storage_path = Path(storage_path) if storage_path else Path("~/.ark/cl/curriculum.json").expanduser()
        
        self.current_level: int = 0
        self.successes_at_level: int = 0
        self.failures_at_level: int = 0
        self.history: List[Dict] = []
        self.scheduler = DifficultyScheduler()
        
        self._load()
    
    def _load(self) -> None:
        """Load curriculum state."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                self.current_level = data.get("current_level", 0)
                self.successes_at_level = data.get("successes_at_level", 0)
                self.failures_at_level = data.get("failures_at_level", 0)
                self.history = data.get("history", [])
                self.scheduler.current_step = data.get("total_steps", 0)
            except Exception as e:
                logger.warning("Failed to load curriculum: %s", e)
    
    def _save(self) -> None:
        """Save curriculum state."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_level": self.current_level,
                "successes_at_level": self.successes_at_level,
                "failures_at_level": self.failures_at_level,
                "history": self.history[-100:],  # Keep last 100
                "total_steps": self.scheduler.current_step,
                "saved_at": time.time()
            }
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save curriculum: %s", e)
    
    @property
    def current_difficulty(self) -> DifficultyLevel:
        """Get current difficulty level."""
        return self.levels[min(self.current_level, len(self.levels) - 1)]
    
    def get_gate_thresholds(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust gate thresholds based on current curriculum level.
        
        Lower levels = more lenient thresholds.
        """
        strictness = self.current_difficulty.gate_strictness
        return {
            name: threshold * strictness
            for name, threshold in base_thresholds.items()
        }
    
    def filter_by_difficulty(self, experiences: List[Any], difficulty_fn: Callable) -> List[Any]:
        """
        Filter experiences to match current difficulty level.
        
        Args:
            experiences: List of experiences
            difficulty_fn: Function(exp) -> float returning difficulty score
        """
        max_difficulty = self.current_difficulty.entropy_max
        return [e for e in experiences if difficulty_fn(e) <= max_difficulty]
    
    def record_result(self, success: bool, cmp: float, entropy_total: float) -> bool:
        """
        Record commit result and check for level advancement.
        
        Returns True if level changed.
        """
        self.scheduler.step()
        
        if success:
            self.successes_at_level += 1
        else:
            self.failures_at_level += 1
        
        self.history.append({
            "level": self.current_level,
            "success": success,
            "cmp": cmp,
            "entropy": entropy_total,
            "timestamp": time.time()
        })
        
        # Check for advancement
        level_changed = False
        current = self.current_difficulty
        
        if self.successes_at_level >= current.min_successes:
            # Check success rate
            total = self.successes_at_level + self.failures_at_level
            success_rate = self.successes_at_level / total if total > 0 else 0
            
            if success_rate >= 0.7 and self.current_level < len(self.levels) - 1:
                # Advance to next level
                self.current_level += 1
                self.successes_at_level = 0
                self.failures_at_level = 0
                level_changed = True
                logger.info("Advanced to level %d: %s", 
                           self.current_level, self.current_difficulty.name)
        
        # Check for demotion (too many failures)
        if self.failures_at_level > current.min_successes * 2 and self.current_level > 0:
            self.current_level -= 1
            self.successes_at_level = 0
            self.failures_at_level = 0
            level_changed = True
            logger.info("Demoted to level %d: %s",
                       self.current_level, self.current_difficulty.name)
        
        self._save()
        return level_changed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        total_attempts = self.successes_at_level + self.failures_at_level
        success_rate = self.successes_at_level / total_attempts if total_attempts > 0 else 0
        
        return {
            "current_level": self.current_level,
            "level_name": self.current_difficulty.name,
            "successes": self.successes_at_level,
            "failures": self.failures_at_level,
            "success_rate": success_rate,
            "needed_for_advance": self.current_difficulty.min_successes,
            "learning_rate": self.scheduler.get_lr(),
            "difficulty_multiplier": self.scheduler.get_difficulty_multiplier(),
            "total_steps": self.scheduler.current_step
        }
    
    def reset(self) -> None:
        """Reset curriculum to beginning."""
        self.current_level = 0
        self.successes_at_level = 0
        self.failures_at_level = 0
        self.history.clear()
        self.scheduler = DifficultyScheduler()
        self._save()
        logger.info("Curriculum reset to level 0")
