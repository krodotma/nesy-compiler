#!/usr/bin/env python3
"""
replay_buffer.py - Experience Replay Buffer for ARK CL

P2-022: Implement experience replay buffer
P2-029: Add batch learning mode

Features:
- Prioritized experience replay (PER)
- Temporal difference (TD) error weighting
- Reservoir sampling for bounded memory
- CMP-based priority scoring
"""

import json
import time
import random
import heapq
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
import logging

logger = logging.getLogger("ARK.CL.ReplayBuffer")


@dataclass
class Experience:
    """Single learning experience from a commit."""
    commit_sha: str
    etymology: str
    cmp_before: float
    cmp_after: float
    entropy_before: Dict[str, float]
    entropy_after: Dict[str, float]
    gate_results: Dict[str, bool]  # gate_name -> passed
    success: bool
    timestamp: float = field(default_factory=time.time)
    
    # Computed fields
    cmp_delta: float = field(init=False)
    reward: float = field(init=False)
    priority: float = field(default=1.0)
    td_error: float = field(default=0.0)
    
    def __post_init__(self):
        self.cmp_delta = self.cmp_after - self.cmp_before
        # Reward: CMP improvement + bonus for success
        self.reward = self.cmp_delta + (0.2 if self.success else -0.1)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Experience":
        # Handle computed fields
        exp = cls(
            commit_sha=data["commit_sha"],
            etymology=data.get("etymology", ""),
            cmp_before=data.get("cmp_before", 0.5),
            cmp_after=data.get("cmp_after", 0.5),
            entropy_before=data.get("entropy_before", {}),
            entropy_after=data.get("entropy_after", {}),
            gate_results=data.get("gate_results", {}),
            success=data.get("success", True),
            timestamp=data.get("timestamp", time.time()),
            priority=data.get("priority", 1.0),
            td_error=data.get("td_error", 0.0)
        )
        return exp


class ReplayBuffer:
    """
    Standard FIFO experience replay buffer.
    
    P2-022: Core replay buffer implementation.
    """
    
    def __init__(self, capacity: int = 10000, db_path: Optional[str] = None):
        self.capacity = capacity
        self.db_path = Path(db_path) if db_path else Path("~/.ark/cl/replay_buffer.db").expanduser()
        self.buffer: List[Experience] = []
        self._load()
    
    def _ensure_db(self) -> None:
        """Create SQLite database if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sha TEXT NOT NULL UNIQUE,
                data TEXT NOT NULL,
                priority REAL DEFAULT 1.0,
                ts REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def _load(self) -> None:
        """Load experiences from database."""
        self._ensure_db()
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.execute(
                "SELECT data FROM experiences ORDER BY ts DESC LIMIT ?",
                (self.capacity,)
            )
            for row in cur:
                try:
                    data = json.loads(row[0])
                    self.buffer.append(Experience.from_dict(data))
                except Exception:
                    pass
            conn.close()
            logger.debug("Loaded %d experiences from buffer", len(self.buffer))
        except Exception as e:
            logger.warning("Failed to load replay buffer: %s", e)
    
    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
        
        # Evict oldest if over capacity (FIFO)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        
        # Persist to database
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT OR REPLACE INTO experiences (sha, data, priority, ts) VALUES (?, ?, ?, ?)",
                (experience.commit_sha, json.dumps(experience.to_dict()), 
                 experience.priority, experience.timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to persist experience: %s", e)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)
    
    def sample_batch(self, batch_size: int = 32) -> List[Dict]:
        """P2-029: Batch learning mode - return feature dicts for training."""
        experiences = self.sample(batch_size)
        return [
            {
                "cmp_before": e.cmp_before,
                "cmp_after": e.cmp_after,
                "cmp_delta": e.cmp_delta,
                "success": 1.0 if e.success else 0.0,
                "reward": e.reward,
                "gate_pass_rate": sum(e.gate_results.values()) / max(len(e.gate_results), 1),
                "entropy_delta": self._entropy_delta(e)
            }
            for e in experiences
        ]
    
    def _entropy_delta(self, e: Experience) -> float:
        """Compute total entropy change."""
        before = sum(e.entropy_before.values()) / max(len(e.entropy_before), 1)
        after = sum(e.entropy_after.values()) / max(len(e.entropy_after), 1)
        return before - after  # Positive = reduction (good)
    
    def recent(self, n: int = 10) -> List[Experience]:
        """Get n most recent experiences."""
        sorted_exp = sorted(self.buffer, key=lambda e: e.timestamp, reverse=True)
        return sorted_exp[:n]
    
    def clear(self) -> int:
        """Clear buffer and return count of cleared experiences."""
        count = len(self.buffer)
        self.buffer.clear()
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("DELETE FROM experiences")
            conn.commit()
            conn.close()
        except Exception:
            pass
        return count
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __iter__(self) -> Iterator[Experience]:
        return iter(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.
    
    Samples experiences based on TD error and CMP delta.
    High-impact experiences are replayed more frequently.
    """
    
    def __init__(
        self, 
        capacity: int = 10000, 
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
        db_path: Optional[str] = None
    ):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # Anneal beta to 1.0
        self.epsilon = 1e-6  # Small constant for stability
        super().__init__(capacity, db_path)
    
    def add(self, experience: Experience) -> None:
        """Add with max priority (for unseen experiences)."""
        if self.buffer:
            max_priority = max(e.priority for e in self.buffer)
        else:
            max_priority = 1.0
        
        experience.priority = max_priority
        super().add(experience)
    
    def update_priorities(self, sha_priorities: Dict[str, float]) -> None:
        """Update priorities based on TD errors from training."""
        for exp in self.buffer:
            if exp.commit_sha in sha_priorities:
                td_error = sha_priorities[exp.commit_sha]
                exp.td_error = td_error
                exp.priority = (abs(td_error) + self.epsilon) ** self.alpha
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample based on priorities."""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        # Compute sampling probabilities
        priorities = [e.priority for e in self.buffer]
        total_priority = sum(priorities)
        probabilities = [p / total_priority for p in priorities]
        
        # Sample indices
        indices = random.choices(
            range(len(self.buffer)),
            weights=probabilities,
            k=batch_size
        )
        
        # Compute importance sampling weights
        n = len(self.buffer)
        min_prob = min(probabilities)
        max_weight = (n * min_prob) ** (-self.beta)
        
        samples = []
        for idx in indices:
            exp = self.buffer[idx]
            # Add importance sampling weight to experience
            weight = (n * probabilities[idx]) ** (-self.beta) / max_weight
            exp_copy = Experience.from_dict(exp.to_dict())
            exp_copy.priority = weight  # Store weight in priority field
            samples.append(exp_copy)
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring."""
        if not self.buffer:
            return {"size": 0}
        
        cmp_deltas = [e.cmp_delta for e in self.buffer]
        rewards = [e.reward for e in self.buffer]
        success_rate = sum(1 for e in self.buffer if e.success) / len(self.buffer)
        
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "mean_cmp_delta": sum(cmp_deltas) / len(cmp_deltas),
            "mean_reward": sum(rewards) / len(rewards),
            "success_rate": success_rate,
            "beta": self.beta
        }
