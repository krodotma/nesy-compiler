"""
Theia In-Context Learning from Screenshots — CTM-Derived ICL.

ATTRIBUTION:
Temporal processing concepts derived from CTM (arXiv:2505.05522).
ICL patterns informed by "Hopfield Networks is All You Need" (Ramsauer et al.).

This module enables self-teaching from screenshot sequences:
    - Store successful action→screenshot pairs
    - Retrieve similar past experiences
    - Learn temporal patterns for prescience
"""

import numpy as np
import base64
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from theia.memory.hopfield import HopfieldMemory, store_patterns
from theia.memory.temporal_sync import TemporalHierarchy, ARCPrescienceEngine


@dataclass
class ScreenshotExample:
    """
    Screenshot example for ICL.
    
    Stores correlation between visual state and action taken.
    """
    id: str
    image_embedding: np.ndarray  # Encoded screenshot
    action: str                  # Action taken
    outcome: str                 # 'success', 'failure', 'neutral'
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_screenshot(
        cls,
        image_b64: str,
        action: str,
        outcome: str = "neutral",
        embed_fn: Optional[callable] = None,
    ) -> "ScreenshotExample":
        """Create example from base64 screenshot."""
        # Default embedding: hash-based for demo
        if embed_fn is None:
            import hashlib
            h = hashlib.sha256(image_b64.encode()).digest()
            embedding = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
        else:
            embedding = embed_fn(image_b64)
        
        return cls(
            id=f"ex-{int(time.time()*1000)}",
            image_embedding=embedding,
            action=action,
            outcome=outcome,
            timestamp=time.time(),
        )


class ScreenshotICL:
    """
    In-Context Learning from Screenshots.
    
    Combines mHC pattern retrieval with temporal sync for:
    1. Storing successful interactions as patterns
    2. Retrieving similar past situations
    3. Learning temporal patterns for action prediction
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        max_examples: int = 100,
        beta: float = 5.0,
    ):
        self.embedding_dim = embedding_dim
        self.max_examples = max_examples
        self.beta = beta
        
        self.examples: List[ScreenshotExample] = []
        self._memory: Optional[HopfieldMemory] = None
        
        # Temporal sync for pattern detection
        self.temporal = TemporalHierarchy()
    
    def add_example(self, example: ScreenshotExample) -> None:
        """Add example to ICL memory."""
        self.examples.append(example)
        
        # Trim if too many
        if len(self.examples) > self.max_examples:
            # Keep most recent successful ones
            successful = [e for e in self.examples if e.outcome == "success"]
            recent = sorted(self.examples, key=lambda e: e.timestamp)[-self.max_examples//2:]
            self.examples = list(set(successful + recent))[:self.max_examples]
        
        # Rebuild memory
        self._rebuild_memory()
    
    def _rebuild_memory(self) -> None:
        """Rebuild Hopfield memory from examples."""
        if not self.examples:
            self._memory = None
            return
        
        patterns = [e.image_embedding for e in self.examples]
        self._memory = store_patterns(patterns, beta=self.beta)
    
    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
    ) -> List[Tuple[ScreenshotExample, float]]:
        """
        Retrieve k most similar examples.
        
        Uses mHC soft retrieval for similarity.
        """
        if self._memory is None or not self.examples:
            return []
        
        # Compute similarities via Hopfield retrieval
        similarities = self.beta * (self._memory.patterns @ query_embedding)
        weights = np.exp(similarities - np.max(similarities))
        weights = weights / np.sum(weights)
        
        # Get top-k
        top_k_idx = np.argsort(weights)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            if idx < len(self.examples):
                results.append((self.examples[idx], float(weights[idx])))
        
        return results
    
    def suggest_action(
        self,
        current_embedding: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Suggest action based on similar past experiences.
        
        Returns (action, confidence).
        """
        similar = self.retrieve_similar(current_embedding, k=3)
        
        if not similar:
            return "explore", 0.0
        
        # Vote among successful similar experiences
        action_votes: Dict[str, float] = {}
        for example, similarity in similar:
            if example.outcome == "success":
                action = example.action
                action_votes[action] = action_votes.get(action, 0) + similarity
        
        if not action_votes:
            return "explore", 0.0
        
        best_action = max(action_votes, key=action_votes.get)
        confidence = action_votes[best_action] / sum(action_votes.values())
        
        return best_action, confidence
    
    def update_temporal(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Update temporal sync with new embedding."""
        return self.temporal.step(embedding)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard visualization."""
        return {
            "n_examples": len(self.examples),
            "success_rate": (
                sum(1 for e in self.examples if e.outcome == "success")
                / max(1, len(self.examples))
            ),
            "temporal_coherence": (
                self.temporal.scales[0].compute_coherence()
                if self.temporal.scales else 0.0
            ),
            "memory_capacity": self.max_examples,
            "beta_temperature": self.beta,
        }


__all__ = [
    "ScreenshotExample",
    "ScreenshotICL",
]
