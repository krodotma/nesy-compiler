"""
VIL ICL+ Selection Strategies
Multi-strategy example selection for ICL.

Features:
1. Novelty-based selection (exploration)
2. Diversity-based selection (balanced)
3. Recency-based selection (exploitation)
4. CMP-driven selection (fitness-based)
5. Hybrid selection (combined strategies)

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from nucleus.vil.icl.buffer import (
    ICLExample,
    ICLExampleType,
    GeometricICLBuffer,
    create_geometric_icl_buffer,
)
from nucleus.vil.cmp.manager import PHI


class SelectionStrategy(str, Enum):
    """ICL example selection strategies."""

    NOVEL = "novel"  # High novelty (exploration)
    DIVERSE = "diverse"  # Geometric diversity (balanced)
    RECENT = "recent"  # Recently used (exploitation)
    SIMILAR = "similar"  # Similar to query (standard)
    HIGH_FITNESS = "high_fitness"  # CMP fitness-driven
    HYBRID = "hybrid"  # Combined strategies


@dataclass
class SelectionConfig:
    """
    Configuration for example selection.

    Contains:
    - Strategy weights for hybrid selection
    - Selection parameters
    - Diversity threshold
    """

    strategy_weights: Dict[SelectionStrategy, float] = field(default_factory=lambda: {
        SelectionStrategy.NOVEL: 0.2,
        SelectionStrategy.DIVERSE: 0.3,
        SelectionStrategy.RECENT: 0.2,
        SelectionStrategy.HIGH_FITNESS: 0.3,
    })
    diversity_threshold: float = 0.7  # Minimum cosine distance
    novelty_bonus: float = 0.5  # Bonus for novelty
    recency_halflife: float = 3600.0  # 1 hour
    fitness_weight: float = PHI  # Golden ratio weighting

    def normalize_weights(self) -> Dict[SelectionStrategy, float]:
        """Normalize strategy weights to sum to 1."""
        total = sum(self.strategy_weights.values())
        if total == 0:
            return {s: 0.25 for s in SelectionStrategy if s != SelectionStrategy.HYBRID}
        return {s: w / total for s, w in self.strategy_weights.items()}


@dataclass
class SelectionResult:
    """
    Result from example selection.

    Contains:
    - Selected examples
    - Selection scores
    - Strategy used
    - Diversity metrics
    """

    examples: List[ICLExample]
    scores: List[float]
    strategy: SelectionStrategy
    diversity_score: float
    avg_fitness: float
    avg_novelty: float
    selection_time_ms: float


class ICLExampleSelector:
    """
    Multi-strategy ICL example selector.

    Features:
    1. Novelty-based: Exploration of high novelty examples
    2. Diversity-based: Geometrically diverse batch
    3. Recency-based: Exploitation of recent knowledge
    4. Fitness-based: CMP-driven fitness maximization
    5. Hybrid: Weighted combination of strategies
    """

    def __init__(
        self,
        buffer: GeometricICLBuffer,
        config: Optional[SelectionConfig] = None,
        bus_emitter: Optional[callable] = None,
    ):
        self.buffer = buffer
        self.config = config or SelectionConfig()
        self.bus_emitter = bus_emitter

        # Selection history
        self.selection_history: List[SelectionResult] = []

        # Statistics
        self.stats = {
            "selections_per_strategy": {s.value: 0 for s in SelectionStrategy},
            "avg_diversity": 0.0,
            "avg_selection_time": 0.0,
        }

    def select(
        self,
        query: str,
        k: int = 5,
        strategy: SelectionStrategy = SelectionStrategy.DIVERSE,
        query_embedding: Optional[np.ndarray] = None,
    ) -> SelectionResult:
        """
        Select ICL examples using specified strategy.

        Args:
            query: Query input
            k: Number of examples to select
            strategy: Selection strategy
            query_embedding: Optional query embedding

        Returns:
            SelectionResult with selected examples
        """
        start_time = time.time()

        # Select based on strategy
        if strategy == SelectionStrategy.NOVEL:
            examples = self._select_novel(k)
        elif strategy == SelectionStrategy.DIVERSE:
            examples = self._select_diverse(query_embedding or self._get_query_embedding(query), k)
        elif strategy == SelectionStrategy.RECENT:
            examples = self._select_recent(k)
        elif strategy == SelectionStrategy.SIMILAR:
            examples = self._select_similar(query_embedding or self._get_query_embedding(query), k)
        elif strategy == SelectionStrategy.HIGH_FITNESS:
            examples = self._select_high_fitness(k)
        elif strategy == SelectionStrategy.HYBRID:
            examples = self._select_hybrid(query_embedding or self._get_query_embedding(query), k)
        else:
            examples = self._select_recent(k)

        # Calculate scores
        scores = self._calculate_selection_scores(examples, strategy)

        # Calculate metrics
        diversity = self._calculate_diversity(examples)
        avg_fitness = np.mean([e.fitness for e in examples]) if examples else 0.0
        avg_novelty = np.mean([e.novelty_score for e in examples]) if examples else 0.0

        selection_time = (time.time() - start_time) * 1000

        result = SelectionResult(
            examples=examples,
            scores=scores,
            strategy=strategy,
            diversity_score=diversity,
            avg_fitness=avg_fitness,
            avg_novelty=avg_novelty,
            selection_time_ms=selection_time,
        )

        # Update history and stats
        self.selection_history.append(result)
        self.stats["selections_per_strategy"][strategy.value] += 1
        self.stats["avg_diversity"] = (
            (self.stats["avg_diversity"] * (len(self.selection_history) - 1) + diversity) /
            len(self.selection_history)
        )
        self.stats["avg_selection_time"] = (
            (self.stats["avg_selection_time"] * (len(self.selection_history) - 1) + selection_time) /
            len(self.selection_history)
        )

        # Emit event
        self._emit_selection_event(result)

        return result

    def _select_novel(self, k: int) -> List[ICLExample]:
        """Select high novelty examples."""
        sorted_examples = sorted(
            self.buffer.examples.values(),
            key=lambda e: e.novelty_score,
            reverse=True,
        )
        return sorted_examples[:k]

    def _select_diverse(self, query_embedding: np.ndarray, k: int) -> List[ICLExample]:
        """Select geometrically diverse examples."""
        if not self.buffer.examples:
            return []

        retrieved = self.buffer.retrieve_examples("", strategy="diverse", k=k, query_embedding=query_embedding)
        return retrieved

    def _select_recent(self, k: int) -> List[ICLExample]:
        """Select recently used examples."""
        sorted_examples = sorted(
            self.buffer.examples.values(),
            key=lambda e: max(e.timestamp, e.last_used),
            reverse=True,
        )
        return sorted_examples[:k]

    def _select_similar(self, query_embedding: np.ndarray, k: int) -> List[ICLExample]:
        """Select most similar examples."""
        retrieved = self.buffer.retrieve_examples("", strategy="similar", k=k, query_embedding=query_embedding)
        return retrieved

    def _select_high_fitness(self, k: int) -> List[ICLExample]:
        """Select highest fitness examples."""
        sorted_examples = sorted(
            self.buffer.examples.values(),
            key=lambda e: e.fitness,
            reverse=True,
        )
        return sorted_examples[:k]

    def _select_hybrid(self, query_embedding: np.ndarray, k: int) -> List[ICLExample]:
        """Select using hybrid strategy (weighted combination)."""
        weights = self.config.normalize_weights()

        # Calculate number of examples per strategy
        alloc = {}
        remaining = k
        for strategy, weight in weights.items():
            count = max(1, int(k * weight))
            alloc[strategy] = count
            remaining -= count

        # Distribute remaining
        if remaining > 0:
            alloc[SelectionStrategy.DIVERSE] += remaining

        # Select from each strategy
        selected = set()
        for strategy, count in alloc.items():
            if count <= 0:
                continue

            if strategy == SelectionStrategy.NOVEL:
                examples = self._select_novel(count * 2)
            elif strategy == SelectionStrategy.DIVERSE:
                examples = self._select_diverse(query_embedding, count * 2)
            elif strategy == SelectionStrategy.RECENT:
                examples = self._select_recent(count * 2)
            elif strategy == SelectionStrategy.HIGH_FITNESS:
                examples = self._select_high_fitness(count * 2)
            else:
                examples = []

            # Add without duplicates
            for ex in examples:
                if ex.example_id not in selected:
                    selected.add(ex.example_id)
                    if len(selected) >= k:
                        break

            if len(selected) >= k:
                break

        return [self.buffer.examples[eid] for eid in list(selected)[:k]]

    def _calculate_selection_scores(
        self,
        examples: List[ICLExample],
        strategy: SelectionStrategy,
    ) -> List[float]:
        """Calculate selection scores for examples."""
        scores = []

        for example in examples:
            if strategy == SelectionStrategy.NOVEL:
                score = example.novelty_score
            elif strategy == SelectionStrategy.HIGH_FITNESS:
                score = example.fitness
            elif strategy == SelectionStrategy.RECENT:
                age = time.time() - max(example.timestamp, example.last_used)
                score = 1.0 / (1.0 + age / self.config.recency_halflife)
            else:
                score = example.confidence

            scores.append(score)

        return scores

    def _calculate_diversity(self, examples: List[ICLExample]) -> float:
        """Calculate geometric diversity of selected examples."""
        if len(examples) <= 1:
            return 0.0

        # Compute pairwise distances
        distances = []
        for i, ex1 in enumerate(examples):
            for ex2 in examples[i+1:]:
                if ex1.embedding_spherical is not None and ex2.embedding_spherical is not None:
                    # Cosine distance
                    sim = np.dot(ex1.embedding_spherical, ex2.embedding_spherical)
                    distance = 1 - float(sim)
                    distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get or compute query embedding."""
        # In production, compute actual embedding
        embedding_dim = self.buffer.embedding_dim
        return np.random.randn(embedding_dim)

    def recommend_strategy(
        self,
        clade_id: Optional[str] = None,
        current_fitness: float = 0.5,
    ) -> SelectionStrategy:
        """
        Recommend selection strategy based on context.

        Mapping:
        - High fitness (> PHI): Use "novel" (explore)
        - Medium fitness: Use "diverse" (balanced)
        - Low fitness: Use "recent" (exploit known good)
        """
        if current_fitness > PHI:
            return SelectionStrategy.NOVEL
        elif current_fitness > 0.8:
            return SelectionStrategy.DIVERSE
        else:
            return SelectionStrategy.RECENT

    def get_stats(self) -> Dict[str, Any]:
        """Get selection statistics."""
        return {
            **self.stats,
            "total_selections": len(self.selection_history),
            "recent_diversity": self.selection_history[-1].diversity_score if self.selection_history else 0.0,
        }

    def _emit_selection_event(self, result: SelectionResult) -> None:
        """Emit selection event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.icl.selection",
            "data": {
                "strategy": result.strategy.value,
                "num_examples": len(result.examples),
                "diversity_score": result.diversity_score,
                "avg_fitness": result.avg_fitness,
                "selection_time_ms": result.selection_time_ms,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[ICLExampleSelector] Bus emission error: {e}")


def create_icl_example_selector(
    buffer: Optional[GeometricICLBuffer] = None,
    config: Optional[SelectionConfig] = None,
    bus_emitter: Optional[callable] = None,
) -> ICLExampleSelector:
    """Create ICL example selector with default config."""
    if buffer is None:
        buffer = create_geometric_icl_buffer(bus_emitter=bus_emitter)

    return ICLExampleSelector(
        buffer=buffer,
        config=config,
        bus_emitter=bus_emitter,
    )


__all__ = [
    "SelectionStrategy",
    "SelectionConfig",
    "SelectionResult",
    "ICLExampleSelector",
    "create_icl_example_selector",
]
