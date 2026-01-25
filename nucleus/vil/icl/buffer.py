"""
VIL ICL+ Buffer Module
Geometric In-Context Learning buffer with manifold embeddings.

Features:
1. Spherical/hyperbolic example embeddings
2. Geometric similarity-based retrieval
3. H* entropy-based quality scoring
4. CMP-driven buffer pruning
5. Multi-strategy example selection

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from nucleus.vil.cmp.manager import VisionCMPMetrics, PHI


class ICLExampleType(str, Enum):
    """Types of ICL examples."""

    VISION = "vision"  # Vision-to-code example
    META = "meta"  # Metalearning example
    SYNTHESIS = "synthesis"  # Program synthesis example
    GEOMETRIC = "geometric"  # Geometric learning example


@dataclass
class ICLExample:
    """
    Single ICL example with geometric embedding.

    Contains:
    - Example data (input, output)
    - Geometric embedding (spherical/hyperbolic)
    - Quality metrics (H* entropy, confidence)
    - Timestamp and usage stats
    """

    example_id: str
    example_type: ICLExampleType
    input_data: str  # Image or prompt
    output_data: str  # Code or response

    # Geometric embedding
    embedding_spherical: Optional[np.ndarray] = None  # S^n embedding
    embedding_hyperbolic: Optional[np.ndarray] = None  # H^n embedding

    # Quality metrics
    h_star_entropy: float = 0.5  # H* normalized entropy
    confidence: float = 0.8
    novelty_score: float = 0.5

    # Metadata
    timestamp: float = field(default_factory=time.time)
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    fitness: float = 0.0

    # CMP linkage
    clade_id: Optional[str] = None
    cmp_fitness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "example_id": self.example_id,
            "type": self.example_type.value,
            "input_length": len(self.input_data),
            "output_length": len(self.output_data),
            "h_star_entropy": self.h_star_entropy,
            "confidence": self.confidence,
            "novelty_score": self.novelty_score,
            "usage_count": self.usage_count,
            "fitness": self.fitness,
            "clade_id": self.clade_id,
        }


@dataclass
class ICLBufferStats:
    """ICL buffer statistics."""

    total_examples: int = 0
    vision_examples: int = 0
    meta_examples: int = 0
    synthesis_examples: int = 0
    geometric_examples: int = 0
    avg_entropy: float = 0.0
    avg_confidence: float = 0.0
    avg_novelty: float = 0.0
    buffer_utilization: float = 0.0


class GeometricICLBuffer:
    """
    Geometric ICL buffer with manifold embeddings.

    Features:
    1. Spherical embeddings (S^n) for directional similarity
    2. Hyperbolic embeddings (H^n) for hierarchical similarity
    3. H* entropy-based quality filtering
    4. CMP-driven pruning for fitness maximization
    5. Multi-strategy retrieval (novel, diverse, recent)
    """

    def __init__(
        self,
        max_size: int = 100,
        embedding_dim: int = 64,
        use_spherical: bool = True,
        use_hyperbolic: bool = True,
        bus_emitter: Optional[callable] = None,
    ):
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.use_spherical = use_spherical
        self.use_hyperbolic = use_hyperbolic
        self.bus_emitter = bus_emitter

        # Example storage
        self.examples: Dict[str, ICLExample] = {}
        self._example_counter = 0

        # Geometric index
        self.spherical_embeddings: Dict[str, np.ndarray] = {}
        self.hyperbolic_embeddings: Dict[str, np.ndarray] = {}

        # H* entropy tracker
        self.entropy_baseline = 0.5
        self.entropy_history: List[float] = []

        # Statistics
        self.stats = ICLBufferStats()

    def add_example(
        self,
        input_data: str,
        output_data: str,
        example_type: ICLExampleType,
        h_star_entropy: float = 0.5,
        confidence: float = 0.8,
        novelty_score: float = 0.5,
        clade_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """
        Add example to ICL buffer.

        Args:
            input_data: Example input (image or prompt)
            output_data: Example output (code or response)
            example_type: Type of example
            h_star_entropy: H* normalized entropy
            confidence: Confidence score
            novelty_score: Novelty score
            clade_id: Associated clade ID
            embedding: Optional pre-computed embedding

        Returns:
            Example ID
        """
        # Check buffer size
        if len(self.examples) >= self.max_size:
            self._prune_buffer()

        # Create example ID
        example_id = f"icl_{self._example_counter}"
        self._example_counter += 1

        # Compute embeddings if not provided
        if embedding is None:
            embedding = np.random.randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize

        spherical_emb = embedding.copy() if self.use_spherical else None
        hyperbolic_emb = self._to_hyperbolic(embedding) if self.use_hyperbolic else None

        # Calculate fitness
        fitness = self._calculate_fitness(h_star_entropy, confidence, novelty_score)

        # Create example
        example = ICLExample(
            example_id=example_id,
            example_type=example_type,
            input_data=input_data,
            output_data=output_data,
            embedding_spherical=spherical_emb,
            embedding_hyperbolic=hyperbolic_emb,
            h_star_entropy=h_star_entropy,
            confidence=confidence,
            novelty_score=novelty_score,
            fitness=fitness,
            clade_id=clade_id,
        )

        # Store
        self.examples[example_id] = example
        if spherical_emb is not None:
            self.spherical_embeddings[example_id] = spherical_emb
        if hyperbolic_emb is not None:
            self.hyperbolic_embeddings[example_id] = hyperbolic_emb

        # Update stats
        self._update_stats()

        # Emit event
        self._emit_add_event(example)

        return example_id

    def retrieve_examples(
        self,
        query: str,
        strategy: str = "diverse",
        k: int = 5,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[ICLExample]:
        """
        Retrieve examples using specified strategy.

        Strategies:
        - "novel": High novelty score examples (exploration)
        - "diverse": Geometrically diverse examples (balanced)
        - "recent": Recently added/used examples (exploitation)
        - "similar": Most similar to query (standard)
        - "high_fitness": Highest fitness examples (CMP-driven)

        Args:
            query: Query input
            strategy: Retrieval strategy
            k: Number of examples to retrieve
            query_embedding: Optional query embedding

        Returns:
            List of retrieved examples
        """
        if not self.examples:
            return []

        # Compute query embedding if not provided
        if query_embedding is None:
            query_embedding = np.random.randn(self.embedding_dim)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Select based on strategy
        if strategy == "novel":
            retrieved = self._retrieve_novel(k)
        elif strategy == "diverse":
            retrieved = self._retrieve_diverse(query_embedding, k)
        elif strategy == "recent":
            retrieved = self._retrieve_recent(k)
        elif strategy == "similar":
            retrieved = self._retrieve_similar(query_embedding, k)
        elif strategy == "high_fitness":
            retrieved = self._retrieve_high_fitness(k)
        else:
            retrieved = self._retrieve_recent(k)

        # Update usage stats
        for example in retrieved:
            example.usage_count += 1
            example.last_used = time.time()

        return retrieved

    def _retrieve_novel(self, k: int) -> List[ICLExample]:
        """Retrieve high novelty examples."""
        sorted_examples = sorted(
            self.examples.values(),
            key=lambda e: e.novelty_score,
            reverse=True,
        )
        return sorted_examples[:k]

    def _retrieve_diverse(self, query_embedding: np.ndarray, k: int) -> List[ICLExample]:
        """Retrieve geometrically diverse examples."""
        if not self.examples:
            return []

        # Compute similarities
        similarities = {}
        for eid, emb in self.spherical_embeddings.items():
            sim = float(np.dot(query_embedding, emb))
            similarities[eid] = sim

        # Select diverse batch (farthest first sampling)
        retrieved = []
        remaining = set(self.examples.keys())

        for _ in range(min(k, len(self.examples))):
            if not remaining:
                break

            if retrieved:
                # Find example farthest from already retrieved
                best_eid = max(
                    remaining,
                    key=lambda eid: min(
                        1 - float(np.dot(
                            self.spherical_embeddings[eid],
                            self.spherical_embeddings[rid]
                        ))
                        for rid in [e.example_id for e in retrieved]
                    )
                )
            else:
                # First example: highest similarity
                best_eid = max(remaining, key=lambda eid: similarities.get(eid, 0))

            retrieved.append(self.examples[best_eid])
            remaining.remove(best_eid)

        return retrieved

    def _retrieve_recent(self, k: int) -> List[ICLExample]:
        """Retrieve recently added/used examples."""
        sorted_examples = sorted(
            self.examples.values(),
            key=lambda e: max(e.timestamp, e.last_used),
            reverse=True,
        )
        return sorted_examples[:k]

    def _retrieve_similar(self, query_embedding: np.ndarray, k: int) -> List[ICLExample]:
        """Retrieve most similar examples."""
        similarities = []
        for eid, emb in self.spherical_embeddings.items():
            sim = float(np.dot(query_embedding, emb))
            similarities.append((sim, eid))

        similarities.sort(reverse=True)
        return [self.examples[eid] for _, eid in similarities[:k]]

    def _retrieve_high_fitness(self, k: int) -> List[ICLExample]:
        """Retrieve highest fitness examples (CMP-driven)."""
        sorted_examples = sorted(
            self.examples.values(),
            key=lambda e: e.fitness,
            reverse=True,
        )
        return sorted_examples[:k]

    def _prune_buffer(self) -> None:
        """
        Prune buffer to maintain size limit.

        Pruning strategy (multi-factor):
        1. Low fitness (CMP-driven)
        2. Low novelty
        3. Old (not recently used)
        4. Low H* entropy (poor quality)
        """
        if len(self.examples) <= self.max_size:
            return

        # Calculate pruning scores
        scores = {}
        for eid, example in self.examples.items():
            # Lower score = more likely to prune
            age = time.time() - max(example.timestamp, example.last_used)
            score = (
                example.fitness * PHI +  # Fitness weighted by PHI
                example.novelty_score * 1.0 +
                example.h_star_entropy * 0.5 +
                (1.0 / (1.0 + age / 3600)) * 0.3  # Recency bonus
            )
            scores[eid] = score

        # Sort by score (ascending)
        sorted_ids = sorted(scores.keys(), key=lambda eid: scores[eid])

        # Remove lowest scoring examples
        num_to_remove = len(self.examples) - self.max_size + 1
        for eid in sorted_ids[:num_to_remove]:
            self._remove_example(eid)

    def _remove_example(self, example_id: str) -> None:
        """Remove example from buffer."""
        if example_id in self.examples:
            del self.examples[example_id]
            if example_id in self.spherical_embeddings:
                del self.spherical_embeddings[example_id]
            if example_id in self.hyperbolic_embeddings:
                del self.hyperbolic_embeddings[example_id]
            self._update_stats()

    def _calculate_fitness(
        self,
        h_star_entropy: float,
        confidence: float,
        novelty_score: float,
    ) -> float:
        """Calculate example fitness (phi-weighted)."""
        # Use golden ratio for importance weighting
        fitness = (
            confidence * PHI +
            h_star_entropy * 1.0 +
            novelty_score * (1 / PHI)
        )
        return fitness

    def _to_hyperbolic(self, spherical_emb: np.ndarray) -> np.ndarray:
        """
        Convert spherical embedding to hyperbolic (Poincaré ball).

        Uses exponential map for S^n -> H^n conversion.
        """
        # Simplified conversion: scale by norm
        # In production, use proper Riemannian exponential map
        norm = np.linalg.norm(spherical_emb)
        if norm == 0:
            return spherical_emb

        # Poincaré ball projection
        epsilon = 0.1  # Poincaré ball radius
        hyperbolic = spherical_emb * np.tanh(norm / epsilon) / norm
        return hyperbolic

    def _update_stats(self) -> None:
        """Update buffer statistics."""
        self.stats.total_examples = len(self.examples)
        self.stats.buffer_utilization = len(self.examples) / self.max_size

        # Count by type
        type_counts = {t: 0 for t in ICLExampleType}
        for example in self.examples.values():
            type_counts[example.example_type] += 1

        self.stats.vision_examples = type_counts[ICLExampleType.VISION]
        self.stats.meta_examples = type_counts[ICLExampleType.META]
        self.stats.synthesis_examples = type_counts[ICLExampleType.SYNTHESIS]
        self.stats.geometric_examples = type_counts[ICLExampleType.GEOMETRIC]

        # Compute averages
        if self.examples:
            entropies = [e.h_star_entropy for e in self.examples.values()]
            confidences = [e.confidence for e in self.examples.values()]
            novelties = [e.novelty_score for e in self.examples.values()]

            self.stats.avg_entropy = np.mean(entropies)
            self.stats.avg_confidence = np.mean(confidences)
            self.stats.avg_novelty = np.mean(novelties)

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "total_examples": self.stats.total_examples,
            "buffer_utilization": self.stats.buffer_utilization,
            "avg_entropy": self.stats.avg_entropy,
            "avg_confidence": self.stats.avg_confidence,
            "avg_novelty": self.stats.avg_novelty,
            "type_counts": {
                "vision": self.stats.vision_examples,
                "meta": self.stats.meta_examples,
                "synthesis": self.stats.synthesis_examples,
                "geometric": self.stats.geometric_examples,
            },
        }

    def _emit_add_event(self, example: ICLExample) -> None:
        """Emit example addition event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.icl.example_added",
            "data": {
                "example_id": example.example_id,
                "type": example.example_type.value,
                "h_star_entropy": example.h_star_entropy,
                "confidence": example.confidence,
                "novelty_score": example.novelty_score,
                "fitness": example.fitness,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[GeometricICLBuffer] Bus emission error: {e}")


def create_geometric_icl_buffer(
    max_size: int = 100,
    embedding_dim: int = 64,
    bus_emitter: Optional[callable] = None,
) -> GeometricICLBuffer:
    """Create geometric ICL buffer with default config."""
    return GeometricICLBuffer(
        max_size=max_size,
        embedding_dim=embedding_dim,
        bus_emitter=bus_emitter,
    )


__all__ = [
    "ICLExampleType",
    "ICLExample",
    "ICLBufferStats",
    "GeometricICLBuffer",
    "create_geometric_icl_buffer",
]
