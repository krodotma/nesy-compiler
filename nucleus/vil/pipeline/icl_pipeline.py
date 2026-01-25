"""
VIL Vision-to-ICL Pipeline
Connects vision frames to in-context learning examples.

This pipeline:
1. Captures vision frames with H* quality filtering
2. Converts frames to ICL examples with embeddings
3. Manages ICL buffer with novelty-based selection
4. Emits VIL learning events for each example
5. Integrates with VILCoordinator for CMP tracking

Based on ICL+ (In-Context Learning Plus) from VIL plan.

Version: 1.0
Date: 2026-01-25
"""

import time
import base64
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    from theia.vlm.icl import ScreenshotICL, ScreenshotExample
    THEIA_ICL_AVAILABLE = True
except ImportError:
    THEIA_ICL_AVAILABLE = False

from nucleus.vil.events import (
    VILEventType,
    LearningEvent,
    create_learning_event,
    create_trace_id,
    CmpMetrics,
)
from nucleus.vil.pipeline.entropy import (
    EntropyMetrics,
    compute_h_star,
    EntropyNormalizer,
)


class ICLStrategy(str, Enum):
    """ICL example selection strategies."""

    NEAREST = "nearest"  # Closest in embedding space
    DIVERSE = "diverse"  # Maximize diversity
    RECENT = "recent"  # Most recent
    RANDOM = "random"  # Random sampling
    NOVEL = "novel"  # High novelty (high H* drift)


@dataclass
class ICLExample:
    """
    ICL example from vision frame.

    Combines:
    - Visual embedding
    - Text prompt/response
    - H* entropy metrics
    - Success outcome
    """

    id: str
    image_data: str  # Base64 encoded
    embedding: np.ndarray
    prompt: str
    response: str
    success: bool
    timestamp: float
    entropy: EntropyMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large fields)."""
        return {
            "id": self.id,
            "prompt": self.prompt[:100],
            "response": self.response[:100],
            "success": self.success,
            "timestamp": self.timestamp,
            "h_star": self.entropy.h_star,
            "quality": self.entropy.quality_score,
            "novelty": self.entropy.novelty_score,
            "complexity": self.entropy.complexity,
            "embedding_dim": len(self.embedding),
        }


class VisionToICLPipeline:
    """
    Vision-to-ICL conversion pipeline.

    Flow:
    1. Capture vision frame
    2. Compute H* entropy
    3. Filter by quality
    4. Create embedding
    5. Store as ICL example
    6. Emit learning event
    """

    def __init__(
        self,
        buffer_size: int = 5,
        quality_threshold: float = 0.6,
        novelty_threshold: float = 0.7,
        embedding_dim: int = 32,
        bus_emitter: Optional[callable] = None,
    ):
        self.buffer_size = buffer_size
        self.quality_threshold = quality_threshold
        self.novelty_threshold = novelty_threshold
        self.embedding_dim = embedding_dim
        self.bus_emitter = bus_emitter

        # ICL buffer
        self.examples: List[ICLExample] = []

        # Entropy normalizer
        self.entropy_normalizer = EntropyNormalizer()

        # Theia ICL (if available)
        if THEIA_ICL_AVAILABLE:
            self.theia_icl = ScreenshotICL(max_examples=buffer_size)
        else:
            self.theia_icl = None

        # Statistics
        self.stats = {
            "frames_seen": 0,
            "frames_accepted": 0,
            "frames_rejected": 0,
            "icl_examples": 0,
            "avg_h_star": 0.0,
            "avg_quality": 0.0,
        }

    async def process_frame(
        self,
        image_data: str,
        prompt: str,
        response: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ICLExample]:
        """
        Process vision frame into ICL example.

        Args:
            image_data: Base64-encoded image
            prompt: Text prompt
            response: Model response
            success: Whether outcome was successful
            metadata: Additional metadata

        Returns:
            ICLExample if quality threshold met, None otherwise
        """
        self.stats["frames_seen"] += 1

        # Compute H* entropy
        entropy_metrics = compute_h_star(image_data=image_data)

        # Update stats
        self.stats["avg_h_star"] = (
            (self.stats["avg_h_star"] * (self.stats["frames_seen"] - 1) + entropy_metrics.h_star) /
            self.stats["frames_seen"]
        )
        self.stats["avg_quality"] = (
            (self.stats["avg_quality"] * (self.stats["frames_seen"] - 1) + entropy_metrics.quality_score) /
            self.stats["frames_seen"]
        )

        # Filter by quality
        if entropy_metrics.quality_score < self.quality_threshold:
            self.stats["frames_rejected"] += 1
            return None

        # Create embedding
        embedding = self._create_embedding(image_data)

        # Create ICL example
        example = ICLExample(
            id=self._create_id(),
            image_data=image_data,
            embedding=embedding,
            prompt=prompt,
            response=response,
            success=success,
            timestamp=time.time(),
            entropy=entropy_metrics,
            metadata=metadata or {},
        )

        # Add to buffer
        self._add_to_buffer(example)

        # Also add to Theia ICL if available
        if self.theia_icl and success:
            theia_example = ScreenshotExample.from_screenshot(
                image_b64=image_data,
                action=response[:100],
                outcome="success" if success else "failure",
            )
            self.theia_icl.add_example(theia_example)

        # Emit learning event
        await self._emit_learning_event(example)

        self.stats["frames_accepted"] += 1
        self.stats["icl_examples"] = len(self.examples)

        return example

    def select_examples(
        self,
        query_embedding: Optional[np.ndarray] = None,
        k: int = 3,
        strategy: ICLStrategy = ICLStrategy.NEAREST,
    ) -> List[ICLExample]:
        """
        Select k ICL examples using strategy.

        Args:
            query_embedding: Query embedding for similarity
            k: Number of examples to return
            strategy: Selection strategy

        Returns:
            List of selected examples
        """
        if not self.examples:
            return []

        if strategy == ICLStrategy.RECENT:
            # Most recent
            return sorted(self.examples, key=lambda e: e.timestamp, reverse=True)[:k]

        elif strategy == ICLStrategy.RANDOM:
            # Random sampling
            indices = np.random.choice(len(self.examples), min(k, len(self.examples)), replace=False)
            return [self.examples[i] for i in indices]

        elif strategy == ICLStrategy.NOVEL:
            # High novelty (high drift)
            return sorted(self.examples, key=lambda e: e.entropy.novelty_score, reverse=True)[:k]

        elif strategy == ICLStrategy.NEAREST:
            # Closest in embedding space
            if query_embedding is None:
                return self.examples[:k]

            similarities = [
                self._cosine_sim(query_embedding, e.embedding)
                for e in self.examples
            ]
            top_k_idx = np.argsort(similarities)[-k:][::-1]
            return [self.examples[i] for i in top_k_idx]

        elif strategy == ICLStrategy.DIVERSE:
            # Maximize diversity (farthest apart)
            if len(self.examples) <= k:
                return self.examples[:]

            selected = [self.examples[0]]
            remaining = self.examples[1:]

            while len(selected) < k and remaining:
                # Find farthest from selected
                farthest_idx = max(
                    range(len(remaining)),
                    key=lambda i: min(
                        self._cosine_sim(remaining[i].embedding, s.embedding)
                        for s in selected
                    )
                )
                selected.append(remaining.pop(farthest_idx))

            return selected

        else:
            return self.examples[:k]

    def get_icl_context(
        self,
        query_embedding: Optional[np.ndarray] = None,
        k: int = 3,
        strategy: ICLStrategy = ICLStrategy.NEAREST,
    ) -> str:
        """
        Get ICL context string for prompting.

        Returns formatted examples for LLM context.
        """
        examples = self.select_examples(query_embedding, k, strategy)

        if not examples:
            return "No relevant past examples found."

        context = "Relevant past examples:\n\n"
        for i, ex in enumerate(examples, 1):
            context += f"Example {i}:\n"
            context += f"  Prompt: {ex.prompt[:150]}...\n"
            context += f"  Response: {ex.response[:150]}...\n"
            context += f"  Quality: {ex.entropy.quality_score:.2f} (H*: {ex.entropy.h_star:.2f})\n"
            context += f"  Success: {ex.success}\n\n"

        return context

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "buffer_size": len(self.examples),
            "max_buffer_size": self.buffer_size,
            "acceptance_rate": (
                self.stats["frames_accepted"] / max(1, self.stats["frames_seen"])
            ),
            "baseline_entropy": self.entropy_normalizer.get_baseline_stats(),
            "theia_icl_available": THEIA_ICL_AVAILABLE and self.theia_icl is not None,
        }

    # === Private Methods ===

    def _create_embedding(self, image_data: str) -> np.ndarray:
        """Create embedding from base64 image."""
        # Simple hash-based embedding (would use real vision encoder)
        import hashlib
        h = hashlib.sha256(image_data.encode()).digest()
        embedding = np.frombuffer(h[:self.embedding_dim], dtype=np.uint8).astype(np.float32) / 255.0
        return embedding

    def _create_id(self) -> str:
        """Create unique example ID."""
        return f"icl_{int(time.time() * 1000)}_{len(self.examples)}"

    def _add_to_buffer(self, example: ICLExample) -> None:
        """Add example to buffer with size management."""
        self.examples.append(example)

        # Maintain size
        if len(self.examples) > self.buffer_size:
            # Keep high-quality and novel examples
            self.examples = sorted(
                self.examples,
                key=lambda e: (e.entropy.quality_score, e.entropy.novelty_score),
                reverse=True
            )[:self.buffer_size]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    async def _emit_learning_event(self, example: ICLExample) -> None:
        """Emit learning event to bus."""
        if not self.bus_emitter:
            return

        event = LearningEvent(
            event_type=VILEventType.LEARN_ICL_EXAMPLE,
            task_id=example.id,
            icl_examples=len(self.examples),
            icl_buffer_size=self.buffer_size,
            source="vil-icl-pipeline",
            data={
                "example": example.to_dict(),
                "h_star": example.entropy.h_star,
                "quality": example.entropy.quality_score,
            },
        )

        try:
            self.bus_emitter(event.to_bus_event())
        except Exception as e:
            print(f"[VisionToICLPipeline] Bus emission error: {e}")


def create_icl_pipeline(
    buffer_size: int = 5,
    bus_emitter: Optional[callable] = None,
) -> VisionToICLPipeline:
    """Create vision-to-ICL pipeline with default config."""
    return VisionToICLPipeline(
        buffer_size=buffer_size,
        bus_emitter=bus_emitter,
    )


__all__ = [
    "ICLStrategy",
    "ICLExample",
    "VisionToICLPipeline",
    "create_icl_pipeline",
]
