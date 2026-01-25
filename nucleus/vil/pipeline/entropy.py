"""
VIL Entropy Normalization (H* - H-star)
Visual entropy normalization for vision frames.

Based on MetaIngest Driftflow: Semantic drift tracker with entropy normalization.

H* = H / H_max where:
- H: Shannon entropy of frame
- H_max: Maximum possible entropy (log2(N) for N pixels)

Applications:
1. Frame quality assessment
2. Novelty detection for ICL selection
3. CMP capture quality metric
4. Attractor basin identification

Version: 1.0
Date: 2026-01-25
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import base64


@dataclass
class EntropyMetrics:
    """Entropy metrics for a vision frame."""
    h_raw: float  # Raw Shannon entropy
    h_star: float  # Normalized entropy [0, 1]
    novelty_score: float  # Novelty for ICL selection
    quality_score: float  # Frame quality [0, 1]
    complexity: str  # "low", "medium", "high", "very_high"
    drift: float = 0.0  # Semantic drift from baseline


class EntropyNormalizer:
    """
    H* (H-star) entropy normalization.

    Normalizes visual entropy to [0, 1] range for:
    - Quality assessment
    - Novelty detection
    - CMP tracking
    """

    def __init__(
        self,
        baseline_window: int = 10,
        quality_threshold: float = 0.6,
        novelty_threshold: float = 0.7,
    ):
        self.baseline_window = baseline_window
        self.quality_threshold = quality_threshold
        self.novelty_threshold = novelty_threshold

        # Baseline entropy history
        self.baseline_entropies: list = []

    def compute_entropy(
        self,
        image_data: Optional[str] = None,
        pixels: Optional[np.ndarray] = None,
    ) -> EntropyMetrics:
        """
        Compute H* normalized entropy.

        Args:
            image_data: Base64-encoded image (optional)
            pixels: Raw pixel array (optional)

        Returns:
            EntropyMetrics with H*, novelty, quality
        """
        # Get pixel data
        if pixels is None:
            pixels = self._decode_image(image_data)

        if pixels is None:
            # Return zero metrics if no image
            return EntropyMetrics(
                h_raw=0.0,
                h_star=0.0,
                novelty_score=0.0,
                quality_score=0.0,
                complexity="unknown",
            )

        # Compute raw Shannon entropy
        h_raw = self._shannon_entropy(pixels)

        # Normalize to [0, 1]
        h_max = np.log2(256)  # Max entropy for 8-bit grayscale
        h_star = min(h_raw / h_max, 1.0)

        # Update baseline
        self._update_baseline(h_star)

        # Compute drift from baseline
        baseline_mean = np.mean(self.baseline_entropies) if self.baseline_entropies else 0.5
        drift = abs(h_star - baseline_mean)

        # Novelty: high drift = high novelty
        novelty_score = min(drift * 2, 1.0)

        # Quality: mid-range entropy is best
        # Too low = boring/static, Too high = noise
        quality_score = self._compute_quality(h_star)

        # Complexity classification
        complexity = self._classify_complexity(h_star)

        return EntropyMetrics(
            h_raw=h_raw,
            h_star=h_star,
            novelty_score=novelty_score,
            quality_score=quality_score,
            complexity=complexity,
            drift=drift,
        )

    def _decode_image(self, image_data: Optional[str]) -> Optional[np.ndarray]:
        """Decode base64 image to pixel array."""
        if image_data is None:
            return None

        try:
            # Decode base64
            img_bytes = base64.b64decode(image_data)

            # Simple decoding: treat as grayscale
            # In production, use PIL/OpenCV
            pixels = np.frombuffer(img_bytes, dtype=np.uint8)
            return pixels[:min(10000, len(pixels))]  # Limit size for performance
        except Exception:
            return None

    def _shannon_entropy(self, pixels: np.ndarray) -> float:
        """
        Compute Shannon entropy: H = -Σ p(x) log2(p(x))

        Args:
            pixels: Pixel value array

        Returns:
            Entropy in bits
        """
        if len(pixels) == 0:
            return 0.0

        # Compute histogram (probability distribution)
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
        probs = hist / len(pixels)

        # Remove zero probabilities
        probs = probs[probs > 0]

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    def _update_baseline(self, h_star: float) -> None:
        """Update rolling baseline of entropies."""
        self.baseline_entropies.append(h_star)
        if len(self.baseline_entropies) > self.baseline_window:
            self.baseline_entropies.pop(0)

    def _compute_quality(self, h_star: float) -> float:
        """
        Compute quality score from H*.

        Optimal range: 0.5-0.8 (enough detail, not too noisy)
        """
        if 0.5 <= h_star <= 0.8:
            # Sweet spot
            return 1.0
        elif 0.3 <= h_star < 0.5:
            # A bit simple
            return 0.7 + (h_star - 0.3) * 1.5
        elif 0.8 < h_star <= 0.95:
            # A bit noisy
            return 1.0 - (h_star - 0.8) * 3
        else:
            # Too simple or too noisy
            return max(0.0, 1.0 - abs(h_star - 0.65) * 2)

    def _classify_complexity(self, h_star: float) -> str:
        """Classify visual complexity from H*."""
        if h_star < 0.3:
            return "low"
        elif h_star < 0.5:
            return "medium"
        elif h_star < 0.7:
            return "high"
        else:
            return "very_high"

    def is_novel(self, metrics: EntropyMetrics) -> bool:
        """Check if frame is novel (high drift)."""
        return metrics.novelty_score > self.novelty_threshold

    def is_high_quality(self, metrics: EntropyMetrics) -> bool:
        """Check if frame is high quality."""
        return metrics.quality_score > self.quality_threshold

    def get_baseline_stats(self) -> Dict[str, float]:
        """Get baseline entropy statistics."""
        if not self.baseline_entropies:
            return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5}

        arr = np.array(self.baseline_entropies)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }


# Global normalizer instance
_global_normalizer: Optional[EntropyNormalizer] = None


def get_entropy_normalizer() -> EntropyNormalizer:
    """Get global entropy normalizer instance."""
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = EntropyNormalizer()
    return _global_normalizer


def compute_h_star(
    image_data: Optional[str] = None,
    pixels: Optional[np.ndarray] = None,
) -> EntropyMetrics:
    """
    Compute H* entropy using global normalizer.

    Convenience function for quick entropy computation.
    """
    normalizer = get_entropy_normalizer()
    return normalizer.compute_entropy(image_data, pixels)


__all__ = [
    "EntropyMetrics",
    "EntropyNormalizer",
    "compute_h_star",
    "get_entropy_normalizer",
]
