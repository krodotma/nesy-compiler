"""
VIL ICL+ Quality Scoring
H* entropy-based quality assessment for ICL examples.

Features:
1. H* normalized entropy calculation
2. Quality scoring with multi-factor metrics
3. Novelty detection from entropy drift
4. Deduplication via entropy fingerprinting
5. CMP-weighted quality ranking

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.stats import entropy as scipy_entropy

from nucleus.vil.pipeline.entropy import EntropyNormalizer, H_STAR_MAX
from nucleus.vil.cmp.manager import PHI


class QualityThreshold(str, Enum):
    """Quality threshold levels."""

    EXCELLENT = "excellent"  # H* in [0.4, 0.7], high confidence
    GOOD = "good"  # H* in [0.3, 0.8], moderate confidence
    FAIR = "fair"  # H* in [0.2, 0.9], low confidence
    POOR = "poor"  # H* outside acceptable range


@dataclass
class QualityMetrics:
    """
    Quality metrics for ICL example.

    Contains:
    - H* normalized entropy
    - Confidence score
    - Novelty score
    - Quality threshold
    - Deduplication fingerprint
    """

    h_star_entropy: float = 0.5
    raw_entropy: float = 0.0
    entropy_drift: float = 0.0
    confidence: float = 0.8
    novelty_score: float = 0.5
    quality_threshold: QualityThreshold = QualityThreshold.GOOD
    fingerprint: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "h_star_entropy": self.h_star_entropy,
            "raw_entropy": self.raw_entropy,
            "entropy_drift": self.entropy_drift,
            "confidence": self.confidence,
            "novelty_score": self.novelty_score,
            "quality_threshold": self.quality_threshold.value,
        }


@dataclass
class QualityReport:
    """
    Quality assessment report.

    Contains:
    - Overall quality score
    - Per-metric scores
    - Recommendations
    - CMP fitness estimate
    """

    overall_score: float
    metrics: QualityMetrics
    is_acceptable: bool
    recommendations: List[str]
    cmp_fitness_estimate: float
    assessment_time_ms: float


class ICLQualityScorer:
    """
    Quality scorer for ICL examples.

    Features:
    1. H* entropy calculation
    2. Multi-factor quality scoring
    3. Novelty detection from drift
    4. Deduplication via fingerprints
    5. CMP fitness estimation
    """

    def __init__(
        self,
        entropy_normalizer: Optional[EntropyNormalizer] = None,
        h_star_range: Tuple[float, float] = (0.4, 0.7),
        novelty_threshold: float = 0.2,
        bus_emitter: Optional[callable] = None,
    ):
        self.entropy_normalizer = entropy_normalizer or EntropyNormalizer()
        self.h_star_range = h_star_range  # Optimal H* range
        self.novelty_threshold = novelty_threshold
        self.bus_emitter = bus_emitter

        # Entropy baseline
        self.entropy_baseline = 0.5
        self.entropy_history: List[float] = []

        # Deduplication registry
        self.fingerprint_registry: Dict[str, float] = {}  # fingerprint -> timestamp

        # Statistics
        self.stats = {
            "assessments": 0,
            "acceptable_count": 0,
            "excellent_count": 0,
            "good_count": 0,
            "fair_count": 0,
            "poor_count": 0,
            "avg_h_star": 0.0,
            "avg_novelty": 0.0,
        }

    def assess_quality(
        self,
        image_data: Optional[str] = None,
        text_data: Optional[str] = None,
        confidence: float = 0.8,
        update_baseline: bool = True,
    ) -> QualityReport:
        """
        Assess quality of ICL example.

        Args:
            image_data: Optional base64 image
            text_data: Optional text input
            confidence: VLM confidence score
            update_baseline: Whether to update entropy baseline

        Returns:
            QualityReport
        """
        start_time = time.time()

        # Calculate H* entropy
        h_star, raw_entropy = self._calculate_h_star(image_data, text_data)

        # Calculate entropy drift
        entropy_drift = self._calculate_drift(h_star)
        if update_baseline:
            self._update_baseline(h_star)

        # Calculate novelty score
        novelty_score = self._calculate_novelty(h_star, entropy_drift)

        # Determine quality threshold
        quality_threshold = self._determine_threshold(h_star, confidence)

        # Generate fingerprint for deduplication
        fingerprint = self._generate_fingerprint(h_star, raw_entropy, text_data)

        # Create metrics
        metrics = QualityMetrics(
            h_star_entropy=h_star,
            raw_entropy=raw_entropy,
            entropy_drift=entropy_drift,
            confidence=confidence,
            novelty_score=novelty_score,
            quality_threshold=quality_threshold,
            fingerprint=fingerprint,
        )

        # Calculate overall quality score
        overall_score = self._calculate_overall_score(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        # Estimate CMP fitness
        cmp_fitness = self._estimate_cmp_fitness(metrics)

        # Determine if acceptable
        is_acceptable = quality_threshold in [QualityThreshold.EXCELLENT, QualityThreshold.GOOD]

        assessment_time = (time.time() - start_time) * 1000

        report = QualityReport(
            overall_score=overall_score,
            metrics=metrics,
            is_acceptable=is_acceptable,
            recommendations=recommendations,
            cmp_fitness_estimate=cmp_fitness,
            assessment_time_ms=assessment_time,
        )

        # Update stats
        self._update_stats(metrics, is_acceptable)

        # Emit event
        self._emit_quality_event(report)

        return report

    def _calculate_h_star(
        self,
        image_data: Optional[str],
        text_data: Optional[str],
    ) -> Tuple[float, float]:
        """Calculate H* normalized entropy."""
        if image_data:
            # Use entropy normalizer for image
            h_star = self.entropy_normalizer.compute_entropy(image_data=image_data)
            raw_entropy = self.entropy_normalizer.raw_history[-1] if self.entropy_normalizer.raw_history else 0.0
        elif text_data:
            # Calculate text entropy
            text_bytes = text_data.encode('utf-8')
            byte_counts = np.bincount([b for b in text_bytes], minlength=256)
            byte_probs = byte_counts / len(text_bytes)
            raw_entropy = float(scipy_entropy(byte_probs + 1e-10))
            h_star = min(raw_entropy / 8.0, 1.0)  # Normalize to [0, 1]
        else:
            # Default
            h_star = 0.5
            raw_entropy = 4.0

        return h_star, raw_entropy

    def _calculate_drift(self, h_star: float) -> float:
        """Calculate entropy drift from baseline."""
        return abs(h_star - self.entropy_baseline)

    def _update_baseline(self, h_star: float) -> None:
        """Update entropy baseline with EMA."""
        alpha = 0.1
        self.entropy_baseline = alpha * h_star + (1 - alpha) * self.entropy_baseline
        self.entropy_history.append(h_star)

        # Keep history bounded
        if len(self.entropy_history) > 1000:
            self.entropy_history = self.entropy_history[-1000:]

    def _calculate_novelty(self, h_star: float, drift: float) -> float:
        """Calculate novelty score from entropy and drift."""
        # High drift = high novelty
        # Medium H* = good information content
        h_score = 1.0 - abs(h_star - 0.5) * 2  # Peak at 0.5
        novelty = (h_score * 0.5 + drift * 0.5)
        return float(np.clip(novelty, 0.0, 1.0))

    def _determine_threshold(
        self,
        h_star: float,
        confidence: float,
    ) -> QualityThreshold:
        """Determine quality threshold."""
        h_min, h_max = self.h_star_range

        if h_min <= h_star <= h_max and confidence > 0.8:
            return QualityThreshold.EXCELLENT
        elif 0.3 <= h_star <= 0.8 and confidence > 0.6:
            return QualityThreshold.GOOD
        elif 0.2 <= h_star <= 0.9 and confidence > 0.4:
            return QualityThreshold.FAIR
        else:
            return QualityThreshold.POOR

    def _generate_fingerprint(
        self,
        h_star: float,
        raw_entropy: float,
        text_data: Optional[str],
    ) -> str:
        """Generate deduplication fingerprint."""
        # Combine H* entropy and text hash
        if text_data:
            text_hash = hash(text_data[:100])  # Hash first 100 chars
            combined = f"{h_star:.3f}_{raw_entropy:.3f}_{text_hash}"
        else:
            combined = f"{h_star:.3f}_{raw_entropy:.3f}"

        import hashlib
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score."""
        # Phi-weighted combination
        score = (
            metrics.confidence * PHI +
            metrics.h_star_entropy * 1.0 +
            metrics.novelty_score * (1 / PHI)
        )

        # Normalize
        max_score = PHI + 1.0 + (1 / PHI)
        return float(score / max_score)

    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        if metrics.h_star_entropy < self.h_star_range[0]:
            recommendations.append("Increase visual complexity or diversity")
        elif metrics.h_star_entropy > self.h_star_range[1]:
            recommendations.append("Reduce visual noise or clutter")

        if metrics.confidence < 0.7:
            recommendations.append("Improve VLM confidence with clearer prompts")

        if metrics.novelty_score < 0.3:
            recommendations.append("Example may be duplicate of existing ICL")

        if metrics.entropy_drift > 0.3:
            recommendations.append("High entropy drift - consider reviewing example")

        return recommendations

    def _estimate_cmp_fitness(self, metrics: QualityMetrics) -> float:
        """Estimate CMP fitness from quality metrics."""
        # Use golden ratio weighting
        fitness = (
            metrics.confidence * PHI +
            metrics.h_star_entropy * 1.0 +
            metrics.novelty_score * (1 / PHI)
        )
        return fitness

    def is_duplicate(
        self,
        fingerprint: str,
        time_window: float = 3600.0,
    ) -> bool:
        """
        Check if example is duplicate based on fingerprint.

        Args:
            fingerprint: Example fingerprint
            time_window: Time window for deduplication (seconds)

        Returns:
            True if duplicate found
        """
        if fingerprint not in self.fingerprint_registry:
            return False

        last_seen = self.fingerprint_registry[fingerprint]
        age = time.time() - last_seen

        return age < time_window

    def register_fingerprint(self, fingerprint: str) -> None:
        """Register fingerprint for deduplication."""
        self.fingerprint_registry[fingerprint] = time.time()

    def cleanup_fingerprints(self, max_age: float = 86400.0) -> int:
        """Clean up old fingerprints."""
        current_time = time.time()
        to_remove = [
            fp for fp, ts in self.fingerprint_registry.items()
            if current_time - ts > max_age
        ]

        for fp in to_remove:
            del self.fingerprint_registry[fp]

        return len(to_remove)

    def _update_stats(self, metrics: QualityMetrics, is_acceptable: bool) -> None:
        """Update quality statistics."""
        self.stats["assessments"] += 1

        if is_acceptable:
            self.stats["acceptable_count"] += 1

        threshold = metrics.quality_threshold
        self.stats[f"{threshold.value}_count"] += 1

        # Update averages
        n = self.stats["assessments"]
        self.stats["avg_h_star"] = (
            (self.stats["avg_h_star"] * (n - 1) + metrics.h_star_entropy) / n
        )
        self.stats["avg_novelty"] = (
            (self.stats["avg_novelty"] * (n - 1) + metrics.novelty_score) / n
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get quality statistics."""
        acceptance_rate = (
            self.stats["acceptable_count"] / self.stats["assessments"]
            if self.stats["assessments"] > 0 else 0.0
        )

        return {
            **self.stats,
            "acceptance_rate": acceptance_rate,
            "entropy_baseline": self.entropy_baseline,
            "fingerprint_registry_size": len(self.fingerprint_registry),
        }

    def _emit_quality_event(self, report: QualityReport) -> None:
        """Emit quality assessment event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.icl.quality_assessment",
            "data": {
                "overall_score": report.overall_score,
                "h_star_entropy": report.metrics.h_star_entropy,
                "confidence": report.metrics.confidence,
                "novelty_score": report.metrics.novelty_score,
                "quality_threshold": report.metrics.quality_threshold.value,
                "is_acceptable": report.is_acceptable,
                "cmp_fitness_estimate": report.cmp_fitness_estimate,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[ICLQualityScorer] Bus emission error: {e}")


def create_icl_quality_scorer(
    h_star_range: Tuple[float, float] = (0.4, 0.7),
    novelty_threshold: float = 0.2,
    bus_emitter: Optional[callable] = None,
) -> ICLQualityScorer:
    """Create ICL quality scorer with default config."""
    return ICLQualityScorer(
        h_star_range=h_star_range,
        novelty_threshold=novelty_threshold,
        bus_emitter=bus_emitter,
    )


__all__ = [
    "QualityThreshold",
    "QualityMetrics",
    "QualityReport",
    "ICLQualityScorer",
    "create_icl_quality_scorer",
]
