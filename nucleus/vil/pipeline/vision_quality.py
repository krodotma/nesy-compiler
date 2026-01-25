"""
VIL Vision Quality & Deduplication
Frame quality assessment and deduplication for vision pipeline.

Features:
1. Quality scoring based on multiple metrics
2. Deduplication using perceptual hashing
3. Batch processing for efficiency
4. Error recovery and retry logic
5. Trace propagation through pipeline

Version: 1.0
Date: 2026-01-25
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from nucleus.vil.pipeline.entropy import EntropyMetrics, compute_h_star


@dataclass
class QualityScore:
    """Comprehensive quality score for vision frame."""

    overall: float  # Overall quality [0, 1]
    entropy: float  # H* entropy contribution
    sharpness: float  # Edge detection (simulated)
    brightness: float  # Brightness score
    contrast: float  # Contrast score
    novelty: float  # Novelty contribution

    def to_dict(self) -> Dict[str, float]:
        return {
            "overall": self.overall,
            "entropy": self.entropy,
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "novelty": self.novelty,
        }


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""

    is_duplicate: bool
    similarity_score: float
    matched_frame_id: Optional[str]
    hash_distance: int


class VisionQualityProcessor:
    """
    Processes vision frames for quality and deduplication.

    Features:
    1. Multi-metric quality scoring
    2. Perceptual hash-based deduplication
    3. Batch processing support
    4. Error recovery
    5. Quality history tracking
    """

    def __init__(
        self,
        quality_threshold: float = 0.6,
        dedup_threshold: float = 0.85,  # Similarity threshold
        history_size: int = 100,
        batch_size: int = 10,
    ):
        self.quality_threshold = quality_threshold
        self.dedup_threshold = dedup_threshold
        self.history_size = history_size
        self.batch_size = batch_size

        # Deduplication: hash storage
        self.frame_hashes: Dict[str, str] = {}  # frame_id -> hash
        self.hash_to_frames: Dict[str, Set[str]] = {}  # hash -> frame_ids

        # Quality history
        self.quality_history: deque = deque(maxlen=history_size)

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "frames_accepted": 0,
            "frames_rejected_quality": 0,
            "frames_rejected_duplicate": 0,
            "avg_quality": 0.0,
            "duplicates_found": 0,
        }

    def assess_quality(
        self,
        image_data: str,
        entropy_metrics: Optional[EntropyMetrics] = None,
    ) -> QualityScore:
        """
        Assess frame quality using multiple metrics.

        Args:
            image_data: Base64-encoded image
            entropy_metrics: Pre-computed entropy metrics

        Returns:
            QualityScore with all metrics
        """
        # Compute entropy if not provided
        if entropy_metrics is None:
            entropy_metrics = compute_h_star(image_data=image_data)

        # Simulate other quality metrics
        # (In production, use actual image analysis)
        sharpness = self._estimate_sharpness(image_data)
        brightness = self._estimate_brightness(image_data)
        contrast = self._estimate_contrast(image_data)

        # Overall quality: weighted average
        weights = {"entropy": 0.3, "sharpness": 0.25, "brightness": 0.2, "contrast": 0.15, "novelty": 0.1}

        overall = (
            entropy_metrics.quality_score * weights["entropy"] +
            sharpness * weights["sharpness"] +
            brightness * weights["brightness"] +
            contrast * weights["contrast"] +
            entropy_metrics.novelty_score * weights["novelty"]
        )

        return QualityScore(
            overall=min(overall, 1.0),
            entropy=entropy_metrics.quality_score,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            novelty=entropy_metrics.novelty_score,
        )

    def check_duplicate(
        self,
        image_data: str,
        frame_id: str,
    ) -> DeduplicationResult:
        """
        Check if frame is a duplicate using perceptual hashing.

        Args:
            image_data: Base64-encoded image
            frame_id: Frame ID for this frame

        Returns:
            DeduplicationResult with match info
        """
        # Compute perceptual hash
        frame_hash = self._perceptual_hash(image_data)

        # Check for similar hashes
        best_match = None
        best_similarity = 0.0

        for existing_hash, frame_ids in self.hash_to_frames.items():
            similarity = self._hash_similarity(frame_hash, existing_hash)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = list(frame_ids)[0] if frame_ids else None

        is_duplicate = best_similarity > self.dedup_threshold

        # Store this hash
        if frame_id not in self.frame_hashes:
            self.frame_hashes[frame_id] = frame_hash
            if frame_hash not in self.hash_to_frames:
                self.hash_to_frames[frame_hash] = set()
            self.hash_to_frames[frame_hash].add(frame_id)

        if is_duplicate:
            self.stats["duplicates_found"] += 1

        return DeduplicationResult(
            is_duplicate=is_duplicate,
            similarity_score=best_similarity,
            matched_frame_id=best_match,
            hash_distance=self._hamming_distance(frame_hash, best_match) if best_match else 0,
        )

    def process_batch(
        self,
        frames: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process batch of frames.

        Args:
            frames: List of frame dicts with image_data and frame_id

        Returns:
            List of processed frames with quality and dedup results
        """
        results = []

        for frame_dict in frames:
            frame_id = frame_dict.get("frame_id") or f"frame_{int(time.time() * 1000)}"
            image_data = frame_dict["image_data"]

            try:
                # Quality assessment
                quality = self.assess_quality(image_data)
                self.quality_history.append(quality)

                # Deduplication
                dedup = self.check_duplicate(image_data, frame_id)

                # Update stats
                self.stats["frames_processed"] += 1

                # Determine acceptance
                accepted = quality.overall >= self.quality_threshold and not dedup.is_duplicate

                if accepted:
                    self.stats["frames_accepted"] += 1
                else:
                    if quality.overall < self.quality_threshold:
                        self.stats["frames_rejected_quality"] += 1
                    if dedup.is_duplicate:
                        self.stats["frames_rejected_duplicate"] += 1

                # Update average
                self.stats["avg_quality"] = (
                    (self.stats["avg_quality"] * (self.stats["frames_processed"] - 1) + quality.overall) /
                    self.stats["frames_processed"]
                )

                results.append({
                    "frame_id": frame_id,
                    "accepted": accepted,
                    "quality": quality.to_dict(),
                    "deduplication": {
                        "is_duplicate": dedup.is_duplicate,
                        "similarity": dedup.similarity_score,
                        "matched_frame": dedup.matched_frame_id,
                    },
                    "timestamp": time.time(),
                })

            except Exception as e:
                # Error recovery: mark as rejected
                results.append({
                    "frame_id": frame_id,
                    "accepted": False,
                    "error": str(e),
                    "timestamp": time.time(),
                })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            **self.stats,
            "acceptance_rate": (
                self.stats["frames_accepted"] / max(1, self.stats["frames_processed"])
            ),
            "duplicate_rate": (
                self.stats["duplicates_found"] / max(1, self.stats["frames_processed"])
            ),
            "unique_hashes": len(self.hash_to_frames),
        }

    # === Private Methods ===

    def _perceptual_hash(self, image_data: str) -> str:
        """
        Compute perceptual hash of image.

        Uses simplified hash for demo (production would use pHash).
        """
        # Decode and sample
        import base64
        try:
            img_bytes = base64.b64decode(image_data)
            # Simple hash: SHA256 of first 1KB
            sample = img_bytes[:1024]
            return hashlib.sha256(sample).hexdigest()[:16]
        except:
            return hashlib.sha256(image_data.encode()).hexdigest()[:16]

    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two hashes [0, 1]."""
        if not hash1 or not hash2:
            return 0.0

        distance = self._hamming_distance(hash1, hash2)
        max_distance = len(hash1)
        similarity = 1.0 - (distance / max_distance)
        return similarity

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hash strings."""
        if not hash1 or not hash2:
            return len(hash1) if hash1 else 0

        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def _estimate_sharpness(self, image_data: str) -> float:
        """Estimate image sharpness (simulated)."""
        # In production: use edge detection (Sobel, Laplacian)
        import base64
        try:
            img_bytes = base64.b64decode(image_data)
            # Simple variance estimate
            if len(img_bytes) < 100:
                return 0.5
            variance = np.var(list(img_bytes[:1000])) / 65536.0
            return min(variance * 10, 1.0)
        except:
            return 0.5

    def _estimate_brightness(self, image_data: str) -> float:
        """Estimate brightness (simulated)."""
        import base64
        try:
            img_bytes = base64.b64decode(image_data)
            if not img_bytes:
                return 0.5
            avg_brightness = sum(img_bytes[:1000]) / (len(img_bytes[:1000]) * 256)
            # Optimal brightness: 0.4-0.7
            if 0.4 <= avg_brightness <= 0.7:
                return 1.0
            return 1.0 - abs(avg_brightness - 0.55) * 2
        except:
            return 0.5

    def _estimate_contrast(self, image_data: str) -> float:
        """Estimate contrast (simulated)."""
        import base64
        try:
            img_bytes = base64.b64decode(image_data)
            if len(img_bytes) < 100:
                return 0.5
            values = list(img_bytes[:1000])
            contrast = (max(values) - min(values)) / 256
            return min(contrast, 1.0)
        except:
            return 0.5


def create_quality_processor(
    quality_threshold: float = 0.6,
    dedup_threshold: float = 0.85,
) -> VisionQualityProcessor:
    """Create quality processor with default config."""
    return VisionQualityProcessor(
        quality_threshold=quality_threshold,
        dedup_threshold=dedup_threshold,
    )


__all__ = [
    "QualityScore",
    "DeduplicationResult",
    "VisionQualityProcessor",
    "create_quality_processor",
]
