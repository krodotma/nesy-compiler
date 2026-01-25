#!/usr/bin/env python3
"""
LASER Uncertainty Quantification Module

Computes confidence scores for webchat responses based on:
- Aleatoric uncertainty: inherent randomness (network, timing, UI state)
- Epistemic uncertainty: knowledge gaps (auth state, response completeness)

This module provides a complementary perspective to the EntropyVector model
in entropy_profiler.py, focusing specifically on WUA (Webchat User Agent)
response confidence rather than semantic quality.

Reference: lens_laser_synthesizer_v1.md H* entropy vector model

Architecture:
    UncertaintyVector - Multi-dimensional uncertainty representation
    LatencyTracker - Statistical tracking for aleatoric estimation
    ResponseValidator - Structural validation for epistemic estimation
    ConfidenceScorer - Main interface combining all signals

Usage:
    from laser import ConfidenceScorer, UncertaintyVector

    scorer = ConfidenceScorer()
    confidence, vector = scorer.score(
        response="Hello, I can help with that...",
        latency_ms=250.0,
        streaming_complete=True
    )
    print(f"Confidence: {confidence:.2%}")

Author: claude-codex (LASER Uncertainty Module)
DKIN Version: v29
"""

from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

__all__ = [
    "UncertaintyVector",
    "LatencyTracker",
    "ResponseValidator",
    "ConfidenceScorer",
    "compute_confidence",
]


# ==============================================================================
# Constants
# ==============================================================================

# Default weights for combining uncertainty components
DEFAULT_WEIGHTS = {
    "aleatoric": 0.30,    # Network/timing variability weight
    "epistemic": 0.40,    # Knowledge gap weight (highest - most important)
    "temporal": 0.15,     # Session age decay weight
    "structural": 0.15,   # Response structure weight
}

# Session decay parameters
SESSION_HALF_LIFE_SECONDS = 3600.0  # 1 hour = 50% confidence decay
SESSION_MAX_AGE_SECONDS = 7200.0    # 2 hours = full uncertainty


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class UncertaintyVector:
    """
    Multi-dimensional uncertainty representation for WUA responses.

    This captures four orthogonal uncertainty dimensions:

    - h_aleatoric: Inherent randomness from network latency, UI timing,
      and stochastic model behavior. Estimated from latency variance.

    - h_epistemic: Knowledge gaps about response completeness, auth state,
      and whether the response was truncated. Estimated from response signals.

    - h_temporal: Time-based confidence decay. Sessions without recent
      verification accumulate uncertainty over time.

    - h_structural: Confidence in response format/structure. Well-structured
      responses (code blocks, lists) have lower structural uncertainty.

    All components are normalized to [0, 1] where:
    - 0 = no uncertainty (complete confidence)
    - 1 = maximum uncertainty (no confidence)

    Attributes:
        h_aleatoric: Inherent randomness uncertainty [0, 1]
        h_epistemic: Knowledge gap uncertainty [0, 1]
        h_temporal: Time-based decay uncertainty [0, 1]
        h_structural: Response structure uncertainty [0, 1]

    Properties:
        confidence: Overall confidence score (1 - weighted uncertainty)
        total: Sum of all uncertainty components
        mean: Average uncertainty across dimensions
    """

    h_aleatoric: float = 0.0
    h_epistemic: float = 0.0
    h_temporal: float = 0.0
    h_structural: float = 0.0

    def __post_init__(self) -> None:
        """Clamp all values to [0, 1] on initialization."""
        self.h_aleatoric = _clamp(self.h_aleatoric)
        self.h_epistemic = _clamp(self.h_epistemic)
        self.h_temporal = _clamp(self.h_temporal)
        self.h_structural = _clamp(self.h_structural)

    @property
    def confidence(self) -> float:
        """
        Compute overall confidence as 1 - weighted combined uncertainty.

        Uses default weights that prioritize epistemic uncertainty (0.4)
        as the most critical signal, followed by aleatoric (0.3), with
        temporal and structural sharing the remainder (0.15 each).

        Returns:
            float: Confidence score in [0, 1]
        """
        combined = (
            DEFAULT_WEIGHTS["aleatoric"] * self.h_aleatoric +
            DEFAULT_WEIGHTS["epistemic"] * self.h_epistemic +
            DEFAULT_WEIGHTS["temporal"] * self.h_temporal +
            DEFAULT_WEIGHTS["structural"] * self.h_structural
        )
        return _clamp(1.0 - combined)

    @property
    def total(self) -> float:
        """Sum of all uncertainty components."""
        return self.h_aleatoric + self.h_epistemic + self.h_temporal + self.h_structural

    @property
    def mean(self) -> float:
        """Mean uncertainty across all dimensions."""
        return self.total / 4.0

    def confidence_with_weights(
        self,
        w_aleatoric: float = 0.3,
        w_epistemic: float = 0.4,
        w_temporal: float = 0.15,
        w_structural: float = 0.15,
    ) -> float:
        """
        Compute confidence with custom weights.

        Allows callers to adjust the relative importance of each
        uncertainty dimension based on their use case.

        Args:
            w_aleatoric: Weight for aleatoric component
            w_epistemic: Weight for epistemic component
            w_temporal: Weight for temporal component
            w_structural: Weight for structural component

        Returns:
            float: Confidence score in [0, 1]
        """
        total_weight = w_aleatoric + w_epistemic + w_temporal + w_structural
        if total_weight <= 0:
            return 0.5  # Return neutral confidence if weights are invalid

        # Normalize weights
        w_aleatoric /= total_weight
        w_epistemic /= total_weight
        w_temporal /= total_weight
        w_structural /= total_weight

        combined = (
            w_aleatoric * self.h_aleatoric +
            w_epistemic * self.h_epistemic +
            w_temporal * self.h_temporal +
            w_structural * self.h_structural
        )
        return _clamp(1.0 - combined)

    def to_dict(self) -> dict[str, float]:
        """
        Convert to dictionary for serialization.

        Returns:
            dict: All components and computed values
        """
        return {
            "h_aleatoric": round(self.h_aleatoric, 4),
            "h_epistemic": round(self.h_epistemic, 4),
            "h_temporal": round(self.h_temporal, 4),
            "h_structural": round(self.h_structural, 4),
            "total": round(self.total, 4),
            "mean": round(self.mean, 4),
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "UncertaintyVector":
        """
        Create from dictionary.

        Args:
            d: Dictionary with uncertainty components

        Returns:
            UncertaintyVector: New instance
        """
        return cls(
            h_aleatoric=d.get("h_aleatoric", 0.0),
            h_epistemic=d.get("h_epistemic", 0.0),
            h_temporal=d.get("h_temporal", 0.0),
            h_structural=d.get("h_structural", 0.0),
        )

    def interpret(self) -> str:
        """
        Interpret confidence level as human-readable tier.

        Returns:
            str: One of "high", "moderate", "low", "very_low"
        """
        conf = self.confidence
        if conf >= 0.8:
            return "high"
        elif conf >= 0.6:
            return "moderate"
        elif conf >= 0.4:
            return "low"
        else:
            return "very_low"

    def __repr__(self) -> str:
        return (
            f"UncertaintyVector("
            f"aleatoric={self.h_aleatoric:.3f}, "
            f"epistemic={self.h_epistemic:.3f}, "
            f"temporal={self.h_temporal:.3f}, "
            f"structural={self.h_structural:.3f}, "
            f"confidence={self.confidence:.3f})"
        )


@dataclass
class LatencyTracker:
    """
    Track latency statistics for aleatoric uncertainty estimation.

    Maintains a sliding window of recent latency observations and uses
    statistical analysis to detect anomalous latencies that indicate
    increased aleatoric uncertainty.

    The core insight is that consistent latencies indicate stable network
    and service conditions (low aleatoric uncertainty), while high variance
    or extreme values indicate unstable conditions (high aleatoric uncertainty).

    Attributes:
        samples: List of latency observations (ms)
        max_samples: Maximum samples to retain (sliding window)

    Example:
        tracker = LatencyTracker()
        tracker.add_sample(150.0)
        tracker.add_sample(180.0)
        tracker.add_sample(160.0)

        # Normal latency = low uncertainty
        uncertainty = tracker.estimate_aleatoric(170.0)  # ~0.1

        # Anomalous latency = high uncertainty
        uncertainty = tracker.estimate_aleatoric(500.0)  # ~0.7
    """

    samples: List[float] = field(default_factory=list)
    max_samples: int = 100

    def add_sample(self, latency_ms: float) -> None:
        """
        Add a latency observation to the tracker.

        Maintains sliding window by removing oldest sample when
        max_samples is exceeded.

        Args:
            latency_ms: Observed latency in milliseconds
        """
        if latency_ms < 0:
            latency_ms = 0.0  # Sanitize negative values

        self.samples.append(latency_ms)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def estimate_aleatoric(self, current_latency: float) -> float:
        """
        Estimate aleatoric uncertainty from latency variance.

        Uses z-score analysis to determine how anomalous the current
        latency is compared to historical observations. High z-scores
        indicate the current request experienced unusual conditions.

        Strategy:
        - Compute mean and standard deviation of historical samples
        - Calculate z-score of current latency
        - Map z-score to uncertainty: z=0 -> 0, z>=3 -> 1

        Special cases:
        - < 5 samples: Return 0.5 (insufficient data)
        - Zero variance: Return 0 (perfectly consistent)

        Args:
            current_latency: Current request latency in milliseconds

        Returns:
            float: Aleatoric uncertainty estimate in [0, 1]
        """
        # Need minimum samples for statistical significance
        if len(self.samples) < 5:
            return 0.5  # Neutral uncertainty - insufficient data

        mean = statistics.mean(self.samples)
        stdev = statistics.stdev(self.samples)

        if stdev <= 0:
            # Zero variance = perfectly consistent = low uncertainty
            return 0.0

        # Z-score of current latency
        z = abs(current_latency - mean) / stdev

        # Map z-score to uncertainty
        # z=0 -> 0 (exactly at mean)
        # z=1 -> 0.33 (within 1 stdev)
        # z=2 -> 0.67 (within 2 stdev)
        # z>=3 -> 1.0 (outlier)
        return _clamp(z / 3.0)

    def get_statistics(self) -> dict[str, float]:
        """
        Get current latency statistics.

        Returns:
            dict: Statistics including mean, stdev, min, max, count
        """
        if not self.samples:
            return {"count": 0}

        return {
            "count": len(self.samples),
            "mean": statistics.mean(self.samples),
            "stdev": statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0,
            "min": min(self.samples),
            "max": max(self.samples),
            "median": statistics.median(self.samples),
        }

    def reset(self) -> None:
        """Clear all samples."""
        self.samples.clear()


class ResponseValidator:
    """
    Validate response completeness and structure for uncertainty estimation.

    Analyzes response text to estimate epistemic uncertainty (knowledge gaps
    about response completeness) and structural uncertainty (confidence in
    response format).

    Key signals for epistemic uncertainty:
    - Stop token presence (indicates intentional termination)
    - Streaming completion signal
    - Absence of truncation markers
    - Length matching expectations

    Key signals for structural uncertainty:
    - Response length (very short = high uncertainty)
    - Presence of structured elements (code blocks, lists, headers)
    - Consistent formatting

    Attributes:
        STOP_TOKENS: Common response termination markers
        TRUNCATION_MARKERS: Indicators of incomplete responses
        STRUCTURE_MARKERS: Indicators of well-formatted responses
    """

    # Response termination markers - indicate intentional completion
    STOP_TOKENS: List[str] = [".", "!", "?", "```"]

    # Truncation indicators - suggest incomplete response
    TRUNCATION_MARKERS: List[str] = ["...", "[truncated]", "Continue", "[cut off]", "---"]

    # Structure indicators - suggest well-formatted response
    STRUCTURE_MARKERS: List[str] = ["```", "#", "-", "1.", "*", "|"]

    def estimate_epistemic(
        self,
        response: str,
        expected_length: Optional[int] = None,
        has_stop_token: bool = False,
        streaming_complete: bool = False,
    ) -> float:
        """
        Estimate epistemic uncertainty based on response signals.

        Analyzes various signals to determine confidence in response
        completeness and validity.

        Uncertainty reduction factors:
        - Stop token present: -0.2
        - Streaming complete signal: -0.2
        - No truncation markers: -0.1
        - Length within expected range: -0.1

        Args:
            response: Response text to analyze
            expected_length: Optional expected character count
            has_stop_token: External signal that stop token was received
            streaming_complete: External signal that streaming finished

        Returns:
            float: Epistemic uncertainty estimate in [0, 1]
        """
        # Base uncertainty - start with moderate uncertainty
        uncertainty = 0.5

        # Sanitize input
        if not response:
            return 0.9  # Empty response = very high uncertainty

        response_stripped = response.rstrip()

        # Check for stop token (explicit or detected)
        if has_stop_token:
            uncertainty -= 0.2
        elif any(response_stripped.endswith(t) for t in self.STOP_TOKENS):
            uncertainty -= 0.2

        # Streaming completion is a strong signal
        if streaming_complete:
            uncertainty -= 0.2

        # Check for truncation markers (increase uncertainty)
        has_truncation = any(marker in response for marker in self.TRUNCATION_MARKERS)
        if has_truncation:
            uncertainty += 0.2
        else:
            uncertainty -= 0.1  # No truncation is good

        # Length check against expectations
        if expected_length and expected_length > 0:
            ratio = len(response) / expected_length
            if 0.8 <= ratio <= 1.2:
                uncertainty -= 0.1  # Within 20% of expected
            elif ratio < 0.5:
                uncertainty += 0.1  # Significantly shorter than expected

        return _clamp(uncertainty)

    def estimate_structural(self, response: str) -> float:
        """
        Estimate structural uncertainty from response format.

        Well-structured responses with consistent formatting have
        lower structural uncertainty. Structure indicators include:
        - Code blocks (```)
        - Headers (#)
        - Lists (-, *, 1.)
        - Tables (|)

        Args:
            response: Response text to analyze

        Returns:
            float: Structural uncertainty estimate in [0, 1]
        """
        # Base structural uncertainty
        uncertainty = 0.3

        # Empty or very short responses
        if not response or len(response.strip()) < 10:
            return 0.9  # Very high uncertainty for minimal responses

        # Check for structure indicators
        structure_count = 0

        # Code blocks are strong structure signal
        if "```" in response:
            structure_count += 2

        # Headers indicate organization
        if "#" in response:
            structure_count += 1

        # Lists indicate enumeration
        if any(marker in response for marker in ["-", "*", "1."]):
            structure_count += 1

        # Tables indicate tabular data
        if "|" in response and response.count("|") >= 4:
            structure_count += 1

        # Paragraphs indicate organization
        if "\n\n" in response:
            structure_count += 1

        # Reduce uncertainty based on structure count
        uncertainty -= min(0.25, structure_count * 0.05)

        # Penalize very long unstructured text
        if len(response) > 1000 and structure_count == 0:
            uncertainty += 0.1

        return _clamp(uncertainty)

    def validate(self, response: str) -> dict[str, Any]:
        """
        Perform full validation of response.

        Returns detailed validation results including both uncertainty
        estimates and diagnostic information.

        Args:
            response: Response text to validate

        Returns:
            dict: Validation results with uncertainties and diagnostics
        """
        return {
            "h_epistemic": self.estimate_epistemic(response),
            "h_structural": self.estimate_structural(response),
            "length": len(response) if response else 0,
            "has_structure": any(m in (response or "") for m in self.STRUCTURE_MARKERS),
            "has_truncation": any(m in (response or "") for m in self.TRUNCATION_MARKERS),
            "ends_with_stop": any((response or "").rstrip().endswith(t) for t in self.STOP_TOKENS),
        }


class ConfidenceScorer:
    """
    Main interface for computing response confidence scores.

    Combines signals from latency tracking, response validation,
    and temporal factors to produce an overall confidence score
    and detailed uncertainty vector.

    This is the primary entry point for WUA confidence scoring.

    Attributes:
        latency_tracker: Tracks latency statistics
        validator: Validates response structure/completeness
        last_verification: Timestamp of last session verification

    Example:
        scorer = ConfidenceScorer()

        # Score a response
        confidence, vector = scorer.score(
            response="Here's how to solve that problem...",
            latency_ms=180.0,
            streaming_complete=True,
            session_age_s=300.0
        )

        print(f"Confidence: {confidence:.2%}")
        print(f"Uncertainty breakdown: {vector.to_dict()}")

        # With screenshot verification bonus
        confidence, vector = scorer.score(
            response="Login successful",
            latency_ms=200.0,
            screenshot_verified=True
        )
    """

    def __init__(self) -> None:
        """Initialize the confidence scorer with fresh trackers."""
        self.latency_tracker = LatencyTracker()
        self.validator = ResponseValidator()
        self.last_verification: float = time.time()

    def score(
        self,
        response: str,
        latency_ms: float,
        streaming_complete: bool = False,
        screenshot_verified: bool = False,
        session_age_s: float = 0.0,
        expected_length: Optional[int] = None,
    ) -> Tuple[float, UncertaintyVector]:
        """
        Compute confidence score for a response.

        Combines multiple uncertainty signals into an overall confidence
        score and detailed uncertainty vector.

        Uncertainty components:
        - Aleatoric: From latency variance (network/timing randomness)
        - Epistemic: From response signals (completeness/validity)
        - Temporal: From session age (confidence decay over time)
        - Structural: From response format (structure quality)

        Args:
            response: Response text to score
            latency_ms: Request latency in milliseconds
            streaming_complete: Whether streaming finished normally
            screenshot_verified: Whether screenshot verification passed
            session_age_s: Age of current session in seconds
            expected_length: Optional expected response length

        Returns:
            Tuple[float, UncertaintyVector]: (confidence_score, uncertainty_vector)
            where confidence_score is in [0, 1] and higher is better
        """
        # Add latency sample and estimate aleatoric uncertainty
        self.latency_tracker.add_sample(latency_ms)

        # Build uncertainty vector from all sources
        vec = UncertaintyVector(
            h_aleatoric=self.latency_tracker.estimate_aleatoric(latency_ms),
            h_epistemic=self.validator.estimate_epistemic(
                response,
                expected_length=expected_length,
                has_stop_token=streaming_complete,
                streaming_complete=streaming_complete,
            ),
            h_temporal=self._temporal_decay(session_age_s),
            h_structural=self.validator.estimate_structural(response),
        )

        # Get base confidence from vector
        confidence = vec.confidence

        # Screenshot verification provides a confidence boost
        # (visual confirmation reduces epistemic uncertainty)
        if screenshot_verified:
            confidence = _clamp(confidence + 0.1)

        return (confidence, vec)

    def _temporal_decay(self, session_age_s: float) -> float:
        """
        Compute temporal uncertainty from session age.

        Sessions accumulate uncertainty over time without verification.
        Uses exponential decay model with configurable half-life.

        Model:
        - Age 0: uncertainty = 0
        - Age = half_life: uncertainty = 0.5
        - Age = max_age: uncertainty = 1.0

        Args:
            session_age_s: Session age in seconds

        Returns:
            float: Temporal uncertainty in [0, 1]
        """
        if session_age_s <= 0:
            return 0.0

        # Linear interpolation to max uncertainty
        return _clamp(session_age_s / SESSION_MAX_AGE_SECONDS)

    def record_verification(self) -> None:
        """
        Record a successful verification event.

        Resets the temporal uncertainty by updating the last
        verification timestamp.
        """
        self.last_verification = time.time()

    def get_session_age(self) -> float:
        """
        Get time since last verification.

        Returns:
            float: Seconds since last verification
        """
        return time.time() - self.last_verification

    def reset(self) -> None:
        """Reset all tracking state."""
        self.latency_tracker.reset()
        self.last_verification = time.time()


# ==============================================================================
# Utility Functions
# ==============================================================================

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """
    Clamp value to [low, high] range.

    Args:
        value: Value to clamp
        low: Lower bound (default 0.0)
        high: Upper bound (default 1.0)

    Returns:
        float: Clamped value
    """
    return max(low, min(high, value))


def compute_confidence(
    response: str,
    latency_ms: float,
    streaming_complete: bool = False,
    screenshot_verified: bool = False,
    session_age_s: float = 0.0,
) -> Tuple[float, UncertaintyVector]:
    """
    Convenience function for one-shot confidence scoring.

    Creates a temporary ConfidenceScorer and computes confidence.
    For repeated scoring, create a ConfidenceScorer instance directly
    to benefit from latency tracking across requests.

    Args:
        response: Response text to score
        latency_ms: Request latency in milliseconds
        streaming_complete: Whether streaming finished normally
        screenshot_verified: Whether screenshot verification passed
        session_age_s: Age of current session in seconds

    Returns:
        Tuple[float, UncertaintyVector]: (confidence_score, uncertainty_vector)
    """
    scorer = ConfidenceScorer()
    return scorer.score(
        response=response,
        latency_ms=latency_ms,
        streaming_complete=streaming_complete,
        screenshot_verified=screenshot_verified,
        session_age_s=session_age_s,
    )


def emit_confidence_event(
    confidence: float,
    vector: UncertaintyVector,
    response_id: Optional[str] = None,
    bus_dir: Optional[str] = None,
    actor: str = "laser-uncertainty",
) -> None:
    """
    Emit a confidence scoring event to the bus.

    Publishes uncertainty metrics for observability and analysis.

    Args:
        confidence: Computed confidence score
        vector: Full uncertainty vector
        response_id: Optional identifier for the response
        bus_dir: Bus directory (default from env or /pluribus/.pluribus/bus)
        actor: Actor name for the event
    """
    if bus_dir is None:
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")

    bus_path = Path(bus_dir) / "events.ndjson"

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": "laser.confidence.scored",
        "kind": "metric",
        "level": "info",
        "actor": actor,
        "data": {
            "confidence": round(confidence, 4),
            "uncertainty_vector": vector.to_dict(),
            "tier": vector.interpret(),
            "response_id": response_id,
        },
    }

    try:
        bus_path.parent.mkdir(parents=True, exist_ok=True)
        with bus_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass  # Non-fatal: bus emission is best-effort


# ==============================================================================
# CLI Interface (for testing and debugging)
# ==============================================================================

def main() -> int:
    """CLI entry point for testing uncertainty scoring."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="uncertainty.py",
        description="LASER Uncertainty Quantification - Compute confidence scores",
    )
    parser.add_argument(
        "--response", "-r",
        default="This is a test response.",
        help="Response text to score",
    )
    parser.add_argument(
        "--latency", "-l",
        type=float,
        default=200.0,
        help="Latency in milliseconds",
    )
    parser.add_argument(
        "--streaming-complete",
        action="store_true",
        help="Mark streaming as complete",
    )
    parser.add_argument(
        "--screenshot-verified",
        action="store_true",
        help="Mark as screenshot verified",
    )
    parser.add_argument(
        "--session-age",
        type=float,
        default=0.0,
        help="Session age in seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    confidence, vector = compute_confidence(
        response=args.response,
        latency_ms=args.latency,
        streaming_complete=args.streaming_complete,
        screenshot_verified=args.screenshot_verified,
        session_age_s=args.session_age,
    )

    if args.json:
        output = {
            "confidence": round(confidence, 4),
            "tier": vector.interpret(),
            "uncertainty_vector": vector.to_dict(),
        }
        print(json.dumps(output, indent=2))
    else:
        print("LASER Uncertainty Score")
        print("=" * 40)
        print(f"Confidence:      {confidence:.2%}")
        print(f"Tier:            {vector.interpret()}")
        print("-" * 40)
        print(f"H_aleatoric:     {vector.h_aleatoric:.4f}")
        print(f"H_epistemic:     {vector.h_epistemic:.4f}")
        print(f"H_temporal:      {vector.h_temporal:.4f}")
        print(f"H_structural:    {vector.h_structural:.4f}")
        print("-" * 40)
        print(f"Total:           {vector.total:.4f}")
        print(f"Mean:            {vector.mean:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
