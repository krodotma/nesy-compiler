#!/usr/bin/env python3
"""
confidence_scorer.py - Confidence Scorer (Step 25)

Score confidence in research answers based on multiple factors.
Provides uncertainty quantification for research results.

PBTSO Phase: DISTILL, ITERATE

Bus Topics:
- a2a.research.confidence.score
- a2a.research.confidence.explain
- research.confidence.calibrate

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..bootstrap import AgentBus
from ..search.query_planner import QueryIntent


# ============================================================================
# Configuration
# ============================================================================


class ConfidenceLevel(Enum):
    """Discrete confidence levels."""
    VERY_HIGH = "very_high"   # 0.9+ - Almost certain
    HIGH = "high"             # 0.7-0.9 - Confident
    MEDIUM = "medium"         # 0.5-0.7 - Reasonably confident
    LOW = "low"               # 0.3-0.5 - Uncertain
    VERY_LOW = "very_low"     # 0.0-0.3 - Very uncertain


class ConfidenceFactor(Enum):
    """Factors affecting confidence."""
    RESULT_QUALITY = "result_quality"     # Quality of search results
    RESULT_AGREEMENT = "result_agreement" # Agreement between results
    QUERY_CLARITY = "query_clarity"       # Clarity of the query
    COVERAGE = "coverage"                 # Coverage of search space
    FRESHNESS = "freshness"               # How recent the data is
    SPECIFICITY = "specificity"           # How specific the answer is


@dataclass
class ScorerConfig:
    """Configuration for confidence scorer."""

    # Factor weights
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        ConfidenceFactor.RESULT_QUALITY.value: 0.30,
        ConfidenceFactor.RESULT_AGREEMENT.value: 0.25,
        ConfidenceFactor.QUERY_CLARITY.value: 0.15,
        ConfidenceFactor.COVERAGE.value: 0.15,
        ConfidenceFactor.SPECIFICITY.value: 0.15,
    })

    # Thresholds for confidence levels
    level_thresholds: Dict[str, float] = field(default_factory=lambda: {
        ConfidenceLevel.VERY_HIGH.value: 0.9,
        ConfidenceLevel.HIGH.value: 0.7,
        ConfidenceLevel.MEDIUM.value: 0.5,
        ConfidenceLevel.LOW.value: 0.3,
    })

    # Intent-specific adjustments
    intent_adjustments: Dict[str, float] = field(default_factory=lambda: {
        QueryIntent.FIND_SYMBOL.value: 1.0,      # High confidence possible
        QueryIntent.FIND_USAGE.value: 0.95,      # May miss some usages
        QueryIntent.EXPLAIN_CODE.value: 0.85,   # Subjective element
        QueryIntent.SEARCH_CONCEPT.value: 0.80, # More uncertain
        QueryIntent.DEBUG_ISSUE.value: 0.75,    # Complex, uncertain
    })

    # Penalties
    no_results_penalty: float = 0.5
    ambiguous_penalty: float = 0.2
    stale_data_penalty: float = 0.15

    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class FactorScore:
    """Score for a single confidence factor."""

    factor: ConfidenceFactor
    score: float              # 0.0 to 1.0
    weight: float             # Factor weight
    contribution: float       # Weighted contribution
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor": self.factor.value,
            "score": round(self.score, 4),
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 4),
            "explanation": self.explanation,
        }


@dataclass
class ConfidenceScore:
    """Complete confidence assessment."""

    overall_score: float
    level: ConfidenceLevel
    factor_scores: List[FactorScore]
    adjustments: List[Tuple[str, float]]  # (reason, adjustment)
    explanation: str
    calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": round(self.overall_score, 4),
            "level": self.level.value,
            "factors": [f.to_dict() for f in self.factor_scores],
            "adjustments": self.adjustments,
            "explanation": self.explanation,
            "calibrated": self.calibrated,
        }

    @property
    def percentage(self) -> int:
        """Return confidence as percentage."""
        return int(round(self.overall_score * 100))


@dataclass
class CalibrationData:
    """Data for confidence calibration."""

    predictions: List[float] = field(default_factory=list)  # Predicted confidences
    outcomes: List[bool] = field(default_factory=list)      # Actual correctness
    buckets: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add_sample(self, predicted: float, correct: bool) -> None:
        """Add a calibration sample."""
        self.predictions.append(predicted)
        self.outcomes.append(correct)

    def calibration_error(self) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if len(self.predictions) < 10:
            return 0.0

        # Create buckets
        n_buckets = 10
        bucket_predictions: List[List[float]] = [[] for _ in range(n_buckets)]
        bucket_outcomes: List[List[bool]] = [[] for _ in range(n_buckets)]

        for pred, outcome in zip(self.predictions, self.outcomes):
            bucket = min(int(pred * n_buckets), n_buckets - 1)
            bucket_predictions[bucket].append(pred)
            bucket_outcomes[bucket].append(outcome)

        # Calculate ECE
        ece = 0.0
        total = len(self.predictions)

        for preds, outcomes in zip(bucket_predictions, bucket_outcomes):
            if not preds:
                continue
            avg_pred = sum(preds) / len(preds)
            avg_outcome = sum(outcomes) / len(outcomes)
            weight = len(preds) / total
            ece += weight * abs(avg_pred - avg_outcome)

        return ece


# ============================================================================
# Confidence Scorer
# ============================================================================


class ConfidenceScorer:
    """
    Score confidence in research answers.

    Evaluates multiple factors to produce calibrated confidence scores:
    - Result quality and agreement
    - Query clarity and specificity
    - Search coverage
    - Data freshness

    PBTSO Phase: DISTILL, ITERATE

    Example:
        scorer = ConfidenceScorer()

        confidence = scorer.score(
            results=search_results,
            query="where is UserService defined?",
            intent=QueryIntent.FIND_SYMBOL
        )

        print(f"Confidence: {confidence.percentage}% ({confidence.level.value})")
        print(confidence.explanation)
    """

    def __init__(
        self,
        config: Optional[ScorerConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the confidence scorer.

        Args:
            config: Scorer configuration
            bus: AgentBus for event emission
        """
        self.config = config or ScorerConfig()
        self.bus = bus or AgentBus()

        # Factor extractors
        self._factor_extractors: Dict[ConfidenceFactor, Callable] = {}
        self._register_default_extractors()

        # Calibration data
        self._calibration = CalibrationData()

    def score(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceScore:
        """
        Score confidence in research results.

        Args:
            results: Search results
            query: Original query
            intent: Query intent
            context: Additional context

        Returns:
            ConfidenceScore with detailed breakdown
        """
        self._emit_with_lock({
            "topic": "a2a.research.confidence.score",
            "kind": "confidence",
            "data": {
                "query": query[:100],
                "intent": intent.value,
                "result_count": len(results),
            }
        })

        # Extract factor scores
        factor_scores = []
        for factor in ConfidenceFactor:
            extractor = self._factor_extractors.get(factor)
            if extractor:
                raw_score = extractor(results, query, intent, context)
                weight = self.config.factor_weights.get(factor.value, 0.1)
                contribution = raw_score * weight

                factor_scores.append(FactorScore(
                    factor=factor,
                    score=raw_score,
                    weight=weight,
                    contribution=contribution,
                    explanation=self._explain_factor(factor, raw_score),
                ))

        # Calculate base score
        base_score = sum(f.contribution for f in factor_scores)

        # Apply adjustments
        adjustments = []

        # Intent adjustment
        intent_mult = self.config.intent_adjustments.get(intent.value, 1.0)
        if intent_mult != 1.0:
            adjustments.append((f"Intent ({intent.value})", intent_mult - 1.0))
            base_score *= intent_mult

        # Penalty for no results
        if not results:
            adjustments.append(("No results", -self.config.no_results_penalty))
            base_score -= self.config.no_results_penalty

        # Penalty for ambiguous results
        if self._is_ambiguous(results):
            adjustments.append(("Ambiguous results", -self.config.ambiguous_penalty))
            base_score -= self.config.ambiguous_penalty

        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, base_score))

        # Determine confidence level
        level = self._score_to_level(final_score)

        # Generate explanation
        explanation = self._generate_explanation(final_score, factor_scores, adjustments)

        return ConfidenceScore(
            overall_score=final_score,
            level=level,
            factor_scores=factor_scores,
            adjustments=adjustments,
            explanation=explanation,
        )

    def score_quick(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> float:
        """
        Quick confidence score without detailed breakdown.

        Args:
            results: Search results
            query: Original query

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not results:
            return 0.2

        # Quick heuristic based on top result score
        top_score = results[0].get("score", results[0].get("final_score", 0.5))

        # Adjust based on result count
        if len(results) == 1 and top_score > 0.8:
            return min(0.95, top_score)
        elif len(results) > 10:
            return top_score * 0.85

        return top_score

    def explain(self, confidence: ConfidenceScore) -> str:
        """
        Generate detailed explanation of confidence score.

        Args:
            confidence: Confidence score to explain

        Returns:
            Detailed explanation string
        """
        lines = [
            f"Confidence Score: {confidence.percentage}% ({confidence.level.value})",
            "",
            "Factor Breakdown:",
        ]

        for factor in sorted(confidence.factor_scores, key=lambda f: -f.contribution):
            bar = "=" * int(factor.score * 20)
            lines.append(f"  {factor.factor.value:20} [{bar:20}] {factor.score:.2f} (x{factor.weight:.2f})")
            if factor.explanation:
                lines.append(f"    {factor.explanation}")

        if confidence.adjustments:
            lines.append("")
            lines.append("Adjustments:")
            for reason, adj in confidence.adjustments:
                sign = "+" if adj > 0 else ""
                lines.append(f"  {reason}: {sign}{adj:.2f}")

        lines.append("")
        lines.append(f"Summary: {confidence.explanation}")

        self._emit_with_lock({
            "topic": "a2a.research.confidence.explain",
            "kind": "confidence",
            "data": {"score": confidence.overall_score}
        })

        return "\n".join(lines)

    def calibrate(self, predicted: float, actual_correct: bool) -> None:
        """
        Add a calibration sample.

        Args:
            predicted: Predicted confidence
            actual_correct: Whether the answer was actually correct
        """
        self._calibration.add_sample(predicted, actual_correct)

        self._emit_with_lock({
            "topic": "research.confidence.calibrate",
            "kind": "calibration",
            "data": {
                "predicted": predicted,
                "correct": actual_correct,
                "samples": len(self._calibration.predictions),
            }
        })

    def get_calibration_error(self) -> float:
        """Get current Expected Calibration Error."""
        return self._calibration.calibration_error()

    def register_factor_extractor(
        self,
        factor: ConfidenceFactor,
        extractor: Callable[[List[Dict], str, QueryIntent, Optional[Dict]], float],
    ) -> None:
        """Register a custom factor extractor."""
        self._factor_extractors[factor] = extractor

    # ========================================================================
    # Factor Extractors
    # ========================================================================

    def _register_default_extractors(self) -> None:
        """Register default factor extractors."""
        self._factor_extractors[ConfidenceFactor.RESULT_QUALITY] = self._extract_result_quality
        self._factor_extractors[ConfidenceFactor.RESULT_AGREEMENT] = self._extract_result_agreement
        self._factor_extractors[ConfidenceFactor.QUERY_CLARITY] = self._extract_query_clarity
        self._factor_extractors[ConfidenceFactor.COVERAGE] = self._extract_coverage
        self._factor_extractors[ConfidenceFactor.SPECIFICITY] = self._extract_specificity

    def _extract_result_quality(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract result quality score."""
        if not results:
            return 0.2

        # Average of top result scores
        scores = []
        for r in results[:5]:
            score = r.get("score", r.get("final_score", 0.5))
            scores.append(score)

        if not scores:
            return 0.5

        # Weighted average favoring top results
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight

    def _extract_result_agreement(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract result agreement score."""
        if len(results) < 2:
            return 0.7  # Single result - moderate agreement

        # Check if top results agree on key aspects
        top_results = results[:5]

        # Agreement on kind/type
        kinds = [r.get("kind", r.get("type", "unknown")) for r in top_results]
        kind_agreement = len(set(kinds)) <= 2

        # Agreement on path (same file)
        paths = [r.get("path", "") for r in top_results]
        path_agreement = len(set(paths)) <= 3

        # Score consistency (similar scores)
        scores = [r.get("score", 0.5) for r in top_results]
        if scores:
            score_std = self._std_dev(scores)
            score_agreement = score_std < 0.2

            agreement_count = sum([kind_agreement, path_agreement, score_agreement])
            return 0.5 + (agreement_count * 0.15)

        return 0.5

    def _extract_query_clarity(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract query clarity score."""
        score = 0.5

        # Clear if short and specific
        words = query.split()
        if 2 <= len(words) <= 6:
            score += 0.2

        # Clear if contains identifiers (CamelCase, snake_case)
        import re
        if re.search(r"[A-Z][a-z]+[A-Z]|[a-z]+_[a-z]+", query):
            score += 0.2

        # Clear if intent was clearly detected
        if intent != QueryIntent.SEARCH_CONCEPT:
            score += 0.1

        return min(1.0, score)

    def _extract_coverage(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract coverage score."""
        # Based on result count and diversity
        if not results:
            return 0.2

        # More results = better coverage (up to a point)
        count_score = min(1.0, len(results) / 20)

        # File diversity
        paths = set(r.get("path", "") for r in results)
        diversity_score = min(1.0, len(paths) / 5)

        return (count_score + diversity_score) / 2

    def _extract_specificity(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract specificity score."""
        if not results:
            return 0.3

        # Exact match in top result
        top = results[0]
        query_terms = set(query.lower().split())

        name = top.get("name", "").lower()
        if name and any(term in name for term in query_terms):
            return 0.9

        # Has specific location
        if top.get("line") and top.get("path"):
            return 0.8

        return 0.5

    # ========================================================================
    # Helpers
    # ========================================================================

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        thresholds = self.config.level_thresholds

        if score >= thresholds[ConfidenceLevel.VERY_HIGH.value]:
            return ConfidenceLevel.VERY_HIGH
        elif score >= thresholds[ConfidenceLevel.HIGH.value]:
            return ConfidenceLevel.HIGH
        elif score >= thresholds[ConfidenceLevel.MEDIUM.value]:
            return ConfidenceLevel.MEDIUM
        elif score >= thresholds[ConfidenceLevel.LOW.value]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _is_ambiguous(self, results: List[Dict[str, Any]]) -> bool:
        """Check if results are ambiguous."""
        if len(results) < 2:
            return False

        # Check if top two results have similar scores but different targets
        if len(results) >= 2:
            score1 = results[0].get("score", results[0].get("final_score", 0))
            score2 = results[1].get("score", results[1].get("final_score", 0))

            if abs(score1 - score2) < 0.1:
                path1 = results[0].get("path", "")
                path2 = results[1].get("path", "")
                if path1 != path2:
                    return True

        return False

    def _explain_factor(self, factor: ConfidenceFactor, score: float) -> str:
        """Generate explanation for a factor score."""
        explanations = {
            ConfidenceFactor.RESULT_QUALITY: {
                (0.8, 1.0): "High-quality results with strong match scores",
                (0.6, 0.8): "Good quality results",
                (0.4, 0.6): "Moderate quality results",
                (0.0, 0.4): "Low quality or uncertain matches",
            },
            ConfidenceFactor.RESULT_AGREEMENT: {
                (0.8, 1.0): "Results strongly agree on the answer",
                (0.6, 0.8): "Results mostly agree",
                (0.4, 0.6): "Some disagreement between results",
                (0.0, 0.4): "Significant disagreement in results",
            },
            ConfidenceFactor.QUERY_CLARITY: {
                (0.8, 1.0): "Query is clear and specific",
                (0.6, 0.8): "Query is reasonably clear",
                (0.4, 0.6): "Query could be more specific",
                (0.0, 0.4): "Query is vague or ambiguous",
            },
            ConfidenceFactor.COVERAGE: {
                (0.8, 1.0): "Comprehensive coverage of search space",
                (0.6, 0.8): "Good coverage",
                (0.4, 0.6): "Partial coverage",
                (0.0, 0.4): "Limited search coverage",
            },
            ConfidenceFactor.SPECIFICITY: {
                (0.8, 1.0): "Found exact, specific answer",
                (0.6, 0.8): "Answer is fairly specific",
                (0.4, 0.6): "Answer is somewhat general",
                (0.0, 0.4): "Answer lacks specificity",
            },
        }

        factor_explanations = explanations.get(factor, {})
        for (low, high), text in factor_explanations.items():
            if low <= score < high:
                return text

        return ""

    def _generate_explanation(
        self,
        score: float,
        factors: List[FactorScore],
        adjustments: List[Tuple[str, float]],
    ) -> str:
        """Generate overall explanation."""
        level = self._score_to_level(score)

        if level == ConfidenceLevel.VERY_HIGH:
            base = "Very high confidence in this answer."
        elif level == ConfidenceLevel.HIGH:
            base = "High confidence in this answer."
        elif level == ConfidenceLevel.MEDIUM:
            base = "Moderate confidence - answer is likely correct but verify important details."
        elif level == ConfidenceLevel.LOW:
            base = "Low confidence - consider additional research."
        else:
            base = "Very uncertain - this answer may not be reliable."

        # Add top factor
        if factors:
            top_factor = max(factors, key=lambda f: f.contribution)
            if top_factor.score > 0.7:
                base += f" Strong support from {top_factor.factor.value}."
            elif top_factor.score < 0.4:
                base += f" Concern: {top_factor.explanation}"

        return base

    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def _emit_with_lock(self, event: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        import socket
        from datetime import datetime, timezone
        import uuid

        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": event.get("topic", "unknown"),
            "kind": event.get("kind", "event"),
            "level": event.get("level", "info"),
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": event.get("data", {}),
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(full_event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Confidence Scorer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Confidence Scorer (Step 25)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Score command
    score_parser = subparsers.add_parser("score", help="Score confidence in results")
    score_parser.add_argument("--input", "-i", required=True, help="Input JSON file with results")
    score_parser.add_argument("--query", "-q", required=True, help="Original query")
    score_parser.add_argument("--intent", choices=[i.value for i in QueryIntent],
                             default="search_concept", help="Query intent")
    score_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Explain confidence score")
    explain_parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    explain_parser.add_argument("--query", "-q", required=True, help="Original query")
    explain_parser.add_argument("--intent", default="search_concept", help="Query intent")

    args = parser.parse_args()

    scorer = ConfidenceScorer()

    if args.command == "score":
        with open(args.input) as f:
            results = json.load(f)

        intent = QueryIntent(args.intent)
        confidence = scorer.score(results, args.query, intent)

        if args.json:
            print(json.dumps(confidence.to_dict(), indent=2))
        else:
            print(f"Confidence: {confidence.percentage}% ({confidence.level.value})")
            print(f"Explanation: {confidence.explanation}")

    elif args.command == "explain":
        with open(args.input) as f:
            results = json.load(f)

        intent = QueryIntent(args.intent)
        confidence = scorer.score(results, args.query, intent)
        print(scorer.explain(confidence))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
