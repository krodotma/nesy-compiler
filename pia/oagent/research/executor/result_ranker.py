#!/usr/bin/env python3
"""
result_ranker.py - Result Ranker (Step 22)

Rank and prioritize research results using multiple signals.
Supports configurable ranking factors and score normalization.

PBTSO Phase: DISTILL, ITERATE

Bus Topics:
- a2a.research.rank.start
- a2a.research.rank.complete
- research.rank.scored

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


# ============================================================================
# Configuration
# ============================================================================


class RankingFactor(Enum):
    """Factors used in ranking."""
    RELEVANCE = "relevance"       # Semantic/textual relevance
    RECENCY = "recency"           # How recently modified
    AUTHORITY = "authority"       # Importance of source file
    CONTEXT_MATCH = "context"     # Match with user context
    SYMBOL_TYPE = "symbol_type"   # Type of symbol (class > function > variable)
    DOCUMENTATION = "documentation"  # Has documentation
    USAGE_FREQUENCY = "usage"     # How often used in codebase


@dataclass
class RankerConfig:
    """Configuration for result ranker."""

    # Factor weights (should sum to 1.0 for normalization)
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        RankingFactor.RELEVANCE.value: 0.35,
        RankingFactor.CONTEXT_MATCH.value: 0.25,
        RankingFactor.AUTHORITY.value: 0.15,
        RankingFactor.SYMBOL_TYPE.value: 0.10,
        RankingFactor.DOCUMENTATION.value: 0.10,
        RankingFactor.RECENCY.value: 0.05,
    })

    # Scoring parameters
    min_score: float = 0.0
    max_results: int = 100
    diversity_penalty: float = 0.1  # Penalty for results from same file
    boost_exact_match: float = 1.5  # Boost for exact name matches

    # File authority scores (by path pattern)
    authority_patterns: Dict[str, float] = field(default_factory=lambda: {
        "src/": 1.0,
        "lib/": 0.9,
        "core/": 1.0,
        "test": 0.5,
        "example": 0.4,
        "vendor/": 0.3,
        "node_modules/": 0.2,
    })

    # Symbol type scores
    symbol_type_scores: Dict[str, float] = field(default_factory=lambda: {
        "class": 1.0,
        "interface": 0.95,
        "function": 0.9,
        "method": 0.85,
        "constant": 0.8,
        "variable": 0.7,
        "import": 0.4,
    })

    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class RankingSignals:
    """Signals used for ranking a result."""

    relevance_score: float = 0.0
    recency_score: float = 0.0
    authority_score: float = 0.0
    context_score: float = 0.0
    symbol_type_score: float = 0.0
    documentation_score: float = 0.0
    usage_score: float = 0.0

    # Boosts and penalties
    exact_match_boost: float = 1.0
    diversity_penalty: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RankedResult:
    """A ranked search result."""

    original: Dict[str, Any]
    final_score: float
    signals: RankingSignals
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **self.original,
            "final_score": self.final_score,
            "rank": self.rank,
            "signals": self.signals.to_dict(),
        }


@dataclass
class RankingReport:
    """Report of ranking operation."""

    input_count: int
    output_count: int
    ranking_time_ms: float
    score_distribution: Dict[str, int]  # Score buckets
    top_factors: List[Tuple[str, float]]  # Most influential factors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "ranking_time_ms": self.ranking_time_ms,
            "score_distribution": self.score_distribution,
            "top_factors": self.top_factors,
        }


# ============================================================================
# Result Ranker
# ============================================================================


class ResultRanker:
    """
    Rank and prioritize research results.

    Uses multiple signals to compute a final relevance score:
    - Semantic relevance from search
    - File authority based on path patterns
    - Symbol type importance
    - Context match with user query
    - Documentation presence

    PBTSO Phase: DISTILL, ITERATE

    Example:
        ranker = ResultRanker()
        ranked = ranker.rank(results, query="find UserService", context={"file": "auth.py"})
        for r in ranked[:10]:
            print(f"{r.rank}. {r.original['name']} (score: {r.final_score:.3f})")
    """

    def __init__(
        self,
        config: Optional[RankerConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the result ranker.

        Args:
            config: Ranking configuration
            bus: AgentBus for event emission
        """
        self.config = config or RankerConfig()
        self.bus = bus or AgentBus()

        # Custom signal extractors
        self._signal_extractors: Dict[RankingFactor, Callable] = {}
        self._register_default_extractors()

    def rank(
        self,
        results: List[Dict[str, Any]],
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_results: Optional[int] = None,
    ) -> List[RankedResult]:
        """
        Rank a list of results.

        Args:
            results: Raw search results
            query: Original search query
            context: User context (current file, selection, etc.)
            max_results: Maximum results to return

        Returns:
            List of RankedResult sorted by score
        """
        start_time = time.time()

        self._emit_with_lock({
            "topic": "a2a.research.rank.start",
            "kind": "rank",
            "data": {
                "input_count": len(results),
                "query_length": len(query) if query else 0,
            }
        })

        max_results = max_results or self.config.max_results

        # Score each result
        ranked_results: List[RankedResult] = []
        seen_paths: Dict[str, int] = {}  # Track results per path for diversity

        for result in results:
            signals = self._extract_signals(result, query, context)

            # Apply diversity penalty
            path = result.get("path", "")
            if path in seen_paths:
                signals.diversity_penalty = 1.0 - (
                    self.config.diversity_penalty * min(seen_paths[path], 3)
                )
            seen_paths[path] = seen_paths.get(path, 0) + 1

            # Compute final score
            final_score = self._compute_final_score(signals)

            if final_score >= self.config.min_score:
                ranked_results.append(RankedResult(
                    original=result,
                    final_score=final_score,
                    signals=signals,
                ))

        # Sort by score
        ranked_results.sort(key=lambda r: r.final_score, reverse=True)

        # Assign ranks and limit
        for i, result in enumerate(ranked_results[:max_results]):
            result.rank = i + 1

        ranked_results = ranked_results[:max_results]

        # Generate report
        ranking_time = (time.time() - start_time) * 1000
        report = self._generate_report(results, ranked_results, ranking_time)

        self._emit_with_lock({
            "topic": "a2a.research.rank.complete",
            "kind": "rank",
            "data": report.to_dict()
        })

        return ranked_results

    def rank_with_feedback(
        self,
        results: List[Dict[str, Any]],
        feedback: Dict[str, float],
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RankedResult]:
        """
        Rank results with user feedback adjustment.

        Args:
            results: Raw search results
            feedback: Dict mapping result keys to feedback scores (-1 to 1)
            query: Original search query
            context: User context

        Returns:
            List of RankedResult adjusted by feedback
        """
        # First do normal ranking
        ranked = self.rank(results, query, context)

        # Apply feedback adjustments
        for result in ranked:
            key = self._result_key(result.original)
            if key in feedback:
                adjustment = feedback[key]
                result.final_score *= (1.0 + adjustment * 0.3)

        # Re-sort and re-rank
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    def explain_ranking(self, result: RankedResult) -> str:
        """
        Generate human-readable explanation of ranking.

        Args:
            result: Ranked result to explain

        Returns:
            Explanation string
        """
        lines = [
            f"Ranking Explanation for: {result.original.get('name', result.original.get('path', 'unknown'))}",
            f"Final Score: {result.final_score:.4f}",
            "",
            "Signal Breakdown:",
        ]

        signals = result.signals
        weights = self.config.factor_weights

        contributions = []
        for factor, weight in weights.items():
            signal_value = getattr(signals, f"{factor}_score", 0)
            contribution = signal_value * weight
            contributions.append((factor, signal_value, weight, contribution))

        contributions.sort(key=lambda x: x[3], reverse=True)

        for factor, signal, weight, contrib in contributions:
            lines.append(f"  {factor}: {signal:.3f} x {weight:.2f} = {contrib:.4f}")

        if signals.exact_match_boost != 1.0:
            lines.append(f"\n  Exact match boost: x{signals.exact_match_boost:.2f}")

        if signals.diversity_penalty != 1.0:
            lines.append(f"  Diversity penalty: x{signals.diversity_penalty:.2f}")

        return "\n".join(lines)

    def register_signal_extractor(
        self,
        factor: RankingFactor,
        extractor: Callable[[Dict[str, Any], Optional[str], Optional[Dict]], float],
    ) -> None:
        """
        Register a custom signal extractor.

        Args:
            factor: Ranking factor
            extractor: Function(result, query, context) -> float
        """
        self._signal_extractors[factor] = extractor

    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        Update factor weights.

        Args:
            weights: New weights (partial update allowed)
        """
        self.config.factor_weights.update(weights)

    # ========================================================================
    # Signal Extraction
    # ========================================================================

    def _register_default_extractors(self) -> None:
        """Register default signal extractors."""
        self._signal_extractors[RankingFactor.RELEVANCE] = self._extract_relevance
        self._signal_extractors[RankingFactor.RECENCY] = self._extract_recency
        self._signal_extractors[RankingFactor.AUTHORITY] = self._extract_authority
        self._signal_extractors[RankingFactor.CONTEXT_MATCH] = self._extract_context_match
        self._signal_extractors[RankingFactor.SYMBOL_TYPE] = self._extract_symbol_type
        self._signal_extractors[RankingFactor.DOCUMENTATION] = self._extract_documentation
        self._signal_extractors[RankingFactor.USAGE_FREQUENCY] = self._extract_usage

    def _extract_signals(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> RankingSignals:
        """Extract all ranking signals from a result."""
        signals = RankingSignals()

        # Extract each signal
        for factor, extractor in self._signal_extractors.items():
            value = extractor(result, query, context)
            setattr(signals, f"{factor.value}_score", value)

        # Check for exact match boost
        if query:
            name = result.get("name", "")
            if name.lower() == query.lower():
                signals.exact_match_boost = self.config.boost_exact_match
            elif query.lower() in name.lower():
                signals.exact_match_boost = 1.2

        return signals

    def _extract_relevance(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract relevance score."""
        # Use provided score if available
        if "score" in result:
            return min(1.0, max(0.0, result["score"]))

        # Otherwise compute from name similarity
        if query and "name" in result:
            return self._string_similarity(result["name"], query)

        return 0.5  # Default

    def _extract_recency(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract recency score based on file modification time."""
        path = result.get("path")
        if not path:
            return 0.5

        try:
            mtime = Path(path).stat().st_mtime
            # Score based on age (higher for recent files)
            age_days = (time.time() - mtime) / (24 * 3600)
            # Exponential decay: score of 1.0 for today, ~0.5 for 30 days ago
            return math.exp(-age_days / 30)
        except Exception:
            return 0.5

    def _extract_authority(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract authority score based on file path."""
        path = result.get("path", "")

        for pattern, score in self.config.authority_patterns.items():
            if pattern in path:
                return score

        return 0.7  # Default for unmatched paths

    def _extract_context_match(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract context match score."""
        if not context:
            return 0.5

        score = 0.5
        path = result.get("path", "")

        # Boost if in same directory as context file
        context_file = context.get("current_file", context.get("file"))
        if context_file:
            context_dir = str(Path(context_file).parent)
            if path.startswith(context_dir):
                score += 0.3

        # Boost if symbol mentioned in context selection
        selection = context.get("selection", "")
        if selection:
            name = result.get("name", "")
            if name and name in selection:
                score += 0.2

        return min(1.0, score)

    def _extract_symbol_type(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract symbol type importance score."""
        kind = result.get("kind", result.get("type", ""))

        return self.config.symbol_type_scores.get(kind, 0.5)

    def _extract_documentation(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract documentation presence score."""
        docstring = result.get("docstring", "")

        if docstring and len(docstring) > 50:
            return 1.0
        elif docstring:
            return 0.7
        else:
            return 0.3

    def _extract_usage(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Extract usage frequency score."""
        # This would typically come from pre-computed analysis
        usage_count = result.get("usage_count", 0)

        if usage_count > 100:
            return 1.0
        elif usage_count > 50:
            return 0.9
        elif usage_count > 20:
            return 0.8
        elif usage_count > 5:
            return 0.7
        elif usage_count > 0:
            return 0.6
        else:
            return 0.5  # Unknown

    # ========================================================================
    # Score Computation
    # ========================================================================

    def _compute_final_score(self, signals: RankingSignals) -> float:
        """Compute final score from signals."""
        weights = self.config.factor_weights

        weighted_sum = 0.0
        for factor_name, weight in weights.items():
            signal_attr = f"{factor_name}_score"
            signal_value = getattr(signals, signal_attr, 0.0)
            weighted_sum += signal_value * weight

        # Apply boosts and penalties
        final_score = weighted_sum * signals.exact_match_boost * signals.diversity_penalty

        return max(0.0, min(1.0, final_score))

    # ========================================================================
    # Helpers
    # ========================================================================

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity (Jaccard on words)."""
        words1 = set(s1.lower().split("_"))
        words2 = set(s2.lower().split())

        # Also split camelCase
        import re
        words1.update(re.findall(r"[A-Z][a-z]+|[a-z]+", s1))
        words2.update(re.findall(r"[A-Z][a-z]+|[a-z]+", s2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _result_key(self, result: Dict[str, Any]) -> str:
        """Generate unique key for a result."""
        path = result.get("path", "")
        line = result.get("line", "")
        name = result.get("name", "")
        return f"{path}:{line}:{name}"

    def _generate_report(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[RankedResult],
        ranking_time: float,
    ) -> RankingReport:
        """Generate ranking report."""
        # Score distribution
        distribution = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for r in outputs:
            if r.final_score < 0.2:
                distribution["0.0-0.2"] += 1
            elif r.final_score < 0.4:
                distribution["0.2-0.4"] += 1
            elif r.final_score < 0.6:
                distribution["0.4-0.6"] += 1
            elif r.final_score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1

        # Top factors (by average contribution)
        if outputs:
            factor_sums: Dict[str, float] = {}
            for r in outputs:
                for factor, weight in self.config.factor_weights.items():
                    signal = getattr(r.signals, f"{factor}_score", 0)
                    factor_sums[factor] = factor_sums.get(factor, 0) + signal * weight

            top_factors = sorted(
                [(f, s / len(outputs)) for f, s in factor_sums.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        else:
            top_factors = []

        return RankingReport(
            input_count=len(inputs),
            output_count=len(outputs),
            ranking_time_ms=ranking_time,
            score_distribution=distribution,
            top_factors=top_factors,
        )

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
    """CLI entry point for Result Ranker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Result Ranker (Step 22)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Rank command
    rank_parser = subparsers.add_parser("rank", help="Rank results from JSON input")
    rank_parser.add_argument("--input", "-i", required=True, help="Input JSON file with results")
    rank_parser.add_argument("--query", "-q", help="Original query")
    rank_parser.add_argument("--max", type=int, default=20, help="Maximum results")
    rank_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Explain ranking")
    explain_parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    explain_parser.add_argument("--query", "-q", help="Original query")
    explain_parser.add_argument("--index", type=int, default=0, help="Result index to explain")

    # Weights command
    weights_parser = subparsers.add_parser("weights", help="Show or update weights")
    weights_parser.add_argument("--set", nargs=2, action="append",
                                metavar=("FACTOR", "WEIGHT"), help="Set weight")

    args = parser.parse_args()

    ranker = ResultRanker()

    if args.command == "rank":
        with open(args.input) as f:
            results = json.load(f)

        ranked = ranker.rank(results, query=args.query, max_results=args.max)

        if args.json:
            print(json.dumps([r.to_dict() for r in ranked], indent=2))
        else:
            print(f"Ranked {len(ranked)} results:")
            for r in ranked:
                name = r.original.get("name", r.original.get("path", "unknown"))
                print(f"  {r.rank:3}. {name:40} (score: {r.final_score:.4f})")

    elif args.command == "explain":
        with open(args.input) as f:
            results = json.load(f)

        ranked = ranker.rank(results, query=args.query)
        if args.index < len(ranked):
            print(ranker.explain_ranking(ranked[args.index]))
        else:
            print(f"Index {args.index} out of range (0-{len(ranked)-1})")

    elif args.command == "weights":
        if args.set:
            for factor, weight in args.set:
                ranker.update_weights({factor: float(weight)})

        print("Current factor weights:")
        for factor, weight in sorted(ranker.config.factor_weights.items()):
            print(f"  {factor}: {weight:.3f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
