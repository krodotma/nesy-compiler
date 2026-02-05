#!/usr/bin/env python3
"""
feedback_integrator.py - Feedback Integrator (Step 26)

Learn from user feedback to improve research quality.
Adjusts ranking weights, caches positive results, and tracks patterns.

PBTSO Phase: ITERATE, LEARN

Bus Topics:
- a2a.research.feedback.receive
- a2a.research.feedback.apply
- research.feedback.learn
- research.feedback.stats

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class FeedbackType(Enum):
    """Types of feedback."""
    POSITIVE = "positive"         # Good result
    NEGATIVE = "negative"         # Bad result
    IRRELEVANT = "irrelevant"    # Not what user wanted
    MISSING = "missing"           # Result should have been included
    CORRECTION = "correction"     # User provides correct answer


class FeedbackSource(Enum):
    """Source of feedback."""
    EXPLICIT = "explicit"         # User explicitly gave feedback
    IMPLICIT = "implicit"         # Inferred from behavior
    AUTOMATED = "automated"       # System-generated


@dataclass
class FeedbackConfig:
    """Configuration for feedback integrator."""

    db_path: Optional[str] = None
    learning_rate: float = 0.1      # How quickly to adjust weights
    decay_factor: float = 0.95      # Decay old feedback influence
    min_samples: int = 5            # Minimum samples before adjusting
    max_history: int = 10000        # Maximum feedback history
    enable_learning: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.db_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.db_path = f"{pluribus_root}/.pluribus/research/feedback.db"
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Feedback:
    """A feedback entry."""

    id: str
    feedback_type: FeedbackType
    source: FeedbackSource
    query: str
    result_key: str               # Key identifying the result
    timestamp: float = field(default_factory=time.time)
    score_adjustment: float = 0.0 # How much to adjust score
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "feedback_type": self.feedback_type.value,
            "source": self.source.value,
            "query": self.query,
            "result_key": self.result_key,
            "timestamp": self.timestamp,
            "score_adjustment": self.score_adjustment,
        }


@dataclass
class FeedbackPattern:
    """Learned pattern from feedback."""

    pattern_type: str             # query_type, path_pattern, symbol_kind, etc.
    pattern_value: str
    positive_count: int = 0
    negative_count: int = 0
    total_adjustment: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def net_score(self) -> float:
        """Net positive/negative score."""
        total = self.positive_count + self.negative_count
        if total == 0:
            return 0.0
        return (self.positive_count - self.negative_count) / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "pattern_value": self.pattern_value,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "net_score": self.net_score,
            "total_adjustment": self.total_adjustment,
        }


@dataclass
class FeedbackStats:
    """Statistics on feedback."""

    total_feedback: int = 0
    positive_count: int = 0
    negative_count: int = 0
    accuracy_estimate: float = 0.0
    patterns_learned: int = 0
    weight_adjustments: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Feedback Integrator
# ============================================================================


class FeedbackIntegrator:
    """
    Integrate user feedback to improve research quality.

    Features:
    - Track positive/negative feedback on results
    - Learn patterns from feedback
    - Adjust ranking weights dynamically
    - Cache high-quality results

    PBTSO Phase: ITERATE, LEARN

    Example:
        integrator = FeedbackIntegrator()

        # Record feedback
        integrator.record_feedback(
            query="find UserService",
            result_key="src/user.py:45:UserService",
            feedback_type=FeedbackType.POSITIVE
        )

        # Get adjustments for ranking
        adjustments = integrator.get_ranking_adjustments(results)
    """

    def __init__(
        self,
        config: Optional[FeedbackConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the feedback integrator.

        Args:
            config: Feedback configuration
            bus: AgentBus for event emission
        """
        self.config = config or FeedbackConfig()
        self.bus = bus or AgentBus()

        # Database connection
        self._db: Optional[sqlite3.Connection] = None
        self._init_db()

        # In-memory caches
        self._patterns: Dict[str, FeedbackPattern] = {}
        self._weight_adjustments: Dict[str, float] = {}

        # Load existing patterns
        self._load_patterns()

    def record_feedback(
        self,
        query: str,
        result_key: str,
        feedback_type: FeedbackType,
        source: FeedbackSource = FeedbackSource.EXPLICIT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record feedback on a result.

        Args:
            query: Original query
            result_key: Key identifying the result (path:line:name)
            feedback_type: Type of feedback
            source: Source of feedback
            metadata: Additional metadata

        Returns:
            Feedback ID
        """
        import uuid
        feedback_id = str(uuid.uuid4())[:8]

        # Determine score adjustment
        if feedback_type == FeedbackType.POSITIVE:
            adjustment = 0.1
        elif feedback_type == FeedbackType.NEGATIVE:
            adjustment = -0.15
        elif feedback_type == FeedbackType.IRRELEVANT:
            adjustment = -0.1
        elif feedback_type == FeedbackType.MISSING:
            adjustment = 0.2  # Boost missing results
        else:
            adjustment = 0.0

        feedback = Feedback(
            id=feedback_id,
            feedback_type=feedback_type,
            source=source,
            query=query,
            result_key=result_key,
            score_adjustment=adjustment,
            metadata=metadata or {},
        )

        # Store in database
        self._store_feedback(feedback)

        # Update patterns
        if self.config.enable_learning:
            self._update_patterns(feedback)

        self._emit_with_lock({
            "topic": "a2a.research.feedback.receive",
            "kind": "feedback",
            "data": feedback.to_dict()
        })

        return feedback_id

    def record_implicit_feedback(
        self,
        query: str,
        clicked_results: List[str],
        all_results: List[str],
    ) -> None:
        """
        Record implicit feedback from user behavior.

        Clicked results are positive, skipped top results may be negative.

        Args:
            query: Original query
            clicked_results: Result keys that user clicked
            all_results: All result keys shown
        """
        # Positive feedback for clicked results
        for key in clicked_results:
            self.record_feedback(
                query=query,
                result_key=key,
                feedback_type=FeedbackType.POSITIVE,
                source=FeedbackSource.IMPLICIT,
            )

        # Weak negative for top results that weren't clicked
        for i, key in enumerate(all_results[:3]):
            if key not in clicked_results:
                self.record_feedback(
                    query=query,
                    result_key=key,
                    feedback_type=FeedbackType.IRRELEVANT,
                    source=FeedbackSource.IMPLICIT,
                    metadata={"rank": i},
                )

    def get_ranking_adjustments(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, float]:
        """
        Get ranking adjustments based on learned patterns.

        Args:
            results: List of results to adjust
            query: Current query

        Returns:
            Dict mapping result keys to adjustment factors
        """
        adjustments = {}

        for result in results:
            key = self._make_result_key(result)
            adjustment = 0.0

            # Check for direct feedback on this result
            direct = self._get_direct_adjustment(key)
            adjustment += direct

            # Check for pattern-based adjustments
            pattern_adj = self._get_pattern_adjustment(result, query)
            adjustment += pattern_adj

            if adjustment != 0.0:
                adjustments[key] = adjustment

        self._emit_with_lock({
            "topic": "a2a.research.feedback.apply",
            "kind": "feedback",
            "data": {"adjustments": len(adjustments)}
        })

        return adjustments

    def get_high_quality_results(
        self,
        query: str,
        limit: int = 10,
    ) -> List[str]:
        """
        Get results that received positive feedback for similar queries.

        Args:
            query: Current query
            limit: Maximum results

        Returns:
            List of result keys
        """
        cursor = self._db.execute("""
            SELECT result_key, COUNT(*) as count
            FROM feedback
            WHERE feedback_type = 'positive'
            AND query LIKE ?
            GROUP BY result_key
            ORDER BY count DESC
            LIMIT ?
        """, (f"%{query}%", limit))

        return [row[0] for row in cursor.fetchall()]

    def get_stats(self) -> FeedbackStats:
        """Get feedback statistics."""
        cursor = self._db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN feedback_type = 'positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN feedback_type IN ('negative', 'irrelevant') THEN 1 ELSE 0 END) as negative
            FROM feedback
        """)
        row = cursor.fetchone()

        total = row[0] or 0
        positive = row[1] or 0
        negative = row[2] or 0

        accuracy = positive / total if total > 0 else 0.0

        return FeedbackStats(
            total_feedback=total,
            positive_count=positive,
            negative_count=negative,
            accuracy_estimate=accuracy,
            patterns_learned=len(self._patterns),
            weight_adjustments=self._weight_adjustments.copy(),
        )

    def apply_weight_adjustments(
        self,
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply learned weight adjustments to base weights.

        Args:
            base_weights: Original factor weights

        Returns:
            Adjusted weights
        """
        adjusted = base_weights.copy()

        for factor, adjustment in self._weight_adjustments.items():
            if factor in adjusted:
                new_weight = adjusted[factor] * (1.0 + adjustment * self.config.learning_rate)
                adjusted[factor] = max(0.05, min(0.5, new_weight))  # Clamp to reasonable range

        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def forget_old_feedback(self, days: int = 90) -> int:
        """
        Remove feedback older than specified days.

        Args:
            days: Age threshold

        Returns:
            Number of entries removed
        """
        cutoff = time.time() - (days * 24 * 3600)
        cursor = self._db.execute(
            "DELETE FROM feedback WHERE timestamp < ?",
            (cutoff,)
        )
        self._db.commit()
        return cursor.rowcount

    def export_feedback(self, path: str) -> None:
        """Export all feedback to JSON file."""
        cursor = self._db.execute("SELECT * FROM feedback")
        feedback_list = []

        for row in cursor:
            feedback_list.append({
                "id": row[0],
                "feedback_type": row[1],
                "source": row[2],
                "query": row[3],
                "result_key": row[4],
                "timestamp": row[5],
                "score_adjustment": row[6],
            })

        with open(path, "w") as f:
            json.dump(feedback_list, f, indent=2)

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(str(db_path))
        self._db.row_factory = sqlite3.Row

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                feedback_type TEXT NOT NULL,
                source TEXT NOT NULL,
                query TEXT NOT NULL,
                result_key TEXT NOT NULL,
                timestamp REAL NOT NULL,
                score_adjustment REAL DEFAULT 0,
                metadata TEXT
            )
        """)

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_type TEXT NOT NULL,
                pattern_value TEXT NOT NULL,
                positive_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                total_adjustment REAL DEFAULT 0,
                last_updated REAL,
                PRIMARY KEY (pattern_type, pattern_value)
            )
        """)

        self._db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_result ON feedback(result_key)")
        self._db.commit()

    def _store_feedback(self, feedback: Feedback) -> None:
        """Store feedback in database."""
        self._db.execute("""
            INSERT OR REPLACE INTO feedback
            (id, feedback_type, source, query, result_key, timestamp, score_adjustment, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.id,
            feedback.feedback_type.value,
            feedback.source.value,
            feedback.query,
            feedback.result_key,
            feedback.timestamp,
            feedback.score_adjustment,
            json.dumps(feedback.metadata),
        ))
        self._db.commit()

    def _load_patterns(self) -> None:
        """Load patterns from database."""
        cursor = self._db.execute("SELECT * FROM patterns")

        for row in cursor:
            key = f"{row['pattern_type']}:{row['pattern_value']}"
            self._patterns[key] = FeedbackPattern(
                pattern_type=row["pattern_type"],
                pattern_value=row["pattern_value"],
                positive_count=row["positive_count"],
                negative_count=row["negative_count"],
                total_adjustment=row["total_adjustment"],
                last_updated=row["last_updated"] or time.time(),
            )

    def _update_patterns(self, feedback: Feedback) -> None:
        """Update learned patterns based on feedback."""
        # Extract patterns from result key
        parts = feedback.result_key.split(":")
        if len(parts) >= 1:
            path = parts[0]

            # Path pattern (directory)
            dir_path = str(Path(path).parent)
            self._update_pattern("path", dir_path, feedback)

            # Extension pattern
            ext = Path(path).suffix
            if ext:
                self._update_pattern("extension", ext, feedback)

        # Query pattern (first word/intent)
        query_words = feedback.query.lower().split()
        if query_words:
            self._update_pattern("query_start", query_words[0], feedback)

        # Emit learning event
        self._emit_with_lock({
            "topic": "research.feedback.learn",
            "kind": "feedback",
            "data": {
                "feedback_id": feedback.id,
                "patterns_updated": 3,
            }
        })

    def _update_pattern(
        self,
        pattern_type: str,
        pattern_value: str,
        feedback: Feedback,
    ) -> None:
        """Update a single pattern."""
        key = f"{pattern_type}:{pattern_value}"

        if key not in self._patterns:
            self._patterns[key] = FeedbackPattern(
                pattern_type=pattern_type,
                pattern_value=pattern_value,
            )

        pattern = self._patterns[key]

        if feedback.feedback_type == FeedbackType.POSITIVE:
            pattern.positive_count += 1
        elif feedback.feedback_type in (FeedbackType.NEGATIVE, FeedbackType.IRRELEVANT):
            pattern.negative_count += 1

        pattern.total_adjustment += feedback.score_adjustment
        pattern.last_updated = time.time()

        # Store in database
        self._db.execute("""
            INSERT OR REPLACE INTO patterns
            (pattern_type, pattern_value, positive_count, negative_count, total_adjustment, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_type,
            pattern.pattern_value,
            pattern.positive_count,
            pattern.negative_count,
            pattern.total_adjustment,
            pattern.last_updated,
        ))
        self._db.commit()

    def _get_direct_adjustment(self, result_key: str) -> float:
        """Get direct adjustment for a result based on past feedback."""
        cursor = self._db.execute("""
            SELECT AVG(score_adjustment) as avg_adj
            FROM feedback
            WHERE result_key = ?
        """, (result_key,))

        row = cursor.fetchone()
        return row[0] if row and row[0] else 0.0

    def _get_pattern_adjustment(
        self,
        result: Dict[str, Any],
        query: str,
    ) -> float:
        """Get pattern-based adjustment for a result."""
        adjustment = 0.0
        patterns_matched = 0

        path = result.get("path", "")

        # Check path pattern
        if path:
            dir_path = str(Path(path).parent)
            key = f"path:{dir_path}"
            if key in self._patterns:
                pattern = self._patterns[key]
                if pattern.positive_count + pattern.negative_count >= self.config.min_samples:
                    adjustment += pattern.net_score * 0.1
                    patterns_matched += 1

            # Check extension pattern
            ext = Path(path).suffix
            if ext:
                key = f"extension:{ext}"
                if key in self._patterns:
                    pattern = self._patterns[key]
                    if pattern.positive_count + pattern.negative_count >= self.config.min_samples:
                        adjustment += pattern.net_score * 0.05
                        patterns_matched += 1

        # Average if multiple patterns matched
        if patterns_matched > 0:
            adjustment /= patterns_matched

        return adjustment

    def _make_result_key(self, result: Dict[str, Any]) -> str:
        """Generate a unique key for a result."""
        path = result.get("path", "")
        line = result.get("line", "")
        name = result.get("name", "")
        return f"{path}:{line}:{name}"

    def _emit_with_lock(self, event: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        import socket
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
    """CLI entry point for Feedback Integrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Feedback Integrator (Step 26)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Record command
    record_parser = subparsers.add_parser("record", help="Record feedback")
    record_parser.add_argument("--query", "-q", required=True, help="Query")
    record_parser.add_argument("--result", "-r", required=True, help="Result key (path:line:name)")
    record_parser.add_argument("--type", "-t", choices=[t.value for t in FeedbackType],
                              default="positive", help="Feedback type")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show feedback statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export feedback")
    export_parser.add_argument("--output", "-o", required=True, help="Output file")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old feedback")
    cleanup_parser.add_argument("--days", type=int, default=90, help="Age threshold in days")

    args = parser.parse_args()

    integrator = FeedbackIntegrator()

    if args.command == "record":
        feedback_type = FeedbackType(args.type)
        feedback_id = integrator.record_feedback(
            query=args.query,
            result_key=args.result,
            feedback_type=feedback_type,
        )
        print(f"Recorded feedback: {feedback_id}")

    elif args.command == "stats":
        stats = integrator.get_stats()
        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("Feedback Statistics:")
            print(f"  Total Feedback: {stats.total_feedback}")
            print(f"  Positive: {stats.positive_count}")
            print(f"  Negative: {stats.negative_count}")
            print(f"  Accuracy Estimate: {stats.accuracy_estimate:.1%}")
            print(f"  Patterns Learned: {stats.patterns_learned}")

    elif args.command == "export":
        integrator.export_feedback(args.output)
        print(f"Exported feedback to {args.output}")

    elif args.command == "cleanup":
        removed = integrator.forget_old_feedback(args.days)
        print(f"Removed {removed} entries older than {args.days} days")

    integrator.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
