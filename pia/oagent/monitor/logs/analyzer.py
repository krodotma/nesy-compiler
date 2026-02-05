#!/usr/bin/env python3
"""
Log Analyzer - Step 255

Analyzes logs for patterns, anomalies, and classifications.

PBTSO Phase: VERIFY

Bus Topics:
- monitor.logs.analyze (subscribed)
- monitor.logs.patterns (emitted)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

from .collector import LogCollector, LogEntry, LogLevel, get_collector


class PatternType(Enum):
    """Types of log patterns."""
    ERROR = "error"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_FAILURE = "auth_failure"
    CONNECTION = "connection"
    RESOURCE = "resource"
    CUSTOM = "custom"


@dataclass
class LogPattern:
    """A detected log pattern.

    Attributes:
        pattern_type: Type of pattern
        regex: Regular expression
        description: Human-readable description
        severity: Pattern severity
        count: Number of matches
        examples: Example log entries
        first_seen: First occurrence timestamp
        last_seen: Last occurrence timestamp
    """
    pattern_type: PatternType
    regex: str
    description: str
    severity: LogLevel = LogLevel.WARN
    count: int = 0
    examples: List[str] = field(default_factory=list)
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "regex": self.regex,
            "description": self.description,
            "severity": self.severity.value,
            "count": self.count,
            "examples": self.examples[:5],  # Limit examples
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class AnalysisResult:
    """Result of log analysis.

    Attributes:
        patterns: Detected patterns
        anomalies: Detected anomalies
        statistics: Log statistics
        recommendations: Recommended actions
        timestamp: Analysis timestamp
    """
    patterns: List[LogPattern] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patterns": [p.to_dict() for p in self.patterns],
            "anomalies": self.anomalies,
            "statistics": self.statistics,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


class LogAnalyzer:
    """
    Analyze logs for patterns, anomalies, and insights.

    The analyzer:
    - Detects common error patterns
    - Identifies anomalous log volumes
    - Classifies log entries
    - Provides recommendations

    Example:
        analyzer = LogAnalyzer()
        result = analyzer.analyze(window_s=3600)
        for pattern in result.patterns:
            print(f"{pattern.pattern_type.value}: {pattern.count} occurrences")
    """

    BUS_TOPICS = {
        "analyze": "monitor.logs.analyze",
        "patterns": "monitor.logs.patterns",
    }

    # Built-in patterns to detect
    BUILTIN_PATTERNS: List[Tuple[PatternType, str, str]] = [
        (PatternType.ERROR, r"(?i)\b(error|err|failed|failure|fatal)\b", "Generic error"),
        (PatternType.EXCEPTION, r"(?i)(exception|traceback|stack\s*trace)", "Exception/traceback"),
        (PatternType.TIMEOUT, r"(?i)(timeout|timed?\s*out|deadline\s*exceeded)", "Timeout"),
        (PatternType.RATE_LIMIT, r"(?i)(rate\s*limit|throttl|too\s*many\s*requests|429)", "Rate limiting"),
        (PatternType.AUTH_FAILURE, r"(?i)(auth.*fail|unauthorized|forbidden|403|401|access\s*denied)", "Authentication failure"),
        (PatternType.CONNECTION, r"(?i)(connection\s*(refused|reset|closed)|ECONNREFUSED|ECONNRESET)", "Connection issue"),
        (PatternType.RESOURCE, r"(?i)(out\s*of\s*memory|OOM|disk\s*full|no\s*space|resource\s*exhausted)", "Resource exhaustion"),
    ]

    def __init__(
        self,
        collector: Optional[LogCollector] = None,
        bus_dir: Optional[str] = None
    ):
        """Initialize log analyzer.

        Args:
            collector: Log collector to analyze
            bus_dir: Directory for bus events
        """
        self._collector = collector or get_collector()
        self._custom_patterns: List[Tuple[PatternType, str, str]] = []
        self._compiled_patterns: List[Tuple[PatternType, Pattern, str]] = []

        # Compile builtin patterns
        for ptype, regex, desc in self.BUILTIN_PATTERNS:
            self._compiled_patterns.append((ptype, re.compile(regex), desc))

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def add_pattern(
        self,
        pattern_type: PatternType,
        regex: str,
        description: str
    ) -> None:
        """Add a custom pattern.

        Args:
            pattern_type: Pattern type
            regex: Regular expression
            description: Description
        """
        self._custom_patterns.append((pattern_type, regex, description))
        self._compiled_patterns.append((pattern_type, re.compile(regex), description))

    def analyze(
        self,
        window_s: int = 3600,
        source: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze logs in time window.

        Args:
            window_s: Time window in seconds
            source: Optional source filter

        Returns:
            Analysis result
        """
        start_time = time.time() - window_s
        logs = self._collector.search(
            start_time=start_time,
            source=source,
            limit=100000
        )

        result = AnalysisResult()

        # Detect patterns
        result.patterns = self._detect_patterns(logs)

        # Detect anomalies
        result.anomalies = self._detect_anomalies(logs)

        # Compute statistics
        result.statistics = self._compute_statistics(logs)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def detect_patterns(
        self,
        logs: List[LogEntry]
    ) -> List[LogPattern]:
        """Detect patterns in logs.

        Args:
            logs: Log entries to analyze

        Returns:
            Detected patterns
        """
        return self._detect_patterns(logs)

    def classify_entry(self, entry: LogEntry) -> List[PatternType]:
        """Classify a log entry by pattern types.

        Args:
            entry: Log entry

        Returns:
            List of matching pattern types
        """
        matches = []
        for ptype, regex, _ in self._compiled_patterns:
            if regex.search(entry.message):
                matches.append(ptype)
        return matches

    def get_error_rate(self, window_s: int = 300) -> float:
        """Calculate error rate in time window.

        Args:
            window_s: Time window

        Returns:
            Error rate (0-1)
        """
        start_time = time.time() - window_s
        all_logs = self._collector.search(start_time=start_time, limit=100000)
        error_logs = self._collector.search(
            start_time=start_time,
            min_level=LogLevel.ERROR,
            limit=100000
        )

        if not all_logs:
            return 0.0

        return len(error_logs) / len(all_logs)

    def get_top_errors(
        self,
        window_s: int = 3600,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top error messages by frequency.

        Args:
            window_s: Time window
            limit: Number of results

        Returns:
            List of error info
        """
        start_time = time.time() - window_s
        errors = self._collector.search(
            start_time=start_time,
            min_level=LogLevel.ERROR,
            limit=10000
        )

        # Group by normalized message
        counter: Counter = Counter()
        examples: Dict[str, LogEntry] = {}

        for entry in errors:
            # Normalize message (remove numbers, UUIDs, etc.)
            normalized = self._normalize_message(entry.message)
            counter[normalized] += 1
            if normalized not in examples:
                examples[normalized] = entry

        # Return top errors
        results = []
        for msg, count in counter.most_common(limit):
            example = examples[msg]
            results.append({
                "normalized_message": msg,
                "count": count,
                "example": example.message,
                "source": example.source,
                "level": example.level.value,
            })

        return results

    def handle_analyze_request(self, event: Dict[str, Any]) -> AnalysisResult:
        """Handle analysis request from bus.

        Args:
            event: Bus event

        Returns:
            Analysis result
        """
        data = event.get("data", {})
        window_s = data.get("window_s", 3600)
        source = data.get("source")

        result = self.analyze(window_s, source)
        self.emit_patterns_event(result)

        return result

    def emit_patterns_event(self, result: AnalysisResult) -> str:
        """Emit patterns detected event.

        Args:
            result: Analysis result

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS["patterns"],
            "kind": "event",
            "level": "info",
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": result.to_dict(),
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _detect_patterns(self, logs: List[LogEntry]) -> List[LogPattern]:
        """Detect patterns in logs.

        Args:
            logs: Log entries

        Returns:
            Detected patterns
        """
        pattern_matches: Dict[str, LogPattern] = {}

        for entry in logs:
            for ptype, regex, desc in self._compiled_patterns:
                if regex.search(entry.message):
                    key = f"{ptype.value}:{desc}"
                    if key not in pattern_matches:
                        pattern_matches[key] = LogPattern(
                            pattern_type=ptype,
                            regex=regex.pattern,
                            description=desc,
                            severity=entry.level,
                            first_seen=entry.timestamp,
                        )

                    pattern = pattern_matches[key]
                    pattern.count += 1
                    pattern.last_seen = entry.timestamp
                    if len(pattern.examples) < 5:
                        pattern.examples.append(entry.message[:200])

        return sorted(
            pattern_matches.values(),
            key=lambda p: p.count,
            reverse=True
        )

    def _detect_anomalies(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect anomalies in log patterns.

        Args:
            logs: Log entries

        Returns:
            Detected anomalies
        """
        anomalies = []

        # Check for sudden spikes in error rate
        if logs:
            # Split into time buckets
            bucket_size = 60  # 1 minute buckets
            buckets: Dict[int, List[LogEntry]] = defaultdict(list)

            for entry in logs:
                bucket = int(entry.timestamp / bucket_size)
                buckets[bucket].append(entry)

            # Calculate error rate per bucket
            error_rates = []
            for bucket_time, bucket_logs in sorted(buckets.items()):
                errors = sum(1 for e in bucket_logs if e.level >= LogLevel.ERROR)
                rate = errors / len(bucket_logs) if bucket_logs else 0
                error_rates.append((bucket_time, rate, len(bucket_logs)))

            # Detect spikes (rate > 2 * average)
            if error_rates:
                avg_rate = sum(r[1] for r in error_rates) / len(error_rates)
                for bucket_time, rate, count in error_rates:
                    if rate > avg_rate * 2 and rate > 0.1:
                        anomalies.append({
                            "type": "error_spike",
                            "timestamp": bucket_time * bucket_size,
                            "error_rate": rate,
                            "average_rate": avg_rate,
                            "log_count": count,
                        })

        # Check for source going quiet
        source_counts: Dict[str, int] = defaultdict(int)
        for entry in logs:
            source_counts[entry.source] += 1

        # Sources with very few logs might be problematic
        avg_count = sum(source_counts.values()) / len(source_counts) if source_counts else 0
        for source, count in source_counts.items():
            if count < avg_count * 0.1 and avg_count > 10:
                anomalies.append({
                    "type": "low_volume",
                    "source": source,
                    "count": count,
                    "average_count": avg_count,
                })

        return anomalies

    def _compute_statistics(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Compute log statistics.

        Args:
            logs: Log entries

        Returns:
            Statistics dictionary
        """
        if not logs:
            return {
                "total_logs": 0,
                "by_level": {},
                "by_source": {},
                "error_rate": 0,
            }

        by_level: Dict[str, int] = defaultdict(int)
        by_source: Dict[str, int] = defaultdict(int)

        for entry in logs:
            by_level[entry.level.value] += 1
            by_source[entry.source] += 1

        error_count = sum(
            by_level.get(level.value, 0)
            for level in [LogLevel.ERROR, LogLevel.FATAL]
        )

        return {
            "total_logs": len(logs),
            "by_level": dict(by_level),
            "by_source": dict(by_source),
            "error_rate": error_count / len(logs),
            "unique_sources": len(by_source),
            "time_range_s": (
                logs[-1].timestamp - logs[0].timestamp if len(logs) > 1 else 0
            ),
        }

    def _generate_recommendations(self, result: AnalysisResult) -> List[str]:
        """Generate recommendations from analysis.

        Args:
            result: Analysis result

        Returns:
            List of recommendations
        """
        recommendations = []

        # High error rate
        if result.statistics.get("error_rate", 0) > 0.05:
            recommendations.append(
                f"High error rate detected ({result.statistics['error_rate']:.1%}). "
                "Investigate top errors and recent deployments."
            )

        # Pattern-specific recommendations
        for pattern in result.patterns:
            if pattern.count >= 10:
                if pattern.pattern_type == PatternType.TIMEOUT:
                    recommendations.append(
                        f"Timeout pattern detected ({pattern.count} occurrences). "
                        "Check downstream service health and timeout configurations."
                    )
                elif pattern.pattern_type == PatternType.RATE_LIMIT:
                    recommendations.append(
                        f"Rate limiting detected ({pattern.count} occurrences). "
                        "Consider increasing rate limits or implementing backoff."
                    )
                elif pattern.pattern_type == PatternType.RESOURCE:
                    recommendations.append(
                        f"Resource exhaustion detected ({pattern.count} occurrences). "
                        "Check memory/disk usage and consider scaling."
                    )

        # Anomaly-specific recommendations
        for anomaly in result.anomalies:
            if anomaly.get("type") == "error_spike":
                recommendations.append(
                    f"Error spike detected at {anomaly['timestamp']}. "
                    "Correlate with deployments and external events."
                )

        return recommendations

    def _normalize_message(self, message: str) -> str:
        """Normalize a log message for grouping.

        Args:
            message: Original message

        Returns:
            Normalized message
        """
        # Remove UUIDs
        normalized = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '<UUID>',
            message,
            flags=re.IGNORECASE
        )
        # Remove hex strings
        normalized = re.sub(r'\b[0-9a-f]{16,}\b', '<HEX>', normalized, flags=re.IGNORECASE)
        # Remove numbers
        normalized = re.sub(r'\b\d+\b', '<N>', normalized)
        # Remove paths
        normalized = re.sub(r'/[\w/.-]+', '<PATH>', normalized)

        return normalized


# Singleton instance
_analyzer: Optional[LogAnalyzer] = None


def get_analyzer() -> LogAnalyzer:
    """Get or create the log analyzer singleton.

    Returns:
        LogAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = LogAnalyzer()
    return _analyzer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log Analyzer (Step 255)")
    parser.add_argument("--analyze", action="store_true", help="Run full analysis")
    parser.add_argument("--window", type=int, default=3600, help="Window in seconds")
    parser.add_argument("--source", help="Filter by source")
    parser.add_argument("--errors", action="store_true", help="Show top errors")
    parser.add_argument("--error-rate", action="store_true", help="Show error rate")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    analyzer = get_analyzer()

    if args.analyze:
        result = analyzer.analyze(args.window, args.source)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("Log Analysis Results:")
            print(f"  Total logs: {result.statistics.get('total_logs', 0)}")
            print(f"  Error rate: {result.statistics.get('error_rate', 0):.1%}")
            print(f"\nPatterns detected: {len(result.patterns)}")
            for p in result.patterns[:10]:
                print(f"  - {p.pattern_type.value}: {p.count} ({p.description})")
            print(f"\nRecommendations:")
            for r in result.recommendations:
                print(f"  * {r}")

    if args.errors:
        errors = analyzer.get_top_errors(args.window)
        if args.json:
            print(json.dumps(errors, indent=2))
        else:
            print("Top Errors:")
            for e in errors:
                print(f"  [{e['count']}] {e['normalized_message'][:80]}")

    if args.error_rate:
        rate = analyzer.get_error_rate(args.window)
        if args.json:
            print(json.dumps({"error_rate": rate}))
        else:
            print(f"Error rate: {rate:.2%}")
