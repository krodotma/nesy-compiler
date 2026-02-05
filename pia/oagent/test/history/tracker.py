#!/usr/bin/env python3
"""
Step 123: Test History Tracker

Tracks and analyzes historical test data for trends and patterns.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.history.record (subscribes)
- test.history.query (subscribes)
- test.history.trend (emits)

Dependencies: Steps 101-122 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

class TrendDirection(Enum):
    """Direction of a trend."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    UNKNOWN = "unknown"


class AggregationType(Enum):
    """Type of aggregation for queries."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    RUN = "run"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class HistoryRecord:
    """A single test result history record."""
    run_id: str
    test_name: str
    status: str
    duration_ms: float
    timestamp: float
    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    error_message: Optional[str] = None
    coverage_percent: Optional[float] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "test_name": self.test_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "error_message": self.error_message,
            "coverage_percent": self.coverage_percent,
            "file_path": self.file_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryRecord":
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            test_name=data.get("test_name", ""),
            status=data.get("status", "unknown"),
            duration_ms=data.get("duration_ms", 0),
            timestamp=data.get("timestamp", time.time()),
            commit_sha=data.get("commit_sha"),
            branch=data.get("branch"),
            error_message=data.get("error_message"),
            coverage_percent=data.get("coverage_percent"),
            file_path=data.get("file_path"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HistoryQuery:
    """Query parameters for historical data."""
    test_name: Optional[str] = None
    test_pattern: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    limit: int = 100
    offset: int = 0
    aggregation: Optional[AggregationType] = None
    order_by: str = "timestamp"
    order_desc: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "test_pattern": self.test_pattern,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "limit": self.limit,
            "offset": self.offset,
        }


@dataclass
class TrendData:
    """Trend analysis data."""
    metric: str
    direction: TrendDirection
    current_value: float
    previous_value: float
    change_percent: float
    data_points: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    period_days: int = 7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "change_percent": self.change_percent,
            "data_points": self.data_points,
            "confidence": self.confidence,
            "period_days": self.period_days,
        }


@dataclass
class HistoryConfig:
    """
    Configuration for history tracking.

    Attributes:
        db_path: Path to SQLite database
        retention_days: Days to retain history
        enable_trends: Enable trend analysis
        trend_window_days: Window for trend calculation
        batch_size: Batch size for writes
        output_dir: Output directory for reports
    """
    db_path: str = ".pluribus/test-agent/history/test_history.db"
    retention_days: int = 90
    enable_trends: bool = True
    trend_window_days: int = 7
    batch_size: int = 100
    output_dir: str = ".pluribus/test-agent/history"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "db_path": self.db_path,
            "retention_days": self.retention_days,
            "enable_trends": self.enable_trends,
            "trend_window_days": self.trend_window_days,
        }


@dataclass
class HistoryStats:
    """Statistics from history."""
    total_runs: int = 0
    total_tests: int = 0
    unique_tests: int = 0
    pass_rate: float = 0.0
    avg_duration_ms: float = 0.0
    flaky_tests: int = 0
    most_failed: List[Tuple[str, int]] = field(default_factory=list)
    slowest_tests: List[Tuple[str, float]] = field(default_factory=list)
    trends: List[TrendData] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_runs": self.total_runs,
            "total_tests": self.total_tests,
            "unique_tests": self.unique_tests,
            "pass_rate": self.pass_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "flaky_tests": self.flaky_tests,
            "most_failed": self.most_failed,
            "slowest_tests": self.slowest_tests,
            "trends": [t.to_dict() for t in self.trends],
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class HistoryBus:
    """Bus interface for history tracking with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Test History Tracker
# ============================================================================

class TestHistoryTracker:
    """
    Tracks and analyzes historical test data.

    Features:
    - SQLite-based persistent storage
    - Trend analysis
    - Query interface
    - Automatic cleanup of old data
    - Aggregation support

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.history.record, test.history.query, test.history.trend
    """

    BUS_TOPICS = {
        "record": "test.history.record",
        "query": "test.history.query",
        "trend": "test.history.trend",
        "stats": "test.history.stats",
    }

    def __init__(self, bus=None, config: Optional[HistoryConfig] = None):
        """
        Initialize the history tracker.

        Args:
            bus: Optional bus instance
            config: History configuration
        """
        self.bus = bus or HistoryBus()
        self.config = config or HistoryConfig()
        self._db: Optional[sqlite3.Connection] = None
        self._write_buffer: List[HistoryRecord] = []

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row

        # Create tables
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                timestamp REAL NOT NULL,
                commit_sha TEXT,
                branch TEXT,
                error_message TEXT,
                coverage_percent REAL,
                file_path TEXT,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_test_name ON test_results(test_name);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON test_results(timestamp);
            CREATE INDEX IF NOT EXISTS idx_run_id ON test_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_status ON test_results(status);

            CREATE TABLE IF NOT EXISTS run_summaries (
                run_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                total_tests INTEGER,
                passed INTEGER,
                failed INTEGER,
                skipped INTEGER,
                duration_s REAL,
                coverage_percent REAL,
                commit_sha TEXT,
                branch TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_run_timestamp ON run_summaries(timestamp);
        """)

        self._db.commit()

    def record(self, record: HistoryRecord) -> None:
        """
        Record a test result.

        Args:
            record: History record to store
        """
        self._write_buffer.append(record)

        if len(self._write_buffer) >= self.config.batch_size:
            self._flush_buffer()

    def record_batch(self, records: List[HistoryRecord]) -> None:
        """Record multiple test results at once."""
        self._write_buffer.extend(records)
        self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush write buffer to database."""
        if not self._write_buffer or not self._db:
            return

        cursor = self._db.cursor()

        for record in self._write_buffer:
            cursor.execute("""
                INSERT INTO test_results
                (run_id, test_name, status, duration_ms, timestamp,
                 commit_sha, branch, error_message, coverage_percent, file_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id,
                record.test_name,
                record.status,
                record.duration_ms,
                record.timestamp,
                record.commit_sha,
                record.branch,
                record.error_message,
                record.coverage_percent,
                record.file_path,
                json.dumps(record.metadata),
            ))

        self._db.commit()
        self._write_buffer.clear()

        # Emit event
        self._emit_event("record", {"count": cursor.rowcount})

    def record_run_summary(
        self,
        run_id: str,
        total_tests: int,
        passed: int,
        failed: int,
        skipped: int,
        duration_s: float,
        coverage_percent: Optional[float] = None,
        commit_sha: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> None:
        """Record a run summary."""
        if not self._db:
            return

        cursor = self._db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO run_summaries
            (run_id, timestamp, total_tests, passed, failed, skipped,
             duration_s, coverage_percent, commit_sha, branch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            time.time(),
            total_tests,
            passed,
            failed,
            skipped,
            duration_s,
            coverage_percent,
            commit_sha,
            branch,
        ))

        self._db.commit()

    def query(self, query: HistoryQuery) -> List[HistoryRecord]:
        """
        Query historical records.

        Args:
            query: Query parameters

        Returns:
            List of matching records
        """
        if not self._db:
            return []

        sql = "SELECT * FROM test_results WHERE 1=1"
        params = []

        if query.test_name:
            sql += " AND test_name = ?"
            params.append(query.test_name)

        if query.test_pattern:
            sql += " AND test_name LIKE ?"
            params.append(query.test_pattern.replace("*", "%"))

        if query.status:
            sql += " AND status = ?"
            params.append(query.status)

        if query.start_time:
            sql += " AND timestamp >= ?"
            params.append(query.start_time)

        if query.end_time:
            sql += " AND timestamp <= ?"
            params.append(query.end_time)

        if query.commit_sha:
            sql += " AND commit_sha = ?"
            params.append(query.commit_sha)

        if query.branch:
            sql += " AND branch = ?"
            params.append(query.branch)

        # Ordering
        order = "DESC" if query.order_desc else "ASC"
        sql += f" ORDER BY {query.order_by} {order}"

        # Pagination
        sql += " LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])

        cursor = self._db.cursor()
        cursor.execute(sql, params)

        records = []
        for row in cursor.fetchall():
            metadata = {}
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    pass

            records.append(HistoryRecord(
                run_id=row["run_id"],
                test_name=row["test_name"],
                status=row["status"],
                duration_ms=row["duration_ms"],
                timestamp=row["timestamp"],
                commit_sha=row["commit_sha"],
                branch=row["branch"],
                error_message=row["error_message"],
                coverage_percent=row["coverage_percent"],
                file_path=row["file_path"],
                metadata=metadata,
            ))

        return records

    def get_test_history(self, test_name: str, limit: int = 50) -> List[HistoryRecord]:
        """Get history for a specific test."""
        query = HistoryQuery(test_name=test_name, limit=limit)
        return self.query(query)

    def get_stats(self, days: int = 7) -> HistoryStats:
        """
        Get statistics for the specified period.

        Args:
            days: Number of days to analyze

        Returns:
            HistoryStats with computed statistics
        """
        if not self._db:
            return HistoryStats()

        start_time = time.time() - (days * 24 * 3600)
        cursor = self._db.cursor()

        # Basic counts
        cursor.execute("""
            SELECT
                COUNT(*) as total_tests,
                COUNT(DISTINCT run_id) as total_runs,
                COUNT(DISTINCT test_name) as unique_tests,
                AVG(duration_ms) as avg_duration,
                SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM test_results
            WHERE timestamp >= ?
        """, (start_time,))

        row = cursor.fetchone()
        total = row["total_tests"]
        passed = row["passed"] or 0
        pass_rate = (passed / total * 100) if total > 0 else 0

        stats = HistoryStats(
            total_runs=row["total_runs"] or 0,
            total_tests=total,
            unique_tests=row["unique_tests"] or 0,
            pass_rate=pass_rate,
            avg_duration_ms=row["avg_duration"] or 0,
        )

        # Most failed tests
        cursor.execute("""
            SELECT test_name, COUNT(*) as fail_count
            FROM test_results
            WHERE status = 'failed' AND timestamp >= ?
            GROUP BY test_name
            ORDER BY fail_count DESC
            LIMIT 10
        """, (start_time,))

        stats.most_failed = [(row["test_name"], row["fail_count"]) for row in cursor.fetchall()]

        # Slowest tests
        cursor.execute("""
            SELECT test_name, AVG(duration_ms) as avg_duration
            FROM test_results
            WHERE timestamp >= ?
            GROUP BY test_name
            ORDER BY avg_duration DESC
            LIMIT 10
        """, (start_time,))

        stats.slowest_tests = [(row["test_name"], row["avg_duration"]) for row in cursor.fetchall()]

        # Flaky tests (tests with both pass and fail in recent history)
        cursor.execute("""
            SELECT test_name
            FROM test_results
            WHERE timestamp >= ?
            GROUP BY test_name
            HAVING COUNT(DISTINCT status) > 1
        """, (start_time,))

        stats.flaky_tests = len(cursor.fetchall())

        # Compute trends
        if self.config.enable_trends:
            stats.trends = self._compute_trends(days)

        return stats

    def _compute_trends(self, days: int) -> List[TrendData]:
        """Compute trend data for various metrics."""
        if not self._db:
            return []

        trends = []
        now = time.time()
        cursor = self._db.cursor()

        # Pass rate trend
        current_start = now - (days * 24 * 3600)
        previous_start = current_start - (days * 24 * 3600)

        # Current period pass rate
        cursor.execute("""
            SELECT
                SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(COUNT(*), 0) as pass_rate
            FROM test_results
            WHERE timestamp >= ?
        """, (current_start,))
        current_rate = cursor.fetchone()["pass_rate"] or 0

        # Previous period pass rate
        cursor.execute("""
            SELECT
                SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(COUNT(*), 0) as pass_rate
            FROM test_results
            WHERE timestamp >= ? AND timestamp < ?
        """, (previous_start, current_start))
        previous_rate = cursor.fetchone()["pass_rate"] or 0

        change = current_rate - previous_rate
        direction = TrendDirection.STABLE
        if change > 1:
            direction = TrendDirection.IMPROVING
        elif change < -1:
            direction = TrendDirection.DECLINING

        trends.append(TrendData(
            metric="pass_rate",
            direction=direction,
            current_value=current_rate,
            previous_value=previous_rate,
            change_percent=change,
            period_days=days,
        ))

        # Duration trend
        cursor.execute("""
            SELECT AVG(duration_ms) as avg_duration
            FROM test_results WHERE timestamp >= ?
        """, (current_start,))
        current_duration = cursor.fetchone()["avg_duration"] or 0

        cursor.execute("""
            SELECT AVG(duration_ms) as avg_duration
            FROM test_results
            WHERE timestamp >= ? AND timestamp < ?
        """, (previous_start, current_start))
        previous_duration = cursor.fetchone()["avg_duration"] or 0

        duration_change = 0
        if previous_duration > 0:
            duration_change = ((current_duration - previous_duration) / previous_duration) * 100

        direction = TrendDirection.STABLE
        if duration_change < -5:
            direction = TrendDirection.IMPROVING  # Faster is better
        elif duration_change > 5:
            direction = TrendDirection.DECLINING

        trends.append(TrendData(
            metric="avg_duration_ms",
            direction=direction,
            current_value=current_duration,
            previous_value=previous_duration,
            change_percent=duration_change,
            period_days=days,
        ))

        return trends

    def cleanup_old_data(self) -> int:
        """Remove data older than retention period."""
        if not self._db:
            return 0

        cutoff = time.time() - (self.config.retention_days * 24 * 3600)

        cursor = self._db.cursor()
        cursor.execute("DELETE FROM test_results WHERE timestamp < ?", (cutoff,))
        cursor.execute("DELETE FROM run_summaries WHERE timestamp < ?", (cutoff,))

        deleted = cursor.rowcount
        self._db.commit()

        return deleted

    def get_run_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get history of test runs."""
        if not self._db:
            return []

        cursor = self._db.cursor()
        cursor.execute("""
            SELECT * FROM run_summaries
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def export_history(self, output_path: str, format: str = "json") -> str:
        """Export history to file."""
        if not self._db:
            return ""

        cursor = self._db.cursor()
        cursor.execute("SELECT * FROM test_results ORDER BY timestamp DESC")

        records = [dict(row) for row in cursor.fetchall()]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_file, "w") as f:
                json.dump(records, f, indent=2)
        elif format == "csv":
            import csv
            with open(output_file, "w", newline="") as f:
                if records:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)

        return str(output_file)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.history.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "history",
            "actor": "test-agent",
            "data": data,
        })

    def close(self) -> None:
        """Close database connection."""
        if self._write_buffer:
            self._flush_buffer()
        if self._db:
            self._db.close()
            self._db = None


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test History Tracker."""
    import argparse

    parser = argparse.ArgumentParser(description="Test History Tracker")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--days", type=int, default=7, help="Days to analyze")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query history")
    query_parser.add_argument("--test", help="Test name")
    query_parser.add_argument("--pattern", help="Test name pattern")
    query_parser.add_argument("--status", help="Filter by status")
    query_parser.add_argument("--limit", type=int, default=20)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export history")
    export_parser.add_argument("output", help="Output file path")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old data")
    cleanup_parser.add_argument("--days", type=int, default=90, help="Retention days")

    # Runs command
    runs_parser = subparsers.add_parser("runs", help="List recent runs")
    runs_parser.add_argument("--limit", type=int, default=20)

    # Common arguments
    parser.add_argument("--db", default=".pluribus/test-agent/history/test_history.db")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = HistoryConfig(db_path=args.db)
    tracker = TestHistoryTracker(config=config)

    try:
        if args.command == "stats":
            stats = tracker.get_stats(args.days)
            if args.json:
                print(json.dumps(stats.to_dict(), indent=2))
            else:
                print(f"\n{'='*60}")
                print(f"Test History Statistics (Last {args.days} days)")
                print(f"{'='*60}")
                print(f"Total Runs: {stats.total_runs}")
                print(f"Total Tests: {stats.total_tests}")
                print(f"Unique Tests: {stats.unique_tests}")
                print(f"Pass Rate: {stats.pass_rate:.1f}%")
                print(f"Avg Duration: {stats.avg_duration_ms:.0f}ms")
                print(f"Flaky Tests: {stats.flaky_tests}")

                if stats.most_failed:
                    print(f"\nMost Failed Tests:")
                    for name, count in stats.most_failed[:5]:
                        print(f"  - {name}: {count} failures")

                if stats.trends:
                    print(f"\nTrends:")
                    for trend in stats.trends:
                        print(f"  - {trend.metric}: {trend.direction.value} "
                              f"({trend.change_percent:+.1f}%)")

        elif args.command == "query":
            query = HistoryQuery(
                test_name=args.test,
                test_pattern=args.pattern,
                status=args.status,
                limit=args.limit,
            )
            records = tracker.query(query)

            if args.json:
                print(json.dumps([r.to_dict() for r in records], indent=2))
            else:
                print(f"\nFound {len(records)} records:")
                for record in records:
                    dt = datetime.fromtimestamp(record.timestamp)
                    print(f"  [{record.status}] {record.test_name} "
                          f"({record.duration_ms:.0f}ms) - {dt.strftime('%Y-%m-%d %H:%M')}")

        elif args.command == "export":
            output = tracker.export_history(args.output, args.format)
            print(f"Exported to: {output}")

        elif args.command == "cleanup":
            config.retention_days = args.days
            deleted = tracker.cleanup_old_data()
            print(f"Deleted {deleted} old records")

        elif args.command == "runs":
            runs = tracker.get_run_history(args.limit)
            if args.json:
                print(json.dumps(runs, indent=2))
            else:
                print(f"\nRecent Test Runs:")
                for run in runs:
                    dt = datetime.fromtimestamp(run["timestamp"])
                    print(f"  {run['run_id'][:8]}... - "
                          f"{run['passed']}/{run['total_tests']} passed - "
                          f"{dt.strftime('%Y-%m-%d %H:%M')}")

        else:
            parser.print_help()

    finally:
        tracker.close()


if __name__ == "__main__":
    main()
