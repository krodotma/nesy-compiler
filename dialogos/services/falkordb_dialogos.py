#!/usr/bin/env python3
"""
FalkorDB Dialogos Session System
=================================

Graph-based session management for Dialogos conversations.
Stores conversation turns, provides analytics, timeline queries, and replay.

Phase 4: Dialogos Session Graph (Steps 20-26)

Features:
- Session and turn graph storage
- Session analytics and metrics
- Timeline queries for conversation history
- Integration with dialogosd daemon
- Session export/replay functionality

Usage:
    from nucleus.tools.falkordb_dialogos import DialogosService

    dialogos = DialogosService()
    dialogos.create_session(session_id, actor)
    dialogos.record_turn(session_id, role, content)
    timeline = dialogos.get_session_timeline(session_id)

CLI:
    python3 nucleus/tools/falkordb_dialogos.py sessions --actor claude
    python3 nucleus/tools/falkordb_dialogos.py timeline SESSION_ID
    python3 nucleus/tools/falkordb_dialogos.py analytics
    python3 nucleus/tools/falkordb_dialogos.py export SESSION_ID
    python3 nucleus/tools/falkordb_dialogos.py sync --source trace
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True


# =============================================================================
# Data Models (Step 20: Dialogos Session Schema)
# =============================================================================

@dataclass
class Session:
    """Dialogos conversation session."""
    id: str
    actor: str
    started_ts: float
    ended_ts: Optional[float] = None
    turn_count: int = 0
    total_tokens: int = 0
    status: str = "active"  # active, completed, interrupted
    model: Optional[str] = None
    parent_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """Single conversation turn."""
    id: str
    session_id: str
    role: str  # user, assistant, system
    content_hash: str
    ts: float
    token_count: int = 0
    latency_ms: Optional[float] = None
    tool_calls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Dialogos Service (Steps 20-26)
# =============================================================================

class DialogosService:
    """
    FalkorDB-backed Dialogos session management.

    Provides:
    - Session/turn CRUD with graph storage
    - Timeline queries
    - Analytics and metrics
    - Export/replay functionality
    - Dialogos daemon integration
    """

    DIALOGOS_TRACE_PATH = Path(".pluribus/dialogos/trace.ndjson")

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
    ):
        self.host = host or os.environ.get("FALKORDB_HOST", "localhost")
        self.port = int(port or os.environ.get("FALKORDB_PORT", "6379"))
        self.database = database or os.environ.get("FALKORDB_DATABASE", "pluribus_kg")
        self._graph = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to FalkorDB."""
        if self._connected:
            return True

        try:
            from falkordb import FalkorDB

            client = FalkorDB(host=self.host, port=self.port)
            self._graph = client.select_graph(self.database)
            self._graph.query("RETURN 1")
            self._connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}", file=sys.stderr)
            return False

    def _ensure_connected(self) -> bool:
        """Ensure connection is established."""
        if not self._connected:
            return self.connect()
        return True

    # =========================================================================
    # Session Management (Step 20)
    # =========================================================================

    def create_session(
        self,
        session_id: str,
        actor: str,
        model: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Session]:
        """Create a new Dialogos session."""
        if not self._ensure_connected():
            return None

        try:
            ts = time.time()
            metadata_str = json.dumps(metadata or {})

            query = """
                MERGE (s:dialogos_session {id: $id})
                SET s.actor = $actor,
                    s.started_ts = $ts,
                    s.status = 'active',
                    s.turn_count = 0,
                    s.total_tokens = 0,
                    s.model = $model,
                    s.parent_session_id = $parent_id,
                    s.metadata = $metadata
                RETURN s.id
            """
            self._graph.query(query, {
                "id": session_id,
                "actor": actor,
                "ts": ts,
                "model": model,
                "parent_id": parent_session_id,
                "metadata": metadata_str,
            })

            # Link to parent if exists
            if parent_session_id:
                self._graph.query("""
                    MATCH (p:dialogos_session {id: $pid}), (c:dialogos_session {id: $cid})
                    MERGE (p)-[:SPAWNED_SESSION]->(c)
                """, {"pid": parent_session_id, "cid": session_id})

            # Also create lineage node
            self._graph.query("""
                MERGE (l:lineage:session {id: $id})
                SET l.node_type = 'session',
                    l.actor = $actor,
                    l.ts = $ts
            """, {"id": session_id, "actor": actor, "ts": ts})

            return Session(
                id=session_id,
                actor=actor,
                started_ts=ts,
                model=model,
                parent_session_id=parent_session_id,
                metadata=metadata or {},
            )

        except Exception as e:
            print(f"Failed to create session: {e}", file=sys.stderr)
            return None

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        if not self._ensure_connected():
            return None

        try:
            result = self._graph.query("""
                MATCH (s:dialogos_session {id: $id})
                RETURN s.id, s.actor, s.started_ts, s.ended_ts, s.turn_count,
                       s.total_tokens, s.status, s.model, s.parent_session_id, s.metadata
            """, {"id": session_id})

            if not result.result_set or not result.result_set[0]:
                return None

            row = result.result_set[0]
            metadata = {}
            try:
                if row[9]:
                    metadata = json.loads(row[9])
            except (json.JSONDecodeError, TypeError):
                pass

            return Session(
                id=row[0],
                actor=row[1] or "unknown",
                started_ts=row[2] or 0,
                ended_ts=row[3],
                turn_count=row[4] or 0,
                total_tokens=row[5] or 0,
                status=row[6] or "unknown",
                model=row[7],
                parent_session_id=row[8],
                metadata=metadata,
            )

        except Exception as e:
            print(f"Failed to get session: {e}", file=sys.stderr)
            return None

    def end_session(self, session_id: str, status: str = "completed") -> bool:
        """End a session."""
        if not self._ensure_connected():
            return False

        try:
            ts = time.time()
            self._graph.query("""
                MATCH (s:dialogos_session {id: $id})
                SET s.ended_ts = $ts, s.status = $status
            """, {"id": session_id, "ts": ts, "status": status})
            return True
        except Exception:
            return False

    # =========================================================================
    # Turn Management (Step 21)
    # =========================================================================

    def record_turn(
        self,
        session_id: str,
        turn_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        latency_ms: Optional[float] = None,
        tool_calls: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Turn]:
        """Record a conversation turn."""
        if not self._ensure_connected():
            return None

        try:
            ts = time.time()
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            tool_calls_str = json.dumps(tool_calls or [])
            metadata_str = json.dumps(metadata or {})

            # Create turn node
            query = """
                MERGE (t:dialogos_turn {id: $id})
                SET t.session_id = $session_id,
                    t.role = $role,
                    t.content_hash = $content_hash,
                    t.ts = $ts,
                    t.token_count = $token_count,
                    t.latency_ms = $latency_ms,
                    t.tool_calls = $tool_calls,
                    t.metadata = $metadata
                RETURN t.id
            """
            self._graph.query(query, {
                "id": turn_id,
                "session_id": session_id,
                "role": role,
                "content_hash": content_hash,
                "ts": ts,
                "token_count": token_count,
                "latency_ms": latency_ms,
                "tool_calls": tool_calls_str,
                "metadata": metadata_str,
            })

            # Link to session
            self._graph.query("""
                MATCH (s:dialogos_session {id: $sid}), (t:dialogos_turn {id: $tid})
                MERGE (s)-[:HAS_TURN]->(t)
            """, {"sid": session_id, "tid": turn_id})

            # Link to previous turn
            self._graph.query("""
                MATCH (s:dialogos_session {id: $sid})-[:HAS_TURN]->(prev:dialogos_turn)
                WHERE prev.id <> $tid AND prev.ts < $ts
                WITH prev ORDER BY prev.ts DESC LIMIT 1
                MATCH (curr:dialogos_turn {id: $tid})
                MERGE (prev)-[:FOLLOWED_BY]->(curr)
            """, {"sid": session_id, "tid": turn_id, "ts": ts})

            # Update session stats
            self._graph.query("""
                MATCH (s:dialogos_session {id: $sid})
                SET s.turn_count = s.turn_count + 1,
                    s.total_tokens = s.total_tokens + $tokens
            """, {"sid": session_id, "tokens": token_count})

            return Turn(
                id=turn_id,
                session_id=session_id,
                role=role,
                content_hash=content_hash,
                ts=ts,
                token_count=token_count,
                latency_ms=latency_ms,
                tool_calls=tool_calls or [],
                metadata=metadata or {},
            )

        except Exception as e:
            print(f"Failed to record turn: {e}", file=sys.stderr)
            return None

    def get_turn(self, turn_id: str) -> Optional[Turn]:
        """Get a turn by ID."""
        if not self._ensure_connected():
            return None

        try:
            result = self._graph.query("""
                MATCH (t:dialogos_turn {id: $id})
                RETURN t.id, t.session_id, t.role, t.content_hash, t.ts,
                       t.token_count, t.latency_ms, t.tool_calls, t.metadata
            """, {"id": turn_id})

            if not result.result_set or not result.result_set[0]:
                return None

            row = result.result_set[0]
            tool_calls = []
            metadata = {}
            try:
                if row[7]:
                    tool_calls = json.loads(row[7])
                if row[8]:
                    metadata = json.loads(row[8])
            except (json.JSONDecodeError, TypeError):
                pass

            return Turn(
                id=row[0],
                session_id=row[1] or "",
                role=row[2] or "unknown",
                content_hash=row[3] or "",
                ts=row[4] or 0,
                token_count=row[5] or 0,
                latency_ms=row[6],
                tool_calls=tool_calls,
                metadata=metadata,
            )

        except Exception:
            return None

    # =========================================================================
    # Timeline Queries (Step 23)
    # =========================================================================

    def get_session_timeline(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[Turn]:
        """Get chronological timeline of turns in a session."""
        if not self._ensure_connected():
            return []

        try:
            result = self._graph.query("""
                MATCH (s:dialogos_session {id: $id})-[:HAS_TURN]->(t:dialogos_turn)
                RETURN t.id, t.session_id, t.role, t.content_hash, t.ts,
                       t.token_count, t.latency_ms, t.tool_calls
                ORDER BY t.ts ASC
                LIMIT $limit
            """, {"id": session_id, "limit": limit})

            turns = []
            if result.result_set:
                for row in result.result_set:
                    tool_calls = []
                    try:
                        if row[7]:
                            tool_calls = json.loads(row[7])
                    except (json.JSONDecodeError, TypeError):
                        pass

                    turns.append(Turn(
                        id=row[0],
                        session_id=row[1] or session_id,
                        role=row[2] or "unknown",
                        content_hash=row[3] or "",
                        ts=row[4] or 0,
                        token_count=row[5] or 0,
                        latency_ms=row[6],
                        tool_calls=tool_calls,
                    ))

            return turns

        except Exception as e:
            print(f"Failed to get timeline: {e}", file=sys.stderr)
            return []

    def get_recent_sessions(
        self,
        actor: Optional[str] = None,
        limit: int = 50,
        since_ts: Optional[float] = None,
    ) -> List[Session]:
        """Get recent sessions."""
        if not self._ensure_connected():
            return []

        try:
            conditions = []
            params = {"limit": limit}

            if actor:
                conditions.append("s.actor = $actor")
                params["actor"] = actor

            if since_ts:
                conditions.append("s.started_ts >= $since")
                params["since"] = since_ts

            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

            result = self._graph.query(f"""
                MATCH (s:dialogos_session)
                {where_clause}
                RETURN s.id, s.actor, s.started_ts, s.ended_ts, s.turn_count,
                       s.total_tokens, s.status, s.model
                ORDER BY s.started_ts DESC
                LIMIT $limit
            """, params)

            sessions = []
            if result.result_set:
                for row in result.result_set:
                    sessions.append(Session(
                        id=row[0],
                        actor=row[1] or "unknown",
                        started_ts=row[2] or 0,
                        ended_ts=row[3],
                        turn_count=row[4] or 0,
                        total_tokens=row[5] or 0,
                        status=row[6] or "unknown",
                        model=row[7],
                    ))

            return sessions

        except Exception as e:
            print(f"Failed to get sessions: {e}", file=sys.stderr)
            return []

    # =========================================================================
    # Analytics (Step 22)
    # =========================================================================

    def get_analytics(self) -> Dict[str, Any]:
        """Get session analytics and metrics."""
        if not self._ensure_connected():
            return {"error": "Not connected"}

        try:
            analytics = {}

            # Total sessions and turns
            result = self._graph.query("""
                MATCH (s:dialogos_session)
                RETURN count(s) AS sessions,
                       sum(s.turn_count) AS total_turns,
                       sum(s.total_tokens) AS total_tokens
            """)
            if result.result_set and result.result_set[0]:
                analytics["total_sessions"] = result.result_set[0][0] or 0
                analytics["total_turns"] = result.result_set[0][1] or 0
                analytics["total_tokens"] = result.result_set[0][2] or 0

            # By status
            result = self._graph.query("""
                MATCH (s:dialogos_session)
                RETURN s.status, count(s) AS cnt
            """)
            analytics["by_status"] = {}
            if result.result_set:
                for row in result.result_set:
                    analytics["by_status"][row[0] or "unknown"] = row[1]

            # By actor
            result = self._graph.query("""
                MATCH (s:dialogos_session)
                RETURN s.actor, count(s) AS cnt, sum(s.turn_count) AS turns
                ORDER BY cnt DESC
                LIMIT 10
            """)
            analytics["by_actor"] = {}
            if result.result_set:
                for row in result.result_set:
                    analytics["by_actor"][row[0] or "unknown"] = {
                        "sessions": row[1],
                        "turns": row[2] or 0,
                    }

            # Average session stats
            result = self._graph.query("""
                MATCH (s:dialogos_session)
                WHERE s.turn_count > 0
                RETURN avg(s.turn_count) AS avg_turns,
                       avg(s.total_tokens) AS avg_tokens
            """)
            if result.result_set and result.result_set[0]:
                analytics["avg_turns_per_session"] = round(result.result_set[0][0] or 0, 2)
                analytics["avg_tokens_per_session"] = round(result.result_set[0][1] or 0, 2)

            # Sessions in last 24h
            day_ago = time.time() - 86400
            result = self._graph.query("""
                MATCH (s:dialogos_session)
                WHERE s.started_ts >= $since
                RETURN count(s) AS cnt
            """, {"since": day_ago})
            analytics["sessions_last_24h"] = result.result_set[0][0] if result.result_set else 0

            # Turn latency stats
            result = self._graph.query("""
                MATCH (t:dialogos_turn)
                WHERE t.latency_ms IS NOT NULL
                RETURN avg(t.latency_ms) AS avg_latency,
                       max(t.latency_ms) AS max_latency
            """)
            if result.result_set and result.result_set[0]:
                analytics["avg_latency_ms"] = round(result.result_set[0][0] or 0, 2)
                analytics["max_latency_ms"] = result.result_set[0][1] or 0

            return analytics

        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Export/Replay (Step 25)
    # =========================================================================

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export session with all turns for replay or analysis."""
        if not self._ensure_connected():
            return {"error": "Not connected"}

        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        turns = self.get_session_timeline(session_id, limit=10000)

        return {
            "session": asdict(session),
            "turns": [asdict(t) for t in turns],
            "export_ts": time.time(),
            "export_format": "dialogos_v1",
        }

    def import_session(self, data: Dict[str, Any]) -> bool:
        """Import a session from exported data."""
        if not self._ensure_connected():
            return False

        try:
            session_data = data.get("session", {})
            turns_data = data.get("turns", [])

            # Create session
            self.create_session(
                session_id=session_data.get("id"),
                actor=session_data.get("actor", "unknown"),
                model=session_data.get("model"),
                metadata=session_data.get("metadata", {}),
            )

            # Record turns
            for turn_data in turns_data:
                self.record_turn(
                    session_id=session_data.get("id"),
                    turn_id=turn_data.get("id"),
                    role=turn_data.get("role", "unknown"),
                    content="",  # Content not stored, only hash
                    token_count=turn_data.get("token_count", 0),
                    latency_ms=turn_data.get("latency_ms"),
                    tool_calls=turn_data.get("tool_calls", []),
                    metadata=turn_data.get("metadata", {}),
                )

            return True

        except Exception as e:
            print(f"Failed to import session: {e}", file=sys.stderr)
            return False

    # =========================================================================
    # Dialogos Daemon Integration (Step 24)
    # =========================================================================

    def sync_from_trace(self, since_ts: Optional[float] = None) -> Dict[str, int]:
        """
        Sync sessions from Dialogos trace file.

        Returns count of synced sessions and turns.
        """
        if not self._ensure_connected():
            return {"error": "Not connected"}

        if not self.DIALOGOS_TRACE_PATH.exists():
            return {"error": "Trace file not found"}

        stats = {"sessions": 0, "turns": 0, "skipped": 0}
        sessions_seen = set()

        with self.DIALOGOS_TRACE_PATH.open("r") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = event.get("ts", 0)
                if since_ts and ts < since_ts:
                    stats["skipped"] += 1
                    continue

                # Extract session ID
                session_id = event.get("session_id") or event.get("id", "")[:36]
                actor = event.get("actor", "unknown")

                # Create session if new
                if session_id and session_id not in sessions_seen:
                    existing = self.get_session(session_id)
                    if not existing:
                        self.create_session(
                            session_id=session_id,
                            actor=actor,
                            metadata={"source": "trace_sync"}
                        )
                        stats["sessions"] += 1
                    sessions_seen.add(session_id)

                # Record turn
                kind = event.get("kind", "")
                if kind in ("request", "response", "user_prompt", "assistant_stop"):
                    turn_id = event.get("id") or hashlib.sha256(
                        f"{session_id}:{ts}".encode()
                    ).hexdigest()[:16]

                    role = "user" if kind in ("request", "user_prompt") else "assistant"
                    content = event.get("content", "")

                    self.record_turn(
                        session_id=session_id,
                        turn_id=turn_id,
                        role=role,
                        content=content if isinstance(content, str) else "",
                        token_count=event.get("token_count", 0),
                        latency_ms=event.get("latency_ms"),
                    )
                    stats["turns"] += 1

        return stats

    def get_active_sessions(self) -> List[Session]:
        """Get currently active sessions."""
        if not self._ensure_connected():
            return []

        try:
            result = self._graph.query("""
                MATCH (s:dialogos_session)
                WHERE s.status = 'active'
                RETURN s.id, s.actor, s.started_ts, s.turn_count, s.total_tokens, s.model
                ORDER BY s.started_ts DESC
            """)

            sessions = []
            if result.result_set:
                for row in result.result_set:
                    sessions.append(Session(
                        id=row[0],
                        actor=row[1] or "unknown",
                        started_ts=row[2] or 0,
                        turn_count=row[3] or 0,
                        total_tokens=row[4] or 0,
                        status="active",
                        model=row[5],
                    ))

            return sessions

        except Exception:
            return []

    # =========================================================================
    # Query Helpers
    # =========================================================================

    def search_sessions(
        self,
        actor: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        min_turns: Optional[int] = None,
        limit: int = 100,
    ) -> List[Session]:
        """Search sessions with filters."""
        if not self._ensure_connected():
            return []

        try:
            conditions = []
            params = {"limit": limit}

            if actor:
                conditions.append("s.actor = $actor")
                params["actor"] = actor

            if model:
                conditions.append("s.model = $model")
                params["model"] = model

            if status:
                conditions.append("s.status = $status")
                params["status"] = status

            if min_turns:
                conditions.append("s.turn_count >= $min_turns")
                params["min_turns"] = min_turns

            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

            result = self._graph.query(f"""
                MATCH (s:dialogos_session)
                {where_clause}
                RETURN s.id, s.actor, s.started_ts, s.ended_ts, s.turn_count,
                       s.total_tokens, s.status, s.model
                ORDER BY s.started_ts DESC
                LIMIT $limit
            """, params)

            sessions = []
            if result.result_set:
                for row in result.result_set:
                    sessions.append(Session(
                        id=row[0],
                        actor=row[1] or "unknown",
                        started_ts=row[2] or 0,
                        ended_ts=row[3],
                        turn_count=row[4] or 0,
                        total_tokens=row[5] or 0,
                        status=row[6] or "unknown",
                        model=row[7],
                    ))

            return sessions

        except Exception:
            return []


def main() -> int:
    parser = argparse.ArgumentParser(description="FalkorDB Dialogos Session System")
    parser.add_argument("command", choices=["sessions", "timeline", "analytics", "export", "sync", "active"],
                        help="Command to execute")
    parser.add_argument("session_id", nargs="?", help="Session ID")
    parser.add_argument("--actor", help="Filter by actor")
    parser.add_argument("--status", help="Filter by status")
    parser.add_argument("--limit", type=int, default=50, help="Result limit")
    parser.add_argument("--since", help="Since timestamp or duration (e.g., 24h)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-o", "--output", help="Output file for export")

    args = parser.parse_args()

    service = DialogosService()

    # Parse since
    since_ts = None
    if args.since:
        since_str = args.since.lower()
        if since_str.endswith("h"):
            since_ts = time.time() - (float(since_str[:-1]) * 3600)
        elif since_str.endswith("d"):
            since_ts = time.time() - (float(since_str[:-1]) * 86400)

    if args.command == "sessions":
        sessions = service.search_sessions(
            actor=args.actor,
            status=args.status,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps([asdict(s) for s in sessions], indent=2))
        else:
            print(f"Sessions ({len(sessions)}):")
            for s in sessions:
                ts_str = datetime.fromtimestamp(s.started_ts).strftime("%Y-%m-%d %H:%M")
                print(f"  [{s.status:10}] {s.id[:30]}... ({s.actor}) {s.turn_count} turns @ {ts_str}")
        return 0

    elif args.command == "timeline":
        if not args.session_id:
            print("Error: session_id required", file=sys.stderr)
            return 1
        turns = service.get_session_timeline(args.session_id, limit=args.limit)
        if args.json:
            print(json.dumps([asdict(t) for t in turns], indent=2))
        else:
            print(f"Timeline for {args.session_id} ({len(turns)} turns):")
            for t in turns:
                ts_str = datetime.fromtimestamp(t.ts).strftime("%H:%M:%S")
                print(f"  [{ts_str}] {t.role:10} ({t.token_count} tokens)")
        return 0

    elif args.command == "analytics":
        analytics = service.get_analytics()
        if args.json:
            print(json.dumps(analytics, indent=2))
        else:
            print("Dialogos Analytics:")
            print(f"  Total sessions: {analytics.get('total_sessions', 0)}")
            print(f"  Total turns: {analytics.get('total_turns', 0)}")
            print(f"  Total tokens: {analytics.get('total_tokens', 0)}")
            print(f"  Sessions (24h): {analytics.get('sessions_last_24h', 0)}")
            print(f"  Avg turns/session: {analytics.get('avg_turns_per_session', 0)}")
            print(f"  Avg latency: {analytics.get('avg_latency_ms', 0)}ms")
        return 0

    elif args.command == "export":
        if not args.session_id:
            print("Error: session_id required", file=sys.stderr)
            return 1
        data = service.export_session(args.session_id)
        output = json.dumps(data, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Exported to {args.output}")
        else:
            print(output)
        return 0

    elif args.command == "sync":
        stats = service.sync_from_trace(since_ts=since_ts)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Synced from trace:")
            print(f"  Sessions: {stats.get('sessions', 0)}")
            print(f"  Turns: {stats.get('turns', 0)}")
            print(f"  Skipped: {stats.get('skipped', 0)}")
        return 0

    elif args.command == "active":
        sessions = service.get_active_sessions()
        if args.json:
            print(json.dumps([asdict(s) for s in sessions], indent=2))
        else:
            print(f"Active sessions ({len(sessions)}):")
            for s in sessions:
                print(f"  {s.id[:30]}... ({s.actor}) {s.turn_count} turns")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
