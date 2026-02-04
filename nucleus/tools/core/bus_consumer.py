#!/usr/bin/env python3
"""
Bus Consumer Core Library (v3)
==============================

Robust, persistent, and efficient consumer for the Pluribus event bus.

Features:
- **Persistence**: Tracks read offset and file identity (inode) to resume correctly after restarts.
- **Rotation Awareness**: Detects file rotation (inode change) and handles it seamlessly.
- **Efficient Filtering**: Supports low-level grep filtering for scanning large histories.
- **Generator API**: Pythonic `consume()` iterator.
- **FalkorDB Integration**: Optionally persist events to FalkorDB graph for querying.

Usage:
    consumer = BusConsumer(bus_path, state_path=Path("my_state.json"))
    for event in consumer.consume(tail=True):
        process(event)
        consumer.ack() # Persist state

FalkorDB Usage:
    consumer = BusConsumer(bus_path, falkordb_enabled=True)
    consumer.persist_to_graph(event)  # Write event to graph
    events = consumer.query_from_graph(topic="git.commit", limit=100)  # Query events
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import logging
from pathlib import Path
from typing import Callable, Optional, Iterator, Any, Dict, Generator, List

# Configure logging
logger = logging.getLogger("BusConsumer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class BusConsumer:
    """Unified consumer for the Pluribus event bus with FalkorDB integration."""

    def __init__(
        self,
        bus_path: Path,
        state_path: Optional[Path] = None,
        falkordb_enabled: bool = False,
        falkordb_host: Optional[str] = None,
        falkordb_port: Optional[int] = None,
        falkordb_database: Optional[str] = None,
    ):
        self.bus_path = Path(bus_path)
        self.state_path = Path(state_path) if state_path else None

        # State
        self._last_inode: int = 0
        self._last_offset: int = 0
        self._last_id: Optional[str] = None

        # FalkorDB integration
        self._falkordb_enabled = falkordb_enabled
        self._falkordb_host = falkordb_host or os.environ.get("FALKORDB_HOST", "localhost")
        self._falkordb_port = int(falkordb_port or os.environ.get("FALKORDB_PORT", "6379"))
        self._falkordb_database = falkordb_database or os.environ.get("FALKORDB_DATABASE", "pluribus_kg")
        self._falkordb_graph = None
        self._falkordb_stats = {"persisted": 0, "errors": 0, "queries": 0}

        self.load_state()

        if self._falkordb_enabled:
            self._init_falkordb()

    def load_state(self):
        """Load state from file."""
        if self.state_path and self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self._last_inode = data.get("inode", 0)
                self._last_offset = data.get("offset", 0)
                self._last_id = data.get("last_id")
                logger.debug(f"Loaded state: inode={self._last_inode}, offset={self._last_offset}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def save_state(self):
        """Persist current state."""
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "inode": self._last_inode,
                "offset": self._last_offset,
                "last_id": self._last_id,
                "ts": time.time()
            }
            # Atomic write
            tmp_path = self.state_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(state))
            tmp_path.replace(self.state_path)

    def ack(self):
        """Explicitly acknowledge processing up to the current point."""
        self.save_state()

    def _open_bus(self):
        """Open the bus file, handling rotation logic."""
        if not self.bus_path.exists():
            raise FileNotFoundError(f"Bus file not found: {self.bus_path}")

        f = self.bus_path.open("r", encoding="utf-8", errors="replace")
        fd = f.fileno()
        stats = os.fstat(fd)
        current_inode = stats.st_ino
        current_size = stats.st_size

        # Rotation/Reset Detection
        if self._last_inode != 0 and current_inode != self._last_inode:
            logger.info(f"File rotation detected (inode {self._last_inode} -> {current_inode}). Resetting offset.")
            self._last_offset = 0
        elif self._last_offset > current_size:
            logger.warning(f"File truncated (offset {self._last_offset} > size {current_size}). Resetting offset.")
            self._last_offset = 0
        
        # Seek to position
        f.seek(self._last_offset)
        
        # Update inode in state
        self._last_inode = current_inode
        return f

    def consume(self, tail: bool = False, topic_filter: Optional[str] = None, poll_interval: float = 0.5) -> Generator[Optional[dict], None, None]:
        """
        Yields events from the bus.
        
        Args:
            tail: If True, keep listening for new events indefinitely.
            topic_filter: Optional regex string for pre-filtering (optimization).
            poll_interval: Seconds to wait when tailing and file is empty.
        
        Yields:
            dict: Event data.
            None: If tailing and no data is available (heartbeat for maintenance tasks).
        """
        try:
            f = self._open_bus()
        except FileNotFoundError:
            if tail:
                logger.info("Bus file not found, waiting...")
                while not self.bus_path.exists():
                    time.sleep(poll_interval)
                    yield None
                f = self._open_bus()
            else:
                return

        while True:
            line = f.readline()
            if not line:
                if not tail:
                    break
                
                # Tailing logic
                current_pos = f.tell()
                
                # Check for rotation
                try:
                    stats = self.bus_path.stat()
                    if stats.st_ino != self._last_inode:
                        logger.info("Rotation detected during tail. Reopening.")
                        f.close()
                        f = self._open_bus()
                        continue
                except FileNotFoundError:
                    pass

                time.sleep(poll_interval)
                f.seek(current_pos)
                yield None
                continue

            # Update offset *after* reading the line but *before* yielding
            # (In case of crash during processing, we might re-process this line, which is safer than skipping)
            self._last_offset = f.tell()
            
            # 1. Fast String Filter (avoid JSON parse if possible)
            if topic_filter and topic_filter not in line:
                 # Note: This is a loose check. For strict regex, we'd need to parse or use re.
                 # But avoiding json.loads is the main perf win.
                 # If topic_filter is simple (e.g. "git.commit"), this is fine.
                 # If it's complex regex, we might need to parse.
                 pass

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 2. Precise Filter
            # (Implement strict logic here if needed, but for now relying on consumer to filter)
            
            self._last_id = event.get("id")
            yield event

    def grep_history(self, pattern: str, limit: int = 1000) -> Generator[dict, None, None]:
        """
        Efficiently grep historical events using system `grep`.
        Does NOT update state/offset (read-only scan).
        """
        cmd = ["grep", "-E", pattern, str(self.bus_path)]
        # optionally add tail -n limit if needed, but grep doesn't support it easily.
        # We'll limit in python.
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        count = 0
        try:
            if proc.stdout:
                for line in proc.stdout:
                    try:
                        yield json.loads(line)
                        count += 1
                        if count >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        finally:
            proc.terminate()

    # =========================================================================
    # FalkorDB Integration Methods
    # =========================================================================

    def _init_falkordb(self) -> bool:
        """Initialize FalkorDB connection."""
        try:
            from falkordb import FalkorDB

            client = FalkorDB(host=self._falkordb_host, port=self._falkordb_port)
            self._falkordb_graph = client.select_graph(self._falkordb_database)
            # Test connection
            self._falkordb_graph.query("RETURN 1")
            logger.info(f"FalkorDB connected: {self._falkordb_database}")
            return True
        except Exception as e:
            logger.warning(f"FalkorDB init failed: {e}")
            self._falkordb_enabled = False
            return False

    def persist_to_graph(self, event: Dict[str, Any]) -> bool:
        """
        Persist a bus event to FalkorDB as an event node.

        Creates an :event node with properties from the event dict.
        Links to :actor node if actor field present.
        Links to related entities via trace_id.

        Args:
            event: Bus event dictionary

        Returns:
            True if persisted successfully
        """
        if not self._falkordb_graph:
            return False

        try:
            event_id = event.get("id") or hashlib.sha256(
                json.dumps(event, sort_keys=True).encode()
            ).hexdigest()[:16]
            topic = event.get("topic", "unknown")
            actor = event.get("actor", "system")
            ts = event.get("ts") or time.time()
            trace_id = event.get("trace_id")
            data_json = json.dumps(event.get("data", {}), separators=(",", ":"))

            # Create event node
            query = """
                MERGE (e:bus_event {id: $id})
                SET e.topic = $topic,
                    e.actor = $actor,
                    e.ts = $ts,
                    e.trace_id = $trace_id,
                    e.data_json = $data_json,
                    e.indexed_ts = timestamp()
                RETURN e.id
            """
            params = {
                "id": event_id,
                "topic": topic,
                "actor": actor,
                "ts": ts,
                "trace_id": trace_id,
                "data_json": data_json,
            }
            self._falkordb_graph.query(query, params)

            # Link to actor if present
            if actor and actor != "system":
                actor_query = """
                    MERGE (a:agent {id: $actor_id})
                    WITH a
                    MATCH (e:bus_event {id: $event_id})
                    MERGE (a)-[:EMITTED]->(e)
                """
                self._falkordb_graph.query(
                    actor_query, {"actor_id": actor, "event_id": event_id}
                )

            # Link via trace_id for event correlation
            if trace_id:
                trace_query = """
                    MATCH (e1:bus_event {trace_id: $trace_id})
                    WHERE e1.id <> $event_id
                    WITH e1 ORDER BY e1.ts LIMIT 1
                    MATCH (e2:bus_event {id: $event_id})
                    MERGE (e1)-[:FOLLOWED_BY]->(e2)
                """
                self._falkordb_graph.query(
                    trace_query, {"trace_id": trace_id, "event_id": event_id}
                )

            self._falkordb_stats["persisted"] += 1
            return True

        except Exception as e:
            logger.warning(f"Failed to persist event to graph: {e}")
            self._falkordb_stats["errors"] += 1
            return False

    def query_from_graph(
        self,
        topic: Optional[str] = None,
        actor: Optional[str] = None,
        trace_id: Optional[str] = None,
        since_ts: Optional[float] = None,
        until_ts: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query events from FalkorDB graph.

        Args:
            topic: Filter by topic (supports prefix match with *)
            actor: Filter by actor
            trace_id: Filter by trace ID
            since_ts: Filter events after this timestamp
            until_ts: Filter events before this timestamp
            limit: Maximum events to return

        Returns:
            List of event dictionaries
        """
        if not self._falkordb_graph:
            return []

        try:
            conditions = []
            params = {"limit": limit}

            if topic:
                if topic.endswith("*"):
                    conditions.append("e.topic STARTS WITH $topic_prefix")
                    params["topic_prefix"] = topic[:-1]
                else:
                    conditions.append("e.topic = $topic")
                    params["topic"] = topic

            if actor:
                conditions.append("e.actor = $actor")
                params["actor"] = actor

            if trace_id:
                conditions.append("e.trace_id = $trace_id")
                params["trace_id"] = trace_id

            if since_ts:
                conditions.append("e.ts >= $since_ts")
                params["since_ts"] = since_ts

            if until_ts:
                conditions.append("e.ts <= $until_ts")
                params["until_ts"] = until_ts

            where_clause = " AND ".join(conditions) if conditions else "true"

            query = f"""
                MATCH (e:bus_event)
                WHERE {where_clause}
                RETURN e.id, e.topic, e.actor, e.ts, e.trace_id, e.data_json
                ORDER BY e.ts DESC
                LIMIT $limit
            """

            result = self._falkordb_graph.query(query, params)
            self._falkordb_stats["queries"] += 1

            events = []
            if result.result_set:
                for row in result.result_set:
                    payload = {}
                    try:
                        if row[5]:
                            payload = json.loads(row[5])
                    except (json.JSONDecodeError, TypeError):
                        pass

                    events.append({
                        "id": row[0],
                        "topic": row[1],
                        "actor": row[2],
                        "ts": row[3],
                        "trace_id": row[4],
                        "data": payload,
                    })

            return events

        except Exception as e:
            logger.warning(f"Failed to query events from graph: {e}")
            return []

    def get_event_trace(self, trace_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all events in a trace, ordered by timestamp.

        Args:
            trace_id: The trace ID to query
            limit: Maximum events to return

        Returns:
            List of events in chronological order
        """
        return self.query_from_graph(trace_id=trace_id, limit=limit)

    def get_actor_activity(
        self, actor: str, since_ts: Optional[float] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent events from a specific actor.

        Args:
            actor: Actor ID
            since_ts: Optional timestamp to filter from
            limit: Maximum events

        Returns:
            List of events from the actor
        """
        return self.query_from_graph(actor=actor, since_ts=since_ts, limit=limit)

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get FalkorDB statistics for this consumer.

        Returns:
            Dict with persisted count, errors, queries, and graph stats
        """
        stats = dict(self._falkordb_stats)
        stats["enabled"] = self._falkordb_enabled

        if self._falkordb_graph:
            try:
                result = self._falkordb_graph.query(
                    "MATCH (e:bus_event) RETURN count(e) AS cnt"
                )
                stats["event_count"] = (
                    result.result_set[0][0] if result.result_set else 0
                )
            except Exception:
                stats["event_count"] = "error"

        return stats
