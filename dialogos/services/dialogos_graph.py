#!/usr/bin/env python3
"""
Dialogos Graph Writer
=====================

Persists Dialogos conversation sessions as graph structures in FalkorDB.
Enables graph-based search, entity linking, and conversation analytics.

Phase 4 Step 21: Implement Session Graph Writer

Schema:
    Node: :session
        - id: STRING
        - actor: STRING
        - start_ts: FLOAT
        - end_ts: FLOAT
        - message_count: INT
        - status: STRING
    
    Node: :message
        - id: STRING (UUID)
        - role: STRING (user, assistant, system)
        - content_hash: STRING
        - ts: FLOAT
    
    Relationships:
        (:session)-[:OWNED_BY]->(:agent)
        (:session)-[:CONTAINS]->(:message)
        (:message)-[:FOLLOWS]->(:message)
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import logging
import hashlib
from typing import List, Dict, Optional, Any

sys.dont_write_bytecode = True

# Ensure repo root is in sys.path
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools.core.falkordb_pool import get_graph
    POOL_AVAILABLE = True
except ImportError:
    POOL_AVAILABLE = False

logger = logging.getLogger("DialogosGraph")

class DialogosGraphWriter:
    """Write dialogos sessions to FalkorDB."""

    def __init__(self, db_name: Optional[str] = None):
        if not POOL_AVAILABLE:
            raise RuntimeError("FalkorDB Pool not available")
        self.db_name = db_name or os.environ.get("FALKORDB_DATABASE", "pluribus_kg")

    def _now(self) -> float:
        return time.time()

    def start_session(self, session_id: str, actor: str) -> None:
        """Initialize a new session in the graph."""
        cypher = """
        MERGE (s:session {id: $id})
        ON CREATE SET 
            s.actor = $actor,
            s.start_ts = $ts,
            s.status = 'active',
            s.message_count = 0
        WITH s
        MERGE (a:agent {id: $actor})
        MERGE (s)-[:OWNED_BY]->(a)
        """
        
        params = {
            "id": session_id,
            "actor": actor,
            "ts": self._now()
        }
        
        with get_graph(self.db_name) as graph:
            graph.query(cypher, params)

    def add_message(self, session_id: str, 
                    role: str, 
                    content: str, 
                    msg_id: Optional[str] = None, 
                    ts: Optional[float] = None) -> str:
        """Add a message to a session, linking it to the previous message."""
        if not msg_id:
            msg_id = str(uuid.uuid4())
        if not ts:
            ts = self._now()
            
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        cypher = """
        MATCH (s:session {id: $session_id})
        
        CREATE (m:message {
            id: $msg_id,
            role: $role,
            content_hash: $content_hash,
            ts: $ts
        })
        
        CREATE (s)-[:CONTAINS]->(m)
        
        WITH m, s
        
        // Find previous message in this session to link
        // We use a subquery or optional match pattern to find the most recent previous message
        OPTIONAL MATCH (prev:message)<-[:CONTAINS]-(s)
        WHERE prev.ts < $ts AND prev.id <> m.id
        WITH m, s, prev
        ORDER BY prev.ts DESC 
        LIMIT 1
        
        FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
            CREATE (m)-[:FOLLOWS]->(prev)
        )
        
        // Update session stats
        SET s.message_count = coalesce(s.message_count, 0) + 1,
            s.end_ts = $ts
            
        RETURN m.id
        """
        
        params = {
            "session_id": session_id,
            "msg_id": msg_id,
            "role": role,
            "content_hash": content_hash,
            "ts": ts
        }
        
        with get_graph(self.db_name) as graph:
            res = graph.query(cypher, params)
            return res.result_set[0][0]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        cypher = """
        MATCH (s:session {id: $id})
        RETURN s
        """
        with get_graph(self.db_name) as graph:
            res = graph.query(cypher, {"id": session_id})
            if not res.result_set:
                return None
            node = res.result_set[0][0]
            return node.properties if hasattr(node, 'properties') else dict(node)

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for a session (chronological)."""
        cypher = """
        MATCH (s:session {id: $id})-[:CONTAINS]->(m:message)
        RETURN m
        ORDER BY m.ts ASC
        LIMIT $limit
        """
        messages = []
        with get_graph(self.db_name) as graph:
            res = graph.query(cypher, {"id": session_id, "limit": limit})
            for row in res.result_set:
                node = row[0]
                props = node.properties if hasattr(node, 'properties') else dict(node)
                messages.append(props)
        return messages

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "msg", "get", "list"])
    parser.add_argument("--session", help="Session ID")
    parser.add_argument("--actor", default="gemini")
    parser.add_argument("--role", default="user")
    parser.add_argument("--content", default="Hello")
    
    args = parser.parse_args()
    writer = DialogosGraphWriter()
    
    if args.command == "start":
        sid = args.session or str(uuid.uuid4())
        writer.start_session(sid, args.actor)
        print(f"Session started: {sid}")
    elif args.command == "msg":
        if not args.session:
            sys.exit("Session ID required")
        mid = writer.add_message(args.session, args.role, args.content)
        print(f"Message added: {mid}")
    elif args.command == "get":
        if not args.session:
            sys.exit("Session ID required")
        print(writer.get_session(args.session))
        print("Messages:")
        for m in writer.get_messages(args.session):
            print(f"  [{m['role']}] {m['ts']}")
