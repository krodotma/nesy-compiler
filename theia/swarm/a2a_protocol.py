"""
Theia DKIN v29 Compliance — A2A Protocol Implementation.

Implements canonical PLURIBUS v1 A2A protocol:
- Mandatory ack within 60s
- Codeword liveness heartbeat every 5min
- Multi-channel aggregation (bus + HEXIS + tmux + git)
- Protocol headers: DKIN v29 | PAIP v15 | CITIZEN v1
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from enum import Enum, auto


# Canonical protocol header
PROTO_HEADER = "DKIN v29 | PAIP v15 | CITIZEN v1"


class CollabStatus(Enum):
    """Collaboration status per DKIN v29."""
    PROPOSED = auto()
    ACCEPTED = auto()
    DECLINED = auto()
    BUSY = auto()
    ACTIVE = auto()
    BLOCKED = auto()
    COMPLETING = auto()
    DISSOCIATED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class Collaboration:
    """Active A2A collaboration tracking."""
    collab_id: str
    codeword: str
    initiator: str
    target_agents: List[str]
    scope: str
    status: CollabStatus = CollabStatus.PROPOSED
    iteration: int = 0
    iterations_planned: int = 10
    ttl_s: int = 3600
    mode: str = "sequential"  # or "parallel"
    sync_points: List[int] = field(default_factory=list)
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_heartbeat: Dict[str, float] = field(default_factory=dict)
    ack_received: Dict[str, bool] = field(default_factory=dict)


class A2AProtocol:
    """
    DKIN v29 A2A Collaboration Protocol.
    
    Implements:
    - Handshake with mandatory 60s ack
    - Heartbeat every 5 minutes
    - Multi-channel liveness detection
    - Codeword tracking
    """
    
    ACK_TIMEOUT_S = 60
    HEARTBEAT_INTERVAL_S = 300
    MAX_RETRIES = 3
    
    def __init__(
        self,
        agent_id: str = "theia",
        bus_path: str = "/pluribus/.pluribus/bus/events.ndjson",
    ):
        self.agent_id = agent_id
        self.bus_path = Path(bus_path)
        self.active_collabs: Dict[str, Collaboration] = {}
    
    def propose_collaboration(
        self,
        codeword: str,
        targets: List[str],
        scope: str,
        iterations: int = 10,
        mode: str = "sequential",
        sync_points: Optional[List[int]] = None,
    ) -> str:
        """
        Propose A2A collaboration per DKIN v29.
        
        Emits: a2a.handshake.propose
        Returns: collab_id
        """
        collab_id = str(uuid.uuid4())
        
        collab = Collaboration(
            collab_id=collab_id,
            codeword=codeword,
            initiator=self.agent_id,
            target_agents=targets,
            scope=scope,
            iterations_planned=iterations,
            mode=mode,
            sync_points=sync_points or [],
        )
        
        self.active_collabs[collab_id] = collab
        
        self._emit({
            "topic": "a2a.handshake.propose",
            "kind": "request",
            "data": {
                "codeword": codeword,
                "collab_id": collab_id,
                "target_agents": targets,
                "scope": scope,
                "iterations_planned": iterations,
                "ttl_s": collab.ttl_s,
                "mode": mode,
                "sync_points": sync_points or [],
            }
        })
        
        return collab_id
    
    def acknowledge(
        self,
        collab_id: str,
        codeword: str,
        status: str = "accepted",
        channel: str = "bus",
    ) -> None:
        """
        Acknowledge collaboration proposal (MANDATORY within 60s).
        
        Emits: a2a.handshake.ack
        """
        self._emit({
            "topic": "a2a.handshake.ack",
            "kind": "response",
            "data": {
                "codeword": codeword,
                "collab_id": collab_id,
                "status": status,
                "channel_preference": channel,
                "heartbeat_interval_s": self.HEARTBEAT_INTERVAL_S,
            }
        })
    
    def heartbeat(self, collab_id: str, status: str = "active", last_action: str = "") -> None:
        """
        Emit liveness heartbeat (required every 5 min during collaboration).
        
        Emits: a2a.heartbeat
        """
        collab = self.active_collabs.get(collab_id)
        if not collab:
            return
        
        self._emit({
            "topic": "a2a.heartbeat",
            "kind": "metric",
            "data": {
                "codeword": collab.codeword,
                "collab_id": collab_id,
                "iteration": collab.iteration,
                "status": status,
                "last_action": last_action,
                "channel": "bus",
            }
        })
        
        collab.last_heartbeat[self.agent_id] = time.time()
    
    def complete_iteration(self, collab_id: str, next_agent: Optional[str] = None) -> None:
        """
        Signal iteration completion.
        
        Emits: a2a.iteration.complete
        """
        collab = self.active_collabs.get(collab_id)
        if not collab:
            return
        
        collab.iteration += 1
        
        self._emit({
            "topic": "a2a.iteration.complete",
            "kind": "event",
            "data": {
                "codeword": collab.codeword,
                "collab_id": collab_id,
                "iteration": collab.iteration,
                "next_agent": next_agent,
            }
        })
    
    def complete_collaboration(self, collab_id: str, summary: str = "") -> None:
        """
        End collaboration successfully.
        
        Emits: a2a.collab.complete
        """
        collab = self.active_collabs.get(collab_id)
        if not collab:
            return
        
        collab.status = CollabStatus.COMPLETED
        
        self._emit({
            "topic": "a2a.collab.complete",
            "kind": "event",
            "data": {
                "codeword": collab.codeword,
                "collab_id": collab_id,
                "summary": summary,
                "participants": [collab.initiator] + collab.target_agents,
                "iterations_completed": collab.iteration,
                "duration_s": time.time() - collab.created_at,
            }
        })
        
        del self.active_collabs[collab_id]
    
    def check_dissociation(self, collab_id: str) -> List[str]:
        """
        Check for dissociated agents (3+ missed heartbeats).
        
        Returns list of dissociated agent IDs.
        """
        collab = self.active_collabs.get(collab_id)
        if not collab:
            return []
        
        now = time.time()
        threshold = 3 * self.HEARTBEAT_INTERVAL_S
        dissociated = []
        
        all_agents = [collab.initiator] + collab.target_agents
        for agent in all_agents:
            last = collab.last_heartbeat.get(agent, collab.created_at)
            if now - last > threshold:
                dissociated.append(agent)
        
        if dissociated:
            self._emit({
                "topic": "a2a.dissociation.detected",
                "kind": "alert",
                "level": "warn",
                "data": {
                    "codeword": collab.codeword,
                    "collab_id": collab_id,
                    "missing_agents": dissociated,
                    "action": "Initiating recovery scan",
                }
            })
        
        return dissociated
    
    def _emit(self, event: Dict[str, Any]) -> None:
        """Emit DKIN v29 compliant event."""
        event["id"] = str(uuid.uuid4())
        event["ts"] = time.time()
        event["iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        event["actor"] = self.agent_id
        event["proto"] = PROTO_HEADER
        event["level"] = event.get("level", "info")
        
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[A2AProtocol] Failed to emit: {e}")
    
    def status(self) -> Dict[str, Any]:
        """Get protocol status."""
        return {
            "agent_id": self.agent_id,
            "proto": PROTO_HEADER,
            "active_collabs": len(self.active_collabs),
            "codewords": [c.codeword for c in self.active_collabs.values()],
        }


__all__ = [
    "PROTO_HEADER",
    "CollabStatus",
    "Collaboration",
    "A2AProtocol",
]
