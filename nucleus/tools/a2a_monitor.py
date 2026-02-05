#!/usr/bin/env python3
"""
a2a_monitor.py - Agent-to-Agent Collaboration Monitor

Monitors active A2A collaborations to detect and recover from dissociation.
Implements DKIN v30 A2A protocol. Integrates with OHM (Omega Heart Monitor)
on VPS for live agent liveness data.

Ring: 1 (Infrastructure)
Protocol: DKIN v30 | PAIP v16 | Citizen v1

OHM Integration:
    - VPS OHM at 69.169.104.17 emits ohm.status every ~30s
    - OHM tracks `agents.last_seen` with per-agent timestamps
    - OHM tracks `a2a.requests` and `a2a.responses`
    - This monitor extends OHM by adding codeword-specific tracking

Usage:
    python3 a2a_monitor.py watch <codeword>           # Monitor a codeword
    python3 a2a_monitor.py status [<codeword>]        # Show status
    python3 a2a_monitor.py recover <codeword> <agent> # Attempt recovery
    python3 a2a_monitor.py daemon                     # Run as daemon
    python3 a2a_monitor.py ohm-status                 # Query VPS OHM

Bus Topics:
    a2a.handshake.propose
    a2a.handshake.ack
    a2a.handshake.timeout
    a2a.heartbeat
    a2a.dissociation.detected
    a2a.dissociation.recovered
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

# Import agent_bus for proper file-locked event emission
try:
    from agent_bus import emit_event as emit_bus_event, resolve_bus_paths
    AGENT_BUS_AVAILABLE = True
except ImportError:
    AGENT_BUS_AVAILABLE = False


# DR-gating for NDJSON reads (DKIN v30 bus policy)
def _ndjson_read_allowed() -> bool:
    mode = (os.environ.get("PLURIBUS_NDJSON_MODE") or "").strip().lower()
    if not mode or mode in {"allow", "enabled", "on"}:
        return True
    if mode in {"dr", "disaster", "recovery"}:
        return os.environ.get("PLURIBUS_DR_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    return mode not in {"off", "disabled", "deny", "no"}


# Configuration
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))
HEXIS_DIR = Path(os.environ.get("PLURIBUS_HEXIS_DIR", "/tmp/hexis"))
COLLAB_STATE_FILE = Path(".pluribus/a2a/collaborations.json")

HEARTBEAT_INTERVAL_S = 300  # 5 minutes
HEARTBEAT_TIMEOUT_S = 900   # 15 minutes (3 missed heartbeats)
HANDSHAKE_TIMEOUT_S = 60
MAX_HANDSHAKE_RETRIES = 3


@dataclass
class A2AParticipant:
    """A participant in an A2A collaboration."""
    agent_id: str
    status: str = "pending"  # pending, active, dissociated, completed
    last_heartbeat_ts: float = 0.0
    channel: str = "bus"
    iteration: int = 0


@dataclass
class A2ACollaboration:
    """An active A2A collaboration."""
    collab_id: str
    codeword: str
    initiator: str
    participants: Dict[str, A2AParticipant] = field(default_factory=dict)
    mode: str = "turn_based"  # turn_based, parallel
    sync_points: List[int] = field(default_factory=list)
    current_iteration: int = 0
    created_ts: float = field(default_factory=time.time)
    ttl_s: int = 3600
    status: str = "active"  # active, completing, completed, failed


class A2AMonitor:
    """
    Monitor and manage A2A collaborations.
    
    Responsibilities:
    - Track active codewords and participants
    - Monitor heartbeats across channels (bus, HEXIS, tmux, git)
    - Detect and alert on dissociation
    - Attempt recovery via multi-channel discovery
    """

    def __init__(self, bus_dir: Path = None):
        self.bus_dir = bus_dir or BUS_DIR
        self.bus_path = self.bus_dir / "events.ndjson"
        self.state_file = COLLAB_STATE_FILE
        self.collaborations: Dict[str, A2ACollaboration] = {}
        self._load_state()

    def _load_state(self):
        """Load collaboration state from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                for cid, cdata in data.get("collaborations", {}).items():
                    participants = {
                        pid: A2AParticipant(**pdata)
                        for pid, pdata in cdata.pop("participants", {}).items()
                    }
                    self.collaborations[cid] = A2ACollaboration(
                        **{k: v for k, v in cdata.items() if k in A2ACollaboration.__dataclass_fields__},
                        participants=participants
                    )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load state: {e}")

    def _save_state(self):
        """Save collaboration state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "collaborations": {
                cid: {
                    **asdict(collab),
                    "participants": {
                        pid: asdict(p) for pid, p in collab.participants.items()
                    }
                }
                for cid, collab in self.collaborations.items()
            },
            "updated_ts": time.time()
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def _emit_bus_event(self, topic: str, data: Dict[str, Any], level: str = "info"):
        """Emit event to the Pluribus bus with proper file locking."""
        if AGENT_BUS_AVAILABLE:
            # Use agent_bus for proper fcntl.flock() protected writes
            try:
                paths = resolve_bus_paths(str(self.bus_path.parent))
                return emit_bus_event(
                    paths,
                    topic=topic,
                    kind="event",
                    level=level,
                    actor="a2a_monitor",
                    data=data,
                    trace_id=None,
                    run_id=None,
                    durable=False,
                )
            except Exception:
                pass  # Fall through to direct write

        # Fallback: direct write (no file locking)
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "kind": "event",
            "level": level,
            "actor": "a2a_monitor",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event["id"]

    def propose_collaboration(
        self,
        codeword: str,
        initiator: str,
        target_agents: List[str],
        scope: str,
        iterations: int = 10,
        mode: str = "parallel",
        ttl_s: int = 3600,
    ) -> str:
        """
        Propose a new A2A collaboration.
        
        Returns the collab_id.
        """
        collab_id = f"collab-{uuid.uuid4().hex[:8]}"
        
        collab = A2ACollaboration(
            collab_id=collab_id,
            codeword=codeword,
            initiator=initiator,
            mode=mode,
            ttl_s=ttl_s,
        )
        
        # Add initiator as participant
        collab.participants[initiator] = A2AParticipant(
            agent_id=initiator,
            status="active",
            last_heartbeat_ts=time.time(),
        )
        
        # Add targets as pending
        for agent in target_agents:
            collab.participants[agent] = A2AParticipant(agent_id=agent)
        
        self.collaborations[collab_id] = collab
        self._save_state()
        
        self._emit_bus_event("a2a.handshake.propose", {
            "codeword": codeword,
            "collab_id": collab_id,
            "initiator": initiator,
            "target_agents": target_agents,
            "scope": scope,
            "iterations_planned": iterations,
            "mode": mode,
            "ttl_s": ttl_s,
        })
        
        return collab_id

    def acknowledge_handshake(
        self,
        collab_id: str,
        agent_id: str,
        status: str = "accepted",
        channel: str = "bus",
    ) -> bool:
        """
        Acknowledge a handshake proposal.
        """
        collab = self.collaborations.get(collab_id)
        if not collab:
            return False
        
        if agent_id not in collab.participants:
            return False
        
        participant = collab.participants[agent_id]
        participant.status = "active" if status == "accepted" else "declined"
        participant.last_heartbeat_ts = time.time()
        participant.channel = channel
        
        self._save_state()
        
        self._emit_bus_event("a2a.handshake.ack", {
            "codeword": collab.codeword,
            "collab_id": collab_id,
            "agent": agent_id,
            "status": status,
            "channel": channel,
        })
        
        return True

    def emit_heartbeat(
        self,
        codeword: str,
        agent_id: str,
        iteration: int = 0,
        channel: str = "bus",
        last_action: str = "",
    ):
        """Emit a heartbeat for an agent."""
        # Find collaboration by codeword
        collab = None
        for c in self.collaborations.values():
            if c.codeword == codeword and c.status == "active":
                collab = c
                break
        
        if collab and agent_id in collab.participants:
            collab.participants[agent_id].last_heartbeat_ts = time.time()
            collab.participants[agent_id].iteration = iteration
            collab.participants[agent_id].channel = channel
            collab.participants[agent_id].status = "active"
            self._save_state()
        
        self._emit_bus_event("a2a.heartbeat", {
            "codeword": codeword,
            "collab_id": collab.collab_id if collab else None,
            "agent": agent_id,
            "iteration": iteration,
            "channel": channel,
            "last_action": last_action[:200],
        })

    def check_liveness(self, codeword: str) -> Dict[str, Dict[str, Any]]:
        """
        Check liveness of all participants in a collaboration.
        
        Returns dict of {agent_id: {alive: bool, last_seen: iso, channel: str}}
        """
        collab = None
        for c in self.collaborations.values():
            if c.codeword == codeword:
                collab = c
                break
        
        if not collab:
            return {}
        
        now = time.time()
        result = {}
        
        for agent_id, participant in collab.participants.items():
            age = now - participant.last_heartbeat_ts
            alive = age < HEARTBEAT_TIMEOUT_S
            
            result[agent_id] = {
                "alive": alive,
                "status": participant.status,
                "last_heartbeat_ts": participant.last_heartbeat_ts,
                "last_heartbeat_iso": datetime.fromtimestamp(participant.last_heartbeat_ts).isoformat() if participant.last_heartbeat_ts else None,
                "age_s": int(age),
                "channel": participant.channel,
                "iteration": participant.iteration,
            }
            
            # Check for dissociation
            if not alive and participant.status == "active":
                participant.status = "dissociated"
                self._emit_bus_event("a2a.dissociation.detected", {
                    "codeword": codeword,
                    "collab_id": collab.collab_id,
                    "missing_agent": agent_id,
                    "last_heartbeat_iso": result[agent_id]["last_heartbeat_iso"],
                    "age_s": int(age),
                })
                self._save_state()
        
        return result

    def _check_bus_activity(self, agent_id: str, window_s: int = 300) -> bool:
        """Check if agent has recent bus activity."""
        if not self.bus_path.exists():
            return False
        if not _ndjson_read_allowed():
            return False  # Cannot read NDJSON in non-DR mode

        cutoff = time.time() - window_s
        try:
            for line in self.bus_path.read_text().strip().split("\n")[-500:]:
                if line:
                    event = json.loads(line)
                    if event.get("actor") == agent_id and event.get("ts", 0) > cutoff:
                        return True
        except (json.JSONDecodeError, IOError):
            pass
        return False

    def _check_hexis_activity(self, agent_id: str, window_s: int = 300) -> bool:
        """Check if agent has recent HEXIS activity."""
        hexis_inbox = HEXIS_DIR / agent_id / "inbox.ndjson"
        hexis_outbox = HEXIS_DIR / agent_id / "outbox.ndjson"
        
        cutoff = time.time() - window_s
        for path in [hexis_inbox, hexis_outbox]:
            if path.exists():
                try:
                    mtime = path.stat().st_mtime
                    if mtime > cutoff:
                        return True
                except OSError:
                    pass
        return False

    def _check_tmux_activity(self, agent_id: str, window_s: int = 300) -> bool:
        """Check if agent has recent tmux activity via session hash."""
        session_name = f"pluribus_agent_{agent_id}"
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", session_name, "-p", "-S", "-50"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # If there's content in the pane, consider it active
                # (A more sophisticated check would hash content over time)
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return False

    def recover_dissociated(self, codeword: str, agent_id: str) -> bool:
        """
        Attempt to recover a dissociated agent via multi-channel discovery.
        """
        # Check all channels
        if self._check_bus_activity(agent_id, 600):
            channel = "bus"
        elif self._check_hexis_activity(agent_id, 600):
            channel = "hexis"
        elif self._check_tmux_activity(agent_id, 600):
            channel = "tmux"
        else:
            return False
        
        # Found activity - recover
        collab = None
        for c in self.collaborations.values():
            if c.codeword == codeword:
                collab = c
                break
        
        if collab and agent_id in collab.participants:
            collab.participants[agent_id].status = "active"
            collab.participants[agent_id].last_heartbeat_ts = time.time()
            collab.participants[agent_id].channel = channel
            self._save_state()
            
            self._emit_bus_event("a2a.dissociation.recovered", {
                "codeword": codeword,
                "collab_id": collab.collab_id,
                "recovered_agent": agent_id,
                "channel": channel,
            })
            return True
        
        return False

    def status(self, codeword: str = None) -> Dict[str, Any]:
        """Get status of one or all collaborations."""
        if codeword:
            for collab in self.collaborations.values():
                if collab.codeword == codeword:
                    return {
                        "collab_id": collab.collab_id,
                        "codeword": collab.codeword,
                        "status": collab.status,
                        "initiator": collab.initiator,
                        "mode": collab.mode,
                        "iteration": collab.current_iteration,
                        "participants": {
                            pid: {
                                "status": p.status,
                                "iteration": p.iteration,
                                "channel": p.channel,
                            }
                            for pid, p in collab.participants.items()
                        }
                    }
            return {"error": f"Codeword not found: {codeword}"}
        
        return {
            "active_count": len([c for c in self.collaborations.values() if c.status == "active"]),
            "collaborations": [
                {
                    "codeword": c.codeword,
                    "status": c.status,
                    "participants": len(c.participants),
                }
                for c in self.collaborations.values()
            ]
        }


def cmd_watch(args):
    """Watch a codeword for activity."""
    monitor = A2AMonitor()
    print(f"Watching codeword: {args.codeword}")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            liveness = monitor.check_liveness(args.codeword)
            if liveness:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Liveness check:")
                for agent, data in liveness.items():
                    status = "✓" if data["alive"] else "✗"
                    print(f"  {status} {agent}: {data['status']} (iter {data['iteration']}, {data['age_s']}s ago)")
            else:
                print(f"No collaboration found for codeword: {args.codeword}")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopped")
    return 0


def cmd_status(args):
    """Show status."""
    monitor = A2AMonitor()
    status = monitor.status(args.codeword)
    print(json.dumps(status, indent=2))
    return 0


def cmd_recover(args):
    """Attempt recovery."""
    monitor = A2AMonitor()
    success = monitor.recover_dissociated(args.codeword, args.agent)
    if success:
        print(f"✓ Recovered {args.agent} for {args.codeword}")
    else:
        print(f"✗ Could not recover {args.agent}")
    return 0 if success else 1


def cmd_heartbeat(args):
    """Emit a heartbeat."""
    monitor = A2AMonitor()
    monitor.emit_heartbeat(
        codeword=args.codeword,
        agent_id=args.agent,
        iteration=args.iteration or 0,
        channel=args.channel or "bus",
        last_action=args.action or "",
    )
    print(f"Heartbeat emitted for {args.agent} on {args.codeword}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="A2A Collaboration Monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # watch
    p_watch = subparsers.add_parser("watch", help="Watch a codeword")
    p_watch.add_argument("codeword", help="Codeword to watch")
    
    # status
    p_status = subparsers.add_parser("status", help="Show status")
    p_status.add_argument("codeword", nargs="?", help="Codeword (optional)")
    
    # recover
    p_recover = subparsers.add_parser("recover", help="Attempt recovery")
    p_recover.add_argument("codeword", help="Codeword")
    p_recover.add_argument("agent", help="Agent ID to recover")
    
    # heartbeat
    p_hb = subparsers.add_parser("heartbeat", help="Emit heartbeat")
    p_hb.add_argument("codeword", help="Codeword")
    p_hb.add_argument("agent", help="Agent ID")
    p_hb.add_argument("--iteration", type=int, help="Iteration number")
    p_hb.add_argument("--channel", help="Channel (bus/hexis/tmux)")
    p_hb.add_argument("--action", help="Last action description")
    
    args = parser.parse_args()
    
    if args.command == "watch":
        return cmd_watch(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "recover":
        return cmd_recover(args)
    elif args.command == "heartbeat":
        return cmd_heartbeat(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
