#!/usr/bin/env python3
"""
tbtso_a2a.py - PBTSO Agent-to-Agent Coordination Module (legacy filename)

Implements PBTSO Steps 31-40: A2A Coordination Foundation

Provides:
- Integration with existing a2a_monitor.py and a2a_bridge.py
- Swarm-level A2A coordination with codeword tracking
- Dialogos lane progress emission
- Heartbeat automation for swarm agents

Architecture:
- Wraps A2AMonitor for PBTSO swarm context
- Emits dialogos.lanes.progress events
- Provides CLI for swarm coordination operations
- Integrates with tmux_swarm_orchestrator.py

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2

Bus Topics:
- tbtso.a2a.swarm.init
- tbtso.a2a.swarm.heartbeat
- tbtso.a2a.swarm.complete
- tbtso.a2a.lane.progress
- pbtso.a2a.swarm.init (canonical mirror)
- pbtso.a2a.swarm.heartbeat (canonical mirror)
- pbtso.a2a.swarm.complete (canonical mirror)
- pbtso.a2a.lane.progress (canonical mirror)
- a2a.handshake.propose (via a2a_monitor)
- a2a.heartbeat (via a2a_monitor)
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Protocol constants
A2A_HEARTBEAT_INTERVAL_S = 300  # 5 minutes
A2A_HEARTBEAT_TIMEOUT_S = 900   # 15 minutes (3 missed heartbeats)
A2A_HANDSHAKE_TIMEOUT_S = 60


class A2AMode:
    """A2A collaboration mode constants."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    SYNC_POINTS = "sync_points"
    TURN_BASED = "turn_based"


class SwarmState:
    """Swarm state constants."""
    INITIALIZING = "initializing"
    HANDSHAKING = "handshaking"
    RUNNING = "running"
    COMPLETING = "completing"
    COMPLETE = "complete"
    FAILED = "failed"


def _get_pluribus_root() -> Path:
    return Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))


def _get_bus_path() -> Path:
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(_get_pluribus_root() / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _get_lanes_path() -> Path:
    return _get_pluribus_root() / "nucleus" / "state" / "lanes.json"

def _mirror_topic(topic: str) -> Optional[str]:
    if topic.startswith("tbtso."):
        return "pbtso." + topic[len("tbtso."):]
    if topic.startswith("pbtso."):
        return "tbtso." + topic[len("pbtso."):]
    return None


@dataclass
class SwarmParticipant:
    """A participant in a PBTSO swarm."""
    agent_id: str
    role: str = "worker"
    status: str = "pending"  # pending, active, dissociated, complete
    last_heartbeat_ts: float = 0.0
    iteration: int = 0
    channel: str = "bus"
    runner: str = "claude"  # claude, codex, gemini, etc.

    def is_alive(self, timeout_s: float = A2A_HEARTBEAT_TIMEOUT_S) -> bool:
        """Check if participant is alive based on heartbeat timeout."""
        return (time.time() - self.last_heartbeat_ts) < timeout_s


@dataclass
class SwarmCoordination:
    """PBTSO swarm coordination state."""
    swarm_id: str
    codeword: str
    collab_id: str
    initiator: str
    participants: Dict[str, SwarmParticipant] = field(default_factory=dict)
    state: str = SwarmState.INITIALIZING
    mode: str = A2AMode.PARALLEL
    lane_id: Optional[str] = None
    created_ts: float = field(default_factory=time.time)
    iterations_planned: int = 10
    current_iteration: int = 0
    ttl_s: int = 3600
    scope: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "swarm_id": self.swarm_id,
            "codeword": self.codeword,
            "collab_id": self.collab_id,
            "initiator": self.initiator,
            "participants": {k: asdict(v) for k, v in self.participants.items()},
            "state": self.state,
            "mode": self.mode,
            "lane_id": self.lane_id,
            "created_ts": self.created_ts,
            "iterations_planned": self.iterations_planned,
            "current_iteration": self.current_iteration,
            "ttl_s": self.ttl_s,
            "scope": self.scope,
        }


class TBTSO_A2A:
    """
    PBTSO Agent-to-Agent Coordination Manager.

    Integrates with:
        - a2a_monitor.py: Collaboration state and heartbeat tracking
        - a2a_bridge.py: Bus event forwarding
        - tmux_swarm_orchestrator.py: Swarm spawning
        - lanes.json: Dialogos lane progress
    """

    def __init__(self, bus_path: Optional[Path] = None, lanes_path: Optional[Path] = None):
        self.bus_path = bus_path or _get_bus_path()
        self.lanes_path = lanes_path or _get_lanes_path()
        self.actor = os.environ.get("PLURIBUS_ACTOR", "tbtso_a2a")
        self.swarms: Dict[str, SwarmCoordination] = {}
        self._last_heartbeat_ts: Dict[str, float] = {}
        self._load_existing_swarms()

    def _load_existing_swarms(self) -> None:
        """Load existing swarm state from state directory."""
        state_dir = _get_pluribus_root() / ".pluribus" / "tbtso" / "swarms"
        if not state_dir.exists():
            return
        for state_file in state_dir.glob("*.json"):
            try:
                with state_file.open("r") as f:
                    data = json.load(f)
                    swarm_id = data.get("swarm_id")
                    if swarm_id:
                        participants = {}
                        for pid, pdata in data.get("participants", {}).items():
                            participants[pid] = SwarmParticipant(**pdata)
                        data["participants"] = participants
                        self.swarms[swarm_id] = SwarmCoordination(**data)
            except Exception:
                pass

    def _save_swarm_state(self, swarm: SwarmCoordination) -> None:
        """Save swarm state to disk."""
        state_dir = _get_pluribus_root() / ".pluribus" / "tbtso" / "swarms"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / f"{swarm.swarm_id}.json"
        with state_file.open("w") as f:
            json.dump(swarm.to_dict(), f, indent=2)

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to the Pluribus bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.actor,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        mirror_topic = _mirror_topic(topic)
        if mirror_topic:
            mirror_event = dict(event)
            mirror_event["id"] = str(uuid.uuid4())
            mirror_event["topic"] = mirror_topic
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(mirror_event) + "\n")

        return event_id

    def init_swarm(
        self,
        agent_ids: List[str],
        scope: str,
        lane_id: Optional[str] = None,
        mode: str = A2AMode.PARALLEL,
        iterations: int = 10,
        ttl_s: int = 3600,
    ) -> SwarmCoordination:
        """
        Initialize a new PBTSO swarm with A2A coordination.

        Steps 31-33: Initialize swarm, generate codeword, create coordination state.
        """
        swarm_id = f"swarm-{uuid.uuid4().hex[:8]}"
        codeword = f"tbtso-{swarm_id[-8:]}-{int(time.time()) % 10000}"
        collab_id = f"collab-{uuid.uuid4().hex[:8]}"

        # Create participants
        participants = {}
        for agent_id in agent_ids:
            participants[agent_id] = SwarmParticipant(
                agent_id=agent_id,
                status="pending",
            )

        swarm = SwarmCoordination(
            swarm_id=swarm_id,
            codeword=codeword,
            collab_id=collab_id,
            initiator=self.actor,
            participants=participants,
            state=SwarmState.INITIALIZING,
            mode=mode,
            lane_id=lane_id,
            iterations_planned=iterations,
            ttl_s=ttl_s,
            scope=scope,
        )

        self.swarms[swarm_id] = swarm
        self._save_swarm_state(swarm)

        # Emit init event
        self._emit_bus_event("tbtso.a2a.swarm.init", {
            "swarm_id": swarm_id,
            "codeword": codeword,
            "collab_id": collab_id,
            "agents": agent_ids,
            "scope": scope,
            "mode": mode,
            "iterations": iterations,
            "lane_id": lane_id,
        })

        return swarm

    def propose_handshake(self, swarm: SwarmCoordination) -> str:
        """
        Propose A2A handshake for swarm participants.

        Step 34: Emit a2a.handshake.propose before spawning.
        """
        swarm.state = SwarmState.HANDSHAKING
        self._save_swarm_state(swarm)

        # Emit handshake propose (mirrors a2a_monitor.py format)
        event_id = self._emit_bus_event("a2a.handshake.propose", {
            "codeword": swarm.codeword,
            "collab_id": swarm.collab_id,
            "initiator": swarm.initiator,
            "target_agents": list(swarm.participants.keys()),
            "scope": swarm.scope,
            "iterations_planned": swarm.iterations_planned,
            "mode": swarm.mode,
            "ttl_s": swarm.ttl_s,
            "swarm_id": swarm.swarm_id,
        })

        return event_id

    def acknowledge_participant(
        self,
        swarm_id: str,
        agent_id: str,
        accepted: bool = True,
        channel: str = "bus"
    ) -> bool:
        """
        Acknowledge a participant joining the swarm.

        Step 35: Handle handshake acknowledgments.
        """
        swarm = self.swarms.get(swarm_id)
        if not swarm or agent_id not in swarm.participants:
            return False

        participant = swarm.participants[agent_id]
        participant.status = "active" if accepted else "declined"
        participant.last_heartbeat_ts = time.time()
        participant.channel = channel

        self._save_swarm_state(swarm)

        # Check if all participants acknowledged
        active_count = sum(1 for p in swarm.participants.values() if p.status == "active")
        if active_count == len(swarm.participants):
            swarm.state = SwarmState.RUNNING
            self._save_swarm_state(swarm)

        # Emit ack event
        self._emit_bus_event("a2a.handshake.ack", {
            "swarm_id": swarm_id,
            "codeword": swarm.codeword,
            "collab_id": swarm.collab_id,
            "agent": agent_id,
            "status": "accepted" if accepted else "declined",
            "channel": channel,
        })

        return True

    def emit_heartbeat(
        self,
        swarm_id: str,
        agent_id: str,
        iteration: int = 0,
        last_action: str = "",
    ) -> bool:
        """
        Emit heartbeat for swarm participant.

        Step 36: Heartbeat emission with rate limiting.
        """
        swarm = self.swarms.get(swarm_id)
        if not swarm or agent_id not in swarm.participants:
            return False

        # Rate limit heartbeats
        cache_key = f"{swarm_id}:{agent_id}"
        last_hb = self._last_heartbeat_ts.get(cache_key, 0)
        if time.time() - last_hb < A2A_HEARTBEAT_INTERVAL_S - 30:
            return True  # Already sent recently

        participant = swarm.participants[agent_id]
        participant.last_heartbeat_ts = time.time()
        participant.iteration = iteration
        participant.status = "active"

        self._last_heartbeat_ts[cache_key] = time.time()
        self._save_swarm_state(swarm)

        # Emit heartbeat
        self._emit_bus_event("tbtso.a2a.swarm.heartbeat", {
            "swarm_id": swarm_id,
            "codeword": swarm.codeword,
            "collab_id": swarm.collab_id,
            "agent": agent_id,
            "iteration": iteration,
            "last_action": last_action[:200],
        })

        # Also emit standard a2a.heartbeat
        self._emit_bus_event("a2a.heartbeat", {
            "codeword": swarm.codeword,
            "collab_id": swarm.collab_id,
            "agent": agent_id,
            "iteration": iteration,
            "channel": participant.channel,
        })

        return True

    def check_liveness(self, swarm_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Check liveness of all swarm participants.

        Step 37: Dissociation detection.
        """
        swarm = self.swarms.get(swarm_id)
        if not swarm:
            return {}

        result = {}
        now = time.time()

        for agent_id, participant in swarm.participants.items():
            age = now - participant.last_heartbeat_ts
            alive = participant.is_alive()

            result[agent_id] = {
                "alive": alive,
                "status": participant.status,
                "last_heartbeat_ts": participant.last_heartbeat_ts,
                "age_s": int(age),
                "iteration": participant.iteration,
                "channel": participant.channel,
            }

            # Detect dissociation
            if not alive and participant.status == "active":
                participant.status = "dissociated"
                self._save_swarm_state(swarm)

                self._emit_bus_event("a2a.dissociation.detected", {
                    "swarm_id": swarm_id,
                    "codeword": swarm.codeword,
                    "collab_id": swarm.collab_id,
                    "agent": agent_id,
                    "age_s": int(age),
                }, level="warn")

        return result

    def emit_lane_progress(
        self,
        swarm_id: str,
        wip_pct: int,
        note: str = "",
    ) -> bool:
        """
        Emit lane progress for swarm.

        Step 38: Dialogos lane integration.
        """
        swarm = self.swarms.get(swarm_id)
        if not swarm or not swarm.lane_id:
            return False

        # Update lanes.json if exists
        if self.lanes_path.exists():
            try:
                with self.lanes_path.open("r") as f:
                    lanes_data = json.load(f)

                for lane in lanes_data.get("lanes", []):
                    if lane.get("id") == swarm.lane_id:
                        lane["wip_pct"] = wip_pct
                        lane.setdefault("history", []).append({
                            "ts": datetime.now(timezone.utc).isoformat() + "Z",
                            "wip_pct": wip_pct,
                            "note": note or f"Swarm {swarm.swarm_id} progress update",
                        })
                        break

                lanes_data["updated"] = datetime.now(timezone.utc).isoformat() + "Z"

                with self.lanes_path.open("w") as f:
                    json.dump(lanes_data, f, indent=2)
            except Exception:
                pass

        # Emit progress event
        self._emit_bus_event("tbtso.a2a.lane.progress", {
            "swarm_id": swarm_id,
            "lane_id": swarm.lane_id,
            "wip_pct": wip_pct,
            "note": note,
            "codeword": swarm.codeword,
        }, kind="metric")

        return True

    def complete_swarm(
        self,
        swarm_id: str,
        success: bool = True,
        summary: str = "",
    ) -> bool:
        """
        Complete swarm coordination.

        Step 39-40: Completion with evidence emission.
        """
        swarm = self.swarms.get(swarm_id)
        if not swarm:
            return False

        swarm.state = SwarmState.COMPLETE if success else SwarmState.FAILED
        for participant in swarm.participants.values():
            if participant.status == "active":
                participant.status = "complete"

        self._save_swarm_state(swarm)

        # Emit completion event
        self._emit_bus_event("tbtso.a2a.swarm.complete", {
            "swarm_id": swarm_id,
            "codeword": swarm.codeword,
            "collab_id": swarm.collab_id,
            "success": success,
            "summary": summary,
            "participants": list(swarm.participants.keys()),
            "iterations_completed": swarm.current_iteration,
            "lane_id": swarm.lane_id,
        })

        # Also emit a2a.collab.complete
        self._emit_bus_event("a2a.collab.complete", {
            "codeword": swarm.codeword,
            "collab_id": swarm.collab_id,
            "initiator": swarm.initiator,
            "participants": list(swarm.participants.keys()),
            "summary": summary,
            "success": success,
        })

        return True

    def status(self, swarm_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of one or all swarms."""
        if swarm_id:
            swarm = self.swarms.get(swarm_id)
            if not swarm:
                return {"error": f"Swarm not found: {swarm_id}"}
            return swarm.to_dict()

        return {
            "active_swarms": len([s for s in self.swarms.values() if s.state == SwarmState.RUNNING]),
            "swarms": [
                {
                    "swarm_id": s.swarm_id,
                    "codeword": s.codeword,
                    "state": s.state,
                    "participants": len(s.participants),
                    "lane_id": s.lane_id,
                }
                for s in self.swarms.values()
            ]
        }


def main() -> int:
    """CLI entry point for tbtso_a2a."""
    import argparse

    parser = argparse.ArgumentParser(description="PBTSO A2A Coordination (Steps 31-40)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize new swarm")
    init_parser.add_argument("--agents", required=True, help="Comma-separated agent IDs")
    init_parser.add_argument("--scope", required=True, help="Scope of swarm task")
    init_parser.add_argument("--lane", help="Lane ID for progress tracking")
    init_parser.add_argument("--mode", default="parallel", choices=["parallel", "sequential", "turn_based"])
    init_parser.add_argument("--iterations", type=int, default=10, help="Planned iterations")
    init_parser.add_argument("--json", action="store_true", help="JSON output")

    # status command
    status_parser = subparsers.add_parser("status", help="Show swarm status")
    status_parser.add_argument("swarm_id", nargs="?", help="Swarm ID (optional)")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # heartbeat command
    hb_parser = subparsers.add_parser("heartbeat", help="Emit heartbeat")
    hb_parser.add_argument("swarm_id", help="Swarm ID")
    hb_parser.add_argument("agent_id", help="Agent ID")
    hb_parser.add_argument("--iteration", type=int, default=0, help="Iteration number")
    hb_parser.add_argument("--action", default="", help="Last action description")

    # liveness command
    liveness_parser = subparsers.add_parser("liveness", help="Check liveness")
    liveness_parser.add_argument("swarm_id", help="Swarm ID")
    liveness_parser.add_argument("--json", action="store_true", help="JSON output")

    # progress command
    progress_parser = subparsers.add_parser("progress", help="Emit lane progress")
    progress_parser.add_argument("swarm_id", help="Swarm ID")
    progress_parser.add_argument("wip_pct", type=int, help="WIP percentage (0-100)")
    progress_parser.add_argument("--note", default="", help="Progress note")

    # complete command
    complete_parser = subparsers.add_parser("complete", help="Complete swarm")
    complete_parser.add_argument("swarm_id", help="Swarm ID")
    complete_parser.add_argument("--success", action="store_true", default=True, help="Success status")
    complete_parser.add_argument("--failed", action="store_true", help="Mark as failed")
    complete_parser.add_argument("--summary", default="", help="Completion summary")

    args = parser.parse_args()

    coordinator = TBTSO_A2A()

    if args.command == "init":
        agents = [a.strip() for a in args.agents.split(",") if a.strip()]
        swarm = coordinator.init_swarm(
            agent_ids=agents,
            scope=args.scope,
            lane_id=args.lane,
            mode=args.mode,
            iterations=args.iterations,
        )
        coordinator.propose_handshake(swarm)

        if args.json:
            print(json.dumps(swarm.to_dict(), indent=2))
        else:
            print(f"Swarm initialized: {swarm.swarm_id}")
            print(f"  Codeword: {swarm.codeword}")
            print(f"  Participants: {list(swarm.participants.keys())}")
            print(f"  State: {swarm.state}")
        return 0

    elif args.command == "status":
        status = coordinator.status(args.swarm_id)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            if "error" in status:
                print(status["error"])
                return 1
            if args.swarm_id:
                print(f"Swarm: {status['swarm_id']}")
                print(f"  State: {status['state']}")
                print(f"  Codeword: {status['codeword']}")
                print(f"  Participants: {len(status['participants'])}")
            else:
                print(f"Active swarms: {status['active_swarms']}")
                for s in status["swarms"]:
                    print(f"  {s['swarm_id']}: {s['state']} ({s['participants']} agents)")
        return 0

    elif args.command == "heartbeat":
        success = coordinator.emit_heartbeat(
            swarm_id=args.swarm_id,
            agent_id=args.agent_id,
            iteration=args.iteration,
            last_action=args.action,
        )
        if success:
            print(f"Heartbeat emitted for {args.agent_id}")
        else:
            print(f"Failed to emit heartbeat")
            return 1
        return 0

    elif args.command == "liveness":
        liveness = coordinator.check_liveness(args.swarm_id)
        if args.json:
            print(json.dumps(liveness, indent=2))
        else:
            if not liveness:
                print(f"No liveness data for swarm: {args.swarm_id}")
                return 1
            for agent, data in liveness.items():
                status_icon = "V" if data["alive"] else "X"
                print(f"  {status_icon} {agent}: {data['status']} (age: {data['age_s']}s)")
        return 0

    elif args.command == "progress":
        success = coordinator.emit_lane_progress(
            swarm_id=args.swarm_id,
            wip_pct=args.wip_pct,
            note=args.note,
        )
        if success:
            print(f"Progress updated: {args.wip_pct}%")
        else:
            print("Failed to update progress (no lane_id?)")
            return 1
        return 0

    elif args.command == "complete":
        success = coordinator.complete_swarm(
            swarm_id=args.swarm_id,
            success=not args.failed,
            summary=args.summary,
        )
        if success:
            print(f"Swarm completed: {args.swarm_id}")
        else:
            print(f"Failed to complete swarm")
            return 1
        return 0

    return 1


# Canonical alias (PBTSO) for legacy TBTSO naming.
PBTSO_A2A = TBTSO_A2A


if __name__ == "__main__":
    sys.exit(main())
