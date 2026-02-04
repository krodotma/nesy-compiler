#!/usr/bin/env python3
"""
rep_protocol.py - REP Coordination Protocol

DUALITY-BIND E5: Each agent broadcasts compact coordination state.

Ring: 1 (Operator)
Protocol: DKIN v29
"""

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import os

BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))
REP_DIR = Path(os.environ.get("PLURIBUS_REP_DIR", ".pluribus/rep"))


@dataclass
class CoordinationState:
    """Compact coordination state for an agent."""
    agent_id: str
    goal_sketch: str  # Brief description of current goal
    plan_synopsis: str  # Brief plan description
    sensitivities: Dict[str, float] = field(default_factory=dict)  # Variable sensitivities
    resource_hints: Dict[str, Any] = field(default_factory=dict)  # Resource requirements
    timestamp: float = field(default_factory=time.time)
    ttl_s: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl_s


class REPProtocol:
    """REP: Replicated Execution Protocol for coordination."""
    
    def __init__(self, rep_dir: Path = None):
        self.rep_dir = rep_dir or REP_DIR
        self.rep_dir.mkdir(parents=True, exist_ok=True)
        self.states: Dict[str, CoordinationState] = {}
    
    def broadcast(self, state: CoordinationState) -> str:
        """Broadcast coordination state."""
        self.states[state.agent_id] = state
        
        # Persist
        path = self.rep_dir / f"{state.agent_id}.json"
        with open(path, "w") as f:
            json.dump(state.to_dict(), f)
        
        return state.agent_id
    
    def receive(self, agent_id: str) -> Optional[CoordinationState]:
        """Receive coordination state for an agent."""
        path = self.rep_dir / f"{agent_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            state = CoordinationState(**data)
            if not state.is_expired():
                return state
        return None
    
    def get_all_states(self) -> List[CoordinationState]:
        """Get all non-expired coordination states."""
        states = []
        for path in self.rep_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                state = CoordinationState(**data)
                if not state.is_expired():
                    states.append(state)
            except:
                continue
        return states
    
    def compute_consensus(self) -> Dict[str, float]:
        """Compute consensus across all states."""
        states = self.get_all_states()
        if not states:
            return {}
        
        # Aggregate sensitivities
        all_sensitivities = {}
        for state in states:
            for k, v in state.sensitivities.items():
                if k not in all_sensitivities:
                    all_sensitivities[k] = []
                all_sensitivities[k].append(v)
        
        # Return mean sensitivities
        return {k: sum(v) / len(v) for k, v in all_sensitivities.items()}


def main():
    parser = argparse.ArgumentParser(description="REP Coordination Protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_broadcast = subparsers.add_parser("broadcast", help="Broadcast state")
    p_broadcast.add_argument("--agent", required=True)
    p_broadcast.add_argument("--goal", required=True)
    p_broadcast.add_argument("--plan", default="")
    
    p_receive = subparsers.add_parser("receive", help="Receive state")
    p_receive.add_argument("agent_id")
    
    subparsers.add_parser("consensus", help="Compute consensus")
    subparsers.add_parser("list", help="List all states")
    
    args = parser.parse_args()
    rep = REPProtocol()
    
    if args.command == "broadcast":
        state = CoordinationState(
            agent_id=args.agent,
            goal_sketch=args.goal,
            plan_synopsis=args.plan,
        )
        rep.broadcast(state)
        print(f"✅ Broadcast state for {args.agent}")
        return 0
    elif args.command == "receive":
        state = rep.receive(args.agent_id)
        if state:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            print(f"No state for {args.agent_id}")
        return 0
    elif args.command == "consensus":
        consensus = rep.compute_consensus()
        print(f"Consensus: {json.dumps(consensus, indent=2)}")
        return 0
    elif args.command == "list":
        states = rep.get_all_states()
        for s in states:
            print(f"  {s.agent_id}: {s.goal_sketch}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
