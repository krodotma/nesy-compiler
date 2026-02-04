#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Dict, Any

# Minimal TBTSO A2A restoration

class A2AManager:
    def __init__(self):
        self.state_file = "/pluribus/.pluribus/a2a_state.json"
        
    def load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_file):
            return {"swarms": {}}
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except:
            return {"swarms": {}}

    def save_state(self, state: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_status(self, swarm_id: str = None):
        state = self.load_state()
        if swarm_id:
            return state["swarms"].get(swarm_id, {})
        return state

    def update_swarm(self, swarm_id: str, data: Dict[str, Any]):
        state = self.load_state()
        if swarm_id not in state["swarms"]:
            state["swarms"][swarm_id] = {"id": swarm_id, "status": "initialized", "agents": []}
        
        state["swarms"][swarm_id].update(data)
        state["swarms"][swarm_id]["last_updated"] = time.time()
        self.save_state(state)

def main():
    parser = argparse.ArgumentParser(description="TBTSO A2A Manager")
    subparsers = parser.add_subparsers(dest="command")
    
    # Status
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--swarm")
    status_parser.add_argument("--json", action="store_true")
    
    # Update
    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--swarm", required=True)
    update_parser.add_argument("--status")
    
    args = parser.parse_args()
    mgr = A2AManager()
    
    if args.command == "status":
        status = mgr.get_status(args.swarm)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"A2A Status: {len(status.get('swarms', []))} active swarms")
            
    elif args.command == "update":
        mgr.update_swarm(args.swarm, {"status": args.status})
        print(f"Updated swarm {args.swarm} to {args.status}")

if __name__ == "__main__":
    main()
