#!/usr/bin/env python3
"""
dialogos_simple_worker.py - Minimal agent worker using Dialogos protocol
Demonstrates proper LASER/LENS/Dialogos usage
"""
import argparse
import json
import time
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cagent-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Agent: {args.cagent_id}")
    print(f"Task: {args.task}")
    print(f"Protocol: Dialogos (LASER/LENS routing)")
    print(f"{'='*60}\n")
    
    # Direct NDJSON bus write (simplified for demo)
    bus_path = Path("/pluribus/.pluribus/bus/events.ndjson")
    
    for i in range(args.iterations):
        req_id = f"{args.cagent_id}_{int(time.time())}_{i}"
        
        # Submit LLM request via Dialogos
        print(f"[{i+1}/{args.iterations}] Submitting dialogos request: {req_id}")
        
        # Create bus event
        import uuid
        from datetime import datetime
        
        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.utcnow().isoformat() + "Z",
            "topic": f"dialogos.submit.{args.cagent_id}",
            "kind": "event",
            "level": "info",
            "actor": args.cagent_id,
            "data": {
                "req_id": req_id,
                "prompt": args.task,
                "provider_affinity": ["claude-web", "gemini-web"],  # LASER web sessions
                "max_tokens": 2000,
            }
        }
        
        # Append to bus
        with open(bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        
        # In a real implementation, we'd listen for dialogos.cell.{cagent_id}
        # For demo purposes, just show the pattern
        print(f"  ✅ Request emitted (ID: {event['id'][:8]}...)")
        print(f"  🔍 LASER/LENS will route to web sessions")
        print(f"  📡 Monitoring for dialogos.cell.{args.cagent_id}...")
        
        time.sleep(5)  # Simulate work
    
    print(f"\n{'='*60}")
    print(f"✅ Worker complete: {args.cagent_id}")
    print(f"⚠️  NOTE: This is a demo pattern")
    print(f"   Real implementation would listen for dialogos.cell responses")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
