#!/usr/bin/env python3
"""
SKY Signal Tool: The Network of the Cell
========================================

Implements the "Sky" signaling protocol over the Pluribus Bus.
Allows agents to discover peers, exchange capabilities, and signal
intent using the `pluribus.sky.v1` schema semantics (mapped to JSON).

Usage:
    python3 sky_signal.py hello --swarm default --label "gemini-cli"
    python3 sky_signal.py listen --swarm default
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True

# Constants from constants.ts / sky.proto
SKY_MAGIC_V1 = 0x534B5931
SKY_VERSION_V1 = 1

def now_ts_ms() -> int:
    return int(time.time() * 1000)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def get_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "sky-signal"

def emit_bus(bus_dir: Path, topic: str, data: dict, actor: str) -> None:
    """Emit event to the bus."""
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": "signal",  # Distinct from 'metric' or 'artifact'
        "level": "info",
        "actor": actor,
        "data": data,
    }
    
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def make_envelope(
    body_type: str,
    body_payload: dict,
    swarm_id: str,
    target_peer_id: str = "",
    trace_id: str = "",
) -> dict:
    """Create a JSON representation of a SkyEnvelope."""
    return {
        "magic": SKY_MAGIC_V1,
        "version": SKY_VERSION_V1,
        "ts_ms": now_ts_ms(),
        "trace_id": trace_id or str(uuid.uuid4()),
        "swarm_id": swarm_id,
        "source_peer_id": get_actor(),
        "target_peer_id": target_peer_id,
        "body_type": body_type,
        body_type: body_payload
    }

def cmd_hello(args):
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    
    payload = {
        "capabilities": ["ws+pb", "bus+json"],
        "preferred_transport": "bus",
        "label": args.label or get_actor(),
    }
    
    envelope = make_envelope("hello", payload, args.swarm)
    
    # Emit to the bus as a sky.signal event
    emit_bus(bus_dir, "sky.signal", envelope, get_actor())
    print(json.dumps({"status": "sent", "type": "hello", "envelope": envelope}, indent=2))

def cmd_listen(args):
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_path = bus_dir / "events.ndjson"
    
    print(f"Listening for SKY signals on swarm '{args.swarm}'... (Ctrl+C to stop)")
    
    # Simple tail
    try:
        if not events_path.exists():
            print("Bus not found.")
            return

        with events_path.open("r", encoding="utf-8") as f:
            # Seek to end
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                try:
                    event = json.loads(line)
                    if event.get("topic") == "sky.signal":
                        data = event.get("data", {})
                        if data.get("swarm_id") == args.swarm:
                            # Filter my own messages? Optional.
                            print(json.dumps(event, indent=2))
                            if args.once:
                                return
                except:
                    continue
    except KeyboardInterrupt:
        print("\nStopped.")

def main():
    parser = argparse.ArgumentParser(description="SKY Signaling Tool")
    parser.add_argument("--bus-dir", help="Bus directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Hello
    hello_p = subparsers.add_parser("hello", help="Broadcast Hello")
    hello_p.add_argument("--swarm", default="default", help="Swarm ID")
    hello_p.add_argument("--label", help="Peer label")
    
    # Listen
    listen_p = subparsers.add_parser("listen", help="Listen for signals")
    listen_p.add_argument("--swarm", default="default", help="Swarm ID")
    listen_p.add_argument("--once", action="store_true", help="Exit after first message")
    
    args = parser.parse_args()
    
    if args.command == "hello":
        cmd_hello(args)
    elif args.command == "listen":
        cmd_listen(args)

if __name__ == "__main__":
    main()
