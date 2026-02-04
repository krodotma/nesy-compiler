#!/usr/bin/env python3
"""
PBTEST semantic operator: wrapper for rigorous neurosymbolic TDD verification.

Broadcasting this intent signals that an agent is about to perform a Reality Check.
It enforces the grammar of Scope -> Mode -> Browser.

Usage:
  python3 nucleus/tools/pbtest_operator.py --scope nucleus/dashboard --mode live --intent "Verify MemIngestBar render"
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Try to import agent_bus for partitioned logging
try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None

# Try to import paip_isolation
try:
    import paip_isolation
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    try:
        import paip_isolation
    except ImportError:
        paip_isolation = None

# Bus configuration
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
ACTOR = os.environ.get("PLURIBUS_ACTOR", "pbtest-operator")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbtest_operator.py", description="PBTEST: Neurosymbolic TDD Verification Operator")
    p.add_argument("--scope", required=True, help="Target file/module/feature")
    p.add_argument("--mode", choices=["unit", "live", "soak", "full"], required=True, help="Verification rigor level")
    p.add_argument("--browser", choices=["chromium", "webkit", "firefox", "laser", "none"], default="chromium", help="Browser engine for live checks")
    p.add_argument("--intent", required=True, help="Description of the test goal")
    p.add_argument("--bus-dir", default=str(BUS_DIR), help="Override bus directory")
    return p

def emit_bus(bus_dir: Path, topic: str, kind: str, data: dict):
    if agent_bus is not None:
        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level="info",
            actor=ACTOR,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
        return

    if not bus_dir.exists():
        return

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": "info",
        "actor": ACTOR,
        "data": data
    }

    try:
        events_path = bus_dir / "events.ndjson"
        with open(events_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        sys.stderr.write(f"Failed to emit bus event: {e}\n")

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    bus_dir = Path(args.bus_dir)
    
    # 1. Emit Request
    payload = {
        "scope": args.scope,
        "mode": args.mode,
        "browser": args.browser,
        "intent": args.intent,
        "phase": "init"
    }
    
    print(f"PBTEST: Initiating {args.mode.upper()} check on {args.scope} via {args.browser}...")
    emit_bus(bus_dir, "operator.pbtest.request", "request", payload)
    
    # 2. Output Instructions for the Agent (Stdout is the prompt)
    print("\n--- PBTEST INSTRUCTIONS ---")
    print(f"1. You have declared intent to verify: '{args.intent}'")
    if args.mode in ["live", "full"]:
        print("2. REQUIREMENT: You MUST execute a live browser verification (Playwright/Laser).")
        print("3. REQUIREMENT: You MUST check 'telemetry.client.error' in the bus for silent failures.")

        # PAIP Isolation Logic
        if paip_isolation:
            iso_config = paip_isolation.get_isolated_config()
            print("\n--- PAIP v13 ISOLATION (MANDATORY) ---")
            print(f"# Use these exports to prevent port/display collisions (Slot {iso_config['PAIP_SLOT']}):")
            for k, v in iso_config.items():
                print(f"export {k}={v}")
            print("--------------------------------------")

    print("4. If the test harness fails, fix the harness. Do not skip.")
    print("---------------------------\n")

if __name__ == "__main__":
    main()
