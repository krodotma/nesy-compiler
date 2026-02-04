#!/usr/bin/env python3
"""
PBCLITEST semantic operator: CLI verification for tools and scripts.

Usage:
  python3 nucleus/tools/pbclitest_operator.py --scope nucleus/tools/ohm.py --mode full --intent "Verify OHM CLI output"
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
ACTOR = os.environ.get("PLURIBUS_ACTOR", "pbclitest-operator")

try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbclitest_operator.py", description="PBCLITEST: CLI Verification Operator")
    p.add_argument("--scope", required=True, help="Target file/module/tool")
    p.add_argument(
        "--mode",
        choices=["unit", "integration", "e2e", "live", "full"],
        required=True,
        help="Verification rigor: unit|integration|e2e|live|full",
    )
    p.add_argument("--intent", required=True, help="Description of test goal")
    p.add_argument("--command", help="Command or harness invocation to run")
    p.add_argument("--bus-dir", default=str(BUS_DIR), help="Override bus directory")
    return p


def emit_bus(bus_dir: Path, topic: str, kind: str, data: dict) -> None:
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
        "data": data,
    }
    try:
        events_path = bus_dir / "events.ndjson"
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        sys.stderr.write(f"Failed to emit bus event: {e}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    bus_dir = Path(args.bus_dir)

    payload = {
        "scope": args.scope,
        "mode": args.mode,
        "intent": args.intent,
        "command": args.command,
        "phase": "init",
    }

    print(f"PBCLITEST: Initiating {args.mode.upper()} check on {args.scope}...")
    emit_bus(bus_dir, "operator.pbclitest.request", "request", payload)

    print("\n--- PBCLITEST INSTRUCTIONS ---")
    print(f"1. You have declared intent to verify: '{args.intent}'")
    if args.command:
        print(f"2. Run the declared CLI command: {args.command}")
    else:
        print("2. Run the appropriate CLI/unit/e2e harness for this scope.")
    print("3. Capture pass/fail evidence and emit pbclitest.result/verdict on the bus.")
    print("4. If the harness fails, fix the harness. Do not skip.")
    print("------------------------------\n")


if __name__ == "__main__":
    main()
