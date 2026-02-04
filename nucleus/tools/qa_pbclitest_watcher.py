#!/usr/bin/env python3
"""
QA PBCLITEST Watcher
Monitors the bus for 'operator.pbclitest.request' and related results.
"""

import json
import time
from pathlib import Path

BUS_DIR = Path("/pluribus/.pluribus/bus")
EVENTS_FILE = BUS_DIR / "events.ndjson"


def tail_bus() -> None:
    if not EVENTS_FILE.exists():
        print("Waiting for bus...")
        return

    with open(EVENTS_FILE, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            process_event(event)


def process_event(event: dict) -> None:
    topic = event.get("topic", "")
    data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

    if topic == "operator.pbclitest.request":
        print(f"\n[QA] PBCLITEST REQUEST: {data.get('intent')} (Mode: {data.get('mode')})")
        print(f"     Scope: {data.get('scope')} | Command: {data.get('command')}")
        print("     -> Awaiting pbclitest result/verdict...")
    elif topic == "operator.pbclitest.result":
        print(f"[QA] PBCLITEST RESULT: {data.get('status')} {data.get('details', '')}")
    elif topic == "operator.pbclitest.verdict":
        print(f"[QA] PBCLITEST VERDICT: {data.get('status')}")


if __name__ == "__main__":
    print("QA PBCLITEST Watcher Active...")
    tail_bus()
