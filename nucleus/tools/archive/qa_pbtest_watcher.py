#!/usr/bin/env python3
"""
QA PBTEST Watcher
Monitors the bus for 'operator.pbtest.request' and correlates with 'telemetry.client.*'.
"""

import json
import time
import sys
from pathlib import Path

BUS_DIR = Path("/pluribus/.pluribus/bus")
EVENTS_FILE = BUS_DIR / "events.ndjson"

def tail_bus():
    if not EVENTS_FILE.exists():
        print("Waiting for bus...")
        return

    with open(EVENTS_FILE, "r") as f:
        # Seek to end
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            try:
                event = json.loads(line)
                process_event(event)
            except json.JSONDecodeError:
                pass

def process_event(event):
    topic = event.get("topic", "")
    
    if topic == "operator.pbtest.request":
        data = event.get("data", {})
        print(f"\n[QA] 🧪 PBTEST REQUEST: {data.get('intent')} (Mode: {data.get('mode')})")
        print(f"     Scope: {data.get('scope')} | Browser: {data.get('browser')}")
        print("     -> Monitoring telemetry for verdict...")

    elif topic.startswith("telemetry.client.error"):
        print(f"[QA] 🚨 CLIENT ERROR: {event.get('data', {}).get('message')}")
        # Logic to correlate with active test could go here

    elif topic == "qa.verdict":
        print(f"[QA] ⚖️ VERDICT: {event.get('data', {}).get('status')}")

if __name__ == "__main__":
    print("QA PBTEST Watcher Active...")
    tail_bus()
