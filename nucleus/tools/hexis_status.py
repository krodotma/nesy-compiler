#!/usr/bin/env python3
"""
Emit HEXIS buffer status to the bus.
Summarizes pending counts per buffer in /tmp/*.buffer.
"""
from __future__ import annotations
import json
import os
import time
import fcntl
from pathlib import Path
import argparse

DEFAULT_BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def collect_counts():
    counts = {}
    oldest = {}
    for buf in Path("/tmp").glob("*.buffer"):
        try:
            with buf.open() as f:
                lines = [l for l in f if l.strip()]
            counts[buf.stem] = len(lines)
            if lines:
                try:
                    oldest[buf.stem] = json.loads(lines[0]).get("iso")
                except Exception:
                    oldest[buf.stem] = None
        except Exception:
            continue
    return counts, oldest

def resolve_events_path(bus_dir: str) -> Path:
    try:
        import agent_bus  # type: ignore

        paths = agent_bus.resolve_bus_paths(bus_dir)
        return Path(paths.events_path)
    except Exception:
        return Path(bus_dir) / "events.ndjson"


def emit(topic: str, actor: str, data: dict, bus_dir: str):
    events_path = resolve_events_path(bus_dir)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "id": os.urandom(16).hex(),
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": "metric",
        "level": "info",
        "actor": actor,
        "data": data,
    }
    with events_path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus-dir", default=DEFAULT_BUS_DIR)
    parser.add_argument("--actor", default="hexis-status")
    args = parser.parse_args()
    counts, oldest = collect_counts()
    emit("hexis.buffer.status", args.actor, {"pending": counts, "oldest_iso": oldest}, args.bus_dir)

if __name__ == "__main__":
    main()
