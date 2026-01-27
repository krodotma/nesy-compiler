#!/usr/bin/env python3
"""
header_snapshot_updater.py - Updates IRKG header snapshot from bus events.

Reads bus events and produces a summarized snapshot for agent_header.py.
Maintains the DR ring (100-cap NDJSON) as a fallback.

Usage:
    python3 header_snapshot_updater.py [--once] [--interval SECONDS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

SNAPSHOT_PATH = Path(os.environ.get("PLURIBUS_HEADER_SNAPSHOT", "/pluribus/.pluribus/index/irkg/header_snapshot.json"))
DR_RING_PATH = Path(os.environ.get("PLURIBUS_HEADER_DR_RING", "/pluribus/.pluribus/dr/header_events.ndjson"))
BUS_PATH = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")) / "events.ndjson"
DR_RING_CAP = 100
DEFAULT_INTERVAL = 300  # 5 minutes


def tail_lines(path: Path, max_lines: int = 500, max_bytes: int = 500_000) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - max_bytes)
            handle.seek(start)
            chunk = handle.read()
    except OSError:
        return []
    if start > 0:
        nl = chunk.find(b"\n")
        if nl != -1:
            chunk = chunk[nl + 1:]
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return [line for line in lines if line.strip()]


def categorize_topic(topic: str) -> tuple[str, str]:
    """Return (category, subcategory) for a topic."""
    parts = topic.split(".")
    if len(parts) < 2:
        return ("sys", "other")

    prefix = parts[0].lower()
    sub = parts[1].lower() if len(parts) > 1 else "other"

    # A2A topics
    if prefix in ("agent", "a2a"):
        if sub in ("notify", "for", "forward"):
            return ("a2a", "for")
        elif sub in ("collab", "collaboration"):
            return ("a2a", "col")
        elif sub in ("internal", "int"):
            return ("a2a", "int")
        elif sub in ("neg", "negotiate"):
            return ("a2a", "neg")
        elif sub in ("task", "assign"):
            return ("a2a", "task")
        return ("a2a", "int")

    # Ops topics
    if prefix in ("ops", "pb", "pbtest", "pblock", "pbresume", "pbflush"):
        return ("ops", prefix if prefix.startswith("pb") else sub)
    if prefix == "hygiene" or sub == "hygiene":
        return ("ops", "hyg")

    # QA topics
    if prefix in ("qa", "quality", "alert", "anomaly"):
        return ("qa", sub if sub in ("alert", "anom", "rem", "verd", "live", "lchk", "act") else "other")

    # System topics
    if prefix in ("sys", "system", "telemetry", "dialogos", "dashboard", "omega"):
        return ("sys", sub[:4] if len(sub) > 4 else sub)

    return ("sys", "other")


def compute_metrics(lines: list[str]) -> dict:
    """Compute bus metrics from event lines."""
    cats = {
        "a2a": defaultdict(int),
        "ops": defaultdict(int),
        "qa": defaultdict(int),
        "sys": defaultdict(int),
    }

    for line in lines:
        try:
            event = json.loads(line)
            topic = event.get("topic", "sys.unknown")
            cat, sub = categorize_topic(topic)
            cats[cat][sub] += 1
        except json.JSONDecodeError:
            continue

    return {
        "a2a": dict(cats["a2a"]),
        "ops": dict(cats["ops"]),
        "qa": dict(cats["qa"]),
        "sys": dict(cats["sys"]),
    }


def update_snapshot() -> dict:
    """Read bus, compute metrics, write snapshot."""
    lines = tail_lines(BUS_PATH)
    raw_metrics = compute_metrics(lines)

    # Map to agent_header.py expected keys
    a2a = raw_metrics.get("a2a", {})
    ops = raw_metrics.get("ops", {})
    qa = raw_metrics.get("qa", {})
    sys_m = raw_metrics.get("sys", {})

    # agent_header.py expects these exact keys
    bus_metrics = {
        "a2a": {
            "for": a2a.get("for", 0),
            "col": a2a.get("col", 0),
            "int": a2a.get("int", 0),
            "neg": a2a.get("neg", 0),
            "task": a2a.get("task", 0),
        },
        "ops": {
            "pbtest": ops.get("pbtest", 0),
            "pblock": ops.get("pblock", 0),
            "pbresume": ops.get("pbresume", 0),
            "pblive": ops.get("pblive", 0),
            "pbflush": ops.get("pbflush", 0),
            "pbiud": ops.get("pbiud", 0),
            "pbcli": ops.get("pbcli", 0),
            "hyg": ops.get("hyg", 0) + ops.get("hygiene", 0),
        },
        "qa": {
            "alert": qa.get("alert", 0),
            "anom": qa.get("anom", 0),
            "rem": qa.get("rem", 0),
            "verd": qa.get("verd", 0),
            "live": qa.get("live", 0),
            "lchk": qa.get("lchk", 0),
            "act": qa.get("act", 0),
            "hyg": qa.get("hyg", 0),
        },
        "sys": {
            "tel": sys_m.get("tel", 0) + sys_m.get("tele", 0),
            "task": sys_m.get("task", 0),
            "agent": sys_m.get("agent", 0),
            "dlg": sys_m.get("dlg", 0) + sys_m.get("disp", 0),
            "ohm": sys_m.get("ohm", 0),
            "omg": sys_m.get("omg", 0),
            "prov": sys_m.get("prov", 0),
            "dash": sys_m.get("dash", 0),
            "brow": sys_m.get("brow", 0),
        },
        "total": sum(sum(v.values()) for v in raw_metrics.values()),
    }

    snapshot = {
        "version": "1.0",
        "updated_ts": time.time(),
        "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bus": bus_metrics,
        "tasks": {
            "active": 0,
            "total": 0,
            "label": "none",
            "progress": None,
        },
        "hexis": {
            "buffer_count": 0,
            "state": "quiet",
        },
    }

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    tmp.replace(SNAPSHOT_PATH)

    return snapshot


def update_dr_ring() -> int:
    """Maintain 100-cap DR ring from bus events."""
    lines = tail_lines(BUS_PATH, max_lines=DR_RING_CAP)
    DR_RING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DR_RING_PATH.open("w", encoding="utf-8") as f:
        for line in lines[-DR_RING_CAP:]:
            f.write(line + "\n")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update IRKG header snapshot")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Update interval in seconds")
    args = parser.parse_args()

    while True:
        try:
            snapshot = update_snapshot()
            dr_count = update_dr_ring()
            print(f"[{time.strftime('%H:%M:%S')}] Snapshot updated | DR ring: {dr_count} events")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}", file=sys.stderr)

        if args.once:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
