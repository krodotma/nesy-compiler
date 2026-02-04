#!/usr/bin/env python3
"""
MABSWARM Operator: The Intelligent Membrane (Protocol v10)
==========================================================

Probes the Multi-Agent Bus (MAB) for patterns and instigates
meta-level control actions (Nudge, Reflect, Backoff, Break).

Usage:
    python3 mabswarm.py --window 60
    python3 mabswarm.py --window 60 --emit-bus
    python3 mabswarm.py --daemon --emit-bus
"""

import argparse
import fcntl
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

# Thresholds
MIN_VELOCITY_MPS = 0.1  # Messages per second (idle check)
MAX_ERROR_RATE = 0.1    # 10% errors triggers REFLECT
LATENCY_SLA_MS = 2000   # 2s latency triggers BACKOFF

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def get_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "mabswarm"

def emit_bus(bus_dir: Path, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def tail_lines(path: Path, *, max_bytes: int = 2_000_000) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    start = max(0, size - max_bytes)
    with path.open("rb") as f:
        f.seek(start)
        data = f.read()
    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    return [ln.decode("utf-8", errors="replace") for ln in lines]

def analyze_window(bus_dir: Path, window: int) -> dict:
    events_path = bus_dir / "events.ndjson"
    if not events_path.exists():
        return {}

    now = time.time()
    cutoff = now - window
    
    total = 0
    errors = 0
    latencies = []
    infer_pending: set[str] = set()
    infer_acked: set[str] = set()
    
    lines = tail_lines(events_path)
        
    for line in reversed(lines):
        try:
            e = json.loads(line)
            ts = float(e.get("ts", 0))
            if ts < cutoff:
                break
            
            total += 1
            if e.get("level") == "error":
                errors += 1
            
            # Extract latency if available (e.g. from plurichat.response)
            data = e.get("data", {})
            if "latency_ms" in data:
                latencies.append(data["latency_ms"])

            topic = str(e.get("topic") or "")
            if topic == "infer_sync.request":
                rid = str((data or {}).get("req_id") or "")
                if rid:
                    infer_pending.add(rid)
            elif topic == "infer_sync.response":
                rid = str((data or {}).get("req_id") or "")
                if rid:
                    infer_acked.add(rid)
                
        except Exception:
            continue
            
    velocity = total / window
    error_rate = errors / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    queue_depth = len([rid for rid in infer_pending if rid not in infer_acked])
    
    return {
        "velocity": velocity,
        "error_rate": error_rate,
        "avg_latency": avg_latency,
        "total": total,
        "queue_depth": queue_depth,
    }

def decide_action(metrics: dict) -> dict:
    actions = []
    
    if metrics.get("velocity", 0) < MIN_VELOCITY_MPS and metrics.get("queue_depth", 0) > 0:
        actions.append({
            "type": "NUDGE",
            "topic": "mabswarm.nudge",
            "reason": "velocity_low",
            "data": {"current": metrics["velocity"], "threshold": MIN_VELOCITY_MPS, "queue_depth": metrics.get("queue_depth", 0)}
        })
        
    if metrics.get("error_rate", 0) > MAX_ERROR_RATE:
        actions.append({
            "type": "REFLECT",
            "topic": "mabswarm.reflect",
            "reason": "error_spike",
            "data": {"current": metrics["error_rate"], "threshold": MAX_ERROR_RATE}
        })
        
    if metrics.get("avg_latency", 0) > LATENCY_SLA_MS:
        actions.append({
            "type": "BACKOFF",
            "topic": "mabswarm.backoff",
            "reason": "latency_violation",
            "data": {"current": metrics["avg_latency"], "threshold": LATENCY_SLA_MS}
        })
        
    return actions

def cmd_run(args):
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    actor = get_actor()
    
    while True:
        metrics = analyze_window(bus_dir, args.window)
        actions = decide_action(metrics)
        
        print(
            f"[{now_iso()}] MABSWARM Probe: v={metrics.get('velocity',0):.2f} "
            f"e={metrics.get('error_rate',0):.2f} l={metrics.get('avg_latency',0):.0f} "
            f"q={metrics.get('queue_depth',0)} | Actions: {len(actions)}"
        )
        
        # Always emit a probe metric if asked (safe).
        if args.emit_bus:
            emit_bus(
                bus_dir,
                "mabswarm.probe",
                "metric",
                "info",
                actor,
                {"window_s": args.window, "metrics": metrics, "actions": [{"type": a["type"], "topic": a["topic"], "reason": a["reason"]} for a in actions]},
            )

        if args.emit_bus:
            for action in actions:
                emit_bus(bus_dir, action["topic"], "request", "warn", actor, action["data"])
                print(f"  -> Emitted {action['type']}")
        else:
            for action in actions:
                print(f"  -> [DRY] Would emit {action['type']}")
                
        if not args.daemon:
            break
        
        time.sleep(args.interval)

def main():
    parser = argparse.ArgumentParser(description="MABSWARM Operator")
    parser.add_argument("--bus-dir", help="Bus directory")
    parser.add_argument("--window", type=int, default=60, help="Analysis window seconds")
    parser.add_argument("--interval", type=int, default=10, help="Daemon interval")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--emit-bus", action="store_true", help="Emit probe + control requests to bus")
    
    args = parser.parse_args()
    cmd_run(args)

if __name__ == "__main__":
    main()
