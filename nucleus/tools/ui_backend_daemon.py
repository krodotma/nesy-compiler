#!/usr/bin/env python3
"""
UI Backend Daemon: The Heart of the Interface
=============================================

Periodically probes system state (Providers, Agents, Art) and emits 
normalized bus events to drive the WebUI and TUI.

Fills the gap between "Passive Tools" and "Active Dashboards".

Usage:
    python3 ui_backend_daemon.py --interval 30
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

# Import PluriChat for provider checks
sys.path.append(str(Path(__file__).parent))
try:
    import plurichat
except ImportError:
    plurichat = None

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def emit_bus(bus_dir: Path, topic: str, data: dict, actor: str = "ui-backend") -> None:
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": "metric",
        "level": "info",
        "actor": actor,
        "data": data,
    }
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _tail_events(events_path: Path, *, max_lines: int = 2500) -> list[dict]:
    try:
        lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except Exception:
        return []
    out: list[dict] = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _compute_entropy_and_mood(bus_dir: Path) -> tuple[float, str]:
    """Derive Art Dept signals from recent bus activity (no RNG)."""
    events_path = bus_dir / "events.ndjson"
    events = _tail_events(events_path)
    now = time.time()
    window_s = 120.0
    recent = []
    for e in events:
        try:
            ts = float(e.get("ts") or 0.0)
        except Exception:
            ts = 0.0
        if ts and (now - ts) <= window_s:
            recent.append(e)
    rate = len(recent) / max(1.0, window_s)  # events per second
    error_rate = 0.0
    if recent:
        error_rate = len([e for e in recent if str(e.get("level") or "").lower() in {"error", "warn"}]) / float(len(recent))

    # Entropy is a bounded function of event rate + error_rate.
    entropy = min(1.0, 0.08 + (rate * 1.8) + (error_rate * 0.6))
    if error_rate >= 0.15:
        mood = "anxious"
    elif rate >= 0.08:
        mood = "focused"
    else:
        mood = "calm"
    return float(entropy), mood


def check_providers(bus_dir: Path):
    """Probe all providers and emit status."""
    if not plurichat:
        return
    
    # Use the unified JSON status API
    statuses = plurichat.get_status_json()
    
    for name, st in statuses.items():
        # Emit event expected by Dashboard (see index.tsx handleBusEvent)
        emit_bus(bus_dir, f"provider.status.{name}", {
            "provider": name,
            "available": st.get("available", False),
            "model": st.get("model"),
            "error": st.get("error"),
            "note": st.get("blocker")
        }, "ui-backend")

def check_art_dept(bus_dir: Path):
    """Emit Art Dept signals (Entropy/Mood)."""
    entropy, mood = _compute_entropy_and_mood(bus_dir)
    
    emit_bus(bus_dir, "art.state.update", {
        "entropy": entropy,
        "mood": mood,
        "active_layer": "GenerativeBackground"
    }, "art-director")

def run_daemon(args):
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    print(f"UI Backend Daemon running (interval={args.interval}s)...")
    
    while True:
        try:
            check_providers(bus_dir)
            check_art_dept(bus_dir)
            # Add other checks (Agents, Services) here
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus-dir")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    run_daemon(args)

if __name__ == "__main__":
    main()
