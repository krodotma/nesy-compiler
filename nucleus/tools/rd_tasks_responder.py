#!/usr/bin/env python3
from __future__ import annotations

"""
RD Tasks responder daemon.

Tails the bus for `rd.tasks.dispatch` and emits:
- `rd.tasks.ack` (kind=response) with correlated req_id
- optional mirror: `infer_sync.response` for existing dashboards

This is the “no messenger” loop for task dispatch.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_bus_dir(args_bus_dir: str | None) -> str:
    return args_bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"


def default_actor(args_actor: str | None) -> str:
    return args_actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "rd-tasks-responder"


def ensure_events_file(bus_dir: str) -> Path:
    p = Path(bus_dir) / "events.ndjson"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def tail_events(events_path: Path, *, since_ts: float, poll_s: float, stop_at_ts: float | None):
    # robust tail: read from start, filter by timestamp
    if not events_path.exists():
        return
    
    current_pos = 0
    while True:
        if stop_at_ts is not None and time.time() >= stop_at_ts:
            return
            
        # Check if file grew
        try:
            stat = events_path.stat()
            if stat.st_size > current_pos:
                with events_path.open("rb") as f:
                    f.seek(current_pos)
                    lines = f.readlines()
                    current_pos = f.tell()
                
                for line in lines:
                    try:
                        obj = json.loads(line.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if obj.get("topic") != "rd.tasks.dispatch":
                        continue
                    try:
                        ts = float(obj.get("ts") or 0.0)
                    except Exception:
                        ts = 0.0
                    if ts < since_ts:
                        continue
                    yield obj
            else:
                time.sleep(max(0.05, poll_s))
        except FileNotFoundError:
            time.sleep(max(0.05, poll_s))


import uuid

def append_ndjson(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

def emit_bus(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    events_path = Path(bus_dir) / "events.ndjson"
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    try:
        append_ndjson(events_path, evt)
    except Exception:
        pass


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="rd_tasks_responder.py", description="Daemon: respond to rd.tasks.dispatch with rd.tasks.ack.")
    ap.add_argument("--bus-dir", default=None)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--poll", default="0.25")
    ap.add_argument("--run-for-s", default="0", help="Run for N seconds then exit (0 = run forever).")
    ap.add_argument("--since-ts", default=None, help="Only respond to triggers >= this UNIX timestamp.")
    ap.add_argument("--target", default=None, help="Only ack if this actor is included in payload.targets (optional).")
    args = ap.parse_args(argv)

    actor = default_actor(args.actor)
    bus_dir = default_bus_dir(args.bus_dir)
    poll_s = max(0.05, float(args.poll))
    run_for_s = max(0.0, float(args.run_for_s))
    since_ts = float(args.since_ts) if args.since_ts is not None else time.time()
    stop_at_ts = None if run_for_s <= 0 else (time.time() + run_for_s)
    target = (args.target or "").strip().lower()

    events_path = ensure_events_file(bus_dir)
    seen_ids: set[str] = set()

    emit_bus(
        bus_dir,
        topic="rd.tasks.responder.ready",
        kind="artifact",
        level="info",
        actor=actor,
        data={"since_ts": since_ts, "iso": now_iso_utc(), "pid": os.getpid()},
    )

    for trig in tail_events(events_path, since_ts=since_ts, poll_s=poll_s, stop_at_ts=stop_at_ts):
        trig_id = str(trig.get("id") or "")
        if trig_id and trig_id in seen_ids:
            continue
        if trig_id:
            seen_ids.add(trig_id)

        data = trig.get("data") if isinstance(trig.get("data"), dict) else {}
        req_id = str(data.get("req_id") or "")
        targets = data.get("targets") if isinstance(data.get("targets"), list) else []
        if target and target not in {str(t).strip().lower() for t in targets}:
            continue

        payload = {
            "req_id": req_id,
            "ack_iso": now_iso_utc(),
            "status": "acknowledged",
            "task_id": data.get("task_id"),
            "intent": data.get("intent"),
            "next": "await_ckin",
            "notes": "rd.tasks.dispatch received; agent will follow task graph and report via CKIN/BEAM.",
        }
        emit_bus(bus_dir, topic="rd.tasks.ack", kind="response", level="info", actor=actor, data=payload)
        if req_id:
            emit_bus(bus_dir, topic="infer_sync.response", kind="response", level="info", actor=actor, data={**payload, "source": "rd_tasks_responder"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

