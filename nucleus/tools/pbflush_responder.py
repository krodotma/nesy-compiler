#!/usr/bin/env python3
from __future__ import annotations

"""
PBFLUSH responder daemon.

Tails the bus for `operator.pbflush.request` and emits:
  - `operator.pbflush.ack` (kind=response)
  - `infer_sync.response` (kind=response, same req_id)

This is an opt-in “no messenger” path similar to PLURIBUSCHECK responder.
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
    return args_actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pbflush-responder"


def ensure_events_file(bus_dir: str) -> Path:
    p = Path(bus_dir) / "events.ndjson"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def tail_events(events_path: Path, *, since_ts: float, poll_s: float, stop_at_ts: float | None):
    def emit_backfill() -> list[dict]:
        # Backfill avoids missing fast triggers that land before we start following.
        try:
            max_bytes = 512 * 1024
            with events_path.open("rb") as bf:
                bf.seek(0, os.SEEK_END)
                end = bf.tell()
                start = max(0, end - max_bytes)
                bf.seek(start)
                data = bf.read(end - start)
            lines = data.splitlines()
            out: list[dict] = []
            for b in lines[-2000:]:
                try:
                    obj = json.loads(b.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if obj.get("topic") != "operator.pbflush.request":
                    continue
                try:
                    ts = float(obj.get("ts") or 0.0)
                except Exception:
                    ts = 0.0
                if ts < since_ts:
                    continue
                out.append(obj)
            return out
        except Exception:
            return []

    for obj in emit_backfill():
        yield obj

    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            if stop_at_ts is not None and time.time() >= stop_at_ts:
                return
            line = f.readline()
            if not line:
                time.sleep(max(0.05, poll_s))
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("topic") != "operator.pbflush.request":
                continue
            try:
                ts = float(obj.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            if ts < since_ts:
                continue
            yield obj


def emit_bus(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            kind,
            "--level",
            level,
            "--actor",
            actor,
            "--data",
            json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pbflush_responder.py", description="Daemon: respond to operator.pbflush.request with ack/response events.")
    ap.add_argument("--bus-dir", default=None)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--poll", default="0.25")
    ap.add_argument("--run-for-s", default="0", help="Run for N seconds then exit (0 = run forever).")
    ap.add_argument("--since-ts", default=None, help="Only respond to triggers >= this UNIX timestamp.")
    args = ap.parse_args(argv)

    actor = default_actor(args.actor)
    bus_dir = default_bus_dir(args.bus_dir)
    poll_s = max(0.05, float(args.poll))
    run_for_s = max(0.0, float(args.run_for_s))
    since_ts = float(args.since_ts) if args.since_ts is not None else time.time()
    stop_at_ts = None if run_for_s <= 0 else (time.time() + run_for_s)

    events_path = ensure_events_file(bus_dir)
    seen_ids: set[str] = set()

    emit_bus(
        bus_dir,
        topic="operator.pbflush.responder.ready",
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
        payload = {
            "req_id": req_id,
            "ack_iso": now_iso_utc(),
            "status": "acknowledged",
            "posture": "await_ckin",
            "notes": "PBFLUSH received; finish local tasks and await next CKIN planning/co-execution.",
        }
        emit_bus(bus_dir, topic="operator.pbflush.ack", kind="response", level="info", actor=actor, data=payload)
        if req_id:
            emit_bus(bus_dir, topic="infer_sync.response", kind="response", level="info", actor=actor, data={**payload, "source": "pbflush_responder"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
