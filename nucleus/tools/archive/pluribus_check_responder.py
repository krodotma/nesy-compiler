#!/usr/bin/env python3
from __future__ import annotations

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
    return args_actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "unknown"


def ensure_events_file(bus_dir: str) -> Path:
    p = Path(bus_dir) / "events.ndjson"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def tail_events(
    events_path: Path,
    *,
    since_ts: float,
    poll_s: float,
    stop_at_ts: float | None,
):
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
                    if obj.get("topic") != "pluribus.check.trigger":
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


def respond(*, bus_dir: str, actor: str, trigger: dict) -> int:
    tools_dir = Path(__file__).resolve().parent
    report_tool = tools_dir / "pluribus_check.py"
    trigger_data = (trigger.get("data") or {}) if isinstance(trigger.get("data"), dict) else {}
    env = {
        **os.environ,
        "PLURIBUS_BUS_DIR": bus_dir,
        "PLURIBUS_ACTOR": actor,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PLURIBUSCHECK_TRIGGER_ID": str(trigger.get("id") or ""),
        "PLURIBUSCHECK_TRIGGER_ISO": str(trigger.get("iso") or ""),
        "PLURIBUSCHECK_TRIGGER_TS": str(trigger.get("ts") or ""),
        "PLURIBUSCHECK_TRIGGER_ACTOR": str(trigger.get("actor") or ""),
        "PLURIBUSCHECK_TRIGGER_MESSAGE": str(trigger_data.get("message") or ""),
        "PLURIBUSCHECK_TRIGGER_REQ_ID": str(trigger_data.get("req_id") or ""),
    }
    res = subprocess.run(
        [
            sys.executable,
            str(report_tool),
            "--bus-dir",
            bus_dir,
            "report",
            "--goal",
            "PLURIBUSCHECK",
            "--step",
            "auto-responder",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        sys.stderr.write(f"pluribus_check.py failed: {res.stderr}\n")
    return res.returncode


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pluribus_check_responder.py", description="Daemon: respond to pluribus.check.trigger by emitting pluribus.check.report.")
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

    # Emit a “ready” artifact once (append-only evidence).
    try:
        tool = Path(__file__).with_name("agent_bus.py")
        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--bus-dir",
                bus_dir,
                "pub",
                "--topic",
                "pluribus.check.responder.ready",
                "--kind",
                "artifact",
                "--level",
                "info",
                "--actor",
                actor,
                "--data",
                json.dumps({"since_ts": since_ts, "iso": now_iso_utc(), "pid": os.getpid()}, ensure_ascii=False),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "PYTHONDONTWRITE_BYTECODE": "1"},
        )
    except Exception:
        pass

    for trig in tail_events(events_path, since_ts=since_ts, poll_s=poll_s, stop_at_ts=stop_at_ts):
        trig_id = str(trig.get("id") or "")
        if trig_id and trig_id in seen_ids:
            continue
        if trig_id:
            seen_ids.add(trig_id)
        respond(bus_dir=bus_dir, actor=actor, trigger=trig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
