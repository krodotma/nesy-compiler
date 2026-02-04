#!/usr/bin/env python3
from __future__ import annotations

"""
PBFLUSH — broadcast “finish & await” semantic operator.

Meaning:
  - Stop further iterative cycles for the current work epoch.
  - Finish/flush remaining local tasks (append-only artifacts/evidence).
  - Transition into “await next CKIN” posture.

Mechanics:
  - Emits `operator.pbflush.request` (kind=request) to the bus.
  - Also emits `infer_sync.request` (intent=pbflush) for agents using the InferSync channel.

This tool does not force-kill processes; it coordinates via append-only bus requests.
"""

import argparse
import getpass
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
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
    append_ndjson(bus_dir / "events.ndjson", evt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbflush_operator.py", description="PBFLUSH semantic operator: broadcast finish+await posture.")
    p.add_argument("--bus-dir", default=None)
    p.add_argument("--actor", default=None)
    p.add_argument("--subproject", default="ops", help="Subproject tag for infer_sync requests.")
    p.add_argument("--message", default="PBFLUSH", help="Optional operator message/context.")
    p.add_argument("--reason", default="operator_pbflush", help="Short reason code.")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "pbflush"
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()
    ensure_dir(bus_dir)
    (bus_dir / "events.ndjson").touch(exist_ok=True)

    req_id = str(uuid.uuid4())
    payload = {
        "req_id": req_id,
        "subproject": str(args.subproject),
        "intent": "pbflush",
        "message": str(args.message),
        "reason": str(args.reason),
        "iso": now_iso_utc(),
    }

    emit_bus(bus_dir, topic="operator.pbflush.request", kind="request", level="warn", actor=actor, data=payload)
    emit_bus(bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data=payload)

    sys.stdout.write(req_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

