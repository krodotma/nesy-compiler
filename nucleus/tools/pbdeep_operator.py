#!/usr/bin/env python3
from __future__ import annotations

"""
PBDEEP — deep audit request operator.

Meaning:
  - Broadcast a non-blocking deep audit request that inventories branches,
    lost_and_found artifacts, untracked files, and doc/code drift.
  - This tool does not execute scans; it emits structured bus requests only.
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
    p = argparse.ArgumentParser(prog="pbdeep_operator.py", description="PBDEEP semantic operator: deep audit request.")
    p.add_argument("--bus-dir", default=None)
    p.add_argument("--actor", default=None)
    p.add_argument("--instruction", default="", help="PBDEEP instruction body.")
    p.add_argument("--scope", default="repo", help="Scope tag (default: repo).")
    p.add_argument("--reason", default="operator_pbdeep", help="Short reason code.")
    p.add_argument("--no-infer-sync", action="store_true", help="Disable infer_sync mirror.")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "pbdeep"
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()
    ensure_dir(bus_dir)
    (bus_dir / "events.ndjson").touch(exist_ok=True)

    instruction = str(args.instruction or "").strip()
    if not instruction:
        sys.stderr.write("PBDEEP requires instruction text.\n")
        return 2

    req_id = str(uuid.uuid4())
    payload = {
        "req_id": req_id,
        "scope": str(args.scope),
        "intent": "pbdeep",
        "instruction": instruction,
        "reason": str(args.reason),
        "iso": now_iso_utc(),
        "targets": [
            "branches_index",
            "final_assertion_scan",
            "lost_and_found_inventory",
            "untracked_mystery_scan",
            "doc_code_drift_check",
            "rag_index_update",
            "kg_index_update",
            "rhizome_ingest",
            "emit_report",
        ],
        "constraints": {
            "append_only": True,
            "non_blocking": True,
            "tests_first": True,
            "read_only": True,
        },
    }

    emit_bus(bus_dir, topic="operator.pbdeep.request", kind="request", level="info", actor=actor, data=payload)
    if not args.no_infer_sync:
        emit_bus(bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data=payload)

    sys.stdout.write(req_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
