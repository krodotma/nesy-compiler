#!/usr/bin/env python3
"""
beam_append.py

Append-only discourse writer for BEAM 10× iterations with file locking and bus evidence.

Why:
- Multiple agents may append concurrently; this tool provides a safe append protocol.
- Every append also emits an immutable bus artifact (`beam.10x.appended`).

Usage:
  PLURIBUS_BUS_DIR=/pluribus/.pluribus/bus \\
  python3 /pluribus/nucleus/tools/beam_append.py \\
    --file /pluribus/agent_reports/2025-12-15_beam_10x_discourse.md \\
    --iteration 1 \\
    --subagent-id 3 \\
    --scope infercell \\
    --tags V I \\
    --refs nucleus/tools/lens_collimator.py aefb04c \\
    --claim \"Tabs are trace forks\" \\
    --next-check \"Implement trace fork/merge events\" \\
    --text \"Optional extra markdown...\"\n+"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import uuid
from pathlib import Path
from typing import List

sys_dont_write_bytecode = True


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_ts() -> float:
    return time.time()


def emit_bus_event(topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    bus_path = bus_dir / "events.ndjson"
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "id": str(uuid.uuid4()),
        "ts": now_ts(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }

    with bus_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="beam_append.py")
    ap.add_argument("--file", required=True, help="Append-only discourse file path")
    ap.add_argument("--iteration", required=True, type=int, help="Iteration number (0..10)")
    ap.add_argument("--subagent-id", required=True, help="Subagent identifier (name or 1..10)")
    ap.add_argument("--scope", required=True, help="Focus scope (lens_collimator|plurichat|infercell|...)")
    ap.add_argument("--tags", nargs="+", default=[], help="Tags: V R I G")
    ap.add_argument("--refs", nargs="*", default=[], help="File/commit references")
    ap.add_argument("--claim", action="append", default=[], help="Claim line (repeatable)")
    ap.add_argument("--next-check", action="append", default=[], help="Verification/falsifier line (repeatable)")
    ap.add_argument("--text", default="", help="Extra markdown appended verbatim")
    args = ap.parse_args(argv)

    actor = os.environ.get("PLURIBUS_ACTOR", "unknown")
    ts_iso = now_iso()
    entry_id = str(uuid.uuid4())

    discourse_path = Path(args.file)
    discourse_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "",
        f"## Entry {entry_id} — {actor} — {ts_iso}",
        "",
        f"iteration: {args.iteration}",
        f"subagent_id: {args.subagent_id}",
        f"actor: {actor}",
        f"scope: {args.scope}",
        f"tags: [{', '.join(args.tags)}]" if args.tags else "tags: []",
        "refs:",
    ]
    if args.refs:
        header += [f"- `{r}`" for r in args.refs]
    else:
        header += ["- (none)"]

    claims = ["", "claims:"]
    claims += [f"- {c}" for c in (args.claim or [])] or ["- (none)"]

    checks = ["", "next_checks:"]
    checks += [f"- {c}" for c in (args.next_check or [])] or ["- (none)"]

    extra = []
    if args.text.strip():
        extra = ["", "notes:", args.text.rstrip()]

    block = "\n".join(header + claims + checks + extra) + "\n"

    # Append with lock (prevents interleaving).
    with discourse_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(block)
        fcntl.flock(f, fcntl.LOCK_UN)

    emit_bus_event(
        topic="beam.10x.appended",
        kind="artifact",
        level="info",
        actor=actor,
        data={
            "entry_id": entry_id,
            "iteration": args.iteration,
            "subagent_id": args.subagent_id,
            "scope": args.scope,
            "tags": args.tags,
            "refs": args.refs,
            "claims_count": len(args.claim or []),
            "next_checks_count": len(args.next_check or []),
            "file": str(discourse_path),
        },
    )

    print(entry_id)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

