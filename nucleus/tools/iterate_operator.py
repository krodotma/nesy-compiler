#!/usr/bin/env python3
"""
ITERATE — Semantic Operator (Coordination / Evolution)

Purpose:
- When the operator types "iterate" (alone), broadcast a non-blocking request that
  participating agents should:
    1) emit a CKIN dashboard snapshot
    2) append one BEAM entry (and/or mark [V] verifications)
    3) optionally respond on infer_sync.response with a concise summary

This tool is intentionally append-only and bus-first.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

sys_dont_write_bytecode = True  # set below without importing sys in hot path
import sys  # noqa: E402

sys.dont_write_bytecode = True

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> str:
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    path = bus_dir / "events.ndjson"
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(evt, ensure_ascii=False, separators=(",", ":")) + "\n")
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_UN)
    return evt_id


def publish_iterate(
    *,
    bus_dir: Path,
    actor: str,
    req_id: str | None,
    subproject: str,
    intent: str,
    response_topic: str,
    window_s: int,
) -> str:
    rid = req_id or str(uuid.uuid4())
    constraints = {"append_only": True, "tests_first": True, "non_blocking": True}
    inputs = {
        "operator": "iterate",
        "window_s": int(window_s),
        "charter_ref": "agent_reports/2025-12-15_beam_10x_charter.md",
        "beam_ref": "agent_reports/2025-12-15_beam_10x_discourse.md",
        "golden_ref": "nucleus/docs/GOLDEN_SYNTHESIS_DISCOURSE.md",
        "requested_actions": [
            "emit_ckin",
            "append_beam_entry",
            "cross_verify_one_claim",
            "reply_infer_sync_response_optional",
        ],
    }

    emit_bus(
        bus_dir,
        topic="infer_sync.request",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": rid,
            "subproject": subproject,
            "intent": intent,
            "inputs": inputs,
            "constraints": constraints,
            "response_topic": response_topic,
            "iso": now_iso_utc(),
        },
    )

    emit_bus(
        bus_dir,
        topic="operator.iterate.request",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": rid,
            "subproject": subproject,
            "intent": intent,
            "window_s": int(window_s),
            "response_topic": response_topic,
            "iso": now_iso_utc(),
        },
    )

    return rid


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="iterate_operator.py", description="ITERATE semantic operator (broadcast coordination request).")
    ap.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    ap.add_argument("--agent", default=default_actor())
    ap.add_argument("--req-id", default=None)
    ap.add_argument("--subproject", default="beam_10x")
    ap.add_argument("--intent", default="iterate")
    ap.add_argument("--response-topic", default="infer_sync.response")
    ap.add_argument("--window-s", type=int, default=900)
    args = ap.parse_args(argv)

    bus_dir = Path(args.bus_dir)
    rid = publish_iterate(
        bus_dir=bus_dir,
        actor=args.agent,
        req_id=args.req_id,
        subproject=args.subproject,
        intent=args.intent,
        response_topic=args.response_topic,
        window_s=args.window_s,
    )
    sys.stdout.write(rid + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

