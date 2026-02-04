#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any, Dict

sys.dont_write_bytecode = True

try:
    from .agent_bus import resolve_bus_paths, emit_event, default_actor
except Exception:  # pragma: no cover
    from agent_bus import resolve_bus_paths, emit_event, default_actor


def build_payload(size: int) -> str:
    if size <= 0:
        return ""
    return "x" * size


def run_soak(
    *,
    bus_dir: str,
    topic: str,
    total_events: int,
    rate: float,
    payload_bytes: int,
    emit_bus_summary: bool,
) -> Dict[str, Any]:
    paths = resolve_bus_paths(bus_dir)
    actor = default_actor()
    payload = build_payload(payload_bytes)
    run_id = str(uuid.uuid4())

    start = time.time()
    sent = 0
    sleep_s = (1.0 / rate) if rate and rate > 0 else 0.0

    for idx in range(total_events):
        emit_event(
            paths,
            topic=topic,
            kind="metric",
            level="info",
            actor=actor,
            data={"seq": idx, "payload": payload},
            trace_id=None,
            run_id=run_id,
            durable=False,
        )
        sent += 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    elapsed = max(0.0001, time.time() - start)
    eps = sent / elapsed

    summary = {
        "run_id": run_id,
        "events": sent,
        "duration_s": round(elapsed, 4),
        "eps": round(eps, 2),
        "topic": topic,
        "payload_bytes": payload_bytes,
    }

    if emit_bus_summary:
        emit_event(
            paths,
            topic="qa.soak.complete",
            kind="metric",
            level="info",
            actor=actor,
            data=summary,
            trace_id=None,
            run_id=run_id,
            durable=False,
        )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="QA bus soak/load generator")
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus")
    parser.add_argument("--topic", default="qa.soak.event")
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--rate", type=float, default=50.0, help="Events per second; 0 = no throttling")
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--emit-bus", action="store_true")
    args = parser.parse_args()

    summary = run_soak(
        bus_dir=args.bus_dir,
        topic=args.topic,
        total_events=max(0, args.events),
        rate=args.rate,
        payload_bytes=max(0, args.payload_bytes),
        emit_bus_summary=args.emit_bus,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
