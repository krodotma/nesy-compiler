#!/usr/bin/env python3
"""
MBAD — Multi-Agent Bus (Membrane) Diagnostics

Purpose:
- Produce a compact “membrane view” of the multi-agent bus without secrets.
- Surface coordination backlog (infer_sync pending), buffer backlog (hexis),
  and live activity (kind/prefix counts) in a single snapshot.
- Optional bus metric emission: `mbad.snapshot`.

Usage:
  python3 mbad.py --window 900
  python3 mbad.py --window 900 --emit-bus
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SPARK = "▁▂▃▄▅▆▇█"


def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "mbad"


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


def parse_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        s = (line or "").strip()
        if not s:
            continue
        try:
            e = json.loads(s)
        except Exception:
            continue
        if isinstance(e, dict) and e.get("topic"):
            out.append(e)
    return out


def sparkline(values: list[int]) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return SPARK[0] * len(values)
    out: list[str] = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(SPARK) - 1))
        out.append(SPARK[max(0, min(len(SPARK) - 1, idx))])
    return "".join(out)


def hexis_status(hexis_dir: Path) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for p in sorted(hexis_dir.glob("*.buffer")):
        agent = p.name.replace(".buffer", "")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        pending = len([ln for ln in lines if ln.strip()])
        oldest_topic = None
        oldest_iso = None
        if pending:
            try:
                first = json.loads(lines[0])
                oldest_topic = first.get("topic")
                oldest_iso = first.get("iso")
            except Exception:
                oldest_topic = None
                oldest_iso = None
        status[agent] = {"pending": pending, "oldest_topic": oldest_topic, "oldest_iso": oldest_iso, "path": str(p)}
    return status


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> str:
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": now_ts(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    path = bus_dir / "events.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(evt, ensure_ascii=False, separators=(",", ":")) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    return evt_id


def build_snapshot(*, bus_dir: Path, window_s: int, width: int, hexis_dir: Path) -> dict[str, Any]:
    events_path = bus_dir / "events.ndjson"
    events = parse_events(tail_lines(events_path))

    now = now_ts()
    cutoff = now - float(window_s)
    recent: list[dict[str, Any]] = []
    for e in events:
        try:
            ts_f = float(e.get("ts"))
        except Exception:
            continue
        if ts_f >= cutoff:
            recent.append(e)

    kinds = Counter()
    prefixes = Counter()
    for e in recent:
        kinds[str(e.get("kind") or "unknown")] += 1
        topic = str(e.get("topic") or "")
        pref = topic.split(".", 1)[0] if "." in topic else topic
        if pref:
            prefixes[pref] += 1

    width = max(8, int(width))
    bin_s = max(1.0, float(window_s) / float(width))
    bins = [0 for _ in range(width)]
    for e in recent:
        try:
            ts_f = float(e.get("ts"))
        except Exception:
            continue
        idx = int((ts_f - cutoff) / bin_s)
        if 0 <= idx < width:
            bins[idx] += 1

    req_ts: dict[str, float] = {}
    resp_seen: set[str] = set()
    for e in events:
        topic = str(e.get("topic") or "")
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        ts = e.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        if topic == "infer_sync.request":
            rid = str(data.get("req_id") or "")
            if rid:
                req_ts[rid] = float(ts)
        elif topic == "infer_sync.response":
            rid = str(data.get("req_id") or "")
            if rid:
                resp_seen.add(rid)

    pending = sorted(
        [{"req_id": rid, "age_s": max(0.0, now - ts)} for rid, ts in req_ts.items() if rid not in resp_seen],
        key=lambda x: x["age_s"],
        reverse=True,
    )

    hexis = hexis_status(hexis_dir)
    hexis_pending_total = sum(int(v.get("pending") or 0) for v in hexis.values())

    return {
        "iso": now_iso(),
        "bus_dir": str(bus_dir),
        "window_s": int(window_s),
        "events": len(recent),
        "kinds": dict(kinds),
        "topic_prefixes": dict(prefixes),
        "spark": sparkline(bins),
        "infer_sync_pending": len(pending),
        "infer_sync_oldest_age_s": float(pending[0]["age_s"]) if pending else 0.0,
        "hexis_pending_total": hexis_pending_total,
        "hexis_by_agent": {k: {"pending": int(v["pending"]), "oldest_topic": v.get("oldest_topic")} for k, v in hexis.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="MBAD (Multi-Agent Bus Diagnostics)")
    ap.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    ap.add_argument("--hexis-dir", default=os.environ.get("HEXIS_BUFFER_DIR", "/tmp"))
    ap.add_argument("--window", type=int, default=900)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--emit-bus", action="store_true")
    args = ap.parse_args()

    bus_dir = Path(args.bus_dir)
    hexis_dir = Path(args.hexis_dir)
    snap = build_snapshot(bus_dir=bus_dir, window_s=int(args.window), width=int(args.width), hexis_dir=hexis_dir)

    print(f"MBAD snapshot @ {snap['iso']}")
    print(f"bus_dir: {snap['bus_dir']}")
    print(f"window: last {snap['window_s']}s | events: {snap['events']} | spark: {snap['spark']}")
    print("")
    print("MEMBRANE")
    kinds = snap["kinds"]
    print("  kinds: " + ", ".join([f"{k}={kinds[k]}" for k in sorted(kinds.keys())]))
    prefs = snap["topic_prefixes"]
    top = sorted(prefs.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("  prefixes: " + ", ".join([f"{k}={v}" for k, v in top]))
    print(f"  infer_sync.pending: {snap['infer_sync_pending']} (oldest_age_s={int(snap['infer_sync_oldest_age_s'])})")
    print(f"  hexis.pending_total: {snap['hexis_pending_total']}")

    if args.emit_bus:
        emit_bus(
            bus_dir,
            topic="mbad.snapshot",
            kind="metric",
            level="info",
            actor=default_actor(),
            data={k: v for k, v in snap.items() if k not in {"bus_dir"}},
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

