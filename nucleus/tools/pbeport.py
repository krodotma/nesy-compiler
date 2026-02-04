#!/usr/bin/env python3
"""
PBEPORT — Pluribus Bus Executive Report (snapshot distillation)

Purpose:
- Provide a compact, metrics-first status distillation of live Pluribus flows
  (past / present / future) from the append-only bus.
- Produce small “sparkline” subgraphics suitable for TUI/Web embedding.
- Act as a semantic operator: invoked as a keyword in PluriChat (PBEPORT)
  or as a standalone CLI.

Inputs:
- Bus event tape: $PLURIBUS_BUS_DIR/events.ndjson (system of record)

Outputs:
- Human-readable snapshot printed to stdout
- Optional bus metric event: `pbeport.snapshot`
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.dont_write_bytecode = True

SPARK = "▁▂▃▄▅▆▇█"


def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pbeport"


def human_int(n: int) -> str:
    return f"{n:,}"

def count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except Exception:
        return 0


def count_beam_entries(path: Path) -> int:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0
    return sum(1 for ln in txt.splitlines() if ln.startswith("## Entry "))


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
        line = (line or "").strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if isinstance(e, dict) and e.get("topic"):
            out.append(e)
    return out


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


@dataclass(frozen=True)
class Snapshot:
    window_s: int
    width: int
    now_iso: str
    bus_dir: str
    events_total: int
    kinds: dict[str, int]
    topic_prefixes: dict[str, int]
    spark_bins: list[int]
    last_checkins: dict[str, dict[str, Any]]
    pending_req_ids: list[str]
    provider_incidents: list[dict[str, Any]]
    git_recent: list[dict[str, Any]]


def build_snapshot(*, bus_dir: Path, window_s: int, width: int) -> Snapshot:
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

    last_checkins: dict[str, dict[str, Any]] = {}
    for e in reversed(recent):
        if str(e.get("topic")) != "infer_sync.checkin":
            continue
        actor = str(e.get("actor") or "unknown")
        if actor in last_checkins:
            continue
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        last_checkins[actor] = {
            "status": data.get("status"),
            "done": data.get("done"),
            "open": data.get("open"),
            "blocked": data.get("blocked"),
            "errors": data.get("errors"),
            "next": data.get("next"),
            "subproject": data.get("subproject"),
            "iso": e.get("iso"),
        }

    req_ids: dict[str, dict[str, Any]] = {}
    seen_resp: set[str] = set()
    for e in recent:
        topic = str(e.get("topic") or "")
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        if topic == "infer_sync.request":
            rid = data.get("req_id")
            if isinstance(rid, str) and rid:
                req_ids[rid] = {"intent": data.get("intent"), "subproject": data.get("subproject"), "iso": e.get("iso")}
        elif topic == "infer_sync.response":
            rid = data.get("req_id")
            if isinstance(rid, str) and rid:
                seen_resp.add(rid)
    pending = [rid for rid in req_ids.keys() if rid not in seen_resp][:20]

    incidents: list[dict[str, Any]] = []
    for e in reversed(recent):
        if str(e.get("topic")) != "providers.incident":
            continue
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        incidents.append(
            {
                "iso": e.get("iso"),
                "provider": data.get("provider"),
                "blocker": data.get("blocker"),
                "exit_code": data.get("exit_code"),
                "cooldown_s": data.get("cooldown_s"),
            }
        )
        if len(incidents) >= 8:
            break

    git_recent: list[dict[str, Any]] = []
    for e in reversed(recent):
        topic = str(e.get("topic") or "")
        if not (topic.startswith("git.") or topic.startswith("iso_git.")):
            continue
        if topic not in {"git.commit", "git.reset"} and not topic.startswith("git.evo."):
            continue
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        git_recent.append(
            {
                "iso": e.get("iso"),
                "topic": topic,
                "data": {k: data.get(k) for k in ["sha", "applied_commit", "source_commit", "target", "branch"] if k in data},
            }
        )
        if len(git_recent) >= 6:
            break

    return Snapshot(
        window_s=int(window_s),
        width=width,
        now_iso=now_iso(),
        bus_dir=str(bus_dir),
        events_total=len(recent),
        kinds=dict(kinds),
        topic_prefixes=dict(prefixes),
        spark_bins=bins,
        last_checkins=last_checkins,
        pending_req_ids=pending,
        provider_incidents=incidents,
        git_recent=git_recent,
    )


def render_snapshot(s: Snapshot) -> str:
    def fmt_iso(v: Any) -> str:
        return str(v) if v else "-"

    top_prefixes = sorted(s.topic_prefixes.items(), key=lambda x: (-x[1], x[0]))[:10]
    top_kinds = sorted(s.kinds.items(), key=lambda x: (-x[1], x[0]))

    lines: list[str] = []
    lines.append(f"PBEPORT snapshot @ {s.now_iso}")
    lines.append(f"bus_dir: {s.bus_dir}")
    lines.append(f"window: last {s.window_s}s | events: {human_int(s.events_total)} | spark: {sparkline(s.spark_bins)}")

    lines.append("")
    lines.append("PRESENT")
    lines.append("  kinds: " + ", ".join([f"{k}={v}" for k, v in top_kinds]) if top_kinds else "  kinds: (none)")
    lines.append("  topics: " + ", ".join([f"{k}={v}" for k, v in top_prefixes]) if top_prefixes else "  topics: (none)")

    if s.last_checkins:
        lines.append("")
        lines.append("  infer_sync.checkin (latest per actor)")
        for actor, d in sorted(s.last_checkins.items(), key=lambda x: str(x[0]))[:12]:
            lines.append(
                "  - "
                + actor
                + f" status={d.get('status')} done={d.get('done')} open={d.get('open')} blocked={d.get('blocked')} errors={d.get('errors')} subproject={d.get('subproject')} iso={fmt_iso(d.get('iso'))}"
            )

    lines.append("")
    lines.append("PAST")
    if s.git_recent:
        for e in s.git_recent:
            lines.append(f"  - {fmt_iso(e.get('iso'))} {e.get('topic')} {json.dumps(e.get('data', {}), ensure_ascii=False)}")
    else:
        lines.append("  - (no git events in window)")

    if s.provider_incidents:
        lines.append("")
        lines.append("  providers.incident (recent)")
        for inc in s.provider_incidents:
            lines.append(
                f"  - {fmt_iso(inc.get('iso'))} provider={inc.get('provider')} blocker={inc.get('blocker')} code={inc.get('exit_code')} cooldown_s={inc.get('cooldown_s')}"
            )

    lines.append("")
    lines.append("FUTURE")
    if s.pending_req_ids:
        lines.append(f"  pending infer_sync req_ids (no response in window): {len(s.pending_req_ids)}")
        for rid in s.pending_req_ids[:10]:
            lines.append(f"  - {rid}")
    else:
        lines.append("  pending infer_sync req_ids: 0")

    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pbeport.py")
    ap.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    ap.add_argument("--window", type=int, default=900, help="Window seconds (default 900)")
    ap.add_argument("--width", type=int, default=24, help="Sparkline bins (default 24)")
    ap.add_argument("--emit-bus", action="store_true", help="Emit pbeport.snapshot metric to bus")
    args = ap.parse_args(argv)

    bus_dir = Path(args.bus_dir)
    snap = build_snapshot(bus_dir=bus_dir, window_s=int(args.window), width=int(args.width))
    sys.stdout.write(render_snapshot(snap))

    if args.emit_bus:
        emit_bus(
            bus_dir,
            topic="pbeport.snapshot",
            kind="metric",
            level="info",
            actor=default_actor(),
            data={
                "window_s": snap.window_s,
                "events_total": snap.events_total,
                "kinds": snap.kinds,
                "topic_prefixes": dict(sorted(snap.topic_prefixes.items(), key=lambda x: -x[1])[:20]),
                "pending_infer_sync": len(snap.pending_req_ids),
                "provider_incidents": len(snap.provider_incidents),
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
