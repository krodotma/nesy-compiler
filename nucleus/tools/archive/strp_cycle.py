#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def emit_bus(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        raise SystemExit(f"missing agent_bus.py at {tool}")
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


def parse_jsonish(s: str | None):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s == "-":
        return json.load(sys.stdin)
    if s.startswith("@"):
        return json.loads(Path(s[1:]).read_text(encoding="utf-8"))
    return json.loads(s)


def render_table(*, title: str, rows: list[tuple[str, str]]) -> str:
    left_w = max([len(k) for k, _v in rows] + [0])
    right_w = max([len(v) for _k, v in rows] + [0])
    inner_w = max(len(title), left_w + 3 + right_w)
    top = "┌" + ("─" * (inner_w + 2)) + "┐"
    hdr = "│ " + title.ljust(inner_w) + " │"
    mid = "├" + ("─" * (inner_w + 2)) + "┤"
    body = []
    for k, v in rows:
        body.append("│ " + (k.ljust(left_w) + " : " + v).ljust(inner_w) + " │")
    bot = "└" + ("─" * (inner_w + 2)) + "┘"
    return "\n".join([top, hdr, mid, *body, bot])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="strp_cycle.py", description="Emit STRp cycle completion metrics (append-only evidence) + optional human summary.")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    ap.add_argument("--cycle", required=True, help="Cycle name, e.g. sota_curation|rd_ingest|kg_sync")
    ap.add_argument("--curated-count", type=int, default=0)
    ap.add_argument("--kg-nodes-added", type=int, default=0)
    ap.add_argument("--kg-edges-added", type=int, default=0)
    ap.add_argument("--sources", default=None, help="JSON dict of source->count, '-' stdin, '@file.json'.")
    ap.add_argument("--high-signal", default=None, help="JSON list of short strings, '-' stdin, '@file.json'.")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--statusline", default=None, help="Optional STATUSLINE string to attach.")
    ap.add_argument("--topic", default="strp.cycle.complete")
    ap.add_argument("--no-print", action="store_true", help="Do not print the human-readable table.")
    args = ap.parse_args(argv)

    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"

    sources = parse_jsonish(args.sources)
    if sources is None:
        sources = {}
    high_signal = parse_jsonish(args.high_signal)
    if high_signal is None:
        high_signal = []

    payload = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "cycle": args.cycle,
        "curated_count": int(args.curated_count),
        "kg_nodes_added": int(args.kg_nodes_added),
        "kg_edges_added": int(args.kg_edges_added),
        "sources": sources,
        "high_signal": high_signal,
        "notes": args.notes,
        "statusline": args.statusline,
    }

    emit_bus(bus_dir, topic=args.topic, kind="metric", level="info", actor=actor, data=payload)

    if not args.no_print:
        rows = [
            ("Cycle", str(args.cycle)),
            ("Curated Items", str(int(args.curated_count))),
            ("KG Nodes Added", str(int(args.kg_nodes_added))),
            ("KG Edges Added", str(int(args.kg_edges_added))),
            ("Sources", json.dumps(sources, ensure_ascii=False)),
            ("High-Signal", str(len(high_signal))),
            ("Evidence", f"{bus_dir}/events.ndjson"),
        ]
        sys.stdout.write(render_table(title="STRp CYCLE COMPLETE", rows=rows) + "\n")
        if high_signal:
            sys.stdout.write("\nHIGH-SIGNAL ITEMS:\n")
            for item in high_signal:
                sys.stdout.write(f"- {str(item)}\n")
        if args.statusline:
            sys.stdout.write("\n" + args.statusline.strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

