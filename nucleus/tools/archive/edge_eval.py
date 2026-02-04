#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
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


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
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


def run_once(cmd: list[str], *, timeout_s: float | None) -> tuple[int, float, bool]:
    start = time.perf_counter()
    try:
        p = subprocess.run(cmd, check=False, timeout=timeout_s)
        dur = time.perf_counter() - start
        return int(p.returncode), dur, False
    except subprocess.TimeoutExpired:
        dur = time.perf_counter() - start
        return 124, dur, True


def cmd_run(args: argparse.Namespace) -> int:
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    run_id = str(uuid.uuid4())

    cmd = list(args.cmd)
    if not cmd:
        return 2

    timeout_s = float(args.timeout_s) if args.timeout_s is not None else None
    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))

    record = {
        "id": run_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "edge_eval",
        "name": args.name,
        "actor": actor,
        "cmd": cmd,
        "timeout_s": timeout_s,
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "results": {"warmup": [], "runs": []},
        "tags": [t for t in (args.tag or []) if t.strip()],
    }

    emit_bus(bus_dir, topic="edge_eval.start", kind="log", level="info", actor=actor, data={"id": run_id, "name": args.name, "cmd": cmd})

    for _ in range(warmup):
        code, dur, timed_out = run_once(cmd, timeout_s=timeout_s)
        record["results"]["warmup"].append({"exit_code": code, "duration_s": dur, "timed_out": timed_out})

    failures = 0
    for _ in range(repeat):
        code, dur, timed_out = run_once(cmd, timeout_s=timeout_s)
        record["results"]["runs"].append({"exit_code": code, "duration_s": dur, "timed_out": timed_out})
        if code != 0 or timed_out:
            failures += 1

    record["summary"] = {
        "runs": repeat,
        "failures": failures,
        "p50_s": percentile([r["duration_s"] for r in record["results"]["runs"]], 50),
        "p95_s": percentile([r["duration_s"] for r in record["results"]["runs"]], 95),
        "max_s": max((r["duration_s"] for r in record["results"]["runs"]), default=None),
        "min_s": min((r["duration_s"] for r in record["results"]["runs"]), default=None),
    }

    level = "info" if failures == 0 else "error"
    emit_bus(bus_dir, topic="edge_eval.end", kind="metric", level=level, actor=actor, data=record)

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0 if failures == 0 else 1


def percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(len(values) - 1, f + 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edge_eval.py", description="Edge inference evaluation harness (runner-agnostic).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run a command repeatedly and emit timing metrics.")
    r.add_argument("--name", required=True)
    r.add_argument("--repeat", default="10")
    r.add_argument("--warmup", default="2")
    r.add_argument("--timeout-s", default=None)
    r.add_argument("--out", default=None, help="Write JSON record to a file.")
    r.add_argument("--tag", action="append", default=[])
    r.add_argument("--emit-bus", action="store_true", help="Emit start/end events to the bus.")
    r.add_argument("--", dest="dashdash", action="store_true")  # placeholder for help formatting
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    r.set_defaults(func=cmd_run)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    # Only emit bus if requested; otherwise pass bus_dir through but no-op.
    if not getattr(args, "emit_bus", False):
        args.bus_dir = None
    # Strip leading '--' from remainder if present.
    if getattr(args, "cmd", None) and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

