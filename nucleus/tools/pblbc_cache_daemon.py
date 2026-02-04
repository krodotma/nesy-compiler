#!/usr/bin/env python3
"""
PBLBC Cache Daemon
=================

Refreshes the PBLBC cache on a fixed cadence (default 15 minutes).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools import pblbc_operator


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="pblbc_cache_daemon.py", description="PBLBC cache updater")
    ap.add_argument("--policy", default=str(pblbc_operator.DEFAULT_POLICY_PATH), help="Retention policy JSON path")
    ap.add_argument("--bus-dir", default=str(pblbc_operator.DEFAULT_BUS_DIR), help="Bus directory")
    ap.add_argument("--interval-s", type=int, default=0, help="Override update interval seconds")
    ap.add_argument("--once", action="store_true", help="Run a single update then exit")
    ap.add_argument("--emit-bus", action="store_true", help="Emit operator.pblbc.report and pblbc.cache.updated")
    ap.add_argument("--actor", default=pblbc_operator.default_actor(), help="Actor for bus emission")
    return ap.parse_args(argv)


def run_once(policy: dict, *, bus_dir: Path, emit_bus: bool, actor: str) -> None:
    report = pblbc_operator.build_report(policy, bus_dir=bus_dir, max_bytes=2_000_000)
    cache_cfg = report.get("cache") if isinstance(report.get("cache"), dict) else {}
    cache_json = Path(cache_cfg.get("json_path") or "/pluribus/.pluribus/cache/pblbc_report.json")
    cache_md = Path(cache_cfg.get("md_path") or "/pluribus/.pluribus/cache/pblbc_report.md")
    pblbc_operator.write_cache(report, json_path=cache_json, md_path=cache_md)
    if emit_bus:
        pblbc_operator.emit_bus(report, actor=actor, bus_dir=bus_dir)
        pblbc_operator.emit_cache_updated(report, actor=actor, bus_dir=bus_dir)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    policy = pblbc_operator.load_policy(Path(args.policy))
    cache_cfg = policy.get("cache") if isinstance(policy.get("cache"), dict) else {}
    interval_s = args.interval_s or int(cache_cfg.get("interval_s") or 900)
    bus_dir = Path(args.bus_dir)

    if args.once:
        run_once(policy, bus_dir=bus_dir, emit_bus=args.emit_bus, actor=args.actor)
        return 0

    while True:
        run_once(policy, bus_dir=bus_dir, emit_bus=args.emit_bus, actor=args.actor)
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
