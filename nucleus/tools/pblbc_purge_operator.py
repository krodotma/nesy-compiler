#!/usr/bin/env python3
"""
PBLBCPURGE - Pluribus Log/Buffer/Cache Purge
============================================

Generates a purge plan from policy and optionally applies it.
Default is dry-run with expected size reductions.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools import agent_bus
from nucleus.tools import pblbc_operator


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pblbc-purge"


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="pblbc_purge_operator.py", description="PBLBC purge operator")
    ap.add_argument("--policy", default=str(pblbc_operator.DEFAULT_POLICY_PATH), help="Retention policy JSON path")
    ap.add_argument("--bus-dir", default=str(pblbc_operator.DEFAULT_BUS_DIR), help="Bus directory")
    ap.add_argument("--targets", default="qa_backups,bus_archive,fallback_bus", help="Comma-separated targets")
    ap.add_argument("--apply", action="store_true", help="Apply purge (default is dry-run)")
    ap.add_argument("--emit-bus", action="store_true", help="Emit operator.pblbc.purge bus events")
    ap.add_argument("--refresh-cache", action="store_true", help="Refresh PBLBC cache after purge")
    ap.add_argument("--actor", default=default_actor(), help="Actor for bus emission")
    return ap.parse_args(argv)


def collect_targets(plan: dict[str, dict], targets: list[str]) -> dict[str, dict]:
    out = {}
    for target in targets:
        key = target.strip()
        if not key:
            continue
        if key in plan:
            out[key] = plan[key]
    return out


def purge_files(paths: list[str]) -> tuple[int, int]:
    removed = 0
    failed = 0
    for path in paths:
        try:
            p = Path(path)
            if p.exists() and p.is_file():
                p.unlink()
                removed += 1
        except Exception:
            failed += 1
    return removed, failed


def emit_bus(topic: str, *, actor: str, bus_dir: Path, data: dict, kind: str = "request") -> str:
    paths = agent_bus.resolve_bus_paths(str(bus_dir))
    return agent_bus.emit_event(
        paths,
        topic=topic,
        kind=kind,
        level="info",
        actor=actor,
        data=data,
        trace_id=None,
        run_id=None,
        durable=False,
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    policy = pblbc_operator.load_policy(Path(args.policy))
    plan = pblbc_operator.build_purge_plan(policy, include_candidates=True)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    selected = collect_targets(plan, targets)

    expected = sum(item["summary"]["total_bytes"] for item in selected.values())
    expected_human = pblbc_operator.human_bytes(expected)

    if args.emit_bus:
        emit_bus(
            "operator.pblbc.purge",
            actor=args.actor,
            bus_dir=Path(args.bus_dir),
            kind="request",
            data={
                "mode": "apply" if args.apply else "dry-run",
                "targets": targets,
                "expected_reduction_bytes": expected,
                "expected_reduction_human": expected_human,
                "policy": str(args.policy),
            },
        )

    removed_total = 0
    failed_total = 0
    results = {}
    if args.apply:
        for key, item in selected.items():
            candidates = item["summary"].get("candidates", [])
            removed, failed = purge_files(candidates)
            removed_total += removed
            failed_total += failed
            results[key] = {"removed": removed, "failed": failed}

    if args.emit_bus:
        emit_bus(
            "operator.pblbc.purge.report",
            actor=args.actor,
            bus_dir=Path(args.bus_dir),
            kind="metric",
            data={
                "mode": "apply" if args.apply else "dry-run",
                "targets": targets,
                "expected_reduction_bytes": expected,
                "expected_reduction_human": expected_human,
                "removed_total": removed_total,
                "failed_total": failed_total,
                "results": results,
            },
        )

    if args.refresh_cache:
        report = pblbc_operator.build_report(policy, bus_dir=Path(args.bus_dir), max_bytes=2_000_000)
        cache_cfg = report.get("cache") if isinstance(report.get("cache"), dict) else {}
        cache_json = Path(cache_cfg.get("json_path") or "/pluribus/.pluribus/cache/pblbc_report.json")
        cache_md = Path(cache_cfg.get("md_path") or "/pluribus/.pluribus/cache/pblbc_report.md")
        pblbc_operator.write_cache(report, json_path=cache_json, md_path=cache_md)

    sys.stdout.write(f"PBLBCPURGE {'applied' if args.apply else 'dry-run'} expected_reduction={expected_human}\n")
    if args.apply:
        sys.stdout.write(f"removed={removed_total} failed={failed_total}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
