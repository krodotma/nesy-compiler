#!/usr/bin/env python3
"""
PBLBC - Pluribus Log/Buffer/Cache Report
========================================

Snapshot of active log/buffer/cache footprints with hygiene outlook.
Reads from a fixed cache (updated on a 15-minute cadence) or refreshes it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools import agent_bus

DEFAULT_POLICY_PATH = REPO_ROOT / "nucleus" / "specs" / "pblbc_retention_policy.json"
DEFAULT_BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_ts() -> float:
    return time.time()


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pblbc"


def human_bytes(value: int) -> str:
    if value <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}".rstrip("0").rstrip(".") + unit
        size /= 1024.0
    return f"{int(value)}B"


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def du_bytes(path: Path) -> int:
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True)
        raw = out.strip().split()[0]
        return int(raw)
    except Exception:
        return 0


def stat_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
    except Exception:
        return 0
    return 0


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return stat_bytes(path)
    size = du_bytes(path)
    if size:
        return size
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except Exception:
                    continue
    except Exception:
        return 0
    return total


def tail_lines(path: Path, *, max_bytes: int) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    start = max(0, size - max_bytes)
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read()
    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    return [ln.decode("utf-8", errors="replace") for ln in lines]


def parse_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        line = (line or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("topic"):
            events.append(obj)
    return events


def recent_hygiene_events(bus_dir: Path, topics: list[str], *, max_bytes: int) -> list[dict[str, Any]]:
    if not topics:
        return []
    events_path = bus_dir / "events.ndjson"
    events = parse_events(tail_lines(events_path, max_bytes=max_bytes))
    last_by_topic: dict[str, dict[str, Any]] = {}
    for event in events:
        topic = str(event.get("topic") or "")
        if topic in topics:
            last_by_topic[topic] = event
    recent = []
    for topic in topics:
        event = last_by_topic.get(topic)
        if event:
            recent.append(
                {
                    "topic": topic,
                    "iso": event.get("iso"),
                    "actor": event.get("actor"),
                }
            )
    return recent


def list_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    files: list[Path] = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            files.append(Path(root) / name)
    return files


def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def file_age_days(path: Path) -> float:
    return max(0.0, (time.time() - file_mtime(path)) / 86400.0)


def build_purge_plan(policy: dict[str, Any], *, include_candidates: bool = False) -> dict[str, Any]:
    retention = policy.get("retention") if isinstance(policy.get("retention"), dict) else {}
    qa_dir = Path("/pluribus/.pluribus/index/qa")
    bus_archive_dir = Path("/pluribus/.pluribus/bus/archive")
    fallback_dir = Path("/pluribus/.pluribus_local/bus")

    qa_glob = str(retention.get("qa_backups_glob") or "qa_ir.ndjson.*.bak")
    qa_keep = int(retention.get("qa_keep_last") or 0)
    archive_keep = int(retention.get("bus_archive_keep_last") or 0)
    archive_max_age = float(retention.get("bus_archive_max_age_days") or 0)
    fallback_keep_days = float(retention.get("fallback_bus_keep_days") or 0)

    qa_files = sorted(qa_dir.glob(qa_glob), key=file_mtime, reverse=True)
    qa_candidates = qa_files[qa_keep:] if qa_keep > 0 else qa_files

    archive_files = sorted(list_files(bus_archive_dir), key=file_mtime, reverse=True)
    archive_candidates = archive_files[archive_keep:] if archive_keep > 0 else archive_files
    if archive_max_age > 0:
        archive_candidates = [f for f in archive_candidates if file_age_days(f) >= archive_max_age]

    fallback_files = list_files(fallback_dir)
    fallback_candidates = []
    if fallback_keep_days > 0:
        fallback_candidates = [f for f in fallback_files if file_age_days(f) >= fallback_keep_days]

    def summarize(items: list[Path]) -> dict[str, Any]:
        total = 0
        sample: list[dict[str, Any]] = []
        for item in items:
            size = stat_bytes(item) if item.is_file() else 0
            total += size
            if len(sample) < 5:
                sample.append(
                    {
                        "path": str(item),
                        "bytes": size,
                        "human": human_bytes(size),
                        "age_days": round(file_age_days(item), 2),
                    }
                )
        return {
            "count": len(items),
            "total_bytes": total,
            "total_human": human_bytes(total),
            "sample": sample,
            "candidates": [str(p) for p in items] if include_candidates else [],
        }

    plan = {
        "qa_backups": {
            "keep_last": qa_keep,
            "glob": qa_glob,
            "summary": summarize(qa_candidates),
        },
        "bus_archive": {
            "keep_last": archive_keep,
            "max_age_days": archive_max_age,
            "summary": summarize(archive_candidates),
        },
        "fallback_bus": {
            "keep_days": fallback_keep_days,
            "summary": summarize(fallback_candidates),
        },
    }
    expected = sum(item["summary"]["total_bytes"] for item in plan.values())
    plan["expected_reduction_bytes"] = expected
    plan["expected_reduction_human"] = human_bytes(expected)
    return plan


def build_report(
    policy: dict[str, Any],
    *,
    bus_dir: Path,
    max_bytes: int,
    report_ts: float | None = None,
) -> dict[str, Any]:
    report_ts = report_ts or now_ts()
    cache_cfg = policy.get("cache") if isinstance(policy.get("cache"), dict) else {}
    paths_cfg = policy.get("paths") if isinstance(policy.get("paths"), list) else []
    bus_topics = policy.get("bus_topics") if isinstance(policy.get("bus_topics"), dict) else {}
    recent_topics = bus_topics.get("recent_hygiene") if isinstance(bus_topics.get("recent_hygiene"), list) else []

    sizes: list[dict[str, Any]] = []
    total_bytes = 0
    for entry in paths_cfg:
        if not isinstance(entry, dict):
            continue
        if not entry.get("active", True):
            continue
        path = Path(str(entry.get("path") or "")).expanduser()
        size = path_size_bytes(path)
        total_bytes += size
        sizes.append(
            {
                "id": entry.get("id"),
                "label": entry.get("label") or entry.get("id"),
                "path": str(path),
                "kind": entry.get("kind"),
                "exists": path.exists(),
                "bytes": size,
                "human": human_bytes(size),
            }
        )

    sizes_sorted = sorted(sizes, key=lambda x: int(x.get("bytes") or 0), reverse=True)
    purge_plan = build_purge_plan(policy)

    recent = recent_hygiene_events(bus_dir, [str(t) for t in recent_topics], max_bytes=max_bytes)

    next_hygiene = [
        {
            "id": "qa_backups_prune",
            "title": f"QA backups keep_last={purge_plan['qa_backups']['keep_last']}",
            "expected_reduction_bytes": purge_plan["qa_backups"]["summary"]["total_bytes"],
            "expected_reduction_human": purge_plan["qa_backups"]["summary"]["total_human"],
        },
        {
            "id": "bus_archive_prune",
            "title": f"Bus archive keep_last={purge_plan['bus_archive']['keep_last']}, max_age_days={purge_plan['bus_archive']['max_age_days']}",
            "expected_reduction_bytes": purge_plan["bus_archive"]["summary"]["total_bytes"],
            "expected_reduction_human": purge_plan["bus_archive"]["summary"]["total_human"],
        },
        {
            "id": "fallback_bus_prune",
            "title": f"Fallback bus keep_days={purge_plan['fallback_bus']['keep_days']}",
            "expected_reduction_bytes": purge_plan["fallback_bus"]["summary"]["total_bytes"],
            "expected_reduction_human": purge_plan["fallback_bus"]["summary"]["total_human"],
        },
    ]

    report = {
        "operator": "PBLBC",
        "generated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(report_ts)),
        "generated_ts": report_ts,
        "cache": {
            "json_path": cache_cfg.get("json_path"),
            "md_path": cache_cfg.get("md_path"),
            "interval_s": cache_cfg.get("interval_s"),
            "stale_after_s": cache_cfg.get("stale_after_s"),
        },
        "totals": {
            "active_bytes": total_bytes,
            "active_human": human_bytes(total_bytes),
            "expected_reduction_bytes": purge_plan["expected_reduction_bytes"],
            "expected_reduction_human": purge_plan["expected_reduction_human"],
        },
        "sizes": sizes_sorted,
        "hygiene": {
            "recent": recent,
            "next": next_hygiene,
            "expected_reduction_bytes": purge_plan["expected_reduction_bytes"],
            "expected_reduction_human": purge_plan["expected_reduction_human"],
        },
        "purge_plan": purge_plan,
        "policy": {
            "updated_iso": policy.get("updated_iso"),
            "path": str(DEFAULT_POLICY_PATH),
        },
    }
    return report


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"PBLBC report @ {report.get('generated_iso')}")
    totals = report.get("totals", {})
    lines.append(f"active_total: {totals.get('active_human')} (expected_reduction={totals.get('expected_reduction_human')})")
    policy = report.get("policy", {})
    lines.append(f"policy: {policy.get('path')} updated={policy.get('updated_iso')}")

    lines.append("")
    lines.append("SIZES")
    for item in report.get("sizes", [])[:12]:
        label = item.get("label") or item.get("id")
        lines.append(f"- {label}: {item.get('human')} ({item.get('path')})")

    lines.append("")
    lines.append("HYGIENE_RECENT")
    recent = report.get("hygiene", {}).get("recent", [])
    if recent:
        for item in recent:
            lines.append(f"- {item.get('topic')} @ {item.get('iso')} actor={item.get('actor')}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("HYGIENE_NEXT")
    for item in report.get("hygiene", {}).get("next", []):
        lines.append(f"- {item.get('title')} expected_reduction={item.get('expected_reduction_human')}")

    return "\n".join(lines) + "\n"


def write_cache(report: dict[str, Any], *, json_path: Path, md_path: Path | None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_report(report), encoding="utf-8")


def load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def emit_bus(report: dict[str, Any], *, actor: str, bus_dir: Path) -> str:
    paths = agent_bus.resolve_bus_paths(str(bus_dir))
    data = {
        "generated_iso": report.get("generated_iso"),
        "active_bytes": report.get("totals", {}).get("active_bytes"),
        "expected_reduction_bytes": report.get("totals", {}).get("expected_reduction_bytes"),
        "cache_path": report.get("cache", {}).get("json_path"),
    }
    return agent_bus.emit_event(
        paths,
        topic="operator.pblbc.report",
        kind="metric",
        level="info",
        actor=actor,
        data=data,
        trace_id=None,
        run_id=None,
        durable=False,
    )


def emit_cache_updated(report: dict[str, Any], *, actor: str, bus_dir: Path) -> str:
    paths = agent_bus.resolve_bus_paths(str(bus_dir))
    data = {
        "generated_iso": report.get("generated_iso"),
        "cache_path": report.get("cache", {}).get("json_path"),
    }
    return agent_bus.emit_event(
        paths,
        topic="pblbc.cache.updated",
        kind="artifact",
        level="info",
        actor=actor,
        data=data,
        trace_id=None,
        run_id=None,
        durable=False,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="pblbc_operator.py", description="PBLBC log/buffer/cache report")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY_PATH), help="Retention policy JSON path")
    ap.add_argument("--bus-dir", default=str(DEFAULT_BUS_DIR), help="Bus directory")
    ap.add_argument("--cache-json", default=None, help="Override cache JSON path")
    ap.add_argument("--cache-md", default=None, help="Override cache MD path")
    ap.add_argument("--refresh-cache", action="store_true", help="Rebuild cache before reporting")
    ap.add_argument("--max-bytes", type=int, default=2_000_000, help="Max bus bytes to read for hygiene tail")
    ap.add_argument("--emit-bus", action="store_true", help="Emit operator.pblbc.report bus metric")
    ap.add_argument("--json", action="store_true", help="Output JSON report")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout")
    ap.add_argument("--actor", default=default_actor(), help="Actor for bus emissions")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    policy = load_policy(Path(args.policy))
    cache_cfg = policy.get("cache") if isinstance(policy.get("cache"), dict) else {}
    cache_json = Path(args.cache_json or cache_cfg.get("json_path") or "/pluribus/.pluribus/cache/pblbc_report.json")
    cache_md = Path(args.cache_md or cache_cfg.get("md_path") or "/pluribus/.pluribus/cache/pblbc_report.md")

    report = None
    if args.refresh_cache:
        report = build_report(policy, bus_dir=Path(args.bus_dir), max_bytes=int(args.max_bytes))
        write_cache(report, json_path=cache_json, md_path=cache_md)
        if args.emit_bus:
            emit_bus(report, actor=args.actor, bus_dir=Path(args.bus_dir))
            emit_cache_updated(report, actor=args.actor, bus_dir=Path(args.bus_dir))
    else:
        report = load_cache(cache_json)
        if report is None:
            report = build_report(policy, bus_dir=Path(args.bus_dir), max_bytes=int(args.max_bytes))
            write_cache(report, json_path=cache_json, md_path=cache_md)
            if args.emit_bus:
                emit_bus(report, actor=args.actor, bus_dir=Path(args.bus_dir))
                emit_cache_updated(report, actor=args.actor, bus_dir=Path(args.bus_dir))

    if report is None:
        return 1

    if not args.quiet:
        if args.json:
            sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(render_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
