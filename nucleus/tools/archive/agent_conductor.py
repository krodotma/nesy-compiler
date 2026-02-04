#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from agent_bus import default_actor, emit_event, resolve_bus_paths
except ImportError:  # pragma: no cover - fallback for direct invocation
    sys.path.append(str(Path(__file__).resolve().parent))
    from agent_bus import default_actor, emit_event, resolve_bus_paths

try:
    import task_ledger
except ImportError:  # pragma: no cover - fallback for direct invocation
    sys.path.append(str(Path(__file__).resolve().parent))
    import task_ledger

OPEN_STATUSES = {"planned", "in_progress", "blocked"}
STATUS_ORDER = {"blocked": 0, "in_progress": 1, "planned": 2}
PROGRESS_LOG_HEADER = "## Progress Log (append-only)"

DEFAULT_PLAN_TEMPLATE = """# Conductor Plan (living, append-only)

Purpose: maintain a single plan artifact and append progress updates from the bus
and task ledger. Do not rewrite history; only append.

## Progress Log (append-only)
"""


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_plan_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    return Path(__file__).resolve().parents[1] / "docs" / "conductor" / "plan.md"


def resolve_ledger_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    try:
        return task_ledger.default_ledger_path()
    except Exception:
        return Path(__file__).resolve().parents[2] / ".pluribus" / "index" / "task_ledger.ndjson"


def ensure_plan_exists(plan_path: Path) -> None:
    if plan_path.exists():
        return
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(DEFAULT_PLAN_TEMPLATE, encoding="utf-8")


def plan_has_progress_log(plan_path: Path) -> bool:
    if not plan_path.exists():
        return False
    with plan_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() == PROGRESS_LOG_HEADER:
                return True
    return False


def append_text(plan_path: Path, text: str) -> None:
    needs_newline = False
    if plan_path.exists() and plan_path.stat().st_size > 0:
        with plan_path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            last = handle.read(1)
            needs_newline = last not in (b"\n", b"\r")
    with plan_path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def ensure_progress_log_section(plan_path: Path) -> None:
    if plan_has_progress_log(plan_path):
        return
    append_text(plan_path, f"{PROGRESS_LOG_HEADER}\n")


def load_ledger_entries(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    entries: list[dict] = []
    with ledger_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def latest_entries_by_req_id(entries: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for entry in entries:
        req_id = entry.get("req_id")
        if not req_id:
            continue
        ts = entry.get("ts", 0)
        prev = latest.get(req_id)
        if prev is None or ts >= prev.get("ts", 0):
            latest[req_id] = entry
    return latest


def normalize_status_filter(statuses_arg: str | None) -> set[str]:
    if statuses_arg is None:
        return set(OPEN_STATUSES)
    statuses = {item.strip() for item in statuses_arg.split(",") if item.strip()}
    return statuses or set(OPEN_STATUSES)


def truncate_desc(desc: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return desc
    if len(desc) <= max_chars:
        return desc
    if max_chars <= 3:
        return desc[:max_chars]
    return desc[: max_chars - 3].rstrip() + "..."


def collect_open_tasks(
    entries: list[dict],
    max_tasks: int = 20,
    *,
    statuses: set[str] | None = None,
    max_desc_chars: int | None = 120,
) -> tuple[list[dict], dict[str, int]]:
    latest = latest_entries_by_req_id(entries)
    open_tasks: list[dict] = []
    status_counts: dict[str, int] = {}
    statuses = statuses or set(OPEN_STATUSES)
    for req_id, entry in latest.items():
        status = entry.get("status")
        if status not in statuses:
            continue
        status_counts[status] = status_counts.get(status, 0) + 1
        meta = entry.get("meta") or {}
        desc = meta.get("desc") or meta.get("note") or ""
        desc = truncate_desc(str(desc).strip(), max_desc_chars)
        desc = " ".join(desc.split())
        open_tasks.append({"req_id": req_id, "status": status, "desc": desc})

    open_tasks.sort(key=lambda item: (STATUS_ORDER.get(item["status"], 99), item["req_id"]))
    return open_tasks[:max_tasks], status_counts


def render_plan_update(
    now_iso: str,
    note: str,
    open_tasks: list[dict],
    *,
    status_counts: dict[str, int] | None = None,
    total_open: int | None = None,
) -> str:
    header = f"- {now_iso}: {note}" if note else f"- {now_iso}: sync"
    lines = [header]
    total_open = total_open if total_open is not None else sum((status_counts or {}).values())
    if total_open:
        counts = status_counts or {}
        ordered_statuses = sorted(counts.keys(), key=lambda s: STATUS_ORDER.get(s, 99))
        counts_str = ", ".join(f"{status}={counts[status]}" for status in ordered_statuses)
        summary = f"  Open tasks ({total_open})"
        if counts_str:
            summary += f": {counts_str}"
        lines.append(summary)
        for task in open_tasks:
            desc = f" - {task['desc']}" if task["desc"] else ""
            lines.append(f"  - {task['req_id']} [{task['status']}] {desc}".rstrip())
        if total_open > len(open_tasks):
            remainder = total_open - len(open_tasks)
            lines.append(f"  ... {remainder} more (increase --max-tasks to list all)")
    else:
        lines.append("  Open tasks: none")
    return "\n".join(lines) + "\n"


def append_plan_update(plan_path: Path, update_text: str) -> None:
    ensure_plan_exists(plan_path)
    ensure_progress_log_section(plan_path)
    append_text(plan_path, update_text)


def emit_plan_event(*, bus_dir: str | None, actor: str, data: dict) -> str:
    paths = resolve_bus_paths(bus_dir)
    return emit_event(
        paths,
        topic="conductor.plan.update",
        kind="artifact",
        level="info",
        actor=actor,
        data=data,
        trace_id=None,
        run_id=None,
        durable=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append Conductor plan updates from the task ledger.")
    parser.add_argument("--plan-path", help="Path to the Conductor plan file")
    parser.add_argument("--ledger-path", help="Path to task_ledger.ndjson")
    parser.add_argument("--note", default="sync", help="Short note for the update entry")
    parser.add_argument("--max-tasks", type=int, default=20, help="Max open tasks to include")
    parser.add_argument("--max-desc-chars", type=int, default=120, help="Trim task descriptions")
    parser.add_argument(
        "--statuses",
        help="Comma-separated statuses to include (default: planned,in_progress,blocked)",
    )
    parser.add_argument("--now-iso", help="Override ISO timestamp (tests/automation)")
    parser.add_argument("--dry-run", action="store_true", help="Print update without writing")
    parser.add_argument("--init", action="store_true", help="Create plan file if missing")
    parser.add_argument("--no-bus", action="store_true", help="Skip emitting bus event")
    parser.add_argument("--bus-dir", help="Override bus directory")
    parser.add_argument("--actor", help="Override actor for bus event")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = resolve_plan_path(args.plan_path)
    ledger_path = resolve_ledger_path(args.ledger_path)

    if args.init:
        ensure_plan_exists(plan_path)
        ensure_progress_log_section(plan_path)

    entries = load_ledger_entries(ledger_path)
    statuses = normalize_status_filter(args.statuses)
    open_tasks, status_counts = collect_open_tasks(
        entries,
        max_tasks=args.max_tasks,
        statuses=statuses,
        max_desc_chars=args.max_desc_chars,
    )
    total_open = sum(status_counts.values())
    now_iso = args.now_iso or now_iso_utc()
    update_text = render_plan_update(
        now_iso,
        args.note,
        open_tasks,
        status_counts=status_counts,
        total_open=total_open,
    )

    if args.dry_run:
        print(update_text.rstrip())
    else:
        append_plan_update(plan_path, update_text)

    if not args.no_bus:
        actor = args.actor or default_actor()
        emit_plan_event(
            bus_dir=args.bus_dir,
            actor=actor,
            data={
                "plan_path": str(plan_path),
                "ledger_path": str(ledger_path),
                "note": args.note,
                "open_tasks": open_tasks,
                "status_counts": status_counts,
                "total_open": total_open,
                "statuses": sorted(statuses),
                "max_tasks": args.max_tasks,
                "max_desc_chars": args.max_desc_chars,
                "update_iso": now_iso,
                "dry_run": args.dry_run,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
