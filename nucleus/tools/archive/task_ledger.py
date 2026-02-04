#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools import agent_bus

STATUS_VALUES = {
    "planned",
    "in_progress",
    "blocked",
    "completed",
    "abandoned",
}

REQUIRED_FIELDS = ("req_id", "actor", "topic", "status")
LEDGER_FILE_MODE = 0o666


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_state_dir(*, for_write: bool) -> Path:
    repo = repo_root()
    primary = repo / ".pluribus" / "index"
    fallback = repo / ".pluribus_local" / "index"

    if for_write:
        try:
            primary.mkdir(parents=True, exist_ok=True)
            ledger_path = primary / "task_ledger.ndjson"
            fd = os.open(str(ledger_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, LEDGER_FILE_MODE)
            try:
                os.fchmod(fd, LEDGER_FILE_MODE)
            except (AttributeError, PermissionError, OSError):
                _ensure_mode(ledger_path, LEDGER_FILE_MODE)
            finally:
                os.close(fd)
            return primary
        except (PermissionError, OSError):
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    primary_path = primary / "task_ledger.ndjson"
    fallback_path = fallback / "task_ledger.ndjson"
    if primary_path.exists() and fallback_path.exists():
        return primary if primary_path.stat().st_mtime >= fallback_path.stat().st_mtime else fallback
    if primary_path.exists():
        return primary
    if fallback_path.exists():
        return fallback
    return primary


def default_ledger_path(*, for_write: bool = False) -> Path:
    return resolve_state_dir(for_write=for_write) / "task_ledger.ndjson"


def _append_line(path: Path, line: str) -> None:
    agent_bus.append_line(str(path), line, durable=False)
    _ensure_mode(path, LEDGER_FILE_MODE)


def _ensure_mode(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except (PermissionError, OSError):
        return


def normalize_entry(entry: dict, *, run_id: str | None = None) -> dict:
    missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if entry["status"] not in STATUS_VALUES:
        raise ValueError(f"Invalid status '{entry['status']}'")

    normalized = dict(entry)
    normalized.setdefault("id", str(uuid.uuid4()))
    normalized.setdefault("ts", time.time())
    normalized.setdefault("iso", now_iso_utc())
    if run_id and not normalized.get("run_id"):
        normalized["run_id"] = run_id
    return normalized


def append_entry(
    entry: dict,
    *,
    ledger_path: Path | None = None,
    emit_bus: bool = True,
    bus_dir: str | None = None,
    run_id: str | None = None,
) -> dict:
    path = ledger_path or default_ledger_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_entry(entry, run_id=run_id)

    line = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n"
    _append_line(path, line)

    if emit_bus:
        paths = agent_bus.resolve_bus_paths(bus_dir)
        agent_bus.emit_event(
            paths,
            topic="task_ledger.append",
            kind="artifact",
            level="info",
            actor=normalized["actor"],
            data={"entry": normalized, "path": str(path)},
            trace_id=None,
            run_id=normalized.get("run_id"),
            durable=False,
        )

    return normalized


def read_entries(
    ledger_path: Path,
    *,
    actor: str | None = None,
    topic: str | None = None,
    status: str | None = None,
    req_id: str | None = None,
    limit: int | None = None,
):
    if not ledger_path.exists():
        return []

    entries: list[dict] = []
    with ledger_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if actor and obj.get("actor") != actor:
                continue
            if topic and obj.get("topic") != topic:
                continue
            if status and obj.get("status") != status:
                continue
            if req_id and obj.get("req_id") != req_id:
                continue
            entries.append(obj)
            if limit and len(entries) >= limit:
                break
    return entries


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="task_ledger.py", description="Append-only task ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)

    append = sub.add_parser("append", help="Append a task ledger entry")
    append.add_argument("--req-id", required=True)
    append.add_argument("--actor", default=os.environ.get("PLURIBUS_ACTOR") or "unknown")
    append.add_argument("--topic", required=True)
    append.add_argument("--status", required=True)
    append.add_argument("--intent")
    append.add_argument("--run-id")
    append.add_argument("--meta", help="JSON object for additional metadata")
    append.add_argument("--ledger-path")
    append.add_argument("--no-bus", action="store_true")
    append.add_argument("--bus-dir")

    tail = sub.add_parser("tail", help="Read recent task entries")
    tail.add_argument("--ledger-path")
    tail.add_argument("--limit", type=int, default=50)
    tail.add_argument("--actor")
    tail.add_argument("--topic")
    tail.add_argument("--status")
    tail.add_argument("--req-id")

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "append":
        meta = json.loads(args.meta) if args.meta else None
        entry = {
            "req_id": args.req_id,
            "actor": args.actor,
            "topic": args.topic,
            "status": args.status,
        }
        if args.intent:
            entry["intent"] = args.intent
        if meta:
            entry["meta"] = meta
        ledger_path = Path(args.ledger_path) if args.ledger_path else None
        normalized = append_entry(
            entry,
            ledger_path=ledger_path,
            emit_bus=not args.no_bus,
            bus_dir=args.bus_dir,
            run_id=args.run_id,
        )
        print(json.dumps(normalized, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "tail":
        ledger_path = Path(args.ledger_path) if args.ledger_path else default_ledger_path()
        entries = read_entries(
            ledger_path,
            actor=args.actor,
            topic=args.topic,
            status=args.status,
            req_id=args.req_id,
            limit=args.limit,
        )
        for entry in entries:
            print(json.dumps(entry, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
