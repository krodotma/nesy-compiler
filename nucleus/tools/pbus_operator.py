#!/usr/bin/env python3
from __future__ import annotations

"""
PBUS — Pluribus Bus Update operator.

Pulls the shared bus and known comm/log sources to refresh situational awareness.
Emits a non-blocking request and a compact report artifact.
"""

import argparse
import calendar
import getpass
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


DEFAULT_WINDOW_S = 900
DEFAULT_MAX_EVENTS = 200
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_LIST_LIMIT = 20


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_bus_dir() -> Path:
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_UN)


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> str:
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)
    return evt_id


def parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return True


def parse_int(value: str | int | None, default: int, minimum: int = 0) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed >= minimum else default


def iso_from_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    except Exception:
        return ""


def parse_iso_ts(value: str) -> float:
    try:
        return float(calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")))
    except Exception:
        return 0.0


def event_ts(evt: dict, now_ts: float) -> float:
    ts = evt.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    iso = str(evt.get("iso") or "").strip()
    if iso:
        parsed = parse_iso_ts(iso)
        if parsed:
            return parsed
    return now_ts


def read_bus_events(events_path: Path, *, cutoff: float, max_bytes: int, now_ts: float) -> list[dict]:
    if not events_path.exists():
        return []
    try:
        size = events_path.stat().st_size
        start = max(0, size - max_bytes)
        with events_path.open("rb") as f:
            f.seek(start)
            data = f.read()
    except Exception:
        return []
    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    events: list[dict] = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            evt = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(evt, dict):
            continue
        ts = event_ts(evt, now_ts)
        if ts < cutoff:
            continue
        events.append(evt)
    return events


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def list_recent_files(path: Path, *, limit: int, root: Path) -> list[tuple[str, str]]:
    if not path.exists() or not path.is_dir():
        return []
    entries = []
    for entry in path.iterdir():
        if entry.is_file():
            try:
                mtime = entry.stat().st_mtime
            except Exception:
                mtime = 0.0
            entries.append((entry, mtime))
    entries.sort(key=lambda item: item[1], reverse=True)
    out: list[tuple[str, str]] = []
    for entry, mtime in entries[:limit]:
        out.append((relative_to_root(entry, root), iso_from_ts(mtime)))
    return out


def sha256_text(text: str) -> tuple[str, int]:
    payload = text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def build_report(
    *,
    req_id: str,
    actor: str,
    window_s: int,
    generated_at: str,
    sources: dict[str, str],
    summary: dict[str, str | int],
    top_topics: list[tuple[str, int]],
    top_actors: list[tuple[str, int]],
    recent_events: list[str],
    recent_reports: list[tuple[str, str]],
    recent_logs: list[tuple[str, str]],
    notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append("PBUS_REPORT v1")
    lines.append("")
    lines.append("CONTEXT")
    lines.append(f"- req_id: {req_id}")
    lines.append(f"- actor: {actor}")
    lines.append(f"- window_s: {window_s}")
    lines.append(f"- generated_at: {generated_at}")
    lines.append("")
    lines.append("SOURCES")
    lines.append(f"- bus_events: {sources.get('bus_events', 'none')}")
    lines.append(f"- agent_reports: {sources.get('agent_reports', 'none')}")
    lines.append(f"- agent_logs: {sources.get('agent_logs', 'none')}")
    lines.append("")
    lines.append("BUS_SUMMARY")
    lines.append(f"- event_count: {summary.get('event_count', 0)}")
    lines.append(f"- unique_topics: {summary.get('unique_topics', 0)}")
    lines.append(f"- unique_actors: {summary.get('unique_actors', 0)}")
    lines.append(f"- window_start_iso: {summary.get('window_start_iso', '')}")
    lines.append(f"- window_end_iso: {summary.get('window_end_iso', '')}")
    lines.append("")
    lines.append("TOP_TOPICS")
    if top_topics:
        for topic, count in top_topics:
            lines.append(f"- {topic}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("TOP_ACTORS")
    if top_actors:
        for actor_name, count in top_actors:
            lines.append(f"- {actor_name}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("RECENT_EVENTS")
    if recent_events:
        lines.extend(recent_events)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("RECENT_REPORTS")
    if recent_reports:
        for path, mtime in recent_reports:
            lines.append(f"- {path} {mtime}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("RECENT_LOGS")
    if recent_logs:
        for path, mtime in recent_logs:
            lines.append(f"- {path} {mtime}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("NOTES")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbus_operator.py", description="PBUS semantic operator: pull bus + comm logs into an update report.")
    p.add_argument("--bus-dir", default=None)
    p.add_argument("--actor", default=None)
    p.add_argument("--req-id", default=None)
    p.add_argument("--subproject", default="ops", help="Subproject tag for infer_sync mirror.")
    p.add_argument("--message", default="PBUS", help="Operator message/context.")
    p.add_argument("--reason", default="operator_pbus", help="Short reason code.")
    p.add_argument("--window-s", default=str(DEFAULT_WINDOW_S), help="Time window in seconds.")
    p.add_argument("--max-events", default=str(DEFAULT_MAX_EVENTS), help="Max bus events to include.")
    p.add_argument("--max-bytes", default=str(DEFAULT_MAX_BYTES), help="Max bytes to scan from events.ndjson.")
    p.add_argument("--report-dir", default=None, help="Directory for report output (default: agent_reports).")
    p.add_argument("--report-path", default=None, help="Explicit report path (overrides --report-dir).")
    p.add_argument("--include-reports", nargs="?", const="true", default="true", help="Include agent_reports listing.")
    p.add_argument("--include-logs", nargs="?", const="true", default="true", help="Include agent_logs listing.")
    p.add_argument("--no-infer-sync", action="store_true", help="Do not mirror to infer_sync.request.")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "pbus"
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else default_bus_dir()
    ensure_dir(bus_dir)
    (bus_dir / "events.ndjson").touch(exist_ok=True)

    root = repo_root()
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else (root / "agent_reports")
    ensure_dir(report_dir)

    window_s = parse_int(args.window_s, DEFAULT_WINDOW_S, minimum=1)
    max_events = parse_int(args.max_events, DEFAULT_MAX_EVENTS, minimum=1)
    max_bytes = parse_int(args.max_bytes, DEFAULT_MAX_BYTES, minimum=1)
    include_reports = parse_bool(args.include_reports)
    include_logs = parse_bool(args.include_logs)

    req_id = args.req_id or str(uuid.uuid4())
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else (report_dir / f"pbus_report_{stamp_utc()}.txt")

    now_ts = time.time()
    cutoff = now_ts - window_s
    events_path = bus_dir / "events.ndjson"
    events = read_bus_events(events_path, cutoff=cutoff, max_bytes=max_bytes, now_ts=now_ts)

    topic_counts: Counter[str] = Counter()
    actor_counts: Counter[str] = Counter()
    recent_lines: list[str] = []
    for evt in events:
        topic = str(evt.get("topic") or "").strip()
        if topic:
            topic_counts[topic] += 1
        evt_actor = str(evt.get("actor") or "").strip()
        if evt_actor:
            actor_counts[evt_actor] += 1

    trimmed_events = events[-max_events:] if max_events > 0 else events
    for evt in trimmed_events:
        evt_ts = event_ts(evt, now_ts)
        iso = str(evt.get("iso") or "").strip() or iso_from_ts(evt_ts)
        topic = str(evt.get("topic") or "").strip()
        evt_actor = str(evt.get("actor") or "").strip()
        kind = str(evt.get("kind") or "").strip()
        level = str(evt.get("level") or "").strip()
        recent_lines.append(f"- {iso} {topic} {evt_actor} {kind} {level}".strip())

    reports_path = root / "agent_reports"
    logs_path = root / "agent_logs"
    recent_reports = list_recent_files(reports_path, limit=DEFAULT_LIST_LIMIT, root=root) if include_reports else []
    recent_logs = list_recent_files(logs_path, limit=DEFAULT_LIST_LIMIT, root=root) if include_logs else []

    sources = {
        "bus_events": relative_to_root(events_path, root),
        "agent_reports": relative_to_root(reports_path, root) if include_reports else "skipped",
        "agent_logs": relative_to_root(logs_path, root) if include_logs else "skipped",
    }
    summary = {
        "event_count": len(events),
        "unique_topics": len(topic_counts),
        "unique_actors": len(actor_counts),
        "window_start_iso": iso_from_ts(cutoff),
        "window_end_iso": iso_from_ts(now_ts),
    }
    notes = [
        f"max_events: {max_events}",
        f"max_bytes: {max_bytes}",
        f"message: {str(args.message)}",
        f"reason: {str(args.reason)}",
    ]

    report_body = build_report(
        req_id=req_id,
        actor=actor,
        window_s=window_s,
        generated_at=now_iso_utc(),
        sources=sources,
        summary=summary,
        top_topics=topic_counts.most_common(10),
        top_actors=actor_counts.most_common(10),
        recent_events=recent_lines,
        recent_reports=recent_reports,
        recent_logs=recent_logs,
        notes=notes,
    )

    ensure_dir(report_path.parent)
    report_path.write_text(report_body, encoding="utf-8")
    report_sha, report_len = sha256_text(report_body)

    report_rel = relative_to_root(report_path, root)
    payload = {
        "req_id": req_id,
        "subproject": str(args.subproject),
        "intent": "pbus",
        "window_s": window_s,
        "max_events": max_events,
        "max_bytes": max_bytes,
        "message": str(args.message),
        "reason": str(args.reason),
        "report_path": report_rel,
        "report_sha256": report_sha,
        "report_len": report_len,
        "iso": now_iso_utc(),
    }

    emit_bus(bus_dir, topic="operator.pbus.request", kind="request", level="info", actor=actor, data=payload)
    if not args.no_infer_sync:
        emit_bus(bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data=payload)
    emit_bus(bus_dir, topic="operator.pbus.report", kind="artifact", level="info", actor=actor, data=payload)

    sys.stdout.write(req_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
