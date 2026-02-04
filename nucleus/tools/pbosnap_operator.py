#!/usr/bin/env python3
"""
PBOSNAP - Pluribus Bus Snapshot Operator
========================================

Snapshot of active subagent work with dispatch/A2A/omega evidence, omega sandwich signals, and lane summary.
Generates an ASCII report and optionally emits a bus metric event.
"""
from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools import agent_bus

PLURIBUS_ROOT = Path(os.environ.get("PLURIBUS_ROOT", str(REPO_ROOT)))
DEFAULT_BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", str(PLURIBUS_ROOT / ".pluribus" / "bus")))
LANES_PATH = PLURIBUS_ROOT / "nucleus" / "state" / "lanes.json"

ACTIVITY_PREFIXES = (
    "rd.",
    "a2a.",
    "omega.",
    "infer_sync.",
    "task_ledger.",
    "dialogos.",
    "operator.pbtest",
)

OMEGA_SANDWICH_TOPICS = (
    "omega.heartbeat",
    "omega.health",
    "omega.queue.depth",
    "omega.pending.pairs",
    "omega.a2a.pending",
    "omega.providers.scan",
    "dashboard.vps.provider_status",
    "system.boot.log",
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pbosnap"


def sanitize(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def tail_lines(path: Path, *, max_bytes: int, max_lines: int | None) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    if size <= 0:
        return []
    start = max(0, size - max_bytes)
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read()
    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    if max_lines is not None and max_lines > 0:
        lines = lines[-max_lines:]
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


def filter_recent(events: Iterable[dict[str, Any]], window_s: int) -> list[dict[str, Any]]:
    cutoff = time.time() - float(window_s)
    recent: list[dict[str, Any]] = []
    for event in events:
        ts = event.get("ts")
        try:
            ts_f = float(ts)
        except Exception:
            continue
        if ts_f >= cutoff:
            recent.append(event)
    return recent


def is_activity_topic(topic: str) -> bool:
    return any(topic.startswith(prefix) for prefix in ACTIVITY_PREFIXES)


def _event_data(event: dict[str, Any] | None) -> dict[str, Any]:
    if not event:
        return {}
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _latest_by_topic(events: Iterable[dict[str, Any]], topics: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    want = set(topics)
    for event in events:
        topic = str(event.get("topic") or "")
        if topic in want:
            latest[topic] = event
    return latest


def _pending_pairs_summary(by_pair: Any) -> tuple[int, dict[str, dict[str, Any]]]:
    if not isinstance(by_pair, dict):
        return 0, {}
    out: dict[str, dict[str, Any]] = {}
    total = 0
    for pair_id, info in by_pair.items():
        if not isinstance(info, dict):
            continue
        pending = info.get("pending")
        if isinstance(pending, int) and pending > 0:
            out[str(pair_id)] = {"pending": pending, "oldest_age_s": info.get("oldest_age_s")}
            total += pending
    return total, out


def _event_rate(events: Iterable[dict[str, Any]], *, window_s: int) -> float:
    if window_s <= 0:
        return 0.0
    now = time.time()
    count = 0
    for event in events:
        try:
            ts = float(event.get("ts") or 0.0)
        except Exception:
            ts = 0.0
        if ts >= now - window_s:
            count += 1
    return count / float(window_s)


def load_lanes_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"lanes": [], "agents": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"lanes": [], "agents": []}


def compute_overall_wip(lanes: list[dict[str, Any]]) -> int:
    if not lanes:
        return 0
    total = sum(int(lane.get("wip_pct", 0) or 0) for lane in lanes)
    return total // len(lanes)


def render_meter(pct: int, width: int = 28) -> str:
    pct = max(0, min(100, int(pct)))
    filled = pct * width // 100
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {pct:>3}%"


def _normalize_targets(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


@dataclass(frozen=True)
class Snapshot:
    now_iso: str
    window_s: int
    bus_dir: str
    events_total: int
    activity_events: int
    actor_counts: dict[str, int]
    topic_prefixes: dict[str, int]
    dispatches: list[dict[str, Any]]
    a2a_events: list[dict[str, Any]]
    omega_events: list[dict[str, Any]]
    omega_sandwich: dict[str, Any]
    task_entries: list[dict[str, Any]]
    lanes: list[dict[str, Any]]
    agents: list[dict[str, Any]]
    overall_wip: int


def build_snapshot(
    *,
    bus_dir: Path,
    window_s: int,
    max_bytes: int,
    max_lines: int | None,
    limit: int,
    lanes_path: Path,
) -> Snapshot:
    events_path = bus_dir / "events.ndjson"
    lines = tail_lines(events_path, max_bytes=max_bytes, max_lines=max_lines)
    events = parse_events(lines)
    recent = filter_recent(events, window_s=window_s)

    topic_prefixes = Counter()
    actor_counts = Counter()
    activity_events = 0

    dispatches: list[dict[str, Any]] = []
    a2a_events: list[dict[str, Any]] = []
    omega_events: list[dict[str, Any]] = []
    task_entries: list[dict[str, Any]] = []
    last_error: dict[str, Any] | None = None
    worker_delta = 0

    for event in recent:
        topic = str(event.get("topic") or "")
        prefix = topic.split(".", 1)[0] if "." in topic else topic
        if prefix:
            topic_prefixes[prefix] += 1

        if is_activity_topic(topic):
            activity_events += 1
            actor = str(event.get("actor") or "unknown")
            actor_counts[actor] += 1

        if str(event.get("level") or "") == "error":
            last_error = event

        if topic.startswith("strp.worker.start"):
            worker_delta += 1
        elif topic.startswith("strp.worker.end"):
            worker_delta -= 1

        if topic == "rd.tasks.dispatch" and len(dispatches) < limit:
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            dispatches.append(
                {
                    "iso": event.get("iso"),
                    "actor": event.get("actor"),
                    "targets": _normalize_targets(data.get("targets") or data.get("target")),
                    "task_id": data.get("task_id"),
                    "intent": data.get("intent"),
                    "req_id": data.get("req_id"),
                }
            )
            continue

        if topic.startswith("a2a.") and len(a2a_events) < limit:
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            a2a_events.append(
                {
                    "iso": event.get("iso"),
                    "topic": topic,
                    "actor": event.get("actor"),
                    "target": data.get("target") or data.get("to"),
                    "req_id": data.get("req_id") or data.get("request_id"),
                    "intent": data.get("intent"),
                    "status": data.get("status"),
                }
            )
            continue

        if topic.startswith("omega.") and len(omega_events) < limit:
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            omega_events.append(
                {
                    "iso": event.get("iso"),
                    "topic": topic,
                    "actor": event.get("actor"),
                    "target": data.get("target"),
                    "req_id": data.get("req_id"),
                }
            )
            continue

        if topic == "task_ledger.append" and len(task_entries) < limit:
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            entry = data.get("entry") if isinstance(data.get("entry"), dict) else {}
            task_entries.append(
                {
                    "iso": event.get("iso"),
                    "actor": entry.get("actor") or event.get("actor"),
                    "status": entry.get("status"),
                    "topic": entry.get("topic"),
                    "req_id": entry.get("req_id"),
                    "intent": entry.get("intent"),
                }
            )

    omega_latest = _latest_by_topic(recent, OMEGA_SANDWICH_TOPICS)
    hb_data = _event_data(omega_latest.get("omega.heartbeat"))
    health_data = _event_data(omega_latest.get("omega.health"))
    queue_data = _event_data(omega_latest.get("omega.queue.depth"))
    pairs_data = _event_data(omega_latest.get("omega.pending.pairs"))
    a2a_data = _event_data(omega_latest.get("omega.a2a.pending"))
    providers_data = _event_data(omega_latest.get("omega.providers.scan"))
    provider_status_data = _event_data(omega_latest.get("dashboard.vps.provider_status"))
    boot_data = _event_data(omega_latest.get("system.boot.log"))
    pending_pairs_total, pending_pairs = _pending_pairs_summary(pairs_data.get("by_pair"))
    event_rate_10s = _event_rate(recent, window_s=10)

    lanes_state = load_lanes_state(lanes_path)
    lanes = list(lanes_state.get("lanes") or [])
    agents = list(lanes_state.get("agents") or [])
    overall_wip = compute_overall_wip(lanes)

    return Snapshot(
        now_iso=now_iso(),
        window_s=int(window_s),
        bus_dir=str(bus_dir),
        events_total=len(recent),
        activity_events=activity_events,
        actor_counts=dict(actor_counts),
        topic_prefixes=dict(topic_prefixes),
        dispatches=dispatches,
        a2a_events=a2a_events,
        omega_events=omega_events,
        omega_sandwich={
            "heartbeat_iso": omega_latest.get("omega.heartbeat", {}).get("iso") if omega_latest else None,
            "heartbeat_cycle": hb_data.get("cycle"),
            "heartbeat_uptime_s": hb_data.get("uptime_s"),
            "health_iso": omega_latest.get("omega.health", {}).get("iso") if omega_latest else None,
            "health_status": health_data.get("status"),
            "health_cycle": health_data.get("cycle"),
            "queue_pending_requests": queue_data.get("pending_requests"),
            "queue_total_events": queue_data.get("total_events"),
            "pending_pairs_total": pairs_data.get("total", pending_pairs_total),
            "pending_pairs": pending_pairs,
            "pending_a2a_total": a2a_data.get("total"),
            "providers": providers_data.get("providers"),
            "provider_status": provider_status_data,
            "boot_iso": omega_latest.get("system.boot.log", {}).get("iso") if omega_latest else None,
            "boot_message": boot_data.get("message"),
            "event_rate_10s": event_rate_10s,
            "worker_delta": worker_delta,
            "last_error_topic": last_error.get("topic") if last_error else None,
            "last_error_iso": last_error.get("iso") if last_error else None,
        },
        task_entries=task_entries,
        lanes=lanes,
        agents=agents,
        overall_wip=overall_wip,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _fmt_value(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _fmt_providers(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    enabled = [k for k, v in value.items() if v]
    return ",".join(sorted(enabled)) if enabled else "-"


def _fmt_pending_pairs(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    parts: list[str] = []
    for pair_id, info in value.items():
        if not isinstance(info, dict):
            continue
        pending = info.get("pending")
        if isinstance(pending, int):
            parts.append(f"{pair_id}={pending}")
    return ",".join(parts) if parts else "-"


def render_report(snapshot: Snapshot) -> str:
    width = 78
    lines: list[str] = []

    def hr(char: str = "=") -> str:
        return char * width

    def section(title: str) -> None:
        lines.append(hr("-"))
        lines.append(_truncate(title, width))
        lines.append(hr("-"))

    lines.append(hr("="))
    lines.append("PBOSNAP - Pluribus Subagent Snapshot".center(width))
    lines.append(hr("="))
    lines.append(f"Timestamp: {snapshot.now_iso}")
    lines.append(f"Window: last {snapshot.window_s}s | Bus: {snapshot.bus_dir}")
    lines.append(
        f"Events: {snapshot.events_total} | Activity Events: {snapshot.activity_events} | Actors: {len(snapshot.actor_counts)}"
    )
    lines.append(
        f"Dispatches: {len(snapshot.dispatches)} | A2A: {len(snapshot.a2a_events)} | Omega: {len(snapshot.omega_events)}"
    )
    lines.append(f"Lanes: {len(snapshot.lanes)} | Overall WIP: {render_meter(snapshot.overall_wip)}")
    lines.append("")

    section("OMEGA SANDWICH (webui bread)")
    omega = snapshot.omega_sandwich or {}
    if omega:
        heartbeat_iso = omega.get("heartbeat_iso") or "-"
        heartbeat_cycle = _fmt_value(omega.get("heartbeat_cycle"), digits=0)
        heartbeat_uptime = _fmt_value(omega.get("heartbeat_uptime_s"), digits=1)
        health_status = omega.get("health_status") or "-"
        health_cycle = _fmt_value(omega.get("health_cycle"), digits=0)
        queue_pending = _fmt_value(omega.get("queue_pending_requests"), digits=0)
        queue_total = _fmt_value(omega.get("queue_total_events"), digits=0)
        pending_pairs_total = _fmt_value(omega.get("pending_pairs_total"), digits=0)
        pending_pairs = _fmt_pending_pairs(omega.get("pending_pairs"))
        pending_a2a = _fmt_value(omega.get("pending_a2a_total"), digits=0)
        providers = _fmt_providers(omega.get("providers"))
        event_rate = _fmt_value(omega.get("event_rate_10s"), digits=2)
        worker_delta = _fmt_value(omega.get("worker_delta"), digits=0)
        last_error = omega.get("last_error_topic") or "-"
        last_error_iso = omega.get("last_error_iso") or "-"

        lines.append(f"  heartbeat: {heartbeat_iso} cycle={heartbeat_cycle} uptime_s={heartbeat_uptime}")
        lines.append(f"  health: {health_status} cycle={health_cycle}")
        lines.append(f"  queue: pending_requests={queue_pending} total_events={queue_total}")
        lines.append(f"  pending_pairs: total={pending_pairs_total} detail={pending_pairs}")
        lines.append(f"  pending_a2a: total={pending_a2a}")
        lines.append(f"  providers: {providers}")
        lines.append(f"  event_rate_10s: {event_rate}/s | worker_delta: {worker_delta}")
        lines.append(f"  last_error: {sanitize(str(last_error))} @ {last_error_iso}")
    else:
        lines.append("  (no omega sandwich data in window)")
    lines.append("")

    section("ACTIVITY ACTORS (recent)")
    if snapshot.actor_counts:
        for actor, count in sorted(snapshot.actor_counts.items(), key=lambda x: (-x[1], x[0]))[:12]:
            lines.append(f"  {sanitize(actor):<20} {count:>5}")
    else:
        lines.append("  (no recent activity actors)")
    lines.append("")

    section("DISPATCHES (rd.tasks.dispatch)")
    if snapshot.dispatches:
        for item in snapshot.dispatches:
            targets = ",".join([sanitize(t) for t in item.get("targets") or []]) or "-"
            intent = sanitize(str(item.get("intent") or "-"))
            task_id = sanitize(str(item.get("task_id") or "-"))
            line = (
                f"- {item.get('iso') or '-'} actor={sanitize(str(item.get('actor') or '-'))} "
                f"targets={targets} task_id={task_id} intent={intent}"
            )
            lines.append("  " + _truncate(line, width - 2))
    else:
        lines.append("  (no dispatches in window)")
    lines.append("")

    section("A2A NEGOTIATIONS")
    if snapshot.a2a_events:
        for item in snapshot.a2a_events:
            line = (
                f"- {item.get('iso') or '-'} topic={item.get('topic') or '-'} "
                f"actor={sanitize(str(item.get('actor') or '-'))} "
                f"target={sanitize(str(item.get('target') or '-'))} "
                f"status={sanitize(str(item.get('status') or '-'))}"
            )
            lines.append("  " + _truncate(line, width - 2))
    else:
        lines.append("  (no a2a events in window)")
    lines.append("")

    section("OMEGA EVENTS")
    if snapshot.omega_events:
        for item in snapshot.omega_events:
            line = (
                f"- {item.get('iso') or '-'} topic={item.get('topic') or '-'} "
                f"actor={sanitize(str(item.get('actor') or '-'))} "
                f"target={sanitize(str(item.get('target') or '-'))}"
            )
            lines.append("  " + _truncate(line, width - 2))
    else:
        lines.append("  (no omega events in window)")
    lines.append("")

    section("TASK LEDGER (recent)")
    if snapshot.task_entries:
        for item in snapshot.task_entries:
            line = (
                f"- {item.get('iso') or '-'} actor={sanitize(str(item.get('actor') or '-'))} "
                f"status={sanitize(str(item.get('status') or '-'))} "
                f"topic={sanitize(str(item.get('topic') or '-'))} req_id={sanitize(str(item.get('req_id') or '-'))}"
            )
            lines.append("  " + _truncate(line, width - 2))
    else:
        lines.append("  (no task ledger entries in window)")
    lines.append("")

    section("LANES (summary)")
    if snapshot.lanes:
        for lane in snapshot.lanes:
            line = (
                f"- {sanitize(str(lane.get('id') or lane.get('name') or '-'))} "
                f"owner={sanitize(str(lane.get('owner') or '-'))} "
                f"wip={lane.get('wip_pct', 0)} status={sanitize(str(lane.get('status') or '-'))}"
            )
            lines.append("  " + _truncate(line, width - 2))
    else:
        lines.append("  (no lanes state found)")
    lines.append("")

    lines.append(hr("="))
    lines.append("PBOSNAP complete".center(width))
    lines.append(hr("="))

    return "\n".join(lines) + "\n"


def snapshot_to_json(snapshot: Snapshot) -> str:
    payload = {
        "operator": "PBOSNAP",
        "generated": snapshot.now_iso,
        "window_s": snapshot.window_s,
        "bus_dir": snapshot.bus_dir,
        "events_total": snapshot.events_total,
        "activity_events": snapshot.activity_events,
        "actor_counts": snapshot.actor_counts,
        "topic_prefixes": snapshot.topic_prefixes,
        "dispatches": snapshot.dispatches,
        "a2a_events": snapshot.a2a_events,
        "omega_events": snapshot.omega_events,
        "omega_sandwich": snapshot.omega_sandwich,
        "task_entries": snapshot.task_entries,
        "lanes": snapshot.lanes,
        "agents": snapshot.agents,
        "overall_wip": snapshot.overall_wip,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def emit_bus(snapshot: Snapshot, *, actor: str, report_path: str | None) -> str:
    paths = agent_bus.resolve_bus_paths(str(snapshot.bus_dir))
    omega = snapshot.omega_sandwich or {}
    data = {
        "window_s": snapshot.window_s,
        "events_total": snapshot.events_total,
        "activity_events": snapshot.activity_events,
        "actors": snapshot.actor_counts,
        "dispatch_count": len(snapshot.dispatches),
        "a2a_count": len(snapshot.a2a_events),
        "omega_count": len(snapshot.omega_events),
        "lane_count": len(snapshot.lanes),
        "overall_wip": snapshot.overall_wip,
        "omega_sandwich": {
            "heartbeat_iso": omega.get("heartbeat_iso"),
            "heartbeat_cycle": omega.get("heartbeat_cycle"),
            "health_status": omega.get("health_status"),
            "queue_pending_requests": omega.get("queue_pending_requests"),
            "pending_pairs_total": omega.get("pending_pairs_total"),
            "pending_a2a_total": omega.get("pending_a2a_total"),
            "event_rate_10s": omega.get("event_rate_10s"),
            "worker_delta": omega.get("worker_delta"),
            "last_error_topic": omega.get("last_error_topic"),
        },
        "report_path": report_path,
    }
    return agent_bus.emit_event(
        paths,
        topic="operator.pbosnap.report",
        kind="metric",
        level="info",
        actor=actor,
        data=data,
        trace_id=None,
        run_id=None,
        durable=False,
    )


def build_report_path(report_dir: str, report_path: str | None) -> str:
    if report_path:
        return report_path
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return str(Path(report_dir) / f"pbosnap_{stamp}.md")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="pbosnap_operator.py", description="PBOSNAP snapshot operator")
    ap.add_argument("--bus-dir", default=str(DEFAULT_BUS_DIR), help="Bus directory (default: $PLURIBUS_BUS_DIR)")
    ap.add_argument("--window", type=int, default=1800, help="Window seconds (default 1800)")
    ap.add_argument("--max-bytes", type=int, default=2_000_000, help="Max bytes to read from bus tail")
    ap.add_argument("--max-lines", type=int, default=2000, help="Max lines from bus tail")
    ap.add_argument("--limit", type=int, default=12, help="Max entries per section")
    ap.add_argument("--lanes-path", default=str(LANES_PATH), help="Path to lanes.json")
    ap.add_argument("--json", action="store_true", help="Output JSON snapshot")
    ap.add_argument("--emit-bus", action="store_true", help="Emit operator.pbosnap.report metric")
    ap.add_argument("--report-dir", default="agent_reports", help="Directory for report output")
    ap.add_argument("--report-path", default=None, help="Explicit report path (overrides --report-dir)")
    ap.add_argument("--no-report", action="store_true", help="Skip writing report file")
    ap.add_argument("--actor", default=default_actor(), help="Actor for bus emission")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    snapshot = build_snapshot(
        bus_dir=Path(args.bus_dir),
        window_s=args.window,
        max_bytes=args.max_bytes,
        max_lines=args.max_lines,
        limit=args.limit,
        lanes_path=Path(args.lanes_path),
    )

    report_text = render_report(snapshot)
    report_path = None

    if args.json:
        sys.stdout.write(snapshot_to_json(snapshot) + "\n")
    else:
        sys.stdout.write(report_text)

    if not args.no_report:
        report_path = build_report_path(args.report_dir, args.report_path)
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text, encoding="utf-8")

    if args.emit_bus:
        event_id = emit_bus(snapshot, actor=args.actor, report_path=report_path)
        sys.stdout.write(f"\nEmitted operator.pbosnap.report (id: {event_id[:8]})\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
