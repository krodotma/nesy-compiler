#!/usr/bin/env python3
"""Agent Notify Router - mirror agent.notify.request (and aliases) events to HEXIS buffers.

This daemon tails the bus and delivers per-agent inbox messages when
hexis_delivery is unset ("auto"). Use pbnotify_operator.py when possible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, TextIO

sys.dont_write_bytecode = True

try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None

try:
    from nucleus.tools import pbnotify_operator as pbnotify
except Exception:
    import pbnotify_operator as pbnotify  # type: ignore

DEFAULT_TOPICS = [
    "agent.notify.request",
    "agent.notify",
    "notify.request",
    "pbnotify.request",
]
DEFAULT_BOOTSTRAP_BYTES = 65536
DEFAULT_HEXIS_TOPIC = pbnotify.DEFAULT_HEXIS_TOPIC
DEFAULT_EFFECTS = pbnotify.DEFAULT_EFFECTS
DEFAULT_LANE = pbnotify.DEFAULT_LANE
DEFAULT_TOPOLOGY = pbnotify.DEFAULT_TOPOLOGY
DEFAULT_KIND = pbnotify.DEFAULT_KIND
DEFAULT_DISPATCH_TOPIC = pbnotify.DEFAULT_DISPATCH_TOPIC


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def bus_dir_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or ".pluribus/bus").expanduser().resolve()


def bus_events_path(bus_dir: Path) -> Path:
    return bus_dir / "events.ndjson"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if agent_bus is not None:
        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
        return
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_events_path(bus_dir), event)


def _topic_matches(topic: str, patterns: list[str]) -> bool:
    if not topic:
        return False
    for pat in patterns:
        p = (pat or "").strip()
        if not p:
            continue
        if p == topic:
            return True
        if p.endswith("*") and topic.startswith(p[:-1]):
            return True
    return False


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    return []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _resolve_targets(data: dict, class_map: dict[str, list[str]]) -> tuple[list[str], str | None, bool]:
    targets: list[str] = []
    targets.extend(_string_list(data.get("targets")))
    targets.extend(_string_list(data.get("target")))
    targets.extend(_string_list(data.get("to")))
    targets.extend(_string_list(data.get("agent")))

    target_class = data.get("target_class") or data.get("class")
    if isinstance(target_class, str) and target_class:
        targets.extend(class_map.get(target_class, []))
    else:
        target_class = None

    broadcast = bool(data.get("broadcast"))
    if broadcast and not targets:
        targets.extend(class_map.get("broadcast", []))
        if not targets:
            targets.extend(class_map.get("all", []))

    return _dedupe(targets), target_class, broadcast


def _open_events_handle(path: Path, *, start_at_end: bool) -> tuple[TextIO, int]:
    handle = path.open("r", encoding="utf-8", errors="replace")
    if start_at_end:
        handle.seek(0, os.SEEK_END)
    else:
        handle.seek(0, os.SEEK_SET)
    try:
        inode = path.stat().st_ino
    except OSError:
        inode = -1
    return handle, inode


def _reopen_if_rotated(path: Path, handle: TextIO, inode: int) -> tuple[TextIO, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return handle, inode
    if stat.st_ino != inode or stat.st_size < handle.tell():
        handle.close()
        handle = path.open("r", encoding="utf-8", errors="replace")
        inode = stat.st_ino
        handle.seek(0, os.SEEK_SET)
    return handle, inode


def _load_class_map(path: str | None) -> dict[str, list[str]]:
    return pbnotify._load_class_map(path)


def _hexis_mode(data: dict) -> str:
    value = str(data.get("hexis_delivery") or "auto").strip().lower()
    if value not in {"auto", "sent", "skip"}:
        return "auto"
    return value


def _deliver_hexis(
    *,
    actor: str,
    bus_dir: Path,
    event: dict,
    data: dict,
    targets: list[str],
    target_class: str | None,
    broadcast: bool,
    kind: str,
    hexis_topic: str,
    hexis_effects: str,
    hexis_lane: str,
    hexis_topology: str,
    hexis_agent_type: str,
    delivered: set[tuple[str, str]],
) -> int:
    req_id = data.get("req_id") or data.get("request_id")
    trace_id = data.get("trace_id") or ""
    if not isinstance(req_id, str) or not req_id:
        return 0

    hexis_published: dict[str, str] = {}
    hexis_errors: list[dict[str, str]] = []

    for target in targets:
        if (req_id, target) in delivered:
            continue
        try:
            msg_id = pbnotify.publish_hexis(
                target=target,
                actor=actor,
                req_id=req_id,
                trace_id=trace_id or str(uuid.uuid4()),
                topic=hexis_topic,
                kind=kind,
                effects=hexis_effects,
                lane=hexis_lane,
                topology=hexis_topology,
                agent_type=hexis_agent_type,
                payload={
                    "message": data.get("message"),
                    "severity": data.get("severity"),
                    "bus_event_id": event.get("id"),
                    "bus_topic": event.get("topic"),
                    "payload": data.get("payload") if isinstance(data.get("payload"), dict) else {},
                    "reply_to": data.get("reply_to"),
                    "target_class": target_class,
                    "broadcast": broadcast,
                },
            )
            hexis_published[target] = msg_id
            delivered.add((req_id, target))
        except Exception as exc:
            hexis_errors.append({"target": target, "error": str(exc)})

    emit_bus(
        bus_dir,
        topic=DEFAULT_DISPATCH_TOPIC,
        kind="metric",
        level="warn" if hexis_errors else "info",
        actor=actor,
        data={
            "req_id": req_id,
            "trace_id": trace_id,
            "origin_event_id": event.get("id"),
            "origin_topic": event.get("topic"),
            "targets": targets,
            "published": hexis_published,
            "errors": hexis_errors,
            "router": True,
        },
    )
    return 1 if hexis_published or hexis_errors else 0


def process_event(
    *,
    bus_dir: Path,
    actor: str,
    event: dict,
    topics: list[str],
    class_map: dict[str, list[str]],
    hexis_topic: str,
    hexis_effects: str,
    hexis_lane: str,
    hexis_topology: str,
    hexis_agent_type: str,
    delivered: set[tuple[str, str]],
) -> int:
    topic = str(event.get("topic") or "")
    kind = str(event.get("kind") or "")
    if kind != "request":
        return 0
    if not _topic_matches(topic, topics):
        return 0
    data = event.get("data")
    if not isinstance(data, dict):
        return 0

    if _hexis_mode(data) != "auto":
        return 0

    class_map_path = data.get("class_map")
    if isinstance(class_map_path, str) and class_map_path:
        class_map = _load_class_map(class_map_path)

    targets, target_class, broadcast = _resolve_targets(data, class_map)
    if not targets:
        return 0

    return _deliver_hexis(
        actor=actor,
        bus_dir=bus_dir,
        event=event,
        data=data,
        targets=targets,
        target_class=target_class,
        broadcast=broadcast,
        kind=kind or DEFAULT_KIND,
        hexis_topic=hexis_topic,
        hexis_effects=hexis_effects,
        hexis_lane=hexis_lane,
        hexis_topology=hexis_topology,
        hexis_agent_type=hexis_agent_type,
        delivered=delivered,
    )


def _iter_recent_events(path: Path, bootstrap_bytes: int):
    if not path.exists():
        return
    if bootstrap_bytes <= 0:
        bootstrap_bytes = DEFAULT_BOOTSTRAP_BYTES
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        start = max(0, size - bootstrap_bytes)
        handle.seek(start, os.SEEK_SET)
        chunk = handle.read()
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    for line in lines:
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def run_once(
    *,
    bus_dir: Path,
    actor: str,
    topics: list[str],
    class_map: dict[str, list[str]],
    hexis_topic: str,
    hexis_effects: str,
    hexis_lane: str,
    hexis_topology: str,
    hexis_agent_type: str,
    bootstrap_bytes: int,
) -> int:
    events_path = bus_events_path(bus_dir)
    ensure_dir(events_path.parent)
    events_path.touch(exist_ok=True)
    delivered: set[tuple[str, str]] = set()
    processed = 0
    for event in _iter_recent_events(events_path, bootstrap_bytes):
        processed += process_event(
            bus_dir=bus_dir,
            actor=actor,
            event=event,
            topics=topics,
            class_map=class_map,
            hexis_topic=hexis_topic,
            hexis_effects=hexis_effects,
            hexis_lane=hexis_lane,
            hexis_topology=hexis_topology,
            hexis_agent_type=hexis_agent_type,
            delivered=delivered,
        )
    return processed


def run_daemon(
    *,
    bus_dir: Path,
    actor: str,
    topics: list[str],
    class_map: dict[str, list[str]],
    hexis_topic: str,
    hexis_effects: str,
    hexis_lane: str,
    hexis_topology: str,
    hexis_agent_type: str,
    poll_s: float,
    bootstrap_bytes: int,
) -> None:
    events_path = bus_events_path(bus_dir)
    ensure_dir(events_path.parent)
    events_path.touch(exist_ok=True)

    delivered: set[tuple[str, str]] = set()
    for event in _iter_recent_events(events_path, bootstrap_bytes):
        process_event(
            bus_dir=bus_dir,
            actor=actor,
            event=event,
            topics=topics,
            class_map=class_map,
            hexis_topic=hexis_topic,
            hexis_effects=hexis_effects,
            hexis_lane=hexis_lane,
            hexis_topology=hexis_topology,
            hexis_agent_type=hexis_agent_type,
            delivered=delivered,
        )

    handle, inode = _open_events_handle(events_path, start_at_end=True)
    while True:
        line = handle.readline()
        if not line:
            time.sleep(max(0.05, poll_s))
            handle, inode = _reopen_if_rotated(events_path, handle, inode)
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        process_event(
            bus_dir=bus_dir,
            actor=actor,
            event=event,
            topics=topics,
            class_map=class_map,
            hexis_topic=hexis_topic,
            hexis_effects=hexis_effects,
            hexis_lane=hexis_lane,
            hexis_topology=hexis_topology,
            hexis_agent_type=hexis_agent_type,
            delivered=delivered,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_notify_router.py",
        description="Route agent.notify.request (and aliases) events to HEXIS buffers when hexis_delivery=auto.",
    )
    parser.add_argument("--bus-dir", default=None)
    parser.add_argument("--actor", default=None)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    parser.add_argument("--class-map", default=None)
    parser.add_argument("--hexis-topic", default=DEFAULT_HEXIS_TOPIC)
    parser.add_argument("--hexis-effects", default=DEFAULT_EFFECTS)
    parser.add_argument("--hexis-lane", default=DEFAULT_LANE)
    parser.add_argument("--hexis-topology", default=DEFAULT_TOPOLOGY)
    parser.add_argument("--hexis-agent-type", default="worker")
    parser.add_argument("--poll", default=os.environ.get("NOTIFY_POLL_S", "0.2"))
    parser.add_argument("--bootstrap-bytes", default=os.environ.get("NOTIFY_BOOTSTRAP_BYTES", str(DEFAULT_BOOTSTRAP_BYTES)))
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = args.actor or pbnotify.default_actor()
    bus_dir = bus_dir_path(args.bus_dir)
    topics = _dedupe(_string_list(args.topics)) or DEFAULT_TOPICS
    class_map = _load_class_map(args.class_map)
    poll_s = float(args.poll or 0.2)
    bootstrap_bytes = int(args.bootstrap_bytes or DEFAULT_BOOTSTRAP_BYTES)

    if args.once:
        run_once(
            bus_dir=bus_dir,
            actor=actor,
            topics=topics,
            class_map=class_map,
            hexis_topic=args.hexis_topic,
            hexis_effects=args.hexis_effects,
            hexis_lane=args.hexis_lane,
            hexis_topology=args.hexis_topology,
            hexis_agent_type=args.hexis_agent_type,
            bootstrap_bytes=bootstrap_bytes,
        )
        return 0

    run_daemon(
        bus_dir=bus_dir,
        actor=actor,
        topics=topics,
        class_map=class_map,
        hexis_topic=args.hexis_topic,
        hexis_effects=args.hexis_effects,
        hexis_lane=args.hexis_lane,
        hexis_topology=args.hexis_topology,
        hexis_agent_type=args.hexis_agent_type,
        poll_s=poll_s,
        bootstrap_bytes=bootstrap_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
