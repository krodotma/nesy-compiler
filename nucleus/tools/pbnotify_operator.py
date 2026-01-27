#!/usr/bin/env python3
"""PBNOTIFY - Targeted agent notification operator.

Delivers a notification to the bus (SoR) and optionally to HEXIS buffers
for per-agent inbox delivery when the bus is busy.

Usage:
    python3 nucleus/tools/pbnotify_operator.py --message "Ping" --target gemini
    python3 nucleus/tools/pbnotify_operator.py --message "Review commit" --target codex --target claude
    python3 nucleus/tools/pbnotify_operator.py --message "Ops update" --class ops --class-map ops.json
    python3 nucleus/tools/pbnotify_operator.py --message "Broadcast" --broadcast --no-hexis
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

DEFAULT_TOPIC = "agent.notify.request"
DEFAULT_DISPATCH_TOPIC = "agent.notify.dispatch"
DEFAULT_LEVEL = "info"
DEFAULT_KIND = "request"
DEFAULT_HEXIS_TOPIC = "agent.notify"
DEFAULT_EFFECTS = "none"
DEFAULT_LANE = "strp"
DEFAULT_TOPOLOGY = "single"
DEFAULT_CLASS_MAP_ENV = ("PBNOTIFY_CLASS_MAP", "PLURIBUS_NOTIFY_CLASS_MAP")
DEFAULT_CLASS_MAP_PATH = Path(__file__).resolve().parents[1] / "config" / "agent_notify_class_map.json"


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def default_bus_dir() -> Path:
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()


def resolve_bus_events_path(bus_dir: Path) -> Path:
    try:
        from nucleus.tools import agent_bus  # type: ignore

        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        return Path(paths.events_path)
    except Exception:
        return bus_dir / "events.ndjson"


def append_ndjson(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)


def emit_bus(
    bus_dir: Path,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
    trace_id: str | None,
    run_id: str | None,
) -> str:
    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "trace_id": trace_id,
        "run_id": run_id,
        "data": data,
    }
    append_ndjson(resolve_bus_events_path(bus_dir), event)
    return event_id


def _split_csv(values: list[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _read_json(value: str | None) -> dict:
    if not value:
        return {}
    if value.strip() == "-":
        payload = sys.stdin.read()
    else:
        payload = value
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON payload must be an object")
    return data


def _read_targets_file(path: str | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Targets file not found: {file_path}")
    targets: list[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            targets.append(line)
    return targets


def _resolve_class_map_path(path: str | None) -> str | None:
    if path:
        return str(Path(path).expanduser())
    for key in DEFAULT_CLASS_MAP_ENV:
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    if DEFAULT_CLASS_MAP_PATH.exists():
        return str(DEFAULT_CLASS_MAP_PATH)
    return None


def _load_class_map(path: str | None) -> dict[str, list[str]]:
    resolved = _resolve_class_map_path(path)
    if not resolved:
        return {}
    payload = Path(resolved).expanduser().read_text(encoding="utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Class map must be a JSON object mapping class to list")
    out: dict[str, list[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            out[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            out[key] = [value.strip()] if value.strip() else []
    return out


def publish_hexis(
    *,
    target: str,
    actor: str,
    req_id: str,
    trace_id: str,
    topic: str,
    kind: str,
    effects: str,
    lane: str,
    topology: str,
    agent_type: str,
    payload: dict,
) -> str:
    try:
        from nucleus.tools import hexis_buffer  # type: ignore
    except Exception:
        import hexis_buffer  # type: ignore

    msg = hexis_buffer.HexisNotifyMessage(
        actor=actor,
        agent_type=agent_type,
        req_id=req_id,
        trace_id=trace_id,
        topic=topic,
        kind=kind,
        effects=effects,
        lane=lane,
        topology=topology,
        payload=payload,
    )

    path = hexis_buffer.buffer_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(asdict(msg), ensure_ascii=False) + "\n")
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)

    hexis_buffer.emit_bus_event(
        topic="hexis.buffer.published",
        kind="metric",
        level="debug",
        actor=actor,
        data={
            "msg_id": msg.msg_id,
            "req_id": req_id,
            "trace_id": trace_id,
            "agent": target,
            "topic": topic,
            "kind": kind,
            "effects": effects,
            "lane": lane,
            "topology": topology,
        },
    )
    return msg.msg_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pbnotify_operator.py",
        description="Targeted agent notifications with bus + HEXIS delivery.",
    )
    parser.add_argument("--message", required=True, help="Notification message")
    parser.add_argument("--data", default=None, help="JSON payload (object) or '-' for stdin")
    parser.add_argument("--target", action="append", default=[], help="Target agent (repeatable or CSV)")
    parser.add_argument("--targets-file", default=None, help="File with target agents (one per line)")
    parser.add_argument("--class", dest="target_class", default=None, help="Target class label")
    parser.add_argument("--class-map", default=None, help="JSON file mapping class -> list of agents")
    parser.add_argument("--broadcast", action="store_true", help="Broadcast without explicit targets")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Bus topic for notification request")
    parser.add_argument("--kind", default=DEFAULT_KIND, help="Bus kind (request|response|artifact|metric|log)")
    parser.add_argument("--level", default=DEFAULT_LEVEL, help="Bus level (debug|info|warn|error)")
    parser.add_argument("--reply-to", default=None, help="Optional reply topic for acknowledgements")
    parser.add_argument("--req-id", default=None, help="Request ID for correlation")
    parser.add_argument("--trace-id", default=None, help="Trace ID for correlation")
    parser.add_argument("--actor", default=None, help="Override actor (default: PLURIBUS_ACTOR)")
    parser.add_argument("--bus-dir", default=None, help="Override bus directory")
    parser.add_argument("--hexis", dest="hexis", action="store_true", default=True, help="Deliver to HEXIS buffers")
    parser.add_argument("--no-hexis", dest="hexis", action="store_false", help="Disable HEXIS delivery")
    parser.add_argument("--hexis-topic", default=DEFAULT_HEXIS_TOPIC, help="HEXIS message topic")
    parser.add_argument("--hexis-effects", default=DEFAULT_EFFECTS, help="HEXIS effects field")
    parser.add_argument("--hexis-lane", default=DEFAULT_LANE, help="HEXIS lane")
    parser.add_argument("--hexis-topology", default=DEFAULT_TOPOLOGY, help="HEXIS topology")
    parser.add_argument("--hexis-agent-type", default="worker", help="HEXIS agent_type")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    parser.add_argument("--dry-run", action="store_true", help="Print summary only; do not emit")
    parser.add_argument("--strict", action="store_true", help="Non-zero exit on delivery errors")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    targets = _split_csv(args.target)
    targets.extend(_read_targets_file(args.targets_file))
    class_map_path = _resolve_class_map_path(args.class_map)
    class_map = _load_class_map(class_map_path)
    class_targets = class_map.get(args.target_class, []) if args.target_class else []
    targets.extend(class_targets)
    targets = _dedupe(targets)

    if not targets and not args.target_class and not args.broadcast:
        print("Error: provide --target, --class, or --broadcast", file=sys.stderr)
        return 2

    actor = (args.actor or default_actor()).strip()
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else default_bus_dir()
    req_id = args.req_id or str(uuid.uuid4())
    trace_id = args.trace_id or str(uuid.uuid4())
    payload = _read_json(args.data)

    class_unresolved = bool(args.target_class and not class_targets)
    if not args.hexis:
        hexis_delivery = "skip"
    elif targets:
        hexis_delivery = "sent"
    elif args.broadcast:
        hexis_delivery = "auto"
    else:
        hexis_delivery = "skip"

    bus_data = {
        "req_id": req_id,
        "trace_id": trace_id,
        "message": args.message,
        "severity": args.level,
        "targets": targets,
        "target_class": args.target_class,
        "broadcast": bool(args.broadcast),
        "payload": payload,
        "reply_to": args.reply_to,
        "source": {
            "actor": actor,
            "pid": os.getpid(),
            "argv": argv,
        },
        "class_unresolved": class_unresolved,
        "class_map": class_map_path if (args.target_class or args.broadcast) else None,
        "hexis_delivery": hexis_delivery,
    }

    summary = {
        "req_id": req_id,
        "trace_id": trace_id,
        "targets": targets,
        "target_class": args.target_class,
        "broadcast": bool(args.broadcast),
        "bus_topic": args.topic,
        "hexis_enabled": bool(args.hexis and targets),
    }

    if args.dry_run:
        if args.json:
            print(json.dumps({"summary": summary, "bus_data": bus_data}, indent=2))
        else:
            print(json.dumps(summary, indent=2))
        return 0

    bus_event_id = emit_bus(
        bus_dir,
        topic=args.topic,
        kind=args.kind,
        level=args.level,
        actor=actor,
        data=bus_data,
        trace_id=trace_id,
        run_id=req_id,
    )
    summary["bus_event_id"] = bus_event_id

    hexis_published: dict[str, str] = {}
    hexis_errors: list[dict[str, str]] = []

    if args.hexis and targets:
        for target in targets:
            try:
                msg_id = publish_hexis(
                    target=target,
                    actor=actor,
                    req_id=req_id,
                    trace_id=trace_id,
                    topic=args.hexis_topic,
                    kind=args.kind,
                    effects=args.hexis_effects,
                    lane=args.hexis_lane,
                    topology=args.hexis_topology,
                    agent_type=args.hexis_agent_type,
                    payload={
                        "message": args.message,
                        "severity": args.level,
                        "bus_event_id": bus_event_id,
                        "bus_topic": args.topic,
                        "payload": payload,
                        "reply_to": args.reply_to,
                        "target_class": args.target_class,
                    },
                )
                hexis_published[target] = msg_id
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
                "bus_event_id": bus_event_id,
                "targets": targets,
                "published": hexis_published,
                "errors": hexis_errors,
            },
            trace_id=trace_id,
            run_id=req_id,
        )

    summary["hexis_published"] = hexis_published
    summary["hexis_errors"] = hexis_errors

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(bus_event_id)

    if args.strict and hexis_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
