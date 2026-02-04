#!/usr/bin/env python3
"""
Hexis Buffer: Ephemeral FIFO ingress spool for inter-agent communication.

Constitution: The bus remains system of record. Every consumed buffer message
MUST be mirrored as an append-only bus artifact before deletion/ack.

Usage:
    hexis_buffer.py pub --agent <name> --json '{"goal": "..."}'
    hexis_buffer.py pull --agent <name> [--max N]
    hexis_buffer.py ack --agent <name> --msg-id <uuid>
    hexis_buffer.py drain --agent <name> --limit N
    hexis_buffer.py status [--agent <name>]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

BUFFER_DIR = Path(os.environ.get("HEXIS_BUFFER_DIR", "/tmp"))


def resolve_bus_events_path() -> Path:
    """Resolve a writable bus events file, with the same fallback logic as agent_bus.py."""
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"
    try:
        import agent_bus  # type: ignore

        paths = agent_bus.resolve_bus_paths(bus_dir)
        return Path(paths.events_path)
    except Exception:
        return Path(bus_dir) / "events.ndjson"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_ts() -> float:
    return time.time()


@dataclass
class HexisMessage:
    """Infercell packet for inter-agent handoff."""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=now_ts)
    iso: str = field(default_factory=now_iso)
    actor: str = "unknown"
    agent_type: str = "worker"  # ring0|subagent|worker
    req_id: str = ""
    trace_id: str = ""
    topic: str = "hexis.message"
    kind: str = "request"  # request|response|artifact|metric
    effects: str = "none"  # none|file|network|unknown
    lane: str = "strp"  # dialogos|pbpair|strp
    topology: str = "single"  # single|star|peer_debate
    payload: dict = field(default_factory=dict)
    # Flow tracking
    flow: dict = field(default_factory=lambda: {
        "intra": [],  # same agent type
        "inter": [],  # different agent within pluribus
        "extra": [],  # external agents (codex, gemini, etc.)
    })


def buffer_path(agent: str) -> Path:
    """Get buffer file path for agent."""
    return BUFFER_DIR / f"{agent}.buffer"


def emit_bus_event(topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    """Emit event to Pluribus bus (evidence trail)."""
    event = {
        "id": str(uuid.uuid4()),
        "ts": now_ts(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    events_path = resolve_bus_events_path()
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def cmd_pub(args: argparse.Namespace) -> int:
    """Publish message to agent buffer."""
    try:
        payload = json.loads(args.json) if args.json else {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 1

    msg = HexisMessage(
        actor=args.actor or os.environ.get("PLURIBUS_ACTOR", "unknown"),
        agent_type=args.agent_type,
        req_id=args.req_id or str(uuid.uuid4()),
        trace_id=args.trace_id or str(uuid.uuid4()),
        topic=args.topic,
        kind=args.kind,
        effects=args.effects,
        lane=args.lane,
        topology=args.topology,
        payload=payload,
    )

    # Track flow
    if args.flow_intra:
        msg.flow["intra"] = args.flow_intra
    if args.flow_inter:
        msg.flow["inter"] = args.flow_inter
    if args.flow_extra:
        msg.flow["extra"] = args.flow_extra

    path = buffer_path(args.agent)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(asdict(msg), ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    # Emit bus evidence
    emit_bus_event(
        topic="hexis.buffer.published",
        kind="metric",
        level="debug",
        actor=msg.actor,
        data={
            "msg_id": msg.msg_id,
            "req_id": msg.req_id,
            "trace_id": msg.trace_id,
            "agent": args.agent,
            "topic": msg.topic,
            "kind": msg.kind,
            "effects": msg.effects,
            "lane": msg.lane,
            "topology": msg.topology,
        },
    )

    print(msg.msg_id)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Pull messages from agent buffer (FIFO, non-destructive peek)."""
    path = buffer_path(args.agent)
    if not path.exists():
        return 0

    messages = []
    with path.open("r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        for i, line in enumerate(f):
            if args.max and i >= args.max:
                break
            line = line.strip()
            if line:
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        fcntl.flock(f, fcntl.LOCK_UN)

    for msg in messages:
        print(json.dumps(msg, ensure_ascii=False))

    return 0


def cmd_ack(args: argparse.Namespace) -> int:
    """Acknowledge (destructively remove) message from buffer."""
    path = buffer_path(args.agent)
    if not path.exists():
        print("Buffer not found", file=sys.stderr)
        return 1

    remaining = []
    acked_msg = None

    with path.open("r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("msg_id") == args.msg_id:
                    acked_msg = msg
                else:
                    remaining.append(line)
            except json.JSONDecodeError:
                remaining.append(line)
        fcntl.flock(f, fcntl.LOCK_UN)

    if not acked_msg:
        print(f"Message {args.msg_id} not found", file=sys.stderr)
        return 1

    # Mirror to bus BEFORE deletion (constitutional requirement)
    emit_bus_event(
        topic="hexis.buffer.consumed",
        kind="artifact",
        level="info",
        actor=acked_msg.get("actor", "unknown"),
        data={
            "msg_id": args.msg_id,
            "req_id": acked_msg.get("req_id") or "",
            "trace_id": acked_msg.get("trace_id") or "",
            "agent": args.agent,
            "topic": acked_msg.get("topic"),
            "kind": acked_msg.get("kind"),
            "effects": acked_msg.get("effects"),
            "lane": acked_msg.get("lane"),
            "topology": acked_msg.get("topology"),
            "payload": acked_msg.get("payload"),
            "flow": acked_msg.get("flow"),
        },
    )

    # Now safe to delete
    with path.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for line in remaining:
            f.write(line + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    emit_bus_event(
        topic="hexis.buffer.acked",
        kind="metric",
        level="debug",
        actor=acked_msg.get("actor", "unknown"),
        data={
            "msg_id": args.msg_id,
            "req_id": acked_msg.get("req_id") or "",
            "trace_id": acked_msg.get("trace_id") or "",
            "agent": args.agent,
        },
    )

    print(f"Acked: {args.msg_id}")
    return 0


def cmd_drain(args: argparse.Namespace) -> int:
    """Drain up to N messages (pull + ack in one operation)."""
    path = buffer_path(args.agent)
    if not path.exists():
        return 0

    all_lines = []
    with path.open("r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        all_lines = [l.strip() for l in f if l.strip()]
        fcntl.flock(f, fcntl.LOCK_UN)

    to_drain = all_lines[:args.limit]
    remaining = all_lines[args.limit:]

    for line in to_drain:
        try:
            msg = json.loads(line)
            # Mirror to bus before deletion
            emit_bus_event(
                topic="hexis.buffer.consumed",
                kind="artifact",
                level="info",
                actor=msg.get("actor", "unknown"),
                data={
                    "msg_id": msg.get("msg_id"),
                    "req_id": msg.get("req_id") or "",
                    "trace_id": msg.get("trace_id") or "",
                    "agent": args.agent,
                    "topic": msg.get("topic"),
                    "kind": msg.get("kind"),
                    "effects": msg.get("effects"),
                    "lane": msg.get("lane"),
                    "topology": msg.get("topology"),
                    "payload": msg.get("payload"),
                    "flow": msg.get("flow"),
                },
            )
            print(json.dumps(msg, ensure_ascii=False))
        except json.JSONDecodeError:
            continue

    with path.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for line in remaining:
            f.write(line + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show buffer status."""
    if args.agent:
        agents = [args.agent]
    else:
        agents = [p.stem for p in BUFFER_DIR.glob("*.buffer")]

    status = {}
    for agent in agents:
        path = buffer_path(agent)
        if path.exists():
            with path.open("r") as f:
                lines = [l for l in f if l.strip()]
            status[agent] = {
                "pending": len(lines),
                "path": str(path),
            }
            if lines:
                try:
                    oldest = json.loads(lines[0])
                    status[agent]["oldest_iso"] = oldest.get("iso")
                    status[agent]["oldest_topic"] = oldest.get("topic")
                except json.JSONDecodeError:
                    pass

    print(json.dumps(status, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hexis_buffer.py",
        description="Ephemeral FIFO buffer for inter-agent handoff (bus remains SoR).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # pub
    pub = sub.add_parser("pub", help="Publish message to agent buffer")
    pub.add_argument("--agent", required=True, help="Target agent name")
    pub.add_argument("--json", default="{}", help="Payload JSON")
    pub.add_argument("--actor", default=None, help="Source actor")
    pub.add_argument("--agent-type", default="worker", help="ring0|subagent|worker")
    pub.add_argument("--req-id", default=None, help="Request ID for correlation")
    pub.add_argument("--trace-id", default=None, help="Trace ID for distributed tracing")
    pub.add_argument("--topic", default="hexis.message", help="Topic")
    pub.add_argument("--kind", default="request", help="request|response|artifact|metric")
    pub.add_argument("--effects", default="none", help="none|file|network|unknown")
    pub.add_argument("--lane", default="strp", help="dialogos|pbpair|strp")
    pub.add_argument("--topology", default="single", help="single|star|peer_debate")
    pub.add_argument("--flow-intra", nargs="*", default=[], help="Intra-agent flows")
    pub.add_argument("--flow-inter", nargs="*", default=[], help="Inter-agent flows")
    pub.add_argument("--flow-extra", nargs="*", default=[], help="Extra-agent flows")
    pub.set_defaults(func=cmd_pub)

    # pull
    pull = sub.add_parser("pull", help="Pull messages from buffer (non-destructive)")
    pull.add_argument("--agent", required=True, help="Agent name")
    pull.add_argument("--max", type=int, default=None, help="Max messages to pull")
    pull.set_defaults(func=cmd_pull)

    # ack
    ack = sub.add_parser("ack", help="Acknowledge (remove) message")
    ack.add_argument("--agent", required=True, help="Agent name")
    ack.add_argument("--msg-id", required=True, help="Message ID to ack")
    ack.set_defaults(func=cmd_ack)

    # drain
    drain = sub.add_parser("drain", help="Drain N messages (pull + ack)")
    drain.add_argument("--agent", required=True, help="Agent name")
    drain.add_argument("--limit", type=int, required=True, help="Max messages to drain")
    drain.set_defaults(func=cmd_drain)

    # status
    status = sub.add_parser("status", help="Show buffer status")
    status.add_argument("--agent", default=None, help="Specific agent (or all)")
    status.set_defaults(func=cmd_status)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
