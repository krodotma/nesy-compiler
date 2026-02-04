#!/usr/bin/env python3
from __future__ import annotations

"""
RD Tasks Dispatch — bus-first task assignment program (no messenger).

This tool makes `rd.tasks.dispatch` real:
- Emits `rd.tasks.dispatch` (kind=request) with a schema-valid payload + req_id.
- Also mirrors to `infer_sync.request` (intent=rd.tasks.dispatch) so existing
  responders/monitors can surface pending work.

See:
- nucleus/specs/realagents_upgrade_v1.md
- nucleus/specs/rd_tasks_dispatch.schema.json
"""

import argparse
import getpass
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> str:
    event_id = str(uuid.uuid4())
    evt = {
        "id": event_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)
    return event_id


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_default_reagents_upgrade(spec_ref: str, *, req_id: str, iso: str, targets: list[str], message: str) -> dict[str, Any]:
    return {
        "req_id": req_id,
        "task_id": "REALAGENTS_upgrade",
        "intent": "realagents_upgrade",
        "iso": iso,
        "spec_ref": spec_ref,
        "keywords": ["REALAGENTS", "MCP", "A2A", "ADK", "DKIN", "MBAD", "MABSWARM"],
        "targets": targets,
        "constraints": {"append_only": True, "non_blocking": True, "tests_first": True, "no_secrets": True},
        "gates": ["P", "E", "L", "R", "Q", "Ω"],
        "tasks": [
            {
                "id": "T0",
                "title": "Make rd.tasks.dispatch real (dispatcher+responder+CKIN surface)",
                "depends_on": [],
                "deliverables": [
                    "nucleus/tools/rd_tasks_dispatch.py",
                    "nucleus/tools/rd_tasks_responder.py",
                    "ckin_report surfaces dispatch+ack counts",
                ],
                "acceptance": [
                    "rd.tasks.dispatch emits schema-valid payload with req_id",
                    "rd.tasks.ack emitted by responder with same req_id",
                    "CKIN shows last dispatch age + ack ratio",
                ],
                "risk": "low",
                "notes": message,
            },
            {
                "id": "T1",
                "title": "MCP deepening (bus-evidenced tool calls + inventory)",
                "depends_on": ["T0"],
                "deliverables": ["mcp.inventory artifact", "mcp host smoke tests", "effects mapped per tool"],
                "acceptance": [
                    "mcp.host.call → mcp.host.response loop works for built-in servers",
                    "inventory lists servers/tools/effects with provenance",
                ],
                "risk": "med",
                "notes": "",
            },
            {
                "id": "T2",
                "title": "A2A deepening (capabilities + negotiate/decline/redirect)",
                "depends_on": ["T0"],
                "deliverables": ["a2a.capabilities.*", "a2a.negotiate.*", "toy negotiation test"],
                "acceptance": [
                    "non-blocking negotiation yields explicit decline/redirect (no silent drops)",
                    "payloads align with a2a_request_taxonomy kinds",
                ],
                "risk": "med",
                "notes": "",
            },
            {
                "id": "T3",
                "title": "ADK deepening (flow.py DSL + roundtrip + InferCell binding)",
                "depends_on": ["T0"],
                "deliverables": ["sdk/flow.py expanded", "roundtrip tests", "InferCell mapping"],
                "acceptance": ["flowfile ↔ graph spec roundtrip invariant passes", "bindings emit bus evidence"],
                "risk": "high",
                "notes": "",
            },
        ],
    }


def validate_against_schema(payload: dict[str, Any], schema_path: Path) -> tuple[bool, str | None]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return True, None  # optional dependency; schema validity is still enforced by tests in-repo
    schema = load_json(schema_path)
    try:
        jsonschema.validate(payload, schema)
        return True, None
    except Exception as e:
        return False, str(e)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rd_tasks_dispatch.py", description="Publish rd.tasks.dispatch tasks to the Pluribus bus.")
    p.add_argument("--bus-dir", default=None)
    p.add_argument("--actor", default=None)
    p.add_argument("--task", default="REALAGENTS_upgrade", help="Task template to dispatch.")
    p.add_argument("--targets", default="claude,codex,gemini", help="Comma-separated targets.")
    p.add_argument("--message", default="Replace shims with deep implementations (MCP/A2A/ADK) under DKIN/MBAD membrane.", help="Optional message/notes.")
    p.add_argument("--spec-ref", default="nucleus/specs/realagents_upgrade_v1.md", help="Spec anchor path.")
    
    # Calculate default schema path relative to this script
    script_dir = Path(__file__).resolve().parent
    # Assuming standard layout: tools/rd_tasks_dispatch.py -> specs/rd_tasks_dispatch.schema.json is ../../specs/... 
    # But wait, looking at repo structure:
    # /pluribus/nucleus/tools/rd_tasks_dispatch.py
    # /pluribus/nucleus/specs/rd_tasks_dispatch.schema.json
    # So it is ../specs/rd_tasks_dispatch.schema.json relative to tools/
    default_schema = script_dir.parent / "specs" / "rd_tasks_dispatch.schema.json"
    
    p.add_argument("--schema", default=str(default_schema), help="Schema path (best-effort validation).")
    p.add_argument("--no-infer-sync", action="store_true", help="Do not mirror to infer_sync.request.")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "rd-dispatch"
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()
    ensure_dir(bus_dir)
    (bus_dir / "events.ndjson").touch(exist_ok=True)

    req_id = str(uuid.uuid4())
    iso = now_iso_utc()
    targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    spec_ref = str(args.spec_ref)

    if args.task != "REALAGENTS_upgrade":
        sys.stderr.write(f"unknown task template: {args.task}\n")
        return 2

    payload = build_default_reagents_upgrade(spec_ref, req_id=req_id, iso=iso, targets=targets, message=str(args.message))
    ok, err = validate_against_schema(payload, Path(args.schema))
    if not ok:
        sys.stderr.write("payload failed schema validation: " + (err or "unknown") + "\n")
        return 2

    emit_bus(bus_dir, topic="rd.tasks.dispatch", kind="request", level="info", actor=actor, data=payload)
    if not args.no_infer_sync:
        emit_bus(bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data={**payload, "intent": "rd.tasks.dispatch"})

    sys.stdout.write(req_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

