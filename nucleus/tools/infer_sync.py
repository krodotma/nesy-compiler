#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def agent_bus_path() -> Path:
    return Path(__file__).with_name("agent_bus.py")


def publish(*, bus_dir: str | None, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        raise SystemExit("missing bus dir (set PLURIBUS_BUS_DIR or pass --bus-dir)")
    tool = agent_bus_path()
    if not tool.exists():
        raise SystemExit("missing agent_bus.py")
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            kind,
            "--level",
            level,
            "--actor",
            actor,
            "--data",
            json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cmd_checkin(args: argparse.Namespace) -> int:
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    data = {
        "status": args.status,
        "done": int(args.done),
        "open": int(args.open),
        "blocked": int(args.blocked),
        "errors": int(args.errors),
        "next": args.next,
        "subproject": args.subproject,
        "focus": [t for t in (args.focus or []) if t.strip()],
        "iso": now_iso_utc(),
    }
    publish(bus_dir=bus_dir, topic="infer_sync.checkin", kind="metric", level="info", actor=actor, data=data)
    sys.stdout.write("ok\n")
    return 0


def cmd_request(args: argparse.Namespace) -> int:
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    req_id = args.req_id or str(uuid.uuid4())
    inputs = json.loads(args.inputs) if args.inputs else {}
    constraints = json.loads(args.constraints) if args.constraints else {"append_only": True, "tests_first": True}
    data = {
        "req_id": req_id,
        "subproject": args.subproject,
        "intent": args.intent,
        "inputs": inputs,
        "constraints": constraints,
        "response_topic": args.response_topic or "infer_sync.response",
        "iso": now_iso_utc(),
    }
    publish(bus_dir=bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data=data)
    sys.stdout.write(req_id + "\n")
    return 0


def cmd_response(args: argparse.Namespace) -> int:
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    artifacts = [a for a in (args.artifact or []) if a.strip()]
    next_actions = [a for a in (args.next_action or []) if a.strip()]
    risks = [r for r in (args.risk or []) if r.strip()]
    data = {
        "req_id": args.req_id,
        "summary": args.summary,
        "artifacts": artifacts,
        "next_actions": next_actions,
        "risks": risks,
        "iso": now_iso_utc(),
    }
    publish(bus_dir=bus_dir, topic="infer_sync.response", kind="response", level="info", actor=actor, data=data)
    sys.stdout.write("ok\n")
    return 0


def cmd_candidate(args: argparse.Namespace) -> int:
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    tests = [t for t in (args.test or []) if t.strip()]
    evidence = [e for e in (args.evidence or []) if e.strip()]
    data = {
        "candidate_id": args.candidate_id,
        "lineage": {"parent": args.parent, "type": args.lineage_type},
        "claim": args.claim,
        "tests": tests,
        "evidence": evidence,
        "iso": now_iso_utc(),
    }
    publish(bus_dir=bus_dir, topic="infer_sync.candidate", kind="artifact", level="info", actor=actor, data=data)
    sys.stdout.write("ok\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="infer_sync.py", description="infer_sync: standardized bus check-ins and coordination events.")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("checkin", help="Publish infer_sync.checkin")
    c.add_argument("--status", default="working", choices=["working", "idle", "blocked", "error"])
    c.add_argument("--done", default=0, type=int)
    c.add_argument("--open", default=0, type=int)
    c.add_argument("--blocked", default=0, type=int)
    c.add_argument("--errors", default=0, type=int)
    c.add_argument("--next", default="")
    c.add_argument("--subproject", default="other")
    c.add_argument("--focus", action="append", default=[])
    c.set_defaults(func=cmd_checkin)

    r = sub.add_parser("request", help="Publish infer_sync.request")
    r.add_argument("--req-id", default=None)
    r.add_argument("--subproject", default="other")
    r.add_argument("--intent", required=True)
    r.add_argument("--inputs", default=None, help="JSON string for inputs")
    r.add_argument("--constraints", default=None, help="JSON string for constraints")
    r.add_argument("--response-topic", default="infer_sync.response")
    r.set_defaults(func=cmd_request)

    s = sub.add_parser("response", help="Publish infer_sync.response")
    s.add_argument("req_id")
    s.add_argument("--summary", required=True)
    s.add_argument("--artifact", action="append", default=[])
    s.add_argument("--next-action", action="append", default=[])
    s.add_argument("--risk", action="append", default=[])
    s.set_defaults(func=cmd_response)

    k = sub.add_parser("candidate", help="Publish infer_sync.candidate")
    k.add_argument("candidate_id")
    k.add_argument("--lineage-type", default="mutation", choices=["VGT", "HGT", "mutation", "merge"])
    k.add_argument("--parent", default=None)
    k.add_argument("--claim", required=True)
    k.add_argument("--test", action="append", default=[])
    k.add_argument("--evidence", action="append", default=[])
    k.set_defaults(func=cmd_candidate)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

