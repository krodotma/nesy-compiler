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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def resolve_queue_path(root: Path | None) -> Path | None:
    if not root:
        return None
    return root / ".pluribus" / "index" / "requests.ndjson"


def json_load_maybe(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value == "-":
        return json.load(sys.stdin)
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        tool = Path(__file__).resolve().parents[0] / "agent_bus.py"
    if not tool.exists():
        return
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def cmd_request(args: argparse.Namespace) -> int:
    actor = default_actor()
    req_id = str(uuid.uuid4())
    root = Path(args.root).expanduser().resolve() if args.root else find_rhizome_root(Path.cwd())
    queue_path = resolve_queue_path(root)
    payload = {
        "req_id": req_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "actor": actor,
        "goal": args.goal,
        "kind": args.kind,
        "provider_hint": args.provider,
        "parallelizable": bool(args.parallelizable),
        "tool_density": args.tool_density,
        "coord_budget_tokens": args.coord_budget_tokens,
        "topology_hint": args.topology_hint,
        "inputs": json_load_maybe(args.inputs),
        "constraints": json_load_maybe(args.constraints),
        "sextet": json_load_maybe(args.sextet),
        "rhizome_root": str(root) if root else None,
    }
    if queue_path:
        append_ndjson(queue_path, payload)
    if args.emit_bus:
        bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        emit_bus(bus_dir, topic=f"strp.request.{args.kind}", kind="request", level="info", actor=actor, data=payload)
    sys.stdout.write(req_id + "\n")
    return 0


def cmd_respond(args: argparse.Namespace) -> int:
    actor = default_actor()
    resp_id = str(uuid.uuid4())
    payload = {
        "resp_id": resp_id,
        "req_id": args.req_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "actor": actor,
        "provider_used": args.provider,
        "output": json_load_maybe(args.output),
        "notes": args.notes,
        "sextet": json_load_maybe(args.sextet),
    }
    if args.emit_bus:
        bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        emit_bus(bus_dir, topic="strp.response", kind="response", level="info", actor=actor, data=payload)
    sys.stdout.write(resp_id + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="strp_queue.py", description="Publish STRp work requests/responses (bus + optional rhizome log).")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    sub = p.add_subparsers(dest="cmd", required=True)

    req = sub.add_parser("request", help="Publish a work request.")
    req.add_argument("--goal", required=True)
    req.add_argument("--kind", default="distill", help="curate|distill|hypothesize|apply|implement|verify|other")
    req.add_argument("--provider", default="auto", help="auto|gemini|claude|local|web|other")
    req.add_argument("--parallelizable", action="store_true", help="Allow fanout (multi-run orchestration) when policy selects it.")
    req.add_argument("--tool-density", type=float, default=None, help="0..1 heuristic for tool-heaviness; higher => coordination amplifies errors.")
    req.add_argument("--coord-budget-tokens", type=int, default=None, help="Approx coordination budget; low values force single-agent fallback.")
    req.add_argument("--topology-hint", default="auto", help="auto|single|star|peer_debate")
    req.add_argument("--inputs", default=None, help="JSON string; '-' stdin; '@file.json' to load.")
    req.add_argument("--constraints", default=None, help="JSON string; '-' stdin; '@file.json' to load.")
    req.add_argument("--sextet", default=None, help="JSON sextet facets; '-' stdin; '@file.json' to load.")
    req.add_argument("--emit-bus", action="store_true")
    req.add_argument("--bus-dir", default=None)
    req.set_defaults(func=cmd_request)

    resp = sub.add_parser("respond", help="Publish a response to a request.")
    resp.add_argument("req_id")
    resp.add_argument("--provider", default=None)
    resp.add_argument("--output", default=None, help="JSON string; '-' stdin; '@file.json' to load.")
    resp.add_argument("--notes", default=None)
    resp.add_argument("--sextet", default=None)
    resp.add_argument("--emit-bus", action="store_true")
    resp.add_argument("--bus-dir", default=None)
    resp.set_defaults(func=cmd_respond)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
