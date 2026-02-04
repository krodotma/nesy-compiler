#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pluribus_directive import PluribusDirective, detect_pluribus_directive  # noqa: E402


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def _normalize_kind(v: str | None) -> str:
    vv = (v or "").strip().lower()
    if vv in {"distill", "apply", "verify", "audit", "benchmark"}:
        return vv
    return "other"


def _normalize_effects(v: str | None) -> str:
    vv = (v or "").strip().lower()
    if vv in {"none", "file", "network", "unknown"}:
        return vv
    return "unknown"


def _emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    try:
        import agent_bus  # type: ignore

        paths = agent_bus.resolve_bus_paths(bus_dir)
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
    except Exception:
        # Ultra-minimal fallback: append to events.ndjson directly.
        try:
            Path(bus_dir).mkdir(parents=True, exist_ok=True)
            evt = {
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "iso": now_iso_utc(),
                "topic": topic,
                "kind": kind,
                "level": level,
                "actor": actor,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "trace_id": None,
                "run_id": None,
                "data": data,
            }
            (Path(bus_dir) / "events.ndjson").open("a", encoding="utf-8").write(json.dumps(evt, ensure_ascii=False) + "\n")
        except Exception:
            return


def _build_intent_dag(*, req_id: str, kind: str, effects: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def node(node_id: str, *, topic: str, note: str) -> None:
        nodes.append({"id": node_id, "topic": topic, "note": note})

    def edge(src: str, dst: str) -> None:
        edges.append({"from": src, "to": dst})

    node("n0", topic="pluribus.invoke.request", note="Invocation root (potential).")
    node("n1", topic="auom.check", note="AuOM/Sextet compliance checks (lawfulness/observability/provenance/…).")
    node("n2", topic="lens.collimator.plan", note="Deterministic lane/topology/persona plan.")

    edge("n0", "n1")
    edge("n0", "n2")

    if kind in {"apply", "verify", "audit", "benchmark"} or effects in {"file", "network"}:
        node("n3", topic="rhizome.select_or_ingest", note="Select/ingest durable artifacts as rhizome inputs (optional).")
        node("n4", topic="iso_git.plan", note="Bounded transform plan (no native git).")
        node("n5", topic="verify.gates", note="Run tests/Ω/ω checks; record evidence.")
        node("n6", topic="iso_git.commit", note="Commit with provenance trailers + PQC signature.")
        edge("n2", "n3")
        edge("n3", "n4")
        edge("n4", "n5")
        edge("n5", "n6")
    else:
        node("n3", topic="strp.queue", note="Queue async distill/apply/verify work (non-blocking).")
        edge("n2", "n3")

    return {"req_id": req_id, "kind": kind, "effects": effects, "nodes": nodes, "edges": edges}


def _coerce_directive(text: str, *, kind: str, effects: str) -> PluribusDirective:
    found = detect_pluribus_directive(text)
    if found:
        # If CLI flags are provided, allow them to override absent params.
        k = found.kind
        e = found.effects
        if kind and kind != "other" and (not found.params.get("kind")):
            k = kind
        if effects and effects != "unknown" and (not found.params.get("effects")):
            e = effects
        return PluribusDirective(raw=found.raw, form=found.form, params=dict(found.params), goal=found.goal, kind=k, effects=e, role=found.role)
    return PluribusDirective(raw=text.strip(), form="prefix", params={}, goal=text.strip(), kind=kind, effects=effects, role=None)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="pluribus_operator.py", description="Emit a PLURIBUS invocation (intent DAG + evidence) to the bus.")
    p.add_argument("--text", help="Free-form text containing a PLURIBUS directive.")
    p.add_argument("--goal", help="Goal text (if --text omitted).")
    p.add_argument("--kind", default="other", help="distill|apply|verify|audit|benchmark|other")
    p.add_argument("--effects", default="unknown", help="none|file|network|unknown")
    p.add_argument("--req-id", default=None, help="Correlation id (uuid).")
    p.add_argument("--actor", default=None, help="Bus actor name.")
    p.add_argument("--bus-dir", default=None, help="Bus dir (default: $PLURIBUS_BUS_DIR or /pluribus/.pluribus/bus).")
    p.add_argument("--emit-lens", action="store_true", help="Also emit lens.collimator.plan using the local lens implementation.")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout.")
    args = p.parse_args(argv)

    bus_dir = (args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").strip()
    actor = (args.actor or default_actor()).strip() or "pluribus"
    req_id = (args.req_id or "").strip() or str(uuid.uuid4())

    kind = _normalize_kind(args.kind)
    effects = _normalize_effects(args.effects)

    text = (args.text or "").strip()
    if not text:
        text = (args.goal or "").strip()
    if not text:
        if args.json:
            sys.stdout.write(json.dumps({"ok": False, "error": "missing --text or --goal"}) + "\n")
        else:
            sys.stderr.write("error: missing --text or --goal\n")
        return 2

    directive = _coerce_directive(text, kind=kind, effects=effects)

    # 1) Root invocation request
    _emit_bus(
        bus_dir,
        topic="pluribus.invoke.request",
        kind="request",
        level="info",
        actor=actor,
        data={"req_id": req_id, "directive": directive.to_bus_dict()},
    )

    # 2) Intent DAG artifact
    dag = _build_intent_dag(req_id=req_id, kind=directive.kind, effects=directive.effects)
    _emit_bus(
        bus_dir,
        topic="pluribus.invoke.dag",
        kind="artifact",
        level="info",
        actor=actor,
        data=dag,
    )

    # 3) Optional lens plan emission (best-effort)
    if args.emit_lens:
        try:
            from lens_collimator import LensRequest, plan_route, find_pluribus_root, load_vps_session  # type: ignore

            root = find_pluribus_root(Path.cwd())
            session = load_vps_session(root)
            req = LensRequest(
                req_id=req_id,
                goal=directive.goal or "",
                kind=directive.kind,
                effects=directive.effects,
                prefer_providers=["auto"],
                require_model_prefix=None,
            )
            plan = plan_route(req, session=session)
            _emit_bus(bus_dir, topic="lens.collimator.plan", kind="artifact", level="info", actor=actor, data={**plan.__dict__})
            _emit_bus(
                bus_dir,
                topic="lens.collimator.decision",
                kind="metric",
                level="info",
                actor=actor,
                data={"req_id": req_id, "lane": plan.lane, "provider": plan.provider, "persona_id": plan.persona_id, "topology": plan.topology, "fanout": plan.fanout},
            )
        except Exception:
            _emit_bus(
                bus_dir,
                topic="lens.collimator.plan",
                kind="artifact",
                level="warn",
                actor=actor,
                data={"req_id": req_id, "error": "lens emission failed (best-effort)"},
            )

    out = {"ok": True, "req_id": req_id, "directive": directive.to_bus_dict(), "dag": {"nodes": len(dag["nodes"]), "edges": len(dag["edges"])}}
    if args.json:
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(f"OK req_id={req_id} kind={directive.kind} effects={directive.effects} (bus={bus_dir})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

