#!/usr/bin/env python3
from __future__ import annotations

"""
A2A negotiation daemon (bus-native).

Listens for `a2a.negotiate.request` (kind=request) and emits:
- `a2a.capabilities.advertise` (kind=artifact) at startup
- `a2a.negotiate.response` (kind=response) for decisions (agree|reject|negotiate)
- `a2a.decline` (kind=response) for explicit rejections

Design goals:
- Non-blocking, append-only bus evidence.
- Idempotent: never emits a second decision for the same req_id.
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sys

sys.dont_write_bytecode = True


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_bus_dir(v: str | None) -> Path:
    if v:
        return Path(v).expanduser().resolve()
    env = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path("/pluribus/.pluribus/bus").expanduser().resolve()


def _ensure_events_file(bus_dir: Path) -> Path:
    bus_dir.mkdir(parents=True, exist_ok=True)
    p = bus_dir / "events.ndjson"
    p.touch(exist_ok=True)
    return p


def _iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _append_ndjson(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def _emit(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> None:
    _append_ndjson(
        bus_dir / "events.ndjson",
        {
            "id": __import__("uuid").uuid4().hex,
            "ts": time.time(),
            "iso": _now_iso_utc(),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": actor,
            "data": data,
        },
    )


def _already_responded(events, *, req_id: str) -> bool:
    for e in events:
        if e.get("kind") != "response":
            continue
        d = e.get("data")
        if isinstance(d, dict) and (d.get("req_id") == req_id or d.get("request_id") == req_id):
            return True
    return False


def _parse_caps(raw: str) -> list[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="negotiate_daemon.py", description="A2A negotiation daemon for bus-native negotiate/decline flows.")
    p.add_argument("--bus-dir", default=None)
    p.add_argument("--actor", default=None, help="Responder actor (defaults to $PLURIBUS_ACTOR/$USER).")
    p.add_argument("--capabilities", default="", help="Comma-separated capabilities for this actor.")
    p.add_argument("--once", action="store_true", help="Process existing requests once then exit.")
    p.add_argument("--poll", type=float, default=0.25, help="Poll interval seconds (daemon mode).")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "a2a-negotiator").strip()
    bus_dir = _default_bus_dir(args.bus_dir)
    events_path = _ensure_events_file(bus_dir)

    # Late import (keeps this tool usable even if package layout changes).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from a2a.negotiation import Contract, Negotiator, A2AState  # type: ignore
    except Exception:
        from nucleus.tools.a2a.negotiation import Contract, Negotiator, A2AState  # type: ignore

    caps = _parse_caps(args.capabilities or os.environ.get("PLURIBUS_A2A_CAPABILITIES") or "")
    negotiator = Negotiator(agent_name=actor, capabilities=caps)

    _emit(
        bus_dir,
        topic="a2a.capabilities.advertise",
        kind="artifact",
        level="info",
        actor=actor,
        data={"actor": actor, "capabilities": caps, "iso": _now_iso_utc()},
    )

    def process_once() -> int:
        events = list(_iter_ndjson(events_path))
        processed = 0
        for e in events:
            if e.get("topic") != "a2a.negotiate.request" or e.get("kind") != "request":
                continue
            d = e.get("data")
            if not isinstance(d, dict):
                continue
            req_id = str(d.get("req_id") or d.get("request_id") or "").strip()
            if not req_id:
                continue
            if _already_responded(events, req_id=req_id):
                continue
            target = str(d.get("target") or "").strip()
            if target and target.lower() not in {actor.lower()}:
                continue

            contract = Contract(
                contract_id=req_id,
                initiator=str(d.get("initiator") or d.get("from") or e.get("actor") or "unknown"),
                target=actor,
                task_description=str(d.get("task_description") or d.get("task") or ""),
                constraints=d.get("constraints") if isinstance(d.get("constraints"), dict) else {},
                compensation=d.get("compensation") if isinstance(d.get("compensation"), dict) else {},
            )
            contract = negotiator.evaluate_proposal(contract)

            decision = contract.state.value
            payload = {
                "req_id": req_id,
                "decision": decision,
                "contract": asdict(contract) | {"state": contract.state.value},
                "iso": _now_iso_utc(),
            }
            _emit(
                bus_dir,
                topic="a2a.negotiate.response",
                kind="response",
                level="info" if contract.state in {A2AState.AGREE, A2AState.NEGOTIATE} else "warn",
                actor=actor,
                data=payload,
            )
            if contract.state == A2AState.REJECT:
                _emit(
                    bus_dir,
                    topic="a2a.decline",
                    kind="response",
                    level="warn",
                    actor=actor,
                    data={"req_id": req_id, "reason": contract.history[-1]["reason"] if contract.history else "reject", "iso": _now_iso_utc()},
                )
                redirect_to = str(d.get("redirect_to") or "").strip()
                if not redirect_to:
                    alts = d.get("alternatives")
                    if isinstance(alts, list):
                        for cand in alts:
                            c = str(cand or "").strip()
                            if c:
                                redirect_to = c
                                break
                if redirect_to:
                    _emit(
                        bus_dir,
                        topic="a2a.redirect",
                        kind="response",
                        level="info",
                        actor=actor,
                        data={
                            "req_id": req_id,
                            "redirect_to": redirect_to,
                            "reason": contract.history[-1]["reason"] if contract.history else "reject",
                            "iso": _now_iso_utc(),
                        },
                    )
            processed += 1
        return processed

    if args.once:
        n = process_once()
        sys.stdout.write(str(n) + "\n")
        return 0

    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(max(0.05, float(args.poll)))
                continue
            # For responsiveness, just re-run the cheap scan; it stays correct + idempotent.
            process_once()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
