#!/usr/bin/env python3
from __future__ import annotations

"""
SemOps Invoke Responder
======================

Tails the bus for `semops.invoke.request` and emits a structured, non-executing
`semops.invoke.response` with:
  - resolved mode (auto → tool/bus/policy/evolution/ui/agent/app)
  - risk classification derived from `effects`
  - suggested next actions (commands / bus emissions / navigation hints)

This is intentionally *plan-only*: it does not execute tools or mutate state.
Execution should be mediated by bounded transforms and/or explicit operator
approval events.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_bus_dir(args_bus_dir: str | None) -> str:
    return args_bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"


def default_actor(args_actor: str | None) -> str:
    return args_actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "semops-invoke-responder"


def ensure_events_file(bus_dir: str) -> Path:
    p = Path(bus_dir) / "events.ndjson"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def _safe_excerpt(text: str, *, max_chars: int = 240) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "…"


def _sha256_json(obj: object) -> str:
    try:
        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except Exception:
        raw = repr(obj).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def tail_events(events_path: Path, *, since_ts: float, poll_s: float, stop_at_ts: float | None):
    def emit_backfill() -> list[dict]:
        try:
            max_bytes = 512 * 1024
            with events_path.open("rb") as bf:
                bf.seek(0, os.SEEK_END)
                end = bf.tell()
                start = max(0, end - max_bytes)
                bf.seek(start)
                data = bf.read(end - start)
            lines = data.splitlines()
            out: list[dict] = []
            for b in lines[-2000:]:
                try:
                    obj = json.loads(b.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if obj.get("topic") != "semops.invoke.request":
                    continue
                try:
                    ts = float(obj.get("ts") or 0.0)
                except Exception:
                    ts = 0.0
                if ts < since_ts:
                    continue
                out.append(obj)
            return out
        except Exception:
            return []

    for obj in emit_backfill():
        yield obj

    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            if stop_at_ts is not None and time.time() >= stop_at_ts:
                return
            line = f.readline()
            if not line:
                time.sleep(max(0.05, poll_s))
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("topic") != "semops.invoke.request":
                continue
            try:
                ts = float(obj.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            if ts < since_ts:
                continue
            yield obj


def emit_bus(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    tool = Path(__file__).with_name("agent_bus.py")
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


def _normalize_mode(value: object) -> str:
    v = str(value or "").strip().lower()
    return v or "auto"


def _normalize_effects(value: object) -> str:
    v = str(value or "").strip().lower()
    return v or "none"


def _risk_from_effects(effects: str) -> tuple[str, str]:
    e = (effects or "none").strip().lower()
    if e in {"none", "read"}:
        return "low", "No side effects declared."
    if e in {"file"}:
        return "medium", "File mutation possible; prefer bounded transforms + provenance."
    if e in {"network"}:
        return "high", "Network egress possible; ensure provider/profile/policy constraints."
    if e in {"system"}:
        return "critical", "System-level side effects; require explicit operator approval + sandboxing."
    return "medium", "Unknown effects; treat as potentially unsafe until verified."


def _pick_mode(mode_requested: str, operator: dict | None) -> str:
    mode = _normalize_mode(mode_requested)
    if mode != "auto":
        return mode
    op = operator or {}
    hints = op.get("flow_hints")
    if isinstance(hints, list) and hints:
        for h in hints:
            hs = str(h or "").strip().lower()
            if hs:
                return hs
    if op.get("tool"):
        return "tool"
    if op.get("bus_topic"):
        return "bus"
    targets = op.get("targets")
    if isinstance(targets, list):
        for t in targets:
            if not isinstance(t, dict):
                continue
            tt = str(t.get("type") or "").strip().lower()
            if tt in {"tool", "bus", "ui", "policy", "evolution", "agent", "app"}:
                return tt
    domain = str(op.get("domain") or "").strip().lower()
    if domain in {"safety", "policy"}:
        return "policy"
    if domain in {"evolution", "git"}:
        return "evolution"
    if domain in {"ui"}:
        return "ui"
    return "auto"


def _suggest_actions(*, operator_key: str, mode: str, operator: dict | None) -> list[dict]:
    op = operator or {}
    actions: list[dict] = []

    tool_path = op.get("tool")
    if mode == "tool" and isinstance(tool_path, str) and tool_path.strip():
        actions.append(
            {
                "id": "tool.open",
                "kind": "hint",
                "label": "Open tool path",
                "path": tool_path,
            }
        )
        actions.append(
            {
                "id": "tool.help",
                "kind": "hint",
                "label": "Run tool help",
                "command": f"python3 {tool_path} --help",
            }
        )

    if mode == "bus":
        topic = op.get("bus_topic")
        kind = op.get("bus_kind") or "request"
        if isinstance(topic, str) and topic.strip():
            actions.append(
                {
                    "id": "bus.emit",
                    "kind": "hint",
                    "label": f"Emit {topic}",
                    "topic": topic,
                    "bus_kind": str(kind),
                }
            )

    if mode == "policy":
        actions.append(
            {
                "id": "policy.open",
                "kind": "hint",
                "label": "Review gateway policy",
                "path": "/pluribus/.pluribus/gateway_policy.json",
            }
        )

    if mode == "evolution":
        actions.append(
            {
                "id": "evo.iso_git.status",
                "kind": "hint",
                "label": "Inspect repo status (iso_git lane)",
                "command": "node /pluribus/nucleus/tools/iso_git.mjs status /pluribus",
            }
        )

    ui = op.get("ui") if isinstance(op.get("ui"), dict) else {}
    route = ui.get("route") if isinstance(ui, dict) else None
    if mode == "ui" and isinstance(route, str) and route.strip():
        actions.append(
            {
                "id": "ui.navigate",
                "kind": "hint",
                "label": f"Navigate to {route}",
                "route": route,
            }
        )

    actions.append(
        {
            "id": "events.trace",
            "kind": "hint",
            "label": "Trace in Events view (req_id/operator_key)",
            "query": {"topic": "semops.invoke.*", "operator_key": operator_key},
        }
    )
    return actions


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="semops_invoke_responder.py", description="Daemon: respond to semops.invoke.request with semops.invoke.response (plan-only).")
    ap.add_argument("--bus-dir", default=None)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--poll", default="0.25")
    ap.add_argument("--run-for-s", default="0", help="Run for N seconds then exit (0 = run forever).")
    ap.add_argument("--since-ts", default=None, help="Only respond to triggers >= this UNIX timestamp.")
    args = ap.parse_args(argv)

    actor = default_actor(args.actor)
    bus_dir = default_bus_dir(args.bus_dir)
    poll_s = max(0.05, float(args.poll))
    run_for_s = max(0.0, float(args.run_for_s))
    since_ts = float(args.since_ts) if args.since_ts is not None else time.time()
    stop_at_ts = None if run_for_s <= 0 else (time.time() + run_for_s)

    events_path = ensure_events_file(bus_dir)
    seen_ids: set[str] = set()

    emit_bus(
        bus_dir,
        topic="semops.invoke.responder.ready",
        kind="artifact",
        level="info",
        actor=actor,
        data={"since_ts": since_ts, "iso": now_iso_utc(), "pid": os.getpid()},
    )

    for trig in tail_events(events_path, since_ts=since_ts, poll_s=poll_s, stop_at_ts=stop_at_ts):
        trig_id = str(trig.get("id") or "")
        if trig_id and trig_id in seen_ids:
            continue
        if trig_id:
            seen_ids.add(trig_id)

        data = trig.get("data") if isinstance(trig.get("data"), dict) else {}
        req_id = str(data.get("req_id") or "")
        operator_key = str(data.get("operator_key") or "").strip().upper()
        mode_requested = _normalize_mode(data.get("mode"))

        operator = data.get("operator") if isinstance(data.get("operator"), dict) else None
        effects = _normalize_effects(data.get("effects") or (operator or {}).get("effects"))
        mode_resolved = _pick_mode(mode_requested, operator)
        risk_level, risk_reason = _risk_from_effects(effects)

        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        payload_keys = [k for k in payload.keys() if isinstance(k, str)][:30]
        payload_sha = _sha256_json(payload)

        op_name = None
        if isinstance(operator, dict):
            op_name = str(operator.get("name") or "").strip() or None
        summary = f"{operator_key or 'UNKNOWN'} ({op_name or mode_resolved}) → {mode_resolved}"

        response = {
            "req_id": req_id,
            "operator_key": operator_key,
            "mode_requested": mode_requested,
            "mode_resolved": mode_resolved,
            "effects": effects,
            "risk": {"level": risk_level, "reason": risk_reason},
            "summary": summary,
            "operator": {
                "name": op_name,
                "domain": (operator or {}).get("domain") if isinstance(operator, dict) else None,
                "category": (operator or {}).get("category") if isinstance(operator, dict) else None,
                "tool": (operator or {}).get("tool") if isinstance(operator, dict) else None,
                "bus_topic": (operator or {}).get("bus_topic") if isinstance(operator, dict) else None,
                "ui": (operator or {}).get("ui") if isinstance(operator, dict) else None,
            },
            "payload_meta": {
                "keys": payload_keys,
                "sha256": payload_sha,
            },
            "actions": _suggest_actions(operator_key=operator_key, mode=mode_resolved, operator=operator),
            "notes": [
                _safe_excerpt("Plan-only responder: does not execute tools or mutate state."),
                _safe_excerpt("Use SemOps effects as capability hints; gate execution via bounded transforms."),
            ],
            "responded_iso": now_iso_utc(),
        }

        emit_bus(
            bus_dir,
            topic="semops.invoke.response",
            kind="response",
            level="warn" if risk_level in {"high", "critical"} else "info",
            actor=actor,
            data=response,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

