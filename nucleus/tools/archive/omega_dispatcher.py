#!/usr/bin/env python3
from __future__ import annotations

"""
Omega Dispatcher — subagent dispatch tick for rd.tasks.dispatch.

Watches the bus for rd.tasks.dispatch (kind=request) and emits A2A negotiation
requests for targeted subagents when no ack/response is observed. This is a
lightweight, non-blocking dispatcher intended to prevent idle queues while
avoiding unnecessary token churn (cooldown + max attempts).
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
BUS_DEFAULT = "/pluribus/.pluribus/bus"

try:
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore

try:
    from nucleus.tools.ring_guard import RingGuard, Ring  # type: ignore
except Exception:  # pragma: no cover
    RingGuard = None  # type: ignore
    Ring = None  # type: ignore


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_int(v: object, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: object, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def default_actor(args_actor: str | None) -> str:
    return (args_actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "omega-dispatcher").strip()


def emit_bus_event(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> None:
    if agent_bus is not None:
        try:
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
        except Exception:
            pass

    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _tail_bytes(path: Path, max_bytes: int) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    if max_bytes <= 0:
        return []
    try:
        size = path.stat().st_size
    except Exception:
        return []
    start = max(0, size - max_bytes)
    try:
        with path.open("rb") as f:
            if start:
                f.seek(start)
            raw = f.read()
    except Exception:
        return []
    lines = raw.splitlines()
    if start and lines:
        lines = lines[1:]  # drop partial line
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line.decode("utf-8", errors="replace")))
        except Exception:
            continue
    return out


def _read_new_events(path: Path, *, cur: int, last_inode: int | None) -> tuple[list[dict[str, Any]], int, int | None]:
    events: list[dict[str, Any]] = []
    try:
        st = path.stat()
        inode = getattr(st, "st_ino", None)
        if last_inode is None:
            last_inode = inode
        if inode is not None and last_inode is not None and inode != last_inode:
            cur = 0
            last_inode = inode
        if st.st_size < cur:
            cur = 0
        if st.st_size > cur:
            with path.open("rb") as f:
                f.seek(cur)
                lines = f.readlines()
                cur = f.tell()
            for line in lines:
                try:
                    events.append(json.loads(line.decode("utf-8", errors="replace")))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return events, cur, last_inode


def _normalize_targets(raw: object) -> list[str]:
    if isinstance(raw, list):
        targets = [str(t).strip() for t in raw if str(t).strip()]
    elif isinstance(raw, str):
        targets = [t.strip() for t in raw.split(",") if t.strip()]
    else:
        targets = []
    seen: set[str] = set()
    out: list[str] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _load_subagent_targets() -> list[str]:
    personas_path = REPO_ROOT / "nucleus" / "specs" / "personas.json"
    if not personas_path.exists():
        return []
    try:
        payload = json.loads(personas_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    personas = payload.get("personas") if isinstance(payload, dict) else None
    if not isinstance(personas, list):
        return []
    out: list[str] = []
    for p in personas:
        if not isinstance(p, dict):
            continue
        if str(p.get("provider_class") or "").strip() != "subagent":
            continue
        pid = str(p.get("id") or "").strip()
        if pid:
            out.append(pid)
    return out


def _expand_all_targets(targets: list[str], *, all_targets: list[str]) -> list[str]:
    lowered = {t.lower() for t in targets}
    if "all" not in lowered and "*" not in lowered:
        return targets
    expanded = [t for t in targets if t.lower() not in {"all", "*"}]
    for t in all_targets:
        if t not in expanded:
            expanded.append(t)
    return expanded


def _build_task_summary(data: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    task_id = str(data.get("task_id") or "").strip()
    intent = str(data.get("intent") or "").strip()
    tasks = data.get("tasks") if isinstance(data.get("tasks"), list) else []
    outline: list[dict[str, str]] = []
    for t in tasks[:5]:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id") or "").strip()
        title = str(t.get("title") or "").strip()
        if tid or title:
            outline.append({"id": tid, "title": title})
    summary = "rd.tasks.dispatch"
    if task_id or intent:
        summary = f"{task_id}: {intent}".strip(": ").strip()
    return summary, outline


def _init_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_iso": now_iso_utc(),
        "dispatches": {},
    }


def _ensure_dispatch(state: dict[str, Any], *, req_id: str, actor: str, data: dict[str, Any]) -> dict[str, Any]:
    dispatches = state.setdefault("dispatches", {})
    dispatch = dispatches.get(req_id)
    if not isinstance(dispatch, dict):
        dispatch = {
            "req_id": req_id,
            "actor": actor,
            "task_id": data.get("task_id"),
            "intent": data.get("intent"),
            "created_ts": float(data.get("ts") or time.time()),
            "last_event_ts": time.time(),
            "targets": {},
        }
        dispatches[req_id] = dispatch
    return dispatch


def _mark_target(dispatch: dict[str, Any], target: str, *, status: str) -> None:
    targets = dispatch.setdefault("targets", {})
    tstate = targets.get(target)
    if not isinstance(tstate, dict):
        tstate = {
            "status": status,
            "attempts": 0,
            "last_attempt_ts": 0.0,
            "a2a_req_id": f"{dispatch.get('req_id')}:{target}",
        }
        targets[target] = tstate
    else:
        tstate["status"] = status


def _apply_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    topic = str(event.get("topic") or "")
    kind = str(event.get("kind") or "")
    actor = str(event.get("actor") or "")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}

    if topic == "rd.tasks.dispatch" and kind == "request":
        req_id = str(data.get("req_id") or "").strip()
        if not req_id:
            return
        dispatch = _ensure_dispatch(state, req_id=req_id, actor=actor, data={"task_id": data.get("task_id"), "intent": data.get("intent")})
        dispatch["last_event_ts"] = time.time()
        dispatch["task_id"] = data.get("task_id")
        dispatch["intent"] = data.get("intent")
        dispatch["raw_data"] = data
        targets_raw = _normalize_targets(data.get("targets"))
        targets = _expand_all_targets(targets_raw, all_targets=_load_subagent_targets())
        for t in targets:
            _mark_target(dispatch, t, status="pending")

    elif topic == "rd.tasks.ack" and kind == "response":
        req_id = str(data.get("req_id") or "").strip()
        if not req_id:
            return
        dispatch = state.get("dispatches", {}).get(req_id)
        if not isinstance(dispatch, dict):
            return
        target = str(data.get("target") or actor or "").strip()
        if target:
            _mark_target(dispatch, target, status="ack")

    elif topic == "rd.tasks.done" and kind in {"response", "artifact"}:
        req_id = str(data.get("req_id") or "").strip()
        if not req_id:
            return
        dispatch = state.get("dispatches", {}).get(req_id)
        if not isinstance(dispatch, dict):
            return
        for t in list(dispatch.get("targets", {}).keys()):
            _mark_target(dispatch, t, status="done")

    elif topic in {"a2a.negotiate.response", "a2a.decline", "a2a.redirect"} and kind == "response":
        req_id = str(data.get("req_id") or data.get("request_id") or "").strip()
        if not req_id:
            return
        for dispatch in state.get("dispatches", {}).values():
            if not isinstance(dispatch, dict):
                continue
            for t, tstate in (dispatch.get("targets") or {}).items():
                if not isinstance(tstate, dict):
                    continue
                if str(tstate.get("a2a_req_id") or "") == req_id:
                    _mark_target(dispatch, t, status="responded")


def _pending_targets(dispatch: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    pending: list[tuple[str, dict[str, Any]]] = []
    for target, tstate in (dispatch.get("targets") or {}).items():
        if not isinstance(tstate, dict):
            continue
        if tstate.get("status") in {"ack", "responded", "done"}:
            continue
        pending.append((target, tstate))
    return pending


def _dispatch_for_target(
    *,
    bus_dir: Path,
    actor: str,
    dispatch: dict[str, Any],
    target: str,
    tstate: dict[str, Any],
    attempt: int,
    dry_run: bool,
) -> None:
    data = dispatch.get("raw_data") if isinstance(dispatch.get("raw_data"), dict) else {}
    summary, outline = _build_task_summary(data)
    a2a_req_id = str(tstate.get("a2a_req_id") or f"{dispatch.get('req_id')}:{target}")

    # Ring Guard access check (SCI/SAP-style compartmentalization)
    access_granted = True
    access_reason = ""
    if RingGuard is not None:
        try:
            guard = RingGuard()
            task_path = data.get("task_path") or data.get("file") or ""
            if task_path:
                result = guard.check_access(target, task_path)
                access_granted = result.granted
                access_reason = result.reason if hasattr(result, "reason") else ""
                if not access_granted:
                    emit_bus_event(
                        bus_dir,
                        topic="omega.dispatch.access_denied",
                        kind="log",
                        level="warn",
                        actor=actor,
                        data={
                            "target": target,
                            "task_path": task_path,
                            "reason": access_reason,
                            "a2a_req_id": a2a_req_id,
                            "iso": now_iso_utc(),
                        },
                    )
                    tstate["status"] = "access_denied"
                    return
        except Exception as e:
            # Log but don't block on ring_guard failures
            emit_bus_event(
                bus_dir,
                topic="omega.dispatch.ring_check_error",
                kind="log",
                level="warn",
                actor=actor,
                data={"error": str(e), "target": target, "iso": now_iso_utc()},
            )

    payload = {
        "req_id": a2a_req_id,
        "dispatch_req_id": dispatch.get("req_id"),
        "initiator": dispatch.get("actor"),
        "target": target,
        "task_id": dispatch.get("task_id"),
        "task_description": summary,
        "task_outline": outline,
        "constraints": data.get("constraints") if isinstance(data.get("constraints"), dict) else {},
        "gates": data.get("gates") if isinstance(data.get("gates"), list) else [],
        "attempt": attempt,
        "iso": now_iso_utc(),
    }

    if not dry_run:
        emit_bus_event(
            bus_dir,
            topic="a2a.negotiate.request",
            kind="request",
            level="info",
            actor=actor,
            data=payload,
        )
        emit_bus_event(
            bus_dir,
            topic="omega.dispatch.sent",
            kind="metric",
            level="info",
            actor=actor,
            data={
                "dispatch_req_id": dispatch.get("req_id"),
                "a2a_req_id": a2a_req_id,
                "target": target,
                "attempt": attempt,
                "iso": now_iso_utc(),
            },
        )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="omega_dispatcher.py", description="Omega dispatcher: trigger A2A requests for rd.tasks.dispatch targets.")
    ap.add_argument("--bus-dir", default=None, help="Bus directory (default: /pluribus/.pluribus/bus).")
    ap.add_argument("--state-file", default=None, help="State file path (default: /pluribus/.pluribus/index/omega_dispatcher_state.json).")
    ap.add_argument("--actor", default=None, help="Bus actor override.")
    ap.add_argument("--poll", default="0.25", help="Poll interval seconds.")
    ap.add_argument("--tick-s", default="30", help="Tick interval seconds for re-dispatch.")
    ap.add_argument("--resend-after-s", default="900", help="Cooldown seconds before re-dispatch.")
    ap.add_argument("--max-attempts", default="5", help="Max dispatch attempts per target (0 = unlimited).")
    ap.add_argument("--bootstrap-bytes", default="524288", help="Tail bytes to bootstrap state from bus.")
    ap.add_argument("--run-for-s", default="0", help="Run for N seconds then exit (0 = forever).")
    ap.add_argument("--dry-run", action="store_true", help="Do not emit dispatch events.")
    ap.add_argument("--emit-on-empty", action="store_true", help="Emit omega.dispatch.tick even when no pending targets.")
    args = ap.parse_args(argv)

    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or BUS_DEFAULT).expanduser().resolve()
    state_path = Path(args.state_file).expanduser().resolve() if args.state_file else Path("/pluribus/.pluribus/index/omega_dispatcher_state.json")
    actor = default_actor(args.actor)
    poll_s = _safe_float(args.poll, 0.25)
    tick_s = _safe_float(args.tick_s, 30.0)
    resend_after_s = _safe_float(args.resend_after_s, 900.0)
    max_attempts = _safe_int(args.max_attempts, 5)
    bootstrap_bytes = _safe_int(args.bootstrap_bytes, 524288)
    run_for_s = _safe_float(args.run_for_s, 0.0)

    bus_dir.mkdir(parents=True, exist_ok=True)
    events_path = bus_dir / "events.ndjson"
    events_path.touch(exist_ok=True)

    state = _init_state()
    if state_path.exists():
        state = _load_json(state_path) or _init_state()

    # Bootstrap from tail of bus.
    for e in _tail_bytes(events_path, bootstrap_bytes):
        if not isinstance(e, dict):
            continue
        if "topic" not in e:
            continue
        if e.get("topic") == "rd.tasks.dispatch" and e.get("kind") == "request":
            req_id = ((e.get("data") or {}).get("req_id") if isinstance(e.get("data"), dict) else None)
            if isinstance(req_id, str) and req_id:
                dispatch = _ensure_dispatch(state, req_id=req_id, actor=str(e.get("actor") or ""), data=(e.get("data") or {}))
                dispatch["raw_data"] = e.get("data") if isinstance(e.get("data"), dict) else {}
        _apply_event(state, e)

    _save_json(state_path, state)

    start_ts = time.time()
    last_tick = 0.0

    emit_bus_event(
        bus_dir,
        topic="omega.dispatcher.ready",
        kind="artifact",
        level="info",
        actor=actor,
        data={"iso": now_iso_utc(), "pid": os.getpid()},
    )

    cur = 0
    last_inode: int | None = None
    try:
        st = events_path.stat()
        cur = st.st_size
        last_inode = getattr(st, "st_ino", None)
    except Exception:
        cur = 0
        last_inode = None

    while True:
        events, cur, last_inode = _read_new_events(events_path, cur=cur, last_inode=last_inode)
        for e in events:
            _apply_event(state, e)

        if time.time() - last_tick >= tick_s:
            pending_total = 0
            for dispatch in list(state.get("dispatches", {}).values()):
                if not isinstance(dispatch, dict):
                    continue
                for target, tstate in _pending_targets(dispatch):
                    pending_total += 1
                    attempts = _safe_int(tstate.get("attempts"), 0)
                    last_attempt = _safe_float(tstate.get("last_attempt_ts"), 0.0)
                    if max_attempts > 0 and attempts >= max_attempts:
                        continue
                    if last_attempt and (time.time() - last_attempt) < resend_after_s:
                        continue
                    attempts += 1
                    tstate["attempts"] = attempts
                    tstate["last_attempt_ts"] = time.time()
                    _dispatch_for_target(
                        bus_dir=bus_dir,
                        actor=actor,
                        dispatch=dispatch,
                        target=target,
                        tstate=tstate,
                        attempt=attempts,
                        dry_run=bool(args.dry_run),
                    )

            if pending_total > 0 or args.emit_on_empty:
                emit_bus_event(
                    bus_dir,
                    topic="omega.dispatch.tick",
                    kind="metric",
                    level="info",
                    actor=actor,
                    data={
                        "pending_total": pending_total,
                        "resend_after_s": resend_after_s,
                        "max_attempts": max_attempts,
                        "iso": now_iso_utc(),
                    },
                )

            state["updated_iso"] = now_iso_utc()
            _save_json(state_path, state)
            last_tick = time.time()

        if run_for_s > 0 and (time.time() - start_ts) >= run_for_s:
            break

        if not events:
            time.sleep(max(0.05, poll_s))

    state["updated_iso"] = now_iso_utc()
    _save_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
