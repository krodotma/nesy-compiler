#!/usr/bin/env python3
"""
RD Tasks Worker — optional execution substrate for rd.tasks.dispatch

This daemon makes `rd.tasks.dispatch` optionally *actionable*:
- Observes `rd.tasks.dispatch` events (kind=request).
- If payload.constraints.execute == true and the worker is targeted, it executes
  allowlisted actions attached to tasks.
- Emits append-only evidence:
  - rd.tasks.task.start / rd.tasks.task.end
  - rd.tasks.action.start / rd.tasks.action.end
  - rd.tasks.done (kind=response) when all runnable tasks complete

Safety model:
- Default is safe: it will NOT execute unless `constraints.execute` is true.
- Executes only a narrow allowlist (python scripts under nucleus/).

This is intentionally conservative: it is a bridge from coordination to
automation, not a general remote-code execution engine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_bus_dir(args_bus_dir: str | None) -> str:
    return args_bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"


def default_actor(args_actor: str | None) -> str:
    return args_actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "rd-tasks-worker"


def ensure_events_file(bus_dir: str) -> Path:
    p = Path(bus_dir) / "events.ndjson"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def tail_events(events_path: Path, *, since_ts: float, poll_s: float, stop_at_ts: float | None):
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
            if obj.get("topic") != "rd.tasks.dispatch":
                continue
            try:
                ts = float(obj.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            if ts < since_ts:
                continue
            yield obj


def emit_bus(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> None:
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


def _targeted_to_me(*, actor: str, targets: list[str]) -> bool:
    me = actor.strip().lower()
    want = {str(t).strip().lower() for t in (targets or []) if str(t).strip()}
    if not want:
        return True
    return me in want or "all" in want


def _is_allowlisted_action(argv: list[str]) -> tuple[bool, str]:
    """Allowlist only python invocations under /pluribus/nucleus/."""
    if not argv:
        return False, "empty argv"
    exe = str(argv[0])
    if exe not in {"python3", sys.executable}:
        return False, "only python3 actions are allowed"
    if len(argv) < 2:
        return False, "python action missing script path"
    script = Path(argv[1])
    if not script.is_absolute():
        # resolve relative to /pluribus root
        script = (Path("/pluribus") / script).resolve()
    ok_prefix = Path("/pluribus/nucleus").resolve()
    try:
        script_res = script.resolve()
    except Exception:
        return False, "cannot resolve script path"
    if not str(script_res).startswith(str(ok_prefix)):
        return False, f"script not under {ok_prefix}"
    if not script_res.exists():
        return False, "script does not exist"
    return True, "ok"


def run_action(
    *,
    bus_dir: str,
    actor: str,
    req_id: str,
    task_id: str,
    action: dict[str, Any],
) -> dict[str, Any]:
    aid = str(action.get("id") or "")
    argv = action.get("argv") if isinstance(action.get("argv"), list) else []
    argv = [str(x) for x in argv if str(x).strip()]
    cwd = str(action.get("cwd") or "/pluribus")
    try:
        timeout_s = float(action.get("timeout_s") or 120.0)
    except Exception:
        timeout_s = 120.0

    allow_ok, allow_reason = _is_allowlisted_action(argv)
    emit_bus(
        bus_dir,
        topic="rd.tasks.action.start",
        kind="artifact",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "task_id": task_id,
            "action_id": aid,
            "argv": argv,
            "cwd": cwd,
            "allowlisted": allow_ok,
            "allow_reason": allow_reason,
            "iso": now_iso_utc(),
        },
    )
    if not allow_ok:
        emit_bus(
            bus_dir,
            topic="rd.tasks.action.end",
            kind="artifact",
            level="warn",
            actor=actor,
            data={
                "req_id": req_id,
                "task_id": task_id,
                "action_id": aid,
                "ok": False,
                "exit": None,
                "stderr": allow_reason,
                "stdout": "",
                "iso": now_iso_utc(),
            },
        )
        return {"ok": False, "exit": None, "reason": allow_reason}

    try:
        p = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "PLURIBUS_BUS_DIR": bus_dir, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
        ok = int(p.returncode) == 0
        emit_bus(
            bus_dir,
            topic="rd.tasks.action.end",
            kind="artifact",
            level="info" if ok else "error",
            actor=actor,
            data={
                "req_id": req_id,
                "task_id": task_id,
                "action_id": aid,
                "ok": ok,
                "exit": int(p.returncode),
                "stdout": (p.stdout or "")[-8000:],
                "stderr": (p.stderr or "")[-8000:],
                "iso": now_iso_utc(),
            },
        )
        return {"ok": ok, "exit": int(p.returncode)}
    except Exception as e:
        emit_bus(
            bus_dir,
            topic="rd.tasks.action.end",
            kind="artifact",
            level="error",
            actor=actor,
            data={
                "req_id": req_id,
                "task_id": task_id,
                "action_id": aid,
                "ok": False,
                "exit": None,
                "stderr": str(e),
                "stdout": "",
                "iso": now_iso_utc(),
            },
        )
        return {"ok": False, "exit": None, "reason": str(e)}


def handle_dispatch(*, bus_dir: str, actor: str, trig: dict[str, Any]) -> bool:
    """Handle a single rd.tasks.dispatch trigger event. Returns True if processed."""
    data = trig.get("data") if isinstance(trig.get("data"), dict) else {}
    req_id = str(data.get("req_id") or "")
    constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
    execute = bool(constraints.get("execute")) if isinstance(constraints, dict) else False
    targets = data.get("targets") if isinstance(data.get("targets"), list) else []
    if not execute:
        return False
    if not _targeted_to_me(actor=actor, targets=targets):
        return False

    tasks = data.get("tasks") if isinstance(data.get("tasks"), list) else []
    ok_all = True
    completed: list[str] = []
    failed: list[str] = []

    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id") or "")
        actions = t.get("actions") if isinstance(t.get("actions"), list) else []
        emit_bus(
            bus_dir,
            topic="rd.tasks.task.start",
            kind="artifact",
            level="info",
            actor=actor,
            data={"req_id": req_id, "task_id": tid, "iso": now_iso_utc()},
        )
        task_ok = True
        for a in actions:
            if not isinstance(a, dict):
                continue
            res = run_action(bus_dir=bus_dir, actor=actor, req_id=req_id, task_id=tid, action=a)
            if not res.get("ok"):
                task_ok = False
        emit_bus(
            bus_dir,
            topic="rd.tasks.task.end",
            kind="artifact",
            level="info" if task_ok else "error",
            actor=actor,
            data={"req_id": req_id, "task_id": tid, "ok": task_ok, "iso": now_iso_utc()},
        )
        if task_ok:
            completed.append(tid)
        else:
            failed.append(tid)
            ok_all = False

    emit_bus(
        bus_dir,
        topic="rd.tasks.done",
        kind="response",
        level="info" if ok_all else "warn",
        actor=actor,
        data={
            "req_id": req_id,
            "ok": ok_all,
            "completed": completed,
            "failed": failed,
            "iso": now_iso_utc(),
        },
    )
    if req_id:
        emit_bus(
            bus_dir,
            topic="infer_sync.response",
            kind="response",
            level="info" if ok_all else "warn",
            actor=actor,
            data={
                "req_id": req_id,
                "summary": f"rd.tasks.done ok={ok_all} completed={len(completed)} failed={len(failed)}",
                "artifact": None,
                "next_action": "await_ckin",
                "risk": "low" if ok_all else "med",
                "source": "rd_tasks_worker",
            },
        )
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="rd_tasks_worker.py", description="Daemon: optionally execute rd.tasks.dispatch actions (when enabled).")
    ap.add_argument("--bus-dir", default=None)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--poll", default="0.25")
    ap.add_argument("--run-for-s", default="0", help="Run for N seconds then exit (0 = run forever).")
    ap.add_argument("--since-ts", default=None, help="Only react to triggers >= this UNIX timestamp.")
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
        topic="rd.tasks.worker.ready",
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
        handle_dispatch(bus_dir=bus_dir, actor=actor, trig=trig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
