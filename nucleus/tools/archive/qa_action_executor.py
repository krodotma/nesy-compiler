#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.dont_write_bytecode = True

try:
    from .agent_bus import resolve_bus_paths, emit_event, default_actor
    from .qa_tool_queue import enqueue, _default_queue_path
except Exception:  # pragma: no cover
    from agent_bus import resolve_bus_paths, emit_event, default_actor
    from qa_tool_queue import enqueue, _default_queue_path


DEFAULT_ACTIONS: Dict[str, Dict[str, Any]] = {
    "restart.bus-bridge": {
        "command": "systemctl restart pluribus-bus-bridge",
        "requires_root": True,
    },
    "restart.dashboard": {
        "command": "systemctl restart pluribus-dashboard",
        "requires_root": True,
    },
    "restart.vps-session-daemon": {
        "command": "systemctl restart pluribus-vps-session-daemon",
        "requires_root": True,
    },
    "restart.browser-session-daemon": {
        "command": "systemctl restart pluribus-browser-session-daemon",
        "requires_root": True,
    },
    "telemetry.reduce": {
        "command": None,
        "manual": True,
    },
    "inspect.logs": {
        "command": None,
        "manual": True,
    },
}


def _default_state_path() -> str:
    root = os.environ.get("PLURIBUS_ROOT") or "/pluribus"
    return os.path.join(root, ".pluribus", "index", "qa", "action_executor.ndjson")


def _ensure_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "a", encoding="utf-8"):
            pass


def _open_tail(path: str):
    handle = open(path, "r", encoding="utf-8", errors="replace")
    handle.seek(0, os.SEEK_END)
    inode = os.fstat(handle.fileno()).st_ino
    return handle, inode


def _emit(bus_dir: str, topic: str, data: Dict[str, Any], *, level: str = "info") -> None:
    paths = resolve_bus_paths(bus_dir)
    emit_event(
        paths,
        topic=topic,
        kind="metric",
        level=level,
        actor=default_actor(),
        data=data,
        trace_id=data.get("trace_id"),
        run_id=data.get("run_id"),
        durable=False,
    )


def build_allowed_actions(extra_specs: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    allowed = {k: dict(v) for k, v in DEFAULT_ACTIONS.items()}
    for spec in extra_specs:
        if not spec:
            continue
        if "=" in spec:
            action_id, command = spec.split("=", 1)
            action_id = action_id.strip()
            command = command.strip()
            if action_id:
                allowed[action_id] = {
                    "command": command or None,
                    "requires_root": command.startswith("systemctl") if command else False,
                }
        else:
            allowed[spec.strip()] = {"command": None, "manual": True}
    return allowed


def action_key(plan_id: str, action_id: str) -> str:
    return f"{plan_id}:{action_id}"


def load_handled_keys(path: str, limit: int = 2000) -> set[str]:
    if not os.path.exists(path):
        return set()
    recent = deque(maxlen=limit)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                recent.append(json.loads(line))
            except Exception:
                continue
    handled = set()
    for item in recent:
        key = item.get("action_key")
        if key:
            handled.add(str(key))
    return handled


def record_state(path: str, record: Dict[str, Any]) -> None:
    _ensure_file(path)
    line = json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def evaluate_action(
    action: Dict[str, Any],
    allowed: Dict[str, Dict[str, Any]],
    *,
    allow_root: bool,
) -> Tuple[str, Optional[str]]:
    action_id = action.get("id") or action.get("action_id")
    if not action_id:
        return "missing_action_id", None
    if action_id not in allowed:
        return "not_allowed", None
    spec = allowed[action_id]
    command = spec.get("command")
    if spec.get("manual"):
        return "manual", None
    if not command:
        return "no_command", None
    if action.get("command") and action.get("command") != command:
        return "command_mismatch", None
    if spec.get("requires_root") and not allow_root:
        return "root_not_allowed", None
    return "ok", command


def extract_actions(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = event.get("data") or {}
    actions = data.get("actions")
    if isinstance(actions, list):
        return [a for a in actions if isinstance(a, dict)]
    plan = data.get("action_plan") or data.get("actionPlan")
    if isinstance(plan, dict):
        actions = plan.get("actions")
        if isinstance(actions, list):
            return [a for a in actions if isinstance(a, dict)]
    return []


def process_event(
    event: Dict[str, Any],
    *,
    allowed: Dict[str, Dict[str, Any]],
    handled_keys: set[str],
    state_path: str,
    bus_dir: str,
    queue_path: str,
    emit_bus: bool,
    allow_root: bool,
    mode: str,
    timeout_s: int,
) -> None:
    data = event.get("data") or {}
    decision = data.get("decision")
    if decision and decision != "approved":
        return

    plan_id = data.get("action_plan_id") or data.get("actionPlanId") or event.get("id") or "unknown"
    actions = extract_actions(event)
    if not actions:
        record_state(state_path, {"ts": time.time(), "action_key": f"{plan_id}:none", "status": "skipped"})
        if emit_bus:
            _emit(bus_dir, "qa.action.exec.skipped", {"plan_id": plan_id, "reason": "no_actions"})
        return

    for action in actions:
        action_id = action.get("id") or action.get("action_id") or "unknown"
        key = action_key(plan_id, action_id)
        if key in handled_keys:
            if emit_bus:
                _emit(bus_dir, "qa.action.exec.skipped", {"plan_id": plan_id, "action_id": action_id, "reason": "duplicate"})
            continue
        status, command = evaluate_action(action, allowed, allow_root=allow_root)
        if status != "ok" or not command:
            record_state(state_path, {"ts": time.time(), "action_key": key, "status": "skipped", "reason": status})
            handled_keys.add(key)
            if emit_bus:
                _emit(
                    bus_dir,
                    "qa.action.exec.skipped",
                    {"plan_id": plan_id, "action_id": action_id, "reason": status},
                )
            continue

        meta = {
            "action_id": action_id,
            "action_plan_id": plan_id,
            "source_event": event.get("id"),
            "source_topic": event.get("topic"),
            "timeout_s": timeout_s,
        }

        if mode == "queue":
            job = enqueue(queue_path, command, meta)
            record_state(state_path, {"ts": time.time(), "action_key": key, "status": "queued", "job_id": job["id"]})
            handled_keys.add(key)
            if emit_bus:
                _emit(
                    bus_dir,
                    "qa.action.exec.enqueued",
                    {"plan_id": plan_id, "action_id": action_id, "job_id": job["id"], "command": command},
                )
            continue

        try:
            result = subprocess.run(command, shell=True, check=False, timeout=timeout_s)
            exec_status = "completed" if result.returncode == 0 else "failed"
            record_state(state_path, {"ts": time.time(), "action_key": key, "status": exec_status, "returncode": result.returncode})
            handled_keys.add(key)
            if emit_bus:
                _emit(
                    bus_dir,
                    f"qa.action.exec.{exec_status}",
                    {"plan_id": plan_id, "action_id": action_id, "command": command, "returncode": result.returncode},
                    level="info" if exec_status == "completed" else "warn",
                )
        except subprocess.TimeoutExpired:
            record_state(state_path, {"ts": time.time(), "action_key": key, "status": "failed", "reason": "timeout"})
            handled_keys.add(key)
            if emit_bus:
                _emit(
                    bus_dir,
                    "qa.action.exec.failed",
                    {"plan_id": plan_id, "action_id": action_id, "command": command, "reason": "timeout"},
                    level="warn",
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="QA action executor (approval-gated)")
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus")
    parser.add_argument("--state-path", default=_default_state_path())
    parser.add_argument("--queue-path", default=_default_queue_path())
    parser.add_argument("--poll", type=float, default=0.25)
    parser.add_argument("--allow-action", action="append", default=[])
    parser.add_argument("--emit-bus", action="store_true")
    parser.add_argument("--allow-root", action="store_true")
    parser.add_argument("--mode", choices=("queue", "direct"), default=os.environ.get("QA_ACTION_EXEC_MODE", "queue"))
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args()

    allowed = build_allowed_actions(args.allow_action)
    handled_keys = load_handled_keys(args.state_path)

    paths = resolve_bus_paths(args.bus_dir)
    _ensure_file(paths.events_path)
    handle, inode = _open_tail(paths.events_path)

    while True:
        line = handle.readline()
        if line:
            try:
                raw = json.loads(line)
            except Exception:
                continue
            if raw.get("topic") != "qa.action.review.approved":
                continue
            process_event(
                raw,
                allowed=allowed,
                handled_keys=handled_keys,
                state_path=args.state_path,
                bus_dir=args.bus_dir,
                queue_path=args.queue_path,
                emit_bus=args.emit_bus,
                allow_root=args.allow_root,
                mode=args.mode,
                timeout_s=args.timeout_s,
            )
            continue
        time.sleep(args.poll)
        try:
            stats = os.stat(paths.events_path)
        except FileNotFoundError:
            continue
        if stats.st_ino != inode or stats.st_size < handle.tell():
            handle.close()
            handle, inode = _open_tail(paths.events_path)


if __name__ == "__main__":
    raise SystemExit(main())
