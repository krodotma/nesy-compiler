#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

sys.dont_write_bytecode = True

try:
    from .agent_bus import resolve_bus_paths, emit_event, default_actor
except Exception:  # pragma: no cover
    from agent_bus import resolve_bus_paths, emit_event, default_actor


def _default_queue_path() -> str:
    root = os.environ.get("PLURIBUS_ROOT") or "/pluribus"
    return os.path.join(root, ".pluribus", "index", "tool_queue", "queue.ndjson")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_event(path: str, event: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    line = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def load_events(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
            if limit and len(events) >= limit:
                break
    return events


def build_state(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    jobs: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        job_id = ev.get("id")
        if not job_id:
            continue
        jobs[job_id] = {**jobs.get(job_id, {}), **ev}
    counts = {"queued": 0, "started": 0, "completed": 0, "failed": 0}
    queue: List[Dict[str, Any]] = []
    for job in jobs.values():
        status = job.get("status", "queued")
        if status in counts:
            counts[status] += 1
        if status == "queued":
            queue.append(job)
    queue.sort(key=lambda j: j.get("ts", 0))
    active = counts.get("started", 0)
    return {"jobs": jobs, "counts": counts, "active": active, "queue": queue}


def select_next_job(state: Dict[str, Any], max_active: int) -> Optional[Dict[str, Any]]:
    if state["active"] >= max_active:
        return None
    if not state["queue"]:
        return None
    return state["queue"][0]


def _emit(bus_dir: str, topic: str, data: Dict[str, Any]) -> None:
    paths = resolve_bus_paths(bus_dir)
    emit_event(
        paths,
        topic=topic,
        kind="metric",
        level="info",
        actor=default_actor(),
        data=data,
        trace_id=data.get("trace_id"),
        run_id=data.get("run_id"),
        durable=False,
    )


def enqueue(path: str, cmd: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    job = {
        "id": str(uuid.uuid4()),
        "status": "queued",
        "ts": time.time(),
        "cmd": cmd,
        "meta": meta or {},
    }
    append_event(path, job)
    return job


def update_status(path: str, job_id: str, status: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    event = {
        "id": job_id,
        "status": status,
        "ts": time.time(),
        "meta": meta or {},
    }
    append_event(path, event)
    return event


def run_loop(
    path: str,
    max_active: int,
    poll_s: float,
    execute: bool,
    emit_bus: bool,
    bus_dir: str,
    default_timeout_s: int,
) -> None:
    while True:
        events = load_events(path)
        state = build_state(events)
        job = select_next_job(state, max_active=max_active)
        if not job:
            time.sleep(poll_s)
            continue
        update_status(path, job["id"], "started")
        if emit_bus:
            _emit(bus_dir, "qa.tool_queue.started", {"job_id": job["id"], "cmd": job.get("cmd"), "meta": job.get("meta")})
        if execute:
            timeout_s = int(job.get("meta", {}).get("timeout_s") or default_timeout_s)
            try:
                result = subprocess.run(job["cmd"], shell=True, check=False, timeout=timeout_s)
                status = "completed" if result.returncode == 0 else "failed"
                update_status(path, job["id"], status, {"returncode": result.returncode})
                if emit_bus:
                    _emit(
                        bus_dir,
                        f"qa.tool_queue.{status}",
                        {"job_id": job["id"], "returncode": result.returncode, "meta": job.get("meta")},
                    )
            except subprocess.TimeoutExpired:
                update_status(path, job["id"], "failed", {"error": "timeout"})
                if emit_bus:
                    _emit(bus_dir, "qa.tool_queue.failed", {"job_id": job["id"], "error": "timeout", "meta": job.get("meta")})
        time.sleep(poll_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="QA tool queue manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enqueue_cmd = sub.add_parser("enqueue")
    enqueue_cmd.add_argument("--job-cmd", required=True, dest="job_cmd")
    enqueue_cmd.add_argument("--meta", default="{}")
    enqueue_cmd.add_argument("--queue-path", default=_default_queue_path())
    enqueue_cmd.add_argument("--emit-bus", action="store_true")
    enqueue_cmd.add_argument("--bus-dir", default="/pluribus/.pluribus/bus")

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--queue-path", default=_default_queue_path())
    run_cmd.add_argument("--max-active", type=int, default=1)
    run_cmd.add_argument("--poll-s", type=float, default=2.0)
    run_cmd.add_argument("--execute", action="store_true")
    run_cmd.add_argument("--emit-bus", action="store_true")
    run_cmd.add_argument("--bus-dir", default="/pluribus/.pluribus/bus")
    run_cmd.add_argument("--default-timeout-s", type=int, default=300)

    args = parser.parse_args()
    if args.cmd == "enqueue":
        meta = json.loads(args.meta)
        job = enqueue(args.queue_path, args.job_cmd, meta)
        if args.emit_bus:
            _emit(args.bus_dir, "qa.tool_queue.enqueued", {"job_id": job["id"], "cmd": job["cmd"]})
        print(job["id"])
        return 0
    if args.cmd == "run":
        run_loop(
            path=args.queue_path,
            max_active=args.max_active,
            poll_s=args.poll_s,
            execute=args.execute,
            emit_bus=args.emit_bus,
            bus_dir=args.bus_dir,
            default_timeout_s=args.default_timeout_s,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
