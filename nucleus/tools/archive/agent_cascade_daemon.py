#!/usr/bin/env python3
"""
Agent Cascade Daemon - Spawns CLI agents in response to bus events.

This daemon completes the automation loop:
1. Watches for `a2a.negotiate.request` (kind=request) targeting known agents
2. Spawns the appropriate CLI agent with the task
3. Emits `a2a.negotiate.response` (agree) and task completion events

This is the missing piece that makes OHM's dispatch truly automatic.

Environment:
  PLURIBUS_ACTOR: Actor name (default: agent-cascade)
  PLURIBUS_BUS_DIR: Bus directory (default: /pluribus/.pluribus/bus)
  CASCADE_POLL_S: Poll interval (default: 1.0)
  CASCADE_MAX_CONCURRENT: Max concurrent agent spawns (default: 3)
  CASCADE_DRY_RUN: If "1", log but don't spawn (default: 0)

Supported agents:
  - claude: /usr/local/bin/claude
  - codex: /usr/local/bin/codex
  - gemini: /usr/bin/gemini (if available)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import threading
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_bus_dir(v: str | None) -> Path:
    if v:
        return Path(v).expanduser().resolve()
    env = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path("/pluribus/.pluribus/bus")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> None:
    """Emit event to bus."""
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "id": uuid.uuid4().hex,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


@dataclass
class AgentConfig:
    name: str
    path: str
    available: bool = False
    args_template: list[str] = field(default_factory=list)


KNOWN_AGENTS: dict[str, AgentConfig] = {
    "claude": AgentConfig(
        name="claude",
        path="/usr/local/bin/claude",
        args_template=["--dangerously-skip-permissions", "-p"],
    ),
    "codex": AgentConfig(
        name="codex",
        path="/usr/local/bin/codex",
        args_template=["--dangerously-skip-permissions", "-p"],
    ),
    "gemini": AgentConfig(
        name="gemini",
        path="/usr/bin/gemini",
        args_template=["-p"],
    ),
}


def check_agent_availability() -> None:
    """Check which agents are available on the system."""
    for name, cfg in KNOWN_AGENTS.items():
        cfg.available = os.path.isfile(cfg.path) and os.access(cfg.path, os.X_OK)


@dataclass
class PendingTask:
    req_id: str
    target: str
    task_description: str
    task_outline: Optional[str]
    constraints: dict
    dispatch_req_id: str
    created_ts: float
    attempts: int = 0
    last_attempt_ts: float = 0.0
    status: str = "pending"


class AgentCascadeDaemon:
    def __init__(
        self,
        bus_dir: Path,
        actor: str,
        poll_s: float = 1.0,
        max_concurrent: int = 3,
        dry_run: bool = False,
    ):
        self.bus_dir = bus_dir
        self.actor = actor
        self.poll_s = poll_s
        self.max_concurrent = max_concurrent
        self.dry_run = dry_run
        self.events_path = bus_dir / "events.ndjson"

        self.pending: dict[str, PendingTask] = {}
        self.active: dict[str, threading.Thread] = {}
        self.responded: set[str] = set()
        self.running = True
        self.last_file_pos = 0

        check_agent_availability()

    def emit(self, topic: str, kind: str, level: str, data: dict) -> None:
        emit_bus(self.bus_dir, topic=topic, kind=kind, level=level, actor=self.actor, data=data)

    def load_responded_ids(self) -> None:
        """Load already-responded request IDs from bus."""
        if not self.events_path.exists():
            return
        with self.events_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("topic") == "a2a.negotiate.response" and e.get("kind") == "response":
                        d = e.get("data", {})
                        req_id = d.get("req_id") or d.get("request_id")
                        if req_id:
                            self.responded.add(req_id)
                except Exception:
                    continue

    def process_new_events(self) -> None:
        """Process new events since last position."""
        if not self.events_path.exists():
            return

        with self.events_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(self.last_file_pos)
            for line in f:
                try:
                    e = json.loads(line.strip())
                except Exception:
                    continue

                topic = e.get("topic", "")
                kind = e.get("kind", "")

                # Watch for a2a.negotiate.request targeting our known agents
                if topic == "a2a.negotiate.request" and kind == "request":
                    self._handle_negotiate_request(e)

                # Track responses we've seen
                elif topic == "a2a.negotiate.response" and kind == "response":
                    d = e.get("data", {})
                    req_id = d.get("req_id") or d.get("request_id")
                    if req_id:
                        self.responded.add(req_id)

            self.last_file_pos = f.tell()

    def _handle_negotiate_request(self, event: dict) -> None:
        """Handle incoming a2a.negotiate.request."""
        data = event.get("data", {})
        req_id = data.get("req_id") or data.get("request_id", "")
        target = (data.get("target") or "").lower()

        if not req_id or not target:
            return

        # Skip if already responded
        if req_id in self.responded:
            return

        # Skip if not targeting a known agent
        if target not in KNOWN_AGENTS:
            return

        # Skip if agent not available
        if not KNOWN_AGENTS[target].available:
            self.emit(
                "a2a.negotiate.response",
                "response",
                "warn",
                {
                    "req_id": req_id,
                    "decision": "reject",
                    "reason": f"Agent {target} not available on this system",
                    "iso": now_iso_utc(),
                },
            )
            self.responded.add(req_id)
            return

        # Create pending task
        task = PendingTask(
            req_id=req_id,
            target=target,
            task_description=data.get("task_description") or data.get("task") or "",
            task_outline=data.get("task_outline"),
            constraints=data.get("constraints", {}),
            dispatch_req_id=data.get("dispatch_req_id", ""),
            created_ts=time.time(),
        )

        self.pending[req_id] = task
        print(f"[CASCADE] Queued task {req_id} for {target}: {task.task_description[:80]}...")

    def spawn_agent(self, task: PendingTask) -> None:
        """Spawn an agent CLI to handle the task."""
        cfg = KNOWN_AGENTS.get(task.target)
        if not cfg or not cfg.available:
            return

        # Build prompt from task description
        prompt = task.task_description
        if task.task_outline:
            prompt = f"{prompt}\n\nTask outline:\n{task.task_outline}"

        # Add constraints context
        constraints = task.constraints
        if constraints:
            constraint_str = ", ".join(f"{k}={v}" for k, v in constraints.items())
            prompt = f"{prompt}\n\nConstraints: {constraint_str}"

        # Emit acceptance
        self.emit(
            "a2a.negotiate.response",
            "response",
            "info",
            {
                "req_id": task.req_id,
                "decision": "agree",
                "agent": task.target,
                "iso": now_iso_utc(),
            },
        )
        self.responded.add(task.req_id)

        # Emit task start
        self.emit(
            "agent.cascade.task.start",
            "metric",
            "info",
            {
                "req_id": task.req_id,
                "target": task.target,
                "dispatch_req_id": task.dispatch_req_id,
                "iso": now_iso_utc(),
            },
        )

        if self.dry_run:
            print(f"[CASCADE] DRY_RUN: Would spawn {task.target} with prompt: {prompt[:100]}...")
            task.status = "dry_run"
            return

        # Build command
        cmd = [cfg.path] + cfg.args_template + [prompt]

        print(f"[CASCADE] Spawning {task.target} for task {task.req_id}")

        try:
            # Run agent with timeout (30 minutes max)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd="/pluribus",
                env={
                    **os.environ,
                    "PLURIBUS_ACTOR": task.target,
                    "PLURIBUS_CASCADE_REQ_ID": task.req_id,
                },
            )

            task.status = "completed" if result.returncode == 0 else "failed"

            # Emit completion
            self.emit(
                "agent.cascade.task.end",
                "metric",
                "info" if task.status == "completed" else "error",
                {
                    "req_id": task.req_id,
                    "target": task.target,
                    "status": task.status,
                    "returncode": result.returncode,
                    "stdout_lines": len(result.stdout.splitlines()) if result.stdout else 0,
                    "stderr_lines": len(result.stderr.splitlines()) if result.stderr else 0,
                    "iso": now_iso_utc(),
                },
            )

            print(f"[CASCADE] {task.target} completed task {task.req_id} with status {task.status}")

        except subprocess.TimeoutExpired:
            task.status = "timeout"
            self.emit(
                "agent.cascade.task.end",
                "metric",
                "error",
                {
                    "req_id": task.req_id,
                    "target": task.target,
                    "status": "timeout",
                    "iso": now_iso_utc(),
                },
            )
            print(f"[CASCADE] {task.target} timed out on task {task.req_id}")

        except Exception as ex:
            task.status = "error"
            self.emit(
                "agent.cascade.task.end",
                "metric",
                "error",
                {
                    "req_id": task.req_id,
                    "target": task.target,
                    "status": "error",
                    "error": str(ex),
                    "iso": now_iso_utc(),
                },
            )
            print(f"[CASCADE] {task.target} error on task {task.req_id}: {ex}")

    def process_pending(self) -> None:
        """Process pending tasks, spawning agents as capacity allows."""
        # Clean up completed threads
        completed = [k for k, t in self.active.items() if not t.is_alive()]
        for k in completed:
            del self.active[k]

        # Check capacity
        if len(self.active) >= self.max_concurrent:
            return

        # Find next task to process
        for req_id, task in list(self.pending.items()):
            if task.status != "pending":
                continue
            if req_id in self.active:
                continue

            # Spawn in thread
            task.status = "running"
            task.attempts += 1
            task.last_attempt_ts = time.time()

            t = threading.Thread(target=self.spawn_agent, args=(task,), daemon=True)
            t.start()
            self.active[req_id] = t

            if len(self.active) >= self.max_concurrent:
                break

    def run(self) -> None:
        """Main daemon loop."""
        print(f"[CASCADE] Starting agent cascade daemon (actor={self.actor}, dry_run={self.dry_run})")
        print(f"[CASCADE] Available agents: {[k for k, v in KNOWN_AGENTS.items() if v.available]}")

        self.emit(
            "agent.cascade.ready",
            "metric",
            "info",
            {
                "actor": self.actor,
                "available_agents": [k for k, v in KNOWN_AGENTS.items() if v.available],
                "max_concurrent": self.max_concurrent,
                "dry_run": self.dry_run,
                "iso": now_iso_utc(),
            },
        )

        # Load existing responses
        self.load_responded_ids()
        self.last_file_pos = self.events_path.stat().st_size if self.events_path.exists() else 0

        while self.running:
            try:
                self.process_new_events()
                self.process_pending()
                time.sleep(self.poll_s)
            except KeyboardInterrupt:
                self.running = False
            except Exception as ex:
                print(f"[CASCADE] Error in main loop: {ex}")
                time.sleep(5)

        print("[CASCADE] Shutting down...")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_cascade_daemon.py",
        description="Spawns CLI agents in response to a2a.negotiate.request bus events.",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    p.add_argument("--actor", default=None, help="Actor name")
    p.add_argument("--poll", type=float, default=1.0, help="Poll interval seconds")
    p.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent agent spawns")
    p.add_argument("--dry-run", action="store_true", help="Log but don't spawn agents")
    p.add_argument("--once", action="store_true", help="Process once then exit")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    actor = args.actor or os.environ.get("PLURIBUS_ACTOR") or "agent-cascade"
    bus_dir = default_bus_dir(args.bus_dir)
    poll_s = args.poll or float(os.environ.get("CASCADE_POLL_S", "1.0"))
    max_concurrent = args.max_concurrent or int(os.environ.get("CASCADE_MAX_CONCURRENT", "3"))
    dry_run = args.dry_run or os.environ.get("CASCADE_DRY_RUN") == "1"

    daemon = AgentCascadeDaemon(
        bus_dir=bus_dir,
        actor=actor,
        poll_s=poll_s,
        max_concurrent=max_concurrent,
        dry_run=dry_run,
    )

    if args.once:
        daemon.load_responded_ids()
        daemon.last_file_pos = 0  # Process all events
        daemon.process_new_events()
        daemon.process_pending()
        # Wait for active tasks
        for t in daemon.active.values():
            t.join(timeout=60)
        return 0

    daemon.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
