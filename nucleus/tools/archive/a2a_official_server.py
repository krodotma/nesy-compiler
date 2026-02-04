#!/usr/bin/env python3
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools.venv_exec import maybe_reexec_in_realagents_venv  # noqa: E402

maybe_reexec_in_realagents_venv()

import uvicorn  # noqa: E402
from a2a.server.agent_execution.agent_executor import AgentExecutor  # noqa: E402
from a2a.server.apps import A2AStarletteApplication  # noqa: E402
from a2a.server.events.event_queue import EventQueue  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore  # noqa: E402
from a2a.types import (  # noqa: E402
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    Message,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from agent_bus import default_actor, emit_event, resolve_bus_paths  # type: ignore # noqa: E402


Json = dict[str, Any]


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _plurichat_oneshot(prompt: str, *, provider: str, bus_dir: str | None, actor: str) -> Json:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "nucleus" / "tools" / "plurichat.py"),
        "--mode",
        "oneshot",
        "--provider",
        provider,
        "--ask",
        prompt,
        "--json-output",
    ]
    if bus_dir:
        cmd.extend(["--bus-dir", bus_dir])
    if actor:
        cmd.extend(["--actor", actor])
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {"success": False, "error": proc.stderr.strip() or f"plurichat exit={proc.returncode}"}
    try:
        out = json.loads(proc.stdout.strip() or "{}")
        if isinstance(out, dict):
            return out
    except Exception:
        pass
    return {"success": True, "text": proc.stdout.strip()}


class PluribusA2AExecutor(AgentExecutor):
    def __init__(self, *, provider: str, bus_dir: str | None, actor: str):
        self.provider = provider
        self.bus_dir = bus_dir
        self.actor = actor
        self.paths = resolve_bus_paths(bus_dir) if bus_dir else None

    async def execute(self, context, event_queue: EventQueue) -> None:  # type: ignore[override]
        msg = context.message
        task_id = context.task_id or (msg.task_id if msg else None) or "task-" + str(uuid.uuid4())[:8]
        context_id = context.context_id or (msg.context_id if msg else None) or "ctx-" + str(uuid.uuid4())[:8]
        prompt = context.get_user_input()

        if self.paths:
            emit_event(
                self.paths,
                topic="a2a.official.request",
                kind="request",
                level="info",
                actor=self.actor,
                data={"task_id": task_id, "context_id": context_id, "message_id": getattr(msg, "message_id", None), "provider": self.provider},
                trace_id=None,
                run_id=None,
                durable=False,
            )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                final=False,
                status=TaskStatus(state=TaskState.working, timestamp=now_iso_utc()),
            )
        )

        out = _plurichat_oneshot(prompt, provider=self.provider, bus_dir=self.bus_dir, actor=self.actor)
        text = str(out.get("text") or "")
        ok = bool(out.get("success")) and not out.get("error")

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                append=False,
                last_chunk=True,
                artifact=Artifact(
                    artifact_id="reply",
                    name="reply",
                    description="Pluribus response via PluriChat",
                    parts=[TextPart(text=text)],
                    metadata={"provider": self.provider, "ok": ok},
                ),
            )
        )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                final=True,
                status=TaskStatus(
                    state=TaskState.completed if ok else TaskState.failed,
                    timestamp=now_iso_utc(),
                    message=Message(
                        message_id=str(uuid.uuid4()),
                        role="agent",
                        context_id=context_id,
                        task_id=task_id,
                        parts=[TextPart(text="ok" if ok else f"error: {out.get('error')}")],
                    ),
                ),
            )
        )

        if self.paths:
            emit_event(
                self.paths,
                topic="a2a.official.response",
                kind="response",
                level="info" if ok else "error",
                actor=self.actor,
                data={"task_id": task_id, "context_id": context_id, "ok": ok, "provider": self.provider},
                trace_id=None,
                run_id=None,
                durable=False,
            )

    async def cancel(self, context, event_queue: EventQueue) -> None:  # type: ignore[override]
        msg = context.message
        task_id = context.task_id or (msg.task_id if msg else None) or "task-" + str(uuid.uuid4())[:8]
        context_id = context.context_id or (msg.context_id if msg else None) or "ctx-" + str(uuid.uuid4())[:8]
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                final=True,
                status=TaskStatus(state=TaskState.canceled, timestamp=now_iso_utc()),
            )
        )


def build_agent_card(*, url: str) -> AgentCard:
    return AgentCard(
        name="pluribus-a2a",
        description="Pluribus A2A agent (routes prompts via PluriChat + Pluribus router).",
        url=url,
        version="0.1.0",
        protocolVersion="0.3.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True, pushNotifications=False, stateTransitionHistory=True),
        skills=[
            AgentSkill(
                id="plurichat",
                name="PluriChat",
                description="One-shot prompt routing via PluriChat (provider=auto/mock/etc).",
                tags=["pluribus", "router", "plurichat"],
                examples=["Summarize this repo in 5 bullets", "List tools available in the rhizome MCP server"],
            )
        ],
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="a2a_official_server.py", description="Serve Pluribus as an official A2A agent (a2a-sdk).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--provider", default="auto", help="PluriChat provider (auto/mock/gemini-3/claude/codex).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (defaults: $PLURIBUS_BUS_DIR).")
    ap.add_argument("--actor", default=None, help="Actor (defaults: $PLURIBUS_ACTOR or user).")
    args = ap.parse_args(argv)

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    actor = args.actor or os.environ.get("PLURIBUS_ACTOR") or default_actor()

    url = f"http://{args.host}:{args.port}"
    card = build_agent_card(url=url)

    executor = PluribusA2AExecutor(provider=str(args.provider), bus_dir=bus_dir, actor=actor)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    server = A2AStarletteApplication(agent_card=card, http_handler=handler)
    app = server.build()
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
