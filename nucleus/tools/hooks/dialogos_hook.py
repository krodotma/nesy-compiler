#!/usr/bin/env python3
"""Claude Code hook for Dialogos trace integration.

Captures UserPromptSubmit and Stop events, emits dialogos.submit to bus
for ground truth recovery via PBRESUME.

Install: Add to ~/.claude/settings.json or .claude/settings.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus_event(*, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    """Emit event to Pluribus bus."""
    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus")
    events_path = bus_dir / "events.ndjson"

    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": actor,
            "host": os.environ.get("HOSTNAME") or "unknown",
            "pid": os.getpid(),
            "data": data,
        }
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass  # Best-effort


def emit_dialogos_submit(*, req_id: str, prompt: str, session_id: str, actor: str) -> None:
    """Emit dialogos.submit for the prompt."""
    emit_bus_event(
        topic="dialogos.submit",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "mode": "llm",
            "providers": ["claude-code"],
            "prompt": prompt,
            "session_id": session_id,
            "source": "claude-code-hook",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest(),
        },
    )


def emit_dialogos_trace(*, req_id: str, session_id: str, actor: str, event_type: str, data: dict) -> None:
    """Emit trace event to dialogos trace file."""
    trace_path = Path(os.environ.get("PLURIBUS_DIALOGOS_TRACE") or "/pluribus/.pluribus/dialogos/trace.ndjson")

    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "req_id": req_id,
            "session_id": session_id,
            "event_type": event_type,
            "actor": actor,
            "source": "claude-code-hook",
            **data,
        }
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass  # Best-effort


def handle_user_prompt_submit(input_data: dict) -> None:
    """Handle UserPromptSubmit hook event."""
    prompt = input_data.get("prompt", "")
    session_id = input_data.get("session_id", "unknown")
    cwd = input_data.get("cwd", "")

    # Generate unique request ID
    req_id = f"cc-{session_id[:8]}-{int(time.time() * 1000)}"

    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "claude-code"

    # Emit dialogos.submit to bus (dialogosd will NOT process this since provider is claude-code)
    # But we trace it directly for ground truth
    emit_dialogos_trace(
        req_id=req_id,
        session_id=session_id,
        actor=actor,
        event_type="user_prompt",
        data={
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest(),
            "prompt_len": len(prompt),
            "cwd": cwd,
        },
    )

    # Emit bus event for observability
    emit_bus_event(
        topic="dialogos.claude_code.prompt",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "session_id": session_id,
            "prompt_len": len(prompt),
            "prompt_preview": prompt[:200] if len(prompt) > 200 else prompt,
            "cwd": cwd,
        },
    )


def handle_stop(input_data: dict) -> None:
    """Handle Stop hook event - Claude finished responding."""
    session_id = input_data.get("session_id", "unknown")
    transcript_path = input_data.get("transcript_path", "")

    req_id = f"cc-stop-{session_id[:8]}-{int(time.time() * 1000)}"
    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "claude-code"

    # Trace the stop event
    emit_dialogos_trace(
        req_id=req_id,
        session_id=session_id,
        actor=actor,
        event_type="assistant_stop",
        data={
            "transcript_path": transcript_path,
        },
    )

    # Emit bus event
    emit_bus_event(
        topic="dialogos.claude_code.stop",
        kind="response",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "session_id": session_id,
            "transcript_path": transcript_path,
        },
    )


def main() -> int:
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        return 0  # Silent fail on bad input

    event_name = input_data.get("hook_event_name", "")

    if event_name == "UserPromptSubmit":
        handle_user_prompt_submit(input_data)
    elif event_name == "Stop":
        handle_stop(input_data)
    elif event_name == "SessionStart":
        # Emit session start trace
        session_id = input_data.get("session_id", "unknown")
        actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "claude-code"
        emit_dialogos_trace(
            req_id=f"cc-start-{session_id[:8]}-{int(time.time() * 1000)}",
            session_id=session_id,
            actor=actor,
            event_type="session_start",
            data={"cwd": input_data.get("cwd", "")},
        )
    elif event_name == "SessionEnd":
        # Emit session end trace
        session_id = input_data.get("session_id", "unknown")
        actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "claude-code"
        emit_dialogos_trace(
            req_id=f"cc-end-{session_id[:8]}-{int(time.time() * 1000)}",
            session_id=session_id,
            actor=actor,
            event_type="session_end",
            data={},
        )

    # Exit 0 = success, no blocking
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
