#!/usr/bin/env python3
"""
PBRESUME Hook - Auto-trigger recovery on session resume.

This hook runs on SessionStart and checks if the session looks like a resume
(i.e., there was a previous session that ended more than 5 minutes ago).
If so, it automatically invokes PBRESUME with --auto --quiet to recover
incomplete work.

Install: Add to ~/.claude/settings.json hooks configuration

Reference: nucleus/specs/dkin_protocol_v25_resume.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_dialogos_trace(*, session_id: str, actor: str, event_type: str, data: dict) -> None:
    """Emit trace event to dialogos trace file."""
    trace_path = Path(os.environ.get("PLURIBUS_DIALOGOS_TRACE") or "/pluribus/.pluribus/dialogos/trace.ndjson")

    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "session_id": session_id,
            "event_type": event_type,
            "actor": actor,
            "source": "pbresume-hook",
            **data,
        }
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass  # Best-effort


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


def run_pbresume_auto(session_id: str, actor: str, quiet: bool = True) -> int:
    """
    Run pbresume_operator.py --auto --quiet.

    Returns the exit code from the subprocess.
    """
    pbresume_path = Path("/pluribus/nucleus/tools/pbresume_operator.py")

    if not pbresume_path.exists():
        return 1

    cmd = [sys.executable, str(pbresume_path), "--auto"]
    if quiet:
        cmd.append("--quiet")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            env={**os.environ, "PLURIBUS_SESSION_ID": session_id},
        )

        # Log result to dialogos trace
        emit_dialogos_trace(
            session_id=session_id,
            actor=actor,
            event_type="pbresume_hook_result",
            data={
                "exit_code": result.returncode,
                "stdout_len": len(result.stdout),
                "stderr_preview": result.stderr[:200] if result.stderr else "",
            },
        )

        return result.returncode

    except subprocess.TimeoutExpired:
        emit_dialogos_trace(
            session_id=session_id,
            actor=actor,
            event_type="pbresume_hook_timeout",
            data={"timeout_s": 30},
        )
        return 1

    except Exception as e:
        emit_dialogos_trace(
            session_id=session_id,
            actor=actor,
            event_type="pbresume_hook_error",
            data={"error": str(e)},
        )
        return 1


def handle_session_start(input_data: dict) -> None:
    """Handle SessionStart hook event - potentially trigger auto-resume."""
    session_id = input_data.get("session_id", "unknown")
    cwd = input_data.get("cwd", "")

    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pbresume-hook"

    # Log that we're checking for resume
    emit_dialogos_trace(
        session_id=session_id,
        actor=actor,
        event_type="pbresume_hook_check",
        data={"cwd": cwd},
    )

    # Emit bus event for observability
    emit_bus_event(
        topic="operator.pbresume.hook_triggered",
        kind="request",
        level="info",
        actor=actor,
        data={
            "session_id": session_id,
            "cwd": cwd,
            "trigger": "session_start",
        },
    )

    # Run pbresume --auto --quiet
    exit_code = run_pbresume_auto(session_id, actor, quiet=True)

    # Emit result to bus
    emit_bus_event(
        topic="operator.pbresume.hook_complete",
        kind="response",
        level="info",
        actor=actor,
        data={
            "session_id": session_id,
            "exit_code": exit_code,
            "auto_triggered": exit_code == 0,
        },
    )


def main() -> int:
    """Main entry point for the hook."""
    try:
        input_data = json.load(sys.stdin)
    except Exception:
        return 0  # Silent fail on bad input

    event_name = input_data.get("hook_event_name", "")

    if event_name == "SessionStart":
        handle_session_start(input_data)

    # Exit 0 = success, no blocking
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
