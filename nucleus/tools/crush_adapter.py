#!/usr/bin/env python3
"""
Crush Bus Adapter - DKIN v19 Compliant

Wraps Charmbracelet mods/crush CLI to emit Pluribus bus events for
multi-agent coordination. Follows PAIP v12 isolation patterns.

Usage:
    crush_adapter.py --prompt "explain this code" --file main.py
    crush_adapter.py --interactive  # Start interactive session
    echo "code" | crush_adapter.py --prompt "review"

Bus Topics Emitted:
    - crush.session.start    : Session initiated
    - crush.prompt.submit    : Prompt submitted to LLM
    - crush.response.stream  : Streaming response chunks (optional)
    - crush.response.end     : Response completed
    - crush.session.end      : Session terminated
    - crush.error            : Error occurred
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Bus configuration
BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", "/var/lib/pluribus/.pluribus/bus")
EVENTS_FILE = Path(BUS_DIR) / "events.ndjson"
CRUSH_BIN = os.environ.get("CRUSH_BIN", "/usr/local/bin/crush")
ACTOR = os.environ.get("PLURIBUS_ACTOR", "crush/adapter")
PROTOCOL_VERSION = "v19"


def emit_event(topic: str, data: dict, level: str = "info", req_id: Optional[str] = None) -> None:
    """Emit a bus event in NDJSON format."""
    now = datetime.now(timezone.utc)
    event = {
        "ts": int(now.timestamp() * 1000),
        "iso": now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "topic": topic,
        "actor": ACTOR,
        "level": level,
        "data": {
            **data,
            "protocol_version": PROTOCOL_VERSION,
        },
    }
    if req_id:
        event["data"]["req_id"] = req_id

    try:
        EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"[crush_adapter] bus emit failed: {e}", file=sys.stderr)


def run_crush(
    prompt: str,
    model: Optional[str] = None,
    api: Optional[str] = None,
    files: Optional[list] = None,
    raw: bool = False,
    continue_last: bool = False,
    stream_to_bus: bool = False,
) -> tuple[int, str]:
    """
    Run crush/mods CLI with given parameters.
    Returns (exit_code, output).
    """
    req_id = str(uuid.uuid4())[:8]
    session_id = str(uuid.uuid4())[:12]

    # Build command
    cmd = [CRUSH_BIN]
    if model:
        cmd.extend(["--model", model])
    if api:
        cmd.extend(["--api", api])
    if raw:
        cmd.append("--raw")
    if continue_last:
        cmd.append("--continue-last")

    # Add prompt
    cmd.append(prompt)

    # Emit session start
    emit_event("crush.session.start", {
        "session_id": session_id,
        "model": model or "default",
        "api": api or "default",
        "files": files or [],
    }, req_id=req_id)

    # Emit prompt submit
    emit_event("crush.prompt.submit", {
        "session_id": session_id,
        "prompt_preview": prompt[:200] + ("..." if len(prompt) > 200 else ""),
        "prompt_length": len(prompt),
    }, req_id=req_id)

    start_time = time.time()
    output_lines = []

    try:
        # Handle stdin if files provided
        stdin_data = None
        if files:
            stdin_parts = []
            for f in files:
                try:
                    with open(f, "r") as fp:
                        stdin_parts.append(f"--- {f} ---\n{fp.read()}")
                except Exception as e:
                    stdin_parts.append(f"--- {f} (error: {e}) ---")
            stdin_data = "\n\n".join(stdin_parts)

        # Run crush
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if stdin_data:
            stdout, stderr = proc.communicate(input=stdin_data)
        else:
            stdout, stderr = proc.communicate()

        exit_code = proc.returncode
        output = stdout

        if stream_to_bus and output:
            # Emit response chunks (for real-time dashboard updates)
            for i, chunk in enumerate(output.split("\n\n")):
                if chunk.strip():
                    emit_event("crush.response.stream", {
                        "session_id": session_id,
                        "chunk_idx": i,
                        "content": chunk[:500],
                    }, req_id=req_id)

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit response end
        emit_event("crush.response.end", {
            "session_id": session_id,
            "ok": exit_code == 0,
            "duration_ms": duration_ms,
            "output_length": len(output),
            "output_preview": output[:300] + ("..." if len(output) > 300 else ""),
        }, level="info" if exit_code == 0 else "error", req_id=req_id)

        if stderr:
            emit_event("crush.error", {
                "session_id": session_id,
                "stderr": stderr[:500],
            }, level="warn", req_id=req_id)

        # Emit session end
        emit_event("crush.session.end", {
            "session_id": session_id,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
        }, req_id=req_id)

        return exit_code, output

    except Exception as e:
        emit_event("crush.error", {
            "session_id": session_id,
            "error": str(e),
            "error_type": type(e).__name__,
        }, level="error", req_id=req_id)
        return 1, f"Error: {e}"


def interactive_session(model: Optional[str] = None, api: Optional[str] = None) -> None:
    """Run an interactive crush session with bus event emission."""
    session_id = str(uuid.uuid4())[:12]

    emit_event("crush.session.start", {
        "session_id": session_id,
        "mode": "interactive",
        "model": model or "default",
        "api": api or "default",
    })

    print(f"[crush] Interactive session {session_id} (Ctrl+D to exit)")
    print(f"[crush] Model: {model or 'default'}, API: {api or 'default'}")
    print()

    turn = 0
    while True:
        try:
            prompt = input("crush> ").strip()
            if not prompt:
                continue

            turn += 1
            exit_code, output = run_crush(
                prompt,
                model=model,
                api=api,
                continue_last=(turn > 1),
            )
            print(output)
            print()

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n[interrupted]")
            break

    emit_event("crush.session.end", {
        "session_id": session_id,
        "mode": "interactive",
        "turns": turn,
    })
    print(f"\n[crush] Session {session_id} ended ({turn} turns)")


def main():
    parser = argparse.ArgumentParser(
        description="Crush Bus Adapter - DKIN v19 compliant wrapper for Charmbracelet mods/crush"
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send to LLM")
    parser.add_argument("-m", "--model", help="Model to use (e.g., gpt-4, claude-3)")
    parser.add_argument("-a", "--api", help="API provider (openai, anthropic, ollama)")
    parser.add_argument("-f", "--file", action="append", dest="files", help="Files to include as context")
    parser.add_argument("-r", "--raw", action="store_true", help="Raw output (no formatting)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive session mode")
    parser.add_argument("-c", "--continue", dest="continue_last", action="store_true", help="Continue last conversation")
    parser.add_argument("--stream-bus", action="store_true", help="Stream response chunks to bus")
    parser.add_argument("--version", action="store_true", help="Show version info")

    args = parser.parse_args()

    if args.version:
        print(f"crush_adapter {PROTOCOL_VERSION}")
        result = subprocess.run([CRUSH_BIN, "--version"], capture_output=True, text=True)
        print(f"crush/mods: {result.stdout.strip()}")
        print(f"glow: ", end="")
        result = subprocess.run(["glow", "--version"], capture_output=True, text=True)
        print(result.stdout.strip())
        return

    if args.interactive:
        interactive_session(model=args.model, api=args.api)
        return

    if not args.prompt:
        # Read from stdin if no prompt given
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read().strip()
            if stdin_content:
                args.prompt = stdin_content
            else:
                parser.print_help()
                sys.exit(1)
        else:
            parser.print_help()
            sys.exit(1)

    exit_code, output = run_crush(
        args.prompt,
        model=args.model,
        api=args.api,
        files=args.files,
        raw=args.raw,
        continue_last=args.continue_last,
        stream_to_bus=args.stream_bus,
    )

    print(output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


# ============================================================================
# SUPERWORKER Context Injection (DKIN v28 / PAIP v15)
# ============================================================================

def build_superworker_context(scope: str = "lite") -> str:
    """
    Build SUPERWORKER context for injection into LLM system prompt.
    
    Scopes:
        - min: Just CITIZEN reminder
        - lite: CITIZEN + rhizome state
        - full: CITIZEN + rhizome + statusline + recent bus events
    """
    context_parts = []
    root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    
    # 1. CITIZEN Contract (always included)
    citizen_reminder = '''
# PLURIBUS CITIZEN CONTEXT
You are operating as a SUPERWORKER within the Pluribus system.
Protocol: DKIN v28 / PAIP v15 / CITIZEN v1

Core Principles:
- Emit bus events for significant actions
- Honor append-only constraints
- Reference canonical specs at nucleus/specs/
- Coordinate via the agent bus (events.ndjson)
'''
    context_parts.append(citizen_reminder)
    
    if scope in {"lite", "full"}:
        # 2. Rhizome (project genotype)
        rhizome_path = root / ".pluribus" / "rhizome.json"
        if rhizome_path.exists():
            try:
                rhizome_text = rhizome_path.read_text()[:1500]
                context_parts.append(f"# Project Rhizome\n")
            except Exception:
                pass
    
    if scope == "full":
        # 3. Recent bus activity
        events_path = root / ".pluribus" / "bus" / "events.ndjson"
        if events_path.exists():
            try:
                with open(events_path, "r") as f:
                    lines = f.readlines()[-10:]  # Last 10 events
                events_summary = "\n".join([l.strip()[:200] for l in lines])
                context_parts.append(f"# Recent Bus Activity\n")
            except Exception:
                pass
    
    return "\n\n".join(context_parts)


def inject_superworker_prompt(prompt: str, scope: str = "lite") -> str:
    """Wrap user prompt with SUPERWORKER context."""
    context = build_superworker_context(scope)
    return f"{context}\n\n---\n\n# User Request\n{prompt}"


# ============================================================================
# SUPERWORKER Context Injection (DKIN v28 / PAIP v15)
# ============================================================================

def build_superworker_context(scope: str = "lite") -> str:
    """
    Build SUPERWORKER context for injection into LLM system prompt.

    Scopes:
        - min: Just CITIZEN reminder
        - lite: CITIZEN + rhizome state
        - full: CITIZEN + rhizome + statusline + recent bus events
    """
    context_parts = []
    root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))

    # 1. CITIZEN Contract (always included)
    citizen_reminder = """
# PLURIBUS CITIZEN CONTEXT
You are operating as a SUPERWORKER within the Pluribus system.
Protocol: DKIN v28 / PAIP v15 / CITIZEN v1

Core Principles:
- Emit bus events for significant actions
- Honor append-only constraints
- Reference canonical specs at nucleus/specs/
- Coordinate via the agent bus (events.ndjson)
"""
    context_parts.append(citizen_reminder)

    if scope in {"lite", "full"}:
        # 2. Rhizome (project genotype)
        rhizome_path = root / ".pluribus" / "rhizome.json"
        if rhizome_path.exists():
            try:
                rhizome_text = rhizome_path.read_text()[:1500]
                context_parts.append(f"# Project Rhizome\n```json\n{rhizome_text}\n```")
            except Exception:
                pass

    if scope == "full":
        # 3. Recent bus activity
        events_path = root / ".pluribus" / "bus" / "events.ndjson"
        if events_path.exists():
            try:
                with open(events_path, "r") as f:
                    lines = f.readlines()[-10:]  # Last 10 events
                events_summary = "\n".join([l.strip()[:200] for l in lines])
                context_parts.append(f"# Recent Bus Activity\n```\n{events_summary}\n```")
            except Exception:
                pass

    return "\n\n".join(context_parts)


def inject_superworker_prompt(prompt: str, scope: str = "lite") -> str:
    """Wrap user prompt with SUPERWORKER context."""
    context = build_superworker_context(scope)
    return f"{context}\n\n---\n\n# User Request\n{prompt}"
