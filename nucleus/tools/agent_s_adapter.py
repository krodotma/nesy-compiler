#!/usr/bin/env python3
"""
Agent-S Adapter
===============

Bus-integrated wrapper for simular-ai/Agent-S (GUI automation / CUA).
Provides a Pluribus-native CLI to run Agent-S tasks with structured
bus events and JSON output for IsoExecutor.

Effects: R(net), W(bus)
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def emit_bus(
    bus_dir: str | None,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
    trace_id: str | None = None,
) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    cmd = [
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
    ]
    if trace_id:
        cmd.extend(["--trace-id", trace_id])

    subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def resolve_entrypoint() -> list[str] | None:
    override = os.environ.get("AGENT_S_ENTRYPOINT")
    if override:
        return shlex.split(override)

    # Prioritize local membrane source
    local_src = Path(__file__).resolve().parents[2] / "membrane" / "agent-s"
    if local_src.exists():
        python = shutil.which("python3") or shutil.which("python")
        if python:
            # We add the local src to PYTHONPATH via the command execution if needed,
            # or just assume it's installed in editable mode. 
            # For now, let's look for the module entry point inside the local dir.
            if (local_src / "gui_agents" / "s3" / "cli_app.py").exists():
                return [python, str(local_src / "gui_agents" / "s3" / "cli_app.py")]

    agent_s = shutil.which("agent_s")
    if agent_s:
        return [agent_s]

    python = shutil.which("python3") or shutil.which("python")
    if python:
        return [python, "-m", "gui_agents.s3.cli_app"]

    return None


def check_agent_s(entrypoint: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            entrypoint + ["--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            return True, "ok"
        return True, f"exit={result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "not_found"
    except Exception as exc:
        return False, str(exc)


def build_agent_s_args(args: argparse.Namespace) -> list[str]:
    provider = args.provider or os.environ.get("AGENT_S_PROVIDER") or "openai"
    model = args.model or os.environ.get("AGENT_S_MODEL") or "gpt-5-2025-08-07"
    model_url = args.model_url or os.environ.get("AGENT_S_MODEL_URL") or ""
    model_api_key = args.model_api_key or os.environ.get("AGENT_S_MODEL_API_KEY") or ""
    model_temperature = args.model_temperature or os.environ.get("AGENT_S_MODEL_TEMPERATURE")

    ground_provider = args.ground_provider or os.environ.get("AGENT_S_GROUND_PROVIDER")
    ground_url = args.ground_url or os.environ.get("AGENT_S_GROUND_URL")
    ground_api_key = args.ground_api_key or os.environ.get("AGENT_S_GROUND_API_KEY") or ""
    ground_model = args.ground_model or os.environ.get("AGENT_S_GROUND_MODEL")
    grounding_width = args.grounding_width or os.environ.get("AGENT_S_GROUNDING_WIDTH")
    grounding_height = args.grounding_height or os.environ.get("AGENT_S_GROUNDING_HEIGHT")

    missing = []
    if not ground_provider:
        missing.append("ground_provider")
    if not ground_url:
        missing.append("ground_url")
    if not ground_model:
        missing.append("ground_model")
    if not grounding_width:
        missing.append("grounding_width")
    if not grounding_height:
        missing.append("grounding_height")
    if missing:
        raise ValueError("Missing Agent-S grounding config: " + ", ".join(missing))

    args_list = [
        "--provider", provider,
        "--model", model,
        "--model_url", model_url,
        "--model_api_key", model_api_key,
        "--ground_provider", str(ground_provider),
        "--ground_url", str(ground_url),
        "--ground_api_key", ground_api_key,
        "--ground_model", str(ground_model),
        "--grounding_width", str(grounding_width),
        "--grounding_height", str(grounding_height),
    ]

    if model_temperature:
        args_list += ["--model_temperature", str(model_temperature)]

    max_traj = args.max_trajectory_length or os.environ.get("AGENT_S_MAX_TRAJECTORY_LENGTH")
    if max_traj:
        args_list += ["--max_trajectory_length", str(max_traj)]

    enable_reflection = args.enable_reflection
    if enable_reflection is None:
        enable_reflection = os.environ.get("AGENT_S_ENABLE_REFLECTION", "true").lower() in {"1", "true", "yes"}
    if enable_reflection:
        args_list.append("--enable_reflection")

    enable_local_env = args.enable_local_env
    if enable_local_env is None:
        enable_local_env = os.environ.get("AGENT_S_ENABLE_LOCAL_ENV", "false").lower() in {"1", "true", "yes"}
    if enable_local_env:
        args_list.append("--enable_local_env")

    return args_list


def run_agent_s(
    *,
    goal: str,
    entrypoint: list[str],
    args_list: list[str],
    timeout: int,
    bus_dir: str | None = None,
    trace_id: str | None = None,
    actor: str = "agent-s",
) -> tuple[int, str, str]:
    """Runs Agent-S and relays output to the bus for observability."""
    cmd = entrypoint + args_list
    input_payload = f"{goal}\n" + "n\n"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        
        # Send initial goal
        stdout_total = []
        stderr_total = []
        
        # Simple relay loop
        # In a production environment, this would use threading/async
        proc.stdin.write(input_payload)
        proc.stdin.flush()
        
        while proc.poll() is None:
            line = proc.stdout.readline()
            if line:
                stdout_total.append(line)
                if bus_dir:
                    emit_bus(
                        bus_dir,
                        topic="agent_s.log",
                        kind="log",
                        level="info",
                        actor=actor,
                        data={"msg": line.strip()},
                        trace_id=trace_id
                    )
        
        _, stderr = proc.communicate()
        return proc.returncode, "".join(stdout_total), stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", "agent_s not found (pip install gui-agents)"
    except Exception as exc:
        return 1, "", str(exc)


def cmd_check(args: argparse.Namespace) -> int:
    entrypoint = resolve_entrypoint()
    if not entrypoint:
        print("Agent-S not found. Install via: pip install gui-agents", file=sys.stderr)
        return 1
    ok, detail = check_agent_s(entrypoint)
    print(json.dumps({"ok": ok, "detail": detail, "entrypoint": entrypoint}, indent=2))
    return 0 if ok else 2


def cmd_run(args: argparse.Namespace) -> int:
    actor = args.actor or default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    trace_id = args.trace_id or os.environ.get("PLURIBUS_TRACE_ID") or str(uuid.uuid4())
    req_id = args.req_id or str(uuid.uuid4())[:8]

    entrypoint = resolve_entrypoint()
    if not entrypoint:
        emit_bus(
            bus_dir,
            topic="agent_s.run.error",
            kind="response",
            level="error",
            actor=actor,
            data={"req_id": req_id, "error": "agent_s_not_found"},
            trace_id=trace_id,
        )
        print("ERROR: Agent-S not found. Install via: pip install gui-agents", file=sys.stderr)
        return 127

    try:
        args_list = build_agent_s_args(args)
    except ValueError as exc:
        emit_bus(
            bus_dir,
            topic="agent_s.run.error",
            kind="response",
            level="error",
            actor=actor,
            data={"req_id": req_id, "error": "missing_config", "message": str(exc)},
            trace_id=trace_id,
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    emit_bus(
        bus_dir,
        topic="agent_s.run.request",
        kind="request",
        level="info",
        actor=actor,
        data={"req_id": req_id, "goal": args.goal, "at": now_iso_utc()},
        trace_id=trace_id,
    )

    code, stdout, stderr = run_agent_s(
        goal=args.goal,
        entrypoint=entrypoint,
        args_list=args_list,
        timeout=args.timeout,
        bus_dir=bus_dir,
        trace_id=trace_id,
        actor=actor,
    )

    emit_bus(
        bus_dir,
        topic="agent_s.run.response",
        kind="response",
        level="info" if code == 0 else "error",
        actor=actor,
        data={
            "req_id": req_id,
            "exit_code": code,
            "ok": code == 0,
            "stderr": (stderr or "")[:2000],
        },
        trace_id=trace_id,
    )

    if args.json_output:
        print(json.dumps({"status": "ok" if code == 0 else "error", "exit_code": code, "stderr": stderr}))
    else:
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)

    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Pluribus adapter for Agent-S (GUI automation).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Verify Agent-S CLI availability.")
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser("run", help="Run a single Agent-S task.")
    p_run.add_argument("--goal", required=True, help="Task instruction for Agent-S")
    p_run.add_argument("--timeout", type=int, default=900, help="Timeout in seconds")
    p_run.add_argument("--actor")
    p_run.add_argument("--bus-dir")
    p_run.add_argument("--trace-id")
    p_run.add_argument("--req-id")
    p_run.add_argument("--json-output", action="store_true")
    p_run.add_argument("--provider")
    p_run.add_argument("--model")
    p_run.add_argument("--model-url")
    p_run.add_argument("--model-api-key")
    p_run.add_argument("--model-temperature")
    p_run.add_argument("--ground-provider")
    p_run.add_argument("--ground-url")
    p_run.add_argument("--ground-api-key")
    p_run.add_argument("--ground-model")
    p_run.add_argument("--grounding-width")
    p_run.add_argument("--grounding-height")
    p_run.add_argument("--max-trajectory-length")
    p_run.add_argument("--enable-reflection", action="store_true", default=None)
    p_run.add_argument("--enable-local-env", action="store_true", default=None)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
