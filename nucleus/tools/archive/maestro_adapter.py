#!/usr/bin/env python3
"""
Maestro Adapter
===============

Bus-integrated wrapper for Maestro (mobile-dev-inc/maestro).
Runs E2E UI flows and emits structured bus evidence.

Effects: R(file), W(bus)
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
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


def find_maestro() -> str | None:
    # Prioritize local membrane source
    local_maestro = Path(__file__).resolve().parents[2] / "membrane" / "maestro" / "maestro"
    if local_maestro.exists():
        return str(local_maestro)
    return shutil.which("maestro")


def build_command(maestro_path: str, args: argparse.Namespace) -> list[str]:
    mode = args.mode
    if mode == "cloud":
        return [maestro_path, "cloud", args.flow] + args.extra_args
    return [maestro_path, "test", args.flow] + args.extra_args


def run_maestro(
    cmd: list[str], 
    timeout: int,
    bus_dir: str | None = None,
    trace_id: str | None = None,
    actor: str = "maestro",
) -> tuple[int, str, str]:
    """Runs Maestro and relays output to the bus for real-time observability."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    stdout_total = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        
        while proc.poll() is None:
            line = proc.stdout.readline()
            if line:
                stdout_total.append(line)
                if bus_dir:
                    emit_bus(
                        bus_dir,
                        topic="maestro.log",
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
        return 127, "", "maestro not found"
    except Exception as exc:
        return 1, "", str(exc)


def cmd_check(args: argparse.Namespace) -> int:
    path = find_maestro()
    if not path:
        print("Maestro not found. Install: https://maestro.mobile.dev", file=sys.stderr)
        return 1
    result = subprocess.run([path, "--version"], capture_output=True, text=True)
    print(json.dumps({"ok": result.returncode == 0, "path": path, "version": result.stdout.strip()}, indent=2))
    return 0 if result.returncode == 0 else 2


def cmd_run(args: argparse.Namespace) -> int:
    actor = args.actor or default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    trace_id = args.trace_id or os.environ.get("PLURIBUS_TRACE_ID") or str(uuid.uuid4())
    req_id = args.req_id or str(uuid.uuid4())[:8]

    maestro_path = find_maestro()
    if not maestro_path:
        emit_bus(
            bus_dir,
            topic="maestro.run.error",
            kind="response",
            level="error",
            actor=actor,
            data={"req_id": req_id, "error": "maestro_not_found"},
            trace_id=trace_id,
        )
        print("ERROR: Maestro not found. Install: https://maestro.mobile.dev", file=sys.stderr)
        return 127

    cmd = build_command(maestro_path, args)
    emit_bus(
        bus_dir,
        topic="maestro.run.request",
        kind="request",
        level="info",
        actor=actor,
        data={"req_id": req_id, "flow": args.flow, "mode": args.mode, "at": now_iso_utc()},
        trace_id=trace_id,
    )

    code, stdout, stderr = run_maestro(
        cmd, 
        args.timeout,
        bus_dir=bus_dir,
        trace_id=trace_id,
        actor=actor
    )

    emit_bus(
        bus_dir,
        topic="maestro.run.response",
        kind="response",
        level="info" if code == 0 else "error",
        actor=actor,
        data={"req_id": req_id, "exit_code": code, "ok": code == 0},
        trace_id=trace_id,
    )

    if args.json_output:
        print(json.dumps({"status": "ok" if code == 0 else "error", "exit_code": code}))
    else:
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)

    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Pluribus adapter for Maestro UI testing.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Verify Maestro CLI availability.")
    p_check.set_defaults(func=cmd_check)

    p_run = sub.add_parser("run", help="Run a Maestro flow.")
    p_run.add_argument("--flow", required=True, help="Path to Maestro YAML flow")
    p_run.add_argument("--mode", choices=["test", "cloud"], default="test")
    p_run.add_argument("--timeout", type=int, default=900)
    p_run.add_argument("--actor")
    p_run.add_argument("--bus-dir")
    p_run.add_argument("--trace-id")
    p_run.add_argument("--req-id")
    p_run.add_argument("--json-output", action="store_true")
    p_run.add_argument("--extra-args", nargs="*", default=[])
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
