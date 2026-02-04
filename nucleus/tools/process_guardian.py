#!/usr/bin/env python3
"""
Process Guardian - Load monitoring and runaway process termination.

Integrates with OHM for threshold-based automatic cleanup.
Emits events to QA/Hygiene pipeline for restart failures.

Usage:
    python3 nucleus/tools/process_guardian.py --check
    python3 nucleus/tools/process_guardian.py --kill --emit-bus
    python3 nucleus/tools/process_guardian.py --daemon --threshold 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None

# === Configuration ===
DEFAULT_BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
KILLZOMBIES_PATH = REPO_ROOT / "killzombies.zsh"

# Thresholds
LOAD_THRESHOLD_DEFAULT = 50.0
CPU_THRESHOLD_DEFAULT = 150
LONG_CPU_MIN_DEFAULT = 50
ETIME_MIN_DEFAULT = 15
MEM_THRESHOLD_DEFAULT = 30

# Cooldown between kills (seconds)
KILL_COOLDOWN_S = 60

# Protected patterns (must match killzombies.zsh)
AGENT_CLI_PATTERN = re.compile(r"(codex|claude|gemini|qwen|grok)", re.I)
DAEMON_PATTERN = re.compile(
    r"(agent_cascade_daemon|agent_bus|agent_collab_router|dialogosd|dialogos_indexer|"
    r"world_router|omega_heartbeat|omega_guardian|omega_dispatcher|browser_session_daemon|"
    r"ohm\.py|qa_observer|qa_tool_queue|qa_live_checker|bus_mirror_daemon|bus-bridge|"
    r"meta_ingest|git_server|vps_session|websockify|codemaster_agent|pblbc_operator|"
    r"pbhygiene_operator)", re.I
)
INFRA_PATTERN = re.compile(r"(vite|esbuild|playwright|tsx|novnc)", re.I)


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    cpu_pct: float
    mem_pct: float
    elapsed_s: int
    command: str
    is_protected: bool = False
    protection_reason: str = ""


@dataclass
class LoadStatus:
    load_1m: float
    load_5m: float
    load_15m: float
    running_procs: int
    total_procs: int
    above_threshold: bool = False
    candidate_count: int = 0
    candidates: list[ProcessInfo] = field(default_factory=list)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if agent_bus is None:
        # Fallback: direct append
        try:
            evt = {
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "iso": now_iso(),
                "topic": topic,
                "kind": kind,
                "level": level,
                "actor": actor,
                "data": data,
            }
            events_path = bus_dir / "events.ndjson"
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(evt, separators=(",", ":")) + "\n")
        except Exception:
            pass
        return

    try:
        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception:
        pass


def get_load_average() -> tuple[float, float, float]:
    """Get 1, 5, 15 minute load averages."""
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().strip().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return 0.0, 0.0, 0.0


def get_process_count() -> tuple[int, int]:
    """Get running and total process counts."""
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().strip().split()
            procs = parts[3].split("/")
            return int(procs[0]), int(procs[1])
    except Exception:
        return 0, 0


def parse_elapsed(etime: str) -> int:
    """Parse ps elapsed time to seconds."""
    days = 0
    time_part = etime

    if "-" in etime:
        days_str, time_part = etime.split("-", 1)
        days = int(days_str)

    parts = time_part.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        h, m, s = 0, 0, int(parts[0])

    return days * 86400 + h * 3600 + m * 60 + s


def is_protected(cmd: str) -> tuple[bool, str]:
    """Check if a command matches protected patterns."""
    if AGENT_CLI_PATTERN.search(cmd):
        return True, "agent_cli"
    if DAEMON_PATTERN.search(cmd):
        return True, "daemon"
    if INFRA_PATTERN.search(cmd):
        return True, "infrastructure"
    return False, ""


def get_candidates(
    cpu_threshold: int = CPU_THRESHOLD_DEFAULT,
    long_cpu_min: int = LONG_CPU_MIN_DEFAULT,
    etime_min: int = ETIME_MIN_DEFAULT,
    mem_threshold: int = MEM_THRESHOLD_DEFAULT,
) -> list[ProcessInfo]:
    """Get list of candidate processes for termination."""
    candidates = []

    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,pcpu=,pmem=,etime=,command="],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = line.split(None, 5)
            if len(parts) < 6:
                continue

            pid = int(parts[0])
            ppid = int(parts[1])
            cpu_pct = float(parts[2])
            mem_pct = float(parts[3])
            etime = parts[4]
            cmd = parts[5]

            # Only target node/python processes
            if not re.search(r"(^|/)(node|python|python3)(\s|$)", cmd):
                continue

            protected, reason = is_protected(cmd)
            elapsed_s = parse_elapsed(etime)

            # Check thresholds
            is_candidate = (
                cpu_pct >= cpu_threshold or
                (elapsed_s >= etime_min * 60 and cpu_pct >= long_cpu_min) or
                mem_pct >= mem_threshold
            )

            if is_candidate:
                proc = ProcessInfo(
                    pid=pid,
                    ppid=ppid,
                    cpu_pct=cpu_pct,
                    mem_pct=mem_pct,
                    elapsed_s=elapsed_s,
                    command=cmd[:200],
                    is_protected=protected,
                    protection_reason=reason,
                )
                candidates.append(proc)

    except Exception as e:
        print(f"Error getting candidates: {e}", file=sys.stderr)

    return candidates


def check_status(load_threshold: float = LOAD_THRESHOLD_DEFAULT) -> LoadStatus:
    """Check system load and identify candidates."""
    load_1m, load_5m, load_15m = get_load_average()
    running, total = get_process_count()
    candidates = get_candidates()

    return LoadStatus(
        load_1m=load_1m,
        load_5m=load_5m,
        load_15m=load_15m,
        running_procs=running,
        total_procs=total,
        above_threshold=load_1m >= load_threshold,
        candidate_count=len([c for c in candidates if not c.is_protected]),
        candidates=candidates,
    )


def run_killzombies(
    kill: bool = False,
    emit_bus_flag: bool = False,
    restart_daemons: bool = False,
    json_output: bool = True,
) -> dict[str, Any]:
    """Run killzombies.zsh script."""
    if not KILLZOMBIES_PATH.exists():
        return {"error": "killzombies.zsh not found", "success": False}

    cmd = ["/usr/bin/zsh", str(KILLZOMBIES_PATH)]
    if kill:
        cmd.append("--kill")
    if emit_bus_flag:
        cmd.append("--emit-bus")
    if restart_daemons:
        cmd.append("--restart-daemons")
    if json_output:
        cmd.append("--json")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

        output = result.stdout.strip()
        if json_output and output:
            # Parse last JSON line (may have multiple outputs)
            for line in reversed(output.split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass

        return {
            "success": result.returncode == 0,
            "stdout": output,
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def emit_qa_failure(bus_dir: Path, failed_daemons: list[str], error_details: str) -> None:
    """Emit QA failure event for hygiene pipeline to handle."""
    emit_bus(
        bus_dir,
        topic="qa.hygiene.restart_failure",
        kind="log",
        level="error",
        actor="process-guardian",
        data={
            "failed_daemons": failed_daemons,
            "error": error_details,
            "action_required": "ralph-code-pipeline",
            "timestamp": now_iso(),
        },
    )


def daemon_loop(
    bus_dir: Path,
    load_threshold: float,
    check_interval: float = 30.0,
    kill_cooldown: float = KILL_COOLDOWN_S,
) -> None:
    """Run daemon loop checking load and triggering kills."""
    last_kill_time = 0.0

    print(f"Process Guardian daemon starting (threshold={load_threshold})")
    emit_bus(
        bus_dir,
        topic="process.guardian.start",
        kind="log",
        level="info",
        actor="process-guardian",
        data={"load_threshold": load_threshold, "check_interval": check_interval},
    )

    while True:
        try:
            status = check_status(load_threshold)

            if status.above_threshold and status.candidate_count > 0:
                now = time.time()
                if now - last_kill_time >= kill_cooldown:
                    print(f"Load {status.load_1m} >= {load_threshold}, {status.candidate_count} candidates. Killing...")

                    result = run_killzombies(
                        kill=True,
                        emit_bus_flag=True,
                        restart_daemons=True,
                        json_output=True,
                    )

                    last_kill_time = now

                    # Check for restart failures
                    if isinstance(result, dict):
                        if "failed_daemons" in result.get("data", {}):
                            emit_qa_failure(
                                bus_dir,
                                result["data"]["failed_daemons"],
                                "Daemon restart failed after process kill",
                            )
                else:
                    remaining = kill_cooldown - (now - last_kill_time)
                    print(f"Load high but in cooldown ({remaining:.0f}s remaining)")

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print("Shutting down...")
            break
        except Exception as e:
            print(f"Error in daemon loop: {e}", file=sys.stderr)
            time.sleep(check_interval)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="process_guardian.py",
        description="Process Guardian - Load monitoring and runaway process termination",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory")

    action = p.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Check current status")
    action.add_argument("--kill", action="store_true", help="Run killzombies with --kill")
    action.add_argument("--daemon", action="store_true", help="Run daemon loop")

    p.add_argument("--threshold", type=float, default=LOAD_THRESHOLD_DEFAULT,
                   help=f"Load threshold for automatic kill (default: {LOAD_THRESHOLD_DEFAULT})")
    p.add_argument("--interval", type=float, default=30.0,
                   help="Check interval in seconds for daemon mode (default: 30)")
    p.add_argument("--emit-bus", action="store_true", help="Emit bus events")
    p.add_argument("--restart-daemons", action="store_true", help="Restart dead daemons")
    p.add_argument("--json", action="store_true", help="JSON output")

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    bus_dir = Path(args.bus_dir) if args.bus_dir else DEFAULT_BUS_DIR

    if args.daemon:
        daemon_loop(bus_dir, args.threshold, args.interval)
        return 0

    if args.kill:
        result = run_killzombies(
            kill=True,
            emit_bus_flag=args.emit_bus,
            restart_daemons=args.restart_daemons,
            json_output=args.json,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(result)
        return 0 if result.get("success", True) else 1

    # Default: check status
    status = check_status(args.threshold)

    if args.json:
        output = {
            "load_1m": status.load_1m,
            "load_5m": status.load_5m,
            "load_15m": status.load_15m,
            "running_procs": status.running_procs,
            "total_procs": status.total_procs,
            "above_threshold": status.above_threshold,
            "threshold": args.threshold,
            "candidate_count": status.candidate_count,
            "candidates": [
                {
                    "pid": c.pid,
                    "cpu": c.cpu_pct,
                    "mem": c.mem_pct,
                    "elapsed_s": c.elapsed_s,
                    "protected": c.is_protected,
                    "reason": c.protection_reason,
                    "cmd": c.command[:100],
                }
                for c in status.candidates
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"=== Process Guardian Status ===")
        print(f"Load: {status.load_1m:.2f} / {status.load_5m:.2f} / {status.load_15m:.2f}")
        print(f"Procs: {status.running_procs} running / {status.total_procs} total")
        print(f"Threshold: {args.threshold} ({'EXCEEDED' if status.above_threshold else 'OK'})")
        print(f"Candidates: {status.candidate_count} killable")
        print()

        if status.candidates:
            print("Candidate processes:")
            for c in status.candidates[:10]:
                prot = f" [PROTECTED: {c.protection_reason}]" if c.is_protected else ""
                print(f"  PID {c.pid}: CPU {c.cpu_pct}% MEM {c.mem_pct}% {c.command[:60]}{prot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
