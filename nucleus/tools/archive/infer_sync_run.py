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

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def agent_bus_path() -> Path:
    return Path(__file__).with_name("agent_bus.py")


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "infer_sync_run"


def pub(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    tool = agent_bus_path()
    subprocess.run(
        [
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
        ],
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="infer_sync_run.py", description="Run a command and emit infer_sync test/checkin events.")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    p.add_argument("--subproject", default="other")
    p.add_argument("--candidate-id", default=None)
    p.add_argument("--label", default=None, help="Short label for the run.")
    p.add_argument("--timeout-s", type=float, default=None)
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run (use: -- <cmd...>)")
    args = p.parse_args(argv)

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    if not bus_dir:
        sys.stderr.write("missing bus dir (set PLURIBUS_BUS_DIR or pass --bus-dir)\n")
        return 2

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.stderr.write("missing command (use: infer_sync_run.py -- <cmd...>)\n")
        return 2

    actor = default_actor()
    run_id = str(uuid.uuid4())
    started = time.time()
    label = args.label or " ".join(cmd[:3])

    pub(
        bus_dir,
        topic="infer_sync.checkin",
        kind="metric",
        level="info",
        actor=actor,
        data={
            "status": "working",
            "done": 0,
            "open": 1,
            "blocked": 0,
            "errors": 0,
            "next": f"run {label}",
            "subproject": args.subproject,
            "focus": ["test", "retest"],
        },
    )

    pub(
        bus_dir,
        topic="infer_sync.test.start",
        kind="metric",
        level="info",
        actor=actor,
        data={
            "run_id": run_id,
            "subproject": args.subproject,
            "candidate_id": args.candidate_id,
            "label": label,
            "cmd": cmd,
            "started_iso": now_iso_utc(),
        },
    )

    try:
        p_run = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout_s,
        )
        code = int(p_run.returncode)
        out = (p_run.stdout or "").strip()
        err = (p_run.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        code = 124
        out = (e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")) if e.stdout is not None else ""
        err = "timeout"

    dt_s = max(0.0, time.time() - started)
    ok = code == 0

    pub(
        bus_dir,
        topic="infer_sync.test.end",
        kind="metric",
        level="info" if ok else "error",
        actor=actor,
        data={
            "run_id": run_id,
            "subproject": args.subproject,
            "candidate_id": args.candidate_id,
            "label": label,
            "exit_code": code,
            "ok": ok,
            "duration_s": dt_s,
            "stdout_tail": out[-2000:],
            "stderr_tail": err[-2000:],
            "ended_iso": now_iso_utc(),
        },
    )

    pub(
        bus_dir,
        topic="infer_sync.checkin",
        kind="metric",
        level="info" if ok else "error",
        actor=actor,
        data={
            "status": "idle" if ok else "error",
            "done": 1 if ok else 0,
            "open": 0,
            "blocked": 0,
            "errors": 0 if ok else 1,
            "next": "ready" if ok else "fix failing gate",
            "subproject": args.subproject,
            "focus": ["test"],
        },
    )

    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

