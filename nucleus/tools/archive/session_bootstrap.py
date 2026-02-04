#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    ensure_dir(Path(bus_dir))
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="session_bootstrap.py",
        description="Emit an append-only session-start event so other agents can recover shared context (git semantics, bus dir, constraints).",
    )
    ap.add_argument("--root", default=None, help="Pluribus root (default: search upward for .pluribus/rhizome.json, else CWD).")
    ap.add_argument("--bus-dir", default=None, help="Bus directory (or set PLURIBUS_BUS_DIR).")
    ap.add_argument("--actor", default=None, help="Actor name (or set PLURIBUS_ACTOR).")
    ap.add_argument("--note", default="", help="Optional short note to attach.")
    args = ap.parse_args(argv)

    cwd = Path.cwd().resolve()
    root = Path(args.root).resolve() if args.root else cwd
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or str(root / ".pluribus" / "bus")
    actor = args.actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "unknown"

    payload = {
        "ts": time.time(),
        "iso": now_iso_utc(),
        "cwd": str(cwd),
        "root": str(root),
        "bus_dir": str(bus_dir),
        "constraints": {
            "git_semantics": "isomorphic_only",
            "network_policy": os.environ.get("PLURIBUS_NETWORK_POLICY") or "restricted",
        },
        "tools": {
            "iso_git": "/pluribus/nucleus/tools/iso_git.mjs",
            "agent_bus": "/pluribus/nucleus/tools/agent_bus.py",
        },
    }
    if args.note:
        payload["note"] = args.note

    emit_bus(bus_dir, topic="agent.session.start", kind="artifact", level="info", actor=actor, data=payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

