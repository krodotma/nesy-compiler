#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_agent_lifecycle(*, bus_dir: str, actor: str, data: dict) -> None:
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        raise SystemExit(f"missing agent_bus.py at {tool}")
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            "agent.lifecycle",
            "--kind",
            "log",
            "--level",
            "info",
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
    ap = argparse.ArgumentParser(prog="nexus_ack.py", description="Read /pluribus/nexus_bridge/<agent>.md and emit agent.lifecycle assimilated.")
    ap.add_argument("--agent", default="codex", help="codex|claude|gemini (default: codex)")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    ap.add_argument("--status", default="assimilated")
    args = ap.parse_args(argv)

    agent = args.agent.strip().lower()
    path = Path("/pluribus/nexus_bridge") / f"{agent}.md"
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    _ = path.read_text(encoding="utf-8", errors="replace")

    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"
    data = {"status": args.status, "source": str(path), "iso": now_iso_utc()}
    emit_agent_lifecycle(bus_dir=bus_dir, actor=actor, data=data)
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

