#!/usr/bin/env python3
from __future__ import annotations

"""
Studio Flow Roundtrip (ADK Ω-gate)

Runs: parse flowfile → serialize → re-parse → compare canonical graph
and emits `studio.flow.roundtrip` evidence to the bus.

Supports:
- `.py` safe DSL (AST-parsed; no execution)
- `.json` Flow.to_dict format
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    import subprocess

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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="studio_flow_roundtrip.py", description="ADK Ω-gate: flowfile roundtrip + bus evidence.")
    ap.add_argument("--root", default=None, help="Pluribus root (defaults: /pluribus).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (default: $PLURIBUS_BUS_DIR).")
    ap.add_argument("--actor", default=None)
    ap.add_argument("--in", dest="in_path", required=True, help="Input flowfile (.py or .json).")
    ap.add_argument("--out", default=None, help="Output path (default: <root>/.pluribus/index/studio/roundtrip/<req_id>.py).")
    ap.add_argument("--emit-bus", action="store_true", help="Emit studio.flow.roundtrip metric event.")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else Path(os.environ.get("PLURIBUS_ROOT") or "/pluribus").expanduser().resolve()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    actor = (args.actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "studio-flow-roundtrip").strip()

    in_path = Path(args.in_path).expanduser().resolve()
    req_id = str(uuid.uuid4())

    out_path = Path(args.out).expanduser().resolve() if args.out else (root / ".pluribus" / "index" / "studio" / "roundtrip" / f"{req_id}.py")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure we can import `nucleus` even when `--root` points at a temp rhizome.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from nucleus.sdk.flow import load_flow  # noqa: E402

    ok = False
    error: str | None = None
    try:
        f1 = load_flow(in_path)
        f1.save_py(out_path)
        f2 = load_flow(out_path)
        c1 = f1.canonical()
        c2 = f2.canonical()
        # The serialized file name can change (stem), but the graph semantics should not.
        c2["name"] = c1.get("name", c2.get("name"))
        ok = c1 == c2
        if not ok:
            error = "canonical mismatch"
    except Exception as e:
        ok = False
        error = str(e)

    if args.emit_bus:
        emit_bus(
            bus_dir,
            topic="studio.flow.roundtrip",
            kind="metric",
            level="info" if ok else "error",
            actor=actor,
            data={
                "req_id": req_id,
                "ok": ok,
                "in_path": str(in_path),
                "out_path": str(out_path),
                "error": error,
                "iso": now_iso_utc(),
            },
        )

    sys.stdout.write(str(out_path) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
