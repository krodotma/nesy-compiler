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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_loader import load_pluribus_env  # noqa: E402


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).resolve().parents[1] / "agent_bus.py"
    if not tool.exists():
        return
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


def run(cmd: list[str], timeout_s: int) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
        return int(p.returncode), (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout_s}s"


def main(argv: list[str]) -> int:
    load_pluribus_env()
    ap = argparse.ArgumentParser(
        prog="gemini3_fallback_smoke.py",
        description="Test at least two Gemini-3 routes (no downgrade to older models).",
    )
    ap.add_argument("--bus-dir", default=None)
    ap.add_argument("--timeout-s", type=int, default=60)
    ap.add_argument("--min-success", type=int, default=2, help="Minimum number of successful routes required (default: 2).")
    ap.add_argument("--prompt", default="Say ok and state the model name you used.")
    ap.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL") or os.environ.get("VERTEX_GEMINI_MODEL") or "gemini-3-pro-preview",
    )
    args = ap.parse_args(argv)

    if not str(args.model).startswith("gemini-3"):
        sys.stderr.write("refusing to test non-gemini-3 model; set --model gemini-3-...\n")
        return 2

    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "unknown"
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    runs = []
    # 1) Gemini CLI route (web auth)
    cmd1 = [
        sys.executable,
        str(Path(__file__).with_name("gemini_cli_smoke.py")),
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--timeout-s",
        str(int(args.timeout_s)),
    ]
    c1, o1, e1 = run(cmd1, args.timeout_s)
    runs.append({"route": "gemini-cli", "exit_code": c1, "stdout": o1, "stderr": e1})

    # 2) Vertex route (urllib; requires gcloud login + project/quota project)
    cmd2 = [
        sys.executable,
        str(Path(__file__).with_name("vertex_gemini_smoke.py")),
        "--model",
        args.model,
        "--require-gemini3",
        "--prompt",
        args.prompt,
        "--timeout-s",
        str(int(args.timeout_s)),
    ]
    c2, o2, e2 = run(cmd2, args.timeout_s)
    runs.append({"route": "vertex-gemini", "exit_code": c2, "stdout": o2, "stderr": e2})

    # 3) Vertex route (curl; independent HTTP stack)
    cmd3 = [
        sys.executable,
        str(Path(__file__).with_name("vertex_gemini_curl_smoke.py")),
        "--model",
        args.model,
        "--require-gemini3",
        "--prompt",
        args.prompt,
        "--timeout-s",
        str(int(args.timeout_s)),
    ]
    c3, o3, e3 = run(cmd3, args.timeout_s)
    runs.append({"route": "vertex-gemini-curl", "exit_code": c3, "stdout": o3, "stderr": e3})

    payload = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "gemini3_fallback_smoke",
        "model": args.model,
        "prompt": args.prompt,
        "runs": runs,
    }
    emit_bus(bus_dir, topic="providers.smoke.gemini3", kind="log", level="info", actor=actor, data=payload)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    ok = sum(1 for r in runs if int(r.get("exit_code") if r.get("exit_code") is not None else 1) == 0)
    return 0 if ok >= int(args.min_success) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
