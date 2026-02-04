#!/usr/bin/env python3
from __future__ import annotations

"""
SPEAK — write text to the speaker bus endpoint for TTS broadcast.

Default endpoint: /pluribus/speaker_bus.aiff
"""

import argparse
import getpass
import hashlib
import json
import os
import socket
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _bool_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _safe_excerpt(text: str, limit: int = 240) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "…"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speak_operator.py", description="SPEAK semantic operator: write text to /pluribus/speaker_bus.aiff")
    p.add_argument("--file", default="/pluribus/speaker_bus.aiff", help="Speaker bus endpoint file.")
    p.add_argument("--text", default=None, help="Text to write (default: read stdin).")
    p.add_argument("--append", action="store_true", help="Append instead of truncate.")
    p.add_argument("--emit-bus", dest="emit_bus", action="store_true", default=True, help="Emit a bus artifact event (default: on).")
    p.add_argument("--no-emit-bus", dest="emit_bus", action="store_false", help="Disable bus event emission.")
    p.add_argument("--broadcast", action="store_true", help="Attempt SSH broadcast via broadcast.sh (best-effort).")
    p.add_argument("--broadcast-script", default="/pluribus/broadcast.sh", help="Broadcast script path.")
    p.add_argument("--broadcast-timeout", type=float, default=8.0, help="Broadcast timeout in seconds.")
    p.add_argument("--context-json", default=None, help="Optional JSON payload for verbose context.")
    p.add_argument("--source", default="semops.speak", help="Source label for bus context.")
    p.add_argument("--reason", default="operator_speak", help="Reason label for bus context.")
    p.add_argument("--req-id", default=None, help="Optional request id.")
    p.add_argument("--bus-dir", default=None, help="Bus directory (default: $PLURIBUS_BUS_DIR).")
    p.add_argument("--actor", default=None, help="Actor identity (default: $PLURIBUS_ACTOR).")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()
    if text is None:
        text = ""
    if not text.strip():
        sys.stderr.write("SPEAK requires non-empty text.\n")
        return 2

    path = Path(str(args.file)).expanduser().resolve()
    if not path.parent.exists():
        ensure_dir(path.parent)
    if not os.access(path.parent, os.W_OK):
        sys.stderr.write(f"Speaker bus directory not writable: {path.parent}\n")
        return 3

    mode = "a" if args.append else "w"
    with path.open(mode, encoding="utf-8") as f:
        f.write(text)

    broadcast_status: dict | None = None
    should_broadcast = bool(args.broadcast) or _bool_env("PLURIBUS_SPEAK_BROADCAST")
    if should_broadcast:
        script = Path(args.broadcast_script).expanduser().resolve()
        started = time.monotonic()
        if not script.exists():
            broadcast_status = {
                "attempted": True,
                "status": "missing_script",
                "script": str(script),
            }
        else:
            try:
                result = subprocess.run(
                    [str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(1.0, float(args.broadcast_timeout)),
                    env=os.environ.copy(),
                )
                duration_ms = int((time.monotonic() - started) * 1000)
                broadcast_status = {
                    "attempted": True,
                    "status": "ok" if result.returncode == 0 else "failed",
                    "exit_code": result.returncode,
                    "duration_ms": duration_ms,
                    "stdout_excerpt": _safe_excerpt(result.stdout),
                    "stderr_excerpt": _safe_excerpt(result.stderr),
                }
            except subprocess.TimeoutExpired:
                duration_ms = int((time.monotonic() - started) * 1000)
                broadcast_status = {
                    "attempted": True,
                    "status": "timeout",
                    "duration_ms": duration_ms,
                    "timeout_s": float(args.broadcast_timeout),
                }
            except Exception as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                broadcast_status = {
                    "attempted": True,
                    "status": "error",
                    "duration_ms": duration_ms,
                    "error": str(exc)[:240],
                }

    if args.emit_bus:
        actor = (args.actor or default_actor()).strip() or "speak"
        bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()
        ensure_dir(bus_dir)
        (bus_dir / "events.ndjson").touch(exist_ok=True)
        req_id = str(args.req_id or uuid.uuid4())
        context_payload: dict = {}
        context_error = None
        if args.context_json:
            try:
                parsed = json.loads(str(args.context_json))
                if isinstance(parsed, dict):
                    context_payload = parsed
                else:
                    context_error = "context_json_not_object"
            except Exception as exc:
                context_error = str(exc)[:160]
        max_text_chars = 8000
        truncated = len(text) > max_text_chars
        payload_text = text[:max_text_chars] if truncated else text
        emit_bus(
            bus_dir,
            topic="speaker.bus.write",
            kind="artifact",
            level="info",
            actor=actor,
            data={
                "req_id": req_id,
                "path": str(path),
                "text": payload_text,
                "text_excerpt": _safe_excerpt(payload_text, limit=280),
                "text_chars": len(text),
                "text_sha256": sha256_text(text),
                "text_truncated": truncated,
                "append": bool(args.append),
                "source": str(args.source),
                "reason": str(args.reason),
                "host": socket.gethostname(),
                "cwd": os.getcwd(),
                "pid": os.getpid(),
                "context": context_payload,
                "context_error": context_error,
                "broadcast": broadcast_status,
                "iso": now_iso_utc(),
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
