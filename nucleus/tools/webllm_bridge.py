#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from nucleus.tools import agent_bus
except Exception:  # pragma: no cover
    import agent_bus  # type: ignore


@dataclass(frozen=True)
class WebLLMResponse:
    ok: bool
    req_id: str
    topic: str
    data: dict[str, Any]


def resolve_bus_paths(bus_dir: str | None) -> Any:
    return agent_bus.resolve_bus_paths(bus_dir or os.environ.get("PLURIBUS_BUS_DIR"))


def emit(paths: Any, *, topic: str, kind: str, actor: str, data: dict[str, Any]) -> None:
    agent_bus.emit_event(
        paths,
        topic=topic,
        kind=kind,
        level="info",
        actor=actor,
        data=data,
        trace_id=None,
        run_id=None,
        durable=True,
    )


def tail_for_req_id(
    *,
    events_path: Path,
    topic: str,
    req_id: str,
    timeout_s: float,
) -> WebLLMResponse:
    deadline = time.time() + float(timeout_s)
    pos = events_path.stat().st_size if events_path.exists() else 0
    buf = b""

    while time.time() < deadline:
        if not events_path.exists():
            time.sleep(0.2)
            continue

        with events_path.open("rb") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()

        if not chunk:
            time.sleep(0.2)
            continue

        data = buf + chunk
        parts = data.split(b"\n")
        buf = parts.pop() if parts else b""

        for raw in parts:
            if not raw:
                continue
            try:
                ev = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if (ev.get("topic") or "") != topic:
                continue
            payload = ev.get("data") or {}
            if not isinstance(payload, dict):
                continue
            if str(payload.get("req_id") or "") != req_id:
                continue
            ok = bool(payload.get("ok", True))
            return WebLLMResponse(ok=ok, req_id=req_id, topic=topic, data=payload)

    return WebLLMResponse(ok=False, req_id=req_id, topic=topic, data={"req_id": req_id, "ok": False, "error": "timeout"})


def cmd_status(args: argparse.Namespace) -> int:
    req_id = f"webllm-status-{uuid.uuid4()}"
    paths = resolve_bus_paths(args.bus_dir)
    bus_dir = Path(paths.active_dir)
    emit(
        paths,
        topic="webllm.status.request",
        kind="request",
        actor=args.actor,
        data={"req_id": req_id, "at": time.time()},
    )
    resp = tail_for_req_id(
        events_path=bus_dir / "events.ndjson",
        topic="webllm.status.response",
        req_id=req_id,
        timeout_s=float(args.timeout_s),
    )
    print(json.dumps(resp.data, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if resp.ok else 2


def cmd_infer(args: argparse.Namespace) -> int:
    req_id = f"webllm-infer-{uuid.uuid4()}"
    paths = resolve_bus_paths(args.bus_dir)
    bus_dir = Path(paths.active_dir)
    emit(
        paths,
        topic="webllm.infer.request",
        kind="request",
        actor=args.actor,
        data={
            "req_id": req_id,
            "prompt": args.prompt,
            "session_id": args.session_id,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "at": time.time(),
        },
    )
    resp = tail_for_req_id(
        events_path=bus_dir / "events.ndjson",
        topic="webllm.infer.response",
        req_id=req_id,
        timeout_s=float(args.timeout_s),
    )
    print(json.dumps(resp.data, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if resp.ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bus-driven WebLLM bridge (requires dashboard WebLLM enabled).")
    parser.add_argument("--bus-dir", default=None, help="Bus dir override (default: resolve via agent_bus).")
    parser.add_argument("--actor", default=os.environ.get("PLURIBUS_ACTOR") or "webllm-bridge", help="Bus actor.")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Timeout seconds.")

    sub = parser.add_subparsers(dest="cmd", required=True)
    p_status = sub.add_parser("status", help="Request current WebLLM widget status via the bus.")
    p_status.set_defaults(func=cmd_status)

    p_infer = sub.add_parser("infer", help="Send a one-shot prompt to a ready WebLLM session via the bus.")
    p_infer.add_argument("prompt", help="User prompt.")
    p_infer.add_argument("--session-id", default=None, help="Target session id (optional).")
    p_infer.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    p_infer.add_argument("--max-tokens", type=int, default=256, help="Max tokens.")
    p_infer.set_defaults(func=cmd_infer)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

