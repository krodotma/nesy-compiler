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
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def bus_events_path(bus_dir: Path) -> Path:
    return bus_dir / "events.ndjson"


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def default_bus_dir() -> Path:
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or ".pluribus/bus").expanduser().resolve()


def default_trace_path() -> Path:
    """Persistent trace file for ground truth recovery."""
    return Path(os.environ.get("PLURIBUS_DIALOGOS_TRACE") or ".pluribus/dialogos/trace.ndjson").expanduser().resolve()


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "dialogosd"


def append_trace(trace_path: Path, event: dict) -> None:
    """Append to persistent dialogos trace (ground truth for recovery)."""
    ensure_dir(trace_path.parent)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


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
    append_ndjson(bus_events_path(bus_dir), evt)


def emit_infer_sync_checkin(
    bus_dir: Path,
    *,
    actor: str,
    status: str,
    done: int,
    errors: int,
    next_action: str,
) -> None:
    emit_bus(
        bus_dir,
        topic="infer_sync.checkin",
        kind="metric",
        level="info" if status != "error" else "error",
        actor=actor,
        data={
            "status": status,
            "done": int(done),
            "open": 0,
            "blocked": 0,
            "errors": int(errors),
            "next": next_action,
            "subproject": "dialogos",
            "focus": ["dialogosd"],
        },
    )


def _router_path() -> Path:
    return Path(__file__).resolve().parent / "providers" / "router.py"


def run_provider(*, provider: str, prompt: str, timeout_s: float = 120.0) -> tuple[int, str, str]:
    # Never default to `mock` for live runs; use `auto` and let the provider router decide.
    provider = (provider or "auto").strip()
    if provider == "mock":
        return 0, f"[mock] {prompt}", ""

    router = _router_path()
    if not router.exists():
        return 2, "", "missing providers/router.py"

    p = subprocess.run(
        [sys.executable, str(router), "--provider", provider, "--prompt", prompt],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return int(p.returncode), p.stdout, p.stderr


def handle_submit(*, bus_dir: Path, trace_path: Path | None, actor: str, submit_event: dict, emit_infer_sync: bool, counters: dict[str, int]) -> bool:
    data = submit_event.get("data") if isinstance(submit_event, dict) else None
    if not isinstance(data, dict):
        return False

    req_id = data.get("req_id") or data.get("request_id")
    mode = (data.get("mode") or "").strip().lower()
    if not isinstance(req_id, str) or not req_id:
        return False
    if mode not in {"llm", "bus", "strp", "shell"}:
        mode = "llm"

    providers = data.get("providers")
    if not isinstance(providers, list) or not providers:
        providers = ["auto"]
    providers = [str(p).strip() for p in providers if str(p).strip()]
    if not providers:
        providers = ["auto"]

    prompt = data.get("prompt")
    if not isinstance(prompt, str):
        prompt = ""

    emit_bus(
        bus_dir,
        topic="dialogos.cell.start",
        kind="response",
        level="info",
        actor=actor,
        data={"req_id": req_id, "mode": mode, "providers": providers},
    )

    ok = True
    errors: list[str] = []

    if mode == "llm":
        for i, provider in enumerate(providers):
            code, out, err = run_provider(provider=provider, prompt=prompt)
            if code != 0:
                ok = False
                errors.append(f"{provider}:exit={code}")
                content = (err or out or "").strip()
                if not content:
                    content = f"provider_failed: {provider} (exit {code})"
                emit_bus(
                    bus_dir,
                    topic="dialogos.cell.output",
                    kind="response",
                    level="error",
                    actor=actor,
                    data={
                        "req_id": req_id,
                        "provider": provider,
                        "index": i,
                        "type": "error",
                        "content": content,
                    },
                )
                continue
            emit_bus(
                bus_dir,
                topic="dialogos.cell.output",
                kind="response",
                level="info",
                actor=actor,
                data={
                    "req_id": req_id,
                    "provider": provider,
                    "index": i,
                    "type": "text",
                    "content": (out or "").strip(),
                },
            )
    else:
        ok = False
        errors.append(f"unsupported_mode:{mode}")
        emit_bus(
            bus_dir,
            topic="dialogos.cell.output",
            kind="response",
            level="error",
            actor=actor,
            data={"req_id": req_id, "type": "error", "content": f"unsupported mode: {mode}"},
        )

    emit_bus(
        bus_dir,
        topic="dialogos.cell.end",
        kind="response",
        level="info" if ok else "warn",
        actor=actor,
        data={"req_id": req_id, "ok": ok, "errors": errors},
    )

    # Write to persistent trace for ground truth recovery
    if trace_path:
        trace_record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "req_id": req_id,
            "mode": mode,
            "providers": providers,
            "prompt_sha256": __import__("hashlib").sha256(prompt.encode("utf-8", errors="replace")).hexdigest() if prompt else None,
            "prompt_len": len(prompt) if prompt else 0,
            "ok": ok,
            "errors": errors,
            "actor": actor,
        }
        try:
            append_trace(trace_path, trace_record)
        except Exception:
            pass  # Trace write is best-effort

    if emit_infer_sync:
        counters["done"] = int(counters.get("done") or 0) + 1
        if not ok:
            counters["errors"] = int(counters.get("errors") or 0) + 1
        emit_infer_sync_checkin(
            bus_dir,
            actor=actor,
            status="working" if ok else "error",
            done=int(counters.get("done") or 0),
            errors=int(counters.get("errors") or 0),
            next_action="process dialogos.submit",
        )
    return True


def process_events_once(*, bus_dir: Path, trace_path: Path | None, actor: str, emit_infer_sync: bool = True) -> int:
    events = list(iter_ndjson(bus_events_path(bus_dir)))
    done: set[str] = set()
    for e in events:
        if e.get("topic") == "dialogos.cell.end":
            d = e.get("data")
            if isinstance(d, dict) and isinstance(d.get("req_id"), str):
                done.add(d["req_id"])

    processed = 0
    counters: dict[str, int] = {"done": 0, "errors": 0}
    for e in events:
        if e.get("topic") != "dialogos.submit":
            continue
        if e.get("kind") != "request":
            continue
        d = e.get("data")
        if not isinstance(d, dict):
            continue
        req_id = d.get("req_id") or d.get("request_id")
        if not isinstance(req_id, str) or not req_id:
            continue
        if req_id in done:
            continue
        if handle_submit(bus_dir=bus_dir, trace_path=trace_path, actor=actor, submit_event=e, emit_infer_sync=emit_infer_sync, counters=counters):
            processed += 1
    return processed


def run_daemon(*, bus_dir: Path, trace_path: Path | None, actor: str, poll_s: float, emit_infer_sync: bool = True) -> int:
    events_path = bus_events_path(bus_dir)
    ensure_dir(events_path.parent)
    events_path.touch(exist_ok=True)

    done: set[str] = set()

    # Seed with already-ended req_ids to avoid replay loops.
    for e in iter_ndjson(events_path):
        if e.get("topic") == "dialogos.cell.end":
            d = e.get("data")
            if isinstance(d, dict) and isinstance(d.get("req_id"), str):
                done.add(d["req_id"])

    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        counters: dict[str, int] = {"done": 0, "errors": 0}
        if emit_infer_sync:
            emit_infer_sync_checkin(
                bus_dir,
                actor=actor,
                status="working",
                done=0,
                errors=0,
                next_action="tail dialogos.submit",
            )
        while True:
            line = f.readline()
            if not line:
                time.sleep(max(0.05, poll_s))
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if not isinstance(e, dict):
                continue
            if e.get("topic") != "dialogos.submit" or e.get("kind") != "request":
                continue
            d = e.get("data")
            if not isinstance(d, dict):
                continue
            req_id = d.get("req_id") or d.get("request_id")
            if not isinstance(req_id, str) or not req_id or req_id in done:
                continue
            ok = handle_submit(bus_dir=bus_dir, trace_path=trace_path, actor=actor, submit_event=e, emit_infer_sync=emit_infer_sync, counters=counters)
            if ok:
                done.add(req_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dialogosd.py", description="Dialogos daemon: consume dialogos.submit and emit dialogos.cell.* with persistent trace.")
    p.add_argument("--bus-dir", default=None, help="Bus dir (default: $PLURIBUS_BUS_DIR or ./.pluribus/bus).")
    p.add_argument("--trace-path", default=None, help="Persistent trace file (default: $PLURIBUS_DIALOGOS_TRACE or ./.pluribus/dialogos/trace.ndjson).")
    p.add_argument("--actor", default=None)
    p.add_argument("--poll", default="0.1", help="Poll interval seconds (daemon mode).")
    p.add_argument("--once", action="store_true", help="Process pending dialogos.submit events already in the bus file, then exit.")
    p.add_argument("--no-infer-sync", action="store_true", help="Disable infer_sync.checkin emissions.")
    p.add_argument("--no-trace", action="store_true", help="Disable persistent trace file writing.")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else default_bus_dir()
    trace_path = None
    if not args.no_trace:
        trace_path = Path(args.trace_path).expanduser().resolve() if args.trace_path else default_trace_path()
    actor = (args.actor or default_actor()).strip() or "dialogosd"
    if args.once:
        processed = process_events_once(bus_dir=bus_dir, trace_path=trace_path, actor=actor, emit_infer_sync=not bool(args.no_infer_sync))
        sys.stdout.write(f"processed {processed}\n")
        return 0
    run_daemon(bus_dir=bus_dir, trace_path=trace_path, actor=actor, poll_s=float(args.poll), emit_infer_sync=not bool(args.no_infer_sync))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
