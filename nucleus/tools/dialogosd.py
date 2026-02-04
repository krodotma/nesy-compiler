#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
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
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
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
    try:
        import agent_bus  # type: ignore

        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
        )
        return
    except Exception:
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


def _resolve_bus_source(cli_value: str | None) -> str:
    source = (cli_value or os.environ.get("PLURIBUS_DIALOGOS_BUS_SOURCE") or "auto").strip().lower()
    if source in ("ndjson", "falkordb"):
        return source
    backend = os.environ.get("PLURIBUS_BUS_BACKEND", "both").strip().lower()
    if backend in ("falkordb", "both"):
        return "falkordb"
    return "ndjson"


def _iter_falkordb_events(topic: str, *, since_ts: float, limit: int = 500) -> list[dict]:
    try:
        from falkordb_bus_events import BusEventService  # type: ignore

        service = BusEventService()
        return service.get_events_by_topic(topic, since_ts=since_ts, limit=limit, ascending=True)
    except Exception:
        return []


def _parse_event_data(data_json: str | None) -> dict | None:
    if not data_json:
        return None
    try:
        obj = json.loads(data_json)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


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


def _cdp_launcher_path() -> Path:
    return Path(__file__).resolve().parent / "chrome_cdp_launcher.py"


def _port_open(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.35)
            return sock.connect_ex(("127.0.0.1", int(port))) == 0
    except Exception:
        return False


def _maybe_start_cdp(port: int, provider: str | None, url_hint: str | None) -> None:
    launcher = _cdp_launcher_path()
    if not launcher.exists():
        return
    cmd = [sys.executable, str(launcher), "start", "--port", str(int(port))]
    if provider:
        cmd.extend(["--provider", provider])
    if url_hint:
        cmd.extend(["--url", url_hint])
    try:
        subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=15,
        )
    except Exception:
        return


def run_provider(*, provider: str, prompt: str, timeout_s: float = 300.0) -> tuple[int, str, str]:
    # DKIN v29: Increased timeout from 120s to 300s for complex provider chains
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


def run_webchat_cdp(
    *,
    provider: str,
    prompt: str,
    action: str,
    timeout_s: float,
    poll_ms: float,
    port: int,
    autostart: bool = False,
    url_hint: str | None = None,
) -> tuple[bool, str, str | None, float, dict]:
    """Run Chrome CDP bridge for DOM-first webchat I/O (no Playwright)."""
    script = Path(__file__).resolve().parent / "chrome_cdp_bridge.mjs"
    if not script.exists():
        return False, "", "cdp_bridge_missing", 0.0, {}

    if autostart and not _port_open(port):
        _maybe_start_cdp(port, provider, url_hint)
        time.sleep(1.2)

    cmd = [
        "node",
        str(script),
        "--provider",
        provider,
        "--action",
        action,
        "--timeout-ms",
        str(int(timeout_s * 1000)),
        "--poll-ms",
        str(int(poll_ms)),
        "--port",
        str(int(port)),
    ]
    if prompt:
        cmd.extend(["--prompt", prompt])

    started = time.time()
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=timeout_s + 10,
    )
    latency_ms = (time.time() - started) * 1000.0

    raw = (proc.stdout or "").strip()
    if not raw:
        err = (proc.stderr or "").strip() or f"cdp_bridge_exit:{proc.returncode}"
        return False, "", err, latency_ms, {}

    try:
        payload = json.loads(raw)
    except Exception:
        err = (proc.stderr or "").strip() or "cdp_bridge_parse_error"
        return False, "", err, latency_ms, {}

    ok = bool(payload.get("ok"))
    response = str(payload.get("response") or "")
    error = payload.get("error")
    return ok, response, (str(error) if error else None), float(payload.get("latency_ms") or latency_ms), payload


def handle_submit(*, bus_dir: Path, trace_path: Path | None, actor: str, submit_event: dict, emit_infer_sync: bool, counters: dict[str, int]) -> bool:
    data = submit_event.get("data") if isinstance(submit_event, dict) else None
    if not isinstance(data, dict):
        return False

    req_id = data.get("req_id") or data.get("request_id")
    mode = (data.get("mode") or "").strip().lower()
    if not isinstance(req_id, str) or not req_id:
        return False
    if mode not in {"llm", "bus", "strp", "shell", "webchat"}:
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

    webchat_provider: str | None = None
    if mode == "webchat":
        provider_override = data.get("provider") or data.get("provider_id")
        webchat_provider = str(provider_override or "gemini-web").strip() or "gemini-web"
        providers = [webchat_provider]

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
    outputs: list[dict[str, Any]] = []

    if mode == "llm":
        for i, provider in enumerate(providers):
            code, out, err = run_provider(provider=provider, prompt=prompt)
            if code != 0:
                ok = False
                errors.append(f"{provider}:exit={code}")
                content = (err or out or "").strip()
                if not content:
                    content = f"provider_failed: {provider} (exit {code})"
                outputs.append({"provider": provider, "type": "error", "content": content})
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
            outputs.append({"provider": provider, "type": "text", "content": (out or "").strip()})
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
    elif mode == "webchat":
        provider = webchat_provider or "gemini-web"
        action = str(data.get("action") or ("send" if prompt else "capture"))
        timeout_s = float(data.get("timeout_s") or 90.0)
        poll_ms = float(data.get("poll_ms") or 800.0)
        port = int(os.environ.get("PLURIBUS_CDP_PORT") or data.get("port") or 9222)
        autostart = str(data.get("cdp_autostart") or os.environ.get("PLURIBUS_CDP_AUTOSTART") or "").strip().lower() in {"1", "true", "yes", "on"}
        url_hint = data.get("cdp_url") or data.get("url")
        if isinstance(url_hint, str):
            url_hint = url_hint.strip() or None
        else:
            url_hint = None

        ok, response, err, latency_ms, meta = run_webchat_cdp(
            provider=provider,
            prompt=prompt,
            action=action,
            timeout_s=timeout_s,
            poll_ms=poll_ms,
            port=port,
            autostart=autostart,
            url_hint=url_hint,
        )
        content = (response or "").strip()
        if not content:
            ok = False
            err = err or "empty_response"
        outputs.append({
            "provider": provider,
            "type": "text" if ok else "error",
            "content": content if content else (err or ""),
            "latency_ms": round(latency_ms, 2),
            "origin": "webchat",
        })

        emit_bus(
            bus_dir,
            topic="dialogos.cell.output",
            kind="response",
            level="info" if ok else "error",
            actor=actor,
            data={
                "req_id": req_id,
                "provider": provider,
                "type": "text" if ok else "error",
                "content": content if content else (err or ""),
                "latency_ms": round(latency_ms, 2),
                "origin": "webchat",
                "action": action,
                "cdp": meta,
            },
        )
    else:
        ok = False
        errors.append(f"unsupported_mode:{mode}")
        outputs.append({"provider": "system", "type": "error", "content": f"unsupported mode: {mode}"})
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
        output_text = "\n\n".join([str(o.get("content") or "") for o in outputs if isinstance(o, dict)])
        output_len = len(output_text)
        output_sha256 = hashlib.sha256(output_text.encode("utf-8", errors="replace")).hexdigest() if output_text else None
        trace_record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "req_id": req_id,
            "mode": mode,
            "providers": providers,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest() if prompt else None,
            "prompt_len": len(prompt) if prompt else 0,
            "output_len": output_len,
            "output_sha256": output_sha256,
            "outputs": outputs,
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


def process_events_once(
    *,
    bus_dir: Path,
    trace_path: Path | None,
    actor: str,
    emit_infer_sync: bool = True,
    bus_source: str = "ndjson",
) -> int:
    if bus_source == "falkordb":
        since = time.time() - 3600
        events = _iter_falkordb_events("dialogos.submit", since_ts=since, limit=500)
        end_events = _iter_falkordb_events("dialogos.cell.end", since_ts=since, limit=500)
        # Normalize to NDJSON-like event dicts
        def _normalize(e):
            d = _parse_event_data(e.get("data_json"))
            if d is None:
                return None
            return {
                "id": e.get("id"),
                "topic": e.get("topic"),
                "kind": "request",
                "actor": e.get("actor"),
                "ts": e.get("ts"),
                "data": d,
            }

        events = [x for x in (_normalize(e) for e in events) if x]
        end_events = [x for x in (_normalize(e) for e in end_events) if x]
        events_all = end_events + events
    else:
        events_all = list(iter_ndjson(bus_events_path(bus_dir)))

    done: set[str] = set()
    for e in events_all:
        if e.get("topic") == "dialogos.cell.end":
            d = e.get("data")
            if isinstance(d, dict) and isinstance(d.get("req_id"), str):
                done.add(d["req_id"])

    processed = 0
    counters: dict[str, int] = {"done": 0, "errors": 0}
    for e in events_all:
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


def run_daemon(
    *,
    bus_dir: Path,
    trace_path: Path | None,
    actor: str,
    poll_s: float,
    emit_infer_sync: bool = True,
    bus_source: str = "ndjson",
) -> int:
    events_path = bus_events_path(bus_dir)
    ensure_dir(events_path.parent)
    events_path.touch(exist_ok=True)

    done: set[str] = set()

    # Seed with already-ended req_ids to avoid replay loops.
    if bus_source == "falkordb":
        since = time.time() - 3600
        for e in _iter_falkordb_events("dialogos.cell.end", since_ts=since, limit=500):
            d = _parse_event_data(e.get("data_json"))
            if isinstance(d, dict) and isinstance(d.get("req_id"), str):
                done.add(d["req_id"])
    else:
        for e in iter_ndjson(events_path):
            if e.get("topic") == "dialogos.cell.end":
                d = e.get("data")
                if isinstance(d, dict) and isinstance(d.get("req_id"), str):
                    done.add(d["req_id"])

    # DKIN v29 Fix: Process pending events BEFORE seeking to EOF
    # This ensures events submitted before daemon start are not ignored
    pending_processed = process_events_once(
        bus_dir=bus_dir,
        trace_path=trace_path,
        actor=actor,
        emit_infer_sync=emit_infer_sync,
        bus_source=bus_source,
    )
    if pending_processed > 0:
        print(f"[dialogosd] Processed {pending_processed} pending events before tailing", file=sys.stderr)

    f = None
    inode = None
    offset = 0
    if bus_source == "ndjson":
        f = events_path.open("r", encoding="utf-8", errors="replace")
        inode = os.fstat(f.fileno()).st_ino
        f.seek(0, os.SEEK_END)
        offset = f.tell()
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
    try:
        while True:
            if bus_source == "falkordb":
                since = time.time() - 3600
                events = _iter_falkordb_events("dialogos.submit", since_ts=since, limit=200)
                for e in events:
                    d = _parse_event_data(e.get("data_json"))
                    if not isinstance(d, dict):
                        continue
                    req_id = d.get("req_id") or d.get("request_id")
                    if not isinstance(req_id, str) or not req_id or req_id in done:
                        continue
                    submit_event = {
                        "id": e.get("id"),
                        "topic": "dialogos.submit",
                        "kind": "request",
                        "actor": e.get("actor"),
                        "ts": e.get("ts"),
                        "data": d,
                    }
                    ok = handle_submit(
                        bus_dir=bus_dir,
                        trace_path=trace_path,
                        actor=actor,
                        submit_event=submit_event,
                        emit_infer_sync=emit_infer_sync,
                        counters=counters,
                    )
                    if ok:
                        done.add(req_id)
                time.sleep(max(0.2, poll_s))
                continue

            line = f.readline() if f else ""
            if not line:
                time.sleep(max(0.05, poll_s))
                try:
                    st = os.stat(events_path)
                except OSError:
                    continue
                if inode is not None and (st.st_ino != inode or st.st_size < offset):
                    f.close()
                    f = events_path.open("r", encoding="utf-8", errors="replace")
                    inode = os.fstat(f.fileno()).st_ino
                    f.seek(0, os.SEEK_SET)
                    offset = 0
                continue
            offset = f.tell()
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
    finally:
        if f:
            f.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dialogosd.py", description="Dialogos daemon: consume dialogos.submit and emit dialogos.cell.* with persistent trace.")
    p.add_argument("--bus-dir", default=None, help="Bus dir (default: $PLURIBUS_BUS_DIR or ./.pluribus/bus).")
    p.add_argument("--trace-path", default=None, help="Persistent trace file (default: $PLURIBUS_DIALOGOS_TRACE or ./.pluribus/dialogos/trace.ndjson).")
    p.add_argument("--actor", default=None)
    p.add_argument("--poll", default="0.1", help="Poll interval seconds (daemon mode).")
    p.add_argument("--bus-source", default="auto", help="Bus source: auto|ndjson|falkordb (default: auto).")
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
    bus_source = _resolve_bus_source(args.bus_source)
    if args.once:
        processed = process_events_once(
            bus_dir=bus_dir,
            trace_path=trace_path,
            actor=actor,
            emit_infer_sync=not bool(args.no_infer_sync),
            bus_source=bus_source,
        )
        sys.stdout.write(f"processed {processed}\n")
        return 0
    run_daemon(
        bus_dir=bus_dir,
        trace_path=trace_path,
        actor=actor,
        poll_s=float(args.poll),
        emit_infer_sync=not bool(args.no_infer_sync),
        bus_source=bus_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
