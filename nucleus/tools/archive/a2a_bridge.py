#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "a2a-bridge"


def bus_events_path(bus_dir: Path) -> Path:
    return bus_dir / "events.ndjson"


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


def _topic_matches(topic: str, patterns: list[str]) -> bool:
    if not topic:
        return False
    for pat in patterns:
        p = (pat or "").strip()
        if not p:
            continue
        if p == topic:
            return True
        if p.endswith("*") and topic.startswith(p[:-1]):
            return True
    return False


def _compile_prompt(event: dict) -> str:
    topic = str(event.get("topic") or "")
    actor = str(event.get("actor") or "")
    iso = str(event.get("iso") or "")
    data = event.get("data")
    if not isinstance(data, dict):
        data = {}

    req_id = str(data.get("req_id") or data.get("request_id") or "")
    ask = data.get("ask")
    followup = data.get("followup")
    constraints = data.get("constraints")
    context_refs = data.get("context_refs")
    deliverable = data.get("deliverable")

    lines: list[str] = []
    lines.append("You are an agent collaborating via an append-only NDJSON bus.")
    lines.append("Return a concise, implementation-oriented response.")
    lines.append("")
    lines.append(f"Bus request: topic={topic} actor={actor} iso={iso} req_id={req_id}")
    lines.append("")

    if isinstance(ask, list) and ask:
        lines.append("Ask:")
        for i, item in enumerate(ask, start=1):
            lines.append(f"{i}. {str(item)}")
        lines.append("")
    elif isinstance(ask, str) and ask.strip():
        lines.append("Ask:")
        lines.append(ask.strip())
        lines.append("")

    if isinstance(followup, str) and followup.strip():
        lines.append("Follow-up:")
        lines.append(followup.strip())
        lines.append("")

    if constraints is not None:
        lines.append("Constraints:")
        try:
            lines.append(json.dumps(constraints, ensure_ascii=False, indent=2))
        except Exception:
            lines.append(str(constraints))
        lines.append("")

    if context_refs is not None:
        lines.append("Context refs:")
        try:
            lines.append(json.dumps(context_refs, ensure_ascii=False, indent=2))
        except Exception:
            lines.append(str(context_refs))
        lines.append("")

    if deliverable is not None:
        lines.append("Deliverable:")
        try:
            lines.append(json.dumps(deliverable, ensure_ascii=False, indent=2))
        except Exception:
            lines.append(str(deliverable))
        lines.append("")

    lines.append("If you propose actions, include an ordered checklist and a test plan.")
    return "\n".join(lines).strip() + "\n"


def bridge_request_to_dialogos_submit(
    *,
    bus_dir: Path,
    actor: str,
    request_event: dict,
    providers: list[str],
    mode: str,
    emit_link_event: bool,
) -> str | None:
    data = request_event.get("data") if isinstance(request_event, dict) else None
    if not isinstance(data, dict):
        return None
    req_id = data.get("req_id") or data.get("request_id")
    if not isinstance(req_id, str) or not req_id:
        return None

    prompt = _compile_prompt(request_event)

    submit_payload = {
        "req_id": req_id,
        "mode": mode,
        "providers": providers,
        "prompt": prompt,
        "origin": {
            "topic": request_event.get("topic"),
            "id": request_event.get("id"),
            "actor": request_event.get("actor"),
            "iso": request_event.get("iso"),
        },
    }

    emit_bus(
        bus_dir,
        topic="dialogos.submit",
        kind="request",
        level="info",
        actor=actor,
        data=submit_payload,
    )

    if emit_link_event:
        emit_bus(
            bus_dir,
            topic="a2a.bridge.forwarded",
            kind="metric",
            level="info",
            actor=actor,
            data={
                "req_id": req_id,
                "from_topic": request_event.get("topic"),
                "to_topic": "dialogos.submit",
                "providers": providers,
                "mode": mode,
            },
        )

    return req_id


def emit_a2a_response_from_bus(*, bus_dir: Path, actor: str, req_id: str) -> bool:
    events = list(iter_ndjson(bus_events_path(bus_dir)))

    origin_topic: str | None = None
    submit_id: str | None = None

    outputs: list[dict] = []
    end_event: dict | None = None

    for e in events:
        d = e.get("data")
        if not isinstance(d, dict):
            continue
        rid = d.get("req_id") or d.get("request_id")
        if rid != req_id:
            continue
        if e.get("topic") == "dialogos.submit" and e.get("kind") == "request":
            origin = d.get("origin")
            if isinstance(origin, dict) and isinstance(origin.get("topic"), str):
                origin_topic = origin.get("topic")
            submit_id = e.get("id") if isinstance(e.get("id"), str) else submit_id
        if e.get("topic") == "dialogos.cell.output" and e.get("kind") == "response":
            outputs.append(d)
        if e.get("topic") == "dialogos.cell.end" and e.get("kind") == "response":
            end_event = e

    if not origin_topic:
        return False
    if not end_event:
        return False

    # Avoid duplicate a2a responses.
    for e in events:
        if e.get("topic") != origin_topic or e.get("kind") != "response":
            continue
        d = e.get("data")
        if isinstance(d, dict) and (d.get("req_id") or d.get("request_id")) == req_id:
            return False

    end_data = end_event.get("data") if isinstance(end_event.get("data"), dict) else {}
    ok = bool(end_data.get("ok"))
    errors = end_data.get("errors")
    if not isinstance(errors, list):
        errors = []

    excerpts: list[dict] = []
    for o in outputs[:12]:
        if not isinstance(o, dict):
            continue
        content = str(o.get("content") or "")
        if len(content) > 800:
            content = content[:800] + "…"
        excerpts.append(
            {
                "provider": o.get("provider"),
                "index": o.get("index"),
                "type": o.get("type"),
                "content_excerpt": content,
            }
        )

    emit_bus(
        bus_dir,
        topic=origin_topic,
        kind="response",
        level="info" if ok else "warn",
        actor=actor,
        data={
            "req_id": req_id,
            "ok": ok,
            "errors": errors,
            "outputs": excerpts,
            "bridge": {"submit_id": submit_id, "dialogos_req_id": req_id},
        },
    )
    return True


def process_events_once(
    *,
    bus_dir: Path,
    actor: str,
    topics: list[str],
    providers: list[str],
    mode: str,
    emit_link_event: bool = True,
) -> int:
    events_path = bus_events_path(bus_dir)
    events = list(iter_ndjson(events_path))

    already_forwarded: set[str] = set()
    already_done: set[str] = set()

    for e in events:
        d = e.get("data")
        if not isinstance(d, dict):
            continue
        rid = d.get("req_id") or d.get("request_id")
        if not isinstance(rid, str) or not rid:
            continue
        if e.get("topic") == "dialogos.submit" and e.get("kind") == "request":
            already_forwarded.add(rid)
        if e.get("topic") == "dialogos.cell.end" and e.get("kind") == "response":
            already_done.add(rid)

    processed = 0
    for e in events:
        if e.get("kind") != "request":
            continue
        topic = str(e.get("topic") or "")
        if not _topic_matches(topic, topics):
            continue
        d = e.get("data")
        if not isinstance(d, dict):
            continue
        rid = d.get("req_id") or d.get("request_id")
        if not isinstance(rid, str) or not rid:
            continue
        if rid in already_forwarded or rid in already_done:
            continue
        out = bridge_request_to_dialogos_submit(
            bus_dir=bus_dir,
            actor=actor,
            request_event=e,
            providers=providers,
            mode=mode,
            emit_link_event=emit_link_event,
        )
        if out:
            processed += 1
            already_forwarded.add(out)
    return processed


def process_responses_once(*, bus_dir: Path, actor: str) -> int:
    events = list(iter_ndjson(bus_events_path(bus_dir)))
    ended: set[str] = set()
    for e in events:
        if e.get("topic") != "dialogos.cell.end" or e.get("kind") != "response":
            continue
        d = e.get("data")
        if isinstance(d, dict) and isinstance(d.get("req_id"), str) and d.get("req_id"):
            ended.add(d["req_id"])

    emitted = 0
    for rid in sorted(ended):
        if emit_a2a_response_from_bus(bus_dir=bus_dir, actor=actor, req_id=rid):
            emitted += 1
    return emitted


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="a2a_bridge.py", description="A2A bridge: forward selected bus requests into dialogos.submit for execution.")
    p.add_argument("--bus-dir", default=None, help="Bus dir (default: $PLURIBUS_BUS_DIR or ./.pluribus/bus).")
    p.add_argument("--actor", default=None)
    p.add_argument("--topics", default="", help="Comma list of bus topics or prefixes (suffix *).")
    p.add_argument("--providers", default="auto", help="Comma list of providers for dialogosd (default: auto).")
    p.add_argument("--mode", default="llm", help="Dialogos mode (default: llm).")
    p.add_argument("--once", action="store_true", help="Process existing bus events, then exit.")
    p.add_argument("--poll", default="0.2", help="Poll interval seconds (daemon mode).")
    return p


def run_daemon(
    *,
    bus_dir: Path,
    actor: str,
    topics: list[str],
    providers: list[str],
    mode: str,
    poll_s: float,
) -> int:
    events_path = bus_events_path(bus_dir)
    ensure_dir(events_path.parent)
    events_path.touch(exist_ok=True)

    # In daemon mode, we "tail" and only forward new matching requests.
    forwarded: set[str] = set()
    completed: set[str] = set()
    origin_by_req_id: dict[str, str] = {}
    response_emitted: set[str] = set()

    for e in iter_ndjson(events_path):
        d = e.get("data")
        if not isinstance(d, dict):
            continue
        rid = d.get("req_id") or d.get("request_id")
        if isinstance(rid, str) and rid:
            if e.get("topic") == "dialogos.submit" and e.get("kind") == "request":
                forwarded.add(rid)
                origin = d.get("origin")
                if isinstance(origin, dict) and isinstance(origin.get("topic"), str):
                    origin_by_req_id[rid] = origin["topic"]
            if e.get("topic") == "dialogos.cell.end" and e.get("kind") == "response":
                completed.add(rid)
            if e.get("kind") == "response":
                # Any pre-existing response on a topic with same req_id counts as emitted.
                if isinstance(d.get("req_id"), str):
                    response_emitted.add(d["req_id"])

    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
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
            if e.get("kind") != "request":
                # Also watch for dialogos completion to emit a2a response.
                if e.get("topic") == "dialogos.submit" and e.get("kind") == "request":
                    d = e.get("data")
                    if isinstance(d, dict):
                        rid = d.get("req_id") or d.get("request_id")
                        origin = d.get("origin")
                        if isinstance(rid, str) and isinstance(origin, dict) and isinstance(origin.get("topic"), str):
                            origin_by_req_id[rid] = origin["topic"]
                if e.get("topic") == "dialogos.cell.end" and e.get("kind") == "response":
                    d = e.get("data")
                    rid = (d.get("req_id") if isinstance(d, dict) else None)
                    if isinstance(rid, str) and rid and rid in origin_by_req_id and rid not in response_emitted:
                        if emit_a2a_response_from_bus(bus_dir=bus_dir, actor=actor, req_id=rid):
                            response_emitted.add(rid)
                continue
            topic = str(e.get("topic") or "")
            if not _topic_matches(topic, topics):
                continue
            d = e.get("data")
            if not isinstance(d, dict):
                continue
            rid = d.get("req_id") or d.get("request_id")
            if not isinstance(rid, str) or not rid:
                continue
            if rid in forwarded or rid in completed:
                continue
            out = bridge_request_to_dialogos_submit(
                bus_dir=bus_dir,
                actor=actor,
                request_event=e,
                providers=providers,
                mode=mode,
                emit_link_event=True,
            )
            if out:
                forwarded.add(out)


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else default_bus_dir()
    actor = (args.actor or default_actor()).strip() or "a2a-bridge"
    topics = [t.strip() for t in str(args.topics or "").split(",") if t.strip()]
    if not topics:
        # safe defaults: only known research topics; operator can extend via --topics.
        topics = [
            "sky.signaling.research",
            "sky.signaling.research.followup",
            "lens.collimator.router.research",
            "lens.collimator.router.research.followup",
        ]
    providers = [p.strip() for p in str(args.providers or "").split(",") if p.strip()] or ["auto"]
    mode = (args.mode or "llm").strip().lower() or "llm"

    if args.once:
        processed = process_events_once(bus_dir=bus_dir, actor=actor, topics=topics, providers=providers, mode=mode)
        _ = process_responses_once(bus_dir=bus_dir, actor=actor)
        print(f"processed {processed}")
        return 0

    run_daemon(bus_dir=bus_dir, actor=actor, topics=topics, providers=providers, mode=mode, poll_s=float(args.poll))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
