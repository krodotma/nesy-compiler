#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None


DEFAULT_TOPICS = ["agent.collab.request"]
DEFAULT_MODE = "llm"
DEFAULT_PROVIDERS = ["auto"]
OUTPUT_LIMIT = 12
OUTPUT_MAX_CHARS = 1200


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def bus_dir_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or ".pluribus/bus").expanduser().resolve()


def bus_events_path(bus_dir: Path) -> Path:
    return bus_dir / "events.ndjson"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if agent_bus is not None:
        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
        return
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_events_path(bus_dir), event)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    return []


def normalize_providers(target: Any, providers: list[str]) -> list[str]:
    target_list = _string_list(target)
    if target_list:
        normalized = [p for p in target_list if p.lower() not in {"auto", "any", "*"}]
        if normalized:
            return normalized
        return providers or ["auto"]
    return providers or ["auto"]


def _compact_json(data: dict, max_chars: int = 2000) -> str:
    try:
        dumped = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    except Exception:
        dumped = str(data)
    if len(dumped) > max_chars:
        return dumped[: max_chars - 3] + "..."
    return dumped


def build_prompt(event: dict) -> str:
    topic = str(event.get("topic") or "")
    actor = str(event.get("actor") or "")
    iso = str(event.get("iso") or "")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}

    req_id = str(data.get("req_id") or data.get("request_id") or "")
    target = data.get("target") or data.get("provider") or data.get("providers")
    intent = data.get("intent") or data.get("ask") or data.get("request") or data.get("question")
    constraints = data.get("constraints")
    deliverable = data.get("deliverable")

    lines: list[str] = []
    lines.append("You are an agent collaborating via the Pluribus bus.")
    lines.append("Return a concise, implementation-oriented response.")
    lines.append("")
    lines.append(f"Bus request: topic={topic} actor={actor} iso={iso} req_id={req_id}")
    if target:
        lines.append(f"Target: {target}")
    lines.append("")
    if intent:
        lines.append("Intent:")
        lines.append(str(intent))
        lines.append("")
    if constraints is not None:
        lines.append("Constraints:")
        lines.append(_compact_json(constraints))
        lines.append("")
    if deliverable is not None:
        lines.append("Deliverable:")
        lines.append(_compact_json(deliverable))
        lines.append("")
    if data:
        lines.append("Full request data:")
        lines.append(_compact_json(data))
        lines.append("")
    lines.append("If you propose actions, include an ordered checklist and a test plan.")
    return "\n".join(lines).strip() + "\n"


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


def forward_request(
    *,
    bus_dir: Path,
    actor: str,
    event: dict,
    providers: list[str],
    mode: str,
) -> str | None:
    data = event.get("data") if isinstance(event.get("data"), dict) else None
    if not isinstance(data, dict):
        return None
    req_id = data.get("req_id") or data.get("request_id")
    if not isinstance(req_id, str) or not req_id:
        return None

    target = data.get("target") or data.get("provider") or data.get("providers")
    provider_list = normalize_providers(target, providers)
    mode_value = str(data.get("mode") or mode or DEFAULT_MODE).strip().lower() or DEFAULT_MODE
    prompt = build_prompt(event)

    submit_payload = {
        "req_id": req_id,
        "mode": mode_value,
        "providers": provider_list,
        "prompt": prompt,
        "origin": {
            "topic": event.get("topic"),
            "id": event.get("id"),
            "actor": event.get("actor"),
            "iso": event.get("iso"),
        },
    }
    emit_bus(bus_dir, topic="dialogos.submit", kind="request", level="info", actor=actor, data=submit_payload)
    emit_bus(
        bus_dir,
        topic="agent.collab.forwarded",
        kind="metric",
        level="info",
        actor=actor,
        data={"req_id": req_id, "providers": provider_list, "mode": mode_value},
    )
    return req_id


def _collect_response_payload(outputs: list[dict], end_event: dict | None) -> dict:
    end_data = end_event.get("data") if isinstance(end_event, dict) else {}
    ok = bool(end_data.get("ok")) if isinstance(end_data, dict) else False
    errors = end_data.get("errors") if isinstance(end_data, dict) else []
    if not isinstance(errors, list):
        errors = []
    excerpts: list[dict] = []
    for output in outputs[:OUTPUT_LIMIT]:
        content = str(output.get("content") or "")
        if len(content) > OUTPUT_MAX_CHARS:
            content = content[:OUTPUT_MAX_CHARS] + "..."
        excerpts.append(
            {
                "provider": output.get("provider"),
                "index": output.get("index"),
                "type": output.get("type"),
                "content_excerpt": content,
            }
        )
    return {"ok": ok, "errors": errors, "outputs": excerpts}


def emit_collab_response(
    *,
    bus_dir: Path,
    actor: str,
    origin_topic: str,
    req_id: str,
    outputs: list[dict],
    end_event: dict | None,
    submit_id: str | None,
    response_topic: str | None,
) -> None:
    payload = _collect_response_payload(outputs, end_event)
    payload["req_id"] = req_id
    payload["bridge"] = {"submit_id": submit_id, "dialogos_req_id": req_id}
    topic = response_topic or origin_topic
    emit_bus(
        bus_dir,
        topic=topic,
        kind="response",
        level="info" if payload.get("ok") else "warn",
        actor=actor,
        data=payload,
    )


def process_once(
    *,
    bus_dir: Path,
    actor: str,
    topics: list[str],
    providers: list[str],
    mode: str,
    response_topic: str | None,
) -> int:
    events = list(iter_ndjson(bus_events_path(bus_dir)))
    already_forwarded: set[str] = set()
    already_responded: set[str] = set()
    origin_by_req: dict[str, str] = {}
    outputs_by_req: dict[str, list[dict]] = defaultdict(list)
    end_by_req: dict[str, dict] = {}
    submit_id_by_req: dict[str, str] = {}

    for e in events:
        d = e.get("data")
        if not isinstance(d, dict):
            continue
        req_id = d.get("req_id") or d.get("request_id")
        if not isinstance(req_id, str) or not req_id:
            continue
        if e.get("topic") == "dialogos.submit" and e.get("kind") == "request":
            origin = d.get("origin")
            if isinstance(origin, dict) and isinstance(origin.get("topic"), str):
                origin_by_req[req_id] = origin.get("topic")
            submit_id_by_req[req_id] = e.get("id")
            already_forwarded.add(req_id)
        if e.get("topic") == "dialogos.cell.output" and e.get("kind") == "response":
            outputs_by_req[req_id].append(d)
        if e.get("topic") == "dialogos.cell.end" and e.get("kind") == "response":
            end_by_req[req_id] = e
        if e.get("topic") in DEFAULT_TOPICS and e.get("kind") == "response":
            already_responded.add(req_id)

    processed = 0
    for e in events:
        if e.get("kind") != "request":
            continue
        topic = str(e.get("topic") or "")
        if not _topic_matches(topic, topics):
            continue
        data = e.get("data")
        if not isinstance(data, dict):
            continue
        req_id = data.get("req_id") or data.get("request_id")
        if not isinstance(req_id, str) or not req_id:
            continue
        if req_id in already_forwarded:
            continue
        out = forward_request(bus_dir=bus_dir, actor=actor, event=e, providers=providers, mode=mode)
        if out:
            already_forwarded.add(out)
            origin_by_req[out] = topic
            processed += 1

    for req_id, end_event in end_by_req.items():
        if req_id in already_responded:
            continue
        origin_topic = origin_by_req.get(req_id)
        if not origin_topic:
            continue
        emit_collab_response(
            bus_dir=bus_dir,
            actor=actor,
            origin_topic=origin_topic,
            req_id=req_id,
            outputs=outputs_by_req.get(req_id, []),
            end_event=end_event,
            submit_id=submit_id_by_req.get(req_id),
            response_topic=response_topic,
        )
        already_responded.add(req_id)

    return processed


def run_daemon(
    *,
    bus_dir: Path,
    actor: str,
    topics: list[str],
    providers: list[str],
    mode: str,
    poll_s: float,
    response_topic: str | None,
) -> None:
    events_path = bus_events_path(bus_dir)
    ensure_dir(events_path.parent)
    events_path.touch(exist_ok=True)

    forwarded: set[str] = set()
    responded: set[str] = set()
    origin_by_req: dict[str, str] = {}
    outputs_by_req: dict[str, list[dict]] = defaultdict(list)
    end_by_req: dict[str, dict] = {}
    submit_id_by_req: dict[str, str] = {}

    for e in iter_ndjson(events_path):
        d = e.get("data")
        if not isinstance(d, dict):
            continue
        req_id = d.get("req_id") or d.get("request_id")
        if not isinstance(req_id, str) or not req_id:
            continue
        if e.get("topic") == "dialogos.submit" and e.get("kind") == "request":
            origin = d.get("origin")
            if isinstance(origin, dict) and isinstance(origin.get("topic"), str):
                origin_by_req[req_id] = origin.get("topic")
            submit_id_by_req[req_id] = e.get("id")
            forwarded.add(req_id)
        if e.get("topic") == "dialogos.cell.output" and e.get("kind") == "response":
            outputs_by_req[req_id].append(d)
        if e.get("topic") == "dialogos.cell.end" and e.get("kind") == "response":
            end_by_req[req_id] = e
        if e.get("topic") in DEFAULT_TOPICS and e.get("kind") == "response":
            responded.add(req_id)

    with events_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(max(0.05, poll_s))
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict):
                continue
            topic = str(event.get("topic") or "")
            kind = event.get("kind")
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            req_id = data.get("req_id") or data.get("request_id")
            if isinstance(req_id, str) and req_id:
                if topic == "dialogos.submit" and kind == "request":
                    origin = data.get("origin")
                    if isinstance(origin, dict) and isinstance(origin.get("topic"), str):
                        origin_by_req[req_id] = origin.get("topic")
                    submit_id_by_req[req_id] = event.get("id")
                    forwarded.add(req_id)
                if topic == "dialogos.cell.output" and kind == "response":
                    if req_id in forwarded:
                        outputs_by_req[req_id].append(data)
                if topic == "dialogos.cell.end" and kind == "response":
                    end_by_req[req_id] = event
                    if req_id in forwarded and req_id not in responded:
                        origin_topic = origin_by_req.get(req_id)
                        if origin_topic:
                            emit_collab_response(
                                bus_dir=bus_dir,
                                actor=actor,
                                origin_topic=origin_topic,
                                req_id=req_id,
                                outputs=outputs_by_req.get(req_id, []),
                                end_event=event,
                                submit_id=submit_id_by_req.get(req_id),
                                response_topic=response_topic,
                            )
                            responded.add(req_id)
                if topic in DEFAULT_TOPICS and kind == "response":
                    responded.add(req_id)

            if kind != "request":
                continue
            if not _topic_matches(topic, topics):
                continue
            if not isinstance(req_id, str) or not req_id:
                continue
            if req_id in forwarded or req_id in responded:
                continue
            out = forward_request(bus_dir=bus_dir, actor=actor, event=event, providers=providers, mode=mode)
            if out:
                forwarded.add(out)
                origin_by_req[out] = topic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent_collab_router.py", description="Route agent.collab.request into dialogos.submit and emit responses.")
    parser.add_argument("--bus-dir", default=None)
    parser.add_argument("--actor", default=None)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    parser.add_argument("--providers", default=os.environ.get("COLLAB_PROVIDERS", "auto"))
    parser.add_argument("--mode", default=os.environ.get("COLLAB_MODE", DEFAULT_MODE))
    parser.add_argument("--response-topic", default=os.environ.get("COLLAB_RESPONSE_TOPIC", ""))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll", default=os.environ.get("COLLAB_POLL_S", "0.2"))
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    bus_dir = bus_dir_path(args.bus_dir)
    actor = (args.actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "collab-router").strip()
    topics = [t.strip() for t in str(args.topics or "").split(",") if t.strip()]
    providers = _string_list(args.providers) or DEFAULT_PROVIDERS
    mode = str(args.mode or DEFAULT_MODE).strip().lower() or DEFAULT_MODE
    response_topic = str(args.response_topic or "").strip() or None
    poll_s = float(args.poll or 0.2)

    if args.once:
        processed = process_once(
            bus_dir=bus_dir,
            actor=actor,
            topics=topics,
            providers=providers,
            mode=mode,
            response_topic=response_topic,
        )
        print(f"processed {processed}")
        return 0

    run_daemon(
        bus_dir=bus_dir,
        actor=actor,
        topics=topics,
        providers=providers,
        mode=mode,
        poll_s=poll_s,
        response_topic=response_topic,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
