#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from nucleus.tools.meta_repo import MetaRepo, resolve_paths
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from nucleus.tools.meta_repo import MetaRepo, resolve_paths

sys.dont_write_bytecode = True

MAX_PREVIEW = 240
BUS_TAIL_BYTES = 200_000
TRACE_TAIL_BYTES = 200_000
LEDGER_TAIL_BYTES = 200_000
TRANSCRIPT_PARSE_MAX_BYTES = 2_000_000

PROMPT_TOPICS = {
    "dialogos.submit",
    "dialogos.request",
    "dialogos.claude_code.prompt",
}
RESPONSE_PREFIXES = ("dialogos.cell.",)
RESPONSE_TOPICS = {
    "dialogos.claude_code.stop",
}
TASK_TOPICS_SUFFIX = ".task"
VERIFICATION_PREFIXES = ("operator.pbtest", "operator.pbreality")
QA_PREFIXES = ("qa.",)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _preview(text: Optional[str]) -> str:
    if not text:
        return ""
    if len(text) <= MAX_PREVIEW:
        return text
    return text[:MAX_PREVIEW]

def _normalize_role(role: Optional[str]) -> str:
    role = (role or "unknown").strip().lower()
    if role in {"assistant", "ai", "model", "bot"}:
        return "assistant"
    if role in {"user", "human", "client", "customer"}:
        return "user"
    if role in {"system", "sys"}:
        return "system"
    return role or "unknown"


def _extract_content(message: dict) -> str:
    content = message.get("content")
    if content is None:
        content = message.get("text") or message.get("message") or message.get("output")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        content = "\n".join(p for p in parts if p)
    if content is None:
        content = ""
    return str(content)


def _parse_transcript_payload(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("messages", "conversation", "turns", "items", "events"):
            if isinstance(payload.get(key), list):
                return payload.get(key)  # type: ignore[return-value]
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


def _parse_plaintext_transcript(text: str) -> list[dict]:
    messages: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("user:"):
            messages.append({"role": "user", "content": stripped[5:].strip()})
        elif lower.startswith("assistant:"):
            messages.append({"role": "assistant", "content": stripped[10:].strip()})
        elif lower.startswith("system:"):
            messages.append({"role": "system", "content": stripped[7:].strip()})
    if messages:
        return messages
    return [{"role": "assistant", "content": text.strip()}] if text.strip() else []


def _load_transcript(path: Path, max_bytes: int) -> tuple[Optional[Any], Optional[bytes]]:
    if not path.exists():
        return None, None
    size = path.stat().st_size
    if size > max_bytes:
        return None, None
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8", errors="replace")), raw
    except Exception:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return None, raw
        ndjson_records = []
        parse_ok = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ndjson_records.append(json.loads(stripped))
                parse_ok = True
            except Exception:
                parse_ok = False
                ndjson_records = []
                break
        if parse_ok and ndjson_records:
            return ndjson_records, raw
        return text, raw


def ingest_transcript(repo: MetaRepo, *, transcript_path: Path, session_id: Optional[str], actor: str) -> None:
    if not transcript_path.exists():
        return
    stored = repo.store_file(transcript_path)
    if not stored:
        return
    if repo.index is not None and repo.index.has_payload_ref(stored.get("ref", "")):
        return

    meta_event = {
        "kind": "transcript",
        "topic": "dialogos.claude_code.transcript",
        "actor": actor,
        "session_id": session_id,
        "summary": transcript_path.name,
        "payload_ref": stored.get("ref"),
        "payload_bytes": stored.get("bytes"),
        "payload_ext": stored.get("ext"),
        "payload_type": stored.get("content_type"),
        "source": "transcript",
        "tags": ["claude-code", "transcript"],
    }
    repo.append_event(meta_event)

    payload, _raw = _load_transcript(transcript_path, TRANSCRIPT_PARSE_MAX_BYTES)
    if payload is None:
        return

    messages = []
    if isinstance(payload, str):
        messages = _parse_plaintext_transcript(payload)
    else:
        messages = _parse_transcript_payload(payload)
    if not messages:
        return

    base_ref = (stored.get("ref") or "transcript").split(":")[-1][:8]
    turn = 0
    current_req = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _normalize_role(message.get("role") or message.get("author") or message.get("type"))
        content = _extract_content(message)
        if not content:
            continue

        if role in {"user", "system"}:
            turn += 1
            current_req = f"cc-{base_ref}-{turn:03d}"
            kind = "prompt"
        elif role == "assistant":
            if current_req is None:
                turn += 1
                current_req = f"cc-{base_ref}-{turn:03d}"
            kind = "response"
        else:
            continue

        payload_info = _event_payload(repo, {"role": role, "content": content})
        event_out = _base_event(kind, "dialogos.claude_code.transcript", actor=actor, source="transcript")
        event_out.update(
            {
                "req_id": current_req,
                "session_id": session_id,
                "lane": "dialogos",
                "summary": _preview(content),
                "tags": ["transcript", role],
            }
        )
        event_out.update(payload_info)
        repo.append_event(event_out)


@dataclass
class FileCursor:
    path: Path
    offset: int = 0
    inode: int = 0
    partial: bytes = b""

    @classmethod
    def from_state(cls, path: Path, state: dict) -> "FileCursor":
        entry = state.get(str(path), {})
        partial = base64.b64decode(entry["partial_b64"]) if entry.get("partial_b64") else b""
        return cls(
            path=path,
            offset=int(entry.get("offset", 0)),
            inode=int(entry.get("inode", 0)),
            partial=partial,
        )

    def to_state(self) -> dict:
        return {
            "offset": self.offset,
            "inode": self.inode,
            "partial_b64": base64.b64encode(self.partial).decode("ascii") if self.partial else "",
        }


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def _read_new_lines(
    cursor: FileCursor,
    tail_bytes: int,
    *,
    max_bytes: Optional[int] = None,
    max_line_bytes: Optional[int] = None,
) -> list[str]:
    if not cursor.path.exists():
        return []
    try:
        st = cursor.path.stat()
    except OSError:
        return []

    inode = st.st_ino or 0
    size = st.st_size
    if cursor.inode != inode or size < cursor.offset:
        cursor.offset = max(0, size - tail_bytes)
        cursor.partial = b""
        cursor.inode = inode

    if max_line_bytes and cursor.partial and len(cursor.partial) > max_line_bytes:
        cursor.partial = b""

    try:
        with cursor.path.open("rb") as handle:
            handle.seek(cursor.offset)
            if max_bytes and max_bytes > 0:
                chunk = handle.read(max_bytes)
            else:
                chunk = handle.read()
            cursor.offset = handle.tell()
    except OSError:
        return []

    if not chunk:
        return []

    data = cursor.partial + chunk
    lines = data.split(b"\n")
    if data and not data.endswith(b"\n"):
        cursor.partial = lines.pop()
        if max_line_bytes and len(cursor.partial) > max_line_bytes:
            cursor.partial = b""
    else:
        cursor.partial = b""

    output = []
    for line in lines:
        if not line:
            continue
        if max_line_bytes and len(line) > max_line_bytes:
            continue
        output.append(line.decode("utf-8", errors="replace"))
    return output


def _safe_json(line: str) -> Optional[dict]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _base_event(kind: str, topic: str, *, actor: str, source: str) -> dict:
    return {
        "kind": kind,
        "topic": topic,
        "actor": actor,
        "source": source,
    }


def _event_payload(repo: MetaRepo, payload: Any) -> dict:
    stored = repo.store_object(payload) if payload is not None else {}
    if not stored:
        return {}
    return {
        "payload_ref": stored.get("ref"),
        "payload_bytes": stored.get("bytes"),
        "payload_ext": stored.get("ext"),
        "payload_type": stored.get("content_type"),
    }


def ingest_bus_event(repo: MetaRepo, event: dict) -> None:
    topic = str(event.get("topic", ""))
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    actor = event.get("actor") or data.get("actor") or "unknown"
    req_id = data.get("req_id") or data.get("request_id") or event.get("req_id")
    run_id = event.get("run_id") or data.get("run_id")
    session_id = data.get("session_id")
    trace_id = event.get("trace_id") or data.get("trace_id")
    lane = data.get("lane") or ("dialogos" if topic.startswith("dialogos.") else None)

    kind = "event"
    if topic in PROMPT_TOPICS:
        kind = "prompt"
    elif topic.startswith(RESPONSE_PREFIXES) or topic in RESPONSE_TOPICS:
        kind = "response"
    elif topic.endswith(TASK_TOPICS_SUFFIX):
        kind = "task"
    elif topic.startswith(VERIFICATION_PREFIXES):
        kind = "verification"
    elif topic.startswith(QA_PREFIXES):
        kind = "qa"
    elif topic in {"task.dispatch", "rd.tasks.dispatch"}:
        kind = "dispatch"

    if kind == "event":
        return

    payload = None
    preview = ""
    if kind == "prompt":
        prompt = data.get("prompt") or data.get("message")
        payload = {
            "prompt": prompt,
            "messages": data.get("messages"),
            "providers": data.get("providers"),
            "mode": data.get("mode"),
            "source": data.get("source"),
        }
        preview = _preview(prompt or data.get("prompt_preview"))
    elif kind == "response":
        content = data.get("content") or data.get("text") or data.get("message")
        payload = {
            "content": content,
            "type": data.get("type"),
            "status": data.get("status"),
            "references": data.get("references"),
        }
        preview = _preview(content)
        if topic == "dialogos.claude_code.stop":
            payload["transcript_path"] = data.get("transcript_path")
    elif kind == "task":
        payload = data
        preview = _preview(str(data.get("desc") or data.get("step") or ""))
    elif kind in {"verification", "qa"}:
        payload = data
        preview = _preview(str(data.get("intent") or data.get("check_id") or ""))

    payload_info = _event_payload(repo, payload) if payload else {}
    event_out = _base_event(kind, topic, actor=actor, source="bus")
    event_out.update(
        {
            "req_id": req_id,
            "run_id": run_id,
            "trace_id": trace_id,
            "session_id": session_id,
            "lane": lane,
            "summary": preview or topic,
        }
    )
    event_out.update(payload_info)
    repo.append_event(event_out)

    if topic == "dialogos.claude_code.stop":
        transcript_path = data.get("transcript_path")
        if isinstance(transcript_path, str) and transcript_path:
            path = Path(transcript_path).expanduser()
            if not path.is_absolute():
                path = (repo.paths.root / path).resolve()
            ingest_transcript(repo, transcript_path=path, session_id=session_id, actor=actor)


def ingest_trace_event(repo: MetaRepo, record: dict) -> None:
    event_type = str(record.get("event_type") or "trace")
    actor = record.get("actor") or "unknown"
    req_id = record.get("req_id")
    session_id = record.get("session_id")

    kind = "event"
    if event_type == "user_prompt":
        kind = "prompt"
    elif event_type in {"assistant_stop", "assistant_output"}:
        kind = "response"

    payload = dict(record)
    preview = _preview(str(record.get("prompt_preview") or record.get("prompt_sha256") or event_type))

    payload_info = _event_payload(repo, payload)
    event_out = _base_event(kind, f"dialogos.trace.{event_type}", actor=actor, source="dialogos_trace")
    event_out.update(
        {
            "req_id": req_id,
            "session_id": session_id,
            "lane": "dialogos",
            "summary": preview or event_type,
        }
    )
    event_out.update(payload_info)
    repo.append_event(event_out)


def ingest_task_entry(repo: MetaRepo, entry: dict) -> None:
    actor = entry.get("actor") or "unknown"
    status = entry.get("status") or "unknown"
    summary = entry.get("meta", {}).get("desc") if isinstance(entry.get("meta"), dict) else None
    payload_info = _event_payload(repo, entry)
    event_out = _base_event("task", "task_ledger.entry", actor=actor, source="task_ledger")
    event_out.update(
        {
            "req_id": entry.get("req_id"),
            "run_id": entry.get("run_id"),
            "summary": _preview(summary or status),
        }
    )
    event_out.update(payload_info)
    repo.append_event(event_out)


def ingest_file(
    repo: MetaRepo,
    cursor: FileCursor,
    tail_bytes: int,
    handler,
    *,
    max_bytes: Optional[int] = None,
    max_line_bytes: Optional[int] = None,
) -> int:
    lines = _read_new_lines(
        cursor,
        tail_bytes,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
    )
    count = 0
    for line in lines:
        record = _safe_json(line)
        if not record:
            continue
        handler(repo, record)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Meta repo ingestion (bus + dialogos + task ledger)")
    ap.add_argument("--bus", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus") + "/events.ndjson")
    ap.add_argument("--dialogos-trace", default=os.environ.get("PLURIBUS_DIALOGOS_TRACE", "/pluribus/.pluribus/dialogos/trace.ndjson"))
    ap.add_argument("--task-ledger", default=str(resolve_paths().root / ".pluribus" / "index" / "task_ledger.ndjson"))
    ap.add_argument("--state", default=str(resolve_paths().state_path))
    ap.add_argument("--once", action="store_true", help="Process available events once, then exit")
    ap.add_argument("--poll", type=float, default=1.0, help="Poll interval (seconds)")
    ap.add_argument("--max-read-bytes", type=int, default=_env_int("META_INGEST_MAX_READ_BYTES", 1_048_576))
    ap.add_argument("--max-line-bytes", type=int, default=_env_int("META_INGEST_MAX_LINE_BYTES", 262_144))
    ap.add_argument("--no-index", action="store_true", help="Disable sqlite index updates")
    return ap.parse_args()


def run_once(repo: MetaRepo, cursors: dict[str, FileCursor], *, max_bytes: int, max_line_bytes: int) -> dict[str, int]:
    counts = {}
    counts["bus"] = ingest_file(
        repo,
        cursors["bus"],
        BUS_TAIL_BYTES,
        ingest_bus_event,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
    )
    counts["dialogos_trace"] = ingest_file(
        repo,
        cursors["dialogos_trace"],
        TRACE_TAIL_BYTES,
        ingest_trace_event,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
    )
    counts["task_ledger"] = ingest_file(
        repo,
        cursors["task_ledger"],
        LEDGER_TAIL_BYTES,
        ingest_task_entry,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
    )
    return counts


def main() -> int:
    args = parse_args()
    state_path = Path(args.state)
    state = _load_state(state_path)
    cursors = {
        "bus": FileCursor.from_state(Path(args.bus), state),
        "dialogos_trace": FileCursor.from_state(Path(args.dialogos_trace), state),
        "task_ledger": FileCursor.from_state(Path(args.task_ledger), state),
    }
    repo = MetaRepo(enable_index=not args.no_index)

    while True:
        run_once(repo, cursors, max_bytes=args.max_read_bytes, max_line_bytes=args.max_line_bytes)
        state = {str(cursor.path): cursor.to_state() for cursor in cursors.values()}
        _save_state(state_path, state)
        if args.once:
            break
        time.sleep(max(0.2, args.poll))

    if repo.index is not None:
        repo.index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
