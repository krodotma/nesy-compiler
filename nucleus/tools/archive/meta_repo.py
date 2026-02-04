#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-posix fallback
    fcntl = None

sys.dont_write_bytecode = True


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_root() -> Path:
    root = os.environ.get("PLURIBUS_META_ROOT") or os.environ.get("PLURIBUS_ROOT") or "/pluribus"
    return Path(root).expanduser().resolve()


@dataclass
class MetaPaths:
    root: Path
    meta_dir: Path
    objects_dir: Path
    ledger_path: Path
    index_path: Path
    state_path: Path


def resolve_paths(root: Optional[Path] = None) -> MetaPaths:
    root_path = root or resolve_root()
    meta_dir = root_path / ".pluribus" / "meta"
    objects_dir = meta_dir / "objects"
    ledger_path = meta_dir / "events.ndjson"
    index_path = root_path / ".pluribus" / "index" / "meta.sqlite3"
    state_path = root_path / ".pluribus" / "index" / "meta_ingest_state.json"
    return MetaPaths(
        root=root_path,
        meta_dir=meta_dir,
        objects_dir=objects_dir,
        ledger_path=ledger_path,
        index_path=index_path,
        state_path=state_path,
    )


def _lock_file(handle) -> None:
    if fcntl is None:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    if fcntl is None:
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class MetaIndex:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        cache_kb = _env_int("META_INDEX_CACHE_KB", 0)
        if cache_kb > 0:
            self.conn.execute(f"PRAGMA cache_size=-{cache_kb};")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta_events (
                id TEXT PRIMARY KEY,
                ts REAL,
                iso TEXT,
                kind TEXT,
                topic TEXT,
                actor TEXT,
                req_id TEXT,
                run_id TEXT,
                trace_id TEXT,
                session_id TEXT,
                lane TEXT,
                tags TEXT,
                source TEXT,
                summary TEXT,
                payload_ref TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_meta_events_req ON meta_events(req_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_meta_events_kind ON meta_events(kind)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_meta_events_topic ON meta_events(topic)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_meta_events_payload ON meta_events(payload_ref)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta_edges (
                id TEXT PRIMARY KEY,
                ts REAL,
                src TEXT,
                rel TEXT,
                dst TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_meta_edges_src ON meta_edges(src)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_meta_edges_dst ON meta_edges(dst)"
        )
        self.conn.commit()

    def add_event(self, event: dict) -> None:
        tags = event.get("tags")
        if isinstance(tags, list):
            tags = json.dumps(tags, ensure_ascii=False, separators=(",", ":"))
        self.conn.execute(
            """
            INSERT OR IGNORE INTO meta_events (
                id, ts, iso, kind, topic, actor, req_id, run_id, trace_id,
                session_id, lane, tags, source, summary, payload_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.get("id"),
                event.get("ts"),
                event.get("iso"),
                event.get("kind"),
                event.get("topic"),
                event.get("actor"),
                event.get("req_id"),
                event.get("run_id"),
                event.get("trace_id"),
                event.get("session_id"),
                event.get("lane"),
                tags,
                event.get("source"),
                event.get("summary"),
                event.get("payload_ref"),
            ),
        )

        if event.get("kind") == "response" and event.get("req_id"):
            prompt_id = self._find_latest_prompt(event["req_id"])
            if prompt_id:
                edge = {
                    "id": str(uuid.uuid4()),
                    "ts": event.get("ts") or time.time(),
                    "src": prompt_id,
                    "rel": "responds_to",
                    "dst": event.get("id"),
                }
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO meta_edges (id, ts, src, rel, dst)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (edge["id"], edge["ts"], edge["src"], edge["rel"], edge["dst"]),
                )
        self.conn.commit()

    def has_payload_ref(self, payload_ref: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM meta_events WHERE payload_ref = ? LIMIT 1",
            (payload_ref,),
        ).fetchone()
        return bool(row)

    def _find_latest_prompt(self, req_id: str) -> Optional[str]:
        row = self.conn.execute(
            """
            SELECT id FROM meta_events
            WHERE req_id = ? AND kind = 'prompt'
            ORDER BY ts DESC LIMIT 1
            """,
            (req_id,),
        ).fetchone()
        return row[0] if row else None

    def unanswered_prompts(self, since_ts: float) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT id, req_id, summary, ts FROM meta_events p
            WHERE p.kind = 'prompt' AND p.ts >= ?
              AND NOT EXISTS (
                SELECT 1 FROM meta_events r
                WHERE r.req_id = p.req_id AND r.kind = 'response'
              )
            ORDER BY p.ts DESC
            """,
            (since_ts,),
        ).fetchall()
        return [
            {"id": row[0], "req_id": row[1], "summary": row[2], "ts": row[3]}
            for row in rows
        ]

    def close(self) -> None:
        self.conn.close()


class MetaRepo:
    def __init__(self, root: Optional[Path] = None, *, enable_index: bool = True):
        self.paths = resolve_paths(root)
        self.paths.meta_dir.mkdir(parents=True, exist_ok=True)
        self.paths.objects_dir.mkdir(parents=True, exist_ok=True)
        self.paths.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = MetaIndex(self.paths.index_path) if enable_index else None

    def store_object(self, payload: Any, *, content_type: Optional[str] = None) -> dict:
        if payload is None:
            return {}
        ext = "txt"
        if isinstance(payload, (dict, list)):
            data = _json_bytes(payload)
            ext = "json"
            content_type = content_type or "application/json"
        elif isinstance(payload, bytes):
            data = payload
            ext = "bin"
            content_type = content_type or "application/octet-stream"
        else:
            data = str(payload).encode("utf-8", errors="replace")
            content_type = content_type or "text/plain"

        digest = hashlib.sha256(data).hexdigest()
        subdir = self.paths.objects_dir / digest[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        obj_path = subdir / f"{digest}.{ext}"
        if not obj_path.exists():
            tmp_path = obj_path.with_suffix(f".{ext}.tmp")
            with tmp_path.open("wb") as handle:
                handle.write(data)
            tmp_path.replace(obj_path)

        return {
            "ref": f"sha256:{digest}",
            "bytes": len(data),
            "ext": ext,
            "content_type": content_type,
            "path": str(obj_path),
        }

    def store_file(self, path: Path, *, content_type: Optional[str] = None) -> dict:
        if not path.exists():
            return {}
        suffix = path.suffix.lstrip(".").lower()
        ext = suffix or "bin"
        if content_type is None:
            if ext in {"json", "ndjson"}:
                content_type = "application/json"
            elif ext in {"txt", "md", "log"}:
                content_type = "text/plain"
            else:
                content_type = "application/octet-stream"

        hasher = hashlib.sha256()
        tmp_path = self.paths.objects_dir / f"tmp-{uuid.uuid4().hex}.{ext}"
        total = 0
        with path.open("rb") as src, tmp_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                dst.write(chunk)
                total += len(chunk)

        digest = hasher.hexdigest()
        subdir = self.paths.objects_dir / digest[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        obj_path = subdir / f"{digest}.{ext}"
        if obj_path.exists():
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(obj_path)

        return {
            "ref": f"sha256:{digest}",
            "bytes": total,
            "ext": ext,
            "content_type": content_type,
            "path": str(obj_path),
        }

    def append_event(self, event: dict) -> str:
        event.setdefault("id", str(uuid.uuid4()))
        event.setdefault("ts", time.time())
        event.setdefault("iso", now_iso_utc())
        event.setdefault("actor", os.environ.get("PLURIBUS_ACTOR", "unknown"))
        event.setdefault("kind", "event")
        event.setdefault("source", "meta")

        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.paths.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.ledger_path.open("a", encoding="utf-8") as handle:
            _lock_file(handle)
            try:
                handle.write(line)
            finally:
                _unlock_file(handle)

        if self.index is not None:
            self.index.add_event(event)
        return event["id"]
