#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def level_to_severity(level: str | None) -> int:
    level = (level or "").lower()
    if level in ("fatal", "critical"):
        return 5
    if level in ("error",):
        return 4
    if level in ("warn", "warning"):
        return 3
    if level in ("info",):
        return 1
    return 0


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint_event(topic: str, actor: str, data: Any) -> str:
    payload = f"{topic}\n{actor}\n{_stable_json(data)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_entropy_score(data: Any) -> float:
    if data is None:
        return 0.0
    if isinstance(data, (dict, list)):
        raw = _stable_json(data)
    else:
        raw = str(data)
    length = len(raw)
    if length == 0:
        return 0.0
    unique_ratio = len(set(raw)) / float(length)
    score = (length / 2048.0) + (unique_ratio * 0.3)
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return round(score, 4)


def normalize_event(raw: Dict[str, Any], source: str = "bus") -> Dict[str, Any]:
    ts = raw.get("ts") or time.time()
    iso = raw.get("iso") or now_iso_utc()
    topic = raw.get("topic") or "unknown"
    actor = raw.get("actor") or "unknown"
    kind = raw.get("kind") or "log"
    level = raw.get("level") or "info"
    data = raw.get("data")
    fingerprint = fingerprint_event(topic, actor, data)
    severity = level_to_severity(level)
    entropy = compute_entropy_score(data)
    lineage = {
        "req_id": raw.get("req_id"),
        "trace_id": raw.get("trace_id"),
        "run_id": raw.get("run_id"),
        "parent_id": raw.get("parent_id"),
    }
    raw_ref = {
        "bus_id": raw.get("id"),
        "path": raw.get("path"),
        "offset": raw.get("offset"),
    }
    return {
        "id": raw.get("id") or fingerprint,
        "ts": ts,
        "iso": iso,
        "source": source,
        "actor": actor,
        "topic": topic,
        "kind": kind,
        "level": level,
        "severity": severity,
        "entropy_score": entropy,
        "fingerprint": fingerprint,
        "lineage": lineage,
        "raw_ref": raw_ref,
        "metrics": {},
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class QAIrStore:
    base_dir: str
    filename: str = "qa_ir.ndjson"
    max_bytes: int = 500 * 1024 * 1024

    @property
    def path(self) -> str:
        return os.path.join(self.base_dir, self.filename)

    def append(self, event: Dict[str, Any]) -> None:
        _ensure_dir(self.base_dir)
        self._rotate_if_needed()
        line = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    def _rotate_if_needed(self) -> None:
        if not os.path.exists(self.path):
            return
        size = os.path.getsize(self.path)
        if size < self.max_bytes:
            return
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        rotated = f"{self.path}.{stamp}.bak"
        os.rename(self.path, rotated)
