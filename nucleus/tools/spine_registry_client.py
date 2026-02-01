#!/usr/bin/env python3
"""
spine_registry_client.py - Lightweight reader for Spine registry snapshots.

This provides a minimal, non-invasive integration point so Pluribus services
can consume Spine registry exports without hard dependencies.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


DEFAULT_SNAPSHOT = ".pluribus/spine/spine_registry_snapshot.ndjson"


@dataclass(frozen=True)
class SpineRecord:
    kind: str
    path: str
    payload: object


def resolve_snapshot_path(path: Optional[str] = None) -> Path:
    if path:
        return Path(path)
    env_path = os.environ.get("PLURIBUS_SPINE_REGISTRY_SNAPSHOT")
    if env_path:
        return Path(env_path)
    return Path(DEFAULT_SNAPSHOT)


def iter_spine_records(path: Optional[str] = None) -> Iterator[SpineRecord]:
    snapshot = resolve_snapshot_path(path)
    if not snapshot.exists():
        return iter(())
    with snapshot.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(obj.get("kind", "unknown"))
            record_path = str(obj.get("path", ""))
            payload = obj.get("payload")
            yield SpineRecord(kind=kind, path=record_path, payload=payload)


def load_spine_records(path: Optional[str] = None) -> list[SpineRecord]:
    return list(iter_spine_records(path))


def index_spine_records(path: Optional[str] = None) -> dict[str, list[SpineRecord]]:
    index: dict[str, list[SpineRecord]] = {}
    for rec in iter_spine_records(path):
        index.setdefault(rec.path, []).append(rec)
    return index
