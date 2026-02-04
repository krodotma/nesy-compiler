#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def kg_nodes_path(root: Path) -> Path:
    return root / ".pluribus" / "index" / "kg_nodes.ndjson"


def build_sota_kg_node(
    *,
    sota_item: dict[str, Any],
    ref: str,
    actor: str,
    context: str | None,
    extra_tags: list[str] | None = None,
) -> dict[str, Any]:
    item_id = str(sota_item.get("id") or "")
    title = str(sota_item.get("title") or "")
    url = str(sota_item.get("url") or "")
    tags = [str(t).strip() for t in (sota_item.get("tags") or []) if str(t).strip()]
    tags = ["sota", *tags, f"sota_item_id:{item_id}"]
    if extra_tags:
        tags.extend([str(t).strip() for t in extra_tags if str(t).strip()])
    # Dedupe tags deterministically
    tags = list(dict.fromkeys(tags))

    text = f"SOTA: {title}".strip()
    if url and url not in text:
        text = f"{text} ({url})"

    return {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "node",
        "type": "artifact",
        "text": text,
        "ref": ref,
        "tags": tags,
        "provenance": {"added_by": actor, "context": context},
    }


def append_sota_kg_node(*, root: Path, node: dict[str, Any]) -> None:
    append_ndjson(kg_nodes_path(root), node)

