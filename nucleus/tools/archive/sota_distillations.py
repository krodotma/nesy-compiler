#!/usr/bin/env python3
from __future__ import annotations

import json
import time
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


def distillations_root(root: Path) -> Path:
    return root / ".pluribus" / "index" / "distillations" / "sota"


def artifact_path_for(*, root: Path, sota_item_id: str, req_id: str) -> Path:
    return distillations_root(root) / sota_item_id / f"{req_id}.md"


def materialize_sota_distillation(
    *,
    root: Path,
    sota_item_id: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    """Persist a STRp distillation response as an append-only artifact and return an artifact record.

    This function is idempotent: the artifact path is deterministic (sota_item_id/req_id).
    """
    req_id = str(response.get("req_id") or "")
    if not req_id:
        raise ValueError("response missing req_id")

    out = response.get("output")
    content = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=2)
    stderr = response.get("stderr")
    if isinstance(stderr, str) and stderr.strip():
        content = f"{content}\n\n---\n\n## stderr\n\n```\n{stderr.strip()}\n```\n"

    path = artifact_path_for(root=root, sota_item_id=sota_item_id, req_id=req_id)
    ensure_dir(path.parent)
    if not path.exists():
        header = [
            f"# SOTA distillation",
            f"",
            f"- `sota_item_id`: `{sota_item_id}`",
            f"- `req_id`: `{req_id}`",
            f"- `provider`: `{response.get('provider')}`",
            f"- `model`: `{response.get('model')}`",
            f"- `exit_code`: `{response.get('exit_code')}`",
            f"- `created_iso`: `{response.get('iso')}`",
            f"",
            "---",
            "",
        ]
        path.write_text("\n".join(header) + content, encoding="utf-8")

    record = {
        "id": f"sota_distill::{sota_item_id}::{req_id}",
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "artifact",
        "type": "sota_distillation",
        "sota_item_id": sota_item_id,
        "req_id": req_id,
        "path": str(path),
        "provider": response.get("provider"),
        "model": response.get("model"),
        "exit_code": response.get("exit_code"),
        "provenance": {"source_kind": str(response.get("kind") or ""), "source_id": response.get("id")},
    }
    artifacts_path = root / ".pluribus" / "index" / "artifacts.ndjson"
    append_ndjson(artifacts_path, record)
    return record


def snippet_from_markdown(path: Path, *, max_chars: int = 320) -> str:
    try:
        s = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"

