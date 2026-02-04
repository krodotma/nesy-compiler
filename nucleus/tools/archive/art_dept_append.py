#!/usr/bin/env python3
"""
Append-only writer for the Pluribus Art Dept indexes.

- Sources:   nucleus/art_dept/sources/sources.ndjson
- Artifacts: nucleus/art_dept/artifacts/genomes.ndjson

Contract:
- never edits existing lines
- uses an exclusive file lock to avoid race conditions
- emits a bus artifact event for provenance
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import fcntl
import subprocess


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _root_from_env_or_cwd() -> Path:
    if os.environ.get("PLURIBUS_ROOT"):
        return Path(os.environ["PLURIBUS_ROOT"]).resolve()
    return Path.cwd().resolve()


def _default_sources_path(root: Path) -> Path:
    return root / "nucleus" / "art_dept" / "sources" / "sources.ndjson"


def _default_artifacts_path(root: Path) -> Path:
    return root / "nucleus" / "art_dept" / "artifacts" / "genomes.ndjson"


def _append_ndjson(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _emit_bus(topic: str, data: Dict[str, Any], actor: Optional[str] = None) -> None:
    try:
        tools_dir = Path(__file__).resolve().parent
        bus = tools_dir / "agent_bus.py"
        payload = json.dumps(data, ensure_ascii=False)
        env = os.environ.copy()
        if actor:
            env["PLURIBUS_ACTOR"] = actor
        bus_dir = env.get("PLURIBUS_BUS_DIR")
        subprocess.run(
            [
                "python3",
                str(bus),
                "pub",
                *(
                    ["--bus-dir", bus_dir]
                    if bus_dir
                    else []
                ),
                "--topic",
                topic,
                "--kind",
                "artifact",
                "--data",
                payload,
            ],
            env=env,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # Avoid breaking append path; bus is best-effort.
        pass


@dataclass(frozen=True)
class Common:
    id: str
    ts: float
    iso: str


def _common(entry_id: Optional[str]) -> Common:
    ts = time.time()
    return Common(
        id=entry_id or str(uuid.uuid4()),
        ts=ts,
        iso=_iso_now(),
    )


def add_source(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    path = Path(args.sources_path).resolve() if args.sources_path else _default_sources_path(root)
    c = _common(args.id)
    obj: Dict[str, Any] = {
        "id": c.id,
        "ts": c.ts,
        "iso": c.iso,
        "kind": args.kind,
        "title": args.title,
        "url": args.url or "",
        "license": args.license or "",
        "tags": args.tags or [],
        "notes": args.notes or "",
        "ingest": {"status": args.status},
    }
    _append_ndjson(path, obj)
    _emit_bus("art_dept.source.added", {"path": str(path), "source": {k: obj[k] for k in ("id", "kind", "title", "url")}})
    print(c.id)
    return 0


def add_artifact(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    path = Path(args.artifacts_path).resolve() if args.artifacts_path else _default_artifacts_path(root)
    c = _common(args.id)
    obj: Dict[str, Any] = {
        "id": c.id,
        "ts": c.ts,
        "iso": c.iso,
        "type": args.type,
        "engine": args.engine,
        "name": args.name,
        "source_refs": args.source_refs or [],
        "code_path": args.code_path or "",
        "bus_bindings": json.loads(args.bus_bindings) if args.bus_bindings else {},
        "notes": args.notes or "",
    }
    _append_ndjson(path, obj)
    _emit_bus("art_dept.artifact.added", {"path": str(path), "artifact": {k: obj[k] for k in ("id", "type", "engine", "name")}})
    print(c.id)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="art_dept_append.py", description="Append-only writer for Art Dept indexes.")
    p.add_argument("--root", default=str(_root_from_env_or_cwd()), help="Repo root (default: $PLURIBUS_ROOT or CWD)")
    p.add_argument("--sources-path", default=None, help="Override sources.ndjson path")
    p.add_argument("--artifacts-path", default=None, help="Override genomes.ndjson path")

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("source", help="Append a source record")
    s.add_argument("--id", default=None)
    s.add_argument("--kind", required=True, choices=["reference", "shadertoy", "threejs", "glsl", "paper", "prompt"])
    s.add_argument("--title", required=True)
    s.add_argument("--url", default=None)
    s.add_argument("--license", default=None)
    s.add_argument("--status", default="unfetched", choices=["unfetched", "reviewed", "imported", "implemented"])
    s.add_argument("--tags", nargs="*", default=[])
    s.add_argument("--notes", default=None)
    s.set_defaults(func=add_source)

    a = sub.add_parser("artifact", help="Append an artifact/genome record")
    a.add_argument("--id", default=None)
    a.add_argument("--type", required=True, choices=["shader", "palette", "scenelet", "layout"])
    a.add_argument("--engine", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--source-refs", nargs="*", default=[])
    a.add_argument("--code-path", default=None)
    a.add_argument("--bus-bindings", default=None, help="JSON string for bus bindings")
    a.add_argument("--notes", default=None)
    a.set_defaults(func=add_artifact)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
