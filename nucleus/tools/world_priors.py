#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


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


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            kind,
            "--level",
            level,
            "--actor",
            actor,
            "--data",
            json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def resolve_root(raw_root: str | None) -> Path:
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    return find_rhizome_root(Path.cwd()) or Path.cwd().resolve()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return h.hexdigest()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cmd_build(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    idx = root / ".pluribus" / "index"
    kg_nodes = idx / "kg_nodes.ndjson"
    kg_edges = idx / "kg_edges.ndjson"
    curation = idx / "curation.ndjson"
    artifacts = idx / "artifacts.ndjson"

    # Minimal “equilibrium” snapshot: counts + hashed inputs so downstream selectors can treat it as a stable prior.
    node_count = sum(1 for _ in iter_ndjson(kg_nodes))
    edge_count = sum(1 for _ in iter_ndjson(kg_edges))
    curation_count = sum(1 for _ in iter_ndjson(curation))
    artifact_count = sum(1 for _ in iter_ndjson(artifacts))

    snapshot = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "world_priors_snapshot",
        "root": str(root),
        "counts": {
            "kg_nodes": node_count,
            "kg_edges": edge_count,
            "curation": curation_count,
            "artifacts": artifact_count,
        },
        "input_hashes": {
            "kg_nodes": hash_file(kg_nodes),
            "kg_edges": hash_file(kg_edges),
            "curation": hash_file(curation),
            "artifacts": hash_file(artifacts),
        },
        "constraints": {"gates": ["P", "E", "L", "R", "Q"], "sextet": True, "auom_boundary": True},
        "provenance": {"built_by": actor},
    }

    out = idx / "world_priors.ndjson"
    append_ndjson(out, snapshot)
    if args.emit_bus:
        emit_bus(bus_dir, topic="world_priors.built", kind="artifact", level="info", actor=actor, data=snapshot)
    sys.stdout.write(snapshot["id"] + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="world_priors.py", description="Build equilibrium world-prior snapshots from STRp indices (append-only).")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Append a new world-priors snapshot to .pluribus/index/world_priors.ndjson.")
    b.add_argument("--emit-bus", action="store_true")
    b.set_defaults(func=cmd_build)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

