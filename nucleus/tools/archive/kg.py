#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

# DEPRECATION NOTICE: 
# kg.py is legacy as of DKIN Protocol v26 (Epistemic Sovereignty).
# Please transition to 'nucleus/tools/graphiti_bridge.py' for temporal facts 
# and verifiable provenance chains.

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


def paths_for_root(root: Path) -> tuple[Path, Path]:
    idx = root / ".pluribus" / "index"
    return idx / "kg_nodes.ndjson", idx / "kg_edges.ndjson"


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    nodes, edges = paths_for_root(root)
    ensure_dir(nodes.parent)
    nodes.touch(exist_ok=True)
    edges.touch(exist_ok=True)
    sys.stdout.write(f"{nodes}\n{edges}\n")
    return 0


def cmd_add_node(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    nodes, _ = paths_for_root(root)
    actor = default_actor()
    node_id = str(uuid.uuid4())
    tags = [t for t in (args.tag or []) if t.strip()]
    node = {
        "id": node_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "node",
        "type": args.type,
        "text": args.text,
        "ref": args.ref,
        "tags": tags,
        "provenance": {"added_by": actor, "context": args.context},
    }
    append_ndjson(nodes, node)
    emit_bus(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR"), topic="kg.node.added", kind="artifact", level="info", actor=actor, data=node)
    sys.stdout.write(node_id + "\n")
    return 0


def cmd_add_edge(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    _, edges = paths_for_root(root)
    actor = default_actor()
    edge_id = str(uuid.uuid4())
    tags = [t for t in (args.tag or []) if t.strip()]
    edge = {
        "id": edge_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "edge",
        "src": args.src,
        "rel": args.rel,
        "dst": args.dst,
        "tags": tags,
        "provenance": {"added_by": actor, "context": args.context},
    }
    append_ndjson(edges, edge)
    emit_bus(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR"), topic="kg.edge.added", kind="artifact", level="info", actor=actor, data=edge)
    sys.stdout.write(edge_id + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kg.py", description="Append-only knowledge graph (nodes/edges) for STRp rhizomes.")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create the KG files under .pluribus/index/")
    init.set_defaults(func=cmd_init)

    n = sub.add_parser("add-node", help="Append one node.")
    n.add_argument("--type", required=True, help="claim|gap|hypothesis|falsifier|entity|artifact|note|other")
    n.add_argument("--text", required=True)
    n.add_argument("--ref", default=None, help="Optional pointer (curation id, artifact sha/id, URL, etc.)")
    n.add_argument("--tag", action="append", default=[])
    n.add_argument("--context", default=None)
    n.set_defaults(func=cmd_add_node)

    e = sub.add_parser("add-edge", help="Append one edge.")
    e.add_argument("src", help="Source node id")
    e.add_argument("rel", help="supports|contradicts|refines|depends_on|implements|tests|other")
    e.add_argument("dst", help="Destination node id")
    e.add_argument("--tag", action="append", default=[])
    e.add_argument("--context", default=None)
    e.set_defaults(func=cmd_add_edge)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

