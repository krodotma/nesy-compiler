#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def main() -> int:
    root = find_rhizome_root(Path.cwd()) or Path.cwd().resolve()
    idx = root / ".pluribus" / "index"

    def present(p: Path) -> str:
        return "present" if p.exists() else "missing"

    sys.stdout.write("pluribus doctor\n\n")
    sys.stdout.write(f"- rhizome_root: {root}\n")
    sys.stdout.write(f"- bus_dir (PLURIBUS_BUS_DIR): {os.environ.get('PLURIBUS_BUS_DIR') or 'unset'}\n\n")

    sys.stdout.write("Core STRp:\n")
    sys.stdout.write(f"- manifest: {present(root / '.pluribus' / 'rhizome.json')}\n")
    sys.stdout.write(f"- artifacts: {present(idx / 'artifacts.ndjson')}\n")
    sys.stdout.write(f"- requests: {present(idx / 'requests.ndjson')}\n\n")

    sys.stdout.write("Memory:\n")
    sys.stdout.write(f"- RAG DB: {present(idx / 'rag.sqlite3')}\n")
    sys.stdout.write(f"- KG nodes: {present(idx / 'kg_nodes.ndjson')}\n")
    sys.stdout.write(f"- KG edges: {present(idx / 'kg_edges.ndjson')}\n")
    sys.stdout.write(f"- world priors: {present(idx / 'world_priors.ndjson')}\n\n")

    sys.stdout.write("SOTA/CMP:\n")
    sys.stdout.write(f"- sota stream: {present(idx / 'sota.ndjson')}\n")
    sys.stdout.write(f"- cmp_large dir: {present(idx / 'cmp_large')}\n\n")

    sys.stdout.write("Ops:\n")
    sys.stdout.write(f"- mounts config: {present(root / '.pluribus' / 'mounts.json')}\n")
    sys.stdout.write(f"- domains registry: {present(idx / 'domains.ndjson')}\n")
    sys.stdout.write(f"- mcp inventory: {present(root / '.pluribus' / 'mcp' / 'inventory.ndjson')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

