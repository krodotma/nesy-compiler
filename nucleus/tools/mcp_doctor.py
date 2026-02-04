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
    mcp_dir = root / ".pluribus" / "mcp"
    inv = mcp_dir / "inventory.ndjson"

    sys.stdout.write("nucleus mcp doctor\n\n")
    sys.stdout.write(f"- rhizome_root: {root}\n")
    sys.stdout.write(f"- mcp_dir: {mcp_dir} ({'present' if mcp_dir.exists() else 'missing'})\n")
    sys.stdout.write(f"- inventory: {inv} ({'present' if inv.exists() else 'missing'})\n")
    sys.stdout.write("\nNotes:\n")
    sys.stdout.write("- MCP enablement is host-specific (Cursor/Claude Desktop/custom).\n")
    sys.stdout.write("- Conventions: docs/workflows/mcp.md\n")
    sys.stdout.write(f"- PLURIBUS_BUS_DIR: {os.environ.get('PLURIBUS_BUS_DIR') or 'unset'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

