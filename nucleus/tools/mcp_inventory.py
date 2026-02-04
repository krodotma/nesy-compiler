#!/usr/bin/env python3
from __future__ import annotations

"""
MCP Inventory — derive an append-only MCP capability inventory from Pluribus-native servers.

Outputs:
- Append-only file: <root>/.pluribus/mcp/inventory.ndjson
- Optional bus artifact: topic `mcp.inventory`

Notes:
- This intentionally treats the inventory as a *derived index* from server tool lists.
- It is safe-by-default: emits only server names + tool names + coarse effects categories.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure repo root is importable (so `import nucleus...` works when executed by path).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.mcp.stdio_mcp import write_framed_json, read_framed_json

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


class StdioRpc:
    def __init__(self, argv: list[str], *, env: dict[str, str] | None = None):
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            env={**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self._stdin = self.proc.stdin
        self._stdout = self.proc.stdout

    def close(self) -> None:
        try:
            self._stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=1)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def request(self, method: str, params: dict[str, Any]) -> Any:
        req_id = str(uuid.uuid4())
        write_framed_json(self._stdin, {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        while True:
            obj = read_framed_json(self._stdout)
            if obj.get("id") == req_id:
                if "error" in obj:
                    return {"error": obj.get("error")}
                return obj.get("result")


TOOL_EFFECTS: dict[str, list[str]] = {
    # Pluribus rhizome
    "ingest": ["W(file)"],
    "list_artifacts": ["R(file)"],
    "show_artifact": ["R(file)"],
    # Pluribus KG
    "add_node": ["W(file)"],
    "add_edge": ["W(file)"],
    "query": ["R(file)"],
    # Pluribus SOTA
    "add_item": ["W(file)"],
    "list_items": ["R(file)"],
    "tick": ["W(file)"],
}


def append_ndjson(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mcp_inventory.py", description="Derive MCP capability inventory (append-only).")
    ap.add_argument("--root", default=None, help="Pluribus root (defaults: search upward).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir for evidence emission (defaults: $PLURIBUS_BUS_DIR).")
    ap.add_argument("--emit-bus", action="store_true", help="Emit mcp.inventory artifact to bus.")
    ap.add_argument("--out", default=None, help="Inventory ndjson path (default: <root>/.pluribus/mcp/inventory.ndjson).")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "mcp-inventory"

    out_path = Path(args.out).expanduser().resolve() if args.out else (root / ".pluribus" / "mcp" / "inventory.ndjson")

    servers = {
        "pluribus-rhizome": ("nucleus/mcp/rhizome_server.py", ["ingest", "list_artifacts", "show_artifact"]),
        "pluribus-sota": ("nucleus/mcp/sota_server.py", ["add_item", "list_items", "tick"]),
        "pluribus-kg": ("nucleus/mcp/kg_server.py", ["add_node", "add_edge", "query"]),
    }

    inv: dict[str, Any] = {
        "id": "mcp-inventory-" + str(uuid.uuid4())[:8],
        "ts": time.time(),
        "iso": now_iso_utc(),
        "protocol_version": "1.0",
        "root": str(root),
        "servers": [],
    }

    ok = True
    for name, (script, expected_tools) in servers.items():
        rpc = StdioRpc([sys.executable, str(root / script), "--root", str(root), "--stdio"], env={"PLURIBUS_ROOT": str(root)})
        try:
            res = rpc.request("tools/list", {})
            tools = [t for t in (res.get("tools") or []) if isinstance(t, dict)]
            tool_names = [str(t.get("name") or "") for t in tools if t.get("name")]
            inv["servers"].append(
                {
                    "name": name,
                    "entry_point": script,
                    "tools": [
                        {
                            "name": tn,
                            "effects": TOOL_EFFECTS.get(tn, ["R(file)", "W(file)"]),
                            "expected": tn in set(expected_tools),
                        }
                        for tn in tool_names
                    ],
                }
            )
            if set(expected_tools) - set(tool_names):
                ok = False
        except Exception as e:
            ok = False
            inv["servers"].append({"name": name, "entry_point": script, "error": str(e), "tools": []})
        finally:
            rpc.close()

    inv["ok"] = ok
    append_ndjson(out_path, inv)
    emit_bus(bus_dir if args.emit_bus else None, topic="mcp.inventory", kind="artifact", level="info" if ok else "error", actor=actor, data={"path": str(out_path), "ok": ok, "server_count": len(inv["servers"]), "iso": inv["iso"]})

    sys.stdout.write(str(out_path) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
