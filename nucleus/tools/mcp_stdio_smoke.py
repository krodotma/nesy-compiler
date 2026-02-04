#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# Ensure repo root is importable (so `import nucleus...` works when executed by path).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
        from nucleus.mcp.stdio_mcp import write_framed_json, read_framed_json

        req_id = str(uuid.uuid4())
        write_framed_json(self._stdin, {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        while True:
            try:
                obj = read_framed_json(self._stdout)
            except EOFError:
                raise RuntimeError(f"stdio rpc EOF (exit={self.proc.poll()})")
            if obj.get("id") == req_id:
                if "error" in obj:
                    return {"error": obj.get("error")}
                return obj.get("result")


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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="mcp_stdio_smoke.py", description="Socket-free MCP stdio smoke (rhizome/sota/kg).")
    ap.add_argument("--root", default=None, help="Pluribus root (defaults: search upward).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir for evidence emission.")
    ap.add_argument("--emit-bus", action="store_true", help="Emit start/end + per-server results to bus.")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "mcp-stdio-smoke"

    scratch = root / ".pluribus" / "tmp" / "mcp_stdio_smoke" / str(uuid.uuid4())[:8]
    (scratch / ".pluribus" / "index").mkdir(parents=True, exist_ok=True)
    (scratch / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")

    scripts_root = Path(__file__).resolve().parents[1] / "mcp"
    servers = {
        "pluribus-rhizome": scripts_root / "rhizome_server.py",
        "pluribus-sota": scripts_root / "sota_server.py",
        "pluribus-kg": scripts_root / "kg_server.py",
    }

    emit_bus(bus_dir if args.emit_bus else None, topic="mcp.stdio.smoke.start", kind="log", level="info", actor=actor, data={"root": str(root), "scratch": str(scratch), "iso": now_iso_utc()})

    ok = True
    results: dict[str, Any] = {}
    for name, script in servers.items():
        rpc = StdioRpc([sys.executable, str(script), "--root", str(scratch), "--stdio"], env={"PLURIBUS_ROOT": str(scratch)})
        try:
            tools = rpc.request("tools/list", {})
            results[name] = {"tools": [t.get("name") for t in (tools.get("tools") or []) if isinstance(t, dict)]}
        except Exception as e:
            ok = False
            results[name] = {"error": str(e)}
        finally:
            rpc.close()
        emit_bus(bus_dir if args.emit_bus else None, topic="mcp.stdio.smoke.server", kind="metric", level="info" if "error" not in results[name] else "error", actor=actor, data={"server": name, **results[name]})

    emit_bus(
        bus_dir if args.emit_bus else None,
        topic="mcp.stdio.smoke.end",
        kind="metric",
        level="info" if ok else "error",
        actor=actor,
        data={"ok": ok, "scratch": str(scratch), "results": results, "iso": now_iso_utc()},
    )

    if not ok:
        sys.stderr.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        return 2
    sys.stdout.write("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
