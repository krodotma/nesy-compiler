from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class StdioRpc:
    def __init__(self, argv: list[str]):
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
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

    def request(self, method: str, params: dict) -> dict:
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


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / ".pluribus" / "index").mkdir(parents=True)
    (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")
    return root


def test_mcp_rhizome_stdio_roundtrip(tmp_path: Path):
    root = make_root(tmp_path)
    sample = root / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    script = Path(__file__).resolve().parents[2] / "mcp" / "rhizome_server.py"
    rpc = StdioRpc([sys.executable, str(script), "--root", str(root), "--stdio"])
    try:
        init = rpc.request("initialize", {"protocolVersion": "1.0", "clientInfo": {"name": "pytest", "version": "0"}})
        assert "serverInfo" in init
        res = rpc.request("tools/call", {"name": "ingest", "arguments": {"path": str(sample)}})
        assert "sha256" in res
        lst = rpc.request("tools/call", {"name": "list_artifacts", "arguments": {"limit": 10}})
        assert lst["count"] >= 1
    finally:
        rpc.close()


def test_mcp_sota_stdio_roundtrip(tmp_path: Path):
    root = make_root(tmp_path)
    script = Path(__file__).resolve().parents[2] / "mcp" / "sota_server.py"
    rpc = StdioRpc([sys.executable, str(script), "--root", str(root), "--stdio"])
    try:
        init = rpc.request("initialize", {"protocolVersion": "1.0", "clientInfo": {"name": "pytest", "version": "0"}})
        assert "serverInfo" in init
        added = rpc.request(
            "tools/call",
            {"name": "add_item", "arguments": {"name": "x", "url": "https://example.invalid/x", "category": "test"}},
        )
        assert added["added"] == "x"
        lst = rpc.request("tools/call", {"name": "list_items", "arguments": {"limit": 10}})
        assert lst["count"] >= 1
        assert any(i["name"] == "x" for i in lst["items"])
    finally:
        rpc.close()


def test_mcp_kg_stdio_roundtrip(tmp_path: Path):
    root = make_root(tmp_path)
    script = Path(__file__).resolve().parents[2] / "mcp" / "kg_server.py"
    rpc = StdioRpc([sys.executable, str(script), "--root", str(root), "--stdio"])
    try:
        init = rpc.request("initialize", {"protocolVersion": "1.0", "clientInfo": {"name": "pytest", "version": "0"}})
        assert "serverInfo" in init
        a = rpc.request("tools/call", {"name": "add_node", "arguments": {"label": "A", "tags": ["t"]}})
        b = rpc.request("tools/call", {"name": "add_node", "arguments": {"label": "B", "tags": ["t"]}})
        assert "id" in a and "id" in b
        e = rpc.request(
            "tools/call",
            {"name": "add_edge", "arguments": {"source_id": a["id"], "target_id": b["id"], "relation": "rel"}},
        )
        assert "id" in e
        q = rpc.request("tools/call", {"name": "query", "arguments": {"tag": "t", "limit": 10}})
        assert q["count"]["nodes"] >= 2
    finally:
        rpc.close()
