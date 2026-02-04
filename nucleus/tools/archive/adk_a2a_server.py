#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools.venv_exec import maybe_reexec_in_realagents_venv  # noqa: E402

maybe_reexec_in_realagents_venv()

import uvicorn  # noqa: E402
from google.adk.a2a.utils.agent_to_a2a import to_a2a  # noqa: E402
from google.adk.agents.llm_agent import LlmAgent  # noqa: E402
from google.adk.tools.mcp_tool import McpToolset  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from nucleus.tools import adk_pluribus_llm as _adk_pluribus_llm  # noqa: F401,E402


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="adk_a2a_server.py", description="Serve an official ADK agent as an official A2A server.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8142)
    ap.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    ap.add_argument("--provider", default="mock", help="PluriChat provider suffix for model pluribus/<provider> (default: mock).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir to pass into spawned MCP servers.")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    mcp_dir = REPO_ROOT / "nucleus" / "mcp"

    def server_params(script: str) -> StdioServerParameters:
        env = {"PLURIBUS_ROOT": str(root)}
        if bus_dir:
            env["PLURIBUS_BUS_DIR"] = bus_dir
        return StdioServerParameters(command=sys.executable, args=[str(mcp_dir / script), "--root", str(root), "--transport", "stdio"], env=env)

    toolsets = [
        McpToolset(connection_params=server_params("rhizome_fastmcp.py"), tool_name_prefix="rhizome_"),
        McpToolset(connection_params=server_params("sota_fastmcp.py"), tool_name_prefix="sota_"),
        McpToolset(connection_params=server_params("kg_fastmcp.py"), tool_name_prefix="kg_"),
    ]

    agent = LlmAgent(
        name="pluribus_adk",
        description="Pluribus ADK agent (LLM routes through PluriChat; tools via MCP).",
        model=f"pluribus/{str(args.provider)}",
        instruction="You are Pluribus. Use MCP tools when they help. Keep outputs concise and evidence-oriented.",
        tools=toolsets,
    )

    app = to_a2a(agent, host=str(args.host), port=int(args.port), protocol="http")
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
