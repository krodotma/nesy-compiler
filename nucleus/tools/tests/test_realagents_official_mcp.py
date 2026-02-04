#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path


class TestRealAgentsOfficialMcp(unittest.TestCase):
    def test_fastmcp_stdio_list_and_call(self) -> None:
        try:
            import anyio
            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except Exception as e:  # pragma: no cover
            self.skipTest(f"official MCP SDK not available: {e}")

        repo_root = Path(__file__).resolve().parents[3]
        server_script = repo_root / "nucleus" / "mcp" / "kg_fastmcp.py"

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".pluribus").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")
            (root / ".pluribus" / "index").mkdir(parents=True, exist_ok=True)

            async def run() -> None:
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[str(server_script), "--root", str(root), "--transport", "stdio"],
                    env={"PLURIBUS_ROOT": str(root)},
                )
                async with stdio_client(params) as (read_stream, write_stream):
                    session = ClientSession(read_stream, write_stream, read_timeout_seconds=None)
                    async with session:
                        await session.initialize()
                        tools = await session.list_tools()
                        tool_names = [t.name for t in tools.tools]
                        self.assertIn("add_node", tool_names)

                        res = await session.call_tool("add_node", {"label": "hello", "tags": ["t1"]})
                        dumped = res.model_dump()
                        self.assertTrue(dumped.get("content"))

            anyio.run(run)


if __name__ == "__main__":
    unittest.main()
