#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


async def _wait_http_ready(url: str, *, timeout_s: float = 15.0) -> None:
    import httpx

    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(url)
                if r.status_code in {200, 404}:
                    return
            except Exception:
                await asyncio.sleep(0.1)
        raise TimeoutError(f"server not ready: {url}")

def _extract_text(parts) -> str:
    out: list[str] = []
    for p in parts or []:
        root = getattr(p, "root", None)
        if root is not None and getattr(root, "text", None) is not None:
            out.append(str(root.text))
            continue
        if getattr(p, "text", None) is not None:
            out.append(str(p.text))
            continue
        try:
            dumped = p.model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            if isinstance(dumped.get("root"), dict) and dumped["root"].get("text"):
                out.append(str(dumped["root"]["text"]))
            elif dumped.get("text"):
                out.append(str(dumped["text"]))
    return "\n".join(out).strip()


class TestRealAgentsOfficialAdkA2A(unittest.TestCase):
    def test_adk_to_a2a_roundtrip(self) -> None:
        try:
            import httpx  # noqa: F401
            from a2a.client.base_client import ClientConfig
            from a2a.client.card_resolver import A2ACardResolver
            from a2a.client.client_factory import ClientFactory
            from a2a.types import Message, TextPart
        except Exception as e:  # pragma: no cover
            self.skipTest(f"official A2A SDK not available: {e}")

        repo_root = Path(__file__).resolve().parents[3]
        server_script = repo_root / "nucleus" / "tools" / "adk_a2a_server.py"

        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            (root / ".pluribus" / "index").mkdir(parents=True, exist_ok=True)
            (root / ".pluribus" / "rhizome.json").write_text("{}", encoding="utf-8")

            bus_dir = Path(td) / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").touch()

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PLURIBUS_ACTOR": "test-adk"}
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(server_script),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--provider",
                    "mock",
                    "--root",
                    str(root),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                asyncio.run(_wait_http_ready(f"{base_url}/.well-known/agent-card.json"))

                async def run() -> None:
                    import httpx

                    async with httpx.AsyncClient(timeout=10.0) as http:
                        resolver = A2ACardResolver(http, base_url)
                        card = await resolver.get_agent_card()
                        factory = ClientFactory(ClientConfig(streaming=False, httpx_client=http))
                        client = factory.create(card)
                        msg = Message(message_id="m1", role="user", parts=[TextPart(text="hello")])

                        got_text = ""
                        async for ev in client.send_message(msg):
                            if getattr(ev, "kind", "") == "message":
                                if not got_text:
                                    got_text = _extract_text(getattr(ev, "parts", []))
                            if isinstance(ev, tuple):
                                _task, update = ev
                                status = getattr(_task, "status", None)
                                msg2 = getattr(status, "message", None) if status else None
                                if msg2 and not got_text:
                                    got_text = _extract_text(getattr(msg2, "parts", []))
                                artifacts = getattr(_task, "artifacts", None) or []
                                if artifacts and not got_text:
                                    got_text = _extract_text(getattr(artifacts[0], "parts", []))
                                history = getattr(_task, "history", None) or []
                                if history and not got_text:
                                    got_text = _extract_text(getattr(history[-1], "parts", []))
                                if update and getattr(update, "kind", "") == "artifact-update":
                                    if not got_text:
                                        got_text = _extract_text(getattr(update.artifact, "parts", []))
                                if update and getattr(update, "kind", "") == "status-update":
                                    status = getattr(update, "status", None)
                                    msg2 = getattr(status, "message", None) if status else None
                                    if msg2 and not got_text:
                                        got_text = _extract_text(getattr(msg2, "parts", []))
                                    if getattr(update, "final", False):
                                        break
                        self.assertIn("mock", got_text)

                asyncio.run(run())
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()


if __name__ == "__main__":
    unittest.main()
