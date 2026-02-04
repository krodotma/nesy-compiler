#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools.venv_exec import maybe_reexec_in_realagents_venv  # noqa: E402

maybe_reexec_in_realagents_venv()

from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.models.registry import LLMRegistry  # noqa: E402
from google.genai import types  # noqa: E402

from agent_bus import default_actor, emit_event, resolve_bus_paths  # type: ignore # noqa: E402


Json = dict[str, Any]


def _bus_paths() -> Any | None:
    bus_dir = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if not bus_dir:
        return None
    return resolve_bus_paths(bus_dir)


def _prompt_from_contents(contents: list[types.Content]) -> str:
    parts: list[str] = []
    for c in contents:
        if not c or not getattr(c, "parts", None):
            continue
        for p in c.parts or []:
            if getattr(p, "text", None):
                parts.append(str(p.text))
    return "\n".join(parts).strip()


def _provider_from_model(model: str) -> str:
    if model.startswith("pluribus/") and len(model.split("/", 1)) == 2:
        suffix = model.split("/", 1)[1].strip()
        return suffix or "auto"
    return "auto"


def _plurichat_oneshot(prompt: str, *, provider: str, actor: str) -> Json:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "nucleus" / "tools" / "plurichat.py"),
        "--mode",
        "oneshot",
        "--provider",
        provider,
        "--ask",
        prompt,
        "--json-output",
    ]
    bus_dir = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if bus_dir:
        cmd.extend(["--bus-dir", bus_dir])
    if actor:
        cmd.extend(["--actor", actor])
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {"success": False, "error": proc.stderr.strip() or f"plurichat exit={proc.returncode}"}
    try:
        out = json.loads(proc.stdout.strip() or "{}")
        if isinstance(out, dict):
            return out
    except Exception:
        pass
    return {"success": True, "text": proc.stdout.strip()}


class PluribusLlm(BaseLlm):
    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"^pluribus/.*$"]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        actor = os.environ.get("PLURIBUS_ACTOR") or default_actor()
        paths = _bus_paths()
        req_id = str(uuid.uuid4())

        provider = _provider_from_model(self.model or llm_request.model or "pluribus/auto")
        prompt = _prompt_from_contents(llm_request.contents or [])

        if paths:
            emit_event(
                paths,
                topic="adk.pluribus.llm.call",
                kind="request",
                level="info",
                actor=actor,
                data={"req_id": req_id, "provider": provider, "model": self.model, "prompt_chars": len(prompt)},
                trace_id=None,
                run_id=None,
                durable=False,
            )

        started = time.time()
        out = _plurichat_oneshot(prompt, provider=provider, actor=actor)
        latency_ms = (time.time() - started) * 1000.0
        ok = bool(out.get("success")) and not out.get("error")
        text = str(out.get("text") or "")

        if paths:
            emit_event(
                paths,
                topic="adk.pluribus.llm.response",
                kind="response",
                level="info" if ok else "error",
                actor=actor,
                data={"req_id": req_id, "ok": ok, "provider": provider, "latency_ms": latency_ms, "error": out.get("error")},
                trace_id=None,
                run_id=None,
                durable=False,
            )

        content = types.Content(role="model", parts=[types.Part(text=text)])
        yield LlmResponse(
            model_version=self.model,
            content=content,
            partial=False,
            turn_complete=True,
            error_code=None if ok else "pluribus_error",
            error_message=None if ok else str(out.get("error") or "unknown error"),
        )


LLMRegistry.register(PluribusLlm)

