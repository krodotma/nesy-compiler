#!/usr/bin/env python3
"""
PBVW Bridge (PluriChat-VNC-WebProxy Bridge)
===========================================
Exposes an OpenAI-compatible API (http://localhost:8080/v1) that routes
requests to the Pluribus Provider Router (CLI/Browser/VNC).

Allows tools like `crush` (mods) to use Pluribus providers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, parse_qs

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore

try:
    from nucleus.tools.pluribus_directive import detect_pluribus_directive  # type: ignore
except Exception:  # pragma: no cover
    detect_pluribus_directive = None  # type: ignore

DEFAULT_ROUTER_PATH = Path(__file__).resolve().parent / "providers" / "router.py"

def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class BridgeConfig:
    host: str
    port: int
    bus_dir: str | None
    actor: str
    router_path: Path
    router_timeout_s: float
    capture_dir: Path | None
    model_alias: dict[str, str]
    api_key: str | None
    policy: dict[str, Any]


def _provider_profile() -> str:
    return (os.environ.get("PLURIBUS_PROVIDER_PROFILE") or "").strip().lower() or "verified"

def _full_profile() -> bool:
    return _provider_profile() in {"full"}

def _verified_profile() -> bool:
    return _provider_profile() in {"verified", "web+vertex", "web-vertex"}

def _web_only_profile() -> bool:
    return _provider_profile() in {"web-only", "web", "plurichat", "plurichat-web", "web_session_only"}

_RESERVED_MODEL_IDS = {
    "auto",
    "chatgpt-web",
    "claude-web",
    "gemini-web",
    "vertex-gemini",
    "vertex-gemini-curl",
    "vllm",
    "vllm-local",
    "ollama",
    "ollama-local",
    "tensorzero",
}


def _parse_model_alias_map(raw: str) -> dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except Exception:
            return {}
        if not isinstance(obj, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, str):
                continue
            kk = k.strip().lower()
            vv = v.strip()
            if kk and vv:
                out[kk] = vv
        return out

    out: dict[str, str] = {}
    for chunk in re.split(r"[,\n]+", text):
        if not chunk or "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        kk = k.strip().lower()
        vv = v.strip()
        if kk and vv:
            out[kk] = vv
    return out


def _load_model_alias_map() -> dict[str, str]:
    raw = (os.environ.get("PLURIBUS_GATEWAY_MODEL_ALIAS") or "").strip()
    path: Path | None = None
    if raw.startswith("@"):
        cand = Path(raw[1:]).expanduser()
        if cand.exists() and cand.is_file():
            path = cand
    elif raw:
        cand = Path(raw).expanduser()
        if cand.exists() and cand.is_file():
            path = cand
    else:
        cand = REPO_ROOT / ".pluribus" / "gateway_model_alias.json"
        if cand.exists() and cand.is_file():
            path = cand

    if path is not None:
        try:
            return _parse_model_alias_map(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}
    return _parse_model_alias_map(raw)


def _apply_model_alias(model: str, *, alias_map: dict[str, str]) -> tuple[str, str, bool]:
    requested = (model or "").strip() or "auto"
    key = requested.lower()
    if key in _RESERVED_MODEL_IDS:
        return requested, requested, False
    target = alias_map.get(key)
    if not target:
        return requested, requested, False
    mapped = str(target).strip()
    if not mapped:
        return requested, requested, False
    return requested, mapped, True


def _parse_gateway_policy(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _load_gateway_policy() -> dict[str, Any]:
    raw = (os.environ.get("PLURIBUS_GATEWAY_POLICY") or "").strip()
    path: Path | None = None
    if raw.startswith("@"):
        cand = Path(raw[1:]).expanduser()
        if cand.exists() and cand.is_file():
            path = cand
    elif raw:
        cand = Path(raw).expanduser()
        if cand.exists() and cand.is_file():
            path = cand
    else:
        cand = REPO_ROOT / ".pluribus" / "gateway_policy.json"
        if cand.exists() and cand.is_file():
            path = cand

    if path is not None:
        try:
            return _parse_gateway_policy(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}
    return _parse_gateway_policy(raw)


def _policy_decide(
    policy: dict[str, Any],
    *,
    prompt: str,
    protocol: str,
    path: str,
    requested_model: str,
    routed_model: str,
) -> tuple[dict[str, Any], dict[str, str], tuple[int, str] | None]:
    policy_id = str(policy.get("policy_id") or "default").strip() or "default"
    max_len = policy.get("max_prompt_len")
    denylist = policy.get("denylist") or []
    require_local = policy.get("require_local_only") or []

    matched: list[dict[str, str]] = []
    extra_env: dict[str, str] = {}

    deny: tuple[int, str] | None = None
    if isinstance(max_len, int) and max_len > 0 and len(prompt) > int(max_len):
        matched.append({"type": "max_prompt_len", "value": str(max_len)})
        deny = (413, f"prompt too large ({len(prompt)} > {int(max_len)})")

    if deny is None and isinstance(denylist, list):
        for pat in denylist:
            if not isinstance(pat, str) or not pat.strip():
                continue
            try:
                if re.search(pat, prompt, flags=re.IGNORECASE):
                    matched.append({"type": "denylist", "pattern": pat})
                    deny = (403, "blocked by gateway policy (denylist)")
                    break
            except re.error:
                continue

    if deny is None and isinstance(require_local, list):
        for pat in require_local:
            if not isinstance(pat, str) or not pat.strip():
                continue
            try:
                if re.search(pat, prompt, flags=re.IGNORECASE):
                    matched.append({"type": "require_local_only", "pattern": pat})
                    extra_env["PLURIBUS_ROUTER_REQUIRE_LOCAL"] = "1"
                    break
            except re.error:
                continue

    action = "deny" if deny else "allow"
    decision = {
        "policy_id": policy_id,
        "action": action,
        "protocol": protocol,
        "path": path,
        "requested_model": requested_model,
        "routed_model": routed_model,
        "matched": matched,
        "require_local_only": bool(extra_env.get("PLURIBUS_ROUTER_REQUIRE_LOCAL")),
    }
    return decision, extra_env, deny


def _resolve_provider(model: str) -> str:
    m = (model or "").strip().lower()
    if not m or m == "auto":
        return "auto"
    # If the caller already supplies a provider ID, honor it (subject to router policy).
    if m in {"chatgpt-web", "claude-web", "gemini-web"}:
        return m
    # Only explicitly map to non-web providers when the operator opted into the full profile.
    if _full_profile():
        if m in {"vllm", "vllm-local"}:
            return "vllm-local"
        if m in {"ollama", "ollama-local"}:
            return "ollama-local"
        if m == "tensorzero":
            return "tensorzero"

    # Heuristic mapping for common model family names.
    if "claude" in m:
        return "claude-web"
    if m.startswith("gemini-3"):
        # gemini-3 is usually a Vertex-only target; let router choose vertex if configured.
        return "auto"
    if "gemini" in m:
        return "gemini-web"
    if "gpt" in m or "openai" in m:
        return "chatgpt-web"

    # Default: allow router to decide (auto can pick vLLM first, then web sessions).
    return "auto"


def _iter_message_text(content: Any) -> Iterable[str]:
    """Extract text from OpenAI-style message content (string or list-of-parts)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
                continue
            if isinstance(part, dict):
                part_type = str(part.get("type") or "")
                if part_type in {"text", "input_text"}:
                    text = part.get("text") or part.get("input_text")
                    if isinstance(text, str) and text:
                        out.append(text)
                    continue
                # For images/audio/etc, preserve a minimal placeholder to keep prompts stable.
                if part_type:
                    out.append(f"[{part_type}]")
        return out
    return [str(content)]


def _build_prompt(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user").strip() or "user"
        content = m.get("content")
        text = " ".join(s.strip() for s in _iter_message_text(content) if str(s).strip()).strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return "\n".join(lines).strip()

def _build_openai_responses_prompt(body: dict) -> str:
    parts: list[str] = []
    system = body.get("instructions") or body.get("system") or body.get("system_prompt")
    if isinstance(system, str) and system.strip():
        parts.append(f"system: {system.strip()}")

    inp = body.get("input")
    if inp is None:
        inp = body.get("messages")

    if isinstance(inp, str):
        if inp.strip():
            parts.append(inp.strip())
        return "\n".join(parts).strip()

    if isinstance(inp, list):
        if all(isinstance(item, dict) and ("role" in item) for item in inp):
            parts.append(_build_prompt(inp))  # type: ignore[arg-type]
            return "\n".join(p for p in parts if p).strip()

        text_bits: list[str] = []
        for item in inp:
            if isinstance(item, str):
                if item.strip():
                    text_bits.append(item.strip())
                continue
            if isinstance(item, dict):
                itype = str(item.get("type") or "")
                if itype == "input_text" and isinstance(item.get("text"), str):
                    text_bits.append(item["text"].strip())
                    continue
                if itype == "message" and isinstance(item.get("content"), list):
                    nested = item.get("content")
                    if isinstance(nested, list):
                        for part in nested:
                            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                                val = part.get("text") or part.get("input_text")
                                if isinstance(val, str) and val.strip():
                                    text_bits.append(val.strip())
                        continue
                if "text" in item and isinstance(item.get("text"), str) and item["text"].strip():
                    text_bits.append(item["text"].strip())
        if text_bits:
            parts.append("\n".join(text_bits).strip())
        return "\n".join(p for p in parts if p).strip()

    if isinstance(inp, dict):
        if inp.get("type") == "input_text" and isinstance(inp.get("text"), str):
            parts.append(inp["text"].strip())
        elif "text" in inp and isinstance(inp.get("text"), str):
            parts.append(inp["text"].strip())
        return "\n".join(p for p in parts if p).strip()

    if inp is not None:
        parts.append(str(inp))
    return "\n".join(p for p in parts if p).strip()

def _iter_anthropic_text(content: Any) -> Iterable[str]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    out.append(block)
                continue
            if isinstance(block, dict):
                btype = str(block.get("type") or "")
                if btype == "text" and isinstance(block.get("text"), str) and block["text"].strip():
                    out.append(block["text"])
                    continue
                if "text" in block and isinstance(block.get("text"), str) and block["text"].strip():
                    out.append(block["text"])
                    continue
                if btype:
                    out.append(f"[{btype}]")
        return out
    return [str(content)]

def _build_anthropic_prompt(body: dict) -> str:
    lines: list[str] = []
    system = body.get("system")
    if system:
        sys_text = " ".join(s.strip() for s in _iter_anthropic_text(system) if str(s).strip()).strip()
        if sys_text:
            lines.append(f"system: {sys_text}")

    messages = body.get("messages") or []
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip() or "user"
            text = " ".join(s.strip() for s in _iter_anthropic_text(msg.get("content")) if str(s).strip()).strip()
            if not text:
                continue
            lines.append(f"{role}: {text}")
    return "\n".join(lines).strip()

def _iter_gemini_parts(parts: Any) -> Iterable[str]:
    if parts is None:
        return []
    if isinstance(parts, dict):
        parts = [parts]
    if isinstance(parts, list):
        out: list[str] = []
        for part in parts:
            if isinstance(part, str):
                if part.strip():
                    out.append(part)
                continue
            if isinstance(part, dict):
                if isinstance(part.get("text"), str) and part["text"].strip():
                    out.append(part["text"])
                    continue
                if "inlineData" in part:
                    out.append("[inlineData]")
                    continue
                if "fileData" in part:
                    out.append("[fileData]")
                    continue
                ptype = str(part.get("type") or "")
                if ptype:
                    out.append(f"[{ptype}]")
        return out
    return [str(parts)]

def _build_gemini_prompt(body: dict) -> str:
    lines: list[str] = []
    system = body.get("systemInstruction")
    if isinstance(system, dict):
        sys_text = " ".join(s.strip() for s in _iter_gemini_parts(system.get("parts")) if str(s).strip()).strip()
        if sys_text:
            lines.append(f"system: {sys_text}")
    elif isinstance(system, str) and system.strip():
        lines.append(f"system: {system.strip()}")

    contents = body.get("contents") or []
    if isinstance(contents, list):
        for item in contents:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip().lower() or "user"
            if role == "model":
                role = "assistant"
            text = " ".join(s.strip() for s in _iter_gemini_parts(item.get("parts")) if str(s).strip()).strip()
            if not text:
                continue
            lines.append(f"{role}: {text}")
    return "\n".join(lines).strip()


def _safe_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()

def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _emit_bus(cfg: BridgeConfig, *, topic: str, kind: str, level: str, data: dict) -> None:
    if agent_bus is None:
        return
    try:
        paths = agent_bus.resolve_bus_paths(cfg.bus_dir)
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=cfg.actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception:
        return

def _maybe_capture(cfg: BridgeConfig, *, req_id: str, label: str, payload: dict) -> None:
    if cfg.capture_dir is None:
        return
    try:
        cfg.capture_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.capture_dir / f"{int(time.time())}_{req_id}_{label}.json"
        path.write_bytes(_json_dumps(payload) + b"\n")
    except Exception:
        return


def call_router(
    prompt: str,
    model: str,
    *,
    cfg: BridgeConfig,
    req_id: str,
    provider: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str, float, str]:
    resolved_provider = provider or _resolve_provider(model)
    cmd = [
        sys.executable,
        str(cfg.router_path),
        "--provider",
        resolved_provider,
        "--prompt",
        prompt,
        "--model",
        model,
        "--format",
        "json",
    ]
    started = time.monotonic()
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(cfg.router_timeout_s),
            env={
                **os.environ,
                "PLURIBUS_BUS_DIR": cfg.bus_dir or "",
                "PLURIBUS_ACTOR": cfg.actor,
                "PLURIBUS_GATEWAY_REQ_ID": req_id,
                "PYTHONDONTWRITEBYTECODE": "1",
                **(extra_env or {}),
            },
        )
        elapsed = time.monotonic() - started
        provider_used = resolved_provider
        out = res.stdout or ""
        err = res.stderr or ""
        try:
            obj = json.loads(out) if out.lstrip().startswith("{") else None
            if isinstance(obj, dict) and ("ok" in obj or "provider" in obj):
                provider_used = str(obj.get("provider") or provider_used)
                text = obj.get("text")
                if not isinstance(text, str):
                    text = obj.get("stdout") if isinstance(obj.get("stdout"), str) else ""
                obj_err = obj.get("stderr")
                if not isinstance(obj_err, str):
                    obj_err = obj.get("error") if isinstance(obj.get("error"), str) else ""
                combined_err = "\n".join([s for s in [obj_err.strip(), err.strip()] if s]).strip()
                return res.returncode, str(text or ""), combined_err, elapsed, provider_used
        except Exception:
            pass
        return res.returncode, out, err, elapsed, provider_used
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        return 124, "", "router timeout", elapsed, resolved_provider

class PBVWHandler(BaseHTTPRequestHandler):
    server_version = "PluribusGateway/0.3"
    protocol_version = "HTTP/1.1"

    _re_gemini_generate = re.compile(r"^/v1beta/models/(?P<model>[^/:]+):(?P<method>generateContent|streamGenerateContent)$")

    def _send_json(self, status: int, payload: dict, *, headers: dict[str, str] | None = None) -> None:
        body = _json_dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            return

    def _send_sse_headers(self, *, headers: dict[str, str] | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # We emit a finite stream (single delta + done). Close the socket so
        # CLI clients that aren't true streaming consumers don't hang.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()

    def _sse_data(self, obj: Any) -> None:
        line = b"data: " + _json_dumps(obj) + b"\n\n"
        self.wfile.write(line)
        try:
            self.wfile.flush()
        except Exception:
            return

    def _sse_event(self, event: str, obj: Any) -> None:
        chunk = b"event: " + event.encode("utf-8") + b"\n" + b"data: " + _json_dumps(obj) + b"\n\n"
        self.wfile.write(chunk)
        try:
            self.wfile.flush()
        except Exception:
            return

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        cfg: BridgeConfig = self.server.cfg  # type: ignore[attr-defined]
        path = urlsplit(self.path).path
        if path not in {"/health", "/v1/health"} and cfg.api_key:
            if not self._authorized(query=parse_qs(urlsplit(self.path).query or "")):
                return
        if path in {"/health", "/v1/health"}:
            self._send_json(200, {"ok": True, "service": "pbvw-bridge", "ts": int(time.time()), "pid": os.getpid()})
            return
        if path in {"/v1/models", "/models"}:
            created = int(time.time())
            models = [
                "auto",
                "chatgpt-web",
                "claude-web",
                "gemini-web",
            ]
            if _verified_profile():
                models += ["vertex-gemini", "vertex-gemini-curl"]
            # List non-web providers only when operator opted into the full profile.
            if _full_profile():
                models += ["vertex-gemini", "vertex-gemini-curl", "vllm-local", "ollama-local", "tensorzero"]
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": mid, "object": "model", "created": created, "owned_by": "pluribus"} for mid in models],
                },
            )
            return
        if path == "/v1beta/models":
            # Minimal Gemini-compatible model list.
            created = now_iso_utc()
            models = [
                {"name": "models/gemini-2.0-flash", "displayName": "gemini-2.0-flash"},
                {"name": "models/gemini-2.5-pro", "displayName": "gemini-2.5-pro"},
                {"name": "models/gemini-3-pro", "displayName": "gemini-3-pro"},
            ]
            self._send_json(200, {"models": models, "nextPageToken": "", "generatedAt": created})
            return
        self.send_error(404)

    def _authorized(self, *, query: dict[str, list[str]] | None) -> bool:
        cfg: BridgeConfig = self.server.cfg  # type: ignore[attr-defined]
        expected = (cfg.api_key or "").strip()
        if not expected:
            return True

        hdrs = {k.lower(): v for k, v in self.headers.items()}

        auth = str(hdrs.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            tok = auth.split(" ", 1)[1].strip()
            if tok == expected:
                return True

        for k in ("x-api-key", "x-goog-api-key", "anthropic-api-key"):
            tok = str(hdrs.get(k) or "").strip()
            if tok and tok == expected:
                return True

        if query:
            tok_list = query.get("key") or []
            if tok_list and str(tok_list[0] or "").strip() == expected:
                return True

        path = urlsplit(self.path).path
        if path.startswith("/v1/messages"):
            self._send_json(401, {"type": "error", "error": {"type": "authentication_error", "message": "unauthorized"}})
            return False
        if path.startswith("/v1beta/"):
            self._send_json(401, {"error": {"message": "unauthorized", "status": "UNAUTHENTICATED"}})
            return False
        self._send_json(401, {"error": {"message": "unauthorized", "type": "authentication_error"}})
        return False

    def do_POST(self):  # noqa: N802
        cfg: BridgeConfig = self.server.cfg  # type: ignore[attr-defined]
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query or "")

        if path not in {"/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens", "/api/event_logging/batch"} and not path.startswith("/v1beta/"):
            self.send_error(404)
            return
        if cfg.api_key:
            if not self._authorized(query=query):
                return

        length = int(self.headers.get("content-length", 0) or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except Exception:
            self._send_json(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}})
            return

        if path == "/api/event_logging/batch":
            # Claude Code "1P event logging" batches are routed through the base URL when using
            # gateway mode. Ack quickly to avoid long shutdown hangs.
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            return

        # Normalize model based on protocol/endpoint.
        requested_model = str(body.get("model") or "auto")
        protocol = "openai.chat_completions"
        if path == "/v1/responses":
            protocol = "openai.responses"
        elif path == "/v1/messages":
            protocol = "anthropic.messages"
        elif path.startswith("/v1beta/"):
            protocol = "google.genai"

        if path == "/v1/messages/count_tokens":
            prompt = _build_anthropic_prompt(body)
            # Very rough heuristic; Claude uses tokenizer-specific counts, but this is adequate
            # for clients that only need a sanity-check value.
            approx_tokens = max(0, int(len(prompt) / 4))
            self._send_json(200, {"input_tokens": approx_tokens}, headers={"X-Served-By": "pluribus-gateway"})
            return

        stream = bool(body.get("stream")) or (self.headers.get("accept") or "").lower().startswith("text/event-stream")

        # Gemini CLI can pass ?key=... with empty JSON body; accept that.
        if protocol == "google.genai":
            m = self._re_gemini_generate.match(path)
            if not m:
                self._send_json(404, {"error": {"message": "unknown Gemini endpoint", "type": "invalid_request_error"}})
                return
            requested_model = str(m.group("model") or requested_model or "gemini-2.0-flash")
            method = str(m.group("method"))
            if method == "streamGenerateContent":
                stream = True

        req_id = str(uuid.uuid4())

        if protocol == "openai.chat_completions":
            messages = body.get("messages") or []
            if not isinstance(messages, list):
                self._send_json(400, {"error": {"message": "messages must be a list", "type": "invalid_request_error"}})
                return
            prompt = _build_prompt(messages)
        elif protocol == "openai.responses":
            prompt = _build_openai_responses_prompt(body)
        elif protocol == "anthropic.messages":
            prompt = _build_anthropic_prompt(body)
        else:
            prompt = _build_gemini_prompt(body)

        prompt_hash = _safe_sha256(prompt)
        requested_model, routed_model, alias_hit = _apply_model_alias(requested_model, alias_map=cfg.model_alias)
        provider_hint = _resolve_provider(routed_model)

        pluribus_directive = None
        if detect_pluribus_directive is not None:
            try:
                pluribus_directive = detect_pluribus_directive(prompt)
            except Exception:
                pluribus_directive = None

        if pluribus_directive is not None:
            route_plan: dict[str, Any] | None = None
            try:
                from nucleus.tools import lens_collimator  # type: ignore

                root = lens_collimator.find_pluribus_root(Path.cwd())
                session = lens_collimator.load_vps_session(root)
                prefer = [provider_hint]
                pref = None
                if isinstance(getattr(pluribus_directive, "params", None), dict):
                    pref = pluribus_directive.params.get("provider")  # type: ignore[attr-defined]
                if isinstance(pref, str) and pref.strip():
                    prefer = [pref.strip()]
                req = lens_collimator.LensRequest(
                    req_id=req_id,
                    goal=getattr(pluribus_directive, "goal", "") or "",
                    kind=getattr(pluribus_directive, "kind", "other"),
                    effects=getattr(pluribus_directive, "effects", "unknown"),
                    prefer_providers=prefer,
                    require_model_prefix=None,
                )
                plan = lens_collimator.plan_route(req, session=session)
                route_plan = {k: v for k, v in plan.__dict__.items() if v is not None}
            except Exception:
                route_plan = None

            _emit_bus(
                cfg,
                topic="pluribus.directive.detected",
                kind="artifact",
                level="info",
                data={
                    "req_id": req_id,
                    "protocol": protocol,
                    "path": path,
                    "model": requested_model,
                    "routed_model": routed_model,
                    "prompt_sha256": prompt_hash,
                    "provider_hint": provider_hint,
                    "directive": pluribus_directive.to_bus_dict(),  # type: ignore[union-attr]
                    "route_plan": route_plan,
                },
            )

        policy_decision, policy_env, deny = _policy_decide(
            cfg.policy,
            prompt=prompt,
            protocol=protocol,
            path=path,
            requested_model=requested_model,
            routed_model=routed_model,
        )
        if policy_decision.get("matched"):
            _emit_bus(
                cfg,
                topic="gateway.policy.decision",
                kind="artifact",
                level="warn" if deny else "info",
                data={"req_id": req_id, **policy_decision},
            )

        _maybe_capture(
            cfg,
            req_id=req_id,
            label="request",
            payload={"protocol": protocol, "path": path, "query": query, "requested_model": requested_model, "routed_model": routed_model, "body": body},
        )

        _emit_bus(
            cfg,
            topic="pbvw.request",
            kind="request",
            level="info",
            data={
                "req_id": req_id,
                "protocol": protocol,
                "path": path,
                "model": requested_model,
                "routed_model": routed_model,
                "alias_hit": bool(alias_hit),
                "prompt_len": len(prompt),
                "prompt_sha256": prompt_hash,
                "provider": provider_hint,
                "stream": bool(stream),
                "policy_id": policy_decision.get("policy_id"),
                "policy_action": policy_decision.get("action"),
                "policy_require_local_only": bool(policy_decision.get("require_local_only")),
                "pluribus": bool(pluribus_directive is not None),
                "pluribus_kind": getattr(pluribus_directive, "kind", None),
                "pluribus_effects": getattr(pluribus_directive, "effects", None),
            },
        )

        if deny:
            http_status, msg = deny
            _emit_bus(
                cfg,
                topic="pbvw.response",
                kind="response",
                level="warn",
                data={
                    "req_id": req_id,
                    "protocol": protocol,
                    "model": requested_model,
                    "routed_model": routed_model,
                    "alias_hit": bool(alias_hit),
                    "provider": "policy",
                    "provider_hint": provider_hint,
                    "ok": False,
                    "http_status": int(http_status),
                    "elapsed_s": 0.0,
                    "stdout_len": 0,
                    "stderr_len": len(msg),
                    "stdout_preview": "",
                    "stderr_preview": msg[:200],
                    "policy_id": policy_decision.get("policy_id"),
                    "policy_action": "deny",
                },
            )
            if protocol == "anthropic.messages":
                self._send_json(int(http_status), {"type": "error", "error": {"type": "permission_error", "message": msg}})
                return
            if protocol == "google.genai":
                self._send_json(int(http_status), {"error": {"message": msg, "status": "FAILED_PRECONDITION"}})
                return
            self._send_json(int(http_status), {"error": {"message": msg, "type": "policy_error"}})
            return

        code, out, err, elapsed_s, provider_used = call_router(
            prompt,
            routed_model,
            cfg=cfg,
            req_id=req_id,
            provider=provider_hint,
            extra_env=policy_env or None,
        )
        ok = code == 0

        _emit_bus(
            cfg,
            topic="pbvw.response",
            kind="response",
            level="info" if ok else "warn",
            data={
                "req_id": req_id,
                "protocol": protocol,
                "model": requested_model,
                "routed_model": routed_model,
                "alias_hit": bool(alias_hit),
                "provider": provider_used,
                "provider_hint": provider_hint,
                "ok": ok,
                "exit_code": int(code),
                "elapsed_s": round(float(elapsed_s), 3),
                "stdout_len": len(out or ""),
                "stderr_len": len(err or ""),
                "stdout_preview": (out or "")[:200],
                "stderr_preview": (err or "")[:200],
                "policy_id": policy_decision.get("policy_id"),
                "policy_action": policy_decision.get("action"),
                "policy_require_local_only": bool(policy_decision.get("require_local_only")),
            },
        )

        _maybe_capture(cfg, req_id=req_id, label="response_meta", payload={"ok": ok, "exit_code": int(code), "stdout": out, "stderr": err})

        if not ok:
            msg = (err or out or "router error").strip()[:800]
            if protocol == "anthropic.messages":
                self._send_json(502, {"type": "error", "error": {"type": "provider_error", "message": msg}})
                return
            if protocol == "google.genai":
                self._send_json(502, {"error": {"message": msg, "status": "FAILED_PRECONDITION"}})
                return
            self._send_json(500, {"error": {"message": msg, "type": "provider_error", "code": code}})
            return

        response_text = (out or "").strip()

        headers = {
            "X-Request-Id": req_id,
            "X-Served-By-Provider": provider_used,
            "X-Requested-Model": requested_model,
            "X-Served-By-Model": routed_model,
            "X-Model-Alias-Hit": "1" if alias_hit else "0",
            "X-Route-Policy": str(policy_decision.get("policy_id") or "default"),
            "X-Route-Action": str(policy_decision.get("action") or "allow"),
        }

        if protocol == "openai.chat_completions":
            if stream:
                self._send_sse_headers(headers=headers)
                chunk = {
                    "id": f"chatcmpl-{int(time.time())}-{req_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {"content": response_text}, "finish_reason": "stop"}],
                }
                self._sse_data(chunk)
                self.wfile.write(b"data: [DONE]\n\n")
                return
            resp = {
                "id": f"chatcmpl-{int(time.time())}-{req_id[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            }
            self._send_json(200, resp, headers=headers)
            return

        if protocol == "openai.responses":
            resp_id = f"resp-{int(time.time())}-{req_id[:8]}"
            msg_id = f"msg-{int(time.time())}-{req_id[:8]}"
            resp = {
                "id": resp_id,
                "object": "response",
                "created": int(time.time()),
                "model": requested_model,
                "output": [
                    {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": response_text}],
                    }
                ],
            }
            if stream:
                self._send_sse_headers(headers=headers)
                self._sse_data(
                    {"type": "response.created", "response": {"id": resp_id, "object": "response", "created": resp["created"], "model": requested_model}}
                )
                self._sse_data({"type": "response.output_text.delta", "delta": response_text})
                self._sse_data({"type": "response.completed", "response": resp})
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self._send_json(200, resp, headers=headers)
            return

        if protocol == "anthropic.messages":
            msg_id = f"msg_{int(time.time())}_{req_id[:8]}"
            resp = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": requested_model,
                "content": [{"type": "text", "text": response_text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
            if stream:
                self._send_sse_headers(headers=headers)
                self._sse_event(
                    "message_start",
                    {"type": "message_start", "message": {"id": msg_id, "type": "message", "role": "assistant", "model": requested_model}},
                )
                self._sse_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                self._sse_event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": response_text}})
                self._sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                self._sse_event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 0}})
                self._sse_event("message_stop", {"type": "message_stop"})
                return
            self._send_json(200, resp, headers=headers)
            return

        # Gemini generateContent
        candidate = {
            "index": 0,
            "content": {"role": "model", "parts": [{"text": response_text}]},
            "finishReason": "STOP",
        }
        resp = {"candidates": [candidate]}
        if stream:
            self._send_sse_headers(headers=headers)
            self._sse_data(resp)
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._send_json(200, resp, headers=headers)
        return

def main():
    p = argparse.ArgumentParser(prog="pbvw_bridge.py", description="PBVW bridge (OpenAI-compatible) for Pluribus router.")
    p.add_argument("--host", default=os.environ.get("PBVW_HOST") or "0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PBVW_PORT") or "8080"))
    p.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus")
    p.add_argument("--actor", default=os.environ.get("PLURIBUS_ACTOR") or "pbvw-bridge")
    p.add_argument("--router-path", default=str(DEFAULT_ROUTER_PATH))
    p.add_argument("--router-timeout-s", type=float, default=float(os.environ.get("PBVW_ROUTER_TIMEOUT_S") or "90"))
    p.add_argument(
        "--api-key",
        default=os.environ.get("PLURIBUS_GATEWAY_API_KEY") or "",
        help="Optional shared key to require for all requests (Authorization: Bearer, x-api-key, or ?key=... for Gemini).",
    )
    p.add_argument(
        "--capture-dir",
        default=os.environ.get("PLURIBUS_GATEWAY_CAPTURE_DIR") or "",
        help="Optional directory to store raw request/response payloads (sensitive).",
    )
    args = p.parse_args()

    cfg = BridgeConfig(
        host=str(args.host),
        port=int(args.port),
        bus_dir=str(args.bus_dir) if args.bus_dir else None,
        actor=str(args.actor),
        router_path=Path(str(args.router_path)).expanduser().resolve(),
        router_timeout_s=float(args.router_timeout_s),
        capture_dir=Path(str(args.capture_dir)).expanduser().resolve() if str(args.capture_dir).strip() else None,
        model_alias=_load_model_alias_map(),
        api_key=str(args.api_key).strip() or None,
        policy=_load_gateway_policy(),
    )

    server = ThreadingHTTPServer((cfg.host, cfg.port), PBVWHandler)
    server.cfg = cfg  # type: ignore[attr-defined]
    print(f"PBVW Bridge running on {cfg.host}:{cfg.port} (router={cfg.router_path})")
    server.serve_forever()

if __name__ == "__main__":
    main()
