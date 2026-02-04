#!/usr/bin/env python3
"""
vlm_solver.py — Local-only vision challenge solver (Phase 1 MVP)

This tool is intended to be used via the browser daemon's external solver hook:

  export PLURIBUS_SOLVER_CMD='python3 /pluribus/nucleus/tools/vlm_solver.py --provider {provider} --html {html} --screenshot {screenshot}'

Design goals:
  - LOCAL ONLY by default (no remote APIs); safe to run unattended.
  - Emit bus evidence (best-effort).
  - Return structured JSON for the daemon to apply (click/type/wait/etc).
  - Do NOT attempt to bypass CAPTCHAs; detect and escalate instead.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _read_text(path: str) -> str:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _read_image_b64(path: str, *, max_bytes: int) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    raw = p.read_bytes()
    if len(raw) > max_bytes:
        return ""
    return base64.b64encode(raw).decode("ascii")


def _looks_like_captcha(text: str) -> bool:
    t = (text or "").lower()
    patterns = (
        "recaptcha",
        "g-recaptcha",
        "hcaptcha",
        "cf-challenge",
        "cloudflare",
        "turnstile",
        "captcha",
        "i am not a robot",
    )
    return any(p in t for p in patterns)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output."""
    s = (text or "").strip()
    if not s:
        return None
    # Fast path: exact JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Heuristic: take substring from first '{' to last '}'.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _emit_bus(topic: str, *, kind: str = "log", level: str = "info", actor: str = "vlm-solver", data: dict) -> None:
    """Best-effort bus emission without importing heavy deps."""
    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_path = bus_dir / "events.ndjson"
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    payload = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": _now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    try:
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        return


ActionType = Literal["click", "fill", "type", "press", "wait", "goto"]


@dataclass(frozen=True)
class SolverDecision:
    action: Literal["actions", "answer", "noop", "escalate", "need_user_code"]
    reason: str
    confidence: float
    backend: str
    answer: str | None = None
    actions: list[dict[str, Any]] | None = None


def _heuristic_decide(*, provider: str, html: str) -> SolverDecision:
    if _looks_like_captcha(html):
        return SolverDecision(
            action="escalate",
            reason="captcha_detected",
            confidence=1.0,
            backend="heuristic",
        )
    # No actionable inference without a model.
    return SolverDecision(
        action="noop",
        reason="no_local_vlm_available",
        confidence=0.5,
        backend="heuristic",
    )


def _ollama_available(host: str, *, timeout_s: float) -> bool:
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_analyze(
    *,
    host: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout_s: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [image_b64] if image_b64 else [],
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
        },
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    try:
        obj = json.loads(body)
        if isinstance(obj, dict) and isinstance(obj.get("response"), str):
            return obj["response"]
    except Exception:
        pass
    return body


def decide(
    *,
    provider: str,
    screenshot_path: str,
    html_path: str | None,
    reason: str | None,
    backend: str,
    model: str,
    timeout_s: float,
    max_image_bytes: int,
) -> SolverDecision:
    html = _read_text(html_path or "")
    if _looks_like_captcha(html):
        return SolverDecision(action="escalate", reason="captcha_detected", confidence=1.0, backend="heuristic")

    if backend == "none":
        return _heuristic_decide(provider=provider, html=html)

    if backend == "auto":
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if _ollama_available(host, timeout_s=min(1.5, timeout_s)):
            backend = "ollama"
        elif _qwen_available():
            backend = "qwen"
        else:
            backend = "heuristic"

    if backend == "qwen":
        p = Path(screenshot_path)
        if not p.exists() or not p.is_file():
            return SolverDecision(action="noop", reason="screenshot_missing", confidence=0.9, backend="qwen")
        qwen_prompt = (
            "You are a vision-based UI automation helper.\n"
            "If CAPTCHA present, respond: {\"action\":\"escalate\",\"reason\":\"captcha_detected\",\"confidence\":1.0}\n"
            "If 2FA/OTP needed, respond: {\"action\":\"need_user_code\",\"reason\":\"otp_required\",\"confidence\":0.9}\n"
            "Otherwise propose next steps. Output JSON with: action, reason, confidence (0-1), optionally actions[] or answer.\n"
            f"provider={provider}\nreason={reason or ''}\n"
        )
        qwen_model = os.environ.get("PLURIBUS_QWEN_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
        try:
            out = _qwen_analyze(model_id=qwen_model, prompt=qwen_prompt, image_path=screenshot_path, timeout_s=timeout_s)
        except Exception as e:
            return SolverDecision(action="noop", reason=f"qwen_error:{type(e).__name__}", confidence=0.2, backend="qwen")
        obj = _extract_json_object(out)
        if not obj:
            return SolverDecision(action="noop", reason="qwen_non_json_output", confidence=0.2, backend="qwen")
        act = str(obj.get("action") or "").strip().lower()
        reason_out = str(obj.get("reason") or "").strip() or "model_output"
        try:
            conf_f = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
        except Exception:
            conf_f = 0.5
        if act in {"escalate", "need_user_code"}:
            return SolverDecision(action=act, reason=reason_out, confidence=conf_f, backend="qwen")
        if act == "answer":
            ans = str(obj.get("answer") or "").strip()
            return SolverDecision(action="answer" if ans else "noop", reason=reason_out, confidence=conf_f, backend="qwen", answer=ans or None)
        if act == "actions":
            actions = obj.get("actions")
            if isinstance(actions, list) and actions:
                return SolverDecision(action="actions", reason=reason_out, confidence=conf_f, backend="qwen", actions=actions)
        return SolverDecision(action="noop", reason=f"unsupported:{act}", confidence=conf_f, backend="qwen")

    if backend == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        image_b64 = _read_image_b64(screenshot_path, max_bytes=max_image_bytes)
        if not image_b64:
            return SolverDecision(action="noop", reason="screenshot_missing_or_too_large", confidence=0.9, backend="ollama")

        sys_prompt = (
            "You are a vision-based UI automation helper for a login/auth challenge.\n"
            "Rules:\n"
            "- Do NOT solve CAPTCHAs or attempt to bypass bot protections.\n"
            "- If a CAPTCHA/bot check is present, respond with {\"action\":\"escalate\",\"reason\":\"captcha_detected\"}.\n"
            "- If a 2FA/OTP code is required, respond with {\"action\":\"need_user_code\",\"reason\":\"otp_required\",\"selectors\":[...]}.\n"
            "- Otherwise, propose safe next steps (click/type/wait) to proceed.\n"
            "Output MUST be a single JSON object.\n"
            "Allowed action types in actions[]: click, fill, type, press, wait, goto.\n"
        )
        user_prompt = (
            f"{sys_prompt}\n"
            f"provider={provider}\n"
            f"reason={reason or ''}\n"
            "Return JSON with keys: action, reason, confidence (0-1), and optionally actions[] or answer.\n"
        )
        try:
            out = _ollama_analyze(host=host, model=model, prompt=user_prompt, image_b64=image_b64, timeout_s=timeout_s)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return SolverDecision(action="noop", reason=f"ollama_unavailable:{type(e).__name__}", confidence=0.2, backend="ollama")
        except Exception as e:
            return SolverDecision(action="noop", reason=f"ollama_error:{type(e).__name__}", confidence=0.2, backend="ollama")

        obj = _extract_json_object(out)
        if not obj:
            return SolverDecision(action="noop", reason="ollama_non_json_output", confidence=0.2, backend="ollama")

        act = str(obj.get("action") or "").strip().lower()
        reason_out = str(obj.get("reason") or "").strip() or "model_output"
        conf = obj.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else 0.5
        except Exception:
            conf_f = 0.5
        conf_f = max(0.0, min(1.0, conf_f))

        if act in {"escalate", "need_user_code"}:
            return SolverDecision(action=act, reason=reason_out, confidence=conf_f, backend="ollama")

        if act == "answer":
            ans = str(obj.get("answer") or "").strip()
            if ans:
                return SolverDecision(action="answer", reason=reason_out, confidence=conf_f, backend="ollama", answer=ans)
            return SolverDecision(action="noop", reason="model_answer_empty", confidence=conf_f, backend="ollama")

        if act == "actions":
            actions = obj.get("actions")
            if isinstance(actions, list) and actions:
                return SolverDecision(action="actions", reason=reason_out, confidence=conf_f, backend="ollama", actions=actions)  # type: ignore[arg-type]
            return SolverDecision(action="noop", reason="model_actions_empty", confidence=conf_f, backend="ollama")

        # Unknown or unsupported action
        return SolverDecision(action="noop", reason=f"unsupported_model_action:{act}", confidence=conf_f, backend="ollama")

    # heuristic fallback
    return _heuristic_decide(provider=provider, html=html)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Local-only vision challenge solver (VNC/Playwright helper).")
    parser.add_argument("--provider", required=True, help="Provider id (gemini-web, claude-web, chatgpt-web)")
    parser.add_argument("--screenshot", required=True, help="Path to screenshot PNG")
    parser.add_argument("--html", default="", help="Path to HTML dump (optional)")
    parser.add_argument("--reason", default="", help="Challenge reason hint (optional)")
    parser.add_argument("--backend", default=os.environ.get("PLURIBUS_VLM_BACKEND", "auto"), help="auto|ollama|qwen|heuristic|none")
    parser.add_argument("--model", default=os.environ.get("PLURIBUS_VLM_MODEL", "llava"), help="VLM model id (backend-specific)")
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("PLURIBUS_VLM_TIMEOUT_S", "25")), help="Backend timeout seconds")
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=int(os.environ.get("PLURIBUS_VLM_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))),
        help="Reject screenshots larger than this many bytes",
    )
    parser.add_argument("--no-bus", action="store_true", help="Disable bus emission")
    args = parser.parse_args(argv)

    backend = str(args.backend or "auto").strip().lower()
    if backend not in {"auto", "ollama", "qwen", "heuristic", "none"}:
        backend = "auto"

    decision = decide(
        provider=args.provider,
        screenshot_path=args.screenshot,
        html_path=args.html or None,
        reason=(args.reason or None),
        backend=backend,
        model=str(args.model or "llava"),
        timeout_s=max(1.0, float(args.timeout_s)),
        max_image_bytes=max(1_000_000, int(args.max_image_bytes)),
    )

    out: dict[str, Any] = {
        "schema_version": 1,
        "provider": args.provider,
        "backend": decision.backend,
        "action": decision.action,
        "reason": decision.reason,
        "confidence": decision.confidence,
    }
    if decision.answer:
        out["answer"] = decision.answer
    if decision.actions:
        out["actions"] = decision.actions

    if not args.no_bus:
        _emit_bus(
            "browser.vlm_solver.decision",
            kind="artifact",
            actor=os.environ.get("PLURIBUS_ACTOR", "vlm-solver"),
            data={
                "provider": args.provider,
                "backend": decision.backend,
                "action": decision.action,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "screenshot": str(args.screenshot),
                "html": str(args.html),
            },
        )

    # Print *only* JSON on stdout (the daemon captures stdout as the solver output).
    sys.stdout.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
