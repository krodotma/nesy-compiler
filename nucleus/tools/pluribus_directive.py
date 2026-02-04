#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def _normalize_kind(v: str | None) -> str:
    vv = (v or "").strip().lower()
    if vv in {"distill", "apply", "verify", "audit", "benchmark"}:
        return vv
    return "other"


def _normalize_effects(v: str | None) -> str:
    vv = (v or "").strip().lower()
    if vv in {"none", "file", "network", "unknown"}:
        return vv
    return "unknown"


def _parse_params(raw: str) -> dict[str, str]:
    """
    Parse key/value params from a "(k=v, k2=v2)" string (without the outer parens).
    This is intentionally conservative: only simple tokens are accepted.
    """
    out: dict[str, str] = {}
    text = (raw or "").strip()
    if not text:
        return out
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            k, v = chunk.split("=", 1)
        elif ":" in chunk:
            k, v = chunk.split(":", 1)
        else:
            continue
        key = k.strip().lower()
        val = v.strip()
        if not key or not val:
            continue
        if not re.fullmatch(r"[a-z0-9_.-]{1,64}", key):
            continue
        if not re.fullmatch(r"[a-zA-Z0-9_./:-]{1,512}", val):
            continue
        out[key] = val
    return out


@dataclass(frozen=True)
class PluribusDirective:
    """
    Parsed PLURIBUS directive extracted from an arbitrary string.

    Note: `goal` is intentionally kept in-memory only; bus-safe serialization
    should not include the full goal text by default.
    """

    raw: str
    form: str  # "prefix" | "inline" | "json"
    params: dict[str, str] = field(default_factory=dict)
    goal: str = ""
    kind: str = "other"
    effects: str = "unknown"
    role: str | None = None

    @property
    def raw_sha256(self) -> str:
        return _sha256(self.raw)

    @property
    def goal_sha256(self) -> str:
        return _sha256(self.goal)

    @property
    def goal_len(self) -> int:
        return len(self.goal or "")

    def to_bus_dict(self, *, include_goal_preview: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "form": self.form,
            "role": self.role,
            "kind": self.kind,
            "effects": self.effects,
            "params": dict(self.params or {}),
            "raw_sha256": self.raw_sha256,
            "goal_sha256": self.goal_sha256,
            "goal_len": self.goal_len,
        }
        if include_goal_preview:
            d["goal_preview"] = (self.goal or "")[:160]
        return {k: v for k, v in d.items() if v is not None}


_ROLE_PREFIX_RE = re.compile(r"(?i)^\s*(system|user|assistant)\s*:\s*")


def detect_pluribus_directive(text: str) -> PluribusDirective | None:
    """
    Detect a PLURIBUS directive in free-form text.

    Detection precedence:
      1) Line-prefixed directives: "user: PLURIBUS(...): goal"
      2) Inline directives: "... PLURIBUS(...): goal"

    Returns the first match found.
    """

    raw_text = text or ""
    if not raw_text.strip():
        return None

    # 1) Prefix form: start-of-line (optionally after "user:" etc).
    for line in raw_text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue

        role: str | None = None
        m_role = _ROLE_PREFIX_RE.match(line)
        rest = line
        if m_role:
            role = str(m_role.group(1) or "").strip().lower() or None
            rest = line[m_role.end() :]

        if not re.match(r"(?i)^\s*pluribus\b", rest):
            continue

        # Slice off the token itself.
        m_tok = re.match(r"(?i)^\s*pluribus\b", rest)
        assert m_tok is not None
        after = rest[m_tok.end() :].strip()

        params: dict[str, str] = {}
        form = "prefix"
        goal = ""
        raw = line.strip()

        # Structured JSON envelope (single-line support).
        if after.startswith("{") and after.endswith("}"):
            try:
                obj = json.loads(after)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                params = {str(k).lower(): str(v) for k, v in obj.items() if isinstance(k, str)}
                goal = str(obj.get("goal") or obj.get("prompt") or obj.get("task") or "").strip()
                form = "json"
        else:
            # Optional "(k=v,...)" params.
            if after.startswith("(") and ")" in after:
                close = after.find(")")
                params = _parse_params(after[1:close])
                after = after[close + 1 :].strip()

            if after.startswith(":"):
                after = after[1:].strip()
            goal = after.strip()

        kind = _normalize_kind(params.get("kind"))
        effects = _normalize_effects(params.get("effects"))
        return PluribusDirective(raw=raw, form=form, params=params, goal=goal, kind=kind, effects=effects, role=role)

    # 2) Inline form: first occurrence of the token.
    m = re.search(r"(?i)\bpluribus\b", raw_text)
    if not m:
        return None
    tail = raw_text[m.end() :].strip()
    params: dict[str, str] = {}
    form = "inline"
    goal = ""

    if tail.startswith("(") and ")" in tail:
        close = tail.find(")")
        params = _parse_params(tail[1:close])
        tail = tail[close + 1 :].strip()
    if tail.startswith(":"):
        tail = tail[1:].strip()
    goal = tail.strip()

    kind = _normalize_kind(params.get("kind"))
    effects = _normalize_effects(params.get("effects"))
    return PluribusDirective(raw=raw_text[max(0, m.start() - 40) : min(len(raw_text), m.end() + 200)], form=form, params=params, goal=goal, kind=kind, effects=effects)

