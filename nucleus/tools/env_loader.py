#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"')):
        v = v[1:-1]
    return v


def _strip_inline_comment(value: str) -> str:
    s = value
    in_single = False
    in_double = False
    escaped = False
    for i, ch in enumerate(s):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            return s[:i].rstrip()
    return s.rstrip()


def _load_kv_file(path: Path, *, override: bool) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _unquote(_strip_inline_comment(value.strip()))
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value


def load_pluribus_env(*, override: bool = False) -> None:
    """
    Best-effort env bootstrap for CLI tools.

    Loads (in order):
      1) ~/.config/nucleus/secrets.env (shell-style: export KEY=VALUE)
      2) <repo>/nucleus/.env (dotenv-style: KEY=VALUE)

    Never prints secrets. Only sets vars that are missing unless override=True.
    """

    home = Path(os.environ.get("HOME") or "~").expanduser()
    secrets_env = home / ".config" / "nucleus" / "secrets.env"
    _load_kv_file(secrets_env, override=override)

    tools_dir = Path(__file__).resolve().parent
    root = tools_dir.parent
    dotenv = root / ".env"
    _load_kv_file(dotenv, override=override)

    # Compatibility: different Gemini tooling expects different env var names.
    gemini = (os.environ.get("GEMINI_API_KEY") or "").strip()
    google = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if gemini and not google:
        os.environ["GOOGLE_API_KEY"] = gemini
    elif google and not gemini:
        os.environ["GEMINI_API_KEY"] = google

