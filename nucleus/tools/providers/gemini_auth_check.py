#!/usr/bin/env python3
"""Gemini CLI auth check.

Previously this only checked for config files and could claim "available" even
when Gemini CLI would fail at runtime (quota/auth errors). We do a bounded CLI
invocation to avoid false positives in the dashboard.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import subprocess
import time
import selectors
import os as _os
from pathlib import Path

from gemini_oauth_prompt import extract_google_oauth_url, looks_like_oauth_prompt


def _run_streaming(cmd: list[str], *, env: dict, timeout_s: float) -> tuple[int, str, str, bool]:
    """
    Run a subprocess with incremental stdout/stderr capture, returning (rc, stdout, stderr, saw_oauth).

    Critical: Gemini CLI can print the OAuth URL then block on "Enter the authorization code:".
    We must detect that prompt early (without waiting for the full timeout) so operators can
    complete auth without a "hung" health check.
    """
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        return 1, "", "failed to start gemini CLI\n", False

    sel = selectors.DefaultSelector()
    if p.stdout:
        sel.register(p.stdout, selectors.EVENT_READ)
    if p.stderr:
        sel.register(p.stderr, selectors.EVENT_READ)

    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    combined_tail = ""
    saw_oauth = False
    deadline = time.time() + float(timeout_s)

    def _append(buf: list[bytes], chunk: bytes) -> None:
        if not chunk:
            return
        buf.append(chunk)

    while time.time() < deadline:
        if p.poll() is not None and not sel.get_map():
            break

        events = sel.select(timeout=0.2)
        if not events:
            if p.poll() is not None:
                break
            continue

        for key, _ in events:
            try:
                chunk = _os.read(key.fileobj.fileno(), 4096)
            except Exception:
                try:
                    sel.unregister(key.fileobj)
                except Exception:
                    pass
                continue

            if not chunk:
                try:
                    sel.unregister(key.fileobj)
                except Exception:
                    pass
                continue

            if key.fileobj is p.stdout:
                _append(out_chunks, chunk)
            else:
                _append(err_chunks, chunk)

            try:
                combined_tail = (combined_tail + chunk.decode("utf-8", errors="replace"))[-20000:]
            except Exception:
                combined_tail = combined_tail[-20000:]

            if looks_like_oauth_prompt(combined_tail):
                saw_oauth = True
                try:
                    p.kill()
                except Exception:
                    pass
                break

        if saw_oauth:
            break

    if p.poll() is None and not saw_oauth:
        try:
            p.kill()
        except Exception:
            pass
        return 124, b"".join(out_chunks).decode("utf-8", errors="replace"), b"".join(err_chunks).decode("utf-8", errors="replace"), False

    rc = int(p.poll() or 0)
    stdout = b"".join(out_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(err_chunks).decode("utf-8", errors="replace")
    return rc, stdout, stderr, saw_oauth


def main() -> int:
    # Check if gemini CLI exists
    gemini = shutil.which("gemini")
    if not gemini:
        sys.stderr.write("missing gemini CLI\n")
        return 2

    home = os.environ.get("PLURIBUS_GEMINI_HOME") or "/pluribus/.pluribus/agent_homes/gemini"
    try:
        os.makedirs(home, exist_ok=True)
    except Exception:
        pass

    # Fast-path: if Gemini CLI isn't configured in this HOME, don't try to run it (startup can be slow).
    settings = Path(home) / ".gemini" / "settings.json"
    if not settings.exists():
        sys.stderr.write(f"gemini CLI needs login/config ({settings})\n")
        return 1

    # If settings explicitly require env API keys, fail fast if they aren't present.
    # (Avoid slow CLI startup logs that can obscure the actual error.)
    try:
        raw = settings.read_text(encoding="utf-8", errors="replace")
        cfg = json.loads(raw) if raw.strip() else {}
    except Exception:
        cfg = {}
    api_key_source = str(cfg.get("api_key_source") or "").strip().lower()
    if api_key_source in {"env", "environment"}:
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            sys.stderr.write("gemini CLI configured for api_key_source=env but no GOOGLE_API_KEY/GEMINI_API_KEY is set\n")
            return 1

    # Prefer a bounded runtime check (no tokens should be consumed for a trivial prompt).
    env = dict(os.environ)
    env["HOME"] = home
    # If the CLI is NOT configured for env API keys, prefer the CLI's own auth (OAuth/web sessions)
    # by removing key-based paths.
    if api_key_source not in {"env", "environment"}:
        env.pop("GOOGLE_API_KEY", None)
        env.pop("GEMINI_API_KEY", None)

    rc, stdout, stderr, saw_oauth = _run_streaming(
        [gemini, "--output-format", "text", "Reply with OK."],
        env=env,
        timeout_s=20.0,
    )

    if rc == 0:
        print(f"gemini CLI: {gemini}")
        print("auth: ok")
        return 0

    combined = (stderr or "") + (stdout or "")
    if saw_oauth or looks_like_oauth_prompt(combined):
        url = extract_google_oauth_url(combined) or ""
        # Emit a request event so the dashboard/TUIs can surface it immediately.
        try:
            try:
                from nucleus.tools import agent_bus  # type: ignore
            except Exception:
                import agent_bus  # type: ignore
            paths = agent_bus.resolve_bus_paths(os.environ.get("PLURIBUS_BUS_DIR"))
            agent_bus.emit_event(
                paths,
                topic="provider.gemini-cli.auth_required",
                kind="request",
                level="warn",
                actor=os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "operator",
                data={
                    "provider": "gemini-cli",
                    "home": home,
                    "auth_url": url,
                    "hint": "Open auth_url in a browser, complete login, then run gemini interactively to paste the code.",
                    "next_cmd": f'HOME="{home}" gemini',
                },
                trace_id=None,
                run_id=None,
                durable=True,
            )
        except Exception:
            pass
        if url:
            sys.stderr.write(f"gemini CLI requires login: {url}\n")
        else:
            sys.stderr.write("gemini CLI requires login (OAuth prompt detected)\n")
        return 3

    # Quota exhaustion means auth is valid, just rate-limited - treat as "available but warn"
    if "exhausted your daily quota" in combined or "quota" in combined.lower() and "exceeded" in combined.lower():
        print(f"gemini CLI: {gemini}")
        print("auth: ok (quota limited)")
        sys.stderr.write("Gemini quota exhausted - requests may fail until quota resets\n")
        return 0  # Auth is valid, quota is a runtime issue

    err = combined.strip()
    if not err:
        err = "gemini auth check failed"
    # Prefer the most-informative line(s) (often at the end).
    lines = [ln.strip() for ln in err.splitlines() if ln.strip()]
    msg = ""
    if lines:
        # If an explicit auth/config message exists, surface it.
        keywords = ("auth", "login", "api key", "settings", "please", "error", "quota", "exceeded", "forbidden", "unauthorized")
        for ln in reversed(lines):
            lnl = ln.lower()
            if any(k in lnl for k in keywords):
                msg = ln
                break
        if not msg:
            msg = lines[-1]
    else:
        msg = err
    sys.stderr.write(msg[:200] + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
