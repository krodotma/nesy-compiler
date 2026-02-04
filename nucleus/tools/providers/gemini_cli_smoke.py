#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import selectors
import os as _os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_loader import load_pluribus_env  # noqa: E402

from gemini_oauth_prompt import extract_google_oauth_url, looks_like_oauth_prompt


def _run_streaming(cmd: list[str], *, env: dict, timeout_s: float) -> tuple[int, str, str, bool]:
    """
    Run gemini CLI with incremental capture so OAuth prompts surface immediately.
    Returns (rc, stdout, stderr, saw_oauth).
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
    except Exception as e:
        return 1, "", f"failed to start gemini CLI: {e}\n", False

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
                out_chunks.append(chunk)
            else:
                err_chunks.append(chunk)

            combined_tail = (combined_tail + chunk.decode("utf-8", errors="replace"))[-20000:]
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


def main(argv: list[str]) -> int:
    load_pluribus_env()
    p = argparse.ArgumentParser(prog="gemini_cli_smoke.py", description="Gemini smoke test via gemini CLI (web auth).")
    p.add_argument("--prompt", required=True, help="User prompt text")
    p.add_argument("--model", default=os.environ.get("GEMINI_MODEL") or None)
    p.add_argument(
        "--home",
        default=os.environ.get("PLURIBUS_GEMINI_HOME") or "/pluribus/.pluribus/agent_homes/gemini",
        help="HOME directory to use for Gemini CLI OAuth/session state.",
    )
    p.add_argument("--timeout-s", default="120", help="Hard timeout for the CLI call (default: 120).")
    p.add_argument(
        "--allow-api-key",
        action="store_true",
        help="Allow GOOGLE_API_KEY/GEMINI_API_KEY env passthrough (default: stripped to prefer OAuth).",
    )
    args = p.parse_args(argv)

    gemini = shutil.which("gemini")
    if not gemini:
        sys.stderr.write("missing gemini CLI on PATH\n")
        return 2

    cmd: list[str] = [gemini, "--output-format", "text"]
    if args.model:
        cmd += ["--model", args.model]
    cmd += [args.prompt]

    env = dict(os.environ)
    try:
        os.makedirs(args.home, exist_ok=True)
    except Exception:
        pass
    env["HOME"] = args.home
    if not args.allow_api_key:
        # Prefer OAuth-based CLI auth over API keys to avoid Vertex 401 failures.
        env.pop("GOOGLE_API_KEY", None)
        env.pop("GEMINI_API_KEY", None)

    rc, stdout, stderr, saw_oauth = _run_streaming(cmd, env=env, timeout_s=max(1.0, float(args.timeout_s)))
    if rc == 124:
        sys.stderr.write(f"gemini CLI timed out after {args.timeout_s}s\n")
        return 124

    combined = (stderr or "") + (stdout or "")
    if rc != 0 and (saw_oauth or looks_like_oauth_prompt(combined)):
        url = extract_google_oauth_url(combined) or ""
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
                    "home": args.home,
                    "auth_url": url,
                    "hint": "Run gemini interactively with HOME set to complete OAuth.",
                    "next_cmd": f'HOME="{args.home}" gemini',
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
    if stderr:
        # Filter out noisy [STARTUP] logs from gemini CLI
        filtered_stderr = "\n".join(
            line for line in stderr.splitlines()
            if not line.strip().startswith("[STARTUP]")
        )
        if filtered_stderr:
            sys.stderr.write(filtered_stderr + "\n")
    if stdout:
        sys.stdout.write(stdout)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
