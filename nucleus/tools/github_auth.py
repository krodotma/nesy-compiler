#!/usr/bin/env python3
"""
GitHub Auth Helper (gh) — bus-first, no-secrets

Purpose:
- Provide a non-interactive-ish control surface for GitHub auth state via `gh`.
- Emit append-only bus evidence WITHOUT ever emitting tokens.

Notes:
- Interactive `gh auth login` cannot be fully automated from non-interactive runners.
- This tool supports token-based login via env var (GH_TOKEN/GITHUB_TOKEN) and
  emits only booleans/ids/scopes (never credentials).

Bus topics:
- github.auth.status (metric)
- github.auth.login (artifact)
- github.repo.permissions (artifact)
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def default_bus_dir(bus_dir: str | None) -> Path:
    return Path(bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> str:
    bus_dir.mkdir(parents=True, exist_ok=True)
    path = bus_dir / "events.ndjson"
    path.touch(exist_ok=True)
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False, separators=(",", ":")) + "\n")
    return evt_id


def run_gh(argv: list[str], *, stdin_text: str | None = None) -> tuple[int, str, str]:
    p = subprocess.run(
        ["gh", *argv],
        input=stdin_text,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return int(p.returncode), str(p.stdout or ""), str(p.stderr or "")


def gh_hosts_file() -> Path:
    return Path(os.path.expanduser("~/.config/gh/hosts.yml"))


def status(*, host: str) -> dict[str, Any]:
    rc, out, err = run_gh(["auth", "status", "-h", host])
    logged_in = rc == 0
    # Avoid parsing too deeply; gh output changes. Provide minimal safe fields.
    login = None
    if logged_in:
        # Typical: "Logged in to github.com as USER (...)".
        for line in out.splitlines():
            if "Logged in to" in line and " as " in line:
                try:
                    login = line.split(" as ", 1)[1].split(" ", 1)[0].strip()
                except Exception:
                    login = None
                break
            # Newer: "✓ Logged in to github.com account USER (...)".
            if "Logged in to" in line and " account " in line:
                try:
                    login = line.split(" account ", 1)[1].split(" ", 1)[0].strip()
                except Exception:
                    login = None
                break
    return {
        "host": host,
        "logged_in": logged_in,
        "login": login,
        "hosts_file_present": gh_hosts_file().exists(),
        "stderr_hint": (err.strip().splitlines()[-1] if err.strip() else None),
    }


def token_from_env(prefer: list[str]) -> tuple[str | None, str | None]:
    for k in prefer:
        v = os.environ.get(k)
        if v:
            return v, k
    return None, None


def login_with_token(*, host: str, scopes: str, token_env_order: list[str]) -> dict[str, Any]:
    token, env_name = token_from_env(token_env_order)
    if not token or not env_name:
        return {"ok": False, "host": host, "reason": "missing_token_env", "token_env_tried": token_env_order}

    # `gh auth login --with-token` reads a token from stdin. Do not print it.
    argv = ["auth", "login", "--hostname", host, "--with-token"]
    if scopes:
        argv += ["--scopes", scopes]
    rc, out, err = run_gh(argv, stdin_text=token + "\n")
    # `gh` can emit helpful info to stdout; keep it in-process only.
    ok = rc == 0
    return {
        "ok": ok,
        "host": host,
        "token_env_used": env_name,
        "scopes_requested": scopes or None,
        "stdout_hint": out.strip().splitlines()[-1] if out.strip() else None,
        "stderr_hint": err.strip().splitlines()[-1] if err.strip() else None,
    }


def repo_permissions(*, repo: str) -> dict[str, Any]:
    # Avoid query printing raw repo JSON to stdout; parse locally.
    rc, out, err = run_gh(["api", f"repos/{repo}"])
    if rc != 0:
        return {"ok": False, "repo": repo, "error": (err.strip() or out.strip() or "unknown")}
    try:
        obj = json.loads(out)
    except Exception:
        return {"ok": False, "repo": repo, "error": "invalid_json"}
    perms = obj.get("permissions") if isinstance(obj, dict) else None
    perms = perms if isinstance(perms, dict) else {}
    return {
        "ok": True,
        "repo": repo,
        "permissions": {k: bool(perms.get(k)) for k in ["pull", "triage", "push", "maintain", "admin"] if k in perms},
        "default_branch": obj.get("default_branch") if isinstance(obj, dict) else None,
        "private": obj.get("private") if isinstance(obj, dict) else None,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="github_auth.py", description="GitHub auth helper (gh) with bus evidence (no secrets).")
    p.add_argument("--bus-dir", default=None)
    p.add_argument("--actor", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="Report gh auth status for a host.")
    st.add_argument("--host", default="github.com")
    st.add_argument("--emit-bus", action="store_true")

    lg = sub.add_parser("login-with-token", help="Perform gh token login (reads token from env; never emits token).")
    lg.add_argument("--host", default="github.com")
    lg.add_argument("--scopes", default="repo,workflow")
    lg.add_argument("--token-env", default="GH_TOKEN,GITHUB_TOKEN", help="Comma-separated env vars to try in order.")
    lg.add_argument("--emit-bus", action="store_true")

    rp = sub.add_parser("repo-perms", help="Check repo permissions via gh api (requires auth).")
    rp.add_argument("--repo", required=True, help="owner/name")
    rp.add_argument("--emit-bus", action="store_true")

    en = sub.add_parser("ensure", help="Ensure logged in (token env only) and optionally check repo perms.")
    en.add_argument("--host", default="github.com")
    en.add_argument("--scopes", default="repo,workflow")
    en.add_argument("--token-env", default="GH_TOKEN,GITHUB_TOKEN")
    en.add_argument("--repo", default=None, help="Optional owner/name permission check.")
    en.add_argument("--emit-bus", action="store_true")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "github-auth"
    bus_dir = default_bus_dir(args.bus_dir)

    if args.cmd == "status":
        st = status(host=args.host)
        if args.emit_bus:
            emit_bus(bus_dir, topic="github.auth.status", kind="metric", level="info", actor=actor, data=st)
        print(json.dumps(st, indent=2))
        return 0 if st.get("logged_in") else 3

    if args.cmd == "login-with-token":
        token_env_order = [x.strip() for x in str(args.token_env).split(",") if x.strip()]
        res = login_with_token(host=args.host, scopes=str(args.scopes), token_env_order=token_env_order)
        if args.emit_bus:
            emit_bus(bus_dir, topic="github.auth.login", kind="artifact", level="info" if res.get("ok") else "warn", actor=actor, data=res)
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 2

    if args.cmd == "repo-perms":
        res = repo_permissions(repo=str(args.repo))
        if args.emit_bus:
            emit_bus(bus_dir, topic="github.repo.permissions", kind="artifact", level="info" if res.get("ok") else "warn", actor=actor, data=res)
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 2

    if args.cmd == "ensure":
        st0 = status(host=args.host)
        login_res = None
        if not st0.get("logged_in"):
            token_env_order = [x.strip() for x in str(args.token_env).split(",") if x.strip()]
            login_res = login_with_token(host=args.host, scopes=str(args.scopes), token_env_order=token_env_order)
            st1 = status(host=args.host)
        else:
            st1 = st0

        perms = repo_permissions(repo=str(args.repo)) if args.repo else None

        out = {"status_before": st0, "login": login_res, "status_after": st1, "repo_permissions": perms}
        if args.emit_bus:
            emit_bus(bus_dir, topic="github.auth.ensure", kind="artifact", level="info", actor=actor, data=out)
        print(json.dumps(out, indent=2))

        if not st1.get("logged_in"):
            return 3
        if perms and perms.get("ok") and isinstance(perms.get("permissions"), dict):
            # If caller asked for perms, require push for "writable" posture.
            if perms["permissions"].get("push") is not True:
                return 4
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
