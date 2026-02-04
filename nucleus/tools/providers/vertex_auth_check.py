#!/usr/bin/env python3
"""Fast Vertex auth check (no Vertex LLM call).

This checks for:
- gcloud availability
- ability to print an access token from the shared gcloud store
- a configured project (env or gcloud config)

It intentionally does *not* call Vertex APIs; use vertex_gemini*_smoke.py for that.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def gcloud_env() -> dict:
    base = os.environ.get("PLURIBUS_GCLOUD_HOME") or "/pluribus/.pluribus/agent_homes/gcloud"
    cfg = os.environ.get("CLOUDSDK_CONFIG") or str(Path(base) / ".config" / "gcloud")
    env = dict(os.environ)
    env.setdefault("HOME", base)
    env["CLOUDSDK_CONFIG"] = cfg
    env.setdefault("CLOUDSDK_CORE_DISABLE_PROMPTS", "1")
    return env


def have_gcloud() -> bool:
    return bool(shutil.which("gcloud"))


def get_default_project() -> str | None:
    # Prefer reading gcloud config files directly (fast, avoids gcloud hangs).
    base = os.environ.get("PLURIBUS_GCLOUD_HOME") or "/pluribus/.pluribus/agent_homes/gcloud"
    cfg_dir = Path(os.environ.get("CLOUDSDK_CONFIG") or str(Path(base) / ".config" / "gcloud"))
    try:
        active = (cfg_dir / "active_config").read_text(encoding="utf-8", errors="replace").strip() or "default"
    except Exception:
        active = "default"
    cfg_file = cfg_dir / "configurations" / f"config_{active}"
    if cfg_file.exists():
        try:
            text = cfg_file.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in text:
                if line.strip().startswith("project") and "=" in line:
                    val = line.split("=", 1)[1].strip()
                    if val:
                        return val
        except Exception:
            pass

    if not have_gcloud():
        return None
    try:
        p = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=gcloud_env(),
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    val = (p.stdout or "").strip()
    if not val or val.lower() in ("(unset)", "unset"):
        return None
    return val


def have_access_token() -> bool:
    if not have_gcloud():
        return False
    # Fast-path: if the shared gcloud store has an access token DB, assume usable.
    # This avoids slow refreshes and prevents the session daemon from blocking.
    base = os.environ.get("PLURIBUS_GCLOUD_HOME") or "/pluribus/.pluribus/agent_homes/gcloud"
    cfg_dir = Path(os.environ.get("CLOUDSDK_CONFIG") or str(Path(base) / ".config" / "gcloud"))
    token_db = cfg_dir / "access_tokens.db"
    try:
        if token_db.exists() and token_db.stat().st_size > 0:
            return True
    except Exception:
        pass
    try:
        p = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=gcloud_env(),
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        return False
    if p.returncode != 0:
        return False
    return bool((p.stdout or "").strip())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="vertex_auth_check.py", description="Fast Vertex auth/config check.")
    ap.add_argument("--require-curl", action="store_true", help="Also require curl (for vertex-curl route).")
    args = ap.parse_args(argv)

    if args.require_curl and not shutil.which("curl"):
        sys.stderr.write("missing curl\n")
        return 2

    if not have_gcloud():
        sys.stderr.write("missing gcloud\n")
        return 2

    project = (os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip() or get_default_project()
    if not project:
        sys.stderr.write("missing project (set VERTEX_PROJECT/GOOGLE_CLOUD_PROJECT or `gcloud config set project ...`)\n")
        return 1

    if not have_access_token():
        sys.stderr.write("missing gcloud user access token; run: gcloud auth login\n")
        return 1

    sys.stdout.write(f"gcloud: {shutil.which('gcloud')}\n")
    sys.stdout.write(f"project: {project}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
