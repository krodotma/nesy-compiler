#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_loader import load_pluribus_env  # noqa: E402


def have_gcloud() -> bool:
    from shutil import which

    return bool(which("gcloud"))

def gcloud_env() -> dict:
    """
    Ensure we use the same gcloud credential store as bus-gcloud.
    """
    base = os.environ.get("PLURIBUS_GCLOUD_HOME") or "/pluribus/.pluribus/agent_homes/gcloud"
    cfg = os.environ.get("CLOUDSDK_CONFIG") or str(Path(base) / ".config" / "gcloud")
    env = dict(os.environ)
    env.setdefault("HOME", base)
    env["CLOUDSDK_CONFIG"] = cfg
    return env


def get_access_token() -> str | None:
    """
    Uses the local gcloud auth store. Requires:
      - gcloud installed
      - `gcloud auth login` completed
    """
    if not have_gcloud():
        return None
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
        return None
    if p.returncode != 0:
        return None
    tok = (p.stdout or "").strip()
    return tok or None

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
            for line in cfg_file.read_text(encoding="utf-8", errors="replace").splitlines():
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


def main(argv: list[str]) -> int:
    load_pluribus_env()
    ap = argparse.ArgumentParser(prog="vertex_gemini_smoke.py", description="Gemini smoke test via Vertex AI (gcloud user auth, no deps).")
    ap.add_argument("--prompt", required=True, help="User prompt text")
    ap.add_argument("--project", default=os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    ap.add_argument("--location", default=os.environ.get("VERTEX_LOCATION") or "us-central1")
    ap.add_argument("--model", default=os.environ.get("VERTEX_GEMINI_MODEL") or "gemini-3-pro-preview")
    ap.add_argument(
        "--host",
        default=os.environ.get("VERTEX_HOST") or "aiplatform.googleapis.com",
        help="Vertex API host. For some publisher preview models, 'aiplatform.googleapis.com' works when '<region>-aiplatform.googleapis.com' 404s.",
    )
    ap.add_argument(
        "--quota-project",
        default=os.environ.get("VERTEX_QUOTA_PROJECT") or os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT") or None,
        help="Quota project for user creds (sets X-Goog-User-Project). Defaults to --project.",
    )
    ap.add_argument("--require-gemini3", action="store_true", help="Fail unless model name starts with 'gemini-3'.")
    ap.add_argument("--timeout-s", type=int, default=60)
    args = ap.parse_args(argv)

    if not args.project:
        args.project = get_default_project()
    if not args.project:
        sys.stderr.write("missing VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT), and no gcloud default project configured\n")
        return 2

    if args.require_gemini3 and not str(args.model).startswith("gemini-3"):
        sys.stderr.write("refusing non-gemini-3 model (set --model gemini-3-...)\n")
        return 2

    token = get_access_token()
    if not token:
        sys.stderr.write("missing gcloud user access token; run: gcloud auth login\n")
        return 2

    quota_project = (args.quota_project or args.project or "").strip()
    if not quota_project:
        sys.stderr.write("missing quota project (set --quota-project or VERTEX_QUOTA_PROJECT)\n")
        return 2

    model_path = f"projects/{args.project}/locations/{args.location}/publishers/google/models/{args.model}"
    host = str(args.host or "").strip()
    if not host:
        sys.stderr.write("missing host (set --host aiplatform.googleapis.com)\n")
        return 2
    url = f"https://{host}/v1/{urllib.parse.quote(model_path, safe='/')}:generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": args.prompt}]}]}

    req = urllib.request.Request(
        url=url,
        method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
            "X-Goog-User-Project": quota_project,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=int(args.timeout_s)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        sys.stderr.write(f"http error: {e.code}\n{body}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        return 1

    try:
        obj = json.loads(raw)
        text = (
            (((obj.get("candidates") or [])[0].get("content") or {}).get("parts") or [])[0].get("text")
            if isinstance(obj, dict)
            else None
        )
    except Exception:
        obj = None
        text = None

    if text:
        sys.stdout.write(text.strip() + "\n")
        return 0

    sys.stdout.write(raw + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
