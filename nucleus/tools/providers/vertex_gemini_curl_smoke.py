#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_loader import load_pluribus_env  # noqa: E402


def have(cmd: str) -> bool:
    from shutil import which

    return bool(which(cmd))


def gcloud_env() -> dict:
    base = os.environ.get("PLURIBUS_GCLOUD_HOME") or "/pluribus/.pluribus/agent_homes/gcloud"
    cfg = os.environ.get("CLOUDSDK_CONFIG") or str(Path(base) / ".config" / "gcloud")
    env = dict(os.environ)
    env.setdefault("HOME", base)
    env["CLOUDSDK_CONFIG"] = cfg
    return env


def get_access_token() -> str | None:
    if not have("gcloud"):
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

    if not have("gcloud"):
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
    ap = argparse.ArgumentParser(
        prog="vertex_gemini_curl_smoke.py",
        description="Gemini smoke test via Vertex AI using curl (independent HTTP stack).",
    )
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--project", default=os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    ap.add_argument("--location", default=os.environ.get("VERTEX_LOCATION") or "us-central1")
    ap.add_argument("--model", default=os.environ.get("VERTEX_GEMINI_MODEL") or "gemini-3-pro-preview")
    ap.add_argument(
        "--host",
        default=os.environ.get("VERTEX_HOST") or "aiplatform.googleapis.com",
        help="Vertex API host (default: aiplatform.googleapis.com).",
    )
    ap.add_argument("--require-gemini3", action="store_true")
    ap.add_argument(
        "--quota-project",
        default=os.environ.get("VERTEX_QUOTA_PROJECT") or os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT") or None,
        help="Quota project (sets X-Goog-User-Project). Defaults to --project.",
    )
    ap.add_argument("--timeout-s", type=int, default=60)
    args = ap.parse_args(argv)

    if not have("curl"):
        sys.stderr.write("missing curl\n")
        return 2
    if not args.project:
        args.project = get_default_project()
    if not args.project:
        sys.stderr.write("missing VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT), and no gcloud default project configured\n")
        return 2
    if args.require_gemini3 and not str(args.model).startswith("gemini-3"):
        sys.stderr.write("refusing non-gemini-3 model (set --model gemini-3-...)\n")
        return 2

    tok = get_access_token()
    if not tok:
        sys.stderr.write("missing gcloud user access token; run: gcloud auth login\n")
        return 2

    quota_project = (args.quota_project or args.project or "").strip()
    if not quota_project:
        sys.stderr.write("missing quota project (set --quota-project or VERTEX_QUOTA_PROJECT)\n")
        return 2

    host = str(args.host or "").strip()
    if not host:
        sys.stderr.write("missing host (set --host aiplatform.googleapis.com)\n")
        return 2
    url = f"https://{host}/v1/projects/{args.project}/locations/{args.location}/publishers/google/models/{args.model}:generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": args.prompt}]}]}
    body = json.dumps(payload, ensure_ascii=False)

    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "--max-time",
        str(int(args.timeout_s)),
        "-H",
        "Content-Type: application/json; charset=utf-8",
        "-H",
        f"Authorization: Bearer {tok}",
        "-H",
        f"X-Goog-User-Project: {quota_project}",
        "-d",
        body,
        url,
    ]
    p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr or p.stdout)
        if not (p.stderr or p.stdout).endswith("\n"):
            sys.stderr.write("\n")
        return int(p.returncode or 1)

    raw = (p.stdout or "").strip()
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
