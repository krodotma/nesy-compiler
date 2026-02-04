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
    if not have_gcloud():
        return None
    p = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=gcloud_env(),
    )
    if p.returncode != 0:
        return None
    tok = (p.stdout or "").strip()
    return tok or None


def main(argv: list[str]) -> int:
    load_pluribus_env()
    ap = argparse.ArgumentParser(
        prog="vertex_models_list.py", description="List Vertex AI publisher models (google) for discovery (requires gcloud auth)."
    )
    ap.add_argument("--project", default=os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    ap.add_argument("--location", default=os.environ.get("VERTEX_LOCATION") or "us-central1")
    ap.add_argument("--filter", dest="filter_text", default=None, help="Substring filter on model id (e.g., 'gemini-3').")
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="Print raw JSON response (first page only).")
    ap.add_argument(
        "--quota-project",
        default=os.environ.get("VERTEX_QUOTA_PROJECT") or os.environ.get("GOOGLE_CLOUD_QUOTA_PROJECT") or None,
        help="Quota project for gcloud calls (maps to --billing-project). Defaults to --project.",
    )
    args = ap.parse_args(argv)

    if not args.project:
        sys.stderr.write("missing VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT)\n")
        return 2

    # Prefer gcloud Model Garden listing (stable + handles the correct backend endpoint).
    # Note: for user credentials, a quota project is required; gcloud exposes this as --billing-project.
    if have_gcloud():
        quota_project = (args.quota_project or args.project or "").strip()
        cmd = ["gcloud", "ai", "model-garden", "models", "list", "--project", args.project, "--billing-project", quota_project]
        if args.filter_text:
            cmd += ["--model-filter", args.filter_text]
        cmd += ["--format", "value(name)", "--limit", str(int(args.page_size) * int(args.max_pages))]
        p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=gcloud_env())
        if p.returncode != 0:
            sys.stderr.write(p.stderr or p.stdout)
            if not (p.stderr or p.stdout).endswith("\n"):
                sys.stderr.write("\n")
            return int(p.returncode or 1)
        for line in (p.stdout or "").splitlines():
            name = line.strip()
            if not name:
                continue
            # Expected: publishers/google/models/<id>
            if "/models/" in name:
                mid = name.split("/models/")[-1]
            else:
                mid = name
            sys.stdout.write(mid + "\n")
        return 0

    token = get_access_token()
    if not token:
        sys.stderr.write("missing gcloud user access token; run: gcloud auth login\n")
        return 2

    base = f"https://{args.location}-aiplatform.googleapis.com/v1"
    parent = f"projects/{args.project}/locations/{args.location}/publishers/google"
    url = f"{base}/{urllib.parse.quote(parent, safe='/')}/models?pageSize={int(args.page_size)}"

    models: list[dict] = []
    next_page = ""
    for _ in range(int(args.max_pages)):
        u = url + (f"&pageToken={urllib.parse.quote(next_page)}" if next_page else "")
        req = urllib.request.Request(url=u, method="GET", headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            sys.stderr.write(f"http error: {e.code}\n{body}\n")
            return 1
        except Exception as e:
            sys.stderr.write(f"error: {e}\n")
            return 1

        obj = json.loads(raw) if raw.strip().startswith("{") else {"raw": raw}
        if args.json:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
            return 0

        for m in (obj.get("models") or []):
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "")
            mid = name.split("/models/")[-1] if "/models/" in name else name
            models.append(
                {
                    "id": mid,
                    "name": name,
                    "displayName": m.get("displayName"),
                    "description": m.get("description"),
                }
            )

        next_page = str(obj.get("nextPageToken") or "")
        if not next_page:
            break

    if args.filter_text:
        ft = args.filter_text.strip().lower()
        models = [m for m in models if ft in str(m.get("id") or "").lower()]

    for m in models:
        sys.stdout.write(str(m.get("id")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
