#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_loader import load_pluribus_env  # noqa: E402


def main(argv: list[str]) -> int:
    """
    "Vertex Express" smoke test (API-key style).

    This attempts a Vertex AI publisher-model request using an API key (no gcloud token).
    It only succeeds if the backend accepts API-key auth for the target endpoint.
    """
    load_pluribus_env()
    ap = argparse.ArgumentParser(
        prog="vertex_express_gemini_smoke.py",
        description="Gemini smoke test via Vertex AI using API key (\"vertex-express-mode\").",
    )
    ap.add_argument("--prompt", required=True, help="User prompt text")
    ap.add_argument("--api-key", default=os.environ.get("VERTEX_EXPRESS_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None)
    ap.add_argument("--project", default=os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    ap.add_argument("--location", default=os.environ.get("VERTEX_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1")
    ap.add_argument("--model", default=os.environ.get("VERTEX_GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-3-pro-preview")
    ap.add_argument("--host", default=os.environ.get("VERTEX_HOST") or "aiplatform.googleapis.com")
    ap.add_argument("--timeout-s", type=int, default=60)
    args = ap.parse_args(argv)

    if not args.api_key:
        sys.stderr.write("missing api key (set VERTEX_EXPRESS_API_KEY or GEMINI_API_KEY)\n")
        return 2
    if not args.project:
        sys.stderr.write("missing project (set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT)\n")
        return 2
    if not args.host:
        sys.stderr.write("missing host (set VERTEX_HOST=aiplatform.googleapis.com)\n")
        return 2

    model_path = f"projects/{args.project}/locations/{args.location}/publishers/google/models/{args.model}"
    url = f"https://{args.host}/v1/{urllib.parse.quote(model_path, safe='/')}:generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": args.prompt}]}]}

    # Try both common API-key styles (query param + header). If unsupported, we should fail cleanly.
    url_with_key = url + "?" + urllib.parse.urlencode({"key": args.api_key})
    req = urllib.request.Request(
        url=url_with_key,
        method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-goog-api-key": str(args.api_key),
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
        text = None

    if text:
        sys.stdout.write(text.strip() + "\n")
        return 0

    sys.stdout.write(raw + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

