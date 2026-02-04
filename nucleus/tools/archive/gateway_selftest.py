#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _http_json(method: str, url: str, *, payload: dict | None = None, headers: dict[str, str] | None = None, timeout_s: float = 15.0) -> tuple[int, dict[str, str], dict]:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=req_headers, method=method.upper())
    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw) if raw.strip() else {}
            return int(resp.status), hdrs, obj
    except HTTPError as e:
        hdrs = {k.lower(): v for k, v in getattr(e, "headers", {}).items()}
        raw = (e.read() or b"").decode("utf-8", errors="replace")
        try:
            obj = json.loads(raw) if raw.strip() else {}
        except Exception:
            obj = {"raw": raw}
        return int(getattr(e, "code", 0) or 0), hdrs, obj
    except URLError as e:
        return 0, {}, {"error": str(e)}


def _emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if agent_bus is None:
        return
    if not bus_dir:
        return
    try:
        paths = agent_bus.resolve_bus_paths(bus_dir)
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception:
        return


def _router_smoke(*, prompt: str, model: str) -> dict:
    tool = REPO_ROOT / "nucleus" / "tools" / "providers" / "router.py"
    if not tool.exists():
        return {"ok": False, "error": "missing router.py"}
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.run(
            [
                sys.executable,
                str(tool),
                "--provider",
                "auto",
                "--prompt",
                prompt,
                "--model",
                model,
                "--format",
                "json",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "router timeout"}

    out = (p.stdout or "").strip()
    try:
        obj = json.loads(out) if out.startswith("{") else None
    except Exception:
        obj = None
    if isinstance(obj, dict):
        return {"ok": bool(obj.get("ok")), "provider": obj.get("provider"), "model": obj.get("model"), "error": obj.get("error")}
    return {"ok": p.returncode == 0, "exit_code": int(p.returncode), "stdout": out[:200], "stderr": (p.stderr or "").strip()[:200]}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gateway_selftest.py", description="PBVW gateway self-test (smoke).")
    ap.add_argument("--base-url", default=os.environ.get("PLURIBUS_GATEWAY_BASE_URL") or "http://127.0.0.1:8080")
    ap.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR") or "")
    ap.add_argument("--actor", default=os.environ.get("PLURIBUS_ACTOR") or "gateway-selftest")
    ap.add_argument("--api-key", default=os.environ.get("PLURIBUS_GATEWAY_API_KEY") or os.environ.get("GEMINI_API_KEY") or "gw-dev-key")
    ap.add_argument("--emit-bus", action="store_true")
    ap.add_argument("--skip-router", action="store_true")
    ap.add_argument(
        "--router-profile",
        default=(os.environ.get("PLURIBUS_PROVIDER_PROFILE") or "").strip() or "full",
        help="Profile for the direct router.py smoke check (default: full).",
    )
    args = ap.parse_args(argv)

    run_id = str(uuid.uuid4())
    bus_dir = str(args.bus_dir).strip() or None
    base = str(args.base_url).rstrip("/")

    started = _now_iso()
    report: dict = {
        "ok": False,
        "run_id": run_id,
        "started_iso": started,
        "base_url": base,
        "checks": {},
    }

    if args.emit_bus:
        _emit_bus(bus_dir, topic="gateway.selftest.start", kind="metric", level="info", actor=args.actor, data={"run_id": run_id, "base_url": base, "started_iso": started})

    status, _, health = _http_json("GET", f"{base}/health", timeout_s=5)
    report["checks"]["health"] = {"ok": status == 200 and bool(health.get("ok")), "status": status, "body": health}

    gemini_url = f"{base}/v1beta/models/gemini-2.5-pro:generateContent?key={args.api_key}"
    status, hdrs, body = _http_json(
        "POST",
        gemini_url,
        payload={"contents": [{"role": "user", "parts": [{"text": "Reply with exactly OK"}]}]},
        timeout_s=60,
    )
    txt = None
    try:
        txt = (((body.get("candidates") or [])[0].get("content") or {}).get("parts") or [])[0].get("text")
    except Exception:
        txt = None

    report["checks"]["gemini_alias"] = {
        "ok": status == 200 and str(txt or "").strip() == "OK",
        "status": status,
        "x_requested_model": hdrs.get("x-requested-model"),
        "x_served_by_model": hdrs.get("x-served-by-model"),
        "x_served_by_provider": hdrs.get("x-served-by-provider"),
        "x_model_alias_hit": hdrs.get("x-model-alias-hit"),
        "text": str(txt or "")[:200],
    }

    if not args.skip_router:
        prior_profile = os.environ.get("PLURIBUS_PROVIDER_PROFILE")
        try:
            os.environ["PLURIBUS_PROVIDER_PROFILE"] = str(args.router_profile)
            report["checks"]["router_gemini3"] = _router_smoke(prompt="Reply with exactly OK", model="gemini-3-pro-preview")
            report["checks"]["router_gemini3"]["router_profile"] = str(args.router_profile)
        finally:
            if prior_profile is None:
                os.environ.pop("PLURIBUS_PROVIDER_PROFILE", None)
            else:
                os.environ["PLURIBUS_PROVIDER_PROFILE"] = prior_profile

    ok = all(bool(v.get("ok")) for v in report["checks"].values())
    report["ok"] = ok
    report["finished_iso"] = _now_iso()

    if args.emit_bus:
        _emit_bus(
            bus_dir,
            topic="gateway.selftest.report",
            kind="artifact",
            level="info" if ok else "warn",
            actor=args.actor,
            data=report,
        )
    sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
