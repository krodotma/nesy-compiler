#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nucleus.tools import webchat_bridge


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_providers(raw: str) -> list[str]:
    out: list[str] = []
    for part in (raw or "").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _tail_for_req_id(*, events_path: Path, topic: str, req_id: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + float(timeout_s)
    pos = events_path.stat().st_size if events_path.exists() else 0
    buf = b""

    while time.time() < deadline:
        if not events_path.exists():
            time.sleep(0.2)
            continue
        with events_path.open("rb") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        if not chunk:
            time.sleep(0.2)
            continue

        data = buf + chunk
        parts = data.split(b"\n")
        buf = parts.pop() if parts else b""
        for raw in parts:
            if not raw:
                continue
            try:
                ev = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if (ev.get("topic") or "") != topic:
                continue
            payload = ev.get("data") or {}
            if not isinstance(payload, dict):
                continue
            if str(payload.get("req_id") or "") != req_id:
                continue
            return payload

    return {"req_id": req_id, "ok": False, "error": "timeout"}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="3-pillar webchat bridge smoke test (chatgpt-web/claude-web/gemini-web) via VNC Playwright sessions."
    )
    parser.add_argument(
        "--providers",
        default=",".join(webchat_bridge.DEFAULT_WEBCHAT_PROVIDERS),
        help="Comma-separated providers (default: chatgpt-web,claude-web,gemini-web).",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with OK only.",
        help="Prompt to send to each provider (default: 'Reply with OK only.')",
    )
    parser.add_argument("--timeout-s", type=float, default=90.0, help="Per-provider timeout seconds.")
    parser.add_argument("--bus-dir", default=None, help="Bus dir override (default: resolve via agent_bus).")
    parser.add_argument("--actor", default=os.environ.get("PLURIBUS_ACTOR") or "3pillar-test", help="Bus actor.")
    parser.add_argument(
        "--check-webllm",
        action="store_true",
        help="Also request WebLLM widget status via the bus (requires dashboard WebLLM enabled).",
    )
    parser.add_argument("--webllm-timeout-s", type=float, default=10.0, help="WebLLM status timeout seconds.")
    parser.add_argument(
        "--no-bus-evidence",
        action="store_true",
        help="Do not emit bus artifacts (still prints a report).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send prompts even if browser_daemon.json reports tabs not ready.",
    )
    args = parser.parse_args(argv)

    providers = parse_providers(args.providers)
    if not providers:
        providers = list(webchat_bridge.DEFAULT_WEBCHAT_PROVIDERS)

    bus_dir = webchat_bridge.resolve_bus_dir(args.bus_dir)
    run_id = f"3pillar-{uuid.uuid4()}"
    started_iso = utc_now_iso()

    readiness = webchat_bridge.webchat_readiness(providers=providers)
    paths = webchat_bridge.webchat_paths()

    artifact_dir = bus_dir / "artifacts" / "verify" / "3pillar"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    start_payload = {
        "run_id": run_id,
        "started_iso": started_iso,
        "providers": providers,
        "prompt": args.prompt,
        "timeout_s": args.timeout_s,
        "bus_dir": str(bus_dir),
        "readiness": asdict(readiness),
        "paths": asdict(paths),
    }

    webllm_status: dict[str, Any] | None = None
    if args.check_webllm:
        try:
            from nucleus.tools import agent_bus

            req_id = f"3pillar-webllm-status-{uuid.uuid4()}"
            bus_paths = agent_bus.resolve_bus_paths(str(bus_dir))
            agent_bus.emit_event(
                bus_paths,
                topic="webllm.status.request",
                kind="request",
                level="info",
                actor=args.actor,
                data={"req_id": req_id, "at": utc_now_iso()},
                trace_id=None,
                run_id=run_id,
                durable=True,
            )
            webllm_status = _tail_for_req_id(
                events_path=Path(bus_paths.active_dir) / "events.ndjson",
                topic="webllm.status.response",
                req_id=req_id,
                timeout_s=float(args.webllm_timeout_s),
            )
        except Exception as e:
            webllm_status = {"ok": False, "error": str(e)}

    if webllm_status is not None:
        start_payload["webllm_status"] = webllm_status

    if not args.no_bus_evidence:
        webchat_bridge.emit_bus_artifact(bus_dir, topic="verify.3pillar.start", actor=args.actor, data=start_payload)

    print(json.dumps(start_payload, indent=2, sort_keys=True, ensure_ascii=False))

    results: list[dict] = []
    overall_ok = True

    for provider in providers:
        st = readiness.tab_status.get(provider, "missing")
        if not args.force and st != "ready":
            overall_ok = False
            results.append(
                asdict(
                    webchat_bridge.WebchatResult(
                        provider=provider,
                        ok=False,
                        latency_ms=0.0,
                        req_id="",
                        response_preview="",
                        error=f"skipped: tab status {st!r} (open dashboard Auth overlay to login/bootstrap)",
                    )
                )
            )
            continue

        try:
            r = webchat_bridge.send_webchat_prompt(
                prompt=args.prompt,
                provider=provider,
                bus_dir=bus_dir,
                actor=args.actor,
                timeout_s=float(args.timeout_s),
            )
            results.append(asdict(r))
            overall_ok = overall_ok and r.ok
        except Exception as e:
            overall_ok = False
            results.append(
                asdict(
                    webchat_bridge.WebchatResult(
                        provider=provider,
                        ok=False,
                        latency_ms=0.0,
                        req_id="",
                        response_preview="",
                        error=str(e),
                    )
                )
            )

    finished_iso = utc_now_iso()
    summary = {
        "run_id": run_id,
        "started_iso": started_iso,
        "finished_iso": finished_iso,
        "ok": overall_ok,
        "providers": providers,
        "results": results,
        "webllm_status": webllm_status,
        "notes": {
            "browser_state_path": readiness.browser_state_path,
            "browser_data_dir": readiness.browser_data_dir,
            "auth_help": "/?view=browser-auth (dashboard), or click ☁️ Auth",
        },
    }

    report_path = artifact_dir / f"3pillar_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{run_id}.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.no_bus_evidence:
        webchat_bridge.emit_bus_artifact(
            bus_dir,
            topic="verify.3pillar.result",
            actor=args.actor,
            data={"run_id": run_id, "ok": overall_ok, "report_path": str(report_path), "summary": summary},
        )

    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

