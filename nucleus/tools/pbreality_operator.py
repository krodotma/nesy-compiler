#!/usr/bin/env python3
from __future__ import annotations

"""
PBREALITY — Production Baseline Reality operator.

Generates a release-readiness report using the canonical template
and emits append-only bus events for coordination and provenance.
"""

import argparse
import getpass
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


DEFAULT_TEMPLATE = """PBREALITY_REPORT v1

CONTEXT
- baseline_env: <laptop name/specs; first external test subject>
- target_env: <krodotma / kroma.live>
- branch_or_commit: <branch or sha>
- time_window: <last hours of work>
- scope: <what this push covers>

REALITY_GAP (local vs production)
- ux_parity: <match|drift> - <1-2 lines>
- perf_parity: <match|drift> - <1-2 lines>
- feature_parity: <match|drift> - <1-2 lines>
- reliability: <match|drift> - <1-2 lines>

ESSENTIAL_FILES (path -> role -> risk)
- <path>: <role> | risk=<low|med|high>

ESSENTIAL_CONCEPTS (distilled)
- <concept>: <1-2 lines, with reference path>

EXTERNAL_TEST_PLAN
- visual_smoke: <tool + URL + expected>
- console_errors: <how checked>
- network_failures: <how checked>
- provider_auth: <what must be logged in>
- bus_health: <expected signals>

PUBLIC_VALUE (what users experience)
- <3-5 bullets of observable value>

EVIDENCE_REQUIRED
- bus_events: <topics + req_id>
- artifacts: <paths>
- screenshots: <paths>
- logs: <paths>

RISKS_AND_BLOCKERS
- <risk or blocker>

NEXT_ACTIONS (ordered)
1) <action>
2) <action>

SIGN_OFF_CRITERIA
- <objective checks that prove parity>

RULES
- Do not claim tests run unless executed. Mark TODO explicitly.
- Keep it append-only: add evidence, do not overwrite history.
- Prefer SOTA -> reality: curation -> distill -> hypothesize -> apply -> verify -> provenance.
"""


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_bus_dir() -> Path:
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_UN)


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> str:
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
    append_ndjson(bus_dir / "events.ndjson", evt)
    return evt_id


def resolve_branch_or_commit(root: Path) -> str:
    head = root / ".git" / "HEAD"
    try:
        raw = head.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""
    if raw.startswith("ref:"):
        ref = raw.split(":", 1)[1].strip()
        return ref.rsplit("/", 1)[-1]
    return raw[:12]


def read_template(path: Path) -> str:
    if not path.exists():
        return DEFAULT_TEMPLATE
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return DEFAULT_TEMPLATE
    lines = raw.splitlines()
    in_block = False
    block: list[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            if not in_block:
                in_block = True
                continue
            break
        if in_block:
            block.append(line)
    if block:
        return "\n".join(block).strip() + "\n"
    return DEFAULT_TEMPLATE


def apply_overrides(template: str, overrides: dict[str, str]) -> str:
    lines = template.splitlines()
    out: list[str] = []
    for line in lines:
        updated = line
        stripped = line.lstrip()
        for key, value in overrides.items():
            if not value:
                continue
            prefix = f"- {key}:"
            if stripped.startswith(prefix):
                indent = line[: len(line) - len(stripped)]
                updated = f"{indent}{prefix} {value}"
                break
        out.append(updated)
    return "\n".join(out).rstrip() + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_recent_actors(bus_dir: Path, window_s: int = 900, max_bytes: int = 2_000_000) -> set[str]:
    events_path = bus_dir / "events.ndjson"
    if not events_path.exists():
        return set()
    cutoff = time.time() - window_s
    actors: set[str] = set()
    try:
        size = events_path.stat().st_size
        start = max(0, size - max_bytes)
        with events_path.open("rb") as f:
            f.seek(start)
            data = f.read()
        lines = data.splitlines()
        if start > 0 and lines:
            lines = lines[1:]
        for raw in lines:
            try:
                evt = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue
            try:
                ts = float(evt.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            if ts < cutoff:
                continue
            actor = str(evt.get("actor") or "").strip()
            if actor and actor != "unknown":
                actors.add(actor)
    except Exception:
        return set()
    return actors


def check_paip_isolation(*, bus_dir: Path, actor: str, req_id: str, repo: Path) -> dict:
    actors = scan_recent_actors(bus_dir)
    unique_actors = len(actors)
    multi_agent = unique_actors > 1
    repo_root_str = str(repo)
    shared_repo = repo_root_str == "/pluribus"
    violation_emitted = False

    if multi_agent and shared_repo:
        emit_bus(
            bus_dir,
            topic="paip.isolation.violation",
            kind="request",
            level="warn",
            actor=actor,
            data={
                "req_id": req_id,
                "repo_root": repo_root_str,
                "cwd": os.getcwd(),
                "unique_actors": unique_actors,
                "actors": sorted(actors),
                "reason": "multi_agent_shared_repo",
                "iso": now_iso_utc(),
            },
        )
        violation_emitted = True

    return {
        "multi_agent": multi_agent,
        "unique_actors": unique_actors,
        "shared_repo": shared_repo,
        "violation_emitted": violation_emitted,
        "actors": sorted(actors),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbreality_operator.py", description="PBREALITY semantic operator: generate release readiness report.")
    p.add_argument("--bus-dir", default=None)
    p.add_argument("--actor", default=None)
    p.add_argument("--req-id", default=None)
    p.add_argument("--subproject", default="ops", help="Subproject tag for infer_sync mirror.")
    p.add_argument("--template", default=None, help="Template path (default: nucleus/docs/workflows/pbreality.md).")
    p.add_argument("--report-dir", default=None, help="Directory for report output (default: agent_reports).")
    p.add_argument("--report-path", default=None, help="Explicit report path (overrides --report-dir).")
    p.add_argument("--baseline-env", default=None)
    p.add_argument("--target-env", default=None)
    p.add_argument("--branch-or-commit", default=None)
    p.add_argument("--time-window", default=None)
    p.add_argument("--scope", default=None)
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "pbreality"
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else default_bus_dir()
    ensure_dir(bus_dir)
    (bus_dir / "events.ndjson").touch(exist_ok=True)

    root = repo_root()
    template_path = Path(args.template).expanduser().resolve() if args.template else (root / "nucleus" / "docs" / "workflows" / "pbreality.md")
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else (root / "agent_reports")
    ensure_dir(report_dir)

    req_id = args.req_id or str(uuid.uuid4())
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else (report_dir / f"pbreality_report_{stamp_utc()}.txt")

    paip_status = check_paip_isolation(bus_dir=bus_dir, actor=actor, req_id=req_id, repo=root)

    baseline_env = (args.baseline_env or os.environ.get("PLURIBUS_BASELINE_ENV") or platform.node()).strip()
    target_env = (args.target_env or "krodotma / kroma.live").strip()
    branch_or_commit = (args.branch_or_commit or resolve_branch_or_commit(root)).strip()
    time_window = (args.time_window or "last few hours").strip()
    scope = (args.scope or "release readiness checkpoint").strip()

    template = read_template(template_path)
    report_rel = ""
    try:
        report_rel = str(report_path.relative_to(root))
    except Exception:
        report_rel = str(report_path)

    overrides = {
        "baseline_env": baseline_env,
        "target_env": target_env,
        "branch_or_commit": branch_or_commit,
        "time_window": time_window,
        "scope": scope,
        "bus_events": f"operator.pbreality.request + operator.pbreality.report (req_id={req_id})",
        "artifacts": report_rel,
    }
    report_body = apply_overrides(template, overrides)
    ensure_dir(report_path.parent)
    report_path.write_text(report_body, encoding="utf-8")

    report_sha = sha256_text(report_body)
    payload = {
        "req_id": req_id,
        "intent": "pbreality",
        "subproject": str(args.subproject),
        "baseline_env": baseline_env,
        "target_env": target_env,
        "branch_or_commit": branch_or_commit,
        "time_window": time_window,
        "scope": scope,
        "report_path": report_rel,
        "report_sha256": report_sha,
        "template_path": str(template_path),
        "paip": paip_status,
        "iso": now_iso_utc(),
    }

    emit_bus(bus_dir, topic="operator.pbreality.request", kind="request", level="info", actor=actor, data=payload)
    emit_bus(bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data=payload)
    emit_bus(bus_dir, topic="operator.pbreality.report", kind="artifact", level="info", actor=actor, data=payload)

    sys.stdout.write(req_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
