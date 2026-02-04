#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


import uuid

def emit_bus(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data) -> None:
    events_path = Path(bus_dir) / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    try:
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass


def default_bus_dir(args_bus_dir: str | None) -> str:
    return args_bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"


def rhizome_root() -> Path:
    # Prefer the global /pluribus rhizome, else CWD.
    for cand in (Path("/pluribus"), Path.cwd().resolve()):
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return Path.cwd().resolve()


def queue_depth(root: Path) -> int:
    idx = root / ".pluribus" / "index"
    req = idx / "requests.ndjson"
    resp = idx / "responses.ndjson"
    if not req.exists():
        return 0
    try:
        rqs = req.read_text(encoding="utf-8", errors="replace").splitlines()
        rps = resp.read_text(encoding="utf-8", errors="replace").splitlines() if resp.exists() else []
        return max(0, len(rqs) - len(rps))
    except Exception:
        return 0


def file_fingerprint(path: Path) -> dict:
    """
    Small, deterministic fingerprint suitable for cross-agent context checks.
    Avoids hashing large artifacts.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    except Exception as e:
        return {"path": str(path), "exists": False, "error": str(e)}

    info: dict = {
        "path": str(path),
        "exists": True,
        "bytes": int(st.st_size),
        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
    }

    # Only hash reasonably small text artifacts; large binary/log stores are excluded.
    max_hash_bytes = 512 * 1024
    if st.st_size <= max_hash_bytes:
        try:
            data = path.read_bytes()
            info["sha256"] = hashlib.sha256(data).hexdigest()
            try:
                text = data.decode("utf-8", errors="replace")
                info["lines"] = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
            except Exception:
                pass
        except Exception as e:
            info["hash_error"] = str(e)
    return info


def tcp_listening(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def recent_files(*, roots: list[Path], since_ts: float, limit: int = 40) -> list[dict]:
    """
    Small, bounded snapshot of recently modified files. This is used to answer:
    "Are agents looking at the same *current* state?" without requiring git.
    """
    out: list[dict] = []
    skip_dirs = {"node_modules", ".git", ".pluribus", "__pycache__", "dist", "build", ".venv"}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            try:
                st = root.stat()
            except Exception:
                continue
            if st.st_mtime >= since_ts:
                out.append(
                    {
                        "path": str(root),
                        "bytes": int(st.st_size),
                        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                    }
                )
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                except Exception:
                    continue
                if st.st_mtime < since_ts:
                    continue
                out.append(
                    {
                        "path": str(p),
                        "bytes": int(st.st_size),
                        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                    }
                )
    out.sort(key=lambda x: x.get("mtime_iso", ""), reverse=True)
    return out[: max(0, int(limit))]


def follow_topic(
    path: Path,
    *,
    topic: str,
    since_ts: float,
    timeout_s: float,
    poll_s: float,
    follow: bool,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    deadline = None if follow else (time.time() + max(0.0, timeout_s))
    def emit_backfill() -> list[dict]:
        # Backfill helps avoid missing fast reports that land before we start following.
        try:
            max_bytes = 512 * 1024
            with path.open("rb") as bf:
                bf.seek(0, os.SEEK_END)
                end = bf.tell()
                start = max(0, end - max_bytes)
                bf.seek(start)
                data = bf.read(end - start)
            lines = data.splitlines()
            out: list[dict] = []
            for b in lines[-2000:]:
                try:
                    obj = json.loads(b.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if obj.get("topic") != topic:
                    continue
                try:
                    ts = float(obj.get("ts") or 0.0)
                except Exception:
                    ts = 0.0
                if ts < since_ts:
                    continue
                out.append(obj)
            return out
        except Exception:
            return []

    for obj in emit_backfill():
        yield obj

    with path.open("r", encoding="utf-8", errors="replace") as f:
        # Follow new lines from the end.
        f.seek(0, os.SEEK_END)
        while True:
            if deadline is not None and time.time() >= deadline:
                break
            line = f.readline()
            if not line:
                time.sleep(max(0.05, poll_s))
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("topic") != topic:
                continue
            try:
                ts = float(obj.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            if ts < since_ts:
                continue
            yield obj


def parse_expected_actors(v: str | None) -> list[str]:
    if not v:
        return []
    out: list[str] = []
    for p in v.split(","):
        p = p.strip()
        if p:
            out.append(p)
    return out


def render_table(*, title: str, rows: list[tuple[str, str]]) -> str:
    left_w = max([len(k) for k, _v in rows] + [0])
    right_w = max([len(v) for _k, v in rows] + [0])
    inner_w = max(len(title), left_w + 3 + right_w)
    top = "┌" + ("─" * (inner_w + 2)) + "┐"
    hdr = "│ " + title.ljust(inner_w) + " │"
    mid = "├" + ("─" * (inner_w + 2)) + "┤"
    body = []
    for k, v in rows:
        body.append("│ " + (k.ljust(left_w) + " : " + v).ljust(inner_w) + " │")
    bot = "└" + ("─" * (inner_w + 2)) + "┘"
    return "\n".join([top, hdr, mid, *body, bot])


def cmd_trigger(args: argparse.Namespace) -> int:
    actor = default_actor()
    bus_dir = default_bus_dir(args.bus_dir)
    t0 = time.time()
    req_id = str(uuid.uuid4())
    trigger_payload = {"req_id": req_id, "message": args.message, "iso": now_iso_utc(), "ts": t0}
    emit_bus(
        bus_dir,
        topic="pluribus.check.trigger",
        kind="request",
        level="warn",
        actor=actor,
        data=trigger_payload,
    )

    if args.no_watch:
        return 0

    expected = parse_expected_actors(args.expect_actors)
    events_path = Path(bus_dir) / "events.ndjson"
    seen: dict[str, dict] = {}

    def record_report(obj: dict) -> None:
        actor_name = str(obj.get("actor") or "")
        if not actor_name:
            return
        if actor_name not in seen:
            seen[actor_name] = obj

    timeout_s = max(0.0, float(args.timeout_s))
    poll_s = max(0.05, float(args.poll))

    for obj in follow_topic(
        events_path,
        topic="pluribus.check.report",
        since_ts=t0,
        timeout_s=timeout_s,
        poll_s=poll_s,
        follow=bool(args.follow),
    ):
        record_report(obj)
        if expected and all(a in seen for a in expected):
            break

    duration_s = max(0.0, time.time() - t0)
    missing = [a for a in expected if a not in seen]
    result = {
        "trigger": trigger_payload,
        "reports": {"count": len(seen), "actors": sorted(seen.keys())},
        "expected": expected,
        "missing": missing,
        "duration_s": duration_s,
        "bus_dir": bus_dir,
    }
    emit_bus(bus_dir, topic="pluribus.check.result", kind="metric", level="info", actor=actor, data=result)

    if not args.quiet:
        rows = [
            ("Message", str(args.message)),
            ("Reports", str(len(seen))),
            ("Actors", ", ".join(sorted(seen.keys())) or "(none)"),
            ("Expected", ", ".join(expected) or "(none)"),
            ("Missing", ", ".join(missing) or "(none)"),
            ("Bus", f"{bus_dir}/events.ndjson"),
            ("Duration", f"{duration_s:.2f}s"),
        ]
        sys.stdout.write(render_table(title="PLURIBUSCHECK RESULT", rows=rows) + "\n")

    if args.strict and (missing or (args.min_reports and len(seen) < int(args.min_reports))):
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    actor = default_actor()
    bus_dir = default_bus_dir(args.bus_dir)
    root = Path(args.root).expanduser().resolve() if args.root else rhizome_root()
    depth = queue_depth(root)
    status = args.status or ("idle" if depth == 0 else "working")
    trigger_req_id = os.environ.get("PLURIBUSCHECK_TRIGGER_REQ_ID") or None

    # Canonical “what should an agent see” set (repo-visible, stable).
    pluribus_root = Path("/pluribus")
    specs_dir = pluribus_root / "nucleus" / "specs"
    docs_dir = pluribus_root / "nucleus" / "docs"
    dashboard_specs_dir = pluribus_root / "nucleus" / "dashboard" / "specs"
    intake_paths = {
        "agents_root": pluribus_root / "AGENTS.md",
        "agents_nucleus": pluribus_root / "nucleus" / "AGENTS.md",
        "constitution": pluribus_root / "nucleus" / "CONSTITUTION.md",
        "security": pluribus_root / "SECURITY.md",
        "essential_stack": pluribus_root / "ESSENTIAL_STACK.md",
        "agent_fallback_modes": pluribus_root / "agent_fallback_modes.md",
        "lens_collimator_subproject": pluribus_root / "lens-collimator-subproject.md",
        "semops": specs_dir / "semops.json",
        "lexicon": specs_dir / "pluribus_lexicon.md",
        "idiolect": specs_dir / "idiolect.json",
        "personas": specs_dir / "personas.json",
        "lens_spec": specs_dir / "lens_collimator.md",
        "effects_typing_rules": specs_dir / "effects_typing_rules.md",
        "omega_liveness_spec": specs_dir / "omega_liveness_spec.md",
        "git_semantics_isomorphic_only": specs_dir / "git_semantics_isomorphic_only.md",
        "pqc_quine_architecture": specs_dir / "pqc_quine_architecture.md",
        "pqc_git_flow": specs_dir / "pqc_git_flow.md",
        "auth_fallback_policy": specs_dir / "auth_fallback_policy.md",
        "sky_architecture": specs_dir / "sky_architecture.md",
        "sky_signaling_protocol": specs_dir / "sky_signaling_protocol.md",
        "a2a_taxonomy": specs_dir / "a2a_request_taxonomy.md",
        "a2a_workflow": docs_dir / "workflows" / "a2a_bridge.md",
        "sota_catalog": docs_dir / "sota_tools_catalog.md",
        "sota_mapping": docs_dir / "sota_pluribus_mapping.md",
        "sota_integration_plan": docs_dir / "sota_integration_plan.md",
        "unified_dashboard_spec": dashboard_specs_dir / "UNIFIED_DASHBOARD.md",
        "isomorphic_dsl_spec": dashboard_specs_dir / "ISOMORPHIC_DSL.md",
        "iso_git_tool": pluribus_root / "nucleus" / "tools" / "iso_git.mjs",
        "iso_pqc_tool": pluribus_root / "nucleus" / "tools" / "iso_pqc.mjs",
        "dashboard_startup_tool": pluribus_root / "nucleus" / "tools" / "dashboard_start.py",
    }
    semops_meta: dict = {}
    try:
        semops_obj = json.loads((specs_dir / "semops.json").read_text(encoding="utf-8"))
        semops_meta = {
            "protocol_version": semops_obj.get("protocol_version"),
            "schema_version": semops_obj.get("schema_version"),
            "updated_iso": semops_obj.get("updated_iso"),
            "operators": sorted((semops_obj.get("operators") or {}).keys()),
        }
    except Exception:
        semops_meta = {}

    runtime = {
        "ports": {
            "dashboard_dev_5173": tcp_listening("127.0.0.1", 5173),
            "bus_bridge_api_9201": tcp_listening("127.0.0.1", 9201),
            "git_server_9300": tcp_listening("127.0.0.1", 9300),
        }
    }
    idx = root / ".pluribus" / "index"
    store = {
        "bus_events": file_fingerprint(Path(bus_dir) / "events.ndjson"),
        "index_requests": file_fingerprint(idx / "requests.ndjson"),
        "index_responses": file_fingerprint(idx / "responses.ndjson"),
        "index_sota": file_fingerprint(idx / "sota.ndjson"),
        "index_rag": file_fingerprint(idx / "rag.sqlite3"),
        "index_kg_nodes": file_fingerprint(idx / "kg_nodes.ndjson"),
        "index_kg_edges": file_fingerprint(idx / "kg_edges.ndjson"),
    }
    recent = recent_files(
        roots=[
            pluribus_root / "nucleus" / "specs",
            pluribus_root / "nucleus" / "docs",
            pluribus_root / "nucleus" / "tools",
            pluribus_root / "agent_reports",
            pluribus_root / "agent_logs",
            pluribus_root / "SECURITY.md",
        ],
        since_ts=time.time() - (24 * 60 * 60),
        limit=50,
    )

    payload = {
        "topic": "pluribus.check.report",
        "kind": "metric",
        "actor": actor,
        "data": {
            "req_id": trigger_req_id,
            "status": status,
            "queue_depth": depth,
            "current_task": {"goal": args.goal or "unspecified", "step": args.step or "", "started_iso": args.started_iso or ""},
            "blockers": [b for b in (args.blocker or []) if b.strip()],
            "context": {"cwd": os.getcwd(), "rhizome_root": str(root)},
            "intake": {
                "fingerprints": {k: file_fingerprint(p) for k, p in intake_paths.items()},
                "semops": semops_meta,
            },
            "trigger": {
                "req_id": trigger_req_id,
                "id": os.environ.get("PLURIBUSCHECK_TRIGGER_ID") or None,
                "iso": os.environ.get("PLURIBUSCHECK_TRIGGER_ISO") or None,
                "ts": os.environ.get("PLURIBUSCHECK_TRIGGER_TS") or None,
                "actor": os.environ.get("PLURIBUSCHECK_TRIGGER_ACTOR") or None,
                "message": os.environ.get("PLURIBUSCHECK_TRIGGER_MESSAGE") or None,
            },
            "runtime": runtime,
            "stores": store,
            "recent_24h": recent,
            "health": args.health or "nominal",
        },
    }
    emit_bus(bus_dir, topic="pluribus.check.report", kind="metric", level="info", actor=actor, data=payload["data"])
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except BrokenPipeError:
        return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pluribus_check.py", description="PLURIBUSCHECK trigger/report helper (bus-first).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = ap.add_subparsers(dest="cmd", required=False)

    trig = sub.add_parser("trigger", help="Emit pluribus.check.trigger and optionally watch for reports.")
    trig.add_argument("--message", default="Manual trigger", help="Context for the check.")
    trig.add_argument("--no-watch", action="store_true", help="Do not tail for reports.")
    trig.add_argument("--timeout-s", default="5", help="How long to watch for reports (default: 5).")
    trig.add_argument("--poll", default="0.25", help="Poll interval seconds (default: 0.25).")
    trig.add_argument("--follow", action="store_true", help="Follow indefinitely (ignores --timeout-s).")
    trig.add_argument("--expect-actors", default=None, help="Comma-separated actor names expected to report.")
    trig.add_argument("--min-reports", type=int, default=0, help="Minimum reports required when --strict is set (default: 0).")
    trig.add_argument("--strict", action="store_true", help="Exit non-zero if expectations are not met.")
    trig.add_argument("--quiet", action="store_true", help="Do not print the result table.")
    trig.set_defaults(func=cmd_trigger)

    rep = sub.add_parser("report", help="Emit a pluribus.check.report (useful for manual compliance).")
    rep.add_argument("--root", default=None, help="Rhizome root (default: /pluribus if present).")
    rep.add_argument("--status", default=None, help="working|idle|blocked|error (default: inferred from queue depth).")
    rep.add_argument("--health", default=None, help="nominal|degraded|critical")
    rep.add_argument("--goal", default=None)
    rep.add_argument("--step", default=None)
    rep.add_argument("--started-iso", dest="started_iso", default=None)
    rep.add_argument("--blocker", action="append", default=[])
    rep.set_defaults(func=cmd_report)

    ap.set_defaults(func=cmd_trigger)
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
