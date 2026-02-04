#!/usr/bin/env python3
from __future__ import annotations

"""
PBASSIMILATE — assimilation screening operator.

Emits a non-blocking screening request for a project or git target, checks for
overlaps with existing semops/tools/membrane entries, and publishes a summary
packet for consensus + planning.
"""

import argparse
import getpass
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
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
    append_ndjson(bus_dir / "events.ndjson", evt)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")


def parse_target(raw: str) -> dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {"raw": "", "kind": "unknown", "name": "", "host": "", "path": ""}

    is_git = raw.startswith("git@") or "://" in raw or raw.endswith(".git")
    if not is_git:
        return {"raw": raw, "kind": "project", "name": raw.strip(), "host": "", "path": ""}

    host = ""
    path = ""
    if raw.startswith("git@"):
        # git@github.com:org/repo.git
        tail = raw.split("@", 1)[1] if "@" in raw else raw
        host = tail.split(":", 1)[0]
        path = tail.split(":", 1)[1] if ":" in tail else ""
    else:
        parsed = urlparse(raw)
        host = parsed.hostname or ""
        path = parsed.path or ""

    name = Path(path).name
    if name.endswith(".git"):
        name = name[:-4]
    return {"raw": raw, "kind": "git", "name": name, "host": host, "path": path}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scan_semops(semops: dict, target_norm: str, overlaps: list[dict]) -> None:
    ops = semops.get("operators") if isinstance(semops.get("operators"), dict) else {}
    for op_key, op_data in ops.items():
        names = [op_key, op_data.get("name", "")]
        for alias in op_data.get("aliases", []) if isinstance(op_data.get("aliases"), list) else []:
            names.append(alias)
        for name in names:
            name_norm = slugify(str(name))
            if not name_norm:
                continue
            if name_norm == target_norm:
                overlaps.append({"source": "semops.operator", "match": op_key, "severity": "conflict"})
                return
            if target_norm in name_norm or name_norm in target_norm:
                overlaps.append({"source": "semops.operator", "match": op_key, "severity": "overlap"})


def scan_sotatools(manifest: dict, target_norm: str, overlaps: list[dict]) -> None:
    tools = manifest.get("tools") if isinstance(manifest.get("tools"), dict) else {}
    for key, entry in tools.items():
        key_norm = slugify(str(key))
        desc = str(entry.get("description") or "")
        if key_norm == target_norm:
            overlaps.append({"source": "sotatools.manifest", "match": key, "severity": "conflict"})
        elif target_norm in key_norm or target_norm in slugify(desc):
            overlaps.append({"source": "sotatools.manifest", "match": key, "severity": "overlap"})


def scan_clade_manifest(clade: dict, target_norm: str, overlaps: list[dict]) -> None:
    membrane = clade.get("membrane") if isinstance(clade.get("membrane"), dict) else {}
    for key, entry in membrane.items():
        key_norm = slugify(str(key))
        remote = str(entry.get("remote") or "")
        remote_norm = slugify(remote)
        if key_norm == target_norm:
            overlaps.append({"source": "clade.membrane", "match": key, "severity": "conflict"})
        elif target_norm in key_norm or (remote_norm and target_norm in remote_norm):
            overlaps.append({"source": "clade.membrane", "match": key, "severity": "overlap"})


def scan_gitmodules(path: Path, target_norm: str, overlaps: list[dict]) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "url" not in line:
            continue
        line_norm = slugify(line)
        if target_norm and target_norm in line_norm:
            overlaps.append({"source": "gitmodules", "match": line.strip(), "severity": "overlap"})
            return


def scan_membrane_dirs(root: Path, target_norm: str, overlaps: list[dict]) -> None:
    mem_dir = root / "membrane"
    if not mem_dir.exists():
        return
    for entry in mem_dir.iterdir():
        if not entry.is_dir():
            continue
        name_norm = slugify(entry.name)
        if name_norm == target_norm:
            overlaps.append({"source": "membrane.dir", "match": entry.name, "severity": "overlap"})


def scan_adapters(root: Path, target_norm: str, overlaps: list[dict]) -> None:
    tools_dir = root / "nucleus" / "tools"
    if not tools_dir.exists():
        return
    for entry in tools_dir.iterdir():
        if not entry.is_file():
            continue
        if not (entry.name.endswith("_adapter.py") or entry.name.endswith("_bridge.py")):
            continue
        if target_norm and target_norm in slugify(entry.name):
            overlaps.append({"source": "adapter", "match": entry.name, "severity": "overlap"})


def build_screening(target: dict[str, str], *, root: Path) -> dict:
    target_norm = slugify(target.get("name") or target.get("raw") or "")
    overlaps: list[dict] = []

    scan_semops(load_json(root / "nucleus" / "specs" / "semops.json"), target_norm, overlaps)
    scan_sotatools(load_json(root / "nucleus" / "specs" / "sotatools_manifest.json"), target_norm, overlaps)
    scan_clade_manifest(load_json(root / ".clade-manifest.json"), target_norm, overlaps)
    scan_gitmodules(root / ".gitmodules", target_norm, overlaps)
    scan_membrane_dirs(root, target_norm, overlaps)
    scan_adapters(root, target_norm, overlaps)

    conflicts = [item for item in overlaps if item.get("severity") == "conflict"]
    if conflicts:
        recommendation = "reject"
        rationale = "Exact match found in existing operators or membrane entries."
    elif overlaps:
        recommendation = "review"
        rationale = "Partial overlap detected; verify non-redundancy before proceeding."
    else:
        recommendation = "proceed"
        rationale = "No known overlaps detected; candidate appears new."

    integration_candidates = [
        {"type": "submodule", "when": "read-only upstream, minimal local patching"},
        {"type": "subtree", "when": "fork/patch expected, local modifications needed"},
        {"type": "package", "when": "npm/pip dependency with pinned versions"},
        {"type": "rhizome", "when": "single-file or prompt-level artifact"},
    ]

    return {
        "target_norm": target_norm,
        "overlaps": overlaps,
        "conflicts": conflicts,
        "recommendation": recommendation,
        "rationale": rationale,
        "integration_candidates": integration_candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbassimilate_operator.py", description="PBASSIMILATE semantic operator: assimilation screening.")
    p.add_argument("--bus-dir", default=None)
    p.add_argument("--actor", default=None)
    p.add_argument("--req-id", default=None)
    p.add_argument("--target", required=True, help="Project name or git URL to screen.")
    p.add_argument("--purpose", default="", help="Short intent for the assimilation request.")
    p.add_argument("--scope", default="sota", help="Scope tag (default: sota).")
    p.add_argument("--consensus-targets", default="", help="Comma-separated consensus agents to consult.")
    p.add_argument("--report-dir", default="", help="Optional report directory (writes JSON screening artifact).")
    p.add_argument("--report-path", default="", help="Optional report path (overrides report-dir).")
    p.add_argument("--no-infer-sync", action="store_true", help="Disable infer_sync mirror.")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "pbassimilate"
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()
    ensure_dir(bus_dir)
    (bus_dir / "events.ndjson").touch(exist_ok=True)

    target = parse_target(args.target)
    if not target.get("raw"):
        sys.stderr.write("PBASSIMILATE requires a non-empty --target.\n")
        return 2

    root = repo_root()
    screening = build_screening(target, root=root)
    req_id = str(args.req_id or uuid.uuid4())
    consensus_targets = [t.strip() for t in str(args.consensus_targets).split(",") if t.strip()]

    payload = {
        "req_id": req_id,
        "iso": now_iso_utc(),
        "scope": str(args.scope),
        "intent": "pbassimilate",
        "target": target,
        "purpose": str(args.purpose or ""),
        "consensus_targets": consensus_targets,
        "screening": screening,
        "constraints": {
            "append_only": True,
            "non_blocking": True,
            "tests_first": True,
            "no_secrets": True,
            "edp_required": True,
        },
        "next_actions": [
            "request_consensus",
            "draft_plan",
            "select_integration_type",
            "prepare_adapter",
        ],
    }

    emit_bus(bus_dir, topic="operator.pbassimilate.request", kind="request", level="info", actor=actor, data=payload)
    emit_bus(bus_dir, topic="operator.pbassimilate.screening", kind="artifact", level="info", actor=actor, data={"req_id": req_id, "screening": screening})
    if not args.no_infer_sync:
        emit_bus(bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data=payload)

    report_path = Path(str(args.report_path)).expanduser().resolve() if args.report_path else None
    if not report_path and args.report_dir:
        report_dir = Path(args.report_dir).expanduser().resolve()
        report_path = report_dir / f"pbassimilate_{req_id}.json"
    if report_path:
        ensure_dir(report_path.parent)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sys.stdout.write(req_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
