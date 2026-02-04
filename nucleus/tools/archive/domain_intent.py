#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
from domain_registry import append_ndjson, default_actor, emit_bus, iter_ndjson, normalize_domain, registry_path, resolve_root  # noqa: E402


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_intent(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_by_domain(reg_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for obj in iter_ndjson(reg_path):
        d = obj.get("domain")
        if not isinstance(d, str):
            continue
        nd = normalize_domain(d)
        if not nd:
            continue
        latest[nd] = obj
    return latest


def cmd_validate(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    intent_path = Path(args.intent).expanduser().resolve() if args.intent else (root / "nucleus" / "docs" / "ingresses" / "domain_intent.json")
    if not intent_path.exists():
        sys.stderr.write(f"missing intent file: {intent_path}\n")
        return 2

    intent = load_intent(intent_path)
    domains = intent.get("domains")
    if not isinstance(domains, list) or not domains:
        sys.stderr.write("intent must contain non-empty domains[]\n")
        return 2

    reg = registry_path(root)
    latest = latest_by_domain(reg)

    missing: list[str] = []
    mismatched: list[str] = []
    for entry in domains:
        if not isinstance(entry, dict):
            continue
        d = normalize_domain(str(entry.get("domain") or ""))
        if not d:
            continue
        want_tags = [str(t).strip().lower() for t in (entry.get("tags") or []) if str(t).strip()]
        have = latest.get(d)
        if not have:
            missing.append(d)
            continue
        have_tags = [str(t).strip().lower() for t in (have.get("tags") or []) if str(t).strip()]
        if want_tags and not all(t in have_tags for t in want_tags):
            mismatched.append(d)

    ok = len(missing) == 0 and len(mismatched) == 0
    out = {"root": str(root), "intent": str(intent_path), "missing": missing, "tag_mismatch": mismatched}
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return 0 if ok else 1


def cmd_apply(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    intent_path = Path(args.intent).expanduser().resolve() if args.intent else (root / "nucleus" / "docs" / "ingresses" / "domain_intent.json")
    if not intent_path.exists():
        sys.stderr.write(f"missing intent file: {intent_path}\n")
        return 2

    intent = load_intent(intent_path)
    domains = intent.get("domains")
    if not isinstance(domains, list) or not domains:
        sys.stderr.write("intent must contain non-empty domains[]\n")
        return 2

    reg = registry_path(root)
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    appended = 0
    for entry in domains:
        if not isinstance(entry, dict):
            continue
        d = normalize_domain(str(entry.get("domain") or ""))
        if not d:
            continue
        tags = [str(t).strip() for t in (entry.get("tags") or []) if str(t).strip()]
        rec = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "kind": "domain",
            "domain": d,
            "purpose": entry.get("purpose"),
            "tags": tags,
            "source": "intent",
            "notes": entry.get("notes"),
            "provenance": {"added_by": actor, "context": args.context, "intent": str(intent_path)},
        }
        append_ndjson(reg, rec)
        appended += 1
        emit_bus(bus_dir, topic="domains.intent.applied", kind="artifact", level="info", actor=actor, data=rec)

    sys.stdout.write(f"appended {appended}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="domain_intent.py", description="Domain intent (purpose/tags) validator + applier (append-only).")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    p.add_argument("--intent", default=None, help="Path to domain_intent.json (default: nucleus/docs/ingresses/domain_intent.json).")
    p.add_argument("--context", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Validate intent against current registry tags.")
    v.set_defaults(func=cmd_validate)

    a = sub.add_parser("apply", help="Append intent records into the registry (does not mutate).")
    a.set_defaults(func=cmd_apply)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

