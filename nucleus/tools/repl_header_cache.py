#!/usr/bin/env python3
"""
REPL Header Cache - 15m TTL cache for agent header attestations.

Usage:
    repl_header_cache.py status [--json]
    repl_header_cache.py get --agent codex [--allow-expired]
    repl_header_cache.py put --agent codex --header-json '{"..."}' [--ttl 900]
    repl_header_cache.py invalidate --agent codex
    repl_header_cache.py expire-all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_CACHE_PATH = "/pluribus/.pluribus/index/repl_header_cache.json"
DEFAULT_TTL_SECONDS = 900
PROTOCOL_VERSIONS_PATH = "/pluribus/nucleus/specs/cagent_adaptations.json"

try:
    from repl_header_audit import validate_header_obj
except ImportError:
    validate_header_obj = None


def now_iso_utc(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def parse_iso(iso: str) -> Optional[float]:
    try:
        return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def resolve_cache_path() -> Path:
    override = (os.environ.get("REPL_HEADER_CACHE_PATH") or "").strip()
    return Path(override) if override else Path(DEFAULT_CACHE_PATH)


def load_protocol_versions() -> Dict[str, str]:
    try:
        data = json.loads(Path(PROTOCOL_VERSIONS_PATH).read_text(encoding="utf-8"))
        versions = data.get("protocol_versions", {})
        if isinstance(versions, dict):
            return {k: str(v) for k, v in versions.items()}
    except Exception:
        return {}
    return {}


def default_cache(current_versions: Dict[str, str]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "ttl_seconds": DEFAULT_TTL_SECONDS,
        "protocol_versions": current_versions,
        "updated_iso": now_iso_utc(),
        "entries": {},
        "last_reset_reason": None,
        "last_reset_iso": None,
    }


def load_cache(path: Path, current_versions: Dict[str, str]) -> Dict[str, Any]:
    if not path.exists():
        return default_cache(current_versions)
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        cache = default_cache(current_versions)

    cached_versions = cache.get("protocol_versions", {})
    if cached_versions != current_versions:
        cache = default_cache(current_versions)
        cache["last_reset_reason"] = "protocol_versions_changed"
        cache["last_reset_iso"] = now_iso_utc()
    return cache


def save_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache["updated_iso"] = now_iso_utc()
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_header(header: Dict[str, Any]) -> Tuple[bool, str]:
    if validate_header_obj:
        ok, reason = validate_header_obj(header)
        return ok, reason
    required = ["contract", "agent", "dkin_version", "paip_version", "citizen_version", "attestation"]
    for key in required:
        if key not in header:
            return False, f"missing:{key}"
    att = header.get("attestation", {})
    if not isinstance(att, dict):
        return False, "attestation_not_object"
    if att.get("score") != "100/100":
        return False, "attestation_score_not_100"
    return True, "ok"


def put_entry(cache: Dict[str, Any], *, agent: str, header: Dict[str, Any], ttl: int, now_ts: Optional[float] = None) -> Dict[str, Any]:
    if now_ts is None:
        now_ts = time.time()
    cached_at = now_iso_utc(now_ts)
    expires_at = now_iso_utc(now_ts + ttl)
    entry = {
        "agent": agent,
        "header": header,
        "cached_at": cached_at,
        "expires_at": expires_at,
        "attestation_event_id": header.get("attestation", {}).get("event_id"),
    }
    cache.setdefault("entries", {})[agent] = entry
    return entry


def get_entry(cache: Dict[str, Any], *, agent: str, now_ts: Optional[float] = None, allow_expired: bool = False) -> Optional[Dict[str, Any]]:
    if now_ts is None:
        now_ts = time.time()
    entry = cache.get("entries", {}).get(agent)
    if not entry:
        return None
    expires_at = parse_iso(entry.get("expires_at", ""))
    if expires_at is None:
        return None if not allow_expired else entry
    if now_ts >= expires_at and not allow_expired:
        return None
    return entry


def remaining_seconds(entry: Dict[str, Any], now_ts: Optional[float] = None) -> Optional[int]:
    if now_ts is None:
        now_ts = time.time()
    expires_at = parse_iso(entry.get("expires_at", ""))
    if expires_at is None:
        return None
    return max(0, int(expires_at - now_ts))


def print_status(cache: Dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(cache, indent=2))
        return
    print("REPL Header Cache")
    print(f"  updated: {cache.get('updated_iso')}")
    entries = cache.get("entries", {})
    if not entries:
        print("  entries: none")
        return
    for agent, entry in entries.items():
        remaining = remaining_seconds(entry)
        status = "expired" if remaining == 0 else "valid"
        print(f"  {agent}: {status} (remaining_s={remaining})")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="REPL Header Cache (TTL)")
    sub = ap.add_subparsers(dest="command", required=True)

    status_p = sub.add_parser("status")
    status_p.add_argument("--json", action="store_true")

    get_p = sub.add_parser("get")
    get_p.add_argument("--agent", required=True)
    get_p.add_argument("--allow-expired", action="store_true")
    get_p.add_argument("--json", action="store_true")

    put_p = sub.add_parser("put")
    put_p.add_argument("--agent", required=True)
    put_p.add_argument("--header-json", required=True, help="JSON string or '-' for stdin")
    put_p.add_argument("--ttl", type=int, default=DEFAULT_TTL_SECONDS)

    inv_p = sub.add_parser("invalidate")
    inv_p.add_argument("--agent", required=True)

    sub.add_parser("expire-all")

    args = ap.parse_args(argv)
    cache_path = resolve_cache_path()
    current_versions = load_protocol_versions()
    cache = load_cache(cache_path, current_versions)

    if args.command == "status":
        print_status(cache, as_json=args.json)
        return 0

    if args.command == "get":
        entry = get_entry(cache, agent=args.agent, allow_expired=args.allow_expired)
        if not entry:
            return 1
        if args.json:
            print(json.dumps(entry, indent=2))
        else:
            print(json.dumps(entry.get("header", {}), ensure_ascii=False))
        return 0

    if args.command == "put":
        raw = args.header_json
        if raw.strip() == "-":
            raw = sys.stdin.read()
        try:
            header = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"invalid header json: {exc}", file=sys.stderr)
            return 2
        ok, reason = validate_header(header)
        if not ok:
            print(f"header validation failed: {reason}", file=sys.stderr)
            return 2
        agent = args.agent.strip()
        if header.get("agent") and header.get("agent") != agent:
            print("header agent mismatch", file=sys.stderr)
            return 2
        put_entry(cache, agent=agent, header=header, ttl=args.ttl)
        save_cache(cache_path, cache)
        return 0

    if args.command == "invalidate":
        cache.get("entries", {}).pop(args.agent, None)
        save_cache(cache_path, cache)
        return 0

    if args.command == "expire-all":
        cache["entries"] = {}
        save_cache(cache_path, cache)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
