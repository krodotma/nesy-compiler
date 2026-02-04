#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "specs" / "cagent_registry.json"


def load_registry(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_token(value: Optional[str], aliases: Dict[str, str], fallback: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return fallback
    return aliases.get(raw, raw)


def normalize_class(value: Optional[str], registry: Dict[str, Any], fallback: str = "") -> str:
    return _normalize_token(value, registry.get("class_aliases", {}), fallback)


def normalize_tier(value: Optional[str], registry: Dict[str, Any], fallback: str = "") -> str:
    return _normalize_token(value, registry.get("tier_aliases", {}), fallback)


def _parse_allowlist(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def env_overrides() -> Dict[str, Any]:
    return {
        "citizen_class": os.environ.get("PLURIBUS_CAGENT_CLASS"),
        "citizen_tier": os.environ.get("PLURIBUS_CITIZEN_TIER"),
        "bootstrap_profile": os.environ.get("PLURIBUS_BOOTSTRAP_PROFILE"),
        "scope_allowlist": os.environ.get("PLURIBUS_SCOPE_ALLOWLIST"),
    }


@dataclass(frozen=True)
class CitizenProfile:
    citizen_class: str
    citizen_tier: str
    bootstrap_profile: str
    scope_allowlist: List[str]
    source: str


def _find_actor_entry(registry: Dict[str, Any], actor: str) -> Optional[Dict[str, Any]]:
    for entry in registry.get("actors", []):
        if entry.get("actor") == actor:
            return entry
    return None


def resolve_actor(
    actor: str,
    registry: Dict[str, Any],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    allow_override: bool = True,
) -> CitizenProfile:
    defaults = registry.get("defaults", {})
    class_aliases = registry.get("class_aliases", {})
    tier_aliases = registry.get("tier_aliases", {})

    base = dict(defaults)
    source = "defaults"
    entry = _find_actor_entry(registry, actor)
    if entry:
        base.update(entry)
        source = "registry"

    data = dict(base)
    if allow_override and overrides:
        for key in ("citizen_class", "citizen_tier", "bootstrap_profile", "scope_allowlist"):
            if overrides.get(key) is not None:
                data[key] = overrides[key]

    citizen_class = _normalize_token(data.get("citizen_class"), class_aliases, "superworker")
    citizen_tier = _normalize_token(data.get("citizen_tier"), tier_aliases, "limited")
    bootstrap_profile = _normalize_token(
        data.get("bootstrap_profile"),
        {"full": "full", "minimal": "minimal"},
        "minimal",
    )
    scope_allowlist = _parse_allowlist(data.get("scope_allowlist"))

    return CitizenProfile(
        citizen_class=citizen_class,
        citizen_tier=citizen_tier,
        bootstrap_profile=bootstrap_profile,
        scope_allowlist=scope_allowlist,
        source=source,
    )


def validate_registry(registry: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not isinstance(registry, dict):
        return False, ["registry must be object"]
    if registry.get("schema_version") != 1:
        errs.append("schema_version must be 1")
    if not isinstance(registry.get("updated_iso"), str) or not registry.get("updated_iso"):
        errs.append("updated_iso must be non-empty string")
    defaults = registry.get("defaults")
    if not isinstance(defaults, dict):
        errs.append("defaults must be object")
    actors = registry.get("actors")
    if not isinstance(actors, list):
        errs.append("actors must be array")
        return False, errs
    seen = set()
    for entry in actors:
        if not isinstance(entry, dict):
            errs.append("actor entry must be object")
            continue
        actor = entry.get("actor")
        if not isinstance(actor, str) or not actor:
            errs.append("actor entry missing actor")
            continue
        if actor in seen:
            errs.append(f"duplicate actor: {actor}")
        seen.add(actor)
        if entry.get("citizen_class") not in {None, "superagent", "superworker"}:
            errs.append(f"{actor}: citizen_class invalid")
        if entry.get("citizen_tier") not in {None, "full", "limited"}:
            errs.append(f"{actor}: citizen_tier invalid")
    return len(errs) == 0, errs


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    registry = load_registry(path)
    ok, errs = validate_registry(registry)
    out = {"path": str(path), "ok": ok, "errors": errs}
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return 0 if ok else 2


def cmd_resolve(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    registry = load_registry(path)
    overrides = env_overrides()
    profile = resolve_actor(args.actor, registry, overrides=overrides, allow_override=True)
    out = {
        "actor": args.actor,
        "citizen_class": profile.citizen_class,
        "citizen_tier": profile.citizen_tier,
        "bootstrap_profile": profile.bootstrap_profile,
        "scope_allowlist": profile.scope_allowlist,
        "source": profile.source,
        "registry_path": str(path),
    }
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cagent_registry.py", description="CAGENT registry resolver/validator.")
    p.add_argument("--path", default=str(_default_registry_path()), help="Registry JSON path")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="Validate registry")
    v.set_defaults(func=cmd_validate)
    r = sub.add_parser("resolve", help="Resolve citizen profile for actor")
    r.add_argument("--actor", required=True)
    r.set_defaults(func=cmd_resolve)
    return p


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
