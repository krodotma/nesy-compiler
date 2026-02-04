#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def _default_personas_path(root: Path) -> Path:
    return root / "nucleus" / "specs" / "personas.json"


def _find_pluribus_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    if (Path("/pluribus") / ".pluribus" / "rhizome.json").exists():
        return Path("/pluribus")
    return cur


def load_personas(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_ID_RE = re.compile(r"^[a-z0-9_.-]+$")


@dataclass(frozen=True)
class PersonaPick:
    persona_id: str
    reason: str


def validate_registry(obj: dict[str, Any]) -> tuple[bool, list[str]]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return False, ["root must be object"]
    if obj.get("schema_version") not in {1, 2}:
        errs.append("schema_version must be 1 or 2")
    if not isinstance(obj.get("updated_iso"), str) or not obj.get("updated_iso"):
        errs.append("updated_iso must be non-empty string")
    taxonomy = obj.get("taxonomy")
    if not isinstance(taxonomy, dict):
        errs.append("taxonomy must be object")
    personas = obj.get("personas")
    if not isinstance(personas, list) or not personas:
        errs.append("personas must be non-empty array")
        return False, errs

    seen: set[str] = set()
    for p in personas:
        if not isinstance(p, dict):
            errs.append("persona entry must be object")
            continue
        pid = p.get("id")
        if not isinstance(pid, str) or not pid or not _ID_RE.match(pid):
            errs.append("persona.id invalid")
            continue
        if pid in seen:
            errs.append(f"duplicate persona.id: {pid}")
        seen.add(pid)
        if not isinstance(p.get("label"), str) or not p.get("label"):
            errs.append(f"{pid}: label required")
        if p.get("provider_class") not in {"ring0", "subagent", "external", "superworker"}:
            errs.append(f"{pid}: provider_class invalid")
        if p.get("depth") not in {"narrow", "deep"}:
            errs.append(f"{pid}: depth invalid")
        lanes = p.get("lanes")
        if not isinstance(lanes, list) or not lanes or any(l not in {"dialogos", "pbpair", "strp"} for l in lanes):
            errs.append(f"{pid}: lanes invalid")
        effects = p.get("effects_budget")
        if not isinstance(effects, list) or not effects or any(e not in {"none", "file", "network", "unknown"} for e in effects):
            errs.append(f"{pid}: effects_budget invalid")
        dom = p.get("domain_tags")
        if not isinstance(dom, list):
            errs.append(f"{pid}: domain_tags must be array")
        standards = p.get("artifact_standards")
        if not isinstance(standards, list) or not standards:
            errs.append(f"{pid}: artifact_standards must be non-empty array")
    return len(errs) == 0, errs


def _infer_domain_tags(goal: str) -> set[str]:
    g = (goal or "").lower()
    tags: set[str] = set()
    if any(w in g for w in ["kroma", "kroma.live"]):
        tags.add("kroma")
    if any(w in g for w in ["avtr", "avatar", "avtr.you", "avtr.world"]):
        tags.add("avtr")
    if any(w in g for w in ["oooo", "oooo.art"]):
        tags.add("oooo")
    if any(w in g for w in ["cinema", "movie", "video", "kareem.movie"]):
        tags.add("cinema")
    if "mirrorstudios" in g or "mirrorstudios.co" in g:
        tags.add("mirrorstudios")
    if any(w in g for w in ["pqc", "ml-kem", "dilithium", "webauthn"]):
        tags.add("pqc")
    if any(w in g for w in ["webrtc", "stun", "turn", "ice"]):
        tags.add("webrtc")
    if any(w in g for w in ["webgpu", "onnx", "quant", "int8", "int4"]):
        tags.add("webgpu")
    if any(w in g for w in ["rag", "retrieval"]):
        tags.add("rag")
    if any(w in g for w in ["kg", "knowledge graph"]):
        tags.add("kg")
    if any(w in g for w in ["cmp", "hgt", "vgt", "lineage", "gödel", "godel"]):
        tags.add("evolution")
    return tags


def choose_persona(registry: dict[str, Any], *, goal: str, depth: str, kind: str, effects: str) -> PersonaPick:
    ok, errs = validate_registry(registry)
    if not ok:
        return PersonaPick("ring0.architect", f"registry_invalid:{','.join(errs[:3])}")

    inferred_tags = _infer_domain_tags(goal)
    personas: list[dict[str, Any]] = registry.get("personas", [])

    # Deterministic scoring:
    # (1) audit_bonus: ring0 preferred for audits
    # (2) effects_ok: exact effects match
    # (3) tag_overlap: domain tag intersection
    # (4) class_weight: prefer lighter-weight classes for generic tasks (subagent > superworker)
    # (5) stable pid for determinism
    scored: list[tuple[int, int, int, int, str, dict[str, Any]]] = []
    # Provider class weight: subagent preferred for narrow/generic tasks, superworker only when tags match
    _CLASS_WEIGHT = {"ring0": 3, "subagent": 2, "external": 1, "superworker": 0}
    for p in personas:
        pid = str(p.get("id") or "")
        if p.get("depth") != depth:
            continue
        effects_budget = set(p.get("effects_budget") or [])
        if effects not in effects_budget and "unknown" not in effects_budget:
            continue
        dom = set(p.get("domain_tags") or [])
        tag_overlap = len(dom & inferred_tags)
        effects_ok = 1 if effects in effects_budget else 0
        # Prefer ring0 for audits regardless of overlap.
        audit_bonus = 1 if (kind == "audit" and p.get("provider_class") == "ring0") else 0
        # Superworkers should only be preferred if there's explicit tag overlap, otherwise prefer subagent.
        pclass = p.get("provider_class") or "subagent"
        class_weight = _CLASS_WEIGHT.get(pclass, 1)
        # Boost superworker only if tags match; penalize if no overlap.
        if pclass == "superworker" and tag_overlap == 0:
            class_weight = -1  # Penalize superworker for generic tasks
        scored.append((audit_bonus, effects_ok, tag_overlap, class_weight, pid, p))

    if not scored:
        fallback = "subagent.narrow_coder" if depth == "narrow" else "ring0.architect"
        return PersonaPick(fallback, "fallback:no_matching_persona")

    scored.sort(reverse=True)
    best = scored[0][5]
    pid = str(best.get("id"))
    why = f"depth={depth};effects={effects};tags={sorted(inferred_tags)}"
    return PersonaPick(pid, why)


def cmd_validate(args: argparse.Namespace) -> int:
    root = _find_pluribus_root(Path.cwd())
    path = Path(args.path).expanduser().resolve() if args.path else _default_personas_path(root)
    obj = load_personas(path)
    ok, errs = validate_registry(obj)
    out = {"path": str(path), "ok": ok, "errors": errs}
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return 0 if ok else 2


def cmd_choose(args: argparse.Namespace) -> int:
    root = _find_pluribus_root(Path.cwd())
    path = Path(args.path).expanduser().resolve() if args.path else _default_personas_path(root)
    obj = load_personas(path)
    depth = (args.depth or "").strip().lower() or "narrow"
    if depth not in {"narrow", "deep"}:
        depth = "narrow"
    kind = (args.kind or "").strip().lower() or "other"
    effects = (args.effects or os.environ.get("PLURIBUS_EFFECTS") or "unknown").strip().lower()
    if effects not in {"none", "file", "network", "unknown"}:
        effects = "unknown"
    pick = choose_persona(obj, goal=args.goal or "", depth=depth, kind=kind, effects=effects)
    sys.stdout.write(json.dumps({"persona_id": pick.persona_id, "reason": pick.reason}, ensure_ascii=False, indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="persona_registry.py", description="Persona taxonomy + chooser (deterministic).")
    p.add_argument("--path", default=None, help="Personas JSON path (default: nucleus/specs/personas.json)")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="Validate persona registry")
    v.set_defaults(func=cmd_validate)
    c = sub.add_parser("choose", help="Choose a persona for a request")
    c.add_argument("--goal", required=True)
    c.add_argument("--depth", default=None)
    c.add_argument("--kind", default=None)
    c.add_argument("--effects", default=None)
    c.set_defaults(func=cmd_choose)
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

