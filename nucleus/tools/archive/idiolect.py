#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def _find_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    if (Path("/pluribus") / ".pluribus" / "rhizome.json").exists():
        return Path("/pluribus")
    return cur


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_idiolect(obj: dict[str, Any]) -> tuple[bool, list[str]]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return False, ["root must be object"]
    if obj.get("schema_version") != 1:
        errs.append("schema_version must be 1")
    if not isinstance(obj.get("updated_iso"), str) or not obj.get("updated_iso"):
        errs.append("updated_iso must be non-empty string")
    lex = obj.get("lexicon")
    if not isinstance(lex, list) or not lex:
        errs.append("lexicon must be non-empty array")
    grams = obj.get("grammars")
    if not isinstance(grams, dict) or "plurichat_repl" not in grams:
        errs.append("grammars.plurichat_repl required")
    priors = obj.get("priors")
    if not isinstance(priors, dict):
        errs.append("priors required")
    return len(errs) == 0, errs


def cmd_validate(args: argparse.Namespace) -> int:
    root = _find_root(Path.cwd())
    path = Path(args.path).expanduser().resolve() if args.path else (root / "nucleus" / "specs" / "idiolect.json")
    obj = load(path)
    ok, errs = validate_idiolect(obj)
    sys.stdout.write(json.dumps({"path": str(path), "ok": ok, "errors": errs}, ensure_ascii=False, indent=2) + "\n")
    return 0 if ok else 2


def cmd_glossary(args: argparse.Namespace) -> int:
    root = _find_root(Path.cwd())
    path = Path(args.path).expanduser().resolve() if args.path else (root / "nucleus" / "specs" / "idiolect.json")
    obj = load(path)
    ok, errs = validate_idiolect(obj)
    if not ok:
        sys.stderr.write("invalid idiolect: " + "; ".join(errs) + "\n")
        return 2
    for entry in obj.get("lexicon", []):
        term = entry.get("term")
        cat = entry.get("category")
        definition = entry.get("definition")
        if term and definition:
            sys.stdout.write(f"{term} [{cat}]: {definition}\n")
    return 0


def cmd_grammar(args: argparse.Namespace) -> int:
    root = _find_root(Path.cwd())
    path = Path(args.path).expanduser().resolve() if args.path else (root / "nucleus" / "specs" / "idiolect.json")
    obj = load(path)
    ok, errs = validate_idiolect(obj)
    if not ok:
        sys.stderr.write("invalid idiolect: " + "; ".join(errs) + "\n")
        return 2
    g = (obj.get("grammars") or {}).get("plurichat_repl") or {}
    sys.stdout.write(str(g.get("ebnf") or "") + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="idiolect.py", description="Pluribus idiolect/lexicon + grammar utilities.")
    p.add_argument("--path", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.set_defaults(func=cmd_validate)
    g = sub.add_parser("glossary")
    g.set_defaults(func=cmd_glossary)
    eb = sub.add_parser("grammar")
    eb.set_defaults(func=cmd_grammar)
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

