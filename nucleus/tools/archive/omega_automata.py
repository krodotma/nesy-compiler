#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def json_load(spec: str) -> dict:
    spec = spec.strip()
    if spec == "-":
        return json.load(sys.stdin)
    if spec.startswith("@"):
        return json.loads(Path(spec[1:]).read_text(encoding="utf-8"))
    return json.loads(spec)


def validate_buchi(obj: dict) -> list[str]:
    errors: list[str] = []
    if obj.get("type") != "buchi":
        errors.append("type must be 'buchi'")
    states = obj.get("states")
    if not isinstance(states, list) or not states:
        errors.append("states must be a non-empty list")
    initial = obj.get("initial")
    if not isinstance(initial, str) or not initial:
        errors.append("initial must be a string")
    accepting = obj.get("accepting")
    if not isinstance(accepting, list):
        errors.append("accepting must be a list")
    transitions = obj.get("transitions")
    if not isinstance(transitions, list):
        errors.append("transitions must be a list")
    return errors


def validate_rabin(obj: dict) -> list[str]:
    errors: list[str] = []
    if obj.get("type") != "rabin":
        errors.append("type must be 'rabin'")
    states = obj.get("states")
    if not isinstance(states, list) or not states:
        errors.append("states must be a non-empty list")
    initial = obj.get("initial")
    if not isinstance(initial, str) or not initial:
        errors.append("initial must be a string")
    pairs = obj.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        errors.append("pairs must be a non-empty list of {E:[...],F:[...]}")
    transitions = obj.get("transitions")
    if not isinstance(transitions, list):
        errors.append("transitions must be a list")
    return errors


def validate(obj: dict) -> list[str]:
    t = obj.get("type")
    if t == "buchi":
        return validate_buchi(obj)
    if t == "rabin":
        return validate_rabin(obj)
    return ["type must be 'buchi' or 'rabin'"]


def render_dot(obj: dict) -> str:
    states = obj.get("states") or []
    accepting = set(obj.get("accepting") or [])
    dot = ["digraph omega {", "  rankdir=LR;"]
    for s in states:
        if not isinstance(s, str):
            continue
        shape = "doublecircle" if s in accepting and obj.get("type") == "buchi" else "circle"
        dot.append(f'  "{s}" [shape={shape}];')
    init = obj.get("initial")
    if isinstance(init, str) and init:
        dot.append('  "__init" [shape=point];')
        dot.append(f'  "__init" -> "{init}";')
    for tr in obj.get("transitions") or []:
        if not isinstance(tr, dict):
            continue
        src = tr.get("from")
        dst = tr.get("to")
        sym = tr.get("on")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        label = str(sym) if sym is not None else ""
        dot.append(f'  "{src}" -> "{dst}" [label="{label}"];')
    dot.append("}")
    return "\n".join(dot) + "\n"


def cmd_validate(args: argparse.Namespace) -> int:
    obj = json_load(args.spec)
    errs = validate(obj)
    if errs:
        for e in errs:
            sys.stderr.write(e + "\n")
        return 2
    sys.stdout.write("ok\n")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    obj = json_load(args.spec)
    errs = validate(obj)
    if errs:
        for e in errs:
            sys.stderr.write(e + "\n")
        return 2
    sys.stdout.write(render_dot(obj))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="omega_automata.py", description="Minimal ω-automata (Büchi/Rabin) spec validator + DOT renderer.")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate")
    v.add_argument("spec", help="JSON string, '-' stdin, or '@file.json'")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("render-dot")
    r.add_argument("spec", help="JSON string, '-' stdin, or '@file.json'")
    r.set_defaults(func=cmd_render)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

