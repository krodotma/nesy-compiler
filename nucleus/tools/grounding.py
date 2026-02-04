#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def _json_load_maybe(value: str):
    value = (value or "").strip()
    if not value:
        raise ValueError("empty input")
    if value == "-":
        return json.load(sys.stdin)
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8", errors="replace"))
    return json.loads(value)


def _coerce_output(obj: object) -> dict | None:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def extract_citation_refs(output: dict) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    citations = output.get("citations")
    if citations is None:
        return [], []

    refs: list[str] = []

    if isinstance(citations, str):
        c = citations.strip()
        if not c:
            issues.append("empty_citation")
        else:
            refs.append(c)
        return refs, issues

    if not isinstance(citations, list):
        return [], ["invalid_citations_type"]

    for entry in citations:
        if isinstance(entry, str):
            c = entry.strip()
            if not c:
                issues.append("empty_citation")
            else:
                refs.append(c)
            continue
        if isinstance(entry, dict):
            ref = entry.get("ref") or entry.get("url") or entry.get("id") or entry.get("source")
            if isinstance(ref, str) and ref.strip():
                refs.append(ref.strip())
            else:
                issues.append("invalid_citation_entry")
            continue
        issues.append("invalid_citation_entry")

    return refs, issues


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _resolve_root(root: str | None) -> Path | None:
    if not root:
        return None
    try:
        return Path(root).expanduser().resolve()
    except Exception:
        return None


def _rag_db_path(root: Path) -> Path:
    return root / ".pluribus" / "index" / "rag.sqlite3"


def _rhizome_object_path(root: Path, sha256: str) -> Path:
    return root / ".pluribus" / "objects" / sha256


def _citation_kind(ref: str) -> tuple[str, str]:
    r = (ref or "").strip()
    if r.startswith("http://") or r.startswith("https://"):
        return "url", r
    if r.startswith("rag:"):
        return "rag", r[len("rag:") :].strip()
    if r.startswith("sha256:"):
        return "sha256", r[len("sha256:") :].strip()
    if r.startswith("rhizome:"):
        return "sha256", r[len("rhizome:") :].strip()
    if r.startswith("artifact:"):
        return "sha256", r[len("artifact:") :].strip()
    if _HEX64_RE.match(r.lower()):
        return "sha256", r.lower()
    if ("/" in r or r.startswith(".")) and Path(r).expanduser().exists():
        return "path", r
    return "other", r


def _rag_has_doc_id(root: Path, doc_id: str) -> bool:
    db_path = _rag_db_path(root)
    if not db_path.exists():
        return False
    try:
        con = sqlite3.connect(str(db_path))
        try:
            row = con.execute("SELECT 1 FROM docs WHERE doc_id = ? LIMIT 1", (doc_id,)).fetchone()
            return bool(row)
        finally:
            con.close()
    except Exception:
        return False


def _rhizome_has_object(root: Path, sha256: str) -> bool:
    if not _HEX64_RE.match((sha256 or "").lower()):
        return False
    return _rhizome_object_path(root, sha256.lower()).exists()


def verify_grounded_output(
    output: dict | str,
    *,
    require_citations: bool = False,
    validate_refs: bool = False,
    root: str | None = None,
) -> tuple[bool, list[str]]:
    parsed = _coerce_output(output)
    if parsed is None:
        return False, ["invalid_json_object"]

    refs, issues = extract_citation_refs(parsed)
    if require_citations and not refs:
        issues.append("missing_citations")

    if validate_refs and refs:
        rroot = _resolve_root(root)
        if not rroot:
            issues.append("missing_root_for_validation")
        else:
            for ref in refs:
                kind, value = _citation_kind(ref)
                if kind in {"url", "path"}:
                    continue
                if kind == "rag":
                    if not value or not _rag_has_doc_id(rroot, value):
                        issues.append("citation_not_found_rag")
                    continue
                if kind == "sha256":
                    if not value or not _rhizome_has_object(rroot, value):
                        issues.append("citation_not_found_object")
                    continue
                # Unknown local reference types are treated as invalid when validating.
                issues.append("invalid_citation_ref")

    ok = len(issues) == 0
    return ok, issues


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        obj = _json_load_maybe(args.input)
    except Exception as e:
        sys.stderr.write(f"invalid input: {e}\n")
        return 2

    ok, issues = verify_grounded_output(
        obj,
        require_citations=bool(args.require_citations),
        validate_refs=bool(args.validate_refs),
        root=args.root,
    )
    out = {"ok": ok, "issues": issues}
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grounding.py", description="Verify grounded outputs (e.g., citations present).")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="Verify an output JSON payload is grounded.")
    v.add_argument("--input", required=True, help="JSON string; '-' stdin; '@file.json' to load.")
    v.add_argument("--require-citations", action="store_true")
    v.add_argument("--validate-refs", action="store_true", help="Validate local refs (rag:/sha256:) against the given root.")
    v.add_argument("--root", default=None, help="Rhizome root containing .pluribus/ (required for --validate-refs).")
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
