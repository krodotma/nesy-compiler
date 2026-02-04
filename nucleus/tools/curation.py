#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_state_dir() -> str:
    state_home = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return os.path.join(state_home, "nucleus", "curation")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_ndjson(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def iter_ndjson(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "unknown"


def parse_tags(raw: str | None) -> list[str]:
    seen: set[str] = set()
    tags: list[str] = []
    for part in (raw or "").split(","):
        tag = part.strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def resolve_id_prefix(index_path: str, prefix: str) -> str | None:
    prefix = prefix.strip()
    if not prefix:
        return None
    matches: list[str] = []
    for obj in iter_ndjson(index_path):
        item_id = obj.get("id")
        if not isinstance(item_id, str):
            continue
        if item_id == prefix or item_id.startswith(prefix):
            matches.append(item_id)
    if len(matches) == 1:
        return matches[0]
    return None


def infer_kind_from_url(url: str) -> str:
    u = url.lower()
    if "youtu.be/" in u or "youtube.com/" in u:
        return "yt"
    if "arxiv.org/" in u or "doi.org/" in u:
        return "paper"
    if "github.com/" in u:
        return "code"
    return "other"


def emit_bus_item(bus_dir: str, topic: str, item: dict) -> None:
    subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "agent_bus.py"),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            "artifact",
            "--data",
            json.dumps(item, ensure_ascii=False),
        ],
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cmd_add(args: argparse.Namespace) -> int:
    item_id = str(uuid.uuid4())
    tags = parse_tags(args.tags)

    item = {
        "id": item_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": args.kind,
        "url": args.url,
        "title": args.title,
        "tags": tags,
        "summary": args.summary,
        "claims": args.claims,
        "gaps": args.gaps,
        "hypotheses": args.hypotheses,
        "falsifiers": args.falsifiers,
        "uncertainty": {
            "aleatoric": args.aleatoric,
            "epistemic": args.epistemic,
            "economic": args.economic,
        },
        "provenance": {
            "added_by": default_actor(),
            "context": args.context,
        },
    }

    append_ndjson(args.index, item)
    if args.emit_bus:
        bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        if bus_dir:
            emit_bus_item(bus_dir, "curation.item.added", item)
    sys.stdout.write(item_id + "\n")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    resolved = resolve_id_prefix(args.index, args.ref)
    if not resolved:
        sys.stderr.write("not found (or ambiguous id prefix)\n")
        return 1

    ann_id = str(uuid.uuid4())
    tags = parse_tags(args.tags)
    annotation = {
        "id": ann_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "annotation",
        "ref": resolved,
        "note": args.note,
        "tags": tags,
        "claims": args.claims,
        "gaps": args.gaps,
        "hypotheses": args.hypotheses,
        "falsifiers": args.falsifiers,
        "provenance": {
            "added_by": default_actor(),
            "context": args.context,
        },
    }

    append_ndjson(args.index, annotation)
    if args.emit_bus:
        bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        if bus_dir:
            emit_bus_item(bus_dir, "curation.item.annotated", annotation)
    sys.stdout.write(ann_id + "\n")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for obj in iter_ndjson(args.index):
        if obj.get("kind") == "annotation" and args.kind is None:
            continue
        if args.kind and obj.get("kind") != args.kind:
            continue
        if args.tag and args.tag not in (obj.get("tags") or []):
            continue
        sys.stdout.write(f"{obj.get('iso','')}  {obj.get('kind',''):<8}  {obj.get('id','')[:8]}  {obj.get('title','')}\n")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    target = None
    resolved = resolve_id_prefix(args.index, args.id)
    if resolved:
        for obj in iter_ndjson(args.index):
            if obj.get("id") == resolved:
                target = obj
                break
    if not target:
        sys.stderr.write("not found\n")
        return 1

    if not args.with_annotations:
        sys.stdout.write(json.dumps(target, indent=2, ensure_ascii=False) + "\n")
        return 0

    annotations: list[dict] = []
    for obj in iter_ndjson(args.index):
        if obj.get("kind") == "annotation" and obj.get("ref") == resolved:
            annotations.append(obj)
    out = {"item": target, "annotations": annotations}
    sys.stdout.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    return 0


def extract_urls_with_titles(path: str) -> list[tuple[str, str | None]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    out: list[tuple[str, str | None]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.search(r"(https?://\S+)", line)
        if not m:
            i += 1
            continue
        url = m.group(1)
        title = line[m.end(1) :].strip() or None
        if title is None:
            j = i + 1
            while j < len(lines):
                cand = lines[j].strip()
                if not cand:
                    j += 1
                    continue
                if cand.upper() in {"YOUTUBE.COM", "GITHUB.COM"}:
                    j += 1
                    continue
                if cand.lower().startswith("author"):
                    j += 1
                    continue
                if "http://" in cand or "https://" in cand:
                    j += 1
                    continue
                title = cand
                break
        out.append((url, title))
        i += 1
    return out


def cmd_import_urls(args: argparse.Namespace) -> int:
    existing_urls: set[str] = set()
    if args.dedupe:
        for obj in iter_ndjson(args.index):
            url = obj.get("url")
            if isinstance(url, str) and url:
                existing_urls.add(url)

    tags = parse_tags(args.tags)
    imported = 0
    for url, title in extract_urls_with_titles(args.from_file):
        if args.dedupe and url in existing_urls:
            continue
        kind = args.kind or infer_kind_from_url(url)
        item_id = str(uuid.uuid4())
        item = {
            "id": item_id,
            "ts": time.time(),
            "iso": now_iso_utc(),
            "kind": kind,
            "url": url,
            "title": title or url,
            "tags": tags,
            "summary": args.summary,
            "claims": None,
            "gaps": None,
            "hypotheses": None,
            "falsifiers": None,
            "uncertainty": {"aleatoric": None, "epistemic": None, "economic": None},
            "provenance": {"added_by": default_actor(), "context": args.context, "source_file": args.from_file},
        }
        if args.dry_run:
            sys.stdout.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            append_ndjson(args.index, item)
            if args.emit_bus:
                bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
                if bus_dir:
                    emit_bus_item(bus_dir, "curation.item.added", item)
        imported += 1

    if args.print_count:
        sys.stderr.write(f"imported {imported}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="curation.py", description="Append-only discourse curation index (NDJSON).")
    default_index = os.path.join(default_state_dir(), "items.ndjson")
    p.add_argument("--index", default=os.environ.get("PLURIBUS_CURATION_INDEX", default_index))
    sub = p.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add", help="Add one discourse item (append-only).")
    add.add_argument("--kind", required=True, help="yt|paper|blog|talk|code|dataset|other")
    add.add_argument("--url", default=None)
    add.add_argument("--title", required=True)
    add.add_argument("--tags", default="")
    add.add_argument("--summary", default=None)
    add.add_argument("--claims", default=None)
    add.add_argument("--gaps", default=None, help="Epistemic gaps / unknowns")
    add.add_argument("--hypotheses", default=None, help="Testable hypothesis text (or pointer)")
    add.add_argument("--falsifiers", default=None, help="How this could be proven wrong")
    add.add_argument("--aleatoric", type=float, default=None)
    add.add_argument("--epistemic", type=float, default=None)
    add.add_argument("--economic", type=float, default=None)
    add.add_argument("--context", default=None, help="Free-form context (project, branch, experiment, etc.)")
    add.add_argument("--emit-bus", action="store_true", help="Also emit curation.item.added on the agent bus")
    add.add_argument("--bus-dir", default=None, help="Bus dir (only used with --emit-bus)")
    add.set_defaults(func=cmd_add)

    ann = sub.add_parser("annotate", help="Append an annotation for an existing item.")
    ann.add_argument("ref", help="Item id or unique id prefix")
    ann.add_argument("--note", required=True, help="Annotation text (distillation notes, decisions, etc.)")
    ann.add_argument("--tags", default="")
    ann.add_argument("--claims", default=None)
    ann.add_argument("--gaps", default=None)
    ann.add_argument("--hypotheses", default=None)
    ann.add_argument("--falsifiers", default=None)
    ann.add_argument("--context", default=None)
    ann.add_argument("--emit-bus", action="store_true", help="Also emit curation.item.annotated on the agent bus")
    ann.add_argument("--bus-dir", default=None, help="Bus dir (only used with --emit-bus)")
    ann.set_defaults(func=cmd_annotate)

    ls = sub.add_parser("list", help="List items.")
    ls.add_argument("--kind", default=None)
    ls.add_argument("--tag", default=None)
    ls.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="Show an item by id prefix.")
    show.add_argument("id")
    show.add_argument("--with-annotations", action="store_true", help="Include any appended annotations")
    show.set_defaults(func=cmd_show)

    imp = sub.add_parser("import-urls", help="Import URLs (and nearby titles) from a text/markdown file.")
    imp.add_argument("--from", dest="from_file", required=True, help="Path to file containing URLs")
    imp.add_argument("--kind", default=None, help="Force kind for all imports (default: infer from URL)")
    imp.add_argument("--tags", default="")
    imp.add_argument("--summary", default=None)
    imp.add_argument("--context", default=None)
    imp.add_argument("--dedupe", action="store_true", help="Skip URLs already present in the index")
    imp.add_argument("--dry-run", action="store_true", help="Print would-be items (NDJSON) without writing")
    imp.add_argument("--print-count", action="store_true", help="Print imported count to stderr")
    imp.add_argument("--emit-bus", action="store_true", help="Also emit curation.item.added events on the agent bus")
    imp.add_argument("--bus-dir", default=None, help="Bus dir (only used with --emit-bus)")
    imp.set_defaults(func=cmd_import_urls)

    return p


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
