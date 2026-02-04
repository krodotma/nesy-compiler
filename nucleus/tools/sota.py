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
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            kind,
            "--level",
            level,
            "--actor",
            actor,
            "--data",
            json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


@dataclass
class SotaItem:
    id: str
    url: str
    title: str
    org: str = ""
    region: str = ""
    type: str = "other"
    priority: int = 3
    cadence_days: int = 14
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    kind: str = "sota_item"
    ts: float = 0.0
    iso: str = ""
    provenance: dict = field(default_factory=dict)

class SotaManager:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / ".pluribus" / "index" / "sota.ndjson"

    def list_items(self) -> list[SotaItem]:
        items = []
        for obj in iter_ndjson(self.path):
            if obj.get("kind") == "sota_item":
                items.append(SotaItem(
                    id=obj.get("id", ""),
                    url=obj.get("url", ""),
                    title=obj.get("title", ""),
                    org=obj.get("org", ""),
                    region=obj.get("region", ""),
                    type=obj.get("type", "other"),
                    priority=int(obj.get("priority", 3)),
                    cadence_days=int(obj.get("cadence_days", 14)),
                    tags=obj.get("tags", []),
                    notes=obj.get("notes", ""),
                    kind=obj.get("kind", "sota_item"),
                    ts=float(obj.get("ts", 0.0)),
                    iso=obj.get("iso", ""),
                    provenance=obj.get("provenance", {})
                ))
        return items

    def add_item(self, item: SotaItem) -> None:
        if not item.id:
            item.id = str(uuid.uuid4())
        if not item.ts:
            item.ts = time.time()
        if not item.iso:
            item.iso = now_iso_utc()
        
        append_ndjson(self.path, asdict(item))

def resolve_root(raw_root: str | None) -> Path:
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    return find_rhizome_root(Path.cwd()) or Path.cwd().resolve()


def sota_path(root: Path) -> Path:
    return root / ".pluribus" / "index" / "sota.ndjson"


def cmd_init(args: argparse.Namespace) -> int:
    mgr = SotaManager(resolve_root(args.root))
    ensure_dir(mgr.path.parent)
    mgr.path.touch(exist_ok=True)
    sys.stdout.write(str(mgr.path) + "\n")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    mgr = SotaManager(root)
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    tags = [t for t in (args.tag or []) if t.strip()]

    item = SotaItem(
        id=str(uuid.uuid4()),
        url=args.url,
        title=args.title,
        org=args.org,
        region=args.region,
        type=args.kind,
        priority=int(args.priority),
        cadence_days=int(args.cadence_days),
        tags=tags,
        notes=args.notes,
        provenance={"added_by": actor, "context": args.context},
    )
    mgr.add_item(item)
    
    emit_bus(bus_dir, topic="sota.item.added", kind="artifact", level="info", actor=actor, data=asdict(item))
    sys.stdout.write(item.id + "\n")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    mgr = SotaManager(resolve_root(args.root))
    for item in mgr.list_items():
        sys.stdout.write(f"{item.iso}\t{item.priority}\t{item.org}\t{item.type}\t{item.title}\n")
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    requests_path = root / ".pluribus" / "index" / "requests.ndjson"
    ensure_dir(requests_path.parent)

    now = time.time()
    last_by_item: dict[str, float] = {}
    for obj in iter_ndjson(requests_path):
        item_id = (obj.get("inputs") or {}).get("sota_item_id")
        if isinstance(item_id, str):
            last_by_item[item_id] = max(last_by_item.get(item_id, 0.0), float(obj.get("ts") or 0.0))

    emitted = 0
    for raw in iter_ndjson(sota_path(root)):
        if raw.get("kind") != "sota_item":
            continue
        item_id = raw.get("id")
        if not isinstance(item_id, str) or not item_id:
            continue
        cadence = max(1, int(raw.get("cadence_days") or 14))
        last = last_by_item.get(item_id, 0.0)
        due = (now - last) >= (cadence * 86400.0)
        if not due and not args.force:
            continue

        payload = build_sota_distill_request(
            root=root,
            actor=actor,
            provider=args.provider,
            item={
                "id": item_id,
                "url": raw.get("url"),
                "title": raw.get("title"),
            },
        )
        append_ndjson(requests_path, payload)
        emitted += 1
        if args.emit_bus:
            emit_bus(bus_dir, topic="strp.request.distill", kind="request", level="info", actor=actor, data=payload)

    sys.stdout.write(f"emitted {emitted}\n")
    return 0


def build_sota_distill_request(*, root: Path, actor: str, provider: str, item: dict) -> dict:
    """Build a STRp distillation request for a single SOTA item (pure function)."""
    now = time.time()
    req_id = str(uuid.uuid4())
    item_id = str(item.get("id") or "")
    url = str(item.get("url") or "")
    title = str(item.get("title") or "")
    return {
        "req_id": req_id,
        "ts": now,
        "iso": now_iso_utc(),
        "actor": actor,
        "goal": f"Distill + extract levers for SOTA item: {title}",
        "kind": "distill",
        "provider_hint": provider,
        "inputs": {"sota_item_id": item_id, "url": url, "title": title},
        "constraints": {"levers_required": True, "gates": ["P", "E", "L", "R", "Q"]},
        "sextet": None,
        "rhizome_root": str(root),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sota.py", description="SOTA stream tracker: append-only items + STRp request emission (non-blocking).")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="Create .pluribus/index/sota.ndjson if missing.")
    i.set_defaults(func=cmd_init)

    a = sub.add_parser("add", help="Append a SOTA source item.")
    a.add_argument("--url", required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--org", default=None, help="mit|ucla|tencent|... (free-form)")
    a.add_argument("--region", default=None, help="us|cn|de|uk|fr|ru|ua|... (free-form)")
    a.add_argument("--kind", default="other", help="paper|yt|repo|blog|talk|dataset|other")
    a.add_argument("--priority", default="3", help="1=highest .. 5=lowest")
    a.add_argument("--cadence-days", default="14")
    a.add_argument("--tag", action="append", default=["sota"])
    a.add_argument("--notes", default=None)
    a.add_argument("--context", default=None)
    a.set_defaults(func=cmd_add)

    ls = sub.add_parser("list", help="List SOTA items.")
    ls.set_defaults(func=cmd_list)

    t = sub.add_parser("tick", help="Emit STRp requests for due SOTA items (writes .pluribus/index/requests.ndjson).")
    # Default to codex-cli (naive, first-party channel) to avoid cached quota/auth failures from
    # Default to `auto` so the control-plane fallback chain (preferring web sessions) can choose.
    t.add_argument("--provider", default="auto")
    t.add_argument("--emit-bus", action="store_true")
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_tick)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
