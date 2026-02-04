#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


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


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


@dataclass(frozen=True)
class RhizomePaths:
    root: Path
    pluribus_dir: Path
    objects_dir: Path
    index_dir: Path
    manifest_path: Path
    artifacts_index: Path


def resolve_paths(root: str | None) -> RhizomePaths:
    if root:
        r = Path(root).expanduser().resolve()
    else:
        detected = find_rhizome_root(Path.cwd())
        r = detected if detected else Path.cwd().resolve()
    pluribus_dir = r / ".pluribus"
    manifest_path = pluribus_dir / "rhizome.json"
    objects_dir = pluribus_dir / "objects"
    index_dir = pluribus_dir / "index"
    artifacts_index = index_dir / "artifacts.ndjson"
    return RhizomePaths(
        root=r,
        pluribus_dir=pluribus_dir,
        objects_dir=objects_dir,
        index_dir=index_dir,
        manifest_path=manifest_path,
        artifacts_index=artifacts_index,
    )


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            h.update(chunk)
    return h.hexdigest(), total


def is_archive(p: Path) -> bool:
    s = p.name.lower()
    return any(s.endswith(ext) for ext in (".zip", ".tar", ".tar.gz", ".tgz", ".tar.zst", ".zst"))


def artifact_kind(p: Path) -> str:
    if p.is_dir():
        return "dir"
    if p.is_file():
        return "archive" if is_archive(p) else "file"
    return "unknown"


def guess_media_type(p: Path) -> str | None:
    ext = p.suffix.lower().lstrip(".")
    if not ext:
        return None
    if ext in {"md", "txt", "log"}:
        return "text/plain"
    if ext in {"json"}:
        return "application/json"
    if ext in {"pdf"}:
        return "application/pdf"
    if ext in {"png"}:
        return "image/png"
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext in {"mp3"}:
        return "audio/mpeg"
    if ext in {"mp4"}:
        return "video/mp4"
    if ext in {"zst"}:
        return "application/zstd"
    if ext in {"zip"}:
        return "application/zip"
    return None


def emit_bus(*, topic: str, data: dict, bus_dir: str | None) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    argv = [
        sys.executable,
        str(tool),
        "--bus-dir",
        bus_dir,
        "pub",
        "--topic",
        topic,
        "--kind",
        "artifact",
        "--data",
        json.dumps(data, ensure_ascii=False),
    ]
    try:
        import subprocess

        subprocess.run(argv, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return


def cmd_init(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    ensure_dir(paths.pluribus_dir)
    ensure_dir(paths.objects_dir)
    ensure_dir(paths.index_dir)

    if paths.manifest_path.exists() and not args.force:
        sys.stderr.write(f"exists: {paths.manifest_path}\n")
        return 2

    domains = [d.strip() for d in (args.domains or "").split(",") if d.strip()]
    manifest = {
        "schema_version": 1,
        "id": str(uuid.uuid4()),
        "name": args.name,
        "created_iso": now_iso_utc(),
        "purpose": args.purpose,
        "direction": args.direction,
        "domains": domains,
        "public_ip": args.public_ip,
        "stores": {"objects_dir": ".pluribus/objects", "index_dir": ".pluribus/index"},
        "providers": [],
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(str(paths.manifest_path) + "\n")
    return 0


def iter_ingest_targets(targets: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in targets:
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    out.append(child)
        else:
            out.append(p)
    return out


def cmd_ingest(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    if not paths.manifest_path.exists():
        sys.stderr.write("no rhizome manifest found; run: rhizome.py init\n")
        return 2

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    tags = [t for t in (args.tag or []) if t.strip()]
    actor = default_actor()

    ingested = 0
    for p in iter_ingest_targets(args.paths):
        kind = artifact_kind(p)
        if kind not in {"file", "archive"}:
            continue
        sha, size = sha256_file(p)
        obj_path = paths.objects_dir / sha
        if args.store and not obj_path.exists():
            ensure_dir(paths.objects_dir)
            with p.open("rb") as src, obj_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

        ext = p.suffix.lower().lstrip(".") or None
        rec = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "kind": kind,
            "sha256": sha,
            "bytes": size,
            "media_type": guess_media_type(p),
            "ext": ext,
            "tags": tags,
            "sources": [str(p)],
            "provenance": {"added_by": actor},
        }
        append_ndjson(paths.artifacts_index, rec)
        if args.emit_bus:
            emit_bus(topic="strp.artifact.ingested", data=rec, bus_dir=bus_dir)
        ingested += 1

    sys.stdout.write(f"ingested {ingested}\n")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    n = max(1, args.n)
    rows = list(iter_ndjson(paths.artifacts_index))[-n:]
    for obj in rows:
        sys.stdout.write(
            f"{obj.get('iso','')}  {obj.get('kind',''):<7}  {str(obj.get('bytes','')):<10}  {obj.get('sha256','')[:12]}  {','.join(obj.get('tags') or [])}\n"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.root)
    prefix = args.id.strip()
    if not prefix:
        return 2
    for obj in iter_ndjson(paths.artifacts_index):
        oid = obj.get("id") or ""
        sha = obj.get("sha256") or ""
        if (isinstance(oid, str) and oid.startswith(prefix)) or (isinstance(sha, str) and sha.startswith(prefix)):
            sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
            return 0
    sys.stderr.write("not found\n")
    return 1


def cmd_sync(args: argparse.Namespace) -> int:
    """Sync artifacts between local and remote rhizome."""
    import urllib.request
    import urllib.error

    paths = resolve_paths(args.root)
    if not paths.manifest_path.exists():
        sys.stderr.write("no rhizome manifest found; run: rhizome.py init\n")
        return 2

    remote_url = args.remote
    if not remote_url:
        # Try to get from manifest
        try:
            manifest = json.loads(paths.manifest_path.read_text())
            remote_url = manifest.get("remote_url")
        except Exception:
            pass

    if not remote_url:
        sys.stderr.write("no remote URL specified; use --remote or set remote_url in manifest\n")
        return 2

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    # Fetch remote artifact index
    try:
        req = urllib.request.Request(f"{remote_url}/index/artifacts.ndjson")
        with urllib.request.urlopen(req, timeout=30) as resp:
            remote_index = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"failed to fetch remote index: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"sync error: {e}\n")
        return 1

    # Parse remote artifacts
    remote_artifacts: dict[str, dict] = {}
    for line in remote_index.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            sha = obj.get("sha256")
            if sha:
                remote_artifacts[sha] = obj
        except Exception:
            continue

    # Parse local artifacts
    local_artifacts: dict[str, dict] = {}
    for obj in iter_ndjson(paths.artifacts_index):
        sha = obj.get("sha256")
        if sha:
            local_artifacts[sha] = obj

    # Find missing locally (to pull)
    to_pull = set(remote_artifacts.keys()) - set(local_artifacts.keys())
    # Find missing remotely (to push)
    to_push = set(local_artifacts.keys()) - set(remote_artifacts.keys())

    pulled = 0
    pushed = 0

    # Pull missing artifacts
    if args.direction in ("pull", "both"):
        for sha in to_pull:
            obj_url = f"{remote_url}/objects/{sha}"
            local_obj = paths.objects_dir / sha
            try:
                req = urllib.request.Request(obj_url)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                ensure_dir(paths.objects_dir)
                local_obj.write_bytes(data)
                # Add to local index
                remote_obj = remote_artifacts[sha]
                remote_obj["provenance"] = {"synced_from": remote_url, "synced_iso": now_iso_utc()}
                append_ndjson(paths.artifacts_index, remote_obj)
                pulled += 1
                if args.emit_bus:
                    emit_bus(topic="rhizome.artifact.pulled", data={"sha256": sha, "from": remote_url}, bus_dir=bus_dir)
            except Exception as e:
                sys.stderr.write(f"pull failed for {sha[:12]}: {e}\n")

    # Push missing artifacts (if remote accepts uploads)
    if args.direction in ("push", "both") and args.push_enabled:
        for sha in to_push:
            local_obj = paths.objects_dir / sha
            if not local_obj.exists():
                continue
            push_url = f"{remote_url}/upload/{sha}"
            try:
                data = local_obj.read_bytes()
                req = urllib.request.Request(push_url, data=data, method="PUT")
                req.add_header("Content-Type", "application/octet-stream")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp.read()
                pushed += 1
                if args.emit_bus:
                    emit_bus(topic="rhizome.artifact.pushed", data={"sha256": sha, "to": remote_url}, bus_dir=bus_dir)
            except Exception as e:
                sys.stderr.write(f"push failed for {sha[:12]}: {e}\n")

    sys.stdout.write(f"sync complete: pulled {pulled}, pushed {pushed}\n")

    if args.emit_bus:
        emit_bus(
            topic="rhizome.sync.complete",
            data={"remote": remote_url, "pulled": pulled, "pushed": pushed},
            bus_dir=bus_dir,
        )

    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    """Publish artifacts matching tags/filters to a destination."""
    paths = resolve_paths(args.root)
    if not paths.manifest_path.exists():
        sys.stderr.write("no rhizome manifest found; run: rhizome.py init\n")
        return 2

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    filter_tags = set(args.tag) if args.tag else None
    limit = args.limit

    # Collect artifacts to publish
    to_publish = []
    for obj in iter_ndjson(paths.artifacts_index):
        if filter_tags:
            obj_tags = set(obj.get("tags") or [])
            if not obj_tags.intersection(filter_tags):
                continue
        to_publish.append(obj)
        if limit and len(to_publish) >= limit:
            break

    # Output format based on destination
    if args.format == "manifest":
        # Output as JSON manifest for external consumption
        manifest = {
            "schema_version": 1,
            "published_iso": now_iso_utc(),
            "count": len(to_publish),
            "artifacts": to_publish,
        }
        out = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    elif args.format == "ndjson":
        # Output as NDJSON stream
        out = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in to_publish) + "\n"
    else:
        # Default: simple listing
        out = ""
        for obj in to_publish:
            out += f"{obj.get('sha256', '')[:12]}  {','.join(obj.get('tags') or [])}  {obj.get('iso', '')}\n"

    # Write to destination
    if args.output == "-":
        sys.stdout.write(out)
    else:
        output_path = Path(args.output).expanduser()
        ensure_dir(output_path.parent)
        output_path.write_text(out, encoding="utf-8")
        sys.stdout.write(f"published {len(to_publish)} artifacts to {output_path}\n")

    if args.emit_bus:
        emit_bus(
            topic="rhizome.publish.complete",
            data={"count": len(to_publish), "format": args.format, "output": args.output},
            bus_dir=bus_dir,
        )

    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Pull a specific artifact by SHA from remote."""
    import urllib.request
    import urllib.error

    paths = resolve_paths(args.root)
    if not paths.manifest_path.exists():
        sys.stderr.write("no rhizome manifest found; run: rhizome.py init\n")
        return 2

    sha = args.sha
    remote_url = args.remote
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    local_obj = paths.objects_dir / sha
    if local_obj.exists() and not args.force:
        sys.stdout.write(f"artifact {sha[:12]} already exists locally\n")
        return 0

    obj_url = f"{remote_url}/objects/{sha}"
    try:
        req = urllib.request.Request(obj_url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        ensure_dir(paths.objects_dir)
        local_obj.write_bytes(data)

        # Create index entry
        rec = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "kind": "file",
            "sha256": sha,
            "bytes": len(data),
            "provenance": {"pulled_from": remote_url, "pulled_iso": now_iso_utc()},
        }
        append_ndjson(paths.artifacts_index, rec)

        sys.stdout.write(f"pulled {sha[:12]} ({len(data)} bytes)\n")

        if args.emit_bus:
            emit_bus(topic="rhizome.artifact.pulled", data={"sha256": sha, "from": remote_url, "bytes": len(data)}, bus_dir=bus_dir)

    except urllib.error.HTTPError as e:
        sys.stderr.write(f"pull failed: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"pull error: {e}\n")
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rhizome.py", description="STRp rhizome manifest + local artifact ingest (append-only).")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: find .pluribus/rhizome.json upward, else CWD).")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create .pluribus/rhizome.json + dirs.")
    init.add_argument("--name", required=True)
    init.add_argument("--purpose", required=True)
    init.add_argument(
        "--direction",
        default="STRp: curate→distill→hypothesize→apply→implement→verify; append-only evidence; sextet-tagged artifacts.",
    )
    init.add_argument("--domains", default="")
    init.add_argument("--public-ip", default=None)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    ing = sub.add_parser("ingest", help="Ingest files/dirs and append to .pluribus/index/artifacts.ndjson.")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--tag", action="append", default=[])
    ing.add_argument("--store", action="store_true", help="Copy bytes into .pluribus/objects/<sha256>.")
    ing.add_argument("--emit-bus", action="store_true", help="Emit strp.artifact.ingested on the agent bus.")
    ing.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    ing.set_defaults(func=cmd_ingest)

    ls = sub.add_parser("list", help="List most recent artifact records.")
    ls.add_argument("-n", type=int, default=25)
    ls.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="Show one artifact record by id/sha prefix.")
    show.add_argument("id")
    show.set_defaults(func=cmd_show)

    # Sync command
    sync = sub.add_parser("sync", help="Sync artifacts with a remote rhizome.")
    sync.add_argument("--remote", default=None, help="Remote rhizome URL (or set in manifest).")
    sync.add_argument("--direction", choices=["pull", "push", "both"], default="pull", help="Sync direction.")
    sync.add_argument("--push-enabled", action="store_true", help="Allow pushing to remote (requires remote write access).")
    sync.add_argument("--emit-bus", action="store_true", help="Emit bus events.")
    sync.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sync.set_defaults(func=cmd_sync)

    # Publish command
    publish = sub.add_parser("publish", help="Publish artifacts to a file or stdout.")
    publish.add_argument("--output", "-o", default="-", help="Output file (- for stdout).")
    publish.add_argument("--format", choices=["simple", "ndjson", "manifest"], default="simple")
    publish.add_argument("--tag", action="append", default=[], help="Filter by tag (can repeat).")
    publish.add_argument("--limit", type=int, default=None, help="Max artifacts to publish.")
    publish.add_argument("--emit-bus", action="store_true", help="Emit bus events.")
    publish.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    publish.set_defaults(func=cmd_publish)

    # Pull command
    pull = sub.add_parser("pull", help="Pull a specific artifact from remote by SHA.")
    pull.add_argument("sha", help="SHA256 of artifact to pull.")
    pull.add_argument("--remote", required=True, help="Remote rhizome URL.")
    pull.add_argument("--force", action="store_true", help="Overwrite if exists locally.")
    pull.add_argument("--emit-bus", action="store_true", help="Emit bus events.")
    pull.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    pull.set_defaults(func=cmd_pull)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

