#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

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


def resolve_root(raw_root: str | None) -> Path:
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    return find_rhizome_root(Path.cwd()) or Path.cwd().resolve()


def registry_path(root: Path) -> Path:
    return root / ".pluribus" / "index" / "domains.ndjson"


def normalize_domain(raw: str) -> str | None:
    raw = raw.strip().lower().rstrip(".")
    if not raw or " " in raw:
        return None
    raw = raw.split(":")[0]
    if raw.startswith("*."):
        raw = raw[2:]
    if raw.startswith("."):
        raw = raw[1:]
    if not raw:
        return None
    if raw == "localhost":
        return raw
    if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", raw) is None:
        return None
    # Avoid polluting the registry with version strings / IPs like "0.0.0" or "69.169.104.17".
    if re.search(r"[a-z]", raw) is None:
        return None
    return raw


URL_RE = re.compile(r"https?://[^\s<>\"]+")


def extract_domains_from_text(text: str, *, include_bare: bool) -> set[str]:
    out: set[str] = set()
    for m in URL_RE.finditer(text):
        try:
            u = urlparse(m.group(0))
            host = (u.hostname or "").strip().lower()
            d = normalize_domain(host)
            if d:
                out.add(d)
        except Exception:
            continue

    if include_bare:
        for m in re.finditer(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)+\b", text.lower()):
            d = normalize_domain(m.group(0))
            if d:
                out.add(d)
    return out


def scan_system_domains() -> set[str]:
    out: set[str] = set()
    hosts = Path("/etc/hosts")
    if hosts.exists():
        for line in hosts.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            for cand in parts[1:]:
                d = normalize_domain(cand)
                if d:
                    out.add(d)
    nginx_dir = Path("/etc/nginx/sites-enabled")
    if nginx_dir.exists():
        for p in nginx_dir.glob("*"):
            if not p.is_file():
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"server_name\s+([^;]+);", txt):
                for cand in m.group(1).split():
                    d = normalize_domain(cand)
                    if d:
                        out.add(d)
    return out


def scan_workspace_domains(root: Path, *, include_bare: bool, max_files: int, max_bytes: int) -> set[str]:
    out: set[str] = set()
    ingress_hosts_candidates = [
        root / "docs" / "ingresses" / "hosts.txt",
        root / "nucleus" / "docs" / "ingresses" / "hosts.txt",
    ]
    for ingress_hosts in ingress_hosts_candidates:
        if not ingress_hosts.exists():
            continue
        for line in ingress_hosts.read_text(encoding="utf-8", errors="replace").splitlines():
            d = normalize_domain(line)
            if d:
                out.add(d)
        break
    manifest = root / ".pluribus" / "rhizome.json"
    if manifest.exists():
        try:
            obj = json.loads(manifest.read_text(encoding="utf-8"))
            for d in obj.get("domains") or []:
                if isinstance(d, str):
                    nd = normalize_domain(d)
                    if nd:
                        out.add(nd)
        except Exception:
            pass

    exts = {".md", ".txt", ".env", ".json", ".yaml", ".yml"}
    scanned = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        try:
            if p.stat().st_size > max_bytes:
                continue
        except Exception:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        out |= extract_domains_from_text(txt, include_bare=include_bare)
        scanned += 1
        if scanned >= max_files:
            break
    return out


def audit_path(root: Path) -> Path:
    return root / ".pluribus" / "index" / "domain_audit.ndjson"


def resolve_ips(domain: str) -> set[str]:
    out: set[str] = set()
    try:
        infos = socket.getaddrinfo(domain, None)
    except Exception:
        return out
    for _, _, _, _, sockaddr in infos:
        if not sockaddr:
            continue
        ip = sockaddr[0]
        if isinstance(ip, str) and ip:
            out.add(ip)
    return out


def discover_public_ip(timeout_s: float = 5.0) -> str | None:
    import urllib.request

    urls = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                ip = resp.read().decode("utf-8", errors="replace").strip()
            if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", ip):
                return ip
        except Exception:
            continue
    return None


def cmd_audit(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    reg = registry_path(root)
    outp = audit_path(root)
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    desired_ip: str | None = args.ip
    if desired_ip == "auto":
        manifest = root / ".pluribus" / "rhizome.json"
        if manifest.exists():
            try:
                obj = json.loads(manifest.read_text(encoding="utf-8"))
                cand = obj.get("public_ip")
                if isinstance(cand, str) and cand.strip():
                    desired_ip = cand.strip()
            except Exception:
                pass
        if desired_ip == "auto" and args.discover:
            desired_ip = discover_public_ip()
    if desired_ip == "auto":
        desired_ip = None

    if not desired_ip:
        sys.stderr.write("missing target ip; pass --ip <addr> or set rhizome public_ip (or --discover)\n")
        return 2

    def from_registry(*, exclude_auto: bool) -> set[str]:
        out: set[str] = set()
        for obj in iter_ndjson(reg):
            d = obj.get("domain")
            if not isinstance(d, str):
                continue
            if exclude_auto:
                tags = obj.get("tags") or []
                if isinstance(tags, list) and any(str(t).strip().lower() == "auto" for t in tags):
                    continue
                if str(obj.get("source") or "").strip().lower() == "scan":
                    continue
            nd = normalize_domain(d)
            if nd:
                out.add(nd)
        return out

    def from_ingresses() -> set[str]:
        out: set[str] = set()
        ingress_hosts_candidates = [
            root / "docs" / "ingresses" / "hosts.txt",
            root / "nucleus" / "docs" / "ingresses" / "hosts.txt",
        ]
        for ingress_hosts in ingress_hosts_candidates:
            if not ingress_hosts.exists():
                continue
            for line in ingress_hosts.read_text(encoding="utf-8", errors="replace").splitlines():
                d = normalize_domain(line)
                if d:
                    out.add(d)
            break
        return out

    def from_rhizome() -> set[str]:
        out: set[str] = set()
        manifest = root / ".pluribus" / "rhizome.json"
        if not manifest.exists():
            return out
        try:
            obj = json.loads(manifest.read_text(encoding="utf-8"))
            for d in obj.get("domains") or []:
                if isinstance(d, str):
                    nd = normalize_domain(d)
                    if nd:
                        out.add(nd)
        except Exception:
            return out
        return out

    scope = (args.scope or "curated").strip().lower()
    if scope not in {"curated", "registry", "all"}:
        sys.stderr.write("invalid --scope; use curated|registry|all\n")
        return 2

    domains_set: set[str] = set()
    if scope in {"curated", "all"} and not args.no_ingresses:
        domains_set |= from_ingresses()
    if scope in {"curated", "all"}:
        domains_set |= from_rhizome()
    if scope in {"registry", "all"}:
        domains_set |= from_registry(exclude_auto=bool(args.exclude_auto))
    if scope == "curated":
        domains_set |= from_registry(exclude_auto=True)

    domains = sorted(domains_set)
    try:
        cap = int(args.max_domains)
    except Exception:
        cap = 200
    if cap > 0 and len(domains) > cap:
        domains = domains[:cap]

    ok = 0
    mismatch = 0
    unresolved = 0
    for d in domains:
        ips = resolve_ips(d)
        match = desired_ip in ips
        if not ips:
            unresolved += 1
        elif match:
            ok += 1
        else:
            mismatch += 1
        rec = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "kind": "domain_audit",
            "domain": d,
            "desired_ip": desired_ip,
            "resolved_ips": sorted(ips),
            "match": bool(match),
            "provenance": {"added_by": actor, "context": args.context},
        }
        append_ndjson(outp, rec)

    summary = {"root": str(root), "desired_ip": desired_ip, "total": len(domains), "ok": ok, "mismatch": mismatch, "unresolved": unresolved}
    emit_bus(bus_dir, topic="domains.audit.completed", kind="log", level="info", actor=actor, data=summary)
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    reg = registry_path(root)
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    domain = normalize_domain(args.domain)
    if not domain:
        sys.stderr.write("invalid domain\n")
        return 2

    tags = [t for t in (args.tag or []) if t.strip()]
    rec = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "domain",
        "domain": domain,
        "tags": tags,
        "source": args.source,
        "provenance": {"added_by": actor, "context": args.context},
    }
    append_ndjson(reg, rec)
    emit_bus(bus_dir, topic="domains.added", kind="artifact", level="info", actor=actor, data=rec)
    sys.stdout.write(rec["id"] + "\n")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    reg = registry_path(root)
    seen = set()
    for obj in iter_ndjson(reg):
        d = obj.get("domain")
        if not isinstance(d, str):
            continue
        nd = normalize_domain(d)
        if not nd:
            continue
        if nd in seen:
            continue
        seen.add(nd)
        sys.stdout.write(nd + "\n")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    reg = registry_path(root)
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    existing = {obj.get("domain") for obj in iter_ndjson(reg) if isinstance(obj.get("domain"), str)}
    found = scan_workspace_domains(
        root,
        include_bare=bool(args.include_bare),
        max_files=max(1, int(args.max_files)),
        max_bytes=max(1, int(args.max_bytes)),
    )
    if args.include_system:
        found |= scan_system_domains()

    added = 0
    for d in sorted(found):
        if d in existing:
            continue
        rec = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "kind": "domain",
            "domain": d,
            "tags": ["auto"],
            "source": "scan",
            "provenance": {"added_by": actor, "context": args.context},
        }
        append_ndjson(reg, rec)
        added += 1

    emit_bus(bus_dir, topic="domains.scan.completed", kind="log", level="info", actor=actor, data={"root": str(root), "added": added, "include_system": args.include_system})
    sys.stdout.write(f"added {added}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="domain_registry.py", description="Append-only domain registry + detector for STRp rhizomes.")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Add one domain.")
    a.add_argument("domain")
    a.add_argument("--tag", action="append", default=[])
    a.add_argument("--source", default="manual")
    a.add_argument("--context", default=None)
    a.set_defaults(func=cmd_add)

    s = sub.add_parser("scan", help="Scan workspace (and optional system configs) for domains and append them.")
    s.add_argument("--include-system", action="store_true")
    s.add_argument(
        "--include-bare",
        action="store_true",
        help="Also attempt to extract bare domains (not only URL hosts). Can be noisy; default is URL hosts + rhizome domains.",
    )
    s.add_argument("--max-files", default="2000", help="Max files to scan (default: 2000).")
    s.add_argument("--max-bytes", default=str(2_000_000), help="Max bytes per file (default: 2000000).")
    s.add_argument("--context", default=None)
    s.set_defaults(func=cmd_scan)

    l = sub.add_parser("list", help="List unique domains.")
    l.set_defaults(func=cmd_list)

    a2 = sub.add_parser("audit", help="Audit domains: check DNS A/AAAA resolution against a target IP; append audit records.")
    a2.add_argument("--ip", default="auto", help="Target IP, or 'auto' (use rhizome public_ip; optionally --discover).")
    a2.add_argument("--discover", action="store_true", help="If --ip auto and rhizome has no public_ip, discover via ipify/ifconfig.")
    a2.add_argument("--no-ingresses", action="store_true", help="Do not include docs/ingresses/hosts.txt (included by default).")
    a2.add_argument("--scope", default="curated", help="curated|registry|all (default: curated).")
    a2.add_argument("--exclude-auto", action="store_true", help="When auditing registry scope, exclude entries tagged auto or source=scan.")
    a2.add_argument("--max-domains", default="200", help="Safety cap for audit (default: 200). Use 0 for no cap.")
    a2.add_argument("--context", default=None)
    a2.set_defaults(func=cmd_audit)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
