#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class HostRef:
    host: str
    kind: str  # "ip" | "domain"
    ports: tuple[int, ...]
    sources: tuple[str, ...]


_URL_RE = re.compile(r"(?i)^(?P<scheme>[a-z][a-z0-9+.-]*):\\/\\/(?P<rest>.+)$")
_SSH_USERHOST_RE = re.compile(r"^(?P<user>[^@\s]+)@(?P<host>[^\s]+)$")
_HOSTPORT_RE = re.compile(r"^(?P<host>\[[^\]]+\]|[^:\s]+)(?::(?P<port>\d{1,5}))?$")


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _clean_host(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    raw = raw.strip().strip(".")
    return raw.lower()


def _extract_host_and_port(token: str) -> tuple[str, int | None] | None:
    token = token.strip()
    if not token:
        return None

    m = _URL_RE.match(token)
    if m:
        token = m.group("rest").split("/", 1)[0]

    m = _SSH_USERHOST_RE.match(token)
    if m:
        token = m.group("host")

    m = _HOSTPORT_RE.match(token)
    if not m:
        return None

    host = _clean_host(m.group("host"))
    if not host:
        return None

    port: int | None = None
    if m.group("port"):
        p = int(m.group("port"))
        if 1 <= p <= 65535:
            port = p
    return host, port


def iter_tokens(line: str) -> Iterable[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return []
    line = line.split("#", 1)[0]
    return re.split(r"[\s,;]+", line.strip())


def build_refs(lines: list[str]) -> list[HostRef]:
    allow_bare = os.environ.get("INGRESS_REF_ALLOW_BARE_HOSTNAMES", "0") == "1"
    acc: dict[str, dict] = {}
    for idx, line in enumerate(lines, start=1):
        for token in iter_tokens(line):
            extracted = _extract_host_and_port(token)
            if not extracted:
                continue
            host, port = extracted
            if host in {"localhost"}:
                continue
            if not _is_ip(host) and "." not in host and not allow_bare:
                continue
            kind = "ip" if _is_ip(host) else "domain"
            entry = acc.setdefault(host, {"kind": kind, "ports": set(), "sources": set()})
            if port is not None:
                entry["ports"].add(port)
            entry["sources"].add(f"L{idx}")
    out: list[HostRef] = []
    for host, meta in sorted(acc.items(), key=lambda kv: kv[0]):
        out.append(
            HostRef(
                host=host,
                kind=meta["kind"],
                ports=tuple(sorted(meta["ports"])),
                sources=tuple(sorted(meta["sources"])),
            )
        )
    return out


def write_outputs(refs: list[HostRef], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "hosts.txt").write_text("\n".join(r.host for r in refs) + ("\n" if refs else ""), encoding="utf-8")
    (out_dir / "hosts.json").write_text(
        json.dumps([asdict(r) for r in refs], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = []
    for r in refs:
        ports = ",".join(str(p) for p in r.ports) if r.ports else ""
        rows.append(f"| `{r.host}` | `{r.kind}` | `{ports}` | `{','.join(r.sources)}` |")

    md = [
        "# Ingress Host Reference",
        "",
        "This is an auto-generated, deduplicated reference list of hosts mentioned in the provided input.",
        "",
        f"- Total hosts: `{len(refs)}`",
        "",
        "| host | kind | ports | sources |",
        "|---|---:|---:|---:|",
        *rows,
        "",
    ]
    (out_dir / "hosts.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Distill host/ingress references into an official reference set.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input file containing hosts/urls/ssh targets.")
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "ingresses"),
        help="Output directory (default: nucleus/docs/ingresses).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    lines = in_path.read_text(encoding="utf-8", errors="replace").splitlines()
    refs = build_refs(lines)
    write_outputs(refs, out_dir)
    print(f"Wrote {len(refs)} hosts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
