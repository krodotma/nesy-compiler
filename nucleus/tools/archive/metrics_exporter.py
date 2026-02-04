#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.dont_write_bytecode = True


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n


def sqlite_count(db_path: Path, sql: str) -> int:
    if not db_path.exists():
        return 0
    try:
        con = sqlite3.connect(str(db_path))
        try:
            row = con.execute(sql).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        finally:
            con.close()
    except Exception:
        return 0


def tail_ndjson(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except Exception:
                continue
    except Exception:
        return None
    return None


class Metrics:
    def __init__(self, root: Path, bus_dir: Path | None):
        self.root = root
        self.bus_dir = bus_dir
        self._cache_ts = 0.0
        self._cache = ""

    def render(self) -> str:
        now = time.time()
        if now - self._cache_ts < 1.0 and self._cache:
            return self._cache

        idx = self.root / ".pluribus" / "index"
        artifacts = count_lines(idx / "artifacts.ndjson")
        requests = count_lines(idx / "requests.ndjson")
        responses = count_lines(idx / "responses.ndjson")
        domains = count_lines(idx / "domains.ndjson")
        audits = count_lines(idx / "domain_audit.ndjson")
        rag_docs = sqlite_count(idx / "rag.sqlite3", "SELECT COUNT(1) FROM docs")

        mismatch = 0
        last_audit = tail_ndjson(idx / "domain_audit.ndjson")
        if isinstance(last_audit, dict):
            match = last_audit.get("match")
            if match is False:
                mismatch = 1

        bus_events = count_lines(self.bus_dir / "events.ndjson") if self.bus_dir else 0

        lines = []
        lines.append("# HELP strp_artifacts_total Total ingested artifacts (records)")
        lines.append("# TYPE strp_artifacts_total gauge")
        lines.append(f"strp_artifacts_total {artifacts}")
        lines.append("# HELP strp_requests_total Total STRp requests (records)")
        lines.append("# TYPE strp_requests_total gauge")
        lines.append(f"strp_requests_total {requests}")
        lines.append("# HELP strp_responses_total Total STRp responses (records)")
        lines.append("# TYPE strp_responses_total gauge")
        lines.append(f"strp_responses_total {responses}")
        lines.append("# HELP strp_domains_total Total domains (records)")
        lines.append("# TYPE strp_domains_total gauge")
        lines.append(f"strp_domains_total {domains}")
        lines.append("# HELP strp_domain_audit_records_total Total domain audit records")
        lines.append("# TYPE strp_domain_audit_records_total gauge")
        lines.append(f"strp_domain_audit_records_total {audits}")
        lines.append("# HELP strp_rag_docs_total Total docs in SQLite FTS index")
        lines.append("# TYPE strp_rag_docs_total gauge")
        lines.append(f"strp_rag_docs_total {rag_docs}")
        lines.append("# HELP strp_bus_events_total Total events in the bus log")
        lines.append("# TYPE strp_bus_events_total gauge")
        lines.append(f"strp_bus_events_total {bus_events}")
        lines.append("# HELP strp_domain_last_audit_mismatch Whether the latest audit record was a mismatch (0/1)")
        lines.append("# TYPE strp_domain_last_audit_mismatch gauge")
        lines.append(f"strp_domain_last_audit_mismatch {mismatch}")
        lines.append("")

        self._cache_ts = now
        self._cache = "\n".join(lines)
        return self._cache


class Handler(BaseHTTPRequestHandler):
    metrics: Metrics

    def do_GET(self):  # noqa: N802
        if self.path not in {"/metrics", "/"}:
            self.send_response(404)
            self.end_headers()
            return
        body = self.metrics.render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args):  # noqa: A003
        return


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="metrics_exporter.py", description="Prometheus-style /metrics for STRp rhizomes (no deps).")
    ap.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9119)
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    bus_raw = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    bus_dir = Path(bus_raw).expanduser().resolve() if bus_raw else None

    Handler.metrics = Metrics(root=root, bus_dir=bus_dir)
    httpd = HTTPServer((args.listen, int(args.port)), Handler)
    sys.stdout.write(f"listening on http://{args.listen}:{args.port}/metrics  root={root}\n")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

