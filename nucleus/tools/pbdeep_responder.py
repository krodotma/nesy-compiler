#!/usr/bin/env python3
from __future__ import annotations

"""
PBDEEP responder daemon.

Tails the bus for `operator.pbdeep.request` and emits:
  - `operator.pbdeep.progress` (kind=metric)
  - `operator.pbdeep.report` (kind=artifact)

This responder is read-only and append-only; it does not mutate git state.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_bus_dir(args_bus_dir: str | None) -> str:
    return args_bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"


def default_actor(args_actor: str | None) -> str:
    return args_actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pbdeep-responder"


def ensure_events_file(bus_dir: str) -> Path:
    p = Path(bus_dir) / "events.ndjson"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p


def tail_events(events_path: Path, *, since_ts: float, poll_s: float, stop_at_ts: float | None):
    def emit_backfill() -> list[dict]:
        try:
            max_bytes = 512 * 1024
            with events_path.open("rb") as bf:
                bf.seek(0, os.SEEK_END)
                end = bf.tell()
                start = max(0, end - max_bytes)
                bf.seek(start)
                data = bf.read(end - start)
            lines = data.splitlines()
            out: list[dict] = []
            for b in lines[-2000:]:
                try:
                    obj = json.loads(b.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if obj.get("topic") != "operator.pbdeep.request":
                    continue
                try:
                    ts = float(obj.get("ts") or 0.0)
                except Exception:
                    ts = 0.0
                if ts < since_ts:
                    continue
                out.append(obj)
            return out
        except Exception:
            return []

    for obj in emit_backfill():
        yield obj

    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            if stop_at_ts is not None and time.time() >= stop_at_ts:
                return
            line = f.readline()
            if not line:
                time.sleep(max(0.05, poll_s))
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("topic") != "operator.pbdeep.request":
                continue
            try:
                ts = float(obj.get("ts") or 0.0)
            except Exception:
                ts = 0.0
            if ts < since_ts:
                continue
            yield obj


def emit_bus(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
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


def run_cmd(cmd: list[str], *, timeout_s: float, cwd: Path | None = None, env: dict | None = None) -> tuple[int, str, str]:
    try:
        merged_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        if env:
            merged_env.update(env)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:
        return 1, "", str(exc)


def parse_instruction(instruction: str) -> tuple[dict[str, str], str]:
    opts: dict[str, str] = {}
    rest: list[str] = []
    for tok in (instruction or "").split():
        if "=" in tok:
            key, value = tok.split("=", 1)
            key = key.strip().lower()
            if key in {"mode", "max_branches", "max_untracked", "max_lost", "max_refs", "max_graph"}:
                opts[key] = value.strip()
                continue
        rest.append(tok)
    return opts, " ".join(rest).strip()


def list_branches(root: Path, iso_git: Path, timeout_s: float) -> tuple[list[str], str | None, str | None]:
    code, out, err = run_cmd(["node", str(iso_git), "branch", str(root)], timeout_s=timeout_s)
    if code != 0:
        return [], None, err.strip() or None
    branches: list[str] = []
    current: str | None = None
    for line in out.splitlines():
        if line.startswith("* "):
            name = line[2:].strip()
            if name:
                current = name
                branches.append(name)
        elif line.startswith("  "):
            name = line[2:].strip()
            if name:
                branches.append(name)
    return branches, current, None


def list_remotes(root: Path, iso_git: Path, timeout_s: float) -> tuple[list[dict], str | None]:
    code, out, err = run_cmd(["node", str(iso_git), "remote", "list", str(root)], timeout_s=timeout_s)
    if code != 0:
        return [], err.strip() or None
    remotes: list[dict] = []
    for line in out.splitlines():
        if not line.strip() or line.strip() == "(no remotes)":
            continue
        parts = line.split("\t")
        remote = parts[0].strip() if parts else ""
        url = parts[1].strip() if len(parts) > 1 else ""
        if remote:
            remotes.append({"remote": remote, "url": url})
    return remotes, None


def list_untracked(root: Path, iso_git: Path, timeout_s: float, max_untracked: int) -> tuple[list[str], str | None]:
    code, out, err = run_cmd(["node", str(iso_git), "status", str(root)], timeout_s=timeout_s)
    if code != 0 and not out.strip():
        return [], err.strip() or None
    untracked: list[str] = []
    for line in out.splitlines():
        if line.startswith("?? "):
            path = line[3:].strip()
            if path:
                untracked.append(path)
            if len(untracked) >= max_untracked:
                break
    return untracked, None


def scan_lost_and_found(root: Path, max_lost: int) -> list[str]:
    hits: list[str] = []
    skip_dirs = {".git", ".pluribus", ".pluribus_local", "node_modules", ".venv", ".cache", "dist", "build"}
    patterns = {"lost_and_found", "lost+found", "lost-found", "lostandfound"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        base = Path(dirpath).name.lower()
        if base in patterns:
            hits.append(str(Path(dirpath).relative_to(root)))
            if len(hits) >= max_lost:
                break
        for fname in filenames:
            if len(hits) >= max_lost:
                break
            lower = fname.lower()
            if any(p in lower for p in patterns):
                hits.append(str((Path(dirpath) / fname).relative_to(root)))
        if len(hits) >= max_lost:
            break
    return hits


def scan_doc_refs(root: Path, max_refs: int) -> list[dict]:
    refs: list[dict] = []
    doc_dirs = [
        root / "nucleus" / "docs",
        root / "pluribus_next" / "docs",
        root / "nucleus" / "specs",
    ]
    path_re = re.compile(r'(?P<path>(?:nucleus|pluribus_next|pluribus)/[A-Za-z0-9_./-]+)')
    for d in doc_dirs:
        if not d.exists():
            continue
        for doc in d.rglob("*.md"):
            try:
                if doc.stat().st_size > 512 * 1024:
                    continue
                text = doc.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for match in path_re.finditer(text):
                raw = match.group("path").rstrip(").,;:[]{}")
                if not raw:
                    continue
                target = root / raw
                if not target.exists():
                    refs.append({"doc": str(doc.relative_to(root)), "path": raw})
                if len(refs) >= max_refs:
                    return refs
    return refs


def scan_code_doc_refs(root: Path, max_refs: int) -> list[dict]:
    refs: list[dict] = []
    rg = shutil.which("rg")
    pattern = r"nucleus/docs/|pluribus_next/docs/"
    if rg:
        code, out, _ = run_cmd(
            [rg, "-n", pattern, str(root)],
            timeout_s=10.0,
        )
        if code == 0:
            path_re = re.compile(r'(nucleus/docs/[A-Za-z0-9_./-]+|pluribus_next/docs/[A-Za-z0-9_./-]+)')
            for line in out.splitlines():
                match = path_re.search(line)
                if not match:
                    continue
                raw = match.group(0).rstrip(").,;:[]{}")
                if not raw:
                    continue
                target = root / raw
                if not target.exists():
                    parts = line.split(":", 2)
                    refs.append({"source": parts[0] if parts else "", "path": raw})
                if len(refs) >= max_refs:
                    break
        return refs

    # Fallback: limited scan in nucleus/tools and nucleus/dashboard/src.
    scan_dirs = [root / "nucleus" / "tools", root / "nucleus" / "dashboard" / "src"]
    path_re = re.compile(r'(nucleus/docs/[A-Za-z0-9_./-]+|pluribus_next/docs/[A-Za-z0-9_./-]+)')
    for d in scan_dirs:
        if not d.exists():
            continue
        for src in d.rglob("*.[pt]y"):
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            match = path_re.search(text)
            if match:
                raw = match.group(0).rstrip(").,;:[]{}")
                target = root / raw
                if not target.exists():
                    refs.append({"source": str(src.relative_to(root)), "path": raw})
            if len(refs) >= max_refs:
                return refs
    return refs


def build_graph(branches: list[str], lost: list[str], untracked: list[str], doc_missing: list[dict]) -> dict:
    nodes: list[dict] = [{"id": "repo", "label": "pluribus", "type": "root", "weight": 3}]
    edges: list[dict] = []
    for name in branches:
        node_id = f"branch:{name}"
        nodes.append({"id": node_id, "label": name, "type": "branch", "weight": 2 if "FINAL" in name else 1})
        edges.append({"source": "repo", "target": node_id, "kind": "branch"})
    for idx, path in enumerate(lost[:20]):
        node_id = f"lost:{idx}"
        nodes.append({"id": node_id, "label": Path(path).name or path, "type": "lost", "weight": 1})
        edges.append({"source": "repo", "target": node_id, "kind": "lost"})
    for idx, path in enumerate(untracked[:20]):
        node_id = f"untracked:{idx}"
        nodes.append({"id": node_id, "label": Path(path).name or path, "type": "untracked", "weight": 1})
        edges.append({"source": "repo", "target": node_id, "kind": "untracked"})
    for idx, miss in enumerate(doc_missing[:20]):
        node_id = f"drift:{idx}"
        nodes.append({"id": node_id, "label": Path(str(miss.get("path", ""))).name or "missing", "type": "drift", "weight": 1})
        edges.append({"source": "repo", "target": node_id, "kind": "drift"})
    return {"nodes": nodes, "edges": edges}


def trim_lines(items: list[str], max_items: int) -> list[str]:
    return [str(item) for item in items[:max_items]]


def format_doc_ref(item: object) -> str:
    if isinstance(item, dict):
        doc = str(item.get("doc") or item.get("source") or "doc")
        path = str(item.get("path") or "")
        return f"{doc} -> {path}".strip()
    return str(item)


def render_index_text(report: dict, *, max_chars: int, max_items: int) -> str:
    summary = report.get("summary") or {}
    instruction = str(report.get("instruction") or "")
    branches = report.get("branches", {}).get("local", []) if isinstance(report.get("branches"), dict) else []
    lost = report.get("lost_and_found", {}).get("paths", []) if isinstance(report.get("lost_and_found"), dict) else []
    untracked = report.get("untracked", {}).get("paths", []) if isinstance(report.get("untracked"), dict) else []
    doc_missing = report.get("doc_drift", {}).get("docs_missing", []) if isinstance(report.get("doc_drift"), dict) else []
    code_missing = report.get("doc_drift", {}).get("code_missing", []) if isinstance(report.get("doc_drift"), dict) else []

    lines: list[str] = [
        "PBDEEP REPORT",
        f"req_id: {report.get('req_id')}",
        f"iso: {report.get('iso')}",
        f"scope: {report.get('scope')}",
        f"instruction: {instruction}",
        "",
        "SUMMARY",
        f"branches_total: {summary.get('branches_total')}",
        f"final_branches: {summary.get('final_branches')}",
        f"lost_and_found_count: {summary.get('lost_and_found_count')}",
        f"untracked_count: {summary.get('untracked_count')}",
        f"doc_missing_count: {summary.get('doc_missing_count')}",
        f"code_missing_docs: {summary.get('code_missing_docs')}",
        "",
        "BRANCHES",
    ]
    lines.extend(trim_lines(branches, max_items))
    lines.append("")
    lines.append("LOST_AND_FOUND")
    lines.extend(trim_lines(lost, max_items))
    lines.append("")
    lines.append("UNTRACKED")
    lines.extend(trim_lines(untracked, max_items))
    lines.append("")
    lines.append("DOC_MISSING")
    lines.extend(trim_lines([format_doc_ref(item) for item in doc_missing], max_items))
    lines.append("")
    lines.append("CODE_MISSING")
    lines.extend(trim_lines([format_doc_ref(item) for item in code_missing], max_items))

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[truncated]"
    return text


def index_rag_report(root: Path, report: dict, report_path: Path, *, bus_dir: str, actor: str, timeout_s: float) -> tuple[str | None, str | None]:
    tool = root / "nucleus" / "tools" / "rag_index.py"
    if not tool.exists():
        return None, "rag_index_missing"
    req_id = str(report.get("req_id") or "")
    title = f"PBDEEP {req_id}".strip() if req_id else "PBDEEP report"
    text = render_index_text(report, max_chars=16000, max_items=50)
    cmd = [
        sys.executable,
        str(tool),
        "--root",
        str(root),
        "--bus-dir",
        bus_dir,
        "add-text",
        "--title",
        title,
        "--source",
        str(report_path),
        "--kind",
        "pbdeep_report",
        "--tag",
        "pbdeep",
        "--tag",
        "forensics",
        "--text",
        text,
    ]
    code, out, err = run_cmd(cmd, timeout_s=timeout_s, env={"PLURIBUS_ACTOR": actor, "PLURIBUS_EMBED_MODE": "off"})
    if code != 0:
        return None, err.strip() or "rag_index_failed"
    doc_id = (out or "").strip()
    return (doc_id or None), None


def kg_add_node(
    root: Path,
    *,
    bus_dir: str,
    actor: str,
    node_type: str,
    text: str,
    ref: str | None,
    tags: list[str],
    context: str | None,
    timeout_s: float,
) -> tuple[str | None, str | None]:
    tool = root / "nucleus" / "tools" / "kg.py"
    if not tool.exists():
        return None, "kg_missing"
    cmd = [
        sys.executable,
        str(tool),
        "--root",
        str(root),
        "--bus-dir",
        bus_dir,
        "add-node",
        "--type",
        node_type,
        "--text",
        text,
    ]
    if ref:
        cmd.extend(["--ref", ref])
    if context:
        cmd.extend(["--context", context])
    for tag in tags:
        if tag:
            cmd.extend(["--tag", tag])
    code, out, err = run_cmd(cmd, timeout_s=timeout_s, env={"PLURIBUS_ACTOR": actor})
    if code != 0:
        return None, err.strip() or "kg_add_node_failed"
    node_id = (out or "").strip()
    return (node_id or None), None


def kg_add_edge(
    root: Path,
    *,
    bus_dir: str,
    actor: str,
    src: str,
    rel: str,
    dst: str,
    tags: list[str],
    context: str | None,
    timeout_s: float,
) -> str | None:
    tool = root / "nucleus" / "tools" / "kg.py"
    if not tool.exists():
        return "kg_missing"
    cmd = [
        sys.executable,
        str(tool),
        "--root",
        str(root),
        "--bus-dir",
        bus_dir,
        "add-edge",
        src,
        rel,
        dst,
    ]
    if context:
        cmd.extend(["--context", context])
    for tag in tags:
        if tag:
            cmd.extend(["--tag", tag])
    code, _, err = run_cmd(cmd, timeout_s=timeout_s, env={"PLURIBUS_ACTOR": actor})
    if code != 0:
        return err.strip() or "kg_add_edge_failed"
    return None


def index_kg_report(root: Path, report: dict, report_path: Path, *, bus_dir: str, actor: str, timeout_s: float) -> tuple[dict, str | None]:
    instruction = str(report.get("instruction") or "")
    req_id = str(report.get("req_id") or "")
    context = instruction[:240] if instruction else None

    branches = report.get("branches", {}).get("local", []) if isinstance(report.get("branches"), dict) else []
    lost = report.get("lost_and_found", {}).get("paths", []) if isinstance(report.get("lost_and_found"), dict) else []
    untracked = report.get("untracked", {}).get("paths", []) if isinstance(report.get("untracked"), dict) else []
    doc_missing = report.get("doc_drift", {}).get("docs_missing", []) if isinstance(report.get("doc_drift"), dict) else []
    code_missing = report.get("doc_drift", {}).get("code_missing", []) if isinstance(report.get("doc_drift"), dict) else []

    summary = report.get("summary") or {}
    report_label = f"PBDEEP report {req_id}".strip() if req_id else "PBDEEP report"

    report_node, err = kg_add_node(
        root,
        bus_dir=bus_dir,
        actor=actor,
        node_type="artifact",
        text=report_label,
        ref=str(report_path),
        tags=["pbdeep", "report", "forensics"],
        context=context,
        timeout_s=timeout_s,
    )
    if not report_node:
        return {"report_node": None, "nodes": 0, "edges": 0}, err or "kg_report_node_failed"

    total_nodes = 1
    total_edges = 0

    def add_category(name: str, count: int) -> str | None:
        nonlocal total_nodes, total_edges
        node_id, node_err = kg_add_node(
            root,
            bus_dir=bus_dir,
            actor=actor,
            node_type="entity",
            text=f"{name} ({count})",
            ref=None,
            tags=["pbdeep", name],
            context=None,
            timeout_s=timeout_s,
        )
        if node_id:
            total_nodes += 1
            edge_err = kg_add_edge(
                root,
                bus_dir=bus_dir,
                actor=actor,
                src=report_node,
                rel="refines",
                dst=node_id,
                tags=["pbdeep", name],
                context=context,
                timeout_s=timeout_s,
            )
            if edge_err is None:
                total_edges += 1
        return node_id

    branch_cat = add_category("branches", len(branches))
    lost_cat = add_category("lost_and_found", len(lost))
    untracked_cat = add_category("untracked", len(untracked))
    drift_cat = add_category("doc_drift", int(summary.get("doc_missing_count", 0)) + int(summary.get("code_missing_docs", 0)))

    max_items = 20

    def add_items(cat_node: str | None, items: list[object], *, label_prefix: str, tags: list[str]) -> None:
        nonlocal total_nodes, total_edges
        if not cat_node:
            return
        for item in items[:max_items]:
            label = label_prefix + str(item)
            node_id, _ = kg_add_node(
                root,
                bus_dir=bus_dir,
                actor=actor,
                node_type="artifact",
                text=label,
                ref=str(item),
                tags=["pbdeep", *tags],
                context=None,
                timeout_s=timeout_s,
            )
            if not node_id:
                continue
            total_nodes += 1
            edge_err = kg_add_edge(
                root,
                bus_dir=bus_dir,
                actor=actor,
                src=cat_node,
                rel="refines",
                dst=node_id,
                tags=["pbdeep", *tags],
                context=None,
                timeout_s=timeout_s,
            )
            if edge_err is None:
                total_edges += 1

    add_items(branch_cat, [str(b) for b in branches], label_prefix="branch:", tags=["branch"])
    add_items(lost_cat, [str(p) for p in lost], label_prefix="lost:", tags=["lost"])
    add_items(untracked_cat, [str(p) for p in untracked], label_prefix="untracked:", tags=["untracked"])
    add_items(drift_cat, [format_doc_ref(p) for p in doc_missing], label_prefix="doc_missing:", tags=["doc_missing"])
    add_items(drift_cat, [format_doc_ref(p) for p in code_missing], label_prefix="code_missing:", tags=["code_missing"])

    return {"report_node": report_node, "nodes": total_nodes, "edges": total_edges}, None


def render_markdown(report: dict) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# PBDEEP Report",
        "",
        f"- req_id: `{report.get('req_id')}`",
        f"- iso: `{report.get('iso')}`",
        f"- scope: `{report.get('scope')}`",
        f"- instruction: `{report.get('instruction')}`",
        "",
        "## Summary",
        "",
        f"- branches_total: {summary.get('branches_total')}",
        f"- final_branches: {summary.get('final_branches')}",
        f"- lost_and_found_count: {summary.get('lost_and_found_count')}",
        f"- untracked_count: {summary.get('untracked_count')}",
        f"- doc_missing_count: {summary.get('doc_missing_count')}",
        f"- code_missing_docs: {summary.get('code_missing_docs')}",
        "",
        "## Next Actions",
    ]
    for action in summary.get("next_actions", []) or []:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def ingest_rhizome(root: Path, path: Path, bus_dir: str) -> None:
    tool = root / "nucleus" / "tools" / "rhizome.py"
    if not tool.exists():
        return
    run_cmd(
        [sys.executable, str(tool), "ingest", "--root", str(root), "--emit-bus", "--store", str(path)],
        timeout_s=10.0,
    )


def emit_progress(bus_dir: str, actor: str, req_id: str, stage: str, status: str, percent: int, extra: dict | None = None) -> None:
    payload = {
        "req_id": req_id,
        "stage": stage,
        "status": status,
        "percent": percent,
        "iso": now_iso_utc(),
    }
    if extra:
        payload.update(extra)
    emit_bus(bus_dir, topic="operator.pbdeep.progress", kind="metric", level="info", actor=actor, data=payload)


def build_report(root: Path, req_id: str, instruction: str, scope: str, scan_mode: str, limits: dict, iso_git: Path, bus_dir: str, actor: str) -> dict:
    if scan_mode == "noop":
        report = {
            "schema_version": 1,
            "req_id": req_id,
            "iso": now_iso_utc(),
            "scope": scope,
            "instruction": instruction,
            "mode": scan_mode,
            "summary": {
                "branches_total": 0,
                "final_branches": 0,
                "lost_and_found_count": 0,
                "untracked_count": 0,
                "doc_missing_count": 0,
                "code_missing_docs": 0,
                "next_actions": ["Run PBDEEP in fast/deep mode for real scanning."],
            },
            "branches": {"local": [], "current": None},
            "lost_and_found": {"paths": []},
            "untracked": {"paths": []},
            "doc_drift": {"docs_missing": [], "code_missing": []},
            "graph": {"nodes": [], "edges": []},
        }
        return report

    emit_progress(bus_dir, actor, req_id, "branches", "start", 5)
    branches, current, branch_err = list_branches(root, iso_git, limits["timeout_s"])
    remotes, remote_err = list_remotes(root, iso_git, limits["timeout_s"])
    emit_progress(bus_dir, actor, req_id, "branches", "done", 20, {"branches_total": len(branches)})

    emit_progress(bus_dir, actor, req_id, "untracked", "start", 25)
    untracked, untracked_err = list_untracked(root, iso_git, limits["timeout_s"], limits["max_untracked"])
    emit_progress(bus_dir, actor, req_id, "untracked", "done", 40, {"untracked_count": len(untracked)})

    emit_progress(bus_dir, actor, req_id, "lost_and_found", "start", 45)
    lost = scan_lost_and_found(root, limits["max_lost"])
    emit_progress(bus_dir, actor, req_id, "lost_and_found", "done", 60, {"lost_and_found_count": len(lost)})

    emit_progress(bus_dir, actor, req_id, "doc_drift", "start", 65)
    docs_missing = scan_doc_refs(root, limits["max_refs"])
    code_missing = scan_code_doc_refs(root, limits["max_refs"])
    emit_progress(
        bus_dir,
        actor,
        req_id,
        "doc_drift",
        "done",
        80,
        {"doc_missing_count": len(docs_missing), "code_missing_docs": len(code_missing)},
    )

    final_branches = [b for b in branches if "FINAL" in b]
    summary = {
        "branches_total": len(branches),
        "final_branches": len(final_branches),
        "lost_and_found_count": len(lost),
        "untracked_count": len(untracked),
        "doc_missing_count": len(docs_missing),
        "code_missing_docs": len(code_missing),
        "next_actions": [
            "Review FINAL branches for lost_and_found artifacts.",
            "Triage untracked files and decide to add or archive.",
            "Resolve doc/code path mismatches.",
        ],
    }
    graph = build_graph(branches[: limits["max_branches"]], lost, untracked, docs_missing)

    return {
        "schema_version": 1,
        "req_id": req_id,
        "iso": now_iso_utc(),
        "scope": scope,
        "instruction": instruction,
        "mode": scan_mode,
        "branches": {
            "current": current,
            "local": branches,
            "final": final_branches,
            "errors": {"branch": branch_err, "remote": remote_err},
            "remotes": remotes,
        },
        "lost_and_found": {"paths": lost},
        "untracked": {"paths": untracked, "error": untracked_err},
        "doc_drift": {"docs_missing": docs_missing, "code_missing": code_missing},
        "summary": summary,
        "graph": graph,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pbdeep_responder.py", description="Daemon: respond to operator.pbdeep.request with report events.")
    ap.add_argument("--bus-dir", default=None)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--root", default="/pluribus")
    ap.add_argument("--poll", default="0.25")
    ap.add_argument("--run-for-s", default="0", help="Run for N seconds then exit (0 = run forever).")
    ap.add_argument("--since-ts", default=None, help="Only respond to triggers >= this UNIX timestamp.")
    ap.add_argument("--scan-mode", default="fast", choices=["fast", "deep", "noop"])
    ap.add_argument("--max-branches", default="80")
    ap.add_argument("--max-untracked", default="200")
    ap.add_argument("--max-lost", default="200")
    ap.add_argument("--max-refs", default="200")
    ap.add_argument("--timeout-s", default="10")
    ap.add_argument("--no-index", action="store_true", help="Skip rag/kg indexing.")
    args = ap.parse_args(argv)

    actor = default_actor(args.actor)
    bus_dir = default_bus_dir(args.bus_dir)
    poll_s = max(0.05, float(args.poll))
    run_for_s = max(0.0, float(args.run_for_s))
    since_ts = float(args.since_ts) if args.since_ts is not None else time.time()
    stop_at_ts = None if run_for_s <= 0 else (time.time() + run_for_s)
    root = Path(args.root).expanduser().resolve()
    iso_git = root / "nucleus" / "tools" / "iso_git.mjs"

    limits = {
        "max_branches": max(5, int(args.max_branches)),
        "max_untracked": max(10, int(args.max_untracked)),
        "max_lost": max(10, int(args.max_lost)),
        "max_refs": max(10, int(args.max_refs)),
        "timeout_s": max(3.0, float(args.timeout_s)),
    }
    index_timeout_s = min(6.0, limits["timeout_s"])
    enable_index = not args.no_index

    events_path = ensure_events_file(bus_dir)
    seen_ids: set[str] = set()

    emit_bus(
        bus_dir,
        topic="operator.pbdeep.responder.ready",
        kind="artifact",
        level="info",
        actor=actor,
        data={"since_ts": since_ts, "iso": now_iso_utc(), "pid": os.getpid()},
    )

    for trig in tail_events(events_path, since_ts=since_ts, poll_s=poll_s, stop_at_ts=stop_at_ts):
        trig_id = str(trig.get("id") or "")
        if trig_id and trig_id in seen_ids:
            continue
        if trig_id:
            seen_ids.add(trig_id)
        data = trig.get("data") if isinstance(trig.get("data"), dict) else {}
        req_id = str(data.get("req_id") or uuid.uuid4())
        instruction = str(data.get("instruction") or "").strip()
        scope = str(data.get("scope") or "repo")

        opts, cleaned_instruction = parse_instruction(instruction)
        scan_mode = opts.get("mode") or args.scan_mode
        max_branches = int(opts.get("max_branches") or limits["max_branches"])
        max_untracked = int(opts.get("max_untracked") or limits["max_untracked"])
        max_lost = int(opts.get("max_lost") or limits["max_lost"])
        max_refs = int(opts.get("max_refs") or limits["max_refs"])
        limits.update(
            {
                "max_branches": max_branches,
                "max_untracked": max_untracked,
                "max_lost": max_lost,
                "max_refs": max_refs,
            }
        )

        emit_progress(bus_dir, actor, req_id, "pbdeep", "start", 0, {"mode": scan_mode})

        report = build_report(
            root,
            req_id,
            cleaned_instruction or instruction,
            scope,
            scan_mode,
            limits,
            iso_git,
            bus_dir,
            actor,
        )

        report_dir = root / "agent_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"pbdeep_{req_id}.json"
        report_md_path = report_dir / f"pbdeep_{req_id}.md"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        report_md_path.write_text(render_markdown(report), encoding="utf-8")

        ingest_rhizome(root, report_path, bus_dir)

        index_summary = {"status": "skipped", "reason": "disabled"}
        index_path: Path | None = None
        rag_error = None
        kg_error = None
        if enable_index and scan_mode != "noop":
            emit_progress(bus_dir, actor, req_id, "index", "start", 85)
            rag_doc_id, rag_error = index_rag_report(
                root,
                report,
                report_path,
                bus_dir=bus_dir,
                actor=actor,
                timeout_s=index_timeout_s,
            )
            kg_summary, kg_error = index_kg_report(
                root,
                report,
                report_path,
                bus_dir=bus_dir,
                actor=actor,
                timeout_s=index_timeout_s,
            )
            index_summary = {
                "status": "ok" if not rag_error and not kg_error else "partial",
                "rag_doc_id": rag_doc_id,
                "rag_error": rag_error,
                "kg": kg_summary,
                "kg_error": kg_error,
                "embedding_mode": "off",
                "iso": now_iso_utc(),
            }
            index_path = report_dir / f"pbdeep_index_{req_id}.json"
            index_path.write_text(json.dumps(index_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            ingest_rhizome(root, index_path, bus_dir)
            emit_bus(
                bus_dir,
                topic="operator.pbdeep.index.updated",
                kind="artifact",
                level="info",
                actor=actor,
                data={
                    "req_id": req_id,
                    "summary": index_summary,
                    "index_path": str(index_path),
                    "iso": now_iso_utc(),
                },
            )
            emit_progress(bus_dir, actor, req_id, "index", "done", 95)
        elif scan_mode == "noop":
            index_summary = {"status": "skipped", "reason": "noop"}

        emit_progress(bus_dir, actor, req_id, "pbdeep", "done", 100, {"index_status": index_summary.get("status")})
        emit_bus(
            bus_dir,
            topic="operator.pbdeep.report",
            kind="artifact",
            level="info",
            actor=actor,
            data={
                "req_id": req_id,
                "scope": report.get("scope"),
                "mode": report.get("mode"),
                "summary": report.get("summary"),
                "report_path": str(report_path),
                "report_md_path": str(report_md_path),
                "index": index_summary,
                "index_path": str(index_path) if index_path else None,
                "graph": report.get("graph"),
                "iso": report.get("iso"),
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
