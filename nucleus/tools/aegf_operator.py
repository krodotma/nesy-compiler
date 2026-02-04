#!/usr/bin/env python3
"""AEGF operator: Aleatoric-Epistemic Gap Fill analysis.

Produces a lightweight gap analysis report and optionally emits bus events
for downstream tooling. This is a deterministic heuristic pass intended to
prevent silent failures and to seed deeper analysis workflows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None

ROOT = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus")).resolve()

DEFAULT_BUS_TOPIC = "aegf.analysis.request"
SECONDARY_TOPICS = {
    "aleatoric": "aegf.gap.aleatoric",
    "epistemic": "aegf.gap.epistemic",
    "ontological": "aegf.gap.ontological",
    "teleological": "aegf.gap.teleological",
    "hydration": "aegf.hydration.path",
    "superposition": "aegf.superposition.exploit",
    "entanglement": "aegf.entanglement.map",
    "trajectory": "aegf.latent.trajectory",
}

ALLOWED_MODES = {"aleatoric", "epistemic", "ontological", "teleological", "full"}

SCOPE_MAP = {
    "codebase": ROOT,
    "spec": ROOT / "nucleus" / "specs",
    "architecture": ROOT / "docs" / "architecture",
    "integration": ROOT / "docs",
}

SKIP_DIRS = {
    ".git",
    ".pluribus",
    ".pluribus_local",
    "node_modules",
    "site",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
}

SKIP_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".onnx",
    ".bin",
    ".mp4",
    ".mp3",
    ".wav",
    ".aiff",
    ".so",
    ".dylib",
    ".exe",
    ".class",
    ".jar",
    ".sqlite",
}

DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_MAX_FILES = 10_000

PATTERNS = {
    "aleatoric": [
        re.compile(r"\\brandom\\b", re.IGNORECASE),
        re.compile(r"\\bMath\\.random\\b"),
        re.compile(r"\\bnp\\.random\\b"),
        re.compile(r"\\bos\\.urandom\\b"),
        re.compile(r"\\buuid\\b", re.IGNORECASE),
        re.compile(r"\\btime\\.time\\b"),
        re.compile(r"\\bdatetime\\.now\\b"),
        re.compile(r"\\bsleep\\b", re.IGNORECASE),
    ],
    "epistemic": [
        re.compile(r"\\bTODO\\b"),
        re.compile(r"\\bFIXME\\b"),
        re.compile(r"\\bTBD\\b"),
        re.compile(r"\\bXXX\\b"),
        re.compile(r"NotImplemented", re.IGNORECASE),
    ],
    "ontological": [
        re.compile(r"\\brefactor\\b", re.IGNORECASE),
        re.compile(r"\\bhack\\b", re.IGNORECASE),
        re.compile(r"\\bworkaround\\b", re.IGNORECASE),
        re.compile(r"\\blegacy\\b", re.IGNORECASE),
        re.compile(r"\\bdeprecat", re.IGNORECASE),
    ],
    "teleological": [
        re.compile(r"\\bTODO\\b"),
        re.compile(r"\\bFIXME\\b"),
        re.compile(r"\\bintent\\b", re.IGNORECASE),
        re.compile(r"\\brequirement\\b", re.IGNORECASE),
        re.compile(r"\\bgoal\\b", re.IGNORECASE),
        re.compile(r"\\bdrift\\b", re.IGNORECASE),
    ],
}


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "aegf"


def resolve_bus_events_path(bus_dir: Path) -> Path:
    try:
        from nucleus.tools import agent_bus as bus  # type: ignore

        paths = bus.resolve_bus_paths(str(bus_dir))
        return Path(paths.events_path)
    except Exception:
        return bus_dir / "events.ndjson"


def append_ndjson(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)


def emit_bus(
    bus_dir: Path,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
    trace_id: str | None,
    run_id: str | None,
) -> str:
    event_id = str(uuid.uuid4())
    if agent_bus is not None:
        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=trace_id,
            run_id=run_id,
            durable=False,
        )
        return event_id
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "trace_id": trace_id,
        "run_id": run_id,
        "data": data,
    }
    append_ndjson(resolve_bus_events_path(bus_dir), event)
    return event_id


def resolve_scope(scope: str) -> Path:
    scope = (scope or "").strip()
    if scope in SCOPE_MAP:
        return SCOPE_MAP[scope]
    candidate = Path(scope)
    if candidate.exists():
        return candidate.resolve()
    return ROOT


def iter_text_files(base: Path, *, max_bytes: int, max_files: int) -> tuple[dict[str, int], list[tuple[Path, list[str]]]]:
    skipped = {"ignored": 0, "too_large": 0, "binary": 0}
    files: list[tuple[Path, list[str]]] = []
    count = 0
    for path in base.rglob("*"):
        if count >= max_files:
            break
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            skipped["ignored"] += 1
            continue
        if path.suffix.lower() in SKIP_EXT:
            skipped["binary"] += 1
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            skipped["too_large"] += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files.append((path, text.splitlines()))
        count += 1
    return skipped, files


def analyze_files(
    files: list[tuple[Path, list[str]]],
    *,
    scope_root: Path,
    mode: str,
    limit: int,
) -> dict[str, Any]:
    requested = ALLOWED_MODES if mode == "full" else {mode}
    findings: dict[str, list[dict[str, Any]]] = {k: [] for k in PATTERNS}
    counts = {k: 0 for k in PATTERNS}
    truncated = {k: False for k in PATTERNS}
    entanglement: dict[str, set[str]] = {}

    for path, lines in files:
        rel = str(path.relative_to(scope_root)) if scope_root in path.parents or path == scope_root else str(path)
        for idx, line in enumerate(lines, start=1):
            for category in requested:
                if category == "teleological" and path.suffix.lower() not in {".md", ".rst", ".txt"}:
                    continue
                if category != "teleological" and category not in PATTERNS:
                    continue
                for pattern in PATTERNS.get(category, []):
                    if not pattern.search(line):
                        continue
                    counts[category] += 1
                    if len(findings[category]) < limit:
                        findings[category].append(
                            {
                                "file": rel,
                                "line": idx,
                                "text": line.strip()[:200],
                                "match": pattern.pattern,
                            }
                        )
                    else:
                        truncated[category] = True
                    entanglement.setdefault(str(Path(rel).parent), set()).add(category)
                    break

    entanglement_map = [
        {"path": path, "categories": sorted(list(cats))}
        for path, cats in entanglement.items()
        if len(cats) > 1
    ]
    entanglement_map = entanglement_map[:limit]

    return {
        "findings": findings,
        "counts": counts,
        "truncated": truncated,
        "entanglement_map": entanglement_map,
    }


def build_hydration_paths(counts: dict[str, int]) -> list[str]:
    paths: list[str] = []
    if counts.get("epistemic"):
        paths.append("Hydrate epistemic gaps: document assumptions, add tests, and resolve TODO/FIXME markers.")
    if counts.get("ontological"):
        paths.append("Hydrate ontological gaps: refactor boundaries, clarify domain models, reduce legacy/hack debt.")
    if counts.get("teleological"):
        paths.append("Hydrate teleological gaps: reconcile docs/specs with current behavior and align intent.")
    if counts.get("aleatoric"):
        paths.append("Bound aleatoric gaps: add probabilistic tests, jitter budgets, and graceful degradation paths.")
    return paths


def build_superposition_opportunities(epistemic_items: list[dict[str, Any]], limit: int) -> list[str]:
    opportunities = []
    for item in epistemic_items[:limit]:
        opportunities.append(
            f"Explore alternative resolutions for {item['file']}:{item['line']} ({item['text']})"
        )
    return opportunities


def build_latent_trajectory(paths: list[str]) -> list[str]:
    if not paths:
        return []
    return [f"Step {idx + 1}: {path}" for idx, path in enumerate(paths)]


def build_report(*, scope: str, scope_root: Path, mode: str, hydrate: bool, limit: int, skipped: dict[str, int], analysis: dict[str, Any]) -> dict[str, Any]:
    counts = analysis["counts"]
    hydration_paths = build_hydration_paths(counts) if hydrate else []
    superposition = build_superposition_opportunities(analysis["findings"].get("epistemic", []), limit)
    latent = build_latent_trajectory(hydration_paths)

    return {
        "scope": scope,
        "scope_root": str(scope_root),
        "mode": mode,
        "counts": counts,
        "skipped": skipped,
        "findings": analysis["findings"],
        "truncated": analysis["truncated"],
        "hydration_paths": hydration_paths,
        "superposition_opportunities": superposition,
        "entanglement_map": analysis["entanglement_map"],
        "latent_trajectory": latent,
    }


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("AEGF GAP ANALYSIS")
    lines.append(f"Scope: {report['scope']} ({report['scope_root']})")
    lines.append(f"Mode: {report['mode']}")
    lines.append("")
    lines.append("COUNTS")
    for key, value in report["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    for section, label in [
        ("aleatoric", "ALEATORIC_GAPS"),
        ("epistemic", "EPISTEMIC_GAPS"),
        ("ontological", "ONTOLOGICAL_GAPS"),
        ("teleological", "TELEOLOGICAL_GAPS"),
    ]:
        lines.append(label)
        items = report["findings"].get(section, [])
        if not items:
            lines.append("  (none)")
        else:
            for item in items:
                lines.append(f"  - {item['file']}:{item['line']} {item['text']}")
        if report["truncated"].get(section):
            lines.append("  ... truncated")
        lines.append("")
    lines.append("HYDRATION_PATHS")
    if report["hydration_paths"]:
        for path in report["hydration_paths"]:
            lines.append(f"  - {path}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("SUPERPOSITION_OPPORTUNITIES")
    if report["superposition_opportunities"]:
        for item in report["superposition_opportunities"]:
            lines.append(f"  - {item}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("ENTANGLEMENT_MAP")
    if report["entanglement_map"]:
        for entry in report["entanglement_map"]:
            lines.append(f"  - {entry['path']}: {', '.join(entry['categories'])}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("LATENT_TRAJECTORY")
    if report["latent_trajectory"]:
        for entry in report["latent_trajectory"]:
            lines.append(f"  - {entry}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegf_operator.py",
        description="Aleatoric-Epistemic Gap Fill analysis operator.",
    )
    parser.add_argument("--scope", default="codebase", help="Analysis scope (codebase/spec/architecture/integration or path)")
    parser.add_argument("--mode", default="full", help="Gap detection mode: aleatoric|epistemic|ontological|teleological|full")
    parser.add_argument("--emit-bus", action="store_true", help="Emit analysis artifacts to bus")
    parser.add_argument("--hydrate", action="store_true", help="Include hydration path suggestions")
    parser.add_argument("--limit", type=int, default=50, help="Max findings per section")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Max file size to scan")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Max files to scan")
    parser.add_argument("--bus-dir", default=None, help="Bus directory (default: $PLURIBUS_BUS_DIR)")
    parser.add_argument("--actor", default=None, help="Actor identity (default: $PLURIBUS_ACTOR)")
    parser.add_argument("--req-id", default=None, help="Optional request id")
    parser.add_argument("--trace-id", default=None, help="Optional trace id")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    mode = args.mode.strip().lower()
    if mode not in ALLOWED_MODES:
        print(f"Invalid mode: {mode}", file=sys.stderr)
        return 2

    scope_root = resolve_scope(args.scope)
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").resolve()
    actor = (args.actor or default_actor()).strip()
    req_id = args.req_id or str(uuid.uuid4())
    trace_id = args.trace_id or str(uuid.uuid4())

    emit_bus(
        bus_dir,
        topic=DEFAULT_BUS_TOPIC,
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "trace_id": trace_id,
            "scope": args.scope,
            "mode": mode,
            "hydrate": bool(args.hydrate),
        },
        trace_id=trace_id,
        run_id=req_id,
    )

    skipped, files = iter_text_files(scope_root, max_bytes=args.max_bytes, max_files=args.max_files)
    analysis = analyze_files(files, scope_root=scope_root, mode=mode, limit=args.limit)
    report = build_report(
        scope=args.scope,
        scope_root=scope_root,
        mode=mode,
        hydrate=args.hydrate,
        limit=args.limit,
        skipped=skipped,
        analysis=analysis,
    )

    if args.emit_bus:
        for category in PATTERNS:
            if mode != "full" and category != mode:
                continue
            emit_bus(
                bus_dir,
                topic=SECONDARY_TOPICS[category],
                kind="artifact",
                level="info",
                actor=actor,
                data={
                    "req_id": req_id,
                    "trace_id": trace_id,
                    "scope": args.scope,
                    "mode": mode,
                    "count": report["counts"].get(category, 0),
                    "items": report["findings"].get(category, []),
                    "truncated": report["truncated"].get(category, False),
                },
                trace_id=trace_id,
                run_id=req_id,
            )
        emit_bus(
            bus_dir,
            topic=SECONDARY_TOPICS["entanglement"],
            kind="artifact",
            level="info",
            actor=actor,
            data={
                "req_id": req_id,
                "trace_id": trace_id,
                "scope": args.scope,
                "mode": mode,
                "items": report["entanglement_map"],
            },
            trace_id=trace_id,
            run_id=req_id,
        )
        emit_bus(
            bus_dir,
            topic=SECONDARY_TOPICS["superposition"],
            kind="artifact",
            level="info",
            actor=actor,
            data={
                "req_id": req_id,
                "trace_id": trace_id,
                "scope": args.scope,
                "mode": mode,
                "items": report["superposition_opportunities"],
            },
            trace_id=trace_id,
            run_id=req_id,
        )
        if report["hydration_paths"]:
            emit_bus(
                bus_dir,
                topic=SECONDARY_TOPICS["hydration"],
                kind="artifact",
                level="info",
                actor=actor,
                data={
                    "req_id": req_id,
                    "trace_id": trace_id,
                    "scope": args.scope,
                    "mode": mode,
                    "paths": report["hydration_paths"],
                },
                trace_id=trace_id,
                run_id=req_id,
            )
            emit_bus(
                bus_dir,
                topic=SECONDARY_TOPICS["trajectory"],
                kind="artifact",
                level="info",
                actor=actor,
                data={
                    "req_id": req_id,
                    "trace_id": trace_id,
                    "scope": args.scope,
                    "mode": mode,
                    "items": report["latent_trajectory"],
                },
                trace_id=trace_id,
                run_id=req_id,
            )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(render_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
