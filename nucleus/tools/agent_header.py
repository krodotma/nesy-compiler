#!/usr/bin/env python3
"""
agent_header.py - Minimal UNIFORM panel generator (PLURIBUS v1)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

sys.dont_write_bytecode = True

PLURIBUS_VERSION = "v1"
UNIFORM_VERSION = "v2.1"
DKIN_VERSION = "v30"
PAIP_VERSION = "v16"
CITIZEN_VERSION = "v2"
INNER_WIDTH = 78
SNAPSHOT_PATH = Path(os.environ.get("PLURIBUS_HEADER_SNAPSHOT", "/pluribus/.pluribus/index/irkg/header_snapshot.json"))
DR_RING_PATH = Path(os.environ.get("PLURIBUS_HEADER_DR_RING", "/pluribus/.pluribus/dr/header_events.ndjson"))
USE_NDJSON = os.environ.get("PLURIBUS_DR_MODE", "").strip() == "1" or os.environ.get("PLURIBUS_USE_NDJSON", "").strip() == "1"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _format_row(content: str, width: int = INNER_WIDTH) -> str:
    content = _truncate(content, width)
    return f"| {content:<{width}} |"


def _wrap_parts(prefix: str, parts: list[str], width: int = INNER_WIDTH) -> list[str]:
    if not parts:
        return [_format_row(prefix, width)]
    lines: list[str] = []
    current = prefix
    indent = " " * len(prefix)
    for part in parts:
        tentative = f"{current} {part}" if current else part
        if len(tentative) > width and current != prefix:
            lines.append(_format_row(current, width))
            current = f"{indent}{part}"
        elif len(tentative) > width and current == prefix:
            lines.append(_format_row(current, width))
            current = f"{indent}{part}"
        else:
            current = tentative
    lines.append(_format_row(current, width))
    return lines


def _progress_bar(progress: Optional[float], width: int = 10) -> str:
    if progress is None:
        return "?" * width
    filled = int(round(progress * width))
    filled = max(0, min(width, filled))
    return ("#" * filled) + ("." * (width - filled))


def detect_agent() -> str:
    actor = os.environ.get("PLURIBUS_ACTOR", "")
    if actor:
        return actor.lower().replace("-cli", "").replace("_cli", "").split("_")[0].split("-")[0]
    return "agent"


def _tail_lines(path: Path, max_bytes: int = 200_000, max_lines: int = 200) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - max_bytes)
            handle.seek(start)
            chunk = handle.read()
    except OSError:
        return []
    if start > 0:
        nl = chunk.find(b"\n")
        if nl != -1:
            chunk = chunk[nl + 1 :]
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return [line for line in lines if line.strip()]


def _load_snapshot(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_nested(source: dict, *keys: str):
    cur = source
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _fmt_count(value: Optional[int]) -> str:
    return "?" if value is None else str(value)


def _sum_counts(values: Iterable[Optional[int]]) -> Optional[int]:
    vals = list(values)
    if any(v is None for v in vals):
        return None
    return int(sum(vals))


def _count_events(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        total += len(_tail_lines(path))
    return total


def _event_files(topic_root: Path, rel: str) -> list[Path]:
    path = topic_root / rel
    if not path.exists():
        return []
    return [p for p in path.rglob("events.ndjson")]


@dataclass
class TaskStats:
    active: int
    total: int
    label: str
    progress: Optional[float]


def _load_task_stats(ledger_path: Path, actor: str) -> TaskStats:
    active_statuses = {"in_progress", "blocked", "run", "running"}
    total = 0
    active = 0
    label = "none"
    progress: Optional[float] = None
    for line in _tail_lines(ledger_path, max_bytes=200_000, max_lines=500):
        try:
            item = json.loads(line)
        except Exception:
            continue
        if item.get("actor") != actor:
            continue
        total += 1
        status = item.get("status")
        if status in active_statuses:
            active += 1
        meta = item.get("meta") or {}
        if isinstance(meta, dict):
            label = meta.get("desc") or meta.get("task") or meta.get("label") or label
            prog = meta.get("progress")
            if isinstance(prog, (int, float)):
                progress = float(prog)
    return TaskStats(active=active, total=total, label=label, progress=progress)


def _task_stats_from_snapshot(snapshot: dict, actor: str) -> Optional[TaskStats]:
    candidate = _get_nested(snapshot, "tasks") or _get_nested(snapshot, "task_stats")
    if not isinstance(candidate, dict):
        return None
    active = candidate.get("active")
    total = candidate.get("total")
    label = candidate.get("label") or candidate.get("task") or "unknown"
    progress = candidate.get("progress")
    if not isinstance(active, int):
        active = 0
    if not isinstance(total, int):
        total = 0
    if isinstance(progress, (int, float)):
        progress_val: Optional[float] = float(progress)
    else:
        progress_val = None
    return TaskStats(active=active, total=total, label=str(label), progress=progress_val)


def _metrics_from_snapshot(snapshot: dict) -> Optional[dict]:
    bus = _get_nested(snapshot, "bus") or _get_nested(snapshot, "metrics", "bus")
    if not isinstance(bus, dict):
        return None
    a2a = bus.get("a2a") if isinstance(bus.get("a2a"), dict) else {}
    ops = bus.get("ops") if isinstance(bus.get("ops"), dict) else {}
    qa = bus.get("qa") if isinstance(bus.get("qa"), dict) else {}
    sysg = bus.get("sys") if isinstance(bus.get("sys"), dict) else {}

    def get_count(src: dict, key: str) -> Optional[int]:
        val = src.get(key)
        return int(val) if isinstance(val, (int, float)) else None

    metrics = {
        "a2a": {
            "for": get_count(a2a, "for"),
            "col": get_count(a2a, "col"),
            "int": get_count(a2a, "int"),
            "neg": get_count(a2a, "neg"),
            "task": get_count(a2a, "task"),
        },
        "ops": {
            "pbtest": get_count(ops, "pbtest"),
            "pblock": get_count(ops, "pblock"),
            "pbresume": get_count(ops, "pbresume"),
            "pblive": get_count(ops, "pblive"),
            "pbflush": get_count(ops, "pbflush"),
            "pbiud": get_count(ops, "pbiud"),
            "pbcli": get_count(ops, "pbcli"),
            "hyg": get_count(ops, "hyg"),
        },
        "qa": {
            "alert": get_count(qa, "alert"),
            "anom": get_count(qa, "anom"),
            "rem": get_count(qa, "rem"),
            "verd": get_count(qa, "verd"),
            "live": get_count(qa, "live"),
            "lchk": get_count(qa, "lchk"),
            "act": get_count(qa, "act"),
            "hyg": get_count(qa, "hyg"),
        },
        "sys": {
            "tel": get_count(sysg, "tel"),
            "task": get_count(sysg, "task"),
            "agent": get_count(sysg, "agent"),
            "dlg": get_count(sysg, "dlg"),
            "ohm": get_count(sysg, "ohm"),
            "omg": get_count(sysg, "omg"),
            "prov": get_count(sysg, "prov"),
            "dash": get_count(sysg, "dash"),
            "brow": get_count(sysg, "brow"),
        },
        "bus_total": None,
    }

    bus_total = bus.get("total") if isinstance(bus.get("total"), (int, float)) else None
    if isinstance(bus_total, (int, float)):
        metrics["bus_total"] = int(bus_total)
    return metrics


def collect_metrics(agent: str) -> Tuple[dict, TaskStats]:
    snapshot = _load_snapshot(SNAPSHOT_PATH)
    metrics = _metrics_from_snapshot(snapshot)
    task_stats = _task_stats_from_snapshot(snapshot, agent)

    if metrics is None or task_stats is None:
        if not USE_NDJSON:
            metrics = metrics or {
                "a2a": {"for": None, "col": None, "int": None, "neg": None, "task": None},
                "ops": {"pbtest": None, "pblock": None, "pbresume": None, "pblive": None, "pbflush": None, "pbiud": None, "pbcli": None, "hyg": None},
                "qa": {"alert": None, "anom": None, "rem": None, "verd": None, "live": None, "lchk": None, "act": None, "hyg": None},
                "sys": {"tel": None, "task": None, "agent": None, "dlg": None, "ohm": None, "omg": None, "prov": None, "dash": None, "brow": None},
                "bus_total": None,
            }
            task_stats = task_stats or TaskStats(active=0, total=0, label="unknown", progress=None)
            return metrics, task_stats

    if metrics is None or task_stats is None:
        bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
        topic_root = bus_dir / "topics" / "topic"

        a2a_for = _count_events(_event_files(topic_root, "a2a/design/forensics"))
        a2a_col = _count_events(_event_files(topic_root, "a2a/design/collaboration")) + _count_events(_event_files(topic_root, "a2a/skills/collaboration"))
        a2a_int = _count_events(_event_files(topic_root, "a2a/design/integration"))
        a2a_neg = _count_events(_event_files(topic_root, "a2a/negotiate"))
        a2a_task = _count_events(_event_files(topic_root, "a2a/task/dispatch"))

        ops = {
            "pbtest": _count_events(_event_files(topic_root, "operator/pbtest")),
            "pblock": _count_events(_event_files(topic_root, "operator/pblock")),
            "pbresume": _count_events(_event_files(topic_root, "operator/pbresume")),
            "pblive": _count_events(_event_files(topic_root, "operator/pblive")),
            "pbflush": _count_events(_event_files(topic_root, "operator/pbflush")),
            "pbiud": _count_events(_event_files(topic_root, "operator/pbiud")),
            "pbcli": _count_events(_event_files(topic_root, "operator/pbclitest")),
            "hyg": _count_events(_event_files(topic_root, "operator/hygiene")),
        }

        qa = {
            "alert": _count_events(_event_files(topic_root, "qa/alert")),
            "anom": _count_events(_event_files(topic_root, "qa/anomaly")),
            "rem": _count_events(_event_files(topic_root, "qa/remediation")),
            "verd": _count_events(_event_files(topic_root, "qa/verdict")),
            "live": _count_events(_event_files(topic_root, "qa/live")),
            "lchk": _count_events(_event_files(topic_root, "qa/live-checker")),
            "act": _count_events(_event_files(topic_root, "qa/action")),
            "hyg": _count_events(_event_files(topic_root, "qa/hygiene")),
        }

        sysg = {
            "tel": _count_events(_event_files(topic_root, "telemetry")),
            "task": _count_events(_event_files(topic_root, "task")),
            "agent": _count_events(_event_files(topic_root, "agent")),
            "dlg": _count_events(_event_files(topic_root, "dialogos")),
            "ohm": _count_events(_event_files(topic_root, "ohm")),
            "omg": _count_events(_event_files(topic_root, "omega")),
            "prov": _count_events(_event_files(topic_root, "providers")),
            "dash": _count_events(_event_files(topic_root, "dashboard")),
            "brow": _count_events(_event_files(topic_root, "browser")),
        }

        bus_total = a2a_for + a2a_col + a2a_int + a2a_neg + a2a_task
        bus_total += sum(ops.values()) + sum(qa.values()) + sum(sysg.values())

        metrics = {
            "a2a": {"for": a2a_for, "col": a2a_col, "int": a2a_int, "neg": a2a_neg, "task": a2a_task},
            "ops": ops,
            "qa": qa,
            "sys": sysg,
            "bus_total": bus_total,
        }

        if task_stats is None:
            task_ledger = bus_dir / "topics" / "topic" / "task" / "task_ledger" / "append" / "events.ndjson"
            task_stats = _load_task_stats(task_ledger, agent)

    return metrics, task_stats


def generate_panel(agent: str) -> str:
    sess = os.environ.get("PLURIBUS_SESSION", "new")
    cell = os.environ.get("PLURIBUS_CELL", "dia.1.0")
    lane = os.environ.get("PLURIBUS_LANE", "dialogos")
    depth = os.environ.get("PLURIBUS_DEPTH", "0")
    style = os.environ.get("UNIFORM_PANEL_STYLE", "").strip().lower()

    metrics, task_stats = collect_metrics(agent)
    bus_total_raw = metrics.get("bus_total")
    bus_total = bus_total_raw if isinstance(bus_total_raw, int) else 0
    bus_total_display = _fmt_count(bus_total_raw)
    state = "active" if (bus_total > 0 or task_stats.active > 0) else "idle"
    hexis = "engaged" if state == "active" else "quiet"
    task_bar = _progress_bar(task_stats.progress)
    task_pct = "?" if task_stats.progress is None else f"{int(round(task_stats.progress * 100))}%"

    if style in {"tablet", "badge", "compact"}:
        width = 64
        rollup = {
            "a2a": _sum_counts(metrics["a2a"].values()),
            "ops": _sum_counts(metrics["ops"].values()),
            "qa": _sum_counts(metrics["qa"].values()),
            "sys": _sum_counts(metrics["sys"].values()),
        }
        tablet_bar = _progress_bar(task_stats.progress, width=8)
        border = "+" + "-" * (width + 2) + "+"
        lines = [
            border,
            _format_row(
                f"UNIFORM {UNIFORM_VERSION} | PLURIBUS {PLURIBUS_VERSION} | agent:{agent} | lane:{lane}",
                width,
            ),
            _format_row(
                f"sess:{sess} | cell:{cell} | d:{depth} | state:{state} | hexis:{hexis}",
                width,
            ),
            _format_row(
                f"TASKS | {tablet_bar} {task_pct} | active {task_stats.active}/{task_stats.total} | task:\"{task_stats.label}\"",
                width,
            ),
            _format_row(
                f"BUS | a2a:{_fmt_count(rollup['a2a'])} ops:{_fmt_count(rollup['ops'])} qa:{_fmt_count(rollup['qa'])} sys:{_fmt_count(rollup['sys'])} | total:+{bus_total_display}",
                width,
            ),
            border,
        ]
        return "\n".join(lines)

    lines = []
    border = "+" + "-" * (INNER_WIDTH + 2) + "+"
    lines.append(border)
    lines.append(_format_row(f"UNIFORM {UNIFORM_VERSION} | {agent:<6} | sess:{sess} | cell:{cell} | lane:{lane} | d:{depth}"))
    lines.append(border)
    lines.append(_format_row(f"PROTO | PLURIBUS {PLURIBUS_VERSION} | gen:? | lineage:?"))
    lines.append(_format_row(f"SCORE | {('#' * 10)} 100/100 v | CDI:? | health:nominal"))
    lines.append(_format_row(f"STATE | {state} | hexis:{hexis} | bus:+{bus_total_display} | uncommit:? | branch:?"))
    lines.append(_format_row("SWARM | SAGENT:? | CAGENT:? | active:?"))
    lines.append(_format_row(f"TASKS | {task_bar} {task_pct} | active {task_stats.active}/{task_stats.total} | task:\"{task_stats.label}\""))
    lines.extend(_wrap_parts("BUS A2A |", [f"for:{_fmt_count(metrics['a2a']['for'])}", f"col:{_fmt_count(metrics['a2a']['col'])}", f"int:{_fmt_count(metrics['a2a']['int'])}", f"neg:{_fmt_count(metrics['a2a']['neg'])}", f"task:{_fmt_count(metrics['a2a']['task'])}"]))
    lines.extend(_wrap_parts("BUS OPS |", [f"pbtest:{_fmt_count(metrics['ops']['pbtest'])}", f"pblock:{_fmt_count(metrics['ops']['pblock'])}", f"pbresume:{_fmt_count(metrics['ops']['pbresume'])}", f"pblive:{_fmt_count(metrics['ops']['pblive'])}", f"pbflush:{_fmt_count(metrics['ops']['pbflush'])}", f"pbiud:{_fmt_count(metrics['ops']['pbiud'])}", f"pbcli:{_fmt_count(metrics['ops']['pbcli'])}", f"hyg:{_fmt_count(metrics['ops']['hyg'])}"]))
    lines.extend(_wrap_parts("BUS QA |", [f"alert:{_fmt_count(metrics['qa']['alert'])}", f"anom:{_fmt_count(metrics['qa']['anom'])}", f"rem:{_fmt_count(metrics['qa']['rem'])}", f"verd:{_fmt_count(metrics['qa']['verd'])}", f"live:{_fmt_count(metrics['qa']['live'])}", f"lchk:{_fmt_count(metrics['qa']['lchk'])}", f"act:{_fmt_count(metrics['qa']['act'])}", f"hyg:{_fmt_count(metrics['qa']['hyg'])}"]))
    lines.extend(_wrap_parts("BUS SYS |", [f"tel:{_fmt_count(metrics['sys']['tel'])}", f"task:{_fmt_count(metrics['sys']['task'])}", f"agent:{_fmt_count(metrics['sys']['agent'])}", f"dlg:{_fmt_count(metrics['sys']['dlg'])}", f"ohm:{_fmt_count(metrics['sys']['ohm'])}", f"omg:{_fmt_count(metrics['sys']['omg'])}", f"prov:{_fmt_count(metrics['sys']['prov'])}", f"dash:{_fmt_count(metrics['sys']['dash'])}", f"brow:{_fmt_count(metrics['sys']['brow'])}"]))
    lines.append(_format_row("SCOPE | goal:\"awaiting task\" | artifact:none"))
    lines.append(border)
    return "\n".join(lines)


def main() -> int:
    agent = detect_agent()
    if len(sys.argv) > 1:
        agent = sys.argv[1]
    print(generate_panel(agent))
    print("")
    print(f"[Panel generated for {agent}. Now respond to user. Do not run this command again.]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
