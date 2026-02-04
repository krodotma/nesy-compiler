#!/usr/bin/env python3
"""
Pulse Prompt Pane (10 lines, dense, 100%-width best effort).
Reads bus/vps/browser/hexis/art/services snapshots and emits a compact pane.
Intended for interactive shells; keep it lightweight (no network).
"""
from __future__ import annotations
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path("/pluribus")
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
BUS_EVENTS = BUS_DIR / "events.ndjson"
VPS_SESSION = Path("/var/lib/pluribus/.pluribus/vps_session.json")
BROWSER_DAEMON = Path("/var/lib/pluribus/.pluribus/browser_daemon.json")
ART_HISTORY = ROOT / "nucleus/art_dept/artifacts/history.ndjson"
MAX_LINES = 10
MIN_WIDTH = 60
TAIL_EVENT_LIMIT = int(os.environ.get("PULSE_TAIL_LINES", "160"))
SPARK_CHARS = " .:-=+*#"


def clamp(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def format_row(label: str, value: str) -> str:
    return f"{label.upper():<9} {value}"


def make_bar(current, total, width: int = 12) -> str:
    try:
        current_val = int(current)
        total_val = int(total)
    except Exception:
        return "n/a"
    if total_val <= 0:
        return "n/a"
    filled = int(round((current_val / total_val) * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def make_sparkline(values: list[int], width: int = 16) -> str:
    if not values:
        return "-" * width
    if len(values) < width:
        values = values + [0] * (width - len(values))
    if len(values) > width:
        step = max(1, len(values) // width)
        values = [sum(values[i : i + step]) for i in range(0, len(values), step)]
        values = values[:width]
    max_val = max(values) if values else 0
    if max_val <= 0:
        return "." * width
    chars = []
    for v in values[:width]:
        idx = int((v / max_val) * (len(SPARK_CHARS) - 1))
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def tail_lines(path: Path, max_lines: int) -> list[str]:
    if not path.exists() or max_lines <= 0:
        return []
    chunk_size = 8192
    data = b""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            while end > 0 and data.count(b"\n") <= max_lines:
                start = max(0, end - chunk_size)
                f.seek(start)
                data = f.read(end - start) + data
                end = start
    except Exception:
        return []
    lines = [ln.decode("utf-8", errors="ignore") for ln in data.splitlines() if ln.strip()]
    return lines[-max_lines:]


def parse_iso_ts(iso: str) -> float | None:
    try:
        return time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return None


def summarize_events(lines: list[str]) -> dict:
    if not lines:
        return {
            "count": 0,
            "spark": "-" * 16,
            "kinds": [],
            "topics": [],
            "actors": [],
            "error_ratio": 0.0,
        }
    counts_kind: dict[str, int] = {}
    counts_topic: dict[str, int] = {}
    counts_actor: dict[str, int] = {}
    ts_values: list[float] = []
    error_count = 0

    for line in lines:
        try:
            evt = json.loads(line)
        except Exception:
            continue
        kind = evt.get("kind") or "n/a"
        topic = evt.get("topic") or "n/a"
        actor = evt.get("actor") or "n/a"
        counts_kind[kind] = counts_kind.get(kind, 0) + 1
        prefix = topic.split(".")[0] if isinstance(topic, str) else "n/a"
        counts_topic[prefix] = counts_topic.get(prefix, 0) + 1
        counts_actor[actor] = counts_actor.get(actor, 0) + 1
        level = (evt.get("level") or "").lower()
        if level in {"error", "fatal"} or kind in {"error", "fatal"}:
            error_count += 1
        ts = evt.get("ts")
        if isinstance(ts, (int, float)):
            ts_values.append(float(ts))
        elif evt.get("iso"):
            parsed = parse_iso_ts(evt.get("iso"))
            if parsed:
                ts_values.append(parsed)

    buckets = [0] * 16
    if len(ts_values) >= 2:
        min_ts = min(ts_values)
        max_ts = max(ts_values)
        span = max(max_ts - min_ts, 1)
        for ts in ts_values:
            idx = int(((ts - min_ts) / span) * (len(buckets) - 1))
            idx = max(0, min(len(buckets) - 1, idx))
            buckets[idx] += 1
    spark = make_sparkline(buckets, width=16)

    def top_items(mapping: dict[str, int], limit: int = 3) -> list[tuple[str, int]]:
        return sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    total = max(1, len(lines))
    return {
        "count": len(lines),
        "spark": spark,
        "kinds": top_items(counts_kind),
        "topics": top_items(counts_topic),
        "actors": top_items(counts_actor),
        "error_ratio": error_count / total,
    }


def make_box(title: str, rows: list[str], width: int) -> list[str]:
    width = max(MIN_WIDTH, width)
    top = "+" + "-" * (width - 2) + "+"

    def box_line(content: str) -> str:
        content = clamp(content, width - 4)
        padding = " " * max(0, (width - 4) - len(content))
        return f"| {content}{padding} |"

    lines = [top, box_line(title)]
    lines.extend(box_line(row) for row in rows)
    lines.append(top)
    return lines

# Curation index locations (best effort)
CURATION_INDEX_CANDIDATES = [
    Path(os.environ.get("PLURIBUS_CURATION_INDEX", "")),
    Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))) / "nucleus" / "curation" / "items.ndjson",
    Path.home() / ".local" / "state" / "nucleus" / "curation" / "items.ndjson",
]


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def bus_info():
    size = "n/a"
    last_iso = "n/a"
    last_topic = "n/a"
    if BUS_EVENTS.exists():
        try:
            stat = BUS_EVENTS.stat()
            size = f"{stat.st_size/1024/1024:.2f}MB"
            with BUS_EVENTS.open("rb") as f:
                f.seek(max(stat.st_size - 4096, 0))
                tail = f.read().decode(errors="ignore").strip().splitlines()
            for line in reversed(tail):
                try:
                    evt = json.loads(line)
                    last_iso = evt.get("iso") or last_iso
                    last_topic = evt.get("topic") or last_topic
                    break
                except Exception:
                    continue
        except Exception:
            pass
    return size, last_iso, last_topic


def providers_info():
    data = load_json(VPS_SESSION)
    providers = data.get("providers", {})
    def fmt(name):
        ent = providers.get(name, {})
        status = "OK" if ent.get("available") else "blocked"
        if ent.get("error"):
            status += f" ({ent.get('error')})"
        return status
    fb = data.get("active_fallback") or "none"
    return {
        "gemini-web": fmt("gemini-web"),
        "claude-web": fmt("claude-web"),
        "chatgpt-web": fmt("chatgpt-web"),
        "fallback": fb,
    }


def browser_info():
    data = load_json(BROWSER_DAEMON)
    tabs = data.get("tabs", {})
    def status(name):
        tab = tabs.get(name, {})
        return tab.get("status") or "n/a"
    return {
        "gemini-web": status("gemini-web"),
        "claude-web": status("claude-web"),
        "chatgpt-web": status("chatgpt-web"),
    }


def hexis_counts():
    out = {}
    for buf in Path("/tmp").glob("*.buffer"):
        try:
            with buf.open() as f:
                lines = [l for l in f if l.strip()]
            out[buf.stem] = len(lines)
        except Exception:
            continue
    return out


def services_info():
    ServiceRegistry = None
    try:
        from service_registry import ServiceRegistry as _SR  # type: ignore
        ServiceRegistry = _SR
    except Exception:
        try:
            import sys as _sys
            _sys.path.append(str(ROOT))
            from nucleus.tools.service_registry import ServiceRegistry as _SR  # type: ignore
            ServiceRegistry = _SR
        except Exception:
            return {"defs": "n/a", "running": "n/a"}
    reg = ServiceRegistry(ROOT)
    reg.load()
    defs = len(reg.list_services())
    running = len([i for i in reg.list_instances() if i.status == "running"])
    return {"defs": defs, "running": running}


def art_info():
    if not ART_HISTORY.exists():
        return {"scene": "n/a", "mood": "n/a", "ts": "n/a"}
    try:
        with ART_HISTORY.open() as f:
            tail = f.readlines()[-1:]
        if not tail:
            return {"scene": "n/a", "mood": "n/a", "ts": "n/a"}
        evt = json.loads(tail[0])
        return {
            "scene": evt.get("scene_name") or "?",
            "mood": evt.get("mood") or "?",
            "ts": evt.get("iso") or evt.get("ts"),
        }
    except Exception:
        return {"scene": "n/a", "mood": "n/a", "ts": "n/a"}

def curation_info():
    path = None
    for cand in CURATION_INDEX_CANDIDATES:
        try:
            if cand and cand.exists() and cand.is_file():
                path = cand
                break
        except Exception:
            continue
    if not path:
        return {"items": "n/a", "last_iso": "n/a"}
    try:
        lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        items = len(lines)
        last_iso = "n/a"
        if lines:
            try:
                last = json.loads(lines[-1])
                last_iso = last.get("iso") or last_iso
            except Exception:
                pass
        return {"items": items, "last_iso": last_iso}
    except Exception:
        return {"items": "n/a", "last_iso": "n/a"}


def build_pulse_lines(cols: int) -> list[str]:
    size, last_iso, last_topic = bus_info()
    curation = curation_info()
    providers = providers_info()
    browser = browser_info()
    hexis = hexis_counts()
    services = services_info()
    art = art_info()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        size_mb = float(str(size).replace("MB", "")) if str(size).endswith("MB") else 0
    except Exception:
        size_mb = 0
    bus_bar = make_bar(size_mb, 1024, width=10)
    service_bar = make_bar(services.get("running"), services.get("defs"), width=10)
    hexis_str = " ".join(f"{k}:{v}" for k, v in hexis.items()) or "none"

    tail = tail_lines(BUS_EVENTS, TAIL_EVENT_LIMIT)
    summary = summarize_events(tail)
    kind_row = " ".join([f"{k}:{'#' * min(8, v)}" for k, v in summary["kinds"]]) or "n/a"
    topic_row = " ".join([f"{k}:{'#' * min(8, v)}" for k, v in summary["topics"]]) or "n/a"
    actor_row = " ".join([f"{k}:{'#' * min(6, v)}" for k, v in summary["actors"]]) or "n/a"
    error_bar = make_bar(int(summary["error_ratio"] * 100), 100, width=10)

    flow = "BUS->PROV->BROW->SERV->ART"
    rows = [
        format_row("FLOW", flow),
        format_row("BUS", f"[{bus_bar}] last={last_iso} topic={last_topic}"),
        format_row("EVENTS", f"{summary['spark']} err[{error_bar}]"),
        format_row("KINDS", kind_row),
        format_row("TOPICS", topic_row),
        format_row("ACTORS", actor_row),
        format_row("SERVICES", f"[{service_bar}] defs={services['defs']} run={services['running']}"),
        format_row("PROVIDERS", f"g={providers['gemini-web']} c={providers['claude-web']} o={providers['chatgpt-web']} fb={providers['fallback']}"),
        format_row("BROWSER", f"g={browser['gemini-web']} c={browser['claude-web']} o={browser['chatgpt-web']}"),
        format_row("ART", f"{art['scene']} {art['mood']} {art['ts']}"),
    ]

    title = f"PLURIBUS PULSE {now}"
    return make_box(title, rows, cols)


def main():
    cols = shutil.get_terminal_size((100, 20)).columns
    lines = build_pulse_lines(cols)

    for ln in lines[:MAX_LINES]:
        if len(ln) > cols:
            ln = ln[:cols]
        print(ln)

if __name__ == "__main__":
    main()
