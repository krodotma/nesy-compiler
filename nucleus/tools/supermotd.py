#!/usr/bin/env python3
"""
supermotd.py

Unix-boot-style "system stream" for Pluribus: curated, idiolect-aware meta events
from the append-only bus.

This is intentionally a thin reader:
- No provider calls
- No mutations (reads events.ndjson)
- Designed for tmux panes and quick operator situational awareness
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys_dont_write_bytecode = True


@dataclass(frozen=True)
class MotdLine:
    iso: str
    severity: str
    subsystem: str
    message: str
    topic: str
    actor: str


def _as_dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _truncate(s: str, max_chars: int = 220) -> str:
    t = " ".join((s or "").replace("\r", "").split()).strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def event_to_line(e: dict) -> MotdLine | None:
    topic = str(e.get("topic") or "")
    level = str(e.get("level") or "info")
    actor = str(e.get("actor") or "")
    iso = str(e.get("iso") or "")
    data = _as_dict(e.get("data"))

    if topic == "dialogos.cell.output":
        t = str(data.get("type") or "text")
        if level != "error" and t != "error":
            return None
        return MotdLine(
            iso=iso,
            severity="error",
            subsystem="DIALOGOS",
            message=_truncate(str(data.get("content") or "")) or "cell.output error",
            topic=topic,
            actor=actor,
        )

    if topic == "dialogos.cell.end":
        ok = bool(data.get("ok"))
        status = str(data.get("status") or ("success" if ok else "error"))
        providers = data.get("providers") if isinstance(data.get("providers"), list) else []
        p = ",".join([str(x) for x in providers if str(x).strip()])
        msg = f"cell.end status={status}" + (f" providers=[{p}]" if p else "")
        return MotdLine(iso=iso, severity=("ok" if ok else "error"), subsystem="DIALOGOS", message=msg, topic=topic, actor=actor)

    if topic == "dialogos.submit":
        providers = data.get("providers") if isinstance(data.get("providers"), list) else []
        p = ",".join([str(x) for x in providers if str(x).strip()])
        return MotdLine(iso=iso, severity=("warn" if level == "warn" else "info"), subsystem="DIALOGOS", message=f"submit → [{p or 'auto'}]", topic=topic, actor=actor)

    if topic == "plurichat.lens.decision":
        depth = str(data.get("depth") or "")
        lane = str(data.get("lane") or "")
        topo = str(data.get("topology") or "single")
        fanout = data.get("fanout") or 1
        provider = str(data.get("selected_provider") or data.get("provider") or "")
        persona = str(data.get("persona") or "")
        msg = f"LENS depth={depth} lane={lane} topo={topo}×{fanout} → {provider or 'auto'}" + (f" persona={persona}" if persona else "")
        return MotdLine(iso=iso, severity="info", subsystem="LENS", message=msg, topic=topic, actor=actor)

    if topic.startswith("strp."):
        kind = str(data.get("kind") or "")
        goal = str(data.get("goal") or data.get("goal_summary") or "")
        msg = _truncate((f"{kind} — {goal}" if kind and goal else goal or topic), 240)
        return MotdLine(iso=iso, severity=("error" if level == "error" else "info"), subsystem="STRP", message=msg, topic=topic, actor=actor)

    if topic.startswith("verify.") or topic.startswith("tests."):
        msg = _truncate(str(data.get("summary") or data.get("cmd") or e.get("semantic") or topic))
        return MotdLine(iso=iso, severity=("error" if level == "error" else "info"), subsystem="VERIFY", message=msg, topic=topic, actor=actor)

    if topic.startswith("operator.pbflush."):
        rid = str(data.get("req_id") or data.get("request_id") or "").strip()
        req = rid[:8] if rid else ""
        intent = str(data.get("intent") or "").strip()
        msg = _truncate(str(data.get("message") or data.get("reason") or ""), 180)
        base = f"{topic.replace('operator.pbflush.', 'pbflush.')} " + (f"req={req} " if req else "") + (f"intent={intent}" if intent else "")
        base = base.strip()
        line = f"{base} — {msg}" if msg else base
        return MotdLine(
            iso=iso,
            severity=("warn" if topic.endswith(".request") else ("error" if level == "error" else "info")),
            subsystem="PBFLUSH",
            message=line,
            topic=topic,
            actor=actor,
        )

    if topic == "ckin.report":
        pb = _as_dict(data.get("pbflush"))
        latest = _as_dict(pb.get("latest_request"))
        rid = str(latest.get("req_id") or "").strip()
        req = rid[:8] if rid else ""
        acks = pb.get("acks_window") or 0
        reqs = pb.get("requests_window") or 0
        ver = str(data.get("protocol_version") or "").strip() or "?"
        msg = f"CKIN v{ver} pbflush(req={reqs} acks={acks}" + (f" last={req}" if req else "") + ")"
        return MotdLine(iso=iso, severity="info", subsystem="CKIN", message=msg, topic=topic, actor=actor)

    if topic in {"infer_sync.request", "infer_sync.response"} and str(data.get("intent") or "") == "pbflush":
        rid = str(data.get("req_id") or data.get("request_id") or "").strip()
        req = rid[:8] if rid else ""
        msg = _truncate(str(data.get("message") or data.get("reason") or ""), 180)
        base = f"{topic} " + (f"req={req}" if req else "")
        base = base.strip()
        line = f"{base} — {msg}" if msg else base
        return MotdLine(iso=iso, severity=("error" if level == "error" else "info"), subsystem="PBFLUSH", message=line, topic=topic, actor=actor)

    if topic.startswith("supermotd."):
        msg = _truncate(str(data.get("text") or e.get("semantic") or json.dumps(data, ensure_ascii=False)))
        return MotdLine(iso=iso, severity=("error" if level == "error" else "info"), subsystem="SUPERMOTD", message=msg, topic=topic, actor=actor)

    if topic.startswith("mcp."):
        if topic == "mcp.host.call":
            rid = str(data.get("req_id") or "").strip()
            req = rid[:8] if rid else ""
            server = str(data.get("server") or "").strip()
            tool = str(data.get("tool") or "").strip()
            msg = f"host.call " + (f"req={req} " if req else "") + (f"{server}.{tool}" if server and tool else server or tool or "")
            return MotdLine(iso=iso, severity="info", subsystem="MCP", message=msg.strip(), topic=topic, actor=actor)
        if topic == "mcp.host.response":
            rid = str(data.get("req_id") or "").strip()
            req = rid[:8] if rid else ""
            ok = bool(data.get("ok"))
            server = str(data.get("server") or "").strip()
            tool = str(data.get("tool") or "").strip()
            msg = f"host.response " + (f"req={req} " if req else "") + (("ok " if ok else "err ")) + (f"{server}.{tool}" if server and tool else server or tool or "")
            return MotdLine(iso=iso, severity=("ok" if ok else "error"), subsystem="MCP", message=msg.strip(), topic=topic, actor=actor)
        msg = _truncate(str(data.get("status") or data.get("server") or data.get("tool") or data.get("text") or e.get("semantic") or topic))
        return MotdLine(iso=iso, severity=("error" if level == "error" else "info"), subsystem="MCP", message=msg, topic=topic, actor=actor)

    if topic.startswith("a2a."):
        rid = str(data.get("req_id") or data.get("request_id") or "").strip()
        req = rid[:8] if rid else ""
        if topic == "a2a.negotiate.request":
            initiator = str(data.get("initiator") or data.get("from") or actor or "").strip()
            target = str(data.get("target") or "").strip()
            constraints = _as_dict(data.get("constraints"))
            req_caps = constraints.get("required_capabilities")
            caps = req_caps if isinstance(req_caps, list) else []
            msg = f"negotiate.request " + (f"req={req} " if req else "") + (f"from={initiator} " if initiator else "") + (f"to={target} " if target else "") + (f"caps={len(caps)}" if caps else "")
            return MotdLine(iso=iso, severity="info", subsystem="A2A", message=msg.strip(), topic=topic, actor=actor)
        if topic == "a2a.negotiate.response":
            decision = str(data.get("decision") or "").strip()
            msg = f"negotiate.response " + (f"req={req} " if req else "") + (f"decision={decision}" if decision else "")
            sev = "ok" if decision in {"agree"} else ("warn" if decision in {"negotiate"} else "error")
            return MotdLine(iso=iso, severity=sev, subsystem="A2A", message=msg.strip(), topic=topic, actor=actor)
        if topic == "a2a.decline":
            reason = _truncate(str(data.get("reason") or "decline"), 180)
            msg = f"decline " + (f"req={req}" if req else "")
            msg = f"{msg} — {reason}" if reason else msg
            return MotdLine(iso=iso, severity="warn", subsystem="A2A", message=msg.strip(), topic=topic, actor=actor)
        if topic == "a2a.redirect":
            to = str(data.get("redirect_to") or "").strip()
            reason = _truncate(str(data.get("reason") or "redirect"), 180)
            msg = f"redirect " + (f"req={req} " if req else "") + (f"to={to}" if to else "")
            msg = f"{msg} — {reason}" if reason else msg
            return MotdLine(iso=iso, severity="info", subsystem="A2A", message=msg.strip(), topic=topic, actor=actor)
        msg = _truncate(str(data.get("summary") or data.get("text") or json.dumps(data, ensure_ascii=False)))
        return MotdLine(iso=iso, severity=("error" if level == "error" else "info"), subsystem="A2A", message=msg, topic=topic, actor=actor)

    if topic == "studio.flow.roundtrip":
        rid = str(data.get("req_id") or "").strip()
        req = rid[:8] if rid else ""
        ok = bool(data.get("ok"))
        in_path = str(data.get("in_path") or "").strip()
        msg = f"flow.roundtrip " + (f"req={req} " if req else "") + ("ok" if ok else "fail")
        if in_path:
            msg += f" in={Path(in_path).name}"
        return MotdLine(iso=iso, severity=("ok" if ok else "error"), subsystem="STUDIO", message=msg.strip(), topic=topic, actor=actor)

    return None


def iter_events(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _tail_bytes(path: Path, *, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        return b""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start, os.SEEK_SET)
            chunk = f.read()
    except Exception:
        return b""
    if not chunk:
        return b""
    if start > 0:
        # Drop the first partial line so JSON parsing isn't poisoned.
        nl = chunk.find(b"\n")
        if nl >= 0:
            chunk = chunk[nl + 1 :]
    return chunk


def iter_tail_events(path: Path, *, max_bytes: int = 2_000_000) -> Iterable[dict]:
    """Yield recent NDJSON events without scanning the entire log."""
    chunk = _tail_bytes(path, max_bytes=max_bytes)
    if not chunk:
        return
    for raw in chunk.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _estimate_event_count(*, events_path: Path, sample_bytes: int = 2_000_000) -> int:
    """
    Heuristic count: estimate events ~= size_bytes / avg_line_bytes, sampled from the tail.
    Avoids O(n) scans for multi-GB buses.
    """
    try:
        size = events_path.stat().st_size
    except Exception:
        size = 0
    if size <= 0:
        return 0
    chunk = _tail_bytes(events_path, max_bytes=sample_bytes)
    if not chunk:
        return 0
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return 0
    avg = sum(len(ln) for ln in lines) / max(1, len(lines))
    avg = max(32.0, avg)  # guard against tiny/sparse samples
    return int(size / avg)


def format_line(l: MotdLine) -> str:
    ts = l.iso[11:19] if len(l.iso) >= 19 else "--:--:--"
    return f"{ts} [{l.subsystem}] {l.message}"


def cmd_tail(*, events_path: Path, limit: int, follow: bool) -> int:
    lines: list[str] = []
    # Fast path: scan only the tail (multi-GB buses must still render quickly).
    for e in iter_tail_events(events_path, max_bytes=2_000_000):
        ml = event_to_line(e)
        if not ml:
            continue
        lines.append(format_line(ml))
    for x in lines[-limit:]:
        print(x)

    if not follow:
        return 0

    last_size = events_path.stat().st_size if events_path.exists() else 0
    while True:
        try:
            size = events_path.stat().st_size
        except Exception:
            size = 0
        if size < last_size:
            last_size = 0
        if size > last_size:
            with events_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(last_size, os.SEEK_SET)
                chunk = f.read()
            last_size = size
            for raw in chunk.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    e = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(e, dict):
                    continue
                ml = event_to_line(e)
                if not ml:
                    continue
                print(format_line(ml), flush=True)
        time.sleep(0.2)


def get_system_state(root: Path, bus_dir: Path) -> dict:
    """Gather system state for JSON output."""
    import socket

    events_path = bus_dir / "events.ndjson"

    # Count bus events
    event_count = 0
    last_event = ""
    insights: list[str] = []

    if events_path.exists():
        try:
            event_count = _estimate_event_count(events_path=events_path)
            # Get recent events for insights (tail only).
            recent = list(iter_tail_events(events_path, max_bytes=2_000_000))
            for e in recent:
                iso = str(e.get("iso") or "")
                if iso:
                    last_event = iso
                ml = event_to_line(e)
                if ml and ml.severity in ("info", "warn") and len(insights) < 10:
                    insights.append(ml.message)
            insights = insights[-5:]
        except Exception:
            pass

    if not insights:
        insights = ["System operational", "Awaiting activity..."]

    # Check vps_session.json for provider status (OAuth providers)
    vps_session_path = root / ".pluribus" / "vps_session.json"
    providers_available: list[str] = []
    providers_total = 0
    try:
        if vps_session_path.exists():
            import json as jm
            vps = jm.loads(vps_session_path.read_text())
            providers = vps.get("providers", {})
            providers_total = len(providers)
            for name, status in providers.items():
                if status.get("available", False):
                    providers_available.append(name)
    except Exception:
        pass

    # Check infercells
    infercells_active = 0
    cells_dir = root / ".pluribus" / "cells"
    if cells_dir.exists():
        try:
            for cell_dir in cells_dir.iterdir():
                state_file = cell_dir / "state.json"
                if state_file.exists():
                    import json as jm
                    state = jm.loads(state_file.read_text())
                    if state.get("state") == "active":
                        infercells_active += 1
        except Exception:
            pass

    # Uptime from bus file
    uptime_s = 0.0
    try:
        if events_path.exists():
            uptime_s = time.time() - events_path.stat().st_mtime
    except Exception:
        pass

    # Check omega/lineage
    lineage_path = root / ".pluribus" / "lineage.json"
    lineage_id = "genesis"
    generation = 0
    try:
        if lineage_path.exists():
            import json as jm
            lin = jm.loads(lineage_path.read_text())
            lineage_id = lin.get("id", "genesis")[:12]
            generation = lin.get("generation", 0)
    except Exception:
        pass

    return {
        "hostname": socket.gethostname(),
        "uptime_s": uptime_s,
        "rings": {
            "ring0": {
                "status": "sealed",
                "pqc_algorithm": "ML-DSA-65",
                "constitution_hash": "auom-v1"
            },
            "ring1": {
                "lineage_id": lineage_id,
                "generation": generation,
                "transfer_type": "VGT",
                "rhizome_objects": event_count
            },
            "ring2": {
                "infercells_active": infercells_active
            },
            "ring3": {
                "omega_healthy": True,
                "omega_cycle": generation,
                "providers_available": providers_available[:5],
                "providers_total": providers_total
            }
        },
        "bus": {
            "events": event_count,
            "last_event": last_event
        },
        "insights": insights
    }


def cmd_json(root: Path, bus_dir: Path) -> int:
    """Output system state as JSON."""
    import json as jm
    data = get_system_state(root, bus_dir)
    print(jm.dumps(data, indent=2))
    return 0


def cmd_compact(root: Path, bus_dir: Path) -> int:
    """Output compact one-line status."""
    data = get_system_state(root, bus_dir)
    rings = data["rings"]
    r0 = "●" if rings["ring0"]["status"] == "sealed" else "○"
    r1 = "●" if rings["ring1"]["lineage_id"] else "○"
    r2 = "●"
    r3 = "●" if rings["ring3"]["omega_healthy"] else "○"

    gen = rings["ring1"]["generation"]
    bus = data["bus"]["events"]
    omega = rings["ring3"]["omega_cycle"]
    providers = len(rings["ring3"]["providers_available"])
    total = rings["ring3"]["providers_total"]

    print(f"PLURIBUS [{r0}{r1}{r2}{r3}] gen:{gen} bus:{bus} omega:{omega} providers:{providers}/{total}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="supermotd.py", description="Curated Pluribus system stream (boot-style).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (default: $PLURIBUS_BUS_DIR or /pluribus/.pluribus/bus)")
    ap.add_argument("--root", default="/pluribus", help="Pluribus root directory")
    ap.add_argument("--limit", default="40")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--json", action="store_true", help="Output system state as JSON")
    ap.add_argument("--compact", action="store_true", help="Output compact one-line status")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()

    if args.json:
        return cmd_json(root, bus_dir)

    if args.compact:
        return cmd_compact(root, bus_dir)

    events_path = bus_dir / "events.ndjson"
    try:
        limit = int(str(args.limit))
    except Exception:
        limit = 40
    limit = max(1, min(200, limit))
    return cmd_tail(events_path=events_path, limit=limit, follow=bool(args.follow))


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
