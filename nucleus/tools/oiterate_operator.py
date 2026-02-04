#!/usr/bin/env python3
"""
OITERATE — Omega Iterate Autonomous Loop Operator (Protocol v8)

Headless omega-loop that:
- observes progress (BEAM + GOLDEN),
- monitors staleness,
- observes coordination/backlog (infer_sync + hexis),
- emits `oiterate.*` metrics/artifacts,
- triggers inter-agent coordination via `infer_sync.request` + `operator.iterate.request`
  when stale (non-blocking, cooldown-limited).

Usage:
  # Run indefinitely
  python3 oiterate_operator.py --agent <name> --goals 10x10 --tick-interval 30

  # One tick (testing)
  python3 oiterate_operator.py --agent <name> --goals 10x10 --single-tick
"""

from __future__ import annotations

import argparse
import calendar
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

TICK_INTERVAL_DEFAULT = 60
STALENESS_THRESHOLD_DEFAULT = 300
MIN_ITERATE_INTERVAL_DEFAULT = 300
BUS_WINDOW_DEFAULT = 900

BEAM_PATH = Path("/pluribus/agent_reports/2025-12-15_beam_10x_discourse.md")
GOLDEN_PATH = Path("/pluribus/nucleus/docs/GOLDEN_SYNTHESIS_DISCOURSE.md")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_ts() -> float:
    return time.time()


def iso_to_ts(iso: str) -> float | None:
    try:
        t = time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
        return float(calendar.timegm(t))
    except Exception:
        return None


def last_beam_append_iso(beam_text: str) -> str | None:
    for line in reversed(beam_text.splitlines()):
        if not line.startswith("## Entry "):
            continue
        if "—" not in line:
            continue
        parts = [p.strip() for p in line.split("—") if p.strip()]
        if not parts:
            continue
        cand = parts[-1]
        if cand.endswith("Z") and "T" in cand:
            return cand
    return None


def count_verified_entries(beam_text: str) -> int:
    count = 0
    for line in beam_text.splitlines():
        s = line.strip()
        if not s.lower().startswith("tags:"):
            continue
        raw = s.split(":", 1)[1].strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        parts = [p.strip().upper() for p in raw.replace("|", ",").split(",") if p.strip()]
        if "V" in parts:
            count += 1
    return count


def count_generation_headers(golden_text: str) -> int:
    gens: set[int] = set()
    for line in golden_text.splitlines():
        if not line.startswith("## Generation "):
            continue
        rest = line[len("## Generation ") :]
        num = ""
        for ch in rest:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            gens.add(int(num))
    return len(gens)


def emit_bus(bus_dir: Path, topic: str, data: dict[str, Any], actor: str, kind: str = "metric") -> None:
    event = {
        "id": str(uuid.uuid4()),
        "ts": now_ts(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": "info",
        "actor": actor,
        "data": data,
    }
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


@dataclass
class Goal:
    id: str
    name: str
    current: float
    target: float
    weight: float
    mode: str = "atleast"  # atleast|atmost
    achieved: bool = False


@dataclass
class OiterateState:
    agent: str
    session_id: str
    started_iso: str
    tick_count: int = 0
    state: str = "IDLE"  # IDLE|RUNNING|STALE|ACHIEVED
    goals: list[Goal] = field(default_factory=list)
    last_tick_iso: str = ""
    staleness_threshold_s: float = float(STALENESS_THRESHOLD_DEFAULT)
    current_staleness_s: float = 0.0
    staleness: dict[str, float] = field(default_factory=dict)  # progress/liveness/coordination
    agenda: list[dict[str, Any]] = field(default_factory=list)  # pushdown frames
    graph: dict[str, Any] = field(default_factory=dict)  # minimal tree view
    last_iterate_trigger_ts: float = 0.0

    def aggregate_progress(self) -> float:
        total_weight = sum(g.weight for g in self.goals)
        if total_weight <= 0:
            return 0.0
        weighted_sum = 0.0
        for g in self.goals:
            if g.mode == "atmost":
                if g.target < 0:
                    continue
                if g.current <= g.target:
                    score = 1.0
                else:
                    denom = float(max(1.0, g.target + 1.0))
                    score = max(0.0, 1.0 - ((g.current - g.target) / denom))
                weighted_sum += score * g.weight
                continue
            if g.target <= 0:
                weighted_sum += (1.0 if g.current >= g.target else 0.0) * g.weight
            else:
                weighted_sum += (min(g.current, g.target) / g.target) * g.weight
        return weighted_sum / total_weight


class OiterateLoop:
    def __init__(self, *, agent: str, bus_dir: Path) -> None:
        self.agent = agent
        self.bus_dir = bus_dir
        self.state = OiterateState(agent=agent, session_id=f"oit-{uuid.uuid4().hex[:8]}", started_iso=now_iso())
        self.iterate_when_achieved: bool = False
        self.min_iterate_interval_s: float = float(MIN_ITERATE_INTERVAL_DEFAULT)
        self.bus_window_s: int = BUS_WINDOW_DEFAULT
        self.hexis_dir: Path = Path(os.environ.get("HEXIS_BUFFER_DIR", "/tmp"))
        self.subproject: str = "beam_10x"
        self.ui_mode: str = "line"  # line|summary|ckin|both|none

    def _read_ndjson_tail(self, path: Path, max_bytes: int = 512_000) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            size = path.stat().st_size
            start = max(0, size - max_bytes)
            with path.open("rb") as f:
                f.seek(start)
                data = f.read().decode("utf-8", errors="replace")
            lines = [ln for ln in data.splitlines() if ln.strip()]
            events: list[dict[str, Any]] = []
            for ln in lines:
                try:
                    events.append(json.loads(ln))
                except Exception:
                    continue
            return events
        except Exception:
            return []

    def _bus_stats(self) -> dict[str, Any]:
        events_path = self.bus_dir / "events.ndjson"
        now = now_ts()
        events = self._read_ndjson_tail(events_path)

        window: list[dict[str, Any]] = []
        for e in events:
            ts = e.get("ts")
            if isinstance(ts, (int, float)) and (now - float(ts)) <= float(self.bus_window_s):
                window.append(e)

        kind_counts: dict[str, int] = {}
        prefix_counts: dict[str, int] = {}
        last_ts: float | None = None
        last_iso_s: str = ""
        for e in window:
            kind = str(e.get("kind") or "")
            if kind:
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            topic = str(e.get("topic") or "")
            if topic:
                prefix = topic.split(".", 1)[0]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            ts = e.get("ts")
            if isinstance(ts, (int, float)) and (last_ts is None or float(ts) > last_ts):
                last_ts = float(ts)
                last_iso_s = str(e.get("iso") or "")

        # infer_sync pending (best-effort: within tail)
        req_ts_by_id: dict[str, float] = {}
        resp_seen: set[str] = set()
        last_resp_ts: float | None = None
        last_art_ts: float | None = None
        for e in events:
            topic = str(e.get("topic") or "")
            ts = e.get("ts")
            if not isinstance(ts, (int, float)):
                continue
            tsf = float(ts)
            if topic == "infer_sync.request":
                req_id = str((e.get("data") or {}).get("req_id") or "")
                if req_id:
                    req_ts_by_id[req_id] = tsf
            elif topic == "infer_sync.response":
                req_id = str((e.get("data") or {}).get("req_id") or "")
                if req_id:
                    resp_seen.add(req_id)
                    last_resp_ts = tsf if last_resp_ts is None else max(last_resp_ts, tsf)
            elif topic.startswith("art_dept.") or topic.startswith("art."):
                last_art_ts = tsf if last_art_ts is None else max(last_art_ts, tsf)

        pending = sorted(
            [{"req_id": rid, "age_s": max(0.0, now - ts)} for rid, ts in req_ts_by_id.items() if rid not in resp_seen],
            key=lambda x: x["age_s"],
            reverse=True,
        )
        oldest_pending_age = float(pending[0]["age_s"]) if pending else 0.0

        return {
            "events": len(window),
            "kinds": kind_counts,
            "prefixes": prefix_counts,
            "last_event_iso": last_iso_s,
            "liveness_age_s": float(max(0.0, now - last_ts)) if last_ts is not None else 0.0,
            "infer_sync": {
                "pending": len(pending),
                "oldest_pending_age_s": oldest_pending_age,
                "last_response_age_s": float(max(0.0, now - last_resp_ts)) if last_resp_ts is not None else None,
            },
            "art": {"last_age_s": float(max(0.0, now - last_art_ts)) if last_art_ts is not None else None},
        }

    def _hexis_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {}
        try:
            for p in sorted(self.hexis_dir.glob("*.buffer")):
                agent = p.name.replace(".buffer", "")
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                pending = len([ln for ln in lines if ln.strip()])
                oldest_iso = None
                oldest_topic = None
                if pending:
                    try:
                        first = json.loads(lines[0])
                        oldest_iso = first.get("iso")
                        oldest_topic = first.get("topic")
                    except Exception:
                        oldest_iso = None
                status[agent] = {"pending": pending, "path": str(p), "oldest_iso": oldest_iso, "oldest_topic": oldest_topic}
        except Exception:
            pass
        return status

    def load_10x10_goals(self) -> None:
        self.state.goals = [
            Goal("beam", "BEAM Entries", 0, 100, 1.0),
            Goal("golden", "GOLDEN Lines", 0, 500, 1.0),
            Goal("verified", "Verified [V]", 0, 30, 1.0),
            Goal("sections", "Sections", 0, 10, 1.0),
            Goal("infer_sync_pending", "infer_sync pending", 0, 0, 0.5, mode="atmost"),
            Goal("hexis_pending", "Hexis pending", 0, 0, 0.5, mode="atmost"),
        ]

    def update_progress(self) -> None:
        try:
            if BEAM_PATH.exists():
                content = BEAM_PATH.read_text(encoding="utf-8", errors="replace")
                entries = content.count("## Entry")
                verified = count_verified_entries(content)
                for g in self.state.goals:
                    if g.id == "beam":
                        g.current = float(entries)
                    elif g.id == "verified":
                        g.current = float(verified)

            if GOLDEN_PATH.exists():
                golden_text = GOLDEN_PATH.read_text(encoding="utf-8", errors="replace")
                lines = len(golden_text.splitlines())
                sections = count_generation_headers(golden_text)
                for g in self.state.goals:
                    if g.id == "golden":
                        g.current = float(lines)
                    elif g.id == "sections":
                        g.current = float(sections)
        except Exception:
            pass

        # Dynamic coordination/backlog goals (bus + hexis).
        try:
            bus = self._bus_stats()
            infer_pending = int((bus.get("infer_sync") or {}).get("pending") or 0)
            hexis = self._hexis_status()
            hexis_pending_total = sum(int(v.get("pending") or 0) for v in hexis.values())
            for g in self.state.goals:
                if g.id == "infer_sync_pending":
                    g.current = float(infer_pending)
                elif g.id == "hexis_pending":
                    g.current = float(hexis_pending_total)
        except Exception:
            pass

        for g in self.state.goals:
            if g.mode == "atmost":
                g.achieved = g.current <= g.target
            else:
                g.achieved = g.current >= g.target

    def update_staleness(self) -> None:
        now = now_ts()
        beam_age: float | None = None
        golden_age: float | None = None

        if BEAM_PATH.exists():
            try:
                beam_text = BEAM_PATH.read_text(encoding="utf-8", errors="replace")
                iso = last_beam_append_iso(beam_text)
                ts = iso_to_ts(iso) if iso else None
                if ts is not None:
                    beam_age = max(0.0, now - ts)
                else:
                    beam_age = max(0.0, now - BEAM_PATH.stat().st_mtime)
            except Exception:
                beam_age = None

        if GOLDEN_PATH.exists():
            try:
                golden_age = max(0.0, now - GOLDEN_PATH.stat().st_mtime)
            except Exception:
                golden_age = None

        # Staleness is "time since meaningful progress" (not just liveness).
        bus = self._bus_stats()
        liveness_age = float(bus.get("liveness_age_s") or 0.0)
        coordination_age = float((bus.get("infer_sync") or {}).get("oldest_pending_age_s") or 0.0)

        base_ages = [a for a in (beam_age, golden_age) if a is not None]
        extra_ages: list[float] = []
        last_resp_age = (bus.get("infer_sync") or {}).get("last_response_age_s")
        if isinstance(last_resp_age, (int, float)):
            extra_ages.append(float(last_resp_age))
        last_art_age = (bus.get("art") or {}).get("last_age_s")
        if isinstance(last_art_age, (int, float)):
            extra_ages.append(float(last_art_age))
        progress_age = float(min(base_ages + extra_ages)) if (base_ages or extra_ages) else 0.0

        self.state.staleness = {"progress_s": progress_age, "liveness_s": liveness_age, "coordination_s": coordination_age}
        self.state.current_staleness_s = progress_age

        # PDA frames derived from live signals
        frames: list[dict[str, Any]] = []
        hexis = self._hexis_status()
        hexis_pending_total = sum(int(v.get("pending") or 0) for v in hexis.values())
        infer_pending = int((bus.get("infer_sync") or {}).get("pending") or 0)
        if infer_pending > 0:
            frames.append(
                {
                    "frame": "SYNC_ACK",
                    "severity": "high",
                    "signal": {"infer_sync_pending": infer_pending, "oldest_age_s": coordination_age},
                    "action": {"topic": "infer_sync.request", "intent": "acknowledge", "subproject": self.subproject},
                }
            )
        if hexis_pending_total > 0:
            frames.append(
                {
                    "frame": "HEXIS_DRAIN",
                    "severity": "med",
                    "signal": {"hexis_pending_total": hexis_pending_total},
                    "action": {"topic": "hexis_buffer", "intent": "drain"},
                }
            )

        # License uncertainty signal (reference-only artifact note)
        try:
            genomes = Path("/pluribus/nucleus/art_dept/artifacts/genomes.ndjson")
            if genomes.exists() and "reference-only" in genomes.read_text(encoding="utf-8", errors="replace").lower():
                frames.append(
                    {
                        "frame": "LICENSE_AUDIT",
                        "severity": "med",
                        "signal": {"reason": "reference-only artifacts present"},
                        "action": {"topic": "art_dept.license.audit", "intent": "audit"},
                    }
                )
        except Exception:
            pass

        self.state.agenda = frames
        self.state.graph = {
            "nodes": [{"id": "root", "label": "OITERATE"}]
            + [{"id": f"f{i}", "label": fr["frame"], "severity": fr["severity"]} for i, fr in enumerate(frames)],
            "edges": [{"from": "root", "to": f"f{i}", "label": "push"} for i in range(len(frames))],
        }

    def maybe_trigger_iterate(self) -> None:
        now = now_ts()
        achieved = all(g.achieved for g in self.state.goals)
        if achieved and not self.iterate_when_achieved and not self.state.agenda:
            return
        if self.state.current_staleness_s <= self.state.staleness_threshold_s and not self.state.agenda:
            return
        if (now - self.state.last_iterate_trigger_ts) < self.min_iterate_interval_s:
            return

        self.state.last_iterate_trigger_ts = now
        self.state.state = "STALE"
        req_id = str(uuid.uuid4())

        emit_bus(
            self.bus_dir,
            "oiterate.action_triggered",
            {
                "session_id": self.state.session_id,
                "tick": self.state.tick_count,
                "action": "iterate",
                "reason": f"stale>{int(self.state.staleness_threshold_s)}s cooldown={int(self.min_iterate_interval_s)}s",
                "req_id": req_id,
            },
            self.agent,
            kind="request",
        )
        emit_bus(
            self.bus_dir,
            "infer_sync.request",
            {
                "req_id": req_id,
                "subproject": self.subproject,
                "intent": "iterate" if not self.state.agenda else "resolve_agenda",
                "source": "oiterate",
                "agenda": self.state.agenda[:5],
            },
            self.agent,
            kind="request",
        )
        emit_bus(
            self.bus_dir,
            "operator.iterate.request",
            {
                "req_id": req_id,
                "subproject": self.subproject,
                "intent": "iterate" if not self.state.agenda else "resolve_agenda",
                "source": "oiterate",
            },
            self.agent,
            kind="request",
        )

    def tick(self) -> None:
        self.state.tick_count += 1
        self.state.last_tick_iso = now_iso()
        self.state.state = "RUNNING"
        
        self.update_progress()
        self.update_staleness() # Ensure staleness is calculated
        self.maybe_trigger_iterate()
        
        agg = self.state.aggregate_progress()
        # Preserve "STALE" if we just triggered an iterate request; otherwise the UI
        # reads as "ACHIEVED" even while we are actively dispatching work.
        if self.state.state != "STALE" and all(g.achieved for g in self.state.goals) and not self.state.agenda:
            self.state.state = "ACHIEVED"
        
        emit_bus(
            self.bus_dir,
            "oiterate.tick",
            {
                "session_id": self.state.session_id,
                "tick": self.state.tick_count,
                "state": self.state.state,
                "progress": agg,
                "staleness_s": self.state.current_staleness_s,
                "staleness": self.state.staleness,
                "agenda": self.state.agenda,
                "graph": self.state.graph,
                "bus": self._bus_stats(),
                "hexis": self._hexis_status(),
                "goals": [asdict(g) for g in self.state.goals],
            },
            self.agent,
            kind="metric",
        )

        if self.ui_mode == "none":
            return

        if self.ui_mode in ("line", "summary", "both", "ckin"):
            print(
                f"[{self.state.last_tick_iso}] Tick {self.state.tick_count}: "
                f"state={self.state.state} progress={agg*100:.1f}% "
                f"stale(progress)={int(self.state.staleness.get('progress_s', 0))}s "
                f"agenda={len(self.state.agenda)}"
            )

        if self.ui_mode in ("summary", "both"):
            bus = self._bus_stats()
            hexis = self._hexis_status()
            print(
                f"  staleness: progress={int(self.state.staleness.get('progress_s', 0))}s "
                f"liveness={int(self.state.staleness.get('liveness_s', 0))}s "
                f"coord={int(self.state.staleness.get('coordination_s', 0))}s"
            )
            print(
                f"  bus: window={self.bus_window_s}s events={bus.get('events')} "
                f"infer_sync.pending={((bus.get('infer_sync') or {}).get('pending') or 0)} "
                f"last={bus.get('last_event_iso')}"
            )
            hexis_pending_total = sum(int(v.get('pending') or 0) for v in hexis.values())
            print(f"  hexis.pending_total={hexis_pending_total}")
            if self.state.agenda:
                print("  PDA stack (top→bottom):")
                for fr in self.state.agenda:
                    print(f"    - {fr['frame']} ({fr['severity']})")

        if self.ui_mode in ("ckin", "both"):
            here = Path(__file__).resolve().parent
            ckin_tool = here / "ckin_report.py"
            if ckin_tool.exists():
                sys.stdout.flush()
                subprocess.run(
                    [sys.executable, str(ckin_tool), "--agent", self.agent],
                    env={**os.environ, "PLURIBUS_BUS_DIR": str(self.bus_dir), "PYTHONDONTWRITEBYTECODE": "1"},
                    check=False,
                )
            else:
                print(f"(ckin_report.py not found at {ckin_tool})")

    def run(self, *, tick_interval_s: int, stop_on_achieved: bool, max_ticks: int) -> None:
        emit_bus(self.bus_dir, "oiterate.started", {"session_id": self.state.session_id}, self.agent, kind="request")
        achieved_emitted = False
        try:
            while True:
                self.tick()

                if all(g.achieved for g in self.state.goals) and not achieved_emitted:
                    achieved_emitted = True
                    emit_bus(
                        self.bus_dir,
                        "oiterate.goal_achieved",
                        {"session_id": self.state.session_id, "tick": self.state.tick_count},
                        self.agent,
                        kind="artifact",
                    )
                    if stop_on_achieved:
                        break

                if max_ticks and self.state.tick_count >= max_ticks:
                    break

                time.sleep(float(tick_interval_s))
        except KeyboardInterrupt:
            self.state.state = "IDLE"
        finally:
            emit_bus(
                self.bus_dir,
                "oiterate.stopped",
                {"session_id": self.state.session_id, "tick": self.state.tick_count, "state": self.state.state},
                self.agent,
                kind="artifact",
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="OITERATE Operator (Protocol v8)")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--goals", choices=["10x10"], required=True)
    ap.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    ap.add_argument("--subproject", default="beam_10x", help="Subproject for triggered infer_sync requests")
    ap.add_argument("--bus-window", type=int, default=BUS_WINDOW_DEFAULT, help="Bus window seconds for stats")
    ap.add_argument("--hexis-dir", default=os.environ.get("HEXIS_BUFFER_DIR", "/tmp"), help="Hexis buffer dir")
    ap.add_argument("--ui", choices=["none", "line", "summary", "ckin", "both"], default="line", help="Console UI mode")
    ap.add_argument("--interval", type=int, default=TICK_INTERVAL_DEFAULT, help="Tick interval seconds (deprecated; use --tick-interval)")
    ap.add_argument("--tick-interval", type=int, default=None, help="Tick interval seconds")
    ap.add_argument("--staleness-threshold", type=int, default=STALENESS_THRESHOLD_DEFAULT, help="Seconds before STALE triggers ITERATE")
    ap.add_argument("--iterate-when-achieved", action="store_true", help="Allow ITERATE triggers even when goals are achieved")
    ap.add_argument("--min-iterate-interval", type=int, default=MIN_ITERATE_INTERVAL_DEFAULT, help="Minimum seconds between ITERATE triggers")
    ap.add_argument("--stop-on-achieved", action="store_true", help="Stop loop after goals achieved (default: keep running)")
    ap.add_argument("--max-ticks", type=int, default=0, help="Stop after N ticks (0=infinite)")
    ap.add_argument("--single-tick", action="store_true", help="Run one tick and exit")
    args = ap.parse_args()

    bus_dir = Path(args.bus_dir)
    tick_interval_s = int(args.tick_interval) if args.tick_interval is not None else int(args.interval)
    max_ticks = 1 if args.single_tick else int(args.max_ticks)

    loop = OiterateLoop(agent=args.agent, bus_dir=bus_dir)
    loop.bus_window_s = int(args.bus_window)
    loop.hexis_dir = Path(args.hexis_dir)
    loop.subproject = str(args.subproject)
    loop.ui_mode = str(args.ui)
    loop.state.staleness_threshold_s = float(args.staleness_threshold)
    loop.iterate_when_achieved = bool(args.iterate_when_achieved)
    loop.min_iterate_interval_s = float(args.min_iterate_interval)
    loop.load_10x10_goals()

    loop.run(
        tick_interval_s=tick_interval_s,
        stop_on_achieved=bool(args.stop_on_achieved),
        max_ticks=max_ticks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
