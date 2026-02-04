#!/usr/bin/env python3
import time
import os
import sys
import json
import collections
import argparse
import shutil
import subprocess
import shlex
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None

# Configuration
BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")
BUS_FILE = os.path.join(BUS_DIR, "events.ndjson")
MAX_LOG_LINES = 1000
TAIL_BOOTSTRAP_BYTES = 2_000_000  #  2MB to capture recent metrics
STATUS_REFRESH_S = 5.0
DEFAULT_REFRESH_S = 5.0
BUS_SIZE_WARN_MB_DEFAULT = 100.0
WAVE_WIDTH = 48
HEARTBEAT_PATTERN = "__/\\__"
LEDGER_TAIL_BYTES = 2_000_000
LEDGER_TAIL_LINES = 5000
TASK_ACTIVE_STATUSES = {"in_progress", "blocked", "run", "running"}
TASK_DONE_STATUSES = {"completed", "abandoned", "failed", "done", "fail"}
TASK_COMPLETE_KEEP_S = 60
TASK_STALE_DEFAULT_S = 6 * 60 * 60
TASK_PROGRESS_HISTORY_MAX = 20
TASK_PROGRESS_MIN_DELTA = 0.01
TASK_PROGRESS_BAR_WIDTH = 18
TMUX_RIGHT_PCT_DEFAULT = 35
TMUX_LEFT_MIN_COLS_DEFAULT = 110
TMUX_LEFT_MIN_ROWS_DEFAULT = 40
AGENT_ACTIVE_DEFAULT_S = 5 * 60
OHM_EMIT_INTERVAL_DEFAULT_S = 30.0
OHM_STATUS_TOPIC_DEFAULT = "ohm.status"
OHM_ALERT_TOPIC_DEFAULT = "ohm.alert"
ART_TICK_INTERVAL_S = 8640.0  # 144 minutes (10/day)

OMEGA_TOPICS = {
    "omega.heartbeat",
    "omega.queue.depth",
    "omega.pending.pairs",
    "omega.metrics.velocity",
    "omega.metrics.latency",
    "omega.metrics.entropy",
    "omega.providers.scan",
    "omega.providers.health",
    "omega.health",
    "omega.health.composite",
    "omega.guardian.cycle",
    "omega.guardian.warn",
    "omega.dispatcher.ready",
    "omega.dispatch.tick",
    "omega.dispatch.sent",
    "omega.dispatch.access_denied",
    "omega.dispatch.ring_check_error",
}

SERVICE_UNIT_PREFIX = "pluribus-"


class OmegaHeartMonitor:
    def __init__(self, filter_name=None, bootstrap_tasks=True):
        self.running = True
        self.log_buffer = collections.deque(maxlen=MAX_LOG_LINES)
        self.bus_offset = 0
        self.bus_inode = None
        self.bus_partial = b""
        self.bootstrapped = False
        self.last_service_check = 0.0
        self.service_status = {}
        self.service_units = []
        self.wave = collections.deque(maxlen=WAVE_WIDTH)
        self.pulse_pattern = collections.deque()
        self.pending_pulses = 0
        self.heartbeat_intervals = collections.deque(maxlen=12)
        self.heartbeat_last_ts = 0.0
        self.filters = self._build_filters()
        self.filter_index = 0
        self.task_state = {}
        self.task_entry_ids = set()
        self.task_ledger_path = None
        self.task_ledger_mtime = 0.0
        self.task_ledger_offset = 0
        self.task_ledger_inode = None
        self.task_ledger_partial = b""

        refresh_env = os.environ.get("OHM_REFRESH_S")
        try:
            self.refresh_s = float(refresh_env) if refresh_env else DEFAULT_REFRESH_S
        except (TypeError, ValueError):
            self.refresh_s = DEFAULT_REFRESH_S
        if self.refresh_s <= 0:
            self.refresh_s = DEFAULT_REFRESH_S
        self.refresh_s = max(0.5, self.refresh_s)
        warn_env = os.environ.get("OHM_BUS_WARN_MB")
        try:
            self.bus_warn_mb = float(warn_env) if warn_env else BUS_SIZE_WARN_MB_DEFAULT
        except (TypeError, ValueError):
            self.bus_warn_mb = BUS_SIZE_WARN_MB_DEFAULT
        if self.bus_warn_mb <= 0:
            self.bus_warn_mb = BUS_SIZE_WARN_MB_DEFAULT
        stale_env = os.environ.get("OHM_TASK_STALE_S")
        try:
            self.task_stale_s = float(stale_env) if stale_env else TASK_STALE_DEFAULT_S
        except (TypeError, ValueError):
            self.task_stale_s = TASK_STALE_DEFAULT_S
        active_env = os.environ.get("OHM_AGENT_ACTIVE_S")
        try:
            self.agent_active_s = float(active_env) if active_env else AGENT_ACTIVE_DEFAULT_S
        except (TypeError, ValueError):
            self.agent_active_s = AGENT_ACTIVE_DEFAULT_S
        self.show_stale_tasks = os.environ.get("OHM_SHOW_STALE_TASKS") == "1"
        self.agent_expanded = os.environ.get("OHM_AGENT_EXPAND") == "1"
        emit_env = os.environ.get("OHM_EMIT_S")
        try:
            self.emit_interval_s = float(emit_env) if emit_env else OHM_EMIT_INTERVAL_DEFAULT_S
        except (TypeError, ValueError):
            self.emit_interval_s = OHM_EMIT_INTERVAL_DEFAULT_S
        if self.emit_interval_s <= 0:
            self.emit_interval_s = OHM_EMIT_INTERVAL_DEFAULT_S
        self.emit_interval_s = max(1.0, self.emit_interval_s)
        self.emit_topic = os.environ.get("OHM_EMIT_TOPIC", OHM_STATUS_TOPIC_DEFAULT)
        self.alert_topic = os.environ.get("OHM_ALERT_TOPIC", OHM_ALERT_TOPIC_DEFAULT)
        self.emit_alerts = os.environ.get("OHM_EMIT_ALERTS") == "1"
        self._last_emit_ts = 0.0
        self._last_alert_ts = 0.0
        self._last_art_tick_ts = 0.0
        self._last_alert_state = None

        self.metrics = {
            "agents": set(),
            "active_tasks": {},  # task_id -> {agent, timestamp}
            "tasks_started": 0,
            "tasks_completed": 0,
            "dispatches": collections.deque(maxlen=50),
            "responses": collections.deque(maxlen=50),
            "a2a_requests": 0,
            "a2a_responses": 0,
            "collab_requests": 0,
            "collab_responses": 0,
            "events_seen": 0,
            "provider_activity": collections.Counter(),  # v1.13: Per-provider event counts
            "agent_last_seen": {},  # v1.14: {actor: last_ts} for agent heartbeat monitoring
        }

        self.event_counters = collections.Counter()

        self.omega = {
            "heartbeat": {"count": 0, "last_ts": 0.0, "cycle": 0, "uptime_s": 0.0},
            "guardian": {"count": 0, "warn": 0, "last_ts": 0.0, "last_warn_ts": 0.0, "cycle": 0},
            "dispatch": {"tick": 0, "sent": 0, "denied": 0, "errors": 0, "last_ts": 0.0, "pending_total": 0},
            "queue": {"count": 0, "last_ts": 0.0, "pending_requests": 0, "total_events": 0},
            "pairs": {"count": 0, "last_ts": 0.0, "total": 0, "oldest_age_s": 0.0},
            "providers": {"count": 0, "last_ts": 0.0, "providers": {}, "details": {}, "health_score": None},
            "health": {"count": 0, "last_ts": 0.0, "status": "unknown", "score": None},
            "metrics": {"velocity": {}, "latency": {}, "entropy": {}},
        }

        self.set_filter(filter_name or os.environ.get("OHM_FILTER"))
        self._prime_wave()
        if bootstrap_tasks:
            self._bootstrap_task_ledger()

    def _prime_wave(self):
        for _ in range(WAVE_WIDTH):
            self.wave.append("_")

    def _resolve_task_ledger_path(self):
        try:
            from nucleus.tools import task_ledger
            path = task_ledger.default_ledger_path(for_write=False)
            return str(path)
        except Exception:
            return None

    def _tail_file_lines(self, path, max_bytes=LEDGER_TAIL_BYTES, max_lines=LEDGER_TAIL_LINES):
        if not path or not os.path.exists(path):
            return []
        try:
            size = os.path.getsize(path)
            offset = max(0, size - max_bytes)
            with open(path, "rb") as handle:
                if offset:
                    handle.seek(offset, os.SEEK_SET)
                data = handle.read()
            lines = data.splitlines()
            if offset and lines:
                lines = lines[1:]
            if max_lines and len(lines) > max_lines:
                lines = lines[-max_lines:]
            return [line.decode("utf-8", errors="replace") for line in lines]
        except Exception:
            return []

    def _read_task_ledger_lines(self):
        path = self.task_ledger_path or self._resolve_task_ledger_path()
        if not path or not os.path.exists(path):
            return []
        self.task_ledger_path = path
        try:
            st = os.stat(path)
            inode = getattr(st, "st_ino", None)
            if self.task_ledger_inode is None:
                self.task_ledger_inode = inode
            if inode is not None and self.task_ledger_inode is not None and inode != self.task_ledger_inode:
                self.task_ledger_offset = 0
                self.task_ledger_partial = b""
                self.task_ledger_inode = inode
            if st.st_size < self.task_ledger_offset:
                self.task_ledger_offset = 0
                self.task_ledger_partial = b""
            with open(path, "rb") as handle:
                if self.task_ledger_offset == 0 and st.st_size > LEDGER_TAIL_BYTES:
                    self.task_ledger_offset = st.st_size - LEDGER_TAIL_BYTES
                handle.seek(self.task_ledger_offset, os.SEEK_SET)
                chunk = handle.read()
                self.task_ledger_offset = handle.tell()
            if not chunk:
                return []
            data = self.task_ledger_partial + chunk
            lines = data.split(b"\n")
            if data and not data.endswith(b"\n"):
                self.task_ledger_partial = lines.pop()
            else:
                self.task_ledger_partial = b""
            return lines
        except Exception:
            return []

    def _bootstrap_task_ledger(self):
        self.task_ledger_path = self._resolve_task_ledger_path()
        if not self.task_ledger_path:
            return
        lines = self._tail_file_lines(self.task_ledger_path)
        for line in lines:
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            self._ingest_task_entry(entry)
        try:
            st = os.stat(self.task_ledger_path)
            self.task_ledger_inode = getattr(st, "st_ino", None)
            self.task_ledger_offset = st.st_size
            self.task_ledger_partial = b""
        except Exception:
            self.task_ledger_offset = 0
        try:
            self.task_ledger_mtime = os.path.getmtime(self.task_ledger_path)
        except Exception:
            self.task_ledger_mtime = 0.0

    def tail_task_ledger(self):
        lines = self._read_task_ledger_lines()
        if not lines:
            return
        for raw in lines:
            if not raw:
                continue
            try:
                entry = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            self._ingest_task_entry(entry)

    def _ingest_task_entry(self, entry):
        if not isinstance(entry, dict):
            return
        status = entry.get("status")
        if not status:
            return
        status = str(status).lower()
        entry_id = entry.get("id")
        if entry_id:
            if entry_id in self.task_entry_ids:
                return
            self.task_entry_ids.add(entry_id)
        run_id = entry.get("run_id") or entry.get("id") or entry.get("req_id")
        if not run_id:
            return
        actor = entry.get("actor") or "unknown"
        self.metrics["agents"].add(actor)
        ts = self._coerce_ts(entry.get("ts") or entry.get("timestamp"))
        if not ts:
            ts = time.time()
        self.metrics["agent_last_seen"][actor] = ts  # v1.14: Track last activity
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        desc = meta.get("desc") or meta.get("step") or meta.get("intent") or entry.get("desc")
        progress = self._coerce_progress(
            meta.get("progress") or entry.get("progress") or entry.get("pct")
        )

        state = self.task_state.get(run_id)
        is_new_task = state is None
        if not state:
            state = {
                "status": status,
                "actor": actor,
                "ts": ts,
                "start_ts": ts,
                "last_ts": ts,
                "desc": desc,
                "progress": None,
                "history": collections.deque(maxlen=TASK_PROGRESS_HISTORY_MAX),
                "completed_ts": None,
                "_counted_started": False,
                "_counted_completed": False,
            }
            self.task_state[run_id] = state

        # Track dispatches: count when task first seen or first reaches active
        if is_new_task or (status in TASK_ACTIVE_STATUSES and not state.get("_counted_started")):
            self.metrics["tasks_started"] += 1
            state["_counted_started"] = True
            self.metrics["dispatches"].append({"id": run_id, "target": actor})

        state["status"] = status
        if actor:
            state["actor"] = actor
        state["ts"] = ts
        state["last_ts"] = ts
        state.setdefault("start_ts", ts)
        if desc:
            state["desc"] = desc
        if "history" not in state or not isinstance(state["history"], collections.deque):
            state["history"] = collections.deque(state.get("history", []), maxlen=TASK_PROGRESS_HISTORY_MAX)

        if progress is not None:
            last_progress = state.get("progress")
            if last_progress is None or abs(progress - last_progress) >= TASK_PROGRESS_MIN_DELTA:
                state["history"].append((ts, progress))
            state["progress"] = progress

        if status in TASK_DONE_STATUSES:
            if state.get("completed_ts") is None:
                state["completed_ts"] = ts
            # Track completions: count when task first reaches done status
            if not state.get("_counted_completed"):
                self.metrics["tasks_completed"] += 1
                state["_counted_completed"] = True
            if state.get("progress") is None or state.get("progress", 0.0) < 1.0:
                if not state["history"] or state["history"][-1][1] < 1.0:
                    state["history"].append((ts, 1.0))
                state["progress"] = 1.0
        else:
            state["completed_ts"] = None

    def _is_task_stale(self, info, now: float) -> bool:
        if not self.task_stale_s or self.task_stale_s <= 0:
            return False
        status = str(info.get("status", "unknown")).lower()
        if status not in TASK_ACTIVE_STATUSES:
            return False
        last_ts = info.get("last_ts") or info.get("ts") or 0.0
        if not last_ts:
            return False
        return (now - last_ts) > self.task_stale_s

    def _agent_task_counts(self, now: float) -> collections.Counter:
        counts = collections.Counter()
        for info in self.task_state.values():
            status = str(info.get("status", "unknown")).lower()
            if status not in TASK_ACTIVE_STATUSES:
                continue
            if self._is_task_stale(info, now):
                continue
            actor = info.get("actor") or "unknown"
            counts[actor] += 1
        return counts

    def _task_counts(self):
        counts = collections.Counter()
        active = {}
        stale = 0
        now = time.time()
        for task_id, info in self.task_state.items():
            status = str(info.get("status", "unknown")).lower()
            if self._is_task_stale(info, now):
                stale += 1
                if self.show_stale_tasks:
                    active[task_id] = dict(info, status="stale")
                continue
            counts[status] += 1
            if status in TASK_ACTIVE_STATUSES:
                active[task_id] = info
        if stale:
            counts["stale"] = stale
        return counts, active

    def _prune_tasks(self, now=None) -> None:
        if not self.task_state:
            return
        now = now or time.time()
        stale_cutoff = None
        if self.task_stale_s and self.task_stale_s > 0:
            stale_cutoff = self.task_stale_s + TASK_COMPLETE_KEEP_S

        for run_id, info in list(self.task_state.items()):
            status = str(info.get("status", "unknown")).lower()
            last_ts = info.get("last_ts") or info.get("ts") or 0.0
            completed_ts = info.get("completed_ts") or last_ts
            if status in TASK_DONE_STATUSES:
                if completed_ts and (now - completed_ts) > TASK_COMPLETE_KEEP_S:
                    self.task_state.pop(run_id, None)
                continue
            if stale_cutoff is not None and status in TASK_ACTIVE_STATUSES:
                if last_ts and (now - last_ts) > stale_cutoff:
                    self.task_state.pop(run_id, None)
                continue
            if status not in TASK_ACTIVE_STATUSES and status not in TASK_DONE_STATUSES:
                if last_ts and (now - last_ts) > TASK_COMPLETE_KEEP_S:
                    self.task_state.pop(run_id, None)

    def _build_filters(self):
        def _match_topic(prefixes, entry):
            topic = entry.get("topic", "")
            return any(topic.startswith(prefix) for prefix in prefixes)

        def _match_errors(entry):
            level = str(entry.get("level", "")).lower()
            if level in {"error", "warn"}:
                return True
            summary = str(entry.get("summary", "")).lower()
            return "error" in summary or "warn" in summary

        def _match_dispatch(entry):
            topic = entry.get("topic", "")
            return topic.startswith("omega.dispatch") or topic in {"task.dispatch", "rd.tasks.dispatch"}

        def _match_tasks(entry):
            topic = entry.get("topic", "")
            return (
                topic.startswith("task.")
                or topic.startswith("rd.tasks.")
                or topic.startswith("task_ledger.")
                or topic.endswith(".task")
            )

        return [
            ("All", lambda entry: True),
            ("Omega", lambda entry: _match_topic(("omega.",), entry)),
            ("Dispatch", _match_dispatch),
            ("QA", lambda entry: _match_topic(("qa.", "qa_"), entry)),
            ("Tasks", _match_tasks),
            ("Errors", _match_errors),
        ]

    def set_filter(self, name_or_index):
        if name_or_index is None:
            return
        if isinstance(name_or_index, int):
            if 0 <= name_or_index < len(self.filters):
                self.filter_index = name_or_index
            return
        raw = str(name_or_index).strip()
        if not raw:
            return
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(self.filters):
                self.filter_index = idx
            return
        raw = raw.lower()
        for idx, (label, _) in enumerate(self.filters):
            if label.lower() == raw or raw in label.lower():
                self.filter_index = idx
                return

    def _filtered_log_entries(self, limit=None):
        if not self.log_buffer:
            return []
        _, predicate = self.filters[self.filter_index]
        entries = [entry for entry in self.log_buffer if predicate(entry)]
        if limit:
            return entries[-limit:]
        return entries

    def _coerce_ts(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    def _coerce_progress(self, value):
        if value is None:
            return None
        try:
            progress = float(value)
        except (TypeError, ValueError):
            return None
        if progress < 0:
            progress = 0.0
        if progress > 1.0:
            progress = progress / 100.0 if progress <= 100.0 else 1.0
        return max(0.0, min(1.0, progress))

    def _format_age(self, ts: float) -> str:
        if not ts:
            return "n/a"
        age = max(0.0, time.time() - ts)
        if age < 60:
            return f"{age:.0f}s"
        if age < 3600:
            return f"{age/60:.0f}m"
        return f"{age/3600:.1f}h"

    def _format_duration(self, seconds):
        if seconds is None:
            return "n/a"
        seconds = max(0.0, seconds)
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds/60:.0f}m"
        return f"{seconds/3600:.1f}h"

    def _heartbeat_stats(self):
        if not self.heartbeat_intervals:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "jitter": 0.0, "bpm": 0.0}
        intervals = list(self.heartbeat_intervals)
        avg = sum(intervals) / len(intervals)
        min_i = min(intervals)
        max_i = max(intervals)
        jitter = max_i - min_i
        bpm = 60.0 / avg if avg > 0 else 0.0
        return {"avg": avg, "min": min_i, "max": max_i, "jitter": jitter, "bpm": bpm}

    def _update_wave(self):
        if self.pending_pulses > 0 and not self.pulse_pattern:
            self.pulse_pattern.extend(list(HEARTBEAT_PATTERN))
            self.pending_pulses -= 1

        if self.pulse_pattern:
            self.wave.append(self.pulse_pattern.popleft())
        else:
            self.wave.append("_")

    def _progress_bar(self, progress, width=TASK_PROGRESS_BAR_WIDTH):
        progress = max(0.0, min(1.0, progress))
        filled = int(round(progress * width))
        filled = max(0, min(width, filled))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    def _task_eta_seconds(self, task):
        history = task.get("history")
        if not history or len(history) < 2:
            return None
        try:
            t0, p0 = history[0]
            t1, p1 = history[-1]
        except Exception:
            return None
        dt = t1 - t0
        dp = p1 - p0
        if dt <= 0 or dp <= 0:
            return None
        progress = task.get("progress", p1)
        if progress is None:
            progress = p1
        remaining = 1.0 - progress
        if remaining <= 0:
            return 0.0
        speed = dp / dt
        if speed <= 0:
            return None
        return remaining / speed

    def _task_progress_rows(self, inner_width, max_rows):
        now = time.time()
        rows = []
        for task_id, info in self.task_state.items():
            status = str(info.get("status", "unknown")).lower()
            done = status in TASK_DONE_STATUSES
            is_stale = self._is_task_stale(info, now)
            if is_stale and not self.show_stale_tasks:
                continue
            if done:
                completed_ts = info.get("completed_ts") or info.get("last_ts") or info.get("ts") or 0.0
                if not completed_ts or (now - completed_ts) > TASK_COMPLETE_KEEP_S:
                    continue
            elif status not in TASK_ACTIVE_STATUSES and not is_stale:
                continue

            progress = info.get("progress")
            if done:
                progress = 1.0
            progress_value = progress if progress is not None else 0.0
            progress_value = max(0.0, min(1.0, progress_value))
            pct = int(round(progress_value * 100))
            bar = self._progress_bar(progress_value, TASK_PROGRESS_BAR_WIDTH)
            elapsed = self._format_duration(now - (info.get("start_ts") or info.get("ts") or now))
            if done:
                eta_str = "done"
            else:
                eta = self._task_eta_seconds(info)
                eta_str = self._format_duration(eta) if eta is not None else "--"
            if is_stale:
                status_tag = "STL"
                status = "stale"
            else:
                status_tag = "RUN" if status == "in_progress" else ("BLK" if status == "blocked" else "DONE")
            actor = info.get("actor") or "unknown"
            desc = info.get("desc") or ""

            line = f"{status_tag} {bar} {pct:>3}% eta {eta_str:>4} elap {elapsed:>4} {actor}"
            if desc:
                line += f" {desc}"

            max_line = max(10, inner_width - 4)
            if len(line) > max_line:
                line = line[: max_line - 3] + "..."

            order = 0 if status == "in_progress" else (1 if status == "blocked" else (2 if status == "stale" else 3))
            sort_ts = info.get("last_ts") or info.get("ts") or 0
            rows.append(
                {
                    "order": order,
                    "ts": sort_ts,
                    "line": line,
                    "status": status,
                    "task_id": task_id,
                }
            )

        rows.sort(key=lambda item: (item["order"], -item["ts"]))
        if max_rows and len(rows) > max_rows:
            rows = rows[:max_rows]
        return rows

    def _refresh_services(self):
        now = time.time()
        if now - self.last_service_check < STATUS_REFRESH_S:
            return
        self.last_service_check = now
        if not shutil.which("systemctl"):
            return
        try:
            output = subprocess.check_output(
                [
                    "systemctl",
                    "list-units",
                    "--type=service",
                    "--all",
                    f"{SERVICE_UNIT_PREFIX}*",
                    "--no-legend",
                    "--no-pager",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).splitlines()
            services = []
            status_map = {}
            for line in output:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 4)
                if len(parts) < 4:
                    continue
                unit, load, active, sub = parts[:4]
                description = parts[4] if len(parts) > 4 else ""
                services.append(unit)
                status_map[unit] = {
                    "load": load,
                    "active": active,
                    "sub": sub,
                    "description": description,
                }
            if services:
                self.service_units = sorted(services)
                self.service_status = status_map
        except Exception:
            for unit in self.service_units:
                self.service_status.setdefault(unit, {"active": "unknown"})

    def _read_bus_lines(self):
        if not os.path.exists(BUS_FILE):
            return []
        try:
            st = os.stat(BUS_FILE)
            inode = getattr(st, "st_ino", None)
            if self.bus_inode is None:
                self.bus_inode = inode
            if inode is not None and self.bus_inode is not None and inode != self.bus_inode:
                self.bus_offset = 0
                self.bus_partial = b""
                self.bus_inode = inode
            if st.st_size < self.bus_offset:
                self.bus_offset = 0
                self.bus_partial = b""
            with open(BUS_FILE, "rb") as f:
                if not self.bootstrapped and st.st_size > TAIL_BOOTSTRAP_BYTES:
                    self.bus_offset = st.st_size - TAIL_BOOTSTRAP_BYTES
                    self.bootstrapped = True
                f.seek(self.bus_offset, os.SEEK_SET)
                chunk = f.read()
                self.bus_offset = f.tell()
            if not chunk:
                return []
            data = self.bus_partial + chunk
            lines = data.split(b"\n")
            if data and not data.endswith(b"\n"):
                self.bus_partial = lines.pop()
            else:
                self.bus_partial = b""
            return lines
        except Exception:
            return []

    def _record_heartbeat(self, ts: float, data: dict):
        if self.heartbeat_last_ts:
            self.heartbeat_intervals.append(ts - self.heartbeat_last_ts)
        self.heartbeat_last_ts = ts
        self.pending_pulses += 1
        self.omega["heartbeat"]["last_ts"] = ts
        self.omega["heartbeat"]["cycle"] = data.get("cycle", self.omega["heartbeat"]["cycle"])
        self.omega["heartbeat"]["uptime_s"] = data.get("uptime_s", self.omega["heartbeat"]["uptime_s"])

    def _summarize_event(self, topic: str, actor: str, level: str, data: dict) -> str:
        tag = f"{topic}"
        if level:
            tag += f" [{level}]"
        if actor:
            tag += f" ({actor})"
        if topic in {"rd.tasks.dispatch", "task.dispatch"}:
            target = data.get("target") or data.get("target_agent") or data.get("target_agent_id")
            if target:
                tag += f" -> {target}"
        if topic in {"rd.tasks.response", "task.complete"}:
            status = data.get("status") or data.get("result")
            if status:
                tag += f" {str(status)[:30]}"
        if "check_id" in data:
            tag += f" check={str(data.get('check_id'))[:8]}"
        return tag

    def _process_entry(self, entry: dict):
        topic = entry.get("topic", "unknown")
        data = entry.get("data", {})
        if not isinstance(data, dict):
            data = {}
        actor = entry.get("actor") or entry.get("sender") or data.get("sender") or "Unknown"
        level = entry.get("level", "")
        kind = entry.get("kind", "")
        ts = entry.get("ts") or entry.get("timestamp") or time.time()

        self.metrics["events_seen"] += 1
        self.metrics["agents"].add(actor)
        self.metrics["agent_last_seen"][actor] = ts  # v1.14: Track last activity
        self.event_counters[topic] += 1

        # v1.13: Track provider-level activity
        provider = data.get("provider") or ""
        if provider:
            self.metrics["provider_activity"][provider] += 1

        if topic in {"task.dispatch", "rd.tasks.dispatch"}:
            if topic == "task.dispatch":
                task_id = data.get("task_id") or data.get("req_id")
                target = data.get("target_agent") or data.get("target")
                desc = data.get("instruction") or data.get("intent") or data.get("desc")
            else:
                task_id = data.get("req_id") or data.get("task_id")
                target = data.get("target") or data.get("target_agent") or data.get("target_agent_id")
                desc = data.get("intent") or data.get("desc")
            if task_id:
                task_entry = {
                    "id": entry.get("id") or f"dispatch:{task_id}",
                    "run_id": task_id,
                    "status": "in_progress",
                    "actor": target or actor,
                    "ts": ts,
                    "meta": {"desc": desc} if desc else {},
                }
                self._ingest_task_entry(task_entry)
                self.metrics["dispatches"].append({"id": task_id, "target": target or actor})
            self.omega["dispatch"]["tick"] += 1
            self.omega["dispatch"]["last_ts"] = ts
        elif topic in {"rd.tasks.response", "task.complete"}:
            task_id = data.get("req_id") or data.get("task_id") or data.get("id")
            status = data.get("status")
            if not isinstance(status, str):
                status = "completed"
            if task_id:
                task_entry = {
                    "id": entry.get("id") or f"response:{task_id}",
                    "run_id": task_id,
                    "status": status,
                    "actor": data.get("sender") or actor,
                    "ts": ts,
                    "meta": {"desc": data.get("type")} if isinstance(data.get("type"), str) else {},
                }
                self._ingest_task_entry(task_entry)
            self.metrics["responses"].append(entry)
            self.omega["dispatch"]["sent"] += 1

        if topic == "a2a.negotiate.request":
            self.metrics["a2a_requests"] += 1
        elif topic == "a2a.negotiate.response":
            self.metrics["a2a_responses"] += 1
        elif topic == "agent.collab.request":
            if kind == "request":
                self.metrics["collab_requests"] += 1
            elif kind == "response":
                self.metrics["collab_responses"] += 1

        if topic == "task_ledger.append":
            entry = data.get("entry")
            if isinstance(entry, dict):
                self._ingest_task_entry(entry)
        elif topic.endswith(".task") and isinstance(entry, dict):
            if entry.get("status"):
                self._ingest_task_entry(entry)

        if topic in OMEGA_TOPICS:
            if topic == "omega.heartbeat":
                self.omega["heartbeat"]["count"] += 1
                self._record_heartbeat(ts, data)
                
                # COMPLIANCE ENFORCEMENT (v26)
                compliance = data.get("_auom_compliance")
                if compliance:
                    actor = entry.get("actor", "unknown")
                    for check in compliance.get("checks", []):
                        msg = f"![Violation] {actor} failed {check['law']}"
                        self.log_buffer.append(msg)
            elif topic == "omega.guardian.cycle":
                self.omega["guardian"]["count"] += 1
                self.omega["guardian"]["last_ts"] = ts
                self.omega["guardian"]["cycle"] = data.get("cycle", self.omega["guardian"]["cycle"])
            elif topic == "omega.guardian.warn":
                self.omega["guardian"]["warn"] += 1
                self.omega["guardian"]["last_warn_ts"] = ts
            elif topic == "omega.dispatch.tick":
                self.omega["dispatch"]["tick"] += 1
                self.omega["dispatch"]["last_ts"] = ts
                self.omega["dispatch"]["pending_total"] = data.get("pending_total", self.omega["dispatch"]["pending_total"])
            elif topic == "omega.dispatch.sent":
                self.omega["dispatch"]["sent"] += 1
            elif topic == "omega.dispatch.access_denied":
                self.omega["dispatch"]["denied"] += 1
            elif topic == "omega.dispatch.ring_check_error":
                self.omega["dispatch"]["errors"] += 1
            elif topic == "omega.queue.depth":
                self.omega["queue"]["count"] += 1
                self.omega["queue"]["last_ts"] = ts
                self.omega["queue"]["pending_requests"] = data.get("pending_requests", self.omega["queue"]["pending_requests"])
                self.omega["queue"]["total_events"] = data.get("total_events", self.omega["queue"]["total_events"])
            elif topic == "omega.pending.pairs":
                self.omega["pairs"]["count"] += 1
                self.omega["pairs"]["last_ts"] = ts
                self.omega["pairs"]["total"] = data.get("total", self.omega["pairs"]["total"])
                oldest = data.get("oldest_age_s")
                if oldest is None:
                    oldest = 0.0
                    by_pair = data.get("by_pair", {}) if isinstance(data.get("by_pair"), dict) else {}
                    by_topic = data.get("by_topic", {}) if isinstance(data.get("by_topic"), dict) else {}
                    for info in list(by_pair.values()) + list(by_topic.values()):
                        if not isinstance(info, dict):
                            continue
                        oldest = max(oldest, float(info.get("oldest_age_s", 0.0)))
                try:
                    self.omega["pairs"]["oldest_age_s"] = float(oldest)
                except (TypeError, ValueError):
                    self.omega["pairs"]["oldest_age_s"] = 0.0
                by_topic = data.get("by_topic", {}) if isinstance(data.get("by_topic"), dict) else {}
                pending_dispatch = 0
                for key in ("rd.tasks.dispatch", "task.dispatch"):
                    val = by_topic.get(key)
                    if isinstance(val, dict):
                        pending_dispatch += int(val.get("pending_total") or val.get("total") or val.get("count") or 0)
                    elif isinstance(val, (int, float)):
                        pending_dispatch += int(val)
                if pending_dispatch:
                    self.omega["dispatch"]["pending_total"] = pending_dispatch
            elif topic == "omega.providers.scan":
                self.omega["providers"]["count"] += 1
                self.omega["providers"]["last_ts"] = ts
                providers = data.get("providers") if isinstance(data.get("providers"), dict) else {}
                self.omega["providers"]["details"] = providers
                self.omega["providers"]["providers"] = self._normalize_provider_status(providers)
            elif topic == "omega.providers.health":
                self.omega["providers"]["count"] += 1
                self.omega["providers"]["last_ts"] = ts
                providers = data.get("providers") if isinstance(data.get("providers"), dict) else {}
                self.omega["providers"]["details"] = providers
                self.omega["providers"]["health_score"] = data.get("health_score", self.omega["providers"]["health_score"])
                self.omega["providers"]["providers"] = self._normalize_provider_status(providers)
            elif topic == "omega.health":
                self.omega["health"]["count"] += 1
                self.omega["health"]["last_ts"] = ts
                self.omega["health"]["status"] = data.get("status", self.omega["health"]["status"])
                self.omega["health"]["score"] = data.get("score", self.omega["health"]["score"])
            elif topic == "omega.health.composite":
                self.omega["health"]["count"] += 1
                self.omega["health"]["last_ts"] = ts
                self.omega["health"]["status"] = data.get("status", self.omega["health"]["status"])
                self.omega["health"]["score"] = data.get("score", self.omega["health"]["score"])
            elif topic == "omega.metrics.velocity":
                self.omega["metrics"]["velocity"] = data
            elif topic == "omega.metrics.latency":
                self.omega["metrics"]["latency"] = data
            elif topic == "omega.metrics.entropy":
                self.omega["metrics"]["entropy"] = data

        summary = self._summarize_event(topic, actor, level, data)
        self.log_buffer.append(
            {
                "topic": topic,
                "level": level,
                "summary": summary,
            }
        )

    def _emit_bus_event(self, *, topic: str, kind: str, level: str, data: dict) -> None:
        if not topic:
            return
        if agent_bus is not None:
            paths = agent_bus.resolve_bus_paths(BUS_DIR)
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind=kind,
                level=level,
                actor=os.environ.get("PLURIBUS_ACTOR", "ohm"),
                data=data,
                trace_id=None,
                run_id=None,
                durable=False,
            )
            return
        try:
            Path(BUS_FILE).parent.mkdir(parents=True, exist_ok=True)
            event = {
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "topic": topic,
                "kind": kind,
                "level": level,
                "actor": os.environ.get("PLURIBUS_ACTOR", "ohm"),
                "data": data,
            }
            with open(BUS_FILE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            return

    def _status_payload(self) -> dict:
        task_counts, active_tasks = self._task_counts()
        bus_mb = self._bus_size_mb()
        bus_warn = bool(self.bus_warn_mb) and bus_mb >= self.bus_warn_mb
        active_services, total_services = self._service_counts()
        inactive_services = [
            unit
            for unit in self.service_units
            if self.service_status.get(unit, {}).get("active") != "active"
        ]
        # v1.16: Calculate live agent count for status payload
        now = time.time()
        live_agent_count = 0
        for actor in self.metrics["agents"]:
            last_ts = self.metrics["agent_last_seen"].get(actor, 0)
            if last_ts:
                if isinstance(last_ts, str):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
                        last_ts = dt.timestamp()
                    except (ValueError, TypeError):
                        continue
                if (now - float(last_ts)) <= self.agent_active_s:
                    live_agent_count += 1
        return {
            "bus_size_mb": round(bus_mb, 2),
            "bus_warn": bus_warn,
            "services": {
                "active": active_services,
                "total": total_services,
                "inactive": inactive_services[:12],
            },
            "tasks": {
                "counts": dict(task_counts),
                "active": len(active_tasks),
                "dispatched": self.metrics["tasks_started"],
                "completed": self.metrics["tasks_completed"],
            },
            "agents": {
                "seen": len(self.metrics["agents"]),
                "live": live_agent_count,
                "last_seen": self.metrics["agent_last_seen"],
            },
            "a2a": {
                "requests": self.metrics["a2a_requests"],
                "responses": self.metrics["a2a_responses"],
            },
            "collab": {
                "requests": self.metrics["collab_requests"],
                "responses": self.metrics["collab_responses"],
            },
            "omega": {
                "queue": self.omega.get("queue", {}),
                "dispatch": self.omega.get("dispatch", {}),
                "pairs": self.omega.get("pairs", {}),
                "providers": self.omega.get("providers", {}),
                "health": self.omega.get("health", {}),
            },
        }

    def _emit_status(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_emit_ts) < self.emit_interval_s:
            return
        self._last_emit_ts = now
        payload = self._status_payload()
        self._emit_bus_event(topic=self.emit_topic, kind="metric", level="info", data=payload)
        if not self.emit_alerts:
            return
        alert_state = {
            "bus_warn": payload["bus_warn"],
            "inactive_services": payload["services"]["inactive"],
            "stale_tasks": payload["tasks"]["counts"].get("stale", 0),
        }
        if alert_state != self._last_alert_state or (now - self._last_alert_ts) > (self.emit_interval_s * 4):
            self._last_alert_state = alert_state
            self._last_alert_ts = now
            level = "warn" if payload["bus_warn"] or alert_state["inactive_services"] else "info"
            self._emit_bus_event(topic=self.alert_topic, kind="log", level=level, data=alert_state)

    def _emit_art_tick(self) -> None:
        now = time.time()
        if (now - self._last_art_tick_ts) < ART_TICK_INTERVAL_S:
            return
        self._last_art_tick_ts = now
        self._emit_bus_event(
            topic="art.director.tick",
            kind="trigger",
            level="info",
            data={"source": "ohm", "interval_s": ART_TICK_INTERVAL_S}
        )

    def tail_bus(self):
        """Reads new lines from the bus file."""
        lines = self._read_bus_lines()
        if not lines:
            return
        for raw in lines:
            if not raw:
                continue
            try:
                entry = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            self._process_entry(entry)

    def _bus_size_mb(self, path=None) -> float:
        path = path or BUS_FILE
        try:
            size = os.path.getsize(path)
        except (OSError, TypeError):
            return 0.0
        return size / (1024 * 1024)

    def _bus_size_label(self):
        mb = self._bus_size_mb()
        warn = bool(self.bus_warn_mb) and mb >= self.bus_warn_mb
        label = f"BUS: {mb:.1f}MB"
        if warn:
            label += " (rotate)"
        return label, warn

    def _normalize_provider_status(self, providers: dict) -> dict:
        normalized = {}
        for name, value in providers.items():
            if isinstance(value, dict):
                ok = value.get("available")
                if ok is None:
                    ok = value.get("healthy")
                if ok is None:
                    ok = value.get("ok")
                if ok is None:
                    status = str(value.get("status", "")).lower()
                    ok = status in {"ok", "healthy", "available"}
                normalized[name] = bool(ok) if ok is not None else False
            else:
                normalized[name] = bool(value)
        return normalized

    def _provider_string(self) -> str:
        providers = self.omega["providers"].get("providers", {})
        if not providers:
            return "n/a"
        if any(isinstance(value, dict) for value in providers.values()):
            providers = self._normalize_provider_status(providers)
        parts = []
        for name, ok in sorted(providers.items()):
            parts.append(f"{name}:{'on' if ok else 'off'}")
        out = ",".join(parts)
        return out[:46] + "..." if len(out) > 49 else out

    def _services_summary(self) -> str:
        active, total = self._service_counts()
        if total == 0:
            return "n/a"
        return f"{active}/{total} active"

    def _service_counts(self):
        total = len(self.service_units)
        active = 0
        for unit in self.service_units:
            status = self.service_status.get(unit, {})
            if status.get("active") == "active":
                active += 1
        return active, total

    def _service_icon(self, load: str, active: str, sub: str) -> str:
        active = (active or "").lower()
        load = (load or "").lower()
        sub = (sub or "").lower()
        if load not in {"loaded", ""}:
            return "⚠️"
        if active == "active":
            return "⚡"
        if active in {"activating", "reloading"} or sub in {"auto-restart", "start"}:
            return "⏳"
        if active in {"failed", "deactivating"}:
            return "❌"
        if active == "inactive":
            return "⏸"
        return "❔"

    def _service_detail_lines(self, width: int = 72):
        lines = []
        for unit in self.service_units:
            status = self.service_status.get(unit, {})
            load = status.get("load", "unknown")
            active = status.get("active", "unknown")
            sub = status.get("sub", "unknown")
            desc = status.get("description", "")
            icon = self._service_icon(load, active, sub)
            label = f"{icon} {unit}: {load} {active}/{sub} {desc}".strip()
            if len(label) > width:
                label = label[: width - 3] + "..."
            lines.append(
                {
                    "unit": unit,
                    "label": label,
                    "load": load,
                    "active": active,
                    "sub": sub,
                }
            )
        return lines

    def _tmux_list_sessions(self):
        if not shutil.which("tmux"):
            return []
        try:
            output = subprocess.check_output(
                ["tmux", "list-sessions", "-F", "#S|#{?session_attached,attached,detached}|#{session_windows}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).splitlines()
        except subprocess.CalledProcessError:
            return []
        sessions = []
        for line in output:
            parts = line.split("|")
            if not parts:
                continue
            name = parts[0]
            state = parts[1] if len(parts) > 1 else "unknown"
            windows = parts[2] if len(parts) > 2 else "?"
            sessions.append({"name": name, "state": state, "windows": windows})
        return sessions

    def _tmux_list_windows(self, session=None):
        if not shutil.which("tmux"):
            return []
        cmd = ["tmux", "list-windows", "-F", "#W"]
        if session:
            cmd.extend(["-t", session])
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).splitlines()
        except subprocess.CalledProcessError:
            return []

    def _tmux_list_panes(self, target=None):
        if not shutil.which("tmux"):
            return []
        cmd = ["tmux", "list-panes", "-F", "#{pane_id}|#{pane_left}|#{pane_right}|#{pane_top}|#{pane_bottom}"]
        if target:
            cmd.extend(["-t", target])
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).splitlines()
        except subprocess.CalledProcessError:
            return []
        panes = []
        for line in output:
            parts = line.split("|")
            if len(parts) < 5:
                continue
            try:
                panes.append(
                    {
                        "id": parts[0],
                        "left": int(parts[1]),
                        "right": int(parts[2]),
                        "top": int(parts[3]),
                        "bottom": int(parts[4]),
                    }
                )
            except ValueError:
                continue
        return panes

    def _tmux_left_right_panes(self, target):
        panes = self._tmux_list_panes(target)
        if not panes:
            return (None, None)
        panes_sorted = sorted(panes, key=lambda pane: pane["left"])
        left_id = panes_sorted[0]["id"]
        right_id = panes_sorted[-1]["id"] if len(panes_sorted) > 1 else left_id
        return (left_id, right_id)

    def _tmux_display(self, format_str, target=None):
        if not shutil.which("tmux"):
            return None
        cmd = ["tmux", "display-message", "-p", format_str]
        if target:
            cmd.extend(["-t", target])
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            return None

    def _tmux_pane_width(self, target):
        raw = self._tmux_display("#{pane_width}", target=target)
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    def _tmux_split_right(self, target, right_pct, min_left_cols=None, width_override=None):
        width = self._tmux_pane_width(target) or width_override
        if width:
            right_cols = max(10, int(round(width * (right_pct / 100.0))))
            if min_left_cols and width - right_cols < min_left_cols:
                right_cols = max(10, width - min_left_cols)
            subprocess.run(["tmux", "split-window", "-h", "-l", str(right_cols), "-t", target], check=True)
            return
        subprocess.run(["tmux", "split-window", "-h", "-l", "40", "-t", target], check=True)

    def _tmux_resize_pane(self, target, min_cols=None, min_rows=None):
        if not shutil.which("tmux"):
            return
        if min_cols:
            try:
                subprocess.run(["tmux", "resize-pane", "-t", target, "-x", str(min_cols)], check=True)
            except subprocess.CalledProcessError:
                pass
        if min_rows:
            try:
                subprocess.run(["tmux", "resize-pane", "-t", target, "-y", str(min_rows)], check=True)
            except subprocess.CalledProcessError:
                pass

    def _tmux_unique_name(self, base, existing):
        if base not in existing:
            return base
        suffix = time.strftime("%H%M%S", time.gmtime())
        candidate = f"{base}-{suffix}"
        if candidate not in existing:
            return candidate
        return f"{base}-{suffix}-{os.getpid()}"

    def _prompt_tmux_session(self, sessions, session_name=None):
        if session_name:
            if session_name.lower() in {"none", "no", "off"}:
                return None
            return session_name
        if not sessions:
            return None
        if not sys.stdin.isatty():
            return sessions[0]["name"]
        print("Select tmux session to attach in right pane:")
        print("  0) none (plain shell)")
        for idx, session in enumerate(sessions, start=1):
            print(f"  {idx}) {session['name']} [{session['state']}] windows={session['windows']}")
        while True:
            choice = input("Choice: ").strip()
            if not choice:
                return sessions[0]["name"]
            if choice.isdigit():
                idx = int(choice)
                if idx == 0:
                    return None
                if 1 <= idx <= len(sessions):
                    return sessions[idx - 1]["name"]
            print("Invalid selection.")

    def _build_ohm_command(self, filter_name=None):
        cmd = [sys.executable, os.path.abspath(__file__), "--cli"]
        if filter_name:
            cmd.extend(["--filter", filter_name])
        if self.refresh_s:
            cmd.extend(["--interval", f"{self.refresh_s}"])
        if self.agent_expanded:
            cmd.append("--agents-full")
        if self.show_stale_tasks:
            cmd.append("--show-stale-tasks")
        return cmd

    def _tmux_attach_command(self, session_name):
        if not session_name:
            return None
        return ["env", "-u", "TMUX", "tmux", "attach-session", "-t", session_name]

    def run_tmux_split(self, session_name=None, filter_name=None):
        if not shutil.which("tmux"):
            print("tmux not available; install tmux to use --split.")
            return 1
        sessions = self._tmux_list_sessions()
        selected = self._prompt_tmux_session(sessions, session_name=session_name)
        ohm_cmd = self._build_ohm_command(filter_name=filter_name)
        attach_cmd = self._tmux_attach_command(selected)

        right_pct_env = os.environ.get("OHM_SPLIT_RIGHT_PCT")
        min_cols_env = os.environ.get("OHM_SPLIT_MIN_LEFT_COLS")
        min_rows_env = os.environ.get("OHM_SPLIT_MIN_LEFT_ROWS")
        try:
            right_pct = int(right_pct_env) if right_pct_env else TMUX_RIGHT_PCT_DEFAULT
        except ValueError:
            right_pct = TMUX_RIGHT_PCT_DEFAULT
        right_pct = max(20, min(80, right_pct))
        try:
            min_cols = int(min_cols_env) if min_cols_env else TMUX_LEFT_MIN_COLS_DEFAULT
        except ValueError:
            min_cols = TMUX_LEFT_MIN_COLS_DEFAULT
        try:
            min_rows = int(min_rows_env) if min_rows_env else TMUX_LEFT_MIN_ROWS_DEFAULT
        except ValueError:
            min_rows = TMUX_LEFT_MIN_ROWS_DEFAULT

        try:
            if os.environ.get("TMUX"):
                window_names = self._tmux_list_windows()
                window_name = self._tmux_unique_name("ohm", window_names)
                left_pane = subprocess.check_output(
                    ["tmux", "new-window", "-n", window_name, "-P", "-F", "#{pane_id}"] + ohm_cmd,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                target = left_pane or f":{window_name}"
                self._tmux_split_right(target, right_pct, min_left_cols=min_cols)
                left_id, right_id = self._tmux_left_right_panes(target)
                if left_id:
                    self._tmux_resize_pane(left_id, min_cols=min_cols, min_rows=min_rows)
                if attach_cmd and right_id:
                    attach_line = shlex.join(attach_cmd)
                    subprocess.run(["tmux", "send-keys", "-t", right_id, "-l", attach_line], check=True)
                    subprocess.run(["tmux", "send-keys", "-t", right_id, "C-m"], check=True)
                subprocess.run(["tmux", "select-window", "-t", target], check=True)
                return 0

            session_names = [s["name"] for s in sessions]
            session_name = self._tmux_unique_name("ohm", session_names)
            term_size = shutil.get_terminal_size(fallback=(120, 40))
            left_pane = subprocess.check_output(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session_name,
                    "-n",
                    "ohm",
                    "-x",
                    str(term_size.columns),
                    "-y",
                    str(term_size.lines),
                    "-P",
                    "-F",
                    "#{pane_id}",
                ]
                + ohm_cmd,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            target = left_pane or f"{session_name}:ohm"
            self._tmux_split_right(target, right_pct, min_left_cols=min_cols, width_override=term_size.columns)
            left_id, right_id = self._tmux_left_right_panes(target)
            if left_id:
                self._tmux_resize_pane(left_id, min_cols=min_cols, min_rows=min_rows)
            if attach_cmd and right_id:
                attach_line = shlex.join(attach_cmd)
                subprocess.run(["tmux", "send-keys", "-t", right_id, "-l", attach_line], check=True)
                subprocess.run(["tmux", "send-keys", "-t", right_id, "C-m"], check=True)
            subprocess.run(["tmux", "attach-session", "-t", session_name], check=True)
            return 0
        except subprocess.CalledProcessError as exc:
            print(f"tmux split failed: {exc}")
            return 1

    def _agent_dashboard_lines(self, width: int = 60, max_agents: int = 10, show_all: bool = False) -> list:
        """Return agent dashboard lines with last-seen and active task counts."""
        now = time.time()
        last_seen = self.metrics.get("agent_last_seen", {})
        task_counts = self._agent_task_counts(now)

        known_agents = []
        for meta in AGENT_TAXONOMY.values():
            known_agents.extend(meta["agents"])
        known_set = set(known_agents)
        seen_agents = set(self.metrics.get("agents", set()))
        unknown_agents = sorted(seen_agents - known_set)

        def _age_info(actor: str):
            last_ts = last_seen.get(actor, 0)
            if not last_ts:
                return None, "never"
            # Handle string timestamps (ISO format) or numeric
            if isinstance(last_ts, str):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
                    last_ts = dt.timestamp()
                except (ValueError, TypeError):
                    return None, "never"
            age_s = max(0.0, now - float(last_ts))
            if age_s < 60:
                return age_s, f"{int(age_s)}s"
            if age_s < 3600:
                return age_s, f"{int(age_s/60)}m"
            if age_s < 86400:
                return age_s, f"{age_s/3600:.1f}h"
            return age_s, f"{age_s/86400:.1f}d"

        def _status_icon(age_s):
            if age_s is None:
                return "⚫"
            if age_s < 60:
                return "🟢"
            if age_s < 300:
                return "🟡"
            if age_s < 3600:
                return "🟠"
            return "🔴"

        def _format_agent(actor: str):
            _, glyph, _, _ = categorize_agent(actor)
            age_s, age_str = _age_info(actor)
            status = _status_icon(age_s)
            tasks = task_counts.get(actor, 0)
            task_str = f" t{tasks:>2}" if tasks else "    "
            label = f"{glyph} {actor[:18]:<18} {status} {age_str:>6}{task_str}"
            if len(label) > width:
                label = label[: width - 3] + "..."
            return {"label": label, "actor": actor, "age": age_s if age_s is not None else float("inf")}

        lines = []
        if show_all:
            for cat, meta in AGENT_TAXONOMY.items():
                agents = meta.get("agents", [])
                active_count = 0
                for agent in agents:
                    last_ts = last_seen.get(agent, 0)
                    if last_ts and (now - last_ts) <= self.agent_active_s:
                        active_count += 1
                header = f"{meta.get('glyph', '?')} {cat} r{meta.get('ring', '?')} {active_count}/{len(agents)} active"
                lines.append({"label": header[:width], "actor": cat, "age": -1, "is_header": True})
                for agent in agents:
                    lines.append(_format_agent(agent))
            if unknown_agents:
                lines.append({"label": "❔ other", "actor": "other", "age": -1, "is_header": True})
                for agent in unknown_agents:
                    lines.append(_format_agent(agent))
        else:
            for actor in seen_agents:
                lines.append(_format_agent(actor))
            lines.sort(key=lambda x: x.get("age", float("inf")))

        if max_agents and len(lines) > max_agents:
            lines = lines[:max_agents]
        return lines

    def _operator_lines(self, width: int = 60) -> list:
        if not SEMANTIC_OPERATORS:
            return []
        tokens = [f"{meta['glyph']}{name}" for name, meta in SEMANTIC_OPERATORS.items()]
        lines = []
        current = ""
        for token in tokens:
            if current and (len(current) + 1 + len(token)) > width:
                lines.append(current)
                current = token
            else:
                current = token if not current else f"{current} {token}"
        if current:
            lines.append(current)
        return lines


    def print_cli_status(self, clear: bool = True):
        """Prints a compact CLI status (non-curses mode)."""
        self.tail_task_ledger()
        self.tail_bus()
        self._prune_tasks()
        self._refresh_services()
        self._update_wave()
        task_counts, active_tasks = self._task_counts()
        self.metrics["active_tasks"] = active_tasks
        if self.omega["dispatch"]["pending_total"] < len(active_tasks):
            self.omega["dispatch"]["pending_total"] = len(active_tasks)

        term_size = shutil.get_terminal_size(fallback=(120, 40))
        cols = max(60, term_size.columns)
        rows = max(20, term_size.lines)
        width_env = os.environ.get("OHM_BOX_WIDTH")
        try:
            requested_width = int(width_env) if width_env else cols
        except ValueError:
            requested_width = cols
        box_width = min(cols, max(60, requested_width))
        inner_width = box_width - 2
        log_lines_env = os.environ.get("OHM_LOG_LINES")
        try:
            log_lines = int(log_lines_env) if log_lines_env else 0
        except ValueError:
            log_lines = 0

        # ANSI colors for neon effect
        NEON_CYAN = '\033[38;5;51m'
        NEON_MAGENTA = '\033[38;5;201m'
        NEON_GREEN = '\033[38;5;118m'
        NEON_YELLOW = '\033[38;5;226m'
        NEON_ORANGE = '\033[38;5;208m'
        NEON_RED = '\033[38;5;203m'
        NEON_BLUE = '\033[38;5;75m'
        NEON_WHITE = '\033[38;5;255m'
        DIM = '\033[2m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

        def border(left, fill, right):
            return f"{NEON_CYAN}{BOLD}{left}{fill * inner_width}{right}{RESET}"

        border_top = border("╔", "═", "╗")
        border_mid = border("╠", "═", "╣")
        border_thin = border("╠", "─", "╣")
        border_bottom = border("╚", "═", "╝")

        if clear:
            # Clear screen and move cursor to top
            print('\033[2J\033[H', end='')

        stats = self._heartbeat_stats()
        hb = self.omega["heartbeat"]
        guardian = self.omega["guardian"]
        dispatch = self.omega["dispatch"]
        queue = self.omega["queue"]
        pairs = self.omega["pairs"]
        health = self.omega["health"]

        wave = ''.join(self.wave)
        hb_age = self._format_age(hb["last_ts"])
        guardian_age = self._format_age(guardian["last_ts"])
        guardian_warn_age = self._format_age(guardian["last_warn_ts"])
        dispatch_age = self._format_age(dispatch["last_ts"])
        queue_age = self._format_age(queue["last_ts"])
        health_age = self._format_age(health["last_ts"])
        bus_label, bus_warn = self._bus_size_label()

        # Header
        header_text = "OMEGA HEART MONITOR (OHM) v1.16.0 - CLI Mode (Elite)"
        header = header_text.center(inner_width)
        print(border_top)
        print(f"{NEON_CYAN}{BOLD}║{header}║{RESET}")
        print(border_mid)

        agents = list(self.metrics["agents"])
        known_agents = set()
        for meta in AGENT_TAXONOMY.values():
            known_agents.update(meta.get("agents", []))
        active_task_count = len(active_tasks)
        # v1.16: Count "live" agents - those seen within agent_active_s window
        now = time.time()
        live_agent_count = 0
        for actor in agents:
            last_ts = self.metrics["agent_last_seen"].get(actor, 0)
            if last_ts:
                # Handle string timestamps (ISO format) or numeric
                if isinstance(last_ts, str):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
                        last_ts = dt.timestamp()
                    except (ValueError, TypeError):
                        continue
                if (now - float(last_ts)) <= self.agent_active_s:
                    live_agent_count += 1
        print(
            f"{NEON_CYAN}║{RESET} {NEON_MAGENTA}AGENTS:{RESET} {len(agents):<3} ({NEON_GREEN}{live_agent_count} live{RESET}) "
            f"{NEON_CYAN}│{RESET} {NEON_GREEN}TASKS:{RESET} {active_task_count:<3} "
            f"{NEON_CYAN}│{RESET} {NEON_YELLOW}DISPATCHED:{RESET} {self.metrics['tasks_started']:<4} "
            f"{NEON_CYAN}│{RESET} {NEON_GREEN}COMPLETED:{RESET} {self.metrics['tasks_completed']:<4} "
            f"{NEON_CYAN}│{RESET} {NEON_MAGENTA}A2A:{RESET} {self.metrics['a2a_requests']}/{self.metrics['a2a_responses']} "
            f"{NEON_CYAN}│{RESET} {NEON_BLUE}COLLAB:{RESET} {self.metrics['collab_requests']}/{self.metrics['collab_responses']}")

        agent_str = ', '.join(agents[:5]) + ('...' if len(agents) > 5 else '')
        print(f"{NEON_CYAN}║{RESET} {NEON_YELLOW}Agents:{RESET} [{agent_str}]")
        tasks_line = (
            f"run {task_counts.get('in_progress', 0)} "
            f"blocked {task_counts.get('blocked', 0)} "
            f"planned {task_counts.get('planned', 0)} "
            f"done {task_counts.get('completed', 0)} "
            f"fail {task_counts.get('abandoned', 0)}"
        )
        if task_counts.get("stale", 0):
            tasks_line += f" stale {task_counts['stale']}"
        print(f"{NEON_CYAN}║{RESET} {NEON_BLUE}Tasks:{RESET} {NEON_WHITE}{tasks_line}{RESET}")

        task_rows_env = os.environ.get("OHM_TASK_ROWS")
        try:
            max_task_rows = int(task_rows_env) if task_rows_env else 0
        except ValueError:
            max_task_rows = 0
        if max_task_rows <= 0:
            max_task_rows = min(15, max(3, rows - 30))
        task_rows = self._task_progress_rows(inner_width, max_task_rows)
        task_section_lines = 0
        print(f"{NEON_CYAN}║{RESET} {NEON_MAGENTA}{BOLD}ACTIVE TASKS:{RESET}")
        if task_rows:
            task_section_lines = 1 + len(task_rows)
            for row in task_rows:
                status = row.get("status")
                if status == "in_progress":
                    row_color = NEON_GREEN
                elif status == "blocked":
                    row_color = NEON_ORANGE
                elif status == "stale":
                    row_color = NEON_RED
                else:
                    row_color = NEON_CYAN
                print(f"{NEON_CYAN}║{RESET}   {row_color}{row['line']}{RESET}")
        else:
            task_section_lines = 2
            print(f"{NEON_CYAN}║{RESET}   {DIM}{NEON_WHITE}none{RESET}")

        print(border_mid)
        print(f"{NEON_CYAN}║{RESET} {BOLD}{NEON_WHITE}PULSE:{RESET} [{wave}]")
        print(
            f"{NEON_CYAN}║{RESET} {NEON_RED}HB:{RESET} beats {NEON_WHITE}{hb['count']}{RESET} cycle {NEON_WHITE}{hb['cycle']}{RESET} "
            f"bpm {NEON_WHITE}{stats['bpm']:.1f}{RESET} avg {NEON_WHITE}{stats['avg']:.1f}s{RESET} "
            f"jitter {NEON_WHITE}{stats['jitter']:.1f}s{RESET} last {NEON_WHITE}{hb_age}{RESET}")
        print(
            f"{NEON_CYAN}║{RESET} {NEON_MAGENTA}GUARD:{RESET} cycles {NEON_WHITE}{guardian['count']}{RESET} "
            f"warns {NEON_WHITE}{guardian['warn']}{RESET} last {NEON_WHITE}{guardian_age}{RESET} "
            f"warn {NEON_WHITE}{guardian_warn_age}{RESET}")
        print(
            f"{NEON_CYAN}║{RESET} {NEON_YELLOW}DISPATCH:{RESET} ticks {NEON_WHITE}{dispatch['tick']}{RESET} "
            f"sent {NEON_WHITE}{dispatch['sent']}{RESET} pend {NEON_WHITE}{dispatch['pending_total']}{RESET} "
            f"denied {NEON_WHITE}{dispatch['denied']}{RESET} err {NEON_WHITE}{dispatch['errors']}{RESET} "
            f"last {NEON_WHITE}{dispatch_age}{RESET}")
        print(
            f"{NEON_CYAN}║{RESET} {NEON_BLUE}QUEUE:{RESET} pending {NEON_WHITE}{queue['pending_requests']}{RESET} "
            f"events {NEON_WHITE}{queue['total_events']}{RESET} last {NEON_WHITE}{queue_age}{RESET}")
        bus_color = NEON_RED if bus_warn else NEON_CYAN
        print(f"{NEON_CYAN}║{RESET} {bus_color}{bus_label}{RESET}")
        health_status = str(health['status']).lower()
        health_color = NEON_GREEN if health_status in {"nominal", "ok", "healthy"} else (NEON_ORANGE if health_status in {"degraded", "warn", "warning"} else NEON_RED)
        print(
            f"{NEON_CYAN}║{RESET} {NEON_BLUE}PAIRS:{RESET} pending {NEON_WHITE}{pairs['total']}{RESET} "
            f"oldest {NEON_WHITE}{pairs['oldest_age_s']:.1f}s{RESET} "
            f"{NEON_CYAN}│{RESET} {NEON_MAGENTA}HEALTH:{RESET} {health_color}{health['status']}{RESET} "
            f"last {NEON_WHITE}{health_age}{RESET}")
        print(f"{NEON_CYAN}║{RESET} {NEON_CYAN}PROVIDERS:{RESET} {NEON_WHITE}{self._provider_string()}{RESET}")
        # v1.10: Show provider activity counts
        prov_acts = self.metrics["provider_activity"]
        if prov_acts:
            prov_str = " ".join(f"{k}:{v}" for k, v in prov_acts.most_common(5))
            print(f"{NEON_CYAN}║{RESET} {NEON_MAGENTA}ACTIVITY:{RESET} {NEON_WHITE}{prov_str}{RESET}")

        # v1.14: Agent Dashboard grid cell
        agent_lines = self._agent_dashboard_lines(
            max_agents=(0 if self.agent_expanded else 5),
            show_all=self.agent_expanded,
            width=max(10, inner_width - 4),
        )
        if agent_lines:
            print(f"{NEON_CYAN}║{RESET} {NEON_CYAN}AGENTS (Ctrl-A in curses):{RESET}")
            for aline in agent_lines:
                label = aline.get("label", "")
                if aline.get("is_header"):
                    print(f"{NEON_CYAN}║{RESET}   {BOLD}{NEON_WHITE}{label}{RESET}")
                else:
                    print(f"{NEON_CYAN}║{RESET}   {NEON_WHITE}{label}{RESET}")
        if self.agent_expanded:
            op_lines = self._operator_lines(width=max(10, inner_width - 4))
            if op_lines:
                print(f"{NEON_CYAN}║{RESET} {NEON_MAGENTA}OPERATORS:{RESET}")
                for line in op_lines:
                    print(f"{NEON_CYAN}║{RESET}   {NEON_WHITE}{line}{RESET}")

        svc_active, svc_total = self._service_counts()
        if svc_total == 0:
            svc_color = NEON_RED
        elif svc_active == svc_total:
            svc_color = NEON_GREEN
        else:
            svc_color = NEON_ORANGE
        service_lines = self._service_detail_lines(width=max(10, inner_width - 4))
        print(
            f"{NEON_CYAN}║{RESET} {NEON_CYAN}SERVICES:{RESET} {svc_color}{self._services_summary()}{RESET}")
        for svc in service_lines:
            active = svc.get("active", "")
            if active == "active":
                line_color = NEON_GREEN
            elif active in {"inactive", "failed", "deactivating"}:
                line_color = NEON_RED
            else:
                line_color = NEON_YELLOW
            print(f"{NEON_CYAN}║{RESET}   {line_color}{svc['label']}{RESET}")

        print(border_mid)

        # Recent Dispatches
        print(f"{NEON_CYAN}║{RESET} {NEON_YELLOW}{BOLD}PENDING DISPATCHES:{RESET}")
        for d in list(self.metrics["dispatches"])[-3:]:
            print(f"{NEON_CYAN}║{RESET}   → [{d['id']}] -> {d['target']}")

        # Recent Responses
        if self.metrics["responses"]:
            print(border_thin)
            print(f"{NEON_CYAN}║{RESET} {NEON_GREEN}{BOLD}RESPONSES:{RESET}")
            for r in list(self.metrics["responses"])[-3:]:
                sender = r.get('actor', r.get('sender', 'Unknown'))
                rtype = r.get('data', {}).get('type', 'task_response')
                print(f"{NEON_CYAN}║{RESET}   ← [{NEON_GREEN}{sender}{RESET}] {rtype[:40]}")

        print(border_mid)

        # Event Log Tail
        tabs = []
        for idx, (label, _) in enumerate(self.filters):
            if idx == self.filter_index:
                tabs.append(f"{BOLD}{NEON_MAGENTA}[{label}]{RESET}")
            else:
                tabs.append(f"{DIM}{NEON_WHITE}{label}{RESET}")
        print(f"{NEON_CYAN}║{RESET} {NEON_CYAN}{BOLD}FILTERS:{RESET} " + " ".join(tabs))
        available_rows = rows - (20 + len(service_lines) + task_section_lines)
        if self.metrics["responses"]:
            available_rows -= 4
        if log_lines <= 0:
            log_lines = max(3, min(12, available_rows))
        print(f"{NEON_CYAN}║{RESET} {BOLD}{NEON_WHITE}EVENT LOG (last {log_lines}):{RESET}")
        entries = self._filtered_log_entries(limit=log_lines)
        if not entries:
            print(f"{NEON_CYAN}║{RESET} {DIM}{NEON_WHITE}no events for filter{RESET}")
        for entry in entries:
            line = entry.get("summary", "")
            if len(line) > inner_width - 2:
                line = line[: max(10, inner_width - 5)] + '...'
            level = str(entry.get("level", "")).lower()
            topic = entry.get("topic", "")
            if level == "error":
                color = NEON_RED
            elif level == "warn":
                color = NEON_ORANGE
            elif topic.startswith("omega."):
                color = NEON_CYAN
            elif topic.startswith("qa."):
                color = NEON_MAGENTA
            elif topic.startswith("rd.tasks.") or topic.startswith("task."):
                color = NEON_GREEN
            elif "dispatch" in topic:
                color = NEON_YELLOW
            elif "response" in topic:
                color = NEON_BLUE
            else:
                color = NEON_WHITE
            print(f"{NEON_CYAN}║{RESET} {color}{line}{RESET}")

        print(border_bottom)
        print(f"{NEON_CYAN}Press Ctrl+C to exit. Refreshing every {self.refresh_s:.0f}s...{RESET}")

    def run_cli_mode(self, iterations=None):
        """Runs in CLI mode (non-curses)."""
        count = 0
        try:
            while self.running:
                self.print_cli_status(clear=True)
                self._emit_art_tick()
                time.sleep(self.refresh_s)
                count += 1
                if iterations and count >= iterations:
                    break
        except KeyboardInterrupt:
            print("\nOHM Stopped.")

    def run_daemon_mode(self):
        """Runs headless and emits OHM status to the bus."""
        try:
            while self.running:
                self.tail_task_ledger()
                self.tail_bus()
                self._prune_tasks()
                self._refresh_services()
                self._update_wave()
                task_counts, active_tasks = self._task_counts()
                self.metrics["active_tasks"] = active_tasks
                if self.omega["dispatch"]["pending_total"] < len(active_tasks):
                    self.omega["dispatch"]["pending_total"] = len(active_tasks)
                self._emit_status()
                self._emit_art_tick()
                time.sleep(self.refresh_s)
        except KeyboardInterrupt:
            return

    def run_curses_mode(self):
        """Runs in full curses TUI mode."""
        import curses

        def main(stdscr):
            self.stdscr = stdscr
            self.maximized = False

            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_MAGENTA, -1)
            curses.init_pair(3, curses.COLOR_GREEN, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)
            curses.init_pair(5, curses.COLOR_YELLOW, -1)
            curses.init_pair(6, curses.COLOR_BLUE, -1)
            curses.init_pair(7, curses.COLOR_WHITE, -1)

            stdscr.nodelay(True)
            curses.curs_set(0)

            while self.running:
                stdscr.erase()

                try:
                    c = stdscr.getch()
                    if c == ord('q'):
                        self.running = False
                    elif c == ord('M'):
                        self.maximized = not self.maximized
                    elif c == 1:
                        self.agent_expanded = not self.agent_expanded
                    elif c == ord('t'):
                        self.filter_index = (self.filter_index + 1) % len(self.filters)
                    elif ord('1') <= c < ord('1') + len(self.filters):
                        self.filter_index = c - ord('1')
                    elif c == 27:
                        self.maximized = False
                except Exception:
                    pass

                self.tail_task_ledger()
                self.tail_bus()
                self._prune_tasks()
                self._refresh_services()
                self._update_wave()
                task_counts, active_tasks = self._task_counts()
                self.metrics["active_tasks"] = active_tasks
                if self.omega["dispatch"]["pending_total"] < len(active_tasks):
                    self.omega["dispatch"]["pending_total"] = len(active_tasks)

                self._emit_art_tick()

                rows, cols = stdscr.getmaxyx()

                stats = self._heartbeat_stats()
                hb = self.omega["heartbeat"]
                guardian = self.omega["guardian"]
                dispatch = self.omega["dispatch"]
                queue = self.omega["queue"]
                pairs = self.omega["pairs"]
                health = self.omega["health"]

                wave = ''.join(self.wave)
                hb_age = self._format_age(hb["last_ts"])
                guardian_age = self._format_age(guardian["last_ts"])
                guardian_warn_age = self._format_age(guardian["last_warn_ts"])
                dispatch_age = self._format_age(dispatch["last_ts"])
                queue_age = self._format_age(queue["last_ts"])
                health_age = self._format_age(health["last_ts"])
                bus_label, bus_warn = self._bus_size_label()

                health_status = str(health["status"]).lower()
                if health_status in {"nominal", "ok", "healthy"}:
                    health_color = curses.color_pair(3)
                elif health_status in {"degraded", "warn", "warning"}:
                    health_color = curses.color_pair(5)
                else:
                    health_color = curses.color_pair(4)

                svc_active, svc_total = self._service_counts()
                if svc_total == 0:
                    svc_color = curses.color_pair(4)
                elif svc_active == svc_total:
                    svc_color = curses.color_pair(3)
                else:
                    svc_color = curses.color_pair(5)

                known_agents = set()
                for meta in AGENT_TAXONOMY.values():
                    known_agents.update(meta.get("agents", []))
                # v1.16: Count "live" agents - those seen within agent_active_s window
                seen_agents = list(self.metrics["agents"])
                curses_now = time.time()
                live_agent_count = 0
                for actor in seen_agents:
                    last_ts = self.metrics["agent_last_seen"].get(actor, 0)
                    if last_ts:
                        # Handle string timestamps (ISO format) or numeric
                        if isinstance(last_ts, str):
                            try:
                                from datetime import datetime
                                dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
                                last_ts = dt.timestamp()
                            except (ValueError, TypeError):
                                continue
                        if (curses_now - float(last_ts)) <= self.agent_active_s:
                            live_agent_count += 1
                task_line = (
                    f"Tasks: run {task_counts.get('in_progress', 0)} blocked {task_counts.get('blocked', 0)} "
                    f"planned {task_counts.get('planned', 0)} done {task_counts.get('completed', 0)} "
                    f"fail {task_counts.get('abandoned', 0)}"
                )
                if task_counts.get("stale", 0):
                    task_line += f" stale {task_counts['stale']}"

                lines = [
                    ("OMEGA HEART MONITOR (v1.16.0)", curses.color_pair(1) | curses.A_BOLD),
                    (f"Heartbeat: {hb['count']} cycle {hb['cycle']} last {hb_age} bpm {stats['bpm']:.1f} avg {stats['avg']:.1f}s jitter {stats['jitter']:.1f}s", curses.color_pair(3)),
                    (f"Guardian: {guardian['count']} warns {guardian['warn']} last {guardian_age} warn {guardian_warn_age}", curses.color_pair(2)),
                    (f"Dispatch: ticks {dispatch['tick']} sent {dispatch['sent']} pend {dispatch['pending_total']} denied {dispatch['denied']} err {dispatch['errors']} last {dispatch_age}", curses.color_pair(5)),
                    (f"Queue: pending {queue['pending_requests']} events {queue['total_events']} last {queue_age}", curses.color_pair(6)),
                    (bus_label, curses.color_pair(4) if bus_warn else curses.color_pair(1)),
                    (f"Pairs: pending {pairs['total']} oldest {pairs['oldest_age_s']:.1f}s | Health: {health['status']} last {health_age}", health_color),
                    (f"Providers: {self._provider_string()}", curses.color_pair(1)),
                    (f"Pulse: {wave}", curses.color_pair(1)),
                    (f"Services: {self._services_summary()}", svc_color),
                    (task_line, curses.color_pair(6)),
                    (f"Agents: {len(self.metrics['agents'])} ({live_agent_count} live) | Tasks: {len(active_tasks)} | Dispatched: {self.metrics['tasks_started']} | Completed: {self.metrics['tasks_completed']}", curses.color_pair(7)),
                ]
                if self.agent_expanded:
                    lines.append(("Agents (expanded)", curses.color_pair(7) | curses.A_BOLD))
                    agent_lines = self._agent_dashboard_lines(
                        max_agents=max(5, rows - len(lines) - 2),
                        show_all=True,
                        width=max(10, cols - 6),
                    )
                    for aline in agent_lines:
                        label = aline.get("label", "")
                        attr = curses.color_pair(7)
                        if aline.get("is_header"):
                            attr |= curses.A_BOLD
                        lines.append((f"  {label}", attr))
                    op_lines = self._operator_lines(width=max(10, cols - 6))
                    if op_lines:
                        lines.append(("Operators", curses.color_pair(2) | curses.A_BOLD))
                        for line in op_lines:
                            lines.append((f"  {line}", curses.color_pair(7)))
                if self.maximized:
                    for svc in self._service_detail_lines(width=max(10, cols - 6)):
                        active = svc.get("active", "")
                        if active == "active":
                            color = curses.color_pair(3)
                        elif active in {"inactive", "failed", "deactivating"}:
                            color = curses.color_pair(4)
                        else:
                            color = curses.color_pair(5)
                        lines.append((f"  {svc['label']}", color))

                try:
                    info_height = min(rows - 2, len(lines)) if not self.maximized else rows - 2
                    stdscr.addstr(0, 2, f" {lines[0][0]} ", lines[0][1])
                    y = 1
                    for line, color in lines[1:info_height]:
                        if y >= rows - 1:
                            break
                        line = line[:cols - 4] + '..' if len(line) > cols - 4 else line
                        stdscr.addstr(y, 2, line, color)
                        y += 1

                    # Log lines
                    y = info_height
                    if y < rows - 1:
                        x = 2
                        label = "Filters: "
                        stdscr.addstr(y, x, label, curses.color_pair(1) | curses.A_BOLD)
                        x += len(label)
                        for idx, (fname, _) in enumerate(self.filters):
                            tab = f"[{fname}] "
                            attr = curses.color_pair(2 if idx == self.filter_index else 7)
                            if idx == self.filter_index:
                                attr |= curses.A_BOLD | curses.A_REVERSE
                            if x + len(tab) >= cols - 2:
                                break
                            stdscr.addstr(y, x, tab, attr)
                            x += len(tab)
                        y += 1
                    max_logs = max(1, rows - y - 1)
                    entries = self._filtered_log_entries(limit=max_logs)
                    if not entries and y < rows - 1:
                        stdscr.addstr(y, 2, "no events for filter", curses.color_pair(7))
                        y += 1
                    for entry in entries:
                        if y >= rows - 1:
                            break
                        line = entry.get("summary", "")
                        line = line[:cols - 4] + '..' if len(line) > cols - 4 else line
                        level = str(entry.get("level", "")).lower()
                        topic = entry.get("topic", "")
                        if level == "error":
                            color = curses.color_pair(4)
                        elif level == "warn":
                            color = curses.color_pair(5)
                        elif topic.startswith("omega."):
                            color = curses.color_pair(1)
                        elif topic.startswith("qa."):
                            color = curses.color_pair(2)
                        elif topic.startswith("rd.tasks.") or topic.startswith("task."):
                            color = curses.color_pair(3)
                        elif "dispatch" in topic:
                            color = curses.color_pair(5)
                        elif "response" in topic:
                            color = curses.color_pair(6)
                        else:
                            color = curses.color_pair(7)
                        stdscr.addstr(y, 2, line, color)
                        y += 1
                except Exception:
                    pass

                stdscr.refresh()
                time.sleep(self.refresh_s)

        curses.wrapper(main)


# v1.13: Bootstrap status helpers (agent normalization compliance)
def check_bootstrap_compliance() -> dict:
    """Check if agent wrappers are properly configured per bootstrap doctor."""
    import shutil
    from pathlib import Path

    wrappers = ["bus-claude", "bus-codex", "bus-gemini", "bus-qwen"]
    tools_dir = Path(__file__).parent

    status = {"compliant": 0, "missing": 0, "agents": {}}
    for wrapper in wrappers:
        path = tools_dir / wrapper
        agent = wrapper.replace("bus-", "")
        if path.exists():
            content = path.read_text(errors="replace")[:2000]
            uses_common = "agent_wrapper_common.sh" in content
            status["agents"][agent] = {"exists": True, "uses_common": uses_common}
            if uses_common:
                status["compliant"] += 1
            else:
                status["missing"] += 1
        else:
            status["agents"][agent] = {"exists": False, "uses_common": False}
            status["missing"] += 1

    return status


# =============================================================================
# v1.14: AGENT TAXONOMY AND SEMANTIC OPERATOR GRAMMAR
# =============================================================================

AGENT_TAXONOMY = {
    "infrastructure": {
        "glyph": "⚙",
        "ring": 0,
        "agents": ["omega", "omega_guardian", "bus-mirror", "bus-mirror-reverse", "world-router"],
        "color": "cyan",
    },
    "models": {
        "glyph": "🧠",
        "ring": 1,
        "agents": ["claude", "codex", "gemini", "chatgpt", "qwen", "claude-opus", "claude-codex", "claude-reviewer", "claude-hygiene"],
        "color": "magenta",
    },
    "qa": {
        "glyph": "✓",
        "ring": 2,
        "agents": ["qa-live-checker", "smoketest", "ui-benchmark", "ui-benchmark-full"],
        "color": "green",
    },
    "dialogos": {
        "glyph": "💬",
        "ring": 2,
        "agents": ["dialogosd", "dialogos-indexer", "dialogos-search"],
        "color": "yellow",
    },
    "dashboard": {
        "glyph": "📊",
        "ring": 2,
        "agents": ["dashboard", "dashboard-telemetry", "dashboard-user", "dashboard/crush", "load-timing"],
        "color": "blue",
    },
    "browser": {
        "glyph": "🌐",
        "ring": 3,
        "agents": ["browser_daemon", "Antigravity"],
        "color": "orange",
    },
    "ephemeral": {
        "glyph": "⚡",
        "ring": 3,
        "agents": ["root", "unknown", "test", "test-user", "test-actor", "test_claude"],
        "color": "dim",
    },
}

SEMANTIC_OPERATORS = {
    "PLURIBUS": {"glyph": "Ω", "ring": 0, "desc": "Kernel-level modality"},
    "PBLOCK": {"glyph": "⛔", "ring": 0, "desc": "Milestone freeze"},
    "PBDEEP": {"glyph": "🔬", "ring": 0, "desc": "Deep forensic audit"},
    "OITERATE": {"glyph": "∞", "ring": 1, "desc": "Omega-loop with Büchi liveness"},
    "CKIN": {"glyph": "📥", "ring": 1, "desc": "Check-in status"},
    "DKIN": {"glyph": "📋", "ring": 1, "desc": "Dashboard status"},
    "REALAGENTS": {"glyph": "🤖", "ring": 1, "desc": "Deep implementation dispatch"},
    "PBPAIR": {"glyph": "👥", "ring": 1, "desc": "Paired model consultation"},
    "PBLANES": {"glyph": "🛤", "ring": 1, "desc": "Multi-agent lane coordination"},
    "ITERATE": {"glyph": "↻", "ring": 2, "desc": "Non-blocking iteration"},
    "PBEPORT": {"glyph": "📡", "ring": 2, "desc": "Compact bus liveness"},
    "PBTEST": {"glyph": "🧪", "ring": 2, "desc": "Neurosymbolic TDD"},
    "PBCLITEST": {"glyph": "⌨", "ring": 2, "desc": "CLI verification"},
    "CRUSH": {"glyph": "💥", "ring": 2, "desc": "CLI LLM integration"},
    "BEAM": {"glyph": "✏", "ring": 2, "desc": "Discourse ledger append"},
}


def categorize_agent(actor: str) -> tuple:
    """Return (category, glyph, ring, color) for an actor."""
    for cat, meta in AGENT_TAXONOMY.items():
        if actor in meta["agents"]:
            return (cat, meta["glyph"], meta["ring"], meta["color"])
    return ("unknown", "?", 3, "dim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omega Heart Monitor - Pluribus Observability")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no curses)")
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    parser.add_argument("--iterations", type=int, help="Run for N iterations then exit")
    parser.add_argument("--filter", help="Initial filter tab (name or 1-based index)")
    parser.add_argument("--split", action="store_true", help="Launch tmux split view with OHM on the left")
    parser.add_argument("--session", help="tmux session to attach in right pane (optional)")
    parser.add_argument("--interval", type=float, help="Refresh interval seconds (default: OHM_REFRESH_S or 5)")
    parser.add_argument("--agents-full", action="store_true", help="Show full agent taxonomy in CLI output")
    parser.add_argument("--show-stale-tasks", action="store_true", help="Show stale tasks in ACTIVE TASKS")
    parser.add_argument("--task-stale-s", type=float, help="Seconds before active tasks are marked stale")
    parser.add_argument("--daemon", action="store_true", help="Run headless and emit OHM status to the bus")
    parser.add_argument("--emit-interval", type=float, help="Seconds between OHM status emissions")
    parser.add_argument("--emit-topic", help="Bus topic for OHM status emissions")
    parser.add_argument("--emit-alerts", action="store_true", help="Emit OHM alert events on warnings")
    args = parser.parse_args()

    ohm = OmegaHeartMonitor(filter_name=args.filter)
    if args.interval:
        ohm.refresh_s = max(0.5, args.interval)
    if args.task_stale_s is not None:
        ohm.task_stale_s = args.task_stale_s
    if args.agents_full:
        ohm.agent_expanded = True
    if args.show_stale_tasks:
        ohm.show_stale_tasks = True
    if args.emit_interval:
        ohm.emit_interval_s = max(1.0, float(args.emit_interval))
    if args.emit_topic:
        ohm.emit_topic = args.emit_topic
    if args.emit_alerts:
        ohm.emit_alerts = True

    if args.split:
        sys.exit(ohm.run_tmux_split(session_name=args.session, filter_name=args.filter))

    if args.daemon:
        sys.exit(ohm.run_daemon_mode())

    # Auto-detect: use CLI mode if not in a TTY
    if args.cli or args.once or not sys.stdout.isatty():
        if args.once:
            ohm.tail_bus()
            ohm.print_cli_status(clear=False)
        else:
            ohm.run_cli_mode(iterations=args.iterations)
    else:
        try:
            ohm.run_curses_mode()
        except Exception as e:
            print(f"Curses mode failed ({e}), falling back to CLI mode...")
            ohm.run_cli_mode()
