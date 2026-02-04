#!/usr/bin/env python3
"""
Omega Guardian - omega-regular verification loop (Buchi/Rabin/Streett inspired).

Checks:
- Liveness (Buchi): expected events recur within a window.
- Fairness (Streett): request/response pairs clear within max age.
- Safety (Rabin): bad events do not dominate without good recovery signals.
 - Truth/Correctness: malformed bus lines or missing req_id correlations.
 - Verification: periodic pluribus.check.report (optionally auto-triggered).
 - Evolution: stale in_progress/blocked tasks in the task ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def emit_bus(bus_dir: str, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if agent_bus is None:
        return
    try:
        paths = agent_bus.resolve_bus_paths(bus_dir)
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception:
        return


def _normalize_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    out: list[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _resolve_path(path_str: str, *, base: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        return (base / p).resolve()
    return p


def _read_tail_bytes(path: Path, max_bytes: int) -> list[bytes]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            start = max(0, end - max_bytes)
            f.seek(start)
            data = f.read(end - start)
        lines = data.splitlines()
        if start > 0 and lines:
            lines = lines[1:]
        return lines
    except Exception:
        return []


def _parse_ts(obj: dict) -> float:
    ts = obj.get("ts")
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)
    return time.time()


def _extract_req_id(data: dict, keys: list[str]) -> str | None:
    for k in keys:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _read_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1].isdigit():
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1]) * 1024
    except Exception:
        pass
    return info


class OmegaGuardian:
    def __init__(
        self,
        *,
        bus_dir: str,
        rules_path: Path,
        state_path: Path,
        ledger_path: Path,
        interval_s: float,
        warmup_s: float,
        emit_ok: bool,
        once: bool,
    ) -> None:
        self.bus_dir = bus_dir
        self.events_path = Path(bus_dir) / "events.ndjson"
        self.rules_path = rules_path
        self.state_path = state_path
        self.ledger_path = ledger_path
        self.interval_s = max(1.0, interval_s)
        self.warmup_s = max(0.0, warmup_s)
        self.emit_ok = emit_ok
        self.once = once

        self.actor = "omega_guardian"
        self.start_ts = time.time()

        # Streams
        self.bus_cursor = 0
        self.bus_buffer = b""
        self.ledger_cursor = 0
        self.ledger_buffer = b""

        # Rule state
        self.buchi_rules: list[dict] = []
        self.rabin_rules: list[dict] = []
        self.streett_rules: list[dict] = []
        self.error_rules: list[dict] = []
        self.resource_rules: list[dict] = []
        self.verification_rules: list[dict] = []
        self.evolution_rules: list[dict] = []
        self.truth_rules: list[dict] = []
        self.correctness_rules: list[dict] = []

        self.rule_last_seen: dict[str, float] = {}
        self.rabin_state: dict[str, dict[str, deque[float]]] = {}
        self.error_state: dict[str, deque[float]] = {}
        self.verification_state: dict[str, dict[str, float]] = {}

        # Pending pairs
        self.pair_defs: list[dict] = []
        self.pending_pairs: dict[str, dict[str, float]] = {}
        self.streett_ignore_pairs: set[str] = set()

        # Truth/correctness counters
        self.bus_parse_errors: deque[float] = deque()
        self.ledger_parse_errors: deque[float] = deque()
        self.missing_req_id: deque[float] = deque()

        # Task ledger state (evolution)
        self.tasks: dict[str, dict[str, Any]] = {}

        # Config settings
        self.bootstrap_bus_tail_bytes = 512 * 1024
        self.task_ledger_max_bootstrap_bytes = 1024 * 1024

        self._load_rules()
        self._load_state()
        self._bootstrap_streams()

    def _load_rules(self) -> None:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"rules not found: {self.rules_path}")
        data = json.loads(self.rules_path.read_text(encoding="utf-8"))
        checks = data.get("checks") or []
        settings = data.get("settings") or {}
        self.bootstrap_bus_tail_bytes = _safe_int(settings.get("bootstrap_bus_tail_bytes"), self.bootstrap_bus_tail_bytes)
        self.task_ledger_max_bootstrap_bytes = _safe_int(settings.get("task_ledger_max_bootstrap_bytes"), self.task_ledger_max_bootstrap_bytes)

        for rule in checks:
            if not isinstance(rule, dict):
                continue
            rtype = str(rule.get("type") or "").strip().lower()
            rid = str(rule.get("id") or "").strip()
            if not rid or not rtype:
                continue
            rule["id"] = rid
            rule["type"] = rtype
            if rtype == "buchi":
                self.buchi_rules.append(rule)
                self.rule_last_seen.setdefault(rid, 0.0)
            elif rtype == "rabin":
                self.rabin_rules.append(rule)
                self.rabin_state.setdefault(rid, {"good": deque(), "bad": deque()})
            elif rtype == "streett_pairs":
                self.streett_rules.append(rule)
                self.streett_ignore_pairs.update(_normalize_list(rule.get("ignore_pairs")))
            elif rtype == "safety_errors":
                self.error_rules.append(rule)
                self.error_state.setdefault(rid, deque())
            elif rtype == "safety_resources":
                self.resource_rules.append(rule)
            elif rtype == "verification_report":
                self.verification_rules.append(rule)
                self.verification_state.setdefault(rid, {"last_report": 0.0, "last_trigger": 0.0})
            elif rtype == "evolution_stall":
                self.evolution_rules.append(rule)
            elif rtype == "truth_parse":
                self.truth_rules.append(rule)
            elif rtype == "correctness_reqid":
                self.correctness_rules.append(rule)

        # Load pair specs for streett rules
        for rule in self.streett_rules:
            spec_path = str(rule.get("pairs_spec") or "")
            if not spec_path:
                continue
            resolved = _resolve_path(spec_path, base=REPO_ROOT)
            if not resolved.exists():
                continue
            try:
                spec = json.loads(resolved.read_text(encoding="utf-8"))
            except Exception:
                continue
            for pair in spec.get("pairs") or []:
                if not isinstance(pair, dict):
                    continue
                pid = str(pair.get("id") or "").strip()
                req_topics = set(_normalize_list(pair.get("request_topics")))
                resp_topics = set(_normalize_list(pair.get("response_topics")))
                if not pid or not req_topics or not resp_topics:
                    continue
                req_keys = _normalize_list(pair.get("req_id_keys") or ["req_id", "request_id", "id"])
                kinds_req = set(_normalize_list(pair.get("kinds_request") or ["request"]))
                kinds_resp = set(_normalize_list(pair.get("kinds_response") or ["response"]))
                self.pair_defs.append(
                    {
                        "id": pid,
                        "request_topics": req_topics,
                        "response_topics": resp_topics,
                        "req_id_keys": req_keys,
                        "kinds_request": kinds_req,
                        "kinds_response": kinds_resp,
                    }
                )
                self.pending_pairs.setdefault(pid, {})

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.bus_cursor = _safe_int(data.get("bus_cursor"), self.bus_cursor)
        self.ledger_cursor = _safe_int(data.get("ledger_cursor"), self.ledger_cursor)
        self.rule_last_seen.update({k: _safe_float(v, 0.0) for k, v in (data.get("rule_last_seen") or {}).items()})
        for pid, pending in (data.get("pending_pairs") or {}).items():
            if isinstance(pending, dict):
                self.pending_pairs[pid] = {k: _safe_float(v, 0.0) for k, v in pending.items()}
        for rid, state in (data.get("rabin_state") or {}).items():
            if rid not in self.rabin_state:
                continue
            self.rabin_state[rid]["good"] = deque(_safe_float(x, 0.0) for x in (state.get("good") or []))
            self.rabin_state[rid]["bad"] = deque(_safe_float(x, 0.0) for x in (state.get("bad") or []))
        for rid, items in (data.get("error_state") or {}).items():
            if rid not in self.error_state:
                continue
            self.error_state[rid] = deque(_safe_float(x, 0.0) for x in (items or []))
        for rid, vstate in (data.get("verification_state") or {}).items():
            if rid in self.verification_state and isinstance(vstate, dict):
                self.verification_state[rid]["last_report"] = _safe_float(vstate.get("last_report"), 0.0)
                self.verification_state[rid]["last_trigger"] = _safe_float(vstate.get("last_trigger"), 0.0)

    def _save_state(self) -> None:
        payload = {
            "bus_cursor": self.bus_cursor,
            "ledger_cursor": self.ledger_cursor,
            "rule_last_seen": self.rule_last_seen,
            "pending_pairs": self.pending_pairs,
            "rabin_state": {
                rid: {"good": list(state["good"]), "bad": list(state["bad"])}
                for rid, state in self.rabin_state.items()
            },
            "error_state": {rid: list(items) for rid, items in self.error_state.items()},
            "verification_state": self.verification_state,
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    def _bootstrap_streams(self) -> None:
        # Bootstrap bus cursor and last_seen via tail sampling (avoid full scan).
        if self.events_path.exists() and self.bus_cursor == 0:
            tail = _read_tail_bytes(self.events_path, self.bootstrap_bus_tail_bytes)
            for raw in tail:
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                self._process_bus_event(obj, bootstrap=True)
            try:
                self.bus_cursor = self.events_path.stat().st_size
            except Exception:
                pass

        # Bootstrap task ledger if small enough.
        if self.ledger_path.exists() and self.ledger_cursor == 0:
            try:
                if self.ledger_path.stat().st_size <= self.task_ledger_max_bootstrap_bytes:
                    for line in self.ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        self._process_ledger_event(obj)
                    self.ledger_cursor = self.ledger_path.stat().st_size
                else:
                    self.ledger_cursor = self.ledger_path.stat().st_size
            except Exception:
                pass

    def _read_new_bus_events(self) -> None:
        if not self.events_path.exists():
            return
        try:
            size = self.events_path.stat().st_size
            if self.bus_cursor > size:
                self.bus_cursor = size
        except Exception:
            pass
        try:
            with self.events_path.open("rb") as f:
                f.seek(self.bus_cursor)
                chunk = f.read()
                self.bus_cursor = f.tell()
        except Exception:
            return
        if not chunk:
            return
        data = self.bus_buffer + chunk
        lines = data.split(b"\n")
        if data and not data.endswith(b"\n"):
            self.bus_buffer = lines.pop()
        else:
            self.bus_buffer = b""
        for raw in lines:
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                self.bus_parse_errors.append(time.time())
                continue
            self._process_bus_event(obj, bootstrap=False)

    def _read_new_ledger_events(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            size = self.ledger_path.stat().st_size
            if self.ledger_cursor > size:
                self.ledger_cursor = size
        except Exception:
            pass
        try:
            with self.ledger_path.open("rb") as f:
                f.seek(self.ledger_cursor)
                chunk = f.read()
                self.ledger_cursor = f.tell()
        except Exception:
            return
        if not chunk:
            return
        data = self.ledger_buffer + chunk
        lines = data.split(b"\n")
        if data and not data.endswith(b"\n"):
            self.ledger_buffer = lines.pop()
        else:
            self.ledger_buffer = b""
        for raw in lines:
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                self.ledger_parse_errors.append(time.time())
                continue
            self._process_ledger_event(obj)

    def _actor_allowed(self, rule: dict, actor: str) -> bool:
        include = set(_normalize_list(rule.get("include_actors")))
        exclude = set(_normalize_list(rule.get("exclude_actors")))
        if include and actor not in include:
            return False
        if exclude and actor in exclude:
            return False
        return True

    def _process_bus_event(self, obj: dict, *, bootstrap: bool) -> None:
        topic = str(obj.get("topic") or "")
        if not topic:
            return
        actor = str(obj.get("actor") or "")
        level = str(obj.get("level") or "")
        kind = str(obj.get("kind") or "")
        ts = _parse_ts(obj)
        data = obj.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        # Buchi last-seen updates
        for rule in self.buchi_rules:
            rid = rule["id"]
            topics = set(_normalize_list(rule.get("topics")))
            if "*" not in topics and topic not in topics:
                continue
            if not self._actor_allowed(rule, actor):
                continue
            self.rule_last_seen[rid] = ts

        # Rabin good/bad windows
        for rule in self.rabin_rules:
            rid = rule["id"]
            if not self._actor_allowed(rule, actor):
                continue
            good_topics = set(_normalize_list(rule.get("good_topics")))
            bad_topics = set(_normalize_list(rule.get("bad_topics")))
            good_levels = set(_normalize_list(rule.get("good_levels")))
            bad_levels = set(_normalize_list(rule.get("bad_levels")))
            if topic in good_topics or (good_levels and level in good_levels):
                self.rabin_state[rid]["good"].append(ts)
            if topic in bad_topics or (bad_levels and level in bad_levels):
                self.rabin_state[rid]["bad"].append(ts)

        # Safety error windows
        for rule in self.error_rules:
            rid = rule["id"]
            if not self._actor_allowed(rule, actor):
                continue
            levels = set(_normalize_list(rule.get("levels")))
            topics = set(_normalize_list(rule.get("topics")))
            if (levels and level in levels) or (topics and topic in topics):
                self.error_state[rid].append(ts)

        # Verification report tracking
        for rule in self.verification_rules:
            rid = rule["id"]
            if topic == str(rule.get("topic") or ""):
                self.verification_state[rid]["last_report"] = ts

        # Streett-style pending pairs
        if not bootstrap and self.pair_defs:
            for pair in self.pair_defs:
                pid = pair["id"]
                if pid in self.streett_ignore_pairs:
                    continue
                if topic in pair["request_topics"] and kind in pair["kinds_request"]:
                    rid = _extract_req_id(data, pair["req_id_keys"])
                    if rid:
                        if rid not in self.pending_pairs[pid]:
                            self.pending_pairs[pid][rid] = ts
                    else:
                        self.missing_req_id.append(time.time())
                elif topic in pair["response_topics"] and kind in pair["kinds_response"]:
                    rid = _extract_req_id(data, pair["req_id_keys"])
                    if rid:
                        self.pending_pairs[pid].pop(rid, None)
                    else:
                        self.missing_req_id.append(time.time())

    def _process_ledger_event(self, obj: dict) -> None:
        run_id = str(obj.get("run_id") or obj.get("id") or "").strip()
        if not run_id:
            return
        status = str(obj.get("status") or "").strip()
        ts = _safe_float(obj.get("ts"), time.time())
        self.tasks[run_id] = {
            "status": status,
            "ts": ts,
            "req_id": obj.get("req_id"),
            "actor": obj.get("actor"),
            "topic": obj.get("topic"),
            "meta": obj.get("meta") if isinstance(obj.get("meta"), dict) else {},
        }

    def _prune_deque(self, dq: deque[float], *, window_s: float, now: float) -> None:
        cutoff = now - window_s
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _emit_trigger(self, rule: dict, now: float) -> None:
        topic = str(rule.get("autotrigger_topic") or "")
        if not topic:
            return
        req_id = str(uuid.uuid4())
        emit_bus(
            self.bus_dir,
            topic=topic,
            kind="request",
            level="warn",
            actor=self.actor,
            data={
                "req_id": req_id,
                "message": f"omega_guardian auto-trigger: {rule.get('id')}",
                "ts": now,
                "iso": now_iso(),
                "check_id": rule.get("id"),
            },
        )

    def _evaluate_buchi(self, rule: dict, now: float) -> tuple[str, dict]:
        rid = rule["id"]
        max_silence = _safe_float(rule.get("max_silence_s"), 60.0)
        last_seen = self.rule_last_seen.get(rid, 0.0)
        age = now - last_seen if last_seen else float("inf")
        status = "ok" if age <= max_silence else "warn"
        details = {"last_seen_age_s": round(age, 2), "max_silence_s": max_silence}
        return status, details

    def _evaluate_rabin(self, rule: dict, now: float) -> tuple[str, dict]:
        rid = rule["id"]
        window_s = _safe_float(rule.get("window_s"), 300.0)
        bad_threshold = _safe_int(rule.get("bad_threshold"), 1)
        state = self.rabin_state.get(rid, {"good": deque(), "bad": deque()})
        self._prune_deque(state["good"], window_s=window_s, now=now)
        self._prune_deque(state["bad"], window_s=window_s, now=now)
        good_count = len(state["good"])
        bad_count = len(state["bad"])
        status = "ok"
        if bad_count >= bad_threshold and good_count == 0:
            status = "warn"
        details = {"window_s": window_s, "good_count": good_count, "bad_count": bad_count}
        return status, details

    def _evaluate_streett_pairs(self, rule: dict, now: float) -> tuple[str, dict]:
        max_age = _safe_float(rule.get("max_age_s"), 600.0)
        by_pair: dict[str, dict] = {}
        status = "ok"
        for pid, pending in self.pending_pairs.items():
            if pid in self.streett_ignore_pairs:
                continue
            ages = [max(0.0, now - ts) for ts in pending.values() if isinstance(ts, (int, float))]
            oldest = max(ages) if ages else 0.0
            count = len(pending)
            by_pair[pid] = {"pending": count, "oldest_age_s": round(oldest, 2)}
            if count > 0 and oldest > max_age:
                status = "warn"
        details = {"max_age_s": max_age, "by_pair": by_pair}
        return status, details

    def _evaluate_errors(self, rule: dict, now: float) -> tuple[str, dict]:
        rid = rule["id"]
        window_s = _safe_float(rule.get("window_s"), 300.0)
        max_count = _safe_int(rule.get("max_count"), 1)
        dq = self.error_state.get(rid, deque())
        self._prune_deque(dq, window_s=window_s, now=now)
        count = len(dq)
        status = "ok" if count <= max_count else "warn"
        details = {"window_s": window_s, "max_count": max_count, "count": count}
        return status, details

    def _evaluate_resources(self, rule: dict, now: float) -> tuple[str, dict]:
        min_disk_gb = _safe_float(rule.get("min_disk_free_gb"), 5.0)
        max_mem_pct = _safe_float(rule.get("max_mem_used_pct"), 92.0)
        max_load_per_cpu = _safe_float(rule.get("max_load1_per_cpu"), 2.0)
        max_load1 = _safe_float(rule.get("max_load1"), 0.0)

        # Disk
        try:
            usage = shutil.disk_usage("/")
            disk_free_gb = usage.free / (1024 ** 3)
        except Exception:
            disk_free_gb = 0.0

        # Memory
        meminfo = _read_meminfo()
        total = float(meminfo.get("MemTotal", 0))
        avail = float(meminfo.get("MemAvailable", 0))
        mem_used_pct = 0.0
        if total > 0:
            mem_used_pct = ((total - avail) / total) * 100.0

        # Load
        try:
            load1 = os.getloadavg()[0]
        except Exception:
            load1 = 0.0
        cpu_count = os.cpu_count() or 1
        load_per_cpu = load1 / float(cpu_count)

        status = "ok"
        if disk_free_gb < min_disk_gb or mem_used_pct > max_mem_pct:
            status = "warn"
        if max_load1 > 0 and load1 > max_load1:
            status = "warn"
        if max_load1 <= 0 and load_per_cpu > max_load_per_cpu:
            status = "warn"

        details = {
            "disk_free_gb": round(disk_free_gb, 2),
            "min_disk_free_gb": min_disk_gb,
            "mem_used_pct": round(mem_used_pct, 2),
            "max_mem_used_pct": max_mem_pct,
            "load1": round(load1, 2),
            "load1_per_cpu": round(load_per_cpu, 2),
            "max_load1_per_cpu": max_load_per_cpu,
            "max_load1": max_load1,
        }
        return status, details

    def _evaluate_verification(self, rule: dict, now: float) -> tuple[str, dict]:
        rid = rule["id"]
        max_silence = _safe_float(rule.get("max_silence_s"), 86400.0)
        autotrigger_interval = _safe_float(rule.get("autotrigger_interval_s"), 0.0)
        state = self.verification_state.get(rid, {"last_report": 0.0, "last_trigger": 0.0})
        last_report = state.get("last_report", 0.0)
        last_trigger = state.get("last_trigger", 0.0)
        age = now - last_report if last_report else float("inf")

        if autotrigger_interval > 0 and (now - last_trigger) > autotrigger_interval:
            self._emit_trigger(rule, now)
            state["last_trigger"] = now
            self.verification_state[rid] = state

        status = "ok" if age <= max_silence else "warn"
        details = {"last_report_age_s": round(age, 2), "max_silence_s": max_silence}
        return status, details

    def _evaluate_truth_parse(self, rule: dict, now: float) -> tuple[str, dict]:
        window_s = _safe_float(rule.get("window_s"), 300.0)
        max_count = _safe_int(rule.get("max_count"), 0)
        self._prune_deque(self.bus_parse_errors, window_s=window_s, now=now)
        self._prune_deque(self.ledger_parse_errors, window_s=window_s, now=now)
        count = len(self.bus_parse_errors) + len(self.ledger_parse_errors)
        status = "ok" if count <= max_count else "warn"
        details = {"window_s": window_s, "max_count": max_count, "count": count}
        return status, details

    def _evaluate_correctness_reqid(self, rule: dict, now: float) -> tuple[str, dict]:
        window_s = _safe_float(rule.get("window_s"), 300.0)
        max_count = _safe_int(rule.get("max_count"), 0)
        self._prune_deque(self.missing_req_id, window_s=window_s, now=now)
        count = len(self.missing_req_id)
        status = "ok" if count <= max_count else "warn"
        details = {"window_s": window_s, "max_count": max_count, "count": count}
        return status, details

    def _evaluate_evolution(self, rule: dict, now: float) -> tuple[str, dict]:
        max_idle = _safe_float(rule.get("max_idle_s"), 6 * 3600.0)
        max_report = _safe_int(rule.get("max_report"), 10)
        stale: list[tuple[float, str, dict]] = []
        for run_id, info in self.tasks.items():
            status = str(info.get("status") or "")
            if status not in {"in_progress", "blocked"}:
                continue
            age = now - float(info.get("ts") or 0.0)
            if age > max_idle:
                stale.append((age, run_id, info))
        stale.sort(reverse=True, key=lambda x: x[0])
        details = {
            "max_idle_s": max_idle,
            "stale_count": len(stale),
            "stale_samples": [
                {
                    "run_id": rid,
                    "age_s": round(age, 2),
                    "req_id": info.get("req_id"),
                    "actor": info.get("actor"),
                    "topic": info.get("topic"),
                }
                for age, rid, info in stale[:max_report]
            ],
        }
        status = "ok" if not stale else "warn"
        return status, details

    def run_cycle(self, cycle: int) -> None:
        now = time.time()
        self._read_new_bus_events()
        self._read_new_ledger_events()

        results: list[dict] = []
        warn_results: list[dict] = []

        if (now - self.start_ts) >= self.warmup_s:
            for rule in self.buchi_rules:
                status, details = self._evaluate_buchi(rule, now)
                results.append({**rule, "status": status, "details": details})
            for rule in self.rabin_rules:
                status, details = self._evaluate_rabin(rule, now)
                results.append({**rule, "status": status, "details": details})
            for rule in self.streett_rules:
                status, details = self._evaluate_streett_pairs(rule, now)
                results.append({**rule, "status": status, "details": details})
            for rule in self.error_rules:
                status, details = self._evaluate_errors(rule, now)
                results.append({**rule, "status": status, "details": details})
            for rule in self.resource_rules:
                status, details = self._evaluate_resources(rule, now)
                results.append({**rule, "status": status, "details": details})
            for rule in self.verification_rules:
                status, details = self._evaluate_verification(rule, now)
                results.append({**rule, "status": status, "details": details})
            for rule in self.truth_rules:
                status, details = self._evaluate_truth_parse(rule, now)
                results.append({**rule, "status": status, "details": details})
            for rule in self.correctness_rules:
                status, details = self._evaluate_correctness_reqid(rule, now)
                results.append({**rule, "status": status, "details": details})
            for rule in self.evolution_rules:
                status, details = self._evaluate_evolution(rule, now)
                results.append({**rule, "status": status, "details": details})

        for r in results:
            if r.get("status") == "warn":
                warn_results.append(r)

        summary = {
            "cycle": cycle,
            "iso": now_iso(),
            "results_total": len(results),
            "results_warn": len(warn_results),
        }
        emit_bus(self.bus_dir, topic="omega.guardian.cycle", kind="metric", level="info", actor=self.actor, data=summary)

        if self.emit_ok and results:
            emit_bus(self.bus_dir, topic="omega.guardian.ok", kind="log", level="info", actor=self.actor, data={"cycle": cycle, "results": results})

        for r in warn_results:
            emit_bus(
                self.bus_dir,
                topic="omega.guardian.warn",
                kind="log",
                level="warn",
                actor=self.actor,
                data={
                    "cycle": cycle,
                    "check_id": r.get("id"),
                    "category": r.get("category"),
                    "type": r.get("type"),
                    "details": r.get("details"),
                },
            )

        self._save_state()

    def run(self) -> int:
        cycle = 0
        while True:
            cycle += 1
            self.run_cycle(cycle)
            if self.once:
                break
            time.sleep(self.interval_s)
        return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Omega Guardian (omega-regular verification)")
    ap.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    ap.add_argument("--rules", default="/pluribus/nucleus/specs/omega_guardian_rules.json")
    ap.add_argument("--state-path", default="/pluribus/.pluribus/index/omega_guardian_state.json")
    ap.add_argument("--ledger-path", default=None)
    ap.add_argument("--interval-s", default=os.environ.get("OMEGA_GUARDIAN_INTERVAL_S", "10"))
    ap.add_argument("--warmup-s", default=os.environ.get("OMEGA_GUARDIAN_WARMUP_S", "30"))
    ap.add_argument("--emit-ok", action="store_true", help="Emit omega.guardian.ok with full results each cycle.")
    ap.add_argument("--once", action="store_true", help="Run a single cycle then exit.")
    return ap.parse_args(argv)


def resolve_ledger_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    # Mirror task_ledger default resolution
    primary = REPO_ROOT / ".pluribus" / "index" / "task_ledger.ndjson"
    fallback = REPO_ROOT / ".pluribus_local" / "index" / "task_ledger.ndjson"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rules_path = Path(args.rules)
    state_path = Path(args.state_path)
    ledger_path = resolve_ledger_path(args.ledger_path)

    guardian = OmegaGuardian(
        bus_dir=str(args.bus_dir),
        rules_path=rules_path,
        state_path=state_path,
        ledger_path=ledger_path,
        interval_s=_safe_float(args.interval_s, 10.0),
        warmup_s=_safe_float(args.warmup_s, 30.0),
        emit_ok=bool(args.emit_ok),
        once=bool(args.once),
    )
    return guardian.run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
