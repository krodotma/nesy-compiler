#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def tail_lines(path: Path, n: int = 200) -> list[str]:
    if n <= 0 or not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        block = 4096
        data = b""
        pos = end
        while pos > 0 and data.count(b"\n") <= n:
            step = block if pos >= block else pos
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
        lines = data.splitlines()[-n:]
    out: list[str] = []
    for b in lines:
        try:
            out.append(b.decode("utf-8", errors="replace"))
        except Exception:
            continue
    return out


def classify_failure_text(text: str) -> str:
    t = (text or "").lower()
    if not t.strip():
        return "error"
    if "please run /login" in t or "run /login" in t or "invalid api key" in t:
        return "blocked_auth"
    if "resource_exhausted" in t or "quota exceeded" in t or "http error: 429" in t:
        return "blocked_quota"
    if "no provider configured" in t or "missing api key" in t or "set gemini_api_key" in t or "set anthropic_api_key" in t:
        return "blocked_config"
    if "unsupported provider" in t:
        return "blocked_config"
    return "error"


@dataclass(frozen=True)
class Req:
    req_id: str
    ts: float
    iso: str
    goal: str


@dataclass
class RespState:
    attempts: int = 0
    success: bool = False
    last_ts: float = 0.0
    last_iso: str = ""
    last_exit_code: int | None = None
    last_provider: str | None = None
    last_classification: str | None = None
    last_stderr: str | None = None


def emit_bus(bus_dir: str, *, actor: str, topic: str, level: str, data: dict) -> None:
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    import subprocess

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
            "metric",
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


def compute_summary(*, root: Path, bus_dir: Path | None, stuck_after_s: float) -> dict:
    idx = root / ".pluribus" / "index"
    requests_path = idx / "requests.ndjson"
    responses_path = idx / "responses.ndjson"

    reqs: list[Req] = []
    for obj in iter_ndjson(requests_path):
        rid = obj.get("req_id")
        if not isinstance(rid, str) or not rid:
            continue
        ts = float(obj.get("ts") or 0.0)
        iso = str(obj.get("iso") or "")
        goal = str(obj.get("goal") or "")
        reqs.append(Req(req_id=rid, ts=ts, iso=iso, goal=goal))

    states: dict[str, RespState] = {}
    for obj in iter_ndjson(responses_path):
        rid = obj.get("req_id")
        if not isinstance(rid, str) or not rid:
            continue
        st = states.setdefault(rid, RespState())
        st.attempts += 1
        ts = float(obj.get("ts") or 0.0)
        if ts >= st.last_ts:
            st.last_ts = ts
            st.last_iso = str(obj.get("iso") or "")
            try:
                st.last_exit_code = int(obj.get("exit_code"))
            except Exception:
                st.last_exit_code = None
            st.last_provider = obj.get("provider") if isinstance(obj.get("provider"), str) else None
            stderr = obj.get("stderr")
            out = obj.get("output")
            msg = "\n".join([str(stderr or ""), str(out or "")]).strip()
            st.last_stderr = str(stderr or "") if stderr is not None else None
            st.last_classification = (
                obj.get("classification") if isinstance(obj.get("classification"), str) else classify_failure_text(msg)
            )

        if st.last_exit_code == 0:
            st.success = True

    now = time.time()
    done = 0
    open_ = 0
    blocked = 0
    errors = 0
    stuck = 0

    open_reqs: list[Req] = []
    for r in reqs:
        st = states.get(r.req_id)
        if st and st.success:
            done += 1
            continue
        open_ += 1
        open_reqs.append(r)
        if st and st.last_exit_code not in (None, 0):
            if (st.last_classification or "").startswith("blocked_"):
                blocked += 1
            else:
                errors += 1

        last_progress = st.last_ts if st and st.last_ts > 0 else r.ts
        if last_progress > 0 and (now - last_progress) >= stuck_after_s:
            stuck += 1

    status = "idle"
    if blocked > 0:
        status = "blocked"
    elif errors > 0:
        status = "error"
    elif open_ > 0:
        status = "working"

    next_goal = ""
    if open_reqs:
        open_reqs.sort(key=lambda rr: rr.ts or 0.0)
        next_goal = open_reqs[0].goal

    last_error = None
    if bus_dir:
        events_path = bus_dir / "events.ndjson"
        for line in reversed(tail_lines(events_path, n=250)):
            try:
                ev = json.loads(line)
            except Exception:
                continue
            lvl = str(ev.get("level") or "")
            if lvl.lower() in {"error", "fatal"}:
                last_error = {"topic": ev.get("topic"), "iso": ev.get("iso"), "actor": ev.get("actor")}
                break

    return {
        "ts": time.time(),
        "iso": now_iso_utc(),
        "root": str(root),
        "status": status,
        "counts": {"requests": len(reqs), "done": done, "open": open_, "blocked": blocked, "errors": errors, "stuck": stuck},
        "next": {"goal": next_goal} if next_goal else None,
        "last_error": last_error,
    }


def format_line(summary: dict) -> str:
    c = summary.get("counts") or {}
    status = summary.get("status") or "unknown"
    nxt = summary.get("next") or {}
    goal = str(nxt.get("goal") or "").replace("\n", " ").strip()
    if len(goal) > 120:
        goal = goal[:117] + "..."
    return (
        "STATUSLINE: "
        f"status={status}; "
        f"done={c.get('done',0)}; "
        f"open={c.get('open',0)}; "
        f"blocked={c.get('blocked',0)}; "
        f"errors={c.get('errors',0)}; "
        f"stuck={c.get('stuck',0)}; "
        f'next="{goal or "await_task"}"; '
        "tools=[]; blockers=[]"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="statusline.py", description="Compute a short inter/intra-agent status summary (append-only evidence).")
    ap.add_argument("--root", default=None, help="Rhizome root (default: search upward for .pluribus/rhizome.json).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    ap.add_argument("--stuck-after-s", default="600", help="Seconds without progress to consider 'stuck' (default: 600).")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of a one-line statusline.")
    ap.add_argument("--emit-bus", action="store_true", help="Emit metric event to the bus (topic: agent.statusline).")
    ap.add_argument("--topic", default="agent.statusline")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    bus_path = Path(bus_dir).expanduser().resolve() if bus_dir else None
    stuck_after_s = max(0.0, float(args.stuck_after_s))

    summary = compute_summary(root=root, bus_dir=bus_path, stuck_after_s=stuck_after_s)
    if args.emit_bus and bus_path:
        actor = default_actor()
        emit_bus(str(bus_path), actor=actor, topic=args.topic, level="info", data=summary)

    if args.json:
        sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(format_line(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
