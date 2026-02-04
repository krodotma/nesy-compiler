#!/usr/bin/env python3
from __future__ import annotations

"""
Gymnist: Gemini-like UX shim over Pluribus
=========================================

Gymnist is a small CLI that provides a Gemini-CLI-like *one-shot* UX without coupling
the user experience to any single vendor CLI quota surface.

Design:
  - Publishes a `dialogos.submit` request to the Pluribus bus.
  - Uses Lens/Collimator (+ VPS control plane state) to choose provider intent.
  - Tails `dialogos.cell.*` events correlated by `req_id` to print outputs.
  - Writes an append-only transcript under `.pluribus/index/gymnist/transcripts/`.

Non-goals:
  - Not a general REPL: use `plurichat.py` for interactive chat.
  - Not a quota/ToS bypass and does not automate vendor web UIs.
"""

import argparse
import getpass
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lens_collimator  # noqa: E402


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)


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


def tail_for_req_id(*, events_path: Path, req_id: str, timeout_s: float) -> tuple[bool, list[dict]]:
    t0 = time.time()
    seen: list[dict] = []
    done = False
    last_size = 0
    while time.time() - t0 < timeout_s:
        try:
            size = events_path.stat().st_size
        except Exception:
            size = 0

        if size < last_size:
            last_size = 0

        if size > last_size:
            try:
                with events_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(last_size, os.SEEK_SET)
                    chunk = f.read()
            except Exception:
                chunk = ""
            last_size = size
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                d = e.get("data")
                if not isinstance(d, dict):
                    continue
                if (d.get("req_id") or d.get("request_id")) != req_id:
                    continue
                if str(e.get("topic") or "").startswith("dialogos.cell."):
                    seen.append(e)
                    if e.get("topic") == "dialogos.cell.end":
                        done = True
        if done:
            return True, seen
        time.sleep(0.2)
    return False, seen


def write_transcript(root: Path, *, req_id: str, prompt: str, events: list[dict]) -> Path | None:
    try:
        d = root / ".pluribus" / "index" / "gymnist" / "transcripts"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{req_id}.ndjson"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "prompt", "req_id": req_id, "iso": now_iso_utc(), "text": prompt}, ensure_ascii=False) + "\n")
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return path
    except Exception:
        return None


def format_repl_prompt(*, user_prompt: str, history: list[dict], system: str | None) -> str:
    parts: list[str] = []
    if system and str(system).strip():
        parts.append(f"System: {str(system).strip()}")
    if history:
        parts.append("Conversation:")
        for item in history:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant"
            parts.append(f"{label}: {content}")
    parts.append(f"User: {str(user_prompt).strip()}")
    parts.append("Assistant:")
    return "\n".join(parts).strip() + "\n"


def collect_text_for_req_id(*, events: list[dict], req_id: str) -> str:
    chunks: list[str] = []
    for e in events:
        if e.get("topic") != "dialogos.cell.output":
            continue
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        if (d.get("req_id") or d.get("request_id")) != req_id:
            continue
        content = (d.get("content") if isinstance(d, dict) else None) or ""
        content = str(content).strip()
        if not content:
            continue
        chunks.append(content)
    return "\n".join(chunks)


def append_repl_session_event(
    *,
    root: Path,
    session_id: str,
    req_id: str,
    role: str,
    content: str,
    provider: str,
    ok: bool,
) -> Path:
    base = (root / ".pluribus" / "index" / "gymnist" / "sessions").resolve()
    ensure_dir(base)
    path = base / f"{session_id}.ndjson"
    append_ndjson(
        path,
        {
            "type": "turn",
            "iso": now_iso_utc(),
            "session_id": session_id,
            "req_id": req_id,
            "role": role,
            "content": content,
            "provider": provider,
            "ok": bool(ok),
        },
    )
    return path


def run_dialogos_once(*, bus_dir: Path) -> None:
    tool = Path(__file__).with_name("dialogosd.py")
    if not tool.exists():
        return
    import subprocess

    subprocess.run(
        [sys.executable, str(tool), "--bus-dir", str(bus_dir), "--once"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def ask_once(
    *,
    root: Path,
    bus_dir: Path,
    actor: str,
    provider: str,
    kind: str,
    effects: str,
    require_model_prefix: str | None,
    timeout_s: float,
    process_once: bool,
    goal: str,
    prompt: str,
    mode: str = "llm",
) -> tuple[bool, str, str, Path | None, str]:
    req_id = str(uuid.uuid4())
    lr = lens_collimator.LensRequest(
        req_id=req_id,
        goal=str(goal),
        kind=str(kind),
        effects=str(effects),
        prefer_providers=[str(provider)],
        require_model_prefix=str(require_model_prefix) if require_model_prefix else None,
    )
    session = lens_collimator.load_vps_session(root)
    plan = lens_collimator.plan_route(lr, session=session)

    emit_bus(
        bus_dir,
        topic="dialogos.submit",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "mode": mode,
            "providers": [plan.provider],
            "prompt": str(prompt),
            "lens": {
                "depth": plan.depth,
                "lane": plan.lane,
                "context_mode": plan.context_mode,
                "notes": plan.notes,
            },
        },
    )

    if process_once:
        run_dialogos_once(bus_dir=bus_dir)

    ok, events = tail_for_req_id(events_path=bus_dir / "events.ndjson", req_id=req_id, timeout_s=float(timeout_s))
    for e in events:
        if e.get("topic") != "dialogos.cell.output":
            continue
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        content = (d.get("content") if isinstance(d, dict) else None) or ""
        sys.stdout.write(str(content).rstrip() + "\n")

    transcript = write_transcript(root, req_id=req_id, prompt=str(prompt), events=events)
    emit_bus(
        bus_dir,
        topic="gymnist.ask.complete",
        kind="metric",
        level="info" if ok else "warn",
        actor=actor,
        data={"req_id": req_id, "ok": ok, "provider": plan.provider, "transcript": str(transcript) if transcript else None},
    )
    text = collect_text_for_req_id(events=events, req_id=req_id)
    return ok, req_id, plan.provider, transcript, text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gymnist.py",
        description="Gymnist: Gemini-like CLI that routes via Pluribus control plane + Dialogos (no quota/ToS bypass).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="Send one prompt via dialogos.submit and print streamed outputs.")
    ask.add_argument("--bus-dir", default=None)
    ask.add_argument("--root", default=None)
    ask.add_argument("--actor", default=None)
    ask.add_argument("--provider", default="auto", help="auto|codex-cli|gemini|vertex-gemini|vertex-gemini-curl|claude-cli|mock")
    ask.add_argument("--kind", default="other")
    ask.add_argument("--effects", default="unknown")
    ask.add_argument("--require-model-prefix", default=None, help="e.g. gemini-3")
    ask.add_argument("--timeout-s", default="120")
    ask.add_argument("--process-once", action="store_true", help="Run dialogosd --once after publishing (may invoke providers).")
    ask.add_argument("--prompt", required=True)
    ask.add_argument("--mode", default="llm", help="Dialogos mode (default: llm).")

    repl = sub.add_parser("repl", help="Interactive REPL using dialogos.submit (append-only session transcript).")
    repl.add_argument("--bus-dir", default=None)
    repl.add_argument("--root", default=None)
    repl.add_argument("--actor", default=None)
    repl.add_argument("--provider", default="auto", help="auto|codex-cli|gemini|vertex-gemini|vertex-gemini-curl|claude-cli|mock")
    repl.add_argument("--kind", default="other")
    repl.add_argument("--effects", default="unknown")
    repl.add_argument("--require-model-prefix", default=None, help="e.g. gemini-3")
    repl.add_argument("--timeout-s", default="120")
    repl.add_argument("--process-once", action="store_true", help="Run dialogosd --once after publishing (may invoke providers).")
    repl.add_argument("--session-id", default=None, help="Session id for session transcript (default: uuid4)")
    repl.add_argument("--history-turns", default="6", help="Turns of history to include (default: 6)")
    repl.add_argument("--system", default=None, help="Optional system instruction to prepend.")
    repl.add_argument("--mode", default="llm", help="Dialogos mode (default: llm).")

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "gymnist"

    root = Path(args.root).expanduser().resolve() if args.root else lens_collimator.find_pluribus_root(Path.cwd())
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()
    ensure_dir(bus_dir)
    (bus_dir / "events.ndjson").touch(exist_ok=True)

    if args.cmd == "ask":
        ok, _req_id, _provider, _transcript, _text = ask_once(
            root=root,
            bus_dir=bus_dir,
            actor=actor,
            provider=str(args.provider),
            kind=str(args.kind),
            effects=str(args.effects),
            require_model_prefix=str(args.require_model_prefix) if args.require_model_prefix else None,
            timeout_s=float(args.timeout_s),
            process_once=bool(args.process_once),
            goal=str(args.prompt),
            prompt=str(args.prompt),
            mode=str(args.mode),
        )
        return 0 if ok else 2

    if args.cmd == "repl":
        session_id = str(args.session_id).strip() if args.session_id else str(uuid.uuid4())
        try:
            history_turns = int(str(args.history_turns))
        except Exception:
            history_turns = 6
        if history_turns < 0:
            history_turns = 0

        history: list[dict] = []
        print("# Gymnist REPL (/exit to quit, /reset to clear history)")
        while True:
            try:
                line = input("> ")
            except EOFError:
                print("")
                break
            except KeyboardInterrupt:
                print("")
                continue

            line = (line or "").strip()
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                break
            if line == "/reset":
                history = []
                print("# history cleared")
                continue

            hist = history[-(2 * history_turns):] if history_turns else []
            effective_prompt = format_repl_prompt(user_prompt=line, history=hist, system=args.system)
            ok, req_id, chosen_provider, _transcript, text = ask_once(
                root=root,
                bus_dir=bus_dir,
                actor=actor,
                provider=str(args.provider),
                kind=str(args.kind),
                effects=str(args.effects),
                require_model_prefix=str(args.require_model_prefix) if args.require_model_prefix else None,
                timeout_s=float(args.timeout_s),
                process_once=bool(args.process_once),
                goal=str(line),
                prompt=str(effective_prompt),
                mode=str(args.mode),
            )

            append_repl_session_event(
                root=root,
                session_id=session_id,
                req_id=req_id,
                role="user",
                content=line,
                provider=str(chosen_provider),
                ok=ok,
            )
            append_repl_session_event(
                root=root,
                session_id=session_id,
                req_id=req_id,
                role="assistant",
                content=text,
                provider=str(chosen_provider),
                ok=ok,
            )

            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": text})

        emit_bus(
            bus_dir,
            topic="gymnist.repl.complete",
            kind="metric",
            level="info",
            actor=actor,
            data={"session_id": session_id, "history_turns": history_turns},
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
