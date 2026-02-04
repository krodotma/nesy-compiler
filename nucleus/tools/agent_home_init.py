#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
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


def copy_tree(src: Path, dst: Path) -> tuple[int, list[str]]:
    """
    Best-effort copy that tolerates missing src and preserves metadata when possible.
    Returns: (copied_count, warnings)
    """
    warnings: list[str] = []
    if not src.exists():
        return 0, warnings
    ensure_dir(dst.parent)
    copied = 0
    if dst.exists():
        # Merge copy: copy items that don't exist yet.
        for root, dirs, files in os.walk(src):
            rel = Path(root).relative_to(src)
            out_root = dst / rel
            ensure_dir(out_root)
            for d in dirs:
                ensure_dir(out_root / d)
            for f in files:
                s = Path(root) / f
                t = out_root / f
                if t.exists():
                    continue
                try:
                    shutil.copy2(s, t)
                    copied += 1
                except Exception as e:
                    warnings.append(f"copy2 failed: {s} -> {t}: {e}")
        return copied, warnings

    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as e:
        warnings.append(f"copytree failed: {src} -> {dst}: {e}")
        # Fallback to manual walk
        for root, dirs, files in os.walk(src):
            rel = Path(root).relative_to(src)
            out_root = dst / rel
            ensure_dir(out_root)
            for d in dirs:
                ensure_dir(out_root / d)
            for f in files:
                s = Path(root) / f
                t = out_root / f
                if t.exists():
                    continue
                try:
                    shutil.copy2(s, t)
                    copied += 1
                except Exception as e2:
                    warnings.append(f"copy2 failed: {s} -> {t}: {e2}")
        return copied, warnings

    # Count files copied (best-effort)
    for _ in dst.rglob("*"):
        copied += 1
    return copied, warnings


def select_agent_home_base(root: Path, *, bus_dir: str | None, actor: str) -> Path:
    primary = root / ".pluribus" / "agent_homes"
    fallback = root / ".pluribus_local" / "agent_homes"
    try:
        ensure_dir(primary)
        return primary
    except PermissionError as exc:
        ensure_dir(fallback)
        emit_bus(
            bus_dir,
            topic="agent.home.init.fallback",
            kind="log",
            level="warn",
            actor=actor,
            data={"primary": str(primary), "fallback": str(fallback), "error": str(exc)},
        )
        return fallback
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            ensure_dir(fallback)
            emit_bus(
                bus_dir,
                topic="agent.home.init.fallback",
                kind="log",
                level="warn",
                actor=actor,
                data={"primary": str(primary), "fallback": str(fallback), "error": str(exc)},
            )
            return fallback
        raise


def init_one(*, name: str, src_rel: str, base: Path, bus_dir: str | None, actor: str, copy_state: bool) -> int:
    ts = time.time()
    iso = now_iso_utc()
    dst_home = base / name
    ensure_dir(dst_home)

    sentinel = dst_home / ".pluribus_agent_home_initialized"
    if sentinel.exists():
        emit_bus(bus_dir, topic="agent.home.init.skip", kind="log", level="info", actor=actor, data={"name": name, "home": str(dst_home), "ts": ts, "iso": iso})
        return 0

    src_home = Path("/root") / src_rel
    dst_state = dst_home / src_rel
    copied = 0
    warnings: list[str] = []
    notes: list[str] = []
    if copy_state:
        copied, warnings = copy_tree(src_home, dst_state)
    else:
        ensure_dir(dst_state)
        notes.append("copy_skipped")
    sentinel.write_text(
        json.dumps(
            {"name": name, "src": str(src_home), "dst": str(dst_state), "copied": copied, "warnings": warnings, "notes": notes, "ts": ts, "iso": iso},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    emit_bus(
        bus_dir,
        topic="agent.home.init",
        kind="artifact",
        level="info" if not warnings else "warn",
        actor=actor,
        data={
            "name": name,
            "home": str(dst_home),
            "state_dir": str(dst_state),
            "copied": copied,
            "warnings": warnings,
            "notes": notes,
        },
    )
    return 0 if not warnings else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="agent_home_init.py", description="Initialize writable per-agent HOME directories under /pluribus/.pluribus/agent_homes (for CLIs that cannot write to /root).")
    ap.add_argument("--root", default="/pluribus", help="Pluribus root for bus + .pluribus dir.")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    ap.add_argument("--only", action="append", default=[], help="Limit to: codex|claude|gemini|qwen|grok|antigravity (repeatable).")
    ap.add_argument("--no-copy", action="store_true", help="Skip copying state from /root for faster init.")
    args = ap.parse_args(argv)

    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "unknown"
    root = Path(args.root).expanduser().resolve()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or str(root / ".pluribus" / "bus")
    base = select_agent_home_base(root, bus_dir=bus_dir, actor=actor)

    only = {s.strip().lower() for s in (args.only or []) if s.strip()}
    rc = 0
    if not only or "codex" in only:
        rc = max(rc, init_one(name="codex", src_rel=".codex", base=base, bus_dir=bus_dir, actor=actor, copy_state=not args.no_copy))
    if not only or "claude" in only:
        rc = max(rc, init_one(name="claude", src_rel=".claude", base=base, bus_dir=bus_dir, actor=actor, copy_state=not args.no_copy))
    if not only or "gemini" in only:
        # Gemini CLI typically uses XDG dirs under HOME; just ensure the HOME exists.
        gemini_home = base / "gemini"
        ensure_dir(gemini_home)
        sentinel = gemini_home / ".pluribus_agent_home_initialized"
        if not sentinel.exists():
            sentinel.write_text(json.dumps({"name": "gemini", "home": str(gemini_home), "ts": time.time(), "iso": now_iso_utc()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit_bus(bus_dir, topic="agent.home.init", kind="artifact", level="info", actor=actor, data={"name": "gemini", "home": str(gemini_home), "notes": "HOME/XDG overlay for gemini CLI auth/state"})
    if not only or "qwen" in only:
        rc = max(rc, init_one(name="qwen", src_rel=".qwen", base=base, bus_dir=bus_dir, actor=actor, copy_state=not args.no_copy))
    if not only or "grok" in only:
        grok_home = base / "grok"
        ensure_dir(grok_home)
        sentinel = grok_home / ".pluribus_agent_home_initialized"
        if not sentinel.exists():
            sentinel.write_text(json.dumps({"name": "grok", "home": str(grok_home), "ts": time.time(), "iso": now_iso_utc()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit_bus(bus_dir, topic="agent.home.init", kind="artifact", level="info", actor=actor, data={"name": "grok", "home": str(grok_home), "notes": "HOME/XDG overlay for grok CLI auth/state"})
    if not only or "antigravity" in only:
        # Antigravity (IDE meta-tool) uses workspace-local state
        antigravity_home = base / "antigravity"
        ensure_dir(antigravity_home)
        sentinel = antigravity_home / ".pluribus_agent_home_initialized"
        if not sentinel.exists():
            sentinel.write_text(json.dumps({"name": "antigravity", "home": str(antigravity_home), "ts": time.time(), "iso": now_iso_utc()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            emit_bus(bus_dir, topic="agent.home.init", kind="artifact", level="info", actor=actor, data={"name": "antigravity", "home": str(antigravity_home), "notes": "IDE meta-tool state directory"})
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
