#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

from env_loader import load_pluribus_env


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


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


def run_cmd(argv: list[str], *, env: dict | None = None, timeout_s: int = 3) -> dict:
    try:
        p = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=timeout_s)
        return {
            "ok": p.returncode == 0,
            "exit_code": int(p.returncode),
            "stdout": (p.stdout or "").strip(),
            "stderr": (p.stderr or "").strip(),
        }
    except Exception as e:
        return {"ok": False, "exit_code": None, "stdout": "", "stderr": str(e)}


def parse_node_major(version_text: str) -> int | None:
    s = (version_text or "").strip()
    if s.startswith("v"):
        s = s[1:]
    head = s.split(".", 1)[0].strip()
    try:
        return int(head)
    except Exception:
        return None


def main(argv: list[str]) -> int:
    load_pluribus_env()
    ap = argparse.ArgumentParser(prog="mesh_status.py", description="Agent mesh readiness snapshot (Codex/Claude/Gemini + bus + STRp memory).")
    ap.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    ap.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    ap.add_argument("--emit-bus", action="store_true", help="Publish snapshot to the agent bus.")
    ap.add_argument("--timeout", type=int, default=30, help="Per-command timeout seconds.")
    args = ap.parse_args(argv)

    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    idx = root / ".pluribus" / "index"

    home = Path(os.environ.get("HOME") or "~").expanduser()
    node20_dir = home / ".local" / "node20" / "bin"
    node20 = node20_dir / "node"
    node20_present = node20.exists()

    node_path = shutil.which("node") or ""
    node_run = run_cmd([node_path or "node", "-v"], timeout_s=args.timeout) if (node_path or shutil.which("node")) else {"ok": False}
    node_major = parse_node_major(node_run.get("stdout", "")) if node_run.get("ok") else None

    node20_run = run_cmd([str(node20), "-v"], timeout_s=args.timeout) if node20_present else {"ok": False}
    node20_major = parse_node_major(node20_run.get("stdout", "")) if node20_run.get("ok") else None

    gemini_path = shutil.which("gemini")
    gemini_current = run_cmd([gemini_path or "gemini", "--version"], timeout_s=args.timeout) if gemini_path else {"ok": False}
    env_with_node20 = dict(os.environ)
    if node20_present:
        env_with_node20["PATH"] = f"{node20_dir}{os.pathsep}{env_with_node20.get('PATH','')}"
    gemini_with_node20 = run_cmd([gemini_path or "gemini", "--version"], env=env_with_node20, timeout_s=args.timeout) if gemini_path else {"ok": False}

    claude_path = shutil.which("claude")
    codex_path = shutil.which("codex")

    have_gemini_key = bool((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip())
    have_anthropic_key = bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())
    have_openai_key = bool((os.environ.get("OPENAI_API_KEY") or "").strip())

    gemini_cli_usable = bool(gemini_path and (gemini_current.get("ok") or gemini_with_node20.get("ok")))
    gemini_auth_usable = bool(have_gemini_key or gemini_cli_usable)
    claude_ready = bool(claude_path or have_anthropic_key)
    codex_ready = bool(codex_path or have_openai_key)

    # Some environments disallow writes under /root for third-party CLIs; prefer Pluribus-managed HOME overlays.
    agent_homes = {
        "codex": os.environ.get("PLURIBUS_CODEX_HOME") or "/pluribus/.pluribus/agent_homes/codex",
        "claude": os.environ.get("PLURIBUS_CLAUDE_HOME") or "/pluribus/.pluribus/agent_homes/claude",
    }
    agent_home_status = {}
    for name, home_path in agent_homes.items():
        p = Path(home_path)
        agent_home_status[name] = {"path": home_path, "exists": p.exists(), "writable": bool(p.exists() and os.access(p, os.W_OK))}

    missing: list[str] = []
    if not gemini_cli_usable:
        missing.append("gemini-cli")
    if not claude_ready:
        missing.append("claude")
    if not codex_ready:
        missing.append("codex")

    missing_auth: list[str] = []
    if not gemini_auth_usable:
        missing_auth.append("gemini-api-key-or-gemini-cli")
    if not (have_anthropic_key or claude_path):
        missing_auth.append("anthropic-api-key-or-claude-cli")
    if not (have_openai_key or codex_path):
        missing_auth.append("openai-api-key-or-codex-cli")

    snapshot = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "mesh_status",
        "root": str(root),
        "bus_dir": bus_dir,
        "memory": {
            "rag_db": str(idx / "rag.sqlite3"),
            "kg_nodes": str(idx / "kg_nodes.ndjson"),
            "kg_edges": str(idx / "kg_edges.ndjson"),
        },
        "agents": {
            "gemini": {
                "cli_path": gemini_path,
                "cli_version": (gemini_current.get("stdout") or None) if gemini_current.get("ok") else None,
                "cli_version_with_node20": (gemini_with_node20.get("stdout") or None) if gemini_with_node20.get("ok") else None,
                "api_key_set": have_gemini_key,
                "usable": gemini_cli_usable,
                "notes": "Gemini CLI requires Node.js >= 20; prefix PATH with ~/.local/node20/bin if needed.",
            },
            "claude": {
                "cli_path": claude_path,
                "api_key_set": have_anthropic_key,
                "usable": claude_ready,
                "notes": "Claude is usable via ANTHROPIC_API_KEY (API) or the `claude` CLI (web-login).",
            },
            "codex": {
                "cli_path": codex_path,
                "api_key_set": have_openai_key,
                "usable": codex_ready,
                "notes": "Codex is treated as usable if `codex` is on PATH or OPENAI_API_KEY is set.",
            },
        },
        "agent_homes": agent_home_status,
        "network": {},
        "runtime": {
            "node": {"path": node_path or None, "major": node_major, "raw": node_run.get("stdout") if node_run.get("ok") else None},
            "node20": {"path": str(node20) if node20_present else None, "major": node20_major, "raw": node20_run.get("stdout") if node20_run.get("ok") else None},
        },
        "gate": {
            "triad_ready": len(missing) == 0,
            "missing": missing,
            "auth_ready": len(missing_auth) == 0,
            "missing_auth": missing_auth,
        },
    }

    # Network capability (some sandboxes deny socket() entirely).
    net = {"socket_ok": False, "dns_ok": None, "notes": None}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.close()
        net["socket_ok"] = True
    except Exception as e:
        net["socket_ok"] = False
        net["notes"] = f"socket blocked: {type(e).__name__}: {e}"
    snapshot["network"] = net

    if args.emit_bus:
        emit_bus(
            bus_dir,
            topic="mesh.status",
            kind="metric",
            level="info" if (not missing and not missing_auth) else "warn",
            actor=actor,
            data=snapshot,
        )

    sys.stdout.write(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    return 0 if len(missing) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
