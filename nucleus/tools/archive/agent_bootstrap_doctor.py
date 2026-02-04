#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

try:
    from nucleus.tools import agent_bus
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2] / "nucleus" / "tools"))
    import agent_bus  # type: ignore


PROMPT_FLAGS = ("--append-system-prompt", "--system-prompt", "--system", "--instruction")


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_read(path: Path, limit: int = 4096) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def check_wrapper(path: Path) -> dict:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "executable": exists and os.access(path, os.X_OK),
        "uses_common": "agent_wrapper_common.sh" in safe_read(path),
    }


def run_help(bin_path: str) -> str:
    try:
        proc = subprocess.run(
            [bin_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
    except Exception:
        return ""
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def detect_prompt_flag(help_text: str) -> str | None:
    for flag in PROMPT_FLAGS:
        if flag in help_text:
            return flag
    return None


def check_cli(name: str) -> dict:
    bin_path = shutil.which(name)
    info = {"name": name, "path": bin_path, "present": bool(bin_path), "prompt_flag": None}
    if not bin_path:
        return info
    help_text = run_help(bin_path)
    info["prompt_flag"] = detect_prompt_flag(help_text)
    return info


def resolve_agent_home(env_key: str, default_path: str) -> Path:
    raw = (os.environ.get(env_key) or "").strip()
    return Path(raw) if raw else Path(default_path)


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def check_secrets(paths: list[Path]) -> list[str]:
    found = []
    for path in paths:
        if path_exists(path):
            found.append(str(path))
    return found


def build_report(root: Path, bus_dir: str, actor: str) -> dict:
    bus_paths = agent_bus.resolve_bus_paths(bus_dir)
    wrappers = {
        "claude": check_wrapper(root / "nucleus" / "tools" / "bus-claude"),
        "codex": check_wrapper(root / "nucleus" / "tools" / "bus-codex"),
        "gemini": check_wrapper(root / "nucleus" / "tools" / "bus-gemini"),
        "qwen": check_wrapper(root / "nucleus" / "tools" / "bus-qwen"),
        "grok": check_wrapper(root / "nucleus" / "tools" / "bus-grok"),
    }

    agent_homes = {
        "claude": resolve_agent_home("PLURIBUS_CLAUDE_HOME", "/pluribus/.pluribus/agent_homes/claude"),
        "codex": resolve_agent_home("PLURIBUS_CODEX_HOME", "/pluribus/.pluribus/agent_homes/codex"),
        "gemini": resolve_agent_home("PLURIBUS_GEMINI_HOME", "/pluribus/.pluribus/agent_homes/gemini"),
        "qwen": resolve_agent_home("PLURIBUS_QWEN_HOME", "/pluribus/.pluribus/agent_homes/qwen"),
        "grok": resolve_agent_home("PLURIBUS_GROK_HOME", "/pluribus/.pluribus/agent_homes/grok"),
    }

    prompts = {
        "claude": root / "nexus_bridge" / "claude.md",
        "codex": root / "nexus_bridge" / "codex.md",
        "gemini": root / "nexus_bridge" / "gemini.md",
        "qwen": root / "nexus_bridge" / "qwen.md",
        "grok": root / "nexus_bridge" / "grok.md",
    }

    clis = {
        "claude": check_cli("claude"),
        "codex": check_cli("codex"),
        "gemini": check_cli("gemini"),
        "qwen": check_cli("qwen"),
        "grok": check_cli("grok"),
    }

    secrets_paths = [
        Path("~/.config/nucleus/secrets.env").expanduser(),
        Path("~/.config/pluribus_next/secrets.env").expanduser(),
    ]
    agent_secrets_paths = []
    for home in agent_homes.values():
        agent_secrets_paths.append(home / ".config" / "nucleus" / "secrets.env")
        agent_secrets_paths.append(home / ".config" / "pluribus_next" / "secrets.env")

    report = {
        "ts": time.time(),
        "iso": now_iso_utc(),
        "actor": actor,
        "bus": {
            "active_dir": bus_paths.active_dir,
            "primary_dir": bus_paths.primary_dir,
            "fallback_dir": bus_paths.fallback_dir,
            "events_path": bus_paths.events_path,
        },
        "wrappers": wrappers,
        "agents": {
            "claude": {
                "home": str(agent_homes["claude"]),
                "home_exists": path_exists(agent_homes["claude"]),
                "state_dir": str(agent_homes["claude"] / ".claude"),
                "state_exists": path_exists(agent_homes["claude"] / ".claude"),
                "prompt_file": str(prompts["claude"]),
                "prompt_exists": path_exists(prompts["claude"]),
            },
            "codex": {
                "home": str(agent_homes["codex"]),
                "home_exists": path_exists(agent_homes["codex"]),
                "state_file": str(agent_homes["codex"] / ".codex" / "config.toml"),
                "state_exists": path_exists(agent_homes["codex"] / ".codex" / "config.toml"),
                "prompt_file": str(prompts["codex"]),
                "prompt_exists": path_exists(prompts["codex"]),
            },
            "gemini": {
                "home": str(agent_homes["gemini"]),
                "home_exists": path_exists(agent_homes["gemini"]),
                "node20_path": str(Path("~/.local/node20/bin").expanduser()),
                "node20_exists": path_exists(Path("~/.local/node20/bin").expanduser()),
                "prompt_file": str(prompts["gemini"]),
                "prompt_exists": path_exists(prompts["gemini"]),
            },
            "qwen": {
                "home": str(agent_homes["qwen"]),
                "home_exists": path_exists(agent_homes["qwen"]),
                "state_dir": str(agent_homes["qwen"] / ".qwen"),
                "state_exists": path_exists(agent_homes["qwen"] / ".qwen"),
                "prompt_file": str(prompts["qwen"]),
                "prompt_exists": path_exists(prompts["qwen"]),
            },
            "grok": {
                "home": str(agent_homes["grok"]),
                "home_exists": path_exists(agent_homes["grok"]),
                "prompt_file": str(prompts["grok"]),
                "prompt_exists": path_exists(prompts["grok"]),
            },
        },
        "cli": clis,
        "secrets": {
            "global": check_secrets(secrets_paths),
            "agent_overlays": check_secrets(agent_secrets_paths),
            "env": {
                "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
                "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
                "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
            },
        },
    }

    missing_wrappers = [name for name, info in wrappers.items() if not info["exists"] or not info["executable"]]
    missing_common = [name for name, info in wrappers.items() if info["exists"] and not info["uses_common"]]
    missing_clis = [name for name, info in clis.items() if not info["present"]]

    report["warnings"] = []
    if missing_wrappers:
        report["warnings"].append({"kind": "wrapper_missing", "agents": missing_wrappers})
    if missing_common:
        report["warnings"].append({"kind": "wrapper_nonstandard", "agents": missing_common})
    if missing_clis:
        report["warnings"].append({"kind": "cli_missing", "agents": missing_clis})
    if bus_paths.fallback_dir:
        report["warnings"].append({"kind": "bus_fallback", "active_dir": bus_paths.active_dir, "primary_dir": bus_paths.primary_dir})

    return report


def render_report(report: dict) -> str:
    lines = []
    lines.append("Bootstrap Doctor Report")
    lines.append(f"  time: {report.get('iso')}")
    bus = report.get("bus", {})
    lines.append(f"  bus: active={bus.get('active_dir')} primary={bus.get('primary_dir')}")
    if bus.get("fallback_dir"):
        lines.append(f"  bus: fallback={bus.get('fallback_dir')}")
    lines.append("  wrappers:")
    for name, info in report.get("wrappers", {}).items():
        status = "ok" if info.get("exists") and info.get("executable") else "missing"
        common = "common" if info.get("uses_common") else "custom"
        lines.append(f"    - {name}: {status} ({common})")
    lines.append("  cli:")
    for name, info in report.get("cli", {}).items():
        status = "present" if info.get("present") else "missing"
        flag = info.get("prompt_flag") or "none"
        lines.append(f"    - {name}: {status} (prompt_flag={flag})")
    if report.get("warnings"):
        lines.append("  warnings:")
        for warning in report["warnings"]:
            lines.append(f"    - {warning}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="agent_bootstrap_doctor.py",
        description="Audit agent bootstrap readiness (wrappers, bus, CLI, secrets) and emit a report.",
    )
    ap.add_argument("--root", default="/pluribus", help="Workspace root (default: /pluribus).")
    ap.add_argument("--bus-dir", default=None, help="Bus directory (or set PLURIBUS_BUS_DIR).")
    ap.add_argument("--actor", default=None, help="Actor name (or set PLURIBUS_ACTOR).")
    ap.add_argument("--json", action="store_true", help="Print JSON report only.")
    ap.add_argument("--no-bus", action="store_true", help="Do not emit a bus report.")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if critical items are missing.")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    actor = args.actor or os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "unknown"
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or str(root / ".pluribus" / "bus")

    report = build_report(root, bus_dir, actor)

    if not args.no_bus:
        paths = agent_bus.resolve_bus_paths(bus_dir)
        agent_bus.emit_event(
            paths,
            topic="agent.bootstrap.report",
            kind="metric",
            level="info",
            actor=actor,
            data=report,
            trace_id=None,
            run_id=None,
            durable=False,
        )

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(render_report(report))

    if args.strict and report.get("warnings"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
