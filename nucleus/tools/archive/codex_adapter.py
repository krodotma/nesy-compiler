#!/usr/bin/env python3
"""
Codex Adapter for Pluribus.

Provides a minimal bridge to the Codex CLI/NPM package with bus evidence.
This adapter is intentionally lightweight: it checks availability, reports
versions, and can invoke the CLI when explicitly asked.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from agent_bus import resolve_bus_paths, emit_event, default_actor  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
NPM_PACKAGE = "@openai/codex"


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_package_version(package_json: Path) -> str | None:
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    version = data.get("version")
    return version if isinstance(version, str) else None


def _find_local_package(root: Path) -> Path | None:
    candidate = root / "node_modules" / "@openai" / "codex" / "package.json"
    return candidate if candidate.exists() else None


def _cli_version(cli_path: str, timeout_s: int = 5) -> str | None:
    try:
        proc = subprocess.run(
            [cli_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (proc.stdout or proc.stderr or "").strip()
    return out or None


@dataclass
class CodexStatus:
    cli_path: str | None
    cli_version: str | None
    npm_version: str | None
    npm_path: str | None
    ok: bool
    notes: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cli_path": self.cli_path,
            "cli_version": self.cli_version,
            "npm_version": self.npm_version,
            "npm_path": self.npm_path,
            "ok": self.ok,
            "notes": self.notes,
        }


class CodexAdapter:
    def __init__(self, *, bus_dir: str | None = None, actor: str | None = None, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or REPO_ROOT
        self.bus_paths = resolve_bus_paths(bus_dir)
        self.actor = actor or default_actor()

    def _emit(self, topic: str, kind: str, data: dict[str, Any], level: str = "info") -> None:
        emit_event(
            self.bus_paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=self.actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )

    def status(self) -> CodexStatus:
        cli_path = shutil.which("codex")
        cli_version = _cli_version(cli_path, timeout_s=3) if cli_path else None
        npm_package = _find_local_package(self.repo_root)
        npm_version = _read_package_version(npm_package) if npm_package else None
        ok = bool(cli_path or npm_version)
        notes = None if ok else "codex CLI or local npm package not found"

        status = CodexStatus(
            cli_path=cli_path,
            cli_version=cli_version,
            npm_version=npm_version,
            npm_path=str(npm_package) if npm_package else None,
            ok=ok,
            notes=notes,
        )
        self._emit(
            topic="membrane.codex.status",
            kind="metric",
            data={"status": status.to_dict(), "at": now_iso_utc()},
        )
        return status

    def invoke(self, args: list[str], timeout_s: int = 60) -> subprocess.CompletedProcess[str]:
        cli_path = shutil.which("codex")
        if not cli_path:
            err = "codex CLI not found on PATH"
            self._emit("membrane.codex.invoke", "log", {"args": args, "ok": False, "error": err}, level="warn")
            raise RuntimeError(err)

        self._emit("membrane.codex.invoke", "log", {"args": args, "ok": None})
        proc = subprocess.run(
            [cli_path, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        self._emit(
            "membrane.codex.invoke.result",
            "log",
            {
                "args": args,
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
            },
            level="info" if proc.returncode == 0 else "warn",
        )
        return proc


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex adapter (status/invoke).")
    parser.add_argument("--bus-dir", default=None, help="Override bus directory.")
    parser.add_argument("--actor", default=None, help="Actor name for bus events.")
    parser.add_argument("--status", action="store_true", help="Print status JSON and exit.")
    parser.add_argument("--invoke", nargs=argparse.REMAINDER, help="Invoke codex CLI with args.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    adapter = CodexAdapter(bus_dir=args.bus_dir, actor=args.actor)

    if args.status or not args.invoke:
        status = adapter.status()
        print(json.dumps(status.to_dict(), indent=2))
        return 0 if status.ok else 2

    proc = adapter.invoke(args.invoke)
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
