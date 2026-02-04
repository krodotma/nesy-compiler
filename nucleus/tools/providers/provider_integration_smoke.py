#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_loader import load_pluribus_env  # noqa: E402


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).resolve().parents[1] / "agent_bus.py"
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


def run(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    try:
        p = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
        return int(p.returncode), p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout_s}s: {' '.join(argv)}"


def main(argv: list[str]) -> int:
    load_pluribus_env()
    ap = argparse.ArgumentParser(
        prog="provider_integration_smoke.py",
        description="Integration smoke tests for providers (Gemini + Claude API/CLI + optional CLI + mock).",
    )
    ap.add_argument("--bus-dir", default=None, help="Emit results to agent bus.")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--prompt", default="Say 'ok' and one short sentence about STRp.")
    ap.add_argument("--gemini", action="store_true", help="Run Gemini (API if key exists, else CLI).")
    ap.add_argument("--gemini-cli", action="store_true", help="Run Gemini CLI test (requires gemini CLI + OAuth login).")
    ap.add_argument("--gemini3", action="store_true", help="Run Gemini-3 multi-route smoke (gemini CLI + Vertex; refuses downgrade).")
    ap.add_argument("--vertex-gemini", action="store_true", help="Run Gemini via Vertex AI (requires gcloud auth + VERTEX_PROJECT).")
    ap.add_argument("--claude-api", action="store_true", help="Run Claude API test (requires ANTHROPIC_API_KEY).")
    ap.add_argument("--claude-cli", action="store_true", help="Run Claude Code CLI test (requires claude CLI + setup-token).")
    ap.add_argument("--codex-cli", action="store_true", help="Run Codex CLI test (requires codex CLI login).")
    ap.add_argument("--mock", action="store_true", help="Run mock provider test (no network/auth).")
    args = ap.parse_args(argv)

    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "unknown"
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    want_any = args.gemini or args.gemini_cli or args.gemini3 or args.vertex_gemini or args.claude_api or args.claude_cli or args.codex_cli or args.mock
    if not want_any:
        args.gemini = True
        args.gemini3 = True
        args.vertex_gemini = True
        args.claude_api = True
        args.claude_cli = True
        args.codex_cli = True
        args.mock = True

    tool_dir = Path(__file__).resolve().parent
    results: list[dict] = []

    def record(name: str, code: int, out: str, err: str) -> None:
        rec = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": now_iso_utc(),
            "kind": "provider_smoke",
            "provider": name,
            "exit_code": code,
            "stdout": out,
            "stderr": err,
        }
        results.append(rec)
        emit_bus(bus_dir, topic=f"providers.smoke.{name}", kind="log", level="info" if code == 0 else "error", actor=actor, data=rec)

    prefer_cli = (os.environ.get("PLURIBUS_GEMINI_PREFER_CLI") or "").strip().lower() in {"1", "true", "yes", "on"}
    if args.gemini:
        have_key = bool((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip())
        if have_key and not (prefer_cli and shutil.which("gemini")):
            code, out, err = run([sys.executable, str(tool_dir / "gemini_smoke.py"), "--prompt", args.prompt], args.timeout)
            # Free-tier quota keys can be present but unusable.
            if code != 0 and ("http error: 429" in err or "RESOURCE_EXHAUSTED" in err or "Quota exceeded" in err):
                code2, out2, err2 = run([sys.executable, str(tool_dir / "gemini_cli_smoke.py"), "--prompt", args.prompt], args.timeout)
                record("gemini", code, out, err)
                record("gemini-cli", code2, out2, err2)
            else:
                record("gemini", code, out, err)
        else:
            code, out, err = run([sys.executable, str(tool_dir / "gemini_cli_smoke.py"), "--prompt", args.prompt], args.timeout)
            record("gemini-cli", code, out, err)

    if args.gemini_cli:
        code, out, err = run([sys.executable, str(tool_dir / "gemini_cli_smoke.py"), "--prompt", args.prompt], args.timeout)
        record("gemini-cli", code, out, err)

    if args.gemini3:
        code, out, err = run(
            [sys.executable, str(tool_dir / "gemini3_fallback_smoke.py"), "--prompt", args.prompt], max(args.timeout, 90)
        )
        record("gemini3", code, out, err)

    if args.vertex_gemini:
        code, out, err = run([sys.executable, str(tool_dir / "vertex_gemini_smoke.py"), "--prompt", args.prompt], args.timeout)
        record("vertex-gemini", code, out, err)

    if args.claude_api:
        code, out, err = run([sys.executable, str(tool_dir / "claude_smoke.py"), "--prompt", args.prompt], args.timeout)
        record("claude-api", code, out, err)

    if args.claude_cli:
        home = os.environ.get("PLURIBUS_CLAUDE_HOME") or "/pluribus/.pluribus/agent_homes/claude"
        code, out, err = run([sys.executable, str(tool_dir / "claude_cli_smoke.py"), "--prompt", args.prompt, "--home", home], args.timeout)
        record("claude-cli", code, out, err)

    if args.codex_cli:
        # codex_cli_smoke enforces its own 120s timeout; don't undercut it here.
        code, out, err = run([sys.executable, str(tool_dir / "codex_cli_smoke.py"), "--prompt", args.prompt], max(args.timeout, 130))
        record("codex-cli", code, out, err)

    if args.mock:
        code, out, err = run([sys.executable, str(tool_dir / "mock_smoke.py"), "--prompt", args.prompt], args.timeout)
        record("mock", code, out, err)

    sys.stdout.write(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
