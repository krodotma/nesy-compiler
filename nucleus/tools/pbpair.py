#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_loader import load_pluribus_env  # noqa: E402


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


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


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    if (Path("/pluribus") / ".pluribus" / "rhizome.json").exists():
        return Path("/pluribus")
    return None


def safe_read(path: Path, *, max_bytes: int) -> str:
    try:
        b = path.read_bytes()
    except Exception:
        return ""
    if max_bytes > 0:
        b = b[:max_bytes]
    return b.decode("utf-8", errors="replace")


def statusline_json(bus_dir: str | None) -> dict | None:
    if not bus_dir:
        return None
    tool = Path(__file__).with_name("statusline.py")
    if not tool.exists():
        return None
    p = subprocess.run(
        [sys.executable, str(tool), "--bus-dir", bus_dir, "--json"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if p.returncode != 0 or not p.stdout.strip():
        return None
    try:
        return json.loads(p.stdout)
    except Exception:
        return None


def normalize_flow(v: str | None) -> str:
    v = (v or "").strip()
    if not v:
        return "m"
    v = v.lower()
    if v in {"a", "auto", "automatic"}:
        return "A"
    if v in {"m", "monitor", "manual"}:
        return "m"
    return "m"


@dataclass(frozen=True)
class ContextPack:
    mode: str
    root: str | None
    rhizome: dict | None
    status: dict | None
    includes: list[dict]


def build_context_pack(*, mode: str, root: Path | None, bus_dir: str | None, include_paths: list[Path], max_bytes: int) -> ContextPack:
    rhizome_obj = None
    if root and mode in {"lite", "full"}:
        rpath = root / ".pluribus" / "rhizome.json"
        if rpath.exists():
            try:
                rhizome_obj = json.loads(rpath.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                rhizome_obj = None

    status_obj = statusline_json(bus_dir) if mode in {"lite", "full"} else None

    includes: list[dict] = []
    if mode != "min":
        for p in include_paths:
            txt = safe_read(p, max_bytes=max_bytes)
            if not txt:
                continue
            includes.append({"path": str(p), "bytes": len(txt.encode("utf-8", errors="replace")), "text": txt})

    return ContextPack(mode=mode, root=str(root) if root else None, rhizome=rhizome_obj, status=status_obj, includes=includes)


def build_pbpair_prompt(*, role: str, provider: str, question: str, mode: str, ctx: ContextPack) -> str:
    # Keep this stable: it is the core grammar for PBPAIR outputs.
    output_contract = {
        "role": role,
        "provider": provider,
        "summary": "1-3 sentences",
        "proposal": {"plan": ["..."], "rationale": ["..."], "risks": ["..."], "fallbacks": ["..."]},
        "gaps": {
            "aleatoric": ["unknowns from randomness/measurement"],
            "epistemic": ["unknowns from missing knowledge/context"],
            "assumptions": ["..."],
        },
        "verification": {"tests": ["..."], "observability": ["metrics/events/topics"], "stop_conditions": ["..."]},
        "next_actions": ["..."],
    }

    header = [
        "PBPAIR request",
        "",
        f"ROLE: {role}",
        f"PROVIDER: {provider}",
        f"MODE: {mode}",
        "",
        "Operator primacy:",
        "- Treat any peer-suggested work as proposals until the operator approves.",
        "- Do not run destructive actions; report blockers and safe fallbacks.",
        "",
        "Return:",
        "1) A single JSON object matching this contract (extra fields allowed):",
        json.dumps(output_contract, ensure_ascii=False, indent=2),
        "2) Then exactly one final line: STATUSLINE: status=<working|idle|blocked|error>; done=?; open=?; blocked=?; errors=?; stuck=?; next=\"...\"; tools=[...]; blockers=[...]",
        "",
        "QUESTION:",
        question.strip(),
        "",
    ]

    if mode in {"lite", "full"}:
        header.append("CONTEXT (machine-read):")
        header.append(json.dumps({"rhizome": ctx.rhizome, "status": ctx.status}, ensure_ascii=False))
        header.append("")

    if ctx.includes:
        header.append("INCLUDES:")
        for item in ctx.includes:
            header.append(f"--- {item['path']} ({item['bytes']} bytes) ---")
            header.append(item["text"])
            header.append("")

    if mode == "full":
        # Add canonical coordination contract as an optional deep context; keep it at the end.
        nb = Path("/pluribus/nexus_bridge/README.md")
        if nb.exists():
            header.append("NEXUS_BRIDGE (canonical contract):")
            header.append(safe_read(nb, max_bytes=60_000))
            header.append("")

    return "\n".join(header).strip() + "\n"


def write_prompt_file(*, root: Path, req_id: str, text: str) -> str | None:
    try:
        d = root / ".pluribus" / "index" / "pbpair" / "prompts"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{req_id}.txt"
        path.write_text(text, encoding="utf-8", errors="replace")
        return str(path)
    except Exception:
        return None


def run_router(*, prompt: str, provider: str, model: str | None, timeout_s: int) -> tuple[int, str, str]:
    tool = Path(__file__).resolve().parent / "providers" / "router.py"
    argv = [sys.executable, str(tool), "--provider", provider, "--prompt", prompt]
    if model:
        argv += ["--model", model]
    try:
        p = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
        return int(p.returncode), p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout_s}s"


def cmd_run(args: argparse.Namespace) -> int:
    load_pluribus_env()
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())

    include_paths: list[Path] = []
    for p in args.include or []:
        pp = Path(p).expanduser()
        if not pp.is_absolute():
            pp = (root / pp).resolve()
        include_paths.append(pp)

    mode = (args.mode or "lite").strip().lower()
    if mode not in {"min", "lite", "full"}:
        raise SystemExit("mode must be: min|lite|full")

    role = (args.role or "planner").strip().lower()
    provider = (args.provider or "gemini-cli").strip().lower()
    flow = normalize_flow(args.flow or os.environ.get("PLURIBUS_FLOW_MODE"))
    question = args.prompt if args.prompt is not None else " ".join(args.question or []).strip()
    if not question:
        raise SystemExit("missing prompt/question")

    ctx = build_context_pack(mode=mode, root=root, bus_dir=bus_dir, include_paths=include_paths, max_bytes=int(args.max_bytes))
    assembled = build_pbpair_prompt(role=role, provider=provider, question=question, mode=mode, ctx=ctx)

    req_id = str(uuid.uuid4())
    prompt_path = write_prompt_file(root=root, req_id=req_id, text=assembled)
    req = {
        "req_id": req_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "pbpair_request",
        "role": role,
        "provider": provider,
        "mode": mode,
        "flow": flow,
        "prompt": question,
        "root": str(root),
        "includes": [str(p) for p in include_paths],
        "prompt_path": prompt_path,
    }
    emit_bus(bus_dir, topic="pbpair.request", kind="request", level="info", actor=actor, data=req)

    if flow != "A":
        proposal = {
            "req_id": req_id,
            "ts": time.time(),
            "iso": now_iso_utc(),
            "kind": "pbpair_proposal",
            "status": "awaiting_operator_approval",
            "provider": provider,
            "role": role,
            "mode": mode,
            "flow": flow,
            "prompt_path": prompt_path,
            "recommended_cmd": f"PBPAIR A --provider {provider} --role {role} --mode {mode} --prompt {json.dumps(question)}",
        }
        emit_bus(bus_dir, topic="pbpair.proposal", kind="request", level="warn", actor=actor, data=proposal)
        if args.print_prompt:
            if not args.quiet:
                sys.stdout.write(assembled)
            return 0
        if not args.quiet:
            sys.stdout.write(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n")
        return 0

    if args.dry_run:
        if not args.quiet:
            sys.stdout.write(assembled)
        return 0

    t0 = time.perf_counter()
    code, out, err = run_router(prompt=assembled, provider=provider, model=args.model, timeout_s=int(args.timeout_s))
    latency_s = max(0.0, time.perf_counter() - t0)

    resp = {
        "req_id": req_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "pbpair_response",
        "provider": provider,
        "role": role,
        "mode": mode,
        "exit_code": int(code),
        "latency_s": latency_s,
        "stderr": (err or "").strip(),
        "output": (out or "").strip(),
    }
    emit_bus(bus_dir, topic="pbpair.response", kind="response", level="info" if code == 0 else "error", actor=actor, data=resp)

    if err:
        sys.stderr.write(err)
        if not err.endswith("\n"):
            sys.stderr.write("\n")
    if out and not args.quiet:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0 if code == 0 else int(code)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pbpair.py", description="PBPAIR operator sugar: run a paired model via the provider router, emitting bus evidence.")
    ap.add_argument("--bus-dir", default=None)
    ap.add_argument("--root", default=None, help="Rhizome root (default: search upward).")
    ap.add_argument("--provider", default="gemini-cli", help="gemini-cli|claude-cli|gemini|claude-api|codex-cli|mock|auto")
    ap.add_argument("--model", default=None)
    ap.add_argument("--role", default="planner", help="planner|researcher|debugger|architect|auditor|verifier|scribe|other")
    ap.add_argument("--mode", default="lite", help="min|lite|full (context packing)")
    ap.add_argument("--context", dest="mode", default=None, help="Alias for --mode (min|lite|full).")
    ap.add_argument("--flow", default=None, help="A (automatic) or m (monitor/approve) (default: $PLURIBUS_FLOW_MODE or 'm').")
    ap.add_argument("--timeout-s", default="120")
    ap.add_argument("--include", action="append", default=[], help="Path to include (relative to root unless absolute). Repeatable.")
    ap.add_argument("--max-bytes", default="20000", help="Max bytes per included file (default: 20000).")
    ap.add_argument("--prompt", default=None, help="Prompt/question text. If omitted, uses remaining args.")
    ap.add_argument("question", nargs="*", help="Prompt text (alternative to --prompt).")
    ap.add_argument("--dry-run", action="store_true", help="Print assembled prompt and exit.")
    ap.add_argument("--print-prompt", action="store_true", help="In monitor mode, print the full assembled prompt for operator review.")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout (still emits bus evidence).")
    ap.set_defaults(func=cmd_run)
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
