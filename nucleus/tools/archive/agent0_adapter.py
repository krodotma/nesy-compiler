#!/usr/bin/env python3
"""
Agent0 Adapter
==============

Integration layer for aiming-lab/Agent0 (self-evolving agents).
Maps Agent0's curriculum/executor paradigm into Pluribus task ledger
and bus evidence, with optional script execution when the repo is
present and configured.

Effects: R(file), W(file), W(bus)
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from nucleus.tools import agent_bus
from nucleus.tools import task_ledger

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_agent0_root() -> Path | None:
    env_root = os.environ.get("AGENT0_ROOT")
    if env_root:
        return Path(env_root)
    candidate = repo_root() / "membrane" / "agent0"
    if candidate.exists():
        return candidate
    return None


def check_agent0_install(root: Path | None) -> tuple[bool, str]:
    if not root:
        return False, "agent0_root_missing"
    curriculum = root / "Agent0" / "curriculum_train"
    executor = root / "Agent0" / "executor_train"
    if not curriculum.exists():
        return False, "curriculum_train_missing"
    if not executor.exists():
        return False, "executor_train_missing"
    return True, "ok"


def plan_path(root: Path) -> Path:
    directory = root / ".pluribus" / "index" / "agent0_plans"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"agent0_plan_{now_iso_utc().replace(':', '').replace('-', '')}.json"


def create_plan(
    *,
    goal: str,
    iterations: int,
    actor: str,
    req_id: str,
    run_id: str | None,
    ledger_path: Path | None = None,
    emit_bus: bool = True,
) -> dict:
    entries: list[dict] = []
    for i in range(1, iterations + 1):
        entries.append(
            task_ledger.append_entry(
                {
                    "req_id": req_id,
                    "actor": actor,
                    "topic": "agent0.curriculum",
                    "status": "planned",
                    "intent": f"Iteration {i}: curriculum proposal for goal: {goal}",
                    "meta": {"iteration": i, "role": "curriculum"},
                },
                ledger_path=ledger_path,
                emit_bus=emit_bus,
                run_id=run_id,
            )
        )
        entries.append(
            task_ledger.append_entry(
                {
                    "req_id": req_id,
                    "actor": actor,
                    "topic": "agent0.executor",
                    "status": "planned",
                    "intent": f"Iteration {i}: executor solves curriculum tasks",
                    "meta": {"iteration": i, "role": "executor"},
                },
                ledger_path=ledger_path,
                emit_bus=emit_bus,
                run_id=run_id,
            )
        )

    plan = {
        "id": str(uuid.uuid4()),
        "req_id": req_id,
        "goal": goal,
        "iterations": iterations,
        "actor": actor,
        "created_iso": now_iso_utc(),
        "entries": entries,
    }
    return plan


def run_script(script_path: Path, args: list[str], timeout: int) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["bash", str(script_path)] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"Script not found: {script_path}"
    except Exception as exc:
        return 1, "", str(exc)


def cmd_check(args: argparse.Namespace) -> int:
    root = resolve_agent0_root()
    ok, detail = check_agent0_install(root)
    print(json.dumps({"ok": ok, "detail": detail, "root": str(root) if root else None}, indent=2))
    return 0 if ok else 1


def cmd_plan(args: argparse.Namespace) -> int:
    actor = args.actor or default_actor()
    req_id = args.req_id or str(uuid.uuid4())[:8]
    run_id = args.run_id

    plan = create_plan(
        goal=args.goal,
        iterations=args.iterations,
        actor=actor,
        req_id=req_id,
        run_id=run_id,
        ledger_path=Path(args.ledger_path) if args.ledger_path else None,
    )

    root = repo_root()
    out_path = plan_path(root)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    paths = agent_bus.resolve_bus_paths(args.bus_dir)
    agent_bus.emit_event(
        paths,
        topic="agent0.plan.created",
        kind="artifact",
        level="info",
        actor=actor,
        data={"req_id": req_id, "plan_path": str(out_path), "iterations": args.iterations},
        trace_id=None,
        run_id=run_id,
        durable=False,
    )

    payload = {"status": "ok", "plan_path": str(out_path), "entries": len(plan["entries"])}
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    root = resolve_agent0_root()
    ok, detail = check_agent0_install(root)
    if not ok or not root:
        print(f"ERROR: Agent0 repo not ready: {detail}", file=sys.stderr)
        return 1

    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = root / script_path

    code, stdout, stderr = run_script(script_path, args.script_args, args.timeout)
    paths = agent_bus.resolve_bus_paths(args.bus_dir)
    agent_bus.emit_event(
        paths,
        topic="agent0.train.result",
        kind="artifact",
        level="info" if code == 0 else "error",
        actor=args.actor or default_actor(),
        data={
            "script": str(script_path),
            "exit_code": code,
            "stderr": (stderr or "")[:2000],
        },
        trace_id=None,
        run_id=args.run_id,
        durable=False,
    )

    if args.json_output:
        print(json.dumps({"status": "ok" if code == 0 else "error", "exit_code": code}))
    else:
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Pluribus adapter for Agent0 self-evolving agents.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Verify Agent0 repo availability.")
    p_check.set_defaults(func=cmd_check)

    p_plan = sub.add_parser("plan", help="Create curriculum/executor plan entries.")
    p_plan.add_argument("--goal", required=True)
    p_plan.add_argument("--iterations", type=int, default=1)
    p_plan.add_argument("--actor")
    p_plan.add_argument("--req-id")
    p_plan.add_argument("--run-id")
    p_plan.add_argument("--bus-dir")
    p_plan.add_argument("--ledger-path")
    p_plan.add_argument("--json-output", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_train = sub.add_parser("train", help="Run a specific Agent0 training script.")
    p_train.add_argument("--script", required=True, help="Script path (absolute or relative to AGENT0_ROOT).")
    p_train.add_argument("--script-args", nargs="*", default=[])
    p_train.add_argument("--timeout", type=int, default=3600)
    p_train.add_argument("--actor")
    p_train.add_argument("--run-id")
    p_train.add_argument("--bus-dir")
    p_train.add_argument("--json-output", action="store_true")
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
