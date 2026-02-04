#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def run(argv: list[str], *, timeout_s: int = 120, env: dict | None = None) -> tuple[int, str, str]:
    p = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s, env=env)
    return int(p.returncode), (p.stdout or "").strip(), (p.stderr or "").strip()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ops_smoke.py", description="Operational smoke run: agent homes + mesh snapshot + provider smoke + mock STRp worker run.")
    ap.add_argument("--root", default="/pluribus")
    ap.add_argument("--bus-dir", default="/pluribus/.pluribus/bus")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--providers", default="mock,gemini", help="Comma list: mock,gemini,gemini-cli,claude-cli")
    ap.add_argument("--domains", action="store_true", help="Run domain intent validation + DNS audit for curated domains.")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    bus_dir = args.bus_dir
    tool = root / "nucleus" / "tools"

    run_id = str(uuid.uuid4())
    results: dict = {"id": run_id, "ts": time.time(), "steps": []}

    def step(name: str, cmd: list[str], *, env: dict | None = None, timeout_s: int | None = None) -> None:
        t0 = time.time()
        code, out, err = run(cmd, timeout_s=timeout_s or args.timeout, env=env)
        results["steps"].append(
            {
                "name": name,
                "cmd": cmd,
                "exit_code": code,
                "duration_s": max(0.0, time.time() - t0),
                "stdout": out[:4000],
                "stderr": err[:4000],
            }
        )

    step("agent_home_init", [sys.executable, str(tool / "agent_home_init.py"), "--root", str(root), "--bus-dir", bus_dir])
    step("mesh_status", [sys.executable, str(tool / "mesh_status.py"), "--root", str(root), "--bus-dir", bus_dir, "--emit-bus"])

    if args.domains:
        step("domain_intent_validate", [sys.executable, str(tool / "domain_intent.py"), "--root", str(root), "--bus-dir", bus_dir, "validate"], timeout_s=30)
        step(
            "domain_dns_audit_curated",
            [
                sys.executable,
                str(tool / "domain_registry.py"),
                "--root",
                str(root),
                "--bus-dir",
                bus_dir,
                "audit",
                "--ip",
                "auto",
                "--discover",
                "--scope",
                "curated",
                "--context",
                "ops_smoke",
            ],
            timeout_s=60,
        )

    prov = [p.strip() for p in args.providers.split(",") if p.strip()]
    if prov:
        cmd = [sys.executable, str(tool / "providers" / "provider_integration_smoke.py"), "--bus-dir", bus_dir, "--timeout", str(args.timeout), "--prompt", "Say 'ok' and one short sentence about STRp."]
        for p in prov:
            cmd.append(f"--{p}")
        step("provider_smoke", cmd)

    # Mock STRp worker run in a temp rhizome (validates topology + response writing without network).
    with tempfile.TemporaryDirectory(prefix="pluribus_strp_smoke_") as td:
        tmp_root = Path(td)
        step("rhizome_init", [sys.executable, str(tool / "rhizome.py"), "--root", str(tmp_root), "init", "--name", "ops_smoke", "--purpose", "smoke"], timeout_s=30)
        req_id_cmd = [
            sys.executable,
            str(tool / "strp_queue.py"),
            "--root",
            str(tmp_root),
            "request",
            "--kind",
            "distill",
            "--goal",
            "smoke test request",
            "--parallelizable",
            "--tool-density",
            "0.1",
            "--coord-budget-tokens",
            "8000",
        ]
        code, out, err = run(req_id_cmd, timeout_s=30)
        results["steps"].append({"name": "strp_queue_request", "cmd": req_id_cmd, "exit_code": code, "stdout": out[:200], "stderr": err[:4000], "duration_s": 0.0})
        step("strp_worker_once_mock", [sys.executable, str(tool / "strp_worker.py"), "--root", str(tmp_root), "--bus-dir", bus_dir, "--provider", "mock", "--once"], timeout_s=args.timeout)
        resp_path = tmp_root / ".pluribus" / "index" / "responses.ndjson"
        results["steps"].append({"name": "responses_present", "cmd": ["stat", str(resp_path)], "exit_code": 0 if resp_path.exists() else 2, "stdout": str(resp_path.exists()), "stderr": "", "duration_s": 0.0})

    sys.stdout.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(s["exit_code"] == 0 for s in results["steps"] if s["name"] != "responses_present") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
