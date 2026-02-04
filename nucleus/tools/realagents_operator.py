#!/usr/bin/env python3
"""
REALAGENTS Operator: The Deep Dispatcher
========================================

Implements the REALAGENTS semantic operator.

Contract (non-optional):
- Emits schema-valid `rd.tasks.dispatch` (kind=request) as the assignment SoR.
- Mirrors to `infer_sync.request` to wake up OITERATE/monitors.

See:
- nucleus/specs/realagents_upgrade_v1.md
- nucleus/specs/rd_tasks_dispatch.schema.json
"""
import argparse
import json
import os
import sys
import time
import uuid
import subprocess
from pathlib import Path

sys.dont_write_bytecode = True

def default_bus_dir(d: str | None = None) -> str:
    return d or os.environ.get("PLURIBUS_BUS_DIR") or str(Path(__file__).resolve().parents[2] / ".pluribus" / "bus")

def emit_bus_event(*, bus_dir: str, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    tool = Path(__file__).with_name("agent_bus.py")
    cmd = [
        sys.executable, str(tool),
        "--bus-dir", str(bus_dir),
        "pub",
        "--topic", topic,
        "--kind", kind,
        "--level", level,
        "--actor", actor,
        "--data", json.dumps(data, ensure_ascii=False)
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _normalize_targets(raw: str) -> list[str]:
    return [t.strip() for t in str(raw or "").split(",") if t.strip()]

def _looks_like_task_id(s: str) -> bool:
    # Allows typical stable identifiers like REALAGENTS_upgrade / mcp_deepening / A2A-v1.
    s = (s or "").strip()
    if not s:
        return False
    if " " in s or "\t" in s or "\n" in s:
        return False
    return True

def build_reagents_upgrade_payload(*, req_id: str, iso: str, spec_ref: str, targets: list[str], intent: str, actor: str) -> dict:
    # Keep this payload schema-compatible with rd_tasks_dispatch.schema.json.
    return {
        "req_id": req_id,
        "task_id": "REALAGENTS_upgrade",
        "intent": "realagents_upgrade",
        "iso": iso,
        "actor": actor,
        "spec_ref": spec_ref,
        "keywords": ["REALAGENTS", "NO_SHIMS", "MCP", "A2A", "ADK", "CKIN_v12_1", "DKIN_v12_1", "MCP_INTEROP"],
        "targets": targets,
        "constraints": {"append_only": True, "non_blocking": True, "tests_first": True, "no_secrets": True},
        "gates": ["P", "E", "L", "R", "Q", "Ω"],
        "tasks": [
            {
                "id": "T0",
                "title": "Make rd.tasks.dispatch real (dispatcher+responder+CKIN surface)",
                "depends_on": [],
                "deliverables": [
                    "nucleus/tools/rd_tasks_dispatch.py",
                    "nucleus/tools/rd_tasks_responder.py",
                    "nucleus/tools/ckin_report.py (dispatch/ack visibility)",
                ],
                "acceptance": [
                    "rd.tasks.dispatch emits schema-valid payload with req_id",
                    "rd.tasks.ack emitted by responder with same req_id",
                    "CKIN shows last dispatch age + ack ratio",
                ],
                "risk": "low",
                "notes": intent,
            },
            {
                "id": "T1",
                "title": "MCP deepening (bus-evidenced tool calls + inventory)",
                "depends_on": ["T0"],
                "deliverables": ["mcp.inventory artifact", "mcp host smoke tests", "effects mapped per tool"],
                "acceptance": [
                    "mcp.host.call → mcp.host.response loop works for built-in servers",
                    "inventory lists servers/tools/effects with provenance",
                ],
                "risk": "med",
                "notes": "",
            },
            {
                "id": "T2",
                "title": "A2A deepening (capabilities + negotiate/decline/redirect)",
                "depends_on": ["T0"],
                "deliverables": ["a2a.capabilities.*", "a2a.negotiate.*", "toy negotiation test"],
                "acceptance": [
                    "non-blocking negotiation yields explicit decline/redirect (no silent drops)",
                    "payloads align with a2a_request_taxonomy kinds",
                ],
                "risk": "med",
                "notes": "",
            },
            {
                "id": "T3",
                "title": "ADK deepening (flow.py DSL + roundtrip + InferCell binding)",
                "depends_on": ["T0"],
                "deliverables": ["sdk/flow.py expanded", "roundtrip tests", "InferCell mapping"],
                "acceptance": ["flowfile ↔ graph spec roundtrip invariant passes", "bindings emit bus evidence"],
                "risk": "high",
                "notes": "",
            },
        ],
    }

def build_custom_payload(*, req_id: str, iso: str, spec_ref: str, targets: list[str], task_id: str, title: str, intent: str, actor: str) -> dict:
    return {
        "req_id": req_id,
        "task_id": task_id,
        "intent": "realagents_custom",
        "iso": iso,
        "actor": actor,
        "spec_ref": spec_ref,
        "keywords": ["REALAGENTS", "NO_SHIMS", "CKIN_v12_1", "DKIN_v12_1"],
        "targets": targets,
        "constraints": {"append_only": True, "non_blocking": True, "tests_first": True, "no_secrets": True},
        "gates": ["P", "E", "L", "R", "Q"],
        "tasks": [
            {
                "id": "T0",
                "title": title,
                "depends_on": [],
                "deliverables": ["implementation", "tests", "bus evidence"],
                "acceptance": ["tests pass", "bus evidence emitted", "operator can PBFLUSH and await CKIN"],
                "risk": "unknown",
                "notes": intent,
            }
        ],
    }

def main():
    parser = argparse.ArgumentParser(description="REALAGENTS Dispatch Operator")
    parser.add_argument("--bus-dir", default=None, help="Bus directory")
    parser.add_argument("--task", default="REALAGENTS_upgrade", help="Task id (e.g. REALAGENTS_upgrade) OR freeform title (legacy).")
    parser.add_argument("--task-id", default=None, help="Explicit stable task id (overrides --task when set).")
    parser.add_argument("--targets", default=None, help="Target agent(s), comma-separated (preferred).")
    parser.add_argument("--target", default=None, help="Alias for --targets (legacy).")
    parser.add_argument("--intent", required=True, help="Description of the deep work required")
    parser.add_argument("--spec-ref", default="nucleus/specs/realagents_upgrade_v1.md", help="Spec anchor path (payload.spec_ref).")
    args = parser.parse_args()

    bus_dir = default_bus_dir(args.bus_dir)
    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "realagents-op"
    
    req_id = str(uuid.uuid4())
    iso = now_iso()
    targets_raw = args.targets if args.targets is not None else args.target
    targets = _normalize_targets(targets_raw or "claude,codex,gemini")
    spec_ref = str(args.spec_ref)

    task_id = (args.task_id or "").strip()
    raw_task = str(args.task or "").strip()
    if not task_id:
        task_id = raw_task if _looks_like_task_id(raw_task) else "REALAGENTS_custom"

    if task_id == "REALAGENTS_upgrade":
        payload = build_reagents_upgrade_payload(req_id=req_id, iso=iso, spec_ref=spec_ref, targets=targets, intent=str(args.intent), actor=actor)
    else:
        title = raw_task if raw_task else task_id
        payload = build_custom_payload(
            req_id=req_id,
            iso=iso,
            spec_ref=spec_ref,
            targets=targets,
            task_id=task_id,
            title=title,
            intent=str(args.intent),
            actor=actor,
        )

    # 1. Primary Dispatch (The Assignment)
    emit_bus_event(
        bus_dir=bus_dir,
        topic="rd.tasks.dispatch",
        kind="request",
        level="info",
        actor=actor,
        data=payload
    )

    # 2. Mirror to Infer Sync (The Wake-up Call)
    # This ensures OITERATE loops see it even if they don't watch rd.tasks.*
    emit_bus_event(
        bus_dir=bus_dir,
        topic="infer_sync.request",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "intent": "rd.tasks.dispatch",
            "subproject": "realagents",
            "inputs": payload
        }
    )

    print(req_id)

if __name__ == "__main__":
    main()
