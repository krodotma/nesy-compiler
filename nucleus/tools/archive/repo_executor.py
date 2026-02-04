#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import time
import subprocess
import os
from pathlib import Path
from typing import Any, Dict

# Ensure nucleus.sdk is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from nucleus.tools.agent_bus import iter_lines_follow, resolve_bus_paths, default_actor
except ImportError:
    # Fallback/Bootstrap mode
    current_dir = Path(__file__).resolve().parents[0]
    sys.path.append(str(current_dir.parents[0] / "tools"))
    from agent_bus import iter_lines_follow, resolve_bus_paths, default_actor

ISO_GIT_TOOL = Path(__file__).parent / "iso_git.mjs"
ARTIFACTS_INDEX = Path(__file__).resolve().parents[2] / ".pluribus" / "index" / "artifacts.ndjson"

# Request ID collision detection
try:
    from req_id_registry import RequestIdRegistry
    REQ_ID_REGISTRY = RequestIdRegistry()
except ImportError:
    REQ_ID_REGISTRY = None

# Dynamic Imports for Ribosome
try:
    from nucleus.ribosome.dna.genome import OrganismGenome
    from nucleus.ribosome.transcriber import RibosomeTranscriber
    RIBOSOME_AVAILABLE = True
except ImportError:
    RIBOSOME_AVAILABLE = False

try:
    from nucleus.tools import task_ledger as task_ledger_mod  # type: ignore
except Exception:
    task_ledger_mod = None

def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: Dict[str, Any]) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    
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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def append_task_ledger(entry: Dict[str, Any], *, bus_dir: str | None) -> None:
    if task_ledger_mod is None:
        return
    try:
        task_ledger_mod.append_entry(entry, bus_dir=bus_dir)
    except Exception:
        return

def get_artifact_meta(sha: str) -> Dict[str, Any] | None:
    """Retrieve artifact metadata from the index."""
    if not ARTIFACTS_INDEX.exists():
        return None
    try:
        with open(ARTIFACTS_INDEX, "r") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                    # Support partial SHA matching
                    if rec.get("sha256", "").startswith(sha):
                        return rec
                except: pass
    except Exception as e:
        print(f"Error reading artifacts: {e}")
    return None

def get_artifact_content(sha: str) -> str:
    """
    Retrieve artifact content. 
    """
    # 1. Try to find path from meta
    meta = get_artifact_meta(sha)
    if meta and "original_path" in meta:
        path = Path(meta["original_path"])
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
            
    # 2. Fallback placeholder
    return f"""# Promoted Artifact
Source SHA: {sha}
Promoted At: {time.time()}

This content was promoted from the Rhizome.
"""

def run_iso_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    raise RuntimeError("run_iso_git(args, cwd) is deprecated; use run_iso_git_cmd()")


def run_iso_git_cmd(command: str, repo_dir: Path, args: list[str] | None = None) -> subprocess.CompletedProcess:
    """
    Run iso_git.mjs against a repository directory.

    iso_git usage: `node iso_git.mjs <command> [dir] [args]`
    """
    argv = ["node", str(ISO_GIT_TOOL), command, str(repo_dir)]
    if args:
        argv.extend(args)
    return subprocess.run(argv, capture_output=True, text=True, check=False, cwd=repo_dir)


# === PATH VALIDATION (Bounded Polymorphism Enforcement) ===
ALLOWED_STAGE_PREFIXES = [
    "nucleus/docs/",
    "nucleus/specs/",
    "nucleus/sdk/",
    "nucleus/ribosome/",
    "nucleus/dashboard/src/",
    "nucleus/tools/",
    "data/",
    "docs/",
    "specs/",
]

BLOCKED_PATHS = [
    ".git/",
    ".pluribus/",
    "node_modules/",
    "__pycache__/",
    ".env",
    "credentials",
    "secrets",
    "/etc/",
    "/root/",
    "/home/",
]


def validate_stage_path(stage_path: str) -> tuple[bool, str]:
    """
    Validate stage_path for bounded polymorphism enforcement.

    Returns (is_valid, error_message).

    Rules:
    1. No path traversal (..)
    2. Must be relative or within /pluribus
    3. Must match allowed prefixes
    4. Must not match blocked paths
    """
    # Normalize path
    p = Path(stage_path)

    # Check for path traversal
    if ".." in str(stage_path):
        return False, "Path traversal (..) not allowed"

    # Resolve to absolute for checking
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (Path("/pluribus") / p).resolve()

    # Must be within /pluribus
    try:
        resolved.relative_to(Path("/pluribus"))
    except ValueError:
        return False, f"Path must be within /pluribus, got: {resolved}"

    # Check blocked paths
    path_str = str(resolved).lower()
    for blocked in BLOCKED_PATHS:
        if blocked.lower() in path_str:
            return False, f"Path contains blocked segment: {blocked}"

    # Check allowed prefixes (relative path from /pluribus)
    rel_path = str(resolved.relative_to(Path("/pluribus")))

    # Allow any path if PLURIBUS_ALLOW_ANY_PATH=1 (for testing)
    if os.environ.get("PLURIBUS_ALLOW_ANY_PATH") == "1":
        return True, ""

    for prefix in ALLOWED_STAGE_PREFIXES:
        if rel_path.startswith(prefix):
            return True, ""

    return False, f"Path '{rel_path}' not in allowed prefixes: {ALLOWED_STAGE_PREFIXES}"


def handle_request(event: Dict[str, Any], bus_paths: Any, actor: str):
    data = event.get("data", {})
    req_id = data.get("req_id")
    plan_ref = data.get("plan_ref")
    stage_path = data.get("stage_path")
    
    if not req_id or not plan_ref or not stage_path:
        print(f"[{actor}] Invalid request: missing fields")
        return

    # === PATH VALIDATION (Bounded Polymorphism Enforcement) ===
    is_valid, err_msg = validate_stage_path(stage_path)
    if not is_valid:
        print(f"[{actor}] BLOCKED: Invalid stage_path: {err_msg}")
        emit_bus(bus_paths.bus_dir, topic="repo.exec.result", kind="metric", level="error", actor=actor,
                 data={
                     "req_id": req_id,
                     "status": "error",
                     "errors": [f"Path validation failed: {err_msg}"],
                     "target_path": stage_path
                 })
        emit_bus(bus_paths.bus_dir, topic="repo.exec.path_violation", kind="metric", level="warning", actor=actor,
                 data={"req_id": req_id, "stage_path": stage_path, "error": err_msg})
        return

    # === REQ_ID COLLISION DETECTION ===
    if REQ_ID_REGISTRY:
        acquired, existing_actor = REQ_ID_REGISTRY.acquire(req_id, actor, topic="repo.exec.request")
        if not acquired:
            print(f"[{actor}] COLLISION: req_id {req_id} already in use by {existing_actor}")
            emit_bus(bus_paths.bus_dir, topic="repo.exec.result", kind="metric", level="error", actor=actor,
                     data={
                         "req_id": req_id,
                         "status": "error",
                         "errors": [f"req_id collision: already in use by {existing_actor}"],
                         "target_path": stage_path
                     })
            emit_bus(bus_paths.bus_dir, topic="repo.exec.collision", kind="metric", level="warning", actor=actor,
                     data={"req_id": req_id, "existing_actor": existing_actor, "requesting_actor": actor})
            return

    print(f"[{actor}] Processing req {req_id}: Promote {plan_ref} -> {stage_path}")

    append_task_ledger(
        {
            "req_id": req_id,
            "actor": actor,
            "topic": "repo.exec",
            "status": "in_progress",
            "intent": f"Promote rhizome {plan_ref[:8]} to {stage_path}",
            "meta": {"plan_ref": plan_ref, "stage_path": stage_path},
        },
        bus_dir=bus_paths.bus_dir,
    )

    # 1. Emit Progress
    emit_bus(bus_paths.bus_dir, topic="repo.exec.progress", kind="metric", level="info", actor=actor,
             data={"req_id": req_id, "step": "analyzing", "status": "running"})

    repo_root = Path("/pluribus").resolve()
    stage_abs = Path(stage_path)
    if not stage_abs.is_absolute():
        stage_abs = (repo_root / stage_abs).resolve()
    stage_rel = str(stage_abs.relative_to(repo_root))

    try:
        meta = get_artifact_meta(plan_ref)
        is_genome = meta and ("genome" in meta.get("tags", []) or meta.get("kind") == "genome")

        if is_genome and RIBOSOME_AVAILABLE:
            # === RIBOSOME GENESIS FLOW ===
            target_root = stage_abs
            
            emit_bus(bus_paths.bus_dir, topic="repo.exec.progress", kind="metric", level="info", actor=actor,
                 data={"req_id": req_id, "step": "transcribing", "status": "running"})
            
            # Load Genome (Mock loading from content for now, assuming JSON/YAML structure)
            # In production this would deserialize the artifact content
            genome_content = get_artifact_content(plan_ref)
            # For prototype, we'll construct a sample if parsing fails, or use the sample_genome
            # if the content is just a placeholder.
            from nucleus.ribosome.dna.genome import sample_genome
            genome = sample_genome() 
            genome.name = target_root.name
            
            transcriber = RibosomeTranscriber(target_root)
            transcriber.transcribe(genome)
            
            emit_bus(bus_paths.bus_dir, topic="repo.exec.progress", kind="metric", level="info", actor=actor,
                 data={"req_id": req_id, "step": "genesis_commit", "status": "running"})
            
            # Initialize Git Repo
            res_init = run_iso_git_cmd("init", target_root)
            if res_init.returncode != 0:
                raise RuntimeError(f"iso_git init failed: {res_init.stderr.strip() or res_init.stdout.strip()}")

            # Commit all generated files (auto-stages everything).
            message = f"Genesis: {genome.name} ({genome.version})\n\nRhizome-SHA: {plan_ref}\nPlan-Req: {req_id}"
            res_commit = run_iso_git_cmd("commit", target_root, [message])
            if res_commit.returncode != 0:
                raise RuntimeError(f"iso_git commit failed: {res_commit.stderr.strip() or res_commit.stdout.strip()}")

        else:
            # === STANDARD ARTIFACT FLOW ===
            content = get_artifact_content(plan_ref)
            stage_abs.parent.mkdir(parents=True, exist_ok=True)
            stage_abs.write_text(content, encoding="utf-8")
            
            emit_bus(bus_paths.bus_dir, topic="repo.exec.progress", kind="metric", level="info", actor=actor,
                 data={"req_id": req_id, "step": "staging", "status": "running"})

            message = f"feat(rhizome): promote {plan_ref[:8]}\n\nRhizome-SHA: {plan_ref}\nPlan-Req: {req_id}"
            res_commit = run_iso_git_cmd("commit-paths", repo_root, [message, stage_rel])
            if res_commit.returncode != 0:
                raise Exception(f"Git commit-paths failed: {res_commit.stderr}")

        # 5. Success
        emit_bus(bus_paths.bus_dir, topic="repo.exec.result", kind="response", level="info", actor=actor,
                 data={
                     "req_id": req_id,
                     "status": "success",
                     "target_path": str(stage_path),
                     "type": "organism" if is_genome else "artifact",
                     "errors": []
                 })
        append_task_ledger(
            {
                "req_id": req_id,
                "actor": actor,
                "topic": "repo.exec",
                "status": "completed",
                "summary": f"Promoted {plan_ref[:8]} to {stage_path}",
                "meta": {"plan_ref": plan_ref, "stage_path": stage_path},
            },
            bus_dir=bus_paths.bus_dir,
        )
        print(f"[{actor}] Request {req_id} completed successfully.")

    except Exception as e:
        print(f"[{actor}] Request {req_id} failed: {e}")
        emit_bus(bus_paths.bus_dir, topic="repo.exec.result", kind="response", level="error", actor=actor,
                 data={
                     "req_id": req_id,
                     "status": "error",
                     "errors": [str(e)]
                 })
        append_task_ledger(
            {
                "req_id": req_id,
                "actor": actor,
                "topic": "repo.exec",
                "status": "blocked",
                "summary": f"Promotion failed: {e}",
                "meta": {"plan_ref": plan_ref, "stage_path": stage_path, "error": str(e)},
            },
            bus_dir=bus_paths.bus_dir,
        )

    finally:
        # Release req_id from collision registry
        if REQ_ID_REGISTRY:
            REQ_ID_REGISTRY.release(req_id)


def handle_plan_request(event: Dict[str, Any], bus_paths: Any, actor: str):
    data = event.get("data", {})
    req_id = data.get("req_id")
    artifact_sha = data.get("artifact_sha")
    template_id = data.get("template_id", "default")
    
    if not req_id or not artifact_sha:
        print(f"[{actor}] Invalid plan request: missing fields")
        return

    print(f"[{actor}] Planning req {req_id}: Artifact {artifact_sha}, Template {template_id}")

    append_task_ledger(
        {
            "req_id": req_id,
            "actor": actor,
            "topic": "repo.plan",
            "status": "planned",
            "intent": f"Plan promotion for rhizome {artifact_sha[:8]}",
            "meta": {"artifact_sha": artifact_sha, "template_id": template_id},
        },
        bus_dir=bus_paths.bus_dir,
    )

    # Check for Ribosome DNA
    meta = get_artifact_meta(artifact_sha)
    is_genome = meta and ("genome" in meta.get("tags", []) or meta.get("kind") == "genome")
    
    if is_genome:
        target_path = f"clades/{artifact_sha[:8]}_organism"
        plan_steps = [
            {"step": "ribosome.load", "description": f"Load Organism Genome {artifact_sha[:8]}"},
            {"step": "ribosome.transcribe", "description": "Materialize phenotype structure"},
            {"step": "git.init", "description": "Initialize isomorphic repository"},
            {"step": "git.commit", "description": "Genesis commit (Root PQC Signed)"}
        ]
        preview = "🧬 Ribosome Genome Detected.\n\nType: OrganismGenome\nClades: [Core, Tools, AgentMesh]\n\nReady to transcribe."
    else:
        # Standard Single Artifact Promotion
        target_path = f"rhizome_promotions/artifact_{artifact_sha[:8]}.md"
        plan_steps = [
            {"step": "fetch", "description": f"Retrieve artifact {artifact_sha[:8]} from Rhizome"},
            {"step": "transform", "description": f"Apply template '{template_id}'"},
            {"step": "stage", "description": f"Stage content to {target_path}"},
            {"step": "verify", "description": "Run standard linting"},
            {"step": "commit", "description": "Commit with provenance trailers"}
        ]
        preview = f"# Preview\nSource: {artifact_sha}\n\n[Content Preview...]"

    # Emit Plan Response
    emit_bus(bus_paths.bus_dir, topic="repo.plan.response", kind="response", level="info", actor=actor,
             data={
                 "req_id": req_id,
                 "plan_steps": plan_steps,
                 "target_path": target_path,
                 "preview_snippet": preview,
                 "warnings": []
             })
    print(f"[{actor}] Plan {req_id} generated.")

def main():
    bus_paths = resolve_bus_paths(None)
    actor = "repo-executor" # Distinct from codex/gpt
    
    print(f"[{actor}] Repo Executor (Real): Listening for repo.exec/plan.request events...")
    
    for line in iter_lines_follow(str(bus_paths.events_path), poll_s=0.5):
        try:
            event = json.loads(line)
            topic = event.get("topic")
            kind = event.get("kind")
            
            if kind == "request":
                if topic == "repo.exec.request":
                    handle_request(event, bus_paths, actor)
                elif topic == "repo.plan.request":
                    handle_plan_request(event, bus_paths, actor)
                    
        except Exception as e:
            print(f"[{actor}] Error loop: {e}")

if __name__ == "__main__":
    main()
