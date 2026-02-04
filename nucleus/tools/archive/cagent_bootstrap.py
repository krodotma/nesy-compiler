#!/usr/bin/env python3
"""
CAGENT Bootstrap Daemon - Citizen Agent Unified Initialization

DKIN v28 compliant bootstrap sequence for all Pluribus agents.
Implements the 9-step bootstrap from cagent_protocol_v1.md.

Usage:
    cagent_bootstrap.py --actor claude [--class sagent] [--persona architect]
    cagent_bootstrap.py --actor codex --class swagent --scope "nucleus/tools/*"
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Canonical paths
PLURIBUS_ROOT = os.environ.get("PLURIBUS_ROOT", "/pluribus")
BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", f"{PLURIBUS_ROOT}/.pluribus/bus")
SPECS_DIR = f"{PLURIBUS_ROOT}/nucleus/specs"

# Registry files
CITIZEN_MD = f"{SPECS_DIR}/CITIZEN.md"
CAGENT_REGISTRY = f"{SPECS_DIR}/cagent_registry.json"
CAGENT_ADAPTATIONS = f"{SPECS_DIR}/cagent_adaptations.json"
CAGENT_PATHS = f"{SPECS_DIR}/cagent_paths.json"


def emit_bus_event(topic: str, kind: str, data: dict, level: str = "info"):
    """Emit event to Pluribus bus."""
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", "cagent_bootstrap"),
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "data": data,
    }
    bus_file = f"{BUS_DIR}/events.ndjson"
    try:
        with open(bus_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"Warning: Could not emit to bus: {e}", file=sys.stderr)
    return event["id"]


def load_json_file(path: str) -> dict:
    """Load JSON file with error handling."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {path} not found", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path}: {e}", file=sys.stderr)
        return {}


def resolve_citizen_class(actor: str, requested_class: Optional[str] = None) -> dict:
    """
    Resolve citizen class from registry.

    Returns dict with: citizen_class, citizen_tier, bootstrap_profile, scope_allowlist
    """
    registry = load_json_file(CAGENT_REGISTRY)
    defaults = registry.get("defaults", {
        "citizen_class": "superworker",
        "citizen_tier": "limited",
        "bootstrap_profile": "minimal",
        "scope_allowlist": []
    })

    # Find actor in registry
    actors = registry.get("actors", [])
    actor_config = None
    for a in actors:
        if a.get("actor") == actor:
            actor_config = a
            break

    if actor_config:
        result = {
            "citizen_class": actor_config.get("citizen_class", defaults["citizen_class"]),
            "citizen_tier": actor_config.get("citizen_tier", defaults["citizen_tier"]),
            "bootstrap_profile": actor_config.get("bootstrap_profile", defaults["bootstrap_profile"]),
            "scope_allowlist": actor_config.get("scope_allowlist", defaults["scope_allowlist"]),
        }
    else:
        result = defaults.copy()

    # Override with requested class if provided
    if requested_class:
        class_aliases = registry.get("class_aliases", {})
        normalized = class_aliases.get(requested_class.lower(), requested_class.lower())
        result["citizen_class"] = normalized
        if normalized == "superagent":
            result["citizen_tier"] = "full"
            result["bootstrap_profile"] = "full"
        else:
            result["citizen_tier"] = "limited"
            result["bootstrap_profile"] = "minimal"

    return result


def load_adaptations(actor: str) -> dict:
    """Load model-specific adaptations."""
    adaptations = load_json_file(CAGENT_ADAPTATIONS)
    agent_adapt = adaptations.get("adaptations", {}).get(actor, {})
    if not agent_adapt:
        agent_adapt = adaptations.get("defaults", {}).get("unknown_agent", {})
    return agent_adapt


def compute_constitution_hash() -> str:
    """Compute SHA256 hash of CITIZEN.md."""
    try:
        with open(CITIZEN_MD, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except FileNotFoundError:
        return "NOT_FOUND"


def step_1_identity_resolution(actor: str, actor_version: str) -> dict:
    """Step 1: Identity Resolution."""
    os.environ["PLURIBUS_ACTOR"] = actor
    os.environ["PLURIBUS_ACTOR_VERSION"] = actor_version

    return {
        "step": 1,
        "name": "identity_resolution",
        "actor": actor,
        "actor_version": actor_version,
        "status": "complete"
    }


def step_2_classification_binding(actor: str, requested_class: Optional[str], scope: Optional[str]) -> dict:
    """Step 2: Classification Binding."""
    classification = resolve_citizen_class(actor, requested_class)

    # Override scope if provided
    if scope:
        classification["scope_allowlist"] = [s.strip() for s in scope.split(",")]

    # Set environment variables
    os.environ["PLURIBUS_CAGENT_CLASS"] = classification["citizen_class"]
    os.environ["PLURIBUS_CITIZEN_TIER"] = classification["citizen_tier"]
    os.environ["PLURIBUS_BOOTSTRAP_PROFILE"] = classification["bootstrap_profile"]
    if classification["scope_allowlist"]:
        os.environ["PLURIBUS_SCOPE_ALLOWLIST"] = ",".join(classification["scope_allowlist"])

    return {
        "step": 2,
        "name": "classification_binding",
        **classification,
        "status": "complete"
    }


def step_3_path_canonicalization() -> dict:
    """Step 3: Path Canonicalization."""
    paths = load_json_file(CAGENT_PATHS)
    roots = paths.get("roots", {})

    # Set root environment variables
    for key, value in roots.items():
        os.environ[key] = value

    # Verify critical paths exist
    critical_paths = [
        CITIZEN_MD,
        CAGENT_REGISTRY,
        CAGENT_ADAPTATIONS,
        f"{BUS_DIR}/events.ndjson"
    ]

    verified = []
    missing = []
    for p in critical_paths:
        if os.path.exists(p):
            verified.append(p)
        else:
            missing.append(p)

    return {
        "step": 3,
        "name": "path_canonicalization",
        "verified_count": len(verified),
        "missing": missing,
        "status": "complete" if not missing else "warning"
    }


def step_4_constitution_loading() -> dict:
    """Step 4: Constitution Loading."""
    constitution_hash = compute_constitution_hash()

    # Read principles count
    principles_count = 0
    try:
        with open(CITIZEN_MD) as f:
            content = f.read()
            principles_count = content.count("### ")
    except FileNotFoundError:
        pass

    emit_bus_event("cagent.constitution.loaded", "log", {
        "path": CITIZEN_MD,
        "hash": constitution_hash,
        "principles": principles_count
    })

    return {
        "step": 4,
        "name": "constitution_loading",
        "constitution_hash": constitution_hash,
        "principles_loaded": principles_count,
        "status": "complete"
    }


def step_5_persona_binding(actor: str, persona: Optional[str]) -> dict:
    """Step 5: Persona Binding."""
    adaptations = load_adaptations(actor)

    bound_persona = persona or adaptations.get("archetype", "Worker")

    emit_bus_event("cagent.persona.bound", "log", {
        "actor": actor,
        "persona": bound_persona,
        "archetype": adaptations.get("archetype"),
        "preferred_lanes": adaptations.get("preferred_lanes", [])
    })

    return {
        "step": 5,
        "name": "persona_binding",
        "persona": bound_persona,
        "archetype": adaptations.get("archetype"),
        "sampling": adaptations.get("sampling", {}),
        "status": "complete"
    }


def step_6_skills_loading(actor: str) -> dict:
    """Step 6: Skills Loading."""
    agent_home = os.environ.get("PLURIBUS_HOME", f"{PLURIBUS_ROOT}/.pluribus/agent_homes/{actor}")
    skills_dir = f"{agent_home}/skills"

    skills_loaded = []
    if os.path.isdir(skills_dir):
        for skill_file in os.listdir(skills_dir):
            if skill_file.endswith(".md") or skill_file.endswith(".json"):
                skills_loaded.append(skill_file)

    return {
        "step": 6,
        "name": "skills_loading",
        "skills_dir": skills_dir,
        "skills_loaded": skills_loaded,
        "count": len(skills_loaded),
        "status": "complete"
    }


def step_7_hooks_registration(actor: str) -> dict:
    """Step 7: Hooks Registration."""
    adaptations = load_adaptations(actor)
    special_hooks = adaptations.get("special_hooks", [])

    emit_bus_event("cagent.hooks.registered", "log", {
        "actor": actor,
        "hooks": special_hooks
    })

    return {
        "step": 7,
        "name": "hooks_registration",
        "hooks_registered": special_hooks,
        "status": "complete"
    }


def step_8_protocol_handshake(manifest: dict) -> dict:
    """Step 8: Protocol Handshake."""
    # Emit citizen ready event
    event_id = emit_bus_event("cagent.citizen.ready", "artifact", manifest)

    return {
        "step": 8,
        "name": "protocol_handshake",
        "event_id": event_id,
        "status": "complete"
    }


def compute_phi_score(steps: list) -> float:
    """Compute compliance score (φ) from bootstrap steps."""
    total = len(steps)
    complete = sum(1 for s in steps if s.get("status") == "complete")
    return round(complete / total, 2) if total > 0 else 0.0


def bootstrap(
    actor: str,
    actor_version: str = "unknown",
    citizen_class: Optional[str] = None,
    persona: Optional[str] = None,
    scope: Optional[str] = None,
    emit_only: bool = False
) -> dict:
    """
    Execute full CAGENT bootstrap sequence.

    Args:
        actor: Agent identifier (claude, codex, gemini, etc.)
        actor_version: Model version string
        citizen_class: Override citizen class (sagent/swagent)
        persona: Override persona binding
        scope: Comma-separated scope allowlist for superworkers
        emit_only: Only emit events, don't set env vars

    Returns:
        Bootstrap manifest with all step results
    """
    steps = []

    # Step 1: Identity Resolution
    steps.append(step_1_identity_resolution(actor, actor_version))

    # Step 2: Classification Binding
    step2 = step_2_classification_binding(actor, citizen_class, scope)
    steps.append(step2)

    # Step 3: Path Canonicalization
    steps.append(step_3_path_canonicalization())

    # Step 4: Constitution Loading
    steps.append(step_4_constitution_loading())

    # Step 5: Persona Binding
    steps.append(step_5_persona_binding(actor, persona))

    # Step 6: Skills Loading
    steps.append(step_6_skills_loading(actor))

    # Step 7: Hooks Registration
    steps.append(step_7_hooks_registration(actor))

    # Compute φ score
    phi_score = compute_phi_score(steps)

    # Build manifest
    manifest = {
        "citizen_version": "1.0.0",
        "dkin_version": "v28",
        "actor": actor,
        "actor_version": actor_version,
        "citizen_class": step2["citizen_class"],
        "citizen_tier": step2["citizen_tier"],
        "bootstrap_profile": step2["bootstrap_profile"],
        "scope_allowlist": step2.get("scope_allowlist", []),
        "constitution_hash": steps[3].get("constitution_hash"),
        "persona": steps[4].get("persona"),
        "archetype": steps[4].get("archetype"),
        "sampling": steps[4].get("sampling", {}),
        "skills_loaded": steps[5].get("skills_loaded", []),
        "hooks_registered": steps[6].get("hooks_registered", []),
        "phi_score": phi_score,
        "steps": steps,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Step 8: Protocol Handshake
    steps.append(step_8_protocol_handshake(manifest))

    # Update manifest with final step
    manifest["steps"] = steps
    manifest["bootstrap_complete"] = True

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="CAGENT Bootstrap Daemon - Citizen Agent Initialization"
    )
    parser.add_argument("--actor", required=True, help="Agent identifier")
    parser.add_argument("--version", default="unknown", help="Agent version")
    parser.add_argument("--class", dest="citizen_class",
                        choices=["sagent", "swagent", "superagent", "superworker"],
                        help="Override citizen class")
    parser.add_argument("--persona", help="Override persona binding")
    parser.add_argument("--scope", help="Scope allowlist (comma-separated)")
    parser.add_argument("--json", action="store_true", help="Output JSON manifest")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    manifest = bootstrap(
        actor=args.actor,
        actor_version=args.version,
        citizen_class=args.citizen_class,
        persona=args.persona,
        scope=args.scope
    )

    if args.json:
        print(json.dumps(manifest, indent=2))
    elif not args.quiet:
        print(f"CAGENT Bootstrap Complete")
        print(f"  Actor: {manifest['actor']} ({manifest['actor_version']})")
        print(f"  Class: {manifest['citizen_class']} ({manifest['citizen_tier']})")
        print(f"  Profile: {manifest['bootstrap_profile']}")
        print(f"  Persona: {manifest['persona']} ({manifest['archetype']})")
        print(f"  φ Score: {manifest['phi_score']}")
        print(f"  Constitution: {manifest['constitution_hash']}")
        print(f"  Skills: {len(manifest['skills_loaded'])}")
        print(f"  Hooks: {manifest['hooks_registered']}")

    # Exit with appropriate code
    sys.exit(0 if manifest["phi_score"] >= 0.7 else 1)


if __name__ == "__main__":
    main()


# ============================================================================
# Attestation Header Injection (v2 - Visual Format)
# ============================================================================

def inject_attestation_prompt(agent_id: str, scope: str = "session_start") -> str:
    """
    Generate attestation header instruction to inject into agent system prompt.
    
    This ensures every agent response starts with a visual attestation header.
    """
    from nucleus.tools.attestation_render import render_attestation
    
    example_header = render_attestation(agent_id, score=100, compact=True)
    
    instruction = f"""
# REQUIRED: Attestation Header (Every Response)

You MUST begin EVERY response with a visual attestation header.
Use this format:

{example_header}

Rules:
- Start EVERY response with this header (compact format)
- Update score based on your confidence in protocol compliance
- Score 100% = fully compliant, 80% = minor issues, <50% = degraded
- After header, add one blank line, then your response

Example response structure:
```
⟦PLURIBUS⟧ {agent_id} │ DKIN:v28 PAIP:v15 │ ██████████ 100% │ ✓2025-12-30

[Your actual response here...]
```
"""
    return instruction
