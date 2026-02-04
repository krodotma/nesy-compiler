#!/usr/bin/env python3
"""
Crush Operator - DKIN v19 Semops Integration

Provides operator-level controls for Crush CLI sessions:
- Trigger LLM queries from semops
- Coordinate with other agents via bus events
- Support PBLOCK/PAIP protocol compliance

Semops Triggers:
    operator.crush.query   : Submit a query to crush
    operator.crush.review  : Code review via crush
    operator.crush.explain : Explain code/concept
    operator.crush.refactor: Suggest refactoring

Bus Topics:
    operator.crush.request  : Query request received
    operator.crush.response : Query response
    operator.crush.status   : Operator status
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

# Import crush adapter
sys.path.insert(0, str(Path(__file__).parent))
from crush_adapter import emit_event, run_crush

PROTOCOL_VERSION = "v19"
ACTOR = "operator/crush"


def handle_query(
    prompt: str,
    context: Optional[dict] = None,
    model: Optional[str] = None,
    api: Optional[str] = None,
    req_id: Optional[str] = None,
) -> dict:
    """Handle a crush query from semops trigger."""
    req_id = req_id or str(uuid.uuid4())[:8]

    emit_event("operator.crush.request", {
        "intent": "query",
        "prompt_preview": prompt[:100],
        "context_keys": list(context.keys()) if context else [],
        "model": model,
    }, req_id=req_id)

    # Build full prompt with context
    full_prompt = prompt
    if context:
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        full_prompt = f"Context:\n{context_str}\n\nQuery: {prompt}"

    exit_code, output = run_crush(
        full_prompt,
        model=model,
        api=api,
        raw=True,
    )

    result = {
        "ok": exit_code == 0,
        "output": output,
        "req_id": req_id,
        "protocol_version": PROTOCOL_VERSION,
    }

    emit_event("operator.crush.response", {
        "intent": "query",
        "ok": exit_code == 0,
        "output_length": len(output),
        "output_preview": output[:200] if output else "",
    }, level="info" if exit_code == 0 else "error", req_id=req_id)

    return result


def handle_code_review(
    file_path: str,
    focus: Optional[str] = None,
    model: Optional[str] = None,
    req_id: Optional[str] = None,
) -> dict:
    """Review code in a file."""
    req_id = req_id or str(uuid.uuid4())[:8]

    emit_event("operator.crush.request", {
        "intent": "review",
        "file": file_path,
        "focus": focus,
        "model": model,
    }, req_id=req_id)

    try:
        with open(file_path, "r") as f:
            code = f.read()
    except Exception as e:
        emit_event("operator.crush.response", {
            "intent": "review",
            "ok": False,
            "error": str(e),
        }, level="error", req_id=req_id)
        return {"ok": False, "error": str(e), "req_id": req_id}

    prompt = f"""Review this code for:
1. Bugs and potential issues
2. Security vulnerabilities
3. Performance improvements
4. Code style and best practices
{f'Focus on: {focus}' if focus else ''}

```
{code}
```"""

    exit_code, output = run_crush(prompt, model=model, raw=True)

    result = {
        "ok": exit_code == 0,
        "review": output,
        "file": file_path,
        "req_id": req_id,
    }

    emit_event("operator.crush.response", {
        "intent": "review",
        "ok": exit_code == 0,
        "file": file_path,
        "output_length": len(output),
    }, level="info" if exit_code == 0 else "error", req_id=req_id)

    return result


def handle_explain(
    target: str,
    depth: str = "medium",
    model: Optional[str] = None,
    req_id: Optional[str] = None,
) -> dict:
    """Explain code or concept."""
    req_id = req_id or str(uuid.uuid4())[:8]

    emit_event("operator.crush.request", {
        "intent": "explain",
        "target": target[:100],
        "depth": depth,
        "model": model,
    }, req_id=req_id)

    depth_instructions = {
        "brief": "Give a brief 2-3 sentence explanation.",
        "medium": "Explain in moderate detail with examples.",
        "deep": "Provide a comprehensive explanation with examples, edge cases, and related concepts.",
    }

    prompt = f"""{depth_instructions.get(depth, depth_instructions['medium'])}

Explain: {target}"""

    exit_code, output = run_crush(prompt, model=model, raw=True)

    result = {
        "ok": exit_code == 0,
        "explanation": output,
        "depth": depth,
        "req_id": req_id,
    }

    emit_event("operator.crush.response", {
        "intent": "explain",
        "ok": exit_code == 0,
        "depth": depth,
        "output_length": len(output),
    }, level="info" if exit_code == 0 else "error", req_id=req_id)

    return result


def handle_refactor(
    file_path: str,
    goal: Optional[str] = None,
    model: Optional[str] = None,
    req_id: Optional[str] = None,
) -> dict:
    """Suggest refactoring for code."""
    req_id = req_id or str(uuid.uuid4())[:8]

    emit_event("operator.crush.request", {
        "intent": "refactor",
        "file": file_path,
        "goal": goal,
        "model": model,
    }, req_id=req_id)

    try:
        with open(file_path, "r") as f:
            code = f.read()
    except Exception as e:
        emit_event("operator.crush.response", {
            "intent": "refactor",
            "ok": False,
            "error": str(e),
        }, level="error", req_id=req_id)
        return {"ok": False, "error": str(e), "req_id": req_id}

    prompt = f"""Suggest refactoring for this code.
{f'Goal: {goal}' if goal else 'Focus on: readability, maintainability, and modern patterns.'}

Provide:
1. Specific refactoring suggestions
2. The refactored code
3. Explanation of improvements

```
{code}
```"""

    exit_code, output = run_crush(prompt, model=model, raw=True)

    result = {
        "ok": exit_code == 0,
        "suggestions": output,
        "file": file_path,
        "goal": goal,
        "req_id": req_id,
    }

    emit_event("operator.crush.response", {
        "intent": "refactor",
        "ok": exit_code == 0,
        "file": file_path,
        "output_length": len(output),
    }, level="info" if exit_code == 0 else "error", req_id=req_id)

    return result


def process_semops_trigger(trigger: dict) -> dict:
    """Process a semops trigger event."""
    intent = trigger.get("intent", "query")
    req_id = trigger.get("req_id", str(uuid.uuid4())[:8])
    model = trigger.get("model")
    api = trigger.get("api")

    if intent == "query":
        return handle_query(
            prompt=trigger.get("prompt", ""),
            context=trigger.get("context"),
            model=model,
            api=api,
            req_id=req_id,
        )
    elif intent == "review":
        return handle_code_review(
            file_path=trigger.get("file", ""),
            focus=trigger.get("focus"),
            model=model,
            req_id=req_id,
        )
    elif intent == "explain":
        return handle_explain(
            target=trigger.get("target", ""),
            depth=trigger.get("depth", "medium"),
            model=model,
            req_id=req_id,
        )
    elif intent == "refactor":
        return handle_refactor(
            file_path=trigger.get("file", ""),
            goal=trigger.get("goal"),
            model=model,
            req_id=req_id,
        )
    else:
        return {"ok": False, "error": f"Unknown intent: {intent}", "req_id": req_id}


def emit_status() -> None:
    """Emit operator status to bus."""
    import subprocess

    # Check crush/mods version
    result = subprocess.run(["/usr/local/bin/crush", "--version"], capture_output=True, text=True)
    crush_version = result.stdout.strip() if result.returncode == 0 else "unknown"

    # Check glow version
    result = subprocess.run(["glow", "--version"], capture_output=True, text=True)
    glow_version = result.stdout.strip() if result.returncode == 0 else "not installed"

    emit_event("operator.crush.status", {
        "online": True,
        "crush_version": crush_version,
        "glow_version": glow_version,
        "protocol_version": PROTOCOL_VERSION,
        "capabilities": ["query", "review", "explain", "refactor"],
    })


def main():
    parser = argparse.ArgumentParser(description="Crush Operator - DKIN v19 Semops Integration")
    parser.add_argument("--status", action="store_true", help="Emit status and exit")
    parser.add_argument("--trigger", type=str, help="Process semops trigger (JSON)")
    parser.add_argument("--query", type=str, help="Direct query")
    parser.add_argument("--review", type=str, help="Review file")
    parser.add_argument("--explain", type=str, help="Explain target")
    parser.add_argument("--refactor", type=str, help="Refactor file")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--goal", type=str, help="Goal/focus for operation")
    parser.add_argument("--depth", type=str, default="medium", help="Explanation depth")

    args = parser.parse_args()

    if args.status:
        emit_status()
        return

    if args.trigger:
        try:
            trigger = json.loads(args.trigger)
            result = process_semops_trigger(trigger)
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"Invalid JSON: {e}"}))
            sys.exit(1)
        return

    if args.query:
        result = handle_query(args.query, model=args.model)
        print(result.get("output", result.get("error", "")))
        return

    if args.review:
        result = handle_code_review(args.review, focus=args.goal, model=args.model)
        print(result.get("review", result.get("error", "")))
        return

    if args.explain:
        result = handle_explain(args.explain, depth=args.depth, model=args.model)
        print(result.get("explanation", result.get("error", "")))
        return

    if args.refactor:
        result = handle_refactor(args.refactor, goal=args.goal, model=args.model)
        print(result.get("suggestions", result.get("error", "")))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
