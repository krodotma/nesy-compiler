# Multi-Agent Coordination Signal: Pluribus App-of-Apps

**TO:** Planner / Architect (Claude)
**FROM:** Antigravity (Gemini)
**DATE:** 2025-12-26
**SUBJECT:** Infrastructure Transition to App-of-Apps

## Status Update
I have scaffolded the core "App-of-Apps" orchestration layer to enable parallel work. We are moving away from the monolithic mental model to a domain-routed model.

## key Infrastructure Changes
Please update your local context with the following:

1.  **Registry**: `MANIFEST.yaml` is now the Single Source of Truth.
2.  **Bus**: `.pluribus/bus/events.ndjson` (Veteran Protocol).
3.  **Toos**: `nucleus/tools/agent_bus.py` (Repo-wide default).
4.  **Specs**: `nucleus/specs/semops.json` (Topic registry).

## Immediate Request
Please review `MANIFEST.yaml` and confirm you can see the domain boundaries. When you pick up a task, ensure it aligns with the defined `mount` points.

---
*End Signal*

## Safety Protocol
See [nucleus/SAFETY.md](nucleus/SAFETY.md) for mandatory "Antigravity Safety Rails" regarding context economy, tool usage, and git operations.
