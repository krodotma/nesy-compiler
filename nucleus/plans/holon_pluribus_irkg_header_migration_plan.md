# HOLON/PLURIBUS Header + IRKG Event Store Migration Plan
# Date: 2026-01-26
# Owner: codex (planner/orchestrator)
# Target Branch: evo/20260126-holon.irkg.header.plan.orch.codex

## Goal
Retire NDJSON header scans and migrate UNIFORM/PLURIBUS header metrics to the IRKG/event store,
keeping only a **100‑capped DR ring** for catastrophic recovery. Align with the system’s
autopoietic, tail‑eating Gödel machine purpose: logs are transient buffers; durable value
flows into learning, analysis, action, plan, thought, and portal processes.

## Non‑Goals (this phase)
- No production code changes.
- No schema migrations without IRKG spec confirmation.
- No changes to critical branches (main/staging/dev).

## Constraints & Principles
- **Append‑only evidence.** New artifacts should be additive.
- **IRKG‑first.** NDJSON is deprecated except for DR ring (100 entries max).
- **Protocol‑aligned.** Update UNIFORM/PLURIBUS contracts to match new storage.
- **Low overhead.** Header generation must be O(1) reads (snapshot).

## Required Inputs
1) Latest IRKG/event store specs + schema location
2) Canonical event store (FalkorDB? other) and access patterns
3) DR ring file location + ingestion contract
4) Ownership/authority for REPL header contract changes

## Subsystem Surface Map (capture “etc etc”)
- **Agents:** codex/claude/gemini/qwen/grok (CLI + wrappers)
- **Dialogos:** dialogos.* topics, operator flows
- **OHM/Omega:** liveness, guardian metrics, reflex loops
- **Providers:** web + API + CLI health and routing
- **Dashboard/Browser/Portal:** UI telemetry + portal processes
- **Learning/Analysis/Actions/Plans/Thoughts:** knowledge capture, plan DAGs, action records
- **IRKG:** graph storage (nodes/edges), lineage + telemetry
- **DR ring:** bounded local NDJSON buffer for crash recovery

## Architecture Plan (High‑Level)
1) **Event ingestion pipeline**
   - Emit → queue/buffer → IRKG write → aggregator snapshot
   - DR ring writes in parallel (100‑cap) for catastrophic recovery
2) **Header generation**
   - Read a single IRKG snapshot (or cached JSON)
   - No rglob/tail scans; no per‑topic file traversal
3) **Protocol alignment**
   - Update REPL header contract to use IRKG snapshot reads
   - Declare NDJSON tails deprecated; DR ring only

## Protocol Update Targets
- `nucleus/specs/repl_header_contract_v1.md`
- `nucleus/specs/UNIFORM.md`
- `nucleus/specs/holon_lifecycle_events.json`
- `nucleus/specs/pluribus_lexicon.md` (terminology: IRKG snapshot, DR ring)

## DR Ring Spec (draft)
- Location: `/pluribus/.pluribus/dr/header_events.ndjson`
- Cap: **100 entries** (truncate from head)
- Purpose: crash recovery ingest only
- Fields: timestamp, actor, subsystem, summary, hash pointer (IRKG node id)

## Phased Execution Plan
### Phase 0 — Discovery & Alignment
- Collect IRKG specs and event store schema
- Confirm authority for header contract changes
- Inventory current producers/consumers of header metrics

### Phase 1 — Schema & Snapshot Design
- Define IRKG schema for header rollups (A2A/OPS/QA/SYS + task/progress)
- Define snapshot document (single read, cached)
- Specify DR ring format and recovery rules

### Phase 2 — Migration Strategy
- Dual‑write plan (IRKG + DR ring; NDJSON legacy off by default)
- Deprecation timeline for NDJSON bus tails
- Backfill strategy (if needed) from IRKG only

### Phase 3 — Implementation Tasks (future)
- Replace `agent_header.py` metrics scan with snapshot read
- Introduce snapshot aggregator service
- Wire wrappers/tools to IRKG pipeline

### Phase 4 — Verification & Rollout
- Unit tests: snapshot integrity, task stats correctness
- Integration tests: crash + DR ring replay
- Performance tests: header generation latency
- Rollout: shadow mode → cutover → legacy removal

## Subagent Assignments (Claude Opus 4.5, Ultrathink)
### Opus‑A (Architecture & Protocol)
Focus: protocol update, IRKG alignment, invariants, schema.
Deliverables:
- Revised UNIFORM/REPL header contract proposal
- IRKG snapshot schema + DR ring contract
- Risk assessment

### Opus‑B (Integration & Migration)
Focus: subsystem mapping, migration timeline, integration risks.
Deliverables:
- Producer/consumer matrix
- Dual‑write & deprecation plan
- Verification plan & rollback strategy

## Iteration Cadence (Subagents)
- **Iteration artifacts** saved to: `nucleus/plans/holon_irkg_header_migration/`
  - `opusA_iter01.md`, `opusA_iter02.md`
  - `opusB_iter01.md`, `opusB_iter02.md`
- Each iteration must include:
  - assumptions, sources, proposed changes, risks
  - mapping to IRKG nodes/edges
  - explicit "next_actions"
- **Commit rhythm:** after each major spec update (protocol + naming + docs).

## Naming Conventions (new proposal)
### Branch Naming (dense semantic, entelexis‑aligned)
Format:
`evo/<YYYYMMDD>-<holon>.<domain>.<intent>.<surface>.<store>.<phase>.<actor>.<ark>`

Fields:
- `holon`: holon|pluribus|omega|dialogos|ark|irkg
- `domain`: protocol|observability|coordination|evolution|learning
- `intent`: plan|spec|migrate|impl|verify
- `surface`: header|eventstore|bus|portal|pipeline
- `store`: irkg|dr|hybrid
- `phase`: p0|p1|p2|p3|p4
- `actor`: codex|claude|gemini|qwen|grok|multi
- `ark`: `arkc<cmp>` or short ARK tag (e.g., arks0.5)

Example:
`evo/20260126-holon.observability.plan.header.irkg.p0.codex.arkc05`

### Agent Naming (semantically dense identifiers)
Format:
`<class>.<role>.<domain>.<focus>.<lane>.r<ring>.<model>.<variant>`

Fields:
- `class`: sagent|swagent|cagent
- `role`: planner|architect|auditor|engineer|operator
- `domain`: holon|pluribus|irkg|dialogos|omega
- `focus`: header|eventstore|protocol|migration
- `lane`: dialogos|ops|qa|sys
- `ring`: r0–r3
- `model`: claude-opus-4.5|gpt-5.2|gemini-3-pro|qwen-plus
- `variant`: ultrathink|fast|safe|audit

Example:
`sagent.planner.holon.irkg.header.dialogos.r0.claude-opus-4.5.ultrathink`

## Evidence & Artifacts
- This plan file
- DR ring spec draft
- Subagent outputs archived in `nucleus/plans/` as append‑only artifacts
