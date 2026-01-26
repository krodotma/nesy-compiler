# HOLON/PLURIBUS Header → IRKG Migration (Plan)
# Date: 2026-01-26
# Owner: codex (planner/orchestrator)

## Summary
We are migrating UNIFORM/PLURIBUS header metrics away from NDJSON bus tails
to an IRKG-first event store. NDJSON becomes a bounded **DR ring only**
(100 entries max) for catastrophic recovery.

## Why
- Header metrics are frequently **inaccurate** due to tail bias and wrapper bugs.
- Per-response scans create **I/O overhead** and can destabilize iterative flows.
- System alignment requires **learning/analysis/action** to flow into IRKG,
  not into unbounded logs.

## Core Shift
**From:** per-response NDJSON tail scans  
**To:** single IRKG snapshot read (cached) + DR ring fallback

## Surface Coverage ("etc etc")
Agents, Dialogos, OHM/Omega, Providers, Dashboard/Browser/Portal processes,
Learning/Analysis/Action/Plan/Thought pipelines, IRKG graph, DR ring.

## Deliverables
1) Updated header contract (IRKG snapshot + DR ring only)
2) Semantic naming conventions for branches and agents
3) IRKG snapshot schema + DR ring spec
4) Migration plan and verification checklist

## Subagent Iteration (Opus 4.5, Ultrathink)
### Opus‑A: Architecture/Protocol
- Produce: IRKG snapshot schema, DR ring contract, header contract update
- Risk analysis and invariants

### Opus‑B: Integration/Migration
- Map producers/consumers across surfaces
- Dual‑write/deprecation timeline
- Verification + rollback plan

## Phases
1) **Discovery**: confirm IRKG store + schema authority
2) **Schema**: define snapshot + DR ring formats
3) **Migration**: dual‑write strategy and deprecation gates
4) **Implementation**: header generator reads snapshot only
5) **Verification**: crash recovery and performance checks

## Evidence
- `nucleus/specs/repl_header_contract_v1.md`
- `nucleus/specs/holon_semantic_naming_v1.md`
- `nucleus/specs/pluribus_lexicon.md`
- `nucleus/docs/isomorphic_git_reference.md`
- `nucleus/docs/agents/index.md`
