# Opus 4.5 Subagent Prompts (Ultrathink)

## Opus-A (Architecture/Protocol)
Goal: Produce IRKG snapshot schema + DR ring contract + header contract revisions.
Focus:
- Replace NDJSON tailing with IRKG snapshot read
- Define DR ring (100-cap) format + recovery rules
- Protocol invariants and risks
Deliverables:
- `opusA_iter01.md` with schema + invariants
- `opusA_iter02.md` with revised protocol text + risk matrix

Suggested prompt:
```
You are Opus-A (architecture/protocol). Ultrathink.
Produce: IRKG header snapshot schema (fields + types), DR ring contract (100-cap),
and recommended edits to repl_header_contract_v1.md + UNIFORM.md.
Include risks, invariants, and mapping to IRKG nodes/edges.
```

## Opus-B (Integration/Migration)
Goal: Map producers/consumers and design migration/deprecation plan.
Focus:
- Subsystem surface map (agents, dialogos, omega, providers, dashboard, browser, portal)
- Dual-write vs cutover strategy
- Verification and rollback plan
Deliverables:
- `opusB_iter01.md` with matrix + migration stages
- `opusB_iter02.md` with verification plan + rollout gates

Suggested prompt:
```
You are Opus-B (integration/migration). Ultrathink.
Map producers/consumers, propose a dual-write migration timeline,
and produce verification + rollback plan. Reference IRKG-first direction
and NDJSON DR-only policy.
```

## GLM-A (Header Overhead & Stability)
Goal: Analyze header overhead and stability risk; propose mitigations.
Focus:
- Worst-case CPU/mem/time scaling for header scans
- Caching or snapshot strategy
- Risk, tests, rollout
Deliverables:
- `glmA_iter01.md` with analysis + mitigations
- `glmA_iter02.md` with test/rollout plan

Suggested prompt:
```
You are GLM-A subagent (analysis). Context and tasks:
- Repo: /pluribus
- Observed: agent_header.py scans bus event.ndjson files; ~257 event files, ~49MB read; ~150ms per header.
- Task ledger parse bug: ledger entries are nested under data.entry.*; agent_header expects top-level fields, causing zeros/? in header.
- Direction: move away from NDJSON except 100-entry DR ring; use IRKG snapshot files for header metrics.
- Files of interest: /pluribus/nucleus/tools/agent_header.py, /pluribus/nucleus/specs/repl_header_contract_v1.md, /pluribus/nucleus/specs/UNIFORM.md.
Deliverables (concise bullets):
1) Is header overhead likely to cause instability? Explain.
2) Worst-case CPU/mem/time overhead and scaling with event volume.
3) Specific changes to reduce overhead (caching, snapshots, incremental counters, file caps).
4) Risks, tests, and rollout notes.
```

## Codex-B (Plan & Naming Review)
Goal: Review HOLON/PLURIBUS IRKG migration plan + semantic naming spec; identify gaps.
Focus:
- Spec contradictions (UNIFORM vs repl_header_contract)
- Naming grammar alignment
- Branch/agent ID conformance rules
Deliverables:
- `codexB_iter01.md` with issues + fixes
- `codexB_iter02.md` with patch-ready recommendations

Suggested prompt:
```
You are Codex-B subagent. Goal: review HOLON/PLURIBUS IRKG migration plan + semantic naming spec; identify gaps, inconsistencies, improvements.
Context (current files):
- /pluribus/nucleus/plans/holon_pluribus_irkg_header_migration_plan.md
- /pluribus/nucleus/docs/plans/holon_irkg_header_migration.md
- /pluribus/nucleus/specs/holon_semantic_naming_v1.md
- /pluribus/nucleus/specs/pluribus_lexicon.md (naming grammar)
- /pluribus/nucleus/specs/repl_header_contract_v1.md (IRKG snapshot + DR ring)
- /pluribus/nucleus/specs/UNIFORM.md (snapshot usage)
- /pluribus/nucleus/docs/isomorphic_git_reference.md (branch naming)
- /pluribus/docs/agents/index.md (agent ID naming)
Constraints:
- Move away from NDJSON except 100-cap DR ring; IRKG snapshot is source of truth.
- Avoid hand-wavy claims; propose concrete edits or additions.
Deliverables (concise bullets):
1) Issues / gaps / contradictions.
2) Recommended fixes with exact file references.
3) Ready-to-apply change list (short).
```
