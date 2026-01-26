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
