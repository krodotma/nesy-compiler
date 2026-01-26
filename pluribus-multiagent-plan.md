# Pluribus v1 Normalization: 50-Step Multi-Agent Execution Plan

**Goal:** Establish a fully sequestered, multi-agent workflow to normalize Pluribus v1 specs, integrate ISO tooling, and produce a formal whitepaper.
**Agents:** Opus 1 (Architect), Gemini Pro 3 (Spec/Docs), Codex 5.2 (Tooling), Opus 2 (QA/Sim).
**Constraints:** Sequestered PAIP clones, no localhost, strict log hygiene, iso_git commits only.

## Phase 1: Initialization & Sequester (Steps 1-10)
**Owner:** Opus 1 (Architect)
1.  Verify host environment health (IO, memory, disk).
2.  Provision PAIP temp clones for Opus 1, Gemini, and Codex.
3.  Establish `bus.sub` channels for inter-agent coordination.
4.  Generate `UNIFORM v2.1` headers for all agent sessions.
5.  Audit current `GEMINI.md`, `CLAUDE.md`, `CODEX.md` for v1 compliance.
6.  Sequester Opus 1: Clone `nucleus` to `/tmp/paip-opus-1`.
7.  Sequester Gemini: Clone `nucleus` to `/tmp/paip-gemini`.
8.  Sequester Codex: Clone `nucleus` to `/tmp/paip-codex`.
9.  Run `agent_bootstrap_doctor.py` in each clone.
10. Broadcast `sys.ready` event to bus with clone paths.

## Phase 2: Spec Normalization (Steps 11-20)
**Owner:** Gemini Pro 3 (Spec/Docs)
11. Ingest `pluribus-v1-normalization-plan.md`.
12. Scan all `*.md` files in `nucleus/specs` for deprecated v0 terminology.
13. Update `repl_header_contract_v1.md` to final `UNIFORM v2.1` syntax.
14. Normalize `AGENTS.md` and `nucleus/AGENTS.md`.
15. Draft `PLURIBUS_PROTOCOL_V1_WHITEPAPER.md` skeleton.
16. Integrate `Antigravity Safety Rails` into standard `specs/SAFETY.md`.
17. Verify no "Antigravity" references remain in active spec files (archive only).
18. Run `pbdocs_link_checker.py` on normalized specs.
19. Commit spec updates to `feature/pluribus-v1-specs` (via iso_git).
20. Handoff to Codex for tooling updates.

## Phase 3: Tooling & ISO Integration (Steps 21-30)
**Owner:** Codex 5.2 (Tooling)
21. Ingest updated `repl_header_contract_v1.md`.
22. Refactor `agent_header.py` to enforce `UNIFORM v2.1`.
23. Update `iso_git.mjs` to require `iso_pqc.mjs` strictly.
24. Implement `log_hygiene.py` hooks into `agent_wrapper_common.sh`.
25. Create `iso_wrapper.sh` to alias `git` to `iso_git.mjs` in sequestered envs.
26. Verify `node nucleus/tools/iso_git.mjs status` in strict mode.
27. Update `agent_bus.py` to emit `v1` protocol events.
28. Run unit tests for `agent_header.py` and `iso_git.mjs`.
29. Commit tooling updates to `feature/pluribus-v1-tooling`.
30. Handoff to Opus 2 for verification.

## Phase 4: Verification & Simulation (Steps 31-40)
**Owner:** Opus 2 (QA/Sim)
31. Merge `feature/pluribus-v1-specs` and `feature/pluribus-v1-tooling` into `dev`.
32. Simulate a full agent bootstrap sequence (init -> ready -> task).
33. Verify `UNIFORM` panel output matches spec exactly.
34. Verify `iso_git` rejects non-signed commits.
35. Verify `log_hygiene` prevents log bloat during simulation.
36. Run `du -sm` check (mocked if necessary) to confirm cleanup.
37. Test "Antigravity Safety Rails" by attempting a large find/grep (expect block).
38. Compile verification report `pluribus-v1-verification.md`.
39. Flag any regressions to Gemini/Codex.
40. Sign off on V1 Release Candidate.

## Phase 5: Finalization & Delivery (Steps 41-50)
**Owner:** Gemini Pro 3 (Finalizer)
41. Aggregate all findings into `pluribus-normalization-iter.md`.
42. Update `progress.md` and `task_plan.md` to "Complete".
43. Finalize `PLURIBUS_PROTOCOL_V1_WHITEPAPER.md`.
44. Create final `iso_git` commit: "feat(release): Pluribus v1 Normalization".
45. Tag commit `v1.0.0-pluribus`.
46. Push to remote `origin/main`.
47. Archive old v0 specs to `nucleus/specs/archive/v0`.
48. Clean up `/tmp/paip-*` clones.
49. Broadcast `sys.release` event to bus.
50. Hibernate session.
