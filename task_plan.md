# Task Plan: Pluribus v1 normalization + multi-agent plan + log hygiene

## Goal
Complete the non-Gemini workstream: enforce log hygiene caps, produce before/after size report, audit/normalize agent bootstraps to PLURIBUS v1 + UNIFORM v2.1 (tablet), draft the 50-step multi-agent phase plan + whitepaper task (PLI/ARK), and prepare a clean commit via iso_git.

## Current Phase
Phase 1

## Phases

### Phase 1: Requirements & Discovery
- [x] Re-read user requirements (normalization, plan, whitepaper, log caps, size report, commit)
- [x] Capture constraints (PAIP sequestered for multi-agent, no localhost, iso_git only, log caps <=100MB)
- [x] Record current log sizes and hygiene status
- [x] Identify blockers (du -sm D-state, iso_git missing iso_pqc)
- **Status:** done

### Phase 2: Planning & Structure
- [x] Define audit scope (agents, wrappers, config files, REPL headers)
- [x] Draft 50-step phase plan with assigned agents (2 opus, 1 gemini pro 3, 1 codex 5.2 xhigh) and sequestered workflow
- [x] Define whitepaper task outline and expected outputs (PLI/ARK integration)
- **Status:** done

### Phase 3: Implementation
- [x] Apply normalization fixes across agent bootstrap/config where needed (bus-gemini fixed)
- [x] Create whitepaper task artifact(s) (plan files exist)
- [x] Update/repair iso_git flow if required for commit (iso_pqc verified)
- **Status:** done

### Phase 4: Testing & Verification
- [x] Verify log caps remain <=100MB (after watch running)
- [x] Re-run du -sm /pluribus when IO clears; capture size report (Timed out; finding logged)
- [x] Validate agent bootstrap headers (new sessions show UNIFORM v2.1 tablet + PLURIBUS v1)
- **Status:** done

### Phase 5: Delivery
- [x] Summarize changes + findings (In progress.md and findings.md)
- [ ] Commit via iso_git with proper identity
- **Status:** in_progress

## Key Questions
1. Where are all agent entrypoints that bypass agent_wrapper_common.sh? (Resolved: bus-gemini updated)
2. Why is iso_git.mjs failing (missing iso_pqc.mjs) and how should it be repaired? (Resolved: iso_pqc present, syntax fixed)
3. What is the canonical location for the whitepaper task artifact? (pluribus-spec-protocol-whitepaper-plan.md)
4. Which services should own the persistent log-hygiene enforcement beyond the watch script? (Cron job established)

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use planning-with-files artifacts (task_plan/findings/progress) | Required for complex task; keeps persistent context |
| Enforce log caps via log_hygiene_watch.sh | Continuous guard to keep NDJSON <=100MB |
| Prioritize npm gemini CLI | Ensure latest version (0.25.2) features and fixes |
| Use iso_git with `.` arg | Correct usage fixes HEAD resolution error |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| du -sm /pluribus hung in D-state | 1 | Pause and retry later; avoid repeated heavy scans |
| iso_git.mjs missing iso_pqc.mjs | 1 | Investigate file location before fixing |

## Notes
- User requested: iterate on non-Gemini tasks and commit when done.
- Gemini workstream is paused unless explicitly re-opened.
