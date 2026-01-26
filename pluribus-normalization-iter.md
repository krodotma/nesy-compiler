# Pluribus v1 Normalization Iteration

## Goal
Normalize agent bootstraps and documentation to PLURIBUS v1/UNIFORM v2.1, deliver the 50-step whitepaper plan, and prepare a clean commit.

## Tasks
- [x] Capture requirements and constraints in planning files → Verify: `task_plan.md`, `findings.md`, `progress.md` exist.
- [x] Update normalization + whitepaper plan docs → Verify: `pluribus-v1-normalization-plan.md` and `pluribus-spec-protocol-whitepaper-plan.md` updated.
- [x] Route claude/glm aliases through PLURIBUS wrappers → Verify: `/root/.local/bin/agent-wrapper` uses `bus-claude`/`bus-glm`.
- [x] Resolve iso_git PQC dependency → Verify: `node /pluribus/nucleus/tools/iso_git.mjs status /pluribus` succeeds.
- [~] Produce size report including `du -sm /pluribus` → Verify: `log_hygiene_report.md` updated with du result (timed out due to IO, noted in report).
- [x] Produce 50-step multi-agent plan → Verify: `pluribus-multiagent-plan.md` created.
- [x] Validate bootstrap headers for all agents → Verify: new sessions show UNIFORM v2.1 tablet + PLURIBUS v1 (verified via code inspection of wrappers).
- [ ] Commit changes via iso_git → Verify: commit exists with correct message.

## Done When
- [ ] All tasks above checked and a commit is created with iso_git.

## Notes
- User requested 50-step plan (delivered separately) and sequestered execution for multi-agent work.
- Gemini workstream is paused unless explicitly re-opened.
