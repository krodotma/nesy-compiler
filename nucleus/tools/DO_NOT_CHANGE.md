# DO NOT CHANGE - ORCHESTRATION KERNEL

The following files are CRITICAL for the "Hybrid Bridge" architecture (Python Orchestrator -> TypeScript Agents).
Any regression here breaks the ability to spawn agents.

## Protected Files
1.  **`nucleus/tools/pli.py`**: The CLI entry point. Must support `monitor`, `iterate --start`, `swarm`.
2.  **`nucleus/tools/tmux_swarm_orchestrator.py`**: The process manager. Must use `bus-codex` wrapper and Heredoc prompts.
3.  **`nucleus/tools/agent_wrapper_common.sh`**: The auth bridge. Must load secrets from `~/.config/agent-wrapper/keys`.

## Mandate
- **NO MOCKS**: Never replace these with stubs.
- **NO REVERT**: Do not restore "Golden" versions older than 2026-02-05 without manual review.
- **TEST FIRST**: Any change must pass `docs/specs/pbtso_tooling_contract.md`.
