# GEMINI AGENT INSTRUCTIONS FOR PLURIBUS

## PROTOCOL ZERO: NO MOCKS, NO SIMULATIONS

**MANDATE:** You are FORBIDDEN from simulating, mocking, or "pretending" to run multi-agent swarms.
**REALITY:** If you are asked to "dispatch", "swarm", or "orchestrate", you MUST use the verified tooling:
1.  `pli swarm <manifest>`
2.  `tmux_swarm_orchestrator.py`
3.  `bus-codex` / `bus-claude` wrappers

**VIOLATION:** Summarizing a "hypothetical" swarm session is a protocol violation. You must spawn the processes, monitor the bus, and report actual `exit_code` and `files_created`.

---

## ARCHITECTURE: HYBRID NDJSON + IRKG

**DO NOT CONFUSE WRITE PATH WITH QUERY PATH.**

1.  **WRITE PATH (Append-Only):**
    *   `agent_conductor.py` writes to `task_ledger.ndjson`.
    *   Agents write to `.pluribus/bus/events.ndjson`.
    *   **NO FalkorDB code here.** This is intentional for speed and simplicity.

2.  **SYNC PATH (Background):**
    *   `falkordb_bus_events.py` (Daemon) reads NDJSON and writes to FalkorDB (Port 6380).

3.  **QUERY PATH (Structured):**
    *   `falkordb_bus_events.py timeline` queries the Graph.

**Implication:** Do not search for `falkordb` imports in `agent_conductor.py`. They don't exist.

---

## OPERATIONAL NUANCES

### 1. Authentication - NO API KEYS
*   **Codex/Claude/Gemini:** Use OAUTH via `bus-*` wrappers.
*   **GLM:** Only exception (API Key).
*   **Mechanism:** `agent_wrapper_common.sh` loads secrets from `~/.config/agent-wrapper/keys` or browser session.

### 2. Manifest Rules
*   **NO MODEL FIELDS:** Remove `"model": "..."` from manifests. Let the wrapper decide.
*   **Runners:** Use `"runner": "claude"` or `"runner": "codex"`.

### 3. Orchestration Quirks
*   **Double Tap:** Claude agents in `tmux` often need **Two Enters** (`C-m C-m`) to flush the prompt buffer. If an agent stalls at the prompt, send keys again.
*   **Heredoc:** Large prompts are passed via `cat << EOF | codex ...` (handled by `tmux_swarm_orchestrator.py`).

### 4. Default Models
*   **Codex:** `gpt-5.2-codex` (XHigh Reasoning).
*   **Claude:** `claude-4.5-opus` (UltraThink).

---

## EXECUTION CHECKLIST
1.  **Spawn:** `pli swarm manifest.json`
2.  **Kick:** `tmux send-keys ... C-m C-m` (if stalled)
3.  **Monitor:** `tail -f .pluribus/bus/events.ndjson`
4.  **Verify:** `pli a2a --status`

---

## NEW INFRASTRUCTURE (50-Step Enhancement - DO NOT BREAK)

### Circuit Breaker (quota_manager.py)
Per-provider failure isolation. After 5 consecutive failures, provider is circuit-broken for 60s.
```bash
python3 nucleus/tools/quota_manager.py circuit claude    # Check circuit status
python3 nucleus/tools/quota_manager.py reset-circuit claude  # Manual reset
```

### Exponential Backoff (agent_wrapper_common.sh)
Retries with jitter for transient failures. Used automatically by bus-run.
```bash
plu_retry --max-attempts 3 --initial-delay 1.0 -- <command>
```

### NDJSON Rotation (10MB threshold)
Auto-rotates events.ndjson at 10MB, retains 5MB tail, archives to .pluribus/bus/archive/.
```bash
python3 nucleus/tools/agent_bus.py rotate-status         # Check rotation status
python3 nucleus/tools/agent_bus.py rotate-status --force # Force rotation
```

### Secret Vault (secret_vault.py)
Unified secret management supporting env vars, SOPS, and plain files.
```bash
python3 nucleus/tools/secret_vault.py list    # List available secrets
python3 nucleus/tools/secret_vault.py check   # Check vault backends
```

### Git Worktree (git_worktree.py)
Replace PAIP temp dirs with worktrees (10x faster, shared .git).
```bash
python3 nucleus/tools/git_worktree.py create agent-task-123 main
python3 nucleus/tools/git_worktree.py list
python3 nucleus/tools/git_worktree.py cleanup 24  # Remove worktrees older than 24h
```
