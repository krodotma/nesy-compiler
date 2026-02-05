# PBTSO TOOLING RESTORATION CONTRACT
**Status:** REQUIRED
**Owner:** Gemini-Subagent-1 (Toolsmith)
**Constraint:** NO MOCKS. NO STUBS. NO SIMULATIONS.

## 1. THE OBJECTIVE
Restore `nucleus/tools/pli.py` and `nucleus/tools/tmux_swarm_orchestrator.py` to a fully functional state where:
1.  `pli monitor` launches a working TUI (using `bus_monitor_tui.py`).
2.  `pli iterate --start` correctly triggers the orchestrator AND registers the swarm in `tbtso_a2a.py`.
3.  `pli swarm <manifest>` spawns actual processes that persist.

## 2. THE TEST CONTRACT
The fix is ONLY accepted if the following sequence executes successfully on the live system:

```bash
# 1. Spawn a test swarm
python3 nucleus/tools/pli.py swarm nucleus/manifests/tooling_test.json

# 2. Verify Liveness (Must show PIDs)
python3 nucleus/tools/pli.py a2a --status --json | grep "running"

# 3. Trigger Iteration (Must emit event AND update state)
python3 nucleus/tools/pli.py iterate --start --swarm tooling-test-swarm --scope "Verification"

# 4. Monitor (Must not crash)
python3 nucleus/tools/pli.py monitor --timeout 5
```

## 3. IMPLEMENTATION REQUIREMENTS
1.  **Merge:** You must merge the "Rich" legacy `pli.py` commands with the "New" `iterate/monitor` arguments. Do not overwrite one with the other.
2.  **Pathing:** Ensure `sys.path` correctly includes `nucleus/tools` so `agent_bus` imports work.
3.  **Argparse:** Fix the subcommand conflict in `pli.py`.

## 4. EXECUTION
- **Analyze:** Read `pli.py`, `tmux_swarm_orchestrator.py`, `bus_monitor_tui.py`.
- **Fix:** Rewrite `pli.py` to satisfy the contract.
- **Verify:** Run the sequence in Section 2.
- **Report:** Emit `tooling.restoration.complete`.
