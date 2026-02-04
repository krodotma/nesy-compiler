#!/usr/bin/env python3
"""
swarm_dashboard.py - Enhanced Tmux Swarm Dashboard with Dialogos Monitor (v1.1.2)
"""
import subprocess
import sys
import os
import time
import shutil

VERSION = "1.1.2"
SESSION_NAME = "pluribus_dashboard"
TMUX_BIN = shutil.which("tmux") or "/usr/bin/tmux"
SYSTEM_BUS_PATH = "/pluribus/.pluribus/bus/events.ndjson"

CLI_BINS = {
    "claude": shutil.which("claude") or "/usr/local/bin/claude",
    "gemini": shutil.which("gemini") or "/usr/bin/gemini",
    "codex": shutil.which("codex") or "/usr/local/bin/codex",
}

def run_tmux(args):
    res = subprocess.run([TMUX_BIN] + args, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: tmux {' '.join(args)} -> {res.stderr.strip()}")
    return res.stdout.strip()

def setup_pane(idx, cmd):
    run_tmux(["send-keys", "-t", f"{SESSION_NAME}:dashboard.{idx}", cmd, "Enter"])

def spawn_in_pane(pane_index, model, prompt, working_dir, loop_count=1):
    cli_bin = CLI_BINS.get(model.lower(), CLI_BINS["claude"])
    
    # CLI-specific safe flags
    safe_prompt = prompt.replace("'", "'\\''")
    if "codex" in model.lower():
        # Codex: exec subcommand + dangerous bypass 
        # Attempting as root. If this fails, we genuinely need a user, but we'd need to cp creds.
        cmd_str = f"{cli_bin} exec '{safe_prompt}' --dangerously-bypass-approvals-and-sandbox"
    elif "gemini" in model.lower():
        # Gemini: Revert to known working -y from Step 1467, plus --no-sandbox if possible
        # Step 1467 output showed "YOLO mode enabled" with just -y.
        cmd_str = f"{cli_bin} '{safe_prompt}' -y --no-sandbox"
    else:
        # Claude: Remove dangerous flag to allow root execution.
        # Omega is mostly coordinating/reading in this demo.
        cmd_str = f"{cli_bin} -p '{safe_prompt}'"
    
    # Run as current user (root)
    full_cmd = f"cd {working_dir} && {cmd_str}"
    
    if loop_count > 1:
        # Wrap in shell loop
        full_cmd = f"cd {working_dir} && for i in $(seq 1 {loop_count}); do echo \"\\033[1;33m--- ITERATION $i/{loop_count} ---\\033[0m\"; {cmd_str}; sleep 3; done"

    setup_pane(pane_index, full_cmd)

def create_layout(working_dir="/pluribus"):
    if subprocess.run([TMUX_BIN, "has-session", "-t", SESSION_NAME], capture_output=True).returncode == 0:
        run_tmux(["kill-session", "-t", SESSION_NAME])
    
    # 1. Create session (Pane 0)
    run_tmux(["new-session", "-d", "-s", SESSION_NAME, "-n", "dashboard", "-c", working_dir, "-x", "200", "-y", "60"])
    time.sleep(2.0)
    
    # 2. Split Right (Pane 1) - Monitor
    run_tmux(["split-window", "-t", f"{SESSION_NAME}:dashboard", "-h", "-l", "35%", "-c", working_dir])
    
    # 3. Select Left (Pane 0) and Split Down 
    run_tmux(["select-pane", "-t", f"{SESSION_NAME}:dashboard.0"])
    run_tmux(["split-window", "-t", f"{SESSION_NAME}:dashboard.0", "-v", "-c", working_dir])
    
    # 4. Select Bot Left (Pane 1) and Split Right
    run_tmux(["select-pane", "-t", f"{SESSION_NAME}:dashboard.1"])
    run_tmux(["split-window", "-t", f"{SESSION_NAME}:dashboard.1", "-h", "-c", working_dir])
    
    # Setup Titles
    setup_pane(0, f"cd {working_dir} && clear && echo '=== OMEGA CLAUDE MONITOR (LOOPING 5x) ==='")
    setup_pane(1, f"cd {working_dir} && clear && echo '=== GEMINI WORKER ==='")
    setup_pane(2, f"cd {working_dir} && clear && echo '=== CODEX WORKER ==='")
    
    # Monitor (run as root is fine for reading bus)
    monitor_cmd = (
        f"clear && echo '╔══════════════════════════════╗' && "
        f"echo '║  PBLOOP / LIVE MONITOR       ║' && "
        f"echo '╠══════════════════════════════╣' && "
        f"echo '║ [P]ause  [R]esume  [S]top    ║' && "
        f"echo '╚══════════════════════════════╝' && "
        f"echo '' && echo 'Monitoring {SYSTEM_BUS_PATH}...' && "
        f"tail -f {SYSTEM_BUS_PATH} 2>/dev/null | jq -c 'select(.topic | test(\"dialogos|swarm|agent\")) | {{ts: .iso[:19], topic: .topic, actor: .actor}}'"
    )
    setup_pane(3, monitor_cmd)

def run_demo():
    demo_dir = f"/tmp/pluribus_test_{int(time.time())}"
    os.makedirs(demo_dir, exist_ok=True)
    subprocess.run(["git", "init", demo_dir], check=False)
    
    with open(f"{demo_dir}/PBLIVETEST.md", "w") as f: 
        f.write("# PBLIVETEST ACTIVE\n\nGoal: Create a python calculate module and a test for it.")
    
    create_layout(demo_dir)
    print(f"[PBLIVETEST] Layout created. Spawning agents (as root)...")
    time.sleep(1)
    
    # OMEGA: Loop 5 times
    omega_prompt = (
        "You are OMEGA (Omega Orchestrator). Tasks in current dir. "
        "1. Check if calc.py exists. "
        "2. Check if test_calc.py exists. "
        "3. If files missing, instruct user (simulated) to wait. "
        "4. If files exist, review them. "
        "Report status."
    )
    spawn_in_pane(0, "claude", omega_prompt, demo_dir, loop_count=5)
    
    # GEMINI: Create calc.py
    gemini_prompt = "You are GEMINI. Create 'calc.py' with add/sub functions."
    spawn_in_pane(1, "gemini", gemini_prompt, demo_dir)
    
    # CODEX: Create test_calc.py
    codex_prompt = "You are CODEX. Create 'test_calc.py' using unittest for calc.py."
    spawn_in_pane(2, "codex", codex_prompt, demo_dir)
    
    print(f"[PBLIVETEST] Demo running in {demo_dir}")
    print(f"[PBLIVETEST] Attach: tmux attach-session -t {SESSION_NAME}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        create_layout()
