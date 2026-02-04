#!/usr/bin/env python3
"""
tmux_swarm_orchestrator.py - Claude Multi-Agent Swarm via TMUX
Version: 1.1.0
Ring: 0 (Kernel)
Status: Active
Protocol: DKIN v28 / PAIP v15 / Citizen v1

Features:
- Spawns N tmux panes, each running a Claude CLI instance
- CAGENT bootstrap context injection (REPL Header, Ring, Protocol Stack)
- Ralph Loop Mode: Autonomous coding loops until completion promise (PBLOOP)
- LSP capability awareness for enhanced code intelligence

Semops:
- PBTSWARM: Multi-agent swarm orchestration
- PBLOOP: Autonomous coding loop (Ralph Wiggum pattern)

Usage:
    python3 tmux_swarm_orchestrator.py test                    # 2-agent test
    python3 tmux_swarm_orchestrator.py spawn <manifest>        # Spawn from manifest
    python3 tmux_swarm_orchestrator.py loop "<prompt>" [opts]  # Ralph loop mode
"""

import subprocess
import sys
import os
import time
import json
import re
from typing import List, Dict, Any, Optional

# Version
VERSION = "1.1.0"

# Paths (system-specific, discovered on macOS with Homebrew)
TMUX_BIN = "/opt/homebrew/bin/tmux"
CLAUDE_BIN = "/opt/homebrew/bin/claude"

# Default completion promise for Ralph loops
DEFAULT_COMPLETION_PROMISE = "DONE"
DEFAULT_MAX_ITERATIONS = 50

# LSP Configuration - Language servers that Claude Code can leverage
LSP_CAPABILITIES = {
    "python": {"server": "pyright-lsp", "features": ["go-to-definition", "find-references", "diagnostics"]},
    "typescript": {"server": "vtsls", "features": ["go-to-definition", "find-references", "diagnostics", "hover"]},
    "javascript": {"server": "vtsls", "features": ["go-to-definition", "find-references", "diagnostics"]},
    "go": {"server": "gopls", "features": ["go-to-definition", "find-references", "diagnostics"]},
    "rust": {"server": "rust-analyzer", "features": ["go-to-definition", "find-references", "diagnostics"]},
}

# CAGENT Context Template - Injected into every agent's system prompt
CAGENT_CONTEXT_TEMPLATE = """
# PLURIBUS CAGENT BOOTSTRAP CONTEXT

You are a **{agent_role}** in the Pluribus multi-agent system.

## Constitutional Adherence
- **Protocol Stack**: DKIN v28 (CAGENT) | PAIP v15 | Citizen v1
- **Ring Level**: {ring_level} ({ring_name})
- **Sextet Laws**: Lawfulness | Observability | Provenance | Recurrence | Evolvability | Boundedness

## REPL Header Contract v1
Every response MUST begin with:
```
[AGENT: {agent_id}]
[RING: {ring_level}]
[TASK: <current_task>]
[STATUS: <pending|in_progress|complete|blocked>]
```

## Working Directory
{working_dir}

## LSP Capabilities Available
{lsp_info}

## Your Assignment
{task_prompt}

---
Execute your assignment. Emit bus events for observability. Create all files with full content.
When complete, output a JSON summary: {{"files_created": [...], "status": "complete"}}
"""

# Ralph Loop Context Template - For autonomous coding loops
RALPH_LOOP_CONTEXT_TEMPLATE = """
# PLURIBUS AUTONOMOUS CODING LOOP (PBLOOP)

You are operating in **Ralph Loop Mode** - an autonomous, iterative coding loop.

## Loop Mechanics
- This prompt will repeat until you output the completion promise
- Your previous work persists in files and git history
- Each iteration, check files for your past progress
- Build incrementally on previous iterations

## Completion Promise
When your task is FULLY COMPLETE, output exactly:
<promise>{completion_promise}</promise>

Do NOT output the promise until ALL requirements are met:
{requirements}

## Current Iteration
Iteration: {iteration} / {max_iterations}

## Working Directory
{working_dir}

## Task
{task_prompt}

---
Check your previous work in files. Continue from where you left off.
Test your code. Fix failures. Iterate until complete.
"""

RING_NAMES = {
    0: "KERNEL",
    1: "OPERATOR", 
    2: "APPLICATION",
    3: "EPHEMERAL"
}


class TmuxSwarmOrchestrator:
    """Manages multi-agent Claude swarms via tmux panes."""
    
    def __init__(self, session_name: str = "pluribus_swarm", working_dir: str = None):
        self.session_name = session_name
        self.working_dir = working_dir or os.getcwd()
        self.agents = []
        
    def _run_tmux(self, args: List[str]) -> str:
        """Execute a tmux command."""
        cmd = [TMUX_BIN] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()

    def _run_claude(self, prompt: str, model: str = None, print_mode: bool = True) -> subprocess.CompletedProcess:
        """Execute Claude CLI with a prompt."""
        cmd = [CLAUDE_BIN]
        if print_mode:
            cmd.append("-p")
        if model:
            cmd.extend(["--model", model])
        cmd.append(prompt)
        
        return subprocess.run(cmd, capture_output=True, text=True, cwd=self.working_dir)

    def _session_exists(self) -> bool:
        """Check if our tmux session already exists."""
        check = subprocess.run([TMUX_BIN, "has-session", "-t", self.session_name], capture_output=True)
        return check.returncode == 0

    def _detect_lsp_capabilities(self) -> str:
        """Detect available LSP servers and format for prompt."""
        available = []
        for lang, config in LSP_CAPABILITIES.items():
            # Check if server binary exists
            server = config["server"]
            if subprocess.run(["which", server], capture_output=True).returncode == 0:
                features = ", ".join(config["features"])
                available.append(f"- {lang}: {server} ({features})")
        
        if available:
            return "Claude Code LSP available:\\n" + "\\n".join(available)
        return "No LSP servers detected. Consider installing pyright-lsp, vtsls, etc."

    def init_session(self):
        """Create the swarm session if it doesn't exist."""
        if self._session_exists():
            print(f"[Swarm] Session '{self.session_name}' already exists.")
            return
        
        # Create new detached session with explicit local shell
        self._run_tmux([
            "new-session", "-d", "-s", self.session_name, "-n", "control",
            "-c", self.working_dir
        ])
        # Send a cd command to ensure we're in the right directory
        time.sleep(0.5)
        self._run_tmux([
            "send-keys", "-t", f"{self.session_name}:control",
            f"cd {self.working_dir} && echo 'Swarm Control Ready - v{VERSION}'", "Enter"
        ])
        print(f"[Swarm] Created session: {self.session_name}")

    def _build_cagent_prompt(self, agent_id: str, ring_level: int, role: str, task_prompt: str) -> str:
        """Build a CAGENT-compliant prompt with context injection."""
        return CAGENT_CONTEXT_TEMPLATE.format(
            agent_id=agent_id,
            agent_role=role,
            ring_level=ring_level,
            ring_name=RING_NAMES.get(ring_level, "UNKNOWN"),
            working_dir=self.working_dir,
            lsp_info=self._detect_lsp_capabilities(),
            task_prompt=task_prompt
        )

    def _build_ralph_prompt(self, task_prompt: str, requirements: str, 
                            completion_promise: str, iteration: int, max_iterations: int) -> str:
        """Build a Ralph Loop prompt for autonomous coding."""
        return RALPH_LOOP_CONTEXT_TEMPLATE.format(
            task_prompt=task_prompt,
            requirements=requirements,
            completion_promise=completion_promise,
            iteration=iteration,
            max_iterations=max_iterations,
            working_dir=self.working_dir
        )

    def spawn_agent(self, agent_id: str, prompt: str, ring_level: int = 2, 
                    role: str = "Agent", model: str = None) -> Dict[str, Any]:
        """
        Spawn a new Claude agent in a new tmux window with CAGENT context.
        """
        # Build CAGENT-compliant prompt
        full_prompt = self._build_cagent_prompt(agent_id, ring_level, role, prompt)
        
        # Escape for shell
        escaped_prompt = full_prompt.replace('"', '\\"').replace("'", "'\\''")
        
        # Create new window
        self._run_tmux([
            "new-window", "-t", self.session_name, "-n", agent_id, 
            "-c", self.working_dir
        ])
        
        # Build Claude command
        model_flag = f"--model {model}" if model else ""
        claude_cmd = f'{CLAUDE_BIN} -p {model_flag} "{escaped_prompt}"'
        
        # Send command to window
        time.sleep(0.3)
        self._run_tmux([
            "send-keys", "-t", f"{self.session_name}:{agent_id}", 
            f"cd {self.working_dir} && {claude_cmd}", "Enter"
        ])
        
        agent_info = {
            "id": agent_id,
            "window": f"{self.session_name}:{agent_id}",
            "ring": ring_level,
            "role": role,
            "model": model or "default",
            "mode": "standard",
            "status": "running",
            "spawned_at": time.time()
        }
        self.agents.append(agent_info)
        print(f"[Swarm] Spawned agent: {agent_id} (Ring {ring_level}, {role})")
        return agent_info

    def spawn_ralph_loop(self, agent_id: str, task_prompt: str, requirements: str = "",
                         completion_promise: str = DEFAULT_COMPLETION_PROMISE,
                         max_iterations: int = DEFAULT_MAX_ITERATIONS,
                         model: str = None) -> Dict[str, Any]:
        """
        Spawn a Ralph Loop agent - autonomous coding loop until completion.
        
        This implements the Ralph Wiggum pattern: iterate with the same prompt
        until the completion promise is detected in output.
        """
        print(f"[Ralph] Starting autonomous loop: {agent_id}")
        print(f"[Ralph] Completion promise: '{completion_promise}'")
        print(f"[Ralph] Max iterations: {max_iterations}")
        
        # Create window for this agent
        self._run_tmux([
            "new-window", "-t", self.session_name, "-n", agent_id, 
            "-c", self.working_dir
        ])
        
        for iteration in range(1, max_iterations + 1):
            print(f"[Ralph] Iteration {iteration}/{max_iterations}")
            
            # Build the Ralph loop prompt
            ralph_prompt = self._build_ralph_prompt(
                task_prompt=task_prompt,
                requirements=requirements, 
                completion_promise=completion_promise,
                iteration=iteration,
                max_iterations=max_iterations
            )
            
            # Escape for shell
            escaped = ralph_prompt.replace('"', '\\"').replace("'", "'\\''")
            model_flag = f"--model {model}" if model else ""
            claude_cmd = f'{CLAUDE_BIN} -p {model_flag} "{escaped}"'
            
            # Run Claude in the tmux pane
            self._run_tmux([
                "send-keys", "-t", f"{self.session_name}:{agent_id}", 
                f"cd {self.working_dir} && {claude_cmd}", "Enter"
            ])
            
            # Wait for execution (crude but effective for v1.1)
            # In production, we'd poll the pane output
            time.sleep(30)  # Adjust based on task complexity
            
            # Capture output to check for completion
            output = self.capture_output(agent_id, lines=100)
            
            # Check for completion promise
            if f"<promise>{completion_promise}</promise>" in output:
                print(f"[Ralph] ✅ Completion promise detected at iteration {iteration}!")
                break
            
            # Check for explicit DONE markers
            if completion_promise in output and ("complete" in output.lower() or "finished" in output.lower()):
                print(f"[Ralph] ✅ Task appears complete at iteration {iteration}")
                break
                
            print(f"[Ralph] Iteration {iteration} complete, continuing...")
        
        agent_info = {
            "id": agent_id,
            "window": f"{self.session_name}:{agent_id}",
            "mode": "ralph_loop",
            "iterations_run": iteration,
            "max_iterations": max_iterations,
            "completion_promise": completion_promise,
            "status": "complete" if iteration < max_iterations else "max_iterations_reached",
            "model": model or "default"
        }
        self.agents.append(agent_info)
        return agent_info

    def spawn_from_manifest(self, manifest_path: str):
        """Spawn agents from a JSON manifest file."""
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        self.init_session()
        
        for agent_def in manifest.get("agents", []):
            # Check if this is a Ralph loop agent
            if agent_def.get("mode") == "ralph_loop":
                self.spawn_ralph_loop(
                    agent_id=agent_def["id"],
                    task_prompt=agent_def.get("prompt", ""),
                    requirements=agent_def.get("requirements", ""),
                    completion_promise=agent_def.get("completion_promise", DEFAULT_COMPLETION_PROMISE),
                    max_iterations=agent_def.get("max_iterations", DEFAULT_MAX_ITERATIONS),
                    model=agent_def.get("model")
                )
            else:
                self.spawn_agent(
                    agent_id=agent_def["id"],
                    prompt=agent_def.get("prompt", agent_def.get("phase_focus", "Execute your role.")),
                    ring_level=agent_def.get("ring", 2),
                    role=agent_def.get("role", "Agent"),
                    model=agent_def.get("model")
                )
        
        print(f"[Swarm] Spawned {len(self.agents)} agents from manifest")

    def spawn_swarm(self, tasks: List[Dict[str, Any]]):
        """Spawn multiple agents in parallel from a task list."""
        self.init_session()
        for task in tasks:
            self.spawn_agent(
                agent_id=task["agent_id"],
                prompt=task["prompt"],
                ring_level=task.get("ring", 2),
                role=task.get("role", "Agent"),
                model=task.get("model")
            )
        print(f"[Swarm] Spawned {len(tasks)} agents in session '{self.session_name}'")

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all windows/agents in the swarm."""
        windows = self._run_tmux([
            "list-windows", "-t", self.session_name, "-F", "#{window_name}|#{window_activity}"
        ])
        if not windows:
            return []
        
        agents = []
        for line in windows.split("\n"):
            parts = line.split("|")
            if len(parts) >= 1:
                agents.append({
                    "window": parts[0],
                    "last_activity": parts[1] if len(parts) > 1 else "unknown"
                })
        return agents

    def capture_output(self, agent_id: str, lines: int = 50) -> str:
        """Capture recent output from an agent's pane."""
        result = self._run_tmux([
            "capture-pane", "-t", f"{self.session_name}:{agent_id}", 
            "-p", "-S", f"-{lines}"
        ])
        return result

    def kill_session(self):
        """Terminate the entire swarm session."""
        self._run_tmux(["kill-session", "-t", self.session_name])
        print(f"[Swarm] Killed session: {self.session_name}")

    def attach(self):
        """Attach to the swarm session (interactive)."""
        subprocess.run([TMUX_BIN, "attach-session", "-t", self.session_name])


def run_test():
    """Run a small 2-agent test to verify the orchestrator works."""
    print("=" * 60)
    print(f"TMUX SWARM ORCHESTRATOR v{VERSION} - CAGENT TEST MODE")
    print("=" * 60)
    
    orchestrator = TmuxSwarmOrchestrator(
        session_name="pluribus_test_swarm",
        working_dir="/Users/kroma/pluribus_evolution"
    )
    
    test_tasks = [
        {
            "agent_id": "agent_alpha",
            "prompt": "Respond with your REPL header and confirm: 'Agent Alpha reporting. CAGENT bootstrap complete.'",
            "ring": 2,
            "role": "Test Agent Alpha"
        },
        {
            "agent_id": "agent_beta", 
            "prompt": "Respond with your REPL header and confirm: 'Agent Beta online. Protocol stack verified.'",
            "ring": 2,
            "role": "Test Agent Beta"
        }
    ]
    
    orchestrator.spawn_swarm(test_tasks)
    
    time.sleep(2)
    agents = orchestrator.list_agents()
    print(f"\n[Test] Active agents: {json.dumps(agents, indent=2)}")
    
    print(f"\n[Test] Swarm spawned. Check session with:")
    print(f"  {TMUX_BIN} attach-session -t pluribus_test_swarm")
    print(f"\nTo kill the test session:")
    print(f"  {TMUX_BIN} kill-session -t pluribus_test_swarm")


def run_loop(prompt: str, completion_promise: str = DEFAULT_COMPLETION_PROMISE, 
             max_iterations: int = DEFAULT_MAX_ITERATIONS, model: str = None):
    """Run a single Ralph Loop agent."""
    print("=" * 60)
    print(f"TMUX SWARM ORCHESTRATOR v{VERSION} - RALPH LOOP MODE (PBLOOP)")
    print("=" * 60)
    
    orchestrator = TmuxSwarmOrchestrator(
        session_name="pluribus_ralph_loop",
        working_dir=os.getcwd()
    )
    orchestrator.init_session()
    
    result = orchestrator.spawn_ralph_loop(
        agent_id="ralph_worker",
        task_prompt=prompt,
        completion_promise=completion_promise,
        max_iterations=max_iterations,
        model=model
    )
    
    print(f"\n[Ralph] Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"TMUX Swarm Orchestrator v{VERSION}")
        print("\nUsage:")
        print("  python3 tmux_swarm_orchestrator.py test                          # Run 2-agent test")
        print("  python3 tmux_swarm_orchestrator.py spawn <manifest>              # Spawn from manifest")
        print("  python3 tmux_swarm_orchestrator.py loop \"<prompt>\" [--promise X] [--max N]  # Ralph loop")
        print("  python3 tmux_swarm_orchestrator.py list                          # List active agents")
        print("  python3 tmux_swarm_orchestrator.py kill                          # Kill swarm session")
        print("\nSemops: PBTSWARM (swarm mode), PBLOOP (Ralph loop mode)")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "test":
        run_test()
    elif cmd == "spawn" and len(sys.argv) > 2:
        orch = TmuxSwarmOrchestrator(working_dir=os.getcwd())
        orch.spawn_from_manifest(sys.argv[2])
    elif cmd == "loop":
        # Parse loop arguments
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Complete the task."
        promise = DEFAULT_COMPLETION_PROMISE
        max_iter = DEFAULT_MAX_ITERATIONS
        model = None
        
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--promise" and i + 1 < len(sys.argv):
                promise = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--max" and i + 1 < len(sys.argv):
                max_iter = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
                model = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        run_loop(prompt, promise, max_iter, model)
    elif cmd == "list":
        orch = TmuxSwarmOrchestrator()
        print(json.dumps(orch.list_agents(), indent=2))
    elif cmd == "kill":
        orch = TmuxSwarmOrchestrator()
        orch.kill_session()
    elif cmd == "version":
        print(f"TMUX Swarm Orchestrator v{VERSION}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
