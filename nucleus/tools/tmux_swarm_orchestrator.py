#!/usr/bin/env python3
"""
tmux_swarm_orchestrator.py - Claude Multi-Agent Swarm via TMUX (Verbessern Protocols)
Version: 1.2.0
Ring: 0 (Kernel)
Status: Active
Protocol: DKIN v28 / PAIP v15 / Citizen v1 / Verbessern v1

Features:
- Spawns N tmux panes, each running a Claude CLI instance
- CAGENT bootstrap context injection (REPL Header, Ring, Protocol Stack)
- Ralph Loop Mode: Autonomous coding loops until completion promise (PBLOOP)
- Autoclaude Monitor: Dedicated pane for rate-limit supervision
- LSP capability awareness for enhanced code intelligence

Semops:
- PBTSWARM: Multi-agent swarm orchestration
- PBLOOP: Autonomous coding loop (Ralph Wiggum pattern)
- PBMONITOR: Autoclaude supervision

Usage:
    python3 tmux_swarm_orchestrator.py test                    # 2-agent test
    python3 tmux_swarm_orchestrator.py spawn <manifest>        # Spawn from manifest
    python3 tmux_swarm_orchestrator.py loop "<prompt>" [opts]  # Ralph loop mode
    python3 tmux_swarm_orchestrator.py monitor                 # Start Autoclaude monitor
"""

import subprocess
import sys
import os
import time
import json
import re
import shutil
import shlex
from typing import List, Dict, Any, Optional

# Version
VERSION = "1.2.0"

def resolve_agent_bin(name: str, fallback: str) -> str:
    """Resolve a CLI binary, preferring bus wrappers and env overrides."""
    env_key = f"PLURIBUS_{name.upper()}_BIN"
    env_val = os.environ.get(env_key) or os.environ.get(f"{name.upper()}_BIN")
    if env_val:
        return env_val
    wrapper = os.path.join(os.path.dirname(__file__), f"bus-{name}")
    if os.path.isfile(wrapper) and os.access(wrapper, os.X_OK):
        return wrapper
    return fallback or name

# Paths (system-specific, discovered on macOS with Homebrew)
TMUX_BIN = os.environ.get("TMUX_BIN") or shutil.which("tmux") or "/opt/homebrew/bin/tmux"
CLAUDE_BIN = resolve_agent_bin("claude", shutil.which("claude") or "/opt/homebrew/bin/claude")
CODEX_BIN = resolve_agent_bin("codex", shutil.which("codex") or "/usr/local/bin/codex")
GEMINI_BIN = resolve_agent_bin("gemini", shutil.which("gemini") or "/usr/bin/gemini")
QWEN_BIN = resolve_agent_bin("qwen", shutil.which("qwen") or "/usr/local/bin/qwen")
GROK_BIN = resolve_agent_bin("grok", shutil.which("grok") or "/usr/local/bin/grok")

RUNNER_CONFIG = {
    "claude": {"bin": CLAUDE_BIN, "default_mode": "oneshot", "style": "claude_prompt"},
    "codex": {"bin": CODEX_BIN, "default_mode": "exec", "style": "codex_exec"},
    "gemini": {"bin": GEMINI_BIN, "default_mode": "interactive", "style": "interactive"},
    "qwen": {"bin": QWEN_BIN, "default_mode": "interactive", "style": "interactive"},
    "grok": {"bin": GROK_BIN, "default_mode": "interactive", "style": "interactive"},
}

# Default completion promise for Ralph loops
DEFAULT_COMPLETION_PROMISE = "DONE"
DEFAULT_MAX_ITERATIONS = 50

# Autoclaude Configuration
AUTOCLAUDE_CHECK_INTERVAL = 10
AUTOCLAUDE_RATE_LIMIT_BACKOFF = 300  # 5 minutes

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
- **Protocol Stack**: DKIN v29 (CAGENT) | PAIP v15 | Citizen v1 | Verbessern v1
- **Ring Level**: {ring_level} ({ring_name})
- **Control Plane**: Ralph Loop (Iterative) + Kanban (Planning)
- **Sextet Laws**: Lawfulness | Observability | Provenance | Recurrence | Evolvability | Boundedness

## REPL Header Contract (UNIFORM v1.4)
Every response MUST begin with the UNIFORM panel defined in:
- /pluribus/nucleus/specs/repl_header_contract_v1.md
Do NOT use the legacy [AGENT: ...] header.

## Verbessern Workflow (Ralph Protocol)
1. **Source of Truth**: Read `kanban.md` for tasks.
2. **Exit Condition**: Task is DONE only when `kanban.md` card moves + tests pass.
3. **Safety**: Do NOT run commands as root. Confirm destructive actions.

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
When your task is FULLY COMPLETE (Kanban updated + Tests pass), output exactly:
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

DEFAULT_BOOTSTRAP_PROMPT = (
    "PLURIBUS BOOTSTRAP: Start every response with the UNIFORM v1.4 panel. "
    "Use: python3 /pluribus/nucleus/tools/agent_header.py {agent} to view the template. "
    "Do not run tools or read files before the panel. Acknowledge with BOOTSTRAP_OK."
)


class TmuxSwarmOrchestrator:
    """Manages multi-agent Claude swarms via tmux panes."""
    
    def __init__(self, session_name: Optional[str] = None, working_dir: str = None):
        if not session_name:
            session_name = os.environ.get("PLURIBUS_SWARM_SESSION", "pluribus_swarm")
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

        env = self._build_claude_env()
        return subprocess.run(cmd, capture_output=True, text=True, cwd=self.working_dir, env=env or None)

    def _build_claude_env(self) -> Optional[Dict[str, str]]:
        claude_home = os.environ.get("CLAUDE_HOME")
        clear_keys = os.environ.get("CLAUDE_CLEAR_KEYS") == "1"
        if not claude_home and not clear_keys:
            return None
        env = os.environ.copy()
        if claude_home:
            env["HOME"] = claude_home
            env["XDG_CONFIG_HOME"] = os.environ.get("CLAUDE_XDG_CONFIG_HOME") or os.path.join(claude_home, ".config")
            env["XDG_STATE_HOME"] = os.environ.get("CLAUDE_XDG_STATE_HOME") or os.path.join(claude_home, ".local/state")
        if clear_keys:
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("CLAUDE_API_KEY", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
        return env

    def _claude_env_prefix(self) -> str:
        prefix = os.environ.get("CLAUDE_ENV_PREFIX", "").strip()
        if prefix:
            return f"{prefix} "
        claude_home = os.environ.get("CLAUDE_HOME")
        clear_keys = os.environ.get("CLAUDE_CLEAR_KEYS") == "1"
        if not claude_home and not clear_keys:
            return ""
        parts = []
        if claude_home:
            xdg_config = os.environ.get("CLAUDE_XDG_CONFIG_HOME") or os.path.join(claude_home, ".config")
            xdg_state = os.environ.get("CLAUDE_XDG_STATE_HOME") or os.path.join(claude_home, ".local/state")
            parts.append(f"HOME={shlex.quote(claude_home)}")
            parts.append(f"XDG_CONFIG_HOME={shlex.quote(xdg_config)}")
            parts.append(f"XDG_STATE_HOME={shlex.quote(xdg_state)}")
        if clear_keys:
            parts.extend(["ANTHROPIC_API_KEY=", "CLAUDE_API_KEY=", "ANTHROPIC_AUTH_TOKEN="])
        return " ".join(parts) + " "

    def _shell_safe_prompt(self, prompt: str) -> str:
        """Collapse and sanitize prompt for safe shell embedding."""
        collapsed = " ".join(line.strip() for line in prompt.splitlines() if line.strip())
        collapsed = collapsed.replace("`", "'")
        return collapsed.replace("\\", "\\\\").replace('"', '\\"')

    def _resolve_runner(self, runner: Optional[str]) -> Dict[str, str]:
        """Resolve runner config for a given agent runner."""
        if not runner:
            runner = "claude"
        return RUNNER_CONFIG.get(runner, RUNNER_CONFIG["claude"])

    def _build_runner_command(self, runner: str, prompt: str, model: Optional[str], use_heredoc: bool = False) -> str:
        """Build a command for one-shot/exec runners.

        If use_heredoc=True, returns a heredoc-style command for large prompts.
        """
        cfg = self._resolve_runner(runner)
        env_prefix = self._claude_env_prefix()
        model_flag = f"--model {model}" if model else ""

        if cfg.get("style") == "codex_exec":
            workspace_mode = os.environ.get("PLURIBUS_CODEX_WORKSPACE", "workspace-write")
            # Prefer bus-codex wrapper for proper secret loading and bus events
            wrapper_path = os.path.join(os.path.dirname(__file__), "bus-codex")
            if os.path.isfile(wrapper_path) and os.access(wrapper_path, os.X_OK):
                codex_bin = wrapper_path
            else:
                codex_bin = cfg["bin"]

            if use_heredoc:
                # Use heredoc for large prompts - escape for heredoc safety
                heredoc_safe = prompt.replace("'", "'\\''")
                return (
                    f"cat << 'PBTSO_EOF' | {codex_bin} exec --full-auto "
                    f"-s {workspace_mode} {model_flag} -\n{heredoc_safe}\nPBTSO_EOF"
                )
            else:
                escaped_prompt = self._shell_safe_prompt(prompt)
                return (
                    f'{codex_bin} exec --full-auto '
                    f'-s {workspace_mode} {model_flag} "{escaped_prompt}"'
                )

        # Default to claude-style one-shot prompt
        escaped_prompt = self._shell_safe_prompt(prompt)
        return f'{env_prefix}{cfg["bin"]} -p {model_flag} "{escaped_prompt}"'

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

    def spawn_monitor_pane(self):
        """Spawn the Autoclaude Monitor pane."""
        if not self._session_exists():
            self.init_session()
            
        print("[Swarm] Spawning Autoclaude Monitor...")
        self._run_tmux([
            "new-window", "-t", self.session_name, "-n", "monitor",
            "-c", self.working_dir
        ])
        
        # In a real setup, this would run the actual 'autoclaude' tool.
        # Here we simulate it with a python loop that tails logs.
        monitor_script = f"""
while true; do
    echo "[Autoclaude] Monitoring swarm active..."
    # Check for 429 errors in recent output
    # ... logic here ...
    sleep {AUTOCLAUDE_CHECK_INTERVAL}
done
"""
        escaped_script = monitor_script.replace('"', '\\"').replace("'", "'\\''")
        self._run_tmux([
            "send-keys", "-t", f"{self.session_name}:monitor",
            f"echo 'Starting Monitor' && {monitor_script}", "Enter"
        ])

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
                    role: str = "Agent", model: str = None, runner: str = "claude",
                    mode: Optional[str] = None, bootstrap_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Spawn a new agent in a tmux window with CAGENT context.
        """
        # Build CAGENT-compliant prompt
        full_prompt = self._build_cagent_prompt(agent_id, ring_level, role, prompt)

        runner_cfg = self._resolve_runner(runner)
        spawn_mode = mode or runner_cfg.get("default_mode", "oneshot")

        # Create new window
        self._run_tmux([
            "new-window", "-t", self.session_name, "-n", agent_id, 
            "-c", self.working_dir
        ])

        # Send command to window
        time.sleep(1.5)  # INCREASED WAIT
        if spawn_mode == "interactive":
            env_prefix = self._claude_env_prefix()
            runner_cmd = f"{env_prefix}{runner_cfg['bin']}"
            print(f"[Orchestrator] Interactive CMD: {runner_cmd}")
            self._run_tmux([
                "send-keys", "-t", f"{self.session_name}:{agent_id}",
                f"cd {self.working_dir} && {runner_cmd}", "Enter"
            ])
            time.sleep(1.0)
            bootstrap = bootstrap_prompt or DEFAULT_BOOTSTRAP_PROMPT.format(agent=runner)
            if bootstrap:
                self._run_tmux([
                    "send-keys", "-t", f"{self.session_name}:{agent_id}",
                    self._shell_safe_prompt(bootstrap), "Enter"
                ])
                time.sleep(0.2)
            self._run_tmux([
                "send-keys", "-t", f"{self.session_name}:{agent_id}",
                self._shell_safe_prompt(full_prompt), "Enter"
            ])
        else:
            # Use heredoc for large prompts (CAGENT context can be > 1000 chars)
            use_heredoc = len(full_prompt) > 500
            runner_cmd = self._build_runner_command(runner, full_prompt, model, use_heredoc=use_heredoc)
            print(f"[Orchestrator] Runner CMD (heredoc={use_heredoc}): {runner_cmd[:200]}...")
            self._run_tmux([
                "send-keys", "-t", f"{self.session_name}:{agent_id}",
                f"cd {self.working_dir} && {runner_cmd}", "Enter"
            ])
        
        agent_info = {
            "id": agent_id,
            "window": f"{self.session_name}:{agent_id}",
            "ring": ring_level,
            "role": role,
            "model": model or "default",
            "mode": spawn_mode,
            "runner": runner,
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
            escaped = self._shell_safe_prompt(ralph_prompt)
            model_flag = f"--model {model}" if model else ""
            env_prefix = self._claude_env_prefix()
            claude_cmd = f'{env_prefix}{CLAUDE_BIN} -p {model_flag} "{escaped}"'
            
            # Run Claude in the tmux pane
            self._run_tmux([
                "send-keys", "-t", f"{self.session_name}:{agent_id}", 
                f"cd {self.working_dir} && {claude_cmd}", "Enter"
            ])
            
            # Wait for execution (crude but effective for v1.2)
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
        self.spawn_monitor_pane()  # Always spawn monitor in v1.2
        
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
                    model=agent_def.get("model"),
                    runner=agent_def.get("runner") or agent_def.get("cli", "claude"),
                    mode=agent_def.get("mode"),
                    bootstrap_prompt=agent_def.get("bootstrap")
                )
        
        print(f"[Swarm] Spawned {len(self.agents)} agents from manifest")

    def spawn_swarm(self, tasks: List[Dict[str, Any]]):
        """Spawn multiple agents in parallel from a task list."""
        self.init_session()
        self.spawn_monitor_pane()
        
        for task in tasks:
            self.spawn_agent(
                agent_id=task["agent_id"],
                prompt=task["prompt"],
                ring_level=task.get("ring", 2),
                role=task.get("role", "Agent"),
                model=task.get("model"),
                runner=task.get("runner", "claude"),
                mode=task.get("mode"),
                bootstrap_prompt=task.get("bootstrap")
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
    orchestrator.spawn_monitor_pane()
    
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
        print("  python3 tmux_swarm_orchestrator.py monitor                       # Start Autoclaude monitor")
        print("  python3 tmux_swarm_orchestrator.py list                          # List active agents")
        print("  python3 tmux_swarm_orchestrator.py kill                          # Kill swarm session")
        print("\nSemops: PBTSWARM (swarm mode), PBLOOP (Ralph loop mode), PBMONITOR (Autoclaude)")
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
    elif cmd == "monitor":
        orch = TmuxSwarmOrchestrator()
        orch.spawn_monitor_pane()
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
