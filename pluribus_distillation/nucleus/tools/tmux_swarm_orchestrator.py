#!/usr/bin/env python3
"""
tmux_swarm_orchestrator.py - Claude Multi-Agent Swarm via TMUX (Verbessern Protocols)
Version: 1.3.0
Ring: 0 (Kernel)
Status: Active
Protocol: DKIN v30 / PAIP v16 / Citizen v2 / Verbessern v1 / A2A v30

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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# Version
VERSION = "1.5.0"

# Import QuotaManager for quota-aware orchestration
try:
    from quota_manager import QuotaManager, Provider, GatingDecision, TaskComplexity
    QUOTA_AWARE = True
except ImportError:
    QUOTA_AWARE = False
    QuotaManager = None

# A2A v29 Configuration
A2A_HEARTBEAT_INTERVAL_S = 300  # 5 minutes
A2A_HANDSHAKE_TIMEOUT_S = 60
BUS_PATH = Path(os.environ.get("PLURIBUS_BUS_PATH", ".pluribus/bus/events.ndjson"))

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
- **Protocol Stack**: DKIN v30 (CAGENT) | PAIP v16 | Citizen v2 | Verbessern v1
- **Ring Level**: {ring_level} ({ring_name})
- **Control Plane**: Ralph Loop (Iterative) + Kanban (Planning)
- **Sextet Laws**: Lawfulness | Observability | Provenance | Recurrence | Evolvability | Boundedness

## REPL Header Contract (UNIFORM v2.0)
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
    "PLURIBUS BOOTSTRAP: Start every response with the UNIFORM v2.0 panel. "
    "Use: python3 /pluribus/nucleus/tools/agent_header.py {agent} to view the template. "
    "Do not run tools or read files before the panel. Acknowledge with BOOTSTRAP_OK."
)


class TmuxSwarmOrchestrator:
    """Manages multi-agent Claude swarms via tmux panes."""

    def __init__(self, session_name: Optional[str] = None, working_dir: str = None,
                 quota_tier: str = "pro", quota_aware: bool = True):
        if not session_name:
            session_name = os.environ.get("PLURIBUS_SWARM_SESSION", "pluribus_swarm")
        self.session_name = session_name
        self.working_dir = working_dir or os.getcwd()
        self.agents = []
        # A2A v29 state
        self.collab_id: Optional[str] = None
        self.codeword: Optional[str] = None
        self.a2a_initiator: str = "tmux_swarm_orchestrator"
        self._last_heartbeat_ts: float = 0.0

        # Quota Protocol v1 integration
        self.quota_aware = quota_aware and QUOTA_AWARE
        self.quota_tier = quota_tier
        self.quota_manager: Optional[QuotaManager] = None
        if self.quota_aware:
            self.quota_manager = QuotaManager(tier=quota_tier, working_dir=self.working_dir)
            print(f"[Swarm] Quota-aware mode enabled (tier: {quota_tier})")

    # =========================================================================
    # A2A v29 Protocol Methods
    # =========================================================================

    def _emit_bus_event(self, topic: str, data: Dict[str, Any], level: str = "info", kind: str = "event") -> str:
        """Emit event to the Pluribus bus (NDJSON append)."""
        bus_path = BUS_PATH
        if not bus_path.is_absolute():
            bus_path = Path(self.working_dir) / bus_path
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = uuid.uuid4().hex
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.a2a_initiator,
            "host": os.uname().nodename if hasattr(os, 'uname') else "localhost",
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def emit_a2a_handshake(self, target_agents: List[str], scope: str, iterations: int = 10, mode: str = "parallel") -> str:
        """
        Emit A2A handshake proposal before spawning swarm.

        DKIN v30 A2A Protocol:
        - Emit a2a.handshake.propose with codeword, target agents, scope
        - Wait for acks (handled by a2a_monitor daemon)
        - Store collab_id for later heartbeats and completion
        """
        self.collab_id = f"collab-{uuid.uuid4().hex[:8]}"
        self.codeword = f"swarm-{self.session_name}-{uuid.uuid4().hex[:6]}"

        self._emit_bus_event("a2a.handshake.propose", {
            "codeword": self.codeword,
            "collab_id": self.collab_id,
            "initiator": self.a2a_initiator,
            "target_agents": target_agents,
            "scope": scope,
            "iterations_planned": iterations,
            "mode": mode,
            "ttl_s": 3600,
            "session_name": self.session_name,
        })

        print(f"[A2A] Handshake proposed: codeword={self.codeword}, collab_id={self.collab_id}")
        return self.collab_id

    def emit_a2a_heartbeat(self, iteration: int = 0, status: str = "active") -> None:
        """
        Emit A2A heartbeat (should be called every 5 minutes during swarm execution).

        DKIN v30 A2A Protocol:
        - Heartbeat every 5 minutes to confirm liveness
        - Include codeword, iteration, and current status
        """
        if not self.codeword or not self.collab_id:
            return  # No active collaboration

        now = time.time()
        if now - self._last_heartbeat_ts < A2A_HEARTBEAT_INTERVAL_S - 30:
            return  # Rate limit heartbeats

        self._last_heartbeat_ts = now

        self._emit_bus_event("a2a.heartbeat", {
            "codeword": self.codeword,
            "collab_id": self.collab_id,
            "agent_id": self.a2a_initiator,
            "iteration": iteration,
            "status": status,
            "active_agents": len(self.agents),
            "session_name": self.session_name,
            "channel": "bus",
        })

        print(f"[A2A] Heartbeat: codeword={self.codeword}, iteration={iteration}, status={status}")

    def emit_a2a_collab_complete(self, summary: str = "", success: bool = True) -> None:
        """
        Emit A2A collaboration complete event on swarm termination.

        DKIN v30 A2A Protocol:
        - Emit a2a.collab.complete when swarm finishes
        - Include summary and participant list
        """
        if not self.codeword or not self.collab_id:
            return  # No active collaboration

        participants = [a.get("agent_id", "unknown") for a in self.agents]

        self._emit_bus_event("a2a.collab.complete", {
            "codeword": self.codeword,
            "collab_id": self.collab_id,
            "initiator": self.a2a_initiator,
            "participants": participants,
            "summary": summary or f"Swarm {self.session_name} completed with {len(self.agents)} agents",
            "success": success,
            "final_iteration": len(self.agents),
            "session_name": self.session_name,
        })

        print(f"[A2A] Collaboration complete: codeword={self.codeword}, success={success}")

        # Clear A2A state
        self.collab_id = None
        self.codeword = None
        
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

    def _build_runner_command(self, runner: str, prompt: str, model: Optional[str]) -> str:
        """Build a command for one-shot/exec runners."""
        cfg = self._resolve_runner(runner)
        env_prefix = self._claude_env_prefix()
        escaped_prompt = self._shell_safe_prompt(prompt)
        model_flag = f"--model {model}" if model else ""

        if cfg.get("style") == "codex_exec":
            workspace_mode = os.environ.get("PLURIBUS_CODEX_WORKSPACE", "workspace-write")
            return (
                f'{env_prefix}{cfg["bin"]} exec --full-auto --ask-for-approval never '
                f'-s {workspace_mode} {model_flag} "{escaped_prompt}"'
            )

        # Default to claude-style one-shot prompt
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

    def spawn_monitor_pane(self, providers: Optional[List[str]] = None):
        """Spawn the Quota Monitor pane (Quota Protocol v1)."""
        if not self._session_exists():
            self.init_session()

        print("[Swarm] Spawning Quota Monitor (Quota Protocol v1)...")
        self._run_tmux([
            "new-window", "-t", self.session_name, "-n", "monitor",
            "-c", self.working_dir
        ])

        # Build provider list for monitoring
        provider_str = ",".join(providers) if providers else "claude,codex,gemini"

        # Run the quota_monitor.py daemon
        monitor_script_path = Path(__file__).parent / "quota_monitor.py"
        if monitor_script_path.exists():
            monitor_cmd = (
                f"python3 {monitor_script_path} "
                f"--providers {provider_str} "
                f"--tier {self.quota_tier} "
                f"--interval {AUTOCLAUDE_CHECK_INTERVAL}"
            )
        else:
            # Fallback to basic monitoring if quota_monitor.py not available
            monitor_cmd = f"""
while true; do
    echo "[QuotaMonitor] Monitoring swarm (fallback mode)..."
    echo "  Tip: Install quota_monitor.py for full quota tracking"
    sleep {AUTOCLAUDE_CHECK_INTERVAL}
done
"""
        self._run_tmux([
            "send-keys", "-t", f"{self.session_name}:monitor",
            f"cd {self.working_dir} && {monitor_cmd}", "Enter"
        ])

        self._emit_bus_event("pbtswarm.monitor.started", {
            "session_name": self.session_name,
            "providers": provider_str,
            "tier": self.quota_tier,
            "quota_aware": self.quota_aware
        })

    def _build_cagent_prompt(self, agent_id: str, ring_level: int, role: str,
                             task_prompt: str, working_dir: Optional[str] = None) -> str:
        """Build a CAGENT-compliant prompt with context injection."""
        resolved_dir = working_dir or self.working_dir
        return CAGENT_CONTEXT_TEMPLATE.format(
            agent_id=agent_id,
            agent_role=role,
            ring_level=ring_level,
            ring_name=RING_NAMES.get(ring_level, "UNKNOWN"),
            working_dir=resolved_dir,
            lsp_info=self._detect_lsp_capabilities(),
            task_prompt=task_prompt
        )

    def _build_ralph_prompt(self, task_prompt: str, requirements: str,
                            completion_promise: str, iteration: int, max_iterations: int,
                            working_dir: Optional[str] = None) -> str:
        """Build a Ralph Loop prompt for autonomous coding."""
        resolved_dir = working_dir or self.working_dir
        return RALPH_LOOP_CONTEXT_TEMPLATE.format(
            task_prompt=task_prompt,
            requirements=requirements,
            completion_promise=completion_promise,
            iteration=iteration,
            max_iterations=max_iterations,
            working_dir=resolved_dir
        )

    def spawn_agent(self, agent_id: str, prompt: str, ring_level: int = 2,
                    role: str = "Agent", model: str = None, runner: str = "claude",
                    mode: Optional[str] = None, bootstrap_prompt: Optional[str] = None,
                    working_dir: Optional[str] = None,
                    estimated_tokens: int = 5000) -> Dict[str, Any]:
        """
        Spawn a new agent in a tmux window with CAGENT context.

        Quota Protocol v1 Integration:
        - Pre-spawn quota check
        - Provider downgrade recommendations
        - Model switching on low budget
        """
        agent_working_dir = working_dir or self.working_dir

        # Quota-aware pre-spawn check
        actual_runner = runner
        actual_model = model
        gating_decision = None

        if self.quota_aware and self.quota_manager:
            # Map runner to Provider enum
            provider_map = {
                "claude": Provider.CLAUDE,
                "codex": Provider.CODEX,
                "gemini": Provider.GEMINI,
                "grok": Provider.GROK,
            }
            provider = provider_map.get(runner, Provider.CLAUDE)

            # Check quota
            result = self.quota_manager.can_proceed(
                provider,
                tokens_in=estimated_tokens,
                task_complexity=TaskComplexity.MEDIUM
            )
            gating_decision = result.decision

            if result.decision == GatingDecision.REJECT:
                print(f"[Swarm] WARNING: Quota rejected for {runner}: {result.reason}")
                # Try to find alternative provider
                alt_provider, alt_result = self.quota_manager.select_optimal_provider(
                    tokens_in=estimated_tokens,
                    exclude_providers=[provider]
                )
                if alt_result.decision == GatingDecision.PROCEED:
                    print(f"[Swarm] Switching to alternative provider: {alt_provider.value}")
                    actual_runner = alt_provider.value
                    self._emit_bus_event("pbtswarm.agent.provider_switch", {
                        "agent_id": agent_id,
                        "original_runner": runner,
                        "new_runner": actual_runner,
                        "reason": result.reason
                    })
                else:
                    self._emit_bus_event("pbtswarm.agent.quota_blocked", {
                        "agent_id": agent_id,
                        "runner": runner,
                        "reason": result.reason
                    }, level="warning")
                    return {
                        "id": agent_id,
                        "status": "quota_blocked",
                        "reason": result.reason
                    }

            elif result.decision == GatingDecision.DOWNGRADE:
                # Use recommended cheaper model
                if result.recommended_model:
                    print(f"[Swarm] Downgrading model for {agent_id}: {result.recommended_model}")
                    actual_model = result.recommended_model
                    self._emit_bus_event("pbtswarm.agent.model_downgrade", {
                        "agent_id": agent_id,
                        "original_model": model,
                        "new_model": actual_model,
                        "reason": result.reason
                    })
                # Or switch provider
                elif result.recommended_provider:
                    actual_runner = result.recommended_provider.value
                    self._emit_bus_event("pbtswarm.agent.provider_switch", {
                        "agent_id": agent_id,
                        "original_runner": runner,
                        "new_runner": actual_runner,
                        "reason": result.reason
                    })

            elif result.decision == GatingDecision.QUEUE:
                print(f"[Swarm] Queuing agent {agent_id} (wait {result.wait_seconds}s)")
                # For now, we proceed after logging; future: implement actual queue
                self._emit_bus_event("pbtswarm.agent.queued", {
                    "agent_id": agent_id,
                    "wait_seconds": result.wait_seconds,
                    "reason": result.reason
                })

        # Build CAGENT-compliant prompt
        full_prompt = self._build_cagent_prompt(agent_id, ring_level, role, prompt, working_dir=agent_working_dir)

        # Use actual_runner (may have been switched by quota manager)
        runner_cfg = self._resolve_runner(actual_runner)
        spawn_mode = mode or runner_cfg.get("default_mode", "oneshot")

        # Create new window
        self._run_tmux([
            "new-window", "-t", self.session_name, "-n", agent_id,
            "-c", agent_working_dir
        ])

        # Send command to window
        time.sleep(0.3)
        if spawn_mode == "interactive":
            env_prefix = self._claude_env_prefix()
            runner_cmd = f"{env_prefix}{runner_cfg['bin']}"
            self._run_tmux([
                "send-keys", "-t", f"{self.session_name}:{agent_id}",
                f"cd {agent_working_dir} && {runner_cmd}", "Enter"
            ])
            time.sleep(1.0)
            bootstrap = bootstrap_prompt or DEFAULT_BOOTSTRAP_PROMPT.format(agent=actual_runner)
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
            # Use actual_model (may have been downgraded by quota manager)
            runner_cmd = self._build_runner_command(actual_runner, full_prompt, actual_model)
            self._run_tmux([
                "send-keys", "-t", f"{self.session_name}:{agent_id}",
                f"cd {agent_working_dir} && {runner_cmd}", "Enter"
            ])

        # Record consumption in quota manager
        if self.quota_aware and self.quota_manager:
            provider_map = {
                "claude": Provider.CLAUDE,
                "codex": Provider.CODEX,
                "gemini": Provider.GEMINI,
                "grok": Provider.GROK,
            }
            provider = provider_map.get(actual_runner, Provider.CLAUDE)
            self.quota_manager.record_consumption(
                provider=provider,
                tokens_in=estimated_tokens,
                tokens_out=int(estimated_tokens * 0.5),  # Estimate
                agent_id=agent_id
            )

        agent_info = {
            "id": agent_id,
            "window": f"{self.session_name}:{agent_id}",
            "ring": ring_level,
            "role": role,
            "model": actual_model or "default",
            "mode": spawn_mode,
            "runner": actual_runner,
            "original_runner": runner if runner != actual_runner else None,
            "working_dir": agent_working_dir,
            "status": "running",
            "spawned_at": time.time(),
            "gating_decision": gating_decision.value if gating_decision else None,
            "quota_aware": self.quota_aware
        }
        self.agents.append(agent_info)

        status_msg = f"[Swarm] Spawned agent: {agent_id} (Ring {ring_level}, {role})"
        if runner != actual_runner:
            status_msg += f" [switched: {runner}->{actual_runner}]"
        if model != actual_model and actual_model:
            status_msg += f" [downgraded: {actual_model}]"
        print(status_msg)

        return agent_info

    def spawn_ralph_loop(self, agent_id: str, task_prompt: str, requirements: str = "",
                         completion_promise: str = DEFAULT_COMPLETION_PROMISE,
                         max_iterations: int = DEFAULT_MAX_ITERATIONS,
                         model: str = None, working_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Spawn a Ralph Loop agent - autonomous coding loop until completion.
        """
        agent_working_dir = working_dir or self.working_dir
        print(f"[Ralph] Starting autonomous loop: {agent_id}")
        print(f"[Ralph] Completion promise: '{completion_promise}'")
        print(f"[Ralph] Max iterations: {max_iterations}")
        
        # Create window for this agent
        self._run_tmux([
            "new-window", "-t", self.session_name, "-n", agent_id, 
            "-c", agent_working_dir
        ])
        
        for iteration in range(1, max_iterations + 1):
            print(f"[Ralph] Iteration {iteration}/{max_iterations}")
            
            # Build the Ralph loop prompt
            ralph_prompt = self._build_ralph_prompt(
                task_prompt=task_prompt,
                requirements=requirements, 
                completion_promise=completion_promise,
                iteration=iteration,
                max_iterations=max_iterations,
                working_dir=agent_working_dir
            )
            
            # Escape for shell
            escaped = self._shell_safe_prompt(ralph_prompt)
            model_flag = f"--model {model}" if model else ""
            env_prefix = self._claude_env_prefix()
            claude_cmd = f'{env_prefix}{CLAUDE_BIN} -p {model_flag} "{escaped}"'
            
            # Run Claude in the tmux pane
            self._run_tmux([
                "send-keys", "-t", f"{self.session_name}:{agent_id}", 
                f"cd {agent_working_dir} && {claude_cmd}", "Enter"
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

        # A2A v29: Emit handshake before spawning
        agent_ids = [a["id"] for a in manifest.get("agents", [])]
        scope = manifest.get("purpose", manifest.get("name", "swarm task"))
        iterations = sum(a.get("max_iterations", 10) for a in manifest.get("agents", []))
        self.emit_a2a_handshake(
            target_agents=agent_ids,
            scope=scope,
            iterations=iterations,
            mode="parallel"
        )

        self.init_session()
        self.spawn_monitor_pane()  # Always spawn monitor in v1.2

        agents = manifest.get("agents", [])
        workdirs = []
        for agent_def in agents:
            workdirs.append(
                agent_def.get("working_dir")
                or agent_def.get("workdir")
                or agent_def.get("cwd")
                or self.working_dir
            )
        unique_workdirs = {wd for wd in workdirs if wd}
        if len(agents) > 1 and len(unique_workdirs) <= 1:
            print("[Swarm] WARNING: multiple agents share one working_dir; PAIP expects per-agent clones under /tmp.")

        for agent_def in agents:
            agent_working_dir = (
                agent_def.get("working_dir")
                or agent_def.get("workdir")
                or agent_def.get("cwd")
                or self.working_dir
            )
            # Check if this is a Ralph loop agent
            if agent_def.get("mode") == "ralph_loop":
                self.spawn_ralph_loop(
                    agent_id=agent_def["id"],
                    task_prompt=agent_def.get("prompt", ""),
                    requirements=agent_def.get("requirements", ""),
                    completion_promise=agent_def.get("completion_promise", DEFAULT_COMPLETION_PROMISE),
                    max_iterations=agent_def.get("max_iterations", DEFAULT_MAX_ITERATIONS),
                    model=agent_def.get("model"),
                    working_dir=agent_working_dir
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
                    bootstrap_prompt=agent_def.get("bootstrap"),
                    working_dir=agent_working_dir
                )
        
        # Emit pbtswarm.spawn event
        self._emit_bus_event("pbtswarm.spawn", {
            "session_name": self.session_name,
            "agents_spawned": len(self.agents),
            "agent_ids": [a.get("id", "unknown") for a in self.agents],
            "codeword": self.codeword,
            "collab_id": self.collab_id,
            "manifest_path": manifest_path,
        })

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
                bootstrap_prompt=task.get("bootstrap"),
                working_dir=task.get("working_dir")
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
        # A2A v29: Emit collab complete before killing
        self.emit_a2a_collab_complete(
            summary=f"Session {self.session_name} terminated by kill command",
            success=False  # Killed = not successful completion
        )
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
