#!/usr/bin/env python3
"""
IsoExecutor: Isolated Process Execution for Pluribus Agents.
============================================================

Implements the "Dynamic Isolated Agents" pattern (SOTA Part 3).
Spawns ephemeral CLI processes for "Specialist" agents to prevent
context bleed and ensure clean execution environments.

Supported Agents:
    - plurichat (default): PluriChat with persona routing
    - opencode: OpenCode TUI agent for coding tasks
    - pr-agent: PR-Agent for code review

Usage:
    executor = IsoExecutor()
    result = executor.spawn(
        agent="opencode",
        goal="Implement HGT",
        context_files=["/tmp/ctx.md"],
        trace_id="abc-123"
    )
"""

import os
import sys
import json
import subprocess
import uuid
import time
from pathlib import Path
from dataclasses import dataclass

sys.dont_write_bytecode = True


@dataclass
class IsoResult:
    exit_code: int
    stdout: str
    stderr: str
    artifacts: list[str]


# Agent routing table: maps agent names to their wrapper scripts
AGENT_ROUTES = {
    # Default: PluriChat with personas
    "plurichat": "plurichat.py",
    "ring0.architect": "plurichat.py",
    "sota.researcher": "plurichat.py",
    "pr.reviewer": "plurichat.py",
    "codex.peer": "plurichat.py",
    # SOTA Agents
    "opencode": "opencode_wrapper.py",
    "pr-agent": "pr_agent_wrapper.py",
    "agent-s": "agent_s_adapter.py",
    "agent0": "agent0_adapter.py",
}


class IsoExecutor:
    def __init__(self, bus_dir: str = None):
        self.bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        self.env = os.environ.copy()

    def _resolve_agent(self, agent: str) -> tuple[str, str, list[str]]:
        """
        Resolve agent name to wrapper script and build arguments.

        Returns:
            Tuple of (wrapper_path, agent_type, extra_args)
        """
        tools_dir = Path(__file__).parent

        # Check if it's a SOTA agent
        if agent in {"opencode", "opencode-agent"}:
            wrapper = tools_dir / "opencode_wrapper.py"
            return str(wrapper), "opencode", ["run"]

        if agent in {"pr-agent", "pr_agent"}:
            wrapper = tools_dir / "pr_agent_wrapper.py"
            return str(wrapper), "pr-agent", ["check"]

        if agent in {"agent-s", "agent_s", "agent-s3"}:
            wrapper = tools_dir / "agent_s_adapter.py"
            return str(wrapper), "agent-s", ["run"]

        if agent in {"agent0", "agent-0", "agent_0"}:
            wrapper = tools_dir / "agent0_adapter.py"
            return str(wrapper), "agent0", ["plan"]

        # Default to PluriChat with persona
        wrapper = tools_dir / "plurichat.py"
        return str(wrapper), "plurichat", ["--mode", "oneshot", "--persona", agent]

    def spawn(
        self,
        agent: str,
        goal: str,
        context_files: list[str] = None,
        trace_id: str = None,
        parent_id: str = None,
        timeout: int = 120
    ) -> IsoResult:
        """
        Spawn an isolated specialist agent process.

        Args:
            agent: The persona/tool to invoke (e.g. 'opencode', 'pr-agent', 'codex-cli').
            goal: The specific task description.
            context_files: List of file paths to inject as context.
            trace_id: Distributed trace ID for correlation.
            parent_id: Causal parent event ID.
            timeout: Max execution time in seconds.
        """
        run_id = str(uuid.uuid4())

        # 1. Prepare Environment (Isolation)
        child_env = self.env.copy()
        if trace_id:
            child_env["PLURIBUS_TRACE_ID"] = trace_id
        if parent_id:
            child_env["PLURIBUS_PARENT_ID"] = parent_id
        child_env["PLURIBUS_RUN_ID"] = run_id
        child_env["PLURIBUS_ISOLATION"] = "process"
        if self.bus_dir:
            child_env["PLURIBUS_BUS_DIR"] = self.bus_dir

        # 2. Resolve Agent and Construct Command
        wrapper_path, agent_type, base_args = self._resolve_agent(agent)

        # Construct the context payload
        prompt = f"Goal: {goal}\n"
        if context_files:
            prompt += "\nContext:\n"
            for fpath in context_files:
                prompt += f"- {fpath}\n"

        # Build command based on agent type
        if agent_type == "opencode":
            cmd = [
                sys.executable,
                wrapper_path,
                "run",
                "--goal", goal,
                "--json-output",
            ]
            if trace_id:
                cmd.extend(["--trace-id", trace_id])
            if context_files:
                cmd.extend(["--files"] + context_files)

        elif agent_type == "pr-agent":
            # PR-Agent requires a PR URL, so we parse it from goal
            # Format: "Review PR: https://github.com/..."
            pr_url = None
            for word in goal.split():
                if "github.com" in word and "/pull/" in word:
                    pr_url = word.strip()
                    break
                if "gitlab.com" in word and "/merge_requests/" in word:
                    pr_url = word.strip()
                    break

            if pr_url:
                cmd = [
                    sys.executable,
                    wrapper_path,
                    "review",
                    "--pr-url", pr_url,
                    "--json-output",
                ]
                if trace_id:
                    cmd.extend(["--trace-id", trace_id])
            else:
                # Fallback: just run check
                cmd = [sys.executable, wrapper_path, "check"]

        elif agent_type == "agent-s":
            cmd = [
                sys.executable,
                wrapper_path,
                "run",
                "--goal", goal,
                "--json-output",
            ]
            if trace_id:
                cmd.extend(["--trace-id", trace_id])

        elif agent_type == "agent0":
            cmd = [
                sys.executable,
                wrapper_path,
                "plan",
                "--goal", goal,
                "--json-output",
            ]
            if trace_id:
                cmd.extend(["--run-id", trace_id])

        else:
            # Default: PluriChat
            cmd = [
                sys.executable,
                wrapper_path,
                "--mode", "oneshot",
                "--persona", agent,
                "--prompt", prompt,
                "--json-output",
            ]

        # 3. Execute (Transduction)
        try:
            start_ts = time.time()
            proc = subprocess.run(
                cmd,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            duration = time.time() - start_ts

            # 4. Parse artifacts from JSON output
            artifacts = []
            try:
                data = json.loads(proc.stdout)
                if isinstance(data, dict):
                    artifacts = data.get("files_modified", []) or data.get("artifacts", [])
            except (json.JSONDecodeError, TypeError):
                pass

            return IsoResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                artifacts=artifacts
            )

        except subprocess.TimeoutExpired:
            return IsoResult(
                exit_code=124,
                stdout="",
                stderr=f"Timeout after {timeout}s",
                artifacts=[]
            )
        except Exception as e:
            return IsoResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                artifacts=[]
            )

    def spawn_opencode(
        self,
        goal: str,
        working_dir: str = None,
        trace_id: str = None,
        timeout: int = 300
    ) -> IsoResult:
        """
        Convenience method to spawn OpenCode specifically.

        Args:
            goal: Task description for OpenCode.
            working_dir: Optional working directory.
            trace_id: Trace ID for correlation.
            timeout: Max execution time.
        """
        child_env = self.env.copy()
        if trace_id:
            child_env["PLURIBUS_TRACE_ID"] = trace_id
        if self.bus_dir:
            child_env["PLURIBUS_BUS_DIR"] = self.bus_dir

        wrapper = Path(__file__).parent / "opencode_wrapper.py"
        cmd = [
            sys.executable,
            str(wrapper),
            "run",
            "--goal", goal,
            "--json-output",
        ]
        if trace_id:
            cmd.extend(["--trace-id", trace_id])
        if working_dir:
            cmd.extend(["--working-dir", working_dir])

        try:
            proc = subprocess.run(
                cmd,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
            )
            return IsoResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                artifacts=[]
            )
        except subprocess.TimeoutExpired:
            return IsoResult(exit_code=124, stdout="", stderr=f"Timeout after {timeout}s", artifacts=[])
        except Exception as e:
            return IsoResult(exit_code=1, stdout="", stderr=str(e), artifacts=[])

    def spawn_pr_agent(
        self,
        pr_url: str,
        command: str = "review",
        trace_id: str = None,
        timeout: int = 300
    ) -> IsoResult:
        """
        Convenience method to spawn PR-Agent specifically.

        Args:
            pr_url: GitHub/GitLab PR URL.
            command: review, improve, describe, or test.
            trace_id: Trace ID for correlation.
            timeout: Max execution time.
        """
        child_env = self.env.copy()
        if trace_id:
            child_env["PLURIBUS_TRACE_ID"] = trace_id
        if self.bus_dir:
            child_env["PLURIBUS_BUS_DIR"] = self.bus_dir

        wrapper = Path(__file__).parent / "pr_agent_wrapper.py"
        cmd = [
            sys.executable,
            str(wrapper),
            command,
            "--pr-url", pr_url,
            "--json-output",
        ]
        if trace_id:
            cmd.extend(["--trace-id", trace_id])

        try:
            proc = subprocess.run(
                cmd,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return IsoResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                artifacts=[]
            )
        except subprocess.TimeoutExpired:
            return IsoResult(exit_code=124, stdout="", stderr=f"Timeout after {timeout}s", artifacts=[])
        except Exception as e:
            return IsoResult(exit_code=1, stdout="", stderr=str(e), artifacts=[])

def main():
    # Simple CLI test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--agent", default="codex-cli")
    args = parser.parse_args()
    
    exc = IsoExecutor()
    res = exc.spawn(args.agent, args.goal, trace_id=str(uuid.uuid4()))
    print(json.dumps(res.__dict__, indent=2))

if __name__ == "__main__":
    main()
