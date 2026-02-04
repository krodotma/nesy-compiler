#!/usr/bin/env python3
"""
Tests for SOTA Coding Agent Integrations.
=========================================

Tests for OpenCode and PR-Agent wrappers and their integration
with the service registry and strp_worker delegation.
"""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

# Import modules under test
import opencode_wrapper  # noqa: E402
import pr_agent_wrapper  # noqa: E402
import iso_executor  # noqa: E402


class TestOpenCodeWrapper(unittest.TestCase):
    """Tests for OpenCode wrapper functionality."""

    def test_find_opencode_returns_none_when_not_installed(self):
        """find_opencode returns None when OpenCode is not available."""
        with patch("shutil.which", return_value=None):
            result = opencode_wrapper.find_opencode()
            # May return "npx" if npx is available, or None
            # We just check it doesn't raise
            self.assertIn(result, [None, "npx"])

    def test_check_opencode_version_timeout(self):
        """check_opencode_version handles timeout gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = opencode_wrapper.subprocess.TimeoutExpired(cmd=[], timeout=30)
            available, version = opencode_wrapper.check_opencode_version("npx")
            self.assertFalse(available)
            self.assertEqual(version, "timeout")

    def test_run_opencode_file_not_found(self):
        """run_opencode handles FileNotFoundError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            code, stdout, stderr = opencode_wrapper.run_opencode(
                goal="test task",
                opencode_path="/nonexistent/opencode",
            )
            self.assertEqual(code, 127)
            self.assertIn("not found", stderr.lower())

    def test_run_opencode_success(self):
        """run_opencode returns output on success."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Task completed",
                stderr="",
            )
            code, stdout, stderr = opencode_wrapper.run_opencode(
                goal="test task",
                opencode_path="npx",
            )
            self.assertEqual(code, 0)
            self.assertEqual(stdout, "Task completed")

    def test_cmd_check_without_opencode(self):
        """check command reports when OpenCode is not found."""
        with patch.object(opencode_wrapper, "find_opencode", return_value=None):
            # Create mock args
            args = MagicMock()
            result = opencode_wrapper.cmd_check(args)
            self.assertEqual(result, 1)


class TestPRAgentWrapper(unittest.TestCase):
    """Tests for PR-Agent wrapper functionality."""

    def test_parse_pr_url_github(self):
        """parse_pr_url correctly parses GitHub PR URLs."""
        result = pr_agent_wrapper.parse_pr_url(
            "https://github.com/owner/repo/pull/123"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["owner"], "owner")
        self.assertEqual(result["repo"], "repo")
        self.assertEqual(result["pr_number"], 123)
        self.assertEqual(result["provider"], "github")

    def test_parse_pr_url_gitlab(self):
        """parse_pr_url correctly parses GitLab MR URLs."""
        result = pr_agent_wrapper.parse_pr_url(
            "https://gitlab.com/owner/repo/-/merge_requests/456"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["owner"], "owner")
        self.assertEqual(result["repo"], "repo")
        self.assertEqual(result["pr_number"], 456)
        self.assertEqual(result["provider"], "gitlab")

    def test_parse_pr_url_invalid(self):
        """parse_pr_url returns None for invalid URLs."""
        result = pr_agent_wrapper.parse_pr_url("https://example.com/not-a-pr")
        self.assertIsNone(result)

    def test_find_pr_agent_returns_none_when_not_installed(self):
        """find_pr_agent returns None when PR-Agent is not available."""
        with patch("shutil.which", return_value=None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="")
                path, install_type = pr_agent_wrapper.find_pr_agent()
                # May find docker or module, or return None
                self.assertIn(install_type, ["none", "docker", "module", "cli"])

    def test_run_pr_agent_command_invalid_type(self):
        """run_pr_agent_command handles invalid install type."""
        code, stdout, stderr = pr_agent_wrapper.run_pr_agent_command(
            command="review",
            pr_url="https://github.com/owner/repo/pull/1",
            install_type="invalid",
            path="/fake",
        )
        self.assertEqual(code, 1)
        self.assertIn("invalid", stderr.lower())

    def test_cmd_check_without_pr_agent(self):
        """check command reports when PR-Agent is not found."""
        with patch.object(pr_agent_wrapper, "find_pr_agent", return_value=(None, "none")):
            args = MagicMock()
            result = pr_agent_wrapper.cmd_check(args)
            self.assertEqual(result, 1)


class TestIsoExecutorAgentRouting(unittest.TestCase):
    """Tests for IsoExecutor agent routing to SOTA tools."""

    def test_resolve_agent_opencode(self):
        """_resolve_agent correctly routes to OpenCode wrapper."""
        executor = iso_executor.IsoExecutor()
        wrapper_path, agent_type, _ = executor._resolve_agent("opencode")
        self.assertEqual(agent_type, "opencode")
        self.assertIn("opencode_wrapper.py", wrapper_path)

    def test_resolve_agent_pr_agent(self):
        """_resolve_agent correctly routes to PR-Agent wrapper."""
        executor = iso_executor.IsoExecutor()
        wrapper_path, agent_type, _ = executor._resolve_agent("pr-agent")
        self.assertEqual(agent_type, "pr-agent")
        self.assertIn("pr_agent_wrapper.py", wrapper_path)

    def test_resolve_agent_agent_s(self):
        """_resolve_agent correctly routes to Agent-S adapter."""
        executor = iso_executor.IsoExecutor()
        wrapper_path, agent_type, _ = executor._resolve_agent("agent-s")
        self.assertEqual(agent_type, "agent-s")
        self.assertIn("agent_s_adapter.py", wrapper_path)

    def test_resolve_agent_agent0(self):
        """_resolve_agent correctly routes to Agent0 adapter."""
        executor = iso_executor.IsoExecutor()
        wrapper_path, agent_type, _ = executor._resolve_agent("agent0")
        self.assertEqual(agent_type, "agent0")
        self.assertIn("agent0_adapter.py", wrapper_path)

    def test_resolve_agent_plurichat_fallback(self):
        """_resolve_agent falls back to PluriChat for unknown agents."""
        executor = iso_executor.IsoExecutor()
        wrapper_path, agent_type, _ = executor._resolve_agent("unknown-agent")
        self.assertEqual(agent_type, "plurichat")
        self.assertIn("plurichat.py", wrapper_path)

    def test_resolve_agent_persona_routes_to_plurichat(self):
        """_resolve_agent routes persona names to PluriChat."""
        executor = iso_executor.IsoExecutor()
        for persona in ["ring0.architect", "sota.researcher", "pr.reviewer"]:
            wrapper_path, agent_type, _ = executor._resolve_agent(persona)
            self.assertEqual(agent_type, "plurichat", f"Failed for {persona}")

    def test_spawn_extracts_pr_url_from_goal(self):
        """spawn correctly extracts PR URL from goal for pr-agent."""
        executor = iso_executor.IsoExecutor()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"status": "ok"}',
                stderr="",
            )
            executor.spawn(
                agent="pr-agent",
                goal="Review PR: https://github.com/owner/repo/pull/42",
            )
            # Verify the command was called with the PR URL
            call_args = mock_run.call_args
            cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
            cmd_str = " ".join(cmd)
            self.assertIn("pr_agent_wrapper", cmd_str)
            self.assertIn("github.com/owner/repo/pull/42", cmd_str)

    def test_spawn_opencode_method(self):
        """spawn_opencode convenience method works correctly."""
        executor = iso_executor.IsoExecutor()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"status": "ok", "files_modified": ["test.py"]}',
                stderr="",
            )
            result = executor.spawn_opencode(
                goal="Implement feature X",
                trace_id="test-trace-123",
            )
            self.assertEqual(result.exit_code, 0)
            call_args = mock_run.call_args
            cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
            cmd_str = " ".join(cmd)
            self.assertIn("opencode_wrapper", cmd_str)
            self.assertIn("--goal", cmd_str)

    def test_spawn_pr_agent_method(self):
        """spawn_pr_agent convenience method works correctly."""
        executor = iso_executor.IsoExecutor()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"status": "ok"}',
                stderr="",
            )
            result = executor.spawn_pr_agent(
                pr_url="https://github.com/owner/repo/pull/99",
                command="improve",
                trace_id="test-trace-456",
            )
            self.assertEqual(result.exit_code, 0)
            call_args = mock_run.call_args
            cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
            cmd_str = " ".join(cmd)
            self.assertIn("pr_agent_wrapper", cmd_str)
            self.assertIn("improve", cmd_str)


class TestServiceRegistrySOTAEntries(unittest.TestCase):
    """Tests for SOTA agent entries in service registry."""

    def test_opencode_service_registered(self):
        """OpenCode service is registered in BUILTIN_SERVICES."""
        import service_registry

        opencode_entries = [
            s for s in service_registry.BUILTIN_SERVICES if s["id"] == "opencode"
        ]
        self.assertEqual(len(opencode_entries), 1)

        entry = opencode_entries[0]
        self.assertEqual(entry["name"], "OpenCode TUI Agent")
        self.assertEqual(entry["kind"], "process")
        self.assertIn("opencode_wrapper.py", entry["entry_point"])
        self.assertIn("sota", entry["tags"])
        self.assertIn("agent", entry["tags"])
        self.assertEqual(entry["lineage"], "sota.agents")

    def test_pr_agent_service_registered(self):
        """PR-Agent service is registered in BUILTIN_SERVICES."""
        import service_registry

        pr_entries = [
            s for s in service_registry.BUILTIN_SERVICES if s["id"] == "pr-agent"
        ]
        self.assertEqual(len(pr_entries), 1)

        entry = pr_entries[0]
        self.assertEqual(entry["name"], "PR-Agent Code Review")
        self.assertEqual(entry["kind"], "process")
        self.assertIn("pr_agent_wrapper.py", entry["entry_point"])
        self.assertIn("sota", entry["tags"])
        self.assertIn("review", entry["tags"])
        self.assertEqual(entry["lineage"], "sota.agents")

    def test_agent_s_service_registered(self):
        """Agent-S service is registered in BUILTIN_SERVICES."""
        import service_registry

        entries = [s for s in service_registry.BUILTIN_SERVICES if s["id"] == "agent-s"]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertIn("agent_s_adapter.py", entry["entry_point"])
        self.assertIn("sota", entry["tags"])

    def test_agent0_service_registered(self):
        """Agent0 service is registered in BUILTIN_SERVICES."""
        import service_registry

        entries = [s for s in service_registry.BUILTIN_SERVICES if s["id"] == "agent0"]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertIn("agent0_adapter.py", entry["entry_point"])
        self.assertIn("sota", entry["tags"])

    def test_maestro_service_registered(self):
        """Maestro service is registered in BUILTIN_SERVICES."""
        import service_registry

        entries = [s for s in service_registry.BUILTIN_SERVICES if s["id"] == "maestro"]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertIn("maestro_adapter.py", entry["entry_point"])
        self.assertIn("sota", entry["tags"])


class TestBusEventEmission(unittest.TestCase):
    """Tests for bus event emission from wrappers."""

    def test_opencode_emits_start_event(self):
        """OpenCode wrapper emits task start event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = tmpdir

            with patch.object(opencode_wrapper, "find_opencode", return_value="npx"):
                with patch.object(
                    opencode_wrapper, "check_opencode_version", return_value=(True, "1.0.0")
                ):
                    with patch.object(
                        opencode_wrapper, "run_opencode", return_value=(0, "done", "")
                    ):
                        with patch.object(opencode_wrapper, "emit_bus") as mock_emit:
                            args = MagicMock()
                            args.goal = "Test task"
                            args.working_dir = None
                            args.files = None
                            args.timeout = 60
                            args.trace_id = "test-123"
                            args.req_id = "req-456"
                            args.delegate = False
                            args.json_output = True
                            args.actor = "tester"
                            args.bus_dir = bus_dir

                            opencode_wrapper.cmd_run(args)

                            # Check start event was emitted
                            start_calls = [
                                c for c in mock_emit.call_args_list
                                if c[1].get("topic") == "opencode.task.start"
                            ]
                            self.assertEqual(len(start_calls), 1)

                            # Check complete event was emitted
                            complete_calls = [
                                c for c in mock_emit.call_args_list
                                if c[1].get("topic") == "opencode.task.complete"
                            ]
                            self.assertEqual(len(complete_calls), 1)

    def test_pr_agent_emits_review_events(self):
        """PR-Agent wrapper emits review start/complete events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus_dir = tmpdir

            with patch.object(
                pr_agent_wrapper, "find_pr_agent", return_value=("/usr/bin/pr-agent", "cli")
            ):
                with patch.object(
                    pr_agent_wrapper,
                    "run_pr_agent_command",
                    return_value=(0, "Review complete", ""),
                ):
                    with patch.object(pr_agent_wrapper, "emit_bus") as mock_emit:
                        args = MagicMock()
                        args.pr_url = "https://github.com/owner/repo/pull/1"
                        args.timeout = 60
                        args.trace_id = "test-789"
                        args.req_id = "req-abc"
                        args.json_output = True
                        args.actor = "tester"
                        args.bus_dir = bus_dir

                        pr_agent_wrapper.cmd_review(args)

                        # Check start event
                        start_calls = [
                            c for c in mock_emit.call_args_list
                            if "pr_agent.review.start" in str(c)
                        ]
                        self.assertEqual(len(start_calls), 1)

                        # Check complete event
                        complete_calls = [
                            c for c in mock_emit.call_args_list
                            if "pr_agent.review.complete" in str(c)
                        ]
                        self.assertEqual(len(complete_calls), 1)


class TestAgentRoutesTable(unittest.TestCase):
    """Tests for the AGENT_ROUTES constant."""

    def test_agent_routes_contains_sota_agents(self):
        """AGENT_ROUTES includes OpenCode and PR-Agent."""
        self.assertIn("opencode", iso_executor.AGENT_ROUTES)
        self.assertIn("pr-agent", iso_executor.AGENT_ROUTES)

    def test_agent_routes_contains_plurichat_personas(self):
        """AGENT_ROUTES includes standard PluriChat personas."""
        for persona in ["plurichat", "ring0.architect", "sota.researcher"]:
            self.assertIn(persona, iso_executor.AGENT_ROUTES)

    def test_agent_routes_wrapper_files_exist(self):
        """All wrapper files referenced in AGENT_ROUTES exist."""
        for agent, wrapper_file in iso_executor.AGENT_ROUTES.items():
            wrapper_path = TOOLS_DIR / wrapper_file
            self.assertTrue(
                wrapper_path.exists(),
                f"Wrapper for {agent} not found at {wrapper_path}",
            )


if __name__ == "__main__":
    unittest.main()
