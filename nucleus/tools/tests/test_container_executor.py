#!/usr/bin/env python3
"""
Tests for ContainerExecutor - Microsandbox-ready containerized execution.

Tests cover:
- Runtime detection
- Isolation levels
- Network policies
- Ring 0 path protection
- Overlay filesystem
- Mock/dry-run mode
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from container_executor import (
    ContainerExecutor,
    ContainerResult,
    IsolationLevel,
    NetworkPolicy,
    ResourceLimits,
    OverlayConfig,
    RuntimeDetector,
    SeccompProfile,
    RING0_PATHS,
)


class TestRuntimeDetector:
    """Tests for RuntimeDetector."""

    def test_detect_best_runtime_no_runtimes(self):
        """Without any runtimes, should return mock."""
        with patch.object(RuntimeDetector, "has_kvm", return_value=False), \
             patch.object(RuntimeDetector, "has_firecracker", return_value=False), \
             patch.object(RuntimeDetector, "has_gvisor", return_value=False), \
             patch.object(RuntimeDetector, "has_docker", return_value=False), \
             patch.object(RuntimeDetector, "has_podman", return_value=False):
            runtime, level = RuntimeDetector.detect_best_runtime()
            assert runtime == "mock"
            assert level == IsolationLevel.MOCK

    def test_detect_best_runtime_docker_only(self):
        """With only Docker, should return docker/standard."""
        with patch.object(RuntimeDetector, "has_kvm", return_value=False), \
             patch.object(RuntimeDetector, "has_firecracker", return_value=False), \
             patch.object(RuntimeDetector, "has_gvisor", return_value=False), \
             patch.object(RuntimeDetector, "has_docker", return_value=True), \
             patch.object(RuntimeDetector, "has_podman", return_value=False):
            runtime, level = RuntimeDetector.detect_best_runtime()
            assert runtime == "docker"
            assert level == IsolationLevel.STANDARD

    def test_detect_best_runtime_firecracker(self):
        """With KVM and Firecracker, should prefer firecracker."""
        with patch.object(RuntimeDetector, "has_kvm", return_value=True), \
             patch.object(RuntimeDetector, "has_firecracker", return_value=True), \
             patch.object(RuntimeDetector, "has_gvisor", return_value=True), \
             patch.object(RuntimeDetector, "has_docker", return_value=True), \
             patch.object(RuntimeDetector, "has_podman", return_value=True):
            runtime, level = RuntimeDetector.detect_best_runtime()
            assert runtime == "firecracker"
            assert level == IsolationLevel.STRICT

    def test_detect_best_runtime_gvisor(self):
        """With gVisor (no KVM), should prefer gVisor."""
        with patch.object(RuntimeDetector, "has_kvm", return_value=False), \
             patch.object(RuntimeDetector, "has_firecracker", return_value=False), \
             patch.object(RuntimeDetector, "has_gvisor", return_value=True), \
             patch.object(RuntimeDetector, "has_docker", return_value=True), \
             patch.object(RuntimeDetector, "has_podman", return_value=False):
            runtime, level = RuntimeDetector.detect_best_runtime()
            assert runtime == "gvisor"
            assert level == IsolationLevel.STRICT

    def test_get_capabilities(self):
        """Capabilities report should include all fields."""
        caps = RuntimeDetector.get_capabilities()
        assert "kvm" in caps
        assert "firecracker" in caps
        assert "gvisor" in caps
        assert "docker" in caps
        assert "podman" in caps
        assert "recommended" in caps


class TestSeccompProfile:
    """Tests for SeccompProfile."""

    def test_generate_strict_profile(self):
        """Strict profile should have deny-by-default."""
        profile = SeccompProfile.generate_strict()
        assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
        assert "syscalls" in profile
        assert len(profile["syscalls"]) > 0
        assert profile["syscalls"][0]["action"] == "SCMP_ACT_ALLOW"

    def test_write_profile(self):
        """Profile should be writable to file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            SeccompProfile.write_profile(path)
            assert path.exists()
            content = json.loads(path.read_text())
            assert "defaultAction" in content
        finally:
            path.unlink()


class TestContainerExecutor:
    """Tests for ContainerExecutor."""

    def test_init_mock_mode(self):
        """Executor should initialize in mock mode without runtimes."""
        with patch.object(RuntimeDetector, "detect_best_runtime",
                         return_value=("mock", IsolationLevel.MOCK)):
            executor = ContainerExecutor()
            assert executor.dry_run is True
            assert executor.runtime == "mock"

    def test_spawn_dry_run(self):
        """Dry-run spawn should return mock result."""
        executor = ContainerExecutor(runtime="mock")
        result = executor.spawn(
            image="alpine:latest",
            command=["echo", "hello"],
            trace_id="test-123"
        )

        assert result.exit_code == 0
        assert result.runtime == "mock"
        assert "dry_run" in result.stdout

        # Parse stdout as JSON
        data = json.loads(result.stdout)
        assert data["status"] == "dry_run"
        assert data["image"] == "alpine:latest"
        assert data["trace_id"] == "test-123"

    def test_spawn_with_env(self):
        """Environment variables should be passed."""
        executor = ContainerExecutor(runtime="mock")
        result = executor.spawn(
            image="alpine:latest",
            command=["env"],
            env={"MY_VAR": "my_value"},
            trace_id="env-test"
        )

        data = json.loads(result.stdout)
        assert "MY_VAR" in data["env_keys"]
        assert "PLURIBUS_TRACE_ID" in data["env_keys"]

    def test_spawn_with_isolation_level(self):
        """Isolation level should be recorded."""
        executor = ContainerExecutor(runtime="mock")
        result = executor.spawn(
            image="alpine:latest",
            command=["echo"],
            isolation_level=IsolationLevel.STRICT,
        )

        assert result.isolation_level == "strict"

    def test_spawn_with_network_policy(self):
        """Network policy should be recorded."""
        executor = ContainerExecutor(runtime="mock")
        result = executor.spawn(
            image="alpine:latest",
            command=["echo"],
            network_policy=NetworkPolicy.NONE,
        )

        assert result.network_policy == "none"

    def test_validate_mounts_ring0_protection(self):
        """Ring 0 paths should be filtered from mounts."""
        executor = ContainerExecutor(runtime="mock")

        mounts = {
            "/safe/path": "/container/safe",
            "/path/to/.pluribus/constitution.md": "/container/constitution",
            "/other/AGENTS.md": "/container/agents",
        }

        safe = executor._validate_mounts(mounts)

        # Only safe path should pass
        assert "/safe/path" in safe
        # Ring 0 paths should be filtered
        assert len(safe) == 1

    def test_setup_overlay(self):
        """Overlay setup should create directories."""
        executor = ContainerExecutor(runtime="mock")
        config = OverlayConfig(enabled=True)

        overlay_root = executor._setup_overlay(config)

        assert overlay_root is not None
        assert Path(overlay_root).exists()
        assert config.upper_dir is not None
        assert Path(config.upper_dir).exists()
        assert config.work_dir is not None
        assert Path(config.work_dir).exists()

        # Cleanup
        import shutil
        shutil.rmtree(overlay_root)

    def test_build_network_args_none(self):
        """Network none should disable networking."""
        executor = ContainerExecutor(runtime="mock")
        args = executor._build_network_args(NetworkPolicy.NONE)
        assert "--network" in args
        assert "none" in args

    def test_build_network_args_full(self):
        """Network full should have no restrictions."""
        executor = ContainerExecutor(runtime="mock")
        args = executor._build_network_args(NetworkPolicy.FULL)
        assert args == []

    def test_build_resource_args(self):
        """Resource limits should be converted to CLI args."""
        executor = ContainerExecutor(runtime="mock")
        limits = ResourceLimits(cpu_cores=2.0, memory_mb=1024, pids_limit=200)
        args = executor._build_resource_args(limits)

        assert "--cpus" in args
        assert "2.0" in args
        assert "--memory" in args
        assert "1024m" in args
        assert "--pids-limit" in args
        assert "200" in args

    def test_execute_code_python(self):
        """Code execution should use correct image and command."""
        executor = ContainerExecutor(runtime="mock")
        result = executor.execute_code(
            code="print('hello')",
            language="python",
            trace_id="code-test"
        )

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "python" in data["image"]
        assert "python3" in data["cmd"][0]

    def test_execute_code_node(self):
        """Node code execution should use node image."""
        executor = ContainerExecutor(runtime="mock")
        result = executor.execute_code(
            code="console.log('hello')",
            language="node",
        )

        data = json.loads(result.stdout)
        assert "node" in data["image"]

    def test_get_capabilities(self):
        """Capabilities should include current runtime info."""
        executor = ContainerExecutor(runtime="mock")
        caps = executor.get_capabilities()

        assert "current_runtime" in caps
        assert "current_isolation" in caps
        assert "dry_run" in caps
        assert caps["dry_run"] is True


class TestContainerResult:
    """Tests for ContainerResult dataclass."""

    def test_result_fields(self):
        """Result should have all expected fields."""
        result = ContainerResult(
            exit_code=0,
            stdout="output",
            stderr="error",
            container_id="test-123",
            runtime="docker",
            isolation_level="standard",
            network_policy="none",
            duration_ms=100,
            overlay_artifacts="/tmp/overlay",
        )

        assert result.exit_code == 0
        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.container_id == "test-123"
        assert result.runtime == "docker"
        assert result.isolation_level == "standard"
        assert result.network_policy == "none"
        assert result.duration_ms == 100
        assert result.overlay_artifacts == "/tmp/overlay"


class TestIsolationLevels:
    """Tests for isolation level enum."""

    def test_isolation_values(self):
        """Isolation levels should have correct values."""
        assert IsolationLevel.STRICT.value == "strict"
        assert IsolationLevel.STANDARD.value == "standard"
        assert IsolationLevel.RELAXED.value == "relaxed"
        assert IsolationLevel.MOCK.value == "mock"


class TestNetworkPolicy:
    """Tests for network policy enum."""

    def test_network_values(self):
        """Network policies should have correct values."""
        assert NetworkPolicy.NONE.value == "none"
        assert NetworkPolicy.EGRESS_ONLY.value == "egress"
        assert NetworkPolicy.ALLOW_DNS.value == "dns"
        assert NetworkPolicy.FULL.value == "full"


class TestRing0Protection:
    """Tests for Ring 0 path protection."""

    def test_ring0_paths_defined(self):
        """Ring 0 paths should be defined."""
        assert len(RING0_PATHS) > 0
        assert ".pluribus/constitution.md" in RING0_PATHS
        assert "AGENTS.md" in RING0_PATHS

    def test_ring0_not_mountable(self):
        """Ring 0 paths should not be mountable."""
        executor = ContainerExecutor(runtime="mock")

        for r0_path in RING0_PATHS:
            mounts = {f"/some/prefix/{r0_path}": "/container/path"}
            safe = executor._validate_mounts(mounts)
            assert len(safe) == 0, f"Ring 0 path {r0_path} should be blocked"


class TestIntegration:
    """Integration tests (require actual runtime)."""

    @pytest.mark.skipif(
        not RuntimeDetector.has_docker() and not RuntimeDetector.has_podman(),
        reason="No container runtime available"
    )
    def test_real_container_spawn(self):
        """Test actual container spawn if runtime available."""
        executor = ContainerExecutor()

        if executor.dry_run:
            pytest.skip("No container runtime")

        result = executor.spawn(
            image="alpine:latest",
            command=["echo", "pluribus-test"],
            timeout=30,
        )

        assert result.exit_code == 0
        assert "pluribus-test" in result.stdout

    @pytest.mark.skipif(
        not RuntimeDetector.has_docker() and not RuntimeDetector.has_podman(),
        reason="No container runtime available"
    )
    def test_network_isolation(self):
        """Test network isolation prevents connectivity."""
        executor = ContainerExecutor()

        if executor.dry_run:
            pytest.skip("No container runtime")

        result = executor.spawn(
            image="alpine:latest",
            command=["ping", "-c", "1", "-W", "1", "8.8.8.8"],
            network_policy=NetworkPolicy.NONE,
            timeout=10,
        )

        # Should fail due to network isolation
        assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
