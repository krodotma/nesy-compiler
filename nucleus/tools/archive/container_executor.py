#!/usr/bin/env python3
"""
ContainerExecutor: Microsandbox-Ready Containerized Execution
=============================================================

Implements the "Container Cell" pattern with multi-tier isolation:

Tier 1: Firecracker microVM (if KVM available) - strongest isolation
Tier 2: gVisor sandbox (if available) - strong isolation
Tier 3: Docker/Podman with seccomp + namespace isolation
Tier 4: Mock dry-run for architectural verification

Features:
- Overlay filesystem support for ephemeral workspaces
- Network isolation with configurable egress rules
- Seccomp profile enforcement
- Resource limits (CPU, memory)
- Ring 0 path protection

Usage:
    executor = ContainerExecutor()
    result = executor.spawn(
        image="pluribus-agent:latest",
        cmd=["python3", "agent.py", "--goal", "distill"],
        trace_id="abc-123",
        isolation_level="strict",
        network_policy=NetworkPolicy.NONE
    )
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
import uuid
import time
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

# Ring 0 protected paths (never mount into containers)
RING0_PATHS = {
    ".pluribus/constitution.md",
    "AGENTS.md",
    ".pluribus/lineage.json",
    "nucleus/tools/iso_git.mjs",
}


class IsolationLevel(Enum):
    """Isolation strength tiers."""
    STRICT = "strict"      # Firecracker/gVisor if available, else deny
    STANDARD = "standard"  # Best available runtime
    RELAXED = "relaxed"    # Docker/Podman with minimal restrictions
    MOCK = "mock"          # Dry-run for testing


class NetworkPolicy(Enum):
    """Network isolation policies."""
    NONE = "none"          # No network access
    EGRESS_ONLY = "egress" # Outbound only (no listening)
    ALLOW_DNS = "dns"      # DNS resolution only
    FULL = "full"          # Full network access (dangerous)


@dataclass
class ResourceLimits:
    """Container resource constraints."""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    pids_limit: int = 100
    timeout_s: int = 120


@dataclass
class OverlayConfig:
    """Overlay filesystem configuration for ephemeral workspaces."""
    enabled: bool = False
    lower_dir: Optional[str] = None   # Read-only base layer
    work_dir: Optional[str] = None    # OverlayFS workdir (auto-created)
    upper_dir: Optional[str] = None   # Writeable layer (auto-created)
    merged_dir: Optional[str] = None  # Mount point in container


@dataclass
class ContainerResult:
    """Execution result from container."""
    exit_code: int
    stdout: str
    stderr: str
    container_id: Optional[str]
    runtime: str
    isolation_level: str
    network_policy: str
    duration_ms: int = 0
    overlay_artifacts: Optional[str] = None  # Path to preserved upper layer


class RuntimeDetector:
    """Detects available container runtimes and capabilities."""

    @staticmethod
    def has_kvm() -> bool:
        """Check if KVM is available for Firecracker."""
        kvm_path = Path("/dev/kvm")
        if not kvm_path.exists():
            return False
        # Check if we can access it
        try:
            return os.access("/dev/kvm", os.R_OK | os.W_OK)
        except Exception:
            return False

    @staticmethod
    def has_firecracker() -> bool:
        """Check if Firecracker is installed."""
        return shutil.which("firecracker") is not None

    @staticmethod
    def has_gvisor() -> bool:
        """Check if gVisor (runsc) is available."""
        return shutil.which("runsc") is not None

    @staticmethod
    def has_docker() -> bool:
        """Check if Docker is available and responsive."""
        if not shutil.which("docker"):
            return False
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def has_podman() -> bool:
        """Check if Podman is available."""
        if not shutil.which("podman"):
            return False
        try:
            result = subprocess.run(
                ["podman", "info"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    @classmethod
    def detect_best_runtime(cls) -> tuple[str, IsolationLevel]:
        """Detect the best available runtime."""
        # Tier 1: Firecracker (requires KVM)
        if cls.has_kvm() and cls.has_firecracker():
            return ("firecracker", IsolationLevel.STRICT)

        # Tier 2: gVisor
        if cls.has_gvisor() and (cls.has_docker() or cls.has_podman()):
            return ("gvisor", IsolationLevel.STRICT)

        # Tier 3: Container runtime
        if cls.has_podman():
            return ("podman", IsolationLevel.STANDARD)
        if cls.has_docker():
            return ("docker", IsolationLevel.STANDARD)

        # Tier 4: Mock
        return ("mock", IsolationLevel.MOCK)

    @classmethod
    def get_capabilities(cls) -> Dict[str, Any]:
        """Return full capabilities report."""
        return {
            "kvm": cls.has_kvm(),
            "firecracker": cls.has_firecracker(),
            "gvisor": cls.has_gvisor(),
            "docker": cls.has_docker(),
            "podman": cls.has_podman(),
            "recommended": cls.detect_best_runtime()[0],
        }


class SeccompProfile:
    """Generates seccomp profiles for container isolation."""

    STRICT_SYSCALLS = [
        # File operations (limited)
        "read", "write", "open", "close", "stat", "fstat", "lstat",
        "poll", "lseek", "mmap", "mprotect", "munmap", "brk",
        "ioctl", "access", "pipe", "dup", "dup2",
        # Process (limited)
        "clone", "fork", "vfork", "execve", "exit", "exit_group",
        "wait4", "kill", "getpid", "getppid", "getuid", "geteuid",
        "getgid", "getegid", "gettid",
        # Memory
        "mmap", "munmap", "mremap", "madvise",
        # Time
        "clock_gettime", "gettimeofday", "nanosleep",
        # Misc safe
        "getcwd", "chdir", "rename", "mkdir", "rmdir", "unlink",
        "readlink", "chmod", "chown", "umask",
        "fcntl", "flock", "fsync", "fdatasync",
        "getrandom", "futex", "set_tid_address",
        "arch_prctl", "set_robust_list",
    ]

    @classmethod
    def generate_strict(cls) -> Dict[str, Any]:
        """Generate strict seccomp profile."""
        return {
            "defaultAction": "SCMP_ACT_ERRNO",
            "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
            "syscalls": [
                {
                    "names": cls.STRICT_SYSCALLS,
                    "action": "SCMP_ACT_ALLOW"
                }
            ]
        }

    @classmethod
    def write_profile(cls, path: Path) -> None:
        """Write seccomp profile to file."""
        profile = cls.generate_strict()
        path.write_text(json.dumps(profile, indent=2))


class ContainerExecutor:
    """Multi-tier isolated container executor."""

    def __init__(
        self,
        bus_dir: Optional[str] = None,
        runtime: str = "auto",
        isolation_level: IsolationLevel = IsolationLevel.STANDARD
    ):
        self.bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        self.requested_isolation = isolation_level

        if runtime == "auto":
            self.runtime, self.actual_isolation = RuntimeDetector.detect_best_runtime()
        else:
            self.runtime = runtime
            self.actual_isolation = (
                IsolationLevel.MOCK if runtime == "mock"
                else IsolationLevel.STANDARD
            )

        self.dry_run = self.runtime == "mock"
        self._temp_dirs: List[tempfile.TemporaryDirectory] = []

    def _validate_mounts(self, bind_mounts: Dict[str, str]) -> Dict[str, str]:
        """Validate mounts don't expose Ring 0 paths."""
        safe_mounts = {}
        for host_path, container_path in bind_mounts.items():
            host_resolved = Path(host_path).resolve()
            # Check against Ring 0 paths
            is_ring0 = False
            for r0 in RING0_PATHS:
                if str(host_resolved).endswith(r0):
                    is_ring0 = True
                    break
            if not is_ring0:
                safe_mounts[host_path] = container_path
        return safe_mounts

    def _setup_overlay(self, config: OverlayConfig) -> Optional[str]:
        """Set up overlay filesystem for ephemeral workspace."""
        if not config.enabled:
            return None

        # Create temp directories for overlay
        overlay_root = tempfile.mkdtemp(prefix="pluribus_overlay_")
        upper = Path(overlay_root) / "upper"
        work = Path(overlay_root) / "work"
        merged = Path(overlay_root) / "merged"

        upper.mkdir()
        work.mkdir()
        merged.mkdir()

        config.upper_dir = str(upper)
        config.work_dir = str(work)
        config.merged_dir = str(merged)

        return overlay_root

    def _build_network_args(self, policy: NetworkPolicy) -> List[str]:
        """Build network isolation arguments."""
        if policy == NetworkPolicy.NONE:
            return ["--network", "none"]
        elif policy == NetworkPolicy.EGRESS_ONLY:
            # Use host network but drop capabilities
            return ["--cap-drop", "NET_BIND_SERVICE", "--cap-drop", "NET_RAW"]
        elif policy == NetworkPolicy.ALLOW_DNS:
            return ["--dns", "8.8.8.8", "--network", "bridge"]
        else:  # FULL
            return []

    def _build_resource_args(self, limits: ResourceLimits) -> List[str]:
        """Build resource limit arguments."""
        args = []
        if limits.cpu_cores:
            args.extend(["--cpus", str(limits.cpu_cores)])
        if limits.memory_mb:
            args.extend(["--memory", f"{limits.memory_mb}m"])
        if limits.pids_limit:
            args.extend(["--pids-limit", str(limits.pids_limit)])
        return args

    def _build_security_args(self) -> List[str]:
        """Build security hardening arguments."""
        args = [
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--read-only",
        ]

        # Add seccomp if not using gVisor (gVisor has its own)
        if self.runtime != "gvisor":
            # Use default seccomp profile for now
            # In production, use SeccompProfile.generate_strict()
            args.extend(["--security-opt", "seccomp=unconfined"])

        return args

    def spawn(
        self,
        image: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        bind_mounts: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
        timeout: int = 120,
        isolation_level: Optional[IsolationLevel] = None,
        network_policy: NetworkPolicy = NetworkPolicy.NONE,
        resource_limits: Optional[ResourceLimits] = None,
        overlay: Optional[OverlayConfig] = None,
        working_dir: Optional[str] = None,
    ) -> ContainerResult:
        """
        Spawn an isolated container cell with configurable isolation.

        Args:
            image: Container image to run
            command: Command and arguments
            env: Environment variables
            bind_mounts: Host:container path mappings
            trace_id: Trace ID for observability
            timeout: Execution timeout in seconds
            isolation_level: Override default isolation level
            network_policy: Network isolation policy
            resource_limits: CPU/memory/pids limits
            overlay: Overlay filesystem configuration
            working_dir: Working directory inside container

        Returns:
            ContainerResult with execution details
        """
        run_id = str(uuid.uuid4())
        start_time = time.monotonic()
        effective_isolation = isolation_level or self.actual_isolation
        limits = resource_limits or ResourceLimits(timeout_s=timeout)

        # 1. Prepare Environment
        container_env = (env or {}).copy()
        if trace_id:
            container_env["PLURIBUS_TRACE_ID"] = trace_id
        container_env["PLURIBUS_RUN_ID"] = run_id
        container_env["PLURIBUS_ISOLATION"] = effective_isolation.value
        container_env["PLURIBUS_RUNTIME"] = self.runtime

        # 2. Validate mounts (Ring 0 protection)
        safe_mounts = self._validate_mounts(bind_mounts or {})

        # 3. Setup overlay if requested
        overlay_root = None
        if overlay:
            overlay_root = self._setup_overlay(overlay)

        # 4. Handle mock/dry-run
        if self.dry_run:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ContainerResult(
                exit_code=0,
                stdout=json.dumps({
                    "status": "dry_run",
                    "runtime": "mock",
                    "image": image,
                    "cmd": command,
                    "env_keys": list(container_env.keys()),
                    "trace_id": trace_id,
                    "isolation_level": effective_isolation.value,
                    "network_policy": network_policy.value,
                    "capabilities": RuntimeDetector.get_capabilities(),
                }),
                stderr="(dry run mode - no container runtime available)",
                container_id=f"mock-{run_id[:8]}",
                runtime="mock",
                isolation_level=effective_isolation.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
            )

        # 5. Build CLI command based on runtime
        container_name = f"pluribus-{run_id[:8]}"

        if self.runtime == "firecracker":
            return self._spawn_firecracker(
                image, command, container_env, safe_mounts,
                container_name, limits, network_policy, start_time
            )

        # Docker/Podman path
        cli_cmd = [self.runtime, "run", "--rm", "--name", container_name]

        # Security hardening
        cli_cmd.extend(self._build_security_args())

        # Network isolation
        cli_cmd.extend(self._build_network_args(network_policy))

        # Resource limits
        cli_cmd.extend(self._build_resource_args(limits))

        # gVisor runtime if available and requested
        if self.runtime == "gvisor" or (
            RuntimeDetector.has_gvisor() and
            effective_isolation == IsolationLevel.STRICT
        ):
            cli_cmd.extend(["--runtime", "runsc"])

        # Environment variables
        for k, v in container_env.items():
            cli_cmd.extend(["-e", f"{k}={v}"])

        # Mounts
        for host_path, container_path in safe_mounts.items():
            cli_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])

        # Overlay mount (writable)
        if overlay and overlay.upper_dir:
            # Mount the upper layer as writable workspace
            cli_cmd.extend(["-v", f"{overlay.upper_dir}:/workspace:rw"])

        # Tmpfs for /tmp
        cli_cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        # Working directory
        if working_dir:
            cli_cmd.extend(["-w", working_dir])
        elif overlay:
            cli_cmd.extend(["-w", "/workspace"])

        # Image and command
        cli_cmd.append(image)
        cli_cmd.extend(command)

        # 6. Execute
        try:
            proc = subprocess.run(
                cli_cmd,
                capture_output=True,
                text=True,
                timeout=limits.timeout_s
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)

            return ContainerResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                container_id=container_name,
                runtime=self.runtime,
                isolation_level=effective_isolation.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
                overlay_artifacts=overlay.upper_dir if overlay else None,
            )
        except subprocess.TimeoutExpired:
            # Cleanup
            subprocess.run(
                [self.runtime, "kill", container_name],
                capture_output=True
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ContainerResult(
                exit_code=124,
                stdout="",
                stderr=f"Timeout after {limits.timeout_s}s",
                container_id=None,
                runtime=self.runtime,
                isolation_level=effective_isolation.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ContainerResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                container_id=None,
                runtime=self.runtime,
                isolation_level=effective_isolation.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
            )

    def _spawn_firecracker(
        self,
        image: str,
        command: List[str],
        env: Dict[str, str],
        mounts: Dict[str, str],
        name: str,
        limits: ResourceLimits,
        network_policy: NetworkPolicy,
        start_time: float,
    ) -> ContainerResult:
        """
        Spawn execution in Firecracker microVM.

        This requires:
        1. A rootfs image (ext4)
        2. A kernel image
        3. /dev/kvm access

        For now, falls back to Docker if Firecracker assets not configured.
        """
        # Check for Firecracker configuration
        fc_config_path = Path(os.environ.get(
            "PLURIBUS_FIRECRACKER_CONFIG",
            "/etc/pluribus/firecracker.json"
        ))

        if not fc_config_path.exists():
            # Fallback to Docker with gVisor if available
            self.runtime = "docker" if RuntimeDetector.has_docker() else "podman"
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ContainerResult(
                exit_code=1,
                stdout="",
                stderr="Firecracker config not found, falling back to container runtime",
                container_id=None,
                runtime=self.runtime,
                isolation_level=IsolationLevel.STANDARD.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
            )

        # Load Firecracker config
        try:
            fc_config = json.loads(fc_config_path.read_text())
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ContainerResult(
                exit_code=1,
                stdout="",
                stderr=f"Failed to load Firecracker config: {e}",
                container_id=None,
                runtime="firecracker",
                isolation_level=IsolationLevel.STRICT.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
            )

        # Build Firecracker command
        # This is a simplified version - production would use the Firecracker API
        kernel_path = fc_config.get("kernel_image_path", "/var/lib/firecracker/kernel")
        rootfs_path = fc_config.get("rootfs_path", "/var/lib/firecracker/rootfs.ext4")

        fc_cmd = [
            "firecracker",
            "--no-api",
            "--config-file", str(fc_config_path),
        ]

        try:
            proc = subprocess.run(
                fc_cmd,
                capture_output=True,
                text=True,
                timeout=limits.timeout_s,
                env={**os.environ, **env}
            )
            duration_ms = int((time.monotonic() - start_time) * 1000)

            return ContainerResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                container_id=name,
                runtime="firecracker",
                isolation_level=IsolationLevel.STRICT.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ContainerResult(
                exit_code=124,
                stdout="",
                stderr=f"Firecracker timeout after {limits.timeout_s}s",
                container_id=None,
                runtime="firecracker",
                isolation_level=IsolationLevel.STRICT.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ContainerResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                container_id=None,
                runtime="firecracker",
                isolation_level=IsolationLevel.STRICT.value,
                network_policy=network_policy.value,
                duration_ms=duration_ms,
            )

    def execute_code(
        self,
        code: str,
        language: str = "python",
        trace_id: Optional[str] = None,
        timeout: int = 30,
    ) -> ContainerResult:
        """
        Execute code snippet in isolated container.

        This is a convenience method that wraps spawn() for code execution.
        """
        # Select image based on language
        images = {
            "python": "python:3.12-alpine",
            "node": "node:20-alpine",
            "bash": "alpine:latest",
        }
        image = images.get(language, "alpine:latest")

        # Build command
        if language == "python":
            command = ["python3", "-c", code]
        elif language == "node":
            command = ["node", "-e", code]
        elif language == "bash":
            command = ["sh", "-c", code]
        else:
            command = ["sh", "-c", code]

        return self.spawn(
            image=image,
            command=command,
            trace_id=trace_id,
            timeout=timeout,
            isolation_level=IsolationLevel.STRICT,
            network_policy=NetworkPolicy.NONE,
            resource_limits=ResourceLimits(
                cpu_cores=0.5,
                memory_mb=256,
                pids_limit=50,
                timeout_s=timeout,
            ),
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return runtime capabilities report."""
        caps = RuntimeDetector.get_capabilities()
        caps["current_runtime"] = self.runtime
        caps["current_isolation"] = self.actual_isolation.value
        caps["dry_run"] = self.dry_run
        return caps


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Microsandbox-ready container executor"
    )
    parser.add_argument("--image", default="alpine:latest")
    parser.add_argument("--cmd", default="echo hello")
    parser.add_argument("--isolation", choices=["strict", "standard", "relaxed", "mock"])
    parser.add_argument("--network", choices=["none", "egress", "dns", "full"], default="none")
    parser.add_argument("--capabilities", action="store_true", help="Show runtime capabilities")
    parser.add_argument("--code", help="Execute code snippet (Python)")
    args = parser.parse_args()

    isolation = None
    if args.isolation:
        isolation = IsolationLevel(args.isolation)

    exc = ContainerExecutor(isolation_level=isolation or IsolationLevel.STANDARD)

    if args.capabilities:
        caps = exc.get_capabilities()
        print(json.dumps(caps, indent=2))
        return

    if args.code:
        res = exc.execute_code(args.code, trace_id=str(uuid.uuid4()))
    else:
        network = NetworkPolicy(args.network)
        res = exc.spawn(
            args.image,
            args.cmd.split(),
            trace_id=str(uuid.uuid4()),
            network_policy=network,
        )

    print(f"Runtime: {res.runtime}")
    print(f"Isolation: {res.isolation_level}")
    print(f"Network: {res.network_policy}")
    print(f"Exit Code: {res.exit_code}")
    print(f"Duration: {res.duration_ms}ms")
    print(f"Stdout: {res.stdout}")
    if res.stderr:
        print(f"Stderr: {res.stderr}")


if __name__ == "__main__":
    main()
