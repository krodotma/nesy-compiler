#!/usr/bin/env python3
"""
orchestrator.py - Build Orchestrator (Step 202)

PBTSO Phase: ITERATE
A2A Integration: Orchestrates builds via deploy.build.start

Provides:
- BuildStatus: Enum for build states
- BuildResult: Result of a build operation
- BuildConfig: Configuration for a build
- BuildOrchestrator: Orchestrates build processes

Bus Topics:
- deploy.build.start
- deploy.build.complete
- deploy.build.failed
- deploy.build.progress

Protocol: DKIN v30
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_BUILD_TIMEOUT_S = 600  # 10 minutes
DEFAULT_OUTPUT_DIR = "dist"


# ==============================================================================
# Bus Emission Helper
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "build-orchestrator"
) -> str:
    """Emit an event to the Pluribus bus."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class BuildStatus(Enum):
    """Build status states."""
    PENDING = "pending"
    BUILDING = "building"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class BuildType(Enum):
    """Types of builds supported."""
    NPM = "npm"
    YARN = "yarn"
    PYTHON = "python"
    DOCKER = "docker"
    MAKE = "make"
    CUSTOM = "custom"


@dataclass
class BuildConfig:
    """
    Configuration for a build.

    Attributes:
        build_type: Type of build (npm, yarn, python, docker, make, custom)
        command: Build command to execute (overrides build_type default)
        output_dir: Directory where build artifacts are produced
        timeout_s: Build timeout in seconds
        env: Additional environment variables
        pre_build: Commands to run before build
        post_build: Commands to run after build
        cache_enabled: Whether to use build cache
        cache_key: Key for caching (e.g., hash of dependencies)
    """
    build_type: Union[BuildType, str] = BuildType.CUSTOM
    command: str = ""
    output_dir: str = DEFAULT_OUTPUT_DIR
    timeout_s: int = DEFAULT_BUILD_TIMEOUT_S
    env: Dict[str, str] = field(default_factory=dict)
    pre_build: List[str] = field(default_factory=list)
    post_build: List[str] = field(default_factory=list)
    cache_enabled: bool = True
    cache_key: str = ""

    def __post_init__(self):
        if isinstance(self.build_type, str):
            try:
                self.build_type = BuildType(self.build_type)
            except ValueError:
                self.build_type = BuildType.CUSTOM

        # Set default commands based on build type
        if not self.command:
            defaults = {
                BuildType.NPM: "npm run build",
                BuildType.YARN: "yarn build",
                BuildType.PYTHON: "python -m build",
                BuildType.DOCKER: "docker build -t app .",
                BuildType.MAKE: "make",
                BuildType.CUSTOM: "echo 'No build command specified'",
            }
            self.command = defaults.get(self.build_type, "")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "build_type": self.build_type.value if isinstance(self.build_type, BuildType) else self.build_type,
            "command": self.command,
            "output_dir": self.output_dir,
            "timeout_s": self.timeout_s,
            "env": self.env,
            "pre_build": self.pre_build,
            "post_build": self.post_build,
            "cache_enabled": self.cache_enabled,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BuildResult:
    """
    Result of a build operation.

    Attributes:
        build_id: Unique identifier for this build
        status: Build status
        artifacts: List of artifact paths produced
        duration_ms: Build duration in milliseconds
        logs: Build output logs
        exit_code: Process exit code
        error: Error message if failed
        metadata: Additional metadata
    """
    build_id: str
    status: BuildStatus
    artifacts: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    logs: str = ""
    exit_code: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "build_id": self.build_id,
            "status": self.status.value,
            "artifacts": self.artifacts,
            "duration_ms": self.duration_ms,
            "logs": self.logs,
            "exit_code": self.exit_code,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildResult":
        """Create from dictionary."""
        data = dict(data)
        if "status" in data:
            data["status"] = BuildStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Build Orchestrator (Step 202)
# ==============================================================================

class BuildOrchestrator:
    """
    Build Orchestrator - orchestrates build processes.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Execute build commands
    - Collect artifacts
    - Track build state
    - Emit build events to A2A bus

    Example:
        >>> orchestrator = BuildOrchestrator("/path/to/project")
        >>> config = BuildConfig(build_type=BuildType.NPM)
        >>> result = await orchestrator.build(config)
        >>> print(f"Build {result.status.value}: {result.artifacts}")
    """

    BUS_TOPICS = {
        "start": "deploy.build.start",
        "complete": "deploy.build.complete",
        "failed": "deploy.build.failed",
        "progress": "deploy.build.progress",
    }

    def __init__(self, working_dir: str, actor_id: str = "build-orchestrator"):
        """
        Initialize the build orchestrator.

        Args:
            working_dir: Directory containing the project to build
            actor_id: Actor identifier for bus events
        """
        self.working_dir = Path(working_dir).resolve()
        self.actor_id = actor_id
        self._active_builds: Dict[str, BuildResult] = {}

    async def build(self, config: Union[BuildConfig, Dict[str, Any]]) -> BuildResult:
        """
        Execute a build.

        Args:
            config: Build configuration

        Returns:
            BuildResult with status and artifacts
        """
        if isinstance(config, dict):
            config = BuildConfig.from_dict(config)

        build_id = f"build-{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # Initialize result
        result = BuildResult(
            build_id=build_id,
            status=BuildStatus.PENDING,
            metadata={
                "config": config.to_dict(),
                "working_dir": str(self.working_dir),
                "started_at": start_time,
            },
        )
        self._active_builds[build_id] = result

        # Emit start event
        _emit_bus_event(
            self.BUS_TOPICS["start"],
            {
                "build_id": build_id,
                "config": config.to_dict(),
                "working_dir": str(self.working_dir),
            },
            actor=self.actor_id,
        )

        try:
            result.status = BuildStatus.BUILDING

            # Run pre-build commands
            for pre_cmd in config.pre_build:
                pre_result = await self._run_command(pre_cmd, config.env, config.timeout_s)
                if pre_result.returncode != 0:
                    raise RuntimeError(f"Pre-build command failed: {pre_cmd}")

            # Run main build command
            proc_result = await self._run_command(
                config.command,
                config.env,
                config.timeout_s,
            )

            result.exit_code = proc_result.returncode
            result.logs = proc_result.stdout + proc_result.stderr

            if proc_result.returncode != 0:
                result.status = BuildStatus.FAILED
                result.error = f"Build command exited with code {proc_result.returncode}"
            else:
                # Run post-build commands
                for post_cmd in config.post_build:
                    post_result = await self._run_command(post_cmd, config.env, config.timeout_s)
                    if post_result.returncode != 0:
                        raise RuntimeError(f"Post-build command failed: {post_cmd}")

                # Collect artifacts
                result.artifacts = self._collect_artifacts(config.output_dir)
                result.status = BuildStatus.SUCCESS

        except asyncio.TimeoutError:
            result.status = BuildStatus.TIMEOUT
            result.error = f"Build timed out after {config.timeout_s}s"

        except Exception as e:
            result.status = BuildStatus.FAILED
            result.error = str(e)

        finally:
            # Calculate duration
            result.duration_ms = (time.time() - start_time) * 1000
            result.metadata["completed_at"] = time.time()

            # Emit completion event
            topic = self.BUS_TOPICS["complete"] if result.status == BuildStatus.SUCCESS else self.BUS_TOPICS["failed"]
            _emit_bus_event(
                topic,
                {
                    "build_id": build_id,
                    "status": result.status.value,
                    "artifacts": result.artifacts,
                    "duration_ms": result.duration_ms,
                    "exit_code": result.exit_code,
                    "error": result.error,
                },
                level="info" if result.status == BuildStatus.SUCCESS else "error",
                actor=self.actor_id,
            )

        return result

    def build_sync(self, config: Union[BuildConfig, Dict[str, Any]]) -> BuildResult:
        """Synchronous wrapper for build()."""
        return asyncio.get_event_loop().run_until_complete(self.build(config))

    async def _run_command(
        self,
        command: str,
        env: Dict[str, str],
        timeout_s: int,
    ) -> subprocess.CompletedProcess:
        """
        Run a shell command asynchronously.

        Args:
            command: Command to execute
            env: Additional environment variables
            timeout_s: Timeout in seconds

        Returns:
            CompletedProcess with output
        """
        # Merge environment
        full_env = os.environ.copy()
        full_env.update(env)

        # Run command
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.working_dir),
            env=full_env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        return subprocess.CompletedProcess(
            args=command,
            returncode=proc.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    def _collect_artifacts(self, output_dir: str) -> List[str]:
        """
        Collect artifact paths from the output directory.

        Args:
            output_dir: Directory containing build artifacts

        Returns:
            List of artifact paths relative to working directory
        """
        artifacts = []
        output_path = self.working_dir / output_dir

        if not output_path.exists():
            return artifacts

        for item in output_path.rglob("*"):
            if item.is_file():
                artifacts.append(str(item.relative_to(self.working_dir)))

        return artifacts

    def get_build(self, build_id: str) -> Optional[BuildResult]:
        """Get a build result by ID."""
        return self._active_builds.get(build_id)

    def list_builds(self) -> List[BuildResult]:
        """List all tracked builds."""
        return list(self._active_builds.values())

    def cancel_build(self, build_id: str) -> bool:
        """
        Cancel an active build.

        Args:
            build_id: Build ID to cancel

        Returns:
            True if cancelled
        """
        if build_id not in self._active_builds:
            return False

        result = self._active_builds[build_id]
        if result.status == BuildStatus.BUILDING:
            result.status = BuildStatus.CANCELLED
            result.error = "Build cancelled by user"
            return True

        return False


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for build orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Orchestrator (Step 202)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build command
    build_parser = subparsers.add_parser("build", help="Run a build")
    build_parser.add_argument("--dir", "-d", default=".", help="Working directory")
    build_parser.add_argument("--type", "-t", default="custom", choices=["npm", "yarn", "python", "docker", "make", "custom"])
    build_parser.add_argument("--command", "-c", default="", help="Build command")
    build_parser.add_argument("--output", "-o", default="dist", help="Output directory")
    build_parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    build_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List builds")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if args.command == "build":
        config = BuildConfig(
            build_type=args.type,
            command=args.command,
            output_dir=args.output,
            timeout_s=args.timeout,
        )

        orchestrator = BuildOrchestrator(args.dir)
        result = orchestrator.build_sync(config)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status_icon = "OK" if result.status == BuildStatus.SUCCESS else "FAIL"
            print(f"[{status_icon}] Build {result.build_id}")
            print(f"  Status: {result.status.value}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
            print(f"  Artifacts: {len(result.artifacts)}")
            if result.error:
                print(f"  Error: {result.error}")

        return 0 if result.status == BuildStatus.SUCCESS else 1

    elif args.command == "list":
        # Just show help - builds are ephemeral in this implementation
        print("No persisted builds in this session")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
