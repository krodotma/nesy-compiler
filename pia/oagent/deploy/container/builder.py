#!/usr/bin/env python3
"""
builder.py - Container Builder (Step 204)

PBTSO Phase: ITERATE
A2A Integration: Builds containers via deploy.container.build

Provides:
- ContainerImage: Metadata about a container image
- ContainerBuildResult: Result of a container build
- ContainerBuilder: Builds container images

Bus Topics:
- deploy.container.build
- deploy.container.pushed
- deploy.container.failed

Protocol: DKIN v30
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    actor: str = "container-builder"
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

class ContainerRuntime(Enum):
    """Supported container runtimes."""
    DOCKER = "docker"
    PODMAN = "podman"
    BUILDAH = "buildah"


class BuildStatus(Enum):
    """Container build status."""
    PENDING = "pending"
    BUILDING = "building"
    PUSHING = "pushing"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class ContainerImage:
    """
    Metadata about a container image.

    Attributes:
        image_id: Unique image identifier (short hash)
        full_id: Full image identifier (SHA256)
        name: Image name
        tag: Image tag
        registry: Registry URL
        size_bytes: Image size in bytes
        created_at: Timestamp when created
        labels: Image labels
        layers: List of layer digests
    """
    image_id: str
    full_id: str = ""
    name: str = ""
    tag: str = "latest"
    registry: str = ""
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    layers: List[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        """Get the full image name with registry and tag."""
        parts = []
        if self.registry:
            parts.append(self.registry)
        parts.append(self.name)
        return "/".join(parts) + f":{self.tag}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "full_id": self.full_id,
            "name": self.name,
            "tag": self.tag,
            "registry": self.registry,
            "full_name": self.full_name,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "labels": self.labels,
            "layers": self.layers,
        }


@dataclass
class ContainerBuildResult:
    """
    Result of a container build operation.

    Attributes:
        build_id: Unique build identifier
        status: Build status
        image: Built image metadata (if successful)
        duration_ms: Build duration in milliseconds
        logs: Build output logs
        error: Error message if failed
        pushed: Whether image was pushed to registry
        push_digest: Push digest if pushed
    """
    build_id: str
    status: BuildStatus
    image: Optional[ContainerImage] = None
    duration_ms: float = 0.0
    logs: str = ""
    error: Optional[str] = None
    pushed: bool = False
    push_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "build_id": self.build_id,
            "status": self.status.value,
            "image": self.image.to_dict() if self.image else None,
            "duration_ms": self.duration_ms,
            "logs": self.logs,
            "error": self.error,
            "pushed": self.pushed,
            "push_digest": self.push_digest,
        }


# ==============================================================================
# Container Builder (Step 204)
# ==============================================================================

class ContainerBuilder:
    """
    Container Builder - builds container images.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Build container images from Dockerfile
    - Push images to registry
    - Track build state
    - Emit build events to A2A bus

    Example:
        >>> builder = ContainerBuilder()
        >>> result = await builder.build(
        ...     context_dir="/app",
        ...     image_name="myapp",
        ...     tag="v1.0.0"
        ... )
        >>> if result.status == BuildStatus.SUCCESS:
        ...     await builder.push(result.image)
    """

    BUS_TOPICS = {
        "build": "deploy.container.build",
        "pushed": "deploy.container.pushed",
        "failed": "deploy.container.failed",
    }

    def __init__(
        self,
        runtime: ContainerRuntime = ContainerRuntime.DOCKER,
        registry: str = "",
        actor_id: str = "container-builder",
    ):
        """
        Initialize the container builder.

        Args:
            runtime: Container runtime to use (docker, podman, buildah)
            registry: Default registry URL
            actor_id: Actor identifier for bus events
        """
        self.runtime = runtime
        self.registry = registry
        self.actor_id = actor_id
        self._active_builds: Dict[str, ContainerBuildResult] = {}

    async def build(
        self,
        context_dir: str,
        image_name: str,
        tag: str = "latest",
        dockerfile: str = "Dockerfile",
        build_args: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        target: Optional[str] = None,
        no_cache: bool = False,
        timeout_s: int = 900,
    ) -> ContainerBuildResult:
        """
        Build a container image.

        Args:
            context_dir: Build context directory
            image_name: Image name
            tag: Image tag
            dockerfile: Dockerfile path relative to context
            build_args: Build arguments
            labels: Image labels
            target: Multi-stage build target
            no_cache: Disable build cache
            timeout_s: Build timeout in seconds

        Returns:
            ContainerBuildResult with status and image
        """
        build_id = f"cbuild-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        context_path = Path(context_dir).resolve()

        # Initialize result
        result = ContainerBuildResult(
            build_id=build_id,
            status=BuildStatus.PENDING,
        )
        self._active_builds[build_id] = result

        # Full image name with registry
        full_name = image_name
        if self.registry:
            full_name = f"{self.registry}/{image_name}"

        # Emit build event
        _emit_bus_event(
            self.BUS_TOPICS["build"],
            {
                "build_id": build_id,
                "image_name": image_name,
                "tag": tag,
                "context_dir": str(context_path),
                "dockerfile": dockerfile,
                "registry": self.registry,
            },
            actor=self.actor_id,
        )

        try:
            result.status = BuildStatus.BUILDING

            # Build command
            cmd = self._build_command(
                full_name=full_name,
                tag=tag,
                dockerfile=dockerfile,
                build_args=build_args or {},
                labels=labels or {},
                target=target,
                no_cache=no_cache,
            )

            # Execute build
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(context_path),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"Build timed out after {timeout_s}s")

            result.logs = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                result.status = BuildStatus.FAILED
                result.error = f"Build failed with exit code {proc.returncode}"
            else:
                # Get image info
                image = await self._get_image_info(full_name, tag)
                image.labels = labels or {}
                result.image = image
                result.status = BuildStatus.SUCCESS

        except Exception as e:
            result.status = BuildStatus.FAILED
            result.error = str(e)

        finally:
            result.duration_ms = (time.time() - start_time) * 1000

            if result.status == BuildStatus.FAILED:
                _emit_bus_event(
                    self.BUS_TOPICS["failed"],
                    {
                        "build_id": build_id,
                        "image_name": image_name,
                        "tag": tag,
                        "error": result.error,
                    },
                    level="error",
                    actor=self.actor_id,
                )

        return result

    def build_sync(
        self,
        context_dir: str,
        image_name: str,
        tag: str = "latest",
        **kwargs,
    ) -> ContainerBuildResult:
        """Synchronous wrapper for build()."""
        return asyncio.get_event_loop().run_until_complete(
            self.build(context_dir, image_name, tag, **kwargs)
        )

    async def push(
        self,
        image: ContainerImage,
        registry: Optional[str] = None,
        timeout_s: int = 600,
    ) -> ContainerBuildResult:
        """
        Push an image to a registry.

        Args:
            image: Image to push
            registry: Override registry URL
            timeout_s: Push timeout in seconds

        Returns:
            Updated ContainerBuildResult
        """
        build_id = f"push-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        registry = registry or self.registry or image.registry

        result = ContainerBuildResult(
            build_id=build_id,
            status=BuildStatus.PUSHING,
            image=image,
        )

        try:
            # Tag for registry if needed
            if registry and registry != image.registry:
                new_name = f"{registry}/{image.name}:{image.tag}"
                tag_cmd = f"{self.runtime.value} tag {image.full_name} {new_name}"
                proc = await asyncio.create_subprocess_shell(
                    tag_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.wait()
                image.registry = registry

            # Push
            push_cmd = f"{self.runtime.value} push {image.full_name}"
            proc = await asyncio.create_subprocess_shell(
                push_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"Push timed out after {timeout_s}s")

            result.logs = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                result.status = BuildStatus.FAILED
                result.error = f"Push failed with exit code {proc.returncode}"
            else:
                result.status = BuildStatus.SUCCESS
                result.pushed = True
                # Extract digest from output
                digest_match = re.search(r"(sha256:[a-f0-9]{64})", result.logs)
                if digest_match:
                    result.push_digest = digest_match.group(1)

                # Emit pushed event
                _emit_bus_event(
                    self.BUS_TOPICS["pushed"],
                    {
                        "build_id": build_id,
                        "image_name": image.name,
                        "tag": image.tag,
                        "registry": registry,
                        "full_name": image.full_name,
                        "digest": result.push_digest,
                    },
                    actor=self.actor_id,
                )

        except Exception as e:
            result.status = BuildStatus.FAILED
            result.error = str(e)

        finally:
            result.duration_ms = (time.time() - start_time) * 1000

        return result

    def _build_command(
        self,
        full_name: str,
        tag: str,
        dockerfile: str,
        build_args: Dict[str, str],
        labels: Dict[str, str],
        target: Optional[str],
        no_cache: bool,
    ) -> str:
        """Construct the build command."""
        cmd_parts = [self.runtime.value, "build"]

        # Image name and tag
        cmd_parts.extend(["-t", f"{full_name}:{tag}"])

        # Dockerfile
        cmd_parts.extend(["-f", dockerfile])

        # Build args
        for key, value in build_args.items():
            cmd_parts.extend(["--build-arg", f"{key}={value}"])

        # Labels
        for key, value in labels.items():
            cmd_parts.extend(["--label", f"{key}={value}"])

        # Target
        if target:
            cmd_parts.extend(["--target", target])

        # No cache
        if no_cache:
            cmd_parts.append("--no-cache")

        # Context (current directory)
        cmd_parts.append(".")

        return " ".join(cmd_parts)

    async def _get_image_info(self, name: str, tag: str) -> ContainerImage:
        """Get image information."""
        full_ref = f"{name}:{tag}"

        # Get image ID
        proc = await asyncio.create_subprocess_shell(
            f"{self.runtime.value} images -q {full_ref}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        image_id = stdout.decode().strip()[:12]

        # Get full ID
        proc = await asyncio.create_subprocess_shell(
            f"{self.runtime.value} inspect --format='{{{{.Id}}}}' {full_ref}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        full_id = stdout.decode().strip()

        # Get size
        proc = await asyncio.create_subprocess_shell(
            f"{self.runtime.value} inspect --format='{{{{.Size}}}}' {full_ref}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        try:
            size_bytes = int(stdout.decode().strip())
        except ValueError:
            size_bytes = 0

        return ContainerImage(
            image_id=image_id,
            full_id=full_id,
            name=name,
            tag=tag,
            registry=self.registry,
            size_bytes=size_bytes,
        )

    def get_build(self, build_id: str) -> Optional[ContainerBuildResult]:
        """Get a build result by ID."""
        return self._active_builds.get(build_id)

    def list_builds(self) -> List[ContainerBuildResult]:
        """List all tracked builds."""
        return list(self._active_builds.values())


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for container builder."""
    import argparse

    parser = argparse.ArgumentParser(description="Container Builder (Step 204)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build command
    build_parser = subparsers.add_parser("build", help="Build a container image")
    build_parser.add_argument("context", help="Build context directory")
    build_parser.add_argument("--name", "-n", required=True, help="Image name")
    build_parser.add_argument("--tag", "-t", default="latest", help="Image tag")
    build_parser.add_argument("--file", "-f", default="Dockerfile", help="Dockerfile path")
    build_parser.add_argument("--registry", "-r", default="", help="Registry URL")
    build_parser.add_argument("--runtime", default="docker", choices=["docker", "podman", "buildah"])
    build_parser.add_argument("--push", action="store_true", help="Push after build")
    build_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if args.command == "build":
        builder = ContainerBuilder(
            runtime=ContainerRuntime(args.runtime),
            registry=args.registry,
        )

        result = builder.build_sync(
            context_dir=args.context,
            image_name=args.name,
            tag=args.tag,
            dockerfile=args.file,
        )

        if result.status == BuildStatus.SUCCESS and args.push:
            push_result = asyncio.get_event_loop().run_until_complete(
                builder.push(result.image)
            )
            result.pushed = push_result.pushed
            result.push_digest = push_result.push_digest

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status_icon = "OK" if result.status == BuildStatus.SUCCESS else "FAIL"
            print(f"[{status_icon}] Build {result.build_id}")
            print(f"  Status: {result.status.value}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
            if result.image:
                print(f"  Image: {result.image.full_name}")
                print(f"  ID: {result.image.image_id}")
            if result.pushed:
                print(f"  Pushed: {result.push_digest[:24]}...")
            if result.error:
                print(f"  Error: {result.error}")

        return 0 if result.status == BuildStatus.SUCCESS else 1

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
