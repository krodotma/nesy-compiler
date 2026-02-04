#!/usr/bin/env python3
"""
packager.py - Artifact Packager (Step 203)

PBTSO Phase: DISTILL
A2A Integration: Packages artifacts via deploy.artifact.package

Provides:
- ArtifactManifest: Manifest describing packaged artifacts
- PackageFormat: Supported package formats
- ArtifactPackager: Packages build artifacts for deployment

Bus Topics:
- deploy.artifact.package
- deploy.artifact.ready
- deploy.artifact.failed

Protocol: DKIN v30
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import socket
import tarfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
    actor: str = "artifact-packager"
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

class PackageFormat(Enum):
    """Supported package formats."""
    TAR_GZ = "tar.gz"
    ZIP = "zip"
    TAR = "tar"
    RAW = "raw"  # No compression, directory copy


@dataclass
class ArtifactManifest:
    """
    Manifest describing packaged artifacts.

    Attributes:
        artifact_id: Unique identifier for this artifact package
        version: Version string
        build_id: Associated build ID
        format: Package format
        files: List of files included
        size_bytes: Total size in bytes
        checksum: SHA256 checksum
        created_at: Timestamp when created
        metadata: Additional metadata
    """
    artifact_id: str
    version: str
    build_id: str = ""
    format: PackageFormat = PackageFormat.TAR_GZ
    files: List[str] = field(default_factory=list)
    size_bytes: int = 0
    checksum: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "version": self.version,
            "build_id": self.build_id,
            "format": self.format.value if isinstance(self.format, PackageFormat) else self.format,
            "files": self.files,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactManifest":
        """Create from dictionary."""
        data = dict(data)
        if "format" in data:
            data["format"] = PackageFormat(data["format"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Artifact Packager (Step 203)
# ==============================================================================

class ArtifactPackager:
    """
    Artifact Packager - packages build artifacts for deployment.

    PBTSO Phase: DISTILL

    Responsibilities:
    - Package artifacts into distributable formats
    - Generate manifest with checksums
    - Store artifacts in registry
    - Emit packaging events to A2A bus

    Example:
        >>> packager = ArtifactPackager("/artifacts")
        >>> manifest = packager.package(
        ...     source_dir="/build/dist",
        ...     version="1.0.0",
        ...     format=PackageFormat.TAR_GZ
        ... )
        >>> print(f"Artifact: {manifest.artifact_id}, Size: {manifest.size_bytes}")
    """

    BUS_TOPICS = {
        "package": "deploy.artifact.package",
        "ready": "deploy.artifact.ready",
        "failed": "deploy.artifact.failed",
    }

    def __init__(self, artifacts_dir: str, actor_id: str = "artifact-packager"):
        """
        Initialize the artifact packager.

        Args:
            artifacts_dir: Directory to store packaged artifacts
            actor_id: Actor identifier for bus events
        """
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.actor_id = actor_id
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def package(
        self,
        source_dir: str,
        version: str,
        build_id: str = "",
        format: Union[PackageFormat, str] = PackageFormat.TAR_GZ,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactManifest:
        """
        Package artifacts from a source directory.

        Args:
            source_dir: Directory containing artifacts to package
            version: Version string for the artifact
            build_id: Associated build ID
            format: Package format (tar.gz, zip, tar, raw)
            include_patterns: Glob patterns to include (default: all)
            exclude_patterns: Glob patterns to exclude
            metadata: Additional metadata to include in manifest

        Returns:
            ArtifactManifest describing the packaged artifact
        """
        if isinstance(format, str):
            format = PackageFormat(format)

        artifact_id = f"artifact-{uuid.uuid4().hex[:12]}"
        source_path = Path(source_dir).resolve()

        # Emit package start event
        _emit_bus_event(
            self.BUS_TOPICS["package"],
            {
                "artifact_id": artifact_id,
                "version": version,
                "build_id": build_id,
                "source_dir": str(source_path),
                "format": format.value,
            },
            actor=self.actor_id,
        )

        try:
            # Collect files to package
            files = self._collect_files(source_path, include_patterns, exclude_patterns)

            # Create package
            if format == PackageFormat.TAR_GZ:
                package_path, size_bytes = self._create_tar_gz(artifact_id, version, source_path, files)
            elif format == PackageFormat.ZIP:
                package_path, size_bytes = self._create_zip(artifact_id, version, source_path, files)
            elif format == PackageFormat.TAR:
                package_path, size_bytes = self._create_tar(artifact_id, version, source_path, files)
            else:  # RAW
                package_path, size_bytes = self._create_raw(artifact_id, version, source_path, files)

            # Compute checksum
            checksum = self._compute_checksum(package_path)

            # Create manifest
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                version=version,
                build_id=build_id,
                format=format,
                files=[str(f) for f in files],
                size_bytes=size_bytes,
                checksum=checksum,
                metadata=metadata or {},
            )

            # Save manifest
            self._save_manifest(manifest, package_path)

            # Emit ready event
            _emit_bus_event(
                self.BUS_TOPICS["ready"],
                {
                    "artifact_id": artifact_id,
                    "version": version,
                    "format": format.value,
                    "size_bytes": size_bytes,
                    "checksum": checksum,
                    "file_count": len(files),
                    "path": str(package_path),
                },
                actor=self.actor_id,
            )

            return manifest

        except Exception as e:
            # Emit failed event
            _emit_bus_event(
                self.BUS_TOPICS["failed"],
                {
                    "artifact_id": artifact_id,
                    "version": version,
                    "error": str(e),
                },
                level="error",
                actor=self.actor_id,
            )
            raise

    def _collect_files(
        self,
        source_path: Path,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
    ) -> List[Path]:
        """Collect files matching the patterns."""
        files = []
        exclude_patterns = exclude_patterns or []

        if include_patterns:
            for pattern in include_patterns:
                files.extend(source_path.glob(pattern))
        else:
            files.extend(source_path.rglob("*"))

        # Filter files
        result = []
        for f in files:
            if not f.is_file():
                continue

            rel_path = str(f.relative_to(source_path))

            # Check exclusions
            excluded = False
            for pattern in exclude_patterns:
                if f.match(pattern):
                    excluded = True
                    break

            if not excluded:
                result.append(f.relative_to(source_path))

        return sorted(set(result))

    def _create_tar_gz(
        self,
        artifact_id: str,
        version: str,
        source_path: Path,
        files: List[Path],
    ) -> tuple[Path, int]:
        """Create a tar.gz archive."""
        filename = f"{artifact_id}-{version}.tar.gz"
        package_path = self.artifacts_dir / filename

        with tarfile.open(package_path, "w:gz") as tar:
            for f in files:
                tar.add(source_path / f, arcname=str(f))

        return package_path, package_path.stat().st_size

    def _create_zip(
        self,
        artifact_id: str,
        version: str,
        source_path: Path,
        files: List[Path],
    ) -> tuple[Path, int]:
        """Create a zip archive."""
        filename = f"{artifact_id}-{version}.zip"
        package_path = self.artifacts_dir / filename

        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(source_path / f, arcname=str(f))

        return package_path, package_path.stat().st_size

    def _create_tar(
        self,
        artifact_id: str,
        version: str,
        source_path: Path,
        files: List[Path],
    ) -> tuple[Path, int]:
        """Create an uncompressed tar archive."""
        filename = f"{artifact_id}-{version}.tar"
        package_path = self.artifacts_dir / filename

        with tarfile.open(package_path, "w") as tar:
            for f in files:
                tar.add(source_path / f, arcname=str(f))

        return package_path, package_path.stat().st_size

    def _create_raw(
        self,
        artifact_id: str,
        version: str,
        source_path: Path,
        files: List[Path],
    ) -> tuple[Path, int]:
        """Create a raw directory copy."""
        dirname = f"{artifact_id}-{version}"
        package_path = self.artifacts_dir / dirname
        package_path.mkdir(parents=True, exist_ok=True)

        total_size = 0
        for f in files:
            src = source_path / f
            dst = package_path / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            total_size += dst.stat().st_size

        return package_path, total_size

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file or directory."""
        sha256 = hashlib.sha256()

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
        else:
            # For directories, hash all files
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    with open(f, "rb") as fh:
                        for chunk in iter(lambda: fh.read(8192), b""):
                            sha256.update(chunk)

        return sha256.hexdigest()

    def _save_manifest(self, manifest: ArtifactManifest, package_path: Path) -> None:
        """Save manifest alongside the package."""
        if package_path.is_file():
            manifest_path = package_path.with_suffix(package_path.suffix + ".manifest.json")
        else:
            manifest_path = package_path / "manifest.json"

        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactManifest]:
        """Get an artifact manifest by ID."""
        for manifest_file in self.artifacts_dir.glob("*.manifest.json"):
            try:
                with open(manifest_file, "r") as f:
                    data = json.load(f)
                    if data.get("artifact_id") == artifact_id:
                        return ArtifactManifest.from_dict(data)
            except (json.JSONDecodeError, IOError):
                continue

        # Also check raw directories
        for dir_path in self.artifacts_dir.iterdir():
            if dir_path.is_dir():
                manifest_file = dir_path / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r") as f:
                            data = json.load(f)
                            if data.get("artifact_id") == artifact_id:
                                return ArtifactManifest.from_dict(data)
                    except (json.JSONDecodeError, IOError):
                        continue

        return None

    def list_artifacts(self) -> List[ArtifactManifest]:
        """List all artifacts."""
        artifacts = []

        # Check manifest files
        for manifest_file in self.artifacts_dir.glob("*.manifest.json"):
            try:
                with open(manifest_file, "r") as f:
                    data = json.load(f)
                    artifacts.append(ArtifactManifest.from_dict(data))
            except (json.JSONDecodeError, IOError):
                continue

        # Check raw directories
        for dir_path in self.artifacts_dir.iterdir():
            if dir_path.is_dir():
                manifest_file = dir_path / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r") as f:
                            data = json.load(f)
                            artifacts.append(ArtifactManifest.from_dict(data))
                    except (json.JSONDecodeError, IOError):
                        continue

        return artifacts

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact by ID."""
        manifest = self.get_artifact(artifact_id)
        if not manifest:
            return False

        # Find and delete the package
        for path in self.artifacts_dir.iterdir():
            if artifact_id in str(path):
                if path.is_file():
                    path.unlink()
                    # Also delete manifest
                    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
                    if manifest_path.exists():
                        manifest_path.unlink()
                else:
                    shutil.rmtree(path)
                return True

        return False


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for artifact packager."""
    import argparse

    parser = argparse.ArgumentParser(description="Artifact Packager (Step 203)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # package command
    pkg_parser = subparsers.add_parser("package", help="Package artifacts")
    pkg_parser.add_argument("source", help="Source directory")
    pkg_parser.add_argument("--version", "-v", required=True, help="Version string")
    pkg_parser.add_argument("--build-id", "-b", default="", help="Build ID")
    pkg_parser.add_argument("--format", "-f", default="tar.gz", choices=["tar.gz", "zip", "tar", "raw"])
    pkg_parser.add_argument("--output", "-o", default="./artifacts", help="Output directory")
    pkg_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List artifacts")
    list_parser.add_argument("--dir", "-d", default="./artifacts", help="Artifacts directory")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # get command
    get_parser = subparsers.add_parser("get", help="Get artifact details")
    get_parser.add_argument("artifact_id", help="Artifact ID")
    get_parser.add_argument("--dir", "-d", default="./artifacts", help="Artifacts directory")
    get_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if args.command == "package":
        packager = ArtifactPackager(args.output)
        manifest = packager.package(
            source_dir=args.source,
            version=args.version,
            build_id=args.build_id,
            format=args.format,
        )

        if args.json:
            print(json.dumps(manifest.to_dict(), indent=2))
        else:
            print(f"Artifact packaged: {manifest.artifact_id}")
            print(f"  Version: {manifest.version}")
            print(f"  Format: {manifest.format.value}")
            print(f"  Size: {manifest.size_bytes} bytes")
            print(f"  Files: {len(manifest.files)}")
            print(f"  Checksum: {manifest.checksum[:16]}...")

        return 0

    elif args.command == "list":
        packager = ArtifactPackager(args.dir)
        artifacts = packager.list_artifacts()

        if args.json:
            print(json.dumps([a.to_dict() for a in artifacts], indent=2))
        else:
            if not artifacts:
                print("No artifacts found")
            else:
                for a in artifacts:
                    print(f"{a.artifact_id} ({a.version}) - {a.format.value}, {a.size_bytes} bytes")

        return 0

    elif args.command == "get":
        packager = ArtifactPackager(args.dir)
        manifest = packager.get_artifact(args.artifact_id)

        if not manifest:
            print(f"Artifact not found: {args.artifact_id}")
            return 1

        if args.json:
            print(json.dumps(manifest.to_dict(), indent=2))
        else:
            print(f"Artifact: {manifest.artifact_id}")
            print(f"  Version: {manifest.version}")
            print(f"  Build ID: {manifest.build_id or 'N/A'}")
            print(f"  Format: {manifest.format.value}")
            print(f"  Size: {manifest.size_bytes} bytes")
            print(f"  Files: {len(manifest.files)}")
            print(f"  Checksum: {manifest.checksum}")
            print(f"  Created: {manifest.created_at}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
