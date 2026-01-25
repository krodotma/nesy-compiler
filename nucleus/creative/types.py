"""
Shared Types for Creative Section
==================================

Type definitions used across all subsystems.
"""

from __future__ import annotations

from typing import TypeVar, Generic, Union, Optional, Callable, Any, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy.typing as npt

# Type aliases
ImageArray = npt.NDArray  # HxWx3 or HxWx4 uint8/float32
AudioArray = npt.NDArray  # Samples x Channels float32
MeshVertices = npt.NDArray  # Nx3 float32

# Callback types
ProgressCallback = Callable[[str, float], None]

# Status types
StatusType = str  # "pending" | "running" | "completed" | "failed"
QualityLevel = str  # "low" | "medium" | "high" | "ultra"
ProcessingMode = str  # "sync" | "async" | "batch"

# Result types
T = TypeVar("T")
E = TypeVar("E", bound=Exception)


@dataclass
class Success(Generic[T]):
    """Successful result wrapper."""
    value: T

    def is_success(self) -> bool:
        return True

    def is_failure(self) -> bool:
        return False


@dataclass
class Failure(Generic[E]):
    """Failure result wrapper."""
    error: E

    def is_success(self) -> bool:
        return False

    def is_failure(self) -> bool:
        return True


Result = Union[Success[T], Failure[E]]


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BoundingBox:
    """2D or 3D bounding box."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: Optional[float] = None
    max_z: Optional[float] = None


@dataclass
class TimeRange:
    """A time range with start and end."""
    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class OperationMetrics:
    """Metrics for an operation."""
    duration_ms: float
    memory_mb: float
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
