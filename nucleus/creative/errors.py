"""
Error Types for Creative Section
=================================

Custom exceptions for all Creative subsystems.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, List, Any
from dataclasses import dataclass, field


class ErrorCode(Enum):
    """Error codes for Creative operations."""
    # General errors
    UNKNOWN = auto()
    VALIDATION_FAILED = auto()
    PROCESSING_FAILED = auto()
    RESOURCE_NOT_FOUND = auto()
    RESOURCE_EXHAUSTED = auto()
    CONFIGURATION_ERROR = auto()

    # Provider errors
    PROVIDER_ERROR = auto()
    PROVIDER_UNAVAILABLE = auto()
    PROVIDER_RATE_LIMITED = auto()

    # Subsystem-specific errors
    GRAMMARS_PARSE_ERROR = auto()
    GRAMMARS_EVOLUTION_ERROR = auto()
    CINEMA_GENERATION_ERROR = auto()
    CINEMA_TEMPORAL_ERROR = auto()
    VISUAL_GENERATION_ERROR = auto()
    VISUAL_STYLE_ERROR = auto()
    AURALUX_SYNTHESIS_ERROR = auto()
    AURALUX_RECOGNITION_ERROR = auto()
    AVATARS_EXTRACTION_ERROR = auto()
    AVATARS_RENDERING_ERROR = auto()
    DITS_EVALUATION_ERROR = auto()
    DITS_TRANSITION_ERROR = auto()


class CreativeError(Exception):
    """Base exception for all Creative errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        details: Optional[dict] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "code": self.code.name,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }


class ValidationError(CreativeError):
    """Validation failed."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCode.VALIDATION_FAILED, **kwargs)
        self.field = field


class ProcessingError(CreativeError):
    """Processing operation failed."""

    def __init__(self, message: str, stage: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCode.PROCESSING_FAILED, **kwargs)
        self.stage = stage


class ResourceError(CreativeError):
    """Resource-related error."""

    def __init__(self, message: str, resource: Optional[str] = None, **kwargs):
        if "code" not in kwargs:
            kwargs["code"] = ErrorCode.RESOURCE_NOT_FOUND
        super().__init__(message, **kwargs)
        self.resource = resource


class ConfigurationError(CreativeError):
    """Configuration error."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCode.CONFIGURATION_ERROR, **kwargs)
        self.config_key = config_key


class ProviderError(CreativeError):
    """External provider error."""

    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        if "code" not in kwargs:
            kwargs["code"] = ErrorCode.PROVIDER_ERROR
        super().__init__(message, **kwargs)
        self.provider = provider


# Subsystem-specific errors

class GrammarsError(CreativeError):
    """Grammars subsystem error."""
    pass


class CinemaError(CreativeError):
    """Cinema subsystem error."""
    pass


class VisualError(CreativeError):
    """Visual subsystem error."""
    pass


class AuraluxError(CreativeError):
    """Auralux subsystem error."""
    pass


class AvatarsError(CreativeError):
    """Avatars subsystem error."""
    pass


class DiTSError(CreativeError):
    """DiTS subsystem error."""
    pass


def wrap_error(exc: Exception, wrapper_type: type = CreativeError) -> CreativeError:
    """Wrap a generic exception in a CreativeError."""
    if isinstance(exc, CreativeError):
        return exc
    return wrapper_type(str(exc), cause=exc)


def format_error_chain(error: Exception, max_depth: int = 5) -> str:
    """Format an error chain as a readable string."""
    parts = []
    current = error
    depth = 0

    while current and depth < max_depth:
        parts.append(f"{'  ' * depth}{type(current).__name__}: {current}")
        if isinstance(current, CreativeError):
            current = current.cause
        elif hasattr(current, "__cause__"):
            current = current.__cause__
        else:
            break
        depth += 1

    return "\n".join(parts)
