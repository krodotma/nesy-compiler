#!/usr/bin/env python3
"""
Error Handler - Comprehensive Error Handling (Step 85)

Provides error handling and recovery capabilities.
"""

from .error_handler import (
    CodeAgentError,
    ConfigurationError,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorRecovery,
    ErrorReport,
    ErrorSeverity,
    OperationError,
    PluginError,
    ValidationError,
    main,
    recoverable,
    with_error_handling,
)

__all__ = [
    "CodeAgentError",
    "ConfigurationError",
    "ErrorCategory",
    "ErrorContext",
    "ErrorHandler",
    "ErrorRecovery",
    "ErrorReport",
    "ErrorSeverity",
    "OperationError",
    "PluginError",
    "ValidationError",
    "main",
    "recoverable",
    "with_error_handling",
]
