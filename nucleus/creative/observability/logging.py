"""
Structured Logging Module for Observability
============================================

Provides JSON-formatted structured logging with context propagation
and correlation IDs for distributed tracing integration.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, TextIO, TypeVar

# Import tracing for correlation
try:
    from .tracing import get_current_trace_id
except ImportError:
    def get_current_trace_id() -> Optional[str]:
        return None

T = TypeVar("T")


class LogLevel(Enum):
    """
    Log severity levels.

    Follows standard logging conventions with numeric values
    compatible with Python's logging module.

    Attributes:
        DEBUG: Detailed diagnostic information (10).
        INFO: General operational information (20).
        WARNING: Indication of potential issues (30).
        ERROR: Error conditions that need attention (40).
        CRITICAL: Severe errors requiring immediate action (50).
    """
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """
        Convert string to LogLevel.

        Args:
            level: Case-insensitive level name.

        Returns:
            Corresponding LogLevel.

        Raises:
            ValueError: If level name is not recognized.
        """
        level_upper = level.upper()
        try:
            return cls[level_upper]
        except KeyError:
            raise ValueError(f"Unknown log level: {level}")

    def to_logging_level(self) -> int:
        """Convert to Python logging module level."""
        return self.value


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for Python logging handlers.

    Formats log records as JSON objects for structured logging
    compatibility with log aggregation systems.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the JSON formatter.

        Args:
            include_timestamp: Include timestamp in output.
            include_level: Include log level in output.
            include_logger: Include logger name in output.
            extra_fields: Additional fields to include in every log.
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_entry: Dict[str, Any] = {}

        if self.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.include_level:
            log_entry["level"] = record.levelname

        if self.include_logger:
            log_entry["logger"] = record.name

        log_entry["message"] = record.getMessage()

        # Add extra fields from formatter config
        log_entry.update(self.extra_fields)

        # Add extra fields from record
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_entry.update(record.extra)

        # Add trace correlation if available
        trace_id = get_current_trace_id()
        if trace_id:
            log_entry["trace_id"] = trace_id

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """
    A structured logger with context binding.

    Provides JSON-formatted log output with support for
    bound context, trace correlation, and log levels.

    Attributes:
        name: Logger name/identifier.

    Example:
        >>> logger = StructuredLogger("my-service")
        >>> logger.info("Request received", method="GET", path="/api")
        >>>
        >>> # With bound context
        >>> request_logger = logger.bind(request_id="abc123")
        >>> request_logger.info("Processing")  # includes request_id
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        output: Optional[TextIO] = None,
    ) -> None:
        """
        Initialize the structured logger.

        Args:
            name: Logger name.
            level: Minimum log level to emit.
            output: Output stream (defaults to stderr).
        """
        self.name = name
        self.level = level
        self._context: Dict[str, Any] = {}
        self._output = output or sys.stderr
        self._logger = logging.getLogger(name)
        self._lock = threading.Lock()

        # Configure underlying logger
        self._logger.setLevel(level.value)
        if not self._logger.handlers:
            handler = logging.StreamHandler(self._output)
            handler.setFormatter(JsonFormatter())
            self._logger.addHandler(handler)

    def bind(self, **context: Any) -> "StructuredLogger":
        """
        Create a new logger with additional bound context.

        All subsequent log calls from the returned logger will
        include the bound context fields.

        Args:
            **context: Key-value pairs to bind.

        Returns:
            New StructuredLogger with merged context.

        Example:
            >>> logger = StructuredLogger("app")
            >>> request_logger = logger.bind(request_id="abc", user="john")
            >>> request_logger.info("Action")  # includes request_id and user
        """
        new_logger = StructuredLogger(self.name, self.level, self._output)
        new_logger._context = {**self._context, **context}
        new_logger._logger = self._logger
        return new_logger

    def unbind(self, *keys: str) -> "StructuredLogger":
        """
        Create a new logger without specified context keys.

        Args:
            *keys: Context keys to remove.

        Returns:
            New StructuredLogger without specified keys.
        """
        new_logger = StructuredLogger(self.name, self.level, self._output)
        new_logger._context = {k: v for k, v in self._context.items() if k not in keys}
        new_logger._logger = self._logger
        return new_logger

    def set_level(self, level: LogLevel) -> None:
        """
        Set the minimum log level.

        Args:
            level: New minimum log level.
        """
        self.level = level
        self._logger.setLevel(level.value)

    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """
        Internal log method.

        Args:
            level: Log level.
            message: Log message.
            **kwargs: Additional fields to include.
        """
        if level.value < self.level.value:
            return

        record_dict: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.name,
            "logger": self.name,
            "message": message,
            **self._context,
            **kwargs,
        }

        # Add trace correlation
        trace_id = get_current_trace_id()
        if trace_id:
            record_dict["trace_id"] = trace_id

        with self._lock:
            self._output.write(json.dumps(record_dict, default=str) + "\n")
            self._output.flush()

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log a debug message.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log an info message.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log a warning message.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.WARNING, message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> None:
        """Alias for warning()."""
        self.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log an error message.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """
        Log a critical message.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def exception(self, message: str, exc: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log an error with exception details.

        Args:
            message: Log message.
            exc: Exception to log (uses current exception if None).
            **kwargs: Additional fields.
        """
        if exc:
            kwargs["exception_type"] = type(exc).__name__
            kwargs["exception_message"] = str(exc)
        self._log(LogLevel.ERROR, message, **kwargs)

    def log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """
        Log at a specific level.

        Args:
            level: Log level.
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(level, message, **kwargs)


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}
_loggers_lock = threading.Lock()

# Default log level
_default_level: LogLevel = LogLevel.INFO


def get_logger(name: str, level: Optional[LogLevel] = None) -> StructuredLogger:
    """
    Get or create a structured logger by name.

    Args:
        name: Logger name.
        level: Optional log level (uses default if not specified).

    Returns:
        StructuredLogger instance.

    Example:
        >>> logger = get_logger("my-module")
        >>> logger.info("Starting up")
    """
    with _loggers_lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(name, level or _default_level)
        return _loggers[name]


def set_default_level(level: LogLevel) -> None:
    """
    Set the default log level for new loggers.

    Args:
        level: New default log level.
    """
    global _default_level
    _default_level = level


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    output: Optional[TextIO] = None,
    format_json: bool = True,
) -> None:
    """
    Configure global logging settings.

    Args:
        level: Minimum log level.
        output: Output stream.
        format_json: Use JSON formatting.
    """
    global _default_level
    _default_level = level

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level.value)

    # Clear existing handlers
    root.handlers.clear()

    # Add new handler
    handler = logging.StreamHandler(output or sys.stderr)
    if format_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    root.addHandler(handler)


class LogContext:
    """
    Thread-local log context manager.

    Allows setting context that will be included in all log
    messages within the context scope.

    Example:
        >>> with LogContext(request_id="abc123"):
        ...     get_logger("app").info("Processing")  # includes request_id
    """

    _local = threading.local()

    def __init__(self, **context: Any) -> None:
        """
        Initialize log context.

        Args:
            **context: Context key-value pairs.
        """
        self._context = context
        self._previous: Dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        """Enter the context."""
        if not hasattr(self._local, "context"):
            self._local.context = {}
        self._previous = dict(self._local.context)
        self._local.context.update(self._context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context."""
        self._local.context = self._previous

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get the current thread-local context."""
        if not hasattr(cls._local, "context"):
            cls._local.context = {}
        return dict(cls._local.context)


def logged(
    level: LogLevel = LogLevel.INFO,
    message: Optional[str] = None,
    **extra_fields: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to log function entry and exit.

    Args:
        level: Log level for messages.
        message: Optional custom message.
        **extra_fields: Extra fields to include.

    Returns:
        Decorator function.

    Example:
        >>> @logged(level=LogLevel.DEBUG)
        ... def process_data(data):
        ...     return data.upper()
    """
    import functools
    import time

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        logger = get_logger(func.__module__ or "unknown")
        func_name = func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            msg = message or f"Calling {func_name}"
            logger.log(level, f"{msg} - start", function=func_name, **extra_fields)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log(
                    level,
                    f"{msg} - end",
                    function=func_name,
                    duration_ms=round(duration_ms, 2),
                    success=True,
                    **extra_fields,
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log(
                    LogLevel.ERROR,
                    f"{msg} - error",
                    function=func_name,
                    duration_ms=round(duration_ms, 2),
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                    **extra_fields,
                )
                raise

        return wrapper

    return decorator


__all__ = [
    "LogLevel",
    "StructuredLogger",
    "JsonFormatter",
    "LogContext",
    "get_logger",
    "set_default_level",
    "configure_logging",
    "logged",
]
