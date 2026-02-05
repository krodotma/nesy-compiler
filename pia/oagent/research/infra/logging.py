#!/usr/bin/env python3
"""
logging.py - Structured Logging System (Step 34)

Structured logging with JSON output, log levels, and bus integration.
Supports context propagation and log rotation.

PBTSO Phase: MONITOR

Bus Topics:
- a2a.research.log.entry
- research.log.rotate

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import io
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Convert string to LogLevel."""
        return cls[level.upper()]


@dataclass
class LogConfig:
    """Configuration for structured logger."""

    level: LogLevel = LogLevel.INFO
    format: str = "json"  # json, text, compact
    output: str = "both"  # stdout, file, both, bus
    log_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    include_stacktrace: bool = True
    include_context: bool = True
    emit_to_bus: bool = True
    bus_path: Optional[str] = None
    color_enabled: bool = True
    timestamp_format: str = "iso"  # iso, unix, human

    def __post_init__(self):
        if self.log_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.log_path = f"{pluribus_root}/.pluribus/research/logs/research.log"
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class LogEntry:
    """A structured log entry."""

    level: LogLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    logger_name: str = "research"
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    source_function: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "level": self.level.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat() + "Z",
            "logger": self.logger_name,
        }

        if self.context:
            data["context"] = self.context
        if self.exception:
            data["exception"] = self.exception
        if self.stack_trace:
            data["stack_trace"] = self.stack_trace
        if self.source_file:
            data["source"] = {
                "file": self.source_file,
                "line": self.source_line,
                "function": self.source_function,
            }
        if self.correlation_id:
            data["correlation_id"] = self.correlation_id

        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# ============================================================================
# Formatters
# ============================================================================


class JSONFormatter:
    """Format log entries as JSON."""

    def format(self, entry: LogEntry) -> str:
        """Format entry as JSON line."""
        return entry.to_json()


class TextFormatter:
    """Format log entries as human-readable text."""

    COLORS = {
        LogLevel.DEBUG: "\033[36m",    # Cyan
        LogLevel.INFO: "\033[32m",     # Green
        LogLevel.WARNING: "\033[33m",  # Yellow
        LogLevel.ERROR: "\033[31m",    # Red
        LogLevel.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, color_enabled: bool = True):
        self.color_enabled = color_enabled and sys.stdout.isatty()

    def format(self, entry: LogEntry) -> str:
        """Format entry as text."""
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = entry.level.name.ljust(8)

        if self.color_enabled:
            color = self.COLORS.get(entry.level, "")
            level = f"{color}{level}{self.RESET}"

        parts = [f"{ts} {level} [{entry.logger_name}] {entry.message}"]

        if entry.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in entry.context.items())
            parts.append(f"  context: {ctx_str}")

        if entry.exception:
            parts.append(f"  exception: {entry.exception}")

        if entry.stack_trace:
            parts.append(f"  stack:\n{entry.stack_trace}")

        return "\n".join(parts)


class CompactFormatter:
    """Format log entries in compact single-line format."""

    def format(self, entry: LogEntry) -> str:
        """Format entry as compact line."""
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
        level = entry.level.name[0]  # First letter only
        return f"{ts} {level} {entry.message}"


# ============================================================================
# Log Context
# ============================================================================


class LogContext:
    """
    Thread-local log context for propagating context through calls.

    Example:
        with LogContext.bind(request_id="abc123"):
            logger.info("Processing request")  # Includes request_id
    """

    _local = threading.local()

    @classmethod
    def bind(cls, **kwargs) -> "LogContextManager":
        """Bind context values for current thread."""
        return LogContextManager(kwargs)

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get current context."""
        if not hasattr(cls._local, "context"):
            cls._local.context = {}
        return cls._local.context.copy()

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a context value."""
        if not hasattr(cls._local, "context"):
            cls._local.context = {}
        cls._local.context[key] = value

    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        cls._local.context = {}


class LogContextManager:
    """Context manager for log context binding."""

    def __init__(self, values: Dict[str, Any]):
        self.values = values
        self.previous: Dict[str, Any] = {}

    def __enter__(self):
        # Store previous values
        ctx = LogContext.get()
        self.previous = {k: ctx.get(k) for k in self.values}

        # Set new values
        for k, v in self.values.items():
            LogContext.set(k, v)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous values
        for k, v in self.previous.items():
            if v is None:
                if hasattr(LogContext._local, "context") and k in LogContext._local.context:
                    del LogContext._local.context[k]
            else:
                LogContext.set(k, v)


# ============================================================================
# Structured Logger
# ============================================================================


class StructuredLogger:
    """
    Structured logging with JSON output and bus integration.

    Features:
    - Multiple output targets (stdout, file, bus)
    - JSON and text formatting
    - Context propagation
    - Log rotation
    - Correlation IDs

    PBTSO Phase: MONITOR

    Example:
        logger = StructuredLogger("research.query")
        logger.info("Processing query", query="test", results=10)

        with LogContext.bind(request_id="abc"):
            logger.debug("Context included automatically")
    """

    _instances: Dict[str, "StructuredLogger"] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        name: str = "research",
        config: Optional[LogConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the structured logger.

        Args:
            name: Logger name
            config: Logger configuration
            bus: AgentBus for event emission
        """
        self.name = name
        self.config = config or LogConfig()
        self.bus = bus or AgentBus()

        # Formatters
        self._json_formatter = JSONFormatter()
        self._text_formatter = TextFormatter(self.config.color_enabled)
        self._compact_formatter = CompactFormatter()

        # File handler
        self._file_handler: Optional[RotatingFileHandler] = None
        if self.config.output in ("file", "both"):
            self._init_file_handler()

        # Correlation ID
        self._correlation_id: Optional[str] = None

    @classmethod
    def get_logger(
        cls,
        name: str = "research",
        config: Optional[LogConfig] = None,
    ) -> "StructuredLogger":
        """Get or create a logger instance."""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, config)
            return cls._instances[name]

    def set_level(self, level: LogLevel) -> None:
        """Set logging level."""
        self.config.level = level

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for log entries."""
        self._correlation_id = correlation_id

    def debug(self, message: str, **kwargs) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def warn(self, message: str, **kwargs) -> None:
        """Alias for warning."""
        self.warning(message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log at ERROR level."""
        exception = None
        stack_trace = None

        if exc_info:
            exc_type, exc_value, exc_tb = sys.exc_info()
            if exc_value:
                exception = str(exc_value)
                if self.config.include_stacktrace:
                    stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

        self._log(LogLevel.ERROR, message, exception=exception, stack_trace=stack_trace, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log at CRITICAL level."""
        exception = None
        stack_trace = None

        if exc_info:
            exc_type, exc_value, exc_tb = sys.exc_info()
            if exc_value:
                exception = str(exc_value)
                if self.config.include_stacktrace:
                    stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

        self._log(LogLevel.CRITICAL, message, exception=exception, stack_trace=stack_trace, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with stack trace."""
        self.error(message, exc_info=True, **kwargs)

    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Log at specified level."""
        self._log(level, message, **kwargs)

    def _log(
        self,
        level: LogLevel,
        message: str,
        exception: Optional[str] = None,
        stack_trace: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Internal logging method."""
        if level.value < self.config.level.value:
            return

        # Get source info
        source_file = None
        source_line = None
        source_function = None

        frame = sys._getframe(2)
        if frame:
            source_file = frame.f_code.co_filename
            source_line = frame.f_lineno
            source_function = frame.f_code.co_name

        # Build context
        context = {}
        if self.config.include_context:
            context.update(LogContext.get())
        context.update(kwargs)

        # Create entry
        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            context=context,
            exception=exception,
            stack_trace=stack_trace,
            source_file=source_file,
            source_line=source_line,
            source_function=source_function,
            correlation_id=self._correlation_id,
        )

        # Output
        self._output(entry)

    def _output(self, entry: LogEntry) -> None:
        """Output log entry to configured destinations."""
        # Format based on config
        if self.config.format == "json":
            formatted = self._json_formatter.format(entry)
        elif self.config.format == "compact":
            formatted = self._compact_formatter.format(entry)
        else:
            formatted = self._text_formatter.format(entry)

        # Write to stdout
        if self.config.output in ("stdout", "both"):
            print(formatted)

        # Write to file
        if self._file_handler and self.config.output in ("file", "both"):
            self._file_handler.emit(
                logging.LogRecord(
                    name=self.name,
                    level=entry.level.value,
                    pathname="",
                    lineno=0,
                    msg=self._json_formatter.format(entry),
                    args=(),
                    exc_info=None,
                )
            )

        # Write to bus
        if self.config.emit_to_bus:
            self._emit_to_bus(entry)

    def _init_file_handler(self) -> None:
        """Initialize file handler with rotation."""
        log_path = Path(self.config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count,
        )

    def _emit_to_bus(self, entry: LogEntry) -> None:
        """Emit log entry to bus."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": entry.timestamp,
            "iso": datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).isoformat() + "Z",
            "topic": "a2a.research.log.entry",
            "kind": "log",
            "level": entry.level.name.lower(),
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": entry.to_dict(),
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ============================================================================
# Convenience Functions
# ============================================================================


def get_logger(name: str = "research", config: Optional[LogConfig] = None) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger.get_logger(name, config)


def debug(message: str, **kwargs) -> None:
    """Log debug message to default logger."""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs) -> None:
    """Log info message to default logger."""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs) -> None:
    """Log warning message to default logger."""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs) -> None:
    """Log error message to default logger."""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs) -> None:
    """Log critical message to default logger."""
    get_logger().critical(message, **kwargs)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Logging."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Logging (Step 34)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run logging demo")
    demo_parser.add_argument("--level", default="DEBUG", help="Log level")
    demo_parser.add_argument("--format", choices=["json", "text", "compact"], default="text")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test logging output")
    test_parser.add_argument("message", help="Message to log")
    test_parser.add_argument("--level", default="INFO", help="Log level")

    args = parser.parse_args()

    if args.command == "demo":
        config = LogConfig(
            level=LogLevel.from_string(args.level),
            format=args.format,
            output="stdout",
            emit_to_bus=False,
        )
        logger = StructuredLogger("demo", config)

        print("Running logging demo...\n")

        logger.debug("This is a debug message", component="demo")
        logger.info("This is an info message", user="test", action="demo")
        logger.warning("This is a warning message", threshold=0.8)
        logger.error("This is an error message", error_code=500)

        # With context
        with LogContext.bind(request_id="abc123", session="xyz"):
            logger.info("Message with context")
            logger.debug("Context is propagated")

        # Exception logging
        try:
            raise ValueError("Demo exception")
        except ValueError:
            logger.exception("Caught an exception")

    elif args.command == "test":
        config = LogConfig(
            level=LogLevel.from_string(args.level),
            output="stdout",
            emit_to_bus=True,
        )
        logger = StructuredLogger("test", config)
        logger.log(LogLevel.from_string(args.level), args.message)

    return 0


if __name__ == "__main__":
    sys.exit(main())
