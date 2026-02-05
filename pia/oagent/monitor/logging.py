#!/usr/bin/env python3
"""
Monitor Logging - Step 284

Structured logging system for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.log.* (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def numeric(self) -> int:
        """Get numeric level."""
        return {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50,
        }[self]


class OutputFormat(Enum):
    """Log output formats."""
    JSON = "json"
    TEXT = "text"
    COMPACT = "compact"


@dataclass
class LogContext:
    """Logging context with structured fields.

    Attributes:
        fields: Context fields
        operation: Current operation name
        trace_id: Trace ID for correlation
        span_id: Span ID for distributed tracing
    """
    fields: Dict[str, Any] = field(default_factory=dict)
    operation: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def with_field(self, key: str, value: Any) -> "LogContext":
        """Create new context with additional field.

        Args:
            key: Field key
            value: Field value

        Returns:
            New context
        """
        new_fields = dict(self.fields)
        new_fields[key] = value
        return LogContext(
            fields=new_fields,
            operation=self.operation,
            trace_id=self.trace_id,
            span_id=self.span_id,
        )

    def with_fields(self, **kwargs: Any) -> "LogContext":
        """Create new context with additional fields.

        Args:
            **kwargs: Fields to add

        Returns:
            New context
        """
        new_fields = dict(self.fields)
        new_fields.update(kwargs)
        return LogContext(
            fields=new_fields,
            operation=self.operation,
            trace_id=self.trace_id,
            span_id=self.span_id,
        )


@dataclass
class LogEntry:
    """A structured log entry.

    Attributes:
        level: Log level
        message: Log message
        timestamp: Entry timestamp
        context: Log context
        error: Error information
        caller: Caller information
    """
    level: LogLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    context: LogContext = field(default_factory=LogContext)
    error: Optional[Dict[str, Any]] = None
    caller: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat() + "Z",
        }

        if self.context.fields:
            result["context"] = self.context.fields
        if self.context.operation:
            result["operation"] = self.context.operation
        if self.context.trace_id:
            result["trace_id"] = self.context.trace_id
        if self.context.span_id:
            result["span_id"] = self.context.span_id
        if self.error:
            result["error"] = self.error
        if self.caller:
            result["caller"] = self.caller

        return result

    def to_text(self) -> str:
        """Convert to text format."""
        ts = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        level = self.level.value.upper()[:4]
        msg = f"[{ts}] {level} {self.message}"

        if self.context.operation:
            msg = f"[{ts}] {level} [{self.context.operation}] {self.message}"

        if self.context.fields:
            fields = " ".join(f"{k}={v}" for k, v in self.context.fields.items())
            msg = f"{msg} {fields}"

        if self.error:
            msg = f"{msg} error={self.error.get('message', 'unknown')}"

        return msg

    def to_compact(self) -> str:
        """Convert to compact format."""
        level = self.level.value[0].upper()
        ts = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
        return f"{level} {ts} {self.message}"


class LogHandler:
    """Base log handler."""

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """Initialize handler.

        Args:
            level: Minimum log level
        """
        self.level = level

    def should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged.

        Args:
            level: Log level

        Returns:
            True if should log
        """
        return level.numeric >= self.level.numeric

    def emit(self, entry: LogEntry) -> None:
        """Emit a log entry.

        Args:
            entry: Log entry
        """
        pass


class ConsoleHandler(LogHandler):
    """Handler that writes to console."""

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        format: OutputFormat = OutputFormat.TEXT,
        stream: Optional[TextIO] = None,
    ):
        """Initialize console handler.

        Args:
            level: Minimum log level
            format: Output format
            stream: Output stream
        """
        super().__init__(level)
        self.format = format
        self.stream = stream or sys.stderr

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to console."""
        if not self.should_log(entry.level):
            return

        if self.format == OutputFormat.JSON:
            output = json.dumps(entry.to_dict())
        elif self.format == OutputFormat.COMPACT:
            output = entry.to_compact()
        else:
            output = entry.to_text()

        self.stream.write(output + "\n")
        self.stream.flush()


class FileHandler(LogHandler):
    """Handler that writes to file."""

    def __init__(
        self,
        path: str,
        level: LogLevel = LogLevel.INFO,
        format: OutputFormat = OutputFormat.JSON,
        max_size_mb: int = 100,
        backup_count: int = 5,
    ):
        """Initialize file handler.

        Args:
            path: Log file path
            level: Minimum log level
            format: Output format
            max_size_mb: Maximum file size in MB
            backup_count: Number of backup files
        """
        super().__init__(level)
        self.path = Path(path)
        self.format = format
        self.max_size = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self._lock = threading.Lock()

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to file."""
        if not self.should_log(entry.level):
            return

        if self.format == OutputFormat.JSON:
            output = json.dumps(entry.to_dict())
        elif self.format == OutputFormat.COMPACT:
            output = entry.to_compact()
        else:
            output = entry.to_text()

        with self._lock:
            self._rotate_if_needed()

            with open(self.path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(output + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _rotate_if_needed(self) -> None:
        """Rotate log file if needed."""
        if not self.path.exists():
            return

        if self.path.stat().st_size < self.max_size:
            return

        # Rotate files
        for i in range(self.backup_count - 1, 0, -1):
            src = Path(f"{self.path}.{i}")
            dst = Path(f"{self.path}.{i + 1}")
            if src.exists():
                src.rename(dst)

        # Move current to .1
        self.path.rename(Path(f"{self.path}.1"))


class BusHandler(LogHandler):
    """Handler that writes to Pluribus bus."""

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        bus_dir: Optional[str] = None,
    ):
        """Initialize bus handler.

        Args:
            level: Minimum log level
            bus_dir: Bus directory
        """
        super().__init__(level)

        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to bus."""
        if not self.should_log(entry.level):
            return

        event = {
            "id": str(uuid.uuid4()),
            "ts": entry.timestamp,
            "iso": datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).isoformat() + "Z",
            "topic": f"monitor.log.{entry.level.value}",
            "kind": "log",
            "level": entry.level.value,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": entry.to_dict(),
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass


class MonitorLogger:
    """
    Structured logging system for the Monitor Agent.

    Provides:
    - Structured log entries with context
    - Multiple output handlers
    - Log level filtering
    - Error tracking
    - Performance-optimized logging

    Example:
        logger = MonitorLogger("metrics")

        # Basic logging
        logger.info("Processing metrics")
        logger.error("Failed to process", error=e)

        # With context
        ctx = LogContext(fields={"agent": "code"})
        logger.with_context(ctx).info("Agent metrics collected")

        # With fields
        logger.with_field("metric", "cpu").info("Metric recorded")
    """

    BUS_TOPICS = {
        "debug": "monitor.log.debug",
        "info": "monitor.log.info",
        "warning": "monitor.log.warning",
        "error": "monitor.log.error",
        "critical": "monitor.log.critical",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[LogHandler]] = None,
    ):
        """Initialize logger.

        Args:
            name: Logger name
            level: Minimum log level
            handlers: Log handlers
        """
        self.name = name
        self.level = level
        self._handlers = handlers or [ConsoleHandler()]
        self._context = LogContext()
        self._last_heartbeat = time.time()

        # Statistics
        self._log_counts: Dict[str, int] = {level.value: 0 for level in LogLevel}

    def add_handler(self, handler: LogHandler) -> None:
        """Add a log handler.

        Args:
            handler: Log handler
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a log handler.

        Args:
            handler: Log handler
        """
        if handler in self._handlers:
            self._handlers.remove(handler)

    def with_context(self, context: LogContext) -> "MonitorLogger":
        """Create logger with context.

        Args:
            context: Log context

        Returns:
            Logger with context
        """
        new_logger = MonitorLogger(
            name=self.name,
            level=self.level,
            handlers=self._handlers,
        )
        new_logger._context = context
        new_logger._log_counts = self._log_counts
        return new_logger

    def with_field(self, key: str, value: Any) -> "MonitorLogger":
        """Create logger with additional field.

        Args:
            key: Field key
            value: Field value

        Returns:
            Logger with field
        """
        return self.with_context(self._context.with_field(key, value))

    def with_fields(self, **kwargs: Any) -> "MonitorLogger":
        """Create logger with additional fields.

        Args:
            **kwargs: Fields to add

        Returns:
            Logger with fields
        """
        return self.with_context(self._context.with_fields(**kwargs))

    def with_operation(self, operation: str) -> "MonitorLogger":
        """Create logger with operation name.

        Args:
            operation: Operation name

        Returns:
            Logger with operation
        """
        new_context = LogContext(
            fields=self._context.fields,
            operation=operation,
            trace_id=self._context.trace_id,
            span_id=self._context.span_id,
        )
        return self.with_context(new_context)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Log error message.

        Args:
            message: Log message
            error: Exception
            **kwargs: Additional fields
        """
        self._log(LogLevel.ERROR, message, error=error, **kwargs)

    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Log critical message.

        Args:
            message: Log message
            error: Exception
            **kwargs: Additional fields
        """
        self._log(LogLevel.CRITICAL, message, error=error, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        exc_info = sys.exc_info()
        if exc_info[1]:
            self._log(
                LogLevel.ERROR,
                message,
                error=exc_info[1],
                traceback=traceback.format_exc(),
                **kwargs,
            )
        else:
            self._log(LogLevel.ERROR, message, **kwargs)

    def _log(
        self,
        level: LogLevel,
        message: str,
        error: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Internal log method.

        Args:
            level: Log level
            message: Log message
            error: Exception
            **kwargs: Additional fields
        """
        if level.numeric < self.level.numeric:
            return

        # Build context with additional fields
        context = self._context
        if kwargs:
            tb = kwargs.pop("traceback", None)
            context = context.with_fields(**kwargs)

        # Build error info
        error_info = None
        if error:
            error_info = {
                "type": type(error).__name__,
                "message": str(error),
            }
            if "tb" in dir() and tb:
                error_info["traceback"] = tb

        # Create entry
        entry = LogEntry(
            level=level,
            message=message,
            context=context,
            error=error_info,
            caller=self._get_caller(),
        )

        # Update statistics
        self._log_counts[level.value] += 1

        # Emit to handlers
        for handler in self._handlers:
            try:
                handler.emit(entry)
            except Exception:
                pass

    def _get_caller(self) -> Dict[str, str]:
        """Get caller information."""
        try:
            frame = sys._getframe(3)  # Skip _log, log method, and this method
            return {
                "file": frame.f_code.co_filename,
                "function": frame.f_code.co_name,
                "line": str(frame.f_lineno),
            }
        except Exception:
            return {}

    def get_stats(self) -> Dict[str, int]:
        """Get logging statistics.

        Returns:
            Log counts by level
        """
        return dict(self._log_counts)

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        self.info("Heartbeat", stats=self._log_counts)
        return True


# Singleton instances
_loggers: Dict[str, MonitorLogger] = {}


def get_logger(name: str = "monitor") -> MonitorLogger:
    """Get or create a logger.

    Args:
        name: Logger name

    Returns:
        MonitorLogger instance
    """
    if name not in _loggers:
        _loggers[name] = MonitorLogger(name)
    return _loggers[name]


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    console: bool = True,
    file_path: Optional[str] = None,
    bus: bool = True,
    format: OutputFormat = OutputFormat.TEXT,
) -> None:
    """Configure global logging.

    Args:
        level: Log level
        console: Enable console output
        file_path: Log file path
        bus: Enable bus output
        format: Output format
    """
    handlers: List[LogHandler] = []

    if console:
        handlers.append(ConsoleHandler(level=level, format=format))

    if file_path:
        handlers.append(FileHandler(path=file_path, level=level, format=OutputFormat.JSON))

    if bus:
        handlers.append(BusHandler(level=level))

    for name in _loggers:
        _loggers[name]._handlers = handlers
        _loggers[name].level = level


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Logging (Step 284)")
    parser.add_argument("--level", default="info", help="Log level")
    parser.add_argument("--message", "-m", help="Log a message")
    parser.add_argument("--error", action="store_true", help="Log as error")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--format", default="text", help="Output format")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    logger = get_logger("cli")
    logger.add_handler(ConsoleHandler(
        level=LogLevel.DEBUG,
        format=OutputFormat(args.format),
    ))

    if args.message:
        if args.error:
            logger.error(args.message)
        else:
            logger.info(args.message)

    if args.stats:
        stats = logger.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Log Statistics:")
            for level, count in stats.items():
                print(f"  {level}: {count}")

    if not args.message and not args.stats:
        # Demo logging
        logger.debug("Debug message")
        logger.info("Info message", component="demo")
        logger.warning("Warning message")
        logger.with_field("metric", "cpu").info("Metric logged")
        logger.with_operation("test").info("Operation log")
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.error("Error occurred", error=e)
