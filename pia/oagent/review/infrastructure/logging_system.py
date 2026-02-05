#!/usr/bin/env python3
"""
Review Structured Logging System (Step 184)

Structured logging system for the Review Agent with context propagation,
log correlation, and multiple output formats.

PBTSO Phase: OBSERVE, DISTILL
Bus Topics: review.log.entry, review.log.error

Log Features:
- Structured JSON logging
- Context propagation (request ID, review ID, etc.)
- Log levels with filtering
- Multiple outputs (console, file, bus)
- Log correlation across components

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union
import threading


# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class LogLevel(Enum):
    """Log severity levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50

    def __ge__(self, other: "LogLevel") -> bool:
        return self.value >= other.value

    def __le__(self, other: "LogLevel") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "LogLevel") -> bool:
        return self.value > other.value

    def __lt__(self, other: "LogLevel") -> bool:
        return self.value < other.value


class LogFormat(Enum):
    """Output formats for logs."""
    JSON = "json"
    TEXT = "text"
    COMPACT = "compact"


class LogOutput(Enum):
    """Log output destinations."""
    CONSOLE = "console"
    FILE = "file"
    BUS = "bus"


@dataclass
class LogContext:
    """
    Context for log correlation.

    Attributes:
        request_id: HTTP/API request ID
        review_id: Review operation ID
        trace_id: Distributed trace ID
        span_id: Span ID within trace
        user_id: User identifier
        component: Component name
        custom: Custom context fields
    """
    request_id: Optional[str] = None
    review_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (non-None fields only)."""
        result = {}
        if self.request_id:
            result["request_id"] = self.request_id
        if self.review_id:
            result["review_id"] = self.review_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.component:
            result["component"] = self.component
        result.update(self.custom)
        return result

    def merge(self, other: "LogContext") -> "LogContext":
        """Merge with another context (other takes precedence)."""
        return LogContext(
            request_id=other.request_id or self.request_id,
            review_id=other.review_id or self.review_id,
            trace_id=other.trace_id or self.trace_id,
            span_id=other.span_id or self.span_id,
            user_id=other.user_id or self.user_id,
            component=other.component or self.component,
            custom={**self.custom, **other.custom},
        )


@dataclass
class LogEntry:
    """
    A log entry.

    Attributes:
        timestamp: Log timestamp (ISO format)
        level: Log level
        message: Log message
        context: Log context
        data: Additional structured data
        error: Error information (if applicable)
        source: Source file/line
        logger_name: Logger name
    """
    timestamp: str
    level: LogLevel
    message: str
    context: LogContext = field(default_factory=LogContext)
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    logger_name: str = "review"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "timestamp": self.timestamp,
            "level": self.level.name.lower(),
            "message": self.message,
            "logger": self.logger_name,
        }

        ctx = self.context.to_dict()
        if ctx:
            result["context"] = ctx

        if self.data:
            result["data"] = self.data

        if self.error:
            result["error"] = self.error

        if self.source:
            result["source"] = self.source

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def to_text(self) -> str:
        """Convert to human-readable text."""
        level_colors = {
            LogLevel.TRACE: "",
            LogLevel.DEBUG: "",
            LogLevel.INFO: "",
            LogLevel.WARN: "[WARN]",
            LogLevel.ERROR: "[ERROR]",
            LogLevel.FATAL: "[FATAL]",
        }

        level_str = level_colors.get(self.level, f"[{self.level.name}]")
        ts = self.timestamp[:23]  # Truncate to milliseconds

        parts = [f"{ts} {level_str:8} {self.message}"]

        if self.context.review_id:
            parts[0] += f" [review:{self.context.review_id}]"

        if self.data:
            parts.append(f"  data: {json.dumps(self.data)}")

        if self.error:
            parts.append(f"  error: {self.error.get('type')}: {self.error.get('message')}")
            if self.error.get("traceback"):
                parts.append(f"  {self.error['traceback']}")

        return "\n".join(parts)

    def to_compact(self) -> str:
        """Convert to compact single-line format."""
        level = self.level.name[0]
        ts = self.timestamp[11:23]  # Time only
        ctx_parts = []
        if self.context.review_id:
            ctx_parts.append(f"r:{self.context.review_id[:8]}")
        if self.context.request_id:
            ctx_parts.append(f"q:{self.context.request_id[:8]}")
        ctx_str = f"[{','.join(ctx_parts)}]" if ctx_parts else ""
        return f"{ts} {level} {ctx_str} {self.message}"


@dataclass
class LogConfig:
    """
    Configuration for the logging system.

    Attributes:
        level: Minimum log level
        format: Output format
        outputs: Output destinations
        log_file: Log file path (if file output enabled)
        max_file_size_mb: Maximum log file size
        max_files: Maximum number of log files (rotation)
        include_source: Include source file/line
        colorize: Enable colored output
        bus_min_level: Minimum level for bus events
    """
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.JSON
    outputs: List[LogOutput] = field(default_factory=lambda: [LogOutput.CONSOLE])
    log_file: str = ""
    max_file_size_mb: int = 10
    max_files: int = 5
    include_source: bool = False
    colorize: bool = True
    bus_min_level: LogLevel = LogLevel.WARN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.name.lower(),
            "format": self.format.value,
            "outputs": [o.value for o in self.outputs],
            "log_file": self.log_file,
            "max_file_size_mb": self.max_file_size_mb,
            "max_files": self.max_files,
            "include_source": self.include_source,
            "colorize": self.colorize,
            "bus_min_level": self.bus_min_level.name.lower(),
        }


# ============================================================================
# Context Storage (Thread-Local)
# ============================================================================

_context_storage = threading.local()


def get_current_context() -> LogContext:
    """Get the current log context."""
    return getattr(_context_storage, "context", LogContext())


def set_current_context(context: LogContext) -> None:
    """Set the current log context."""
    _context_storage.context = context


@contextmanager
def log_context(**kwargs):
    """
    Context manager for setting log context.

    Example:
        with log_context(review_id="abc123", user_id="user1"):
            logger.info("Processing review")  # Includes context
    """
    old_context = get_current_context()
    new_context = old_context.merge(LogContext(custom=kwargs))

    # Handle known fields
    if "request_id" in kwargs:
        new_context.request_id = kwargs["request_id"]
    if "review_id" in kwargs:
        new_context.review_id = kwargs["review_id"]
    if "trace_id" in kwargs:
        new_context.trace_id = kwargs["trace_id"]
    if "component" in kwargs:
        new_context.component = kwargs["component"]

    set_current_context(new_context)
    try:
        yield new_context
    finally:
        set_current_context(old_context)


# ============================================================================
# Structured Logger
# ============================================================================

class StructuredLogger:
    """
    Structured logging with context propagation.

    Example:
        logger = StructuredLogger("my-component")

        # Simple logging
        logger.info("Operation completed", data={"count": 42})

        # With context
        with log_context(review_id="abc123"):
            logger.debug("Processing file", data={"file": "test.py"})

        # Error logging
        try:
            risky_operation()
        except Exception as e:
            logger.error("Operation failed", exc=e)
    """

    BUS_TOPICS = {
        "entry": "review.log.entry",
        "error": "review.log.error",
    }

    def __init__(
        self,
        name: str = "review",
        config: Optional[LogConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            config: Logger configuration
            bus_path: Path to event bus file
        """
        self.name = name
        self.config = config or LogConfig()
        self.bus_path = bus_path or self._get_bus_path()
        self._file_handle: Optional[TextIO] = None
        self._file_size = 0
        self._lock = threading.Lock()
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_bus_event(self, entry: LogEntry) -> None:
        """Emit log entry to bus."""
        if LogOutput.BUS not in self.config.outputs:
            return
        if entry.level < self.config.bus_min_level:
            return

        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        topic = self.BUS_TOPICS["error"] if entry.level >= LogLevel.ERROR else self.BUS_TOPICS["entry"]

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": entry.timestamp,
            "topic": topic,
            "kind": "log",
            "actor": self.name,
            "data": entry.to_dict(),
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _format_entry(self, entry: LogEntry) -> str:
        """Format log entry based on config."""
        if self.config.format == LogFormat.JSON:
            return entry.to_json()
        elif self.config.format == LogFormat.COMPACT:
            return entry.to_compact()
        else:
            return entry.to_text()

    def _write_console(self, entry: LogEntry, formatted: str) -> None:
        """Write to console."""
        if LogOutput.CONSOLE not in self.config.outputs:
            return

        stream = sys.stderr if entry.level >= LogLevel.ERROR else sys.stdout

        if self.config.colorize and stream.isatty():
            colors = {
                LogLevel.TRACE: "\033[90m",    # Gray
                LogLevel.DEBUG: "\033[36m",    # Cyan
                LogLevel.INFO: "\033[32m",     # Green
                LogLevel.WARN: "\033[33m",     # Yellow
                LogLevel.ERROR: "\033[31m",    # Red
                LogLevel.FATAL: "\033[35m",    # Magenta
            }
            reset = "\033[0m"
            color = colors.get(entry.level, "")
            formatted = f"{color}{formatted}{reset}"

        print(formatted, file=stream)

    def _write_file(self, formatted: str) -> None:
        """Write to log file."""
        if LogOutput.FILE not in self.config.outputs:
            return
        if not self.config.log_file:
            return

        with self._lock:
            # Check rotation
            if self._file_handle and self._file_size >= self.config.max_file_size_mb * 1024 * 1024:
                self._rotate_file()

            # Open file if needed
            if self._file_handle is None:
                log_path = Path(self.config.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._file_handle = open(log_path, "a")
                self._file_size = log_path.stat().st_size if log_path.exists() else 0

            # Write
            line = formatted + "\n"
            self._file_handle.write(line)
            self._file_handle.flush()
            self._file_size += len(line.encode())

    def _rotate_file(self) -> None:
        """Rotate log files."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        log_path = Path(self.config.log_file)

        # Rotate existing files
        for i in range(self.config.max_files - 1, 0, -1):
            old_path = log_path.with_suffix(f".{i}{log_path.suffix}")
            new_path = log_path.with_suffix(f".{i + 1}{log_path.suffix}")
            if old_path.exists():
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)

        # Rotate current file
        if log_path.exists():
            new_path = log_path.with_suffix(f".1{log_path.suffix}")
            log_path.rename(new_path)

        self._file_size = 0

    def _create_entry(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exc: Optional[Exception] = None,
        context: Optional[LogContext] = None,
    ) -> LogEntry:
        """Create a log entry."""
        # Merge contexts
        current_ctx = get_current_context()
        if context:
            ctx = current_ctx.merge(context)
        else:
            ctx = current_ctx

        # Add component if not set
        if not ctx.component:
            ctx.component = self.name

        # Error info
        error = None
        if exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

        # Source info
        source = None
        if self.config.include_source:
            frame = sys._getframe(3)  # Caller's frame
            source = f"{frame.f_code.co_filename}:{frame.f_lineno}"

        return LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            level=level,
            message=message,
            context=ctx,
            data=data or {},
            error=error,
            source=source,
            logger_name=self.name,
        )

    def _log(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exc: Optional[Exception] = None,
        context: Optional[LogContext] = None,
    ) -> None:
        """Internal log method."""
        if level < self.config.level:
            return

        entry = self._create_entry(level, message, data, exc, context)
        formatted = self._format_entry(entry)

        self._write_console(entry, formatted)
        self._write_file(formatted)
        self._emit_bus_event(entry)

    def trace(self, message: str, **kwargs) -> None:
        """Log at TRACE level."""
        self._log(LogLevel.TRACE, message, data=kwargs.get("data"), exc=kwargs.get("exc"))

    def debug(self, message: str, **kwargs) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, data=kwargs.get("data"), exc=kwargs.get("exc"))

    def info(self, message: str, **kwargs) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, data=kwargs.get("data"), exc=kwargs.get("exc"))

    def warn(self, message: str, **kwargs) -> None:
        """Log at WARN level."""
        self._log(LogLevel.WARN, message, data=kwargs.get("data"), exc=kwargs.get("exc"))

    def error(self, message: str, **kwargs) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, data=kwargs.get("data"), exc=kwargs.get("exc"))

    def fatal(self, message: str, **kwargs) -> None:
        """Log at FATAL level."""
        self._log(LogLevel.FATAL, message, data=kwargs.get("data"), exc=kwargs.get("exc"))

    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Log at specified level."""
        self._log(level, message, data=kwargs.get("data"), exc=kwargs.get("exc"))

    def child(self, name: str, **context_kwargs) -> "StructuredLogger":
        """
        Create a child logger with additional context.

        Args:
            name: Child logger name
            **context_kwargs: Additional context

        Returns:
            Child logger instance
        """
        child_logger = StructuredLogger(
            name=f"{self.name}.{name}",
            config=self.config,
            bus_path=self.bus_path,
        )
        # Set context for child
        child_context = get_current_context().merge(LogContext(custom=context_kwargs))
        set_current_context(child_context)
        return child_logger

    def close(self) -> None:
        """Close the logger and release resources."""
        with self._lock:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": f"logger-{self.name}",
            "healthy": True,
            "level": self.config.level.name.lower(),
            "outputs": [o.value for o in self.config.outputs],
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_bus_event(self._create_entry(
            LogLevel.DEBUG,
            "Heartbeat",
            data=status,
        ))
        return status


# ============================================================================
# Global Logger Instance
# ============================================================================

_default_logger: Optional[StructuredLogger] = None


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name (None = default logger)

    Returns:
        Logger instance
    """
    global _default_logger
    if name:
        return StructuredLogger(name)
    if _default_logger is None:
        _default_logger = StructuredLogger("review")
    return _default_logger


def configure_logging(config: LogConfig) -> StructuredLogger:
    """
    Configure the default logger.

    Args:
        config: Logger configuration

    Returns:
        Configured logger
    """
    global _default_logger
    _default_logger = StructuredLogger("review", config)
    return _default_logger


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Logging System."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Logging System (Step 184)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Log command
    log_parser = subparsers.add_parser("log", help="Write a log entry")
    log_parser.add_argument("message", help="Log message")
    log_parser.add_argument("--level", choices=["trace", "debug", "info", "warn", "error", "fatal"],
                           default="info", help="Log level")
    log_parser.add_argument("--data", help="Additional data (JSON)")
    log_parser.add_argument("--review-id", help="Review ID context")

    # Config command
    subparsers.add_parser("config", help="Show logging configuration")

    parser.add_argument("--format", choices=["json", "text", "compact"], default="text",
                       help="Output format")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    config = LogConfig(
        format=LogFormat[args.format.upper()],
        outputs=[LogOutput.CONSOLE],
        level=LogLevel.DEBUG,
    )
    logger = StructuredLogger("cli", config)

    if args.command == "log":
        level = LogLevel[args.level.upper()]
        data = json.loads(args.data) if args.data else None

        with log_context(review_id=args.review_id) if args.review_id else contextmanager(lambda: (yield))():
            logger.log(level, args.message, data=data)

    elif args.command == "config":
        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print("Logging Configuration")
            for k, v in config.to_dict().items():
                print(f"  {k}: {v}")

    else:
        # Default: demo logging
        logger.info("Demo info message", data={"demo": True})
        logger.warn("Demo warning", data={"code": 42})
        with log_context(review_id="demo123"):
            logger.debug("Message with context")

    return 0


if __name__ == "__main__":
    sys.exit(main())
