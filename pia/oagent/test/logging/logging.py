#!/usr/bin/env python3
"""
Step 134: Test Logging

Structured logging system for the Test Agent.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.log.entry (emits)
- test.log.error (emits)

Dependencies: Steps 101-133 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO


# ============================================================================
# Constants
# ============================================================================

class LogLevel(Enum):
    """Log levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_string(cls, level: str) -> 'LogLevel':
        """Convert string to LogLevel."""
        return cls[level.upper()]


class LogDestination(Enum):
    """Log output destinations."""
    CONSOLE = "console"
    FILE = "file"
    JSON_FILE = "json_file"
    BUS = "bus"
    SYSLOG = "syslog"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class LogEntry:
    """
    A structured log entry.

    Attributes:
        level: Log level
        message: Log message
        timestamp: Entry timestamp
        logger_name: Logger name
        context: Additional context
        exception: Exception information
        trace_id: Trace ID for correlation
    """
    level: LogLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    logger_name: str = "test-agent"
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    source_location: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "logger": self.logger_name,
            "context": self.context,
            "exception": self.exception,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "source": self.source_location,
        }

    def format_text(self, include_timestamp: bool = True, colorize: bool = False) -> str:
        """Format as text."""
        parts = []

        if include_timestamp:
            dt = datetime.fromtimestamp(self.timestamp)
            parts.append(f"[{dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")

        level_str = f"[{self.level.name:8}]"
        if colorize:
            level_str = self._colorize_level(level_str)
        parts.append(level_str)

        parts.append(f"[{self.logger_name}]")
        parts.append(self.message)

        if self.context:
            context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"({context_str})")

        if self.exception:
            parts.append(f"\n  Exception: {self.exception.get('type', 'Unknown')}: {self.exception.get('message', '')}")
            if self.exception.get('traceback'):
                parts.append(f"\n{self.exception['traceback']}")

        return " ".join(parts)

    def _colorize_level(self, text: str) -> str:
        """Add ANSI color to level."""
        colors = {
            LogLevel.DEBUG: "\033[36m",     # Cyan
            LogLevel.INFO: "\033[32m",      # Green
            LogLevel.WARNING: "\033[33m",   # Yellow
            LogLevel.ERROR: "\033[31m",     # Red
            LogLevel.CRITICAL: "\033[35m",  # Magenta
        }
        reset = "\033[0m"
        return f"{colors.get(self.level, '')}{text}{reset}"


class LogFormatter:
    """Log entry formatter."""

    def __init__(
        self,
        format_type: str = "text",
        include_timestamp: bool = True,
        colorize: bool = False,
        include_context: bool = True,
    ):
        self.format_type = format_type
        self.include_timestamp = include_timestamp
        self.colorize = colorize
        self.include_context = include_context

    def format(self, entry: LogEntry) -> str:
        """Format a log entry."""
        if self.format_type == "json":
            return json.dumps(entry.to_dict())
        else:
            return entry.format_text(
                include_timestamp=self.include_timestamp,
                colorize=self.colorize,
            )


@dataclass
class LogConfig:
    """
    Configuration for the logging system.

    Attributes:
        level: Minimum log level
        destinations: Output destinations
        format_type: Log format (text, json)
        output_dir: Directory for log files
        max_file_size_mb: Max log file size
        max_files: Max number of log files
        include_context: Include context in logs
        colorize: Enable ANSI colors
    """
    level: LogLevel = LogLevel.INFO
    destinations: List[LogDestination] = field(default_factory=lambda: [
        LogDestination.CONSOLE,
        LogDestination.FILE,
    ])
    format_type: str = "text"
    output_dir: str = ".pluribus/test-agent/logs"
    log_file: str = "test-agent.log"
    max_file_size_mb: int = 10
    max_files: int = 5
    include_context: bool = True
    colorize: bool = True
    async_logging: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.name,
            "destinations": [d.value for d in self.destinations],
            "format_type": self.format_type,
            "output_dir": self.output_dir,
            "max_file_size_mb": self.max_file_size_mb,
            "colorize": self.colorize,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class LogBus:
    """Bus interface for logging with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Log Handlers
# ============================================================================

class ConsoleHandler:
    """Console log handler."""

    def __init__(self, formatter: LogFormatter, stream: TextIO = None):
        self.formatter = formatter
        self.stream = stream or sys.stderr

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to console."""
        try:
            message = self.formatter.format(entry)
            self.stream.write(message + "\n")
            self.stream.flush()
        except Exception:
            pass


class FileHandler:
    """File log handler with rotation."""

    def __init__(
        self,
        formatter: LogFormatter,
        file_path: Path,
        max_size_bytes: int = 10 * 1024 * 1024,
        max_files: int = 5,
    ):
        self.formatter = formatter
        self.file_path = file_path
        self.max_size_bytes = max_size_bytes
        self.max_files = max_files
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to file."""
        try:
            self._maybe_rotate()

            message = self.formatter.format(entry)

            with open(self.file_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(message + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception:
            pass

    def _maybe_rotate(self) -> None:
        """Rotate log file if needed."""
        if not self.file_path.exists():
            return

        if self.file_path.stat().st_size < self.max_size_bytes:
            return

        # Rotate files
        for i in range(self.max_files - 1, 0, -1):
            src = self.file_path.with_suffix(f".{i}")
            dst = self.file_path.with_suffix(f".{i + 1}")
            if src.exists():
                src.rename(dst)

        # Rename current file
        self.file_path.rename(self.file_path.with_suffix(".1"))


class JsonFileHandler:
    """JSON file log handler (NDJSON format)."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry as JSON."""
        try:
            with open(self.file_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry.to_dict()) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass


class BusHandler:
    """Bus log handler."""

    def __init__(self, bus: LogBus):
        self.bus = bus

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to bus."""
        topic = "test.log.error" if entry.level.value >= LogLevel.ERROR.value else "test.log.entry"
        self.bus.emit({
            "topic": topic,
            "kind": "log",
            "actor": entry.logger_name,
            "data": entry.to_dict(),
        })


# ============================================================================
# Test Logger
# ============================================================================

class TestLogger:
    """
    Structured logging system for the Test Agent.

    Features:
    - Multiple log levels
    - Structured context
    - Multiple output destinations
    - Log rotation
    - Trace correlation
    - Exception handling

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.log.entry, test.log.error
    """

    def __init__(self, name: str = "test-agent", bus=None, config: Optional[LogConfig] = None):
        """
        Initialize the logger.

        Args:
            name: Logger name
            bus: Optional bus instance
            config: Logger configuration
        """
        self.name = name
        self.bus = bus or LogBus()
        self.config = config or LogConfig()
        self._handlers: List[Any] = []
        self._trace_id: Optional[str] = None
        self._span_id: Optional[str] = None
        self._default_context: Dict[str, Any] = {}

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up log handlers."""
        formatter = LogFormatter(
            format_type=self.config.format_type,
            colorize=self.config.colorize,
            include_context=self.config.include_context,
        )

        for dest in self.config.destinations:
            if dest == LogDestination.CONSOLE:
                self._handlers.append(ConsoleHandler(formatter))

            elif dest == LogDestination.FILE:
                file_path = Path(self.config.output_dir) / self.config.log_file
                self._handlers.append(FileHandler(
                    formatter,
                    file_path,
                    max_size_bytes=self.config.max_file_size_mb * 1024 * 1024,
                    max_files=self.config.max_files,
                ))

            elif dest == LogDestination.JSON_FILE:
                file_path = Path(self.config.output_dir) / "logs.ndjson"
                self._handlers.append(JsonFileHandler(file_path))

            elif dest == LogDestination.BUS:
                self._handlers.append(BusHandler(self.bus))

    def log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
    ) -> LogEntry:
        """
        Log a message.

        Args:
            level: Log level
            message: Log message
            context: Additional context
            exception: Exception to log

        Returns:
            Created log entry
        """
        if level.value < self.config.level.value:
            return None

        # Build context
        full_context = {**self._default_context}
        if context:
            full_context.update(context)

        # Build exception info
        exc_info = None
        if exception:
            exc_info = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }

        # Get source location
        source_location = None
        try:
            frame = sys._getframe(2)
            source_location = {
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
            }
        except Exception:
            pass

        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            context=full_context,
            exception=exc_info,
            trace_id=self._trace_id,
            span_id=self._span_id,
            source_location=source_location,
        )

        # Emit to handlers
        for handler in self._handlers:
            try:
                handler.emit(entry)
            except Exception:
                pass

        return entry

    def debug(self, message: str, **context) -> LogEntry:
        """Log debug message."""
        return self.log(LogLevel.DEBUG, message, context)

    def info(self, message: str, **context) -> LogEntry:
        """Log info message."""
        return self.log(LogLevel.INFO, message, context)

    def warning(self, message: str, **context) -> LogEntry:
        """Log warning message."""
        return self.log(LogLevel.WARNING, message, context)

    def error(self, message: str, exception: Optional[Exception] = None, **context) -> LogEntry:
        """Log error message."""
        return self.log(LogLevel.ERROR, message, context, exception)

    def critical(self, message: str, exception: Optional[Exception] = None, **context) -> LogEntry:
        """Log critical message."""
        return self.log(LogLevel.CRITICAL, message, context, exception)

    def exception(self, message: str, **context) -> LogEntry:
        """Log exception with traceback."""
        exc_info = sys.exc_info()
        exc = exc_info[1] if exc_info[1] else None
        return self.log(LogLevel.ERROR, message, context, exc)

    def set_context(self, **context) -> None:
        """Set default context for all logs."""
        self._default_context.update(context)

    def clear_context(self) -> None:
        """Clear default context."""
        self._default_context.clear()

    def set_trace(self, trace_id: str, span_id: Optional[str] = None) -> None:
        """Set trace and span IDs for correlation."""
        self._trace_id = trace_id
        self._span_id = span_id

    def clear_trace(self) -> None:
        """Clear trace information."""
        self._trace_id = None
        self._span_id = None

    def child(self, name: str) -> 'TestLogger':
        """Create a child logger with inherited context."""
        child_name = f"{self.name}.{name}"
        child_logger = TestLogger(child_name, self.bus, self.config)
        child_logger._default_context = self._default_context.copy()
        child_logger._trace_id = self._trace_id
        child_logger._span_id = self._span_id
        return child_logger

    def timed(self, message: str, **context) -> 'LogTimer':
        """Create a timed logging context."""
        return LogTimer(self, message, context)

    def get_child(self, name: str) -> 'TestLogger':
        """Alias for child()."""
        return self.child(name)

    async def log_async(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
    ) -> LogEntry:
        """Async version of log."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.log, level, message, context, exception)


class LogTimer:
    """Context manager for timed logging."""

    def __init__(self, logger: TestLogger, message: str, context: Dict[str, Any]):
        self.logger = logger
        self.message = message
        self.context = context
        self.start_time: Optional[float] = None

    def __enter__(self) -> 'LogTimer':
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.message}", **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = time.time() - (self.start_time or 0)
        if exc_type:
            self.logger.error(
                f"Failed: {self.message}",
                duration_s=duration,
                exception=exc_val,
                **self.context,
            )
        else:
            self.logger.info(
                f"Completed: {self.message}",
                duration_s=duration,
                **self.context,
            )


# ============================================================================
# Global Logger
# ============================================================================

_global_logger: Optional[TestLogger] = None


def get_logger(name: str = "test-agent") -> TestLogger:
    """Get or create a global logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = TestLogger(name)
    return _global_logger


def configure_logging(config: LogConfig) -> TestLogger:
    """Configure global logging."""
    global _global_logger
    _global_logger = TestLogger(config=config)
    return _global_logger


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Logging."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Logging")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Log command
    log_parser = subparsers.add_parser("log", help="Log a message")
    log_parser.add_argument("message", help="Log message")
    log_parser.add_argument("--level", choices=["debug", "info", "warning", "error", "critical"],
                            default="info")
    log_parser.add_argument("--context", type=json.loads, default={})

    # View command
    view_parser = subparsers.add_parser("view", help="View log file")
    view_parser.add_argument("--lines", "-n", type=int, default=20, help="Number of lines")
    view_parser.add_argument("--level", choices=["debug", "info", "warning", "error", "critical"])
    view_parser.add_argument("--follow", "-f", action="store_true", help="Follow log file")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear log file")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/logs")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")

    args = parser.parse_args()

    config = LogConfig(
        output_dir=args.output,
        colorize=not args.no_color,
    )
    logger = TestLogger(config=config)

    if args.command == "log":
        level = LogLevel.from_string(args.level)
        entry = logger.log(level, args.message, args.context)

        if args.json:
            print(json.dumps(entry.to_dict(), indent=2))
        else:
            print("Logged successfully")

    elif args.command == "view":
        log_file = Path(args.output) / "test-agent.log"
        if not log_file.exists():
            print("No log file found")
            return

        if args.follow:
            import subprocess
            subprocess.run(["tail", "-f", str(log_file)])
        else:
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines[-args.lines:]:
                    if args.level:
                        if f"[{args.level.upper()}" in line:
                            print(line.rstrip())
                    else:
                        print(line.rstrip())

    elif args.command == "clear":
        log_file = Path(args.output) / "test-agent.log"
        if log_file.exists():
            log_file.unlink()
            print("Log file cleared")
        else:
            print("No log file to clear")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
