#!/usr/bin/env python3
"""
logging_system.py - Structured Logging System (Step 84)

PBTSO Phase: All Phases

Provides:
- Structured JSON logging
- Log levels and filtering
- Multiple output handlers (console, file, bus)
- Context propagation
- Log rotation and retention

Bus Topics:
- code.log.entry
- code.log.error
- code.log.audit

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from threading import Lock, local
from typing import Any, Callable, Dict, Generator, List, Optional, TextIO, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class LogLevel(IntEnum):
    """Log level severity."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: LogLevel
    message: str
    logger_name: str
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    host: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "level": self.level.name,
            "level_value": int(self.level),
            "message": self.message,
            "logger": self.logger_name,
            "context": self.context,
            "exception": self.exception,
            "stack_trace": self.stack_trace,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "host": self.host,
            "pid": self.pid,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


@dataclass
class LogConfig:
    """Configuration for logging system."""
    level: LogLevel = LogLevel.INFO
    format: str = "json"  # json, text
    output: str = "console"  # console, file, bus, all
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    max_files: int = 5
    enable_colors: bool = True
    include_caller: bool = False
    include_trace: bool = True
    bus_enabled: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "format": self.format,
            "output": self.output,
            "file_path": self.file_path,
            "max_file_size_mb": self.max_file_size_mb,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Log Formatters
# =============================================================================

class LogFormatter(ABC):
    """Abstract base class for log formatters."""

    @abstractmethod
    def format(self, entry: LogEntry) -> str:
        """Format a log entry."""
        pass


class JsonFormatter(LogFormatter):
    """Format logs as JSON."""

    def format(self, entry: LogEntry) -> str:
        return entry.to_json()


class TextFormatter(LogFormatter):
    """Format logs as readable text."""

    LEVEL_COLORS = {
        LogLevel.TRACE: "\033[37m",     # White
        LogLevel.DEBUG: "\033[36m",     # Cyan
        LogLevel.INFO: "\033[32m",      # Green
        LogLevel.WARNING: "\033[33m",   # Yellow
        LogLevel.ERROR: "\033[31m",     # Red
        LogLevel.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors

    def format(self, entry: LogEntry) -> str:
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = entry.level.name.ljust(8)

        if self.use_colors:
            color = self.LEVEL_COLORS.get(entry.level, "")
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


# =============================================================================
# Log Handlers
# =============================================================================

class LogHandler(ABC):
    """Abstract base class for log handlers."""

    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self._lock = Lock()

    def should_handle(self, entry: LogEntry) -> bool:
        """Check if entry should be handled."""
        return entry.level >= self.level

    @abstractmethod
    def handle(self, entry: LogEntry) -> None:
        """Handle a log entry."""
        pass

    def close(self) -> None:
        """Close the handler."""
        pass


class ConsoleHandler(LogHandler):
    """Log to console (stdout/stderr)."""

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        formatter: Optional[LogFormatter] = None,
        stream: Optional[TextIO] = None,
        use_stderr_for_errors: bool = True,
    ):
        super().__init__(level)
        self.formatter = formatter or TextFormatter()
        self.stream = stream or sys.stdout
        self.use_stderr_for_errors = use_stderr_for_errors

    def handle(self, entry: LogEntry) -> None:
        if not self.should_handle(entry):
            return

        formatted = self.formatter.format(entry)
        stream = sys.stderr if (
            self.use_stderr_for_errors and entry.level >= LogLevel.ERROR
        ) else self.stream

        with self._lock:
            print(formatted, file=stream)


class FileHandler(LogHandler):
    """Log to file with rotation."""

    def __init__(
        self,
        file_path: str,
        level: LogLevel = LogLevel.INFO,
        formatter: Optional[LogFormatter] = None,
        max_size_mb: int = 100,
        max_files: int = 5,
    ):
        super().__init__(level)
        self.file_path = Path(file_path)
        self.formatter = formatter or JsonFormatter()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_files = max_files
        self._current_size = 0

        self._ensure_dir()
        self._check_rotation()

    def _ensure_dir(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _check_rotation(self) -> None:
        """Check if log rotation is needed."""
        if self.file_path.exists():
            self._current_size = self.file_path.stat().st_size
            if self._current_size >= self.max_size_bytes:
                self._rotate()

    def _rotate(self) -> None:
        """Rotate log files."""
        # Remove oldest if at max
        oldest = self.file_path.with_suffix(f".{self.max_files}.log")
        if oldest.exists():
            oldest.unlink()

        # Shift existing files
        for i in range(self.max_files - 1, 0, -1):
            old = self.file_path.with_suffix(f".{i}.log")
            new = self.file_path.with_suffix(f".{i + 1}.log")
            if old.exists():
                old.rename(new)

        # Rotate current file
        if self.file_path.exists():
            self.file_path.rename(self.file_path.with_suffix(".1.log"))

        self._current_size = 0

    def handle(self, entry: LogEntry) -> None:
        if not self.should_handle(entry):
            return

        formatted = self.formatter.format(entry) + "\n"
        data = formatted.encode("utf-8")

        with self._lock:
            # Check rotation
            if self._current_size + len(data) >= self.max_size_bytes:
                self._rotate()

            # Write with file locking
            fd = os.open(str(self.file_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, data)
                self._current_size += len(data)
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


class BusHandler(LogHandler):
    """Log to agent bus."""

    BUS_TOPICS = {
        "entry": "code.log.entry",
        "error": "code.log.error",
        "audit": "code.log.audit",
    }

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        bus: Optional[LockedAgentBus] = None,
    ):
        super().__init__(level)
        self.bus = bus or LockedAgentBus()

    def handle(self, entry: LogEntry) -> None:
        if not self.should_handle(entry):
            return

        topic = self.BUS_TOPICS["error"] if entry.level >= LogLevel.ERROR else self.BUS_TOPICS["entry"]

        self.bus.emit({
            "topic": topic,
            "kind": "log",
            "level": entry.level.name.lower(),
            "actor": entry.logger_name,
            "data": entry.to_dict(),
        })


# =============================================================================
# Context Manager
# =============================================================================

_context = local()


def get_context() -> Dict[str, Any]:
    """Get current logging context."""
    if not hasattr(_context, "data"):
        _context.data = {}
    return _context.data


def set_context(**kwargs: Any) -> None:
    """Set logging context values."""
    ctx = get_context()
    ctx.update(kwargs)


def clear_context() -> None:
    """Clear logging context."""
    _context.data = {}


@contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager for logging context."""
    ctx = get_context()
    old_values = {k: ctx.get(k) for k in kwargs}

    ctx.update(kwargs)
    try:
        yield
    finally:
        for k, v in old_values.items():
            if v is None:
                ctx.pop(k, None)
            else:
                ctx[k] = v


# =============================================================================
# Logger
# =============================================================================

class Logger:
    """
    Logger instance for a specific name/component.

    Usage:
        logger = Logger("my-component")
        logger.info("Message", key="value")
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[LogHandler]] = None,
    ):
        self.name = name
        self.level = level
        self.handlers = handlers or []
        self._trace_id: Optional[str] = None
        self._span_id: Optional[str] = None

    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level."""
        self.level = level

    def add_handler(self, handler: LogHandler) -> None:
        """Add a handler."""
        self.handlers.append(handler)

    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)

    def _log(
        self,
        level: LogLevel,
        message: str,
        exc_info: bool = False,
        **context: Any,
    ) -> None:
        """Internal log method."""
        if level < self.level:
            return

        # Build entry
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name=self.name,
            context={**get_context(), **context},
            trace_id=self._trace_id,
            span_id=self._span_id,
        )

        # Add exception info
        if exc_info:
            exc = sys.exc_info()
            if exc[1]:
                entry.exception = str(exc[1])
                entry.stack_trace = "".join(traceback.format_exception(*exc))

        # Send to handlers
        for handler in self.handlers:
            try:
                handler.handle(entry)
            except Exception:
                pass

    def trace(self, message: str, **context: Any) -> None:
        """Log at TRACE level."""
        self._log(LogLevel.TRACE, message, **context)

    def debug(self, message: str, **context: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **context)

    def error(self, message: str, exc_info: bool = False, **context: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **context)

    def critical(self, message: str, exc_info: bool = False, **context: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, exc_info=exc_info, **context)

    def exception(self, message: str, **context: Any) -> None:
        """Log exception with stack trace."""
        self._log(LogLevel.ERROR, message, exc_info=True, **context)

    @contextmanager
    def span(self, name: str, **context: Any) -> Generator["Logger", None, None]:
        """Create a span for tracing."""
        old_span = self._span_id
        self._span_id = f"{name}-{uuid.uuid4().hex[:8]}"

        self.debug(f"Span start: {name}", span=self._span_id, **context)
        start = time.time()

        try:
            yield self
        finally:
            duration = time.time() - start
            self.debug(f"Span end: {name}", span=self._span_id, duration_ms=duration * 1000)
            self._span_id = old_span


# =============================================================================
# Structured Logger (Main Class)
# =============================================================================

class StructuredLogger:
    """
    Main structured logging system.

    PBTSO Phase: All Phases

    Features:
    - Structured JSON logging
    - Multiple handlers (console, file, bus)
    - Context propagation
    - Log levels
    - Log rotation

    Usage:
        logging = StructuredLogger(config)
        logger = logging.get_logger("my-component")
        logger.info("Message", key="value")
    """

    def __init__(
        self,
        config: Optional[LogConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or LogConfig()
        self.bus = bus or LockedAgentBus()
        self._loggers: Dict[str, Logger] = {}
        self._handlers: List[LogHandler] = []

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup handlers based on config."""
        formatter: LogFormatter

        if self.config.format == "json":
            formatter = JsonFormatter()
        else:
            formatter = TextFormatter(use_colors=self.config.enable_colors)

        # Console handler
        if self.config.output in ("console", "all"):
            self._handlers.append(ConsoleHandler(
                level=self.config.level,
                formatter=formatter,
            ))

        # File handler
        if self.config.output in ("file", "all") and self.config.file_path:
            self._handlers.append(FileHandler(
                file_path=self.config.file_path,
                level=self.config.level,
                formatter=JsonFormatter(),
                max_size_mb=self.config.max_file_size_mb,
                max_files=self.config.max_files,
            ))

        # Bus handler
        if self.config.bus_enabled and self.config.output in ("bus", "all"):
            self._handlers.append(BusHandler(
                level=self.config.level,
                bus=self.bus,
            ))

    def get_logger(self, name: str) -> Logger:
        """Get or create a logger for a component."""
        if name not in self._loggers:
            logger = Logger(name, self.config.level, self._handlers.copy())
            self._loggers[name] = logger
        return self._loggers[name]

    def set_level(self, level: LogLevel) -> None:
        """Set global log level."""
        self.config.level = level
        for logger in self._loggers.values():
            logger.set_level(level)

    def add_handler(self, handler: LogHandler) -> None:
        """Add a handler to all loggers."""
        self._handlers.append(handler)
        for logger in self._loggers.values():
            logger.add_handler(handler)

    def close(self) -> None:
        """Close all handlers."""
        for handler in self._handlers:
            handler.close()


# =============================================================================
# Global Logger
# =============================================================================

_global_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "code-agent") -> Logger:
    """Get a logger from the global logging system."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger()
    return _global_logger.get_logger(name)


def configure_logging(config: LogConfig) -> StructuredLogger:
    """Configure global logging system."""
    global _global_logger
    _global_logger = StructuredLogger(config)
    return _global_logger


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Logging System."""
    import argparse

    parser = argparse.ArgumentParser(description="Logging System (Step 84)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test command
    test_parser = subparsers.add_parser("test", help="Test logging")
    test_parser.add_argument("--level", "-l", default="info", choices=["trace", "debug", "info", "warning", "error"])
    test_parser.add_argument("--format", "-f", default="text", choices=["text", "json"])

    # demo command
    subparsers.add_parser("demo", help="Run logging demo")

    args = parser.parse_args()

    level_map = {
        "trace": LogLevel.TRACE,
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
    }

    if args.command == "test":
        config = LogConfig(
            level=level_map[args.level],
            format=args.format,
            output="console",
        )
        logging = StructuredLogger(config)
        logger = logging.get_logger("test")

        logger.trace("This is a trace message")
        logger.debug("This is a debug message")
        logger.info("This is an info message", key="value")
        logger.warning("This is a warning message")
        logger.error("This is an error message")

        return 0

    elif args.command == "demo":
        config = LogConfig(
            level=LogLevel.DEBUG,
            format="text",
            output="console",
        )
        logging = StructuredLogger(config)
        logger = logging.get_logger("demo")

        print("Running logging demo...\n")

        logger.info("Application started")

        with log_context(request_id="req-123", user="alice"):
            logger.info("Processing request")

            with logger.span("database_query"):
                time.sleep(0.1)
                logger.debug("Executing query", table="users")

            logger.info("Request completed", status=200)

        try:
            raise ValueError("Something went wrong")
        except Exception:
            logger.exception("Error occurred")

        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
