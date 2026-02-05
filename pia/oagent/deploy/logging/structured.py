#!/usr/bin/env python3
"""
structured.py - Deploy Logging System (Step 234)

PBTSO Phase: VERIFY, ITERATE
A2A Integration: Structured logging for deployments via deploy.logging.*

Provides:
- LogLevel: Log severity levels
- LogFormat: Output format types
- LogContext: Contextual logging data
- LogEntry: Structured log entry
- LogSink: Log output destination
- StructuredLogger: Logger implementation
- DeployLoggingSystem: Main logging system

Bus Topics:
- deploy.logging.write
- deploy.logging.error
- deploy.logging.audit

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import abc
import asyncio
import fcntl
import json
import logging
import os
import socket
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union


# ==============================================================================
# Bus Emission Helper with File Locking (DKIN v30)
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
    actor: str = "logging-system"
) -> str:
    """Emit an event to the Pluribus bus with fcntl.flock() file locking."""
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
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Context Variables for Request Tracing
# ==============================================================================

_log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class LogLevel(IntEnum):
    """Log severity levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AUDIT = 60  # Special level for audit logs


class LogFormat(Enum):
    """Output format types."""
    JSON = "json"
    TEXT = "text"
    CONSOLE = "console"  # Colored console output
    NDJSON = "ndjson"  # Newline delimited JSON


class LogCategory(Enum):
    """Log categories."""
    DEPLOYMENT = "deployment"
    BUILD = "build"
    HEALTH = "health"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    ERROR = "error"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class LogContext:
    """
    Contextual logging data.

    Attributes:
        trace_id: Distributed trace ID
        span_id: Current span ID
        deployment_id: Associated deployment
        service_name: Service name
        environment: Target environment
        user_id: User/actor identifier
        custom: Custom context fields
    """
    trace_id: str = ""
    span_id: str = ""
    deployment_id: str = ""
    service_name: str = ""
    environment: str = ""
    user_id: str = ""
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "deployment_id": self.deployment_id,
            "service_name": self.service_name,
            "environment": self.environment,
            "user_id": self.user_id,
        }
        result.update(self.custom)
        return {k: v for k, v in result.items() if v}

    def with_span(self, span_id: str) -> "LogContext":
        """Create new context with different span."""
        return LogContext(
            trace_id=self.trace_id,
            span_id=span_id,
            deployment_id=self.deployment_id,
            service_name=self.service_name,
            environment=self.environment,
            user_id=self.user_id,
            custom=dict(self.custom),
        )


@dataclass
class LogEntry:
    """
    Structured log entry.

    Attributes:
        entry_id: Unique entry identifier
        timestamp: Log timestamp
        level: Log level
        category: Log category
        message: Log message
        logger_name: Logger name
        context: Log context
        data: Additional structured data
        exception: Exception information
        source: Source file/line info
    """
    entry_id: str
    timestamp: float
    level: LogLevel
    category: LogCategory
    message: str
    logger_name: str = ""
    context: Optional[LogContext] = None
    data: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    source: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.entry_id,
            "ts": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat() + "Z",
            "level": self.level.name.lower(),
            "level_num": int(self.level),
            "category": self.category.value,
            "message": self.message,
            "logger": self.logger_name,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }

        if self.context:
            result["context"] = self.context.to_dict()

        if self.data:
            result["data"] = self.data

        if self.exception:
            result["exception"] = self.exception

        if self.source:
            result["source"] = self.source

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def to_text(self) -> str:
        """Convert to text format."""
        iso = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        text = f"{iso} [{self.level.name:8}] [{self.category.value}] {self.message}"

        if self.context and self.context.trace_id:
            text += f" trace={self.context.trace_id}"

        if self.exception:
            text += f"\n{self.exception}"

        return text


# ==============================================================================
# Log Sinks
# ==============================================================================

class LogSink(abc.ABC):
    """Abstract base class for log sinks."""

    @abc.abstractmethod
    def write(self, entry: LogEntry) -> None:
        """Write a log entry."""
        pass

    def flush(self) -> None:
        """Flush pending writes."""
        pass

    def close(self) -> None:
        """Close the sink."""
        pass


class FileSink(LogSink):
    """File-based log sink with rotation."""

    def __init__(
        self,
        path: str,
        format: LogFormat = LogFormat.NDJSON,
        max_size_mb: int = 100,
        max_files: int = 10,
    ):
        self.path = Path(path)
        self.format = format
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_files = max_files
        self._file: Optional[TextIO] = None
        self._current_size = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open_file()

    def _open_file(self) -> None:
        """Open the log file."""
        if self.path.exists():
            self._current_size = self.path.stat().st_size
        else:
            self._current_size = 0

        self._file = open(self.path, "a")

    def _rotate_if_needed(self) -> None:
        """Rotate log file if size exceeds limit."""
        if self._current_size < self.max_size_bytes:
            return

        if self._file:
            self._file.close()

        # Rotate existing files
        for i in range(self.max_files - 1, 0, -1):
            old_path = self.path.with_suffix(f"{self.path.suffix}.{i}")
            new_path = self.path.with_suffix(f"{self.path.suffix}.{i + 1}")
            if old_path.exists():
                if new_path.exists():
                    new_path.unlink()
                old_path.rename(new_path)

        # Rename current to .1
        backup_path = self.path.with_suffix(f"{self.path.suffix}.1")
        if self.path.exists():
            self.path.rename(backup_path)

        # Delete oldest if exceeds max_files
        oldest = self.path.with_suffix(f"{self.path.suffix}.{self.max_files}")
        if oldest.exists():
            oldest.unlink()

        self._open_file()

    def write(self, entry: LogEntry) -> None:
        """Write a log entry."""
        if not self._file:
            return

        self._rotate_if_needed()

        if self.format == LogFormat.NDJSON or self.format == LogFormat.JSON:
            line = entry.to_json() + "\n"
        else:
            line = entry.to_text() + "\n"

        fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        try:
            self._file.write(line)
            self._current_size += len(line.encode())
        finally:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)

    def flush(self) -> None:
        """Flush pending writes."""
        if self._file:
            self._file.flush()

    def close(self) -> None:
        """Close the sink."""
        if self._file:
            self._file.close()
            self._file = None


class ConsoleSink(LogSink):
    """Console log sink with optional colors."""

    COLORS = {
        LogLevel.TRACE: "\033[90m",    # Gray
        LogLevel.DEBUG: "\033[36m",    # Cyan
        LogLevel.INFO: "\033[32m",     # Green
        LogLevel.WARNING: "\033[33m",  # Yellow
        LogLevel.ERROR: "\033[31m",    # Red
        LogLevel.CRITICAL: "\033[35m", # Magenta
        LogLevel.AUDIT: "\033[34m",    # Blue
    }
    RESET = "\033[0m"

    def __init__(
        self,
        stream: TextIO = sys.stderr,
        format: LogFormat = LogFormat.CONSOLE,
        use_colors: bool = True,
    ):
        self.stream = stream
        self.format = format
        self.use_colors = use_colors and stream.isatty()

    def write(self, entry: LogEntry) -> None:
        """Write a log entry."""
        if self.format == LogFormat.JSON or self.format == LogFormat.NDJSON:
            line = entry.to_json()
        else:
            line = self._format_console(entry)

        self.stream.write(line + "\n")

    def _format_console(self, entry: LogEntry) -> str:
        """Format for console output."""
        ts = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        level = entry.level.name[:4]

        if self.use_colors:
            color = self.COLORS.get(entry.level, "")
            level_str = f"{color}{level:4}{self.RESET}"
        else:
            level_str = f"{level:4}"

        line = f"{ts} {level_str} | {entry.message}"

        if entry.context and entry.context.deployment_id:
            line += f" [deploy:{entry.context.deployment_id[:8]}]"

        if entry.data:
            data_str = " ".join(f"{k}={v}" for k, v in entry.data.items())
            line += f" | {data_str}"

        if entry.exception:
            line += f"\n{entry.exception}"

        return line

    def flush(self) -> None:
        """Flush pending writes."""
        self.stream.flush()


class CallbackSink(LogSink):
    """Sink that calls a callback function."""

    def __init__(self, callback: Callable[[LogEntry], None]):
        self.callback = callback

    def write(self, entry: LogEntry) -> None:
        """Write a log entry."""
        self.callback(entry)


class BufferSink(LogSink):
    """Buffer sink for batching writes."""

    def __init__(
        self,
        target_sink: LogSink,
        buffer_size: int = 100,
        flush_interval_s: float = 5.0,
    ):
        self.target_sink = target_sink
        self.buffer_size = buffer_size
        self.flush_interval_s = flush_interval_s
        self._buffer: List[LogEntry] = []
        self._last_flush = time.time()

    def write(self, entry: LogEntry) -> None:
        """Write a log entry."""
        self._buffer.append(entry)

        if len(self._buffer) >= self.buffer_size:
            self.flush()
        elif time.time() - self._last_flush >= self.flush_interval_s:
            self.flush()

    def flush(self) -> None:
        """Flush pending writes."""
        for entry in self._buffer:
            self.target_sink.write(entry)
        self.target_sink.flush()
        self._buffer.clear()
        self._last_flush = time.time()

    def close(self) -> None:
        """Close the sink."""
        self.flush()
        self.target_sink.close()


# ==============================================================================
# Structured Logger
# ==============================================================================

class StructuredLogger:
    """
    Structured logger implementation.

    Provides structured logging with context, levels, and categories.
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        category: LogCategory = LogCategory.CUSTOM,
        sinks: Optional[List[LogSink]] = None,
        context: Optional[LogContext] = None,
    ):
        self.name = name
        self.level = level
        self.category = category
        self.sinks = sinks or []
        self._context = context

    def with_context(self, context: LogContext) -> "StructuredLogger":
        """Create logger with specific context."""
        return StructuredLogger(
            name=self.name,
            level=self.level,
            category=self.category,
            sinks=self.sinks,
            context=context,
        )

    def with_category(self, category: LogCategory) -> "StructuredLogger":
        """Create logger with specific category."""
        return StructuredLogger(
            name=self.name,
            level=self.level,
            category=category,
            sinks=self.sinks,
            context=self._context,
        )

    def _get_context(self) -> LogContext:
        """Get current logging context."""
        if self._context:
            return self._context

        # Get from context var
        ctx_data = _log_context.get()
        if ctx_data:
            return LogContext(**ctx_data)

        return LogContext()

    def _log(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exception: Optional[BaseException] = None,
        category: Optional[LogCategory] = None,
    ) -> Optional[LogEntry]:
        """Internal log method."""
        if level < self.level:
            return None

        entry = LogEntry(
            entry_id=f"log-{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            level=level,
            category=category or self.category,
            message=message,
            logger_name=self.name,
            context=self._get_context(),
            data=data or {},
            exception=traceback.format_exc() if exception else None,
            source=self._get_source_info(),
        )

        for sink in self.sinks:
            try:
                sink.write(entry)
            except Exception:
                pass

        return entry

    def _get_source_info(self) -> Dict[str, Any]:
        """Get source file/line information."""
        try:
            frame = sys._getframe(3)
            return {
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
            }
        except Exception:
            return {}

    def trace(self, message: str, **data) -> Optional[LogEntry]:
        """Log trace message."""
        return self._log(LogLevel.TRACE, message, data)

    def debug(self, message: str, **data) -> Optional[LogEntry]:
        """Log debug message."""
        return self._log(LogLevel.DEBUG, message, data)

    def info(self, message: str, **data) -> Optional[LogEntry]:
        """Log info message."""
        return self._log(LogLevel.INFO, message, data)

    def warning(self, message: str, **data) -> Optional[LogEntry]:
        """Log warning message."""
        return self._log(LogLevel.WARNING, message, data)

    def error(self, message: str, exception: Optional[BaseException] = None, **data) -> Optional[LogEntry]:
        """Log error message."""
        return self._log(LogLevel.ERROR, message, data, exception)

    def critical(self, message: str, exception: Optional[BaseException] = None, **data) -> Optional[LogEntry]:
        """Log critical message."""
        return self._log(LogLevel.CRITICAL, message, data, exception)

    def audit(self, message: str, **data) -> Optional[LogEntry]:
        """Log audit message."""
        return self._log(LogLevel.AUDIT, message, data, category=LogCategory.AUDIT)

    def exception(self, message: str, **data) -> Optional[LogEntry]:
        """Log exception with traceback."""
        return self._log(LogLevel.ERROR, message, data, exception=sys.exc_info()[1])


# ==============================================================================
# Deploy Logging System (Step 234)
# ==============================================================================

class DeployLoggingSystem:
    """
    Deploy Logging System - structured logging for deployments.

    PBTSO Phase: VERIFY, ITERATE

    Responsibilities:
    - Provide structured logging across deployment operations
    - Support multiple output sinks (file, console, remote)
    - Maintain logging context (trace IDs, deployment IDs)
    - Support log levels and categories
    - Enable audit logging for compliance

    Example:
        >>> logging_system = DeployLoggingSystem()
        >>> logger = logging_system.get_logger("deploy.orchestrator")
        >>> with logging_system.context(deployment_id="deploy-123"):
        ...     logger.info("Starting deployment", version="v1.0.0")
        >>> logger.audit("Deployment approved", user="admin")
    """

    BUS_TOPICS = {
        "write": "deploy.logging.write",
        "error": "deploy.logging.error",
        "audit": "deploy.logging.audit",
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "logging-system",
        default_level: LogLevel = LogLevel.INFO,
        console_output: bool = True,
        file_output: bool = True,
        bus_output: bool = True,
    ):
        """
        Initialize the logging system.

        Args:
            state_dir: Directory for log files
            actor_id: Actor identifier for bus events
            default_level: Default log level
            console_output: Enable console output
            file_output: Enable file output
            bus_output: Emit logs to bus
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "logs"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.default_level = default_level
        self.bus_output = bus_output

        # Initialize sinks
        self._sinks: List[LogSink] = []
        self._loggers: Dict[str, StructuredLogger] = {}

        if console_output:
            self._sinks.append(ConsoleSink())

        if file_output:
            self._sinks.append(FileSink(
                str(self.state_dir / "deploy.log"),
                format=LogFormat.NDJSON,
            ))

        # Bus sink for important logs
        if bus_output:
            self._sinks.append(CallbackSink(self._emit_to_bus))

    def get_logger(
        self,
        name: str,
        level: Optional[LogLevel] = None,
        category: LogCategory = LogCategory.CUSTOM,
    ) -> StructuredLogger:
        """
        Get or create a logger.

        Args:
            name: Logger name
            level: Log level (defaults to system level)
            category: Log category

        Returns:
            StructuredLogger instance
        """
        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(
                name=name,
                level=level or self.default_level,
                category=category,
                sinks=self._sinks,
            )
        return self._loggers[name]

    def set_level(self, level: LogLevel) -> None:
        """Set default log level."""
        self.default_level = level
        for logger in self._loggers.values():
            logger.level = level

    def add_sink(self, sink: LogSink) -> None:
        """Add a log sink."""
        self._sinks.append(sink)
        for logger in self._loggers.values():
            logger.sinks.append(sink)

    def remove_sink(self, sink: LogSink) -> bool:
        """Remove a log sink."""
        if sink in self._sinks:
            self._sinks.remove(sink)
            for logger in self._loggers.values():
                if sink in logger.sinks:
                    logger.sinks.remove(sink)
            return True
        return False

    class _ContextManager:
        """Context manager for logging context."""

        def __init__(self, logging_system: "DeployLoggingSystem", **kwargs):
            self.logging_system = logging_system
            self.kwargs = kwargs
            self._token = None

        def __enter__(self):
            current = _log_context.get()
            new_context = {**current, **self.kwargs}
            self._token = _log_context.set(new_context)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._token:
                _log_context.reset(self._token)

    def context(
        self,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        service_name: Optional[str] = None,
        environment: Optional[str] = None,
        user_id: Optional[str] = None,
        **custom,
    ) -> _ContextManager:
        """
        Create a logging context.

        Args:
            trace_id: Distributed trace ID
            span_id: Current span ID
            deployment_id: Deployment ID
            service_name: Service name
            environment: Environment
            user_id: User ID
            **custom: Custom context fields

        Returns:
            Context manager

        Example:
            >>> with logging_system.context(deployment_id="deploy-123"):
            ...     logger.info("Inside context")
        """
        kwargs = {}
        if trace_id:
            kwargs["trace_id"] = trace_id
        if span_id:
            kwargs["span_id"] = span_id
        if deployment_id:
            kwargs["deployment_id"] = deployment_id
        if service_name:
            kwargs["service_name"] = service_name
        if environment:
            kwargs["environment"] = environment
        if user_id:
            kwargs["user_id"] = user_id
        kwargs.update(custom)

        return self._ContextManager(self, **kwargs)

    def new_trace(self) -> str:
        """Generate a new trace ID."""
        return uuid.uuid4().hex

    def _emit_to_bus(self, entry: LogEntry) -> None:
        """Emit log entry to bus."""
        if not self.bus_output:
            return

        # Only emit errors and above to bus
        if entry.level < LogLevel.ERROR:
            return

        topic = self.BUS_TOPICS["write"]
        if entry.level >= LogLevel.ERROR:
            topic = self.BUS_TOPICS["error"]
        if entry.category == LogCategory.AUDIT:
            topic = self.BUS_TOPICS["audit"]

        _emit_bus_event(
            topic,
            {
                "entry_id": entry.entry_id,
                "level": entry.level.name.lower(),
                "category": entry.category.value,
                "message": entry.message,
                "logger": entry.logger_name,
                "context": entry.context.to_dict() if entry.context else {},
            },
            level=entry.level.name.lower() if entry.level != LogLevel.AUDIT else "info",
            actor=self.actor_id,
        )

    def flush(self) -> None:
        """Flush all sinks."""
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        """Close all sinks."""
        for sink in self._sinks:
            sink.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==============================================================================
# Integration with Python logging
# ==============================================================================

class StructuredHandler(logging.Handler):
    """Python logging handler that integrates with DeployLoggingSystem."""

    LEVEL_MAP = {
        logging.DEBUG: LogLevel.DEBUG,
        logging.INFO: LogLevel.INFO,
        logging.WARNING: LogLevel.WARNING,
        logging.ERROR: LogLevel.ERROR,
        logging.CRITICAL: LogLevel.CRITICAL,
    }

    def __init__(self, logging_system: DeployLoggingSystem):
        super().__init__()
        self.logging_system = logging_system
        self._logger = logging_system.get_logger("python.logging")

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        level = self.LEVEL_MAP.get(record.levelno, LogLevel.INFO)
        message = self.format(record)

        self._logger._log(
            level=level,
            message=message,
            data={"python_logger": record.name},
            exception=record.exc_info[1] if record.exc_info else None,
        )


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for logging system."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Logging System (Step 234)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # log command
    log_parser = subparsers.add_parser("log", help="Write a log entry")
    log_parser.add_argument("message", help="Log message")
    log_parser.add_argument("--level", "-l", default="info",
                            choices=["trace", "debug", "info", "warning", "error", "critical", "audit"])
    log_parser.add_argument("--category", "-c", default="custom",
                            choices=["deployment", "build", "health", "security", "audit", "error", "system", "custom"])
    log_parser.add_argument("--logger", "-n", default="cli", help="Logger name")
    log_parser.add_argument("--deployment-id", "-d", help="Deployment ID")
    log_parser.add_argument("--service", "-s", help="Service name")
    log_parser.add_argument("--data", help="JSON data")

    # tail command
    tail_parser = subparsers.add_parser("tail", help="Tail log file")
    tail_parser.add_argument("--lines", "-n", type=int, default=20, help="Number of lines")
    tail_parser.add_argument("--level", "-l", help="Filter by level")
    tail_parser.add_argument("--category", "-c", help="Filter by category")
    tail_parser.add_argument("--follow", "-f", action="store_true", help="Follow log file")

    # search command
    search_parser = subparsers.add_parser("search", help="Search logs")
    search_parser.add_argument("pattern", help="Search pattern")
    search_parser.add_argument("--level", "-l", help="Filter by level")
    search_parser.add_argument("--hours", type=int, default=24, help="Time window in hours")
    search_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    logging_system = DeployLoggingSystem(console_output=False)

    if args.command == "log":
        logger = logging_system.get_logger(
            args.logger,
            category=LogCategory(args.category.upper()) if args.category != "custom" else LogCategory.CUSTOM,
        )

        # Set context
        ctx_kwargs = {}
        if args.deployment_id:
            ctx_kwargs["deployment_id"] = args.deployment_id
        if args.service:
            ctx_kwargs["service_name"] = args.service

        data = {}
        if args.data:
            data = json.loads(args.data)

        with logging_system.context(**ctx_kwargs):
            level = LogLevel[args.level.upper()]
            entry = logger._log(level, args.message, data)

        if entry:
            print(entry.to_text())

        logging_system.close()
        return 0

    elif args.command == "tail":
        log_file = logging_system.state_dir / "deploy.log"
        if not log_file.exists():
            print("No log file found")
            return 1

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                tail_lines = lines[-args.lines:]

                for line in tail_lines:
                    try:
                        entry_data = json.loads(line)

                        # Apply filters
                        if args.level and entry_data.get("level") != args.level:
                            continue
                        if args.category and entry_data.get("category") != args.category:
                            continue

                        # Format output
                        ts = entry_data.get("iso", "")[:19]
                        level = entry_data.get("level", "info").upper()[:4]
                        message = entry_data.get("message", "")
                        print(f"{ts} [{level:4}] {message}")

                    except json.JSONDecodeError:
                        print(line.strip())

        except IOError as e:
            print(f"Error reading log file: {e}")
            return 1

        return 0

    elif args.command == "search":
        log_file = logging_system.state_dir / "deploy.log"
        if not log_file.exists():
            print("No log file found")
            return 1

        cutoff = time.time() - (args.hours * 3600)
        matches = []

        try:
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        entry_data = json.loads(line)

                        # Time filter
                        if entry_data.get("ts", 0) < cutoff:
                            continue

                        # Level filter
                        if args.level and entry_data.get("level") != args.level:
                            continue

                        # Pattern filter
                        message = entry_data.get("message", "")
                        if args.pattern.lower() in message.lower():
                            matches.append(entry_data)

                    except json.JSONDecodeError:
                        continue

        except IOError as e:
            print(f"Error reading log file: {e}")
            return 1

        if args.json:
            print(json.dumps(matches, indent=2))
        else:
            for entry in matches:
                ts = entry.get("iso", "")[:19]
                level = entry.get("level", "info").upper()[:4]
                message = entry.get("message", "")
                print(f"{ts} [{level:4}] {message}")

            print(f"\nFound {len(matches)} matching entries")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
