"""
superworker_context.py - Superworker Context Injection

Provides runtime context injection for superworker tasks,
enabling access to shared state, configuration, and services.

Phase 6.1 - App Routing Mesh
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import os


@dataclass
class SuperworkerContext:
    """
    Runtime context for superworker execution.
    Provides access to shared resources and configuration.
    """
    worker_id: str
    task_id: Optional[str] = None
    ring: int = 2
    actor: str = "unknown"
    
    # Runtime state
    env: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    start_time: float = 0.0
    timeout_seconds: int = 300
    
    def __post_init__(self):
        # Load environment
        self.env = {
            "PLURIBUS_BUS_DIR": os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"),
            "PLURIBUS_RING": str(self.ring),
            "PLURIBUS_ACTOR": self.actor,
            "PLURIBUS_WORKER_ID": self.worker_id,
        }
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a registered service by name."""
        return self.services.get(name)
    
    def register_service(self, name: str, service: Any):
        """Register a service for context injection."""
        self.services[name] = service
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value


# Thread-local storage for context
_context_local = threading.local()


def get_current_context() -> Optional[SuperworkerContext]:
    """Get the current superworker context for this thread."""
    return getattr(_context_local, "context", None)


def set_current_context(ctx: SuperworkerContext):
    """Set the current superworker context for this thread."""
    _context_local.context = ctx


def clear_current_context():
    """Clear the current superworker context for this thread."""
    _context_local.context = None


@contextmanager
def superworker_context(
    worker_id: str,
    task_id: Optional[str] = None,
    ring: int = 2,
    actor: str = "unknown",
    **config
):
    """
    Context manager for superworker execution.
    
    Usage:
        with superworker_context("worker-123", task_id="task-456", actor="codex") as ctx:
            # Access ctx.env, ctx.config, ctx.services
            do_work()
    """
    ctx = SuperworkerContext(
        worker_id=worker_id,
        task_id=task_id,
        ring=ring,
        actor=actor,
        config=config,
    )
    
    old_context = get_current_context()
    set_current_context(ctx)
    
    try:
        yield ctx
    finally:
        if old_context:
            set_current_context(old_context)
        else:
            clear_current_context()


def inject_context(func: Callable) -> Callable:
    """
    Decorator to inject superworker context into function.
    The decorated function will receive 'ctx' as first argument.
    """
    def wrapper(*args, **kwargs):
        ctx = get_current_context()
        if ctx is None:
            raise RuntimeError("No superworker context available")
        return func(ctx, *args, **kwargs)
    return wrapper


# Factory for creating contexts
class ContextFactory:
    """Factory for creating superworker contexts with defaults."""
    
    def __init__(self, default_ring: int = 2, default_actor: str = "system"):
        self.default_ring = default_ring
        self.default_actor = default_actor
        self._counter = 0
    
    def create(
        self,
        task_id: Optional[str] = None,
        ring: Optional[int] = None,
        actor: Optional[str] = None,
        **config
    ) -> SuperworkerContext:
        """Create a new context with defaults."""
        self._counter += 1
        return SuperworkerContext(
            worker_id=f"worker-{self._counter}",
            task_id=task_id,
            ring=ring or self.default_ring,
            actor=actor or self.default_actor,
            config=config,
        )


# Default factory instance
_factory = ContextFactory()


def create_context(**kwargs) -> SuperworkerContext:
    """Create a context using the default factory."""
    return _factory.create(**kwargs)


if __name__ == "__main__":
    # Self-test
    with superworker_context("test-worker", task_id="test-task", actor="gemini") as ctx:
        print(f"Worker: {ctx.worker_id}")
        print(f"Actor: {ctx.actor}")
        print(f"Ring: {ctx.ring}")
        print(f"Env: {ctx.env}")
