#!/usr/bin/env python3
"""
Monitor Rate Limiter - Step 288

API rate limiting for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.ratelimit.exceeded (emitted)
- monitor.ratelimit.reset (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"           # Fixed time windows
    SLIDING_WINDOW = "sliding_window"       # Sliding time windows
    TOKEN_BUCKET = "token_bucket"           # Token bucket algorithm
    LEAKY_BUCKET = "leaky_bucket"           # Leaky bucket algorithm


class RateLimitScope(Enum):
    """Rate limit scope."""
    GLOBAL = "global"     # Applies to all requests
    USER = "user"         # Per-user limits
    ENDPOINT = "endpoint" # Per-endpoint limits
    IP = "ip"            # Per-IP limits


@dataclass
class RateLimitConfig:
    """Rate limit configuration.

    Attributes:
        name: Limit name
        requests_per_window: Max requests
        window_s: Window size in seconds
        strategy: Rate limit strategy
        scope: Rate limit scope
        burst_allowance: Extra burst capacity
    """
    name: str
    requests_per_window: int
    window_s: int = 60
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    scope: RateLimitScope = RateLimitScope.GLOBAL
    burst_allowance: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "requests_per_window": self.requests_per_window,
            "window_s": self.window_s,
            "strategy": self.strategy.value,
            "scope": self.scope.value,
            "burst_allowance": self.burst_allowance,
        }


@dataclass
class RateLimitState:
    """State for a rate limiter.

    Attributes:
        key: Unique key (e.g., user_id, endpoint)
        request_count: Current request count
        window_start: Window start time
        tokens: Available tokens (for token bucket)
        last_refill: Last token refill time
        exceeded_count: Times limit was exceeded
    """
    key: str
    request_count: int = 0
    window_start: float = field(default_factory=time.time)
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    exceeded_count: int = 0
    timestamps: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "request_count": self.request_count,
            "window_start": self.window_start,
            "tokens": self.tokens,
            "exceeded_count": self.exceeded_count,
        }


@dataclass
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: Whether request is allowed
        remaining: Remaining requests
        reset_at: When limit resets
        retry_after_s: Seconds to wait before retry
        limit_name: Name of the limit
    """
    allowed: bool
    remaining: int
    reset_at: float
    retry_after_s: float = 0.0
    limit_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "retry_after_s": self.retry_after_s,
            "limit_name": self.limit_name,
        }

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.remaining + (0 if self.allowed else 1)),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(int(self.retry_after_s))
        return headers


class MonitorRateLimiter:
    """
    API rate limiting for the Monitor Agent.

    Provides:
    - Multiple rate limiting strategies
    - Per-user, per-endpoint, and global limits
    - Configurable burst allowance
    - Limit state persistence

    Example:
        limiter = MonitorRateLimiter()

        # Configure a limit
        limiter.configure_limit(RateLimitConfig(
            name="api_requests",
            requests_per_window=100,
            window_s=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        ))

        # Check rate limit
        result = limiter.check_limit("api_requests", key="user-123")
        if result.allowed:
            # Process request
            pass
        else:
            # Return 429 Too Many Requests
            pass
    """

    BUS_TOPICS = {
        "exceeded": "monitor.ratelimit.exceeded",
        "reset": "monitor.ratelimit.reset",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        cleanup_interval_s: int = 60,
        state_ttl_s: int = 3600,
        bus_dir: Optional[str] = None,
    ):
        """Initialize rate limiter.

        Args:
            cleanup_interval_s: Cleanup interval
            state_ttl_s: State TTL in seconds
            bus_dir: Bus directory
        """
        self._cleanup_interval = cleanup_interval_s
        self._state_ttl = state_ttl_s
        self._last_heartbeat = time.time()
        self._last_cleanup = time.time()

        # Limits and state
        self._limits: Dict[str, RateLimitConfig] = {}
        self._state: Dict[str, Dict[str, RateLimitState]] = {}  # limit_name -> key -> state
        self._lock = threading.RLock()

        # Statistics
        self._total_requests = 0
        self._exceeded_requests = 0

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default limits
        self._register_default_limits()

    def configure_limit(self, config: RateLimitConfig) -> None:
        """Configure a rate limit.

        Args:
            config: Rate limit configuration
        """
        with self._lock:
            self._limits[config.name] = config
            self._state[config.name] = {}

    def remove_limit(self, name: str) -> bool:
        """Remove a rate limit.

        Args:
            name: Limit name

        Returns:
            True if removed
        """
        with self._lock:
            if name in self._limits:
                del self._limits[name]
                if name in self._state:
                    del self._state[name]
                return True
            return False

    def check_limit(
        self,
        name: str,
        key: str = "global",
        cost: int = 1,
    ) -> RateLimitResult:
        """Check if request is allowed.

        Args:
            name: Limit name
            key: Rate limit key
            cost: Request cost

        Returns:
            Rate limit result
        """
        self._total_requests += 1

        with self._lock:
            config = self._limits.get(name)
            if not config:
                return RateLimitResult(
                    allowed=True,
                    remaining=float("inf"),
                    reset_at=0,
                    limit_name=name,
                )

            # Get or create state
            if name not in self._state:
                self._state[name] = {}

            if key not in self._state[name]:
                self._state[name][key] = RateLimitState(
                    key=key,
                    tokens=config.requests_per_window + config.burst_allowance,
                )

            state = self._state[name][key]

            # Apply rate limiting strategy
            if config.strategy == RateLimitStrategy.FIXED_WINDOW:
                result = self._check_fixed_window(config, state, cost)
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                result = self._check_sliding_window(config, state, cost)
            elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                result = self._check_token_bucket(config, state, cost)
            elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                result = self._check_leaky_bucket(config, state, cost)
            else:
                result = RateLimitResult(
                    allowed=True,
                    remaining=config.requests_per_window,
                    reset_at=time.time() + config.window_s,
                    limit_name=name,
                )

            result.limit_name = name

            if not result.allowed:
                self._exceeded_requests += 1
                state.exceeded_count += 1
                self._emit_bus_event(
                    self.BUS_TOPICS["exceeded"],
                    {
                        "limit": name,
                        "key": key,
                        "retry_after_s": result.retry_after_s,
                    },
                    level="warning",
                )

        # Periodic cleanup
        self._maybe_cleanup()

        return result

    def consume(
        self,
        name: str,
        key: str = "global",
        cost: int = 1,
    ) -> RateLimitResult:
        """Consume rate limit quota.

        Args:
            name: Limit name
            key: Rate limit key
            cost: Request cost

        Returns:
            Rate limit result
        """
        return self.check_limit(name, key, cost)

    def reset_limit(self, name: str, key: str = "global") -> bool:
        """Reset a rate limit state.

        Args:
            name: Limit name
            key: Rate limit key

        Returns:
            True if reset
        """
        with self._lock:
            if name in self._state and key in self._state[name]:
                config = self._limits.get(name)
                if config:
                    self._state[name][key] = RateLimitState(
                        key=key,
                        tokens=config.requests_per_window + config.burst_allowance,
                    )
                    self._emit_bus_event(
                        self.BUS_TOPICS["reset"],
                        {"limit": name, "key": key},
                    )
                    return True
        return False

    def get_remaining(
        self,
        name: str,
        key: str = "global",
    ) -> int:
        """Get remaining requests.

        Args:
            name: Limit name
            key: Rate limit key

        Returns:
            Remaining requests
        """
        with self._lock:
            config = self._limits.get(name)
            if not config:
                return float("inf")

            state = self._state.get(name, {}).get(key)
            if not state:
                return config.requests_per_window

            if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return int(state.tokens)
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                now = time.time()
                valid_timestamps = [
                    t for t in state.timestamps
                    if now - t < config.window_s
                ]
                return config.requests_per_window - len(valid_timestamps)
            else:
                return config.requests_per_window - state.request_count

    def list_limits(self) -> List[Dict[str, Any]]:
        """List all configured limits.

        Returns:
            List of limit configurations
        """
        with self._lock:
            return [config.to_dict() for config in self._limits.values()]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            state_counts = {
                name: len(states)
                for name, states in self._state.items()
            }

            exceeded_by_limit = {}
            for name, states in self._state.items():
                exceeded_by_limit[name] = sum(s.exceeded_count for s in states.values())

            return {
                "total_requests": self._total_requests,
                "exceeded_requests": self._exceeded_requests,
                "exceed_rate": (
                    self._exceeded_requests / self._total_requests
                    if self._total_requests > 0 else 0.0
                ),
                "limits_configured": len(self._limits),
                "active_states": state_counts,
                "exceeded_by_limit": exceeded_by_limit,
            }

    def _check_fixed_window(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
    ) -> RateLimitResult:
        """Check fixed window rate limit."""
        now = time.time()
        window_end = state.window_start + config.window_s

        # Check if window has expired
        if now >= window_end:
            state.window_start = now
            state.request_count = 0

        max_requests = config.requests_per_window + config.burst_allowance
        remaining = max_requests - state.request_count

        if remaining >= cost:
            state.request_count += cost
            return RateLimitResult(
                allowed=True,
                remaining=remaining - cost,
                reset_at=state.window_start + config.window_s,
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=state.window_start + config.window_s,
                retry_after_s=window_end - now,
            )

    def _check_sliding_window(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
    ) -> RateLimitResult:
        """Check sliding window rate limit."""
        now = time.time()
        window_start = now - config.window_s

        # Remove old timestamps
        state.timestamps = [t for t in state.timestamps if t > window_start]

        max_requests = config.requests_per_window + config.burst_allowance
        remaining = max_requests - len(state.timestamps)

        if remaining >= cost:
            # Add new timestamp(s)
            for _ in range(cost):
                state.timestamps.append(now)
            return RateLimitResult(
                allowed=True,
                remaining=remaining - cost,
                reset_at=now + config.window_s,
            )
        else:
            # Find when oldest request expires
            if state.timestamps:
                oldest = min(state.timestamps)
                retry_after = oldest + config.window_s - now
            else:
                retry_after = 0

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + config.window_s,
                retry_after_s=max(0, retry_after),
            )

    def _check_token_bucket(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
    ) -> RateLimitResult:
        """Check token bucket rate limit."""
        now = time.time()
        max_tokens = config.requests_per_window + config.burst_allowance
        refill_rate = config.requests_per_window / config.window_s

        # Refill tokens
        elapsed = now - state.last_refill
        state.tokens = min(max_tokens, state.tokens + elapsed * refill_rate)
        state.last_refill = now

        if state.tokens >= cost:
            state.tokens -= cost
            return RateLimitResult(
                allowed=True,
                remaining=int(state.tokens),
                reset_at=now + (max_tokens - state.tokens) / refill_rate,
            )
        else:
            tokens_needed = cost - state.tokens
            retry_after = tokens_needed / refill_rate

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + retry_after,
                retry_after_s=retry_after,
            )

    def _check_leaky_bucket(
        self,
        config: RateLimitConfig,
        state: RateLimitState,
        cost: int,
    ) -> RateLimitResult:
        """Check leaky bucket rate limit."""
        now = time.time()
        leak_rate = config.requests_per_window / config.window_s
        max_capacity = config.requests_per_window + config.burst_allowance

        # Leak tokens
        elapsed = now - state.last_refill
        leaked = elapsed * leak_rate
        state.tokens = max(0, state.tokens - leaked)
        state.last_refill = now

        if state.tokens + cost <= max_capacity:
            state.tokens += cost
            return RateLimitResult(
                allowed=True,
                remaining=int(max_capacity - state.tokens),
                reset_at=now + state.tokens / leak_rate,
            )
        else:
            overflow = state.tokens + cost - max_capacity
            retry_after = overflow / leak_rate

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + retry_after,
                retry_after_s=retry_after,
            )

    def _register_default_limits(self) -> None:
        """Register default rate limits."""
        self.configure_limit(RateLimitConfig(
            name="api_global",
            requests_per_window=1000,
            window_s=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            scope=RateLimitScope.GLOBAL,
            burst_allowance=100,
        ))

        self.configure_limit(RateLimitConfig(
            name="api_user",
            requests_per_window=100,
            window_s=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            scope=RateLimitScope.USER,
            burst_allowance=20,
        ))

        self.configure_limit(RateLimitConfig(
            name="metrics_ingest",
            requests_per_window=10000,
            window_s=60,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            scope=RateLimitScope.GLOBAL,
            burst_allowance=1000,
        ))

    def _maybe_cleanup(self) -> None:
        """Cleanup old state if needed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now

        with self._lock:
            for name in list(self._state.keys()):
                for key in list(self._state[name].keys()):
                    state = self._state[name][key]
                    # Remove if no activity for TTL
                    if now - state.window_start > self._state_ttl:
                        del self._state[name][key]

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        stats = self.get_stats()

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_rate_limiter",
                "status": "healthy",
                "total_requests": stats["total_requests"],
                "exceed_rate": stats["exceed_rate"],
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-rate-limiter",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
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

        return event_id


# Singleton instance
_limiter: Optional[MonitorRateLimiter] = None


def get_rate_limiter() -> MonitorRateLimiter:
    """Get or create the rate limiter singleton.

    Returns:
        MonitorRateLimiter instance
    """
    global _limiter
    if _limiter is None:
        _limiter = MonitorRateLimiter()
    return _limiter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Rate Limiter (Step 288)")
    parser.add_argument("--check", metavar="NAME", help="Check a rate limit")
    parser.add_argument("--key", default="global", help="Rate limit key")
    parser.add_argument("--remaining", metavar="NAME", help="Get remaining quota")
    parser.add_argument("--reset", metavar="NAME", help="Reset a rate limit")
    parser.add_argument("--list", action="store_true", help="List configured limits")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--test", metavar="N", type=int, help="Test N requests")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    limiter = get_rate_limiter()

    if args.check:
        result = limiter.check_limit(args.check, args.key)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "allowed" if result.allowed else "denied"
            print(f"Rate Limit Check: {status}")
            print(f"  Remaining: {result.remaining}")
            print(f"  Reset at: {result.reset_at}")
            if not result.allowed:
                print(f"  Retry after: {result.retry_after_s:.1f}s")

    if args.remaining:
        remaining = limiter.get_remaining(args.remaining, args.key)
        if args.json:
            print(json.dumps({"limit": args.remaining, "key": args.key, "remaining": remaining}))
        else:
            print(f"Remaining for {args.remaining} ({args.key}): {remaining}")

    if args.reset:
        success = limiter.reset_limit(args.reset, args.key)
        if args.json:
            print(json.dumps({"reset": success}))
        else:
            print(f"Reset {args.reset}: {'success' if success else 'failed'}")

    if args.list:
        limits = limiter.list_limits()
        if args.json:
            print(json.dumps(limits, indent=2))
        else:
            print("Configured Limits:")
            for limit in limits:
                print(f"  {limit['name']}: {limit['requests_per_window']}/{limit['window_s']}s ({limit['strategy']})")

    if args.stats:
        stats = limiter.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Rate Limiter Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    if args.test:
        limit_name = args.check or "api_global"
        allowed = 0
        denied = 0
        for i in range(args.test):
            result = limiter.check_limit(limit_name, args.key)
            if result.allowed:
                allowed += 1
            else:
                denied += 1
        if args.json:
            print(json.dumps({"allowed": allowed, "denied": denied}))
        else:
            print(f"Test Results ({args.test} requests):")
            print(f"  Allowed: {allowed}")
            print(f"  Denied: {denied}")
