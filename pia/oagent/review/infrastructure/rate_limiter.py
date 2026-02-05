#!/usr/bin/env python3
"""
Review Rate Limiter (Step 188)

API rate limiting system for the Review Agent using token bucket
algorithm with multiple rate limiting strategies.

PBTSO Phase: VERIFY, SEQUESTER
Bus Topics: review.ratelimit.check, review.ratelimit.exceeded

Rate Limiting Features:
- Token bucket algorithm
- Multiple strategies (fixed window, sliding window)
- Per-client rate limiting
- Burst allowance
- Rate limit headers

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"      # Token bucket algorithm
    FIXED_WINDOW = "fixed_window"      # Fixed time window
    SLIDING_WINDOW = "sliding_window"  # Sliding time window
    LEAKY_BUCKET = "leaky_bucket"      # Leaky bucket algorithm


class RateLimitScope(Enum):
    """Scope for rate limiting."""
    GLOBAL = "global"        # Global rate limit
    PER_CLIENT = "per_client"  # Per-client rate limit
    PER_ENDPOINT = "per_endpoint"  # Per-endpoint rate limit
    PER_USER = "per_user"    # Per-user rate limit


@dataclass
class RateLimitResult:
    """
    Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed
        limit: Total limit
        remaining: Remaining requests
        reset_at: When the limit resets (timestamp)
        retry_after: Seconds until retry allowed
        client_id: Client identifier
        scope: Rate limit scope
    """
    allowed: bool
    limit: int
    remaining: int
    reset_at: float
    retry_after: float = 0
    client_id: str = ""
    scope: RateLimitScope = RateLimitScope.GLOBAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "reset_at_iso": datetime.fromtimestamp(self.reset_at, tz=timezone.utc).isoformat() + "Z",
            "retry_after": round(self.retry_after, 1),
            "client_id": self.client_id,
            "scope": self.scope.value,
        }

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting.

    Attributes:
        strategy: Rate limiting strategy
        requests_per_minute: Requests allowed per minute
        requests_per_hour: Requests allowed per hour
        requests_per_day: Requests allowed per day
        burst_size: Maximum burst size
        scope: Rate limit scope
        enabled: Enable rate limiting
        whitelist: Whitelisted client IDs
        blacklist: Blacklisted client IDs
    """
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    scope: RateLimitScope = RateLimitScope.GLOBAL
    enabled: bool = True
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "strategy": self.strategy.value,
            "scope": self.scope.value,
        }


@dataclass
class RateLimitRule:
    """
    A rate limit rule.

    Attributes:
        name: Rule name
        pattern: Pattern to match (endpoint, client, etc.)
        requests_per_minute: RPM limit
        requests_per_hour: RPH limit
        burst_size: Burst allowance
        scope: Rule scope
        priority: Rule priority (higher = checked first)
    """
    name: str
    pattern: str
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    scope: RateLimitScope = RateLimitScope.GLOBAL
    priority: int = 0

    def matches(self, key: str) -> bool:
        """Check if rule matches a key."""
        if self.pattern == "*":
            return True
        if self.pattern.endswith("*"):
            return key.startswith(self.pattern[:-1])
        return key == self.pattern

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "scope": self.scope.value,
        }


# ============================================================================
# Token Bucket
# ============================================================================

class TokenBucket:
    """
    Token bucket rate limiter.

    Tokens are added at a constant rate up to a maximum capacity.
    Each request consumes one token.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        initial_tokens: Optional[int] = None,
    ):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens (bucket size)
            refill_rate: Tokens added per second
            initial_tokens: Initial token count (default: capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait for tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait (0 if tokens available)
        """
        self._refill()

        if self.tokens >= tokens:
            return 0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate

    @property
    def available_tokens(self) -> int:
        """Get available tokens."""
        self._refill()
        return int(self.tokens)


# ============================================================================
# Sliding Window Counter
# ============================================================================

class SlidingWindowCounter:
    """
    Sliding window rate limiter.

    Uses a sliding window to smooth out rate limiting.
    """

    def __init__(self, limit: int, window_seconds: int):
        """
        Initialize sliding window.

        Args:
            limit: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: List[float] = []

    def _cleanup(self) -> None:
        """Remove expired requests."""
        cutoff = time.time() - self.window_seconds
        self._requests = [t for t in self._requests if t > cutoff]

    def record(self) -> bool:
        """
        Record a request.

        Returns:
            True if request is within limit
        """
        self._cleanup()

        if len(self._requests) >= self.limit:
            return False

        self._requests.append(time.time())
        return True

    @property
    def count(self) -> int:
        """Get current request count."""
        self._cleanup()
        return len(self._requests)

    @property
    def remaining(self) -> int:
        """Get remaining requests."""
        return max(0, self.limit - self.count)

    @property
    def reset_at(self) -> float:
        """Get reset timestamp."""
        if not self._requests:
            return time.time() + self.window_seconds
        return self._requests[0] + self.window_seconds


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """
    Rate limiting system for the Review Agent.

    Example:
        limiter = RateLimiter()

        # Check rate limit
        result = await limiter.check("client-123")

        if result.allowed:
            # Process request
            pass
        else:
            # Return 429 Too Many Requests
            print(f"Retry after {result.retry_after} seconds")

        # Add custom rules
        limiter.add_rule(RateLimitRule(
            name="premium",
            pattern="premium-*",
            requests_per_minute=120,
        ))
    """

    BUS_TOPICS = {
        "check": "review.ratelimit.check",
        "exceeded": "review.ratelimit.exceeded",
    }

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            config: Rate limit configuration
            bus_path: Path to event bus file
        """
        self.config = config or RateLimitConfig()
        self.bus_path = bus_path or self._get_bus_path()

        # Rate limiters by key
        self._buckets: Dict[str, TokenBucket] = {}
        self._windows: Dict[str, SlidingWindowCounter] = {}

        # Rules
        self._rules: List[RateLimitRule] = []

        # Statistics
        self._stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_requests": 0,
        }
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "ratelimit") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "rate-limiter",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _get_key(self, client_id: str, endpoint: Optional[str] = None) -> str:
        """Generate rate limit key."""
        if self.config.scope == RateLimitScope.GLOBAL:
            return "global"
        elif self.config.scope == RateLimitScope.PER_ENDPOINT and endpoint:
            return f"endpoint:{endpoint}"
        elif self.config.scope == RateLimitScope.PER_CLIENT:
            return f"client:{client_id}"
        elif self.config.scope == RateLimitScope.PER_USER:
            return f"user:{client_id}"
        return f"client:{client_id}"

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create token bucket for key."""
        if key not in self._buckets:
            # Calculate tokens per second
            tokens_per_second = self.config.requests_per_minute / 60.0
            self._buckets[key] = TokenBucket(
                capacity=self.config.burst_size,
                refill_rate=tokens_per_second,
            )
        return self._buckets[key]

    def _get_window(self, key: str) -> SlidingWindowCounter:
        """Get or create sliding window for key."""
        if key not in self._windows:
            self._windows[key] = SlidingWindowCounter(
                limit=self.config.requests_per_minute,
                window_seconds=60,
            )
        return self._windows[key]

    def _get_rule_for_key(self, key: str) -> Optional[RateLimitRule]:
        """Get matching rule for key."""
        matching_rules = [r for r in self._rules if r.matches(key)]
        if matching_rules:
            # Return highest priority rule
            return max(matching_rules, key=lambda r: r.priority)
        return None

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limit rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)

    def remove_rule(self, name: str) -> bool:
        """Remove a rate limit rule."""
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < original_len

    async def check(
        self,
        client_id: str,
        endpoint: Optional[str] = None,
        cost: int = 1,
    ) -> RateLimitResult:
        """
        Check rate limit for a request.

        Args:
            client_id: Client identifier
            endpoint: Optional endpoint being accessed
            cost: Request cost (default: 1)

        Returns:
            RateLimitResult

        Emits:
            review.ratelimit.check
            review.ratelimit.exceeded (if exceeded)
        """
        self._stats["total_requests"] += 1

        # Check whitelist/blacklist
        if client_id in self.config.whitelist:
            return RateLimitResult(
                allowed=True,
                limit=999999,
                remaining=999999,
                reset_at=time.time() + 60,
                client_id=client_id,
                scope=self.config.scope,
            )

        if client_id in self.config.blacklist:
            self._stats["rejected_requests"] += 1
            return RateLimitResult(
                allowed=False,
                limit=0,
                remaining=0,
                reset_at=time.time() + 3600,
                retry_after=3600,
                client_id=client_id,
                scope=self.config.scope,
            )

        if not self.config.enabled:
            return RateLimitResult(
                allowed=True,
                limit=self.config.requests_per_minute,
                remaining=self.config.requests_per_minute,
                reset_at=time.time() + 60,
                client_id=client_id,
                scope=self.config.scope,
            )

        # Get key and apply rules
        key = self._get_key(client_id, endpoint)
        rule = self._get_rule_for_key(key)

        if rule:
            limit = rule.requests_per_minute
            burst = rule.burst_size
        else:
            limit = self.config.requests_per_minute
            burst = self.config.burst_size

        # Check based on strategy
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            result = self._check_token_bucket(key, cost, limit, burst, client_id)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            result = self._check_sliding_window(key, cost, limit, client_id)
        else:
            result = self._check_token_bucket(key, cost, limit, burst, client_id)

        # Update stats and emit events
        if result.allowed:
            self._stats["allowed_requests"] += 1
        else:
            self._stats["rejected_requests"] += 1
            self._emit_event(self.BUS_TOPICS["exceeded"], {
                "client_id": client_id,
                "endpoint": endpoint,
                "limit": result.limit,
                "retry_after": result.retry_after,
            })

        self._emit_event(self.BUS_TOPICS["check"], {
            "client_id": client_id,
            "allowed": result.allowed,
            "remaining": result.remaining,
        })

        return result

    def _check_token_bucket(
        self,
        key: str,
        cost: int,
        limit: int,
        burst: int,
        client_id: str,
    ) -> RateLimitResult:
        """Check using token bucket strategy."""
        bucket = self._get_bucket(key)

        # Adjust bucket if limit changed
        if bucket.capacity != burst:
            tokens_per_second = limit / 60.0
            bucket.capacity = burst
            bucket.refill_rate = tokens_per_second

        allowed = bucket.consume(cost)
        wait_time = bucket.get_wait_time(cost) if not allowed else 0

        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=bucket.available_tokens,
            reset_at=time.time() + 60,
            retry_after=wait_time,
            client_id=client_id,
            scope=self.config.scope,
        )

    def _check_sliding_window(
        self,
        key: str,
        cost: int,
        limit: int,
        client_id: str,
    ) -> RateLimitResult:
        """Check using sliding window strategy."""
        window = self._get_window(key)

        # Adjust window if limit changed
        if window.limit != limit:
            window.limit = limit

        allowed = all(window.record() for _ in range(cost))

        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=window.remaining,
            reset_at=window.reset_at,
            retry_after=window.reset_at - time.time() if not allowed else 0,
            client_id=client_id,
            scope=self.config.scope,
        )

    def reset(self, client_id: Optional[str] = None) -> int:
        """
        Reset rate limits.

        Args:
            client_id: Specific client to reset (None = all)

        Returns:
            Number of limiters reset
        """
        if client_id:
            key = self._get_key(client_id)
            count = 0
            if key in self._buckets:
                del self._buckets[key]
                count += 1
            if key in self._windows:
                del self._windows[key]
                count += 1
            return count
        else:
            count = len(self._buckets) + len(self._windows)
            self._buckets.clear()
            self._windows.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            **self._stats,
            "active_buckets": len(self._buckets),
            "active_windows": len(self._windows),
            "rules_count": len(self._rules),
            "rejection_rate": (
                self._stats["rejected_requests"] / self._stats["total_requests"] * 100
                if self._stats["total_requests"] > 0 else 0
            ),
        }

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        stats = self.get_stats()
        status = {
            "agent": "rate-limiter",
            "healthy": True,
            "enabled": self.config.enabled,
            "strategy": self.config.strategy.value,
            "requests_handled": stats["total_requests"],
            "rejection_rate": round(stats["rejection_rate"], 2),
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# Decorator
# ============================================================================

def rate_limit(
    limiter: RateLimiter,
    cost: int = 1,
    key_fn: Optional[Callable[..., str]] = None,
):
    """
    Decorator for rate limiting functions.

    Args:
        limiter: Rate limiter instance
        cost: Request cost
        key_fn: Function to extract client key from args

    Example:
        @rate_limit(limiter, cost=1)
        async def my_endpoint(client_id: str):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract client ID
            if key_fn:
                client_id = key_fn(*args, **kwargs)
            elif "client_id" in kwargs:
                client_id = kwargs["client_id"]
            elif args:
                client_id = str(args[0])
            else:
                client_id = "anonymous"

            result = await limiter.check(client_id, cost=cost)

            if not result.allowed:
                raise Exception(f"Rate limit exceeded. Retry after {result.retry_after}s")

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Rate Limiter."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Rate Limiter (Step 188)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check rate limit")
    check_parser.add_argument("client_id", help="Client identifier")
    check_parser.add_argument("--endpoint", help="Endpoint")
    check_parser.add_argument("--cost", type=int, default=1, help="Request cost")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset rate limits")
    reset_parser.add_argument("--client", help="Specific client to reset")

    # Config command
    subparsers.add_parser("config", help="Show configuration")

    # Rules command
    rules_parser = subparsers.add_parser("rules", help="Manage rules")
    rules_parser.add_argument("--add", nargs=3, metavar=("NAME", "PATTERN", "RPM"),
                              help="Add rule")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    limiter = RateLimiter()

    if args.command == "check":
        result = asyncio.run(limiter.check(
            args.client_id,
            endpoint=args.endpoint,
            cost=args.cost,
        ))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "ALLOWED" if result.allowed else "REJECTED"
            print(f"{status}: {result.remaining}/{result.limit} remaining")
            if not result.allowed:
                print(f"  Retry after: {result.retry_after:.1f}s")

    elif args.command == "stats":
        stats = limiter.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Rate Limiter Statistics")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Allowed: {stats['allowed_requests']}")
            print(f"  Rejected: {stats['rejected_requests']}")
            print(f"  Rejection Rate: {stats['rejection_rate']:.1f}%")
            print(f"  Active Limiters: {stats['active_buckets'] + stats['active_windows']}")

    elif args.command == "reset":
        count = limiter.reset(args.client)
        print(f"Reset {count} limiters")

    elif args.command == "config":
        if args.json:
            print(json.dumps(limiter.config.to_dict(), indent=2))
        else:
            print("Rate Limiter Configuration")
            for k, v in limiter.config.to_dict().items():
                print(f"  {k}: {v}")

    elif args.command == "rules":
        if args.add:
            name, pattern, rpm = args.add
            limiter.add_rule(RateLimitRule(
                name=name,
                pattern=pattern,
                requests_per_minute=int(rpm),
            ))
            print(f"Added rule: {name}")
        else:
            rules = [r.to_dict() for r in limiter._rules]
            if args.json:
                print(json.dumps(rules, indent=2))
            else:
                print(f"Rules: {len(rules)}")
                for r in rules:
                    print(f"  {r['name']}: {r['pattern']} ({r['requests_per_minute']} RPM)")

    else:
        # Default: show status
        status = limiter.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Rate Limiter: {status['strategy']}, {status['requests_handled']} requests")

    return 0


if __name__ == "__main__":
    sys.exit(main())
