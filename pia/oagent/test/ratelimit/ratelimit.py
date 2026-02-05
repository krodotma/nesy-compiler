#!/usr/bin/env python3
"""
Step 138: Test Rate Limiter

API rate limiting for the Test Agent.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.ratelimit.allowed (emits)
- test.ratelimit.denied (emits)
- test.ratelimit.reset (emits)

Dependencies: Steps 101-137 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Scope for rate limiting."""
    GLOBAL = "global"
    CLIENT = "client"
    ENDPOINT = "endpoint"
    USER = "user"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class RateLimitBucket:
    """
    A rate limit bucket.

    Attributes:
        key: Bucket key
        tokens: Available tokens
        last_refill: Last refill timestamp
        requests: Request timestamps (for sliding window)
        max_tokens: Maximum tokens
        refill_rate: Tokens per second
    """
    key: str
    tokens: float = 0
    last_refill: float = field(default_factory=time.time)
    requests: List[float] = field(default_factory=list)
    max_tokens: int = 100
    refill_rate: float = 10.0  # tokens per second

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens."""
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "refill_rate": self.refill_rate,
            "last_refill": self.last_refill,
        }


@dataclass
class RateLimitResult:
    """
    Result of a rate limit check.

    Attributes:
        allowed: Whether request is allowed
        remaining: Remaining requests/tokens
        reset_at: When limit resets
        retry_after: Seconds until retry
        limit: The limit value
        bucket_key: The bucket key used
    """
    allowed: bool
    remaining: int = 0
    reset_at: Optional[float] = None
    retry_after: Optional[float] = None
    limit: int = 0
    bucket_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "retry_after": self.retry_after,
            "limit": self.limit,
            "bucket_key": self.bucket_key,
        }

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if self.reset_at:
            headers["X-RateLimit-Reset"] = str(int(self.reset_at))
        if not self.allowed and self.retry_after:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


@dataclass
class RateLimitStats:
    """
    Rate limiting statistics.

    Attributes:
        total_requests: Total requests
        allowed_requests: Allowed requests
        denied_requests: Denied requests
        active_buckets: Number of active buckets
    """
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    active_buckets: int = 0
    peak_qps: float = 0

    @property
    def denial_rate(self) -> float:
        """Calculate denial rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.denied_requests / self.total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "denied_requests": self.denied_requests,
            "denial_rate": self.denial_rate,
            "active_buckets": self.active_buckets,
            "peak_qps": self.peak_qps,
        }


@dataclass
class RateLimitRule:
    """
    A rate limiting rule.

    Attributes:
        name: Rule name
        limit: Request limit
        window_s: Time window in seconds
        strategy: Rate limiting strategy
        scope: Rate limit scope
        pattern: URL pattern to match
        enabled: Whether rule is enabled
    """
    name: str
    limit: int = 100
    window_s: int = 60
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    scope: RateLimitScope = RateLimitScope.GLOBAL
    pattern: str = "*"
    enabled: bool = True
    burst_limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "limit": self.limit,
            "window_s": self.window_s,
            "strategy": self.strategy.value,
            "scope": self.scope.value,
            "pattern": self.pattern,
            "enabled": self.enabled,
            "burst_limit": self.burst_limit,
        }


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting.

    Attributes:
        default_limit: Default request limit
        default_window_s: Default time window
        strategy: Default strategy
        rules: Rate limit rules
        enabled: Whether rate limiting is enabled
        whitelist: Whitelisted clients
        output_dir: Output directory
    """
    default_limit: int = 100
    default_window_s: int = 60
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    rules: List[RateLimitRule] = field(default_factory=list)
    enabled: bool = True
    whitelist: List[str] = field(default_factory=list)
    output_dir: str = ".pluribus/test-agent/ratelimit"
    bucket_cleanup_interval_s: int = 300
    max_buckets: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_limit": self.default_limit,
            "default_window_s": self.default_window_s,
            "strategy": self.strategy.value,
            "rules": [r.to_dict() for r in self.rules],
            "enabled": self.enabled,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class RateLimitBus:
    """Bus interface for rate limiting with file locking."""

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
# Test Rate Limiter
# ============================================================================

class TestRateLimiter:
    """
    API rate limiting for the Test Agent.

    Features:
    - Multiple rate limiting strategies
    - Per-client and per-endpoint limits
    - Token bucket implementation
    - Sliding window implementation
    - Whitelist support

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.ratelimit.allowed, test.ratelimit.denied, test.ratelimit.reset
    """

    BUS_TOPICS = {
        "allowed": "test.ratelimit.allowed",
        "denied": "test.ratelimit.denied",
        "reset": "test.ratelimit.reset",
    }

    def __init__(self, bus=None, config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.

        Args:
            bus: Optional bus instance
            config: Rate limit configuration
        """
        self.bus = bus or RateLimitBus()
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, RateLimitBucket] = {}
        self._stats = RateLimitStats()
        self._lock = threading.RLock()
        self._qps_counter: List[float] = []

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Add default rules if none provided
        if not self.config.rules:
            self._add_default_rules()

    def _add_default_rules(self) -> None:
        """Add default rate limiting rules."""
        self.config.rules = [
            RateLimitRule(
                name="api_default",
                limit=100,
                window_s=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.CLIENT,
            ),
            RateLimitRule(
                name="test_run",
                limit=10,
                window_s=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.GLOBAL,
                pattern="/api/*/tests/run",
            ),
        ]

    def check(
        self,
        client_id: str = "default",
        endpoint: str = "*",
        tokens: int = 1,
    ) -> RateLimitResult:
        """
        Check if a request should be allowed.

        Args:
            client_id: Client identifier
            endpoint: Request endpoint
            tokens: Number of tokens to consume

        Returns:
            RateLimitResult with decision
        """
        if not self.config.enabled:
            return RateLimitResult(allowed=True, limit=0, remaining=0)

        # Check whitelist
        if client_id in self.config.whitelist:
            return RateLimitResult(allowed=True, limit=0, remaining=0)

        # Find matching rule
        rule = self._find_matching_rule(endpoint)
        if rule is None:
            rule = RateLimitRule(
                name="default",
                limit=self.config.default_limit,
                window_s=self.config.default_window_s,
                strategy=self.config.strategy,
            )

        # Get bucket key
        bucket_key = self._get_bucket_key(client_id, endpoint, rule)

        # Execute rate limit check based on strategy
        with self._lock:
            if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                result = self._check_token_bucket(bucket_key, rule, tokens)
            elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                result = self._check_sliding_window(bucket_key, rule, tokens)
            elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                result = self._check_fixed_window(bucket_key, rule, tokens)
            else:
                result = self._check_leaky_bucket(bucket_key, rule, tokens)

            # Update stats
            self._stats.total_requests += 1
            if result.allowed:
                self._stats.allowed_requests += 1
                self._emit_event("allowed", {
                    "client_id": client_id,
                    "endpoint": endpoint,
                    "remaining": result.remaining,
                })
            else:
                self._stats.denied_requests += 1
                self._emit_event("denied", {
                    "client_id": client_id,
                    "endpoint": endpoint,
                    "retry_after": result.retry_after,
                })

            # Update QPS tracking
            self._update_qps()

        return result

    def _find_matching_rule(self, endpoint: str) -> Optional[RateLimitRule]:
        """Find a matching rate limit rule."""
        import fnmatch

        for rule in self.config.rules:
            if not rule.enabled:
                continue
            if rule.pattern == "*" or fnmatch.fnmatch(endpoint, rule.pattern):
                return rule
        return None

    def _get_bucket_key(self, client_id: str, endpoint: str, rule: RateLimitRule) -> str:
        """Generate bucket key based on scope."""
        if rule.scope == RateLimitScope.GLOBAL:
            return f"global:{rule.name}"
        elif rule.scope == RateLimitScope.CLIENT:
            return f"client:{client_id}:{rule.name}"
        elif rule.scope == RateLimitScope.ENDPOINT:
            return f"endpoint:{endpoint}:{rule.name}"
        else:
            return f"user:{client_id}:{rule.name}"

    def _get_or_create_bucket(self, key: str, rule: RateLimitRule) -> RateLimitBucket:
        """Get or create a rate limit bucket."""
        if key not in self._buckets:
            # Check max buckets
            if len(self._buckets) >= self.config.max_buckets:
                self._cleanup_buckets()

            self._buckets[key] = RateLimitBucket(
                key=key,
                tokens=rule.burst_limit or rule.limit,
                max_tokens=rule.burst_limit or rule.limit,
                refill_rate=rule.limit / rule.window_s,
            )
        return self._buckets[key]

    def _check_token_bucket(
        self,
        bucket_key: str,
        rule: RateLimitRule,
        tokens: int,
    ) -> RateLimitResult:
        """Check using token bucket algorithm."""
        bucket = self._get_or_create_bucket(bucket_key, rule)
        allowed = bucket.consume(tokens)

        return RateLimitResult(
            allowed=allowed,
            remaining=int(bucket.tokens),
            reset_at=time.time() + rule.window_s if not allowed else None,
            retry_after=(tokens - bucket.tokens) / bucket.refill_rate if not allowed else None,
            limit=rule.limit,
            bucket_key=bucket_key,
        )

    def _check_sliding_window(
        self,
        bucket_key: str,
        rule: RateLimitRule,
        tokens: int,
    ) -> RateLimitResult:
        """Check using sliding window algorithm."""
        bucket = self._get_or_create_bucket(bucket_key, rule)
        now = time.time()
        window_start = now - rule.window_s

        # Remove old requests
        bucket.requests = [ts for ts in bucket.requests if ts > window_start]

        # Check if allowed
        if len(bucket.requests) + tokens <= rule.limit:
            # Add new requests
            for _ in range(tokens):
                bucket.requests.append(now)

            return RateLimitResult(
                allowed=True,
                remaining=rule.limit - len(bucket.requests),
                limit=rule.limit,
                bucket_key=bucket_key,
            )
        else:
            # Calculate retry after
            oldest = min(bucket.requests) if bucket.requests else now
            retry_after = oldest + rule.window_s - now

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=oldest + rule.window_s,
                retry_after=max(0, retry_after),
                limit=rule.limit,
                bucket_key=bucket_key,
            )

    def _check_fixed_window(
        self,
        bucket_key: str,
        rule: RateLimitRule,
        tokens: int,
    ) -> RateLimitResult:
        """Check using fixed window algorithm."""
        bucket = self._get_or_create_bucket(bucket_key, rule)
        now = time.time()

        # Calculate current window
        window_start = int(now / rule.window_s) * rule.window_s
        window_end = window_start + rule.window_s

        # Reset if new window
        if bucket.last_refill < window_start:
            bucket.tokens = rule.limit
            bucket.last_refill = window_start

        # Check if allowed
        if bucket.tokens >= tokens:
            bucket.tokens -= tokens

            return RateLimitResult(
                allowed=True,
                remaining=int(bucket.tokens),
                reset_at=window_end,
                limit=rule.limit,
                bucket_key=bucket_key,
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=window_end,
                retry_after=window_end - now,
                limit=rule.limit,
                bucket_key=bucket_key,
            )

    def _check_leaky_bucket(
        self,
        bucket_key: str,
        rule: RateLimitRule,
        tokens: int,
    ) -> RateLimitResult:
        """Check using leaky bucket algorithm."""
        # Similar to token bucket but processes at fixed rate
        return self._check_token_bucket(bucket_key, rule, tokens)

    def reset(self, client_id: Optional[str] = None, endpoint: Optional[str] = None) -> int:
        """
        Reset rate limits.

        Args:
            client_id: Client to reset (None for all)
            endpoint: Endpoint to reset (None for all)

        Returns:
            Number of buckets reset
        """
        with self._lock:
            to_remove = []

            for key in self._buckets:
                if client_id and f":{client_id}:" not in key:
                    continue
                if endpoint and f":{endpoint}:" not in key:
                    continue
                to_remove.append(key)

            for key in to_remove:
                del self._buckets[key]

            self._emit_event("reset", {
                "client_id": client_id,
                "endpoint": endpoint,
                "buckets_reset": len(to_remove),
            })

            return len(to_remove)

    def get_stats(self) -> RateLimitStats:
        """Get rate limiting statistics."""
        self._stats.active_buckets = len(self._buckets)
        return self._stats

    def get_bucket_info(self, bucket_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific bucket."""
        bucket = self._buckets.get(bucket_key)
        return bucket.to_dict() if bucket else None

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limit rule."""
        self.config.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rate limit rule."""
        for i, rule in enumerate(self.config.rules):
            if rule.name == name:
                del self.config.rules[i]
                return True
        return False

    def whitelist_add(self, client_id: str) -> None:
        """Add client to whitelist."""
        if client_id not in self.config.whitelist:
            self.config.whitelist.append(client_id)

    def whitelist_remove(self, client_id: str) -> bool:
        """Remove client from whitelist."""
        if client_id in self.config.whitelist:
            self.config.whitelist.remove(client_id)
            return True
        return False

    def _cleanup_buckets(self) -> None:
        """Clean up old buckets."""
        now = time.time()
        cutoff = now - max(r.window_s for r in self.config.rules) * 2

        to_remove = []
        for key, bucket in self._buckets.items():
            if bucket.last_refill < cutoff:
                to_remove.append(key)

        for key in to_remove[:len(self._buckets) // 2]:
            del self._buckets[key]

    def _update_qps(self) -> None:
        """Update QPS tracking."""
        now = time.time()
        self._qps_counter.append(now)

        # Keep only last second
        self._qps_counter = [ts for ts in self._qps_counter if ts > now - 1]

        # Update peak
        current_qps = len(self._qps_counter)
        if current_qps > self._stats.peak_qps:
            self._stats.peak_qps = current_qps

    async def check_async(
        self,
        client_id: str = "default",
        endpoint: str = "*",
        tokens: int = 1,
    ) -> RateLimitResult:
        """Async version of check."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check, client_id, endpoint, tokens)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.ratelimit.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "ratelimit",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# Middleware Decorator
# ============================================================================

def rate_limit(
    limiter: TestRateLimiter,
    limit: int = 100,
    window_s: int = 60,
    scope: RateLimitScope = RateLimitScope.CLIENT,
) -> Callable:
    """
    Decorator for rate limiting functions.

    Usage:
        @rate_limit(limiter, limit=10, window_s=60)
        def my_endpoint(request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        rule = RateLimitRule(
            name=func.__name__,
            limit=limit,
            window_s=window_s,
            scope=scope,
        )
        limiter.add_rule(rule)

        def wrapper(*args, **kwargs):
            client_id = kwargs.get("client_id", "default")
            result = limiter.check(client_id, func.__name__)

            if not result.allowed:
                raise Exception(f"Rate limit exceeded. Retry after {result.retry_after:.0f}s")

            return func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Rate Limiter."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Rate Limiter")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check rate limit")
    check_parser.add_argument("--client", default="default", help="Client ID")
    check_parser.add_argument("--endpoint", default="*", help="Endpoint")
    check_parser.add_argument("--tokens", type=int, default=1, help="Tokens to consume")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset rate limits")
    reset_parser.add_argument("--client", help="Client ID to reset")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")

    # Rules command
    rules_parser = subparsers.add_parser("rules", help="List rules")

    # Whitelist command
    whitelist_parser = subparsers.add_parser("whitelist", help="Manage whitelist")
    whitelist_parser.add_argument("action", choices=["add", "remove", "list"])
    whitelist_parser.add_argument("--client", help="Client ID")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/ratelimit")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = RateLimitConfig(output_dir=args.output)
    limiter = TestRateLimiter(config=config)

    if args.command == "check":
        result = limiter.check(args.client, args.endpoint, args.tokens)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "ALLOWED" if result.allowed else "DENIED"
            print(f"Status: {status}")
            print(f"Remaining: {result.remaining}/{result.limit}")
            if result.retry_after:
                print(f"Retry After: {result.retry_after:.1f}s")

    elif args.command == "reset":
        count = limiter.reset(args.client)
        print(f"Reset {count} buckets")

    elif args.command == "stats":
        stats = limiter.get_stats()

        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("\nRate Limit Statistics:")
            print(f"  Total Requests: {stats.total_requests}")
            print(f"  Allowed: {stats.allowed_requests}")
            print(f"  Denied: {stats.denied_requests}")
            print(f"  Denial Rate: {stats.denial_rate:.1f}%")
            print(f"  Active Buckets: {stats.active_buckets}")
            print(f"  Peak QPS: {stats.peak_qps}")

    elif args.command == "rules":
        rules = [r.to_dict() for r in limiter.config.rules]

        if args.json:
            print(json.dumps(rules, indent=2))
        else:
            print(f"\nRate Limit Rules ({len(rules)}):")
            for rule in limiter.config.rules:
                enabled = "[ON]" if rule.enabled else "[OFF]"
                print(f"  {enabled} {rule.name}: {rule.limit}/{rule.window_s}s ({rule.strategy.value})")
                print(f"      Pattern: {rule.pattern}, Scope: {rule.scope.value}")

    elif args.command == "whitelist":
        if args.action == "list":
            if args.json:
                print(json.dumps(limiter.config.whitelist, indent=2))
            else:
                print(f"\nWhitelist ({len(limiter.config.whitelist)}):")
                for client in limiter.config.whitelist:
                    print(f"  - {client}")

        elif args.action == "add":
            if args.client:
                limiter.whitelist_add(args.client)
                print(f"Added to whitelist: {args.client}")
            else:
                print("--client required")

        elif args.action == "remove":
            if args.client:
                if limiter.whitelist_remove(args.client):
                    print(f"Removed from whitelist: {args.client}")
                else:
                    print(f"Not in whitelist: {args.client}")
            else:
                print("--client required")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
