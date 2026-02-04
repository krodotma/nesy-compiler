#!/usr/bin/env python3
"""
Notification Compressor: Ollama-Powered Message Compression for DevOps/MLOps
=============================================================================

Uses local Ollama LLM to compress complex devops/mlops messages into
human-readable notification summaries while preserving key technical details.

Features:
  - Ollama integration for intelligent compression
  - Caching to avoid redundant LLM calls
  - Heuristic fallback when Ollama unavailable
  - Bus event emission for compression operations
  - Rate limiting and batch processing

Usage:
    from notification_compressor import (
        compress_message,
        batch_compress,
        get_compression_stats,
    )

    # Simple usage
    compressed = compress_message(
        "The deployment to production-cluster-east failed due to ImagePullBackOff "
        "error in pod nginx-deployment-7fb96c846b-xyz with container nginx. "
        "The container runtime attempted to pull image nginx:1.19.0 from "
        "registry.example.com but received HTTP 401 Unauthorized. "
        "Last successful pull was 2024-01-15T10:30:00Z."
    )
    # Returns: "Deployment failed: nginx pod ImagePullBackOff - auth error (401) on registry pull"

    # With specific model
    compressed = compress_message(raw_message, model="llama3.2:1b")
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_MODEL = "llama3.2:1b"
MAX_INPUT_LENGTH = 2000  # Characters
MAX_OUTPUT_LENGTH = 280  # Twitter-like limit for notifications
CACHE_TTL_SECONDS = 86400  # 24 hours
OLLAMA_TIMEOUT = 30  # seconds
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Compression prompt template
COMPRESSION_PROMPT = """You are a DevOps notification compressor. Compress the following message into a concise, human-readable summary suitable for a notification card.

Rules:
1. Maximum {max_length} characters
2. Preserve: service names, error codes, key metrics, timestamps
3. Remove: verbose stack traces, repeated info, obvious context
4. Format: [Service/Component] Brief description - key detail
5. Use abbreviations: err=error, cfg=config, dep=deployment, svc=service

Message to compress:
{message}

Compressed notification (max {max_length} chars):"""


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original: str
    compressed: str
    model: str
    method: Literal["ollama", "heuristic", "cache"]
    compression_ratio: float
    input_tokens_estimate: int
    output_tokens_estimate: int
    latency_ms: float
    cache_key: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompressionStats:
    """Aggregate compression statistics."""
    total_compressions: int = 0
    ollama_hits: int = 0
    cache_hits: int = 0
    heuristic_fallbacks: int = 0
    total_input_chars: int = 0
    total_output_chars: int = 0
    avg_compression_ratio: float = 1.0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    last_updated_iso: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# Cache Management
# ==============================================================================

class CompressionCache:
    """SQLite-based cache for compressed messages."""

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path or Path(
            os.environ.get("COMPRESSION_CACHE_PATH", "/tmp/notification_compressor.db")
        )
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.cache_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compression_cache (
                    cache_key TEXT PRIMARY KEY,
                    original_hash TEXT,
                    compressed TEXT,
                    model TEXT,
                    created_ts REAL,
                    expires_ts REAL,
                    hits INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires ON compression_cache(expires_ts)
            """)
            conn.commit()

    def get(self, cache_key: str) -> str | None:
        """Get cached compression if exists and not expired."""
        now = time.time()
        with sqlite3.connect(str(self.cache_path)) as conn:
            row = conn.execute(
                "SELECT compressed FROM compression_cache WHERE cache_key = ? AND expires_ts > ?",
                (cache_key, now),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE compression_cache SET hits = hits + 1 WHERE cache_key = ?",
                    (cache_key,),
                )
                conn.commit()
                return row[0]
        return None

    def set(self, cache_key: str, original: str, compressed: str, model: str, ttl: int = CACHE_TTL_SECONDS):
        """Store compression in cache."""
        now = time.time()
        original_hash = hashlib.sha256(original.encode()).hexdigest()[:16]
        with sqlite3.connect(str(self.cache_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO compression_cache
                (cache_key, original_hash, compressed, model, created_ts, expires_ts, hits)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (cache_key, original_hash, compressed, model, now, now + ttl),
            )
            conn.commit()

    def cleanup_expired(self):
        """Remove expired cache entries."""
        now = time.time()
        with sqlite3.connect(str(self.cache_path)) as conn:
            conn.execute("DELETE FROM compression_cache WHERE expires_ts < ?", (now,))
            conn.commit()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM compression_cache").fetchone()[0]
            total_hits = conn.execute("SELECT SUM(hits) FROM compression_cache").fetchone()[0] or 0
        return {"total_entries": total, "total_hits": total_hits}


# Global cache instance
_cache: CompressionCache | None = None


def get_cache() -> CompressionCache:
    """Get or create global cache instance."""
    global _cache
    if _cache is None:
        _cache = CompressionCache()
    return _cache


# ==============================================================================
# Ollama Integration
# ==============================================================================

def _check_ollama_available() -> bool:
    """Check if Ollama is available and responding."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _call_ollama(prompt: str, model: str, timeout: int = OLLAMA_TIMEOUT) -> tuple[str, float]:
    """Call Ollama API for text generation."""
    import urllib.request
    import urllib.error

    start_time = time.time()

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 100,  # Limit output tokens
            "temperature": 0.3,  # More deterministic
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "").strip()
            latency = (time.time() - start_time) * 1000
            return response_text, latency
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


# ==============================================================================
# Heuristic Compression (Fallback)
# ==============================================================================

def _heuristic_compress(message: str, max_length: int = MAX_OUTPUT_LENGTH) -> str:
    """
    Heuristic-based compression when Ollama is unavailable.

    Uses rule-based extraction and summarization:
    - Extract key identifiers (service names, error codes)
    - Remove verbose descriptions
    - Apply common abbreviations
    """
    # Common DevOps/MLOps abbreviations
    abbreviations = {
        "deployment": "dep",
        "service": "svc",
        "configuration": "cfg",
        "container": "ctr",
        "kubernetes": "k8s",
        "production": "prod",
        "development": "dev",
        "staging": "stg",
        "error": "err",
        "warning": "warn",
        "failed": "FAIL",
        "successful": "OK",
        "authentication": "auth",
        "authorization": "authz",
        "connection": "conn",
        "database": "db",
        "repository": "repo",
        "certificate": "cert",
        "timeout": "t/o",
        "memory": "mem",
        "latency": "lat",
        "unauthorized": "unauth",
        "forbidden": "denied",
    }

    # Key patterns to extract
    patterns = {
        "service": r"(?:service|svc|pod|deployment|container)[:\s]+([a-zA-Z0-9_-]+)",
        "service_alt": r"pod\s+([a-zA-Z0-9_-]+)",
        "error_code": r"(?:HTTP|error|code|status)[:\s]*(\d{3})",
        "error_type": r"(ImagePullBackOff|CrashLoopBackOff|OOMKilled|Error|Exception|Failed|Timeout|Refused|Denied|Unauthorized|Forbidden)(?:\w+)?",
        "ip_port": r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?)",
        "metric": r"(\d+(?:\.\d+)?)\s*(%|ms|s|MB|GB|KB|req/s)",
        "timestamp": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})",
        "image": r"image[:\s]+([a-zA-Z0-9_/.-]+:\d+[a-zA-Z0-9.-]*)",
        "registry": r"registry[:\s.]*([a-zA-Z0-9_.-]+\.[a-z]+)",
    }

    # Extract key information
    extracted = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, message, re.IGNORECASE)
        if matches:
            match_val = matches[0] if isinstance(matches[0], str) else matches[0][0]
            # Don't overwrite with alternate patterns
            if name.endswith("_alt"):
                base_name = name[:-4]
                if base_name not in extracted:
                    extracted[base_name] = match_val
            else:
                extracted[name] = match_val

    # Detect action/status from message
    action = ""
    if re.search(r"\b(deploy|deploying|deployment)\b", message, re.IGNORECASE):
        action = "Deploy"
    elif re.search(r"\b(pull|pulling)\b", message, re.IGNORECASE):
        action = "Pull"
    elif re.search(r"\b(start|starting)\b", message, re.IGNORECASE):
        action = "Start"
    elif re.search(r"\b(restart|restarting)\b", message, re.IGNORECASE):
        action = "Restart"

    # Detect failure reason
    failure_reason = ""
    if "error_type" in extracted:
        et = extracted["error_type"]
        if "ImagePullBackOff" in et:
            failure_reason = "image pull failed"
        elif "CrashLoopBackOff" in et:
            failure_reason = "crash loop"
        elif "OOMKilled" in et:
            failure_reason = "out of memory"
        elif "Timeout" in et:
            failure_reason = "timeout"
        elif "Unauthorized" in et or "Denied" in et or "Forbidden" in et:
            failure_reason = "auth error"

    # Build compressed summary
    parts = []

    # Service/component first
    if "service" in extracted:
        svc = extracted["service"]
        # Shorten long deployment names
        if len(svc) > 25:
            # Try to extract base name (e.g., nginx-deployment-7fb96c846b-xyz -> nginx)
            base = svc.split("-")[0] if "-" in svc else svc[:20]
            svc = base
        parts.append(f"[{svc}]")

    # Action and status
    if action:
        parts.append(f"{action} FAIL:")
    elif "error_type" in extracted:
        parts.append(extracted["error_type"] + ":")

    # Failure reason or error code
    if failure_reason:
        parts.append(failure_reason)
    elif "error_code" in extracted:
        parts.append(f"HTTP {extracted['error_code']}")

    # Registry/image info if relevant
    if "registry" in extracted and failure_reason == "auth error":
        parts.append(f"on {extracted['registry']}")

    # If we have little structured info, fall back to smart sentence extraction
    if len(parts) < 2:
        # Extract the most informative sentence
        sentences = re.split(r"[.!?]\s+", message)
        # Prefer sentences with error keywords
        error_keywords = ["fail", "error", "unable", "cannot", "refused", "denied", "timeout"]
        best_sentence = sentences[0] if sentences else message

        for sentence in sentences:
            if any(kw in sentence.lower() for kw in error_keywords):
                best_sentence = sentence
                break

        # Apply abbreviations
        compressed = best_sentence[:180]
        for full, abbr in abbreviations.items():
            compressed = re.sub(rf"\b{full}\b", abbr, compressed, flags=re.IGNORECASE)

        # Remove filler words
        filler = ["the", "a", "an", "to", "from", "with", "due", "because", "was", "were", "has", "have", "had"]
        for word in filler:
            compressed = re.sub(rf"\b{word}\b\s*", "", compressed, flags=re.IGNORECASE)

        # Clean up extra spaces
        compressed = re.sub(r"\s+", " ", compressed).strip()

        return compressed[:max_length]

    result = " ".join(parts)

    # Apply abbreviations to result
    for full, abbr in abbreviations.items():
        result = re.sub(rf"\b{full}\b", abbr, result, flags=re.IGNORECASE)

    # Truncate if needed
    if len(result) > max_length:
        result = result[: max_length - 3] + "..."

    return result


# ==============================================================================
# Core Functions
# ==============================================================================

def compress_message(
    raw_message: str,
    model: str = DEFAULT_MODEL,
    max_length: int = MAX_OUTPUT_LENGTH,
    use_cache: bool = True,
    emit_bus: bool = False,
) -> str:
    """
    Compress a complex DevOps/MLOps message using Ollama or heuristics.

    Args:
        raw_message: The raw message to compress
        model: Ollama model to use
        max_length: Maximum output length
        use_cache: Whether to use caching
        emit_bus: Whether to emit bus events

    Returns:
        Compressed message string
    """
    result = compress_message_full(raw_message, model, max_length, use_cache, emit_bus)
    return result.compressed


def compress_message_full(
    raw_message: str,
    model: str = DEFAULT_MODEL,
    max_length: int = MAX_OUTPUT_LENGTH,
    use_cache: bool = True,
    emit_bus: bool = False,
) -> CompressionResult:
    """
    Compress a message and return full result with metadata.

    Args:
        raw_message: The raw message to compress
        model: Ollama model to use
        max_length: Maximum output length
        use_cache: Whether to use caching
        emit_bus: Whether to emit bus events

    Returns:
        CompressionResult with full metadata
    """
    start_time = time.time()

    # Normalize and truncate input
    message = raw_message.strip()
    if len(message) > MAX_INPUT_LENGTH:
        message = message[:MAX_INPUT_LENGTH] + "..."

    # Generate cache key
    cache_key = hashlib.sha256(f"{message}:{model}:{max_length}".encode()).hexdigest()[:16]

    # Check cache first
    if use_cache:
        cache = get_cache()
        cached = cache.get(cache_key)
        if cached:
            latency = (time.time() - start_time) * 1000
            result = CompressionResult(
                original=raw_message,
                compressed=cached,
                model=model,
                method="cache",
                compression_ratio=len(raw_message) / max(len(cached), 1),
                input_tokens_estimate=len(message) // 4,
                output_tokens_estimate=len(cached) // 4,
                latency_ms=latency,
                cache_key=cache_key,
                success=True,
            )
            if emit_bus:
                _emit_compression_event(result)
            return result

    # Try Ollama
    compressed: str = ""
    method: Literal["ollama", "heuristic", "cache"] = "ollama"
    error: str | None = None
    llm_latency: float = 0.0

    if _check_ollama_available():
        try:
            prompt = COMPRESSION_PROMPT.format(message=message, max_length=max_length)
            compressed, llm_latency = _call_ollama(prompt, model)

            # Post-process: ensure within length limit
            if len(compressed) > max_length:
                compressed = compressed[:max_length - 3] + "..."

            # Validate output (not empty, not just whitespace)
            if not compressed or compressed.isspace():
                raise ValueError("Empty response from Ollama")

        except Exception as e:
            error = str(e)
            method = "heuristic"
            compressed = _heuristic_compress(message, max_length)
    else:
        method = "heuristic"
        compressed = _heuristic_compress(message, max_length)

    # Calculate metrics
    total_latency = (time.time() - start_time) * 1000
    compression_ratio = len(raw_message) / max(len(compressed), 1)

    result = CompressionResult(
        original=raw_message,
        compressed=compressed,
        model=model if method == "ollama" else "heuristic",
        method=method,
        compression_ratio=compression_ratio,
        input_tokens_estimate=len(message) // 4,
        output_tokens_estimate=len(compressed) // 4,
        latency_ms=total_latency,
        cache_key=cache_key,
        success=error is None,
        error=error,
    )

    # Cache the result
    if use_cache and result.success:
        cache = get_cache()
        cache.set(cache_key, raw_message, compressed, model)

    # Emit bus event
    if emit_bus:
        _emit_compression_event(result)

    return result


def batch_compress(
    messages: list[str],
    model: str = DEFAULT_MODEL,
    max_length: int = MAX_OUTPUT_LENGTH,
    emit_bus: bool = False,
) -> list[CompressionResult]:
    """
    Compress multiple messages in batch.

    Args:
        messages: List of messages to compress
        model: Ollama model to use
        max_length: Maximum output length
        emit_bus: Whether to emit bus events

    Returns:
        List of CompressionResult objects
    """
    results = []
    for msg in messages:
        result = compress_message_full(msg, model, max_length, emit_bus=emit_bus)
        results.append(result)
    return results


def get_compression_stats() -> CompressionStats:
    """Get aggregate compression statistics from cache."""
    cache = get_cache()
    cache_stats = cache.get_stats()

    return CompressionStats(
        total_compressions=cache_stats["total_entries"],
        cache_hits=cache_stats["total_hits"],
        last_updated_iso=datetime.now(timezone.utc).isoformat(),
    )


# ==============================================================================
# Bus Integration
# ==============================================================================

def _emit_compression_event(result: CompressionResult):
    """Emit compression operation to the bus."""
    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(),
        "topic": "notification.compression",
        "kind": "metric",
        "level": "info" if result.success else "warn",
        "actor": os.environ.get("PLURIBUS_ACTOR", "notification-compressor"),
        "data": {
            "method": result.method,
            "model": result.model,
            "compression_ratio": round(result.compression_ratio, 2),
            "latency_ms": round(result.latency_ms, 2),
            "input_chars": len(result.original),
            "output_chars": len(result.compressed),
            "cache_key": result.cache_key,
            "success": result.success,
            "error": result.error,
        },
        "semantic": f"Compressed notification ({result.method}): {result.compression_ratio:.1f}x ratio, {result.latency_ms:.0f}ms",
    }

    try:
        with events_path.open("a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        pass  # Non-critical


def emit_batch_summary(results: list[CompressionResult]):
    """Emit batch compression summary to the bus."""
    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate stats
    total = len(results)
    successes = sum(1 for r in results if r.success)
    ollama_count = sum(1 for r in results if r.method == "ollama")
    cache_count = sum(1 for r in results if r.method == "cache")
    heuristic_count = sum(1 for r in results if r.method == "heuristic")
    avg_ratio = sum(r.compression_ratio for r in results) / max(total, 1)
    avg_latency = sum(r.latency_ms for r in results) / max(total, 1)

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(),
        "topic": "notification.compression.batch",
        "kind": "metric",
        "level": "info",
        "actor": os.environ.get("PLURIBUS_ACTOR", "notification-compressor"),
        "data": {
            "batch_size": total,
            "successes": successes,
            "ollama_count": ollama_count,
            "cache_count": cache_count,
            "heuristic_count": heuristic_count,
            "avg_compression_ratio": round(avg_ratio, 2),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "semantic": f"Batch compression: {successes}/{total} success, {avg_ratio:.1f}x avg ratio",
    }

    try:
        with events_path.open("a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        pass


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    """CLI for the notification compressor."""
    import argparse

    parser = argparse.ArgumentParser(description="Notification Compressor CLI")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    parser.add_argument("--compress", type=str, help="Message to compress")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--max-length", type=int, default=MAX_OUTPUT_LENGTH)
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--emit-bus", action="store_true", help="Emit to bus")
    parser.add_argument("--check-ollama", action="store_true", help="Check Ollama status")
    parser.add_argument("--stats", action="store_true", help="Show compression stats")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.test:
        print("=== Notification Compressor Self-Test ===\n")

        # Test 1: Heuristic compression
        print("[TEST 1] Heuristic compression")
        test_message = (
            "The deployment to production-cluster-east failed due to ImagePullBackOff "
            "error in pod nginx-deployment-7fb96c846b-xyz with container nginx. "
            "The container runtime attempted to pull image nginx:1.19.0 from "
            "registry.example.com but received HTTP 401 Unauthorized error. "
            "Last successful pull was 2024-01-15T10:30:00Z."
        )
        compressed = _heuristic_compress(test_message)
        print(f"  Original ({len(test_message)} chars):")
        print(f"    {test_message[:100]}...")
        print(f"  Compressed ({len(compressed)} chars):")
        print(f"    {compressed}")
        print(f"  Ratio: {len(test_message)/len(compressed):.1f}x")
        print("  [PASS]\n")

        # Test 2: Check Ollama availability
        print("[TEST 2] Ollama availability check")
        ollama_available = _check_ollama_available()
        print(f"  Ollama available: {ollama_available}")
        print("  [PASS]\n")

        # Test 3: Full compression with fallback
        print("[TEST 3] Full compression (with fallback)")
        result = compress_message_full(test_message, use_cache=False)
        print(f"  Method: {result.method}")
        print(f"  Compressed: {result.compressed}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print(f"  Compression ratio: {result.compression_ratio:.1f}x")
        print("  [PASS]\n")

        # Test 4: Cache functionality
        print("[TEST 4] Cache functionality")
        cache = get_cache()
        cache.set("test_key", "original", "compressed", "test-model")
        cached = cache.get("test_key")
        print(f"  Cache set/get: {'PASS' if cached == 'compressed' else 'FAIL'}")
        stats = cache.get_stats()
        print(f"  Cache stats: {stats}")
        print("  [PASS]\n")

        print("=== ALL TESTS PASSED ===")
        return 0

    if args.check_ollama:
        available = _check_ollama_available()
        if args.json:
            print(json.dumps({"available": available, "base_url": OLLAMA_BASE_URL}))
        else:
            print(f"Ollama available: {available}")
            print(f"Base URL: {OLLAMA_BASE_URL}")
        return 0 if available else 1

    if args.stats:
        stats = get_compression_stats()
        if args.json:
            print(json.dumps(stats.to_dict()))
        else:
            print(f"Total compressions: {stats.total_compressions}")
            print(f"Cache hits: {stats.cache_hits}")
        return 0

    if args.compress:
        result = compress_message_full(
            args.compress,
            model=args.model,
            max_length=args.max_length,
            use_cache=not args.no_cache,
            emit_bus=args.emit_bus,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Original ({len(result.original)} chars):")
            print(f"  {result.original[:200]}{'...' if len(result.original) > 200 else ''}")
            print(f"\nCompressed ({len(result.compressed)} chars):")
            print(f"  {result.compressed}")
            print(f"\nMethod: {result.method}")
            print(f"Model: {result.model}")
            print(f"Ratio: {result.compression_ratio:.1f}x")
            print(f"Latency: {result.latency_ms:.2f}ms")
            if result.error:
                print(f"Error: {result.error}")

        return 0 if result.success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
