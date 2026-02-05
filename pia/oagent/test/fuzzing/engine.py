#!/usr/bin/env python3
"""
Step 112: Fuzzing Framework

Provides input fuzzing capabilities for discovering edge cases and crashes.

PBTSO Phase: TEST, VERIFY
Bus Topics:
- test.fuzz.run (subscribes)
- test.fuzz.crash (emits)
- test.fuzz.complete (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import random
import string
import struct
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


# ============================================================================
# Constants
# ============================================================================

DEFAULT_ITERATIONS = 1000
DEFAULT_TIMEOUT_S = 5
DEFAULT_MAX_INPUT_SIZE = 10000


class FuzzStrategy(Enum):
    """Fuzzing strategies."""
    RANDOM = "random"  # Pure random generation
    MUTATION = "mutation"  # Mutate seed inputs
    GRAMMAR = "grammar"  # Grammar-based generation
    COVERAGE = "coverage"  # Coverage-guided fuzzing


class InputType(Enum):
    """Types of inputs to fuzz."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BYTES = "bytes"
    LIST = "list"
    DICT = "dict"
    JSON = "json"


class CrashType(Enum):
    """Types of crashes/failures detected."""
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    ASSERTION = "assertion"
    MEMORY = "memory"
    HANG = "hang"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class FuzzConfig:
    """
    Configuration for fuzzing.

    Attributes:
        target_function: Function to fuzz
        input_spec: Specification of input types
        iterations: Number of fuzzing iterations
        timeout_s: Timeout per iteration
        strategy: Fuzzing strategy
        seed_inputs: Initial seed inputs
        max_input_size: Maximum input size
        output_dir: Directory for crash reports
    """
    target_function: Optional[Callable] = None
    target_module: Optional[str] = None
    target_name: Optional[str] = None
    input_spec: Dict[str, InputType] = field(default_factory=dict)
    iterations: int = DEFAULT_ITERATIONS
    timeout_s: float = DEFAULT_TIMEOUT_S
    strategy: FuzzStrategy = FuzzStrategy.MUTATION
    seed_inputs: List[Dict[str, Any]] = field(default_factory=list)
    max_input_size: int = DEFAULT_MAX_INPUT_SIZE
    output_dir: str = ".pluribus/test-agent/fuzzing"
    stop_on_crash: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_module": self.target_module,
            "target_name": self.target_name,
            "input_spec": {k: v.value for k, v in self.input_spec.items()},
            "iterations": self.iterations,
            "timeout_s": self.timeout_s,
            "strategy": self.strategy.value,
            "max_input_size": self.max_input_size,
            "output_dir": self.output_dir,
            "stop_on_crash": self.stop_on_crash,
        }


@dataclass
class FuzzCase:
    """A single fuzzing test case."""
    id: str
    inputs: Dict[str, Any]
    success: bool
    duration_s: float
    output: Optional[Any] = None
    exception: Optional[str] = None
    exception_type: Optional[str] = None
    stack_trace: Optional[str] = None
    crash_type: Optional[CrashType] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "inputs": self._serialize_inputs(self.inputs),
            "success": self.success,
            "duration_s": self.duration_s,
            "exception": self.exception,
            "exception_type": self.exception_type,
            "crash_type": self.crash_type.value if self.crash_type else None,
        }

    def _serialize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize inputs for JSON."""
        result = {}
        for k, v in inputs.items():
            if isinstance(v, bytes):
                result[k] = {"__bytes__": v.hex()}
            else:
                try:
                    json.dumps(v)
                    result[k] = v
                except (TypeError, ValueError):
                    result[k] = str(v)
        return result


@dataclass
class FuzzResult:
    """Complete result of a fuzzing campaign."""
    run_id: str
    config: FuzzConfig
    started_at: float
    completed_at: Optional[float] = None
    total_iterations: int = 0
    crashes: int = 0
    timeouts: int = 0
    unique_crashes: int = 0
    crash_cases: List[FuzzCase] = field(default_factory=list)
    all_cases: List[FuzzCase] = field(default_factory=list)
    crash_hashes: Set[str] = field(default_factory=set)

    @property
    def duration_s(self) -> float:
        """Get total duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "total_iterations": self.total_iterations,
            "crashes": self.crashes,
            "timeouts": self.timeouts,
            "unique_crashes": self.unique_crashes,
            "crash_cases": [c.to_dict() for c in self.crash_cases],
        }


# ============================================================================
# Input Generators
# ============================================================================

class InputGenerator:
    """Generates fuzzed inputs."""

    def __init__(self, max_size: int = DEFAULT_MAX_INPUT_SIZE):
        self.max_size = max_size

    def generate(self, input_type: InputType) -> Any:
        """Generate a random input of the given type."""
        generators = {
            InputType.STRING: self._gen_string,
            InputType.INTEGER: self._gen_integer,
            InputType.FLOAT: self._gen_float,
            InputType.BYTES: self._gen_bytes,
            InputType.LIST: self._gen_list,
            InputType.DICT: self._gen_dict,
            InputType.JSON: self._gen_json,
        }
        return generators[input_type]()

    def _gen_string(self) -> str:
        """Generate random string."""
        length = random.randint(0, min(1000, self.max_size))

        # Mix of strategies
        strategy = random.choice([
            "ascii",
            "unicode",
            "special",
            "boundary",
            "format_string",
        ])

        if strategy == "ascii":
            return ''.join(random.choices(string.printable, k=length))
        elif strategy == "unicode":
            # Include various unicode ranges
            chars = []
            for _ in range(length):
                if random.random() < 0.7:
                    chars.append(chr(random.randint(0x0000, 0xFFFF)))
                else:
                    chars.append(chr(random.randint(0x10000, 0x10FFFF)))
            return ''.join(chars)
        elif strategy == "special":
            specials = [
                "",  # Empty
                "\x00",  # Null byte
                "\n" * 100,  # Many newlines
                " " * 1000,  # Many spaces
                "A" * 10000,  # Long string
                "../../../etc/passwd",  # Path traversal
                "<script>alert(1)</script>",  # XSS
                "'; DROP TABLE users; --",  # SQL injection
                "${7*7}",  # Template injection
                "%s%s%s%s%s",  # Format string
            ]
            return random.choice(specials)
        elif strategy == "boundary":
            return random.choice([
                "",
                "a",
                "a" * 255,
                "a" * 256,
                "a" * 65535,
            ])
        else:  # format_string
            format_chars = ["%s", "%d", "%x", "%n", "%.10000s"]
            return ''.join(random.choices(format_chars, k=random.randint(1, 50)))

    def _gen_integer(self) -> int:
        """Generate random integer."""
        boundaries = [
            0, 1, -1,
            127, 128, -128, -129,  # int8
            255, 256, -256,  # uint8
            32767, 32768, -32768, -32769,  # int16
            65535, 65536,  # uint16
            2147483647, 2147483648, -2147483648, -2147483649,  # int32
            4294967295, 4294967296,  # uint32
            2**63 - 1, 2**63, -2**63, -2**63 - 1,  # int64
        ]

        if random.random() < 0.3:
            return random.choice(boundaries)
        else:
            return random.randint(-2**63, 2**63)

    def _gen_float(self) -> float:
        """Generate random float."""
        specials = [
            0.0, -0.0, 1.0, -1.0,
            float('inf'), float('-inf'), float('nan'),
            1e-308, 1e308,  # Near limits
            2.2250738585072014e-308,  # Min normal
            1.7976931348623157e+308,  # Max finite
        ]

        if random.random() < 0.3:
            return random.choice(specials)
        else:
            # Generate random float using struct
            bytes_val = struct.pack('d', random.uniform(-1e100, 1e100))
            return struct.unpack('d', bytes_val)[0]

    def _gen_bytes(self) -> bytes:
        """Generate random bytes."""
        length = random.randint(0, min(1000, self.max_size))

        strategy = random.choice(["random", "null", "pattern"])

        if strategy == "random":
            return bytes(random.randint(0, 255) for _ in range(length))
        elif strategy == "null":
            return b'\x00' * length
        else:
            pattern = bytes([random.randint(0, 255) for _ in range(4)])
            return (pattern * (length // 4 + 1))[:length]

    def _gen_list(self) -> List[Any]:
        """Generate random list."""
        length = random.randint(0, 100)

        if length == 0:
            return []

        element_type = random.choice([
            InputType.STRING, InputType.INTEGER, InputType.FLOAT
        ])
        return [self.generate(element_type) for _ in range(length)]

    def _gen_dict(self) -> Dict[str, Any]:
        """Generate random dictionary."""
        length = random.randint(0, 50)

        result = {}
        for _ in range(length):
            key = self._gen_string()[:100]
            value_type = random.choice([
                InputType.STRING, InputType.INTEGER, InputType.FLOAT
            ])
            result[key] = self.generate(value_type)

        return result

    def _gen_json(self) -> str:
        """Generate random JSON string."""
        data = self._gen_dict()
        try:
            return json.dumps(data)
        except (TypeError, ValueError):
            return "{}"


class InputMutator:
    """Mutates existing inputs."""

    def mutate(self, value: Any) -> Any:
        """Mutate a value."""
        if isinstance(value, str):
            return self._mutate_string(value)
        elif isinstance(value, int):
            return self._mutate_integer(value)
        elif isinstance(value, float):
            return self._mutate_float(value)
        elif isinstance(value, bytes):
            return self._mutate_bytes(value)
        elif isinstance(value, list):
            return self._mutate_list(value)
        elif isinstance(value, dict):
            return self._mutate_dict(value)
        return value

    def _mutate_string(self, s: str) -> str:
        """Mutate a string."""
        if not s:
            return random.choice(["", "x", "\x00", "A" * 100])

        mutation = random.choice([
            "flip_char",
            "insert",
            "delete",
            "duplicate",
            "swap",
        ])

        if mutation == "flip_char":
            idx = random.randint(0, len(s) - 1)
            new_char = chr((ord(s[idx]) + random.randint(1, 255)) % 0x10FFFF)
            return s[:idx] + new_char + s[idx + 1:]
        elif mutation == "insert":
            idx = random.randint(0, len(s))
            return s[:idx] + chr(random.randint(0, 255)) + s[idx:]
        elif mutation == "delete":
            idx = random.randint(0, len(s) - 1)
            return s[:idx] + s[idx + 1:]
        elif mutation == "duplicate":
            return s * random.randint(2, 10)
        else:  # swap
            if len(s) < 2:
                return s
            i, j = random.sample(range(len(s)), 2)
            chars = list(s)
            chars[i], chars[j] = chars[j], chars[i]
            return ''.join(chars)

    def _mutate_integer(self, n: int) -> int:
        """Mutate an integer."""
        mutation = random.choice([
            "add", "subtract", "multiply", "negate", "boundary"
        ])

        if mutation == "add":
            return n + random.randint(1, 1000)
        elif mutation == "subtract":
            return n - random.randint(1, 1000)
        elif mutation == "multiply":
            return n * random.randint(2, 10)
        elif mutation == "negate":
            return -n
        else:  # boundary
            return random.choice([0, 1, -1, 2**31 - 1, -2**31])

    def _mutate_float(self, f: float) -> float:
        """Mutate a float."""
        mutation = random.choice([
            "scale", "negate", "special", "epsilon"
        ])

        if mutation == "scale":
            return f * random.uniform(0.1, 10.0)
        elif mutation == "negate":
            return -f
        elif mutation == "special":
            return random.choice([0.0, float('inf'), float('-inf'), float('nan')])
        else:  # epsilon
            import sys
            return f + sys.float_info.epsilon

    def _mutate_bytes(self, b: bytes) -> bytes:
        """Mutate bytes."""
        if not b:
            return bytes([random.randint(0, 255)])

        b = bytearray(b)
        mutation = random.choice(["flip", "insert", "delete"])

        if mutation == "flip":
            idx = random.randint(0, len(b) - 1)
            b[idx] = b[idx] ^ random.randint(1, 255)
        elif mutation == "insert":
            idx = random.randint(0, len(b))
            b.insert(idx, random.randint(0, 255))
        else:  # delete
            if len(b) > 1:
                del b[random.randint(0, len(b) - 1)]

        return bytes(b)

    def _mutate_list(self, lst: List[Any]) -> List[Any]:
        """Mutate a list."""
        lst = list(lst)

        if not lst:
            return [None]

        mutation = random.choice([
            "mutate_element", "insert", "delete", "shuffle"
        ])

        if mutation == "mutate_element":
            idx = random.randint(0, len(lst) - 1)
            lst[idx] = self.mutate(lst[idx])
        elif mutation == "insert":
            lst.insert(random.randint(0, len(lst)), None)
        elif mutation == "delete":
            if len(lst) > 0:
                del lst[random.randint(0, len(lst) - 1)]
        else:  # shuffle
            random.shuffle(lst)

        return lst

    def _mutate_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a dictionary."""
        d = dict(d)

        if not d:
            return {"key": "value"}

        mutation = random.choice([
            "mutate_value", "add_key", "remove_key"
        ])

        if mutation == "mutate_value":
            key = random.choice(list(d.keys()))
            d[key] = self.mutate(d[key])
        elif mutation == "add_key":
            d[f"fuzz_{random.randint(0, 1000)}"] = random.choice([
                None, 0, "", [], {}
            ])
        else:  # remove_key
            if len(d) > 0:
                del d[random.choice(list(d.keys()))]

        return d


# ============================================================================
# Bus Interface
# ============================================================================

class FuzzBus:
    """Bus interface for fuzzing."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event_with_meta) + "\n")
        except IOError:
            pass


# ============================================================================
# Fuzz Engine
# ============================================================================

class FuzzEngine:
    """
    Orchestrates fuzzing campaigns for discovering edge cases.

    Fuzzing works by:
    1. Generating random/mutated inputs
    2. Running target function with fuzzed inputs
    3. Detecting crashes, exceptions, and unexpected behavior
    4. Minimizing crash inputs for reproducibility

    PBTSO Phase: TEST, VERIFY
    Bus Topics: test.fuzz.run, test.fuzz.crash, test.fuzz.complete
    """

    BUS_TOPICS = {
        "run": "test.fuzz.run",
        "crash": "test.fuzz.crash",
        "complete": "test.fuzz.complete",
        "progress": "test.fuzz.progress",
    }

    def __init__(self, bus=None):
        """
        Initialize the fuzz engine.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or FuzzBus()
        self.generator = InputGenerator()
        self.mutator = InputMutator()

    def fuzz(self, config: FuzzConfig) -> FuzzResult:
        """
        Execute a fuzzing campaign.

        Args:
            config: Fuzzing configuration

        Returns:
            FuzzResult with complete results
        """
        run_id = str(uuid.uuid4())
        result = FuzzResult(
            run_id=run_id,
            config=config,
            started_at=time.time(),
        )

        # Emit start event
        self._emit_event("run", {
            "run_id": run_id,
            "status": "started",
            "iterations": config.iterations,
        })

        # Get target function
        target_fn = self._get_target_function(config)
        if target_fn is None:
            result.completed_at = time.time()
            return result

        # Initialize seed corpus
        corpus = list(config.seed_inputs) if config.seed_inputs else []

        # Run fuzzing iterations
        for i in range(config.iterations):
            # Generate inputs
            inputs = self._generate_inputs(config, corpus)

            # Run test case
            case = self._run_test_case(target_fn, inputs, config)
            result.total_iterations += 1
            result.all_cases.append(case)

            if not case.success:
                # Hash the crash for deduplication
                crash_hash = self._hash_crash(case)

                if crash_hash not in result.crash_hashes:
                    result.crash_hashes.add(crash_hash)
                    result.unique_crashes += 1
                    result.crash_cases.append(case)

                    # Emit crash event
                    self._emit_event("crash", {
                        "run_id": run_id,
                        "crash_id": case.id,
                        "exception_type": case.exception_type,
                        "inputs": case.to_dict()["inputs"],
                    })

                    # Add to corpus for mutation-based fuzzing
                    if config.strategy == FuzzStrategy.MUTATION:
                        corpus.append(inputs)

                if case.crash_type == CrashType.TIMEOUT:
                    result.timeouts += 1
                else:
                    result.crashes += 1

                if config.stop_on_crash:
                    break

            # Progress update
            if i % 100 == 0:
                self._emit_event("progress", {
                    "run_id": run_id,
                    "iteration": i,
                    "crashes": result.crashes,
                    "unique_crashes": result.unique_crashes,
                })

        result.completed_at = time.time()

        # Emit complete event
        self._emit_event("complete", {
            "run_id": run_id,
            "status": "completed",
            "total_iterations": result.total_iterations,
            "crashes": result.crashes,
            "unique_crashes": result.unique_crashes,
            "duration_s": result.duration_s,
        })

        # Save report
        self._save_report(result, config.output_dir)

        return result

    def _get_target_function(self, config: FuzzConfig) -> Optional[Callable]:
        """Get the target function to fuzz."""
        if config.target_function:
            return config.target_function

        if config.target_module and config.target_name:
            import importlib
            try:
                module = importlib.import_module(config.target_module)
                return getattr(module, config.target_name)
            except (ImportError, AttributeError):
                return None

        return None

    def _generate_inputs(
        self,
        config: FuzzConfig,
        corpus: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate inputs for a fuzz case."""
        inputs = {}

        if config.strategy == FuzzStrategy.MUTATION and corpus:
            # Mutate a seed input
            seed = random.choice(corpus)
            for name, input_type in config.input_spec.items():
                if name in seed:
                    inputs[name] = self.mutator.mutate(seed[name])
                else:
                    inputs[name] = self.generator.generate(input_type)
        else:
            # Pure random generation
            for name, input_type in config.input_spec.items():
                inputs[name] = self.generator.generate(input_type)

        return inputs

    def _run_test_case(
        self,
        target_fn: Callable,
        inputs: Dict[str, Any],
        config: FuzzConfig,
    ) -> FuzzCase:
        """Run a single fuzz test case."""
        case_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Run with timeout using signal (Unix) or threading (Windows)
            result = self._run_with_timeout(target_fn, inputs, config.timeout_s)
            duration = time.time() - start_time

            return FuzzCase(
                id=case_id,
                inputs=inputs,
                success=True,
                duration_s=duration,
                output=result,
            )

        except TimeoutError:
            return FuzzCase(
                id=case_id,
                inputs=inputs,
                success=False,
                duration_s=config.timeout_s,
                crash_type=CrashType.TIMEOUT,
                exception="Timeout",
                exception_type="TimeoutError",
            )

        except AssertionError as e:
            return FuzzCase(
                id=case_id,
                inputs=inputs,
                success=False,
                duration_s=time.time() - start_time,
                crash_type=CrashType.ASSERTION,
                exception=str(e),
                exception_type="AssertionError",
                stack_trace=traceback.format_exc(),
            )

        except Exception as e:
            return FuzzCase(
                id=case_id,
                inputs=inputs,
                success=False,
                duration_s=time.time() - start_time,
                crash_type=CrashType.EXCEPTION,
                exception=str(e),
                exception_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
            )

    def _run_with_timeout(
        self,
        fn: Callable,
        inputs: Dict[str, Any],
        timeout_s: float,
    ) -> Any:
        """Run function with timeout."""
        import threading

        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = fn(**inputs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout_s)

        if thread.is_alive():
            raise TimeoutError(f"Execution exceeded {timeout_s}s")

        if exception[0]:
            raise exception[0]

        return result[0]

    def _hash_crash(self, case: FuzzCase) -> str:
        """Hash a crash for deduplication."""
        # Hash based on exception type and stack trace signature
        sig_parts = [
            case.exception_type or "",
            case.exception or "",
        ]

        if case.stack_trace:
            # Extract just the file:line info from stack trace
            import re
            lines = re.findall(r'File "([^"]+)", line (\d+)', case.stack_trace)
            sig_parts.extend([f"{f}:{l}" for f, l in lines[-3:]])

        signature = "|".join(sig_parts)
        return hashlib.md5(signature.encode()).hexdigest()

    def _save_report(self, result: FuzzResult, output_dir: str) -> None:
        """Save fuzzing report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_file = output_path / f"fuzz_report_{result.run_id}.json"
        with open(report_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save crash reproducers
        crashes_dir = output_path / "crashes"
        crashes_dir.mkdir(exist_ok=True)

        for case in result.crash_cases:
            crash_file = crashes_dir / f"crash_{case.id}.json"
            with open(crash_file, "w") as f:
                json.dump(case.to_dict(), f, indent=2)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.fuzz.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "fuzzing",
            "actor": "test-agent",
            "data": data,
        })

    async def fuzz_async(self, config: FuzzConfig) -> FuzzResult:
        """Async version of fuzzing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fuzz, config)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Fuzzing Framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Fuzzing Framework")
    parser.add_argument("module", help="Target module (e.g., mypackage.mymodule)")
    parser.add_argument("function", help="Target function name")
    parser.add_argument("--iterations", "-n", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/fuzzing")
    parser.add_argument("--stop-on-crash", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Auto-detect input spec from function signature
    import importlib
    try:
        module = importlib.import_module(args.module)
        fn = getattr(module, args.function)
        sig = inspect.signature(fn)

        input_spec = {}
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                # Map type annotations to InputType
                if param.annotation == str:
                    input_spec[param_name] = InputType.STRING
                elif param.annotation == int:
                    input_spec[param_name] = InputType.INTEGER
                elif param.annotation == float:
                    input_spec[param_name] = InputType.FLOAT
                elif param.annotation == bytes:
                    input_spec[param_name] = InputType.BYTES
                elif param.annotation == list:
                    input_spec[param_name] = InputType.LIST
                elif param.annotation == dict:
                    input_spec[param_name] = InputType.DICT
                else:
                    input_spec[param_name] = InputType.STRING
            else:
                input_spec[param_name] = InputType.STRING

    except (ImportError, AttributeError) as e:
        print(f"Error loading function: {e}")
        exit(1)

    config = FuzzConfig(
        target_module=args.module,
        target_name=args.function,
        input_spec=input_spec,
        iterations=args.iterations,
        timeout_s=args.timeout,
        output_dir=args.output,
        stop_on_crash=args.stop_on_crash,
    )

    engine = FuzzEngine()
    result = engine.fuzz(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Fuzzing Complete")
        print(f"{'='*60}")
        print(f"Run ID: {result.run_id}")
        print(f"Duration: {result.duration_s:.2f}s")
        print(f"Iterations: {result.total_iterations}")
        print(f"Crashes: {result.crashes}")
        print(f"Unique Crashes: {result.unique_crashes}")
        print(f"Timeouts: {result.timeouts}")
        print(f"{'='*60}")

        if result.unique_crashes > 0:
            print(f"\nCrash reports saved to: {config.output_dir}/crashes/")
            exit(1)


if __name__ == "__main__":
    main()
