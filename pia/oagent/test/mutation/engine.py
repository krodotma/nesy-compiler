#!/usr/bin/env python3
"""
Step 111: Mutation Testing Engine

Orchestrates mutation testing to assess test suite quality by introducing
code mutations and checking if tests detect them.

PBTSO Phase: VERIFY, TEST
Bus Topics:
- test.mutation.run (subscribes)
- test.mutation.results (emits)
- test.mutation.progress (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .generator import MutantGenerator, Mutant, MutationType


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TIMEOUT_S = 60
DEFAULT_PARALLEL_WORKERS = 4
MUTATION_SCORE_THRESHOLD = 0.8


class MutationStatus(Enum):
    """Status of a mutation test."""
    PENDING = "pending"
    KILLED = "killed"  # Test detected the mutation
    SURVIVED = "survived"  # Mutation was not detected
    TIMEOUT = "timeout"
    ERROR = "error"
    EQUIVALENT = "equivalent"  # Functionally equivalent mutation


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class MutationConfig:
    """
    Configuration for mutation testing.

    Attributes:
        source_paths: Paths to source files to mutate
        test_paths: Paths to test files
        mutation_types: Types of mutations to apply
        parallel_workers: Number of parallel mutation workers
        timeout_s: Timeout for each mutation test
        skip_equivalent: Attempt to detect equivalent mutants
        sample_ratio: Ratio of mutants to test (for large codebases)
        output_dir: Directory for mutation reports
    """
    source_paths: List[str]
    test_paths: List[str]
    mutation_types: List[MutationType] = field(default_factory=lambda: [
        MutationType.ARITHMETIC,
        MutationType.RELATIONAL,
        MutationType.LOGICAL,
        MutationType.BOUNDARY,
    ])
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    timeout_s: int = DEFAULT_TIMEOUT_S
    skip_equivalent: bool = True
    sample_ratio: float = 1.0
    output_dir: str = ".pluribus/test-agent/mutations"
    test_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_paths": self.source_paths,
            "test_paths": self.test_paths,
            "mutation_types": [m.value for m in self.mutation_types],
            "parallel_workers": self.parallel_workers,
            "timeout_s": self.timeout_s,
            "skip_equivalent": self.skip_equivalent,
            "sample_ratio": self.sample_ratio,
            "output_dir": self.output_dir,
            "test_command": self.test_command,
        }


@dataclass
class MutantResult:
    """Result of testing a single mutant."""
    mutant_id: str
    mutant: Mutant
    status: MutationStatus
    duration_s: float
    test_output: str = ""
    killing_test: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mutant_id": self.mutant_id,
            "file_path": self.mutant.file_path,
            "line_number": self.mutant.line_number,
            "mutation_type": self.mutant.mutation_type.value,
            "original_code": self.mutant.original_code,
            "mutated_code": self.mutant.mutated_code,
            "status": self.status.value,
            "duration_s": self.duration_s,
            "killing_test": self.killing_test,
            "error_message": self.error_message,
        }


@dataclass
class MutationResult:
    """Complete result of a mutation testing run."""
    run_id: str
    config: MutationConfig
    started_at: float
    completed_at: Optional[float] = None
    total_mutants: int = 0
    killed: int = 0
    survived: int = 0
    timeout: int = 0
    errors: int = 0
    equivalent: int = 0
    mutant_results: List[MutantResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mutation_score(self) -> float:
        """Calculate mutation score (killed / (total - equivalent))."""
        testable = self.total_mutants - self.equivalent
        if testable == 0:
            return 0.0
        return self.killed / testable

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
            "total_mutants": self.total_mutants,
            "killed": self.killed,
            "survived": self.survived,
            "timeout": self.timeout,
            "errors": self.errors,
            "equivalent": self.equivalent,
            "mutation_score": self.mutation_score,
            "mutant_results": [r.to_dict() for r in self.mutant_results],
            "metadata": self.metadata,
        }

    def get_surviving_mutants(self) -> List[MutantResult]:
        """Get list of surviving mutants (potential weak spots in tests)."""
        return [r for r in self.mutant_results if r.status == MutationStatus.SURVIVED]


# ============================================================================
# Bus Interface
# ============================================================================

class MutationBus:
    """Bus interface for mutation testing."""

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
# Mutation Testing Engine
# ============================================================================

class MutationEngine:
    """
    Orchestrates mutation testing to assess test quality.

    Mutation testing works by:
    1. Generating code mutations (small changes)
    2. Running tests against each mutant
    3. If tests fail, the mutant is "killed" (good)
    4. If tests pass, the mutant "survives" (indicates weak tests)

    PBTSO Phase: VERIFY, TEST
    Bus Topics: test.mutation.run, test.mutation.results
    """

    BUS_TOPICS = {
        "run": "test.mutation.run",
        "results": "test.mutation.results",
        "progress": "test.mutation.progress",
        "mutant_tested": "test.mutant.tested",
    }

    def __init__(self, bus=None):
        """
        Initialize the mutation testing engine.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or MutationBus()
        self.generator = MutantGenerator()
        self._active_runs: Dict[str, MutationResult] = {}

    def run_mutation_testing(self, config: MutationConfig) -> MutationResult:
        """
        Execute mutation testing.

        Args:
            config: Mutation testing configuration

        Returns:
            MutationResult with complete results
        """
        run_id = str(uuid.uuid4())
        result = MutationResult(
            run_id=run_id,
            config=config,
            started_at=time.time(),
        )
        self._active_runs[run_id] = result

        # Emit start event
        self._emit_event("run", {
            "run_id": run_id,
            "status": "started",
            "config": config.to_dict(),
        })

        try:
            # Generate all mutants
            all_mutants = self._generate_mutants(config)
            result.total_mutants = len(all_mutants)

            # Sample if needed
            if config.sample_ratio < 1.0:
                import random
                sample_size = int(len(all_mutants) * config.sample_ratio)
                all_mutants = random.sample(all_mutants, sample_size)

            # Test mutants
            if config.parallel_workers > 1:
                mutant_results = self._test_mutants_parallel(
                    all_mutants, config
                )
            else:
                mutant_results = self._test_mutants_sequential(
                    all_mutants, config
                )

            # Aggregate results
            result.mutant_results = mutant_results
            for mr in mutant_results:
                if mr.status == MutationStatus.KILLED:
                    result.killed += 1
                elif mr.status == MutationStatus.SURVIVED:
                    result.survived += 1
                elif mr.status == MutationStatus.TIMEOUT:
                    result.timeout += 1
                elif mr.status == MutationStatus.ERROR:
                    result.errors += 1
                elif mr.status == MutationStatus.EQUIVALENT:
                    result.equivalent += 1

        except Exception as e:
            result.metadata["error"] = str(e)

        result.completed_at = time.time()

        # Emit completion event
        self._emit_event("results", {
            "run_id": run_id,
            "status": "completed",
            "mutation_score": result.mutation_score,
            "total_mutants": result.total_mutants,
            "killed": result.killed,
            "survived": result.survived,
            "duration_s": result.duration_s,
        })

        # Save report
        self._save_report(result, config.output_dir)

        return result

    def _generate_mutants(self, config: MutationConfig) -> List[Mutant]:
        """Generate all mutants for the given source files."""
        all_mutants = []

        for source_path in config.source_paths:
            path = Path(source_path)
            if path.is_file():
                files = [path]
            else:
                files = list(path.rglob("*.py"))

            for file_path in files:
                try:
                    source_code = file_path.read_text()
                    mutants = self.generator.generate_mutants(
                        source_code,
                        str(file_path),
                        config.mutation_types,
                    )
                    all_mutants.extend(mutants)
                except Exception as e:
                    # Skip files that can't be parsed
                    continue

        return all_mutants

    def _test_mutants_sequential(
        self,
        mutants: List[Mutant],
        config: MutationConfig,
    ) -> List[MutantResult]:
        """Test mutants sequentially."""
        results = []

        for i, mutant in enumerate(mutants):
            result = self._test_single_mutant(mutant, config)
            results.append(result)

            # Emit progress
            self._emit_event("progress", {
                "completed": i + 1,
                "total": len(mutants),
                "current_file": mutant.file_path,
                "status": result.status.value,
            })

        return results

    def _test_mutants_parallel(
        self,
        mutants: List[Mutant],
        config: MutationConfig,
    ) -> List[MutantResult]:
        """Test mutants in parallel using process pool."""
        results = []

        with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
            future_to_mutant = {
                executor.submit(
                    _test_mutant_in_process,
                    mutant,
                    config.test_paths,
                    config.test_command,
                    config.timeout_s,
                ): mutant
                for mutant in mutants
            }

            completed = 0
            for future in as_completed(future_to_mutant):
                mutant = future_to_mutant[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(MutantResult(
                        mutant_id=str(uuid.uuid4()),
                        mutant=mutant,
                        status=MutationStatus.ERROR,
                        duration_s=0,
                        error_message=str(e),
                    ))

                completed += 1
                self._emit_event("progress", {
                    "completed": completed,
                    "total": len(mutants),
                    "current_file": mutant.file_path,
                })

        return results

    def _test_single_mutant(
        self,
        mutant: Mutant,
        config: MutationConfig,
    ) -> MutantResult:
        """Test a single mutant."""
        start_time = time.time()
        mutant_id = str(uuid.uuid4())

        # Create temporary directory for mutated code
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="mutation_")

            # Copy source to temp and apply mutation
            original_path = Path(mutant.file_path)
            temp_path = Path(temp_dir) / original_path.name

            # Write mutated code
            temp_path.write_text(mutant.mutated_code)

            # Run tests
            test_command = config.test_command or self._build_test_command(
                config.test_paths
            )

            # Modify PYTHONPATH to use mutated file
            env = os.environ.copy()
            env["PYTHONPATH"] = str(temp_dir) + ":" + env.get("PYTHONPATH", "")

            try:
                result = subprocess.run(
                    test_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_s,
                    env=env,
                    cwd=original_path.parent,
                )

                duration = time.time() - start_time

                if result.returncode != 0:
                    # Tests failed = mutant killed
                    return MutantResult(
                        mutant_id=mutant_id,
                        mutant=mutant,
                        status=MutationStatus.KILLED,
                        duration_s=duration,
                        test_output=result.stdout + result.stderr,
                        killing_test=self._extract_failing_test(result.stdout),
                    )
                else:
                    # Tests passed = mutant survived
                    return MutantResult(
                        mutant_id=mutant_id,
                        mutant=mutant,
                        status=MutationStatus.SURVIVED,
                        duration_s=duration,
                        test_output=result.stdout,
                    )

            except subprocess.TimeoutExpired:
                return MutantResult(
                    mutant_id=mutant_id,
                    mutant=mutant,
                    status=MutationStatus.TIMEOUT,
                    duration_s=config.timeout_s,
                )

        except Exception as e:
            return MutantResult(
                mutant_id=mutant_id,
                mutant=mutant,
                status=MutationStatus.ERROR,
                duration_s=time.time() - start_time,
                error_message=str(e),
            )
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_test_command(self, test_paths: List[str]) -> str:
        """Build default test command."""
        paths = " ".join(test_paths)
        return f"python -m pytest {paths} -x --tb=short -q"

    def _extract_failing_test(self, output: str) -> Optional[str]:
        """Extract the name of the failing test from pytest output."""
        # Look for FAILED test pattern
        import re
        match = re.search(r"FAILED\s+(\S+)", output)
        if match:
            return match.group(1)
        return None

    def _save_report(self, result: MutationResult, output_dir: str) -> None:
        """Save mutation testing report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / f"mutation_report_{result.run_id}.json"
        with open(report_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Generate summary
        summary_file = output_path / f"mutation_summary_{result.run_id}.md"
        with open(summary_file, "w") as f:
            f.write(f"# Mutation Testing Report\n\n")
            f.write(f"**Run ID**: {result.run_id}\n")
            f.write(f"**Duration**: {result.duration_s:.2f}s\n\n")
            f.write(f"## Score\n\n")
            f.write(f"**Mutation Score**: {result.mutation_score:.1%}\n\n")
            f.write(f"| Metric | Count |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Mutants | {result.total_mutants} |\n")
            f.write(f"| Killed | {result.killed} |\n")
            f.write(f"| Survived | {result.survived} |\n")
            f.write(f"| Timeout | {result.timeout} |\n")
            f.write(f"| Errors | {result.errors} |\n\n")

            if result.survived > 0:
                f.write(f"## Surviving Mutants (Weak Spots)\n\n")
                for mr in result.get_surviving_mutants()[:20]:
                    f.write(f"### {mr.mutant.file_path}:{mr.mutant.line_number}\n")
                    f.write(f"- Type: {mr.mutant.mutation_type.value}\n")
                    f.write(f"- Original: `{mr.mutant.original_code}`\n")
                    f.write(f"- Mutated: `{mr.mutant.mutated_code}`\n\n")

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.mutation.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "mutation_testing",
            "actor": "test-agent",
            "data": data,
        })

    async def run_mutation_testing_async(
        self,
        config: MutationConfig,
    ) -> MutationResult:
        """Async version of mutation testing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.run_mutation_testing,
            config,
        )


# ============================================================================
# Helper for Parallel Execution
# ============================================================================

def _test_mutant_in_process(
    mutant: Mutant,
    test_paths: List[str],
    test_command: Optional[str],
    timeout_s: int,
) -> MutantResult:
    """
    Test a mutant in a separate process.

    This function is designed to be called via ProcessPoolExecutor.
    """
    import tempfile
    import shutil
    import subprocess
    import os

    start_time = time.time()
    mutant_id = str(uuid.uuid4())

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="mutation_")

        original_path = Path(mutant.file_path)
        temp_path = Path(temp_dir) / original_path.name

        temp_path.write_text(mutant.mutated_code)

        if test_command:
            cmd = test_command
        else:
            paths = " ".join(test_paths)
            cmd = f"python -m pytest {paths} -x --tb=short -q"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(temp_dir) + ":" + env.get("PYTHONPATH", "")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
                cwd=original_path.parent,
            )

            duration = time.time() - start_time

            if result.returncode != 0:
                return MutantResult(
                    mutant_id=mutant_id,
                    mutant=mutant,
                    status=MutationStatus.KILLED,
                    duration_s=duration,
                    test_output=result.stdout + result.stderr,
                )
            else:
                return MutantResult(
                    mutant_id=mutant_id,
                    mutant=mutant,
                    status=MutationStatus.SURVIVED,
                    duration_s=duration,
                    test_output=result.stdout,
                )

        except subprocess.TimeoutExpired:
            return MutantResult(
                mutant_id=mutant_id,
                mutant=mutant,
                status=MutationStatus.TIMEOUT,
                duration_s=timeout_s,
            )

    except Exception as e:
        return MutantResult(
            mutant_id=mutant_id,
            mutant=mutant,
            status=MutationStatus.ERROR,
            duration_s=time.time() - start_time,
            error_message=str(e),
        )
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Mutation Testing Engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Mutation Testing Engine")
    parser.add_argument("source_paths", nargs="+", help="Source file/directory paths")
    parser.add_argument("--tests", "-t", nargs="+", required=True, help="Test paths")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per mutant")
    parser.add_argument("--sample", type=float, default=1.0, help="Sample ratio (0-1)")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/mutations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = MutationConfig(
        source_paths=args.source_paths,
        test_paths=args.tests,
        parallel_workers=args.workers,
        timeout_s=args.timeout,
        sample_ratio=args.sample,
        output_dir=args.output,
    )

    engine = MutationEngine()
    result = engine.run_mutation_testing(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Mutation Testing Complete")
        print(f"{'='*60}")
        print(f"Run ID: {result.run_id}")
        print(f"Duration: {result.duration_s:.2f}s")
        print(f"Total Mutants: {result.total_mutants}")
        print(f"Killed: {result.killed}")
        print(f"Survived: {result.survived}")
        print(f"Mutation Score: {result.mutation_score:.1%}")
        print(f"{'='*60}")

        if result.mutation_score < MUTATION_SCORE_THRESHOLD:
            print(f"\nWARNING: Mutation score below {MUTATION_SCORE_THRESHOLD:.0%}")
            print("Consider adding more tests for the surviving mutants.")
            exit(1)


if __name__ == "__main__":
    main()
