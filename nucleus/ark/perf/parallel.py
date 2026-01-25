#!/usr/bin/env python3
"""
parallel.py - Parallel Gate Execution

P2-082: Implement gate parallelization
P2-083: Add lazy evaluation for unused gates
P2-088: Implement async commit pipeline

Implements parallel and async execution for gates.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from functools import lru_cache

logger = logging.getLogger("ARK.Perf.Parallel")


@dataclass
class GateResult:
    """Result from gate execution."""
    gate_name: str
    passed: bool
    duration_ms: float
    lazy: bool = False
    error: Optional[str] = None


class LazyGate:
    """
    Lazy gate wrapper - only executes when result is accessed.
    
    P2-083: Add lazy evaluation for unused gates
    """
    
    def __init__(self, gate_fn: Callable, *args, **kwargs):
        self.gate_fn = gate_fn
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._executed = False
    
    @property
    def result(self) -> Any:
        """Execute and return result on first access."""
        if not self._executed:
            self._result = self.gate_fn(*self.args, **self.kwargs)
            self._executed = True
        return self._result
    
    @property
    def executed(self) -> bool:
        return self._executed


class ParallelGateExecutor:
    """
    Execute gates in parallel.
    
    P2-082: Implement gate parallelization
    """
    
    def __init__(self, max_workers: int = 4, timeout: float = 10.0):
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_all(
        self, 
        gates: Dict[str, Callable[[], bool]],
        fail_fast: bool = True
    ) -> Dict[str, GateResult]:
        """
        Execute all gates in parallel.
        
        Args:
            gates: Dict of gate_name -> gate_function
            fail_fast: Stop on first failure if True
        """
        import time
        results = {}
        futures = {}
        
        # Submit all gates
        for name, gate_fn in gates.items():
            future = self.executor.submit(self._run_gate, name, gate_fn)
            futures[future] = name
        
        # Collect results
        for future in as_completed(futures, timeout=self.timeout):
            name = futures[future]
            try:
                result = future.result()
                results[name] = result
                
                if fail_fast and not result.passed:
                    # Cancel remaining
                    for f in futures:
                        f.cancel()
                    break
                    
            except Exception as e:
                results[name] = GateResult(
                    gate_name=name, passed=False, 
                    duration_ms=0, error=str(e)
                )
        
        return results
    
    def _run_gate(self, name: str, gate_fn: Callable) -> GateResult:
        """Run a single gate with timing."""
        import time
        start = time.perf_counter()
        try:
            passed = gate_fn()
            return GateResult(
                gate_name=name,
                passed=bool(passed),
                duration_ms=(time.perf_counter() - start) * 1000
            )
        except Exception as e:
            return GateResult(
                gate_name=name, passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            )
    
    def shutdown(self) -> None:
        """Shutdown executor."""
        self.executor.shutdown(wait=False)


class AsyncCommitPipeline:
    """
    Async commit pipeline for non-blocking commits.
    
    P2-088: Implement async commit pipeline
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.pending: List[asyncio.Task] = []
    
    async def submit(
        self, 
        commit_fn: Callable,
        *args, **kwargs
    ) -> asyncio.Task:
        """Submit async commit."""
        async with self.semaphore:
            task = asyncio.create_task(
                self._run_commit(commit_fn, *args, **kwargs)
            )
            self.pending.append(task)
            return task
    
    async def _run_commit(self, commit_fn: Callable, *args, **kwargs) -> Any:
        """Run commit in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: commit_fn(*args, **kwargs))
    
    async def wait_all(self) -> List[Any]:
        """Wait for all pending commits."""
        if not self.pending:
            return []
        results = await asyncio.gather(*self.pending, return_exceptions=True)
        self.pending.clear()
        return results
    
    def pending_count(self) -> int:
        """Get number of pending commits."""
        return len([t for t in self.pending if not t.done()])
