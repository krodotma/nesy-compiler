#!/usr/bin/env python3
"""
fuzzer.py - Coverage-Guided Fuzzing for ARK

P2-067: Create `ark fuzz` command
P2-068: Implement coverage-guided fuzzing
P2-069: Add adversarial input generation
"""

import random
import hashlib
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum

logger = logging.getLogger("ARK.Testing.Fuzzer")


class FuzzStatus(Enum):
    PASS = "pass"
    CRASH = "crash"
    HANG = "hang"
    INTERESTING = "interesting"


@dataclass
class FuzzInput:
    """A fuzzing input."""
    id: str
    data: Dict[str, Any]
    parent_id: Optional[str] = None
    mutation: Optional[str] = None


@dataclass
class FuzzResult:
    """Result of fuzzing session."""
    total_inputs: int
    crashes: int
    hangs: int
    new_coverage: int
    coverage_percentage: float
    interesting_inputs: List[FuzzInput] = field(default_factory=list)
    crash_inputs: List[FuzzInput] = field(default_factory=list)
    execution_time: float = 0.0


class CoverageFuzzer:
    """Coverage-guided fuzzer for ARK."""
    
    def __init__(self, max_iterations: int = 1000, timeout_ms: float = 1000):
        self.max_iterations = max_iterations
        self.timeout_ms = timeout_ms
        self.corpus: List[FuzzInput] = []
        self.coverage_seen: Set[str] = set()
        self.crash_inputs: List[FuzzInput] = []
        self._input_counter = 0
    
    def fuzz(
        self,
        target: Callable[[Dict[str, Any]], Any],
        seed_inputs: List[Dict[str, Any]],
        coverage_fn: Optional[Callable[[], Set[str]]] = None
    ) -> FuzzResult:
        """Run fuzzing session."""
        start_time = time.time()
        
        for seed_data in seed_inputs:
            self._input_counter += 1
            self.corpus.append(FuzzInput(f"f{self._input_counter}", seed_data))
        
        total, crashes, hangs, new_cov = 0, 0, 0, 0
        interesting = []
        
        for _ in range(self.max_iterations):
            if not self.corpus:
                break
            
            parent = random.choice(self.corpus)
            mutated, mutation = self._mutate(parent.data)
            self._input_counter += 1
            new_input = FuzzInput(f"f{self._input_counter}", mutated, parent.id, mutation)
            total += 1
            
            try:
                target(mutated)
                if coverage_fn:
                    cov_hash = hashlib.md5(str(coverage_fn()).encode()).hexdigest()
                    if cov_hash not in self.coverage_seen:
                        self.coverage_seen.add(cov_hash)
                        interesting.append(new_input)
                        self.corpus.append(new_input)
                        new_cov += 1
            except Exception:
                crashes += 1
                self.crash_inputs.append(new_input)
        
        return FuzzResult(total, crashes, hangs, new_cov, 
                         len(self.coverage_seen)/max(total,1)*100,
                         interesting, self.crash_inputs, time.time()-start_time)
    
    def _mutate(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Mutate input data."""
        mutated = data.copy()
        if not mutated:
            return mutated, "empty"
        
        key = random.choice(list(mutated.keys()))
        value = mutated[key]
        
        if isinstance(value, (int, float)):
            mutated[key] = random.choice([0, 1, -1, value+1, value-1])
        elif isinstance(value, str):
            mutated[key] = random.choice(["", value+"x", "a"*100])
        elif isinstance(value, bool):
            mutated[key] = not value
        
        return mutated, f"mutate:{key}"
    
    def fuzz_gate(self, gate_fn: Callable[[Dict], bool]) -> FuzzResult:
        """Fuzz a DNA gate with entropy inputs."""
        seeds = [
            {f"h_{i}": random.random() for i in range(8)}
            for _ in range(10)
        ]
        return self.fuzz(gate_fn, seeds)
