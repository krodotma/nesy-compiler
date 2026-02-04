#!/usr/bin/env python3
"""
gibbs_sampler.py - Gibbs Sampling for Population Diversity

DUALITY-BIND E7: Blocked, tempered Gibbs for calibrated diversity.

Ring: 1 (Operator)
Protocol: DKIN v29
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
import numpy as np


@dataclass
class PopulationMember:
    """A member of the population."""
    member_id: str
    genome: Dict[str, Any]
    fitness: float = 0.0
    temperature: float = 1.0


class GibbsSampler:
    """Blocked, tempered Gibbs sampling for population maintenance."""
    
    def __init__(
        self,
        block_size: int = 3,
        temperatures: List[float] = None,
        swap_probability: float = 0.1,
    ):
        self.block_size = block_size
        self.temperatures = temperatures or [1.0, 2.0, 4.0]
        self.swap_probability = swap_probability
        self.population: List[PopulationMember] = []
    
    def initialize_population(self, size: int, genome_factory: Callable[[], Dict]):
        """Initialize population with random genomes."""
        self.population = []
        for i in range(size):
            temp_idx = i % len(self.temperatures)
            self.population.append(PopulationMember(
                member_id=f"member-{i}",
                genome=genome_factory(),
                temperature=self.temperatures[temp_idx],
            ))
    
    def compute_energy(self, member: PopulationMember, fitness_fn: Callable) -> float:
        """Compute energy (negative log fitness) for a member."""
        fitness = fitness_fn(member.genome)
        member.fitness = fitness
        return -math.log(fitness + 1e-10)
    
    def gibbs_step(self, fitness_fn: Callable):
        """Perform one Gibbs sweep across all blocks."""
        random.shuffle(self.population)
        
        for i in range(0, len(self.population), self.block_size):
            block = self.population[i:i + self.block_size]
            self._sample_block(block, fitness_fn)
        
        # Parallel tempering: attempt swaps between temperature levels
        self._parallel_tempering_swap(fitness_fn)
    
    def _sample_block(self, block: List[PopulationMember], fitness_fn: Callable):
        """Sample new values for a block."""
        for member in block:
            # Propose mutation
            new_genome = self._mutate(member.genome)
            
            # Compute acceptance probability
            old_energy = self.compute_energy(member, fitness_fn)
            
            temp_member = PopulationMember(
                member_id=member.member_id,
                genome=new_genome,
                temperature=member.temperature,
            )
            new_energy = self.compute_energy(temp_member, fitness_fn)
            
            # Metropolis-Hastings acceptance
            delta = (old_energy - new_energy) / member.temperature
            if delta > 0 or random.random() < math.exp(delta):
                member.genome = new_genome
                member.fitness = temp_member.fitness
    
    def _mutate(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random mutation to genome."""
        new_genome = genome.copy()
        if new_genome:
            key = random.choice(list(new_genome.keys()))
            if isinstance(new_genome[key], (int, float)):
                new_genome[key] = new_genome[key] + random.gauss(0, 0.1)
            elif isinstance(new_genome[key], bool):
                new_genome[key] = not new_genome[key]
        return new_genome
    
    def _parallel_tempering_swap(self, fitness_fn: Callable):
        """Attempt swaps between adjacent temperature levels."""
        by_temp = {}
        for m in self.population:
            if m.temperature not in by_temp:
                by_temp[m.temperature] = []
            by_temp[m.temperature].append(m)
        
        temps = sorted(by_temp.keys())
        for i in range(len(temps) - 1):
            if random.random() < self.swap_probability:
                low_temp = temps[i]
                high_temp = temps[i + 1]
                
                if by_temp[low_temp] and by_temp[high_temp]:
                    low_member = random.choice(by_temp[low_temp])
                    high_member = random.choice(by_temp[high_temp])
                    
                    # Swap temperatures
                    low_member.temperature, high_member.temperature = (
                        high_member.temperature, low_member.temperature
                    )
    
    def get_best(self, n: int = 5) -> List[PopulationMember]:
        """Get top n members by fitness."""
        return sorted(self.population, key=lambda m: m.fitness, reverse=True)[:n]
    
    def compute_diversity(self) -> float:
        """Compute population diversity as entropy."""
        if not self.population:
            return 0.0
        
        fitnesses = [m.fitness for m in self.population]
        total = sum(fitnesses) + 1e-10
        probs = [f / total for f in fitnesses]
        
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        return entropy


def main():
    parser = argparse.ArgumentParser(description="Gibbs Sampler")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    
    sampler = GibbsSampler()
    
    # Initialize with random genomes
    sampler.initialize_population(
        args.population,
        lambda: {"x": random.random(), "y": random.random()}
    )
    
    # Fitness function: maximize x + y
    fitness_fn = lambda g: g.get("x", 0) + g.get("y", 0) + 0.1
    
    print(f"Initial diversity: {sampler.compute_diversity():.3f}")
    
    for step in range(args.steps):
        sampler.gibbs_step(fitness_fn)
    
    print(f"Final diversity: {sampler.compute_diversity():.3f}")
    print(f"Best members:")
    for m in sampler.get_best(3):
        print(f"  {m.member_id}: fitness={m.fitness:.3f}, temp={m.temperature}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
