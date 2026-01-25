#!/usr/bin/env python3
"""
property.py - Property-Based Testing for ARK

P2-062: Implement property-based testing

Implements Hypothesis-style property testing:
- Define properties that should hold for all inputs
- Generate random inputs automatically
- Shrink failing cases to minimal examples
"""

import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from enum import Enum

logger = logging.getLogger("ARK.Testing.Property")

T = TypeVar('T')


class PropertyStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    GAVE_UP = "gave_up"  # Too many invalid inputs
    ERROR = "error"


@dataclass
class PropertyResult:
    """Result of property test execution."""
    status: PropertyStatus
    property_name: str
    examples_tested: int
    counterexample: Optional[Any] = None
    error: Optional[str] = None
    shrunk_example: Optional[Any] = None
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "property_name": self.property_name,
            "examples_tested": self.examples_tested,
            "counterexample": str(self.counterexample) if self.counterexample else None,
            "error": self.error,
            "shrunk_example": str(self.shrunk_example) if self.shrunk_example else None
        }


@dataclass
class Property:
    """A property that should hold for all inputs."""
    name: str
    predicate: Callable[..., bool]
    generators: Dict[str, Callable[[], Any]]  # arg_name -> generator
    precondition: Optional[Callable[..., bool]] = None
    description: str = ""
    
    def check(self, **kwargs) -> bool:
        """Check if property holds for given inputs."""
        if self.precondition and not self.precondition(**kwargs):
            return True  # Skip if precondition not met
        return self.predicate(**kwargs)


class PropertyTester:
    """
    Property-based testing engine.
    
    P2-062: Property-based testing
    """
    
    def __init__(
        self,
        max_examples: int = 100,
        max_shrinks: int = 50,
        seed: Optional[int] = None
    ):
        self.max_examples = max_examples
        self.max_shrinks = max_shrinks
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        # Built-in generators
        self.generators = {
            "int": lambda: random.randint(-1000, 1000),
            "positive_int": lambda: random.randint(1, 1000),
            "float": lambda: random.uniform(-1000, 1000),
            "str": lambda: ''.join(random.choices('abcdefghijklmnop', k=random.randint(0, 20))),
            "bool": lambda: random.choice([True, False]),
            "list_int": lambda: [random.randint(-100, 100) for _ in range(random.randint(0, 10))],
            "dict_str": lambda: {f"k{i}": f"v{i}" for i in range(random.randint(0, 5))}
        }
    
    def test(self, prop: Property) -> PropertyResult:
        """Test a property with random inputs."""
        examples_tested = 0
        gave_up_count = 0
        max_gave_up = self.max_examples * 5
        
        for _ in range(self.max_examples + max_gave_up):
            if examples_tested >= self.max_examples:
                break
            
            # Generate inputs
            try:
                inputs = {
                    arg: gen() for arg, gen in prop.generators.items()
                }
            except Exception as e:
                logger.warning("Generator failed: %s", e)
                gave_up_count += 1
                continue
            
            # Check precondition
            if prop.precondition:
                try:
                    if not prop.precondition(**inputs):
                        gave_up_count += 1
                        continue
                except Exception:
                    gave_up_count += 1
                    continue
            
            examples_tested += 1
            
            # Check property
            try:
                if not prop.check(**inputs):
                    # Found counterexample - try to shrink
                    shrunk = self._shrink(prop, inputs)
                    return PropertyResult(
                        status=PropertyStatus.FAILED,
                        property_name=prop.name,
                        examples_tested=examples_tested,
                        counterexample=inputs,
                        shrunk_example=shrunk
                    )
            except Exception as e:
                return PropertyResult(
                    status=PropertyStatus.ERROR,
                    property_name=prop.name,
                    examples_tested=examples_tested,
                    counterexample=inputs,
                    error=str(e)
                )
        
        if examples_tested < self.max_examples // 2:
            return PropertyResult(
                status=PropertyStatus.GAVE_UP,
                property_name=prop.name,
                examples_tested=examples_tested
            )
        
        return PropertyResult(
            status=PropertyStatus.PASSED,
            property_name=prop.name,
            examples_tested=examples_tested
        )
    
    def _shrink(self, prop: Property, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Shrink counterexample to minimal failing case."""
        current = inputs.copy()
        
        for _ in range(self.max_shrinks):
            # Try shrinking each input
            improved = False
            
            for arg, value in current.items():
                shrunk_value = self._shrink_value(value)
                if shrunk_value != value:
                    test_inputs = {**current, arg: shrunk_value}
                    
                    try:
                        if not prop.check(**test_inputs):
                            current[arg] = shrunk_value
                            improved = True
                    except Exception:
                        current[arg] = shrunk_value
                        improved = True
            
            if not improved:
                break
        
        return current
    
    def _shrink_value(self, value: Any) -> Any:
        """Shrink a single value towards simpler form."""
        if isinstance(value, int):
            if value == 0:
                return 0
            return value // 2
        elif isinstance(value, float):
            if abs(value) < 0.001:
                return 0.0
            return value / 2
        elif isinstance(value, str):
            if len(value) <= 1:
                return value
            return value[:-1]
        elif isinstance(value, list):
            if len(value) <= 1:
                return value
            return value[:-1]
        elif isinstance(value, dict):
            if len(value) <= 1:
                return value
            key = list(value.keys())[-1]
            return {k: v for k, v in value.items() if k != key}
        return value
    
    # --- Built-in properties for ARK gates ---
    
    def cmp_monotonicity(self) -> Property:
        """CMP should improve monotonically over valid commits."""
        return Property(
            name="cmp_monotonicity",
            predicate=lambda before, after: after >= before * 0.9,  # Allow small drops
            generators={
                "before": lambda: random.uniform(0, 1),
                "after": lambda: random.uniform(0, 1)
            },
            precondition=lambda before, after: after > 0,
            description="CMP should not drop significantly on valid commits"
        )
    
    def entropy_bounds(self) -> Property:
        """All entropy values should be in [0, 1]."""
        return Property(
            name="entropy_bounds",
            predicate=lambda h: all(0 <= v <= 1 for v in h.values()),
            generators={
                "h": lambda: {
                    f"h_{i}": random.uniform(-0.1, 1.1)
                    for i in range(8)
                }
            },
            description="Entropy values must be bounded [0, 1]"
        )
    
    def gate_consistency(self) -> Property:
        """Gate decisions should be consistent for same inputs."""
        return Property(
            name="gate_consistency",
            predicate=lambda a, b: a == b,  # Same inputs -> same output
            generators={
                "a": lambda: random.choice([True, False]),
                "b": lambda: True  # Will be overwritten
            },
            description="Gates must be deterministic"
        )
